//! Streaming side of the openai-responses adapter: `processResponsesStream`
//! (upstream `openai-responses-shared.ts`) over the `OpenAI` SDK's event
//! iteration, and the adapter's `stream` (`openai-responses.ts`).
//!

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use bytes::Bytes;
use futures::StreamExt;
use serde_json::{Map, Value};
use tokio_util::sync::CancellationToken;

use super::{OpenAiResponsesOptions, ServiceTier, build_request, configuration_update_effort};
use crate::api::constrained_sampling::GrammarToolInputJsonBuffer;
use crate::api::openai_sdk::{SdkError, SdkRecord, read_record, status_error};
use crate::api::stream_support::{
    STREAM_CAPACITY, closed, finish, header_pairs, initial_message, notify_response,
    request_policy, send,
};
use crate::cost::calculate_cost;
use crate::http::client::{HttpClient, SendError};
use crate::http::error_body::format_provider_error;
use crate::http::retry::jitter;
use crate::http::sse::{EndOfStream, SseEventStream};
use crate::json::is_truthy;
use crate::options::{SimpleStreamOptions, StreamOptions};
use crate::provider::ChatApi;
use crate::stream::{AssistantStream, AssistantStreamSender, DoneReason, Metadata};
use crate::transcript::TranscriptContext;
use crate::types::content::{
    AssistantContentBlock, StopReason, TextContent, ThinkingContent, ToolCall,
};
use crate::types::model::Model;
use crate::types::usage::Usage;

/// Where Sign in with `ChatGPT` users check their usage.
const CHATGPT_USAGE_URL: &str = "https://chatgpt.com/settings/usage";

/// Shown when a `ChatGPT` workspace's per-user limit is reached.
const CHATGPT_WORKSPACE_LIMIT_HINT: &str = "You've reached your ChatGPT workspace's usage limit. Ask your workspace admin to increase it at admin.openai.com → Workspace → Usage limits.";

/// The `openai-responses` chat API.
#[derive(Clone, Debug)]
pub struct OpenAiResponsesApi {
    http: Arc<HttpClient>,
}

impl OpenAiResponsesApi {
    /// An adapter sending requests through the shared `http` pool.
    #[must_use]
    pub const fn new(http: Arc<HttpClient>) -> Self {
        Self { http }
    }

    /// `stream` with the Responses options.
    #[must_use]
    pub fn stream_with(
        &self,
        model: &Model,
        context: TranscriptContext,
        options: OpenAiResponsesOptions,
    ) -> AssistantStream {
        let base = initial_message(model);
        let model = model.clone();
        let http = Arc::clone(&self.http);
        AssistantStream::spawn(base, STREAM_CAPACITY, move |mut sender| async move {
            let signal = options.base.signal.clone().unwrap_or_default();
            let outcome = run(&http, &model, &context, &options, &signal, &mut sender)
                .await
                .map_err(with_usage_limit_hint);
            finish(sender, outcome, signal.is_cancelled()).await;
        })
    }
}

impl ChatApi for OpenAiResponsesApi {
    fn stream(
        &self,
        model: &Model,
        context: TranscriptContext,
        options: StreamOptions,
    ) -> AssistantStream {
        let options = OpenAiResponsesOptions {
            base: options,
            ..OpenAiResponsesOptions::default()
        };
        self.stream_with(model, context, options)
    }

    /// `streamSimple`: the simple reasoning level clamped to the model.
    fn stream_simple(
        &self,
        model: &Model,
        context: TranscriptContext,
        options: SimpleStreamOptions,
    ) -> AssistantStream {
        let options = OpenAiResponsesOptions::from_simple(model, &context, &options);
        self.stream_with(model, context, options)
    }
}

/// Send the request and process the events; the error string is the
/// message upstream's `catch` records.
async fn run(
    http: &HttpClient,
    model: &Model,
    context: &TranscriptContext,
    options: &OpenAiResponsesOptions,
    signal: &CancellationToken,
    sender: &mut AssistantStreamSender,
) -> Result<DoneReason, String> {
    // Upstream writes it on the output before anything can fail.
    if let Some(level) = configuration_update_effort(model, options) {
        send(sender, Metadata::ProviderThinkingLevel(level)).await?;
    }
    let base = &options.base;
    let request = build_request(model, context, options)?;
    let body = crate::api::stream_support::on_payload_body(
        base,
        &Value::Object(request.params.clone()),
        &request.body,
        model,
        crate::api::stream_support::PayloadMerge::Replace,
    )
    .await?;
    let sent = http
        .post_json(
            &request.url,
            &Bytes::from(body),
            &header_pairs(&request.headers),
            &request_policy(base),
            signal,
            &jitter,
        )
        .await;
    let response = match sent {
        Ok(response) => response,
        // `retryProviderRequest` replaces any failure of an aborted request.
        Err(_) if signal.is_cancelled() => return Err(String::from("Request aborted")),
        Err(SendError::Provider(error)) => return Err(catch_message(&status_error(&error), model)),
        Err(SendError::Transport(message)) => return Err(message),
        Err(SendError::Aborted) => return Err(String::from("Request aborted")),
    };
    notify_response(base, &response, model).await;
    sender.start().await.map_err(|_| closed())?;

    let mut state = ResponsesState::new(
        model,
        &request.grammar_tool_input_properties,
        options.service_tier,
    );
    process(&mut state, response, signal, sender, base).await?;
    if signal.is_cancelled() {
        return Err(String::from("Request was aborted"));
    }
    match state.stop_reason {
        StopReason::Pending => Err(String::from(
            "OpenAI Responses stream ended without a stop reason",
        )),
        StopReason::Aborted | StopReason::Error => Err(state
            .error_message
            .unwrap_or_else(|| String::from("An unknown error occurred"))),
        StopReason::Length => Ok(DoneReason::Length),
        StopReason::ToolUse => Ok(DoneReason::ToolUse),
        StopReason::Stop | StopReason::Deferred => Ok(DoneReason::Stop),
    }
}

/// `processResponsesStream`: the event loop and its end-of-stream checks.
async fn process(
    state: &mut ResponsesState<'_>,
    response: reqwest::Response,
    signal: &CancellationToken,
    sender: &mut AssistantStreamSender,
    base: &StreamOptions,
) -> Result<(), String> {
    let mut events = SseEventStream::new(response.bytes_stream(), EndOfStream::Flush);
    loop {
        // An abort ends the SDK's iteration quietly; the checks below report it.
        let next = tokio::select! {
            biased;
            () = signal.cancelled() => break,
            next = events.next() => next,
        };
        let Some(next) = next else { break };
        let sse = next.map_err(|error| error.to_string())?;
        let event = match read_record(&sse).map_err(|error| catch_message(&error, state.model))? {
            SdkRecord::Done => break,
            SdkRecord::Skip => continue,
            SdkRecord::Data(event) => event,
        };
        crate::api::stream_support::on_provider_stream_event(base, &event, state.model).await;
        state.apply(&event, sender).await?;
    }
    if !state.saw_terminal {
        return Err(String::from(
            "OpenAI Responses stream ended before a terminal response event",
        ));
    }
    // The agent runs every tool call in the final message; one whose
    // `output_item.done` never arrived may carry cut-off arguments.
    if state.stop_reason == StopReason::ToolUse
        && let Some((name, id)) = state.unfinished.first()
    {
        return Err(format!(
            "OpenAI Responses stream completed with an unfinished tool call: {name} ({id})"
        ));
    }
    Ok(())
}

/// pi's `catch` for an SDK error: `formatProviderError` with the provider's
/// `API error` prefix.
fn catch_message(error: &SdkError, model: &Model) -> String {
    let provider = model.provider.as_str();
    let name = if provider == "openai" {
        "OpenAI"
    } else {
        provider
    };
    format_provider_error(&error.normalized(), Some(&format!("{name} API error")))
}

/// The hint pi appends when Sign in with `ChatGPT` hits a usage limit
/// shared with other apps or set by a workspace admin.
fn with_usage_limit_hint(message: String) -> String {
    if message.contains("subscription_sharing_usage_limit_exceeded") {
        format!("{message}\nCheck your ChatGPT usage: {CHATGPT_USAGE_URL}")
    } else if message.contains("subscription_sharing_workplace_individual_usage_limit_exceeded") {
        format!("{message}\n{CHATGPT_WORKSPACE_LIMIT_HINT}")
    } else {
        message
    }
}

/// The stream state of a tool call, kept off the content block.
#[derive(Debug)]
enum ToolInput {
    /// JSON arguments, folded by the sender.
    Json,
    /// A grammar tool's input, sent as JSON arguments text.
    Custom {
        property: String,
        buffer: GrammarToolInputJsonBuffer,
        /// JSON text of the item's initial input, sent with the first piece.
        unsent: String,
        /// A piece has been sent; before that the block shows its
        /// `{property: input}` placeholder.
        sent: bool,
    },
}

/// An output item being streamed, by `output_index`.
#[derive(Debug)]
enum Slot {
    Thinking {
        index: usize,
        thinking: String,
    },
    Text {
        index: usize,
        text: String,
    },
    ToolCall {
        index: usize,
        call: Box<ToolCall>,
        input: ToolInput,
    },
}

/// A finished reasoning block, kept for the terminal signature backfill.
#[derive(Debug)]
struct Reasoning {
    index: usize,
    thinking: String,
    signature: String,
}

/// Upstream's `output` fields plus the streaming slots.
struct ResponsesState<'a> {
    model: &'a Model,
    grammar_properties: &'a BTreeMap<String, String>,
    service_tier: Option<ServiceTier>,
    /// Keyed like upstream's `Map`: a missing `output_index` is a key too.
    slots: HashMap<Option<u64>, Slot>,
    /// Tool calls whose `output_item.done` has not arrived, in start order.
    unfinished: Vec<(String, String)>,
    reasoning_by_id: HashMap<String, Reasoning>,
    has_tool_calls: bool,
    stop_reason: StopReason,
    error_message: Option<String>,
    saw_terminal: bool,
}

impl<'a> ResponsesState<'a> {
    fn new(
        model: &'a Model,
        grammar_properties: &'a BTreeMap<String, String>,
        service_tier: Option<ServiceTier>,
    ) -> Self {
        Self {
            model,
            grammar_properties,
            service_tier,
            slots: HashMap::new(),
            unfinished: Vec::new(),
            reasoning_by_id: HashMap::new(),
            has_tool_calls: false,
            stop_reason: StopReason::Pending,
            error_message: None,
            saw_terminal: false,
        }
    }

    /// Fold one Responses stream event.
    async fn apply(
        &mut self,
        event: &Value,
        sender: &mut AssistantStreamSender,
    ) -> Result<(), String> {
        let key = event.get("output_index").and_then(Value::as_u64);
        let text = |name: &str| event.get(name).and_then(Value::as_str).unwrap_or_default();
        match event.get("type").and_then(Value::as_str) {
            Some("response.created") => {
                if let Some(id) = event.pointer("/response/id").and_then(Value::as_str) {
                    send(sender, Metadata::ResponseId(id.to_owned())).await?;
                }
            }
            Some("response.output_item.added") => {
                if let Some(item) = event.get("item") {
                    self.create_slot(key, item, sender).await?;
                }
            }
            Some("response.reasoning_summary_text.delta" | "response.reasoning_text.delta") => {
                self.thinking_delta(key, text("delta"), sender).await?;
            }
            Some("response.reasoning_summary_part.done") => {
                self.thinking_delta(key, "\n\n", sender).await?;
            }
            Some("response.output_text.delta" | "response.refusal.delta") => {
                if let Some(Slot::Text { index, text: block }) = self.slots.get_mut(&key) {
                    let delta = text("delta");
                    block.push_str(delta);
                    sender
                        .text_delta(*index, delta)
                        .await
                        .map_err(|_| closed())?;
                }
            }
            Some("response.function_call_arguments.delta") => {
                if let Some(Slot::ToolCall {
                    index,
                    input: ToolInput::Json,
                    ..
                }) = self.slots.get(&key)
                {
                    sender
                        .toolcall_delta(*index, text("delta"))
                        .await
                        .map_err(|_| closed())?;
                }
            }
            Some("response.function_call_arguments.done") => {
                if let Some(Slot::ToolCall {
                    index,
                    input: ToolInput::Json,
                    ..
                }) = self.slots.get(&key)
                {
                    sender
                        .toolcall_replace(*index, text("arguments"))
                        .await
                        .map_err(|_| closed())?;
                }
            }
            Some("response.custom_tool_call_input.delta") => {
                if let Some(Slot::ToolCall { index, call, input }) = self.slots.get_mut(&key) {
                    let piece = input.append(text("delta"))?;
                    send_piece(sender, *index, call, input, piece).await?;
                }
            }
            Some("response.custom_tool_call_input.done") => {
                if let Some(Slot::ToolCall { index, call, input }) = self.slots.get_mut(&key) {
                    let piece = input.close(event.get("input").and_then(Value::as_str))?;
                    send_piece(sender, *index, call, input, piece).await?;
                }
            }
            Some("response.output_item.done") => {
                if let Some(item) = event.get("item") {
                    self.item_done(key, item, sender).await?;
                }
            }
            Some("response.completed" | "response.incomplete") => {
                let response = event.get("response").unwrap_or(&Value::Null);
                self.finalize(response, sender).await?;
            }
            Some("error") => {
                return Err(format!(
                    "Error Code {}: {}",
                    js_template(event.get("code")),
                    js_template(event.get("message"))
                ));
            }
            Some("response.failed") => return Err(self.failed(event, sender).await?),
            _ => {}
        }
        Ok(())
    }

    async fn thinking_delta(
        &mut self,
        key: Option<u64>,
        delta: &str,
        sender: &mut AssistantStreamSender,
    ) -> Result<(), String> {
        if let Some(Slot::Thinking { index, thinking }) = self.slots.get_mut(&key) {
            thinking.push_str(delta);
            sender
                .thinking_delta(*index, delta)
                .await
                .map_err(|_| closed())?;
        }
        Ok(())
    }

    /// `createSlot`: open the block of an output item; unknown item types
    /// get none.
    async fn create_slot(
        &mut self,
        key: Option<u64>,
        item: &Value,
        sender: &mut AssistantStreamSender,
    ) -> Result<(), String> {
        let field = |name: &str| item.get(name).and_then(Value::as_str);
        let slot = match field("type") {
            Some("reasoning") => Slot::Thinking {
                index: sender.thinking_start().await.map_err(|_| closed())?,
                thinking: String::new(),
            },
            Some("message") => Slot::Text {
                index: sender.text_start().await.map_err(|_| closed())?,
                text: String::new(),
            },
            Some("function_call") => {
                let call = tool_call(item, Map::new());
                let index = sender
                    .toolcall_start(call.clone())
                    .await
                    .map_err(|_| closed())?;
                if let Some(arguments) =
                    field("arguments").filter(|arguments| !arguments.is_empty())
                {
                    sender
                        .toolcall_arguments(index, arguments)
                        .await
                        .map_err(|_| closed())?;
                }
                self.start_tool_call(&call);
                Slot::ToolCall {
                    index,
                    call: Box::new(call),
                    input: ToolInput::Json,
                }
            }
            Some("custom_tool_call") => {
                // `"input"` only stashes the input of a tool the model made up.
                let property = self
                    .grammar_properties
                    .get(field("name").unwrap_or_default())
                    .cloned()
                    .unwrap_or_else(|| String::from("input"));
                let initial = field("input").unwrap_or_default();
                let mut buffer = GrammarToolInputJsonBuffer::default();
                let unsent = buffer
                    .append_delta(&property, initial)
                    .map_err(|error| error.to_string())?
                    .unwrap_or_default();
                let mut arguments = Map::new();
                arguments.insert(property.clone(), Value::from(initial));
                let call = tool_call(item, arguments);
                let index = sender
                    .toolcall_start(call.clone())
                    .await
                    .map_err(|_| closed())?;
                self.start_tool_call(&call);
                Slot::ToolCall {
                    index,
                    call: Box::new(call),
                    input: ToolInput::Custom {
                        property,
                        buffer,
                        unsent,
                        sent: false,
                    },
                }
            }
            _ => return Ok(()),
        };
        self.slots.insert(key, slot);
        Ok(())
    }

    fn start_tool_call(&mut self, call: &ToolCall) {
        self.has_tool_calls = true;
        self.unfinished.push((call.name.clone(), call.id.clone()));
    }

    /// `response.output_item.done`: the item's authoritative content ends
    /// its block; a block that never started starts here.
    async fn item_done(
        &mut self,
        key: Option<u64>,
        item: &Value,
        sender: &mut AssistantStreamSender,
    ) -> Result<(), String> {
        if !self.slots.contains_key(&key) {
            self.create_slot(key, item, sender).await?;
        }
        let kind = item.get("type").and_then(Value::as_str);
        let matches = matches!(
            (kind, self.slots.get(&key)),
            (Some("reasoning"), Some(Slot::Thinking { .. }))
                | (Some("message"), Some(Slot::Text { .. }))
                | (
                    Some("function_call"),
                    Some(Slot::ToolCall {
                        input: ToolInput::Json,
                        ..
                    })
                )
                | (
                    Some("custom_tool_call"),
                    Some(Slot::ToolCall {
                        input: ToolInput::Custom { .. },
                        ..
                    })
                )
        );
        if !matches {
            return Ok(());
        }
        let Some(slot) = self.slots.remove(&key) else {
            return Ok(());
        };
        match slot {
            Slot::Thinking { index, thinking } => {
                self.reasoning_done(index, thinking, item, sender).await
            }
            Slot::Text { index, .. } => {
                let block = TextContent {
                    text: message_text(item),
                    text_signature: Some(text_signature(item)?),
                };
                sender.text_end(index, block).await.map_err(|_| closed())
            }
            Slot::ToolCall {
                index,
                mut call,
                mut input,
            } => {
                // The item's arguments replace the streamed text; a grammar
                // call closes with the provider's final input.
                let arguments = item.get("arguments").and_then(Value::as_str);
                match (&input, arguments) {
                    (ToolInput::Json, Some(arguments)) if !arguments.is_empty() => sender
                        .toolcall_arguments(index, arguments)
                        .await
                        .map_err(|_| closed())?,
                    (ToolInput::Json, _) => {}
                    (ToolInput::Custom { .. }, _) => {
                        let piece = input.close(item.get("input").and_then(Value::as_str))?;
                        send_piece(sender, index, &call, &mut input, piece).await?;
                    }
                }
                if let Some(namespace) = item.get("namespace").and_then(Value::as_str) {
                    call.namespace = Some(namespace.to_owned());
                    sender
                        .toolcall_fields(index, (*call).clone())
                        .await
                        .map_err(|_| closed())?;
                }
                if let Some(position) = self
                    .unfinished
                    .iter()
                    .position(|(name, id)| *name == call.name && *id == call.id)
                {
                    self.unfinished.remove(position);
                }
                sender.toolcall_end(index).await.map_err(|_| closed())
            }
        }
    }

    /// A reasoning item's summary (else its reasoning text) ends the block;
    /// the whole item is the signature replayed on the next turn.
    async fn reasoning_done(
        &mut self,
        index: usize,
        streamed: String,
        item: &Value,
        sender: &mut AssistantStreamSender,
    ) -> Result<(), String> {
        let joined = |name: &str| {
            item.get(name)
                .and_then(Value::as_array)
                .map(|parts| {
                    parts
                        .iter()
                        .map(|part| part.get("text").and_then(Value::as_str).unwrap_or_default())
                        .collect::<Vec<_>>()
                        .join("\n\n")
                })
                .unwrap_or_default()
        };
        let summary = joined("summary");
        let content = joined("content");
        let thinking = [summary, content]
            .into_iter()
            .find(|text| !text.is_empty())
            .unwrap_or(streamed);
        let signature = crate::json::to_string(item).map_err(|error| error.to_string())?;
        self.reasoning_by_id.insert(
            js_template(item.get("id")),
            Reasoning {
                index,
                thinking: thinking.clone(),
                signature: signature.clone(),
            },
        );
        let block = ThinkingContent {
            thinking,
            thinking_signature: Some(signature),
            redacted: None,
        };
        sender
            .thinking_end(index, block)
            .await
            .map_err(|_| closed())
    }

    /// `finalizeResponse`: backfill reasoning signatures, then the response
    /// id, usage, service-tier pricing and stop reason.
    async fn finalize(
        &mut self,
        response: &Value,
        sender: &mut AssistantStreamSender,
    ) -> Result<(), String> {
        self.saw_terminal = true;
        self.backfill_reasoning_signatures(response, sender).await?;
        if let Some(id) = response
            .get("id")
            .and_then(Value::as_str)
            .filter(|id| !id.is_empty())
        {
            send(sender, Metadata::ResponseId(id.to_owned())).await?;
        }
        let mut usage = response
            .get("usage")
            .filter(|usage| is_truthy(usage))
            .map(parse_usage)
            .unwrap_or_default();
        calculate_cost(self.model, &mut usage);
        let tier = response
            .get("service_tier")
            .and_then(Value::as_str)
            .or_else(|| self.service_tier.map(ServiceTier::as_str));
        apply_service_tier_pricing(&mut usage, tier, self.model);
        send(sender, Metadata::Usage(usage)).await?;

        // An incomplete response keeps the provider's reason, so output
        // truncation and content filtering stay distinct.
        let status = response.get("status").and_then(Value::as_str);
        let reason = response
            .pointer("/incomplete_details/reason")
            .and_then(Value::as_str)
            .filter(|reason| !reason.is_empty());
        let raw = match (status, reason) {
            (_, Some(reason)) => Some(format!("{}.{reason}", status.unwrap_or("undefined"))),
            (Some(status), None) => Some(status.to_owned()),
            (None, None) => None,
        };
        if let Some(raw) = raw {
            send(sender, Metadata::RawStopReason(raw)).await?;
        }
        let (stop_reason, error_message) = map_stop_reason(status, reason)?;
        self.stop_reason = if stop_reason == StopReason::Stop && self.has_tool_calls {
            StopReason::ToolUse
        } else {
            stop_reason
        };
        self.error_message = error_message;
        Ok(())
    }

    /// Azure `OpenAI` can omit `encrypted_content` from
    /// `output_item.done` and send it only in the terminal response; it is
    /// copied into the stored signature so `store: false` replay stays
    /// stateless.
    async fn backfill_reasoning_signatures(
        &mut self,
        response: &Value,
        sender: &mut AssistantStreamSender,
    ) -> Result<(), String> {
        let items = response.get("output").and_then(Value::as_array);
        for item in items.into_iter().flatten() {
            let Some(encrypted) = item
                .get("encrypted_content")
                .filter(|content| is_truthy(content))
                .filter(|_| item.get("type").and_then(Value::as_str) == Some("reasoning"))
            else {
                continue;
            };
            let Some(reasoning) = self.reasoning_by_id.get_mut(&js_template(item.get("id"))) else {
                continue;
            };
            let mut stored: Map<String, Value> =
                crate::json::from_str(&reasoning.signature).map_err(|error| error.to_string())?;
            if stored.get("encrypted_content").is_some_and(is_truthy) {
                continue;
            }
            stored.insert("encrypted_content".into(), encrypted.clone());
            reasoning.signature =
                crate::json::to_string(&stored).map_err(|error| error.to_string())?;
            let block = ThinkingContent {
                thinking: reasoning.thinking.clone(),
                thinking_signature: Some(reasoning.signature.clone()),
                redacted: None,
            };
            sender
                .patch(reasoning.index, AssistantContentBlock::Thinking(block))
                .await
                .map_err(|_| closed())?;
        }
        Ok(())
    }

    /// `response.failed`: the raw status, then the provider's error text.
    async fn failed(
        &mut self,
        event: &Value,
        sender: &AssistantStreamSender,
    ) -> Result<String, String> {
        self.saw_terminal = true;
        let response = event.get("response").unwrap_or(&Value::Null);
        if let Some(status) = response.get("status").and_then(Value::as_str) {
            send(sender, Metadata::RawStopReason(status.to_owned())).await?;
        }
        let truthy_text = |value: Option<&Value>, fallback: &str| {
            value
                .filter(|value| is_truthy(value))
                .map_or_else(|| fallback.to_owned(), |value| js_template(Some(value)))
        };
        let reason = response
            .pointer("/incomplete_details/reason")
            .filter(|reason| is_truthy(reason));
        let unknown = || {
            reason.map_or_else(
                || String::from("Unknown error (no error details in response)"),
                |reason| format!("incomplete: {}", js_template(Some(reason))),
            )
        };
        Ok(response
            .get("error")
            .filter(|error| is_truthy(error))
            .map_or_else(unknown, |error| {
                format!(
                    "{}: {}",
                    truthy_text(error.get("code"), "unknown"),
                    truthy_text(error.get("message"), "no message")
                )
            }))
    }
}

impl ToolInput {
    /// `appendCustomInput`: the JSON piece for a grammar input delta.
    fn append(&mut self, delta: &str) -> Result<Option<String>, String> {
        let Self::Custom {
            property,
            buffer,
            unsent,
            ..
        } = self
        else {
            return Ok(None);
        };
        let piece = buffer
            .append_delta(property, delta)
            .map_err(|error| error.to_string())?;
        Ok(take_unsent(unsent, piece))
    }

    /// `closeCustomInput`: the closing JSON piece, with the provider's final
    /// input when it sent one.
    fn close(&mut self, final_input: Option<&str>) -> Result<Option<String>, String> {
        let Self::Custom {
            property,
            buffer,
            unsent,
            ..
        } = self
        else {
            return Ok(None);
        };
        let piece = buffer
            .close(property, final_input)
            .map_err(|error| error.to_string())?;
        Ok(take_unsent(unsent, piece))
    }
}

/// `sendCustomInput`: the initial input's JSON goes out with the first piece.
fn take_unsent(unsent: &mut String, piece: Option<String>) -> Option<String> {
    if unsent.is_empty() {
        return piece;
    }
    let mut sent = std::mem::take(unsent);
    sent.push_str(&piece.unwrap_or_default());
    Some(sent)
}

/// Send a grammar call's JSON piece; before the first one the block's
/// `{property: input}` placeholder is emptied, so the folded text parses
/// to `{property: input}` like upstream's block.
async fn send_piece(
    sender: &mut AssistantStreamSender,
    index: usize,
    call: &ToolCall,
    input: &mut ToolInput,
    piece: Option<String>,
) -> Result<(), String> {
    let Some(piece) = piece else {
        return Ok(());
    };
    if let ToolInput::Custom { sent, .. } = input
        && !*sent
    {
        *sent = true;
        let block = ToolCall {
            arguments: Map::new(),
            ..call.clone()
        };
        sender
            .patch(index, AssistantContentBlock::ToolCall(block))
            .await
            .map_err(|_| closed())?;
    }
    sender
        .toolcall_delta(index, &piece)
        .await
        .map_err(|_| closed())
}

/// A tool call block for a function or custom call item: `call_id|id`.
fn tool_call(item: &Value, arguments: Map<String, Value>) -> ToolCall {
    ToolCall {
        id: format!(
            "{}|{}",
            js_template(item.get("call_id")),
            js_template(item.get("id"))
        ),
        name: item
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned(),
        arguments,
        thought_signature: None,
        namespace: item
            .get("namespace")
            .and_then(Value::as_str)
            .map(str::to_owned),
        arguments_error: None,
    }
}

/// A message item's text: output text and refusals, joined.
fn message_text(item: &Value) -> String {
    item.get("content")
        .and_then(Value::as_array)
        .map(|parts| {
            parts
                .iter()
                .map(|part| {
                    let field = if part.get("type").and_then(Value::as_str) == Some("output_text") {
                        "text"
                    } else {
                        "refusal"
                    };
                    part.get(field).and_then(Value::as_str).unwrap_or_default()
                })
                .collect::<String>()
        })
        .unwrap_or_default()
}

/// `encodeTextSignatureV1`: the item id and phase.
fn text_signature(item: &Value) -> Result<String, String> {
    let mut payload = Map::new();
    payload.insert("v".into(), Value::from(1));
    if let Some(id) = item.get("id").filter(|id| !id.is_null()) {
        payload.insert("id".into(), id.clone());
    }
    if let Some(phase) = item.get("phase").filter(|phase| is_truthy(phase)) {
        payload.insert("phase".into(), phase.clone());
    }
    crate::json::to_string(&payload).map_err(|error| error.to_string())
}

/// Responses usage to pi's: input excludes cached and cache-write tokens,
/// which `input_tokens` includes.
fn parse_usage(raw: &Value) -> Usage {
    let count = |pointer: &str| raw.pointer(pointer).and_then(Value::as_u64).unwrap_or(0);
    let cache_read = count("/input_tokens_details/cached_tokens");
    let cache_write = count("/input_tokens_details/cache_write_tokens");
    Usage {
        input: count("/input_tokens")
            .saturating_sub(cache_read)
            .saturating_sub(cache_write),
        output: count("/output_tokens"),
        cache_read,
        cache_write,
        total_tokens: count("/total_tokens"),
        reasoning: Some(count("/output_tokens_details/reasoning_tokens")),
        ..Usage::default()
    }
}

/// `applyServiceTierPricing`: flex costs half, priority and fast double
/// (2.5x on `gpt-5.5`).
fn apply_service_tier_pricing(usage: &mut Usage, tier: Option<&str>, model: &Model) {
    let multiplier = match tier {
        Some("flex") => 0.5,
        Some("priority" | "fast") if model.id == "gpt-5.5" => 2.5,
        Some("priority" | "fast") => 2.0,
        _ => return,
    };
    let cost = &mut usage.cost;
    cost.input *= multiplier;
    cost.output *= multiplier;
    cost.cache_read *= multiplier;
    cost.cache_write *= multiplier;
    cost.total = cost.input + cost.output + cost.cache_read + cost.cache_write;
}

/// `mapStopReason`: a response status to pi's stop reason.
///
/// # Errors
/// Returns upstream's message for a status it does not know.
fn map_stop_reason(
    status: Option<&str>,
    incomplete_reason: Option<&str>,
) -> Result<(StopReason, Option<String>), String> {
    Ok(match status {
        None | Some("completed" | "in_progress" | "queued") => (StopReason::Stop, None),
        Some("incomplete") => match incomplete_reason {
            Some("max_output_tokens") => (StopReason::Length, None),
            Some(reason) => (
                StopReason::Error,
                Some(format!("Response incomplete: {reason}")),
            ),
            None => (
                StopReason::Error,
                Some(String::from(
                    "Response incomplete without a provider reason",
                )),
            ),
        },
        Some("failed" | "cancelled") => (StopReason::Error, None),
        Some(other) => return Err(format!("Unhandled stop reason: {other}")),
    })
}

/// A JSON value as a JavaScript template literal renders it.
fn js_template(value: Option<&Value>) -> String {
    match value {
        None => String::from("undefined"),
        Some(Value::String(text)) => text.clone(),
        Some(Value::Object(_)) => String::from("[object Object]"),
        Some(other) => crate::json::to_string(other).unwrap_or_default(),
    }
}
