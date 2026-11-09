//! Streaming side of the anthropic-messages adapter: pi's event gating
//! (`iterateAnthropicEvents`) and the mapping of Anthropic stream events onto
//! the assistant stream (`stream` in `anthropic-messages.ts`).
//!
//! pi decodes SSE itself and delivers a record left pending at end of
//! stream, which is [`EndOfStream::Flush`].
//!
//! Not yet ported: `OpenRouter` billed-cost fields in `message_delta` usage,
//! `input_transformations` diagnostics, the strict-tools fallback retry and
//! `onProviderStreamEvent`.

use std::collections::HashMap;

use bytes::Bytes;
use futures::StreamExt;
use serde_json::{Map, Value};
use tokio_util::sync::CancellationToken;

use super::convert::from_claude_code_name;
use super::{AnthropicOptions, build_request, is_oauth_token};
use crate::cost::calculate_cost;
use crate::http::client::{HttpClient, HttpClientConfig, HttpError, SendError};
use crate::http::retry::jitter;
use crate::http::sse::{EndOfStream, SseEvent, SseEventStream};
use crate::json::repair::parse_json_with_repair;
use crate::options::{ResponseMetadata, StreamOptions};
use crate::provider::ChatApi;
use crate::stream::accumulator::wall_ms;
use crate::stream::{AssistantStream, AssistantStreamSender, DoneReason, ErrorReason, Metadata};
use crate::transcript::{TranscriptContext, get_current_tools};
use crate::types::content::{StopReason, TextContent, ThinkingContent, ToolCall};
use crate::types::context::Context;
use crate::types::message::AssistantMessage;
use crate::types::model::Model;
use crate::types::tool::Tool;
use crate::types::usage::Usage;

/// Bounded capacity of the adapter's event channel.
const STREAM_CAPACITY: usize = 64;

/// SSE event names pi forwards; everything else (pings) is skipped.
const ANTHROPIC_MESSAGE_EVENTS: [&str; 6] = [
    "message_start",
    "message_delta",
    "message_stop",
    "content_block_start",
    "content_block_delta",
    "content_block_stop",
];

/// The `anthropic-messages` chat API.
#[derive(Clone, Debug, Default)]
pub struct AnthropicMessagesApi {
    config: HttpClientConfig,
}

impl AnthropicMessagesApi {
    /// An adapter sending requests with `config` (proxy env, defaults).
    #[must_use]
    pub const fn new(config: HttpClientConfig) -> Self {
        Self { config }
    }
}

impl ChatApi for AnthropicMessagesApi {
    fn stream(
        &self,
        model: &Model,
        context: TranscriptContext,
        options: StreamOptions,
    ) -> AssistantStream {
        let base = initial_message(model);
        let model = model.clone();
        let http = HttpClient::new(request_config(&self.config, &options));
        AssistantStream::spawn(base, STREAM_CAPACITY, move |mut sender| async move {
            let signal = options.signal.clone().unwrap_or_default();
            let outcome = run(&http, &model, &context, &options, &signal, &mut sender).await;
            finish(sender, outcome, signal.is_cancelled()).await;
        })
    }
}

/// The `output` literal upstream starts every response from.
fn initial_message(model: &Model) -> AssistantMessage {
    AssistantMessage {
        content: Vec::new(),
        api: model.api.clone(),
        provider: model.provider.clone(),
        model: model.id.clone(),
        usage: Usage::default(),
        stop_reason: StopReason::Pending,
        timestamp: wall_ms(),
        provider_thinking_level: None,
        response_id: None,
        response_model: None,
        raw_stop_reason: None,
        error_message: None,
        diagnostics: None,
        deferred: None,
        end_turn: None,
        duration_ms: None,
        thinking_level: None,
    }
}

/// The per-request client config: request retry options override the
/// adapter defaults like upstream's `retryProviderRequest` options.
fn request_config(defaults: &HttpClientConfig, options: &StreamOptions) -> HttpClientConfig {
    let mut config = defaults.clone();
    if let Some(max_retries) = options.max_retries {
        config.max_retries = max_retries;
    }
    if options.max_retry_delay_ms.is_some() {
        config.max_retry_delay_ms = options.max_retry_delay_ms;
    }
    if let Some(timeout_ms) = options.timeout_ms {
        config.timeout = std::time::Duration::from_millis(timeout_ms);
    }
    config
}

/// Emit the terminal event for `outcome` (upstream's `done` or the `catch`
/// branch: aborted when the signal fired, error otherwise).
async fn finish(sender: AssistantStreamSender, outcome: Result<DoneReason, String>, aborted: bool) {
    let sent = match outcome {
        Ok(reason) => sender.done(reason).await,
        Err(message) => {
            let reason = if aborted {
                ErrorReason::Aborted
            } else {
                ErrorReason::Error
            };
            sender.error(reason, message).await
        }
    };
    // A closed stream means the consumer dropped it; nothing is left to notify.
    sent.ok();
}

/// Send the request and fold the response events; the error string is the
/// message upstream's `catch` records.
async fn run(
    http: &HttpClient,
    model: &Model,
    context: &TranscriptContext,
    options: &StreamOptions,
    signal: &CancellationToken,
    sender: &mut AssistantStreamSender,
) -> Result<DoneReason, String> {
    let oauth = is_oauth_token(options.api_key.as_deref().unwrap_or_default());
    let request_context = Context {
        system_prompt: None,
        messages: context.messages().to_vec(),
        tools: None,
    };
    let anthropic_options = AnthropicOptions {
        base: options.clone(),
        ..AnthropicOptions::default()
    };
    let (url, headers, body) = build_request(model, &request_context, Some(&anthropic_options));
    let url = format!("{}/v1/messages?beta=true", url.trim_end_matches('/'));
    let header_pairs: Vec<(&str, &str)> = headers
        .iter()
        .filter_map(|(name, value)| value.as_str().map(|v| (name.as_str(), v)))
        .collect();
    let response = http
        .post_json(&url, &Bytes::from(body), &header_pairs, signal, &jitter)
        .await
        .map_err(|error| match error {
            SendError::Provider(error) => sdk_error_message(&error),
            SendError::Transport(message) => message,
            SendError::Aborted => String::from("Request was aborted"),
        })?;
    if let Some(on_response) = &options.on_response {
        let metadata = ResponseMetadata {
            status: response.status().as_u16(),
            headers: headers_to_record(response.headers()),
        };
        on_response(&metadata, model).await;
    }
    sender.start().await.map_err(|_| closed())?;

    let tools = get_current_tools(context.messages());
    let mut state = EventState::new(model, oauth, &tools);
    let mut events = SseEventStream::new(response.bytes_stream(), EndOfStream::Flush);
    let mut gate = Gate::default();
    loop {
        let next = tokio::select! {
            () = signal.cancelled() => return Err(String::from("Request was aborted")),
            next = events.next() => next,
        };
        let Some(next) = next else { break };
        let sse = next.map_err(|error| error.to_string())?;
        if let Some(event) = gate.admit(&sse)? {
            state.apply(&event, sender).await?;
        }
    }
    gate.finish()?;
    if signal.is_cancelled() {
        return Err(String::from("Request was aborted"));
    }
    state.outcome()
}

/// The Anthropic SDK's `APIError` message (`makeMessage`): status plus the
/// body's `message`, else the re-serialized JSON body, else the raw text.
fn sdk_error_message(error: &HttpError) -> String {
    let Some(body) = error.body.as_deref() else {
        return error.message.clone();
    };
    let message = crate::json::from_str::<Value>(body).map_or_else(
        |_| body.to_owned(),
        |parsed| match parsed.get("message") {
            Some(Value::String(text)) if !text.is_empty() => text.clone(),
            Some(other) if !other.is_null() && other != &Value::Bool(false) => {
                crate::json::to_string(other).unwrap_or_default()
            }
            _ => crate::json::to_string(&parsed).unwrap_or_default(),
        },
    );
    match (error.status, message.is_empty()) {
        (Some(status), false) => format!("{status} {message}"),
        (Some(status), true) => format!("{status} status code (no body)"),
        (None, _) => message,
    }
}

/// The consumer is gone; the producer stops.
fn closed() -> String {
    String::from("stream consumer closed")
}

/// `headersToRecord`: response headers with lowercase names.
fn headers_to_record(headers: &reqwest::header::HeaderMap) -> Map<String, Value> {
    headers
        .iter()
        .filter_map(|(name, value)| {
            value
                .to_str()
                .ok()
                .map(|v| (name.as_str().to_owned(), Value::from(v)))
        })
        .collect()
}

/// `iterateAnthropicEvents`: SSE records to Anthropic events.
#[derive(Debug, Default)]
struct Gate {
    saw_start: bool,
    saw_stop: bool,
}

impl Gate {
    /// The parsed event for an admitted record, `None` for skipped ones.
    fn admit(&mut self, sse: &SseEvent) -> Result<Option<Value>, String> {
        let name = sse.event.as_deref().unwrap_or_default();
        if name == "error" {
            return Err(sse.data.clone());
        }
        if !ANTHROPIC_MESSAGE_EVENTS.contains(&name) {
            return Ok(None);
        }
        let event = parse_json_with_repair(&sse.data).map_err(|error| {
            format!(
                "Could not parse Anthropic SSE event {name}: {error}; data={}; raw={}",
                sse.data,
                sse.raw.join("\\n"),
            )
        })?;
        match event.get("type").and_then(Value::as_str) {
            Some("message_start") => self.saw_start = true,
            Some("message_stop") => self.saw_stop = true,
            _ => {}
        }
        Ok(Some(event))
    }

    /// The end-of-stream check upstream runs after the last record.
    fn finish(&self) -> Result<(), String> {
        if self.saw_start && !self.saw_stop {
            return Err(String::from("Anthropic stream ended before message_stop"));
        }
        Ok(())
    }
}

/// A content block still receiving deltas, keyed by Anthropic's block index.
#[derive(Debug)]
enum OpenBlock {
    Text {
        index: usize,
        text: String,
    },
    Thinking {
        index: usize,
        block: ThinkingContent,
    },
    ToolCall {
        index: usize,
    },
}

/// The response state upstream keeps on `output` plus the open blocks.
struct EventState<'a> {
    model: &'a Model,
    usage_model: Model,
    oauth: bool,
    tools: &'a [Tool],
    usage: Usage,
    stop_reason: StopReason,
    error_message: Option<String>,
    content_len: usize,
    open: HashMap<u64, OpenBlock>,
}

impl<'a> EventState<'a> {
    fn new(model: &'a Model, oauth: bool, tools: &'a [Tool]) -> Self {
        Self {
            model,
            usage_model: model.clone(),
            oauth,
            tools,
            usage: Usage::default(),
            stop_reason: StopReason::Pending,
            error_message: None,
            content_len: 0,
            open: HashMap::new(),
        }
    }

    /// Fold one Anthropic event into the stream.
    async fn apply(
        &mut self,
        event: &Value,
        sender: &mut AssistantStreamSender,
    ) -> Result<(), String> {
        match event.get("type").and_then(Value::as_str) {
            Some("message_start") => self.message_start(event, sender).await,
            Some("content_block_start") => self.block_start(event, sender).await,
            Some("content_block_delta") => self.block_delta(event, sender).await,
            Some("content_block_stop") => self.block_stop(event, sender).await,
            Some("message_delta") => self.message_delta(event, sender).await,
            _ => Ok(()),
        }
    }

    async fn message_start(
        &mut self,
        event: &Value,
        sender: &AssistantStreamSender,
    ) -> Result<(), String> {
        let message = event.get("message").unwrap_or(&Value::Null);
        if let Some(id) = message.get("id").and_then(Value::as_str) {
            send(sender, Metadata::ResponseId(id.to_owned())).await?;
        }
        let response_model = message.get("model").and_then(Value::as_str);
        if let Some(response_model) = response_model.filter(|m| *m != self.model.id) {
            send(sender, Metadata::ResponseModel(response_model.to_owned())).await?;
            if let Some(cost) = fallback_cost(self.model, response_model) {
                response_model.clone_into(&mut self.usage_model.id);
                self.usage_model.cost = cost;
            }
        }
        let usage = message.get("usage").unwrap_or(&Value::Null);
        self.usage.input = token(usage, "input_tokens").unwrap_or(0);
        self.usage.output = token(usage, "output_tokens").unwrap_or(0);
        self.usage.cache_read = token(usage, "cache_read_input_tokens").unwrap_or(0);
        self.usage.cache_write = token(usage, "cache_creation_input_tokens").unwrap_or(0);
        self.usage.cache_write1h = Some(cache_write_1h(usage).unwrap_or(0));
        self.update_usage(sender).await
    }

    async fn block_start(
        &mut self,
        event: &Value,
        sender: &mut AssistantStreamSender,
    ) -> Result<(), String> {
        let block = event.get("content_block").unwrap_or(&Value::Null);
        let key = event.get("index").and_then(Value::as_u64).unwrap_or(0);
        let open = match block.get("type").and_then(Value::as_str) {
            Some("fallback") => {
                if self.content_len > 0 {
                    return Err(String::from(
                        "Anthropic performed an unsupported mid-output model fallback",
                    ));
                }
                return Ok(());
            }
            Some("text") => {
                let index = sender.text_start().await.map_err(|_| closed())?;
                OpenBlock::Text {
                    index,
                    text: string_field(block, "text"),
                }
            }
            Some("thinking") => {
                let index = sender.thinking_start().await.map_err(|_| closed())?;
                OpenBlock::Thinking {
                    index,
                    block: ThinkingContent {
                        thinking: string_field(block, "thinking"),
                        thinking_signature: Some(string_field(block, "signature")),
                        redacted: None,
                    },
                }
            }
            Some("redacted_thinking") => {
                let index = sender.thinking_start().await.map_err(|_| closed())?;
                OpenBlock::Thinking {
                    index,
                    block: ThinkingContent {
                        thinking: String::from("[Reasoning redacted]"),
                        thinking_signature: block
                            .get("data")
                            .and_then(Value::as_str)
                            .map(str::to_owned),
                        redacted: Some(true),
                    },
                }
            }
            Some("tool_use") => {
                let name = string_field(block, "name");
                let call = ToolCall {
                    id: string_field(block, "id"),
                    name: if self.oauth {
                        from_claude_code_name(&name, self.tools)
                    } else {
                        name
                    },
                    arguments: block
                        .get("input")
                        .and_then(Value::as_object)
                        .cloned()
                        .unwrap_or_default(),
                    thought_signature: None,
                    namespace: None,
                    arguments_error: None,
                };
                let index = sender.toolcall_start(call).await.map_err(|_| closed())?;
                OpenBlock::ToolCall { index }
            }
            _ => return Ok(()),
        };
        self.content_len += 1;
        self.open.insert(key, open);
        Ok(())
    }

    async fn block_delta(
        &mut self,
        event: &Value,
        sender: &mut AssistantStreamSender,
    ) -> Result<(), String> {
        let key = event.get("index").and_then(Value::as_u64).unwrap_or(0);
        let delta = event.get("delta").unwrap_or(&Value::Null);
        let Some(open) = self.open.get_mut(&key) else {
            return Ok(());
        };
        let sent = match (delta.get("type").and_then(Value::as_str), open) {
            (Some("text_delta"), OpenBlock::Text { index, text }) => {
                let piece = string_field(delta, "text");
                text.push_str(&piece);
                sender.text_delta(*index, &piece).await
            }
            (Some("thinking_delta"), OpenBlock::Thinking { index, block }) => {
                let piece = string_field(delta, "thinking");
                block.thinking.push_str(&piece);
                sender.thinking_delta(*index, &piece).await
            }
            (Some("input_json_delta"), OpenBlock::ToolCall { index }) => {
                sender
                    .toolcall_delta(*index, &string_field(delta, "partial_json"))
                    .await
            }
            (Some("signature_delta"), OpenBlock::Thinking { block, .. }) => {
                block
                    .thinking_signature
                    .get_or_insert_with(String::new)
                    .push_str(&string_field(delta, "signature"));
                Ok(())
            }
            _ => Ok(()),
        };
        sent.map_err(|_| closed())
    }

    async fn block_stop(
        &mut self,
        event: &Value,
        sender: &mut AssistantStreamSender,
    ) -> Result<(), String> {
        let key = event.get("index").and_then(Value::as_u64).unwrap_or(0);
        let sent = match self.open.remove(&key) {
            Some(OpenBlock::Text { index, text }) => {
                let block = TextContent {
                    text,
                    text_signature: None,
                };
                sender.text_end(index, block).await
            }
            Some(OpenBlock::Thinking { index, block }) => sender.thinking_end(index, block).await,
            Some(OpenBlock::ToolCall { index }) => sender.toolcall_end(index).await,
            None => Ok(()),
        };
        sent.map_err(|_| closed())
    }

    async fn message_delta(
        &mut self,
        event: &Value,
        sender: &AssistantStreamSender,
    ) -> Result<(), String> {
        let delta = event.get("delta").unwrap_or(&Value::Null);
        if let Some(reason) = delta
            .get("stop_reason")
            .and_then(Value::as_str)
            .filter(|r| !r.is_empty())
        {
            send(sender, Metadata::RawStopReason(reason.to_owned())).await?;
            let (stop_reason, error_message) = map_stop_reason(reason, delta.get("stop_details"))?;
            self.stop_reason = stop_reason;
            if error_message.is_some() {
                self.error_message = error_message;
            }
        }
        if let Some(usage) = event.get("usage").filter(|u| u.is_object()) {
            if let Some(n) = token(usage, "input_tokens") {
                self.usage.input = n;
            }
            if let Some(n) = token(usage, "output_tokens") {
                self.usage.output = n;
            }
            if let Some(n) = token(usage, "cache_read_input_tokens") {
                self.usage.cache_read = n;
            }
            if let Some(n) = token(usage, "cache_creation_input_tokens") {
                self.usage.cache_write = n;
            }
            if let Some(n) = cache_write_1h(usage) {
                self.usage.cache_write1h = Some(n);
            }
            if let Some(n) = usage
                .get("output_tokens_details")
                .and_then(|details| token(details, "thinking_tokens"))
            {
                self.usage.reasoning = Some(n);
            }
        }
        self.update_usage(sender).await
    }

    /// Recompute the total and cost, then publish the usage.
    async fn update_usage(&mut self, sender: &AssistantStreamSender) -> Result<(), String> {
        self.usage.total_tokens =
            self.usage.input + self.usage.output + self.usage.cache_read + self.usage.cache_write;
        calculate_cost(&self.usage_model, &mut self.usage);
        send(sender, Metadata::Usage(self.usage.clone())).await
    }

    /// The terminal decision upstream makes after the event loop.
    fn outcome(&self) -> Result<DoneReason, String> {
        match self.stop_reason {
            StopReason::Pending => {
                Err(String::from("Anthropic stream ended without a stop reason"))
            }
            StopReason::Aborted | StopReason::Error => Err(self
                .error_message
                .clone()
                .unwrap_or_else(|| String::from("An unknown error occurred"))),
            StopReason::Stop => Ok(DoneReason::Stop),
            StopReason::Length => Ok(DoneReason::Length),
            StopReason::ToolUse => Ok(DoneReason::ToolUse),
            StopReason::Deferred => Ok(DoneReason::Deferred),
        }
    }
}

/// Publish one metadata update.
async fn send(sender: &AssistantStreamSender, metadata: Metadata) -> Result<(), String> {
    sender.metadata(metadata).await.map_err(|_| closed())
}

/// `mapStopReason`: Anthropic stop reason to pi's, with the error text for
/// refusals and safety stops.
fn map_stop_reason(
    reason: &str,
    details: Option<&Value>,
) -> Result<(StopReason, Option<String>), String> {
    match reason {
        "end_turn" | "pause_turn" | "stop_sequence" => Ok((StopReason::Stop, None)),
        "max_tokens" => Ok((StopReason::Length, None)),
        "tool_use" => Ok((StopReason::ToolUse, None)),
        "refusal" => {
            let explanation = details
                .and_then(|d| d.get("explanation"))
                .and_then(Value::as_str)
                .filter(|text| !text.is_empty())
                .unwrap_or("The model refused to complete the request");
            Ok((StopReason::Error, Some(explanation.to_owned())))
        }
        "sensitive" => Ok((
            StopReason::Error,
            Some(String::from("Provider stopped with: sensitive")),
        )),
        other => Err(format!("Unhandled stop reason: {other}")),
    }
}

/// The catalog cost of an allowed fallback model the provider answered with.
fn fallback_cost(model: &Model, response_model: &str) -> Option<crate::types::model::ModelCost> {
    let fallbacks = model
        .compat
        .as_ref()?
        .get("allowedFallbackModels")?
        .as_array()?;
    let entry = fallbacks.iter().find(|fallback| {
        fallback.get("provider").and_then(Value::as_str) == Some(model.provider.as_str())
            && fallback.get("model").and_then(Value::as_str) == Some(response_model)
    })?;
    serde_json::from_value(entry.get("cost")?.clone()).ok()
}

/// A non-null token count; JS `|| 0` and `!= null` both treat absence alike.
fn token(usage: &Value, name: &str) -> Option<u64> {
    usage.get(name).and_then(Value::as_u64)
}

/// `cache_creation.ephemeral_1h_input_tokens`.
fn cache_write_1h(usage: &Value) -> Option<u64> {
    usage
        .get("cache_creation")
        .and_then(|creation| token(creation, "ephemeral_1h_input_tokens"))
}

/// A string field, `""` when absent (upstream's `?? ""`).
fn string_field(value: &Value, name: &str) -> String {
    value
        .get(name)
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_owned()
}
