//! Streaming side of the openai-completions adapter (`stream` in
//! `openai-completions.ts`) over the `OpenAI` SDK's chunk iteration.
//!
//! The SDK delivers a record left pending at end of stream, which is
//! [`EndOfStream::Flush`]; `[DONE]` ends the iteration.
//!
//! Not yet ported: `onPayload` and `onProviderStreamEvent`.

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use bytes::Bytes;
use futures::StreamExt;
use serde_json::{Map, Value};
use tokio_util::sync::CancellationToken;

use super::{OpenAiCompletionsCompat, OpenAiCompletionsOptions, build_request, is_truthy};
use crate::api::constrained_sampling::GrammarToolInputJsonBuffer;
use crate::api::stream_support::{
    STREAM_CAPACITY, closed, finish, header_pairs, initial_message, notify_response,
    request_policy, send,
};
use crate::cost::{apply_openrouter_reported_cost, calculate_cost, is_openrouter_model};
use crate::http::client::{HttpClient, HttpError, SendError};
use crate::http::retry::jitter;
use crate::http::sse::{EndOfStream, SseEventStream};
use crate::options::{SimpleStreamOptions, StreamOptions};
use crate::provider::ChatApi;
use crate::stream::{AssistantStream, AssistantStreamSender, DoneReason, Metadata};
use crate::transcript::TranscriptContext;
use crate::types::content::{
    AssistantContentBlock, StopReason, TextContent, ThinkingContent, ToolCall,
};
use crate::types::model::Model;
use crate::types::usage::Usage;

/// The `openai-completions` chat API.
#[derive(Clone, Debug)]
pub struct OpenAiCompletionsApi {
    http: Arc<HttpClient>,
}

impl OpenAiCompletionsApi {
    /// An adapter sending requests through the shared `http` pool.
    #[must_use]
    pub const fn new(http: Arc<HttpClient>) -> Self {
        Self { http }
    }

    /// `stream` with the Chat Completions options.
    #[must_use]
    pub fn stream_with(
        &self,
        model: &Model,
        context: TranscriptContext,
        options: OpenAiCompletionsOptions,
    ) -> AssistantStream {
        let base = initial_message(model);
        let model = model.clone();
        let http = Arc::clone(&self.http);
        AssistantStream::spawn(base, STREAM_CAPACITY, move |mut sender| async move {
            let signal = options.base.signal.clone().unwrap_or_default();
            let outcome = run(&http, &model, &context, &options, &signal, &mut sender).await;
            finish(sender, outcome, signal.is_cancelled()).await;
        })
    }
}

impl ChatApi for OpenAiCompletionsApi {
    fn stream(
        &self,
        model: &Model,
        context: TranscriptContext,
        options: StreamOptions,
    ) -> AssistantStream {
        let options = OpenAiCompletionsOptions {
            base: options,
            ..OpenAiCompletionsOptions::default()
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
        let options = OpenAiCompletionsOptions::from_simple(model, &context, &options);
        self.stream_with(model, context, options)
    }
}

/// Send the request and fold the chunks; the error string is the message
/// upstream's `catch` records.
async fn run(
    http: &HttpClient,
    model: &Model,
    context: &TranscriptContext,
    options: &OpenAiCompletionsOptions,
    signal: &CancellationToken,
    sender: &mut AssistantStreamSender,
) -> Result<DoneReason, String> {
    let base = &options.base;
    let request = build_request(model, context, options)?;
    let sent = http
        .post_json(
            &request.url,
            &Bytes::from(request.body),
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
        Err(SendError::Provider(error)) => return Err(status_error_message(&error)),
        Err(SendError::Transport(message)) => return Err(message),
        Err(SendError::Aborted) => return Err(String::from("Request aborted")),
    };
    notify_response(base, &response, model).await;
    sender.start().await.map_err(|_| closed())?;

    let compat = OpenAiCompletionsCompat::of(model);
    let mut state = ChunkState::new(model, &request.grammar_tool_input_properties);
    let outcome = fold(&mut state, response, signal, sender, &compat).await;
    if outcome.is_err() {
        state.close_thinking_on_error(sender).await?;
    }
    outcome
}

/// The chunk loop and upstream's checks after it.
async fn fold(
    state: &mut ChunkState<'_>,
    response: reqwest::Response,
    signal: &CancellationToken,
    sender: &mut AssistantStreamSender,
    compat: &OpenAiCompletionsCompat,
) -> Result<DoneReason, String> {
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
        if sse.data == "[DONE]" {
            break;
        }
        let data: Value = crate::json::from_str(&sse.data).map_err(|_| {
            String::from("Error reading response: malformed server-sent event JSON.")
        })?;
        if sse
            .event
            .as_deref()
            .is_some_and(|event| event.starts_with("thread."))
        {
            // The SDK yields `{event, data}` envelopes, which carry no choices.
            continue;
        }
        let error = if sse.event.as_deref() == Some("error") {
            Some(data.get("error").filter(|e| !e.is_null()).unwrap_or(&data))
        } else {
            data.get("error").filter(|error| is_truthy(error))
        };
        if let Some(error) = error {
            return Err(stream_error_message(error));
        }
        state.apply(&data, sender).await?;
    }
    state.finish_blocks(sender).await?;
    if signal.is_cancelled() {
        return Err(String::from("Request was aborted"));
    }
    if !state.has_finish_reason && !compat.supports_finish_reason {
        state.stop_reason = if state.has_tool_calls() {
            StopReason::ToolUse
        } else {
            StopReason::Stop
        };
    }
    if state.stop_reason == StopReason::Error {
        return Err(state
            .error_message
            .clone()
            .unwrap_or_else(|| String::from("Provider returned an error stop reason")));
    }
    if (compat.supports_finish_reason && !state.has_finish_reason)
        || state.stop_reason == StopReason::Pending
    {
        return Err(String::from("Stream ended without finish_reason"));
    }
    match state.stop_reason {
        StopReason::Length => Ok(DoneReason::Length),
        StopReason::ToolUse => Ok(DoneReason::ToolUse),
        _ => Ok(DoneReason::Stop),
    }
}

/// An HTTP error as the `OpenAI` SDK throws it (`makeStatusError` from the
/// body text), rendered by pi's `catch`. Without a response the SDK throws
/// its connection error; timeouts are not told apart from it.
fn status_error_message(error: &HttpError) -> String {
    let Some(status) = error.status.filter(|status| *status != 0.0) else {
        return String::from("Connection error.");
    };
    let text = error.body.as_deref().unwrap_or_default();
    let parsed = crate::json::from_str::<Value>(text).ok().filter(is_truthy);
    // `makeStatusError` treats a JSON body without an `error` field as the
    // error itself.
    let (error_field, message) = parsed.as_ref().map_or((None, Some(text)), |json| {
        let field = match json.get("error") {
            None | Some(Value::Null) if json.is_object() || json.is_array() => Some(json),
            field => field,
        };
        (field, None)
    });
    catch_message(
        &api_error_message(Some(status), error_field, message),
        Some(status),
        error_field,
    )
}

/// A stream `error` object, thrown as `APIError(undefined, error)`.
fn stream_error_message(error: &Value) -> String {
    catch_message(
        &api_error_message(None, Some(error), None),
        None,
        Some(error),
    )
}

/// `APIError.makeMessage`.
fn api_error_message(status: Option<f64>, error: Option<&Value>, message: Option<&str>) -> String {
    let error = error.filter(|error| is_truthy(error));
    let msg = error.map_or_else(
        || message.unwrap_or_default().to_owned(),
        |error| match error.get("message").filter(|m| is_truthy(m)) {
            Some(Value::String(text)) => text.clone(),
            Some(other) => crate::json::to_string(other).unwrap_or_default(),
            None => crate::json::to_string(error).unwrap_or_default(),
        },
    );
    match (status, msg.is_empty()) {
        (Some(status), false) => format!("{status} {msg}"),
        (Some(status), true) => format!("{status} status code (no body)"),
        (None, false) => msg,
        (None, true) => String::from("(no status code or body)"),
    }
}

/// pi's `catch`: `formatProviderError(normalizeProviderError(error))`, then
/// `error.error.metadata.raw` unless the message already shows it.
fn catch_message(message: &str, status: Option<f64>, error: Option<&Value>) -> String {
    let fields = serde_json::json!({
        "status": status,
        "error": error.cloned().unwrap_or(Value::Null),
    });
    let normalized = crate::http::error_body::normalize_provider_error(message, Some(&fields));
    let mut rendered = crate::http::error_body::format_provider_error(&normalized, None);
    let raw = error
        .and_then(|error| error.get("metadata"))
        .and_then(|metadata| metadata.get("raw"))
        .filter(|raw| is_truthy(raw))
        .map(js_string);
    if let Some(raw) = raw.filter(|raw| !rendered.contains(raw.as_str())) {
        rendered.push('\n');
        rendered.push_str(&raw);
    }
    rendered
}

/// JavaScript `String(value)` for a JSON value.
fn js_string(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        Value::Object(_) => String::from("[object Object]"),
        Value::Array(items) => items
            .iter()
            .map(|item| {
                if item.is_null() {
                    String::new()
                } else {
                    js_string(item)
                }
            })
            .collect::<Vec<_>>()
            .join(","),
        other => crate::json::to_string(other).unwrap_or_default(),
    }
}

/// The reasoning fields a delta may carry, in upstream's lookup order.
const REASONING_FIELDS: [&str; 3] = ["reasoning_content", "reasoning", "reasoning_text"];

/// A content block in content order.
#[derive(Clone, Copy, Debug)]
enum Slot {
    Text,
    Thinking,
    ToolCall(usize),
}

/// The response's single text block.
#[derive(Debug)]
struct OpenText {
    index: usize,
    text: String,
}

/// The response's single thinking block.
#[derive(Debug)]
struct OpenThinking {
    index: usize,
    thinking: String,
    signature: String,
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
    },
}

/// One streamed tool call.
#[derive(Debug)]
struct OpenToolCall {
    index: usize,
    stream_index: Option<u64>,
    id: String,
    name: String,
    input: ToolInput,
}

/// Upstream's `output` fields plus the streaming blocks.
struct ChunkState<'a> {
    model: &'a Model,
    grammar_properties: &'a BTreeMap<String, String>,
    response_id: Option<String>,
    response_model: Option<String>,
    stop_reason: StopReason,
    error_message: Option<String>,
    has_finish_reason: bool,
    blocks: Vec<Slot>,
    closed: bool,
    text: Option<OpenText>,
    thinking: Option<OpenThinking>,
    details: Option<Vec<ReasoningDetail>>,
    tool_calls: Vec<OpenToolCall>,
    by_index: HashMap<u64, usize>,
    by_id: HashMap<String, usize>,
}

impl<'a> ChunkState<'a> {
    fn new(model: &'a Model, grammar_properties: &'a BTreeMap<String, String>) -> Self {
        Self {
            model,
            grammar_properties,
            response_id: None,
            response_model: None,
            stop_reason: StopReason::Pending,
            error_message: None,
            has_finish_reason: false,
            blocks: Vec::new(),
            closed: false,
            text: None,
            thinking: None,
            details: None,
            tool_calls: Vec::new(),
            by_index: HashMap::new(),
            by_id: HashMap::new(),
        }
    }

    const fn has_tool_calls(&self) -> bool {
        !self.tool_calls.is_empty()
    }

    /// Fold one chunk.
    async fn apply(
        &mut self,
        chunk: &Value,
        sender: &mut AssistantStreamSender,
    ) -> Result<(), String> {
        if !chunk.is_object() {
            return Ok(());
        }
        self.response_fields(chunk, sender).await?;
        let chunk_usage = chunk.get("usage").filter(|usage| is_truthy(usage));
        if let Some(usage) = chunk_usage {
            send(
                sender,
                Metadata::Usage(parse_chunk_usage(usage, self.model)),
            )
            .await?;
        }
        let Some(choice) = chunk
            .get("choices")
            .and_then(Value::as_array)
            .and_then(|choices| choices.first())
            .filter(|choice| is_truthy(choice))
        else {
            return Ok(());
        };
        // Some providers (Moonshot) report usage on the choice instead.
        if chunk_usage.is_none()
            && let Some(usage) = choice.get("usage").filter(|usage| is_truthy(usage))
        {
            send(
                sender,
                Metadata::Usage(parse_chunk_usage(usage, self.model)),
            )
            .await?;
        }
        if let Some(reason) = choice
            .get("finish_reason")
            .and_then(Value::as_str)
            .filter(|reason| !reason.is_empty())
        {
            send(sender, Metadata::RawStopReason(reason.to_owned())).await?;
            let (stop_reason, error_message) = map_stop_reason(reason);
            self.stop_reason = stop_reason;
            if error_message.is_some() {
                self.error_message = error_message;
            }
            self.has_finish_reason = true;
        }
        if let Some(delta) = choice.get("delta").filter(|delta| is_truthy(delta)) {
            self.delta(delta, sender).await?;
        }
        Ok(())
    }

    /// `responseId ||= chunk.id` and `responseModel ||= chunk.model`.
    async fn response_fields(
        &mut self,
        chunk: &Value,
        sender: &AssistantStreamSender,
    ) -> Result<(), String> {
        if self.response_id.as_deref().is_none_or(str::is_empty)
            && let Some(id) = chunk.get("id").and_then(Value::as_str)
        {
            self.response_id = Some(id.to_owned());
            send(sender, Metadata::ResponseId(id.to_owned())).await?;
        }
        if self.response_model.is_none()
            && let Some(response_model) = chunk
                .get("model")
                .and_then(Value::as_str)
                .filter(|m| !m.is_empty() && *m != self.model.id)
        {
            self.response_model = Some(response_model.to_owned());
            send(sender, Metadata::ResponseModel(response_model.to_owned())).await?;
        }
        Ok(())
    }

    /// One delta: reasoning before content, since some proxies (`LiteLLM`)
    /// put both in one delta and the thinking block must come first.
    async fn delta(
        &mut self,
        delta: &Value,
        sender: &mut AssistantStreamSender,
    ) -> Result<(), String> {
        // The first non-empty field only: some providers (chutes.ai) send the
        // same text in two of them.
        let reasoning = REASONING_FIELDS.iter().find_map(|field| {
            delta
                .get(*field)
                .and_then(Value::as_str)
                .filter(|text| !text.is_empty())
                .map(|text| (*field, text))
        });
        if let Some((field, text)) = reasoning {
            let signature = if self.model.provider.as_str() == "opencode-go" && field == "reasoning"
            {
                "reasoning_content"
            } else {
                field
            };
            let index = self.ensure_thinking(signature, sender).await?;
            if let Some(thinking) = &mut self.thinking {
                thinking.thinking.push_str(text);
            }
            sender
                .thinking_delta(index, text)
                .await
                .map_err(|_| closed())?;
        }
        if let Some(details) = delta.get("reasoning_details").and_then(Value::as_array) {
            for detail in details.iter().filter_map(ReasoningDetail::parse) {
                self.ensure_thinking("", sender).await?;
                // Replay data rides in the signature: text and summary deltas
                // merge into logical entries, encrypted entries stay discrete.
                append_reasoning_detail(self.details.get_or_insert_with(Vec::new), detail);
            }
        }
        if let Some(content) = delta
            .get("content")
            .and_then(Value::as_str)
            .filter(|content| !content.is_empty())
        {
            let index = self.ensure_text(sender).await?;
            if let Some(text) = &mut self.text {
                text.text.push_str(content);
            }
            sender
                .text_delta(index, content)
                .await
                .map_err(|_| closed())?;
        }
        if let Some(tool_calls) = delta.get("tool_calls").filter(|calls| is_truthy(calls)) {
            for tool_call in tool_calls
                .as_array()
                .into_iter()
                .flatten()
                .filter(|c| c.is_object())
            {
                self.tool_call_delta(tool_call, sender).await?;
            }
        }
        Ok(())
    }

    async fn ensure_text(&mut self, sender: &mut AssistantStreamSender) -> Result<usize, String> {
        if let Some(text) = &self.text {
            return Ok(text.index);
        }
        let index = sender.text_start().await.map_err(|_| closed())?;
        self.text = Some(OpenText {
            index,
            text: String::new(),
        });
        self.blocks.push(Slot::Text);
        Ok(index)
    }

    async fn ensure_thinking(
        &mut self,
        signature: &str,
        sender: &mut AssistantStreamSender,
    ) -> Result<usize, String> {
        if let Some(thinking) = &self.thinking {
            return Ok(thinking.index);
        }
        let index = sender.thinking_start().await.map_err(|_| closed())?;
        let block = ThinkingContent {
            thinking: String::new(),
            thinking_signature: Some(signature.to_owned()),
            redacted: None,
        };
        sender
            .patch(index, AssistantContentBlock::Thinking(block))
            .await
            .map_err(|_| closed())?;
        self.thinking = Some(OpenThinking {
            index,
            thinking: String::new(),
            signature: signature.to_owned(),
        });
        self.blocks.push(Slot::Thinking);
        Ok(index)
    }

    /// One `tool_calls` entry: find or open its block, then stream its
    /// arguments; a `toolcall_delta` is sent for every entry.
    async fn tool_call_delta(
        &mut self,
        tool_call: &Value,
        sender: &mut AssistantStreamSender,
    ) -> Result<(), String> {
        let function = tool_call.get("function").filter(|f| is_truthy(f));
        let custom = tool_call.get("custom").filter(|c| is_truthy(c));
        let custom_only = custom.is_some() && function.is_none();
        let name = function
            .and_then(|f| f.get("name"))
            .filter(|name| !name.is_null())
            .or_else(|| custom.and_then(|c| c.get("name")))
            .and_then(Value::as_str)
            .unwrap_or_default();
        let id = tool_call
            .get("id")
            .and_then(Value::as_str)
            .filter(|id| !id.is_empty());
        let stream_index = tool_call.get("index").and_then(Value::as_u64);
        let position = self
            .ensure_tool_call(stream_index, id, name, custom_only, sender)
            .await?;

        let Some(call) = self.tool_calls.get_mut(position) else {
            return Ok(());
        };
        let delta = match &call.input {
            ToolInput::Json => function
                .and_then(|f| f.get("arguments"))
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_owned(),
            ToolInput::Custom { .. } => {
                let input = custom
                    .and_then(|c| c.get("input"))
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                let piece = call.append_custom(input)?;
                emit_custom_piece(sender, call, piece)
                    .await?
                    .unwrap_or_default()
            }
        };
        sender
            .toolcall_delta(call.index, &delta)
            .await
            .map_err(|_| closed())
    }

    /// `ensureToolCallBlock`: the call by stream index, else by id, else a
    /// new block; late indices, ids and names are recorded.
    async fn ensure_tool_call(
        &mut self,
        stream_index: Option<u64>,
        id: Option<&str>,
        name: &str,
        custom_only: bool,
        sender: &mut AssistantStreamSender,
    ) -> Result<usize, String> {
        let found = stream_index
            .and_then(|stream_index| self.by_index.get(&stream_index))
            .or_else(|| id.and_then(|id| self.by_id.get(id)))
            .copied();
        let position = match found {
            Some(position) => position,
            None => {
                self.open_tool_call(stream_index, id, name, custom_only, sender)
                    .await?
            }
        };
        if let Some(id) = id {
            self.by_id.insert(id.to_owned(), position);
        }
        let Some(call) = self.tool_calls.get_mut(position) else {
            return Ok(position);
        };
        if let Some(stream_index) = stream_index
            && call.stream_index.is_none()
        {
            call.stream_index = Some(stream_index);
            self.by_index.insert(stream_index, position);
        }
        let late_id = id.filter(|_| call.id.is_empty());
        let late_name = (call.name.is_empty() && !name.is_empty()).then_some(name);
        if let Some(late) = late_id {
            late.clone_into(&mut call.id);
        }
        if let Some(late) = late_name {
            late.clone_into(&mut call.name);
        }
        if late_id.is_some() || late_name.is_some() {
            sender
                .toolcall_identity(call.index, &call.id, &call.name)
                .await
                .map_err(|_| closed())?;
        }
        if custom_only && matches!(call.input, ToolInput::Json) {
            let property = grammar_property(self.grammar_properties, &call.name);
            call.input = ToolInput::Custom {
                property: property.clone(),
                buffer: GrammarToolInputJsonBuffer::default(),
            };
            let mut arguments = Map::new();
            arguments.insert(property, Value::from(""));
            let block = ToolCall {
                arguments,
                ..tool_call_block(call)
            };
            sender
                .patch(call.index, AssistantContentBlock::ToolCall(block))
                .await
                .map_err(|_| closed())?;
        }
        Ok(position)
    }

    async fn open_tool_call(
        &mut self,
        stream_index: Option<u64>,
        id: Option<&str>,
        name: &str,
        custom_only: bool,
        sender: &mut AssistantStreamSender,
    ) -> Result<usize, String> {
        // A grammar tool's input property; `"input"` only stashes the input
        // of a tool the model made up.
        let property = custom_only.then(|| grammar_property(self.grammar_properties, name));
        let mut arguments = Map::new();
        if let Some(property) = &property {
            arguments.insert(property.clone(), Value::from(""));
        }
        let call = ToolCall {
            id: id.unwrap_or_default().to_owned(),
            name: name.to_owned(),
            arguments,
            thought_signature: None,
            namespace: None,
            arguments_error: None,
        };
        let index = sender.toolcall_start(call).await.map_err(|_| closed())?;
        let position = self.tool_calls.len();
        self.tool_calls.push(OpenToolCall {
            index,
            stream_index,
            id: id.unwrap_or_default().to_owned(),
            name: name.to_owned(),
            input: property.map_or(ToolInput::Json, |property| ToolInput::Custom {
                property,
                buffer: GrammarToolInputJsonBuffer::default(),
            }),
        });
        if let Some(stream_index) = stream_index {
            self.by_index.insert(stream_index, position);
        }
        self.blocks.push(Slot::ToolCall(position));
        Ok(position)
    }

    /// `finishBlock` for every block, in content order.
    async fn finish_blocks(&mut self, sender: &mut AssistantStreamSender) -> Result<(), String> {
        self.closed = true;
        for slot in std::mem::take(&mut self.blocks) {
            match slot {
                Slot::Text => self.finish_text(sender).await?,
                Slot::Thinking => self.finish_thinking(sender).await?,
                Slot::ToolCall(position) => self.finish_tool_call(position, sender).await?,
            }
        }
        Ok(())
    }

    async fn finish_thinking(&self, sender: &mut AssistantStreamSender) -> Result<(), String> {
        let Some((index, block)) = self.thinking_block() else {
            return Ok(());
        };
        sender
            .thinking_end(index, block)
            .await
            .map_err(|_| closed())
    }

    /// `finishBlock` for a tool call: a grammar call's closing piece, then
    /// the end event.
    async fn finish_tool_call(
        &mut self,
        position: usize,
        sender: &mut AssistantStreamSender,
    ) -> Result<(), String> {
        let Some(call) = self.tool_calls.get_mut(position) else {
            return Ok(());
        };
        let piece = call.close_custom()?;
        if let Some(delta) = emit_custom_piece(sender, call, piece).await? {
            sender
                .toolcall_delta(call.index, &delta)
                .await
                .map_err(|_| closed())?;
        }
        sender.toolcall_end(call.index).await.map_err(|_| closed())
    }

    async fn finish_text(&self, sender: &mut AssistantStreamSender) -> Result<(), String> {
        let Some(text) = &self.text else {
            return Ok(());
        };
        let block = TextContent {
            text: text.text.clone(),
            text_signature: None,
        };
        sender
            .text_end(text.index, block)
            .await
            .map_err(|_| closed())
    }

    /// The thinking block with its reasoning details serialized into the
    /// signature, as upstream writes it when the block is finalized.
    fn thinking_block(&self) -> Option<(usize, ThinkingContent)> {
        let thinking = self.thinking.as_ref()?;
        let signature = self.details.as_ref().map_or_else(
            || thinking.signature.clone(),
            |details| serialize_reasoning_details(details),
        );
        Some((
            thinking.index,
            ThinkingContent {
                thinking: thinking.thinking.clone(),
                thinking_signature: Some(signature),
                redacted: None,
            },
        ))
    }

    /// Upstream's `catch` applies the streamed reasoning details to a
    /// thinking block that never reached `thinking_end`.
    async fn close_thinking_on_error(
        &self,
        sender: &mut AssistantStreamSender,
    ) -> Result<(), String> {
        if self.closed || self.details.is_none() {
            return Ok(());
        }
        let Some((index, block)) = self.thinking_block() else {
            return Ok(());
        };
        sender
            .patch(index, AssistantContentBlock::Thinking(block))
            .await
            .map_err(|_| closed())
    }
}

/// A piece of a grammar call's JSON arguments text.
struct CustomPiece {
    text: String,
    /// The piece opens the arguments text.
    opens: bool,
}

impl OpenToolCall {
    /// `appendGrammarToolInputJsonDelta` on a grammar call.
    fn append_custom(&mut self, input: &str) -> Result<Option<CustomPiece>, String> {
        let ToolInput::Custom { property, buffer } = &mut self.input else {
            return Ok(None);
        };
        let opened = buffer.started;
        let text = buffer
            .append_delta(property, input)
            .map_err(|error| error.to_string())?;
        Ok(text.map(|text| CustomPiece {
            text,
            opens: !opened,
        }))
    }

    /// `closeGrammarToolInputJson` on a grammar call; `None` for JSON calls.
    fn close_custom(&mut self) -> Result<Option<CustomPiece>, String> {
        let ToolInput::Custom { property, buffer } = &mut self.input else {
            return Ok(None);
        };
        let opened = buffer.started;
        let text = buffer
            .close(property, None)
            .map_err(|error| error.to_string())?;
        Ok(text.map(|text| CustomPiece {
            text,
            opens: !opened,
        }))
    }
}

/// Prepare the fold for a grammar piece and return its text: before the
/// arguments text opens, the block's `{property: ""}` placeholder is
/// emptied so the folded text parses to `{property: input}` like upstream.
async fn emit_custom_piece(
    sender: &mut AssistantStreamSender,
    call: &OpenToolCall,
    piece: Option<CustomPiece>,
) -> Result<Option<String>, String> {
    let Some(piece) = piece else {
        return Ok(None);
    };
    if piece.opens {
        sender
            .patch(
                call.index,
                AssistantContentBlock::ToolCall(tool_call_block(call)),
            )
            .await
            .map_err(|_| closed())?;
    }
    Ok(Some(piece.text))
}

/// The grammar call's input property, `"input"` when the tool is unknown.
fn grammar_property(properties: &BTreeMap<String, String>, name: &str) -> String {
    properties
        .get(name)
        .cloned()
        .unwrap_or_else(|| String::from("input"))
}

/// The content block of a streamed call, with empty arguments.
fn tool_call_block(call: &OpenToolCall) -> ToolCall {
    ToolCall {
        id: call.id.clone(),
        name: call.name.clone(),
        arguments: Map::new(),
        thought_signature: None,
        namespace: None,
        arguments_error: None,
    }
}

/// `mapStopReason`: a `finish_reason` to pi's stop reason, with the error
/// text for the ones pi treats as provider errors.
fn map_stop_reason(reason: &str) -> (StopReason, Option<String>) {
    match reason {
        "stop" | "end" => (StopReason::Stop, None),
        "length" => (StopReason::Length, None),
        "function_call" | "tool_calls" => (StopReason::ToolUse, None),
        other => (
            StopReason::Error,
            Some(format!("Provider finish_reason: {other}")),
        ),
    }
}

/// `parseChunkUsage`: Chat Completions usage to pi's, priced from the
/// catalog and, on `OpenRouter`, from the billed cost.
fn parse_chunk_usage(raw: &Value, model: &Model) -> Usage {
    let count = |value: Option<&Value>| value.and_then(Value::as_u64).unwrap_or(0);
    let details = |name: &str, field: &str| raw.get(name).and_then(|details| details.get(field));
    let prompt = count(raw.get("prompt_tokens"));
    // `cached_tokens` counts cache hits; providers place it differently
    // (OpenAI and OpenRouter in `prompt_tokens_details`, DeepSeek as
    // `prompt_cache_hit_tokens`, Kimi at the top level). Writes are a
    // separate count and are not subtracted from it.
    let cache_read = count(
        [
            details("prompt_tokens_details", "cached_tokens"),
            raw.get("prompt_cache_hit_tokens"),
            raw.get("cached_tokens"),
        ]
        .into_iter()
        .flatten()
        .find(|value| !value.is_null()),
    );
    let cache_write = count(details("prompt_tokens_details", "cache_write_tokens"));
    let input = prompt
        .saturating_sub(cache_read)
        .saturating_sub(cache_write);
    // `completion_tokens` already includes the reasoning tokens.
    let output = count(raw.get("completion_tokens"));
    let mut usage = Usage {
        input,
        output,
        cache_read,
        cache_write,
        total_tokens: input + output + cache_read + cache_write,
        reasoning: Some(count(details(
            "completion_tokens_details",
            "reasoning_tokens",
        ))),
        ..Usage::default()
    };
    calculate_cost(model, &mut usage);
    if is_openrouter_model(model) {
        apply_openrouter_reported_cost(&mut usage, raw);
    }
    usage
}

/// One `reasoning_details` entry as JavaScript holds it: fields in insertion
/// order, `None` for a field assigned `undefined` (kept in place, omitted
/// from JSON).
#[derive(Clone, Debug)]
struct ReasoningDetail {
    fields: Vec<(String, Option<Value>)>,
}

impl ReasoningDetail {
    /// `isOpenAIReasoningDetail`: a plain object with valid common fields
    /// and one of the three detail types.
    fn parse(value: &Value) -> Option<Self> {
        let object = value.as_object()?;
        let field = |name: &str| object.get(name);
        let valid_common = field("id").is_none_or(|id| id.is_null() || id.is_string())
            && field("format").is_none_or(Value::is_string)
            && field("index").is_none_or(Value::is_number);
        let valid_type = match field("type").and_then(Value::as_str) {
            Some("reasoning.summary") => field("summary").is_some_and(Value::is_string),
            Some("reasoning.encrypted") => field("data").is_some_and(Value::is_string),
            Some("reasoning.text") => {
                field("text").is_some_and(Value::is_string)
                    && field("signature").is_none_or(|s| s.is_null() || s.is_string())
            }
            _ => false,
        };
        (valid_common && valid_type).then(|| Self {
            fields: object
                .iter()
                .map(|(name, entry)| (name.clone(), Some(entry.clone())))
                .collect(),
        })
    }

    fn get(&self, name: &str) -> Option<&Value> {
        self.fields
            .iter()
            .find(|(field, _)| field == name)
            .and_then(|(_, value)| value.as_ref())
    }

    fn kind(&self) -> Option<&str> {
        self.get("type").and_then(Value::as_str)
    }

    /// `target[name] = value`: an existing field keeps its position.
    fn set(&mut self, name: &str, value: Option<Value>) {
        match self.fields.iter_mut().find(|(field, _)| field == name) {
            Some((_, slot)) => *slot = value,
            None => self.fields.push((name.to_owned(), value)),
        }
    }

    /// `target[name] ??= source[name]`.
    fn fill_nullish(&mut self, source: &Self, name: &str) {
        if self.get(name).is_none_or(Value::is_null) {
            self.set(name, source.get(name).cloned());
        }
    }

    /// `target[name] ||= source[name]`.
    fn fill_falsy(&mut self, source: &Self, name: &str) {
        if !self.get(name).is_some_and(is_truthy) {
            self.set(name, source.get(name).cloned());
        }
    }

    /// `target[name] += source[name]` for string fields.
    fn concat(&mut self, source: &Self, name: &str) {
        let joined = format!(
            "{}{}",
            self.get(name).and_then(Value::as_str).unwrap_or_default(),
            source.get(name).and_then(Value::as_str).unwrap_or_default()
        );
        self.set(name, Some(Value::from(joined)));
    }

    fn to_value(&self) -> Value {
        Value::Object(
            self.fields
                .iter()
                .filter_map(|(name, value)| value.clone().map(|value| (name.clone(), value)))
                .collect(),
        )
    }
}

/// `appendOpenAIReasoningDetail`: merge consecutive text or summary deltas
/// into the last entry, else append.
fn append_reasoning_detail(details: &mut Vec<ReasoningDetail>, detail: ReasoningDetail) {
    let merged_field = match (
        details.last().and_then(ReasoningDetail::kind),
        detail.kind(),
    ) {
        (Some("reasoning.text"), Some("reasoning.text")) => "text",
        (Some("reasoning.summary"), Some("reasoning.summary")) => "summary",
        _ => {
            details.push(detail);
            return;
        }
    };
    let Some(last) = details.last_mut() else {
        return;
    };
    last.concat(&detail, merged_field);
    if merged_field == "text" {
        last.fill_falsy(&detail, "signature");
    }
    last.fill_nullish(&detail, "id");
    last.fill_falsy(&detail, "format");
    last.fill_nullish(&detail, "index");
}

/// `JSON.stringify(streamedReasoningDetails)`.
fn serialize_reasoning_details(details: &[ReasoningDetail]) -> String {
    let values: Vec<Value> = details.iter().map(ReasoningDetail::to_value).collect();
    crate::json::to_string(&values).unwrap_or_default()
}
