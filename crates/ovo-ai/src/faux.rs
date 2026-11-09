//! `providers/faux.ts` — the scripted test provider.
//!
//! Streams queued `AssistantMessage` responses through the typed
//! event API with chunked deltas, optional pacing, usage estimation,
//! per-`sessionId` prompt-cache accounting, abort handling, and
//! deferred handles. No `http` feature needed.

use std::collections::{BTreeMap, VecDeque};
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde_json::{Map, Value};

use crate::auth::resolve::ApiKeyAuth;
use crate::models_error::ModelsError;
use crate::options::{SimpleStreamOptions, StreamOptions};
use crate::provider::{
    ApiDispatch, ChatApi, Provider, ProviderAuth, ProviderSpec, create_provider,
};
use crate::stream::AssistantStream;
use crate::transcript::TranscriptContext;
use crate::types::content::{
    AssistantContentBlock, ImageContent, StopReason, TextContent, ThinkingContent, ToolCall,
    UserContentBlock,
};
use crate::types::deferred::DeferredHandle;
use crate::types::message::{AssistantMessage, Message, ToolResultMessage};
use crate::types::model::{AnyModel, Model};
use crate::types::usage::{Usage, UsageCost};

const DEFAULT_API: &str = "faux";
const DEFAULT_PROVIDER: &str = "faux";
const DEFAULT_MODEL_ID: &str = "faux-1";
const DEFAULT_MODEL_NAME: &str = "Faux Model";
const DEFAULT_BASE_URL: &str = "http://localhost:0";
const DEFAULT_MIN_TOKEN_SIZE: u32 = 3;
const DEFAULT_MAX_TOKEN_SIZE: u32 = 5;
const ABORTED_MESSAGE: &str = "Request was aborted";

/// The future a `FauxResponseFactory` returns; `Err` is the factory
/// throwing upstream (`error instanceof Error ? error.message : String`).
pub type FauxResponseFactoryFuture =
    Pin<Box<dyn Future<Output = Result<AssistantMessage, String>> + Send>>;

/// `FauxResponseFactory` — `(context, options, state, model) -> message`.
pub type FauxResponseFactory = Box<
    dyn Fn(
            &TranscriptContext,
            &SimpleStreamOptions,
            &FauxState,
            &Model,
        ) -> FauxResponseFactoryFuture
        + Send
        + Sync,
>;

/// `FauxResponseStep` — a fixed message or a factory that sees the
/// request context/options/state.
pub enum FauxResponseStep {
    /// A fixed assistant message.
    Message(Box<AssistantMessage>),
    /// `(context, options, state, model) -> message`.
    Factory(FauxResponseFactory),
    /// `Deferred` — the stream ends `done:deferred` with a handle;
    /// `fetch_deferred` replays `then`.
    Deferred {
        /// `id` of the deferred handle.
        id: String,
        /// The eventual response `fetch_deferred` replays.
        then: Box<AssistantMessage>,
    },
}

/// `FauxProviderState` — observable counters.
#[derive(Clone, Debug, Default)]
pub struct FauxState {
    /// `callCount`.
    pub call_count: usize,
    /// `deferredFetchCount`.
    pub deferred_fetch_count: usize,
    /// `cancelledDeferred`.
    pub cancelled_deferred: Vec<DeferredHandle>,
}

/// `FauxModelDefinition` — `RegisterFauxProviderOptions.models`.
#[derive(Clone, Debug)]
pub struct FauxModelDefinition {
    /// `id`.
    pub id: String,
    /// `name` (default `id`).
    pub name: Option<String>,
    /// `reasoning`.
    pub reasoning: bool,
    /// `contextWindow`.
    pub context_window: u64,
    /// `maxTokens`.
    pub max_tokens: u64,
}

/// `RegisterFauxProviderOptions.tokenSize` — chunk size bounds in
/// estimated tokens; each chunk is `size * 4` UTF-16 units.
#[derive(Clone, Copy, Debug, Default)]
pub struct FauxTokenSize {
    /// `tokenSize.min` (default 3, clamped ≥1 and ≤max).
    pub min: Option<u32>,
    /// `tokenSize.max` (default 5).
    pub max: Option<u32>,
}

/// `RegisterFauxProviderOptions`.
#[derive(Debug, Default)]
pub struct FauxOptions {
    /// `api` — random suffix upstream; fixed string here.
    pub api: Option<String>,
    /// `provider`.
    pub provider: Option<String>,
    /// `models`.
    pub models: Vec<FauxModelDefinition>,
    /// `deferred.pendingFetches` / `pollAfterMs`.
    pub deferred: Option<(usize, Option<u64>)>,
    /// `tokensPerSecond` — pacing delay `estimateTokens(chunk) / tps`
    /// seconds per chunk; `None`/`<=0` yields to the task queue.
    pub tokens_per_second: Option<f64>,
    /// `tokenSize`.
    pub token_size: Option<FauxTokenSize>,
}

/// `fauxAssistantMessage` options.
#[derive(Clone, Debug, Default)]
pub struct FauxMessageOptions {
    /// `stopReason` (default `"stop"`).
    pub stop_reason: Option<StopReason>,
    /// `errorMessage`.
    pub error_message: Option<String>,
    /// `responseId`.
    pub response_id: Option<String>,
    /// `timestamp` (default `Date.now()`).
    pub timestamp: Option<i64>,
}

/// One deferred response slot — message plus remaining fetch count.
#[derive(Debug)]
struct DeferredEntry {
    /// The eventual response.
    message: Box<AssistantMessage>,
    /// `pendingFetches` — the fetches remaining before `then` resolves.
    pending_fetches: usize,
}

/// `FauxProviderHandle` — provider + scripting handles.
pub struct FauxProvider {
    /// `provider` — register via `Models::set_provider`.
    pub provider: Arc<dyn Provider>,
    /// `api`.
    pub api: String,
    /// `models`.
    pub models: Vec<Model>,
    /// `state`.
    pub state: Arc<Mutex<FauxState>>,
    /// `responses` queue.
    pub responses: Arc<Mutex<VecDeque<FauxResponseStep>>>,
}

impl std::fmt::Debug for FauxProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FauxProvider")
            .field("api", &self.api)
            .field("models", &self.models)
            .field("state", &self.state)
            .finish_non_exhaustive()
    }
}

impl std::fmt::Debug for FauxResponseStep {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Message(m) => f.debug_tuple("Message").field(m).finish(),
            Self::Factory(_) => f.debug_tuple("Factory").finish_non_exhaustive(),
            Self::Deferred { id, .. } => f.debug_tuple("Deferred").field(id).finish(),
        }
    }
}

impl FauxProvider {
    /// `setResponses(responses)` — replace the queue.
    pub fn set_responses(&self, responses: Vec<FauxResponseStep>) {
        *self
            .responses
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = responses.into();
    }

    /// `appendResponses(responses)` — add to the queue.
    pub fn append_responses(&self, responses: Vec<FauxResponseStep>) {
        self.responses
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .extend(responses);
    }

    /// `getPendingResponseCount`.
    #[must_use]
    pub fn pending_count(&self) -> usize {
        self.responses
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .len()
    }

    /// `getModel()`.
    #[must_use]
    pub fn get_model(&self) -> &Model {
        self.models
            .first()
            .unwrap_or_else(|| unreachable!("faux models"))
    }

    /// `getModel(modelId)`.
    #[must_use]
    pub fn model_by_id(&self, model_id: &str) -> Option<&Model> {
        self.models.iter().find(|model| model.id == model_id)
    }
}

/// `fauxText` — a text block.
#[must_use]
pub fn faux_text(text: &str) -> AssistantContentBlock {
    AssistantContentBlock::Text(TextContent {
        text: text.to_owned(),
        text_signature: None,
    })
}

/// `fauxThinking` — a thinking block.
#[must_use]
pub fn faux_thinking(thinking: &str) -> AssistantContentBlock {
    AssistantContentBlock::Thinking(ThinkingContent {
        thinking: thinking.to_owned(),
        thinking_signature: None,
        redacted: None,
    })
}

/// `fauxToolCall` — a tool call; `id` defaults to `randomId("tool")`.
#[must_use]
pub fn faux_tool_call(
    name: &str,
    arguments: Map<String, Value>,
    id: Option<&str>,
) -> AssistantContentBlock {
    AssistantContentBlock::ToolCall(ToolCall {
        id: id.map_or_else(|| random_id("tool"), str::to_owned),
        name: name.to_owned(),
        arguments,
        thought_signature: None,
        namespace: None,
        arguments_error: None,
    })
}

/// `fauxAssistantMessage(content)` — a scripted response message.
#[must_use]
pub fn faux_assistant_message(content: Vec<AssistantContentBlock>) -> AssistantMessage {
    faux_assistant_message_with(content, &FauxMessageOptions::default())
}

/// `fauxAssistantMessage(content, options)`.
#[must_use]
pub fn faux_assistant_message_with(
    content: Vec<AssistantContentBlock>,
    options: &FauxMessageOptions,
) -> AssistantMessage {
    AssistantMessage {
        content,
        api: crate::types::api::Api::from(DEFAULT_API),
        provider: crate::types::api::ProviderId::from(DEFAULT_PROVIDER),
        model: String::from(DEFAULT_MODEL_ID),
        usage: default_usage(),
        stop_reason: options.stop_reason.unwrap_or(StopReason::Stop),
        timestamp: options.timestamp.unwrap_or_else(now_ms),
        provider_thinking_level: None,
        response_id: options.response_id.clone(),
        response_model: None,
        raw_stop_reason: None,
        error_message: options.error_message.clone(),
        diagnostics: None,
        deferred: None,
        end_turn: None,
        duration_ms: None,
        thinking_level: None,
    }
}

/// `DEFAULT_USAGE`.
const fn default_usage() -> Usage {
    Usage {
        input: 0,
        output: 0,
        cache_read: 0,
        cache_write: 0,
        total_tokens: 0,
        cost: UsageCost {
            input: 0.0,
            output: 0.0,
            cache_read: 0.0,
            cache_write: 0.0,
            total: 0.0,
            total_from_provider: None,
        },
        cache_write1h: None,
        reasoning: None,
    }
}

fn now_ms() -> i64 {
    i64::try_from(
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_millis()),
    )
    .unwrap_or(0)
}

/// `Math.random()` for chunk sizes and id suffixes — a scripted-test
/// source, not crypto: a counter seeded once from the clock, mixed
/// through splitmix64.
fn next_random() -> u64 {
    static STATE: AtomicU64 = AtomicU64::new(0);
    let seed = || {
        u64::try_from(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| d.as_nanos()),
        )
        .unwrap_or(0x9E37_79B9_7F4A_7C15)
            | 1
    };
    if STATE.load(Ordering::Relaxed) == 0 {
        STATE.store(seed(), Ordering::Relaxed);
    }
    let mut value = STATE.fetch_add(0x9E37_79B9_7F4A_7C15, Ordering::Relaxed);
    value ^= value >> 30;
    value = value.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

/// `randomId(prefix)` — `${prefix}:${Date.now()}:${base36 random}`.
fn random_id(prefix: &str) -> String {
    format!("{prefix}:{}:{:x}", now_ms(), next_random())
}

/// `estimateTokens` — `Math.ceil(text.length / 4)`; `.length` is
/// UTF-16 units.
fn estimate_tokens(text: &str) -> u64 {
    crate::text::js_len(text).div_ceil(4) as u64
}

/// `contentToText` — a string or text/image blocks joined by newlines.
fn content_to_text(content: &crate::types::content::TextOrBlocksWithImages) -> String {
    match content {
        crate::types::content::TextOrBlocksWithImages::Text(text) => text.clone(),
        crate::types::content::TextOrBlocksWithImages::Blocks(blocks) => blocks
            .iter()
            .map(content_block_text)
            .collect::<Vec<_>>()
            .join("\n"),
    }
}

/// One user/tool-result block: text verbatim, image as
/// `[image:{mime}:{data.length}]`.
fn content_block_text(block: &UserContentBlock) -> String {
    match block {
        UserContentBlock::Text(text) => text.text.clone(),
        UserContentBlock::Image(ImageContent { data, mime_type }) => {
            format!("[image:{}:{}]", mime_type, crate::text::js_len(data))
        }
    }
}

/// `assistantContentToText` — text/thinking verbatim, tool calls as
/// `{name}:{JSON.stringify(arguments)}`.
fn assistant_content_to_text(content: &[AssistantContentBlock]) -> String {
    content
        .iter()
        .map(|block| match block {
            AssistantContentBlock::Text(text) => text.text.clone(),
            AssistantContentBlock::Thinking(thinking) => thinking.thinking.clone(),
            AssistantContentBlock::ToolCall(call) => {
                let arguments = crate::json::to_string(&Value::Object(call.arguments.clone()))
                    .unwrap_or_else(|_| String::from("{}"));
                format!("{}:{}", call.name, arguments)
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// `toolResultToText` — `{toolName}\n{content...}`.
fn tool_result_to_text(message: &ToolResultMessage) -> String {
    let mut parts = vec![message.tool_name.clone()];
    for block in &message.content {
        parts.push(content_block_text(block));
    }
    parts.join("\n")
}

/// `messageToText` — the transcript line per message role.
fn message_to_text(message: &Message) -> String {
    match message {
        Message::System(system) => {
            let mut parts = vec![crate::text::get_system_message_text(system)];
            for tool in system.tools_removed.as_deref().unwrap_or(&[]) {
                parts.push(format!(
                    "tool-:{}",
                    crate::json::to_string(tool).unwrap_or_else(|_| String::from("null"))
                ));
            }
            for tool in system.tools_added.as_deref().unwrap_or(&[]) {
                parts.push(format!(
                    "tool+:{}",
                    crate::json::to_string(tool).unwrap_or_else(|_| String::from("null"))
                ));
            }
            parts.retain(|part| !part.is_empty());
            parts.join("\n")
        }
        Message::User(user) => content_to_text(&user.content),
        Message::Assistant(assistant) => assistant_content_to_text(&assistant.content),
        Message::ToolResult(result) => tool_result_to_text(result),
    }
}

/// `joinedLength` — the whole prompt joins lines with blank lines.
fn joined_length(messages: &[String], count: usize) -> usize {
    let mut length = count.saturating_sub(1) * 2;
    for message in messages.iter().take(count) {
        length += crate::text::js_len(message);
    }
    length
}

/// `commonPrefixLength` — shared UTF-16 units at the fronts; the loop
/// stops at the first differing unit or the shorter string's end.
fn common_prefix_length(a: &str, b: &str) -> usize {
    let mut units = 0;
    for (left, right) in a.encode_utf16().zip(b.encode_utf16()) {
        if left != right {
            break;
        }
        units += 1;
    }
    units
}

/// `commonPromptPrefixLength` — equal messages compare whole;
/// characters compare only from the first message that differs.
fn common_prompt_prefix_length(previous: &[String], current: &[String]) -> usize {
    let mut index = 0;
    for (before, after) in previous.iter().zip(current.iter()) {
        if before != after {
            break;
        }
        index += 1;
    }
    let rest = |messages: &[String]| {
        if index == messages.len() {
            String::new()
        } else {
            let tail = messages.get(index..).unwrap_or_default();
            format!(
                "{}{}",
                if index > 0 { "\n\n" } else { "" },
                tail.join("\n\n")
            )
        }
    };
    joined_length(previous, index) + common_prefix_length(&rest(previous), &rest(current))
}

/// `withUsageEstimate` — upstream's usage accounting.
///
/// Prompt tokens = ~4-per-char over `{role}:{messageToText}` lines
/// joined by blank lines; `cacheRead`/`cacheWrite` come from the
/// provider's `promptCache` and the request's `sessionId`/
/// `cacheRetention`.
#[must_use]
pub fn with_usage_estimate<S: std::hash::BuildHasher>(
    message: AssistantMessage,
    context: &TranscriptContext,
    options: &StreamOptions,
    prompt_cache: &Mutex<std::collections::HashMap<String, Vec<String>, S>>,
) -> AssistantMessage {
    let prompt: Vec<String> = context
        .messages()
        .iter()
        .map(|entry| format!("{}:{}", message_role(entry), message_to_text(entry)))
        .collect();
    let prompt_tokens = joined_length(&prompt, prompt.len()).div_ceil(4) as u64;
    let output_tokens = estimate_tokens(&assistant_content_to_text(&message.content));
    let mut input = prompt_tokens;
    let mut cache_read = 0;
    let mut cache_write = 0;
    if let Some(session_id) = &options.session_id
        && options.cache_retention != Some(crate::options::CacheRetention::None)
    {
        let mut cache = prompt_cache
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(previous_prompt) = cache.get(session_id.as_str()) {
            let cached_chars = common_prompt_prefix_length(previous_prompt, &prompt);
            cache_read = cached_chars.div_ceil(4) as u64;
            cache_write = prompt_tokens.saturating_sub(cache_read);
        } else {
            cache_write = prompt_tokens;
        }
        // Every prompt token is either read from or written to the
        // simulated cache.
        input = 0;
        cache.insert(session_id.clone(), prompt);
    }
    let mut estimated = message;
    estimated.usage = Usage {
        input,
        output: output_tokens,
        cache_read,
        cache_write,
        total_tokens: input + output_tokens + cache_read + cache_write,
        cost: UsageCost {
            input: 0.0,
            output: 0.0,
            cache_read: 0.0,
            cache_write: 0.0,
            total: 0.0,
            total_from_provider: None,
        },
        cache_write1h: None,
        reasoning: None,
    };
    estimated
}

/// The wire role name for the `"{role}:{text}"` prompt line.
const fn message_role(message: &Message) -> &'static str {
    match message {
        Message::System(_) => "system",
        Message::User(_) => "user",
        Message::Assistant(_) => "assistant",
        Message::ToolResult(_) => "toolResult",
    }
}

/// `splitStringByTokenSize` — `size * 4` UTF-16 units per chunk, the
/// size drawn from `[min, max]` like `Math.floor(Math.random() * ...)`.
fn split_string_by_token_size(text: &str, min_token_size: u32, max_token_size: u32) -> Vec<String> {
    let mut chunks = Vec::new();
    let mut index = 0;
    while index < crate::text::js_len(text) {
        let spread = u64::from(max_token_size - min_token_size + 1);
        let token_size = u64::from(min_token_size) + next_random() % spread;
        let char_size = usize::try_from((token_size * 4).max(1)).unwrap_or(usize::MAX);
        chunks.push(crate::text::js_slice(text, index, index + char_size).to_owned());
        index += char_size;
    }
    if chunks.is_empty() {
        chunks.push(String::new());
    }
    chunks
}

/// `cloneMessage` — rewrite `api`/`provider`/`model`/`timestamp`/`usage`
/// on the scripted message.
fn clone_message(
    message: &AssistantMessage,
    api: &str,
    provider: &str,
    model_id: &str,
) -> AssistantMessage {
    let mut cloned = message.clone();
    cloned.api = crate::types::api::Api::from(api.to_owned());
    cloned.provider = crate::types::api::ProviderId::from(provider.to_owned());
    model_id.clone_into(&mut cloned.model);
    if cloned.timestamp == 0 {
        cloned.timestamp = now_ms();
    }
    cloned
}

/// `createErrorMessage` — terminal message for a thrown/queued error.
fn create_error_message(
    error: &str,
    api: &str,
    provider: &str,
    model_id: &str,
) -> AssistantMessage {
    let mut message = faux_assistant_message(Vec::new());
    message.api = crate::types::api::Api::from(api.to_owned());
    message.provider = crate::types::api::ProviderId::from(provider.to_owned());
    model_id.clone_into(&mut message.model);
    message.stop_reason = StopReason::Error;
    message.error_message = Some(error.to_owned());
    message
}

/// `scheduleChunk` — `estimateTokens(chunk) / tokensPerSecond` seconds,
/// or a yield upstream's `queueMicrotask` stands in for.
async fn schedule_chunk(chunk: &str, tokens_per_second: Option<f64>) {
    match tokens_per_second {
        Some(tps) if tps > 0.0 => {
            let seconds = estimate_tokens(chunk) as f64 / tps;
            tokio::time::sleep(Duration::from_secs_f64(seconds)).await;
        }
        _ => tokio::task::yield_now().await,
    }
}

fn is_aborted(signal: Option<&tokio_util::sync::CancellationToken>) -> bool {
    signal.is_some_and(tokio_util::sync::CancellationToken::is_cancelled)
}

/// `streamWithDeltas` — `start`, per-block start/delta/end with
/// pacing, abort checks at upstream's two points, and the stop-reason
/// terminal switch.
async fn stream_with_deltas(
    mut sender: crate::stream::AssistantStreamSender,
    message: &AssistantMessage,
    min_token_size: u32,
    max_token_size: u32,
    tokens_per_second: Option<f64>,
    signal: Option<tokio_util::sync::CancellationToken>,
) {
    if is_aborted(signal.as_ref()) {
        abort_stream(sender, message).await;
        return;
    }
    sender.start().await.ok();
    for block in &message.content {
        if is_aborted(signal.as_ref()) {
            abort_stream(sender, message).await;
            return;
        }
        let aborted = match block {
            AssistantContentBlock::Thinking(thinking) => {
                emit_thinking(
                    &mut sender,
                    thinking,
                    min_token_size,
                    max_token_size,
                    tokens_per_second,
                    signal.as_ref(),
                )
                .await
            }
            AssistantContentBlock::Text(text) => {
                emit_text(
                    &mut sender,
                    text,
                    min_token_size,
                    max_token_size,
                    tokens_per_second,
                    signal.as_ref(),
                )
                .await
            }
            AssistantContentBlock::ToolCall(call) => {
                emit_tool_call(
                    &mut sender,
                    call,
                    min_token_size,
                    max_token_size,
                    tokens_per_second,
                    signal.as_ref(),
                )
                .await
            }
        };
        if aborted {
            abort_stream(sender, message).await;
            return;
        }
    }
    if message.stop_reason == StopReason::Pending {
        send_error_message(
            sender,
            create_error_message(
                "Faux response ended without a stop reason",
                message.api.as_str(),
                message.provider.as_str(),
                &message.model,
            ),
        )
        .await;
        return;
    }
    finish_message(sender, message.clone()).await;
}

/// One thinking block: `thinking_start`, paced chunks, `thinking_end`.
/// `true` when the signal fired mid-block — the delta after
/// `scheduleChunk` is upstream's abort checkpoint.
async fn emit_thinking(
    sender: &mut crate::stream::AssistantStreamSender,
    block: &ThinkingContent,
    min_token_size: u32,
    max_token_size: u32,
    tokens_per_second: Option<f64>,
    signal: Option<&tokio_util::sync::CancellationToken>,
) -> bool {
    let index = sender.thinking_start().await.unwrap_or(usize::MAX);
    for chunk in split_string_by_token_size(&block.thinking, min_token_size, max_token_size) {
        schedule_chunk(&chunk, tokens_per_second).await;
        if is_aborted(signal) {
            return true;
        }
        sender.thinking_delta(index, &chunk).await.ok();
    }
    sender.thinking_end(index, block.clone()).await.ok();
    false
}

/// One text block: `text_start`, paced chunks, `text_end`.
async fn emit_text(
    sender: &mut crate::stream::AssistantStreamSender,
    block: &TextContent,
    min_token_size: u32,
    max_token_size: u32,
    tokens_per_second: Option<f64>,
    signal: Option<&tokio_util::sync::CancellationToken>,
) -> bool {
    let index = sender.text_start().await.unwrap_or(usize::MAX);
    for chunk in split_string_by_token_size(&block.text, min_token_size, max_token_size) {
        schedule_chunk(&chunk, tokens_per_second).await;
        if is_aborted(signal) {
            return true;
        }
        sender.text_delta(index, &chunk).await.ok();
    }
    sender.text_end(index, block.clone()).await.ok();
    false
}

/// One tool call: `toolcall_start` with empty arguments (the partial
/// grows like upstream's `{ ...block, arguments: {} }`), serialized
/// arguments chunked through `toolcall_delta`, `toolcall_end`.
async fn emit_tool_call(
    sender: &mut crate::stream::AssistantStreamSender,
    block: &ToolCall,
    min_token_size: u32,
    max_token_size: u32,
    tokens_per_second: Option<f64>,
    signal: Option<&tokio_util::sync::CancellationToken>,
) -> bool {
    let open = ToolCall {
        id: block.id.clone(),
        name: block.name.clone(),
        arguments: Map::new(),
        thought_signature: block.thought_signature.clone(),
        namespace: block.namespace.clone(),
        arguments_error: None,
    };
    let index = sender.toolcall_start(open).await.unwrap_or(usize::MAX);
    let arguments = crate::json::to_string(&Value::Object(block.arguments.clone()))
        .unwrap_or_else(|_| String::from("{}"));
    for chunk in split_string_by_token_size(&arguments, min_token_size, max_token_size) {
        schedule_chunk(&chunk, tokens_per_second).await;
        if is_aborted(signal) {
            return true;
        }
        sender.toolcall_delta(index, &chunk).await.ok();
    }
    sender.toolcall_end(index).await.ok();
    false
}

/// `createAbortedMessage(partial)` — usage metadata then
/// `error:aborted` with `"Request was aborted"`.
async fn abort_stream(
    mut sender: crate::stream::AssistantStreamSender,
    message: &AssistantMessage,
) {
    sender
        .metadata(crate::stream::Metadata::Usage(message.usage.clone()))
        .await
        .ok();
    sender
        .error(
            crate::stream::event::ErrorReason::Aborted,
            ABORTED_MESSAGE.to_owned(),
        )
        .await
        .ok();
}

/// The catch-path terminal — `error:error` with no usage metadata
/// (upstream's `createErrorMessage` carries `DEFAULT_USAGE`).
async fn send_error_message(
    sender: crate::stream::AssistantStreamSender,
    message: AssistantMessage,
) {
    sender
        .error(
            crate::stream::event::ErrorReason::Error,
            message.error_message.unwrap_or_default(),
        )
        .await
        .ok();
}

/// The scripted `ChatApi` — drains the response queue per call.
struct FauxChatApi {
    api: String,
    provider: String,
    responses: Arc<Mutex<VecDeque<FauxResponseStep>>>,
    deferred: Arc<Mutex<std::collections::HashMap<String, DeferredEntry>>>,
    state: Arc<Mutex<FauxState>>,
    prompt_cache: Arc<Mutex<std::collections::HashMap<String, Vec<String>>>>,
    deferred_options: Option<(usize, Option<u64>)>,
    min_token_size: u32,
    max_token_size: u32,
    tokens_per_second: Option<f64>,
}

impl std::fmt::Debug for FauxChatApi {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FauxChatApi").finish_non_exhaustive()
    }
}

impl FauxChatApi {
    fn state_snapshot(&self) -> FauxState {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }
}

impl ChatApi for FauxChatApi {
    fn fetch_deferred(
        &self,
        model: &Model,
        handle: &DeferredHandle,
        options: crate::options::DeferredFetchOptions,
    ) -> Option<AssistantStream> {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .deferred_fetch_count += 1;
        let (message, pending_fetches) = take_fetch(&self.deferred, &handle.id)?;
        let base = base_message(model);
        let model = model.clone();
        let handle = handle.clone();
        Some(AssistantStream::spawn(
            base,
            16,
            move |mut sender| async move {
                emit_blocks(&mut sender, &message).await;
                if let Some(on_response) = &options.on_response {
                    let metadata = crate::options::ResponseMetadata {
                        status: 200,
                        headers: Map::new(),
                    };
                    on_response(&metadata, &model).await;
                }
                if pending_fetches {
                    finish_deferred(sender, &handle, &model).await;
                    return;
                }
                finish_message(sender, message).await;
            },
        ))
    }

    fn cancel_deferred(
        &self,
        model: &Model,
        handle: &DeferredHandle,
        options: crate::options::DeferredCancelOptions,
    ) -> Option<crate::provider::BoxFuture<'static, Result<(), ModelsError>>> {
        let deferred = Arc::clone(&self.deferred);
        let state = Arc::clone(&self.state);
        let handle = handle.clone();
        let model = model.clone();
        Some(Box::pin(async move {
            let id = handle.id.clone();
            if let Some(on_response) = &options.on_response {
                let metadata = crate::options::ResponseMetadata {
                    status: 200,
                    headers: Map::new(),
                };
                on_response(&metadata, &model).await;
            }
            state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .cancelled_deferred
                .push(handle);
            deferred
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .remove(&id);
            Ok(())
        }))
    }

    fn stream(
        &self,
        model: &Model,
        context: TranscriptContext,
        options: StreamOptions,
    ) -> AssistantStream {
        let model = model.clone();
        let step = self
            .responses
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .pop_front();
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .call_count += 1;
        let snapshot = self.state_snapshot();
        let deferred_store = Arc::clone(&self.deferred);
        let prompt_cache = Arc::clone(&self.prompt_cache);
        let deferred_options = self.deferred_options;
        let api = self.api.clone();
        let provider = self.provider.clone();
        let signal = options.signal.clone();
        let min_token_size = self.min_token_size;
        let max_token_size = self.max_token_size;
        let tokens_per_second = self.tokens_per_second;
        let base = base_message(&model);
        AssistantStream::spawn(base, 16, move |mut sender| async move {
            if let Some(on_response) = &options.on_response {
                let metadata = crate::options::ResponseMetadata {
                    status: 200,
                    headers: Map::new(),
                };
                on_response(&metadata, &model).await;
            }
            let Some(step) = step else {
                let mut message = create_error_message(
                    "No more faux responses queued",
                    &api,
                    &provider,
                    &model.id,
                );
                message = with_usage_estimate(message, &context, &options, &prompt_cache);
                sender
                    .metadata(crate::stream::Metadata::Usage(message.usage.clone()))
                    .await
                    .ok();
                send_error_message(sender, message).await;
                return;
            };
            let resolved = resolve_step(&step, &context, &options, &snapshot, &model).await;
            let mut message = match resolved {
                Ok(message) => clone_message(&message, &api, &provider, &model.id),
                Err(error) => {
                    send_error_message(
                        sender,
                        create_error_message(&error, &api, &provider, &model.id),
                    )
                    .await;
                    return;
                }
            };
            // The store keeps the scripted eventual message: the resolved one
            // already carries the deferred stop reason and handle.
            if let FauxResponseStep::Deferred { id, then } = &step {
                deferred_store
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .insert(
                        id.clone(),
                        DeferredEntry {
                            message: then.clone(),
                            pending_fetches: deferred_options.map_or(0, |d| d.0),
                        },
                    );
            }
            message = with_usage_estimate(message, &context, &options, &prompt_cache);
            stream_with_deltas(
                sender,
                &message,
                min_token_size,
                max_token_size,
                tokens_per_second,
                signal,
            )
            .await;
        })
    }
}

/// `fauxProvider(options)` — a `create_provider` assembly with the
/// scripted `ChatApi` and ambient auth.
///
/// # Errors
/// Returns `create_provider`'s error; the spec always supplies `api`, so it
/// does not occur in practice.
pub fn faux_provider(options: FauxOptions) -> Result<FauxProvider, ModelsError> {
    let api = options.api.unwrap_or_else(|| String::from(DEFAULT_API));
    let provider = options
        .provider
        .unwrap_or_else(|| String::from(DEFAULT_PROVIDER));
    let size = options.token_size.unwrap_or_default();
    let min_token_size = size
        .min
        .unwrap_or(DEFAULT_MIN_TOKEN_SIZE)
        .min(size.max.unwrap_or(DEFAULT_MAX_TOKEN_SIZE))
        .max(1);
    let max_token_size = min_token_size.max(size.max.unwrap_or(DEFAULT_MAX_TOKEN_SIZE));
    let responses = Arc::new(Mutex::new(VecDeque::new()));
    let state = Arc::new(Mutex::new(FauxState::default()));
    let deferred = Arc::new(Mutex::new(std::collections::HashMap::new()));
    let prompt_cache = Arc::new(Mutex::new(std::collections::HashMap::new()));
    let chat = Arc::new(FauxChatApi {
        api: api.clone(),
        provider: provider.clone(),
        responses: Arc::clone(&responses),
        deferred: Arc::clone(&deferred),
        state: Arc::clone(&state),
        prompt_cache,
        deferred_options: options.deferred,
        min_token_size,
        max_token_size,
        tokens_per_second: options.tokens_per_second,
    });
    let models: Vec<AnyModel> = if options.models.is_empty() {
        vec![AnyModel::Chat(Box::new(default_model(&api, &provider)))]
    } else {
        options
            .models
            .iter()
            .map(|d| AnyModel::Chat(Box::new(faux_model(d, &api, &provider))))
            .collect()
    };
    let spec = ProviderSpec {
        id: provider,
        name: None,
        base_url: Some(String::from(DEFAULT_BASE_URL)),
        headers: None,
        auth: faux_auth(),
        models,
        api: Some(ApiDispatch::Single(chat)),
        images: BTreeMap::new(),
        classifiers: BTreeMap::new(),
        fetch_models: None,
        filter_models: None,
        filter_all_models: None,
    };
    let built = create_provider(spec)?;
    let chat_models = built.get_models();
    let provider_arc: Arc<dyn Provider> = Arc::new(built);
    Ok(FauxProvider {
        provider: provider_arc,
        api,
        models: chat_models,
        state,
        responses,
    })
}

fn default_model(api: &str, provider: &str) -> Model {
    faux_model(
        &FauxModelDefinition {
            id: String::from(DEFAULT_MODEL_ID),
            name: Some(String::from(DEFAULT_MODEL_NAME)),
            reasoning: false,
            context_window: 128_000,
            max_tokens: 16_384,
        },
        api,
        provider,
    )
}

fn faux_model(def: &FauxModelDefinition, api: &str, provider: &str) -> Model {
    Model {
        id: def.id.clone(),
        name: def.name.clone().unwrap_or_else(|| def.id.clone()),
        api: crate::types::api::Api::from(api.to_owned()),
        provider: crate::types::api::ProviderId::from(provider.to_owned()),
        base_url: String::from(DEFAULT_BASE_URL),
        input: vec![
            crate::types::model::ModelInputModality::Text,
            crate::types::model::ModelInputModality::Image,
        ],
        input_limits: None,
        cost: crate::types::model::ModelCost {
            input: 0.0,
            output: 0.0,
            cache_read: 0.0,
            cache_write: 0.0,
            tiers: None,
        },
        headers: None,
        kind: None,
        reasoning: def.reasoning,
        thinking_level_map: None,
        prompt_cache: None,
        context_window: def.context_window,
        max_tokens: def.max_tokens,
        sampling_params: None,
        sampling_params_by_thinking_level: None,
        compat: None,
        extra: Map::new(),
    }
}

/// `emit_blocks` for `fetch_deferred` — the queued message's blocks
/// without pacing (upstream replays through `streamWithDeltas`; the
/// fetch path is a later milestone, kept working here).
async fn emit_blocks(
    sender: &mut crate::stream::AssistantStreamSender,
    message: &AssistantMessage,
) {
    sender.start().await.ok();
    for block in &message.content {
        match block {
            AssistantContentBlock::Text(t) => {
                let index = sender.text_start().await.unwrap_or(usize::MAX);
                sender.text_delta(index, &t.text).await.ok();
                sender.text_end(index, t.clone()).await.ok();
            }
            AssistantContentBlock::Thinking(t) => {
                let index = sender.thinking_start().await.unwrap_or(usize::MAX);
                sender.thinking_delta(index, &t.thinking).await.ok();
                sender.thinking_end(index, t.clone()).await.ok();
            }
            AssistantContentBlock::ToolCall(tc) => {
                let index = sender
                    .toolcall_start(tc.clone())
                    .await
                    .unwrap_or(usize::MAX);
                sender.toolcall_end(index).await.ok();
            }
        }
    }
}

/// `fetch_deferred`'s pending-deferred terminal — `done:deferred`
/// with the handle when `pendingFetches > 0` remain.
async fn finish_deferred(
    sender: crate::stream::AssistantStreamSender,
    handle: &DeferredHandle,
    model: &Model,
) {
    let mut message = base_message(model);
    message.deferred = Some(handle.clone());
    message.stop_reason = StopReason::Deferred;
    finish_message(sender, message).await;
}

/// The stream's base message: empty content, `pending` like upstream's
/// `partial`, attributed to `model`.
fn base_message(model: &Model) -> AssistantMessage {
    let mut message = faux_assistant_message(Vec::new());
    message.api.clone_from(&model.api);
    message.provider.clone_from(&model.provider);
    message.model.clone_from(&model.id);
    message.stop_reason = StopReason::Pending;
    message
}

/// One `fetchDeferred` step: the scripted message and whether more fetches
/// remain (decrementing the remaining count).
fn take_fetch(
    store: &Mutex<std::collections::HashMap<String, DeferredEntry>>,
    id: &str,
) -> Option<(AssistantMessage, bool)> {
    let mut deferred = store
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let entry = deferred.get_mut(id)?;
    let has_more = entry.pending_fetches > 0;
    if has_more {
        entry.pending_fetches -= 1;
    }
    let message = (*entry.message).clone();
    drop(deferred);
    Some((message, has_more))
}

/// `streamWithDeltas`'s terminal event — the message's usage and deferred
/// handle, then `done`/`error` from its `stopReason`.
async fn finish_message(
    mut sender: crate::stream::AssistantStreamSender,
    message: AssistantMessage,
) {
    sender
        .metadata(crate::stream::Metadata::Usage(message.usage.clone()))
        .await
        .ok();
    if let Some(handle) = message.deferred.clone() {
        sender
            .metadata(crate::stream::Metadata::Deferred(handle))
            .await
            .ok();
    }
    match message.stop_reason {
        StopReason::Error | StopReason::Aborted => {
            let reason = if message.stop_reason == StopReason::Aborted {
                crate::stream::event::ErrorReason::Aborted
            } else {
                crate::stream::event::ErrorReason::Error
            };
            sender
                .error(reason, message.error_message.clone().unwrap_or_default())
                .await
                .ok();
        }
        reason => {
            let done = match reason {
                StopReason::Length => crate::stream::event::DoneReason::Length,
                StopReason::ToolUse => crate::stream::event::DoneReason::ToolUse,
                StopReason::Deferred => crate::stream::event::DoneReason::Deferred,
                _ => crate::stream::event::DoneReason::Stop,
            };
            sender.done(done).await.ok();
        }
    }
}

/// Resolve a queued `FauxResponseStep` to its `AssistantMessage`; the
/// `Err` is the factory's thrown error message.
async fn resolve_step(
    step: &FauxResponseStep,
    context: &TranscriptContext,
    options: &StreamOptions,
    state: &FauxState,
    model: &Model,
) -> Result<AssistantMessage, String> {
    match step {
        FauxResponseStep::Message(m) => Ok(m.as_ref().clone()),
        FauxResponseStep::Factory(f) => {
            f(
                context,
                &SimpleStreamOptions {
                    base: options.clone(),
                    ..SimpleStreamOptions::default()
                },
                state,
                model,
            )
            .await
        }
        FauxResponseStep::Deferred { id, then } => {
            let mut message = then.as_ref().clone();
            message.deferred = Some(DeferredHandle {
                provider: model.provider.as_str().to_owned(),
                model_id: model.id.clone(),
                api: model.api.as_str().to_owned(),
                id: id.clone(),
                expires_at: None,
                poll_after_ms: None,
                data: None,
            });
            message.stop_reason = StopReason::Deferred;
            Ok(message)
        }
    }
}

/// Faux's ambient auth — `resolve: () => ({ auth: {} })` so Models
/// requests run without credentials.
fn faux_auth() -> ProviderAuth {
    ProviderAuth {
        api_key: Some(ApiKeyAuth::custom(
            "Faux",
            Arc::new(|_, _| {
                Ok(Some(crate::auth::types::AuthResult {
                    auth: crate::auth::types::ModelAuth::default(),
                    env: None,
                    source: Some(String::from("faux")),
                }))
            }),
        )),
        oauth: None,
    }
}
