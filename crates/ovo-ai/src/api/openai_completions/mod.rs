//! The `openai-completions` adapter (upstream `api/openai-completions.ts`).
//!
//! [`build_request`](crate::api::openai_completions::build_request) ports
//! `buildParams` and `createClient`: the request body in upstream key order
//! (serialized with `ovo_ai::json`, byte-identical) and the headers in the
//! `OpenAI` SDK's merge order.
//!
//! The streaming side is
//! [`stream`](crate::api::openai_completions::stream).
//!
//! Not yet ported: `onPayload`.

use std::collections::BTreeMap;

use serde_json::{Map, Value, json};

use crate::api::constrained_sampling::create_grammar_tool_input_properties;
use crate::api::openai_prompt_cache::clamp_openai_prompt_cache_key;
use crate::api::openai_sdk::{client_api_key, endpoint_url, request_headers, tool_choice_value};
use crate::json::is_truthy;
use crate::options::{
    CacheRetention, SimpleStreamOptions, StreamOptions, build_base_options, resolve_cache_retention,
};
use crate::simple_options::{
    ThinkingBudgets, clamp_thinking_budget_to_answer_room, resolve_sampling_params,
    thinking_budget_for_level,
};
use crate::transcript::{
    TranscriptContext, get_declared_tools, resolve_transcript, resolve_transcript_tools,
};
use crate::types::content::AssistantContentBlock;
use crate::types::message::Message;
use crate::types::model::{Model, ModelThinkingLevel, ThinkingLevel, ThinkingLevelMapValue};

pub mod compat;
pub mod convert;
#[cfg(feature = "http")]
pub mod stream;

pub use compat::OpenAiCompletionsCompat;
use compat::{MaxTokensField, SessionAffinityFormat, ThinkingFormat};

/// The api identifier this adapter serves.
pub const API: &str = "openai-completions";

/// `OpenAICompletionsOptions`: the stream options plus Chat Completions
/// controls.
#[derive(Clone, Default)]
#[non_exhaustive]
pub struct OpenAiCompletionsOptions {
    /// The shared stream options.
    pub base: StreamOptions,
    /// `auto`, `none`, `required`, or `{ "type": "function", "function": { "name": ... } }`.
    pub tool_choice: Option<Value>,
    /// The reasoning level; `None` turns reasoning off where the format can.
    pub reasoning_effort: Option<ThinkingLevel>,
    /// Token budgets per level, for budget fields and `thinking.budget`.
    pub thinking_budgets: Option<ThinkingBudgets>,
}

impl std::fmt::Debug for OpenAiCompletionsOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpenAiCompletionsOptions")
            .field("tool_choice", &self.tool_choice)
            .field("reasoning_effort", &self.reasoning_effort)
            .field("thinking_budgets", &self.thinking_budgets)
            .finish_non_exhaustive()
    }
}

impl OpenAiCompletionsOptions {
    /// `streamSimple`'s mapping: the simple reasoning level clamped to the
    /// model, `off` sending no effort.
    #[must_use]
    pub fn from_simple(
        model: &Model,
        context: &TranscriptContext,
        options: &SimpleStreamOptions,
    ) -> Self {
        Self {
            base: build_base_options(
                model,
                context,
                Some(options),
                options.base.api_key.as_deref(),
            ),
            tool_choice: options.tool_choice.map(tool_choice_value),
            reasoning_effort: options.reasoning.and_then(|level| {
                crate::cost::clamp_thinking_level(model, level.into()).thinking()
            }),
            thinking_budgets: options.thinking_budgets.clone(),
        }
    }
}

/// One request to the Chat Completions endpoint, ready to send.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OpenAiCompletionsRequest {
    /// `{baseUrl}/chat/completions`, as the SDK's `buildURL` normalizes it.
    pub url: String,
    /// Request headers after the SDK and `createClient` merge order.
    pub headers: Map<String, Value>,
    /// The JSON body, byte-identical to upstream's `buildParams` output.
    pub body: Vec<u8>,
    /// The request params before serialization — `onPayload` sees and
    /// may replace these.
    pub params: Map<String, Value>,
    /// The input property of each tool sent as a grammar tool, by name; the
    /// stream maps `custom` tool-call input onto it.
    pub grammar_tool_input_properties: BTreeMap<String, String>,
}

/// `buildParams` and `createClient`: the request body in upstream key order
/// and the request headers.
///
/// # Errors
/// Returns upstream's message when there is no API key, a tool cannot be
/// sent, a grammar tool call has no string input, the base URL is invalid,
/// or the body fails to serialize.
pub fn build_request(
    model: &Model,
    context: &TranscriptContext,
    options: &OpenAiCompletionsOptions,
) -> Result<OpenAiCompletionsRequest, String> {
    let base = &options.base;
    let api_key = client_api_key(model.provider.as_str(), base)?;
    let compat = OpenAiCompletionsCompat::of(model);
    let transcript = resolve_transcript(context, Some(compat.supports_mid_convo_system_messages));
    let grammar_tool_input_properties = create_grammar_tool_input_properties(
        &get_declared_tools(transcript.messages()),
        compat.supports_openai_grammar_tools,
    )
    .map_err(|error| error.to_string())?;
    let retention = resolve_cache_retention(base.cache_retention, base.env.as_ref());
    let params = build_params(
        model,
        &transcript,
        options,
        &compat,
        retention,
        &grammar_tool_input_properties,
    )?;
    let session_id = base
        .session_id
        .as_deref()
        .filter(|id| !id.is_empty() && retention != CacheRetention::None);
    Ok(OpenAiCompletionsRequest {
        url: endpoint_url(&model.base_url, "chat/completions")?,
        headers: request_headers(
            model,
            transcript.messages(),
            base,
            &api_key,
            &session_affinity(session_id, &compat),
        ),
        body: crate::json::to_vec(&Value::Object(params.clone()))
            .map_err(|error| error.to_string())?,
        params,
        grammar_tool_input_properties,
    })
}

/// `buildParams`: the body in upstream insertion order.
fn build_params(
    model: &Model,
    transcript: &TranscriptContext,
    options: &OpenAiCompletionsOptions,
    compat: &OpenAiCompletionsCompat,
    retention: CacheRetention,
    grammar_tool_input_properties: &BTreeMap<String, String>,
) -> Result<Map<String, Value>, String> {
    let base = &options.base;
    let messages = transcript.messages();
    let request_tools =
        resolve_transcript_tools(messages, convert::tool_support(compat)).request_tools;
    let mut converted =
        convert::convert_messages(model, transcript, compat, grammar_tool_input_properties)?;
    let long = retention == CacheRetention::Long && compat.supports_long_cache_retention;

    let mut tools = if request_tools.is_empty() {
        // Anthropic behind a proxy requires `tools` when the conversation has
        // tool calls or results.
        has_tool_history(messages).then(Vec::new)
    } else {
        Some(convert::convert_tools(&request_tools, compat)?)
    };
    if let Some(cache_control) = cache_control(compat, retention, long) {
        apply_cache_control(&mut converted, tools.as_mut(), &cache_control);
    }

    let mut params = Map::new();
    params.insert("model".into(), Value::from(model.id.clone()));
    params.insert("messages".into(), Value::from(converted));
    params.insert("stream".into(), Value::from(true));
    let cache_key =
        (model.base_url.contains("api.openai.com") && retention != CacheRetention::None) || long;
    if let Some(key) =
        clamp_openai_prompt_cache_key(base.session_id.as_deref()).filter(|_| cache_key)
    {
        params.insert("prompt_cache_key".into(), Value::from(key));
    }
    if long {
        params.insert("prompt_cache_retention".into(), Value::from("24h"));
    }
    if compat.supports_usage_in_streaming {
        params.insert("stream_options".into(), json!({ "include_usage": true }));
    }
    if compat.supports_store {
        params.insert("store".into(), Value::from(false));
    }
    let max_tokens = base.max_tokens.filter(|max_tokens| *max_tokens > 0);
    if let Some(max_tokens) = max_tokens {
        // `max_tokens` is deprecated by OpenAI but some compatible providers
        // only accept it.
        let field = match compat.max_tokens_field {
            MaxTokensField::MaxTokens => "max_tokens",
            MaxTokensField::MaxCompletionTokens => "max_completion_tokens",
        };
        params.insert(field.into(), Value::from(max_tokens));
    }
    if let Some(temperature) = base.temperature {
        params.insert("temperature".into(), Value::from(temperature));
    }
    let has_request_tools = tools.as_ref().is_some_and(|tools| !tools.is_empty());
    if let Some(tools) = tools {
        params.insert("tools".into(), Value::from(tools));
        if has_request_tools && compat.zai_tool_stream {
            params.insert("tool_stream".into(), Value::from(true));
        }
    }
    if let Some(tool_choice) = options
        .tool_choice
        .as_ref()
        .filter(|choice| is_truthy(choice))
    {
        params.insert("tool_choice".into(), tool_choice.clone());
    }
    if let Some(priority) = compat.vllm_priority {
        params.insert("priority".into(), Value::from(priority));
    }

    let budget = thinking_budget(model, options, max_tokens);
    insert_thinking(&mut params, model, options.reasoning_effort, compat, budget);
    // A top-level budget caps reasoning independently of the format: reasoning
    // and the answer share max_tokens, so an uncapped reasoning phase can
    // leave no room for the answer.
    if let (Some(field), Some(budget)) = (compat.budget_field(), budget) {
        params.insert(field.as_str().into(), Value::from(budget));
    }
    insert_routing(&mut params, model);
    // Last, so model and request sampling parameters override named fields.
    let level = options
        .reasoning_effort
        .map_or(ModelThinkingLevel::Off, ModelThinkingLevel::from);
    if let Some(sampling) = resolve_sampling_params(model, level, base.sampling_params.as_ref()) {
        params.extend(sampling);
    }
    Ok(params)
}

/// `hasToolHistory`: the conversation has tool calls or tool results.
fn has_tool_history(messages: &[Message]) -> bool {
    messages.iter().any(|message| match message {
        Message::ToolResult(_) => true,
        Message::Assistant(assistant) => assistant
            .content
            .iter()
            .any(|block| matches!(block, AssistantContentBlock::ToolCall(_))),
        _ => false,
    })
}

/// `resolveClampedThinkingBudget`: the level's budget, leaving answer room
/// under the token ceiling; `None` when off or zero.
fn thinking_budget(
    model: &Model,
    options: &OpenAiCompletionsOptions,
    max_tokens: Option<u64>,
) -> Option<u64> {
    let level = options.reasoning_effort.filter(|_| model.reasoning)?;
    let budget = clamp_thinking_budget_to_answer_room(
        thinking_budget_for_level(level, options.thinking_budgets.as_ref()),
        max_tokens.unwrap_or(model.max_tokens),
    );
    (budget > 0).then_some(budget)
}

/// A `thinkingLevelMap` lookup with JavaScript's three outcomes.
#[derive(Clone, Copy)]
enum Mapped<'a> {
    /// No map or no entry (`undefined`).
    Absent,
    /// The level is unsupported (`null`).
    Null,
    /// The provider value.
    Value(&'a str),
}

impl<'a> Mapped<'a> {
    fn lookup(model: &'a Model, level: &str) -> Self {
        match model
            .thinking_level_map
            .as_ref()
            .and_then(|map| map.value_for(level))
        {
            None => Self::Absent,
            Some(ThinkingLevelMapValue::Unsupported) => Self::Null,
            Some(ThinkingLevelMapValue::Value(value)) => Self::Value(value),
        }
    }

    /// `map[level] ?? fallback`.
    const fn or(self, fallback: &'a str) -> &'a str {
        match self {
            Self::Value(value) => value,
            Self::Absent | Self::Null => fallback,
        }
    }

    /// `mapped === undefined ? fallback : mapped`, kept only when a string.
    const fn or_absent(self, fallback: Option<&'a str>) -> Option<&'a str> {
        match self {
            Self::Absent => fallback,
            Self::Null => None,
            Self::Value(value) => Some(value),
        }
    }
}

/// The reasoning fields of `buildParams` per thinking format; a
/// `reasoning_effort` always follows the format's own field.
fn insert_thinking(
    params: &mut Map<String, Value>,
    model: &Model,
    effort: Option<ThinkingLevel>,
    compat: &OpenAiCompletionsCompat,
    budget: Option<u64>,
) {
    if !model.reasoning {
        return;
    }
    let name = effort.map(ThinkingLevel::as_str);
    let lookup = |level: &str| Mapped::lookup(model, level);
    let off = lookup("off");
    // `map[effort] ?? effort`, where the provider takes `reasoning_effort`.
    let paired = name
        .filter(|_| compat.supports_reasoning_effort)
        .map(|level| lookup(level).or(level));
    let reasoning_effort = match compat.thinking_format {
        ThinkingFormat::Zai => {
            let thinking = if name.is_some() {
                json!({ "type": "enabled", "clear_thinking": false })
            } else {
                json!({ "type": "disabled" })
            };
            params.insert("thinking".into(), thinking);
            name.filter(|_| compat.supports_reasoning_effort)
                .and_then(|level| lookup(level).or_absent(Some(level)))
        }
        ThinkingFormat::Qwen => {
            params.insert("enable_thinking".into(), Value::from(name.is_some()));
            paired
        }
        ThinkingFormat::QwenChatTemplate => {
            params.insert(
                "chat_template_kwargs".into(),
                json!({ "enable_thinking": name.is_some(), "preserve_thinking": true }),
            );
            None
        }
        ThinkingFormat::ChatTemplate => {
            if let Some(values) =
                chat_template_values(model, name, &compat.chat_template_kwargs, budget)
            {
                params.insert("chat_template_kwargs".into(), Value::Object(values));
            }
            None
        }
        ThinkingFormat::Baseten => {
            if let Some(values) =
                chat_template_values(model, name, &compat.chat_template_args, budget)
            {
                params.insert("chat_template_args".into(), Value::Object(values));
            }
            compat
                .supports_reasoning_effort
                .then(|| name.map_or(off, lookup).or_absent(name))
                .flatten()
        }
        ThinkingFormat::Deepseek => {
            if name.is_some() {
                params.insert("thinking".into(), json!({ "type": "enabled" }));
            } else if !matches!(off, Mapped::Null) {
                params.insert("thinking".into(), json!({ "type": "disabled" }));
            }
            paired
        }
        ThinkingFormat::Together => {
            params.insert("reasoning".into(), json!({ "enabled": name.is_some() }));
            paired
        }
        // OpenRouter normalizes reasoning across providers.
        ThinkingFormat::Openrouter => {
            if let Some(value) = effort_or_off(name, lookup, off) {
                params.insert("reasoning".into(), json!({ "effort": value }));
            }
            None
        }
        ThinkingFormat::StringThinking => {
            if let Some(value) = effort_or_off(name, lookup, off) {
                params.insert("thinking".into(), Value::from(value));
            }
            None
        }
        // Without an effort, ant-ling falls through to the generic branch.
        ThinkingFormat::AntLing if name.is_some() => {
            if let Some(Mapped::Value(value)) = name.map(lookup) {
                params.insert("reasoning".into(), json!({ "effort": value }));
            }
            None
        }
        ThinkingFormat::Openai | ThinkingFormat::AntLing | ThinkingFormat::Other => {
            match (name, off) {
                (Some(_), _) => paired,
                (None, Mapped::Value(value)) if compat.supports_reasoning_effort => Some(value),
                (None, _) => None,
            }
        }
    };
    if let Some(value) = reasoning_effort {
        params.insert("reasoning_effort".into(), Value::from(value));
    }
}

/// The effort as mapped, or without one the `off` mapping (default
/// `"none"`) unless the map marks `off` unsupported.
fn effort_or_off<'a>(
    name: Option<&'a str>,
    lookup: impl Fn(&str) -> Mapped<'a>,
    off: Mapped<'a>,
) -> Option<&'a str> {
    match name {
        Some(level) => Some(lookup(level).or(level)),
        None if matches!(off, Mapped::Null) => None,
        None => Some(off.or("none")),
    }
}

/// `buildChatTemplateValues`: the configured values with pi variables
/// resolved; `None` when nothing is left.
fn chat_template_values(
    model: &Model,
    effort: Option<&str>,
    values: &Map<String, Value>,
    budget: Option<u64>,
) -> Option<Map<String, Value>> {
    let resolved: Map<String, Value> = values
        .iter()
        .filter_map(|(key, value)| {
            chat_template_value(model, effort, value, budget).map(|value| (key.clone(), value))
        })
        .collect();
    (!resolved.is_empty()).then_some(resolved)
}

/// `resolveChatTemplateKwargValue`: literals as given, `$var` objects
/// resolved; `None` omits the key.
fn chat_template_value(
    model: &Model,
    effort: Option<&str>,
    value: &Value,
    budget: Option<u64>,
) -> Option<Value> {
    let Value::Object(variable) = value else {
        return Some(value.clone());
    };
    if effort.is_none() && variable.get("omitWhenOff").is_some_and(is_truthy) {
        return None;
    }
    match variable.get("$var").and_then(Value::as_str) {
        Some("thinking.enabled") => Some(Value::from(effort.is_some())),
        Some("thinking.budget") => budget.map(Value::from),
        _ => {
            let mapped = Mapped::lookup(model, effort.unwrap_or("off"));
            mapped.or_absent(effort).map(Value::from)
        }
    }
}

/// Provider routing preferences, read from `model.compat` as given.
fn insert_routing(params: &mut Map<String, Value>, model: &Model) {
    let compat = model.compat.as_ref();
    if let Some(routing) = compat
        .and_then(|compat| compat.get("openRouterRouting"))
        .filter(|routing| is_truthy(routing))
    {
        params.insert("provider".into(), routing.clone());
    }
    let Some(routing) = compat
        .and_then(|compat| compat.get("vercelGatewayRouting"))
        .filter(|routing| is_truthy(routing))
    else {
        return;
    };
    let mut gateway = Map::new();
    for key in ["only", "order"] {
        if let Some(value) = routing.get(key).filter(|value| is_truthy(value)) {
            gateway.insert(key.into(), value.clone());
        }
    }
    if !gateway.is_empty() {
        params.insert("providerOptions".into(), json!({ "gateway": gateway }));
    }
}

/// `getCompatCacheControl`: the Anthropic-style breakpoint, if the compat
/// uses one.
fn cache_control(
    compat: &OpenAiCompletionsCompat,
    retention: CacheRetention,
    long: bool,
) -> Option<Value> {
    if !compat.anthropic_cache_control || retention == CacheRetention::None {
        return None;
    }
    Some(if long {
        json!({ "type": "ephemeral", "ttl": "1h" })
    } else {
        json!({ "type": "ephemeral" })
    })
}

/// `applyAnthropicCacheControl`: breakpoints on the first instruction
/// message, the last tool, and the last conversation message with text.
fn apply_cache_control(
    messages: &mut [Value],
    tools: Option<&mut Vec<Value>>,
    cache_control: &Value,
) {
    if let Some(first) = messages
        .iter_mut()
        .find(|message| matches!(role(message), Some("system" | "developer")))
    {
        mark_text(first, cache_control);
    }
    if let Some(Value::Object(last)) = tools.and_then(|tools| tools.last_mut()) {
        last.insert("cache_control".into(), cache_control.clone());
    }
    for message in messages.iter_mut().rev() {
        if matches!(role(message), Some("user" | "assistant" | "tool"))
            && mark_text(message, cache_control)
        {
            return;
        }
    }
}

fn role(message: &Value) -> Option<&str> {
    message.get("role").and_then(Value::as_str)
}

/// `addCacheControlToTextContent`: mark string content (as a text part) or
/// the last text part; false when there is no text to mark.
fn mark_text(message: &mut Value, cache_control: &Value) -> bool {
    let Some(content) = message.get_mut("content") else {
        return false;
    };
    match content {
        Value::String(text) if !text.is_empty() => {
            let text = std::mem::take(text);
            *content = json!([{ "type": "text", "text": text, "cache_control": cache_control }]);
            true
        }
        Value::Array(parts) => parts
            .iter_mut()
            .rev()
            .find(|part| part.get("type").and_then(Value::as_str) == Some("text"))
            .and_then(Value::as_object_mut)
            .map(|part| part.insert("cache_control".into(), cache_control.clone()))
            .is_some(),
        _ => false,
    }
}

/// The session affinity headers for `sessionId`, when the compat sends them.
fn session_affinity(
    session_id: Option<&str>,
    compat: &OpenAiCompletionsCompat,
) -> Map<String, Value> {
    let mut affinity = Map::new();
    if let Some(session_id) = session_id.filter(|_| compat.send_session_affinity_headers) {
        let names: &[&str] = match compat.session_affinity_format {
            SessionAffinityFormat::Openrouter => &["x-session-id"],
            SessionAffinityFormat::Openai => {
                &["session_id", "x-client-request-id", "x-session-affinity"]
            }
            SessionAffinityFormat::OpenaiNoSession => {
                &["x-client-request-id", "x-session-affinity"]
            }
        };
        for name in names {
            affinity.insert((*name).into(), Value::from(session_id));
        }
    }
    affinity
}
