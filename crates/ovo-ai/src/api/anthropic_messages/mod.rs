//! The `anthropic-messages` adapter (upstream `api/anthropic-messages.ts`).
//!
//! [`build_request`](crate::api::anthropic_messages::build_request) ports `buildParams` and `createClient`: the request body
//! in upstream key order (serialized with `ovo_ai::json`, byte-identical) and
//! the headers in the SDK's merge order. The streaming side lives in
//! the `stream` module (feature `http`).
//!
//! Not yet ported (M5): workload identity federation, GitHub Copilot
//! dynamic headers.

use serde_json::{Map, Value, json};

use crate::api::constrained_sampling::prefers_json_schema_strict_sampling;
use crate::options::{
    CacheRetention, SimpleStreamOptions, StreamOptions, build_base_options, resolve_cache_retention,
};
use crate::simple_options::{adjust_max_tokens_for_thinking, clamp_max_tokens_to_context};
use crate::transcript::{TranscriptContext, get_current_tools, get_initial_system_message};
use crate::types::message::Message;
use crate::types::model::{Model, ThinkingLevel, ThinkingLevelMapValue};
use crate::types::tool::{Tool, ToolChoice};

pub mod convert;
#[cfg(feature = "http")]
pub mod stream;

#[cfg(feature = "http")]
pub use stream::AnthropicMessagesApi;

/// The api identifier this adapter serves.
pub const API: &str = "anthropic-messages";

const CLAUDE_CODE_VERSION: &str = "2.1.280";
const FINE_GRAINED_TOOL_STREAMING_BETA: &str = "fine-grained-tool-streaming-2025-05-14";
const INTERLEAVED_THINKING_BETA: &str = "interleaved-thinking-2025-05-14";
const SERVER_SIDE_FALLBACK_BETA: &str = "server-side-fallback-2026-07-01";
const MID_CONVERSATION_OUTPUT_CONFIG_BETA: &str = "mid-conversation-output-config-2026-07-01";
const THINKING_BINDING_CONTROLS_BETA: &str = "thinking-binding-controls-2026-08-01";
const INLINE_TOOLS_BETA: &str = "inline-tools-2026-09-15";

/// `isOAuthToken`: Claude subscription tokens use bearer auth.
#[must_use]
pub fn is_oauth_token(api_key: &str) -> bool {
    api_key.contains("sk-ant-oat")
}

/// `getAnthropicCompat` plus the other `model.compat` fields the adapter reads.
#[expect(
    clippy::struct_excessive_bools,
    reason = "mirrors upstream's flat boolean compat surface"
)]
#[derive(Clone, Debug)]
pub struct AnthropicMessagesCompat {
    /// `eager_input_streaming: true` on tools.
    pub supports_eager_tool_input_streaming: bool,
    /// `ttl: "1h"` allowed on cache breakpoints.
    pub supports_long_cache_retention: bool,
    /// `cache_control` allowed on the last tool.
    pub supports_cache_control_on_tools: bool,
    /// `temperature` may be sent.
    pub supports_temperature: bool,
    /// Thinking blocks with an empty signature stay thinking blocks.
    pub allow_empty_signature: bool,
    /// Tools may be sent in strict mode.
    pub supports_strict_tools: bool,
    /// Later system messages are sent natively.
    pub supports_mid_convo_system_messages: bool,
    /// Tool changes are sent as `tool_addition`/`tool_removal` blocks.
    pub supports_mid_convo_tool_changes: bool,
    /// The model only supports adaptive thinking.
    pub force_adaptive_thinking: bool,
    /// Effort is set per reply through effort markers.
    pub supports_mid_convo_effort: bool,
    /// `sendSessionAffinityHeaders` — default: `OpenRouter` endpoints only.
    pub send_session_affinity_headers: bool,
    /// `sessionAffinityFormat` — `openrouter` sends `x-session-id`; anything
    /// else sends `x-session-affinity`. Default: `openrouter` on `OpenRouter`.
    pub session_affinity_format: Option<String>,
    /// Models the provider may answer with instead (`allowedFallbackModels`).
    pub allowed_fallback_models: Vec<String>,
}

impl AnthropicMessagesCompat {
    /// The compat view of `model` with upstream defaults.
    #[must_use]
    pub fn of(model: &Model) -> Self {
        let compat = model.compat.as_ref();
        let flag = |key: &str, default: bool| {
            compat
                .and_then(|c| c.get(key))
                .and_then(Value::as_bool)
                .unwrap_or(default)
        };
        let is_openrouter =
            model.provider.as_str() == "openrouter" || model.base_url.contains("openrouter.ai");
        let allowed_fallback_models = compat
            .and_then(|c| c.get("allowedFallbackModels"))
            .and_then(Value::as_array)
            .map(|fallbacks| {
                fallbacks
                    .iter()
                    .filter_map(|fallback| fallback.get("model").and_then(Value::as_str))
                    .map(str::to_owned)
                    .collect()
            })
            .unwrap_or_default();
        Self {
            supports_eager_tool_input_streaming: flag("supportsEagerToolInputStreaming", true),
            supports_long_cache_retention: flag("supportsLongCacheRetention", true),
            supports_cache_control_on_tools: flag("supportsCacheControlOnTools", true),
            supports_temperature: flag("supportsTemperature", true),
            allow_empty_signature: flag("allowEmptySignature", false),
            supports_strict_tools: flag("supportsStrictTools", false),
            supports_mid_convo_system_messages: flag("supportsMidConvoSystemMessages", false),
            supports_mid_convo_tool_changes: flag("supportsMidConvoToolChanges", false),
            force_adaptive_thinking: flag("forceAdaptiveThinking", false),
            supports_mid_convo_effort: flag("supportsMidConvoEffort", false),
            send_session_affinity_headers: flag("sendSessionAffinityHeaders", is_openrouter),
            session_affinity_format: compat
                .and_then(|c| c.get("sessionAffinityFormat"))
                .and_then(Value::as_str)
                .map(str::to_owned)
                .or_else(|| is_openrouter.then(|| String::from("openrouter"))),
            allowed_fallback_models,
        }
    }
}

/// `AnthropicOptions`: the stream options plus Anthropic thinking controls.
#[derive(Clone, Default)]
#[non_exhaustive]
pub struct AnthropicOptions {
    /// The shared stream options.
    pub base: StreamOptions,
    /// Enable extended thinking.
    pub thinking_enabled: Option<bool>,
    /// Thinking budget for budget-based models (1024 when unset).
    pub thinking_budget_tokens: Option<u64>,
    /// Effort for adaptive thinking: `low`, `medium`, `high`, `xhigh` or `max`.
    pub effort: Option<String>,
    /// How thinking is returned: `summarized` (default) or `omitted`.
    pub thinking_display: Option<String>,
    /// Request the interleaved thinking beta on budget-based models (default true).
    pub interleaved_thinking: Option<bool>,
    /// `auto`, `any`, `none`, or `{ "type": "tool", "name": ... }`.
    pub tool_choice: Option<Value>,
}

impl std::fmt::Debug for AnthropicOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AnthropicOptions")
            .field("thinking_enabled", &self.thinking_enabled)
            .field("thinking_budget_tokens", &self.thinking_budget_tokens)
            .field("effort", &self.effort)
            .field("thinking_display", &self.thinking_display)
            .field("interleaved_thinking", &self.interleaved_thinking)
            .field("tool_choice", &self.tool_choice)
            .finish_non_exhaustive()
    }
}

impl AnthropicOptions {
    /// `streamSimple`'s mapping: no reasoning disables thinking, adaptive
    /// models get an effort, older models a thinking budget within
    /// `max_tokens`.
    #[must_use]
    pub fn from_simple(
        model: &Model,
        context: &TranscriptContext,
        options: &SimpleStreamOptions,
    ) -> Self {
        let base = build_base_options(
            model,
            context,
            Some(options),
            options.base.api_key.as_deref(),
        );
        let tool_choice = options.tool_choice.map(|choice| {
            Value::from(match choice {
                ToolChoice::Auto => "auto",
                ToolChoice::None => "none",
            })
        });
        let mut anthropic = Self {
            base,
            tool_choice,
            ..Self::default()
        };
        let Some(level) = options.reasoning else {
            anthropic.thinking_enabled = Some(false);
            return anthropic;
        };
        anthropic.thinking_enabled = Some(true);
        if AnthropicMessagesCompat::of(model).force_adaptive_thinking {
            anthropic.effort = Some(effort_for_level(model, level));
            return anthropic;
        }
        // An unset max_tokens lets the helper use the model cap.
        let adjusted = adjust_max_tokens_for_thinking(
            anthropic.base.max_tokens,
            model.max_tokens,
            level,
            options.thinking_budgets.as_ref(),
        );
        let max_tokens = clamp_max_tokens_to_context(model, context, adjusted.max_tokens);
        anthropic.base.max_tokens = Some(max_tokens);
        anthropic.thinking_budget_tokens = Some(
            adjusted
                .thinking_budget
                .min(max_tokens.saturating_sub(1024)),
        );
        anthropic
    }
}

/// `mapThinkingLevelToEffort`: the model's own mapping, else the nearest
/// Anthropic effort.
fn effort_for_level(model: &Model, level: ThinkingLevel) -> String {
    let name = match level {
        ThinkingLevel::Minimal => "minimal",
        ThinkingLevel::Low => "low",
        ThinkingLevel::Medium => "medium",
        ThinkingLevel::High => "high",
        ThinkingLevel::Xhigh => "xhigh",
        ThinkingLevel::Max => "max",
    };
    if let Some(ThinkingLevelMapValue::Value(mapped)) = model
        .thinking_level_map
        .as_ref()
        .and_then(|map| map.value_for(name))
    {
        return mapped.clone();
    }
    String::from(match level {
        ThinkingLevel::Minimal | ThinkingLevel::Low => "low",
        ThinkingLevel::Medium => "medium",
        _ => "high",
    })
}

/// `getCacheControl`: the ephemeral breakpoint for the retention, if any.
#[must_use]
pub fn get_cache_control(
    model: &Model,
    cache_retention: Option<CacheRetention>,
    env: Option<&Map<String, Value>>,
) -> Option<Value> {
    match resolve_cache_retention(cache_retention, env) {
        CacheRetention::None => None,
        CacheRetention::Long
            if AnthropicMessagesCompat::of(model).supports_long_cache_retention =>
        {
            Some(json!({ "type": "ephemeral", "ttl": "1h" }))
        }
        _ => Some(json!({ "type": "ephemeral" })),
    }
}

/// `getBetaFeatures`: a configured `anthropic-beta` header (last one wins,
/// `null` disables) replaces the composed list.
#[must_use]
pub fn get_beta_features(
    model: &Model,
    messages: &[Message],
    is_oauth_token: bool,
    native_tool_changes: bool,
    options: &AnthropicOptions,
) -> Vec<String> {
    let mut configured: Option<&Value> = None;
    for headers in [model.headers.as_ref(), options.base.headers.as_ref()]
        .into_iter()
        .flatten()
    {
        for (name, value) in headers {
            if name.eq_ignore_ascii_case("anthropic-beta") {
                configured = Some(value);
            }
        }
    }
    match configured {
        Some(Value::String(text)) => {
            return unique(
                text.split(',')
                    .map(str::trim)
                    .filter(|feature| !feature.is_empty())
                    .map(str::to_owned),
            );
        }
        Some(_) => return Vec::new(),
        None => {}
    }
    let compat = AnthropicMessagesCompat::of(model);
    let mut features: Vec<String> = Vec::new();
    if is_oauth_token {
        features.extend([
            String::from("claude-code-20250219"),
            String::from("oauth-2025-04-20"),
        ]);
    }
    if !get_current_tools(messages).is_empty() && !compat.supports_eager_tool_input_streaming {
        features.push(FINE_GRAINED_TOOL_STREAMING_BETA.to_owned());
    }
    if model.reasoning
        && options.thinking_enabled == Some(true)
        && options.interleaved_thinking.unwrap_or(true)
        && !compat.force_adaptive_thinking
    {
        features.push(INTERLEAVED_THINKING_BETA.to_owned());
    }
    if !compat.allowed_fallback_models.is_empty() {
        features.push(SERVER_SIDE_FALLBACK_BETA.to_owned());
    }
    if compat.supports_mid_convo_effort {
        features.extend([
            MID_CONVERSATION_OUTPUT_CONFIG_BETA.to_owned(),
            THINKING_BINDING_CONTROLS_BETA.to_owned(),
        ]);
    }
    if native_tool_changes {
        features.push(INLINE_TOOLS_BETA.to_owned());
    }
    unique(features)
}

/// First-seen order without duplicates, like `[...new Set(list)]`.
fn unique(items: impl IntoIterator<Item = String>) -> Vec<String> {
    let mut seen: Vec<String> = Vec::new();
    for item in items {
        if !seen.contains(&item) {
            seen.push(item);
        }
    }
    seen
}

/// One request to the Messages endpoint, ready to send.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AnthropicRequest {
    /// `{baseUrl}/v1/messages?beta=true` (the SDK's beta messages path).
    pub url: String,
    /// Request headers after the SDK and `createClient` merge order.
    pub headers: Map<String, Value>,
    /// The JSON body, byte-identical to upstream's `buildParams` output.
    pub body: Vec<u8>,
    /// The request params before serialization — `onPayload` sees and
    /// may replace these.
    pub params: Map<String, Value>,
    /// Whether a strict tool is only strict because it prefers it, so a
    /// strict-limit rejection can be retried without strict mode.
    pub has_preferred_strict_tools: bool,
    /// Effort of the final effort marker, recorded on the reply.
    pub provider_thinking_level: Option<String>,
}

/// `buildParams` and `createClient`: the request body in upstream key order
/// and the request headers.
///
/// `disable_preferred_strict_tools` drops strict mode from tools that only
/// prefer it (the retry after a strict-limit rejection).
///
/// # Errors
/// Returns upstream's message when a tool requires strict sampling that
/// cannot be provided, or the body fails to serialize.
pub fn build_request(
    model: &Model,
    context: &TranscriptContext,
    options: &AnthropicOptions,
    disable_preferred_strict_tools: bool,
) -> Result<AnthropicRequest, String> {
    let base = &options.base;
    let compat = AnthropicMessagesCompat::of(model);
    let oauth = is_oauth_token(base.api_key.as_deref().unwrap_or_default());
    let transcript = crate::transcript::resolve_transcript(
        context,
        Some(compat.supports_mid_convo_system_messages),
    );
    let cache_control = get_cache_control(model, base.cache_retention, base.env.as_ref());
    let messages = transcript.messages();
    let transformed = crate::transform::transform_messages(
        messages,
        model,
        Some(&|id, _model, _assistant| convert::normalize_tool_call_id(id)),
    );
    let initial_system = get_initial_system_message(messages);
    let initial_system_text = initial_system
        .map(crate::text::get_system_message_text)
        .unwrap_or_default();
    let conversation = if initial_system.is_some() {
        transformed.get(1..).unwrap_or_default()
    } else {
        transformed.as_slice()
    };

    let tool_definitions = |tools: &[Tool], tool_cache_control: Option<&Value>| {
        convert::convert_tools(
            tools,
            &compat,
            oauth,
            tool_cache_control,
            disable_preferred_strict_tools,
        )
    };
    let initial_tools: &[Tool] = initial_system
        .and_then(|system| system.tools_added.as_deref())
        .unwrap_or_default();
    let native_tool_changes = compat.supports_mid_convo_system_messages
        && compat.supports_mid_convo_tool_changes
        && !initial_tools.is_empty();
    let thinking_disabled = model.reasoning
        && options.thinking_enabled == Some(false)
        && !matches!(
            model
                .thinking_level_map
                .as_ref()
                .and_then(|map| map.value_for("off")),
            Some(ThinkingLevelMapValue::Unsupported)
        );
    // Effort cannot change between messages while thinking is disabled.
    let effort_messages = compat.supports_mid_convo_effort && !thinking_disabled;
    let native_definitions = |tools: &[Tool]| tool_definitions(tools, None);
    let native: convert::ToolDefinitions<'_> = &native_definitions;
    let converted = convert::convert_messages(
        conversation,
        oauth,
        cache_control.as_ref(),
        compat.allow_empty_signature,
        effort_messages.then_some(model.provider.as_str()),
        native_tool_changes.then_some(native),
    )?;
    let active_effort = options
        .effort
        .clone()
        .unwrap_or_else(|| String::from("high"));
    let betas = get_beta_features(model, messages, oauth, native_tool_changes, options);

    let mut params = Map::new();
    params.insert("model".into(), Value::from(model.id.clone()));
    let messages_param = if effort_messages {
        insert_thinking_level_messages(converted, &active_effort)
    } else {
        converted.messages
    };
    params.insert("messages".into(), Value::from(messages_param));
    params.insert(
        "max_tokens".into(),
        Value::from(base.max_tokens.unwrap_or(model.max_tokens)),
    );
    params.insert("stream".into(), Value::from(true));
    if let Some(system) = system_param(oauth, &initial_system_text, cache_control.as_ref()) {
        params.insert("system".into(), system);
    }
    // Temperature is incompatible with extended thinking and unsupported on
    // managed-effort models.
    if let Some(temperature) = base.temperature
        && options.thinking_enabled != Some(true)
        && !compat.supports_mid_convo_effort
        && compat.supports_temperature
    {
        params.insert("temperature".into(), Value::from(temperature));
    }

    let tool_cache_control = cache_control
        .as_ref()
        .filter(|_| compat.supports_cache_control_on_tools);
    let declared: Vec<Tool> = if native_tool_changes {
        initial_tools.to_vec()
    } else {
        get_current_tools(messages)
    };
    let has_preferred_strict_tools = insert_tools(
        &mut params,
        &declared,
        tool_definitions(&declared, tool_cache_control)?,
        native_tool_changes,
    );
    insert_thinking(
        &mut params,
        model,
        options,
        &compat,
        thinking_disabled,
        effort_messages,
    );
    insert_request_controls(&mut params, options, &compat);

    assemble_request(
        model,
        base,
        &compat,
        &betas.join(","),
        params,
        has_preferred_strict_tools,
        effort_messages.then_some(active_effort),
    )
}

/// Serialize `params` and finish the request — `url`/`headers`/`body`.
fn assemble_request(
    model: &Model,
    base: &StreamOptions,
    compat: &AnthropicMessagesCompat,
    betas: &str,
    params: Map<String, Value>,
    has_preferred_strict_tools: bool,
    provider_thinking_level: Option<String>,
) -> Result<AnthropicRequest, String> {
    let body =
        crate::json::to_vec(&Value::Object(params.clone())).map_err(|error| error.to_string())?;
    let url = format!(
        "{}/v1/messages?beta=true",
        model.base_url.trim_end_matches('/')
    );
    // `cacheRetention: none` suppresses the session-affinity headers like
    // upstream's `cacheSessionId` (`resolveCacheRetention` at stream start).
    let session_id = match resolve_cache_retention(base.cache_retention, base.env.as_ref()) {
        CacheRetention::None => None,
        _ => base.session_id.as_deref(),
    };
    let headers = request_headers(model, base, betas, session_id, compat);
    Ok(AnthropicRequest {
        url,
        headers,
        body,
        params,
        has_preferred_strict_tools,
        provider_thinking_level,
    })
}

/// `metadata.user_id`, `tool_choice` and the allowed `fallbacks`.
fn insert_request_controls(
    params: &mut Map<String, Value>,
    options: &AnthropicOptions,
    compat: &AnthropicMessagesCompat,
) {
    if let Some(user_id) = options
        .base
        .metadata
        .as_ref()
        .and_then(|metadata| metadata.get("user_id"))
        .and_then(Value::as_str)
    {
        params.insert("metadata".into(), json!({ "user_id": user_id }));
    }
    if let Some(tool_choice) = &options.tool_choice {
        let choice = match tool_choice {
            Value::String(kind) => json!({ "type": kind }),
            other => other.clone(),
        };
        params.insert("tool_choice".into(), choice);
    }
    if !compat.allowed_fallback_models.is_empty() {
        let fallbacks: Vec<Value> = compat
            .allowed_fallback_models
            .iter()
            .map(|fallback| json!({ "model": fallback }))
            .collect();
        params.insert("fallbacks".into(), Value::from(fallbacks));
    }
}

/// `tools`: the declared definitions, plus the deferred placeholder when tool
/// changes are native. Returns whether a strict tool only prefers strict mode.
fn insert_tools(
    params: &mut Map<String, Value>,
    declared: &[Tool],
    mut definitions: Vec<Value>,
    native_tool_changes: bool,
) -> bool {
    if declared.is_empty() {
        return false;
    }
    let has_preferred_strict_tools = definitions.iter().zip(declared).any(|(definition, tool)| {
        definition.get("strict") == Some(&Value::Bool(true))
            && prefers_json_schema_strict_sampling(tool)
    });
    if native_tool_changes {
        definitions.push(deferred_tool_placeholder());
    }
    params.insert("tools".into(), Value::from(definitions));
    has_preferred_strict_tools
}

/// The `system` blocks: OAuth requests must open with the Claude Code identity.
fn system_param(oauth: bool, text: &str, cache_control: Option<&Value>) -> Option<Value> {
    let block = |content: &str| {
        let mut block =
            json!({ "type": "text", "text": crate::text::sanitize_surrogates(content) });
        if let (Some(cache_control), Some(object)) = (cache_control, block.as_object_mut()) {
            object.insert("cache_control".into(), cache_control.clone());
        }
        block
    };
    let mut blocks = Vec::new();
    if oauth {
        blocks.push(block(
            "You are Claude Code, Anthropic's official CLI for Claude.",
        ));
    }
    if !text.is_empty() {
        blocks.push(block(text));
    }
    (!blocks.is_empty()).then(|| Value::from(blocks))
}

/// `thinking` and `output_config` per the model's thinking mode.
fn insert_thinking(
    params: &mut Map<String, Value>,
    model: &Model,
    options: &AnthropicOptions,
    compat: &AnthropicMessagesCompat,
    thinking_disabled: bool,
    effort_messages: bool,
) {
    let display = options.thinking_display.as_deref().unwrap_or("summarized");
    if thinking_disabled {
        // Anthropic rejects `block_binding` with disabled thinking.
        params.insert("thinking".into(), json!({ "type": "disabled" }));
    } else if effort_messages {
        // Effort markers set every reply's level; the top-level effort is part
        // of the cached prompt and stays "low" because organizations can cap
        // effort and reject any level above the cap.
        params.insert(
            "thinking".into(),
            json!({
                "type": "adaptive",
                "display": display,
                "block_binding": { "prefix_mismatch_behavior": "drop_block" },
            }),
        );
        params.insert("output_config".into(), json!({ "effort": "low" }));
    } else if model.reasoning && options.thinking_enabled == Some(true) {
        if compat.force_adaptive_thinking {
            params.insert(
                "thinking".into(),
                json!({ "type": "adaptive", "display": display }),
            );
            if let Some(effort) = &options.effort {
                params.insert("output_config".into(), json!({ "effort": effort }));
            }
        } else {
            let budget = options
                .thinking_budget_tokens
                .filter(|budget| *budget > 0)
                .unwrap_or(1024);
            params.insert(
                "thinking".into(),
                json!({ "type": "enabled", "budget_tokens": budget, "display": display }),
            );
        }
    }
}

/// `insertThinkingLevelMessages`: an effort marker before each assistant
/// message that recorded one, and the active effort at the end.
fn insert_thinking_level_messages(
    converted: convert::ConvertedMessages,
    active_effort: &str,
) -> Vec<Value> {
    let marker = |effort: &str| json!({ "role": "system", "content": [], "output_config": { "effort": effort } });
    let mut messages = Vec::with_capacity(converted.messages.len() + 1);
    for (index, message) in converted.messages.into_iter().enumerate() {
        if let Some((_, effort)) = converted
            .assistant_levels
            .iter()
            .find(|(position, _)| *position == index)
        {
            messages.push(marker(effort));
        }
        messages.push(message);
    }
    messages.push(marker(active_effort));
    messages
}

/// The never-activated deferred tool that keeps Anthropic's tool-change
/// scaffolding in the cached prefix from the first request.
fn deferred_tool_placeholder() -> Value {
    json!({
        "name": "__pi_deferred_placeholder__",
        "description": "Reserved placeholder. Never available. Never call this.",
        "input_schema": { "type": "object", "properties": {}, "required": [] },
        "defer_loading": true,
    })
}

/// The request headers in the order the Anthropic SDK applies them: its own
/// content type, version and auth first, then pi's `defaultHeaders`
/// (`createClient`: identity or Claude Code headers, then `model.headers`,
/// then the request headers), then the per-request `anthropic-beta`. Later
/// sources win case-insensitively and `null` removes a header.
fn request_headers(
    model: &Model,
    base: &StreamOptions,
    betas: &str,
    session_id: Option<&str>,
    compat: &AnthropicMessagesCompat,
) -> Map<String, Value> {
    let api_key = base.api_key.as_deref().unwrap_or_default();
    let oauth = is_oauth_token(api_key);
    let mut sdk = Map::new();
    sdk.insert("content-type".into(), Value::from("application/json"));
    sdk.insert("anthropic-version".into(), Value::from("2023-06-01"));
    if oauth {
        sdk.insert(
            "authorization".into(),
            Value::from(format!("Bearer {api_key}")),
        );
    } else if !api_key.is_empty() {
        sdk.insert("x-api-key".into(), Value::from(api_key));
    }
    let mut defaults = Map::new();
    if !oauth {
        defaults.insert(
            "User-Agent".into(),
            Value::from(crate::identity::format_user_agent(
                base.identity.as_ref(),
                crate::identity::OVO_PRODUCT,
                crate::identity::OVO_VERSION,
            )),
        );
    }
    defaults.insert("accept".into(), Value::from("application/json"));
    defaults.insert(
        "anthropic-dangerous-direct-browser-access".into(),
        Value::from("true"),
    );
    if oauth {
        defaults.insert(
            "user-agent".into(),
            Value::from(format!("claude-cli/{CLAUDE_CODE_VERSION}")),
        );
        defaults.insert("x-app".into(), Value::from("cli"));
    }
    let mut affinity = Map::new();
    if let (Some(id), true) = (session_id, compat.send_session_affinity_headers) {
        let header = if compat.session_affinity_format.as_deref() == Some("openrouter") {
            "x-session-id"
        } else {
            "x-session-affinity"
        };
        affinity.insert(header.into(), Value::from(id));
    }
    let mut beta = Map::new();
    if !betas.is_empty() {
        beta.insert("anthropic-beta".into(), Value::from(betas));
    }
    crate::http::headers::provider_headers_to_record(&[
        Some(&sdk),
        Some(&defaults),
        Some(&affinity),
        model.headers.as_ref(),
        base.headers.as_ref(),
        Some(&beta),
    ])
    .unwrap_or_default()
}
