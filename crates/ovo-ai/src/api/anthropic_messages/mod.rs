//! `anthropic-messages` — the upstream `anthropic-messages.ts` adapter.
//!
//! Request building only so far — streaming lands in checkpoint B.
//!
//! Ported so far (checkpoint A — request side):
//! - `build_request` — the `MessageCreateParamsStreaming` body built as
//!   typed structs in upstream object-literal key order and serialized
//!   with `json::to_vec` (byte-identical, ADR-011)
//! - `x-api-key` vs `Authorization: Bearer` (OAuth), `anthropic-version`,
//!   `anthropic-beta` composition, `mergeHeaders` model/provider/request
//! - `cache_control` per `cacheRetention` on system/tools/last blocks
//! - `transform_messages` + `normalize_tool_call_id` message conversion
//!
//! Not yet ported (checkpoint B / later):
//! - streaming (`streamAnthropic`, `iterateAnthropicEvents` gating,
//!   `decodeSseLine` event mapping onto `AssistantStreamSender`)
//! - thinking/effort (`insertThinkingLevelMessages`, `effort`,
//!   `output_config`, `thinking`/`thinkingLevelMap`)
//! - OAuth federation (`getAnthropicFederation` OIDC), copilot dynamic
//!   headers, `DEFERRED_TOOL_PLACEHOLDER`/strict-tools fallback,
//!   `onPayload`, `input_transformations`, `context_management`,
//!   `mcp_servers`, `container`, retry-family backoff decisions

use serde_json::{Map, Value, json};

use crate::json::to_vec;
use crate::options::StreamOptions;
use crate::types::context::Context;
use crate::types::model::Model;

/// `isOAuthToken` — upstream `apiKey.includes("sk-ant-oat")`.
#[must_use]
pub fn is_oauth_token(api_key: &str) -> bool {
    api_key.contains("sk-ant-oat")
}

/// `AnthropicMessagesCompat` — `getAnthropicCompat`'s typed view of
/// `model.compat`.
#[expect(
    clippy::struct_excessive_bools,
    reason = "mirrors upstream `getAnthropicCompat`'s flat boolean surface"
)]
#[derive(Clone, Copy, Debug)]
pub struct AnthropicMessagesCompat {
    /// `supportsEagerToolInputStreaming` — `eager_input_streaming: true` on tools.
    pub supports_eager_tool_input_streaming: bool,
    /// `supportsLongCacheRetention` — `ttl: "1h"` allowed.
    pub supports_long_cache_retention: bool,
    /// `supportsCacheControlOnTools` — `cache_control` on the last tool.
    pub supports_cache_control_on_tools: bool,
    /// `supportsTemperature` — `temperature` can be sent.
    pub supports_temperature: bool,
    /// `allowEmptySignature` — empty `signature` thinking stays a thinking block.
    pub allow_empty_signature: bool,
    /// `supportsStrictTools` — `strict` can be sent.
    pub supports_strict_tools: bool,
    /// `supportsMidConvoSystemMessages` — later system updates are native.
    pub supports_mid_convo_system_messages: bool,
    /// `supportsMidConvoToolChanges` — `tool_addition`/`tool_removal` blocks.
    pub supports_mid_convo_tool_changes: bool,
}

impl AnthropicMessagesCompat {
    /// `getAnthropicCompat(model)` — `compat` fields with upstream defaults.
    #[must_use]
    pub fn of(model: &Model) -> Self {
        let compat = model.compat.as_ref();
        let get = |key: &str, default: bool| {
            compat
                .and_then(|c| c.get(key))
                .and_then(Value::as_bool)
                .unwrap_or(default)
        };
        Self {
            supports_eager_tool_input_streaming: get("supportsEagerToolInputStreaming", true),
            supports_long_cache_retention: get("supportsLongCacheRetention", true),
            supports_cache_control_on_tools: get("supportsCacheControlOnTools", true),
            supports_temperature: get("supportsTemperature", true),
            allow_empty_signature: get("allowEmptySignature", false),
            supports_strict_tools: get("supportsStrictTools", false),
            supports_mid_convo_system_messages: get("supportsMidConvoSystemMessages", false),
            supports_mid_convo_tool_changes: get("supportsMidConvoToolChanges", false),
        }
    }
}

/// `AnthropicOptions` — `AnthropicOptions extends StreamOptions`.
#[derive(Clone, Default)]
#[non_exhaustive]
pub struct AnthropicOptions {
    /// `StreamOptions` base.
    pub base: StreamOptions,
    /// `thinkingEnabled`.
    pub thinking_enabled: Option<bool>,
    /// `thinkingBudgetTokens` (default 1024 when enabled).
    pub thinking_budget_tokens: Option<u64>,
    /// `effort` — `"low"|"medium"|"high"|"xhigh"|"max"`.
    pub effort: Option<String>,
    /// `thinkingDisplay` — `"summarized"|"omitted"` (default summarized).
    pub thinking_display: Option<String>,
    /// `interleavedThinking` (default true; skipped on adaptive models).
    pub interleaved_thinking: Option<bool>,
    /// `toolChoice` — `"auto"|"any"|"none"` or `{type:"tool",name}`.
    pub tool_choice: Option<Value>,
}

/// `resolveCacheRetention` — `options.cacheRetention` > `env.PI_CACHE_RETENTION == "long"` > `"short"`.
#[must_use]
pub fn resolve_cache_retention(
    cache_retention: Option<crate::options::CacheRetention>,
    env: Option<&Map<String, Value>>,
) -> &'static str {
    if let Some(r) = cache_retention {
        return match r {
            crate::options::CacheRetention::None => "none",
            crate::options::CacheRetention::Long => "long",
            crate::options::CacheRetention::Short => "short",
        };
    }
    if env
        .and_then(|e| e.get("PI_CACHE_RETENTION"))
        .and_then(Value::as_str)
        .is_some_and(|v| v == "long")
    {
        return "long";
    }
    "short"
}

/// `getCacheControl` — `{type:"ephemeral", ttl?}` per retention.
#[must_use]
pub fn get_cache_control(
    model: &Model,
    cache_retention: Option<crate::options::CacheRetention>,
    env: Option<&Map<String, Value>>,
) -> Option<Value> {
    let retention = resolve_cache_retention(cache_retention, env);
    if retention == "none" {
        return None;
    }
    let compat = AnthropicMessagesCompat::of(model);
    let ttl = if retention == "long" && compat.supports_long_cache_retention {
        Some("1h")
    } else {
        None
    };
    let mut cc = Map::new();
    cc.insert(String::from("type"), Value::from("ephemeral"));
    if let Some(ttl) = ttl {
        cc.insert(String::from("ttl"), Value::from(ttl));
    }
    Some(Value::from(cc))
}

/// `getBetaFeatures` — a caller-configured `anthropic-beta` header wins;
/// otherwise the adapter composes beta flags from model/options.
#[must_use]
pub fn get_beta_features(
    model: &Model,
    is_oauth_token: bool,
    options: Option<&AnthropicOptions>,
    tools_count: usize,
) -> Vec<String> {
    // A configured `anthropic-beta` header on model or options wins.
    for headers in [
        model.headers.as_ref(),
        options.and_then(|o| o.base.headers.as_ref()),
    ] {
        let Some(headers) = headers else {
            continue;
        };
        for (name, value) in headers {
            if !name.eq_ignore_ascii_case("anthropic-beta") {
                continue;
            }
            if let Some(text) = value.as_str() {
                return text
                    .split(',')
                    .map(str::trim)
                    .filter(|f| !f.is_empty())
                    .map(str::to_owned)
                    .collect();
            }
        }
    }
    let compat = AnthropicMessagesCompat::of(model);
    let mut features: Vec<String> = Vec::new();
    if is_oauth_token {
        features.extend([
            String::from("claude-code-20250219"),
            String::from("oauth-2025-04-20"),
        ]);
    }
    // `shouldUseFineGrainedToolStreamingBeta` — tools present and the api
    // doesn't stream arguments eagerly.
    if tools_count > 0 && !compat.supports_eager_tool_input_streaming {
        features.push(String::from(FINE_GRAINED_TOOL_STREAMING_BETA));
    }
    // interleaved thinking — reasoning model + `thinkingEnabled` + not
    // `forceAdaptiveThinking` (checkpoint B wires `options.thinkingEnabled`).
    if model.reasoning
        && options.is_some_and(|o| o.thinking_enabled.unwrap_or(false))
        && !model
            .compat
            .as_ref()
            .and_then(|c| c.get("forceAdaptiveThinking"))
            .and_then(Value::as_bool)
            .unwrap_or(false)
    {
        features.push(String::from(INTERLEAVED_THINKING_BETA));
    }
    if model
        .compat
        .as_ref()
        .and_then(|c| c.get("allowedFallbackModels"))
        .and_then(Value::as_array)
        .is_some_and(|a| !a.is_empty())
    {
        features.push(String::from(SERVER_SIDE_FALLBACK_BETA));
    }
    if model
        .compat
        .as_ref()
        .and_then(|c| c.get("supportsMidConvoEffort"))
        .and_then(Value::as_bool)
        .unwrap_or(false)
    {
        features.extend([
            String::from(MID_CONVERSATION_OUTPUT_CONFIG_BETA),
            String::from(THINKING_BINDING_CONTROLS_BETA),
        ]);
    }
    features
}

/// `buildRequest` — the `MessageCreateParamsStreaming` body in upstream
/// `buildParams` key order plus the request headers.
#[expect(
    clippy::too_many_lines,
    reason = "mirrors upstream `buildParams` — a flat ordered dispatch; splitting it obscures the key order"
)]
#[must_use]
pub fn build_request(
    model: &Model,
    context: &Context,
    options: Option<&AnthropicOptions>,
) -> (String, Map<String, Value>, Vec<u8>) {
    let options = options.cloned().unwrap_or_default();
    let base = options.base.clone();
    let compat = AnthropicMessagesCompat::of(model);
    // Upstream `resolveTranscript(context, ...)` reads `context.messages`
    // only — `context.systemPrompt`/`tools` do not reach the wire via
    // `normalize_context`'s fold. Mirror that: normalize with the prompt
    // and tools cleared so `params.system` comes from `messages` alone.
    let mut context = context.clone();
    context.system_prompt = None;
    context.tools = None;
    let transcript = crate::transcript::normalize_context(&context);
    let normalized = crate::transcript::resolve_transcript(
        &transcript,
        Some(compat.supports_mid_convo_system_messages),
    );
    let transformed = crate::transform::transform_messages(
        normalized.messages(),
        model,
        Some(&|id, _model, _assistant| convert::normalize_tool_call_id(id)),
    );
    let initial_system = crate::transcript::get_initial_system_message(&transformed);
    let initial_system_text = initial_system
        .map(crate::text::get_system_message_text)
        .unwrap_or_default();
    let transformed_messages = if initial_system.is_some() {
        transformed.get(1..).unwrap_or_default()
    } else {
        transformed.as_slice()
    };
    let cache_control = get_cache_control(model, base.cache_retention, base.env.as_ref());
    let messages = convert::convert_messages(
        transformed_messages,
        is_oauth_token(base.api_key.as_deref().unwrap_or_default()),
        compat.allow_empty_signature,
        None,
        cache_control.as_ref(),
    );
    let tools = crate::transcript::get_current_tools(&transformed);
    let tool_cache_control = if compat.supports_cache_control_on_tools {
        cache_control.as_ref()
    } else {
        None
    };
    let converted_tools = convert::convert_tools(
        &tools,
        is_oauth_token(base.api_key.as_deref().unwrap_or_default()),
        compat.supports_eager_tool_input_streaming,
        tool_cache_control,
        false,
    );
    let betas = get_beta_features(
        model,
        is_oauth_token(base.api_key.as_deref().unwrap_or_default()),
        Some(&options),
        tools.len(),
    );

    let mut params = Map::new();
    params.insert(String::from("model"), Value::from(model.id.clone()));
    params.insert(String::from("messages"), Value::from(messages));
    params.insert(
        String::from("max_tokens"),
        Value::from(base.max_tokens.unwrap_or(model.max_tokens)),
    );
    params.insert(String::from("stream"), Value::from(true));
    let betas_header = betas.join(",");
    if is_oauth_token(base.api_key.as_deref().unwrap_or_default()) {
        let mut system: Vec<Value> = vec![json!({
            "type": "text",
            "text": "You are Claude Code, Anthropic's official CLI for Claude.",
        })];
        if let (Some(cc), Some(obj)) = (
            &cache_control,
            system.first_mut().and_then(Value::as_object_mut),
        ) {
            obj.insert(String::from("cache_control"), cc.clone());
        }
        if !initial_system_text.is_empty() {
            let mut block = json!({
                "type": "text",
                "text": crate::text::sanitize_surrogates(&initial_system_text),
            });
            if let (Some(cc), Some(obj)) = (&cache_control, block.as_object_mut()) {
                obj.insert(String::from("cache_control"), cc.clone());
            }
            system.push(block);
        }
        params.insert(String::from("system"), Value::from(system));
    } else if !initial_system_text.is_empty() {
        let mut block = json!({
            "type": "text",
            "text": crate::text::sanitize_surrogates(&initial_system_text),
        });
        if let (Some(cc), Some(obj)) = (&cache_control, block.as_object_mut()) {
            obj.insert(String::from("cache_control"), cc.clone());
        }
        params.insert(String::from("system"), Value::from(vec![block]));
    }
    if base.temperature.is_some()
        && compat.supports_temperature
        && !model
            .compat
            .as_ref()
            .and_then(|c| c.get("supportsMidConvoEffort"))
            .and_then(Value::as_bool)
            .unwrap_or(false)
    {
        params.insert(
            String::from("temperature"),
            base.temperature.map_or(Value::Null, Value::from),
        );
    }
    if !converted_tools.is_empty() {
        params.insert(String::from("tools"), Value::from(converted_tools));
    }
    let body = to_vec(&Value::from(params)).unwrap_or_default();

    // Headers — `anthropic-version` + `anthropic-beta` (joined) +
    // `x-api-key` or `Authorization: Bearer` + model/provider/request
    // headers merged in order.
    let mut headers = Map::new();
    headers.insert(String::from("anthropic-version"), Value::from("2023-06-01"));
    if !betas_header.is_empty() {
        headers.insert(String::from("anthropic-beta"), Value::from(betas_header));
    }
    let api_key = base.api_key.as_deref().unwrap_or_default();
    if !api_key.is_empty() {
        if is_oauth_token(api_key) {
            headers.insert(
                String::from("authorization"),
                Value::from(format!("Bearer {api_key}")),
            );
        } else {
            headers.insert(String::from("x-api-key"), Value::from(api_key));
        }
    }
    (model.base_url.clone(), headers, body)
}

/// `Api` constant — `"anthropic-messages"`.
pub const API: &str = "anthropic-messages";

const FINE_GRAINED_TOOL_STREAMING_BETA: &str = "fine-grained-tool-streaming-2025-05-14";
const INTERLEAVED_THINKING_BETA: &str = "interleaved-thinking-2025-05-14";
const SERVER_SIDE_FALLBACK_BETA: &str = "server-side-fallback-2026-07-01";
const MID_CONVERSATION_OUTPUT_CONFIG_BETA: &str = "mid-conversation-output-config-2026-07-01";
const THINKING_BINDING_CONTROLS_BETA: &str = "thinking-binding-controls-2026-08-01";
pub mod convert;
#[cfg(feature = "http")]
pub mod stream;

#[cfg(feature = "http")]
pub use stream::AnthropicMessagesApi;

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
