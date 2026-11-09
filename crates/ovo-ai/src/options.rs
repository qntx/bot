//! `SimpleStreamOptions`/`StreamOptions` and `buildBaseOptions`
//! (`api/simple-options.ts`).
//!
//! Data only — `onPayload`/`onResponse`/`onProviderStreamEvent`/
//! `signal`/`fetch` arrive with the adapters and the transport
//! (M1-08 B/C).

use serde_json::{Map, Value};

use crate::http::headers::ProviderHeaders;
use crate::identity::ClientIdentity;
use crate::simple_options::{clamp_max_tokens_to_context, resolve_sampling_params};
use crate::transcript::TranscriptContext;
use crate::types::model::{Model, ModelThinkingLevel, ThinkingLevel};

/// `ProviderEnv = Record<string, string>`.
pub type ProviderEnv = Map<String, Value>;

/// `StreamOptions` — every data field upstream carries.
#[derive(Clone, Debug, Default)]
#[non_exhaustive]
pub struct StreamOptions {
    /// `temperature`.
    pub temperature: Option<f64>,
    /// `samplingParams` — provider-specific keys merged into the body.
    pub sampling_params: Option<Map<String, Value>>,
    /// `maxTokens`.
    pub max_tokens: Option<u64>,
    /// `apiKey`.
    pub api_key: Option<String>,
    /// `env` — provider-scoped environment overrides.
    pub env: Option<ProviderEnv>,
    /// `headers` — `null` suppresses a default header.
    pub headers: Option<ProviderHeaders>,
    /// `identity` — `User-Agent` / attribution.
    pub identity: Option<ClientIdentity>,
    /// `timeoutMs`.
    pub timeout_ms: Option<u64>,
    /// `websocketConnectTimeoutMs`.
    pub websocket_connect_timeout_ms: Option<u64>,
    /// `maxRetries` — SDK retry budget (default 2 upstream).
    pub max_retries: Option<u32>,
    /// `maxRetryDelayMs` — cap on server-requested delay (0 disables).
    pub max_retry_delay_ms: Option<f64>,
    /// `metadata` — provider-interpreted request metadata.
    pub metadata: Option<Map<String, Value>>,
    /// `transport`.
    pub transport: Option<Transport>,
    /// `cacheRetention`.
    pub cache_retention: Option<CacheRetention>,
    /// `sessionId`.
    pub session_id: Option<String>,
}

/// `Transport`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Transport {
    /// `sse`.
    Sse,
    /// `websocket`.
    Websocket,
    /// `websocket-cached`.
    WebsocketCached,
    /// `auto`.
    Auto,
}

/// `CacheRetention`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CacheRetention {
    /// `short`.
    Short,
    /// `long`.
    Long,
}

/// `ToolChoice = "auto" | "none"`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ToolChoice {
    /// `auto`.
    Auto,
    /// `none`.
    None,
}

/// `SimpleStreamOptions`: `StreamOptions` plus the simple-request
/// fields.
#[derive(Clone, Debug, Default)]
#[non_exhaustive]
pub struct SimpleStreamOptions {
    /// The underlying `StreamOptions`.
    pub base: StreamOptions,
    /// `toolChoice`.
    pub tool_choice: Option<ToolChoice>,
    /// `reasoning`.
    pub reasoning: Option<ThinkingLevel>,
    /// `deferred`.
    pub deferred: Option<Deferred>,
    /// `thinkingBudgets`.
    pub thinking_budgets: Option<crate::simple_options::ThinkingBudgets>,
}

/// `deferred?: boolean | { window?: "15m" | "1h" | "24h" }`.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(untagged)]
pub enum Deferred {
    /// `true`/`false`.
    Flag(bool),
    /// `{ window?: ... }`.
    Window {
        /// The durability window.
        window: Option<String>,
    },
}

/// `buildBaseOptions(model, context, options?, apiKey?)`.
///
/// Merges simple options into `StreamOptions`, resolving sampling
/// params and clamping `maxTokens` to context.
#[must_use]
pub fn build_base_options(
    model: &Model,
    context: &TranscriptContext,
    options: Option<&SimpleStreamOptions>,
    api_key: Option<&str>,
) -> StreamOptions {
    let reasoning = options
        .and_then(|o| o.reasoning)
        .map_or(ModelThinkingLevel::Off, |level| match level {
            ThinkingLevel::Minimal => ModelThinkingLevel::Minimal,
            ThinkingLevel::Low => ModelThinkingLevel::Low,
            ThinkingLevel::Medium => ModelThinkingLevel::Medium,
            ThinkingLevel::High => ModelThinkingLevel::High,
            ThinkingLevel::Xhigh => ModelThinkingLevel::Xhigh,
            ThinkingLevel::Max => ModelThinkingLevel::Max,
        });
    let sampling_params = resolve_sampling_params(
        model,
        reasoning,
        options.and_then(|o| o.base.sampling_params.as_ref()),
    );
    let max_tokens = clamp_max_tokens_to_context(
        model,
        context,
        options
            .and_then(|o| o.base.max_tokens)
            .unwrap_or(model.max_tokens),
    );
    let base = options.map(|o| o.base.clone()).unwrap_or_default();
    StreamOptions {
        api_key: api_key.map(str::to_owned).or_else(|| {
            base.api_key
                .clone()
                .or_else(|| options.and_then(|o| o.base.api_key.clone()))
        }),
        temperature: options
            .and_then(|o| o.base.temperature)
            .or(base.temperature),
        sampling_params,
        max_tokens: Some(max_tokens),
        env: base.env,
        headers: base.headers,
        identity: base.identity,
        timeout_ms: base.timeout_ms,
        websocket_connect_timeout_ms: base.websocket_connect_timeout_ms,
        max_retries: base.max_retries,
        max_retry_delay_ms: base.max_retry_delay_ms,
        metadata: base.metadata,
        transport: base.transport,
        cache_retention: base.cache_retention,
        session_id: base.session_id,
    }
}
