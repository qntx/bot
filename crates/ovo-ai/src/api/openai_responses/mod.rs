//! The `openai-responses` adapter (upstream `api/openai-responses.ts`).
//!
//! [`build_request`](crate::api::openai_responses::build_request) ports
//! `buildParams` and `createClient`: the request body in upstream key order
//! (serialized with `ovo_ai::json`, byte-identical) and the headers in the
//! `OpenAI` SDK's merge order. [`convert`](crate::api::openai_responses::convert)
//! holds the input and tool conversion other Responses transports share;
//! [`stream`](crate::api::openai_responses::stream) is the streaming side.

use std::collections::BTreeMap;

use serde_json::{Map, Value, json};

use crate::api::constrained_sampling::create_grammar_tool_input_properties;
use crate::api::openai_prompt_cache::clamp_openai_prompt_cache_key;
use crate::api::openai_sdk::{
    SessionAffinityFormat, client_api_key, endpoint_url, request_headers, tool_choice_value,
};
use crate::options::{
    CacheRetention, SimpleStreamOptions, StreamOptions, build_base_options, resolve_cache_retention,
};
use crate::simple_options::resolve_sampling_params;
use crate::transcript::{
    TranscriptContext, get_declared_tools, resolve_transcript, resolve_transcript_tools,
};
use crate::types::model::{Model, ModelThinkingLevel, ThinkingLevel, ThinkingLevelMapValue};

pub mod convert;
#[cfg(feature = "http")]
pub mod stream;

use convert::{ConvertOptions, ToolOptions, convert_messages, convert_tools, tool_support};

/// The api identifier this adapter serves.
pub const API: &str = "openai-responses";

/// Providers whose `call|item` tool call ids keep their item part.
const OPENAI_TOOL_CALL_PROVIDERS: [&str; 3] = ["openai", "openai-codex", "opencode"];

/// The Responses API rejects `max_output_tokens` below 16.
const MIN_OUTPUT_TOKENS: u64 = 16;

/// The fixed `reasoning.effort` while `configuration_update` items carry
/// the thinking level.
const CONFIGURATION_UPDATE_REQUEST_EFFORT: &str = "medium";

/// `reasoning.summary`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ReasoningSummary {
    /// The provider chooses.
    Auto,
    /// A detailed summary.
    Detailed,
    /// A concise summary.
    Concise,
}

impl ReasoningSummary {
    /// The wire value.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Detailed => "detailed",
            Self::Concise => "concise",
        }
    }
}

/// `service_tier`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ServiceTier {
    /// The project's default tier, or `default` when unset.
    Auto,
    /// Standard processing.
    Default,
    /// Lower cost, higher latency.
    Flex,
    /// Scale tier capacity.
    Scale,
    /// Priority processing.
    Priority,
    /// Fast processing.
    Fast,
    /// Ultrafast processing.
    Ultrafast,
}

impl ServiceTier {
    /// The wire value.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Default => "default",
            Self::Flex => "flex",
            Self::Scale => "scale",
            Self::Priority => "priority",
            Self::Fast => "fast",
            Self::Ultrafast => "ultrafast",
        }
    }
}

/// `OpenAIResponsesOptions`: the stream options plus Responses controls.
#[derive(Clone, Default)]
#[non_exhaustive]
pub struct OpenAiResponsesOptions {
    /// The shared stream options.
    pub base: StreamOptions,
    /// The reasoning level; `None` turns reasoning off where the model can.
    pub reasoning_effort: Option<ThinkingLevel>,
    /// `reasoning.summary`; requesting one without an effort implies
    /// `medium`.
    pub reasoning_summary: Option<ReasoningSummary>,
    /// `service_tier`, also used to price the usage.
    pub service_tier: Option<ServiceTier>,
    /// `tool_choice` as sent.
    pub tool_choice: Option<Value>,
}

impl std::fmt::Debug for OpenAiResponsesOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpenAiResponsesOptions")
            .field("reasoning_effort", &self.reasoning_effort)
            .field("reasoning_summary", &self.reasoning_summary)
            .field("service_tier", &self.service_tier)
            .field("tool_choice", &self.tool_choice)
            .finish_non_exhaustive()
    }
}

impl OpenAiResponsesOptions {
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
            reasoning_effort: options.reasoning.and_then(|level| {
                crate::cost::clamp_thinking_level(model, level.into()).thinking()
            }),
            reasoning_summary: None,
            service_tier: None,
            tool_choice: options.tool_choice.map(tool_choice_value),
        }
    }
}

/// The resolved compat settings (`getCompat`): `model.compat` values over
/// upstream's defaults.
#[expect(
    clippy::struct_excessive_bools,
    reason = "mirrors upstream's flat boolean compat surface"
)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OpenAiResponsesCompat {
    /// Instructions use the `developer` role on reasoning models.
    pub supports_developer_role: bool,
    /// Later system messages are sent in place.
    pub supports_mid_convo_system_messages: bool,
    /// The session affinity header shape.
    pub session_affinity_format: SessionAffinityFormat,
    /// `prompt_cache_retention: 24h` (or a `30m` ttl) is allowed.
    pub supports_long_cache_retention: bool,
    /// `strict` is sent on function tools.
    pub supports_strict_mode: bool,
    /// Grammar tools are sent as `custom` tools.
    pub supports_openai_grammar_tools: bool,
    /// Tool changes are declared in `additional_tools` items.
    pub supports_additional_tools: bool,
    /// Tool additions are replayed as client tool searches.
    pub supports_tool_search: bool,
    /// Caching is controlled through `prompt_cache_options`.
    pub supports_explicit_prompt_cache_mode: bool,
    /// `max_output_tokens` is sent.
    pub supports_max_output_tokens: bool,
    /// Thinking levels are sent as `configuration_update` items.
    pub supports_mid_convo_effort: bool,
}

impl OpenAiResponsesCompat {
    /// `getCompat`.
    #[must_use]
    pub fn of(model: &Model) -> Self {
        let overrides = model.compat.as_ref();
        let flag = |key: &str, default: bool| {
            overrides
                .and_then(|compat| compat.get(key))
                .and_then(Value::as_bool)
                .unwrap_or(default)
        };
        let openrouter =
            model.provider.as_str() == "openrouter" || model.base_url.contains("openrouter.ai");
        Self {
            supports_developer_role: flag("supportsDeveloperRole", true),
            supports_mid_convo_system_messages: flag("supportsMidConvoSystemMessages", false),
            session_affinity_format: overrides
                .and_then(|compat| compat.get("sessionAffinityFormat"))
                .and_then(Value::as_str)
                .map_or(
                    if openrouter {
                        SessionAffinityFormat::Openrouter
                    } else {
                        SessionAffinityFormat::Openai
                    },
                    SessionAffinityFormat::parse,
                ),
            supports_long_cache_retention: flag("supportsLongCacheRetention", true),
            supports_strict_mode: flag("supportsStrictMode", false),
            supports_openai_grammar_tools: flag("supportsOpenAIGrammarTools", false),
            supports_additional_tools: flag("supportsAdditionalTools", false),
            supports_tool_search: flag("supportsToolSearch", false),
            supports_explicit_prompt_cache_mode: flag("supportsExplicitPromptCacheMode", false),
            supports_max_output_tokens: flag("supportsMaxOutputTokens", true),
            supports_mid_convo_effort: flag("supportsMidConvoEffort", false),
        }
    }

    const fn tool_options(&self) -> ToolOptions {
        ToolOptions {
            supports_strict_mode: self.supports_strict_mode,
            supports_openai_grammar_tools: self.supports_openai_grammar_tools,
        }
    }
}

/// One request to the Responses endpoint, ready to send.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OpenAiResponsesRequest {
    /// `{baseUrl}/responses`, as the SDK's `buildURL` normalizes it.
    pub url: String,
    /// Request headers after the SDK and `createClient` merge order.
    pub headers: Map<String, Value>,
    /// The JSON body, byte-identical to upstream's `buildParams` output.
    pub body: Vec<u8>,
    /// The request params before serialization — `onPayload` sees and
    /// may replace these.
    pub params: Map<String, Value>,
    /// The input property of each tool sent as a grammar tool, by name; the
    /// stream maps custom tool-call input onto it.
    pub grammar_tool_input_properties: BTreeMap<String, String>,
}

/// `getConfigurationUpdateEffort`: the effort sent in `configuration_update`
/// items (and recorded as the reply's `providerThinkingLevel`), or `None`
/// when it goes in `reasoning.effort`.
#[must_use]
pub fn configuration_update_effort(
    model: &Model,
    options: &OpenAiResponsesOptions,
) -> Option<String> {
    let effort = options.reasoning_effort?;
    // Sign in with ChatGPT rejects configuration_update items.
    if !OpenAiResponsesCompat::of(model).supports_mid_convo_effort
        || is_chatgpt_sign_in(model, &options.base)
    {
        return None;
    }
    Some(mapped_effort(model, effort.as_str()).to_owned())
}

/// `buildParams` and `createClient`: the request body in upstream key order
/// and the request headers.
///
/// # Errors
/// Returns upstream's message when there is no API key, a tool cannot be
/// sent, a grammar tool call has no string input, a replayed reasoning
/// item is not JSON, the base URL is invalid, or the body fails to
/// serialize.
pub fn build_request(
    model: &Model,
    context: &TranscriptContext,
    options: &OpenAiResponsesOptions,
) -> Result<OpenAiResponsesRequest, String> {
    let base = &options.base;
    let compat = OpenAiResponsesCompat::of(model);
    let transcript = resolve_transcript(context, Some(compat.supports_mid_convo_system_messages));
    let api_key = client_api_key(model.provider.as_str(), base)?;
    let retention = resolve_cache_retention(base.cache_retention, base.env.as_ref());
    let grammar_tool_input_properties = create_grammar_tool_input_properties(
        &get_declared_tools(transcript.messages()),
        compat.supports_openai_grammar_tools,
    )
    .map_err(|error| error.to_string())?;
    let session_id = base
        .session_id
        .as_deref()
        .filter(|id| !id.is_empty() && retention != CacheRetention::None);
    let headers = request_headers(
        model,
        transcript.messages(),
        base,
        &api_key,
        &session_affinity(session_id, compat.session_affinity_format),
    );
    let params = build_params(
        model,
        &transcript,
        options,
        &compat,
        retention,
        &grammar_tool_input_properties,
    )?;
    Ok(OpenAiResponsesRequest {
        url: endpoint_url(&model.base_url, "responses")?,
        headers,
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
    options: &OpenAiResponsesOptions,
    compat: &OpenAiResponsesCompat,
    retention: CacheRetention,
    grammar_tool_input_properties: &BTreeMap<String, String>,
) -> Result<Map<String, Value>, String> {
    let base = &options.base;
    let configuration_update = configuration_update_effort(model, options);
    let support = tool_support(
        compat.supports_mid_convo_system_messages,
        compat.supports_additional_tools,
        compat.supports_tool_search,
    );
    let request_tools = resolve_transcript_tools(transcript.messages(), support).request_tools;
    let input = convert_messages(
        model,
        transcript,
        &OPENAI_TOOL_CALL_PROVIDERS,
        &ConvertOptions {
            grammar_tool_input_properties,
            supports_mid_convo_system_messages: compat.supports_mid_convo_system_messages,
            supports_additional_tools: compat.supports_additional_tools,
            supports_tool_search: compat.supports_tool_search,
            tool_options: compat.tool_options(),
            configuration_update_effort: configuration_update.as_deref(),
        },
    )?;
    // Sign in with ChatGPT rejects these request fields.
    let omit_unsupported = is_chatgpt_sign_in(model, base);

    let mut params = Map::new();
    params.insert("model".into(), Value::from(model.id.clone()));
    params.insert("input".into(), Value::from(input));
    params.insert("stream".into(), Value::Bool(true));
    if retention != CacheRetention::None
        && let Some(key) = clamp_openai_prompt_cache_key(base.session_id.as_deref())
    {
        params.insert("prompt_cache_key".into(), Value::from(key));
    }
    if !omit_unsupported {
        let long = retention == CacheRetention::Long && compat.supports_long_cache_retention;
        if long && !compat.supports_explicit_prompt_cache_mode {
            params.insert("prompt_cache_retention".into(), Value::from("24h"));
        }
        if compat.supports_explicit_prompt_cache_mode {
            if retention == CacheRetention::None {
                params.insert("prompt_cache_options".into(), json!({ "mode": "explicit" }));
            } else if long {
                params.insert("prompt_cache_options".into(), json!({ "ttl": "30m" }));
            }
        }
    }
    params.insert("store".into(), Value::Bool(false));
    if let Some(max_tokens) = base
        .max_tokens
        .filter(|max| *max > 0 && compat.supports_max_output_tokens && !omit_unsupported)
    {
        params.insert(
            "max_output_tokens".into(),
            Value::from(max_tokens.max(MIN_OUTPUT_TOKENS)),
        );
    }
    if let Some(temperature) = base.temperature.filter(|_| !omit_unsupported) {
        params.insert("temperature".into(), Value::from(temperature));
    }
    if let Some(tier) = options.service_tier {
        params.insert("service_tier".into(), Value::from(tier.as_str()));
    }
    if !request_tools.is_empty() {
        let tools = convert_tools(&request_tools, compat.tool_options(), false)?;
        params.insert("tools".into(), Value::from(tools));
    }
    if let Some(tool_choice) = &options.tool_choice {
        params.insert("tool_choice".into(), tool_choice.clone());
    }
    let effort = options
        .reasoning_effort
        .map(ThinkingLevel::as_str)
        .or_else(|| options.reasoning_summary.map(|_| "medium"));
    if model.reasoning {
        insert_reasoning(
            &mut params,
            model,
            options,
            effort,
            configuration_update.is_some(),
        );
    }
    // Last, so model and request sampling parameters override named fields.
    let level = match (options.reasoning_effort, options.reasoning_summary) {
        (Some(level), _) => ModelThinkingLevel::from(level),
        (None, Some(_)) => ModelThinkingLevel::Medium,
        (None, None) => ModelThinkingLevel::Off,
    };
    if let Some(sampling) = resolve_sampling_params(model, level, base.sampling_params.as_ref()) {
        params.extend(sampling);
    }
    Ok(params)
}

/// The `reasoning` and `include` fields of a reasoning model.
fn insert_reasoning(
    params: &mut Map<String, Value>,
    model: &Model,
    options: &OpenAiResponsesOptions,
    effort: Option<&str>,
    configuration_update: bool,
) {
    const ENCRYPTED_REASONING: &str = "reasoning.encrypted_content";
    if let Some(effort) = effort {
        let requested = options
            .reasoning_effort
            .map_or(effort, |level| mapped_effort(model, level.as_str()));
        let sent = if configuration_update {
            CONFIGURATION_UPDATE_REQUEST_EFFORT
        } else {
            requested
        };
        let summary = options
            .reasoning_summary
            .map_or("auto", ReasoningSummary::as_str);
        params.insert(
            "reasoning".into(),
            json!({ "effort": sent, "summary": summary }),
        );
        params.insert("include".into(), json!([ENCRYPTED_REASONING]));
    } else if model.provider.as_str() != "github-copilot" {
        let off = model
            .thinking_level_map
            .as_ref()
            .and_then(|map| map.value_for("off"));
        match off {
            Some(ThinkingLevelMapValue::Unsupported) => {}
            Some(ThinkingLevelMapValue::Value(value)) => {
                params.insert("reasoning".into(), json!({ "effort": value }));
            }
            None => {
                params.insert("reasoning".into(), json!({ "effort": "none" }));
            }
        }
    }
    // xAI returns encrypted reasoning only when asked, even without effort.
    if model.provider.as_str() == "xai" {
        params.insert("include".into(), json!([ENCRYPTED_REASONING]));
    }
}

/// `thinkingLevelMap[level] ?? level`.
fn mapped_effort<'a>(model: &'a Model, level: &'a str) -> &'a str {
    match model
        .thinking_level_map
        .as_ref()
        .and_then(|map| map.value_for(level))
    {
        Some(ThinkingLevelMapValue::Value(value)) => value,
        _ => level,
    }
}

/// `isChatGPTSignIn`: a credential other than an `sk-` API key sent to
/// `OpenAI` directly is a Sign in with `ChatGPT` access token.
fn is_chatgpt_sign_in(model: &Model, options: &StreamOptions) -> bool {
    model.provider.as_str() == "openai"
        && model.base_url == "https://api.openai.com/v1"
        && options
            .api_key
            .as_deref()
            .is_some_and(|key| !key.starts_with("sk-"))
}

/// The session affinity headers `createClient` sets for `sessionId`.
fn session_affinity(session_id: Option<&str>, format: SessionAffinityFormat) -> Map<String, Value> {
    let mut affinity = Map::new();
    let Some(session_id) = session_id else {
        return affinity;
    };
    if format == SessionAffinityFormat::Openrouter {
        affinity.insert("x-session-id".into(), Value::from(session_id));
        return affinity;
    }
    if format == SessionAffinityFormat::Openai {
        affinity.insert("session_id".into(), Value::from(session_id));
    }
    affinity.insert("x-client-request-id".into(), Value::from(session_id));
    affinity
}
