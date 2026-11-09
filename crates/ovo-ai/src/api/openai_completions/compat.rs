//! `detectCompat` and `getCompat`: the compat settings of an
//! `openai-completions` model, detected from its provider and base URL and
//! overridden by `model.compat`.

use serde_json::{Map, Value};

pub use crate::api::openai_sdk::SessionAffinityFormat;
use crate::types::model::Model;

/// The resolved compat settings (`ResolvedOpenAICompletionsCompat`).
#[expect(
    clippy::struct_excessive_bools,
    reason = "mirrors upstream's flat boolean compat surface"
)]
#[derive(Clone, Debug, PartialEq)]
pub struct OpenAiCompletionsCompat {
    /// `store: false` is sent.
    pub supports_store: bool,
    /// Instructions use the `developer` role on reasoning models.
    pub supports_developer_role: bool,
    /// `reasoning_effort` may be sent.
    pub supports_reasoning_effort: bool,
    /// `stream_options.include_usage` is sent.
    pub supports_usage_in_streaming: bool,
    /// Streams end with a `finish_reason`; when false the stop reason is
    /// inferred.
    pub supports_finish_reason: bool,
    /// The token-limit field.
    pub max_tokens_field: MaxTokensField,
    /// Tool results carry `name`.
    pub requires_tool_result_name: bool,
    /// A user message after tool results needs an assistant message between.
    pub requires_assistant_after_tool_result: bool,
    /// Replayed thinking is sent as plain text content.
    pub requires_thinking_as_text: bool,
    /// Replayed assistant messages carry `reasoning_content` (empty if none).
    pub requires_reasoning_content_on_assistant_messages: bool,
    /// How reasoning controls are sent.
    pub thinking_format: ThinkingFormat,
    /// Values for `chat_template_kwargs` (`chat-template` format).
    pub chat_template_kwargs: Map<String, Value>,
    /// Values for `chat_template_args` (`baseten` format).
    pub chat_template_args: Map<String, Value>,
    /// z.ai `tool_stream: true` is sent with tools.
    pub zai_tool_stream: bool,
    /// Alias for `thinking_token_budget_field: thinking_token_budget`.
    pub supports_thinking_token_budget: bool,
    /// Top-level field capping reasoning tokens.
    pub thinking_token_budget_field: Option<ThinkingTokenBudgetField>,
    /// `strict` is sent on function tools.
    pub supports_strict_mode: bool,
    /// Grammar tools are sent as `custom` tools.
    pub supports_openai_grammar_tools: bool,
    /// Later system messages are sent in place.
    pub supports_mid_convo_system_messages: bool,
    /// Later system messages may add tools.
    pub supports_mid_convo_tool_additions: bool,
    /// Anthropic-style `cache_control` markers are applied.
    pub anthropic_cache_control: bool,
    /// Session affinity headers are sent for `sessionId`.
    pub send_session_affinity_headers: bool,
    /// The session affinity header shape.
    pub session_affinity_format: SessionAffinityFormat,
    /// Long cache retention (`24h` key retention or a `1h` ttl) is allowed.
    pub supports_long_cache_retention: bool,
    /// vLLM scheduler `priority`.
    pub vllm_priority: Option<f64>,
}

/// `thinkingFormat`: how reasoning controls are sent.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ThinkingFormat {
    /// `reasoning_effort`.
    Openai,
    /// `reasoning.effort`.
    Openrouter,
    /// `thinking.type` plus `reasoning_effort` when supported.
    Deepseek,
    /// `reasoning.enabled` plus `reasoning_effort` when supported.
    Together,
    /// `chat_template_args` plus `reasoning_effort` when supported.
    Baseten,
    /// `thinking.type` plus `reasoning_effort` when supported.
    Zai,
    /// Top-level `enable_thinking` plus `reasoning_effort` when supported.
    Qwen,
    /// Configurable `chat_template_kwargs`.
    ChatTemplate,
    /// `chat_template_kwargs.enable_thinking` and `preserve_thinking`.
    QwenChatTemplate,
    /// Top-level string `thinking`.
    StringThinking,
    /// `reasoning.effort` only for a mapped string effort.
    AntLing,
    /// A value upstream does not know; only the generic branches apply.
    Other,
}

impl ThinkingFormat {
    fn parse(value: &str) -> Self {
        match value {
            "openai" => Self::Openai,
            "openrouter" => Self::Openrouter,
            "deepseek" => Self::Deepseek,
            "together" => Self::Together,
            "baseten" => Self::Baseten,
            "zai" => Self::Zai,
            "qwen" => Self::Qwen,
            "chat-template" => Self::ChatTemplate,
            "qwen-chat-template" => Self::QwenChatTemplate,
            "string-thinking" => Self::StringThinking,
            "ant-ling" => Self::AntLing,
            _ => Self::Other,
        }
    }
}

/// `maxTokensField`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MaxTokensField {
    /// `max_completion_tokens`.
    MaxCompletionTokens,
    /// The deprecated `max_tokens` some compatible providers require.
    MaxTokens,
}

/// `thinkingTokenBudgetField`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ThinkingTokenBudgetField {
    /// `thinking_token_budget` (vLLM).
    TokenBudget,
    /// `thinking_budget` (Qwen, `DashScope`, `SGLang`).
    Budget,
    /// `thinking_budget_tokens` (llama.cpp).
    BudgetTokens,
}

impl ThinkingTokenBudgetField {
    /// The request field name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::TokenBudget => "thinking_token_budget",
            Self::Budget => "thinking_budget",
            Self::BudgetTokens => "thinking_budget_tokens",
        }
    }
}

impl OpenAiCompletionsCompat {
    /// `getCompat`: the detected settings with `model.compat` overrides.
    #[must_use]
    pub fn of(model: &Model) -> Self {
        let mut compat = detect(model);
        let Some(overrides) = model.compat.as_ref() else {
            return compat;
        };
        let flag = |key: &str, slot: &mut bool| {
            if let Some(value) = overrides.get(key).and_then(Value::as_bool) {
                *slot = value;
            }
        };
        flag("supportsStore", &mut compat.supports_store);
        flag("supportsDeveloperRole", &mut compat.supports_developer_role);
        flag(
            "supportsReasoningEffort",
            &mut compat.supports_reasoning_effort,
        );
        flag(
            "supportsUsageInStreaming",
            &mut compat.supports_usage_in_streaming,
        );
        flag("supportsFinishReason", &mut compat.supports_finish_reason);
        flag(
            "requiresToolResultName",
            &mut compat.requires_tool_result_name,
        );
        flag(
            "requiresAssistantAfterToolResult",
            &mut compat.requires_assistant_after_tool_result,
        );
        flag(
            "requiresThinkingAsText",
            &mut compat.requires_thinking_as_text,
        );
        flag(
            "requiresReasoningContentOnAssistantMessages",
            &mut compat.requires_reasoning_content_on_assistant_messages,
        );
        flag("zaiToolStream", &mut compat.zai_tool_stream);
        flag(
            "supportsThinkingTokenBudget",
            &mut compat.supports_thinking_token_budget,
        );
        flag("supportsStrictMode", &mut compat.supports_strict_mode);
        flag(
            "supportsOpenAIGrammarTools",
            &mut compat.supports_openai_grammar_tools,
        );
        flag(
            "supportsMidConvoSystemMessages",
            &mut compat.supports_mid_convo_system_messages,
        );
        flag(
            "supportsMidConvoToolAdditions",
            &mut compat.supports_mid_convo_tool_additions,
        );
        flag(
            "sendSessionAffinityHeaders",
            &mut compat.send_session_affinity_headers,
        );
        flag(
            "supportsLongCacheRetention",
            &mut compat.supports_long_cache_retention,
        );
        let text = |key: &str| overrides.get(key).and_then(Value::as_str);
        if let Some(field) = text("maxTokensField") {
            compat.max_tokens_field = if field == "max_tokens" {
                MaxTokensField::MaxTokens
            } else {
                MaxTokensField::MaxCompletionTokens
            };
        }
        if let Some(format) = text("thinkingFormat") {
            compat.thinking_format = ThinkingFormat::parse(format);
        }
        if let Some(field) = text("thinkingTokenBudgetField") {
            compat.thinking_token_budget_field = match field {
                "thinking_token_budget" => Some(ThinkingTokenBudgetField::TokenBudget),
                "thinking_budget" => Some(ThinkingTokenBudgetField::Budget),
                "thinking_budget_tokens" => Some(ThinkingTokenBudgetField::BudgetTokens),
                _ => None,
            };
        }
        if text("cacheControlFormat") == Some("anthropic") {
            compat.anthropic_cache_control = true;
        }
        if let Some(format) = text("sessionAffinityFormat") {
            compat.session_affinity_format = SessionAffinityFormat::parse(format);
        }
        let record = |key: &str| overrides.get(key).and_then(Value::as_object).cloned();
        if let Some(kwargs) = record("chatTemplateKwargs") {
            compat.chat_template_kwargs = kwargs;
        }
        if let Some(args) = record("chatTemplateArgs") {
            compat.chat_template_args = args;
        }
        compat.vllm_priority = overrides.get("vllmPriority").and_then(Value::as_f64);
        compat
    }

    /// `resolveThinkingTokenBudgetField`: the explicit field, else the alias.
    #[must_use]
    pub const fn budget_field(&self) -> Option<ThinkingTokenBudgetField> {
        match self.thinking_token_budget_field {
            Some(field) => Some(field),
            None if self.supports_thinking_token_budget => {
                Some(ThinkingTokenBudgetField::TokenBudget)
            }
            None => None,
        }
    }
}

/// `detectCompat`: settings inferred from the provider id and base URL.
fn detect(model: &Model) -> OpenAiCompletionsCompat {
    let provider = model.provider.as_str();
    let url = model.base_url.as_str();
    let is = |providers: &[&str], hosts: &[&str]| {
        providers.contains(&provider) || hosts.iter().any(|host| url.contains(host))
    };
    let zai = is(&["zai", "zai-coding-cn"], &["api.z.ai", "open.bigmodel.cn"]);
    let together = is(&["together"], &["api.together.ai", "api.together.xyz"]);
    let moonshot = is(&["moonshotai", "moonshotai-cn"], &["api.moonshot."]);
    let openrouter = is(&["openrouter"], &["openrouter.ai"]);
    let cloudflare_workers = is(&["cloudflare-workers-ai"], &["api.cloudflare.com"]);
    let cloudflare_gateway = is(&["cloudflare-ai-gateway"], &["gateway.ai.cloudflare.com"]);
    let nvidia = is(&["nvidia"], &["integrate.api.nvidia.com"]);
    let ant_ling = is(&["ant-ling"], &["api.ant-ling.com"]);
    let cerebras = is(&["cerebras"], &["cerebras.ai"]);
    let deepseek = provider == "deepseek" || url.to_lowercase().contains("deepseek.com");
    let grok = is(&["xai"], &["api.x.ai"]);
    let chutes = url.contains("chutes.ai");
    let non_standard = nvidia
        || cerebras
        || grok
        || together
        || chutes
        || deepseek
        || zai
        || moonshot
        || is(&["opencode"], &["opencode.ai"])
        || cloudflare_workers
        || cloudflare_gateway
        || ant_ling;
    let use_max_tokens = chutes
        || deepseek
        || moonshot
        || cloudflare_gateway
        || together
        || nvidia
        || ant_ling
        || zai;
    let developer_role_model =
        openrouter && (model.id.starts_with("anthropic/") || model.id.starts_with("openai/"));
    let thinking_format = if deepseek {
        ThinkingFormat::Deepseek
    } else if zai {
        ThinkingFormat::Zai
    } else if together {
        ThinkingFormat::Together
    } else if ant_ling {
        ThinkingFormat::AntLing
    } else if openrouter {
        ThinkingFormat::Openrouter
    } else {
        ThinkingFormat::Openai
    };
    OpenAiCompletionsCompat {
        supports_store: !non_standard,
        supports_developer_role: developer_role_model || (!non_standard && !openrouter),
        supports_reasoning_effort: !grok
            && !zai
            && !moonshot
            && !together
            && !cloudflare_gateway
            && !nvidia
            && !ant_ling,
        supports_usage_in_streaming: true,
        supports_finish_reason: true,
        max_tokens_field: if use_max_tokens {
            MaxTokensField::MaxTokens
        } else {
            MaxTokensField::MaxCompletionTokens
        },
        requires_tool_result_name: false,
        requires_assistant_after_tool_result: false,
        requires_thinking_as_text: false,
        requires_reasoning_content_on_assistant_messages: deepseek,
        thinking_format,
        chat_template_kwargs: Map::new(),
        chat_template_args: Map::new(),
        zai_tool_stream: false,
        supports_thinking_token_budget: false,
        thinking_token_budget_field: None,
        // OpenAI compatibility alone does not imply strict JSON-schema tools.
        supports_strict_mode: false,
        supports_openai_grammar_tools: false,
        supports_mid_convo_system_messages: false,
        supports_mid_convo_tool_additions: false,
        anthropic_cache_control: provider == "openrouter" && model.id.starts_with("anthropic/"),
        send_session_affinity_headers: openrouter,
        session_affinity_format: if openrouter {
            SessionAffinityFormat::Openrouter
        } else {
            SessionAffinityFormat::Openai
        },
        supports_long_cache_retention: !(together
            || cloudflare_workers
            || cloudflare_gateway
            || nvidia
            || ant_ling),
        vllm_priority: None,
    }
}
