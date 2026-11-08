//! Tool declarations and constrained-sampling configuration.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

/// A tool the model may call; `parameters` is the JSON Schema of its
/// arguments.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
#[non_exhaustive]
pub struct Tool {
    /// The tool name the model calls.
    pub name: String,
    /// What the tool does, shown to the model.
    pub description: String,
    /// JSON Schema of the tool's arguments.
    pub parameters: Map<String, Value>,
    /// Optional provider-side constrained sampling config, or `false`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub constrained_sampling: Option<ConstrainedSampling>,
}

/// `false | ConstrainedSamplingConfig` — explicit disable or a config.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ConstrainedSampling {
    /// Explicit `false`.
    Disabled(bool),
    /// A sampling configuration.
    Config(ConstrainedSamplingConfig),
}

/// Provider-side constrained sampling config.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ConstrainedSamplingConfig {
    /// JSON-Schema constrained sampling.
    #[serde(rename = "json_schema")]
    JsonSchema {
        /// Whether the API should prefer or require schema adherence.
        strict: StrictMode,
    },
    /// Grammar-based constrained sampling.
    #[serde(rename = "grammar")]
    Grammar {
        /// Provider-specific encodings of the intended language.
        variants: GrammarVariants,
    },
}

/// JSON-schema strictness.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum StrictMode {
    /// Prefer schema adherence.
    #[serde(rename = "prefer")]
    Prefer,
    /// Require schema adherence.
    #[serde(rename = "require")]
    Require,
}

/// Grammar string per `OpenAI` grammar format.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct GrammarVariants {
    /// Lark-format grammar.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub openai_lark: Option<String>,
    /// Regex-format grammar.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub openai_regex: Option<String>,
}

/// Reference to a tool by name (used by `toolsRemoved`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ToolReference {
    /// The tool name.
    pub name: String,
}

/// Provider-neutral tool selection for simple requests.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ToolChoice {
    /// The provider decides when to call tools.
    #[serde(rename = "auto")]
    Auto,
    /// Tool calls are disabled.
    #[serde(rename = "none")]
    None,
}

/// Custom token budgets for thinking levels.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ThinkingBudgets {
    /// Budget for the `minimal` level.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub minimal: Option<u64>,
    /// Budget for the `low` level.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub low: Option<u64>,
    /// Budget for the `medium` level.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub medium: Option<u64>,
    /// Budget for the `high` level.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub high: Option<u64>,
}

/// Transport for providers that support multiple transports.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Transport {
    /// Server-sent events.
    #[serde(rename = "sse")]
    Sse,
    /// WebSocket streaming.
    #[serde(rename = "websocket")]
    Websocket,
    /// WebSocket with cached connection.
    #[serde(rename = "websocket-cached")]
    WebsocketCached,
    /// Provider chooses the transport.
    #[serde(rename = "auto")]
    Auto,
}
