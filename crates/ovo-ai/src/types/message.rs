//! Transcript messages: `#[serde(tag = "role")]` union plus each role's
//! payload.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use super::api::{Api, ProviderId};
use super::content::{
    AssistantContentBlock, StopReason, TextOrBlocks, TextOrBlocksWithImages, UserContentBlock,
};
use super::deferred::DeferredHandle;
use super::diagnostic::AssistantMessageDiagnostic;
use super::model::ModelThinkingLevel;
use super::tool::{Tool, ToolReference};
use super::usage::Usage;

/// System instructions and tool declarations at one point in the transcript.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
#[non_exhaustive]
pub struct SystemMessage {
    /// Instruction text; the base prompt on the leading message.
    pub content: TextOrBlocks,
    /// Named prompt sections rendered after `content`; `null` removes one.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sections: Option<Map<String, Value>>,
    /// Complete definitions of tools that become available at this point.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools_added: Option<Vec<Tool>>,
    /// Tools that stop being available at this point.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools_removed: Option<Vec<ToolReference>>,
    /// Unix timestamp in milliseconds.
    pub timestamp: i64,
}

/// A user message.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
#[non_exhaustive]
pub struct UserMessage {
    /// Text or text-and-image content.
    pub content: TextOrBlocksWithImages,
    /// Unix timestamp in milliseconds.
    pub timestamp: i64,
}

/// An assistant message.
///
/// Field order follows the upstream construction path: the adapter's base
/// literal, then fields appended in the order anthropic-messages.ts assigns
/// them (`providerThinkingLevel` before the request, `responseId` and
/// `responseModel` at `message_start`, `rawStopReason` and `errorMessage` at
/// stop), then rarer-path fields (`diagnostics` on retries, `deferred`,
/// `endTurn`), then `durationMs` (event-stream.ts:127 at stream end) and
/// `thinkingLevel` (agent-loop.ts:419 after the stream).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
#[non_exhaustive]
pub struct AssistantMessage {
    /// The response content blocks.
    pub content: Vec<AssistantContentBlock>,
    /// API family that produced the message.
    pub api: Api,
    /// Provider that produced the message.
    pub provider: ProviderId,
    /// Requested model id.
    pub model: String,
    /// Token usage and cost of the response.
    pub usage: Usage,
    /// Why the stream terminated.
    pub stop_reason: StopReason,
    /// Unix timestamp in milliseconds when the request started.
    pub timestamp: i64,
    /// Exact provider-native effort level used for this response.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_thinking_level: Option<String>,
    /// Provider-specific response/message identifier.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_id: Option<String>,
    /// Concrete model reported by the provider, when reported.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_model: Option<String>,
    /// Provider-reported stop reason before normalization.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub raw_stop_reason: Option<String>,
    /// Error text when `stopReason` is `"error"` or `"aborted"`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error_message: Option<String>,
    /// Redacted provider/runtime diagnostics for failures and recoveries.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub diagnostics: Option<Vec<AssistantMessageDiagnostic>>,
    /// Handle for a deferred (async) provider response.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub deferred: Option<DeferredHandle>,
    /// Provider indication of whether the model explicitly ended its turn.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub end_turn: Option<bool>,
    /// Milliseconds from `timestamp` until the response ended.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub duration_ms: Option<u64>,
    /// Pi thinking level the agent loop requested.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking_level: Option<ModelThinkingLevel>,
}

/// A tool call another tool made while it ran (e.g. codemode).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
#[non_exhaustive]
pub struct NestedToolCallRecord {
    /// The nested call's identifier.
    pub id: String,
    /// The tool name.
    pub name: String,
    /// The parsed arguments; omitted over the size limits.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arguments: Option<Map<String, Value>>,
    /// UTF-8 size of the arguments as JSON, set when `arguments` is omitted.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arguments_bytes: Option<u64>,
    /// `unfinished`: still running when the calling tool finished.
    pub status: NestedToolCallStatus,
    /// Milliseconds the nested call took.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub duration_ms: Option<u64>,
    /// Error text, truncated.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Completion status of a nested tool call.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum NestedToolCallStatus {
    /// The call completed successfully.
    #[serde(rename = "ok")]
    Ok,
    /// The call failed.
    #[serde(rename = "error")]
    Error,
    /// The call was still running when the calling tool finished.
    #[serde(rename = "unfinished")]
    Unfinished,
}

/// Bounded record of the nested calls a tool made.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct NestedToolCalls {
    /// The recorded calls; results are not recorded.
    pub calls: Vec<NestedToolCallRecord>,
    /// False when calls were dropped, arguments omitted, or calls had not finished.
    pub complete: bool,
}

/// A tool result message.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
#[non_exhaustive]
pub struct ToolResultMessage {
    /// The tool call this result answers.
    pub tool_call_id: String,
    /// The tool that produced the result.
    pub tool_name: String,
    /// Text and image result content.
    pub content: Vec<UserContentBlock>,
    /// Tool-specific structured details.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<Value>,
    /// Usage from the tool execution itself, not main context accounting.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
    /// Calls this tool made to other tools.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub nested_calls: Option<NestedToolCalls>,
    /// Whether the result represents a tool error.
    pub is_error: bool,
    /// Unix timestamp in milliseconds.
    pub timestamp: i64,
    /// Milliseconds the tool's execution took.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub duration_ms: Option<u64>,
}

/// Any transcript message.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "role")]
pub enum Message {
    /// System instructions and tool declarations.
    #[serde(rename = "system")]
    System(SystemMessage),
    /// A user message.
    #[serde(rename = "user")]
    User(UserMessage),
    /// An assistant message.
    #[serde(rename = "assistant")]
    Assistant(Box<AssistantMessage>),
    /// A tool result.
    #[serde(rename = "toolResult")]
    ToolResult(ToolResultMessage),
}
