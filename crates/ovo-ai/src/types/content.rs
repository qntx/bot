//! Content blocks carried by messages and tool calls.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

/// Text block; `textSignature` is provider metadata (legacy id string or
/// `TextSignatureV1` JSON).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TextContent {
    /// The block's text.
    pub text: String,
    /// Provider metadata, e.g. `OpenAI` responses message id.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text_signature: Option<String>,
}

/// Thinking block; `thinkingSignature` replays provider reasoning data.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ThinkingContent {
    /// The model's reasoning text.
    pub thinking: String,
    /// Provider-specific opaque or serialized reasoning replay data.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking_signature: Option<String>,
    /// True when the thinking content was redacted by safety filters.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub redacted: Option<bool>,
}

/// Base64-encoded image block.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ImageContent {
    /// Base64-encoded image data.
    pub data: String,
    /// MIME type, e.g. `"image/jpeg"` or `"image/png"`.
    pub mime_type: String,
}

/// A tool call the model made.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolCall {
    /// The provider's call identifier.
    pub id: String,
    /// The tool name.
    pub name: String,
    /// The parsed arguments object.
    pub arguments: Map<String, Value>,
    /// Google-specific opaque signature for reusing thought context.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thought_signature: Option<String>,
    /// `OpenAI` Responses namespace for namespaced tool calls.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub namespace: Option<String>,
    /// Set when the model's arguments text was not a valid JSON object; `arguments` is then empty.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arguments_error: Option<String>,
}

/// Block allowed in system message content.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum SystemContentBlock {
    /// Text block.
    #[serde(rename = "text")]
    Text(TextContent),
}

/// Block allowed in user and tool-result content.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum UserContentBlock {
    /// Text block.
    #[serde(rename = "text")]
    Text(TextContent),
    /// Image block.
    #[serde(rename = "image")]
    Image(ImageContent),
}

/// Block allowed in assistant content.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum AssistantContentBlock {
    /// Text block.
    #[serde(rename = "text")]
    Text(TextContent),
    /// Thinking block.
    #[serde(rename = "thinking")]
    Thinking(ThinkingContent),
    /// Tool call.
    #[serde(rename = "toolCall")]
    ToolCall(ToolCall),
}

/// `string | TextContent[]` (system message content).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum TextOrBlocks {
    /// Plain instruction text.
    Text(String),
    /// Structured text blocks.
    Blocks(Vec<SystemContentBlock>),
}

/// `string | (TextContent | ImageContent)[]` (user message content).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum TextOrBlocksWithImages {
    /// Plain user text.
    Text(String),
    /// Structured blocks.
    Blocks(Vec<UserContentBlock>),
}

/// Why the assistant stream terminated.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum StopReason {
    /// Response is still being produced.
    #[serde(rename = "pending")]
    Pending,
    /// Natural stop.
    #[serde(rename = "stop")]
    Stop,
    /// Output hit the token limit.
    #[serde(rename = "length")]
    Length,
    /// Model requested tool calls.
    #[serde(rename = "toolUse")]
    ToolUse,
    /// The request failed.
    #[serde(rename = "error")]
    Error,
    /// The request was aborted.
    #[serde(rename = "aborted")]
    Aborted,
    /// The provider returned a deferred-response handle.
    #[serde(rename = "deferred")]
    Deferred,
}
