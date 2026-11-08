//! Wire events of one assistant response and their serde shapes.
//!
//! `partial` and the terminal message payloads of the upstream events are
//! not carried: the consumer's accumulator owns the live message and
//! `AssistantStream::result()` yields it.

use serde::{Deserialize, Serialize};

use crate::types::content::{StopReason, ToolCall};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
/// One wire event of an assistant response stream.
pub enum AssistantMessageEvent {
    /// The response has started.
    #[serde(rename = "start")]
    Start,
    /// A text block opened at `content_index`.
    #[serde(rename = "text_start")]
    TextStart {
        /// Index into `message.content`.
        #[serde(rename = "contentIndex")]
        content_index: usize,
    },
    /// `delta` was appended to the open text block.
    #[serde(rename = "text_delta")]
    TextDelta {
        /// Index into `message.content`.
        #[serde(rename = "contentIndex")]
        content_index: usize,
        /// Text appended to the block.
        delta: String,
    },
    /// The text block closed; `content` is the authoritative text.
    #[serde(rename = "text_end")]
    TextEnd {
        /// Index into `message.content`.
        #[serde(rename = "contentIndex")]
        content_index: usize,
        /// The block's complete text (signatures travel on the block, not here).
        content: String,
    },
    /// A thinking block opened at `content_index`.
    #[serde(rename = "thinking_start")]
    ThinkingStart {
        /// Index into `message.content`.
        #[serde(rename = "contentIndex")]
        content_index: usize,
    },
    /// `delta` was appended to the open thinking block.
    #[serde(rename = "thinking_delta")]
    ThinkingDelta {
        /// Index into `message.content`.
        #[serde(rename = "contentIndex")]
        content_index: usize,
        /// Thinking text appended to the block.
        delta: String,
    },
    /// The thinking block closed; `content` is the authoritative text.
    #[serde(rename = "thinking_end")]
    ThinkingEnd {
        /// Index into `message.content`.
        #[serde(rename = "contentIndex")]
        content_index: usize,
        /// The block's complete thinking text.
        content: String,
    },
    /// A tool-call block opened at `content_index`; the call's id, name and
    /// initial arguments arrive on the block through `partial()` only.
    #[serde(rename = "toolcall_start")]
    ToolcallStart {
        /// Index into `message.content`.
        #[serde(rename = "contentIndex")]
        content_index: usize,
    },
    /// `delta` was appended to the open tool-call's argument text.
    #[serde(rename = "toolcall_delta")]
    ToolcallDelta {
        /// Index into `message.content`.
        #[serde(rename = "contentIndex")]
        content_index: usize,
        /// Argument text appended.
        delta: String,
    },
    /// The tool call closed; `tool_call` is the authoritative block.
    #[serde(rename = "toolcall_end")]
    ToolcallEnd {
        /// Index into `message.content`.
        #[serde(rename = "contentIndex")]
        content_index: usize,
        /// The complete tool call, including `argumentsError` when the
        /// argument text was malformed.
        #[serde(rename = "toolCall", serialize_with = "serialize_tool_call")]
        tool_call: ToolCall,
    },
    /// The response finished normally.
    #[serde(rename = "done")]
    Done {
        /// Why the turn ended.
        reason: DoneReason,
    },
    /// The response failed (or was aborted).
    #[serde(rename = "error")]
    Error {
        /// Why the turn ended.
        reason: ErrorReason,
        /// The error text; upstream carries it inside the message payload.
        #[serde(rename = "errorMessage")]
        error_message: String,
    },
}

/// Wire shape: `toolCall` is a content block and carries its `type` tag.
pub(crate) fn serialize_tool_call<S: serde::Serializer>(
    call: &ToolCall,
    serializer: S,
) -> Result<S::Ok, S::Error> {
    let mut value = crate::json::to_string(call)
        .ok()
        .and_then(|t| serde_json::from_str::<serde_json::Value>(&t).ok())
        .unwrap_or_default();
    if let serde_json::Value::Object(members) = &mut value {
        let mut tagged = serde_json::Map::with_capacity(members.len() + 1);
        tagged.insert("type".to_owned(), serde_json::Value::from("toolCall"));
        tagged.append(members);
        value = serde_json::Value::Object(tagged);
    }
    value.serialize(serializer)
}

/// Reasons a response finished normally (`stop`, `length`, `toolUse`,
/// `deferred`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum DoneReason {
    /// Model finished its turn.
    #[serde(rename = "stop")]
    Stop,
    /// Output hit a length limit.
    #[serde(rename = "length")]
    Length,
    /// Model wants to run a tool.
    #[serde(rename = "toolUse")]
    ToolUse,
    /// Turn continues later (deferred tools).
    #[serde(rename = "deferred")]
    Deferred,
}

/// Reasons a response failed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ErrorReason {
    /// The caller aborted.
    #[serde(rename = "aborted")]
    Aborted,
    /// The provider raised an error.
    #[serde(rename = "error")]
    Error,
}

impl From<DoneReason> for StopReason {
    fn from(reason: DoneReason) -> Self {
        match reason {
            DoneReason::Stop => Self::Stop,
            DoneReason::Length => Self::Length,
            DoneReason::ToolUse => Self::ToolUse,
            DoneReason::Deferred => Self::Deferred,
        }
    }
}

impl From<ErrorReason> for StopReason {
    fn from(reason: ErrorReason) -> Self {
        match reason {
            ErrorReason::Aborted => Self::Aborted,
            ErrorReason::Error => Self::Error,
        }
    }
}
