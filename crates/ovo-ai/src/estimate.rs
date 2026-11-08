//! Context token estimation: upstream `utils/estimate.ts`.

use crate::text::js_len;
use crate::transcript::TranscriptContext;
use crate::types::content::{AssistantContentBlock, StopReason, TextOrBlocksWithImages};
use crate::types::message::Message;
use crate::types::usage::Usage;

const CHARS_PER_TOKEN: f64 = 3.5;

/// `Math.ceil(chars / 3.5)` — non-negative and bounded by message sizes, so
/// the f64-to-u64 conversion never truncates meaningfully.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "upstream Math.ceil of a non-negative ratio; values fit u64"
)]
fn ceil_tokens(chars: f64) -> u64 {
    (chars / CHARS_PER_TOKEN).ceil() as u64
}
const ESTIMATED_IMAGE_CHARS: u64 = 4800;

/// `ContextUsageEstimate`: estimated total context tokens and where the
/// estimate came from.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ContextUsageEstimate {
    /// Estimated total context tokens.
    pub tokens: u64,
    /// Tokens reported by the most recent applicable assistant usage
    /// block.
    pub usage_tokens: u64,
    /// Estimated tokens after the most recent applicable assistant usage
    /// block.
    pub trailing_tokens: u64,
    /// Index of the applicable message that provided usage, or `None`
    /// when none exists.
    pub last_usage_index: Option<usize>,
}

/// `calculateContextTokens`: `totalTokens` when reported, else the sum of
/// the component counts.
#[must_use]
pub const fn calculate_context_tokens(usage: &Usage) -> u64 {
    if usage.total_tokens != 0 {
        usage.total_tokens
    } else {
        usage.input + usage.output + usage.cache_read + usage.cache_write
    }
}

fn safe_json_stringify<T: serde::Serialize>(value: &T) -> String {
    serde_json::to_string(value).unwrap_or_else(|_| "[unserializable]".to_owned())
}

/// `estimateTextTokens`: `ceil(jsLen(text) / 3.5)` — text length is in
/// UTF-16 code units like upstream `.length`.
#[must_use]
pub fn estimate_text_tokens(text: &str) -> u64 {
    ceil_tokens(js_len(text) as f64)
}

fn estimate_blocks_chars(blocks: &[crate::types::content::UserContentBlock]) -> u64 {
    let mut chars = 0u64;
    for block in blocks {
        chars += match block {
            crate::types::content::UserContentBlock::Text(text) => js_len(&text.text) as u64,
            crate::types::content::UserContentBlock::Image(_) => ESTIMATED_IMAGE_CHARS,
        };
    }
    chars
}

/// `estimateTextAndImageContentTokens`: the token estimate of a
/// text/image content value.
#[must_use]
pub fn estimate_text_and_image_content_tokens(content: &TextOrBlocksWithImages) -> u64 {
    let chars = match content {
        TextOrBlocksWithImages::Text(text) => js_len(text) as u64,
        TextOrBlocksWithImages::Blocks(blocks) => estimate_blocks_chars(blocks),
    };
    ceil_tokens(chars as f64)
}

/// `estimateMessageTokens`: the token estimate of one message.
#[must_use]
pub fn estimate_message_tokens(message: &Message) -> u64 {
    match message {
        Message::System(message) => {
            estimate_text_tokens(&crate::text::get_system_message_text(message))
                + estimate_tools_tokens(message.tools_added.as_deref())
                + estimate_tools_tokens(message.tools_removed.as_deref())
        }
        Message::User(user) => estimate_text_and_image_content_tokens(&user.content),
        Message::ToolResult(result) => estimate_text_and_image_content_tokens(
            &TextOrBlocksWithImages::Blocks(result.content.clone()),
        ),
        Message::Assistant(assistant) => {
            let mut chars = 0u64;
            for block in &assistant.content {
                chars += match block {
                    AssistantContentBlock::Text(text) => js_len(&text.text) as u64,
                    AssistantContentBlock::Thinking(thinking) => js_len(&thinking.thinking) as u64,
                    AssistantContentBlock::ToolCall(call) => {
                        (js_len(&call.name) + js_len(&safe_json_stringify(&call.arguments))) as u64
                    }
                };
            }
            ceil_tokens(chars as f64)
        }
    }
}

/// The most recent assistant usage that still applies to the prefix. A
/// newer prefix message (for example a compaction summary) inserted after
/// a response makes its usage stale: the assistant must be at least as
/// recent as every message before it.
fn last_assistant_usage_info(messages: &[Message]) -> Option<(&Usage, usize)> {
    let mut latest_prefix_timestamp = i64::MIN;
    let mut usage_info = None;
    for (index, message) in messages.iter().enumerate() {
        if let Message::Assistant(assistant) = message {
            let usage_applies = assistant.timestamp >= latest_prefix_timestamp
                && assistant.stop_reason != StopReason::Aborted
                && assistant.stop_reason != StopReason::Error
                && calculate_context_tokens(&assistant.usage) > 0;
            if usage_applies {
                usage_info = Some((&assistant.usage, index));
            }
        }
        latest_prefix_timestamp = latest_prefix_timestamp.max(message_timestamp(message));
    }
    usage_info
}

fn message_timestamp(message: &Message) -> i64 {
    match message {
        Message::System(m) => m.timestamp,
        Message::User(m) => m.timestamp,
        Message::Assistant(m) => m.timestamp,
        Message::ToolResult(m) => m.timestamp,
    }
}

/// `estimateContextTokens` on a message list. Upstream also accepts a
/// `TranscriptContext`; pass `context.messages()` for the same result.
#[must_use]
pub fn estimate_context_tokens(messages: &[Message]) -> ContextUsageEstimate {
    if let Some((usage, index)) = last_assistant_usage_info(messages) {
        let usage_tokens = calculate_context_tokens(usage);
        let trailing_tokens: u64 = messages
            .get(index.saturating_add(1)..)
            .unwrap_or_default()
            .iter()
            .map(estimate_message_tokens)
            .sum();
        return ContextUsageEstimate {
            tokens: usage_tokens + trailing_tokens,
            usage_tokens,
            trailing_tokens,
            last_usage_index: Some(index),
        };
    }
    let tokens: u64 = messages.iter().map(estimate_message_tokens).sum();
    ContextUsageEstimate {
        tokens,
        usage_tokens: 0,
        trailing_tokens: tokens,
        last_usage_index: None,
    }
}

/// `estimateContextTokens` on a [`TranscriptContext`].
#[must_use]
pub fn estimate_context_tokens_for_context(context: &TranscriptContext) -> ContextUsageEstimate {
    estimate_context_tokens(context.messages())
}

fn estimate_tools_tokens<T: serde::Serialize>(tools: Option<&[T]>) -> u64 {
    match tools {
        Some(tools) if !tools.is_empty() => estimate_text_tokens(&safe_json_stringify(&tools)),
        _ => 0,
    }
}
