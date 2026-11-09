//! The per-request headers GitHub Copilot expects (upstream
//! `github-copilot-headers.ts`).

use serde_json::{Map, Value};

use crate::types::content::{TextOrBlocksWithImages, UserContentBlock};
use crate::types::message::Message;

/// `buildCopilotDynamicHeaders`: who initiated the request, the intent, and
/// whether it carries images.
#[must_use]
pub fn dynamic_headers(messages: &[Message]) -> Map<String, Value> {
    let mut headers = Map::new();
    headers.insert("X-Initiator".into(), Value::from(initiator(messages)));
    headers.insert("Openai-Intent".into(), Value::from("conversation-edits"));
    if has_vision_input(messages) {
        headers.insert("Copilot-Vision-Request".into(), Value::from("true"));
    }
    headers
}

/// `inferCopilotInitiator`: a request following anything but a user message
/// is an agent follow-up.
const fn initiator(messages: &[Message]) -> &'static str {
    match messages.last() {
        Some(Message::User(_)) | None => "user",
        Some(_) => "agent",
    }
}

/// `hasCopilotVisionInput`: a user message or tool result carries an image.
fn has_vision_input(messages: &[Message]) -> bool {
    let has_image = |blocks: &[UserContentBlock]| {
        blocks
            .iter()
            .any(|block| matches!(block, UserContentBlock::Image(_)))
    };
    messages.iter().any(|message| match message {
        Message::User(user) => match &user.content {
            TextOrBlocksWithImages::Blocks(blocks) => has_image(blocks),
            TextOrBlocksWithImages::Text(_) => false,
        },
        Message::ToolResult(result) => has_image(&result.content),
        Message::System(_) | Message::Assistant(_) => false,
    })
}
