//! Cross-model message transformation: upstream
//! `api/transform-messages.ts`.
//!
//! Lives at `src/transform.rs` rather than `src/api/` while no provider
//! adapters exist yet; it is pure transcript logic and later API modules
//! will import it from here.

use std::collections::{BTreeMap, BTreeSet};

use crate::types::content::{
    AssistantContentBlock, ImageContent, StopReason, TextContent, UserContentBlock,
};
use crate::types::message::{AssistantMessage, Message, ToolResultMessage};
use crate::types::model::{Model, ModelInputModality};

/// `(id, model, source) -> normalized id` — the upstream
/// `normalizeToolCallId` hook.
pub type NormalizeToolCallId<'a> = &'a dyn Fn(&str, &Model, &AssistantMessage) -> String;

const NON_VISION_USER_IMAGE_PLACEHOLDER: &str = "(image omitted: model does not support images)";
const NON_VISION_TOOL_IMAGE_PLACEHOLDER: &str =
    "(tool image omitted: model does not support images)";
const ERROR_TOOL_IMAGE_PLACEHOLDER: &str = "(image attached after the tool results)";

/// `SplitErrorToolResult`: an error tool result split into a text-only
/// result and the images to send after the tool results.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SplitErrorToolResult {
    /// The result content with each run of images replaced by a note.
    pub content: Vec<UserContentBlock>,
    /// Text naming the tool call, sent before `images`.
    pub caption: String,
    /// The extracted images (serialized with the `image` block type tag).
    #[serde(serialize_with = "serialize_images")]
    pub images: Vec<ImageContent>,
}

/// `splitErrorToolResultImages`: split the images out of an error tool
/// result for APIs that reject images in error results. `None` for results
/// that need no split.
#[must_use]
pub fn split_error_tool_result_images(msg: &ToolResultMessage) -> Option<SplitErrorToolResult> {
    if !msg.is_error {
        return None;
    }
    let images: Vec<ImageContent> = msg
        .content
        .iter()
        .filter_map(|block| match block {
            UserContentBlock::Image(image) => Some(image.clone()),
            UserContentBlock::Text(_) => None,
        })
        .collect();
    if images.is_empty() {
        return None;
    }
    Some(SplitErrorToolResult {
        content: replace_images_with_placeholder(&msg.content, ERROR_TOOL_IMAGE_PLACEHOLDER),
        caption: format!(
            "Images from the error result of tool call {} ({}):",
            msg.tool_call_id, msg.tool_name
        ),
        images,
    })
}

fn replace_images_with_placeholder(
    content: &[UserContentBlock],
    placeholder: &str,
) -> Vec<UserContentBlock> {
    let mut result = Vec::new();
    let mut previous_was_placeholder = false;
    for block in content {
        match block {
            UserContentBlock::Image(_) => {
                if !previous_was_placeholder {
                    result.push(UserContentBlock::Text(TextContent {
                        text: placeholder.to_owned(),
                        text_signature: None,
                    }));
                }
                previous_was_placeholder = true;
            }
            UserContentBlock::Text(text) => {
                result.push(block.clone());
                previous_was_placeholder = text.text == placeholder;
            }
        }
    }
    result
}

fn downgrade_unsupported_images(messages: Vec<Message>, model: &Model) -> Vec<Message> {
    if model.input.contains(&ModelInputModality::Image) {
        return messages;
    }
    messages
        .into_iter()
        .map(|mut msg| {
            match &mut msg {
                Message::User(user) => {
                    if let crate::types::content::TextOrBlocksWithImages::Blocks(blocks) =
                        &mut user.content
                    {
                        *blocks = replace_images_with_placeholder(
                            blocks,
                            NON_VISION_USER_IMAGE_PLACEHOLDER,
                        );
                    }
                }
                Message::ToolResult(result) => {
                    result.content = replace_images_with_placeholder(
                        &result.content,
                        NON_VISION_TOOL_IMAGE_PLACEHOLDER,
                    );
                }
                Message::System(_) | Message::Assistant(_) => {}
            }
            msg
        })
        .collect()
}

/// `transformMessages`: downgrade unsupported images, drop other models'
/// thinking, normalize tool call ids, and insert synthetic results for
/// orphaned tool calls.
///
/// `normalize_tool_call_id` is called as `(id, model, source)` like
/// upstream.
#[must_use]
pub fn transform_messages(
    messages: &[Message],
    model: &Model,
    normalize_tool_call_id: Option<NormalizeToolCallId<'_>>,
) -> Vec<Message> {
    let mut tool_call_id_map: BTreeMap<String, String> = BTreeMap::new();
    // Normalize null/undefined content from untyped callers: our `Message`
    // cannot carry missing content, so this pass is a no-op by construction.
    let normalized = messages.to_vec();
    let image_aware = downgrade_unsupported_images(normalized, model);

    let transformed: Vec<Message> = image_aware
        .into_iter()
        .map(|mut msg| {
            if let Message::ToolResult(tool_result) = &mut msg
                && let Some(mapped_id) = tool_call_id_map.get(&tool_result.tool_call_id)
                && mapped_id != &tool_result.tool_call_id
            {
                tool_result.tool_call_id.clone_from(mapped_id);
            }
            if let Message::Assistant(assistant) = &mut msg {
                let same_model = is_same_model(assistant, model);
                let mut content = Vec::with_capacity(assistant.content.len());
                for block in &assistant.content {
                    transform_block(
                        block,
                        assistant,
                        model,
                        same_model,
                        normalize_tool_call_id,
                        &mut tool_call_id_map,
                        &mut content,
                    );
                }
                assistant.content = content;
            }
            msg
        })
        .collect();

    // Second pass: synthetic empty tool results for orphaned tool calls.
    let mut result = Vec::with_capacity(transformed.len());
    let mut pending_tool_calls: Vec<crate::types::content::ToolCall> = Vec::new();
    let mut existing_result_ids: BTreeSet<String> = BTreeSet::new();
    // System messages are transparent to tool-call accounting: one that
    // lands between a call and its results is emitted after the results so
    // it never causes a duplicate result for a call answered later.
    let mut held_system_messages = Vec::new();
    let close_pending = |out: &mut Vec<Message>,
                         pending: &mut Vec<crate::types::content::ToolCall>,
                         existing: &mut BTreeSet<String>,
                         held: &mut Vec<Message>| {
        for call in pending.drain(..) {
            if !existing.contains(&call.id) {
                out.push(Message::ToolResult(ToolResultMessage {
                    tool_call_id: call.id,
                    tool_name: call.name,
                    content: vec![UserContentBlock::Text(TextContent {
                        text: "No result provided".to_owned(),
                        text_signature: None,
                    })],
                    details: None,
                    usage: None,
                    nested_calls: None,
                    is_error: true,
                    timestamp: crate::stream::accumulator::wall_ms(),
                    duration_ms: None,
                }));
            }
        }
        existing.clear();
        out.append(held);
    };

    for msg in transformed {
        match &msg {
            Message::Assistant(assistant) => {
                close_pending(
                    &mut result,
                    &mut pending_tool_calls,
                    &mut existing_result_ids,
                    &mut held_system_messages,
                );
                if matches!(
                    assistant.stop_reason,
                    StopReason::Error | StopReason::Aborted
                ) {
                    continue;
                }
                let tool_calls: Vec<crate::types::content::ToolCall> = assistant
                    .content
                    .iter()
                    .filter_map(|block| match block {
                        AssistantContentBlock::ToolCall(call) => Some(call.clone()),
                        _ => None,
                    })
                    .collect();
                if !tool_calls.is_empty() {
                    pending_tool_calls = tool_calls;
                    existing_result_ids = BTreeSet::new();
                }
                result.push(msg);
            }
            Message::ToolResult(tool_result) => {
                existing_result_ids.insert(tool_result.tool_call_id.clone());
                result.push(msg);
            }
            Message::System(_) => {
                if pending_tool_calls.is_empty() {
                    result.push(msg);
                } else {
                    held_system_messages.push(msg);
                }
            }
            Message::User(_) => {
                close_pending(
                    &mut result,
                    &mut pending_tool_calls,
                    &mut existing_result_ids,
                    &mut held_system_messages,
                );
                result.push(msg);
            }
        }
    }
    close_pending(
        &mut result,
        &mut pending_tool_calls,
        &mut existing_result_ids,
        &mut held_system_messages,
    );
    result
}

fn is_same_model(assistant: &AssistantMessage, model: &Model) -> bool {
    assistant.provider == model.provider
        && assistant.api == model.api
        && assistant.model.eq(&model.id)
}

/// First-pass per-block transformation of one assistant message.
fn transform_block(
    block: &AssistantContentBlock,
    assistant: &AssistantMessage,
    model: &Model,
    is_same_model: bool,
    normalize_tool_call_id: Option<NormalizeToolCallId<'_>>,
    tool_call_id_map: &mut BTreeMap<String, String>,
    content: &mut Vec<AssistantContentBlock>,
) {
    match block {
        AssistantContentBlock::Thinking(thinking) => {
            // Thinking from another model is dropped; signatures and
            // encrypted reasoning are only valid for the model that
            // produced them. Keep redacted and signed blocks even when the
            // thinking text is empty.
            if !is_same_model {
                return;
            }
            let signed = thinking.redacted.is_some() || thinking.thinking_signature.is_some();
            if !signed && crate::text::js_trim(&thinking.thinking).is_empty() {
                return;
            }
            content.push(block.clone());
        }
        AssistantContentBlock::Text(text) => {
            if is_same_model {
                content.push(block.clone());
            } else {
                content.push(AssistantContentBlock::Text(TextContent {
                    text: text.text.clone(),
                    text_signature: None,
                }));
            }
        }
        AssistantContentBlock::ToolCall(tool_call) => {
            let mut normalized = tool_call.clone();
            if !is_same_model {
                normalized.thought_signature = None;
                if let Some(normalize) = normalize_tool_call_id {
                    normalize_tool_call_id_for_model(
                        tool_call,
                        assistant,
                        model,
                        normalize,
                        tool_call_id_map,
                        &mut normalized,
                    );
                }
            }
            content.push(AssistantContentBlock::ToolCall(normalized));
        }
    }
}

/// Cross-model tool call id normalization (`normalize_tool_call_id` hook).
fn normalize_tool_call_id_for_model(
    tool_call: &crate::types::content::ToolCall,
    assistant: &AssistantMessage,
    model: &Model,
    normalize: NormalizeToolCallId<'_>,
    tool_call_id_map: &mut BTreeMap<String, String>,
    normalized: &mut crate::types::content::ToolCall,
) {
    let normalized_id = normalize(&tool_call.id, model, assistant);
    if normalized_id != tool_call.id {
        tool_call_id_map.insert(tool_call.id.clone(), normalized_id.clone());
        normalized.id = normalized_id;
    }
}

/// Wire shape: image blocks carry their `type` tag.
fn serialize_images<S: serde::Serializer>(
    images: &[ImageContent],
    serializer: S,
) -> Result<S::Ok, S::Error> {
    serializer.collect_seq(images.iter().map(|image| {
        let mut value = serde_json::to_value(image).unwrap_or_default();
        if let serde_json::Value::Object(members) = &mut value {
            let mut tagged = serde_json::Map::with_capacity(members.len() + 1);
            tagged.insert("type".to_owned(), serde_json::Value::from("image"));
            tagged.append(members);
            value = serde_json::Value::Object(tagged);
        }
        value
    }))
}
