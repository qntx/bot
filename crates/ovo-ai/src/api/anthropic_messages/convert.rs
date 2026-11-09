//! `anthropic-messages` conversion.
//!
//! Upstream `convertMessages`/`convertTools`/`convertToolResult`/
//! `normalizeToolCallId` ported against `serde_json::Value` in the
//! upstream object-literal key order; `sanitize_surrogates` runs on
//! every text upstream sanitizes.
//!
//! Not yet ported (checkpoint B): `insertThinkingLevelMessages`,
//! `tool_addition`/`tool_removal` native-change blocks, pending-system
//! flush ordering details, federation/copilot/`onPayload`.

use serde_json::{Map, Value, json};

use crate::api::constrained_sampling::{
    json_schema_tool_parameters, prefers_json_schema_strict_sampling,
    resolve_json_schema_strict_sampling,
};
use crate::text::{js_trim, render_system_message_update, sanitize_surrogates};
use crate::types::content::{AssistantContentBlock, TextOrBlocksWithImages, UserContentBlock};
use crate::types::message::{Message, ToolResultMessage};
use crate::types::tool::Tool;

/// `normalizeToolCallId` — `[^a-zA-Z0-9_-]` → `_`, 64-char cap.
#[must_use]
pub fn normalize_tool_call_id(id: &str) -> String {
    let mut out = String::with_capacity(id.len().min(64));
    for ch in id.chars().take(64) {
        out.push(if ch.is_ascii_alphanumeric() || ch == '_' || ch == '-' {
            ch
        } else {
            '_'
        });
    }
    out
}

/// The upstream `claudeCodeTools` canonical casing list.
const CLAUDE_CODE_TOOLS: &[&str] = &[
    "Read",
    "Write",
    "Edit",
    "Bash",
    "Grep",
    "Glob",
    "AskUserQuestion",
    "EnterPlanMode",
    "ExitPlanMode",
    "KillShell",
    "NotebookEdit",
    "Skill",
    "Task",
    "TaskOutput",
    "TodoWrite",
    "WebFetch",
    "WebSearch",
];

/// `toClaudeCodeName` — canonical Claude-Code casing (OAuth only).
#[must_use]
pub fn to_claude_code_name(name: &str) -> String {
    CLAUDE_CODE_TOOLS
        .iter()
        .find(|cc| cc.eq_ignore_ascii_case(name))
        .map_or_else(|| name.to_owned(), |cc| (*cc).to_owned())
}

/// `fromClaudeCodeName` — prefer the transcript-defined tool spelling.
#[must_use]
pub fn from_claude_code_name(name: &str, tools: &[Tool]) -> String {
    tools
        .iter()
        .find(|t| t.name.eq_ignore_ascii_case(name))
        .map_or_else(|| name.to_owned(), |t| t.name.clone())
}

/// `convertImageBlock` — `{type:"image",source:{type:"base64",media_type,data}}`.
fn convert_image_block(image: &crate::types::content::ImageContent) -> Value {
    json!({
        "type": "image",
        "source": {"type": "base64", "media_type": image.mime_type, "data": image.data},
    })
}

/// `convertContentBlocks` — text-only joins to a `"\n"` string; any
/// image emits a block array, and a bare image list gets the
/// `(see attached image)` text placeholder.
#[must_use]
pub fn convert_content_blocks(content: &[UserContentBlock]) -> Value {
    let has_images = content
        .iter()
        .any(|c| matches!(c, UserContentBlock::Image(_)));
    if !has_images {
        let text = content
            .iter()
            .filter_map(|c| match c {
                UserContentBlock::Text(t) => Some(t.text.as_str()),
                UserContentBlock::Image(_) => None,
            })
            .collect::<Vec<_>>()
            .join("\n");
        return Value::from(sanitize_surrogates(&text));
    }
    let mut blocks: Vec<Value> = content
        .iter()
        .map(|block| match block {
            UserContentBlock::Text(t) => {
                json!({"type": "text", "text": sanitize_surrogates(&t.text)})
            }
            UserContentBlock::Image(i) => convert_image_block(i),
        })
        .collect();
    if !blocks.iter().any(|b| b.get("type") == Some(&json!("text"))) {
        blocks.insert(0, json!({"type": "text", "text": "(see attached image)"}));
    }
    Value::from(blocks)
}

/// `convertToolResult` — `{type:"tool_result",tool_use_id,content,is_error}`.
/// `content` overrides `msg.content` when `splitErrorToolResultImages`
/// moved the images out.
#[must_use]
pub fn convert_tool_result(msg: &ToolResultMessage, content: Option<&[UserContentBlock]>) -> Value {
    json!({
        "type": "tool_result",
        "tool_use_id": msg.tool_call_id,
        "content": convert_content_blocks(content.unwrap_or(&msg.content)),
        "is_error": msg.is_error,
    })
}

/// Keywords Anthropic strict tool use rejects with a 400 for the whole request.
const STRICT_UNSUPPORTED: [&str; 11] = [
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "multipleOf",
    "maxItems",
    "uniqueItems",
    "minContains",
    "maxContains",
    "minProperties",
    "maxProperties",
];

/// String formats Anthropic strict tool use accepts.
const STRICT_FORMATS: [&str; 10] = [
    "date-time",
    "time",
    "date",
    "duration",
    "email",
    "hostname",
    "uri",
    "ipv4",
    "ipv6",
    "uuid",
];

/// `isAnthropicStrictUnsupportedKeyword`.
#[expect(clippy::float_cmp, reason = "JS strict equality against 0 and 1")]
fn strict_unsupported(key: &str, value: &Value) -> bool {
    if STRICT_UNSUPPORTED.contains(&key) {
        return true;
    }
    match key {
        "minItems" => !value.as_f64().is_some_and(|n| n == 0.0 || n == 1.0),
        "format" => !value.as_str().is_some_and(|f| STRICT_FORMATS.contains(&f)),
        _ => false,
    }
}

/// `convertTools`: each `Tool` as a `BetaTool` in upstream key order.
///
/// # Errors
/// Returns upstream's message when a tool requires strict sampling that the
/// model or its schema cannot provide.
pub fn convert_tools(
    tools: &[Tool],
    compat: &super::AnthropicMessagesCompat,
    is_oauth_token: bool,
    cache_control: Option<&Value>,
    disable_preferred_strict_tools: bool,
) -> Result<Vec<Value>, String> {
    tools
        .iter()
        .enumerate()
        .map(|(index, tool)| {
            let strict =
                if disable_preferred_strict_tools && prefers_json_schema_strict_sampling(tool) {
                    None
                } else {
                    resolve_json_schema_strict_sampling(
                        tool,
                        compat.supports_strict_tools,
                        Some(strict_unsupported),
                    )
                    .map_err(|error| error.to_string())?
                };
            let parameters =
                json_schema_tool_parameters(tool, strict).map_err(|error| error.to_string())?;
            let mut legacy = Map::new();
            legacy.insert("type".into(), Value::from("object"));
            legacy.insert(
                "properties".into(),
                parameters
                    .get("properties")
                    .cloned()
                    .unwrap_or_else(|| json!({})),
            );
            legacy.insert(
                "required".into(),
                parameters
                    .get("required")
                    .cloned()
                    .unwrap_or_else(|| json!([])),
            );
            let input_schema = if strict == Some(true) {
                let mut merged = parameters;
                merged.extend(legacy);
                merged
            } else {
                legacy
            };
            let mut result = Map::new();
            let name = if is_oauth_token {
                to_claude_code_name(&tool.name)
            } else {
                tool.name.clone()
            };
            result.insert("name".into(), Value::from(name));
            result.insert("description".into(), Value::from(tool.description.clone()));
            if compat.supports_eager_tool_input_streaming {
                result.insert("eager_input_streaming".into(), Value::from(true));
            }
            if strict == Some(true) {
                result.insert("strict".into(), Value::from(true));
            }
            result.insert("input_schema".into(), Value::Object(input_schema));
            if let Some(cache_control) = cache_control
                && index + 1 == tools.len()
            {
                result.insert("cache_control".into(), cache_control.clone());
            }
            Ok(Value::Object(result))
        })
        .collect()
}

/// The converted conversation plus the effort level recorded on each
/// assistant message the managed-effort provider produced.
#[derive(Clone, Debug, Default)]
pub struct ConvertedMessages {
    /// `MessageParam`s in request order.
    pub messages: Vec<Value>,
    /// Index into `messages` of an assistant message and the effort it used.
    pub assistant_levels: Vec<(usize, String)>,
}

/// Converts tool definitions for native `tool_addition` blocks.
pub type ToolDefinitions<'a> = &'a dyn Fn(&[Tool]) -> Result<Vec<Value>, String>;

/// `convertMessages`: transcript messages as Anthropic `MessageParam`s.
///
/// Later system messages are held back until the next assistant message that
/// is sent (or the end), consecutive tool results share one user message, and
/// the last user or system message carries the conversation cache breakpoint.
///
/// # Errors
/// Returns the error of `tool_definitions` for a native tool change.
pub fn convert_messages(
    messages: &[Message],
    is_oauth_token: bool,
    cache_control: Option<&Value>,
    allow_empty_signature: bool,
    managed_provider: Option<&str>,
    tool_definitions: Option<ToolDefinitions<'_>>,
) -> Result<ConvertedMessages, String> {
    let mut converted = ConvertedMessages::default();
    let mut pending_system: Vec<Value> = Vec::new();
    let mut index = 0;
    while let Some(message) = messages.get(index) {
        index += 1;
        match message {
            Message::System(system) => {
                let blocks = system_blocks(system, is_oauth_token, tool_definitions)?;
                if !blocks.is_empty() {
                    pending_system.push(json!({ "role": "system", "content": blocks }));
                }
            }
            Message::User(user) => {
                if let Some(param) = user_param(user) {
                    converted.messages.push(param);
                }
            }
            Message::Assistant(assistant) => {
                let blocks = assistant_blocks(assistant, is_oauth_token, allow_empty_signature);
                if blocks.is_empty() {
                    continue;
                }
                converted.messages.append(&mut pending_system);
                let position = converted.messages.len();
                converted
                    .messages
                    .push(json!({ "role": "assistant", "content": blocks }));
                if let Some(level) = assistant
                    .provider_thinking_level
                    .as_deref()
                    .filter(|level| is_anthropic_effort(level))
                    .filter(|_| {
                        assistant.api.as_str() == super::API
                            && managed_provider == Some(assistant.provider.as_str())
                    })
                {
                    converted
                        .assistant_levels
                        .push((position, level.to_owned()));
                }
            }
            Message::ToolResult(_) => {
                let (param, next) = tool_results_param(messages, index - 1);
                converted.messages.push(param);
                index = next;
            }
        }
    }
    converted.messages.append(&mut pending_system);
    if let Some(cache_control) = cache_control {
        mark_last_message(&mut converted.messages, cache_control);
    }
    Ok(converted)
}

/// `isAnthropicEffort`.
fn is_anthropic_effort(level: &str) -> bool {
    matches!(level, "low" | "medium" | "high" | "xhigh" | "max")
}

/// A later system message: its rendered update text and, when tool changes
/// are native, `tool_removal` and `tool_addition` blocks.
fn system_blocks(
    system: &crate::types::message::SystemMessage,
    is_oauth_token: bool,
    tool_definitions: Option<ToolDefinitions<'_>>,
) -> Result<Vec<Value>, String> {
    let mut blocks = Vec::new();
    let text = render_system_message_update(system);
    if !text.is_empty() {
        blocks.push(json!({ "type": "text", "text": sanitize_surrogates(&text) }));
    }
    let Some(tool_definitions) = tool_definitions else {
        return Ok(blocks);
    };
    let added = system.tools_added.as_deref().unwrap_or_default();
    for tool in system.tools_removed.iter().flatten() {
        // A new definition under the same name replaces the old one.
        if added.iter().any(|addition| addition.name == tool.name) {
            continue;
        }
        let name = if is_oauth_token {
            to_claude_code_name(&tool.name)
        } else {
            tool.name.clone()
        };
        blocks.push(json!({
            "type": "tool_removal",
            "tool": { "type": "tool_reference", "name": name },
        }));
    }
    for definition in tool_definitions(added)? {
        blocks.push(json!({
            "type": "tool_addition",
            "tool": { "type": "tool_definition", "definition": definition },
        }));
    }
    Ok(blocks)
}

/// A user message, `None` when nothing but whitespace remains.
fn user_param(user: &crate::types::message::UserMessage) -> Option<Value> {
    match &user.content {
        TextOrBlocksWithImages::Text(text) => (!js_trim(text).is_empty())
            .then(|| json!({ "role": "user", "content": sanitize_surrogates(text) })),
        TextOrBlocksWithImages::Blocks(items) => {
            let blocks: Vec<Value> = items
                .iter()
                .filter_map(|item| match item {
                    UserContentBlock::Text(text) => (!js_trim(&text.text).is_empty()).then(
                        || json!({ "type": "text", "text": sanitize_surrogates(&text.text) }),
                    ),
                    UserContentBlock::Image(image) => Some(convert_image_block(image)),
                })
                .collect();
            (!blocks.is_empty()).then(|| json!({ "role": "user", "content": blocks }))
        }
    }
}

/// The content blocks of an assistant message.
fn assistant_blocks(
    assistant: &crate::types::message::AssistantMessage,
    is_oauth_token: bool,
    allow_empty_signature: bool,
) -> Vec<Value> {
    let mut blocks = Vec::new();
    for block in &assistant.content {
        match block {
            AssistantContentBlock::Text(text) => {
                if !js_trim(&text.text).is_empty() {
                    blocks.push(json!({ "type": "text", "text": sanitize_surrogates(&text.text) }));
                }
            }
            AssistantContentBlock::Thinking(thinking) => {
                if let Some(converted) = thinking_block(thinking, allow_empty_signature) {
                    blocks.push(converted);
                }
            }
            AssistantContentBlock::ToolCall(call) => {
                let name = if is_oauth_token {
                    to_claude_code_name(&call.name)
                } else {
                    call.name.clone()
                };
                blocks.push(json!({
                    "type": "tool_use",
                    "id": call.id,
                    "name": name,
                    "input": Value::Object(call.arguments.clone()),
                }));
            }
        }
    }
    blocks
}

/// A thinking block: redacted payloads pass back as `redacted_thinking`; a
/// missing signature (for example from an aborted stream) turns the thinking
/// into text unless the model accepts empty signatures.
fn thinking_block(
    thinking: &crate::types::content::ThinkingContent,
    allow_empty_signature: bool,
) -> Option<Value> {
    if thinking.redacted == Some(true) {
        return Some(json!({ "type": "redacted_thinking", "data": thinking.thinking_signature }));
    }
    let signature = thinking
        .thinking_signature
        .as_deref()
        .filter(|signature| !js_trim(signature).is_empty());
    if js_trim(&thinking.thinking).is_empty() && signature.is_none() {
        return None;
    }
    let text = sanitize_surrogates(&thinking.thinking);
    Some(match signature {
        Some(signature) => json!({ "type": "thinking", "thinking": text, "signature": signature }),
        None if allow_empty_signature => {
            json!({ "type": "thinking", "thinking": text, "signature": "" })
        }
        None => json!({ "type": "text", "text": text }),
    })
}

/// The tool results starting at `start` merged into one user message, with
/// error-result images after every result (Anthropic rejects images inside
/// error results); returns the index past the run.
fn tool_results_param(messages: &[Message], start: usize) -> (Value, usize) {
    let mut results: Vec<Value> = Vec::new();
    let mut error_images: Vec<Value> = Vec::new();
    let mut next = start;
    while let Some(Message::ToolResult(result)) = messages.get(next) {
        let split = crate::transform::split_error_tool_result_images(result);
        results.push(convert_tool_result(
            result,
            split.as_ref().map(|split| split.content.as_slice()),
        ));
        if let Some(split) = split {
            error_images.push(json!({ "type": "text", "text": split.caption }));
            error_images.extend(split.images.iter().map(convert_image_block));
        }
        next += 1;
    }
    results.append(&mut error_images);
    (json!({ "role": "user", "content": results }), next)
}

/// Put the conversation cache breakpoint on the last user or system message.
fn mark_last_message(messages: &mut [Value], cache_control: &Value) {
    let Some(last) = messages.last_mut() else {
        return;
    };
    let role = last.get("role").and_then(Value::as_str);
    if role != Some("user") && role != Some("system") {
        return;
    }
    match last.get_mut("content") {
        Some(Value::Array(blocks)) => {
            let Some(Value::Object(block)) = blocks.last_mut() else {
                return;
            };
            let marked = matches!(
                block.get("type").and_then(Value::as_str),
                Some("text" | "image" | "tool_result" | "tool_addition" | "tool_removal")
            );
            if marked {
                block.insert(String::from("cache_control"), cache_control.clone());
            }
        }
        Some(Value::String(text)) => {
            let text = std::mem::take(text);
            last["content"] =
                json!([{ "type": "text", "text": text, "cache_control": cache_control }]);
        }
        _ => {}
    }
}
