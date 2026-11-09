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

use crate::text::render_system_message_update;
use crate::text::sanitize_surrogates;
use crate::types::content::{AssistantContentBlock, UserContentBlock};
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

/// `ANTHROPIC_STRICT_UNSUPPORTED_KEYWORDS` + `STRICT_STRING_FORMATS` —
/// the strict-tool schema blacklist.
const STRICT_UNSUPPORTED: &[&str] = &[
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
const STRICT_FORMATS: &[&str] = &[
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
fn strict_unsupported(key: &str, value: &Value) -> bool {
    if STRICT_UNSUPPORTED.contains(&key) {
        return true;
    }
    if key == "minItems" {
        return value.as_u64().is_none_or(|n| n > 1);
    }
    if key == "format" {
        return value.as_str().is_none_or(|f| !STRICT_FORMATS.contains(&f));
    }
    false
}

/// `resolveJsonSchemaStrictSampling` — `None` when the tool's schema
/// hits a strict-unsupported keyword; `Some(true)` for an explicit
/// strict request; `Some(false)` otherwise.
fn resolve_strict(tool: &Tool) -> Option<bool> {
    if schema_has_unsupported(&Value::from(tool.parameters.clone())) {
        return None;
    }
    Some(false)
}
fn schema_has_unsupported(value: &Value) -> bool {
    match value {
        Value::Object(obj) => obj
            .iter()
            .any(|(k, v)| strict_unsupported(k, v) || schema_has_unsupported(v)),
        Value::Array(items) => items.iter().any(schema_has_unsupported),
        _ => false,
    }
}

/// `getJsonSchemaToolParameters` — the `input_schema` the request sends.
fn tool_parameters(tool: &Tool, strict: Option<bool>) -> Map<String, Value> {
    let mut parameters = tool.parameters.clone();
    if strict == Some(true) {
        parameters.insert(String::from("type"), Value::from("object"));
        parameters
            .entry(String::from("properties"))
            .or_insert(json!({}));
        parameters
            .entry(String::from("required"))
            .or_insert(json!([]));
    }
    parameters
}

/// `convertTools` — each `Tool` → a `BetaTool` in upstream key order.
///
/// (`name`/`description`/`eager_input_streaming?`/`strict?`/`input_schema`/
/// `cache_control?`); `name` uses the OAuth tool-name mapping only when
/// `is_oauth_token`.
#[must_use]
pub fn convert_tools(
    tools: &[Tool],
    is_oauth_token: bool,
    supports_eager_tool_input_streaming: bool,
    cache_control: Option<&Value>,
    disable_preferred_strict_tools: bool,
) -> Vec<Value> {
    let name = |n: &str| {
        if is_oauth_token {
            to_claude_code_name(n)
        } else {
            n.to_owned()
        }
    };
    tools
        .iter()
        .enumerate()
        .map(|(index, tool)| {
            let strict = if disable_preferred_strict_tools {
                None
            } else {
                resolve_strict(tool)
            };
            let parameters = tool_parameters(tool, strict);
            let mut result = Map::new();
            result.insert("name".into(), Value::from(name(&tool.name)));
            result.insert(
                "description".into(),
                Value::from(sanitize_surrogates(&tool.description)),
            );
            if supports_eager_tool_input_streaming {
                result.insert("eager_input_streaming".into(), Value::from(true));
            }
            if strict == Some(true) {
                result.insert("strict".into(), Value::from(true));
            }
            result.insert("input_schema".into(), Value::from(parameters));
            if let Some(cc) = cache_control
                && index == tools.len() - 1
            {
                result.insert("cache_control".into(), cc.clone());
            }
            Value::from(result)
        })
        .collect()
}

/// `convertMessages` — upstream's role-block pipeline in key order.
/// `pendingSystem` merges before the next assistant message (or the end);
/// consecutive `toolResult`s join into one `user` role entry.
#[expect(
    clippy::excessive_nesting,
    clippy::cognitive_complexity,
    clippy::too_many_lines,
    reason = "the structure mirrors the upstream role-block dispatch; splitting it obscures the case order"
)]
#[must_use]
pub fn convert_messages(
    transformed_messages: &[Message],
    is_oauth_token: bool,
    allow_empty_signature: bool,
    _managed_provider: Option<&str>,
    cache_control: Option<&Value>,
) -> Vec<Value> {
    let name = |n: &str| {
        if is_oauth_token {
            to_claude_code_name(n)
        } else {
            n.to_owned()
        }
    };
    let mut params: Vec<Value> = Vec::new();
    let mut pending_system: Vec<Value> = Vec::new();
    let mut index = 0;
    while index < transformed_messages.len() {
        let Some(msg) = transformed_messages.get(index) else {
            break;
        };
        index += 1;
        match msg {
            Message::System(system) => {
                let text = render_system_message_update(system);
                if !text.is_empty() {
                    pending_system.push(json!({
                        "role": "system",
                        "content": [{"type":"text","text":sanitize_surrogates(&text)}],
                    }));
                }
            }
            Message::User(user) => {
                let content = match &user.content {
                    crate::types::content::TextOrBlocksWithImages::Text(text) => {
                        if text.trim().is_empty() {
                            continue;
                        }
                        json!({"role":"user","content":sanitize_surrogates(text)})
                    }
                    crate::types::content::TextOrBlocksWithImages::Blocks(items) => {
                        let blocks: Vec<Value> = items
                            .iter()
                            .map(|item| match item {
                                UserContentBlock::Text(t) => {
                                    json!({"type":"text","text":sanitize_surrogates(&t.text)})
                                }
                                UserContentBlock::Image(i) => convert_image_block(i),
                            })
                            .filter(|b| {
                                b.get("type") != Some(&json!("text"))
                                    || b["text"].as_str().is_none_or(|t| !t.trim().is_empty())
                            })
                            .collect();
                        if blocks.is_empty() {
                            continue;
                        }
                        json!({"role":"user","content":blocks})
                    }
                };
                params.push(content);
            }
            Message::Assistant(assistant) => {
                let mut blocks: Vec<Value> = Vec::new();
                for block in &assistant.content {
                    match block {
                        AssistantContentBlock::Text(t) => {
                            if t.text.trim().is_empty() {
                                continue;
                            }
                            blocks.push(json!({"type":"text","text":sanitize_surrogates(&t.text)}));
                        }
                        AssistantContentBlock::Thinking(t) => {
                            if t.redacted.unwrap_or(false) {
                                blocks.push(
                                    json!({"type":"redacted_thinking","data":t.thinking_signature}),
                                );
                                continue;
                            }
                            let sig = t.thinking_signature.as_deref().unwrap_or_default();
                            let has_sig = !sig.trim().is_empty();
                            if t.thinking.trim().is_empty() && !has_sig {
                                continue;
                            }
                            if has_sig {
                                blocks.push(json!({"type":"thinking","thinking":sanitize_surrogates(&t.thinking),"signature":sig}));
                            } else if allow_empty_signature {
                                blocks.push(json!({"type":"thinking","thinking":sanitize_surrogates(&t.thinking),"signature":""}));
                            } else {
                                blocks.push(
                                    json!({"type":"text","text":sanitize_surrogates(&t.thinking)}),
                                );
                            }
                        }
                        AssistantContentBlock::ToolCall(t) => {
                            blocks.push(json!({"type":"tool_use","id":t.id,"name":name(&t.name),"input":Value::from(t.arguments.clone())}));
                        }
                    }
                }
                if blocks.is_empty() {
                    continue;
                }
                params.append(&mut pending_system);
                params.push(json!({"role":"assistant","content":blocks}));
            }
            Message::ToolResult(_result) => {
                // Merge consecutive toolResult messages into one `user`
                // entry; error-result images follow via
                // `splitErrorToolResultImages` (Anthropic rejects
                // error tool results that contain images).
                let mut tool_results: Vec<Value> = Vec::new();
                let mut error_images: Vec<Value> = Vec::new();
                let mut j = index - 1;
                while j < transformed_messages.len() {
                    let Some(Message::ToolResult(tool_msg)) = transformed_messages.get(j) else {
                        break;
                    };
                    let split = crate::transform::split_error_tool_result_images(tool_msg);
                    tool_results.push(convert_tool_result(
                        tool_msg,
                        split.as_ref().map(|s| s.content.as_slice()),
                    ));
                    if let Some(split) = split {
                        error_images.push(json!({"type":"text","text":split.caption}));
                        error_images.extend(split.images.iter().map(convert_image_block));
                    }
                    j += 1;
                }
                // `index` was already advanced past the current
                // message at the loop top; `j` is the first index past
                // the merged run.
                index = j;
                tool_results.append(&mut error_images);
                params.push(json!({"role":"user","content":tool_results}));
            }
        }
    }
    params.append(&mut pending_system);
    // `cache_control` lands on the last user/system message's final
    // content block (or wraps a string content), matching upstream.
    if let Some(cc) = cache_control
        && let Some(last) = params.last_mut()
        && (last.get("role") == Some(&json!("user")) || last.get("role") == Some(&json!("system")))
    {
        if let Some(Value::Array(blocks)) = last.get_mut("content") {
            if let Some(block) = blocks.last_mut()
                && let Some(Value::String(t)) = block.get("type")
                && matches!(
                    t.as_str(),
                    "text" | "image" | "tool_result" | "tool_addition" | "tool_removal"
                )
                && let Some(obj) = block.as_object_mut()
            {
                obj.insert(String::from("cache_control"), cc.clone());
            }
        } else if let Some(Value::String(text)) = last.get("content") {
            let text = text.clone();
            last["content"] = json!([{"type":"text","text":text,"cache_control":cc}]);
        }
    }
    params
}
