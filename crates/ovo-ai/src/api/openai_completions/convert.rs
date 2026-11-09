//! Message and tool conversion for Chat Completions (`convertMessages`,
//! `convertTools`).

use std::collections::BTreeMap;

use serde_json::{Map, Value, json};

use super::compat::OpenAiCompletionsCompat;
use crate::api::constrained_sampling::{
    GrammarFormat, grammar_tool_input, json_schema_tool_parameters,
    resolve_grammar_constrained_sampling, resolve_json_schema_strict_sampling,
};
use crate::hash::short_hash;
use crate::text::{
    get_system_message_text, js_len, js_slice, js_trim, render_system_message_update,
    replace_non_id_chars, sanitize_surrogates,
};
use crate::transcript::{
    TranscriptContext, TranscriptToolSupport, resolve_transcript, resolve_transcript_tools,
};
use crate::types::content::{
    AssistantContentBlock, TextOrBlocksWithImages, ThinkingContent, ToolCall, UserContentBlock,
};
use crate::types::message::{AssistantMessage, Message, ToolResultMessage};
use crate::types::model::{Model, ModelInputModality};
use crate::types::tool::Tool;

/// Assistant fields a thinking signature may name for raw reasoning replay.
const REASONING_FIELDS: [&str; 3] = ["reasoning", "reasoning_content", "reasoning_text"];

/// Inserted where a provider needs an assistant turn after tool results.
const TOOL_RESULTS_BRIDGE: &str = "I have processed the tool results.";

/// The tool-change support of a compat: additions only when both mid-convo
/// flags are set.
pub(super) const fn tool_support(compat: &OpenAiCompletionsCompat) -> TranscriptToolSupport {
    if compat.supports_mid_convo_system_messages && compat.supports_mid_convo_tool_additions {
        TranscriptToolSupport::Additions
    } else {
        TranscriptToolSupport::None
    }
}

/// `normalizeToolCallId`: Responses-style `call|item` ids become one id of
/// allowed characters within 40 (hashed when longer); `openai` truncates
/// other ids to 40.
#[must_use]
pub fn normalize_tool_call_id(id: &str, model: &Model) -> String {
    if let Some((call, item)) = id.split_once('|') {
        let call_id = replace_non_id_chars(call);
        let item_id = replace_non_id_chars(item);
        let combined = if item_id.is_empty() {
            call_id.clone()
        } else {
            format!("{call_id}_{item_id}")
        };
        if combined.len() <= 40 {
            return combined;
        }
        let hash = short_hash(id);
        let hash = hash.get(..8).unwrap_or(&hash);
        let prefix = call_id
            .get(..40_usize.saturating_sub(hash.len() + 1).max(1))
            .unwrap_or(&call_id);
        return format!("{prefix}_{hash}");
    }
    if model.provider.as_str() == "openai" && js_len(id) > 40 {
        return js_slice(id, 0, 40).to_owned();
    }
    id.to_owned()
}

/// `convertTools`: grammar tools as `custom` tools, the rest as function
/// tools with `strict` where the provider supports it.
///
/// # Errors
/// Returns upstream's message for a grammar or strict tool that cannot be
/// sent.
pub fn convert_tools(
    tools: &[Tool],
    compat: &OpenAiCompletionsCompat,
) -> Result<Vec<Value>, String> {
    tools
        .iter()
        .map(|tool| {
            if let Some(grammar) =
                resolve_grammar_constrained_sampling(tool, compat.supports_openai_grammar_tools)
                    .map_err(|error| error.to_string())?
            {
                let syntax = match grammar.format {
                    GrammarFormat::Lark => "lark",
                    GrammarFormat::Regex => "regex",
                };
                return Ok(json!({
                    "type": "custom",
                    "custom": {
                        "name": tool.name,
                        "description": tool.description,
                        "format": {
                            "type": "grammar",
                            "grammar": { "syntax": syntax, "definition": grammar.definition },
                        },
                    },
                }));
            }
            let strict =
                resolve_json_schema_strict_sampling(tool, compat.supports_strict_mode, None)
                    .map_err(|error| error.to_string())?;
            let parameters =
                json_schema_tool_parameters(tool, strict).map_err(|error| error.to_string())?;
            let mut function = Map::new();
            function.insert("name".into(), Value::from(tool.name.clone()));
            function.insert("description".into(), Value::from(tool.description.clone()));
            function.insert("parameters".into(), Value::Object(parameters));
            // Some providers reject unknown fields, so `strict` is only sent
            // where supported.
            if compat.supports_strict_mode {
                function.insert("strict".into(), Value::from(strict.unwrap_or(false)));
            }
            Ok(json!({ "type": "function", "function": function }))
        })
        .collect()
}

/// `convertMessages`: the transcript as Chat Completions messages.
///
/// `grammar_tool_input_properties` names the input property of each tool
/// sent as a grammar tool, so its calls replay as `custom` tool calls.
///
/// # Errors
/// Returns upstream's message when a tool cannot be sent or a grammar tool
/// call has no string input.
pub fn convert_messages(
    model: &Model,
    context: &TranscriptContext,
    compat: &OpenAiCompletionsCompat,
    grammar_tool_input_properties: &BTreeMap<String, String>,
) -> Result<Vec<Value>, String> {
    let normalized = resolve_transcript(context, Some(compat.supports_mid_convo_system_messages));
    let normalize = |id: &str, _: &Model, _: &AssistantMessage| normalize_tool_call_id(id, model);
    let transformed =
        crate::transform::transform_messages(normalized.messages(), model, Some(&normalize));
    let folded = resolve_transcript_tools(normalized.messages(), tool_support(compat))
        .folded_system_messages;
    let instruction_role = if model.reasoning && compat.supports_developer_role {
        "developer"
    } else {
        "system"
    };

    let mut params: Vec<Value> = Vec::new();
    let mut last_role: Option<&str> = None;
    let mut system_index = 0;
    let mut index = 0;
    while let Some(message) = transformed.get(index) {
        // Some providers reject a user message directly after tool results.
        if compat.requires_assistant_after_tool_result
            && last_role == Some("toolResult")
            && matches!(message, Message::User(_))
        {
            params.push(bridge());
        }
        match message {
            Message::System(system) => {
                let added = if system_index >= folded {
                    system.tools_added.as_deref().unwrap_or_default()
                } else {
                    &[]
                };
                system_index += 1;
                if !added.is_empty() {
                    params
                        .push(json!({ "role": "system", "tools": convert_tools(added, compat)? }));
                }
                let text = if index == 0 {
                    get_system_message_text(system)
                } else {
                    render_system_message_update(system)
                };
                if !text.is_empty() {
                    params.push(
                        json!({ "role": instruction_role, "content": sanitize_surrogates(&text) }),
                    );
                }
            }
            Message::User(user) => {
                let Some(converted) = convert_user(&user.content) else {
                    index += 1;
                    continue;
                };
                params.push(converted);
            }
            Message::Assistant(assistant) => {
                let Some(converted) =
                    convert_assistant(assistant, model, compat, grammar_tool_input_properties)?
                else {
                    index += 1;
                    continue;
                };
                params.push(converted);
            }
            Message::ToolResult(_) => {
                let results: Vec<&ToolResultMessage> = transformed
                    .get(index..)
                    .unwrap_or_default()
                    .iter()
                    .map_while(|next| match next {
                        Message::ToolResult(result) => Some(result),
                        _ => None,
                    })
                    .collect();
                index += results.len();
                last_role = Some(push_tool_results(&mut params, &results, model, compat));
                continue;
            }
        }
        last_role = Some(match message {
            Message::System(_) => "system",
            Message::User(_) => "user",
            Message::Assistant(_) => "assistant",
            Message::ToolResult(_) => "toolResult",
        });
        index += 1;
    }
    Ok(params)
}

fn bridge() -> Value {
    json!({ "role": "assistant", "content": TOOL_RESULTS_BRIDGE })
}

fn image_part(mime_type: &str, data: &str) -> Value {
    json!({ "type": "image_url", "image_url": { "url": format!("data:{mime_type};base64,{data}") } })
}

/// A user message; `None` when every block is empty text.
fn convert_user(content: &TextOrBlocksWithImages) -> Option<Value> {
    let blocks = match content {
        TextOrBlocksWithImages::Text(text) => {
            return Some(json!({ "role": "user", "content": sanitize_surrogates(text) }));
        }
        TextOrBlocksWithImages::Blocks(blocks) => blocks,
    };
    let parts: Vec<Value> = blocks
        .iter()
        .filter_map(|block| match block {
            UserContentBlock::Text(text) if text.text.is_empty() => None,
            UserContentBlock::Text(text) => {
                Some(json!({ "type": "text", "text": sanitize_surrogates(&text.text) }))
            }
            UserContentBlock::Image(image) => Some(image_part(&image.mime_type, &image.data)),
        })
        .collect();
    (!parts.is_empty()).then(|| json!({ "role": "user", "content": parts }))
}

/// A run of consecutive tool results, followed by one user message with
/// their images; returns the role the next message follows.
fn push_tool_results(
    params: &mut Vec<Value>,
    results: &[&ToolResultMessage],
    model: &Model,
    compat: &OpenAiCompletionsCompat,
) -> &'static str {
    let accepts_images = model.input.contains(&ModelInputModality::Image);
    let mut images: Vec<Value> = Vec::new();
    for result in results {
        let text = result
            .content
            .iter()
            .filter_map(|block| match block {
                UserContentBlock::Text(text) => Some(text.text.as_str()),
                UserContentBlock::Image(_) => None,
            })
            .collect::<Vec<_>>()
            .join("\n");
        let has_images = result
            .content
            .iter()
            .any(|block| matches!(block, UserContentBlock::Image(_)));
        let content = if !js_trim(&text).is_empty() {
            text.as_str()
        } else if has_images {
            "(see attached image)"
        } else {
            "(no tool output)"
        };
        let mut message = Map::new();
        message.insert("role".into(), Value::from("tool"));
        message.insert("content".into(), Value::from(sanitize_surrogates(content)));
        message.insert(
            "tool_call_id".into(),
            Value::from(result.tool_call_id.clone()),
        );
        if compat.requires_tool_result_name && !result.tool_name.is_empty() {
            message.insert("name".into(), Value::from(result.tool_name.clone()));
        }
        params.push(Value::Object(message));
        if has_images && accepts_images {
            images.extend(result.content.iter().filter_map(|block| match block {
                UserContentBlock::Image(image) => Some(image_part(&image.mime_type, &image.data)),
                UserContentBlock::Text(_) => None,
            }));
        }
    }
    if images.is_empty() {
        return "toolResult";
    }
    if compat.requires_assistant_after_tool_result {
        params.push(bridge());
    }
    let mut content =
        vec![json!({ "type": "text", "text": "Attached image(s) from tool result:" })];
    content.extend(images);
    params.push(json!({ "role": "user", "content": content }));
    "user"
}

/// An assistant message; `None` when it has neither content nor tool calls
/// (an aborted reply), which providers reject.
fn convert_assistant(
    assistant: &AssistantMessage,
    model: &Model,
    compat: &OpenAiCompletionsCompat,
    grammar_tool_input_properties: &BTreeMap<String, String>,
) -> Result<Option<Value>, String> {
    let mut texts: Vec<String> = Vec::new();
    let mut thinking: Vec<&ThinkingContent> = Vec::new();
    let mut calls: Vec<&ToolCall> = Vec::new();
    for block in &assistant.content {
        match block {
            AssistantContentBlock::Text(text) if !js_trim(&text.text).is_empty() => {
                texts.push(sanitize_surrogates(&text.text).to_owned());
            }
            AssistantContentBlock::Thinking(block) => thinking.push(block),
            AssistantContentBlock::ToolCall(call) => calls.push(call),
            AssistantContentBlock::Text(_) => {}
        }
    }
    let text = texts.concat();
    let preserved_details = reasoning_details(&thinking, &calls);

    let mut message = Map::new();
    message.insert("role".into(), Value::from("assistant"));
    // Some providers reject null content.
    message.insert(
        "content".into(),
        if compat.requires_assistant_after_tool_result {
            Value::from("")
        } else {
            Value::Null
        },
    );
    let thinking: Vec<&ThinkingContent> = thinking
        .into_iter()
        .filter(|block| !js_trim(&block.thinking).is_empty())
        .collect();
    if !thinking.is_empty() && compat.requires_thinking_as_text {
        let joined = thinking
            .iter()
            .map(|block| sanitize_surrogates(&block.thinking))
            .collect::<Vec<_>>()
            .join("\n\n");
        let mut parts = vec![json!({ "type": "text", "text": joined })];
        parts.extend(
            texts
                .iter()
                .map(|part| json!({ "type": "text", "text": part })),
        );
        message.insert("content".into(), Value::from(parts));
    } else {
        // Content is always a plain string: an array of text parts makes some
        // models mirror the structure in their output.
        if !text.is_empty() {
            message.insert("content".into(), Value::from(text));
        }
        if preserved_details.is_none()
            && let Some(field) = reasoning_field(model, &thinking)
        {
            let joined = thinking
                .iter()
                .map(|block| block.thinking.as_str())
                .collect::<Vec<_>>()
                .join("\n");
            message.insert(field.into(), Value::from(joined));
        }
    }
    if !calls.is_empty() {
        let tool_calls = calls
            .iter()
            .map(|call| tool_call(call, grammar_tool_input_properties))
            .collect::<Result<Vec<_>, _>>()?;
        message.insert("tool_calls".into(), Value::from(tool_calls));
    }
    if let Some(details) = preserved_details {
        message.insert("reasoning_details".into(), Value::from(details));
    }
    if compat.requires_reasoning_content_on_assistant_messages
        && model.reasoning
        && !message.contains_key("reasoning_content")
    {
        message.insert("reasoning_content".into(), Value::from(""));
    }
    let has_content = match message.get("content") {
        Some(Value::String(content)) => !content.is_empty(),
        Some(Value::Array(parts)) => !parts.is_empty(),
        _ => false,
    };
    Ok((has_content || message.contains_key("tool_calls")).then_some(Value::Object(message)))
}

/// The raw reasoning field named by the first non-empty thinking block's
/// signature (llama.cpp and gpt-oss style replay).
fn reasoning_field(model: &Model, thinking: &[&ThinkingContent]) -> Option<&'static str> {
    let signature = thinking.first()?.thinking_signature.as_deref()?;
    let signature = if model.provider.as_str() == "opencode-go" && signature == "reasoning" {
        "reasoning_content"
    } else {
        signature
    };
    REASONING_FIELDS
        .into_iter()
        .find(|field| *field == signature)
}

fn tool_call(
    call: &ToolCall,
    grammar_tool_input_properties: &BTreeMap<String, String>,
) -> Result<Value, String> {
    if let Some(property) = grammar_tool_input_properties.get(&call.name) {
        let input = grammar_tool_input(&call.name, &call.arguments, property)
            .map_err(|error| error.to_string())?;
        return Ok(json!({
            "id": call.id,
            "type": "custom",
            "custom": { "name": call.name, "input": sanitize_surrogates(input) },
        }));
    }
    let arguments = crate::json::to_string(&call.arguments).map_err(|error| error.to_string())?;
    Ok(json!({
        "id": call.id,
        "type": "function",
        "function": { "name": call.name, "arguments": arguments },
    }))
}

/// `reasoning_details` to replay: the first thinking signature holding a
/// detail array, else legacy encrypted details from tool-call signatures.
fn reasoning_details(thinking: &[&ThinkingContent], calls: &[&ToolCall]) -> Option<Vec<Value>> {
    let parse = |signature: Option<&str>| -> Option<Value> {
        signature
            .filter(|signature| !signature.is_empty())
            .and_then(|signature| crate::json::from_str::<Value>(signature).ok())
    };
    let signed =
        thinking
            .iter()
            .find_map(|block| match parse(block.thinking_signature.as_deref())? {
                Value::Array(details)
                    if !details.is_empty() && details.iter().all(is_reasoning_detail) =>
                {
                    Some(details)
                }
                _ => None,
            });
    if signed.is_some() {
        return signed;
    }
    let legacy: Vec<Value> = calls
        .iter()
        .filter_map(|call| parse(call.thought_signature.as_deref()))
        .filter(is_legacy_encrypted_detail)
        .collect();
    (!legacy.is_empty()).then_some(legacy)
}

/// `isOpenAIReasoningDetail`.
fn is_reasoning_detail(detail: &Value) -> bool {
    let Value::Object(detail) = detail else {
        return false;
    };
    let common = detail
        .get("id")
        .is_none_or(|id| id.is_null() || id.is_string())
        && detail.get("format").is_none_or(Value::is_string)
        && detail.get("index").is_none_or(Value::is_number);
    let string = |key: &str| detail.get(key).is_some_and(Value::is_string);
    common
        && match detail.get("type").and_then(Value::as_str) {
            Some("reasoning.summary") => string("summary"),
            Some("reasoning.encrypted") => string("data"),
            Some("reasoning.text") => {
                string("text")
                    && detail
                        .get("signature")
                        .is_none_or(|signature| signature.is_null() || signature.is_string())
            }
            _ => false,
        }
}

/// `parseLegacyEncryptedReasoningDetail`: an encrypted detail with an id
/// and data.
fn is_legacy_encrypted_detail(detail: &Value) -> bool {
    let non_empty = |key: &str| {
        detail
            .get(key)
            .and_then(Value::as_str)
            .is_some_and(|value| !value.is_empty())
    };
    is_reasoning_detail(detail)
        && detail.get("type").and_then(Value::as_str) == Some("reasoning.encrypted")
        && non_empty("id")
        && non_empty("data")
}
