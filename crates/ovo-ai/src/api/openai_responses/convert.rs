//! Message and tool conversion for the Responses API (upstream
//! `openai-responses-shared.ts`: `convertResponsesMessages`,
//! `convertResponsesTools`, `getResponsesToolSupport`).

use std::collections::BTreeMap;

use serde_json::{Map, Value, json};

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
    AssistantContentBlock, TextContent, TextOrBlocksWithImages, ToolCall, UserContentBlock,
};
use crate::types::message::{AssistantMessage, Message, SystemMessage, ToolResultMessage};
use crate::types::model::{Model, ModelInputModality};
use crate::types::tool::Tool;

/// How tools are sent: strict schemas and grammar tools where supported.
#[derive(Clone, Copy, Debug, Default)]
pub struct ToolOptions {
    /// `strict` is sent on function tools.
    pub supports_strict_mode: bool,
    /// Grammar tools are sent as `custom` tools.
    pub supports_openai_grammar_tools: bool,
}

/// `ConvertResponsesMessagesOptions`.
#[derive(Clone, Copy, Debug)]
pub struct ConvertOptions<'a> {
    /// The input property of each tool sent as a grammar tool, by name.
    pub grammar_tool_input_properties: &'a BTreeMap<String, String>,
    /// Later system messages are sent in place, not folded into the prompt.
    pub supports_mid_convo_system_messages: bool,
    /// Tool changes are declared in `additional_tools` items.
    pub supports_additional_tools: bool,
    /// Tool additions are replayed as synthetic tool searches.
    pub supports_tool_search: bool,
    /// How tools are sent.
    pub tool_options: ToolOptions,
    /// The effort of the reply being generated; when set, thinking levels
    /// are sent as `configuration_update` items, so earlier input (and the
    /// prompt cache) never changes with the effort.
    pub configuration_update_effort: Option<&'a str>,
}

/// `getResponsesToolSupport`: in-place tool changes with mid-conversation
/// system messages and `additional_tools`, else additions through tool
/// search, else none.
#[must_use]
pub const fn tool_support(
    supports_mid_convo_system_messages: bool,
    supports_additional_tools: bool,
    supports_tool_search: bool,
) -> TranscriptToolSupport {
    if supports_mid_convo_system_messages && supports_additional_tools {
        TranscriptToolSupport::Changes
    } else if supports_tool_search {
        TranscriptToolSupport::Additions
    } else {
        TranscriptToolSupport::None
    }
}

/// `convertResponsesMessages`: the transcript as Responses input items.
///
/// `allowed_tool_call_providers` are the providers whose `call|item` ids
/// keep their item part.
///
/// # Errors
/// Returns the error when a tool cannot be sent, a grammar tool call has no
/// string input, or a replayed reasoning item is not JSON.
pub fn convert_messages(
    model: &Model,
    context: &TranscriptContext,
    allowed_tool_call_providers: &[&str],
    options: &ConvertOptions<'_>,
) -> Result<Vec<Value>, String> {
    let normalized = resolve_transcript(context, Some(options.supports_mid_convo_system_messages));
    let keeps_item_ids = allowed_tool_call_providers.contains(&model.provider.as_str());
    let normalize = |id: &str, _: &Model, source: &AssistantMessage| {
        normalize_tool_call_id(id, model, source, keeps_item_ids)
    };
    let transformed =
        crate::transform::transform_messages(normalized.messages(), model, Some(&normalize));
    let support = tool_support(
        options.supports_mid_convo_system_messages,
        options.supports_additional_tools,
        options.supports_tool_search,
    );
    let folded = resolve_transcript_tools(normalized.messages(), support).folded_system_messages;
    let developer_role = model
        .compat
        .as_ref()
        .and_then(|compat| compat.get("supportsDeveloperRole"))
        != Some(&Value::Bool(false));
    let mut input = Input {
        items: Vec::new(),
        support,
        options,
        instruction_role: if model.reasoning && developer_role {
            "developer"
        } else {
            "system"
        },
        declared_additional_tools: false,
        unavailable: Vec::new(),
    };

    let mut message_index = 0;
    let mut system_index = 0;
    for (source_index, message) in transformed.iter().enumerate() {
        let leading_system = source_index == 0 && matches!(message, Message::System(_));
        let pushed = match message {
            Message::System(system) => {
                // System messages keep their transcript order through
                // `transformMessages`.
                let anchors_tool_changes = system_index >= folded;
                system_index += 1;
                input.system(system, leading_system, anchors_tool_changes, message_index)?;
                true
            }
            Message::User(user) => input.user(&user.content),
            Message::Assistant(assistant) => input.assistant(assistant, model, message_index)?,
            Message::ToolResult(result) => {
                input.tool_result(result, model);
                true
            }
        };
        // Skipped messages (empty user content, empty replies) take no index.
        if pushed && !leading_system {
            message_index += 1;
        }
    }
    if let Some(effort) = options.configuration_update_effort {
        input.items.push(configuration_update(effort));
    }
    Ok(input.items)
}

/// The input items under construction and the tool-change state.
struct Input<'a> {
    items: Vec<Value>,
    support: TranscriptToolSupport,
    options: &'a ConvertOptions<'a>,
    instruction_role: &'static str,
    declared_additional_tools: bool,
    /// Tools announced as removed and not declared again since, in the
    /// order they were removed.
    unavailable: Vec<String>,
}

impl Input<'_> {
    fn system(
        &mut self,
        system: &SystemMessage,
        leading: bool,
        anchors_tool_changes: bool,
        message_index: usize,
    ) -> Result<(), String> {
        if anchors_tool_changes {
            self.tool_additions(system, &format!("system:{message_index}"))?;
        }
        let mut parts = vec![if leading {
            get_system_message_text(system)
        } else {
            render_system_message_update(system)
        }];
        if anchors_tool_changes {
            // A redefinition is a removal plus an addition of the same name;
            // only true removals are announced.
            let added = system.tools_added.as_deref().unwrap_or_default();
            let removed: Vec<String> = system
                .tools_removed
                .iter()
                .flatten()
                .map(|tool| tool.name.clone())
                .filter(|name| !added.iter().any(|tool| &tool.name == name))
                .collect();
            if !removed.is_empty() {
                parts.push(render_tool_removal_notice(&removed));
            }
            // After this message's additions, whose notice precedes this
            // removal notice.
            self.mark_unavailable(removed);
        }
        let text = parts
            .into_iter()
            .filter(|part| !part.is_empty())
            .collect::<Vec<_>>()
            .join("\n\n");
        if !text.is_empty() {
            self.items.push(
                json!({ "role": self.instruction_role, "content": sanitize_surrogates(&text) }),
            );
        }
        Ok(())
    }

    /// Record removed tools; like a `Set`, a name already present keeps
    /// its position.
    fn mark_unavailable(&mut self, names: Vec<String>) {
        for name in names {
            if !self.unavailable.contains(&name) {
                self.unavailable.push(name);
            }
        }
    }

    /// `appendSystemToolAdditions`: a system message's added tools as an
    /// `additional_tools` item or a synthetic client tool search.
    fn tool_additions(&mut self, system: &SystemMessage, seed: &str) -> Result<(), String> {
        let tools = system.tools_added.as_deref().unwrap_or_default();
        self.unavailable
            .retain(|name| !tools.iter().any(|tool| &tool.name == name));
        if tools.is_empty() {
            return Ok(());
        }
        let tool_options = self.options.tool_options;
        match self.support {
            TranscriptToolSupport::Changes => {
                if self.declared_additional_tools {
                    self.items.push(json!({
                        "role": self.instruction_role,
                        "content": render_incremental_tools_notice(&self.unavailable),
                    }));
                }
                self.declared_additional_tools = true;
                self.items.push(json!({
                    "type": "additional_tools",
                    "role": "developer",
                    "tools": convert_tools(tools, tool_options, false)?,
                }));
            }
            TranscriptToolSupport::Additions => {
                let names: Vec<&str> = tools.iter().map(|tool| tool.name.as_str()).collect();
                let call_id = format!(
                    "pi_tool_load_{}",
                    short_hash(&format!("{seed}:{}", names.join(",")))
                );
                self.items.push(json!({
                    "type": "tool_search_call",
                    "call_id": call_id,
                    "execution": "client",
                    "status": "completed",
                    "arguments": { "query": names.join(" "), "limit": names.len() },
                }));
                self.items.push(json!({
                    "type": "tool_search_output",
                    "call_id": call_id,
                    "execution": "client",
                    "status": "completed",
                    "tools": convert_tools(tools, tool_options, true)?,
                }));
            }
            TranscriptToolSupport::None => {}
        }
        Ok(())
    }

    /// A user message; `false` when it has no content blocks.
    fn user(&mut self, content: &TextOrBlocksWithImages) -> bool {
        let parts: Vec<Value> = match content {
            TextOrBlocksWithImages::Text(text) => {
                vec![json!({ "type": "input_text", "text": sanitize_surrogates(text) })]
            }
            TextOrBlocksWithImages::Blocks(blocks) => blocks
                .iter()
                .map(|block| match block {
                    UserContentBlock::Text(text) => {
                        json!({ "type": "input_text", "text": sanitize_surrogates(&text.text) })
                    }
                    UserContentBlock::Image(image) => input_image(&image.mime_type, &image.data),
                })
                .collect(),
        };
        if parts.is_empty() {
            return false;
        }
        self.items.push(json!({ "role": "user", "content": parts }));
        true
    }

    /// An assistant reply as output items; `false` when nothing replays.
    fn assistant(
        &mut self,
        assistant: &AssistantMessage,
        model: &Model,
        message_index: usize,
    ) -> Result<bool, String> {
        let same_provider_and_api =
            assistant.provider == model.provider && assistant.api == model.api;
        let same_model = same_provider_and_api && assistant.model == model.id;
        let different_model = same_provider_and_api && assistant.model != model.id;
        let mut output: Vec<Value> = Vec::new();
        let mut text_index = 0;
        for block in &assistant.content {
            match block {
                // The signature holds the reasoning item to replay.
                AssistantContentBlock::Thinking(thinking) => output.extend(
                    thinking
                        .thinking_signature
                        .as_deref()
                        .filter(|signature| !signature.is_empty())
                        .map(crate::json::from_str::<Value>)
                        .transpose()
                        .map_err(|error| error.to_string())?,
                ),
                AssistantContentBlock::Text(text) => {
                    output.push(output_message(text, message_index, text_index));
                    text_index += 1;
                }
                AssistantContentBlock::ToolCall(call) => {
                    output.push(self.tool_call(call, same_model, different_model)?);
                }
            }
        }
        if output.is_empty() {
            return Ok(false);
        }
        if let Some(level) = assistant
            .provider_thinking_level
            .as_deref()
            .filter(|_| same_model && self.options.configuration_update_effort.is_some())
        {
            self.items.push(configuration_update(level));
        }
        self.items.extend(output);
        Ok(true)
    }

    /// A replayed tool call as a `function_call` or, for grammar tools, a
    /// `custom_tool_call` item.
    fn tool_call(
        &self,
        call: &ToolCall,
        same_model: bool,
        different_model: bool,
    ) -> Result<Value, String> {
        let mut parts = call.id.split('|');
        let call_id = parts.next().unwrap_or_default();
        let custom_property = self.options.grammar_tool_input_properties.get(&call.name);
        // Item ids pair with reasoning items of the same model, and must match
        // the item type: `fc_` for function calls, `ctc_` for custom calls.
        let prefix = if custom_property.is_some() {
            "ctc_"
        } else {
            "fc_"
        };
        let item_id = parts
            .next()
            .filter(|item| !different_model && item.starts_with(prefix));
        let mut item = Map::new();
        if let Some(property) = custom_property {
            let input = grammar_tool_input(&call.name, &call.arguments, property)
                .map_err(|error| error.to_string())?;
            item.insert("type".into(), Value::from("custom_tool_call"));
            if let Some(item_id) = item_id {
                item.insert("id".into(), Value::from(item_id));
            }
            item.insert("call_id".into(), Value::from(call_id));
            item.insert("name".into(), Value::from(call.name.clone()));
            item.insert("input".into(), Value::from(sanitize_surrogates(input)));
        } else {
            let arguments =
                crate::json::to_string(&call.arguments).map_err(|error| error.to_string())?;
            item.insert("type".into(), Value::from("function_call"));
            if let Some(item_id) = item_id {
                item.insert("id".into(), Value::from(item_id));
            }
            item.insert("call_id".into(), Value::from(call_id));
            item.insert("name".into(), Value::from(call.name.clone()));
            item.insert("arguments".into(), Value::from(arguments));
        }
        if let Some(namespace) = call.namespace.as_ref().filter(|_| same_model) {
            item.insert("namespace".into(), Value::from(namespace.clone()));
        }
        Ok(Value::Object(item))
    }

    fn tool_result(&mut self, result: &ToolResultMessage, model: &Model) {
        let call_id = result.tool_call_id.split('|').next().unwrap_or_default();
        let kind = if self
            .options
            .grammar_tool_input_properties
            .contains_key(&result.tool_name)
        {
            "custom_tool_call_output"
        } else {
            "function_call_output"
        };
        self.items.push(json!({
            "type": kind,
            "call_id": call_id,
            "output": tool_result_output(result, model),
        }));
    }
}

/// `normalizeToolCallId`: ids reduced to allowed characters; for providers
/// that keep item ids, a `call|item` id keeps an `fc_` item part, hashed
/// when the call came from another provider or API.
fn normalize_tool_call_id(
    id: &str,
    model: &Model,
    source: &AssistantMessage,
    keeps_item_ids: bool,
) -> String {
    let Some((call_id, rest)) = id.split_once('|').filter(|_| keeps_item_ids) else {
        return normalize_id_part(id);
    };
    let item_id = rest.split('|').next().unwrap_or_default();
    let foreign = source.provider != model.provider || source.api != model.api;
    let mut item_id = if foreign {
        let hashed = format!("fc_{}", short_hash(item_id));
        js_slice(&hashed, 0, 64).to_owned()
    } else {
        normalize_id_part(item_id)
    };
    // The Responses API requires function call item ids to start with `fc`.
    if !item_id.starts_with("fc_") {
        item_id = normalize_id_part(&format!("fc_{item_id}"));
    }
    format!("{}|{item_id}", normalize_id_part(call_id))
}

/// `normalizeIdPart`: allowed characters, at most 64, no trailing `_`.
fn normalize_id_part(part: &str) -> String {
    let sanitized = replace_non_id_chars(part);
    // Sanitized ids are ASCII, so bytes are UTF-16 units.
    let truncated = sanitized.get(..64).unwrap_or(&sanitized);
    truncated.trim_end_matches('_').to_owned()
}

/// A replayed text block as an output `message` item; its id and phase come
/// from the text signature, else a stable fallback id.
fn output_message(text: &TextContent, message_index: usize, text_index: usize) -> Value {
    let signature = parse_text_signature(text.text_signature.as_deref());
    let id = match signature.as_ref().map(|signature| signature.id.as_str()) {
        None | Some("") if text_index == 0 => format!("msg_pi_{message_index}"),
        None | Some("") => format!("msg_pi_{message_index}_{text_index}"),
        // OpenAI caps item ids at 64 characters.
        Some(id) if js_len(id) > 64 => format!("msg_{}", short_hash(id)),
        Some(id) => id.to_owned(),
    };
    let mut item = Map::new();
    item.insert("type".into(), Value::from("message"));
    item.insert("role".into(), Value::from("assistant"));
    item.insert(
        "content".into(),
        json!([{ "type": "output_text", "text": sanitize_surrogates(&text.text), "annotations": [] }]),
    );
    item.insert("status".into(), Value::from("completed"));
    item.insert("id".into(), Value::from(id));
    if let Some(phase) = signature.and_then(|signature| signature.phase) {
        item.insert("phase".into(), Value::from(phase));
    }
    Value::Object(item)
}

/// A parsed text signature.
struct TextSignature {
    id: String,
    phase: Option<&'static str>,
}

/// `parseTextSignature`: a `TextSignatureV1` JSON object, else the legacy
/// plain id.
fn parse_text_signature(signature: Option<&str>) -> Option<TextSignature> {
    let signature = signature.filter(|signature| !signature.is_empty())?;
    if signature.starts_with('{')
        && let Ok(parsed) = crate::json::from_str::<Value>(signature)
        && parsed.get("v").and_then(Value::as_f64) == Some(1.0)
        && let Some(id) = parsed.get("id").and_then(Value::as_str)
    {
        let phase = match parsed.get("phase").and_then(Value::as_str) {
            Some("commentary") => Some("commentary"),
            Some("final_answer") => Some("final_answer"),
            _ => None,
        };
        return Some(TextSignature {
            id: id.to_owned(),
            phase,
        });
    }
    Some(TextSignature {
        id: signature.to_owned(),
        phase: None,
    })
}

/// `convertToolResultOutput`: the text, or text and images when the model
/// accepts images.
fn tool_result_output(result: &ToolResultMessage, model: &Model) -> Value {
    let text = result
        .content
        .iter()
        .filter_map(|block| match block {
            UserContentBlock::Text(text) => Some(text.text.as_str()),
            UserContentBlock::Image(_) => None,
        })
        .collect::<Vec<_>>()
        .join("\n");
    let images: Vec<Value> = result
        .content
        .iter()
        .filter_map(|block| match block {
            UserContentBlock::Image(image) => Some(input_image(&image.mime_type, &image.data)),
            UserContentBlock::Text(_) => None,
        })
        .collect();
    let has_text = !js_trim(&text).is_empty();
    if images.is_empty() || !model.input.contains(&ModelInputModality::Image) {
        let output = if has_text {
            text.as_str()
        } else if images.is_empty() {
            "(no tool output)"
        } else {
            "(see attached image)"
        };
        return Value::from(sanitize_surrogates(output));
    }
    let mut output = Vec::with_capacity(images.len() + 1);
    if has_text {
        output.push(json!({ "type": "input_text", "text": sanitize_surrogates(&text) }));
    }
    output.extend(images);
    Value::Array(output)
}

fn input_image(mime_type: &str, data: &str) -> Value {
    json!({ "type": "input_image", "detail": "auto", "image_url": format!("data:{mime_type};base64,{data}") })
}

fn configuration_update(effort: &str) -> Value {
    json!({ "type": "configuration_update", "reasoning": { "effort": effort } })
}

/// `renderIncrementalToolsNotice`: precedes every `additional_tools` item
/// after the first, so models do not read the latest item as the complete
/// tool set, and restates the tools that stay unavailable.
fn render_incremental_tools_notice(unavailable: &[String]) -> String {
    let notice = "This is an incremental tools update. Previously declared tools remain available unless explicitly marked unavailable. A tool declared again here is available again, and its latest definition replaces the earlier one.";
    if unavailable.is_empty() {
        notice.to_owned()
    } else {
        format!("{notice}\nStill unavailable: {}.", unavailable.join(", "))
    }
}

fn render_tool_removal_notice(names: &[String]) -> String {
    let lines: Vec<String> = names.iter().map(|name| format!("- {name}")).collect();
    format!(
        "The following tools are no longer available. Do not call them:\n{}",
        lines.join("\n")
    )
}

/// `convertResponsesTools`: grammar tools as `custom` tools, the rest as
/// function tools; tools loaded by a tool search are deferred.
///
/// # Errors
/// Returns upstream's message for a grammar or strict tool that cannot be
/// sent.
pub fn convert_tools(
    tools: &[Tool],
    options: ToolOptions,
    tool_search_result: bool,
) -> Result<Vec<Value>, String> {
    tools
        .iter()
        .map(|tool| {
            let mut converted = Map::new();
            if let Some(grammar) =
                resolve_grammar_constrained_sampling(tool, options.supports_openai_grammar_tools)
                    .map_err(|error| error.to_string())?
            {
                let syntax = match grammar.format {
                    GrammarFormat::Lark => "lark",
                    GrammarFormat::Regex => "regex",
                };
                converted.insert("type".into(), Value::from("custom"));
                converted.insert("name".into(), Value::from(tool.name.clone()));
                converted.insert("description".into(), Value::from(tool.description.clone()));
                converted.insert(
                    "format".into(),
                    json!({ "type": "grammar", "syntax": syntax, "definition": grammar.definition }),
                );
                if tool_search_result {
                    converted.insert("defer_loading".into(), Value::Bool(true));
                }
                return Ok(Value::Object(converted));
            }
            let strict =
                resolve_json_schema_strict_sampling(tool, options.supports_strict_mode, None)
                    .map_err(|error| error.to_string())?;
            let parameters =
                json_schema_tool_parameters(tool, strict).map_err(|error| error.to_string())?;
            converted.insert("type".into(), Value::from("function"));
            converted.insert("name".into(), Value::from(tool.name.clone()));
            converted.insert("description".into(), Value::from(tool.description.clone()));
            converted.insert("parameters".into(), Value::Object(parameters));
            if tool_search_result {
                converted.insert("defer_loading".into(), Value::Bool(true));
            }
            if options.supports_strict_mode {
                converted.insert("strict".into(), Value::from(strict.unwrap_or(false)));
            }
            Ok(Value::Object(converted))
        })
        .collect()
}
