//! The transcript fold: upstream `utils/transcript.ts`.
//!
//! A `Context`'s `systemPrompt`/`tools` become a leading system message via
//! `normalize_context`; every provider-facing helper then replays the
//! system messages' tool deltas (`toolsAdded`/`toolsRemoved`) and section
//! patches.

use serde_json::Map;
use std::collections::BTreeMap;

use crate::text::{get_system_message_text, system_content_text};
use crate::types::context::Context;
use crate::types::message::{Message, SystemMessage};
use crate::types::tool::{Tool, ToolReference};

/// A transcript with its leading system message folded in: only
/// [`normalize_context`] produces this.
#[derive(Clone, Debug, PartialEq)]
pub struct TranscriptContext {
    messages: Vec<Message>,
}

impl TranscriptContext {
    /// The folded transcript.
    #[must_use]
    pub fn messages(&self) -> &[Message] {
        &self.messages
    }

    /// Take the folded transcript.
    #[must_use]
    pub fn into_messages(self) -> Vec<Message> {
        self.messages
    }
}

/// `createInitialSystemMessage`: the leading system message for a prompt
/// and tool set; `None` when both are empty so an empty transcript stays
/// empty.
#[must_use]
pub fn create_initial_system_message(
    system_prompt: Option<&str>,
    tools: Option<&[Tool]>,
) -> Option<SystemMessage> {
    let has_prompt = system_prompt.is_some_and(|p| !p.is_empty());
    let has_tools = tools.is_some_and(|t| !t.is_empty());
    if !has_prompt && !has_tools {
        return None;
    }
    Some(SystemMessage {
        content: crate::types::content::TextOrBlocks::Text(
            system_prompt.unwrap_or_default().to_owned(),
        ),
        sections: None,
        tools_added: has_tools.then(|| tools.unwrap_or_default().to_vec()),
        tools_removed: None,
        timestamp: 0,
    })
}

/// `normalizeContext`: fold `Context.system_prompt`/`tools` into a leading
/// system message. This is the only entry point that produces a
/// [`TranscriptContext`]; every provider-facing function expects the result.
#[must_use]
pub fn normalize_context(context: &Context) -> TranscriptContext {
    let initial =
        create_initial_system_message(context.system_prompt.as_deref(), context.tools.as_deref());
    let mut messages = Vec::with_capacity(context.messages.len() + usize::from(initial.is_some()));
    if let Some(initial) = initial {
        messages.push(Message::System(initial));
    }
    messages.extend(context.messages.iter().cloned());
    TranscriptContext { messages }
}

/// `getInitialSystemMessage`: the leading system message, when the
/// transcript starts with one.
#[must_use]
pub const fn get_initial_system_message(messages: &[Message]) -> Option<&SystemMessage> {
    match messages.first() {
        Some(Message::System(message)) => Some(message),
        _ => None,
    }
}

/// `withoutInitialSystemMessage`: drop the leading system message for APIs
/// that carry the prompt outside the message list.
#[must_use]
pub const fn without_initial_system_message(messages: &[Message]) -> &[Message] {
    let Some((first, rest)) = messages.split_first() else {
        return messages;
    };
    if matches!(first, Message::System(_)) {
        rest
    } else {
        messages
    }
}

/// `getCurrentTools`: the tools available after applying every transcript
/// delta in order.
#[must_use]
pub fn get_current_tools(messages: &[Message]) -> Vec<Tool> {
    let mut tools: BTreeMap<String, Tool> = BTreeMap::new();
    // Insertion order is first-declaration order, like upstream's Map.
    let mut order = Vec::new();
    for message in messages {
        let Message::System(message) = message else {
            continue;
        };
        for tool in message.tools_removed.iter().flatten() {
            if tools.remove(&tool.name).is_some() {
                order.retain(|name: &String| name != &tool.name);
            }
        }
        for tool in message.tools_added.iter().flatten() {
            if tools.insert(tool.name.clone(), tool.clone()).is_none() {
                order.push(tool.name.clone());
            }
        }
    }
    order
        .into_iter()
        .filter_map(|name| tools.get(&name).cloned())
        .collect()
}

/// Patch a section map by one message's `sections` (`null` removes).
fn apply_sections(
    sections: &mut Map<String, serde_json::Value>,
    message_sections: &Map<String, serde_json::Value>,
) {
    for (name, value) in message_sections {
        if value.is_null() {
            sections.remove(name);
        } else {
            sections.insert(name.clone(), value.clone());
        }
    }
}

/// `getCurrentSystemMessage`: replay every system message into one leading
/// message holding the current prompt and tools.
///
/// Later `content` is appended to the base prompt, `sections` are patched
/// by name, and tools are resolved with [`get_current_tools`].
#[must_use]
pub fn get_current_system_message(messages: &[Message]) -> Option<SystemMessage> {
    let mut content = Vec::new();
    let mut sections: Map<String, serde_json::Value> = Map::new();
    let mut timestamp = None;
    for message in messages {
        let Message::System(message) = message else {
            continue;
        };
        if timestamp.is_none() {
            timestamp = Some(message.timestamp);
        }
        let text = system_content_text(&message.content);
        if !text.is_empty() {
            content.push(text);
        }
        if let Some(message_sections) = &message.sections {
            apply_sections(&mut sections, message_sections);
        }
    }
    let tools = get_current_tools(messages);
    if timestamp.is_none() && tools.is_empty() {
        return None;
    }
    Some(SystemMessage {
        content: crate::types::content::TextOrBlocks::Text(content.join("\n\n")),
        sections: if sections.is_empty() {
            None
        } else {
            Some(sections)
        },
        tools_added: if tools.is_empty() { None } else { Some(tools) },
        tools_removed: None,
        timestamp: timestamp.unwrap_or(0),
    })
}

/// `getCurrentSystemPrompt`: the current system prompt text after
/// replaying every system message.
#[must_use]
pub fn get_current_system_prompt(messages: &[Message]) -> String {
    get_current_system_message(messages)
        .map(|message| get_system_message_text(&message))
        .unwrap_or_default()
}

/// `collapseSystemMessages`: the replayed system message leads, and every
/// later system message is dropped — for APIs without mid-conversation
/// system messages.
#[must_use]
pub fn collapse_system_messages(context: &TranscriptContext) -> TranscriptContext {
    let head = get_current_system_message(context.messages());
    let mut messages: Vec<Message> = context
        .messages()
        .iter()
        .filter(|message| !matches!(message, Message::System(_)))
        .cloned()
        .collect();
    if let Some(head) = head {
        messages.insert(0, Message::System(head));
    }
    TranscriptContext { messages }
}

/// `resolveTranscript`: keep later system messages in place when the model
/// accepts them; otherwise collapse them.
#[must_use]
pub fn resolve_transcript(
    context: &TranscriptContext,
    supports_mid_convo_system_messages: Option<bool>,
) -> TranscriptContext {
    if supports_mid_convo_system_messages.unwrap_or(false) {
        TranscriptContext {
            messages: context.messages.clone(),
        }
    } else {
        collapse_system_messages(context)
    }
}

/// `toToolDeclaration`: strip executable and display-only fields before
/// transcript comparison or persistence (upstream round-trips through JSON;
/// our `Tool` is already the declaration shape).
#[must_use]
pub fn to_tool_declaration(tool: &Tool) -> Tool {
    tool.clone()
}

/// `declarationsEqual`: whether two tools declare the same interface to the
/// model. Both sides go through [`to_tool_declaration`] first; comparing the
/// serialized declarations is exact (same key order).
#[must_use]
pub fn declarations_equal(left: &Tool, right: &Tool) -> bool {
    matches!(
        (crate::json::to_string(&to_tool_declaration(left)),
         crate::json::to_string(&to_tool_declaration(right))),
        (Ok(left), Ok(right)) if left == right
    )
}

/// `getToolStateChanges`: the difference of two complete tool states. A
/// changed definition is a removal followed by an addition.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ToolStateChanges {
    /// Declarations newly available (or redeclared with different content).
    pub tools_added: Vec<Tool>,
    /// Tools no longer available.
    pub tools_removed: Vec<ToolReference>,
}

/// `getToolStateChanges` on two complete tool states.
#[must_use]
pub fn get_tool_state_changes(previous: &[Tool], current: &[Tool]) -> ToolStateChanges {
    let previous_tools: BTreeMap<&str, &Tool> = previous
        .iter()
        .map(|tool| (tool.name.as_str(), tool))
        .collect();
    let current_tools: BTreeMap<&str, &Tool> = current
        .iter()
        .map(|tool| (tool.name.as_str(), tool))
        .collect();
    ToolStateChanges {
        tools_added: current
            .iter()
            .filter(|tool| {
                previous_tools
                    .get(tool.name.as_str())
                    .is_none_or(|prior| !declarations_equal(prior, tool))
            })
            .map(to_tool_declaration)
            .collect(),
        tools_removed: previous
            .iter()
            .filter(|tool| {
                current_tools
                    .get(tool.name.as_str())
                    .is_none_or(|current_tool| !declarations_equal(tool, current_tool))
            })
            .map(|tool| ToolReference {
                name: tool.name.clone(),
            })
            .collect(),
    }
}

/// `getDeclaredTools`: every definition referenced by transcript tool
/// state, in first-declaration order.
#[must_use]
pub fn get_declared_tools(messages: &[Message]) -> Vec<Tool> {
    let mut order = Vec::new();
    let mut declared: BTreeMap<String, Tool> = BTreeMap::new();
    for message in messages {
        let Message::System(message) = message else {
            continue;
        };
        for tool in message.tools_added.iter().flatten() {
            if declared.insert(tool.name.clone(), tool.clone()).is_none() {
                order.push(tool.name.clone());
            }
        }
    }
    order
        .into_iter()
        .filter_map(|name| declared.get(&name).cloned())
        .collect()
}

/// `hasNonAdditiveToolChanges`: whether tool history contains a removal or
/// same-name redeclaration that an addition-only transport cannot replay.
#[must_use]
pub fn has_non_additive_tool_changes(messages: &[Message]) -> bool {
    let system: Vec<&SystemMessage> = messages
        .iter()
        .filter_map(|message| match message {
            Message::System(message) => Some(message),
            _ => None,
        })
        .collect();
    last_non_additive_tool_change(&system) >= 0
}

/// Index of the last system message with a removal or same-name
/// redeclaration, or -1. A name declared before counts as a redeclaration
/// even after its removal (Kimi K3 ignores a removed tool that a later tool
/// message declares again).
fn last_non_additive_tool_change(system_messages: &[&SystemMessage]) -> isize {
    let mut declared = std::collections::BTreeSet::new();
    let mut last = -1isize;
    for (index, message) in system_messages.iter().enumerate() {
        if message
            .tools_removed
            .as_ref()
            .is_some_and(|r| !r.is_empty())
        {
            last = isize::try_from(index).unwrap_or(isize::MAX);
        }
        for tool in message.tools_added.iter().flatten() {
            if !declared.insert(tool.name.clone()) {
                last = isize::try_from(index).unwrap_or(isize::MAX);
            }
        }
    }
    last
}

/// `TranscriptToolSupport`: how a transport can express tool changes after
/// the first request.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TranscriptToolSupport {
    /// Only the top-level tool list exists; it always holds current tools.
    None,
    /// Tools can be added at a later message, not removed or redefined.
    Additions,
    /// Every tool is declared in place; a later declaration replaces an
    /// earlier one. Removals are announced as text.
    Changes,
}

/// `TranscriptTools`: split tool declarations between the top-level request
/// field and in-place changes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TranscriptTools {
    /// Tools sent in the top-level request field.
    pub request_tools: Vec<Tool>,
    /// Number of leading system messages whose tool changes `request_tools`
    /// already holds; every later system message carries its own changes.
    pub folded_system_messages: usize,
}

/// `resolveTranscriptTools`: split tool declarations between the top-level
/// request field and in-place changes.
#[must_use]
pub fn resolve_transcript_tools(
    messages: &[Message],
    support: TranscriptToolSupport,
) -> TranscriptTools {
    let system_messages: Vec<&SystemMessage> = messages
        .iter()
        .filter_map(|message| match message {
            Message::System(message) => Some(message),
            _ => None,
        })
        .collect();
    match support {
        TranscriptToolSupport::None => TranscriptTools {
            request_tools: get_current_tools(messages),
            folded_system_messages: system_messages.len(),
        },
        TranscriptToolSupport::Changes => TranscriptTools {
            request_tools: Vec::new(),
            folded_system_messages: 0,
        },
        TranscriptToolSupport::Additions => {
            let initial_fold = usize::from(get_initial_system_message(messages).is_some());
            let last = last_non_additive_tool_change(&system_messages);
            let folded = initial_fold.max(usize::try_from(last + 1).unwrap_or(0));
            let prefix: Vec<Message> = system_messages
                .iter()
                .take(folded)
                .map(|message| Message::System((*message).clone()))
                .collect();
            TranscriptTools {
                request_tools: get_current_tools(&prefix),
                folded_system_messages: folded,
            }
        }
    }
}
