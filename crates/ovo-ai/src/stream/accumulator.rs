//! Folds wire events into a live `AssistantMessage`.
//!
//! The accumulator mirrors how upstream adapters mutate `output`: `*_start`
//! pushes an empty block, deltas append, `*_end` sets the authoritative
//! content, and tool-call argument deltas go through
//! `ToolCallArgumentsStream` with their changes applied in place to the
//! block's `arguments`. Metadata mutations arrive as internal channel items
//! (upstream assigns fields of the live `output` object between events).
//!
//! Note for wire folds: on a provider `replace` restart the accumulator's
//! parser is reset through an internal patch as well, but a fold that only
//! replays `toolcall_delta` events sees the transient pre-`toolcall_end`
//! state, same as upstream's wire clients.

use std::time::{Instant, SystemTime, UNIX_EPOCH};

use serde_json::{Map, Value};

use crate::json::streaming::{JsonChange, ToolCallArgumentsStream};
use crate::stream::event::AssistantMessageEvent;
use crate::stream::sender::Metadata;
use crate::types::content::{
    AssistantContentBlock, StopReason, TextContent, ThinkingContent, ToolCall,
};
use crate::types::message::AssistantMessage;

/// One open tool call's fold state: the incremental parser accumulating
/// its argument text. The call itself lives on the block — the parser is
/// the only tool-call state the accumulator keeps.
#[derive(Debug)]
pub(crate) struct FoldToolCall {
    pub(crate) stream: ToolCallArgumentsStream,
}

impl FoldToolCall {
    /// Seed a fold for `call` including its initial arguments text.
    fn new(call: &ToolCall) -> Self {
        let mut stream = ToolCallArgumentsStream::new();
        if let Ok(initial) = crate::json::to_string(&call.arguments)
            && initial != "{}"
        {
            stream.append(&initial);
        }
        Self { stream }
    }
}

/// parity: AI-STREAM-PROTOCOL
/// Folds `AssistantMessageEvent`s into a live `AssistantMessage`, the way
/// upstream's `AssistantMessageEventStream` folds them into `output`.
#[derive(Debug)]
pub struct AssistantMessageAccumulator {
    message: AssistantMessage,
    tool_calls: std::collections::HashMap<usize, FoldToolCall>,
    created: Instant,
    created_wall: i64,
}

impl AssistantMessageAccumulator {
    /// Start folding onto `base`, the adapter's `output` literal.
    #[must_use]
    pub fn new(base: AssistantMessage) -> Self {
        Self {
            message: base,
            tool_calls: std::collections::HashMap::new(),
            created: Instant::now(),
            created_wall: wall_ms(),
        }
    }

    /// The message folded through the last applied event.
    #[must_use]
    pub const fn message(&self) -> &AssistantMessage {
        &self.message
    }

    /// Consume and return the folded message.
    #[must_use]
    pub fn into_message(self) -> AssistantMessage {
        self.message
    }

    /// Apply one wire event exactly as upstream's fold does.
    pub fn apply(&mut self, event: &AssistantMessageEvent) {
        match event {
            AssistantMessageEvent::Start => {}
            AssistantMessageEvent::TextStart { content_index }
            | AssistantMessageEvent::ThinkingStart { content_index }
            | AssistantMessageEvent::ToolcallStart { content_index } => {
                self.open_block(*content_index, event);
            }
            AssistantMessageEvent::TextDelta {
                content_index,
                delta,
            } => {
                if let Some(AssistantContentBlock::Text(block)) =
                    self.message.content.get_mut(*content_index)
                {
                    block.text.push_str(delta);
                }
            }
            AssistantMessageEvent::ThinkingDelta {
                content_index,
                delta,
            } => {
                if let Some(AssistantContentBlock::Thinking(block)) =
                    self.message.content.get_mut(*content_index)
                {
                    block.thinking.push_str(delta);
                }
            }
            AssistantMessageEvent::TextEnd {
                content_index,
                content,
            } => {
                if let Some(AssistantContentBlock::Text(block)) =
                    self.message.content.get_mut(*content_index)
                {
                    block.text.clone_from(content);
                }
            }
            AssistantMessageEvent::ThinkingEnd {
                content_index,
                content,
            } => {
                if let Some(AssistantContentBlock::Thinking(block)) =
                    self.message.content.get_mut(*content_index)
                {
                    block.thinking.clone_from(content);
                }
            }
            AssistantMessageEvent::ToolcallDelta {
                content_index,
                delta,
            } => {
                self.toolcall_delta(*content_index, delta);
            }
            AssistantMessageEvent::ToolcallEnd {
                content_index,
                tool_call,
            } => {
                self.tool_calls.remove(content_index);
                self.ensure_block(
                    *content_index,
                    AssistantContentBlock::ToolCall(tool_call.clone()),
                );
            }
            AssistantMessageEvent::Done { reason } => {
                self.message.stop_reason = StopReason::from(*reason);
                self.time();
            }
            AssistantMessageEvent::Error {
                reason,
                error_message,
            } => {
                self.message.stop_reason = StopReason::from(*reason);
                self.message.error_message = Some(error_message.clone());
                self.time();
            }
        }
    }

    /// Internal: apply a metadata assignment (upstream writes the live
    /// `output` fields between events).
    pub(crate) fn apply_metadata(&mut self, metadata: Metadata) {
        match metadata {
            Metadata::Usage(usage) => self.message.usage = usage,
            Metadata::ResponseId(id) => self.message.response_id = Some(id),
            Metadata::ResponseModel(model) => self.message.response_model = Some(model),
            Metadata::ProviderThinkingLevel(level) => {
                self.message.provider_thinking_level = Some(level);
            }
            Metadata::RawStopReason(reason) => self.message.raw_stop_reason = Some(reason),
            Metadata::EndTurn(end_turn) => self.message.end_turn = Some(end_turn),
            Metadata::Diagnostic(diagnostic) => {
                self.message
                    .diagnostics
                    .get_or_insert_with(Vec::new)
                    .push(diagnostic);
            }
            Metadata::Deferred(handle) => self.message.deferred = Some(handle),
        }
    }

    /// Internal: set the authoritative block fields (`Patch` channel item).
    /// For tool calls this also reseeds the fold's parser with the block's
    /// serialized arguments (initial arguments / provider `replace`
    /// restart), mirroring upstream's `block.arguments = ...` assignment.
    pub(crate) fn ensure_block(&mut self, index: usize, block: AssistantContentBlock) {
        if let Some(slot) = self.message.content.get_mut(index) {
            *slot = block;
        } else {
            // Content indices are contiguous per protocol; a gap is filled
            // with empty text blocks (never taken on a valid stream).
            while self.message.content.len() < index {
                self.message
                    .content
                    .push(AssistantContentBlock::Text(TextContent {
                        text: String::new(),
                        text_signature: None,
                    }));
            }
            self.message.content.push(block);
        }
        if let Some(AssistantContentBlock::ToolCall(call)) =
            self.message.content.get(index).cloned()
        {
            self.tool_calls.insert(index, FoldToolCall::new(&call));
        }
    }

    /// The mutable message for terminal handling.
    pub(crate) const fn message_mut(&mut self) -> &mut AssistantMessage {
        &mut self.message
    }

    /// Time the finished message (upstream sets `durationMs` at the terminal
    /// event when it is absent and the message timestamp is not in the
    /// future relative to stream creation; `Math.round`, clamped ≥ 0).
    /// parity: AI-STREAM-DURATION
    pub(crate) fn time(&mut self) {
        if self.message.duration_ms.is_none() && self.message.timestamp >= self.created_wall {
            let elapsed = self.created.elapsed().as_millis();
            self.message.duration_ms = Some(u64::try_from(elapsed).unwrap_or(u64::MAX));
        }
    }

    /// Expose `time` for the stream's dropped-sender path.
    pub(crate) fn time_public(&mut self) {
        self.time();
    }

    fn open_block(&mut self, index: usize, event: &AssistantMessageEvent) {
        let block = match event {
            AssistantMessageEvent::TextStart { .. } => AssistantContentBlock::Text(TextContent {
                text: String::new(),
                text_signature: None,
            }),
            AssistantMessageEvent::ThinkingStart { .. } => {
                AssistantContentBlock::Thinking(ThinkingContent {
                    thinking: String::new(),
                    thinking_signature: None,
                    redacted: None,
                })
            }
            AssistantMessageEvent::ToolcallStart { .. } => {
                // id/name/initial arguments arrive through the internal
                // Patch; the placeholder keeps indices contiguous.
                let call = ToolCall {
                    id: String::new(),
                    name: String::new(),
                    arguments: Map::new(),
                    thought_signature: None,
                    namespace: None,
                    arguments_error: None,
                };
                self.tool_calls.insert(index, FoldToolCall::new(&call));
                AssistantContentBlock::ToolCall(call)
            }
            _ => return,
        };
        self.ensure_slot(index, block);
    }

    fn ensure_slot(&mut self, index: usize, block: AssistantContentBlock) {
        while self.message.content.len() < index {
            self.message
                .content
                .push(AssistantContentBlock::Text(TextContent {
                    text: String::new(),
                    text_signature: None,
                }));
        }
        if self.message.content.len() == index {
            self.message.content.push(block);
        }
    }

    fn toolcall_delta(&mut self, index: usize, delta: &str) {
        let Some(open) = self.tool_calls.get_mut(&index) else {
            return;
        };
        let changes = open.stream.append(delta);
        if let Some(AssistantContentBlock::ToolCall(call)) = self.message.content.get_mut(index) {
            apply_argument_changes(&mut call.arguments, &changes);
        }
    }

    /// Internal `ToolcallEnd` item: finish the open fold at `index` and
    /// build the public event's `tool_call` — arguments and
    /// `argumentsError` from `finish()`, the rest from the live block.
    pub(crate) fn toolcall_finish(&mut self, index: usize) -> Option<ToolCall> {
        let finished = self.tool_calls.remove(&index)?.stream.finish();
        let Some(AssistantContentBlock::ToolCall(call)) = self.message.content.get_mut(index)
        else {
            return None;
        };
        call.arguments = finished.arguments;
        call.arguments_error = finished.arguments_error;
        Some(call.clone())
    }

    /// Internal `ToolcallFields` item: set the open call's non-argument
    /// fields on the live block; its parser keeps the text streamed so
    /// far (upstream assigns fields on the live `output` block).
    pub(crate) fn toolcall_fields(&mut self, index: usize, fields: ToolCall) {
        if !self.tool_calls.contains_key(&index) {
            return;
        }
        if let Some(AssistantContentBlock::ToolCall(call)) = self.message.content.get_mut(index) {
            call.id = fields.id;
            call.name = fields.name;
            call.thought_signature = fields.thought_signature;
            call.namespace = fields.namespace;
        }
    }

    /// Internal `ToolcallReplace` item: feed `text` to the open fold —
    /// `Continued` yields the delta for the public `toolcall_delta`
    /// event, `Restarted` silently resets the parser and arguments
    /// (upstream's `replace` handling).
    pub(crate) fn toolcall_replace(&mut self, index: usize, text: &str) -> Option<String> {
        let open = self.tool_calls.get_mut(&index)?;
        match open.stream.replace(text) {
            crate::json::streaming::Replaced::Continued { delta, changes } => {
                if let Some(AssistantContentBlock::ToolCall(call)) =
                    self.message.content.get_mut(index)
                {
                    apply_argument_changes(&mut call.arguments, &changes);
                }
                if delta.is_empty() { None } else { Some(delta) }
            }
            crate::json::streaming::Replaced::Restarted { changes } => {
                if let Some(AssistantContentBlock::ToolCall(call)) =
                    self.message.content.get_mut(index)
                {
                    call.arguments.clear();
                    apply_argument_changes(&mut call.arguments, &changes);
                }
                None
            }
        }
    }

    /// Internal: the tool call's argument text is now `text` (a provider
    /// restart, or an assignment without an event): the fold restarts from
    /// it, mirroring upstream's `block.arguments = ...` assignment.
    pub(crate) fn toolcall_reset(&mut self, index: usize, text: &str) {
        let Some(open) = self.tool_calls.get_mut(&index) else {
            return;
        };
        open.stream = ToolCallArgumentsStream::new();
        let changes = open.stream.append(text);
        if let Some(AssistantContentBlock::ToolCall(call)) = self.message.content.get_mut(index) {
            call.arguments.clear();
            apply_argument_changes(&mut call.arguments, &changes);
        }
    }
}

pub(crate) fn wall_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| i64::try_from(d.as_millis()).unwrap_or(i64::MAX))
}

/// Apply streaming-parser changes in place to a tool call's `arguments`
/// (never panics: paths produced by a consistent stream always resolve;
/// an unresolved path is ignored).
fn apply_argument_changes(arguments: &mut Map<String, Value>, changes: &[JsonChange]) {
    let mut target = Value::Object(std::mem::take(arguments));
    for change in changes {
        change.apply(&mut target);
    }
    if let Value::Object(map) = target {
        *arguments = map;
    }
}

/// Whether a `StopReason` is one of the terminal reasons.
pub(crate) const fn is_terminal(reason: StopReason) -> bool {
    matches!(
        reason,
        StopReason::Stop
            | StopReason::Length
            | StopReason::ToolUse
            | StopReason::Deferred
            | StopReason::Aborted
            | StopReason::Error
    )
}
