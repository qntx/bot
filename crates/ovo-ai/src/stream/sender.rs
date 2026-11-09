//! The producer half of `AssistantStream::channel` and its protocol state.

use tokio::sync::mpsc;

use crate::stream::event::{AssistantMessageEvent, DoneReason, ErrorReason};
use crate::types::content::{AssistantContentBlock, TextContent, ThinkingContent, ToolCall};
use crate::types::deferred::DeferredHandle;
use crate::types::diagnostic::AssistantMessageDiagnostic;
use crate::types::usage::Usage;

#[derive(Debug)]
/// A field mutation of the live message (no wire event).
pub enum Metadata {
    /// Token usage and cost.
    Usage(Usage),
    /// Provider response id.
    ResponseId(String),
    /// Provider response model id.
    ResponseModel(String),
    /// Provider-specific thinking level.
    ProviderThinkingLevel(String),
    /// Raw provider stop reason.
    RawStopReason(String),
    /// Whether the provider ended the turn.
    EndTurn(bool),
    /// Provider diagnostics record.
    Diagnostic(AssistantMessageDiagnostic),
    /// A deferred tool call handle.
    Deferred(DeferredHandle),
}

/// The receiving half is gone.
#[derive(Clone, Copy, Debug, thiserror::Error)]
#[error("stream closed")]
pub struct StreamClosed;

/// Queue item: wire events plus metadata mutations, mirroring upstream's
/// live `output` object (fields set between events are visible on it).
#[derive(Debug)]
pub(crate) enum Item {
    Event(AssistantMessageEvent),
    Meta(Metadata),
    /// Authoritative block fields (signatures, tool-call id/name, initial
    /// arguments) which the wire events do not carry — upstream assigns
    /// them to the live `output` object between events.
    Patch {
        index: usize,
        block: AssistantContentBlock,
    },
    /// Provider `replace` restart or field-less assignment: the consumer's
    /// fold restarts from `text` (upstream assigns `block.arguments`
    /// directly).
    ToolcallReset {
        index: usize,
        text: String,
    },
    /// `toolcall_end`: the consumer's fold finishes the call and emits the
    /// public `toolcall_end` event (arguments parsed exactly once, there).
    ToolcallEnd {
        index: usize,
    },
    /// `toolcall_fields`: set the open call's non-argument fields; the
    /// consumer's parser keeps the text streamed so far.
    ToolcallFields {
        index: usize,
        call: ToolCall,
    },
    /// `toolcall_replace`: the consumer feeds `text` to its fold —
    /// `Continued` emits a `toolcall_delta`, `Restarted` silently resets.
    ToolcallReplace {
        index: usize,
        text: String,
    },
}

/// The producer half of `AssistantStream::channel`.
///
/// Adapter code calls its methods and they emit the wire events. Ordering
/// invariants (`start` once, blocks end once, deltas only for open blocks,
/// nothing after the terminal event) are enforced by `debug_assert!` in
/// debug builds; in release builds a violation terminates the stream as
/// `stopReason: error` with an `errorMessage` describing the violation.
#[derive(Debug)]
pub struct AssistantStreamSender {
    sender: mpsc::Sender<Item>,
    started: bool,
    terminated: bool,
    next_index: usize,
    /// The kind of every started block, by content index.
    kinds: Vec<BlockKind>,
    open: Vec<(usize, BlockKind)>,
    violation: Option<String>,
    /// Debug builds `debug_assert!` on protocol violations; tests disable
    /// the assertion to exercise the release path.
    assert_violations: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BlockKind {
    Text,
    Thinking,
    ToolCall,
}

impl AssistantStreamSender {
    /// The producer half of a new channel.
    pub(crate) const fn new(sender: mpsc::Sender<Item>) -> Self {
        Self {
            sender,
            started: false,
            terminated: false,
            next_index: 0,
            kinds: Vec::new(),
            open: Vec::new(),
            violation: None,
            assert_violations: cfg!(debug_assertions),
        }
    }

    /// The one channel send path: terminated errs, a pending violation is
    /// flushed as the terminal `error` event (the item is dropped), else
    /// the item is sent.
    async fn send(&mut self, item: Item) -> Result<(), StreamClosed> {
        if self.terminated {
            return Err(StreamClosed);
        }
        if let Some(violation) = self.violation.take() {
            self.sender
                .send(Item::Event(AssistantMessageEvent::Error {
                    reason: ErrorReason::Error,
                    error_message: violation,
                }))
                .await
                .ok();
            self.terminated = true;
            return Ok(());
        }
        self.sender.send(item).await.map_err(|_| StreamClosed)
    }

    async fn emit(&mut self, event: AssistantMessageEvent) -> Result<(), StreamClosed> {
        self.send(Item::Event(event)).await
    }

    /// Record an ordering violation: `debug_assert!` in debug builds, and in
    /// release the next emit sends a terminal `error` event instead.
    fn check(&mut self, ok: bool, what: &str) {
        if ok {
            return;
        }
        debug_assert!(!self.assert_violations, "stream protocol violation: {what}");
        self.violation = Some(format!("stream protocol violation: {what}"));
    }

    /// The response has started; call once.
    /// # Errors
    /// Returns `Err(StreamClosed)` when the consumer is gone.
    pub async fn start(&mut self) -> Result<(), StreamClosed> {
        self.check(!self.started, "start() called twice");
        self.started = true;
        self.emit(AssistantMessageEvent::Start).await
    }

    /// Open a text block; returns its content index.
    /// # Errors
    /// Returns `Err(StreamClosed)` when the consumer is gone.
    pub async fn text_start(&mut self) -> Result<usize, StreamClosed> {
        self.block_start(BlockKind::Text).await
    }

    /// Open a thinking block; returns its content index.
    /// # Errors
    /// Returns `Err(StreamClosed)` when the consumer is gone.
    pub async fn thinking_start(&mut self) -> Result<usize, StreamClosed> {
        self.block_start(BlockKind::Thinking).await
    }

    async fn block_start(&mut self, kind: BlockKind) -> Result<usize, StreamClosed> {
        self.check(self.started, "block start before start()");
        let index = self.next_index;
        self.next_index += 1;
        self.kinds.push(kind);
        self.open.push((index, kind));
        let event = match kind {
            BlockKind::Text => AssistantMessageEvent::TextStart {
                content_index: index,
            },
            BlockKind::Thinking => AssistantMessageEvent::ThinkingStart {
                content_index: index,
            },
            BlockKind::ToolCall => AssistantMessageEvent::ToolcallStart {
                content_index: index,
            },
        };
        self.emit(event).await?;
        Ok(index)
    }

    /// Append `delta` to the open text block at `index`.
    /// # Errors
    /// Returns `Err(StreamClosed)` when the consumer is gone.
    pub async fn text_delta(&mut self, index: usize, delta: &str) -> Result<(), StreamClosed> {
        self.check(
            self.is_open(index, BlockKind::Text),
            "text_delta for a non-open text block",
        );
        self.emit(AssistantMessageEvent::TextDelta {
            content_index: index,
            delta: delta.to_owned(),
        })
        .await
    }

    /// Close the text block at `index` with its authoritative block
    /// (carrying `textSignature`); the event carries only the text.
    /// # Errors
    /// Returns `Err(StreamClosed)` when the consumer is gone.
    pub async fn text_end(&mut self, index: usize, block: TextContent) -> Result<(), StreamClosed> {
        self.check(
            self.is_open(index, BlockKind::Text),
            "text_end for a non-open text block",
        );
        self.close(index);
        self.patch_block(index, AssistantContentBlock::Text(block.clone()))
            .await?;
        let event = AssistantMessageEvent::TextEnd {
            content_index: index,
            content: block.text,
        };
        self.emit(event).await
    }

    /// Append `delta` to the open thinking block at `index`.
    /// # Errors
    /// Returns `Err(StreamClosed)` when the consumer is gone.
    pub async fn thinking_delta(&mut self, index: usize, delta: &str) -> Result<(), StreamClosed> {
        self.check(
            self.is_open(index, BlockKind::Thinking),
            "thinking_delta for a non-open thinking block",
        );
        self.emit(AssistantMessageEvent::ThinkingDelta {
            content_index: index,
            delta: delta.to_owned(),
        })
        .await
    }

    /// Close the thinking block at `index` (carrying `thinkingSignature`);
    /// the event carries only the thinking text.
    /// # Errors
    /// Returns `Err(StreamClosed)` when the consumer is gone.
    pub async fn thinking_end(
        &mut self,
        index: usize,
        block: ThinkingContent,
    ) -> Result<(), StreamClosed> {
        self.check(
            self.is_open(index, BlockKind::Thinking),
            "thinking_end for a non-open thinking block",
        );
        self.close(index);
        self.patch_block(index, AssistantContentBlock::Thinking(block.clone()))
            .await?;
        let event = AssistantMessageEvent::ThinkingEnd {
            content_index: index,
            content: block.thinking,
        };
        self.emit(event).await
    }

    /// Open a tool-call block with the call's id, name and any initial
    /// arguments text; returns its content index.
    /// # Errors
    /// Returns `Err(StreamClosed)` when the consumer is gone.
    pub async fn toolcall_start(&mut self, call: ToolCall) -> Result<usize, StreamClosed> {
        let index = self.block_start(BlockKind::ToolCall).await?;
        self.patch_block(index, AssistantContentBlock::ToolCall(call))
            .await?;
        Ok(index)
    }

    /// Set the authoritative fields of the block at `index` without a wire
    /// event, like upstream assigning to a live `output` block: a text or
    /// thinking block may already be closed, a tool call must be open and
    /// its argument fold restarts from the patched arguments.
    /// # Errors
    /// Returns `Err(StreamClosed)` when the consumer is gone.
    pub async fn patch(
        &mut self,
        index: usize,
        block: AssistantContentBlock,
    ) -> Result<(), StreamClosed> {
        let kind = match &block {
            AssistantContentBlock::Text(_) => BlockKind::Text,
            AssistantContentBlock::Thinking(_) => BlockKind::Thinking,
            AssistantContentBlock::ToolCall(_) => BlockKind::ToolCall,
        };
        let valid = if kind == BlockKind::ToolCall {
            self.is_open(index, kind)
        } else {
            self.kinds.get(index) == Some(&kind)
        };
        self.check(
            valid,
            "patch for a block that is not a started block of its kind",
        );
        self.patch_block(index, block).await
    }

    /// Append argument text to the open tool call at `index`.
    /// # Errors
    /// Returns `Err(StreamClosed)` when the consumer is gone.
    pub async fn toolcall_delta(&mut self, index: usize, delta: &str) -> Result<(), StreamClosed> {
        self.check(
            self.is_open(index, BlockKind::ToolCall),
            "toolcall_delta for a non-open tool call",
        );
        self.emit(AssistantMessageEvent::ToolcallDelta {
            content_index: index,
            delta: delta.to_owned(),
        })
        .await
    }

    /// Set the open tool call's id, name, thought signature and namespace
    /// from `fields` (its arguments are ignored), keeping the arguments
    /// streamed so far: upstream assigning those fields on the live block.
    /// # Errors
    /// Returns `Err(StreamClosed)` when the consumer is gone.
    pub async fn toolcall_fields(
        &mut self,
        index: usize,
        fields: ToolCall,
    ) -> Result<(), StreamClosed> {
        self.check(
            self.is_open(index, BlockKind::ToolCall),
            "toolcall_fields for a non-open tool call",
        );
        self.send(Item::ToolcallFields {
            index,
            call: fields,
        })
        .await
    }

    /// Replace the open tool call's argument text with `text` without a
    /// wire event, like upstream assigning `block.arguments` from a parser
    /// it fed directly.
    /// # Errors
    /// Returns `Err(StreamClosed)` when the consumer is gone.
    pub async fn toolcall_arguments(
        &mut self,
        index: usize,
        text: &str,
    ) -> Result<(), StreamClosed> {
        self.check(
            self.is_open(index, BlockKind::ToolCall),
            "toolcall_arguments for a non-open tool call",
        );
        self.send_item(Item::ToolcallReset {
            index,
            text: text.to_owned(),
        })
        .await
    }

    /// Send authoritative block fields (no wire event); the consumer folds
    /// them into `partial()` like upstream's live `output` assignments.
    async fn patch_block(
        &mut self,
        index: usize,
        block: AssistantContentBlock,
    ) -> Result<(), StreamClosed> {
        self.send(Item::Patch { index, block }).await
    }

    /// Provider sent its complete argument text: emits a `toolcall_delta`
    /// with the missing piece, or nothing when the text restarted from the
    /// beginning (mirrors upstream adapters' `replace` handling).
    /// # Errors
    /// Returns `Err(StreamClosed)` when the consumer is gone.
    pub async fn toolcall_replace(&mut self, index: usize, text: &str) -> Result<(), StreamClosed> {
        self.check(
            self.is_open(index, BlockKind::ToolCall),
            "toolcall_replace for a non-open tool call",
        );
        self.send(Item::ToolcallReplace {
            index,
            text: text.to_owned(),
        })
        .await
    }

    /// Disable the debug assertion on ordering violations so tests can
    /// exercise the release-path error termination.
    #[cfg(test)]
    pub(crate) const fn skip_violation_asserts(&mut self) {
        self.assert_violations = false;
    }

    /// Close the tool call at `index`: `finish()` supplies the final
    /// arguments and `argumentsError`.
    /// # Errors
    /// Returns `Err(StreamClosed)` when the consumer is gone.
    pub async fn toolcall_end(&mut self, index: usize) -> Result<(), StreamClosed> {
        self.check(
            self.is_open(index, BlockKind::ToolCall),
            "toolcall_end for a non-open tool call",
        );
        self.close(index);
        self.send_item(Item::ToolcallEnd { index }).await
    }

    /// Mutate a non-event field of the message (usage, response id, …).
    /// Produces no wire event; like upstream's live `output` object it is
    /// visible in `partial()` from the next event on.
    /// # Errors
    /// Returns `Err(StreamClosed)` when the consumer is gone.
    pub async fn metadata(&mut self, metadata: Metadata) -> Result<(), StreamClosed> {
        self.send(Item::Meta(metadata)).await
    }

    /// Forward a channel `Item` unchanged — `AssistantStream::forward`'s
    /// pass-through for lazy `Models` dispatch.
    /// # Errors
    /// Returns `Err(StreamClosed)` when the consumer is gone or the stream
    /// already terminated.
    pub(crate) async fn send_item(&mut self, item: Item) -> Result<(), StreamClosed> {
        self.send(item).await
    }

    /// Terminal: the response finished normally.
    /// # Errors
    /// Returns `Err(StreamClosed)` when the consumer is gone.
    pub async fn done(mut self, reason: DoneReason) -> Result<(), StreamClosed> {
        self.check(!self.terminated, "done() after a terminal event");
        if self.terminated {
            return Err(StreamClosed);
        }
        self.emit(AssistantMessageEvent::Done { reason }).await?;
        self.terminated = true;
        Ok(())
    }

    /// Terminal: the response failed or was aborted.
    /// # Errors
    /// Returns `Err(StreamClosed)` when the consumer is gone.
    pub async fn error(
        mut self,
        reason: ErrorReason,
        error_message: String,
    ) -> Result<(), StreamClosed> {
        self.check(!self.terminated, "error() after a terminal event");
        if self.terminated {
            return Err(StreamClosed);
        }
        self.emit(AssistantMessageEvent::Error {
            reason,
            error_message,
        })
        .await?;
        self.terminated = true;
        Ok(())
    }

    fn is_open(&self, index: usize, kind: BlockKind) -> bool {
        self.open.iter().any(|(i, k)| *i == index && *k == kind)
    }

    fn close(&mut self, index: usize) {
        self.open.retain(|(i, _)| *i != index);
    }
}
