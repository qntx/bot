//! Events and folding of one assistant response.
//!
//! Port of `AssistantMessageEventStream` / `EventStream` and the way the
//! provider adapters mutate `output`: the sender emits wire events, the
//! consumer folds them into a live `AssistantMessage` (`partial()`), and
//! `result()` yields the message after the terminal `done`/`error`.
//!
//! Unlike upstream this stream owns no runtime: the producer task and its
//! abort-on-drop arrive with the HTTP layer (M1-08); `channel` only wires
//! a bounded `tokio::sync::mpsc` queue.

pub mod accumulator;
pub mod event;
pub mod sender;

use std::pin::Pin;
use std::task::{Context, Poll};

use tokio::sync::mpsc;

pub use accumulator::AssistantMessageAccumulator;
pub use event::{AssistantMessageEvent, DoneReason, ErrorReason};
pub use sender::{AssistantStreamSender, Metadata, StreamClosed};

use crate::types::content::StopReason;
use std::future::Future;

use accumulator::is_terminal;
use sender::Item;

use crate::types::message::AssistantMessage;

/// The consuming stream of one assistant response (`partial()` + `result()`).
pub struct AssistantStream {
    receiver: mpsc::Receiver<Item>,
    acc: AssistantMessageAccumulator,
    done: bool,
    producer: Option<tokio::task::JoinHandle<()>>,
}

impl AssistantStream {
    /// The channel producer and consumer; `base` is the adapter's
    /// `AssistantMessage` literal and `capacity` bounds the event queue.
    /// The consumer owns the accumulator from creation, so a producer that
    /// fails before `start` still yields `base` as the final message.
    #[must_use]
    pub fn channel(base: AssistantMessage, capacity: usize) -> (AssistantStreamSender, Self) {
        let (sender, receiver) = mpsc::channel(capacity.max(1));
        (
            AssistantStreamSender::new(sender),
            Self {
                receiver,
                acc: AssistantMessageAccumulator::new(base),
                done: false,
                producer: None,
            },
        )
    }

    /// `spawn`: channel + a producer task on the current runtime.
    /// Dropping the stream aborts the producer; the producer is not
    /// awaited (panics are outside the contract).
    #[must_use]
    pub fn spawn<Fut>(
        base: AssistantMessage,
        capacity: usize,
        producer: impl FnOnce(AssistantStreamSender) -> Fut + Send + 'static,
    ) -> Self
    where
        Fut: Future<Output = ()> + Send + 'static,
    {
        let (sender, mut stream) = Self::channel(base, capacity);
        stream.producer = Some(tokio::spawn(producer(sender)));
        stream
    }

    /// The message folded through the last yielded event.
    #[must_use]
    pub const fn partial(&self) -> &AssistantMessage {
        self.acc.message()
    }

    /// The final message: drains the remaining events. When the producer
    /// dropped without a terminal event (an adapter bug path), the message
    /// ends `stopReason: error` with an ovo-specific `errorMessage`.
    pub async fn result(mut self) -> AssistantMessage {
        while let Some(item) = self.receiver.recv().await {
            match item {
                Item::Meta(metadata) => self.acc.apply_metadata(metadata),
                Item::Patch { index, block } => self.acc.ensure_block(index, block),
                Item::ToolcallReset { index, text } => self.acc.toolcall_reset(index, &text),
                Item::Event(event) => self.acc.apply(&event),
            }
        }
        if !self.done {
            let message = self.acc.message_mut();
            if !is_terminal(message.stop_reason) {
                message.stop_reason = StopReason::Error;
                message.error_message = Some(String::from(
                    "producer dropped without a done or error event",
                ));
                self.acc.time_public();
            }
        }
        self.acc.into_message()
    }
}

impl std::fmt::Debug for AssistantStream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AssistantStream").finish_non_exhaustive()
    }
}

impl futures::Stream for AssistantStream {
    type Item = AssistantMessageEvent;

    fn poll_next(mut self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if self.done {
            return Poll::Ready(None);
        }
        loop {
            match self.receiver.poll_recv(context) {
                Poll::Ready(Some(Item::Meta(metadata))) => {
                    self.acc.apply_metadata(metadata);
                }
                Poll::Ready(Some(Item::Patch { index, block })) => {
                    self.acc.ensure_block(index, block);
                }
                Poll::Ready(Some(Item::ToolcallReset { index, text })) => {
                    self.acc.toolcall_reset(index, &text);
                }
                Poll::Ready(Some(Item::Event(event))) => {
                    let terminal = matches!(
                        event,
                        AssistantMessageEvent::Done { .. } | AssistantMessageEvent::Error { .. }
                    );
                    self.done = self.done || terminal;
                    self.acc.apply(&event);
                    return Poll::Ready(Some(event));
                }
                Poll::Ready(None) => return Poll::Ready(None),
                Poll::Pending => return Poll::Pending,
            }
        }
    }
}
#[cfg(test)]
#[expect(
    clippy::panic,
    clippy::indexing_slicing,
    reason = "test code asserts freely"
)]
mod tests {
    use futures::StreamExt;
    use serde_json::json;

    use crate::stream::accumulator::wall_ms;
    use crate::stream::*;
    use crate::types::api::Api;
    use crate::types::content::{AssistantContentBlock, StopReason, TextContent, ToolCall};
    use crate::types::message::AssistantMessage;
    use crate::types::usage::{Usage, UsageCost};

    fn base() -> AssistantMessage {
        AssistantMessage {
            content: Vec::new(),
            api: Api::ANTHROPIC_MESSAGES,
            provider: crate::types::api::ProviderId::ANTHROPIC,
            model: String::from("m"),
            usage: Usage {
                input: 0,
                output: 0,
                cache_read: 0,
                cache_write: 0,
                total_tokens: 0,
                cost: UsageCost {
                    input: 0.0,
                    output: 0.0,
                    cache_read: 0.0,
                    cache_write: 0.0,
                    total: 0.0,
                    total_from_provider: None,
                },
                cache_write1h: None,
                reasoning: None,
            },
            stop_reason: StopReason::Pending,
            timestamp: wall_ms(),
            provider_thinking_level: None,
            response_id: None,
            response_model: None,
            raw_stop_reason: None,
            error_message: None,
            diagnostics: None,
            deferred: None,
            end_turn: None,
            duration_ms: None,
            thinking_level: None,
        }
    }

    /// Upstream "emits text events and resolves with the final message".
    #[test]
    fn text_flow() {
        futures::executor::block_on(async {
            let (mut tx, mut stream) = AssistantStream::channel(base(), 16);
            tx.start().await.map_err(|e| e.to_string()).unwrap();
            let index = tx.text_start().await.unwrap_or(0);
            assert!(tx.text_delta(index, "he").await.is_ok());
            assert!(tx.text_delta(index, "llo").await.is_ok());
            assert!(
                tx.text_end(
                    index,
                    TextContent {
                        text: String::from("hello"),
                        text_signature: None
                    }
                )
                .await
                .is_ok()
            );
            assert!(tx.done(DoneReason::Stop).await.is_ok());
            let mut count = 0;
            while stream.next().await.is_some() {
                count += 1;
            }
            assert_eq!(count, 6);
            let message = stream.result().await;
            assert_eq!(message.content.len(), 1);
            let AssistantContentBlock::Text(block) = &message.content[0] else {
                panic!("expected text block");
            };
            assert_eq!(block.text, "hello");
            assert_eq!(message.stop_reason, StopReason::Stop);
            assert!(message.duration_ms.is_some());
        });
    }

    /// Upstream "tool call deltas update partial arguments".
    #[test]
    fn toolcall_flow() {
        futures::executor::block_on(async {
            let (mut tx, mut stream) = AssistantStream::channel(base(), 16);
            let call = ToolCall {
                id: String::from("call-1"),
                name: String::from("write"),
                arguments: serde_json::Map::new(),
                thought_signature: None,
                namespace: None,
                arguments_error: None,
            };
            tx.start().await.ok();
            let index = tx.toolcall_start(call).await.unwrap_or(0);
            tx.toolcall_delta(index, "{\"path\":\"a").await.ok();
            tx.toolcall_replace(index, "{\"path\":\"ab\",\"x\":1}")
                .await
                .ok();
            tx.toolcall_end(index).await.ok();
            tx.done(DoneReason::ToolUse).await.ok();
            while stream.next().await.is_some() {}
            let message = stream.result().await;
            let AssistantContentBlock::ToolCall(block) = &message.content[0] else {
                panic!("expected tool call");
            };
            assert_eq!(block.id, "call-1");
            assert_eq!(block.arguments["path"], json!("ab"));
            assert_eq!(block.arguments["x"], json!(1));
            assert!(block.arguments_error.is_none());
        });
    }

    /// Upstream "error before start yields the base message as error".
    #[test]
    fn error_before_start() {
        futures::executor::block_on(async {
            let (tx, stream) = AssistantStream::channel(base(), 4);
            drop(tx);
            let message = stream.result().await;
            assert_eq!(message.stop_reason, StopReason::Error);
            assert!(message.error_message.is_some());
        });
    }

    /// Upstream "error event ends the stream with the error message".
    #[test]
    fn error_terminal() {
        futures::executor::block_on(async {
            let (mut tx, stream) = AssistantStream::channel(base(), 4);
            tx.start().await.ok();
            tx.error(ErrorReason::Error, String::from("boom"))
                .await
                .ok();
            let message = stream.result().await;
            assert_eq!(message.stop_reason, StopReason::Error);
            assert_eq!(message.error_message.as_deref(), Some("boom"));
        });
    }

    /// Upstream: a message predating the stream stays untimed.
    #[test]
    fn predated_timestamp_stays_untimed() {
        futures::executor::block_on(async {
            let mut aged = base();
            aged.timestamp = wall_ms() - 60_000;
            let (mut tx, stream) = AssistantStream::channel(aged, 4);
            tx.start().await.ok();
            tx.done(DoneReason::Stop).await.ok();
            let message = stream.result().await;
            assert!(message.duration_ms.is_none());
        });
    }
}

#[cfg(test)]
#[expect(
    clippy::panic,
    clippy::excessive_nesting,
    reason = "test code asserts freely"
)]
mod prop_tests {
    use futures::StreamExt;
    use proptest::prelude::*;
    use proptest::proptest;
    use serde_json::json;

    use crate::stream::*;
    use crate::types::content::{TextContent, ThinkingContent, ToolCall};
    use crate::types::message::AssistantMessage;

    fn base() -> AssistantMessage {
        serde_json::from_value(json!({
            "role": "assistant",
            "content": [],
            "api": "anthropic-messages",
            "provider": "anthropic",
            "model": "m",
            "usage": {
                "input": 0,
                "output": 0,
                "cacheRead": 0,
                "cacheWrite": 0,
                "totalTokens": 0,
                "cost": { "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "total": 0 },
            },
            "stopReason": "pending",
            "timestamp": 0,
        }))
        .unwrap_or_else(|e| panic!("base: {e}"))
    }

    #[derive(Clone, Debug)]
    enum Op {
        Text { deltas: Vec<String> },
        Thinking { deltas: Vec<String> },
        ToolCall { deltas: Vec<String> },
    }

    #[expect(clippy::cognitive_complexity, reason = "op dispatch is a flat match")]
    async fn drive(ops: &[Op]) -> (Vec<AssistantMessageEvent>, AssistantMessage) {
        let (mut tx, mut stream) = AssistantStream::channel(base(), 64);
        tx.start().await.ok();
        for op in ops {
            match op {
                Op::Text { deltas } => {
                    let i = tx.text_start().await.unwrap_or(0);
                    for delta in deltas {
                        tx.text_delta(i, delta).await.ok();
                    }
                    tx.text_end(
                        i,
                        TextContent {
                            text: deltas.concat(),
                            text_signature: None,
                        },
                    )
                    .await
                    .ok();
                }
                Op::Thinking { deltas } => {
                    let i = tx.thinking_start().await.unwrap_or(0);
                    for delta in deltas {
                        tx.thinking_delta(i, delta).await.ok();
                    }
                    tx.thinking_end(
                        i,
                        ThinkingContent {
                            thinking: deltas.concat(),
                            thinking_signature: None,
                            redacted: None,
                        },
                    )
                    .await
                    .ok();
                }
                Op::ToolCall { deltas } => {
                    let call: ToolCall = serde_json::from_value(json!({
                        "id": "c",
                        "name": "t",
                        "arguments": {},
                    }))
                    .unwrap_or_else(|e| panic!("call: {e}"));
                    let i = tx.toolcall_start(call).await.unwrap_or(0);
                    for delta in deltas {
                        tx.toolcall_delta(i, delta).await.ok();
                    }
                    tx.toolcall_end(i).await.ok();
                }
            }
        }
        tx.done(DoneReason::Stop).await.ok();
        let mut events = Vec::new();
        // Property (c): `partial()` after each yielded event equals a fresh
        // external fold of the events so far. Content-wise equality only:
        // metadata (usage/response id/…) may lead the public events since it
        // travels as internal items like upstream's live `output` mutations.
        let mut fold = AssistantMessageAccumulator::new(base());
        while let Some(event) = stream.next().await {
            fold.apply(&event);
            let actual = serde_json::to_value(stream.partial()).unwrap_or_default();
            let expected = serde_json::to_value(fold.message()).unwrap_or_default();
            for key in ["stopReason", "errorMessage"] {
                assert_eq!(
                    actual.get(key),
                    expected.get(key),
                    "partial() diverged at {key}"
                );
            }
            // Content: block fields set through internal patches (id/name,
            // signatures, initial args) lead the event-fold, like upstream's
            // live `output`; compare only the event-derived parts.
            let event_derived = |v: &serde_json::Value| -> serde_json::Value {
                serde_json::Value::Array(
                    v["content"]
                        .as_array()
                        .map(|blocks| {
                            blocks
                                .iter()
                                .map(|block| {
                                    let mut masked = serde_json::Map::new();
                                    masked.insert("type".into(), block["type"].clone());
                                    for key in ["text", "thinking", "arguments"] {
                                        if let Some(value) = block.get(key) {
                                            masked.insert(key.into(), value.clone());
                                        }
                                    }
                                    serde_json::Value::Object(masked)
                                })
                                .collect()
                        })
                        .unwrap_or_default(),
                )
            };
            assert_eq!(
                event_derived(&actual),
                event_derived(&expected),
                "partial() content diverged"
            );
            events.push(event);
        }
        (events, stream.result().await)
    }

    fn delta_strategy() -> impl Strategy<Value = String> {
        prop::collection::vec(any::<char>(), 1..6)
            .prop_map(|chars| chars.into_iter().collect::<String>())
    }

    fn op_strategy() -> impl Strategy<Value = Op> {
        prop_oneof![
            prop::collection::vec(delta_strategy(), 0..4).prop_map(|deltas| Op::Text { deltas }),
            prop::collection::vec(delta_strategy(), 0..4)
                .prop_map(|deltas| Op::Thinking { deltas }),
            prop::collection::vec(delta_strategy(), 0..4)
                .prop_map(|deltas| Op::ToolCall { deltas }),
        ]
    }

    proptest! {
        /// Folding the yielded events with a fresh accumulator equals
        /// `result()` content-wise.
        #[test]
        fn fold_consistency(ops in prop::collection::vec(op_strategy(), 0..4)) {
            let (events, message) = futures::executor::block_on(drive(&ops));
            let mut acc = AssistantMessageAccumulator::new(base());
            for event in &events {
                acc.apply(event);
            }
            assert_eq!(
                serde_json::to_value(acc.message()).unwrap_or_default(),
                serde_json::to_value(&message).unwrap_or_default(),
            );
        }
    }
}

#[cfg(test)]
#[expect(
    clippy::panic,
    clippy::excessive_nesting,
    reason = "test code asserts freely"
)]
mod violation_tests {
    use futures::StreamExt;
    use serde_json::json;

    use crate::stream::*;
    use crate::types::content::StopReason;
    use crate::types::message::AssistantMessage;

    fn base() -> AssistantMessage {
        serde_json::from_value(json!({
            "role": "assistant",
            "content": [],
            "api": "anthropic-messages",
            "provider": "anthropic",
            "model": "m",
            "usage": {
                "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "totalTokens": 0,
                "cost": { "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "total": 0 },
            },
            "stopReason": "pending",
            "timestamp": 0,
        }))
        .unwrap_or_else(|e| panic!("base: {e}"))
    }

    /// Release-path: an ordering violation terminates the stream with
    /// `stopReason: error` and a violation `errorMessage`, never a panic.
    #[test]
    fn delta_before_start_terminates_with_error() {
        futures::executor::block_on(async {
            let (mut tx, mut stream) = AssistantStream::channel(base(), 8);
            tx.skip_violation_asserts();
            tx.text_start().await.ok();
            // No `start` yet, and a delta on an unopened index is invalid.
            tx.text_delta(7, "x").await.ok();
            let mut saw_error = false;
            while let Some(event) = stream.next().await {
                if let AssistantMessageEvent::Error {
                    reason,
                    error_message,
                } = &event
                {
                    saw_error = true;
                    assert_eq!(*reason, ErrorReason::Error);
                    assert!(error_message.contains("stream protocol violation"));
                }
            }
            assert!(saw_error);
            drop(tx);
            let message = stream.result().await;
            assert_eq!(message.stop_reason, StopReason::Error);
            assert!(
                message
                    .error_message
                    .unwrap_or_default()
                    .contains("text_delta")
            );
        });
    }

    /// Events after the terminal are protocol violations too.
    #[test]
    fn events_after_done_terminate_with_error() {
        futures::executor::block_on(async {
            let (mut tx, mut stream) = AssistantStream::channel(base(), 8);
            tx.skip_violation_asserts();
            tx.start().await.ok();
            tx.start().await.ok(); // second start: violation
            let mut saw_error = false;
            while let Some(event) = stream.next().await {
                if matches!(event, AssistantMessageEvent::Error { .. }) {
                    saw_error = true;
                }
            }
            assert!(saw_error);
            drop(tx);
            assert_eq!(stream.result().await.stop_reason, StopReason::Error);
        });
    }
}
