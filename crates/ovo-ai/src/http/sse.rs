//! Server-Sent Events decoding — a verbatim port of the Stainless
//! `LineDecoder` (`internal/decoders/line.js`, `openai` 7.19 semantics)
//! and `SSEDecoder` (`core/streaming.js`) used by both provider SDKs.
//!
//! Written in-house rather than via `sse-stream`: the SDK decoders
//! carry observable quirks the crate does not reproduce — the
//! `skipLeadingLF` state for a `\r` ending the last byte of a chunk,
//! the EOF `flush()` delivering a pending event, `:` comment lines
//! collected into `raw`, and the single-leading-space strip on field
//! values. A spec-exact port is small and vector-verified.

use std::pin::Pin;
use std::task::{Context, Poll};

use futures::Stream;

/// `LineDecoder`: incremental byte-to-line decoding (httpx-style).
///
/// UTF-8 is decoded per complete line, so multi-byte characters split
/// across chunks stay intact. `\n`, `\r\n` and lone `\r` all end lines;
/// a `\r` that ends the buffer defers the decision — `skip_leading_lf`
/// swallows a `\n` arriving at the start of the next chunk.
#[derive(Debug, Default)]
pub struct LineDecoder {
    buffer: Vec<u8>,
    /// The line-ending `\r` fell at the buffer end — skip a `\n` that
    /// starts the next chunk.
    skip_leading_lf: bool,
}

fn find_newline_index(buffer: &[u8], start: usize) -> Option<(usize, usize, bool)> {
    // (preceding, index-after, carriage)
    let mut i = start;
    while i < buffer.len() {
        let Some(&b) = buffer.get(i) else {
            break;
        };
        match b {
            b'\n' => return Some((i, i + 1, false)),
            b'\r' => return Some((i, i + 1, true)),
            _ => i += 1,
        }
    }
    None
}

impl LineDecoder {
    /// `decode(chunk)`: feed bytes, drain completed lines.
    pub fn decode(&mut self, chunk: &[u8]) -> Vec<String> {
        if chunk.is_empty() {
            return Vec::new();
        }
        let mut chunk = chunk;
        if self.skip_leading_lf {
            self.skip_leading_lf = false;
            if chunk.first() == Some(&b'\n') {
                chunk = chunk.get(1..).unwrap_or_default();
            }
            if chunk.is_empty() {
                return Vec::new();
            }
        }
        self.buffer.extend_from_slice(chunk);
        let mut lines = Vec::new();
        let mut start = 0usize;
        let mut search = 0usize;
        while let Some((preceding, index, carriage)) = find_newline_index(&self.buffer, search) {
            let Some(line_bytes) = self.buffer.get(start..preceding) else {
                break;
            };
            lines.push(String::from_utf8_lossy(line_bytes).into_owned());
            start = index;
            if carriage && self.buffer.get(start) == Some(&b'\n') {
                start += 1;
            } else if carriage && start == self.buffer.len() {
                self.skip_leading_lf = true;
            }
            search = start;
        }
        self.buffer.drain(..start);
        lines
    }

    /// `flush()`: at EOF, decode the pending buffer as if `\n` ended it.
    pub fn flush(&mut self) -> Vec<String> {
        if self.buffer.is_empty() {
            return Vec::new();
        }
        let rest = std::mem::take(&mut self.buffer);
        self.buffer = rest;
        self.decode(b"\n")
    }
}

/// `SseEvent`: one decoded record — `{event, data, raw}` per the SDKs.
///
/// `event` is `None` when the record carried no `event:` field; `data`
/// joins `data:` lines with `\n`; `raw` keeps the record's lines.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SseEvent {
    /// `event:` field value (absent when the record carried none).
    pub event: Option<String>,
    /// Joined `data:` lines (`\n`-joined).
    pub data: String,
    /// Raw lines of the record, for diagnostics.
    pub raw: Vec<String>,
}

/// `EndOfStream`: what to do with a record left pending when the
/// stream ends without a trailing blank line — a real difference
/// between the two vendored SDKs, observable on truncated streams.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum EndOfStream {
    /// `flush()` emits the pending record — `openai@7.19` (and pi's
    /// non-SDK adapters that drive `openai`'s `_iterSSEMessages`).
    #[default]
    Flush,
    /// `flush()` discards the pending record — `@anthropic-ai/sdk@0.129`.
    Discard,
}

/// `SSEDecoder`: line-to-record accumulation.
///
/// Blank line dispatches when an event or data was seen; `:` lines
/// are comments kept in `raw`; field values drop one leading space.
/// `policy` selects the SDK's end-of-stream semantics
/// (`anthropic-messages` → [`EndOfStream::Discard`]; `openai-*` →
/// [`EndOfStream::Flush`]).
#[derive(Debug, Default)]
pub struct SseDecoder {
    event: Option<String>,
    data: Vec<String>,
    chunks: Vec<String>,
    policy: EndOfStream,
}

impl SseDecoder {
    /// `new` — `policy` mirrors the vendored SDK's end-of-stream rule.
    #[must_use]
    pub const fn new(policy: EndOfStream) -> Self {
        Self {
            event: None,
            data: Vec::new(),
            chunks: Vec::new(),
            policy,
        }
    }

    /// `decode(line)` — `Some(event)` on dispatch.
    pub fn decode(&mut self, line: &str) -> Option<SseEvent> {
        let line = line.strip_suffix('\r').unwrap_or(line);
        if line.is_empty() {
            if self.event.is_none() && self.data.is_empty() {
                return None;
            }
            let event = SseEvent {
                event: self.event.take(),
                data: self.data.join("\n"),
                raw: std::mem::take(&mut self.chunks),
            };
            self.data.clear();
            return Some(event);
        }
        self.chunks.push(line.to_owned());
        if line.starts_with(':') {
            return None;
        }
        let (fieldname, value) = line.split_once(':').map_or((line, ""), |(f, v)| (f, v));
        let value = value.strip_prefix(' ').unwrap_or(value);
        match fieldname {
            "event" => self.event = Some(value.to_owned()),
            "data" => self.data.push(value.to_owned()),
            _ => {}
        }
        None
    }

    /// `flush()`: under [`EndOfStream::Flush`] (openai@7.19) emit a
    /// pending event when the stream ended without a trailing blank
    /// line; under [`EndOfStream::Discard`] (anthropic@0.129) drop it.
    pub fn flush(&mut self) -> Option<SseEvent> {
        match self.policy {
            EndOfStream::Flush => self.decode(""),
            EndOfStream::Discard => {
                self.event = None;
                self.data.clear();
                self.chunks.clear();
                None
            }
        }
    }
}

/// `sse_events`: adapt a byte stream into [`SseEvent`]s.
///
/// `LineDecoder`/`SseDecoder` flushes run at EOF; `policy` selects the
/// SDK's end-of-stream semantics ([`EndOfStream`]). Cancellation is
/// left to the caller (drop the stream / `select!` against a token).
#[derive(Debug)]
pub struct SseEventStream<S> {
    inner: S,
    lines: LineDecoder,
    decoder: SseDecoder,
    pending: std::collections::VecDeque<SseEvent>,
    done: bool,
}

impl<S, E> SseEventStream<S>
where
    S: Stream<Item = Result<bytes::Bytes, E>> + Unpin,
    E: std::fmt::Debug,
{
    /// Wrap a reqwest-style `bytes` stream under `policy`.
    pub fn new(inner: S, policy: EndOfStream) -> Self {
        Self {
            inner,
            lines: LineDecoder::default(),
            decoder: SseDecoder::new(policy),
            pending: std::collections::VecDeque::new(),
            done: false,
        }
    }
}

/// A decoder-level stream error.
#[derive(Debug)]
pub enum SseStreamError<E> {
    /// The underlying byte stream failed.
    Source(E),
}

impl<E: std::fmt::Debug> std::fmt::Display for SseStreamError<E> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl<E: std::fmt::Debug> std::error::Error for SseStreamError<E> {}

/// Feed a raw chunk: line-decode then event-decode each line.
fn decode_chunk(
    lines: &mut LineDecoder,
    decoder: &mut SseDecoder,
    pending: &mut std::collections::VecDeque<SseEvent>,
    chunk: &[u8],
) {
    for line in lines.decode(chunk) {
        push_decoded(decoder, pending, &line);
    }
}

/// Feed one decoded line through the `SseDecoder`, queueing any
/// dispatched record.
fn push_decoded(
    decoder: &mut SseDecoder,
    pending: &mut std::collections::VecDeque<SseEvent>,
    line: &str,
) {
    if let Some(event) = decoder.decode(line) {
        pending.push_back(event);
    }
}

/// EOF: flush pending line bytes, then the decoder itself.
fn finish_decoder(
    lines: &mut LineDecoder,
    decoder: &mut SseDecoder,
    pending: &mut std::collections::VecDeque<SseEvent>,
) {
    for line in lines.flush() {
        push_decoded(decoder, pending, &line);
    }
    if let Some(event) = decoder.flush() {
        pending.push_back(event);
    }
}

impl<S, E> Stream for SseEventStream<S>
where
    S: Stream<Item = Result<bytes::Bytes, E>> + Unpin,
    E: std::fmt::Debug,
{
    type Item = Result<SseEvent, SseStreamError<E>>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        loop {
            if let Some(event) = this.pending.pop_front() {
                return Poll::Ready(Some(Ok(event)));
            }
            if this.done {
                return Poll::Ready(None);
            }
            match Pin::new(&mut this.inner).poll_next(cx) {
                Poll::Ready(Some(Ok(chunk))) => {
                    decode_chunk(
                        &mut this.lines,
                        &mut this.decoder,
                        &mut this.pending,
                        &chunk,
                    );
                }
                Poll::Ready(Some(Err(e))) => {
                    this.done = true;
                    return Poll::Ready(Some(Err(SseStreamError::Source(e))));
                }
                Poll::Ready(None) => {
                    this.done = true;
                    finish_decoder(&mut this.lines, &mut this.decoder, &mut this.pending);
                }
                Poll::Pending => return Poll::Pending,
            }
        }
    }
}
