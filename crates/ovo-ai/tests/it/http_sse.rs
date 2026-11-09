//! SSE vectors.
//!
//! parity: AI-HTTP-SSE — `LineDecoder`/`SseDecoder` replay the
//! Stainless decoders end to end. Where the two vendored SDKs genuinely
//! differ (EOF flush of a pending record — `anthropic` drops it,
//! `openai` emits it), the vector carries both expectations; the Rust
//! decoder follows the `openai` `flush()` semantics by default.

use ovo_ai::http::sse::{EndOfStream, LineDecoder, SseDecoder};
use serde_json::Value;

#[test]
fn sse_vectors() {
    let text = std::fs::read_to_string(format!(
        "{}/../../vectors/ai/sse.json",
        env!("CARGO_MANIFEST_DIR")
    ))
    .unwrap_or_else(|e| panic!("read sse.json: {e}"));
    let cases: Vec<Value> = serde_json::from_str(&text).unwrap_or_else(|e| panic!("parse: {e}"));
    for case in cases {
        let name = case["name"].as_str().unwrap_or_default();
        let bytes: Vec<u8> = case["bytes_hex"]
            .as_str()
            .unwrap_or_default()
            .as_bytes()
            .chunks(2)
            .map(|h| u8::from_str_radix(std::str::from_utf8(h).unwrap_or(""), 16).unwrap_or(0))
            .collect();
        let splits: Vec<usize> = case["splits"]
            .as_array()
            .map(|l| {
                l.iter()
                    .filter_map(Value::as_u64)
                    .map(|v| v as usize)
                    .collect()
            })
            .unwrap_or_default();
        let mut lines = LineDecoder::default();
        let (expected, policy) = if let Some(e) = case.get("expected_anthropic") {
            (e.clone(), EndOfStream::Discard)
        } else if let Some(e) = case.get("expected_openai") {
            (e.clone(), EndOfStream::Flush)
        } else {
            (case["expected"].clone(), EndOfStream::Flush)
        };
        let mut decoder = SseDecoder::new(policy);
        let mut got: Vec<Value> = Vec::new();
        let mut at = 0usize;
        for end in splits.iter().copied().chain([bytes.len()]) {
            let end = end.min(bytes.len());
            if end > at {
                for line in lines.decode(&bytes[at..end]) {
                    if let Some(e) = decoder.decode(&line) {
                        got.push(serde_json::json!({"event": e.event, "data": e.data, "raw": e.raw}));
                    }
                }
            }
            at = end;
        }
        for line in lines.flush() {
            if let Some(e) = decoder.decode(&line) {
                got.push(serde_json::json!({"event": e.event, "data": e.data, "raw": e.raw}));
            }
        }
        if let Some(e) = decoder.flush() {
            got.push(serde_json::json!({"event": e.event, "data": e.data, "raw": e.raw}));
        }
        assert_eq!(Value::Array(got), expected, "{name}");
    }
}
