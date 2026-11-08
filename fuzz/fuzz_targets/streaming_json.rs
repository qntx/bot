#![no_main]

use libfuzzer_sys::fuzz_target;
use serde_json::Value;

use ovo_ai::json::streaming::StreamingJsonParser;

/// Arbitrary bytes become a lossy UTF-8 document split into random pieces;
/// applying every piece's changes must yield the same value as one append of
/// the whole text, and nothing may panic.
fuzz_target!(|data: &[u8]| {
	let text = String::from_utf8_lossy(data);
	// First bytes choose the cut points (max 7 pieces).
	let cuts = data.len().min(8);
	let mut pieces = Vec::new();
	let mut start = 0usize;
	for &byte in data.iter().take(cuts) {
		let mut cut = usize::from(byte) * text.len() / 256;
		while cut < text.len() && !text.is_char_boundary(cut) {
			cut += 1;
		}
		if cut > start {
			pieces.push(&text[start..cut]);
			start = cut;
		}
	}
	pieces.push(&text[start..]);

	let mut whole = StreamingJsonParser::new();
	let mut whole_value = Value::Null;
	for change in whole.append(&text) {
		change.apply(&mut whole_value);
	}

	let mut parser = StreamingJsonParser::new();
	let mut value = Value::Null;
	for piece in pieces {
		for change in parser.append(piece) {
			change.apply(&mut value);
		}
	}
	assert_eq!(value, whole_value);
});
