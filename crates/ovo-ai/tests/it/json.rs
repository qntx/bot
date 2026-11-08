//! Replays the `AI-JSON-JS` parity vectors generated from Node
//! `JSON.stringify`/`JSON.parse`.

#![allow(clippy::panic, reason = "integration tests fail by panicking")]
#![allow(
    clippy::tests_outside_test_module,
    reason = "integration test files are test modules"
)]

use serde::Deserialize;
use serde_json::Value;

#[derive(Debug, Deserialize)]
struct VectorFile {
    cases: Vec<VectorCase>,
}

#[derive(Debug, Deserialize)]
struct VectorCase {
    name: String,
    input: String,
    expected: String,
    #[serde(default)]
    classes: Vec<String>,
}

fn load_cases() -> Vec<VectorCase> {
    let text = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../vectors/ai/js-json.json"
    ));
    match serde_json::from_str::<VectorFile>(text) {
        Ok(file) => file.cases,
        Err(error) => panic!("invalid vector file: {error}"),
    }
}

/// What the `\u` escape at `index` is for the comparator below.
enum Escape {
    /// Lone surrogate escape: replaced by U+FFFD.
    Lone,
    /// Valid surrogate pair: kept verbatim.
    Pair,
    /// Anything else: copied byte by byte.
    Other,
}

fn classify_escape(bytes: &[u8], index: usize) -> Escape {
    match bytes.get(index + 2..index + 6).and_then(hex4) {
        Some(0xD800..=0xDBFF) => {
            if bytes.get(index + 6..index + 8) == Some(&b"\\u"[..])
                && matches!(
                    bytes.get(index + 8..index + 12).and_then(hex4),
                    Some(0xDC00..=0xDFFF)
                )
            {
                Escape::Pair
            } else {
                Escape::Lone
            }
        }
        Some(0xDC00..=0xDFFF) => Escape::Lone,
        _ => Escape::Other,
    }
}

/// Expected output with every lone surrogate `\uXXXX` escape replaced by the
/// literal U+FFFD character: `JSON.parse` keeps lone surrogates, ovo repairs
/// them to U+FFFD (class `lone-surrogates`).
fn expected_with_repaired_surrogates(expected: &str) -> String {
    let bytes = expected.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0usize;
    while i < bytes.len() {
        match bytes.get(i..i + 2) {
            Some(b"\\\\") => {
                out.extend_from_slice(b"\\\\");
                i += 2;
            }
            Some(b"\\u") => match classify_escape(bytes, i) {
                Escape::Lone => {
                    out.extend_from_slice("\u{FFFD}".as_bytes());
                    i += 6;
                }
                Escape::Pair => {
                    if let Some(pair) = bytes.get(i..i + 12) {
                        out.extend_from_slice(pair);
                    }
                    i += 12;
                }
                Escape::Other => {
                    if let Some(&byte) = bytes.get(i) {
                        out.push(byte);
                    }
                    i += 1;
                }
            },
            _ => {
                if let Some(&byte) = bytes.get(i) {
                    out.push(byte);
                }
                i += 1;
            }
        }
    }
    String::from_utf8(out).unwrap_or_default()
}

const fn hex4(digits: &[u8]) -> Option<u16> {
    if digits.len() != 4 {
        return None;
    }
    let mut value = 0u16;
    let mut i = 0usize;
    while i < 4 {
        let digit = match digits[i] {
            b'0'..=b'9' => digits[i] - b'0',
            b'a'..=b'f' => digits[i] - b'a' + 10,
            b'A'..=b'F' => digits[i] - b'A' + 10,
            _ => return None,
        };
        value = value * 16 + digit as u16;
        i += 1;
    }
    Some(value)
}

/// parity: AI-JSON-JS
#[test]
fn json_round_trips_to_js_bytes() {
    let cases = load_cases();
    assert!(!cases.is_empty(), "vector file has no cases");
    let mut names = std::collections::HashSet::new();
    for case in &cases {
        assert!(
            names.insert(&case.name),
            "duplicate case name {}",
            case.name
        );
    }
    for case in &cases {
        let classes: std::collections::HashSet<&str> =
            case.classes.iter().map(String::as_str).collect();
        if classes.contains("input-limits") {
            assert!(
                ovo_ai::json::from_str::<Value>(&case.input).is_err(),
                "case {} must fail to parse",
                case.name,
            );
            continue;
        }
        let value = match ovo_ai::json::from_str::<Value>(&case.input) {
            Ok(value) => value,
            Err(error) => panic!("case {} failed to parse: {error}", case.name),
        };
        let expected = if classes.contains("lone-surrogates") {
            expected_with_repaired_surrogates(&case.expected)
        } else {
            case.expected.clone()
        };
        match ovo_ai::json::to_string(&value) {
            Ok(actual) => assert_eq!(actual, expected, "case {}", case.name),
            Err(error) => panic!("case {} failed to serialize: {error}", case.name),
        }
    }
}

#[cfg(test)]
mod comparator_tests {
    use super::*;

    #[test]
    fn replaces_lone_surrogate_escapes() {
        assert_eq!(
            expected_with_repaired_surrogates(r#""\ud800x""#),
            "\"\u{FFFD}x\""
        );
        assert_eq!(
            expected_with_repaired_surrogates(r#""\udc00""#),
            "\"\u{FFFD}\""
        );
    }

    #[test]
    fn keeps_pairs_and_other_escapes() {
        assert_eq!(
            expected_with_repaired_surrogates(r#""\ud800\udc00""#),
            r#""\ud800\udc00""#
        );
        assert_eq!(
            expected_with_repaired_surrogates(r#""\\ud800""#),
            r#""\\ud800""#
        );
    }
}
