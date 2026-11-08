//! Lenient JSON repair and finished-argument parsing, ported from
//! `json-parse.ts` (`repairJson`, `parseJsonWithRepair`,
//! `parseToolCallArguments`; `parseStreamingJson` is unused upstream and
//! not ported).

use serde_json::{Map, Value};

use super::JsonError;

/// Repair malformed JSON string literals: escape raw control characters
/// inside strings and double backslashes before invalid escape characters.
#[must_use]
pub fn repair_json(text: &str) -> String {
    let mut repaired = String::with_capacity(text.len() + 8);
    let mut in_string = false;
    let mut chars = text.chars().peekable();
    while let Some(char) = chars.next() {
        if !in_string {
            repaired.push(char);
            if char == '"' {
                in_string = true;
            }
            continue;
        }
        if char == '"' {
            repaired.push(char);
            in_string = false;
            continue;
        }
        if char == '\\' {
            let Some(&next) = chars.peek() else {
                repaired.push_str("\\\\");
                continue;
            };
            if next == 'u'
                && let Some(digits) = unicode_digits(&mut chars)
            {
                repaired.push_str("\\u");
                repaired.push_str(&digits);
                continue;
            }
            if matches!(next, '"' | '\\' | '/' | 'b' | 'f' | 'n' | 'r' | 't') {
                repaired.push('\\');
                repaired.push(next);
                chars.next();
                continue;
            }
            repaired.push_str("\\\\");
            continue;
        }
        repaired.push_str(&escape_control(char));
    }
    repaired
}

/// Consume the `u` plus four hex digits when a valid `\uXXXX` escape
/// follows the backslash.
fn unicode_digits(chars: &mut std::iter::Peekable<std::str::Chars<'_>>) -> Option<String> {
    let digits: String = chars.clone().skip(1).take(4).collect();
    if digits.len() != 4 || !digits.bytes().all(|b| b.is_ascii_hexdigit()) {
        return None;
    }
    for _ in 0..5 {
        chars.next();
    }
    Some(digits)
}

fn escape_control(char: char) -> String {
    let mut buf = [0u8; 4];
    match char {
        c if ('\u{00}'..='\u{1f}').contains(&c) => match c {
            '\u{08}' => String::from("\\b"),
            '\u{0c}' => String::from("\\f"),
            '\n' => String::from("\\n"),
            '\r' => String::from("\\r"),
            '\t' => String::from("\\t"),
            c => format!("\\u{:04x}", c as u32),
        },
        c => c.encode_utf8(&mut buf).to_owned(),
    }
}

/// Parse `text` like upstream `parseJsonWithRepair`: `JSON.parse`
/// semantics first, one `repairJson` retry when it changed the text.
///
/// # Errors
/// Returns the first parse error when the input cannot be parsed even after
/// repair (the error's message is class `engine-error-text`).
pub fn parse_json_with_repair(text: &str) -> Result<Value, JsonError> {
    match super::from_str(text) {
        Ok(value) => Ok(value),
        Err(error) => {
            let repaired = repair_json(text);
            if repaired == text {
                Err(error)
            } else {
                super::from_str(&repaired)
            }
        }
    }
}

/// The parsed arguments of a finished tool call.
#[derive(Debug, PartialEq)]
#[expect(
    clippy::derive_partial_eq_without_eq,
    reason = "Value cannot implement Eq (f64)"
)]
pub struct ToolCallArguments {
    /// The arguments object; empty when the text was malformed.
    pub arguments: Map<String, Value>,
    /// The problem with the text, including the text the model sent.
    pub arguments_error: Option<String>,
}

/// Parse the model's complete arguments text of a finished tool call. Text
/// that is not a valid JSON object means malformed arguments: `arguments`
/// is then empty and `arguments_error` describes the problem.
#[must_use]
pub fn parse_tool_call_arguments(text: &str) -> ToolCallArguments {
    if text.trim().is_empty() {
        return ToolCallArguments {
            arguments: Map::new(),
            arguments_error: None,
        };
    }
    match parse_json_with_repair(text) {
        Ok(Value::Object(arguments)) => ToolCallArguments {
            arguments,
            arguments_error: None,
        },
        Ok(_) => ToolCallArguments {
            arguments: Map::new(),
            arguments_error: Some(format!(
                "Tool arguments must be a JSON object\n\nReceived arguments:\n{text}"
            )),
        },
        Err(error) => ToolCallArguments {
            arguments: Map::new(),
            arguments_error: Some(format!("{error}\n\nReceived arguments:\n{text}")),
        },
    }
}
