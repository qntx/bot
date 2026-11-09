//! JS-semantics string helpers and the upstream `utils/text.ts` +
//! `utils/sanitize-unicode.ts` ports.
//!
//! ECMAScript indexes and slices strings by UTF-16 code units and trims the
//! ECMAScript whitespace set (U+FEFF included, U+0085 excluded). Every later
//! port that mirrors JS string behavior goes through these helpers rather
//! than Rust's `char`/`trim` semantics.

use crate::types::content::{SystemContentBlock, TextContent, UserContentBlock};
use crate::types::message::SystemMessage;

/// The ECMAScript whitespace set: `String.prototype.trim`/`trimStart`/
/// `trimEnd` and the `\s` regex class. Note U+FEFF is included and U+0085
/// (NEL) is not.
#[must_use]
pub const fn is_js_whitespace(c: char) -> bool {
    matches!(
        c,
        '\u{9}' | '\u{b}' | '\u{c}' | '\u{20}' | '\u{a0}' | '\u{1680}' | '\u{2000}'
            ..='\u{200a}'
                | '\u{2028}'
                | '\u{2029}'
                | '\u{202f}'
                | '\u{205f}'
                | '\u{3000}'
                | '\u{feff}'
    )
}

/// `text.length` — the UTF-16 code unit count, not the `char` count.
#[must_use]
pub fn js_len(text: &str) -> usize {
    text.chars().map(char::len_utf16).sum()
}

/// `text.slice(start, end)` with UTF-16 code-unit indices. A cut inside a
/// surrogate pair drops the half character (`lone-surrogates` class: JS
/// strings can hold the half code unit, Rust `String` cannot).
#[must_use]
pub fn js_slice(text: &str, start: usize, end: usize) -> &str {
    let mut from = text.len();
    let mut to = text.len();
    let mut units = 0usize;
    for (offset, c) in text.char_indices() {
        let next = units + c.len_utf16();
        if units <= start && start <= next {
            // A cut inside the pair drops the half character.
            from = if start == units {
                offset
            } else {
                offset + c.len_utf8()
            };
        }
        if units < end && end <= next {
            to = if end == next {
                offset + c.len_utf8()
            } else {
                offset
            };
            break;
        }
        units = next;
    }
    &text[from.min(to)..to]
}

/// `text.slice(start, end)` as an owned `String`, the upstream
/// `ownedSlice`.
///
/// In V8 a long slice keeps the whole original alive, so the upstream
/// helper forces a copy — in Rust `&str` borrows are already the cheap
/// view, so this exists only for parity with upstream call sites.
#[must_use]
pub fn owned_slice(text: &str, start: usize, end: usize) -> String {
    js_slice(text, start, end).to_owned()
}

/// `text.trim()` with the ECMAScript whitespace set.
#[must_use]
pub fn js_trim(text: &str) -> &str {
    text.trim_matches(is_js_whitespace)
}

/// `text.trimStart()` with the ECMAScript whitespace set.
#[must_use]
pub fn js_trim_start(text: &str) -> &str {
    text.trim_start_matches(is_js_whitespace)
}

/// `text.trimEnd()` with the ECMAScript whitespace set.
#[must_use]
pub fn js_trim_end(text: &str) -> &str {
    text.trim_end_matches(is_js_whitespace)
}

/// `replace(/[^a-zA-Z0-9_-]/g, "_")`, the id sanitizing both `OpenAI`
/// adapters apply: the regex matches UTF-16 code units, so a surrogate pair
/// becomes two underscores.
#[must_use]
pub fn replace_non_id_chars(id: &str) -> String {
    let mut out = String::with_capacity(id.len());
    for c in id.chars() {
        if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
            out.push(c);
        } else {
            out.extend(std::iter::repeat_n('_', c.len_utf16()));
        }
    }
    out
}

/// `sanitizeSurrogates` removes unpaired UTF-16 code units from a JS
/// string.
///
/// Rust `String`/`&str` cannot hold lone surrogates, so this is the
/// identity on valid input — the upstream behavior is only observable on
/// values that arrived as raw UTF-16, which the streaming layer handles on
/// its own units.
#[must_use]
pub const fn sanitize_surrogates(text: &str) -> &str {
    text
}

/// Content which carries text among its blocks.
pub trait HasTextBlocks {
    /// The text blocks in order.
    fn text_blocks(&self) -> Vec<&TextContent>;
}

impl HasTextBlocks for [TextContent] {
    fn text_blocks(&self) -> Vec<&TextContent> {
        self.iter().collect()
    }
}

/// `contentText(content, separator)`: the text blocks joined.
///
/// Accepts the block list of any content union that can hold text; string
/// content is returned by the caller side (`system_content_text`).
#[must_use]
pub fn content_text_blocks(blocks: &[impl TextBlock], separator: &str) -> String {
    let mut parts = Vec::new();
    for block in blocks {
        if let Some(text) = block.text() {
            parts.push(text);
        }
    }
    parts.join(separator)
}

/// A content block which may carry text.
pub trait TextBlock {
    /// The text when this is a text block.
    fn text(&self) -> Option<&str>;
}

impl TextBlock for SystemContentBlock {
    fn text(&self) -> Option<&str> {
        match self {
            Self::Text(content) => Some(&content.text),
        }
    }
}

impl TextBlock for UserContentBlock {
    fn text(&self) -> Option<&str> {
        match self {
            Self::Text(content) => Some(&content.text),
            Self::Image(_) => None,
        }
    }
}

impl TextBlock for crate::types::content::AssistantContentBlock {
    fn text(&self) -> Option<&str> {
        match self {
            Self::Text(content) => Some(&content.text),
            Self::Thinking(_) | Self::ToolCall(_) => None,
        }
    }
}

/// `contentText(message.content)` for a system message (`string | blocks`).
#[must_use]
pub fn system_content_text(content: &crate::types::content::TextOrBlocks) -> String {
    match content {
        crate::types::content::TextOrBlocks::Text(text) => text.clone(),
        crate::types::content::TextOrBlocks::Blocks(blocks) => content_text_blocks(blocks, "\n"),
    }
}

/// `getSystemMessageText`: the complete prompt — content followed by each
/// non-null section, joined by blank lines.
#[must_use]
pub fn get_system_message_text(message: &SystemMessage) -> String {
    let mut parts = vec![system_content_text(&message.content)];
    if let Some(sections) = &message.sections {
        for text in sections.values() {
            if let Some(text) = text.as_str() {
                parts.push(text.to_owned());
            }
        }
    }
    parts.retain(|part| !part.is_empty());
    parts.join("\n\n")
}

/// `renderSystemMessageUpdate`: a later system message framed by section
/// name for APIs that accept system messages mid-conversation.
#[must_use]
pub fn render_system_message_update(message: &SystemMessage) -> String {
    let mut parts = Vec::new();
    let text = system_content_text(&message.content);
    if !text.is_empty() {
        parts.push(text);
    }
    if let Some(sections) = &message.sections {
        for (name, value) in sections {
            if let Some(value) = value.as_str() {
                parts.push(format!(
                    "Updated system prompt section \"{name}\":\n\n{value}"
                ));
            } else {
                parts.push(format!("Removed system prompt section \"{name}\"."));
            }
        }
    }
    parts.join("\n\n")
}

/// `jsNumberToString`: `String(number)` as JavaScript writes it — the
/// same shortest-roundtrip form `JSON.stringify` emits, `NaN`,
/// `Infinity`, `-Infinity` spelled out; `-0` is `"0"`.
#[must_use]
pub fn js_number_to_string(n: f64) -> String {
    if n.is_nan() {
        return "NaN".to_owned();
    }
    if n == f64::INFINITY {
        return "Infinity".to_owned();
    }
    if n == f64::NEG_INFINITY {
        return "-Infinity".to_owned();
    }
    if n == 0.0 {
        return "0".to_owned();
    }
    crate::json::to_string(&serde_json::json!(n)).unwrap_or_default()
}

/// `jsNumberFromStr`: ECMAScript `Number(string)` — JS-whitespace trim,
/// empty → 0, `0x`/`0o`/`0b` prefixes, `Infinity`/`+Infinity`/
/// `-Infinity`, decimal literal grammar, anything else → `NaN`.
#[must_use]
pub fn js_number_from_str(text: &str) -> f64 {
    let trimmed = js_trim(text);
    if trimmed.is_empty() {
        return 0.0;
    }
    match trimmed {
        "Infinity" | "+Infinity" => return f64::INFINITY,
        "-Infinity" => return f64::NEG_INFINITY,
        _ => {}
    }
    let (sign, digits) = trimmed.strip_prefix('-').map_or_else(
        || {
            trimmed
                .strip_prefix('+')
                .map_or((1f64, trimmed), |rest| (1f64, rest))
        },
        |rest| (-1f64, rest),
    );
    if let Some(hex) = digits
        .strip_prefix("0x")
        .or_else(|| digits.strip_prefix("0X"))
    {
        if !hex.is_empty() && hex.chars().all(|c| c.is_ascii_hexdigit()) {
            return sign * u64::from_str_radix(hex, 16).map_or(f64::INFINITY, |v| v as f64);
        }
        return f64::NAN;
    }
    if let Some(oct) = digits
        .strip_prefix("0o")
        .or_else(|| digits.strip_prefix("0O"))
    {
        if !oct.is_empty() && oct.chars().all(|c| c.is_digit(8)) {
            return sign * u64::from_str_radix(oct, 8).map_or(f64::INFINITY, |v| v as f64);
        }
        return f64::NAN;
    }
    if let Some(bin) = digits
        .strip_prefix("0b")
        .or_else(|| digits.strip_prefix("0B"))
    {
        if !bin.is_empty() && bin.chars().all(|c| c == '0' || c == '1') {
            return sign * u64::from_str_radix(bin, 2).map_or(f64::INFINITY, |v| v as f64);
        }
        return f64::NAN;
    }
    // Decimal grammar: digits[.digits][e[+-]digits], .digits, or digits.
    let decimal = r"^-?\d*\.?\d*(?:[eE][+-]?\d+)?$";
    if let Ok(re) = regex::Regex::new(decimal)
        && re.is_match(trimmed)
        && trimmed.chars().any(|c| c.is_ascii_digit())
    {
        return trimmed.parse::<f64>().unwrap_or(f64::NAN);
    }
    f64::NAN
}
