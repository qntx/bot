//! Incremental, tolerant parsing of JSON text that arrives in pieces.
//!
//! Port of `streaming-json.ts` for streamed tool-call arguments, minus
//! `snapshot()` and its copy-credit throttling: callers keep their own copy
//! of the value and apply the returned changes in place.
//!
//! Piece boundaries: upstream reads UTF-16 code units, so a provider piece
//! may split a surrogate pair; `&str` pieces here are always whole chars.
//! Escaped `\uXXXX` surrogate halves are buffered as code units, so a pair
//! split across pieces still decodes correctly; a surrogate that never gets
//! its partner materializes as U+FFFD, matching the `lone-surrogates`
//! difference class of `from_str`.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use super::repair::parse_tool_call_arguments;

/// One segment of a path from the root value.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum PathSegment {
    /// Object member name.
    Key(String),
    /// Array index.
    Index(usize),
}

/// One change a piece of text made to the parsed value; see the module docs.
///
/// `Set` fires when a key, an array item or a finished value appears (a new
/// string is set once with its first text, a container empty); `Append` when
/// the piece extended an open string. Unterminated numbers and literals are
/// not reported until they end. A container nested deeper than 32 levels is
/// reported as one `Set` of its whole value when it closes.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[expect(
    clippy::derive_partial_eq_without_eq,
    reason = "Value cannot implement Eq (f64)"
)]
#[serde(tag = "kind")]
pub enum JsonChange {
    /// Assign `value` at `path`.
    #[serde(rename = "set")]
    Set {
        /// Path from the root value.
        path: Vec<PathSegment>,
        /// The value to assign.
        value: Value,
    },
    /// Append `text` to the string at `path`.
    #[serde(rename = "append")]
    Append {
        /// Path from the root value.
        path: Vec<PathSegment>,
        /// The text to append.
        text: String,
    },
}

impl JsonChange {
    /// Apply the change to a value the writer keeps.
    ///
    /// Paths that do not resolve are ignored rather than panicking: for a
    /// consistent change stream this cannot happen — every `Set` for a
    /// container is emitted before the changes inside it, so the parent of
    /// each reported path always exists with the right type. The tolerance
    /// keeps `apply` total for callers replaying recorded changes.
    pub fn apply(&self, target: &mut Value) {
        match self {
            Self::Set { path, value } => apply_set(target, path, value),
            Self::Append { path, text } => {
                let Some(value) = resolve(target, path) else {
                    return;
                };
                if let Value::String(string) = value {
                    string.push_str(text);
                }
            }
        }
    }
}

fn apply_set(target: &mut Value, path: &[PathSegment], value: &Value) {
    if path.is_empty() {
        *target = value.clone();
        return;
    }
    let Some((last, parent)) = resolve_parent(target, path) else {
        return;
    };
    match (last, parent) {
        (PathSegment::Key(key), Value::Object(object)) => {
            object.insert(key.clone(), value.clone());
        }
        (PathSegment::Index(index), Value::Array(array)) => set_index(array, *index, value.clone()),
        _ => {}
    }
}

fn set_index(array: &mut Vec<Value>, index: usize, value: Value) {
    if index == array.len() {
        array.push(value);
    } else if let Some(slot) = array.get_mut(index) {
        *slot = value;
    }
}

fn resolve<'a>(target: &'a mut Value, path: &[PathSegment]) -> Option<&'a mut Value> {
    let mut value = target;
    for segment in path {
        value = match segment {
            PathSegment::Key(key) => value.as_object_mut()?.get_mut(key)?,
            PathSegment::Index(index) => value.as_array_mut()?.get_mut(*index)?,
        };
    }
    Some(value)
}

fn resolve_parent<'a>(
    target: &'a mut Value,
    path: &'a [PathSegment],
) -> Option<(&'a PathSegment, &'a mut Value)> {
    let (last, rest) = path.split_last()?;
    resolve(target, rest).map(|parent| (last, parent))
}

/// Where a value sits in its parent; `None` for a value in an invalid
/// position, which is read but dropped.
#[derive(Clone, Debug)]
enum Slot {
    Root,
    Member(String),
    Element(usize),
}

/// A frame of the open container stack.
#[derive(Debug)]
struct Frame {
    container: Value,
    slot: Option<Slot>,
    /// Where the container sits when its changes are reported; `None` for an
    /// invalid position or inside a too-deep parent.
    path: Option<Vec<PathSegment>>,
    /// Whether changes inside are reported; when not, the whole value is
    /// reported as it closes.
    reports_inside: bool,
    /// The key read last, where the member value goes.
    key: Option<String>,
    phase: Phase,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Phase {
    Key,
    Colon,
    Value,
    Comma,
}

/// An unterminated string.
#[derive(Debug)]
struct OpenString {
    target: StringTarget,
    slot: Option<Slot>,
    /// Accumulated UTF-16 code units, so escaped surrogate halves survive
    /// piece boundaries.
    text: Vec<u16>,
    /// Units already reflected in reported changes.
    emitted: usize,
    /// Escape read so far: empty for none, otherwise `\` or `\u…`.
    escape: String,
    /// Where a value string sits when its changes are reported.
    path: Option<Vec<PathSegment>>,
    /// Index into the current piece's changes of this string's newest change.
    change_index: Option<usize>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum StringTarget {
    Key,
    Value,
}

/// Changes are reported at most this deep, so a path costs O(1) to build.
const MAX_REPORTED_DEPTH: usize = 32;

/// Incremental, tolerant JSON parser; see the module docs for its rules.
#[derive(Debug)]
pub struct StreamingJsonParser {
    has_root: bool,
    done: bool,
    stack: Vec<Frame>,
    string: Option<OpenString>,
    scalar: String,
    scalar_slot: Option<Slot>,
    /// Changes of the piece being read.
    changes: Vec<JsonChange>,
}

impl Default for StreamingJsonParser {
    fn default() -> Self {
        Self::new()
    }
}

impl StreamingJsonParser {
    /// A parser with no text read yet.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            has_root: false,
            done: false,
            stack: Vec::new(),
            string: None,
            scalar: String::new(),
            scalar_slot: None,
            changes: Vec::new(),
        }
    }

    /// Read the next piece of text; returns the changes it made, in order.
    /// Costs time linear in `piece.len()`.
    pub fn append(&mut self, piece: &str) -> Vec<JsonChange> {
        self.changes = Vec::new();
        if let Some(open) = &mut self.string {
            open.change_index = None;
        }
        let mut index = 0usize;
        while index < piece.len() && !self.done {
            if self.string.is_some() {
                index = self.read_string(piece, index);
            } else if !self.scalar.is_empty() {
                index = self.read_scalar(piece, index);
            } else if let Some(char) = piece[index..].chars().next() {
                self.structural(char);
                index += char.len_utf8();
            } else {
                break;
            }
        }
        std::mem::take(&mut self.changes)
    }

    /// Read string characters from `start` until the string ends or the text
    /// does; returns the next index.
    #[expect(
        clippy::excessive_nesting,
        reason = "structure mirrors upstream readString for review fidelity"
    )]
    fn read_string(&mut self, piece: &str, start: usize) -> usize {
        let Some(mut open) = self.string.take() else {
            return start;
        };
        let mut run_start = start;
        let mut index = start;
        while index < piece.len() {
            let Some(&byte) = piece.as_bytes().get(index) else {
                break;
            };
            if !open.escape.is_empty() {
                if self.continue_escape(&mut open, byte as char) {
                    index += (byte as char).len_utf8();
                    run_start = index;
                    continue;
                }
                // The char is not part of the escape: read it normally below.
                run_start = index;
            }
            match byte {
                b'"' => {
                    if index > run_start {
                        self.push_text(&mut open, &piece[run_start..index]);
                    }
                    self.end_string(open);
                    return index + 1;
                }
                b'\\' => {
                    if index > run_start {
                        self.push_text(&mut open, &piece[run_start..index]);
                    }
                    open.escape = String::from("\\");
                    index += 1;
                    run_start = index;
                }
                _ => index += char_width(piece.as_bytes(), index),
            }
        }
        if piece.len() > run_start {
            self.push_text(&mut open, &piece[run_start..]);
        }
        self.string = Some(open);
        piece.len()
    }

    /// Append text to the open string (upstream `#appendString`): accumulate
    /// the units, place the current value, and emit the new units.
    fn push_text(&mut self, open: &mut OpenString, text: &str) {
        let units: Vec<u16> = text.encode_utf16().collect();
        self.push_units(open, units);
    }

    /// Append decoded escape units the same way.
    fn push_units(&mut self, open: &mut OpenString, units: Vec<u16>) {
        open.text.extend(units);
        if open.target != StringTarget::Value {
            return;
        }
        self.place_string(open);
        let mut safe_end = open.text.len();
        while safe_end > open.emitted
            && open
                .text
                .get(safe_end - 1)
                .is_some_and(|&unit| is_high_surrogate(unit))
        {
            safe_end -= 1;
        }
        if safe_end == open.emitted {
            return;
        }
        let text = lossy_slice(&open.text, open.emitted, safe_end);
        open.emitted = safe_end;
        self.report_string_text(open, text);
    }

    /// Store the string so far into its slot's container.
    fn place_string(&mut self, open: &OpenString) {
        let value = Value::String(String::from_utf16_lossy(&open.text));
        self.place(open.slot.as_ref(), value);
    }

    #[expect(
        clippy::excessive_nesting,
        reason = "structure mirrors upstream appendString for review fidelity"
    )]
    fn report_string_text(&mut self, open: &mut OpenString, text: String) {
        let Some(path) = &open.path else {
            return;
        };
        // Extend this string's newest change while nothing came after it;
        // otherwise report an append.
        let last = self.changes.len().wrapping_sub(1);
        if open.change_index == Some(last) {
            if let Some(change) = self.changes.last_mut() {
                match change {
                    JsonChange::Set { value, .. } => {
                        if let Value::String(string) = value {
                            string.push_str(&text);
                        }
                    }
                    JsonChange::Append { text: pending, .. } => pending.push_str(&text),
                }
            }
            return;
        }
        open.change_index = Some(self.changes.len());
        self.changes.push(JsonChange::Append {
            path: path.clone(),
            text,
        });
    }

    /// Continue an escape with `char`; returns false when `char` is not part
    /// of it and must be read as a normal string character.
    fn continue_escape(&mut self, open: &mut OpenString, char: char) -> bool {
        if open.escape == "\\" {
            if let Some(decoded) = simple_escape(char) {
                open.escape.clear();
                self.push_units(open, vec![decoded]);
                return true;
            }
            if char == 'u' {
                open.escape = String::from("\\u");
                return true;
            }
            // Invalid escape: keep the backslash as text and read the
            // character normally, as repairJson() does.
            open.escape.clear();
            self.push_units(open, vec![u16::from(b'\\')]);
            return false;
        }
        if char.is_ascii_hexdigit() {
            open.escape.push(char);
            if open.escape.len() < 6 {
                return true;
            }
            let decoded = open
                .escape
                .get(2..)
                .and_then(|digits| u16::from_str_radix(digits, 16).ok())
                .unwrap_or(0xFFFD);
            open.escape.clear();
            self.push_units(open, vec![decoded]);
            return true;
        }
        // `\u` without four hex digits: keep it as text and read the
        // character normally, as repairJson() does.
        let literal = std::mem::take(&mut open.escape);
        self.push_text(open, &literal);
        false
    }

    /// Read scalar characters from `start` up to the next delimiter.
    fn read_scalar(&mut self, piece: &str, start: usize) -> usize {
        let mut end = start;
        while piece
            .as_bytes()
            .get(end)
            .is_some_and(|&byte| !is_delimiter(byte as char))
        {
            end += char_width(piece.as_bytes(), end);
        }
        if end > start {
            self.scalar.push_str(&piece[start..end]);
        }
        if end < piece.len() {
            self.end_scalar();
        }
        end
    }

    /// Handle one character outside strings.
    fn structural(&mut self, char: char) {
        match char {
            ' ' | '\t' | '\n' | '\r' => {}
            '"' => self.start_string(),
            '{' | '[' => self.open_container(char == '{'),
            '}' | ']' => self.close_container(),
            ':' => self.colon(),
            ',' => self.comma(),
            _ => {
                self.scalar = char.to_string();
                self.scalar_slot = self.value_slot();
            }
        }
    }

    fn colon(&mut self) {
        let Some(frame) = self.stack.last_mut() else {
            return;
        };
        if frame.container.is_object() && frame.phase == Phase::Colon {
            frame.phase = Phase::Value;
        }
    }

    fn comma(&mut self) {
        let Some(frame) = self.stack.last_mut() else {
            return;
        };
        frame.phase = if frame.container.is_array() {
            Phase::Value
        } else {
            Phase::Key
        };
    }

    /// Where a value starting now goes, or `None` when this position takes
    /// no value.
    fn value_slot(&self) -> Option<Slot> {
        let Some(frame) = self.stack.last() else {
            return (!self.has_root).then_some(Slot::Root);
        };
        if frame.container.is_array() {
            let index = frame.container.as_array().map_or(0, Vec::len);
            return Some(Slot::Element(index));
        }
        // A missing colon is tolerated: `{"a" 1}` reads as `{"a": 1}`.
        if matches!(frame.phase, Phase::Value | Phase::Colon)
            && let Some(key) = &frame.key
        {
            return Some(Slot::Member(key.clone()));
        }
        None
    }

    fn start_string(&mut self) {
        if let Some(frame) = self.stack.last() {
            // A missing comma is tolerated: `{"a": 1 "b": 2}` reads the second member.
            if frame.container.is_object() && matches!(frame.phase, Phase::Key | Phase::Comma) {
                self.string = Some(OpenString {
                    target: StringTarget::Key,
                    slot: None,
                    text: Vec::new(),
                    emitted: 0,
                    escape: String::new(),
                    path: None,
                    change_index: None,
                });
                return;
            }
        }
        let slot = self.value_slot();
        let path = self.reported_path(slot.as_ref());
        let mut open = OpenString {
            target: StringTarget::Value,
            slot,
            text: Vec::new(),
            emitted: 0,
            escape: String::new(),
            path,
            change_index: None,
        };
        self.place_string(&open);
        if let Some(set_path) = open.path.take() {
            let index = self.changes.len();
            self.changes.push(JsonChange::Set {
                path: set_path.clone(),
                value: Value::String(String::new()),
            });
            open.change_index = Some(index);
            open.path = Some(set_path);
        }
        self.string = Some(open);
    }

    /// The path of a value at `slot` in the innermost open container, when
    /// its changes are reported.
    fn reported_path(&self, slot: Option<&Slot>) -> Option<Vec<PathSegment>> {
        let slot = slot?;
        if matches!(slot, Slot::Root) {
            return Some(Vec::new());
        }
        let frame = self.stack.last()?;
        if !frame.reports_inside {
            return None;
        }
        let mut path = frame.path.clone()?;
        path.push(match slot {
            Slot::Member(key) => PathSegment::Key(key.clone()),
            Slot::Element(index) => PathSegment::Index(*index),
            Slot::Root => return None,
        });
        Some(path)
    }

    fn end_string(&mut self, mut open: OpenString) {
        // Flush any units held back (a surrogate that never got its partner
        // becomes U+FFFD, matching `from_str`).
        if open.emitted < open.text.len() {
            let text = lossy_slice(&open.text, open.emitted, open.text.len());
            open.emitted = open.text.len();
            self.report_string_text(&mut open, text);
        }
        match open.target {
            StringTarget::Key => {
                if let Some(frame) = self.stack.last_mut() {
                    frame.key = Some(String::from_utf16_lossy(&open.text));
                    frame.phase = Phase::Colon;
                }
            }
            StringTarget::Value => {
                self.place_string(&open);
                self.value_ended();
            }
        }
    }

    fn end_scalar(&mut self) {
        let token = std::mem::take(&mut self.scalar);
        let slot = self.scalar_slot.take();
        if let Some(value) = scalar_value(&token) {
            self.place(slot.as_ref(), value.clone());
            if let Some(path) = self.reported_path(slot.as_ref()) {
                self.changes.push(JsonChange::Set { path, value });
            }
        }
        self.value_ended();
    }

    fn open_container(&mut self, is_object: bool) {
        let slot = self.value_slot();
        let path = self.reported_path(slot.as_ref());
        let container = if is_object {
            Value::Object(Map::new())
        } else {
            Value::Array(Vec::new())
        };
        self.place(slot.as_ref(), container.clone());
        self.stack.push(Frame {
            container: container.clone(),
            slot,
            reports_inside: path.as_ref().is_some_and(|p| p.len() < MAX_REPORTED_DEPTH),
            path: path.clone(),
            key: None,
            phase: if is_object { Phase::Key } else { Phase::Value },
        });
        if let Some(path) = path {
            self.changes.push(JsonChange::Set {
                path,
                value: container,
            });
        }
    }

    fn close_container(&mut self) {
        let Some(frame) = self.stack.pop() else {
            return;
        };
        // Write the finished container into its parent slot (upstream shares
        // the object; here the parent's copy updates at close).
        self.place(frame.slot.as_ref(), frame.container.clone());
        // Too deep to report inside: the whole value, final now.
        if !frame.reports_inside
            && let Some(path) = frame.path
        {
            self.changes.push(JsonChange::Set {
                path,
                value: frame.container,
            });
        }
        self.value_ended();
    }

    /// A value in `frame`, or the root value, has ended.
    fn value_ended(&mut self) {
        if let Some(frame) = self.stack.last_mut() {
            frame.phase = Phase::Comma;
        } else if self.has_root {
            self.done = true;
        }
    }

    /// Set the value at `slot` in the open path; nothing for a value in an
    /// invalid position.
    fn place(&mut self, slot: Option<&Slot>, value: Value) {
        let Some(slot) = slot else {
            return;
        };
        match slot {
            Slot::Root => self.has_root = true,
            Slot::Element(index) => self.place_element(*index, value),
            Slot::Member(key) => self.place_member(key, value),
        }
    }

    fn place_element(&mut self, index: usize, value: Value) {
        let Some(elements) = self
            .stack
            .last_mut()
            .and_then(|frame| frame.container.as_array_mut())
        else {
            return;
        };
        set_index(elements, index, value);
    }

    fn place_member(&mut self, key: &str, value: Value) {
        if let Some(members) = self
            .stack
            .last_mut()
            .and_then(|frame| frame.container.as_object_mut())
        {
            members.insert(key.to_owned(), value);
        }
    }
}

/// The arguments of one streaming tool call: the raw text for the one full
/// parse when the call ends, and a `StreamingJsonParser` for the changes in
/// between.
#[derive(Debug)]
pub struct ToolCallArgumentsStream {
    text: String,
    parser: StreamingJsonParser,
}

/// The result of [`ToolCallArgumentsStream::replace`].
#[derive(Debug, PartialEq)]
pub enum Replaced {
    /// The complete text continues the text so far; `delta` is the missing
    /// piece and `changes` are its changes.
    Continued {
        /// The missing tail of the complete text.
        delta: String,
        /// Changes of `delta` (empty when `delta` is empty).
        changes: Vec<JsonChange>,
    },
    /// The text does not continue the text so far; the caller resets its
    /// value and applies `changes`, which are those of the new full text.
    Restarted {
        /// Changes of the complete text read fresh.
        changes: Vec<JsonChange>,
    },
}

impl Default for ToolCallArgumentsStream {
    fn default() -> Self {
        Self::new()
    }
}

impl ToolCallArgumentsStream {
    /// An empty stream.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            text: String::new(),
            parser: StreamingJsonParser::new(),
        }
    }

    /// The raw argument text received so far.
    #[must_use]
    pub fn text(&self) -> &str {
        &self.text
    }

    /// Append a piece of argument text; returns the changes it made.
    pub fn append(&mut self, delta: &str) -> Vec<JsonChange> {
        self.text.push_str(delta);
        self.parser.append(delta)
    }

    /// Take the provider's complete argument text. When it continues the
    /// text so far returns `Continued` with the missing piece's changes;
    /// otherwise `Restarted`, and the caller resets its copy of the value
    /// and applies the new text's changes.
    pub fn replace(&mut self, text: &str) -> Replaced {
        if let Some(delta) = text.strip_prefix(self.text.as_str()) {
            if delta.is_empty() {
                return Replaced::Continued {
                    delta: String::new(),
                    changes: Vec::new(),
                };
            }
            return Replaced::Continued {
                delta: delta.to_owned(),
                changes: self.append(delta),
            };
        }
        self.text.clear();
        self.parser = StreamingJsonParser::new();
        Replaced::Restarted {
            changes: self.append(text),
        }
    }

    /// The final arguments: one full parse of the complete text, which
    /// reports malformed arguments.
    #[must_use]
    pub fn finish(&self) -> crate::json::repair::ToolCallArguments {
        parse_tool_call_arguments(&self.text)
    }
}

fn simple_escape(char: char) -> Option<u16> {
    Some(match char {
        '"' | '\\' | '/' => u16::from(char as u8),
        'b' => 0x08,
        'f' => 0x0C,
        'n' => u16::from(b'\n'),
        'r' => u16::from(b'\r'),
        't' => u16::from(b'\t'),
        _ => return None,
    })
}

/// `String::from_utf16_lossy` over a unit slice without indexing.
fn lossy_slice(units: &[u16], start: usize, end: usize) -> String {
    units
        .get(start..end)
        .map_or_else(String::new, String::from_utf16_lossy)
}

fn is_high_surrogate(unit: u16) -> bool {
    (0xD800..0xDC00).contains(&unit)
}

const fn is_delimiter(char: char) -> bool {
    matches!(
        char,
        ' ' | '\t' | '\n' | '\r' | ',' | ':' | '{' | '}' | '[' | ']' | '"'
    )
}

fn char_width(bytes: &[u8], index: usize) -> usize {
    let Some(&byte) = bytes.get(index) else {
        return 1;
    };
    if byte < 0x80 {
        1
    } else if byte < 0xE0 {
        2
    } else if byte < 0xF0 {
        3
    } else {
        4
    }
}

/// The value of a complete number or literal token; `None` for anything
/// else, which is dropped.
fn scalar_value(token: &str) -> Option<Value> {
    match token {
        "true" => Some(Value::Bool(true)),
        "false" => Some(Value::Bool(false)),
        "null" => Some(Value::Null),
        _ if is_json_number(token) => {
            // Integers keep the integer representation serde_json uses for
            // `JSON.parse` results; everything else is the JS f64.
            if token.bytes().all(|b| b == b'-' || b.is_ascii_digit()) {
                if let Ok(int) = token.parse::<i64>() {
                    return Some(Value::from(int));
                }
                if let Ok(int) = token.parse::<u64>() {
                    return Some(Value::from(int));
                }
            }
            token
                .parse::<f64>()
                .ok()
                .and_then(serde_json::Number::from_f64)
                .map_or(Some(Value::Null), |number| Some(Value::Number(number)))
        }
        _ => None,
    }
}

/// `^-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?$`
fn is_json_number(token: &str) -> bool {
    let bytes = token.as_bytes();
    let mut index = usize::from(bytes.first() == Some(&b'-'));
    let int = match bytes.get(index) {
        Some(b'0') => index + 1,
        Some(b'1'..=b'9') => {
            let mut end = index + 1;
            while bytes.get(end).is_some_and(u8::is_ascii_digit) {
                end += 1;
            }
            end
        }
        _ => return false,
    };
    index = int;
    if bytes.get(index) == Some(&b'.') {
        index += 1;
        let digits = index;
        while bytes.get(index).is_some_and(u8::is_ascii_digit) {
            index += 1;
        }
        if index == digits {
            return false;
        }
    }
    if matches!(bytes.get(index), Some(b'e' | b'E')) {
        index += 1;
        if matches!(bytes.get(index), Some(b'+' | b'-')) {
            index += 1;
        }
        let digits = index;
        while bytes.get(index).is_some_and(u8::is_ascii_digit) {
            index += 1;
        }
        if index == digits {
            return false;
        }
    }
    index == bytes.len()
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn collect(pieces: &[&str]) -> (Vec<JsonChange>, Value) {
        let mut parser = StreamingJsonParser::new();
        let mut changes = Vec::new();
        for piece in pieces {
            changes.extend(parser.append(piece));
        }
        let mut value = Value::Null;
        for change in &changes {
            change.apply(&mut value);
        }
        (changes, value)
    }

    #[test]
    fn reports_set_and_append_for_strings() {
        let (changes, value) = collect(&["{\"a\":\"he", "llo\",\"b\":1", "2,\"c\":tr", "ue}"]);
        assert_eq!(value, json!({"a": "hello", "b": 12, "c": true}));
        let appended = changes
            .iter()
            .filter(|change| matches!(change, JsonChange::Append { .. }))
            .count();
        assert!(appended >= 1, "expected append changes, got {changes:?}");
    }

    #[test]
    fn tolerates_missing_comma_and_colon() {
        let (_, value) = collect(&["{\"a\":1 2, \"b\" \"x\"}"]);
        // `2` sits in a comma position: dropped; `1` stays.
        assert_eq!(value, json!({"a": 1, "b": "x"}));
    }

    #[test]
    fn drops_values_in_invalid_positions() {
        let (_, first) = collect(&["{\"a\":1} {\"ignored\":2}"]);
        assert_eq!(first, json!({"a": 1}));
        let (_, second) = collect(&["{\"a\":1, \"loose\"}"]);
        assert_eq!(second, json!({"a": 1}));
    }

    #[test]
    fn pending_scalar_reports_only_when_it_ends() {
        let mut parser = StreamingJsonParser::new();
        // The open object and the member key are reported; the unterminated
        // number is not.
        assert_eq!(
            parser.append("{\"n\":1"),
            vec![JsonChange::Set {
                path: vec![],
                value: json!({})
            }],
        );
        assert_eq!(
            parser.append("2}"),
            vec![JsonChange::Set {
                path: vec![PathSegment::Key("n".to_owned())],
                value: json!(12)
            }],
        );
    }

    #[test]
    fn escape_split_across_pieces() {
        let (_, value) = collect(&["{\"a\":\"x", "\\u00", "e9\"}"]);
        assert_eq!(value, json!({"a": "x\u{e9}"}));
    }

    #[test]
    fn invalid_escape_keeps_backslash() {
        let (_, value) = collect(&["{\"a\":\"x \\d \\u12\"}"]);
        assert_eq!(value, json!({"a": "x \\d \\u12"}));
    }

    #[test]
    fn raw_control_characters_stay() {
        let (_, value) = collect(&["{\"code\":\"x\t\t  \"}"]);
        assert_eq!(value, json!({"code": "x\t\t  "}));
    }

    #[test]
    fn astral_escape_pair_split_across_pieces() {
        let (_, value) = collect(&["{\"a\":\"\\ud83d", "\\ude00\"}"]);
        assert_eq!(value, json!({"a": "\u{1F600}"}));
    }

    #[test]
    fn deep_container_reports_once_on_close() {
        let depth = 40;
        let mut text = String::from("[");
        for _ in 0..depth {
            text.push('[');
        }
        text.push_str(&"]".repeat(depth));
        text.push(']');
        let (changes, value) = collect(&[&text]);
        let parsed: Value = serde_json::from_str(&text).unwrap_or_default();
        assert_eq!(value, parsed);
        assert!(
            changes.iter().all(|c| c
                != &JsonChange::Set {
                    path: vec![],
                    value: parsed.clone()
                })
                || changes.last()
                    == Some(&JsonChange::Set {
                        path: vec![],
                        value: parsed
                    }),
            "deep container changes: {}",
            changes.len()
        );
        assert!(changes.len() <= 40, "{} changes", changes.len());
    }

    #[test]
    fn apply_is_total_on_garbage_paths() {
        let mut value = json!({"a": "x"});
        JsonChange::Set {
            path: vec![PathSegment::Key("b".to_owned())],
            value: json!(1),
        }
        .apply(&mut value);
        JsonChange::Append {
            path: vec![
                PathSegment::Key("missing".to_owned()),
                PathSegment::Index(0),
            ],
            text: "y".to_owned(),
        }
        .apply(&mut value);
        assert_eq!(value, json!({"a": "x", "b": 1}));
    }

    #[test]
    fn arguments_stream_finish() {
        let mut stream = ToolCallArgumentsStream::new();
        stream.append("{\"path\":\"a.ts\",\"content\":\"x");
        stream.append("y\"}");
        let finished = stream.finish();
        assert_eq!(finished.arguments.get("path"), Some(&json!("a.ts")));
        assert!(finished.arguments_error.is_none());
    }

    #[test]
    fn arguments_stream_finish_malformed() {
        let mut stream = ToolCallArgumentsStream::new();
        stream.append("{\"a\":");
        let finished = stream.finish();
        assert!(finished.arguments.is_empty());
        let error = finished.arguments_error.unwrap_or_default();
        assert!(error.ends_with("\n\nReceived arguments:\n{\"a\":"));
    }

    #[test]
    fn arguments_stream_replace_continues() {
        let mut stream = ToolCallArgumentsStream::new();
        stream.append("{\"a\":\"he");
        let Replaced::Continued { delta, changes } = stream.replace("{\"a\":\"hello\"}") else {
            unreachable!("continued");
        };
        assert_eq!(delta, "llo\"}");
        assert_ne!(changes, []);
    }

    #[test]
    fn arguments_stream_replace_restarts() {
        let mut stream = ToolCallArgumentsStream::new();
        stream.append("{\"a\":\"he");
        let Replaced::Restarted { changes } = stream.replace("[1,2]") else {
            unreachable!("restarted");
        };
        assert_ne!(changes, []);
        assert_eq!(stream.text(), "[1,2]");
    }

    #[test]
    fn char_at_a_time_matches_whole() {
        let text = "{\"a\":\"hello\",\"b\":[1,2,{\"c\":true}],\"d\":null,\"e\":-1.5e2}";
        let (whole_changes, whole) = collect(&[text]);
        let pieces: Vec<String> = text.chars().map(String::from).collect();
        let refs: Vec<&str> = pieces.iter().map(String::as_str).collect();
        let (_, piecewise) = collect(&refs);
        assert_eq!(whole, piecewise);
        assert_ne!(whole_changes, []);
    }
}

#[cfg(test)]
mod prop_tests {
    use proptest::prelude::*;
    use proptest::proptest;

    use super::*;

    /// Apply every change of `pieces` in order; returns the final value.
    fn apply_all(pieces: &[&str]) -> Value {
        let mut parser = StreamingJsonParser::new();
        let mut value = Value::Null;
        for piece in pieces {
            for change in parser.append(piece) {
                change.apply(&mut value);
            }
        }
        value
    }

    /// Split `text` at the cut points (byte positions are rounded to char
    /// boundaries) into pieces.
    fn split_at(text: &str, mut cuts: Vec<usize>) -> Vec<String> {
        cuts.sort_unstable();
        cuts.dedup();
        let mut pieces = Vec::new();
        let mut start = 0;
        for cut in cuts {
            let mut cut = cut.min(text.len());
            while cut < text.len() && !text.is_char_boundary(cut) {
                cut += 1;
            }
            if cut > start {
                pieces.push(text[start..cut].to_owned());
            }
            start = cut;
        }
        if start < text.len() {
            pieces.push(text[start..].to_owned());
        }
        pieces
    }

    proptest! {
        #[test]
        fn any_chunking_reaches_the_same_value(
            text in "\\{.*\\}",
            cuts in prop::collection::vec(any::<usize>(), 0..8),
        ) {
            let whole = apply_all(&[&text]);
            let pieces = split_at(&text, cuts.clone());
            let refs: Vec<&str> = pieces.iter().map(String::as_str).collect();
            let piecewise = apply_all(&refs);
            assert_eq!(piecewise, whole, "text: {text:?} cuts: {cuts:?}");
        }
    }

    /// Appending a large document one char at a time must stay linear; the
    /// test profile is opt-level 2.
    #[test]
    fn large_document_char_at_a_time() {
        use std::fmt::Write as _;
        let mut text = String::from("{\"data\":[");
        for index in 0..80_000 {
            if index > 0 {
                text.push(',');
            }
            write!(text, "\"item-{index}\"").ok();
        }
        text.push_str("]}");
        assert!(text.len() >= 1_000_000, "text is {} bytes", text.len());
        let start = std::time::Instant::now();
        let mut parser = StreamingJsonParser::new();
        let mut value = Value::Null;
        let mut buf = [0u8; 4];
        for char in text.chars() {
            let piece = char.encode_utf8(&mut buf).to_owned();
            for change in parser.append(&piece) {
                change.apply(&mut value);
            }
        }
        let elapsed = start.elapsed();
        assert!(
            elapsed.as_secs() < 2,
            "char-at-a-time append took {elapsed:?} for {} bytes",
            text.len()
        );
    }
}
