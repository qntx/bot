//! `serde_json` writer and `Formatter` pair that emit `JSON.stringify`
//! semantics: JS number formatting and JS property ordering for objects
//! whose keys are canonical array-index strings.

use std::cell::RefCell;
use std::io::{self, Write};
use std::rc::Rc;

use serde_json::ser::{CompactFormatter, Formatter};

/// Largest key value JS treats as an array index (`2^32 - 2`).
const ARRAY_INDEX_MAX: u64 = 4_294_967_294;

/// Largest integer JS serializes without f64 rounding (`2^53 - 1`).
const MAX_SAFE_INTEGER: u64 = (1 << 53) - 1;

/// Output buffer shared between the serializer's writer and [`JsFormatter`]
/// so the formatter can inspect and rewrite the exact emitted bytes.
pub(crate) struct SharedBuffer(Rc<RefCell<Vec<u8>>>);

impl SharedBuffer {
    /// Create the writer and a handle for the formatter/result side.
    pub(crate) fn pair() -> (Self, Rc<RefCell<Vec<u8>>>) {
        let buffer = Rc::new(RefCell::new(Vec::new()));
        (Self(Rc::clone(&buffer)), buffer)
    }
}

impl Write for SharedBuffer {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.0.borrow_mut().extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

/// Byte range of one serialized `key: value` pair, excluding the comma that
/// precedes it.
#[derive(Clone, Copy, Debug)]
struct Entry {
    /// Offset of the first byte of `"key"` (after the separating comma).
    start: usize,
    /// Offset one past the last byte of the value.
    end: usize,
    /// Numeric value when the key is a canonical array-index string.
    index_key: Option<u64>,
}

/// `serde_json` `Formatter` with `JSON.stringify` semantics. Methods not
/// overridden here keep the trait's compact defaults, matching
/// `JSON.stringify` output.
pub(crate) struct JsFormatter {
    buffer: Rc<RefCell<Vec<u8>>>,
    compact: CompactFormatter,
    /// Recorded entry ranges of every object still open; a shared stack so
    /// objects without index keys cost only bookkeeping.
    entries: Vec<Entry>,
    /// Per open object: base index into `entries` and the offset just after
    /// the opening `{` (start of the gap before the first entry).
    frames: Vec<(usize, usize)>,
    /// Offset of the first byte of the current key.
    key_start: usize,
}

impl JsFormatter {
    /// Create a formatter writing through `buffer`.
    pub(crate) const fn new(buffer: Rc<RefCell<Vec<u8>>>) -> Self {
        Self {
            buffer,
            compact: CompactFormatter,
            entries: Vec::new(),
            frames: Vec::new(),
            key_start: 0,
        }
    }

    fn buffer_len(&self) -> usize {
        self.buffer.borrow().len()
    }

    /// Rewrite the finished object's entries in JS property order: canonical
    /// array-index keys first in ascending numeric order, then the remaining
    /// keys in written order. Gap bytes between entries stay positional, so
    /// the same algorithm works for a pretty formatter later. The rewrite is
    /// in place and length-preserving, so outer entry offsets stay valid.
    fn permute(&self, content_start: usize, entries: &[Entry]) -> io::Result<()> {
        let Some(content_end) = entries.last().map(|entry| entry.end) else {
            return Ok(());
        };
        let mut order: Vec<(usize, bool, u64)> = entries
            .iter()
            .enumerate()
            .map(|(position, entry)| {
                (
                    position,
                    entry.index_key.is_none(),
                    entry.index_key.unwrap_or(0),
                )
            })
            .collect();
        // Stable sort: index-keyed entries keep written order among equals.
        order.sort_by_key(|&(_, not_index, index)| (not_index, index));

        let bytes = {
            let buffer = self.buffer.borrow();
            match buffer.get(content_start..content_end) {
                Some(slice) => slice.to_vec(),
                None => {
                    return Err(range_error());
                }
            }
        };

        // Emit each slot's original gap (bytes between the slot's entries)
        // followed by the entry the permutation moved into the slot.
        let mut rebuilt = Vec::with_capacity(bytes.len());
        let mut cursor = 0usize;
        for (slot, &(position, ..)) in order.iter().enumerate() {
            let (Some(slot_entry), Some(moved_entry)) = (entries.get(slot), entries.get(position))
            else {
                return Err(range_error());
            };
            let (gap_end, moved_start, moved_end) = (
                slot_entry.start - content_start,
                moved_entry.start - content_start,
                moved_entry.end - content_start,
            );
            match (
                bytes.get(cursor..gap_end),
                bytes.get(moved_start..moved_end),
            ) {
                (Some(gap), Some(range)) => {
                    rebuilt.extend_from_slice(gap);
                    rebuilt.extend_from_slice(range);
                }
                _ => return Err(range_error()),
            }
            cursor = slot_entry.end - content_start;
        }

        if rebuilt.len() != bytes.len() {
            return Err(range_error());
        }
        let mut buffer = self.buffer.borrow_mut();
        let target = buffer
            .get_mut(content_start..content_end)
            .ok_or_else(range_error)?;
        target.copy_from_slice(&rebuilt);
        Ok(())
    }
}

fn range_error() -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidData,
        "json: entry range outside output buffer",
    )
}

/// Classify serialized `"key"` bytes (quotes included): `Some(index)` when
/// the key is a canonical array-index string (`0`–`4294967294`, no leading
/// zeros except `"0"` itself).
fn array_index_key(bytes: &[u8]) -> Option<u64> {
    let inner = bytes.strip_prefix(b"\"")?.strip_suffix(b"\"")?;
    if inner.is_empty() || inner.len() > 10 || (inner.len() > 1 && inner.first() == Some(&b'0')) {
        return None;
    }
    let mut value = 0u64;
    for &byte in inner {
        if !byte.is_ascii_digit() {
            return None;
        }
        value = value * 10 + u64::from(byte - b'0');
    }
    (value <= ARRAY_INDEX_MAX).then_some(value)
}

impl Formatter for JsFormatter {
    fn begin_object<W>(&mut self, writer: &mut W) -> io::Result<()>
    where
        W: ?Sized + Write,
    {
        self.compact.begin_object(writer)?;
        self.frames.push((self.entries.len(), self.buffer_len()));
        Ok(())
    }

    fn begin_object_key<W>(&mut self, writer: &mut W, first: bool) -> io::Result<()>
    where
        W: ?Sized + Write,
    {
        self.compact.begin_object_key(writer, first)?;
        self.key_start = self.buffer_len();
        Ok(())
    }

    fn end_object_key<W>(&mut self, writer: &mut W) -> io::Result<()>
    where
        W: ?Sized + Write,
    {
        self.compact.end_object_key(writer)?;
        let index_key = {
            let buffer = self.buffer.borrow();
            buffer.get(self.key_start..).and_then(array_index_key)
        };
        self.entries.push(Entry {
            start: self.key_start,
            end: usize::MAX,
            index_key,
        });
        Ok(())
    }

    fn end_object_value<W>(&mut self, writer: &mut W) -> io::Result<()>
    where
        W: ?Sized + Write,
    {
        self.compact.end_object_value(writer)?;
        let end = self.buffer_len();
        if let Some(entry) = self.entries.last_mut() {
            entry.end = end;
        }
        Ok(())
    }

    fn end_object<W>(&mut self, writer: &mut W) -> io::Result<()>
    where
        W: ?Sized + Write,
    {
        if let Some((base, content_start)) = self.frames.pop() {
            if let Some(frame_entries) = self
                .entries
                .get(base..)
                .filter(|frame_entries| frame_entries.iter().any(|entry| entry.index_key.is_some()))
            {
                self.permute(content_start, frame_entries)?;
            }
            self.entries.truncate(base);
        }
        self.compact.end_object(writer)
    }

    fn write_u64<W>(&mut self, writer: &mut W, value: u64) -> io::Result<()>
    where
        W: ?Sized + Write,
    {
        if value <= MAX_SAFE_INTEGER {
            self.compact.write_u64(writer, value)
        } else {
            self.write_f64(writer, value as f64)
        }
    }

    fn write_i64<W>(&mut self, writer: &mut W, value: i64) -> io::Result<()>
    where
        W: ?Sized + Write,
    {
        if value.unsigned_abs() <= MAX_SAFE_INTEGER {
            self.compact.write_i64(writer, value)
        } else {
            self.write_f64(writer, value as f64)
        }
    }

    fn write_u128<W>(&mut self, writer: &mut W, value: u128) -> io::Result<()>
    where
        W: ?Sized + Write,
    {
        if value <= u128::from(MAX_SAFE_INTEGER) {
            self.compact.write_u128(writer, value)
        } else {
            self.write_f64(writer, value as f64)
        }
    }

    fn write_i128<W>(&mut self, writer: &mut W, value: i128) -> io::Result<()>
    where
        W: ?Sized + Write,
    {
        if value.unsigned_abs() <= u128::from(MAX_SAFE_INTEGER) {
            self.compact.write_i128(writer, value)
        } else {
            self.write_f64(writer, value as f64)
        }
    }

    fn write_f32<W>(&mut self, writer: &mut W, value: f32) -> io::Result<()>
    where
        W: ?Sized + Write,
    {
        self.write_f64(writer, f64::from(value))
    }

    fn write_f64<W>(&mut self, writer: &mut W, value: f64) -> io::Result<()>
    where
        W: ?Sized + Write,
    {
        if !value.is_finite() {
            return self.compact.write_null(writer);
        }
        // JSON.stringify writes "0" for both 0 and -0.
        if value == 0.0 {
            return writer.write_all(b"0");
        }
        writer.write_all(ryu_js::Buffer::new().format_finite(value).as_bytes())
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use serde_json::{Map, Value};

    use super::*;

    #[expect(clippy::panic, reason = "serialization cannot fail for these inputs")]
    fn emit<T: ?Sized + serde::Serialize>(value: &T) -> String {
        match crate::json::to_string(value) {
            Ok(text) => text,
            Err(error) => panic!("serialize failed: {error}"),
        }
    }

    #[test]
    fn classifies_array_index_keys() {
        assert_eq!(array_index_key(b"\"0\""), Some(0));
        assert_eq!(array_index_key(b"\"1\""), Some(1));
        assert_eq!(array_index_key(b"\"10\""), Some(10));
        assert_eq!(array_index_key(b"\"4294967294\""), Some(4_294_967_294));
        assert_eq!(array_index_key(b"\"4294967295\""), None);
        assert_eq!(array_index_key(b"\"42949672940\""), None);
        assert_eq!(array_index_key(b"\"01\""), None);
        assert_eq!(array_index_key(b"\"00\""), None);
        assert_eq!(array_index_key(b"\"-1\""), None);
        assert_eq!(array_index_key(b"\"1.5\""), None);
        assert_eq!(array_index_key(b"\"\""), None);
        assert_eq!(array_index_key(b"\"a\""), None);
    }

    fn object(entries: &[(&str, u64)]) -> Value {
        let mut map = Map::new();
        for &(key, value) in entries {
            map.insert(key.to_owned(), Value::from(value));
        }
        Value::Object(map)
    }

    #[test]
    fn reorders_index_keys_before_other_keys() {
        let value = object(&[("b", 1), ("2", 3), ("1", 4)]);
        assert_eq!(emit(&value), r#"{"1":4,"2":3,"b":1}"#);
    }

    #[test]
    fn reorders_nested_objects_independently() {
        let mut outer = Map::new();
        outer.insert("a".to_owned(), object(&[("b", 1), ("1", 2)]));
        outer.insert("2".to_owned(), object(&[("z", 0), ("0", 9)]));
        assert_eq!(
            emit(&Value::Object(outer)),
            r#"{"2":{"0":9,"z":0},"a":{"1":2,"b":1}}"#
        );
    }

    #[test]
    fn reorders_index_keys_inside_arrays() {
        let mut map = Map::new();
        map.insert(
            "items".to_owned(),
            Value::Array(vec![object(&[("b", 1), ("0", 5)])]),
        );
        assert_eq!(emit(&Value::Object(map)), r#"{"items":[{"0":5,"b":1}]}"#);
    }

    #[test]
    fn classifies_integer_map_keys() {
        let mut map = BTreeMap::new();
        map.insert(1u32, "a");
        map.insert(0u32, "b");
        assert_eq!(emit(&map), r#"{"0":"b","1":"a"}"#);
    }

    #[test]
    fn formats_js_numbers() {
        assert_eq!(emit(&1.0f64), "1");
        assert_eq!(emit(&-0.0f64), "0");
        assert_eq!(emit(&0.0f64), "0");
        assert_eq!(emit(&1.5f32), "1.5");
        assert_eq!(emit(&9_007_199_254_740_991u64), "9007199254740991");
        assert_eq!(emit(&9_007_199_254_740_992u64), "9007199254740992");
        assert_eq!(emit(&9_007_199_254_740_993u64), "9007199254740992");
        assert_eq!(emit(&12_345_678_901_234_567_890u64), "12345678901234567000");
    }
}
