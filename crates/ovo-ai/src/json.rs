//! ECMAScript-flavored JSON read and write (ADR-011).
//!
//! [`to_string`] and [`to_vec`] produce the same bytes as `JSON.stringify`
//! (compact form) of the same value: numbers follow ECMAScript
//! `Number::toString` (`1`, `0`, `12345678901234567000`), object keys that are
//! canonical array-index strings sort numerically before all other keys, and
//! escaping matches `serde_json`'s, which already equals `JSON.stringify` for
//! valid Unicode input.
//!
//! [`from_str`] accepts the same documents as `JSON.parse` with two
//! documented limits (class `input-limits`): numbers outside the `f64` range
//! and nesting deeper than `serde_json`'s 128-level cap are rejected. Lone
//! UTF-16 surrogate escapes (`\ud800` without a low partner, or a lone low
//! one) are replaced with `U+FFFD` on parse (class `lone-surrogates`); pi
//! strips them through `sanitizeSurrogates` before sending.

mod formatter;
pub mod repair;
pub mod streaming;
mod surrogates;

use std::io;
use std::rc::Rc;

use serde::{Serialize, de::DeserializeOwned};
use serde_json::Serializer;

use formatter::{JsFormatter, SharedBuffer};

/// Error from the underlying `serde_json` read or write.
#[derive(Debug, thiserror::Error)]
#[error(transparent)]
pub struct JsonError(serde_json::Error);

/// JavaScript truthiness of a JSON value.
#[must_use]
pub fn is_truthy(value: &serde_json::Value) -> bool {
    use serde_json::Value;
    match value {
        Value::Null => false,
        Value::Bool(flag) => *flag,
        Value::Number(number) => number.as_f64().is_some_and(|n| n != 0.0 && !n.is_nan()),
        Value::String(text) => !text.is_empty(),
        Value::Array(_) | Value::Object(_) => true,
    }
}

/// Serialize `value` into the exact bytes `JSON.stringify(value)` produces.
///
/// # Errors
/// Returns the `serde_json` error when serialization fails.
pub fn to_string<T: ?Sized + Serialize>(value: &T) -> Result<String, JsonError> {
    let bytes = to_vec(value)?;
    // serde_json only ever emits valid UTF-8.
    String::from_utf8(bytes).map_err(|error| {
        JsonError(serde_json::Error::io(io::Error::new(
            io::ErrorKind::InvalidData,
            error,
        )))
    })
}

/// Serialize `value` into the exact bytes `JSON.stringify(value)` produces.
///
/// # Errors
/// Returns the `serde_json` error when serialization fails.
pub fn to_vec<T: ?Sized + Serialize>(value: &T) -> Result<Vec<u8>, JsonError> {
    let (writer, buffer) = SharedBuffer::pair();
    {
        let formatter = JsFormatter::new(Rc::clone(&buffer));
        let mut serializer = Serializer::with_formatter(writer, formatter);
        value.serialize(&mut serializer).map_err(JsonError)?;
    }
    Ok(std::mem::take(&mut *buffer.borrow_mut()))
}

/// Deserialize `text` the way `JSON.parse` would, minus the documented
/// `input-limits` cases.
///
/// # Errors
/// Returns the `serde_json` error when parsing fails. On the first failure
/// the text is scanned once for lone surrogate escapes; if any are repaired
/// with `\ufffd` the parse is retried once, otherwise the original error is
/// returned unchanged.
pub fn from_str<T: DeserializeOwned>(text: &str) -> Result<T, JsonError> {
    match serde_json::from_str(text) {
        Ok(value) => Ok(value),
        Err(error) => surrogates::repair_lone_surrogates(text).map_or_else(
            || Err(JsonError(error)),
            |repaired| serde_json::from_str(&repaired).map_err(JsonError),
        ),
    }
}
