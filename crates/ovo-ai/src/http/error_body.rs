//! `error-body.ts`: normalizing the SDK error object shapes providers
//! surface (Mistral, `openai`, `@google/genai`, AWS Bedrock) and the
//! `formatProviderError` rendering.

use serde_json::Value;

use crate::text::{js_len, owned_slice};

/// `MAX_PROVIDER_ERROR_BODY_CHARS`.
pub const MAX_PROVIDER_ERROR_BODY_CHARS: usize = 4000;

/// `NormalizedProviderError`: the struct each provider composes into
/// its display string.
#[derive(Clone, Debug, PartialEq)]
pub struct NormalizedProviderError {
    /// HTTP status, when one could be extracted.
    pub status: Option<f64>,
    /// Raw HTTP body reason, trimmed and truncated to the cap.
    pub body: Option<String>,
    /// `error.message` (or `safeJsonStringify(error)` upstream — the
    /// caller passes the message).
    pub message: String,
    /// `message` already contains the body (Anthropic's happy path).
    pub message_carries_body: bool,
}

/// `normalizeProviderError` over the JSON view of an SDK error object.
///
/// `message` is the error's message; `fields` carries the enumerable
/// properties the SDK attaches (`status`, `statusCode`, `body`,
/// `error`, `$metadata`, `$response`).
#[must_use]
pub fn normalize_provider_error(message: &str, fields: Option<&Value>) -> NormalizedProviderError {
    let Some(f) = fields.and_then(Value::as_object) else {
        return NormalizedProviderError {
            status: None,
            body: None,
            message: message.to_owned(),
            message_carries_body: false,
        };
    };
    let status = extract_status(f);
    let body = extract_body(f);
    let message_carries_body = body.as_ref().is_none_or(|b| message.contains(b.as_str()));
    NormalizedProviderError {
        status,
        body,
        message: message.to_owned(),
        message_carries_body,
    }
}

fn num_field(f: &serde_json::Map<String, Value>, key: &str) -> Option<f64> {
    f.get(key).and_then(Value::as_f64)
}

fn extract_status(f: &serde_json::Map<String, Value>) -> Option<f64> {
    num_field(f, "statusCode")
        .or_else(|| num_field(f, "status"))
        .or_else(|| {
            f.get("$metadata")
                .and_then(|m| m.get("httpStatusCode"))
                .and_then(Value::as_f64)
        })
        .or_else(|| {
            f.get("$response")
                .and_then(|m| m.get("statusCode"))
                .and_then(Value::as_f64)
        })
}

fn extract_body(f: &serde_json::Map<String, Value>) -> Option<String> {
    let body_text = pick_body_text(f)?;
    let trimmed = body_text.trim();
    if trimmed.is_empty() {
        return None;
    }
    Some(truncate_error_text(trimmed, MAX_PROVIDER_ERROR_BODY_CHARS))
}

fn pick_body_text(f: &serde_json::Map<String, Value>) -> Option<String> {
    if let Some(body) = f.get("body").and_then(Value::as_str) {
        return Some(body.to_owned());
    }
    if let Some(error) = f.get("error")
        && is_plain_non_empty_object(error)
    {
        return Some(crate::json::to_string(error).unwrap_or_default());
    }
    let response_body = f.get("$response").and_then(|r| r.get("body"));
    if let Some(body) = response_body.and_then(Value::as_str) {
        return Some(body.to_owned());
    }
    if let Some(body) = response_body
        && is_plain_non_empty_object(body)
    {
        return Some(crate::json::to_string(body).unwrap_or_default());
    }
    None
}

fn is_plain_non_empty_object(value: &Value) -> bool {
    value.as_object().is_some_and(|m| !m.is_empty())
}

/// `formatProviderError`.
#[must_use]
pub fn format_provider_error(norm: &NormalizedProviderError, prefix: Option<&str>) -> String {
    let status = norm.status;
    if norm.message_carries_body || status.is_none() || norm.body.is_none() {
        return match (prefix, status) {
            (Some(p), Some(s)) => format!("{p} ({s}): {}", norm.message),
            _ => norm.message.clone(),
        };
    }
    let body = norm.body.as_deref().unwrap_or_default();
    let status = status.unwrap_or_default();
    prefix.map_or_else(
        || format!("{status}: {body}"),
        |p| format!("{p} ({status}): {body}"),
    )
}

/// `truncateErrorText` — JS `length`/`slice` (UTF-16 units).
#[must_use]
pub fn truncate_error_text(text: &str, max_chars: usize) -> String {
    if js_len(text) <= max_chars {
        return text.to_owned();
    }
    format!(
        "{}... [truncated {} chars]",
        owned_slice(text, 0, max_chars),
        js_len(text) - max_chars
    )
}
