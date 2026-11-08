//! Agent-level retry classification and policy: upstream `utils/retry.ts`.
//!
//! `retryAssistantCall` itself is deferred to M1-08: cancellation in ovo is
//! `tokio_util::sync::CancellationToken` (ADR-009), and the backoff sleep
//! needs the runtime timer that arrives with the HTTP layer.
//!
//! Error-text classification uses the upstream pattern fragments verbatim,
//! compiled with `regex::bytes` (ASCII `\d`/`\w`/`\s` classes and ASCII
//! case-insensitivity, matching the JS `i` flag on provider error text).
//! The two look-around fragments are rewritten RE2-equivalently for the
//! boolean `.test()` semantics upstream uses; each is commented.

use std::sync::LazyLock;

use serde::{Deserialize, Serialize};

use crate::types::content::StopReason;
use crate::types::message::AssistantMessage;

/// Verbatim upstream `NON_RETRYABLE_PROVIDER_LIMIT_ERROR_PATTERN`
/// fragments, joined with `|` under the `i` flag.
const NON_RETRYABLE_FRAGMENTS: &[&str] = &[
    "GoUsageLimitError",
    "FreeUsageLimitError",
    "Monthly usage limit reached",
    "available balance",
    "insufficient_quota",
    "out of budget",
    "quota exceeded",
    "billing",
    "subscription_sharing_usage_limit_exceeded",
    "subscription_sharing_workplace_individual_usage_limit_exceeded",
    // `"code"\\s*:\\s*"?1308(?![0-9])` — the lookahead consumes one extra
    // byte here; `.test()` only asks whether a match exists.
    r#""code"\s*:\s*"?1308(?:[^0-9]|$)"#,
];

/// Verbatim upstream `RETRYABLE_PROVIDER_ERROR_PATTERN` fragments.
const RETRYABLE_FRAGMENTS: &[&str] = &[
    "overloaded",
    "server_busy",
    "servers are currently busy",
    "currently experiencing high demand",
    "model is at capacity",
    "rate.?limit",
    "too many requests",
    // `(?<![a-z0-9])(?:429|...)(?![a-z0-9])` — lookbehind/lookahead are
    // rewritten as boundary classes; `.test()` semantics are identical.
    r"(?:^|[^a-z0-9])(?:429|500|502|503|504|520|524)(?:[^0-9]|$)",
    "service.?(?:is.?)?(?:temporarily.?)?unavailable",
    "server.?error",
    "internal.?error",
    "provider.?returned.?(?:an.?)?error",
    "exceeded request buffer limit while retrying upstream",
    "unable to forward upstream",
    "upstream.?request.?failed",
    "upstream_stream_read_error",
    "stream was interrupted",
    "http/?2 stream failed",
    "network.?error",
    "connection.?error",
    "connection.?refused",
    "connection.?lost",
    "other side closed",
    "fetch failed",
    r"request to \S+ failed, reason:",
    "getaddrinfo",
    "ENOTFOUND",
    "EAI_AGAIN",
    "upstream.?connect",
    "reset before headers",
    "socket hang up",
    "socket connection was closed",
    "timed? out",
    "timeout",
    "terminated",
    "websocket.?closed",
    "websocket.?error",
    "ended without",
    "stream ended before message_stop",
    "stream ended before a terminal response event",
    "http2 request did not get a response",
    "stream disconnected before completion",
    "pending stream has been canceled",
    "retry delay",
    "you can retry your request",
    "try your request again",
    "please retry your request",
    "ResourceExhausted",
    "subscription_sharing_usage_unavailable",
    "subscription_sharing_user_unavailable",
];

static NON_RETRYABLE: LazyLock<Option<regex::bytes::Regex>> = LazyLock::new(|| {
    regex::bytes::Regex::new(&format!("(?i){}", NON_RETRYABLE_FRAGMENTS.join("|"))).ok()
});

static RETRYABLE: LazyLock<Option<regex::bytes::Regex>> = LazyLock::new(|| {
    regex::bytes::Regex::new(&format!("(?i){}", RETRYABLE_FRAGMENTS.join("|"))).ok()
});

fn matches(pattern: &LazyLock<Option<regex::bytes::Regex>>, text: &str) -> bool {
    pattern
        .as_ref()
        .is_some_and(|regex| regex.is_match(text.as_bytes()))
}

/// `RetryPolicy`: bounded attempts with exponential backoff
/// (`baseDelayMs * 2^(attempt-1)`); `maxAgentDelayMs` caps each delay.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
#[non_exhaustive]
pub struct RetryPolicy {
    /// Whether retries are enabled.
    #[serde(default)]
    pub enabled: bool,
    /// Max retry attempts (0 = no retries). The initial call never counts.
    #[serde(default)]
    pub max_retries: u32,
    /// Base delay in ms; per-attempt delay is `baseDelayMs * 2^(attempt-1)`.
    #[serde(default)]
    pub base_delay_ms: f64,
    /// Optional cap for agent-level retry delays in ms.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_agent_delay_ms: Option<f64>,
}

/// `DEFAULT_MAX_AGENT_RETRY_DELAY_MS`: the default agent retry cap (60 s).
pub const DEFAULT_MAX_AGENT_RETRY_DELAY_MS: f64 = 60_000.0;

/// `retryDelayMs`: `min(baseDelayMs * 2^(attempt-1), maxAgentDelayMs ??
/// 60_000)`, saturating to `Number.MAX_SAFE_INTEGER` like upstream when the
/// product leaves the safe-integer range.
#[must_use]
pub fn retry_delay_ms(policy: &RetryPolicy, attempt: u32) -> f64 {
    let exponent = i32::try_from(attempt.saturating_sub(1)).unwrap_or(i32::MAX);
    let delay = policy.base_delay_ms * 2f64.powi(exponent);
    let is_safe_integer =
        delay.is_finite() && delay.fract() == 0.0 && delay.abs() <= 9_007_199_254_740_991.0;
    let safe_delay = if is_safe_integer {
        delay
    } else {
        9_007_199_254_740_991.0
    };
    safe_delay.min(
        policy
            .max_agent_delay_ms
            .unwrap_or(DEFAULT_MAX_AGENT_RETRY_DELAY_MS),
    )
}

/// `isRetryableAssistantError`: whether a failed assistant message looks
/// like a transient provider or transport error.
#[must_use]
pub fn is_retryable_assistant_error(message: &AssistantMessage) -> bool {
    if message.stop_reason != StopReason::Error {
        return false;
    }
    let Some(error_message) = &message.error_message else {
        return false;
    };
    if matches(&NON_RETRYABLE, error_message) {
        return false;
    }
    matches(&RETRYABLE, error_message)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_pattern_compiles() {
        assert!(
            NON_RETRYABLE.is_some(),
            "non-retryable patterns must compile"
        );
        assert!(RETRYABLE.is_some(), "retryable patterns must compile");
    }
}
