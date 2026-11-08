//! Context-overflow detection: upstream `utils/overflow.ts`.
//!
//! The error-text pattern table is kept verbatim as JS regex fragments and
//! compiled with `regex::bytes`, whose non-Unicode `\d`/`\w`/`\s` classes
//! and ASCII `i` flag match the JS semantics these provider strings rely
//! on (upstream matches plain ASCII error text; JS `\d`/`\w` are ASCII
//! only anyway).

use std::sync::LazyLock;

use regex::bytes::Regex;

use crate::types::content::StopReason;
use crate::types::message::AssistantMessage;

/// Verbatim upstream `OVERFLOW_PATTERNS` fragments (`/.../i`).
const OVERFLOW_PATTERNS: &[&str] = &[
    r"prompt (?:is )?too long",
    r"prompt exceeds max length",
    r"request_too_large",
    r"input is too long for requested model",
    r"exceeds the context window",
    r"exceeds (?:the )?(?:model'?s )?maximum context length(?: of [\d,]+ tokens?|\s*\([\d,]+\))",
    r"context window exceeded",
    r"input token count.*exceeds the maximum",
    r"maximum prompt length is \d+",
    r"reduce the length of the messages",
    r"maximum context length is \d+ tokens",
    r"exceeds (?:the )?maximum allowed input length of [\d,]+ tokens?",
    r"input \(\d+ tokens\) is longer than the model'?s context length \(\d+ tokens\)",
    r"exceeds the limit of \d+",
    r"exceeds the available context size",
    r"greater than the context length",
    r"context window exceeds limit",
    r"exceeded model token limit",
    r"too large for model with \d+ maximum context length",
    r"prompt has [\d,]+ tokens?, but the configured context size is [\d,]+ tokens?",
    r"model_context_window_exceeded",
    r"prompt too long; exceeded (?:max )?context length",
    r"range of input length should be",
    r"context[_ ]length[_ ]exceeded",
    r"too many tokens",
    r"token limit exceeded",
    r"^(?:[\w -]+ API error \(413\):|413(?:\s|:|$))",
];

const CEREBRAS_BODYLESS_OVERFLOW_PATTERN: &str = r"^4(?:00|13)\s*(?:status code)?\s*\(no body\)";

/// Verbatim upstream `NON_OVERFLOW_PATTERNS` fragments (`/.../i`).
const NON_OVERFLOW_PATTERNS: &[&str] = &[
    r"^(Throttling error|Service unavailable):",
    r"rate limit",
    r"too many requests",
];

static OVERFLOW: LazyLock<Option<Regex>> =
    LazyLock::new(|| Regex::new(&format!("(?i){}", OVERFLOW_PATTERNS.join("|"))).ok());

static NON_OVERFLOW: LazyLock<Option<Regex>> =
    LazyLock::new(|| Regex::new(&format!("(?i){}", NON_OVERFLOW_PATTERNS.join("|"))).ok());

static CEREBRAS: LazyLock<Option<Regex>> =
    LazyLock::new(|| Regex::new(&format!("(?i){CEREBRAS_BODYLESS_OVERFLOW_PATTERN}")).ok());

fn matches(pattern: &LazyLock<Option<Regex>>, text: &str) -> bool {
    pattern
        .as_ref()
        .is_some_and(|regex| regex.is_match(text.as_bytes()))
}

/// `isContextOverflow`: whether an assistant message reports a context
/// overflow — pattern-matched error, silent usage overflow (`contextWindow`
/// given), or a length stop with zero output at a filled window.
#[must_use]
pub fn is_context_overflow(message: &AssistantMessage, context_window: Option<u64>) -> bool {
    if message.stop_reason == StopReason::Error
        && let Some(error) = &message.error_message
        && !matches(&NON_OVERFLOW, error)
    {
        if matches(&OVERFLOW, error) {
            return true;
        }
        if message.provider.as_str() == "cerebras" && matches(&CEREBRAS, error) {
            return true;
        }
    }
    let Some(context_window) = context_window else {
        return false;
    };
    // Silent overflow: accepted request whose usage exceeds the window.
    if message.stop_reason == StopReason::Stop {
        let input_tokens = message.usage.input + message.usage.cache_read;
        if input_tokens > context_window {
            return true;
        }
    }
    // Length-stop overflow: the server truncated input to fill the window,
    // leaving no room for output.
    if message.stop_reason == StopReason::Length && message.usage.output == 0 {
        let input_tokens = message.usage.input + message.usage.cache_read;
        if (input_tokens as f64) >= (context_window as f64) * 0.99 {
            return true;
        }
    }
    false
}

/// `isRecoverableLength`: a length stop below the caller's or model's
/// intended output limit — eligible for one bounded compact-and-retry.
#[must_use]
pub fn is_recoverable_length(message: &AssistantMessage, desired_max_output: u64) -> bool {
    message.stop_reason == StopReason::Length
        && desired_max_output > 0
        && message.usage.output < desired_max_output
}

/// `getOverflowPatterns` for testing: the verbatim JS fragment strings.
#[must_use]
pub const fn overflow_patterns() -> &'static [&'static str] {
    OVERFLOW_PATTERNS
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_pattern_compiles() {
        assert!(OVERFLOW.is_some(), "overflow patterns must compile");
        assert!(NON_OVERFLOW.is_some(), "non-overflow patterns must compile");
        assert!(CEREBRAS.is_some(), "cerebras pattern must compile");
    }
}
