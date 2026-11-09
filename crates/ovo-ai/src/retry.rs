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
use std::future::Future;

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
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::sync::{Arc, Mutex};

    use tokio_util::sync::CancellationToken;

    use super::*;
    use crate::types::api::{Api, ProviderId};
    use crate::types::content::{AssistantContentBlock, StopReason, TextContent};
    use crate::types::usage::{Usage, UsageCost};

    fn msg(text: &str, stop_reason: StopReason, error_message: Option<&str>) -> AssistantMessage {
        AssistantMessage {
            content: if text.is_empty() {
                Vec::new()
            } else {
                vec![AssistantContentBlock::Text(TextContent {
                    text: text.to_owned(),
                    text_signature: None,
                })]
            },
            api: Api::ANTHROPIC_MESSAGES,
            provider: ProviderId::ANTHROPIC,
            model: String::from("m"),
            usage: Usage {
                input: 0,
                output: 0,
                cache_read: 0,
                cache_write: 0,
                total_tokens: 0,
                cost: UsageCost {
                    input: 0.0,
                    output: 0.0,
                    cache_read: 0.0,
                    cache_write: 0.0,
                    total: 0.0,
                    total_from_provider: None,
                },
                cache_write1h: None,
                reasoning: None,
            },
            stop_reason,
            timestamp: 0,
            provider_thinking_level: None,
            response_id: None,
            response_model: None,
            raw_stop_reason: None,
            error_message: error_message.map(str::to_owned),
            diagnostics: None,
            deferred: None,
            end_turn: None,
            duration_ms: None,
            thinking_level: None,
        }
    }

    fn ok() -> AssistantMessage {
        msg("ok", StopReason::Stop, None)
    }

    fn err() -> AssistantMessage {
        msg("", StopReason::Error, Some("terminated"))
    }

    fn policy(enabled: bool) -> RetryPolicy {
        RetryPolicy {
            enabled,
            max_retries: 3,
            base_delay_ms: 0.0,
            max_agent_delay_ms: None,
        }
    }

    type Calls = Arc<AtomicU32>;
    type Finished = Arc<Mutex<Option<(bool, u32)>>>;
    type Produce = dyn FnMut() -> std::pin::Pin<Box<dyn Future<Output = AssistantMessage>>>;

    fn counting_produce(calls: &Calls, outcome: fn(u32) -> AssistantMessage) -> Box<Produce> {
        let calls = Arc::clone(calls);
        Box::new(move || {
            let calls = Arc::clone(&calls);
            Box::pin(async move { outcome(calls.fetch_add(1, Ordering::Relaxed) + 1) })
        })
    }

    fn err_then_ok(n: u32) -> AssistantMessage {
        if n < 3 {
            err()
        } else {
            msg("recovered", StopReason::Stop, None)
        }
    }

    fn err_then_aborted(n: u32) -> AssistantMessage {
        if n == 1 {
            err()
        } else {
            msg("", StopReason::Aborted, None)
        }
    }

    fn err_until_five(n: u32) -> AssistantMessage {
        if n < 5 { err() } else { ok() }
    }

    fn event_produce(calls: &Calls, events: &Arc<Mutex<Vec<String>>>) -> Box<Produce> {
        let calls = Arc::clone(calls);
        let events = Arc::clone(events);
        Box::new(move || {
            let calls = Arc::clone(&calls);
            let events = Arc::clone(&events);
            Box::pin(async move {
                let n = calls.fetch_add(1, Ordering::Relaxed);
                events
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .push(format!("produce:{n}"));
                err_then_ok(n + 1)
            })
        })
    }

    fn finished_cb() -> (Finished, RetryFinishedCb) {
        let finished: Finished = Arc::new(Mutex::new(None));
        let shared = Arc::clone(&finished);
        (
            finished,
            Box::new(move |success, attempt, _| {
                *shared
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner) = Some((success, attempt));
            }),
        )
    }

    #[test]
    fn every_pattern_compiles() {
        assert!(
            NON_RETRYABLE.is_some(),
            "non-retryable patterns must compile"
        );
        assert!(RETRYABLE.is_some(), "retryable patterns must compile");
    }

    /// Upstream "returns a successful response immediately without retrying".
    #[tokio::test(start_paused = true)]
    async fn success_immediately() {
        let calls: Calls = Arc::new(AtomicU32::new(0));
        let res = retry_assistant_call(
            counting_produce(&calls, |_| ok()),
            Some(&policy(true)),
            &CancellationToken::new(),
            &mut RetryCallbacks::default(),
        )
        .await;
        assert_eq!(res.content.len(), 1);
        assert_eq!(calls.load(Ordering::Relaxed), 1);
    }

    /// Upstream "does not retry an aborted message".
    #[tokio::test(start_paused = true)]
    async fn aborted_never_retried() {
        let calls: Calls = Arc::new(AtomicU32::new(0));
        let scheduled = Arc::new(AtomicU32::new(0));
        let scheduled_ref = Arc::clone(&scheduled);
        let mut callbacks = RetryCallbacks {
            on_retry_scheduled: Some(Box::new(move |_, _, _, _| {
                scheduled_ref.fetch_add(1, Ordering::Relaxed);
            })),
            ..RetryCallbacks::default()
        };
        let res = retry_assistant_call(
            counting_produce(&calls, |_| msg("", StopReason::Aborted, None)),
            Some(&policy(true)),
            &CancellationToken::new(),
            &mut callbacks,
        )
        .await;
        assert_eq!(res.stop_reason, StopReason::Aborted);
        assert_eq!(calls.load(Ordering::Relaxed), 1);
        assert_eq!(scheduled.load(Ordering::Relaxed), 0);
    }

    /// Upstream "does not retry a non-retryable error (quota/billing)".
    #[tokio::test(start_paused = true)]
    async fn non_retryable_error() {
        let calls: Calls = Arc::new(AtomicU32::new(0));
        let res = retry_assistant_call(
            counting_produce(&calls, |_| {
                msg("", StopReason::Error, Some("insufficient_quota"))
            }),
            Some(&policy(true)),
            &CancellationToken::new(),
            &mut RetryCallbacks::default(),
        )
        .await;
        assert_eq!(res.stop_reason, StopReason::Error);
        assert_eq!(calls.load(Ordering::Relaxed), 1);
    }

    /// Upstream "retries a transient error up to maxRetries then returns the final error".
    #[tokio::test(start_paused = true)]
    async fn exhausts_max_retries() {
        let calls: Calls = Arc::new(AtomicU32::new(0));
        let scheduled = Arc::new(AtomicU32::new(0));
        let scheduled_ref = Arc::clone(&scheduled);
        let (finished, on_finished) = finished_cb();
        let mut callbacks = RetryCallbacks {
            on_retry_scheduled: Some(Box::new(move |_, _, _, _| {
                scheduled_ref.fetch_add(1, Ordering::Relaxed);
            })),
            on_retry_finished: Some(on_finished),
            ..RetryCallbacks::default()
        };
        let res = retry_assistant_call(
            counting_produce(&calls, |_| err()),
            Some(&policy(true)),
            &CancellationToken::new(),
            &mut callbacks,
        )
        .await;
        assert_eq!(res.stop_reason, StopReason::Error);
        assert_eq!(calls.load(Ordering::Relaxed), 4); // 1 initial + 3 retries
        assert_eq!(scheduled.load(Ordering::Relaxed), 3);
        assert_eq!(
            *finished
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner),
            Some((false, 3))
        );
    }

    /// Upstream "reports capped retry delays" (regression #8826).
    #[tokio::test(start_paused = true)]
    async fn capped_delays() {
        let delays: Arc<Mutex<Vec<f64>>> = Arc::new(Mutex::new(Vec::new()));
        let calls: Calls = Arc::new(AtomicU32::new(0));
        let delays_ref = Arc::clone(&delays);
        let mut callbacks = RetryCallbacks {
            on_retry_scheduled: Some(Box::new(move |_, _, delay, _| {
                delays_ref
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .push(delay);
            })),
            ..RetryCallbacks::default()
        };
        let policy = RetryPolicy {
            enabled: true,
            max_retries: 4,
            base_delay_ms: 10.0,
            max_agent_delay_ms: Some(15.0),
        };
        retry_assistant_call(
            counting_produce(&calls, err_until_five),
            Some(&policy),
            &CancellationToken::new(),
            &mut callbacks,
        )
        .await;
        assert_eq!(
            *delays
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner),
            vec![10.0, 15.0, 15.0, 15.0]
        );
    }

    /// Upstream "stops retrying once a call succeeds".
    #[tokio::test(start_paused = true)]
    async fn stops_on_success() {
        let calls: Calls = Arc::new(AtomicU32::new(0));
        let (finished, on_finished) = finished_cb();
        let mut callbacks = RetryCallbacks {
            on_retry_finished: Some(on_finished),
            ..RetryCallbacks::default()
        };
        let res = retry_assistant_call(
            counting_produce(&calls, err_then_ok),
            Some(&policy(true)),
            &CancellationToken::new(),
            &mut callbacks,
        )
        .await;
        assert_eq!(res.content.len(), 1);
        assert_eq!(calls.load(Ordering::Relaxed), 3);
        assert_eq!(
            *finished
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner),
            Some((true, 2))
        );
    }

    /// Upstream "reports an aborted retried call as unsuccessful".
    #[tokio::test(start_paused = true)]
    async fn aborted_retried_call() {
        let calls: Calls = Arc::new(AtomicU32::new(0));
        let (finished, on_finished) = finished_cb();
        let mut callbacks = RetryCallbacks {
            on_retry_finished: Some(on_finished),
            ..RetryCallbacks::default()
        };
        let res = retry_assistant_call(
            counting_produce(&calls, err_then_aborted),
            Some(&policy(true)),
            &CancellationToken::new(),
            &mut callbacks,
        )
        .await;
        assert_eq!(res.stop_reason, StopReason::Aborted);
        assert_eq!(calls.load(Ordering::Relaxed), 2);
        assert_eq!(
            *finished
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner),
            Some((false, 1))
        );
    }

    /// Upstream "does not retry when policy is disabled".
    #[tokio::test(start_paused = true)]
    async fn disabled_policy() {
        let calls: Calls = Arc::new(AtomicU32::new(0));
        let res = retry_assistant_call(
            counting_produce(&calls, |_| err()),
            Some(&policy(false)),
            &CancellationToken::new(),
            &mut RetryCallbacks::default(),
        )
        .await;
        assert_eq!(res.stop_reason, StopReason::Error);
        assert_eq!(calls.load(Ordering::Relaxed), 1);
    }

    /// Upstream "emits onRetryAttemptStart after backoff before each retried call".
    #[tokio::test(start_paused = true)]
    async fn attempt_start_order() {
        let events: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        let calls: Calls = Arc::new(AtomicU32::new(0));
        let produce = event_produce(&calls, &events);
        let events_sched = Arc::clone(&events);
        let events_start = Arc::clone(&events);
        let mut callbacks = RetryCallbacks {
            on_retry_scheduled: Some(Box::new(move |attempt, _, _, _| {
                events_sched
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .push(format!("retry:{attempt}"));
            })),
            on_retry_attempt_start: Some(Box::new(move || {
                events_start
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .push("attempt-start".to_owned());
            })),
            ..RetryCallbacks::default()
        };
        retry_assistant_call(
            produce,
            Some(&policy(true)),
            &CancellationToken::new(),
            &mut callbacks,
        )
        .await;
        assert_eq!(
            *events
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner),
            vec![
                "produce:0",
                "retry:1",
                "attempt-start",
                "produce:1",
                "retry:2",
                "attempt-start",
                "produce:2"
            ]
        );
    }

    /// Aborts during the backoff sleep normalize to an `aborted`
    /// [`AssistantMessage`] without `errorMessage` (upstream path).
    #[tokio::test(start_paused = true)]
    async fn abort_during_sleep() {
        let token = CancellationToken::new();
        let (finished, on_finished) = finished_cb();
        let token_ref = token.clone();
        let mut callbacks = RetryCallbacks {
            on_retry_finished: Some(on_finished),
            on_retry_scheduled: Some(Box::new(move |_, _, _, _| {
                token_ref.cancel();
            })),
            ..RetryCallbacks::default()
        };
        let calls: Calls = Arc::new(AtomicU32::new(0));
        // A nonzero delay keeps `sleep` pending so the cancelled branch
        // wins `select!` deterministically (a 0 ms sleep would already be
        // ready and make the branch choice random).
        let policy = RetryPolicy {
            enabled: true,
            max_retries: 3,
            base_delay_ms: 1_000.0,
            max_agent_delay_ms: None,
        };
        let res = retry_assistant_call(
            counting_produce(&calls, |_| err()),
            Some(&policy),
            &token,
            &mut callbacks,
        )
        .await;
        assert_eq!(res.stop_reason, StopReason::Aborted);
        assert!(res.error_message.is_none());
        assert_eq!(
            *finished
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner),
            Some((false, 1))
        );
    }
}

/// `RetryCallbacks`: emitted around each retry, mirroring upstream.
#[derive(Default)]
pub struct RetryCallbacks {
    /// `onRetryScheduled(attempt, maxAttempts, delayMs, errorMessage)` —
    /// before the backoff sleep of each retry attempt (1-indexed).
    pub on_retry_scheduled: Option<RetryScheduledCb>,
    /// `onRetryAttemptStart` — after the sleep, before the retried call.
    pub on_retry_attempt_start: Option<Box<dyn FnMut() + Send>>,
    /// `onRetryFinished(success, attempt, finalError?)` — once, at the end.
    pub on_retry_finished: Option<RetryFinishedCb>,
}

/// `onRetryScheduled` callback type.
pub type RetryScheduledCb = Box<dyn FnMut(u32, u32, f64, &str) + Send>;
/// `onRetryFinished` callback type.
pub type RetryFinishedCb = Box<dyn FnMut(bool, u32, Option<&str>) + Send>;

impl std::fmt::Debug for RetryCallbacks {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RetryCallbacks")
            .field("on_retry_scheduled", &self.on_retry_scheduled.is_some())
            .field(
                "on_retry_attempt_start",
                &self.on_retry_attempt_start.is_some(),
            )
            .field("on_retry_finished", &self.on_retry_finished.is_some())
            .finish()
    }
}

/// `retryAssistantCall` (`utils/retry.ts`): one assistant-producing
/// call with bounded retry on transient errors.
///
/// Aborts are terminal and never retried; an abort during the backoff
/// sleep is normalized to an `aborted` [`AssistantMessage`] without
/// `errorMessage`. `token` stands in for `AbortSignal` (ovo uses
/// `CancellationToken` per ADR-009). When `policy` is `None` or
/// disabled the first response returns unchanged.
pub async fn retry_assistant_call<F, Fut>(
    mut produce: F,
    policy: Option<&RetryPolicy>,
    token: &tokio_util::sync::CancellationToken,
    callbacks: &mut RetryCallbacks,
) -> AssistantMessage
where
    F: FnMut() -> Fut,
    Fut: Future<Output = AssistantMessage>,
{
    let max_attempts = if policy.is_some_and(|p| p.enabled) {
        policy.map_or(0, |p| p.max_retries)
    } else {
        0
    };

    let mut attempt = 0u32;
    let mut last_retry: Option<(u32, String)> = None;
    loop {
        let response = produce().await;

        // Abort: terminal but not successful. Never retry an aborted message.
        if response.stop_reason == StopReason::Aborted {
            if let Some((finished, _)) = last_retry
                && let Some(cb) = callbacks.on_retry_finished.as_deref_mut()
            {
                cb(false, finished, None);
            }
            return response;
        }

        // Success: non-error, non-abort responses return as-is.
        if response.stop_reason != StopReason::Error {
            if let Some((finished, _)) = last_retry
                && let Some(cb) = callbacks.on_retry_finished.as_deref_mut()
            {
                cb(true, finished, None);
            }
            return response;
        }

        // Non-retryable, or budget exhausted: return the final error message.
        if attempt >= max_attempts || !is_retryable_assistant_error(&response) {
            if let Some((finished, _)) = last_retry
                && let Some(cb) = callbacks.on_retry_finished.as_deref_mut()
            {
                cb(false, finished, response.error_message.as_deref());
            }
            return response;
        }

        attempt += 1;
        let error_message = response
            .error_message
            .clone()
            .unwrap_or_else(|| "Unknown error".to_owned());
        last_retry = Some((attempt, error_message.clone()));
        let delay_ms = retry_delay_ms(
            policy.unwrap_or(&RetryPolicy {
                enabled: true,
                max_retries: 0,
                base_delay_ms: 0.0,
                max_agent_delay_ms: None,
            }),
            attempt,
        );
        if let Some(cb) = callbacks.on_retry_scheduled.as_deref_mut() {
            cb(attempt, max_attempts, delay_ms, &error_message);
        }

        // `abortableSleep`: the token races the timer; an abort is
        // normalized to an `aborted` AssistantMessage (no errorMessage).
        tokio::select! {
            () = tokio::time::sleep(std::time::Duration::from_secs_f64(delay_ms / 1000.0)) => {},
            () = token.cancelled() => {
                if let Some(cb) = callbacks.on_retry_finished.as_deref_mut() {
                    cb(false, attempt, Some(&error_message));
                }
                let mut aborted = response;
                aborted.stop_reason = StopReason::Aborted;
                aborted.error_message = None;
                return aborted;
            },
        }
        if let Some(cb) = callbacks.on_retry_attempt_start.as_deref_mut() {
            cb();
        }
    }
}
