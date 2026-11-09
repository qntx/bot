//! `provider-retry.ts` and the Stainless SDK retry policies the
//! providers wrap (`@anthropic-ai/sdk`, `openai`). Pure decisions —
//! `sleep` and the request loop live with the transport.

use crate::http::headers::HeaderRecord;

/// `DEFAULT_MAX_RETRY_DELAY_MS`: upstream's server-delay cap.
pub const DEFAULT_MAX_RETRY_DELAY_MS: f64 = 60_000.0;

/// `RetryDecision`: what the retry policy does with a failed request.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RetryDecision {
    /// Retry after `delay_ms`.
    Retry {
        /// Milliseconds to wait before the next attempt.
        delay_ms: f64,
    },
    /// Give up — the error is not retryable or the budget ran out.
    GiveUp,
    /// `validateServerRetryDelayMs` overflow: the server asked for a
    /// delay above `maxRetryDelayMs` — upstream throws so a higher layer
    /// fails over instead of sleeping.
    FailOverCap {
        /// The delay the server requested.
        requested_ms: f64,
    },
}

/// `isRetryableProviderError` / the SDKs' `shouldRetry`.
///
/// `x-should-retry` header wins, then 408/409/429/5xx; an absent
/// status (connection failure) is retryable for pi's wrapper — the
/// SDK `shouldRetry` only runs on responses, so `None` maps to the pi
/// path.
#[must_use]
pub fn is_retryable_status(status: Option<u32>, headers: &HeaderRecord) -> bool {
    match headers.get("x-should-retry") {
        Some("true") => return true,
        Some("false") => return false,
        _ => {}
    }
    match status {
        None | Some(408 | 409 | 429) => true,
        Some(s) => s >= 500,
    }
}

/// `retry-after-ms`, `retry-after` (seconds or HTTP-date) and the
/// exponential default — shared header parsing across all three
/// policies. `now_ms` stands in for `Date.now()`.
fn server_delay_ms(headers: &HeaderRecord, now_ms: f64) -> Option<f64> {
    if let Some(v) = headers.get("retry-after-ms")
        && let Ok(n) = v.parse::<f64>()
        && n.is_finite()
    {
        return Some(n);
    }
    if let Some(v) = headers.get("retry-after") {
        if let Ok(n) = v.parse::<f64>() {
            return Some(n * 1000.0);
        }
        let parsed = httpdate_ms(v);
        if let Some(date) = parsed {
            return Some(date - now_ms);
        }
    }
    None
}

/// `Date.parse` for an HTTP-date — RFC 1123 `day, DD mon YYYY HH:MM:SS GMT`.
fn httpdate_ms(text: &str) -> Option<f64> {
    let t = text.trim();
    // RFC 1123: "Sun, 06 Nov 1994 08:49:37 GMT"
    let after_comma = t.split_once(',')?.1.trim();
    let mut parts = after_comma.split_whitespace();
    let day: i64 = parts.next()?.parse().ok()?;
    let month_name = parts.next()?;
    let month = i64::try_from(
        [
            "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
        ]
        .iter()
        .position(|m| *m == month_name)?,
    )
    .ok()?;
    let year: i64 = parts.next()?.parse().ok()?;
    let mut hms = parts.next()?.split(':');
    let h: i64 = hms.next()?.parse().ok()?;
    let min: i64 = hms.next()?.parse().ok()?;
    let s: i64 = hms.next()?.parse().ok()?;
    let days = days_from_civil(year, month + 1, day);
    Some(((days * 86_400) + h * 3600 + min * 60 + s) as f64 * 1000.0)
}

const fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let mp = (m + 9) % 12;
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

/// Exponential backoff with jitter:
/// `min(0.5 * 2^retry_index, 8) * (1 - rand * 0.25)` seconds → ms.
/// `rand` injects `Math.random`.
#[expect(
    clippy::suboptimal_flops,
    reason = "upstream computes `1 - Math.random() * 0.25` in this exact order"
)]
fn default_backoff_ms(retry_index: u32, rand: &dyn Fn() -> f64) -> f64 {
    let seconds = (0.5 * 2f64.powi(retry_index.cast_signed())).min(8.0);
    seconds * (1.0 - rand() * 0.25) * 1000.0
}

/// `getRetryDelayMs` of pi's `provider-retry.ts`.
///
/// Server delay wins (any finite value, capped by
/// `validateServerRetryDelayMs` — `maxRetryDelayMs` default `60_000`,
/// 0 disables); else exponential. `retry_index` is
/// `maxRetries - retriesRemaining`.
#[expect(
    clippy::too_many_arguments,
    reason = "the decision carries the full upstream option set"
)]
#[must_use]
pub fn provider_retry_decision(
    status: Option<u32>,
    headers: &HeaderRecord,
    retry_index: u32,
    retries_remaining: u32,
    no_retry_statuses: &[u32],
    max_retry_delay_ms: Option<f64>,
    now_ms: f64,
    rand: &dyn Fn() -> f64,
) -> RetryDecision {
    if retries_remaining == 0 || !is_retryable_status(status, headers) {
        return RetryDecision::GiveUp;
    }
    if status.is_some_and(|s| no_retry_statuses.contains(&s)) {
        return RetryDecision::GiveUp;
    }
    if let Some(delay) = server_delay_ms(headers, now_ms) {
        let cap = max_retry_delay_ms.unwrap_or(DEFAULT_MAX_RETRY_DELAY_MS);
        if cap > 0.0 && delay > cap {
            return RetryDecision::FailOverCap {
                requested_ms: delay,
            };
        }
        return RetryDecision::Retry { delay_ms: delay };
    }
    RetryDecision::Retry {
        delay_ms: default_backoff_ms(retry_index, rand),
    }
}

/// Anthropic SDK `retryRequest`: server delay accepted only when
/// `> 0 && <= 2^31-1`, else the exponential default.
#[must_use]
pub fn anthropic_retry_decision(
    status: Option<u32>,
    headers: &HeaderRecord,
    retries_remaining: u32,
    max_retries: u32,
    now_ms: f64,
    rand: &dyn Fn() -> f64,
) -> RetryDecision {
    if retries_remaining == 0 || status.is_none() || !is_retryable_status(status, headers) {
        return RetryDecision::GiveUp;
    }
    if let Some(delay) = server_delay_ms(headers, now_ms)
        && delay > 0.0
        && delay <= (1u64 << 31) as f64 - 1.0
    {
        return RetryDecision::Retry { delay_ms: delay };
    }
    RetryDecision::Retry {
        delay_ms: default_backoff_ms(max_retries - retries_remaining, rand),
    }
}

/// `OpenAI` SDK `retryRequest`: server delay accepted only when finite,
/// `>= 0` and `<= 60_000`, else the exponential default.
#[must_use]
pub fn openai_retry_decision(
    status: Option<u32>,
    headers: &HeaderRecord,
    retries_remaining: u32,
    max_retries: u32,
    now_ms: f64,
    rand: &dyn Fn() -> f64,
) -> RetryDecision {
    if retries_remaining == 0 || status.is_none() || !is_retryable_status(status, headers) {
        return RetryDecision::GiveUp;
    }
    if let Some(delay) = server_delay_ms(headers, now_ms)
        && delay.is_finite()
        && (0.0..=60_000.0).contains(&delay)
    {
        return RetryDecision::Retry { delay_ms: delay };
    }
    RetryDecision::Retry {
        delay_ms: default_backoff_ms(max_retries - retries_remaining, rand),
    }
}

/// Uniform `[0, 1)` jitter source for retry backoff (`Math.random` in
/// upstream); not cryptographic.
#[must_use]
pub fn jitter() -> f64 {
    use std::hash::{BuildHasher, RandomState};
    let bits = RandomState::new().hash_one(std::time::Instant::now());
    // The top 53 bits make an exactly representable fraction.
    #[expect(
        clippy::cast_precision_loss,
        reason = "53-bit integer fits f64 exactly"
    )]
    let fraction = (bits >> 11) as f64 / (1u64 << 53) as f64;
    fraction
}
