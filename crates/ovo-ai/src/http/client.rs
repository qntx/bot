//! `HttpClient`: pooled `reqwest` clients (rustls with platform roots, HTTP/2
//! negotiated by ALPN) shared by every adapter.
//!
//! `post_json` sends `ovo_ai::json` bytes, applies the provider retry
//! decisions, and aborts the in-flight request and the backoff sleep on the
//! `CancellationToken`.

use std::collections::HashMap;
use std::sync::{Mutex, PoisonError};
use std::time::Duration;

use bytes::Bytes;
use reqwest::header::{CONTENT_TYPE, HeaderMap, HeaderName, HeaderValue};
use serde_json::{Map, Value};
use tokio_util::sync::CancellationToken;

use crate::http::error_body::{format_provider_error, normalize_provider_error};
use crate::http::headers::HeaderRecord;
use crate::http::proxy::resolve_http_proxy_url_for_target;
use crate::http::retry::{RetryDecision, provider_retry_decision};

/// Transport settings shared by every request.
#[derive(Clone, Debug, Default)]
pub struct HttpClientConfig {
    /// Merged `ProviderEnv` snapshot used for proxy selection.
    pub env: Option<Map<String, Value>>,
}

/// Per-request policy: upstream's `timeoutMs`, `maxRetries` and
/// `maxRetryDelayMs`.
#[derive(Clone, Copy, Debug)]
pub struct RequestPolicy {
    /// Time allowed until the response headers arrive (SDK default 10 minutes).
    /// The streamed body is not limited by it.
    pub timeout: Duration,
    /// Retries after the first attempt (`retryProviderRequest` default 0).
    pub max_retries: u32,
    /// Cap on server-requested retry delays; `0` disables the cap.
    pub max_retry_delay_ms: Option<f64>,
}

impl Default for RequestPolicy {
    fn default() -> Self {
        Self {
            timeout: Duration::from_secs(600),
            max_retries: 0,
            max_retry_delay_ms: None,
        }
    }
}

/// Pooled HTTP clients, one per proxy decision, so connections are reused
/// across requests.
#[derive(Debug, Default)]
pub struct HttpClient {
    config: HttpClientConfig,
    clients: Mutex<HashMap<Option<String>, reqwest::Client>>,
}

/// A normalized provider HTTP error.
#[derive(Debug)]
pub struct HttpError {
    /// The model-facing message (`formatProviderError` output).
    pub message: String,
    /// HTTP status when a response arrived.
    pub status: Option<f64>,
    /// The raw response body, for SDK-family specific error rendering.
    pub body: Option<String>,
}

impl std::fmt::Display for HttpError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for HttpError {}

/// The error `post_json` surfaces.
#[derive(Debug, thiserror::Error)]
pub enum SendError {
    /// Normalized provider error (status and body rendered).
    #[error("{0}")]
    Provider(HttpError),
    /// Transport failure before a response.
    #[error("{0}")]
    Transport(String),
    /// The `CancellationToken` fired (in flight or during backoff).
    #[error("aborted")]
    Aborted,
}

/// What the retry loop does after a failed attempt.
enum Outcome {
    /// Give up or the cap was exceeded: normalize the response.
    Error,
    /// Sleep this many milliseconds, then send the next attempt.
    Wait(f64),
}

/// One attempt's result for the retry decision: status (`None` for a
/// timeout or connect failure), headers, and the response itself.
type Attempt = (Option<u32>, HeaderRecord, Option<reqwest::Response>);

impl HttpClient {
    /// A client pool using `config` for every request.
    #[must_use]
    pub fn new(config: HttpClientConfig) -> Self {
        Self {
            config,
            clients: Mutex::new(HashMap::new()),
        }
    }

    /// POST `body` (built with `crate::json::to_vec`) with `headers`.
    ///
    /// Retries per the provider retry policy (`retry-after(-ms)` delays,
    /// jittered backoff from `rand`, the `maxRetryDelayMs` cap,
    /// `x-should-retry` and the retryable status set) and races the
    /// in-flight request and the backoff sleep against `token`. Returns the
    /// streaming response on success.
    ///
    /// # Errors
    /// [`SendError::Provider`] for a normalized error response,
    /// [`SendError::Transport`] for a transport failure, and
    /// [`SendError::Aborted`] on cancellation.
    pub async fn post_json(
        &self,
        url: &str,
        body: &Bytes,
        headers: &[(&str, &str)],
        policy: &RequestPolicy,
        token: &CancellationToken,
        rand: &(dyn Fn() -> f64 + Send + Sync),
    ) -> Result<reqwest::Response, SendError> {
        let client = self.client_for(url)?;
        let mut retries_remaining = policy.max_retries;
        let mut attempt = 0u32;
        loop {
            let (status, response_headers, response) =
                send_once(&client, url, body, headers, policy, token).await?;
            let response = match response {
                Some(response) if response.status().is_success() => return Ok(response),
                response => response,
            };
            let outcome = decide(
                status,
                &response_headers,
                attempt,
                retries_remaining,
                policy,
                rand,
            );
            match outcome {
                Outcome::Error => return Err(SendError::Provider(finish_error(response).await)),
                Outcome::Wait(delay_ms) => {
                    drop(response);
                    retries_remaining = retries_remaining.saturating_sub(1);
                    backoff(delay_ms, token).await?;
                }
            }
            attempt += 1;
        }
    }

    /// The pooled client for `url`'s proxy decision, built on first use.
    fn client_for(&self, url: &str) -> Result<reqwest::Client, SendError> {
        let proxy = match &self.config.env {
            Some(env) => resolve_http_proxy_url_for_target(url, env)
                .map_err(|error| SendError::Transport(error.message))?
                .map(|proxy| proxy.to_string()),
            None => None,
        };
        let mut clients = self.clients.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(client) = clients.get(&proxy) {
            return Ok(client.clone());
        }
        let mut builder = reqwest::Client::builder();
        if let Some(proxy) = &proxy {
            let proxy = reqwest::Proxy::all(proxy.as_str())
                .map_err(|error| SendError::Transport(error.to_string()))?;
            builder = builder.proxy(proxy);
        }
        let client = builder
            .build()
            .map_err(|error| SendError::Transport(error.to_string()))?;
        clients.insert(proxy, client.clone());
        drop(clients);
        Ok(client)
    }
}

/// The provider retry decision for one failed attempt.
fn decide(
    status: Option<u32>,
    headers: &HeaderRecord,
    attempt: u32,
    retries_remaining: u32,
    policy: &RequestPolicy,
    rand: &(dyn Fn() -> f64 + Send + Sync),
) -> Outcome {
    match provider_retry_decision(
        status,
        headers,
        attempt,
        retries_remaining,
        &[],
        policy.max_retry_delay_ms,
        now_ms(),
        rand,
    ) {
        RetryDecision::Retry { delay_ms } => Outcome::Wait(delay_ms),
        RetryDecision::GiveUp | RetryDecision::FailOverCap { .. } => Outcome::Error,
    }
}

/// One send, cancelled by `token` and limited by `policy.timeout` until the
/// headers arrive. Timeouts and connect failures report no status so the
/// retry decision treats them as retryable.
async fn send_once(
    client: &reqwest::Client,
    url: &str,
    body: &Bytes,
    headers: &[(&str, &str)],
    policy: &RequestPolicy,
    token: &CancellationToken,
) -> Result<Attempt, SendError> {
    let send = tokio::time::timeout(policy.timeout, request(client, url, body, headers).send());
    let outcome = tokio::select! {
        outcome = send => outcome,
        () = token.cancelled() => return Err(SendError::Aborted),
    };
    match outcome {
        Ok(Ok(response)) => Ok((
            Some(u32::from(response.status().as_u16())),
            header_record(response.headers()),
            Some(response),
        )),
        Err(_elapsed) => Ok((None, HeaderRecord::default(), None)),
        Ok(Err(error)) if error.is_timeout() || error.is_connect() => {
            Ok((None, HeaderRecord::default(), None))
        }
        Ok(Err(error)) => Err(SendError::Transport(error.to_string())),
    }
}

/// The backoff sleep, cancelled by `token`.
async fn backoff(delay_ms: f64, token: &CancellationToken) -> Result<(), SendError> {
    let delay = delay_ms.max(0.0);
    if !delay.is_finite() {
        return Ok(());
    }
    tokio::select! {
        () = tokio::time::sleep(Duration::from_secs_f64(delay / 1000.0)) => Ok(()),
        () = token.cancelled() => Err(SendError::Aborted),
    }
}

/// The POST with the caller's headers and a JSON content type unless the
/// caller set one.
fn request(
    client: &reqwest::Client,
    url: &str,
    body: &Bytes,
    headers: &[(&str, &str)],
) -> reqwest::RequestBuilder {
    let mut map = HeaderMap::new();
    for (name, value) in headers {
        if let (Ok(name), Ok(value)) = (
            HeaderName::from_bytes(name.as_bytes()),
            HeaderValue::from_str(value),
        ) {
            map.insert(name, value);
        }
    }
    if !map.contains_key(CONTENT_TYPE) {
        map.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
    }
    client.post(url).headers(map).body(body.clone())
}

fn header_record(headers: &HeaderMap) -> HeaderRecord {
    let map: Map<String, Value> = headers
        .iter()
        .filter_map(|(name, value)| {
            value
                .to_str()
                .ok()
                .map(|v| (name.as_str().to_owned(), Value::from(v)))
        })
        .collect();
    HeaderRecord::from_record(&map)
}

#[expect(
    clippy::cast_precision_loss,
    reason = "milliseconds since the epoch stay below 2^53"
)]
fn now_ms() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|elapsed| elapsed.as_millis() as f64)
        .unwrap_or_default()
}

/// Normalize the final error response like upstream `error-body.ts`, or a
/// bare transport message when no response arrived.
async fn finish_error(response: Option<reqwest::Response>) -> HttpError {
    let Some(response) = response else {
        return HttpError {
            message: String::from("request failed"),
            status: None,
            body: None,
        };
    };
    let status = response.status().as_u16();
    let body = response.text().await.unwrap_or_default();
    let fields = serde_json::json!({ "status": status, "body": body });
    let normalized = normalize_provider_error("", Some(&fields));
    HttpError {
        message: format_provider_error(&normalized, None),
        status: Some(f64::from(status)),
        body: Some(body),
    }
}
