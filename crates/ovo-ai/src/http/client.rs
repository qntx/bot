//! `HttpClient`: a `reqwest::Client` (rustls + native roots, HTTP/2,
//! pooled) built from an injected config.
//!
//! `post_json` sends `ovo_ai::json` bytes, applies the part-A SDK
//! retry decisions, and aborts the in-flight request and backoff
//! sleep on the `CancellationToken`. Compiled under `http` only.

#[cfg(feature = "http")]
mod imp {
    use std::time::Duration;

    use bytes::Bytes;
    use reqwest::header::{HeaderMap, HeaderName, HeaderValue};
    use serde_json::{Map, Value};
    use tokio_util::sync::CancellationToken;

    use crate::http::error_body::{format_provider_error, normalize_provider_error};
    use crate::http::headers::HeaderRecord;
    use crate::http::proxy::resolve_http_proxy_url_for_target;
    use crate::http::retry::{RetryDecision, provider_retry_decision};

    /// `HttpClient` config — proxy selection stays a *decision* (part
    /// A); `env` is the caller's merged `ProviderEnv` snapshot.
    #[derive(Clone, Debug)]
    pub struct HttpClientConfig {
        /// Whole-request timeout (SDK default: 10 minutes).
        pub timeout: Duration,
        /// Merged `ProviderEnv` for proxy resolution.
        pub env: Option<Map<String, Value>>,
        /// `maxRetries` (SDK default 2).
        pub max_retries: u32,
        /// `maxRetryDelayMs` cap on server delays (0 disables).
        pub max_retry_delay_ms: Option<f64>,
    }

    impl Default for HttpClientConfig {
        fn default() -> Self {
            Self {
                timeout: Duration::from_secs(600),
                env: None,
                max_retries: 2,
                max_retry_delay_ms: None,
            }
        }
    }

    /// `HttpClient` — one pooled reqwest client per configured proxy.
    #[derive(Debug)]
    pub struct HttpClient {
        config: HttpClientConfig,
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
            write!(f, "{}", self.message)
        }
    }

    impl std::error::Error for HttpError {}

    /// The error `post_json` surfaces.
    #[derive(Debug)]
    pub enum SendError {
        /// Normalized provider error (status/body rendered).
        Provider(HttpError),
        /// Transport failure before a response.
        Transport(String),
        /// `CancellationToken` fired (in-flight or during backoff).
        Aborted,
    }

    impl std::fmt::Display for SendError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                Self::Provider(e) => write!(f, "{e}"),
                Self::Transport(m) => write!(f, "{m}"),
                Self::Aborted => write!(f, "aborted"),
            }
        }
    }

    impl std::error::Error for SendError {}

    /// The retry-loop outcome: error out or wait then retry.
    enum Outcome {
        /// Give up or the cap was exceeded — normalize the response.
        Error,
        /// Sleep `delay_ms` then send the next attempt.
        Wait(f64),
    }

    impl HttpClient {
        /// Build the client config; the reqwest client is constructed
        /// per `post_json` target so its proxy decision applies.
        #[must_use]
        pub const fn new(config: HttpClientConfig) -> Self {
            Self { config }
        }

        /// `post_json`: POST `body` (built with `crate::json::to_vec`)
        /// with `headers` merged in order.
        ///
        /// Retries per the part-A `provider_retry_decision` policy —
        /// `retry-after(-ms)` delays, backoff with `rand` jitter, the
        /// `maxRetryDelayMs` cap, `x-should-retry`/status sets — and
        /// races both the in-flight request and the backoff sleep
        /// against `token`. Returns the streaming `reqwest::Response`
        /// on success; errors normalize like `error-body.ts`.
        ///
        /// # Errors
        /// [`SendError::Provider`] on a normalized response,
        /// [`SendError::Transport`] on transport failure,
        /// [`SendError::Aborted`] on cancellation.
        #[expect(
            clippy::excessive_nesting,
            reason = "the retry loop's select!/match arms mirror upstream's sequential structure"
        )]
        pub async fn post_json(
            &self,
            url: &str,
            body: &Bytes,
            headers: &[(&str, &str)],
            token: &CancellationToken,
            rand: &(dyn Fn() -> f64 + Send + Sync),
        ) -> Result<reqwest::Response, SendError> {
            let client = self.build(url)?;
            let mut retries_remaining = self.config.max_retries;
            let mut attempt = 0u32;
            let mut last_response;
            loop {
                let (status, response_headers, response) =
                    self.send_once(&client, url, body, headers, token).await?;
                if response.as_ref().is_some_and(|r| r.status().is_success()) {
                    let Some(response) = response else {
                        return Err(SendError::Transport(String::from("no response")));
                    };
                    return Ok(response);
                }
                last_response = response;
                match self.decide(status, &response_headers, attempt, retries_remaining, rand) {
                    Outcome::Error => {
                        return Err(SendError::Provider(
                            finish_error(last_response.take()).await,
                        ));
                    }
                    Outcome::Wait(delay_ms) => {
                        retries_remaining = retries_remaining.saturating_sub(1);
                        drop(last_response.take());
                        self.backoff(delay_ms, token).await?;
                    }
                }
                attempt += 1;
            }
        }

        /// The part-A retry decision for one response.
        fn decide(
            &self,
            status: Option<u32>,
            headers: &HeaderRecord,
            attempt: u32,
            retries_remaining: u32,
            rand: &(dyn Fn() -> f64 + Send + Sync),
        ) -> Outcome {
            match provider_retry_decision(
                status,
                headers,
                attempt,
                retries_remaining,
                &[],
                self.config.max_retry_delay_ms,
                now_ms(),
                rand,
            ) {
                RetryDecision::Retry { delay_ms } => Outcome::Wait(delay_ms),
                RetryDecision::GiveUp | RetryDecision::FailOverCap { .. } => Outcome::Error,
            }
        }

        /// One send, cancelled by `token`. On a non-success response
        /// returns its status/headers for the retry decision plus the
        /// response for error normalization; `None` status marks a
        /// timeout/connect failure.
        async fn send_once(
            &self,
            client: &reqwest::Client,
            url: &str,
            body: &Bytes,
            headers: &[(&str, &str)],
            token: &CancellationToken,
        ) -> Result<(Option<u32>, HeaderRecord, Option<reqwest::Response>), SendError> {
            let request = request(client, url, body, headers);
            let outcome = tokio::select! {
                r = request.send() => r,
                () = token.cancelled() => return Err(SendError::Aborted),
            };
            match outcome {
                Ok(response) => Ok(response_meta(response)),
                Err(error) if error.is_timeout() || error.is_connect() => {
                    Ok((None, HeaderRecord::default(), None))
                }
                Err(error) => Err(SendError::Transport(error.to_string())),
            }
        }

        /// The backoff sleep, cancelled by `token`.
        async fn backoff(&self, delay_ms: f64, token: &CancellationToken) -> Result<(), SendError> {
            let delay = delay_ms.max(0.0);
            if !delay.is_finite() {
                return Ok(());
            }
            tokio::select! {
                () = tokio::time::sleep(Duration::from_secs_f64(delay / 1000.0)) => Ok(()),
                () = token.cancelled() => Err(SendError::Aborted),
            }
        }

        /// Build the reqwest client for `url`'s proxy decision.
        fn build(&self, url: &str) -> Result<reqwest::Client, SendError> {
            let mut builder = reqwest::Client::builder()
                .timeout(self.config.timeout)
                .http2_prior_knowledge();
            if let Some(env) = &self.config.env {
                builder = apply_proxy(builder, url, env)?;
            }
            builder
                .build()
                .map_err(|e| SendError::Transport(e.to_string()))
        }
    }

    /// Apply the part-A proxy decision for `url` to `builder`.
    fn apply_proxy(
        mut builder: reqwest::ClientBuilder,
        url: &str,
        env: &Map<String, Value>,
    ) -> Result<reqwest::ClientBuilder, SendError> {
        let proxy = match resolve_http_proxy_url_for_target(url, env) {
            Ok(p) => p,
            Err(e) => return Err(SendError::Transport(e.message)),
        };
        if let Some(proxy) = proxy {
            let proxy = reqwest::Proxy::all(proxy.as_str())
                .map_err(|e| SendError::Transport(e.to_string()))?;
            builder = builder.proxy(proxy);
        }
        Ok(builder)
    }

    /// The reqwest POST with merged headers and the JSON bytes.
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
        client
            .post(url)
            .headers(map)
            .header("content-type", "application/json")
            .body(body.clone())
    }

    /// (status, headers, response) for the retry decision.
    fn response_meta(
        response: reqwest::Response,
    ) -> (Option<u32>, HeaderRecord, Option<reqwest::Response>) {
        (
            Some(u32::from(response.status().as_u16())),
            header_record(response.headers()),
            Some(response),
        )
    }

    fn header_record(headers: &HeaderMap) -> HeaderRecord {
        let mut map = Map::new();
        for (name, value) in headers {
            if let Ok(v) = value.to_str() {
                map.insert(name.as_str().to_owned(), Value::String(v.to_owned()));
            }
        }
        HeaderRecord::from_record(&map)
    }

    fn now_ms() -> f64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as f64)
            .unwrap_or_default()
    }

    /// `GiveUp`/`FailOverCap`: normalize the response body like
    /// upstream `error-body.ts`, or a bare transport message.
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
        let norm = normalize_provider_error("", Some(&fields));
        HttpError {
            message: format_provider_error(&norm, None),
            status: Some(f64::from(status)),
            body: Some(body),
        }
    }
}

#[cfg(feature = "http")]
pub use imp::*;
