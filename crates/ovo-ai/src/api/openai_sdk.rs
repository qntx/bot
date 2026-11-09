//! The `OpenAI` SDK behavior both `OpenAI` protocols share: credential and
//! URL resolution, the request header merge order, and how the SDK turns
//! HTTP failures and streamed records into errors.

use serde_json::{Map, Value};

use crate::options::StreamOptions;
use crate::types::message::Message;
use crate::types::model::Model;
use crate::types::tool::ToolChoice;

/// `sessionAffinityFormat`: which headers carry `sessionId`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SessionAffinityFormat {
    /// `session_id` plus the request id headers.
    Openai,
    /// The request id headers only.
    OpenaiNoSession,
    /// `x-session-id`.
    Openrouter,
}

impl SessionAffinityFormat {
    /// A `model.compat` value; unknown values behave like `openai`.
    #[must_use]
    pub fn parse(value: &str) -> Self {
        match value {
            "openrouter" => Self::Openrouter,
            "openai-nosession" => Self::OpenaiNoSession,
            _ => Self::Openai,
        }
    }
}

/// A simple tool choice as the `tool_choice` request value.
#[must_use]
pub fn tool_choice_value(choice: ToolChoice) -> Value {
    Value::from(match choice {
        ToolChoice::Auto => "auto",
        ToolChoice::None => "none",
    })
}

/// `getClientApiKey`: the API key, or `"unused"` when a request header
/// carries the credential.
///
/// # Errors
/// Returns upstream's message when neither is present.
pub fn client_api_key(provider: &str, options: &StreamOptions) -> Result<String, String> {
    if let Some(key) = options
        .api_key
        .as_ref()
        .map(secrecy::ExposeSecret::expose_secret)
        .filter(|key| !key.is_empty())
    {
        return Ok(key.to_owned());
    }
    let has_header = |expected: &str| {
        options.headers.iter().flatten().any(|(name, value)| {
            name.eq_ignore_ascii_case(expected)
                && value
                    .as_str()
                    .is_some_and(|value| !crate::text::js_trim(value).is_empty())
        })
    };
    if has_header("authorization") || has_header("cf-aig-authorization") {
        return Ok(String::from("unused"));
    }
    Err(format!("No API key for provider: {provider}"))
}

/// The SDK's `buildURL` for `path` under `base_url`, normalized like
/// `new URL`.
///
/// # Errors
/// Returns `"Invalid URL"` when the joined URL does not parse.
pub fn endpoint_url(base_url: &str, path: &str) -> Result<String, String> {
    let separator = if base_url.ends_with('/') { "" } else { "/" };
    url::Url::parse(&format!("{base_url}{separator}{path}"))
        .map(String::from)
        .map_err(|_| String::from("Invalid URL"))
}

/// The request headers in the order the SDK applies them.
///
/// Accept and bearer auth, then pi's `defaultHeaders` (`createClient`: identity,
/// `model.headers`, GitHub Copilot's per-request headers, the adapter's
/// session `affinity` headers, the request headers), then the body content
/// type. Later sources win
/// case-insensitively and `null` removes a header.
#[must_use]
pub fn request_headers(
    model: &Model,
    messages: &[Message],
    options: &StreamOptions,
    api_key: &str,
    affinity: &Map<String, Value>,
) -> Map<String, Value> {
    let mut sdk = Map::new();
    sdk.insert("Accept".into(), Value::from("application/json"));
    sdk.insert(
        "Authorization".into(),
        Value::from(format!("Bearer {api_key}")),
    );
    let target = crate::identity::ClientIdentityTarget {
        provider: model.provider.as_str().to_owned(),
        base_url: model.base_url.clone(),
    };
    let identity: Map<String, Value> = crate::identity::get_identity_headers(
        &target,
        options.identity.as_ref(),
        crate::identity::OVO_PRODUCT,
        crate::identity::OVO_VERSION,
    )
    .into_iter()
    .map(|(name, value)| (name, Value::from(value)))
    .collect();
    let copilot = (model.provider.as_str() == "github-copilot")
        .then(|| crate::api::github_copilot::dynamic_headers(messages));
    let mut body = Map::new();
    body.insert("Content-Type".into(), Value::from("application/json"));
    crate::http::headers::provider_headers_to_record(&[
        Some(&sdk),
        Some(&identity),
        model.headers.as_ref(),
        copilot.as_ref(),
        Some(affinity),
        options.headers.as_ref(),
        Some(&body),
    ])
    .unwrap_or_default()
}

#[cfg(feature = "http")]
pub use stream::{SdkError, SdkRecord, read_record, status_error};

#[cfg(feature = "http")]
mod stream {
    use serde_json::Value;

    use crate::http::client::HttpError;
    use crate::http::error_body::{NormalizedProviderError, normalize_provider_error};
    use crate::http::sse::SseEvent;
    use crate::json::is_truthy;

    /// An error as the SDK throws it: the `APIError` message with its
    /// `status` and `error` fields, or a plain error's message.
    #[derive(Clone, Debug, PartialEq)]
    pub struct SdkError {
        /// `error.message`.
        pub message: String,
        /// `APIError.status`.
        pub status: Option<f64>,
        /// `APIError.error`: the provider's error object.
        pub error: Option<Value>,
    }

    impl SdkError {
        fn plain(message: &str) -> Self {
            Self {
                message: message.to_owned(),
                status: None,
                error: None,
            }
        }

        /// `new APIError(status, error, message)`.
        fn api(status: Option<f64>, error: Option<&Value>, message: Option<&str>) -> Self {
            let error = error.filter(|error| is_truthy(error));
            Self {
                message: api_error_message(status, error, message),
                status,
                error: error.cloned(),
            }
        }

        /// `normalizeProviderError` of this error.
        #[must_use]
        pub fn normalized(&self) -> NormalizedProviderError {
            let fields = serde_json::json!({
                "status": self.status,
                "error": self.error.clone().unwrap_or(Value::Null),
            });
            normalize_provider_error(&self.message, Some(&fields))
        }

        /// `String(error.error.metadata.raw)` when it is truthy: extra detail
        /// providers behind `OpenRouter` attach.
        #[must_use]
        pub fn raw_metadata(&self) -> Option<String> {
            self.error
                .as_ref()
                .and_then(|error| error.get("metadata"))
                .and_then(|metadata| metadata.get("raw"))
                .filter(|raw| is_truthy(raw))
                .map(js_string)
        }
    }

    /// The error the SDK throws for a failed request: `makeStatusError` from
    /// the response body, or its connection error when no response arrived
    /// (timeouts are not told apart from it).
    #[must_use]
    pub fn status_error(error: &HttpError) -> SdkError {
        let Some(status) = error.status.filter(|status| *status != 0.0) else {
            return SdkError::plain("Connection error.");
        };
        let text = error.body.as_deref().unwrap_or_default();
        crate::json::from_str::<Value>(text)
            .ok()
            .filter(is_truthy)
            .map_or_else(
                || SdkError::api(Some(status), None, Some(text)),
                |json| {
                    // A JSON body without an `error` field is the error itself.
                    let field = match json.get("error") {
                        None | Some(Value::Null) if json.is_object() || json.is_array() => {
                            Some(&json)
                        }
                        field => field,
                    };
                    SdkError::api(Some(status), field, None)
                },
            )
    }

    /// One SSE record as the SDK's stream iteration sees it.
    #[derive(Clone, Debug, PartialEq, Eq)]
    pub enum SdkRecord {
        /// `[DONE]`: the iteration ends.
        Done,
        /// A record the SDK yields as an `{event, data}` envelope, which
        /// carries no payload the adapters read.
        Skip,
        /// The parsed payload.
        Data(Value),
    }

    /// `Stream.fromSSEResponse` for one record.
    ///
    /// # Errors
    /// Returns the SDK's error for malformed JSON, an `error` event, or a
    /// payload with an `error` field.
    pub fn read_record(sse: &SseEvent) -> Result<SdkRecord, SdkError> {
        if sse.data == "[DONE]" {
            return Ok(SdkRecord::Done);
        }
        let data: Value = crate::json::from_str(&sse.data).map_err(|_| {
            SdkError::plain("Error reading response: malformed server-sent event JSON.")
        })?;
        let event = sse.event.as_deref();
        if event.is_some_and(|event| event.starts_with("thread.")) {
            return Ok(SdkRecord::Skip);
        }
        if event == Some("error") {
            let error = data
                .get("error")
                .filter(|error| !error.is_null())
                .unwrap_or(&data);
            return Err(SdkError::api(None, Some(error), None));
        }
        if let Some(error) = data.get("error").filter(|error| is_truthy(error)) {
            return Err(SdkError::api(None, Some(error), None));
        }
        Ok(SdkRecord::Data(data))
    }

    /// `APIError.makeMessage`.
    fn api_error_message(
        status: Option<f64>,
        error: Option<&Value>,
        message: Option<&str>,
    ) -> String {
        let msg = error.map_or_else(
            || message.unwrap_or_default().to_owned(),
            |error| match error.get("message").filter(|m| is_truthy(m)) {
                Some(Value::String(text)) => text.clone(),
                Some(other) => crate::json::to_string(other).unwrap_or_default(),
                None => crate::json::to_string(error).unwrap_or_default(),
            },
        );
        match (status, msg.is_empty()) {
            (Some(status), false) => format!("{status} {msg}"),
            (Some(status), true) => format!("{status} status code (no body)"),
            (None, false) => msg,
            (None, true) => String::from("(no status code or body)"),
        }
    }

    /// JavaScript `String(value)` for a JSON value.
    fn js_string(value: &Value) -> String {
        match value {
            Value::String(text) => text.clone(),
            Value::Object(_) => String::from("[object Object]"),
            Value::Array(items) => items
                .iter()
                .map(|item| {
                    if item.is_null() {
                        String::new()
                    } else {
                        js_string(item)
                    }
                })
                .collect::<Vec<_>>()
                .join(","),
            other => crate::json::to_string(other).unwrap_or_default(),
        }
    }
}
