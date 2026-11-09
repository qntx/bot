//! What every HTTP adapter's `stream` shares: the initial `output`
//! message, the request policy, the `onResponse` call and the terminal
//! event.

use serde_json::{Map, Value};

use crate::http::client::RequestPolicy;
use crate::options::{ResponseMetadata, StreamOptions};
use crate::stream::accumulator::wall_ms;
use crate::stream::{AssistantStreamSender, DoneReason, ErrorReason, Metadata};
use crate::types::content::StopReason;
use crate::types::message::AssistantMessage;
use crate::types::model::Model;
use crate::types::usage::Usage;

/// Bounded capacity of an adapter's event channel.
pub(super) const STREAM_CAPACITY: usize = 64;

/// The `output` literal upstream adapters start every response from.
pub(super) fn initial_message(model: &Model) -> AssistantMessage {
    AssistantMessage {
        content: Vec::new(),
        api: model.api.clone(),
        provider: model.provider.clone(),
        model: model.id.clone(),
        usage: Usage::default(),
        stop_reason: StopReason::Pending,
        timestamp: wall_ms(),
        provider_thinking_level: None,
        response_id: None,
        response_model: None,
        raw_stop_reason: None,
        error_message: None,
        diagnostics: None,
        deferred: None,
        end_turn: None,
        duration_ms: None,
        thinking_level: None,
    }
}

/// The request policy from upstream's `timeoutMs`, `maxRetries` and
/// `maxRetryDelayMs` options; absent ones keep the defaults.
pub(super) fn request_policy(options: &StreamOptions) -> RequestPolicy {
    let defaults = RequestPolicy::default();
    RequestPolicy {
        timeout: options
            .timeout_ms
            .map_or(defaults.timeout, std::time::Duration::from_millis),
        max_retries: options.max_retries.unwrap_or(defaults.max_retries),
        max_retry_delay_ms: options.max_retry_delay_ms,
    }
}

/// String-valued request headers as name/value pairs.
pub(super) fn header_pairs(headers: &Map<String, Value>) -> Vec<(&str, &str)> {
    headers
        .iter()
        .filter_map(|(name, value)| value.as_str().map(|v| (name.as_str(), v)))
        .collect()
}

/// `options.onResponse` with the status and lowercase response headers.
pub(super) async fn notify_response(
    options: &StreamOptions,
    response: &reqwest::Response,
    model: &Model,
) {
    if let Some(on_response) = &options.on_response {
        let metadata = ResponseMetadata {
            status: response.status().as_u16(),
            headers: headers_to_record(response.headers()),
        };
        on_response(&metadata, model).await;
    }
}

/// `headersToRecord`: response headers with lowercase names.
fn headers_to_record(headers: &reqwest::header::HeaderMap) -> Map<String, Value> {
    headers
        .iter()
        .filter_map(|(name, value)| {
            value
                .to_str()
                .ok()
                .map(|v| (name.as_str().to_owned(), Value::from(v)))
        })
        .collect()
}

/// Emit the terminal event for `outcome` (upstream's `done` or the `catch`
/// branch: aborted when the signal fired, error otherwise).
pub(super) async fn finish(
    sender: AssistantStreamSender,
    outcome: Result<DoneReason, String>,
    aborted: bool,
) {
    let sent = match outcome {
        Ok(reason) => sender.done(reason).await,
        Err(message) => {
            let reason = if aborted {
                ErrorReason::Aborted
            } else {
                ErrorReason::Error
            };
            sender.error(reason, message).await
        }
    };
    // A closed stream means the consumer dropped it; nothing is left to notify.
    sent.ok();
}

/// Publish one metadata update.
pub(super) async fn send(
    sender: &mut AssistantStreamSender,
    metadata: Metadata,
) -> Result<(), String> {
    sender.metadata(metadata).await.map_err(|_| closed())
}

/// The consumer is gone; the producer stops.
pub(super) fn closed() -> String {
    String::from("stream consumer closed")
}

/// How `onPayload`'s replacement is folded into the request body.
pub(super) enum PayloadMerge {
    /// `{ ...nextParams, stream: true }` — the Anthropic merge.
    Stream,
    /// `params = nextParams` — the `OpenAI` adapters replace entirely.
    Replace,
}

/// `onPayload` at upstream's call site: `await onPayload(params,
/// model)`; `undefined` keeps the built body.
pub(super) async fn on_payload_body(
    base: &StreamOptions,
    params: &Value,
    body: &[u8],
    model: &Model,
    merge: PayloadMerge,
) -> Result<Vec<u8>, String> {
    let Some(hook) = &base.on_payload else {
        return Ok(body.to_vec());
    };
    let Some(next) = hook(params, model).await else {
        return Ok(body.to_vec());
    };
    let value = match (merge, next) {
        // A non-object spreads to nothing — `{ stream: true }`.
        (PayloadMerge::Stream, Value::Object(mut map)) => {
            map.insert("stream".into(), Value::Bool(true));
            Value::Object(map)
        }
        (PayloadMerge::Stream, _) => serde_json::json!({"stream": true}),
        (PayloadMerge::Replace, value) => value,
    };
    crate::json::to_vec(&value).map_err(|error| error.to_string())
}

/// `onProviderStreamEvent` at upstream's call site: awaited for every
/// SDK-yielded event before the adapter folds it.
pub(super) async fn on_provider_stream_event(base: &StreamOptions, event: &Value, model: &Model) {
    if let Some(hook) = &base.on_provider_stream_event {
        hook(event, model).await;
    }
}
