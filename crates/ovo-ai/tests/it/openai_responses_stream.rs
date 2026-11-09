//! Replays the openai-responses response vectors: each scripted SSE body
//! (or error response) is served by a local server and streamed through
//! `OpenAiResponsesApi`; events and the final message must equal what the
//! published adapter produced.

use std::sync::Arc;

use futures::StreamExt;
use ovo_ai::api::openai_responses::OpenAiResponsesOptions;
use ovo_ai::api::openai_responses::stream::OpenAiResponsesApi;
use ovo_ai::http::client::{HttpClient, HttpClientConfig};
use ovo_ai::options::{SimpleStreamOptions, StreamOptions};
use ovo_ai::provider::ChatApi;
use ovo_ai::stream::AssistantMessageEvent;
use ovo_ai::transcript::normalize_context;
use ovo_ai::types::context::Context;
use ovo_ai::types::message::Message;
use ovo_ai::types::model::Model;
use serde_json::Value;
use tokio_util::sync::CancellationToken;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

use crate::support::js_value_eq;

fn vectors() -> Value {
    let text = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../vectors/ai/openai-responses-stream.json"
    ));
    serde_json::from_str(text).unwrap_or_else(|e| panic!("vector file: {e}"))
}

/// An event as the generator records it: terminal events keep type and reason.
fn recorded(event: &AssistantMessageEvent) -> Value {
    let value: Value = ovo_ai::json::to_string(event)
        .and_then(|text| ovo_ai::json::from_str(&text))
        .unwrap_or_else(|e| panic!("event: {e}"));
    match event {
        AssistantMessageEvent::Done { .. } | AssistantMessageEvent::Error { .. } => {
            serde_json::json!({ "type": value["type"], "reason": value["reason"] })
        }
        _ => value,
    }
}

async fn serve(case: &Value) -> MockServer {
    let server = MockServer::start().await;
    let response = case["status"].as_u64().map_or_else(
        || {
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(case["sse"].as_str().unwrap_or_default())
        },
        |status| {
            ResponseTemplate::new(u16::try_from(status).unwrap_or(500))
                .insert_header("content-type", "application/json")
                .set_body_string(case["body"].as_str().unwrap_or_default())
        },
    );
    Mock::given(method("POST"))
        .and(path("/responses"))
        .respond_with(response)
        .mount(&server)
        .await;
    server
}

/// engine-error-text: the V8 `JSON.parse` message before an
/// `argumentsError`'s "Received arguments" template is masked.
fn mask_engine_errors(value: &mut Value) {
    match value {
        Value::Object(object) => {
            if let Some(Value::String(error)) = object.get_mut("argumentsError")
                && let Some(at) = error.find("\n\nReceived arguments:\n")
            {
                error.replace_range(..at, "<engine>");
            }
            object.values_mut().for_each(mask_engine_errors);
        }
        Value::Array(items) => items.iter_mut().for_each(mask_engine_errors),
        _ => {}
    }
}

/// The options the generator passes: key, no retries, and the abort setup.
fn stream_options(case: &Value) -> StreamOptions {
    let mut options = StreamOptions {
        api_key: Some(secrecy::SecretString::from(String::from("sk-test"))),
        max_retries: Some(0),
        ..StreamOptions::default()
    };
    let signal = CancellationToken::new();
    match case["abort"].as_str() {
        Some("before") => signal.cancel(),
        Some("on-response") => {
            let token = signal.clone();
            options.on_response = Some(Arc::new(move |_, _| {
                token.cancel();
                Box::pin(async {})
            }));
        }
        _ => {}
    }
    options.signal = Some(signal);
    options
}

/// parity: AI-API-OPENAI-RESPONSES
#[tokio::test]
async fn openai_responses_stream_replays_upstream_events() {
    let vectors = vectors();
    let cases = vectors["cases"].as_array().cloned().unwrap_or_default();
    assert!(!cases.is_empty(), "no cases");
    let api = OpenAiResponsesApi::new(Arc::new(HttpClient::new(HttpClientConfig::default())));
    for case in &cases {
        let name = case["name"].as_str().unwrap_or("?");
        let server = serve(case).await;
        let mut model: Model = serde_json::from_value(case["model"].clone())
            .unwrap_or_else(|e| panic!("{name}: model: {e}"));
        model.base_url = server.uri();
        let context: Context = serde_json::from_value(case["context"].clone())
            .unwrap_or_else(|e| panic!("{name}: context: {e}"));
        let context = normalize_context(&context);
        let mut options = stream_options(case);

        let provider_events = Arc::new(std::sync::Mutex::new(Vec::<Value>::new()));
        {
            let captured = Arc::clone(&provider_events);
            options.on_provider_stream_event = Some(Arc::new(move |data, _model| {
                let captured = Arc::clone(&captured);
                let data = data.clone();
                Box::pin(async move {
                    captured
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner)
                        .push(data);
                })
            }));
        }
        let mut stream = if case["simple"].as_bool() == Some(true) {
            let simple = SimpleStreamOptions {
                base: options,
                reasoning: case["options"].get("reasoning").map(|level| {
                    serde_json::from_value(level.clone())
                        .unwrap_or_else(|e| panic!("{name}: reasoning: {e}"))
                }),
                ..SimpleStreamOptions::default()
            };
            api.stream_simple(&model, context, simple)
        } else {
            let responses = OpenAiResponsesOptions {
                base: options,
                reasoning_effort: case["options"].get("reasoningEffort").map(|level| {
                    serde_json::from_value(level.clone())
                        .unwrap_or_else(|e| panic!("{name}: reasoningEffort: {e}"))
                }),
                service_tier: case["options"].get("serviceTier").map(|tier| {
                    serde_json::from_value(tier.clone())
                        .unwrap_or_else(|e| panic!("{name}: serviceTier: {e}"))
                }),
                ..OpenAiResponsesOptions::default()
            };
            api.stream_with(&model, context, responses)
        };
        let mut events = Vec::new();
        while let Some(event) = stream.next().await {
            events.push(recorded(&event));
        }
        let message = Message::Assistant(Box::new(stream.result().await));
        let mut message: Value = ovo_ai::json::to_string(&message)
            .and_then(|text| ovo_ai::json::from_str(&text))
            .unwrap_or_else(|e| panic!("message: {e}"));
        if let Some(object) = message.as_object_mut() {
            object.remove("timestamp");
            object.remove("durationMs");
        }
        let mut events = Value::Array(events);
        let mut want_events = case["events"].clone();
        let mut want_message = case["message"].clone();
        for value in [
            &mut events,
            &mut message,
            &mut want_events,
            &mut want_message,
        ] {
            mask_engine_errors(value);
        }
        assert!(
            js_value_eq(&events, &want_events),
            "{name}: events\n  got:  {events}\n  want: {want_events}"
        );
        let provider_events = {
            let captured = provider_events
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            Value::Array(captured.clone())
        };
        assert!(
            js_value_eq(&provider_events, &case["providerEvents"]),
            "{name}: providerEvents\n  got:  {provider_events}\n  want: {}",
            case["providerEvents"]
        );

        assert!(
            js_value_eq(&message, &want_message),
            "{name}: message\n  got:  {message}\n  want: {want_message}"
        );
    }
}
