//! Replays the anthropic-messages response vectors: each scripted SSE body
//! (or error response) is served by a local server and streamed through
//! `AnthropicMessagesApi`; events and the final message must equal what the
//! published adapter produced.

use futures::StreamExt;
use ovo_ai::api::anthropic_messages::AnthropicMessagesApi;
use ovo_ai::http::client::HttpClientConfig;
use ovo_ai::options::StreamOptions;
use ovo_ai::provider::ChatApi;
use ovo_ai::stream::AssistantMessageEvent;
use ovo_ai::transcript::normalize_context;
use ovo_ai::types::context::Context;
use ovo_ai::types::message::Message;
use ovo_ai::types::model::Model;
use serde_json::Value;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

use crate::support::js_value_eq;

fn vectors() -> Value {
    let text = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../vectors/ai/anthropic-stream.json"
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
        .and(path("/v1/messages"))
        .respond_with(response)
        .mount(&server)
        .await;
    server
}

/// parity: AI-API-ANTHROPIC
#[tokio::test]
async fn anthropic_stream_replays_upstream_events() {
    let vectors = vectors();
    let context: Context = serde_json::from_value(vectors["context"].clone())
        .unwrap_or_else(|e| panic!("context: {e}"));
    let cases = vectors["cases"].as_array().cloned().unwrap_or_default();
    assert!(!cases.is_empty(), "no cases");
    for case in &cases {
        let name = case["name"].as_str().unwrap_or("?");
        let server = serve(case).await;
        let mut model: Model = serde_json::from_value(vectors["model"].clone())
            .unwrap_or_else(|e| panic!("model: {e}"));
        model.base_url = server.uri();
        let api = AnthropicMessagesApi::new(HttpClientConfig {
            max_retries: 0,
            ..HttpClientConfig::default()
        });
        let mut options = StreamOptions::default();
        options.api_key = Some(String::from("sk-ant-api03-test"));
        options.max_retries = Some(0);
        let mut stream = api.stream(&model, normalize_context(&context), options);
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
        let events = Value::Array(events);
        assert!(
            js_value_eq(&events, &case["events"]),
            "{name}: events\n  got:  {events}\n  want: {}",
            case["events"]
        );
        assert!(
            js_value_eq(&message, &case["message"]),
            "{name}: message\n  got:  {message}\n  want: {}",
            case["message"]
        );
    }
}
