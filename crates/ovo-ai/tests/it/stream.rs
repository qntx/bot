//! Replays `AI-STREAM-PROTOCOL` vectors: each case's sender-side ops are
//! replayed through `AssistantStreamSender`; the yielded event sequence must
//! equal upstream's events (minus `partial`), and `result()` the recorded
//! final message minus `role`/`durationMs` (`key-order` class).

use serde_json::Value;

use ovo_ai::stream::{AssistantMessageEvent, AssistantStream, DoneReason, ErrorReason, Metadata};
use ovo_ai::types::content::{TextContent, ThinkingContent, ToolCall};
use ovo_ai::types::deferred::DeferredHandle;
use ovo_ai::types::message::AssistantMessage;

use crate::support::js_value_eq;

fn base_from(message: &Value) -> AssistantMessage {
    // The adapter's output literal minus streamed fields.
    let mut base: AssistantMessage = serde_json::from_value(message.clone())
        .unwrap_or_else(|error| panic!("base message: {error}"));
    base.content.clear();
    base.duration_ms = None;
    base
}

fn _unused() -> AssistantMessage {
    serde_json::from_value(serde_json::json!({
        "role": "assistant",
        "content": [],
        "api": "anthropic-messages",
        "provider": "anthropic",
        "model": "m",
        "usage": {
            "input": 0,
            "output": 0,
            "cacheRead": 0,
            "cacheWrite": 0,
            "totalTokens": 0,
            "cost": { "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "total": 0 },
        },
        "stopReason": "stop",
        "timestamp": 0,
    }))
    .unwrap_or_else(|error| panic!("base message: {error}"))
}

fn decode_ops(ops: &[Value]) -> Vec<(String, Value)> {
    ops.iter()
        .map(|op| (op["op"].as_str().unwrap_or_default().to_owned(), op.clone()))
        .collect()
}

/// Strip upstream-only fields from a recorded event: `message` on `done`
/// and the `error` payload (mapped to our `errorMessage`).
fn normalize_event(mut event: Value) -> Value {
    let Some(object) = event.as_object_mut() else {
        return event;
    };
    object.remove("message");
    let text = object.remove("error").and_then(|e| {
        e.get("errorMessage")
            .and_then(Value::as_str)
            .map(str::to_owned)
    });
    if let Some(text) = text {
        object.insert("errorMessage".to_owned(), Value::from(text));
    }
    event
}

fn last_index(
    last: &std::collections::HashMap<&'static str, usize>,
    kind: &str,
    op: &Value,
) -> usize {
    op["index"].as_u64().map_or_else(
        || *last.get(kind).unwrap_or(&0),
        |i| usize::try_from(i).unwrap_or(usize::MAX),
    )
}

/// Replay one case's ops through the sender, collecting events.
async fn replay_ops(
    name: &str,
    ops: &[(String, Value)],
    base_msg: &Value,
) -> (Vec<AssistantMessageEvent>, AssistantMessage) {
    let (tx0, mut stream) = AssistantStream::channel(base_from(base_msg), 256);
    let mut sender = Some(tx0);
    let mut last: std::collections::HashMap<&'static str, usize> = std::collections::HashMap::new();
    for (op, value) in ops {
        let Some(tx) = sender.as_mut() else {
            break;
        };
        match op.as_str() {
            "start" => tx.start().await.unwrap_or_default(),
            "text_start" => {
                last.insert("text", tx.text_start().await.unwrap_or(0));
            }
            "thinking_start" => {
                last.insert("thinking", tx.thinking_start().await.unwrap_or(0));
            }
            "text_delta" | "thinking_delta" | "toolcall_delta" => {
                send_delta(tx, op, value, &last).await;
            }
            "text_end" => {
                let index = last_index(&last, "text", value);
                let block: TextContent = serde_json::from_value(value["block"].clone())
                    .unwrap_or_else(|e| panic!("{name}: {e}"));
                tx.text_end(index, block).await.unwrap_or_default();
            }
            "thinking_end" => {
                let index = last_index(&last, "thinking", value);
                let block: ThinkingContent = serde_json::from_value(value["block"].clone())
                    .unwrap_or_else(|e| panic!("{name}: {e}"));
                tx.thinking_end(index, block).await.unwrap_or_default();
            }
            "toolcall_start" => {
                let call: ToolCall = serde_json::from_value(value["call"].clone())
                    .unwrap_or_else(|e| panic!("{name}: {e}"));
                last.insert("toolcall", tx.toolcall_start(call).await.unwrap_or(0));
            }
            "toolcall_replace" => {
                let index = last_index(&last, "toolcall", value);
                let text = value["text"].as_str().unwrap_or_default();
                tx.toolcall_replace(index, text).await.unwrap_or_default();
            }
            "toolcall_end" => {
                let index = last_index(&last, "toolcall", value);
                tx.toolcall_end(index).await.unwrap_or_default();
            }
            "metadata" => send_metadata(tx, name, value).await,
            "done" => {
                send_done(&mut sender, name, value).await;
                break;
            }
            "error" => {
                send_error(&mut sender, value).await;
                break;
            }
            other => panic!("{name}: unknown op {other}"),
        }
    }
    drop(sender);
    let mut events = Vec::new();
    while let Some(event) = futures::StreamExt::next(&mut stream).await {
        events.push(event);
    }
    (events, stream.result().await)
}

async fn send_delta(
    tx: &mut ovo_ai::stream::AssistantStreamSender,
    op: &str,
    value: &Value,
    last: &std::collections::HashMap<&'static str, usize>,
) {
    let kind = if op == "toolcall_delta" {
        "toolcall"
    } else if op == "thinking_delta" {
        "thinking"
    } else {
        "text"
    };
    let index = last_index(last, kind, value);
    let delta = value["delta"].as_str().unwrap_or_default();
    let result = match op {
        "text_delta" => tx.text_delta(index, delta).await,
        "thinking_delta" => tx.thinking_delta(index, delta).await,
        _ => tx.toolcall_delta(index, delta).await,
    };
    result.unwrap_or_default();
}

async fn send_metadata(tx: &ovo_ai::stream::AssistantStreamSender, name: &str, value: &Value) {
    let field = value["field"].as_str().unwrap_or_default();
    let raw = value["value"].clone();
    let metadata = match field {
        "usage" => {
            Metadata::Usage(serde_json::from_value(raw).unwrap_or_else(|e| panic!("{name}: {e}")))
        }
        "responseId" => Metadata::ResponseId(raw.as_str().unwrap_or_default().to_owned()),
        "responseModel" => Metadata::ResponseModel(raw.as_str().unwrap_or_default().to_owned()),
        "deferred" => Metadata::Deferred(
            serde_json::from_value::<DeferredHandle>(raw).unwrap_or_else(|e| panic!("{name}: {e}")),
        ),
        other => panic!("{name}: unknown metadata field {other}"),
    };
    tx.metadata(metadata).await.unwrap_or_default();
}

async fn send_done(
    sender: &mut Option<ovo_ai::stream::AssistantStreamSender>,
    name: &str,
    value: &Value,
) {
    let reason = match value["reason"].as_str().unwrap_or_default() {
        "stop" => DoneReason::Stop,
        "length" => DoneReason::Length,
        "toolUse" => DoneReason::ToolUse,
        "deferred" => DoneReason::Deferred,
        other => panic!("{name}: unknown reason {other}"),
    };
    if let Some(s) = sender.take() {
        s.done(reason).await.unwrap_or_default();
    }
}

async fn send_error(sender: &mut Option<ovo_ai::stream::AssistantStreamSender>, value: &Value) {
    let reason = match value["reason"].as_str().unwrap_or_default() {
        "aborted" => ErrorReason::Aborted,
        _ => ErrorReason::Error,
    };
    let message = value["errorMessage"]
        .as_str()
        .unwrap_or_default()
        .to_owned();
    if let Some(s) = sender.take() {
        s.error(reason, message).await.unwrap_or_default();
    }
}

/// parity: AI-STREAM-PROTOCOL
#[test]
fn stream_cases_replay() {
    let text = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../vectors/ai/stream.json"
    ));
    let file: Value = match ovo_ai::json::from_str(text) {
        Ok(value) => value,
        Err(error) => panic!("invalid vector file: {error}"),
    };
    let Some(cases) = file.get("cases").and_then(Value::as_array) else {
        panic!("vector file has no cases");
    };
    assert_ne!(cases.len(), 0);
    for case in cases {
        let name = case["name"].as_str().unwrap_or("?");
        let Some(ops) = case["ops"].as_array() else {
            panic!("{name}: no ops");
        };
        let ops = decode_ops(ops);
        let (events, message) =
            futures::executor::block_on(replay_ops(name, &ops, &case["message"]));
        let Some(expected_events) = case["events"].as_array() else {
            panic!("{name}: no events");
        };
        assert_eq!(
            events.len(),
            expected_events.len(),
            "{name}: event count differs"
        );
        for (index, (event, recorded)) in events.iter().zip(expected_events.iter()).enumerate() {
            let recorded = normalize_event(recorded.clone());
            let actual: Value = serde_json::from_str(
                &ovo_ai::json::to_string(event).unwrap_or_else(|e| panic!("{name}: {e}")),
            )
            .unwrap_or_default();
            assert!(
                js_value_eq(&actual, &recorded),
                "{name}: event {index} differs\n{actual}\n{recorded}"
            );
        }
        // Compare final message: `role` is the tag, `durationMs` is stripped.
        let mut expected_message = case["message"].clone();
        if let Some(object) = expected_message.as_object_mut() {
            object.remove("role");
        }
        let actual = serde_json::to_value(&message).unwrap_or_default();
        assert!(
            js_value_eq(&actual, &expected_message),
            "{name}: message differs\n{actual}"
        );
    }
}
