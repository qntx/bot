//! `AI-API-ANTHROPIC` — request-shape parity between ovo's
//! `build_request` and upstream's `streamAnthropic` (vectors generated
//! from the real dist adapter with a capturing `fetch`).

/// `x-api-key`/`Authorization`/`anthropic-version`/`anthropic-beta` —
/// the pi-set request headers (SDK transport defaults are not compared).
const PI_SET: &[&str] = &[
    "x-api-key",
    "authorization",
    "anthropic-version",
    "anthropic-beta",
];

use serde_json::Value;

fn vectors() -> Value {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../vectors/ai/anthropic-request.json"
    );
    serde_json::from_str(&std::fs::read_to_string(path).unwrap_or_default()).unwrap_or_default()
}

fn base_model() -> ovo_ai::types::model::Model {
    serde_json::from_value(serde_json::json!({
        "id": "claude-test",
        "name": "Test",
        "api": "anthropic-messages",
        "provider": "anthropic",
        "baseUrl": "https://api.anthropic.com",
        "input": ["text", "image"],
        "cost": {"input":0,"output":0,"cacheRead":0,"cacheWrite":0},
        "reasoning": true,
        "contextWindow": 200_000,
        "maxTokens": 8192,
    }))
    .unwrap_or_else(|e| panic!("model: {e}"))
}

#[expect(
    clippy::needless_pass_by_value,
    reason = "messages move into the Context — the by-value call mirrors the generator shape"
)]
fn base_context(messages: Vec<ovo_ai::types::message::Message>) -> ovo_ai::types::context::Context {
    serde_json::from_value(serde_json::json!({"messages": messages}))
        .unwrap_or_else(|e| panic!("context: {e}"))
}

fn base_options() -> ovo_ai::api::anthropic_messages::AnthropicOptions {
    let mut options = ovo_ai::api::anthropic_messages::AnthropicOptions::default();
    options.base.api_key = Some(String::from("sk-ant-api03-test"));
    options
}

fn user(text: &str) -> ovo_ai::types::message::Message {
    serde_json::from_value(serde_json::json!({
        "role": "user",
        "content": text,
        "timestamp": 0,
    }))
    .unwrap_or_else(|e| panic!("user: {e}"))
}

/// Byte-identical request body for every vector.
///
/// parity: AI-API-ANTHROPIC
#[test]
fn anthropic_request_body() {
    for case in vectors().as_array().unwrap_or(&Vec::new()) {
        let name = case["name"].as_str().unwrap_or("?");
        let expected = case["request"]["body"].as_str().unwrap_or("");
        // Rebuild the same context the generator used; each case maps
        // to a hand-coded transcript matching the generator's.
        let (context, options) = build_case(name);
        let (_, headers, body) =
            ovo_ai::api::anthropic_messages::build_request(&base_model(), &context, Some(&options));
        assert_eq!(
            String::from_utf8_lossy(&body),
            expected,
            "{name}: body mismatch\n  got: {}\n  want: {expected}",
            String::from_utf8_lossy(&body)
        );
        let expected_headers = case["request"]["headers"]
            .as_object()
            .cloned()
            .unwrap_or_default();
        // pi sets `x-api-key`/`Authorization`/`anthropic-version`/
        // `anthropic-beta` (plus model/provider/request headers);
        // everything else is SDK client defaults — not compared.
        for (key, value) in &expected_headers {
            if !PI_SET.iter().any(|k| key.eq_ignore_ascii_case(k)) {
                continue;
            }
            assert_eq!(
                headers.get(key).and_then(Value::as_str).unwrap_or(""),
                value.as_str().unwrap_or(""),
                "{name}: header {key} mismatch",
            );
        }
    }
}

#[expect(
    clippy::too_many_lines,
    reason = "one arm per generated vector case — splitting would obscure the case table"
)]
fn build_case(
    name: &str,
) -> (
    ovo_ai::types::context::Context,
    ovo_ai::api::anthropic_messages::AnthropicOptions,
) {
    let mut options = base_options();
    let ctx = match name {
        "plain-text" => base_context(vec![user("hello")]),
        "system-prompt" => {
            let mut c = base_context(vec![user("hello")]);
            c.system_prompt = Some(String::from("you are helpful"));
            c
        },
        "image-only" => base_context(vec![serde_json::from_value(serde_json::json!({
            "role": "user",
            "content": [{"type":"image","data":"aW1hZ2U=","mimeType":"image/png"}],
            "timestamp": 0,
        })).unwrap_or_else(|e| panic!("{e}"))]),
        "mixed-text-image" => base_context(vec![serde_json::from_value(serde_json::json!({
            "role": "user",
            "content": [
                {"type":"text","text":"see this"},
                {"type":"image","data":"aW1hZ2U=","mimeType":"image/jpeg"},
            ],
            "timestamp": 0,
        })).unwrap_or_else(|e| panic!("{e}"))]),
        "assistant-thinking-signature" => base_context(vec![
            user("q"),
            serde_json::from_value(serde_json::json!({
                "role": "assistant",
                "content": [
                    {"type":"thinking","thinking":"hmm","thinkingSignature":"sig-1"},
                    {"type":"text","text":"answer"},
                ],
                "api": "anthropic-messages", "provider": "anthropic", "model": "claude-test",
                "usage": {"input":0,"output":0,"cacheRead":0,"cacheWrite":0,"totalTokens":0,"cost":{"input":0,"output":0,"cacheRead":0,"cacheWrite":0,"total":0}},
                "stopReason": "stop",
                "timestamp": 0,
            })).unwrap_or_else(|e| panic!("{e}")),
            user("followup"),
        ]),
        "redacted-thinking" => base_context(vec![
            user("q"),
            serde_json::from_value(serde_json::json!({
                "role": "assistant",
                "content": [{"type":"thinking","thinking":"","thinkingSignature":"redacted-blob","redacted":true}],
                "api": "anthropic-messages", "provider": "anthropic", "model": "claude-test",
                "usage": {"input":0,"output":0,"cacheRead":0,"cacheWrite":0,"totalTokens":0,"cost":{"input":0,"output":0,"cacheRead":0,"cacheWrite":0,"total":0}},
                "stopReason": "stop",
                "timestamp": 0,
            })).unwrap_or_else(|e| panic!("{e}")),
            user("next"),
        ]),
        "tool-use-and-result" => base_context(vec![
            user("check"),
            serde_json::from_value(serde_json::json!({
                "role": "assistant",
                "content": [{"type":"toolCall","id":"call_1","name":"Read","arguments":{"path":"/tmp/a"}}],
                "api": "anthropic-messages", "provider": "anthropic", "model": "claude-test",
                "usage": {"input":0,"output":0,"cacheRead":0,"cacheWrite":0,"totalTokens":0,"cost":{"input":0,"output":0,"cacheRead":0,"cacheWrite":0,"total":0}},
                "stopReason": "toolUse",
                "timestamp": 0,
            })).unwrap_or_else(|e| panic!("{e}")),
            serde_json::from_value(serde_json::json!({
                "role": "toolResult",
                "toolCallId": "call_1", "toolName": "Read",
                "content": [{"type":"text","text":"file contents"}],
                "isError": false,
                "timestamp": 0,
            })).unwrap_or_else(|e| panic!("{e}")),
        ]),
        "tool-result-error-with-image" => base_context(vec![
            user("check"),
            serde_json::from_value(serde_json::json!({
                "role": "assistant",
                "content": [{"type":"toolCall","id":"call_1","name":"Screenshot","arguments":{}}],
                "api": "anthropic-messages", "provider": "anthropic", "model": "claude-test",
                "usage": {"input":0,"output":0,"cacheRead":0,"cacheWrite":0,"totalTokens":0,"cost":{"input":0,"output":0,"cacheRead":0,"cacheWrite":0,"total":0}},
                "stopReason": "toolUse",
                "timestamp": 0,
            })).unwrap_or_else(|e| panic!("{e}")),
            serde_json::from_value(serde_json::json!({
                "role": "toolResult",
                "toolCallId": "call_1", "toolName": "Screenshot",
                "content": [{"type":"text","text":"failed"},{"type":"image","data":"aW1hZ2U=","mimeType":"image/png"}],
                "isError": true,
                "timestamp": 0,
            })).unwrap_or_else(|e| panic!("{e}")),
        ]),
        "consecutive-tool-results" => base_context(vec![
            user("check"),
            serde_json::from_value(serde_json::json!({
                "role": "assistant",
                "content": [
                    {"type":"toolCall","id":"c1","name":"Read","arguments":{}},
                    {"type":"toolCall","id":"c2","name":"Read","arguments":{}},
                ],
                "api": "anthropic-messages", "provider": "anthropic", "model": "claude-test",
                "usage": {"input":0,"output":0,"cacheRead":0,"cacheWrite":0,"totalTokens":0,"cost":{"input":0,"output":0,"cacheRead":0,"cacheWrite":0,"total":0}},
                "stopReason": "toolUse",
                "timestamp": 0,
            })).unwrap_or_else(|e| panic!("{e}")),
            serde_json::from_value(serde_json::json!({
                "role": "toolResult",
                "toolCallId": "c1", "toolName": "Read",
                "content": [{"type":"text","text":"r1"}],
                "isError": false,
                "timestamp": 0,
            })).unwrap_or_else(|e| panic!("{e}")),
            serde_json::from_value(serde_json::json!({
                "role": "toolResult",
                "toolCallId": "c2", "toolName": "Read",
                "content": [{"type":"text","text":"r2"}],
                "isError": false,
                "timestamp": 0,
            })).unwrap_or_else(|e| panic!("{e}")),
        ]),
        "oauth-token" => {
            options.base.api_key = Some(String::from("sk-ant-oat-oauth-token"));
            let mut c = base_context(vec![user("hi")]);
            c.system_prompt = Some(String::from("system"));
            c
        },
        "cache-retention-long" => {
            options.base.cache_retention = Some(ovo_ai::options::CacheRetention::Long);
            let mut c = base_context(vec![user("hi")]);
            c.system_prompt = Some(String::from("system"));
            c
        },
        "cache-retention-none" => {
            options.base.cache_retention = Some(ovo_ai::options::CacheRetention::None);
            let mut c = base_context(vec![user("hi")]);
            c.system_prompt = Some(String::from("system"));
            c
        },
        "temperature-and-tools" => {
            options.base.temperature = Some(0.5);
            let mut c = base_context(vec![user("hi")]);
            c.system_prompt = Some(String::from("system"));
            c.tools = Some(vec![serde_json::from_value(serde_json::json!({
                "name": "Read",
                "description": "read a file",
                "parameters": {"type":"object","properties":{"path":{"type":"string"}},"required":["path"]},
            })).unwrap_or_else(|e| panic!("{e}"))]);
            c
        },
        _ => panic!("unknown case {name}"),
    };
    (ctx, options)
}
