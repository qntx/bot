//! `AI-API-OPENAI-RESPONSES`: `build_request` against the requests the
//! published `openai-responses` adapter sends.

use ovo_ai::api::openai_responses::{OpenAiResponsesOptions, build_request};
use ovo_ai::options::{CacheRetention, SimpleStreamOptions, StreamOptions};
use ovo_ai::types::context::Context;
use ovo_ai::types::model::Model;
use serde_json::Value;

fn vectors() -> Vec<Value> {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../vectors/ai/openai-responses-request.json"
    );
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let cases: Vec<Value> = serde_json::from_str(&text).unwrap_or_else(|e| panic!("{path}: {e}"));
    assert!(!cases.is_empty(), "no cases");
    cases
}

/// Cases whose expected error text comes from V8's `JSON.parse`
/// (`engine-error-text`): only the failure is compared.
const ENGINE_ERRORS: [&str; 1] = ["assistant-thinking-invalid-signature"];

/// parity: AI-API-OPENAI-RESPONSES
/// `payloadPatch`/`payloadUndefined` — the deterministic `onPayload`
/// the vectors' generator applies before the request is sent.
fn patched_body(
    case: &Value,
    params: &serde_json::Map<String, Value>,
    body: &[u8],
    merge_stream: bool,
) -> Vec<u8> {
    if case["payloadUndefined"].as_bool() == Some(true) || case["payloadPatch"].is_null() {
        return body.to_vec();
    }
    let mut params = params.clone();
    if let Some(keys) = case["payloadPatch"]["delete"].as_array() {
        for key in keys {
            params.remove(key.as_str().unwrap_or_default());
        }
    }
    if let Some(set) = case["payloadPatch"]["set"].as_object() {
        for (key, value) in set {
            params.insert(key.clone(), value.clone());
        }
    }
    if merge_stream {
        params.insert("stream".into(), Value::Bool(true));
    }
    ovo_ai::json::to_vec(&Value::Object(params)).unwrap_or_else(|e| panic!("patched body: {e}"))
}

#[test]
fn openai_responses_request() {
    for case in vectors() {
        let name = case["name"].as_str().unwrap_or("?");
        let model: Model = serde_json::from_value(case["model"].clone())
            .unwrap_or_else(|e| panic!("{name}: model: {e}"));
        let context: Context = serde_json::from_value(case["context"].clone())
            .unwrap_or_else(|e| panic!("{name}: context: {e}"));
        let transcript = ovo_ai::transcript::normalize_context(&context);
        let options = if case["simple"].as_bool() == Some(true) {
            OpenAiResponsesOptions::from_simple(
                &model,
                &transcript,
                &simple_options(&case["options"]),
            )
        } else {
            responses_options(&case["options"])
        };
        let built = build_request(&model, &transcript, &options);
        if let Some(expected) = case["request"]["error"].as_str() {
            if ENGINE_ERRORS.contains(&name) {
                assert!(built.is_err(), "{name}: expected an error");
            } else {
                assert_eq!(built.err().as_deref(), Some(expected), "{name}: error");
            }
            continue;
        }
        let request = built.unwrap_or_else(|e| panic!("{name}: build_request: {e}"));
        assert_eq!(
            request.url,
            case["request"]["url"].as_str().unwrap_or_default(),
            "{name}: url"
        );
        assert_eq!(
            String::from_utf8_lossy(&patched_body(&case, &request.params, &request.body, false)),
            case["request"]["body"].as_str().unwrap_or_default(),
            "{name}: body"
        );
        // The identity `user-agent` names ovo, not pi and Node.
        let mut sent: Vec<(String, String)> = request
            .headers
            .iter()
            .filter(|(header, _)| !header.eq_ignore_ascii_case("user-agent"))
            .map(|(header, value)| {
                (
                    header.to_lowercase(),
                    value.as_str().unwrap_or_default().to_owned(),
                )
            })
            .collect();
        let mut wanted: Vec<(String, String)> = case["request"]["headers"]
            .as_object()
            .into_iter()
            .flatten()
            .map(|(header, value)| {
                (
                    header.clone(),
                    value.as_str().unwrap_or_default().to_owned(),
                )
            })
            .collect();
        sent.sort();
        wanted.sort();
        assert_eq!(sent, wanted, "{name}: headers");
    }
}

/// The stream option fields the vectors set.
fn stream_options(options: &Value) -> StreamOptions {
    StreamOptions {
        api_key: options["apiKey"]
            .as_str()
            .map(|k| secrecy::SecretString::from(k.to_owned())),
        temperature: options["temperature"].as_f64(),
        max_tokens: options["maxTokens"].as_u64(),
        max_retries: options["maxRetries"]
            .as_u64()
            .and_then(|n| u32::try_from(n).ok()),
        cache_retention: options["cacheRetention"]
            .as_str()
            .map(|retention| match retention {
                "long" => CacheRetention::Long,
                "none" => CacheRetention::None,
                _ => CacheRetention::Short,
            }),
        session_id: options["sessionId"].as_str().map(str::to_owned),
        env: options["env"].as_object().cloned(),
        headers: options["headers"].as_object().cloned(),
        sampling_params: options["samplingParams"].as_object().cloned(),
        identity: options.get("identity").map(|identity| {
            serde_json::from_value(identity.clone()).unwrap_or_else(|e| panic!("identity: {e}"))
        }),
        ..StreamOptions::default()
    }
}

fn decoded<T: serde::de::DeserializeOwned>(options: &Value, key: &str) -> Option<T> {
    options
        .get(key)
        .filter(|value| !value.is_null())
        .map(|value| serde_json::from_value(value.clone()).unwrap_or_else(|e| panic!("{key}: {e}")))
}

fn responses_options(options: &Value) -> OpenAiResponsesOptions {
    OpenAiResponsesOptions {
        base: stream_options(options),
        reasoning_effort: decoded(options, "reasoningEffort"),
        reasoning_summary: decoded(options, "reasoningSummary"),
        service_tier: decoded(options, "serviceTier"),
        tool_choice: options.get("toolChoice").cloned(),
    }
}

fn simple_options(options: &Value) -> SimpleStreamOptions {
    SimpleStreamOptions {
        base: stream_options(options),
        reasoning: decoded(options, "reasoning"),
        tool_choice: decoded(options, "toolChoice"),
        ..SimpleStreamOptions::default()
    }
}
