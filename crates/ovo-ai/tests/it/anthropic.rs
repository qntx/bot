//! parity: AI-API-ANTHROPIC
//!
//! `AI-API-ANTHROPIC` — `build_request` parity against the upstream
//! dist `streamAnthropic` (vectors capture the request the SDK sends).
use serde_json::Value;

fn vectors() -> Value {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../vectors/ai/anthropic-request.json"
    );
    serde_json::from_str(&std::fs::read_to_string(path).unwrap_or_default()).unwrap_or_default()
}

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
fn anthropic_request() {
    for case in vectors().as_array().unwrap_or(&Vec::new()) {
        let name = case["name"].as_str().unwrap_or("?");
        let model: ovo_ai::types::model::Model = serde_json::from_value(case["model"].clone())
            .unwrap_or_else(|e| panic!("{name}: model: {e}"));
        let context: ovo_ai::types::context::Context =
            serde_json::from_value(case["context"].clone())
                .unwrap_or_else(|e| panic!("{name}: context: {e}"));
        let options = anthropic_options(&case["options"]);
        // The vectors call the published adapter directly, which reads only
        // `context.messages`; normalize the same messages.
        let mut context = context;
        context.system_prompt = None;
        context.tools = None;
        let transcript = ovo_ai::transcript::normalize_context(&context);
        let built = if case["simple"].as_bool() == Some(true) {
            let simple = simple_options(&case["options"]);
            let mapped = ovo_ai::api::anthropic_messages::AnthropicOptions::from_simple(
                &model,
                &transcript,
                &simple,
            );
            ovo_ai::api::anthropic_messages::build_request(&model, &transcript, &mapped, false)
        } else {
            ovo_ai::api::anthropic_messages::build_request(&model, &transcript, &options, false)
        };
        if let Some(expected_error) = case["request"]["error"].as_str() {
            assert_eq!(
                built.err().as_deref(),
                Some(expected_error),
                "{name}: error"
            );
            continue;
        }
        let request = built.unwrap_or_else(|e| panic!("{name}: build_request: {e}"));
        let body = patched_body(case, &request.params, &request.body, true);
        let (url, headers) = (request.url, request.headers);
        assert_eq!(
            url,
            case["request"]["url"].as_str().unwrap_or_default(),
            "{name}: url"
        );
        let expected = case["request"]["body"].as_str().unwrap_or_default();
        assert_eq!(
            String::from_utf8_lossy(&body),
            expected,
            "{name}: body\n  got: {}\n  want: {expected}",
            String::from_utf8_lossy(&body)
        );
        // pi-set headers only — `accept`/`anthropic-*`/auth/`x-app`; the
        // SDK's `x-stainless-*` and the identity `user-agent` (non-OAuth)
        // carry node-version/platform data — not compared.
        let is_oauth = case["options"]["apiKey"]
            .as_str()
            .unwrap_or_default()
            .contains("sk-ant-oat");
        for (key, value) in case["request"]["headers"]
            .as_object()
            .cloned()
            .unwrap_or_default()
        {
            let compared = matches!(
                key.as_str(),
                "accept"
                    | "anthropic-dangerous-direct-browser-access"
                    | "anthropic-version"
                    | "anthropic-beta"
                    | "x-api-key"
                    | "authorization"
                    | "x-app"
            ) || (is_oauth && key == "user-agent");
            if !compared {
                continue;
            }
            assert_eq!(
                headers
                    .iter()
                    .find(|(header, _)| header.eq_ignore_ascii_case(&key))
                    .and_then(|(_, sent)| sent.as_str())
                    .unwrap_or_default(),
                value.as_str().unwrap_or_default(),
                "{name}: header {key}"
            );
        }
    }
}

/// The stream option fields the vectors set.
fn stream_options(options: &Value) -> ovo_ai::options::StreamOptions {
    let mut base = ovo_ai::options::StreamOptions::default();
    base.api_key = options["apiKey"].as_str().map(str::to_owned);
    base.temperature = options["temperature"].as_f64();
    base.max_tokens = options["maxTokens"].as_u64();
    base.cache_retention = options["cacheRetention"].as_str().map(|r| match r {
        "long" => ovo_ai::options::CacheRetention::Long,
        "none" => ovo_ai::options::CacheRetention::None,
        _ => ovo_ai::options::CacheRetention::Short,
    });
    base.metadata = options["metadata"].as_object().cloned();
    base.headers = options["headers"].as_object().cloned();
    base
}

/// `AnthropicOptions` from a vector case's options.
fn anthropic_options(options: &Value) -> ovo_ai::api::anthropic_messages::AnthropicOptions {
    let mut anthropic = ovo_ai::api::anthropic_messages::AnthropicOptions::default();
    anthropic.base = stream_options(options);
    anthropic.thinking_enabled = options["thinkingEnabled"].as_bool();
    anthropic.thinking_budget_tokens = options["thinkingBudgetTokens"].as_u64();
    anthropic.effort = options["effort"].as_str().map(str::to_owned);
    anthropic.thinking_display = options["thinkingDisplay"].as_str().map(str::to_owned);
    anthropic.interleaved_thinking = options["interleavedThinking"].as_bool();
    anthropic.tool_choice = options.get("toolChoice").filter(|v| !v.is_null()).cloned();
    anthropic
}

/// `SimpleStreamOptions` from a vector case's options.
fn simple_options(options: &Value) -> ovo_ai::options::SimpleStreamOptions {
    let mut simple = ovo_ai::options::SimpleStreamOptions::default();
    simple.base = stream_options(options);
    simple.reasoning = options
        .get("reasoning")
        .and_then(|level| serde_json::from_value(level.clone()).ok());
    simple.tool_choice = options
        .get("toolChoice")
        .and_then(|choice| serde_json::from_value(choice.clone()).ok());
    simple
}
