//! `AI-API-OPENAI-COMPLETIONS`: `build_request` against the requests the
//! published `openai-completions` adapter sends.

use ovo_ai::api::openai_completions::{OpenAiCompletionsOptions, build_request};
use ovo_ai::options::{CacheRetention, SimpleStreamOptions, StreamOptions};
use ovo_ai::types::context::Context;
use ovo_ai::types::model::Model;
use serde_json::Value;

fn vectors() -> Vec<Value> {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../vectors/ai/openai-completions-request.json"
    );
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let cases: Vec<Value> = serde_json::from_str(&text).unwrap_or_else(|e| panic!("{path}: {e}"));
    assert!(!cases.is_empty(), "no cases");
    cases
}

/// parity: AI-API-OPENAI-COMPLETIONS
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
fn openai_completions_request() {
    for case in vectors() {
        let name = case["name"].as_str().unwrap_or("?");
        let model: Model = serde_json::from_value(case["model"].clone())
            .unwrap_or_else(|e| panic!("{name}: model: {e}"));
        let context: Context = serde_json::from_value(case["context"].clone())
            .unwrap_or_else(|e| panic!("{name}: context: {e}"));
        let transcript = ovo_ai::transcript::normalize_context(&context);
        let options = if case["simple"].as_bool() == Some(true) {
            OpenAiCompletionsOptions::from_simple(
                &model,
                &transcript,
                &simple_options(&case["options"]),
            )
        } else {
            completions_options(&case["options"])
        };
        let built = build_request(&model, &transcript, &options);
        if let Some(expected) = case["request"]["error"].as_str() {
            assert_eq!(built.err().as_deref(), Some(expected), "{name}: error");
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
        let expected = case["request"]["headers"]
            .as_object()
            .cloned()
            .unwrap_or_default();
        // The identity `user-agent` names ovo, not pi and Node.
        let sent: Vec<(String, String)> = request
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
        let wanted: Vec<(String, String)> = expected
            .iter()
            .filter(|(header, _)| header.as_str() != "user-agent")
            .map(|(header, value)| {
                (
                    header.clone(),
                    value.as_str().unwrap_or_default().to_owned(),
                )
            })
            .collect();
        let (mut sent, mut wanted) = (sent, wanted);
        sent.sort();
        wanted.sort();
        assert_eq!(sent, wanted, "{name}: headers");
    }
}

/// The stream option fields the vectors set.
fn stream_options(options: &Value) -> StreamOptions {
    let mut base = StreamOptions::default();
    base.api_key = options["apiKey"].as_str().map(str::to_owned);
    base.temperature = options["temperature"].as_f64();
    base.max_tokens = options["maxTokens"].as_u64();
    base.max_retries = options["maxRetries"]
        .as_u64()
        .and_then(|n| u32::try_from(n).ok());
    base.cache_retention = options["cacheRetention"]
        .as_str()
        .map(|retention| match retention {
            "long" => CacheRetention::Long,
            "none" => CacheRetention::None,
            _ => CacheRetention::Short,
        });
    base.session_id = options["sessionId"].as_str().map(str::to_owned);
    base.env = options["env"].as_object().cloned();
    base.headers = options["headers"].as_object().cloned();
    base.sampling_params = options["samplingParams"].as_object().cloned();
    base.identity = options.get("identity").map(|identity| {
        serde_json::from_value(identity.clone()).unwrap_or_else(|e| panic!("identity: {e}"))
    });
    base
}

fn completions_options(options: &Value) -> OpenAiCompletionsOptions {
    let mut completions = OpenAiCompletionsOptions::default();
    completions.base = stream_options(options);
    completions.tool_choice = options.get("toolChoice").cloned();
    completions.reasoning_effort = options.get("reasoningEffort").map(|level| {
        serde_json::from_value(level.clone()).unwrap_or_else(|e| panic!("reasoningEffort: {e}"))
    });
    completions.thinking_budgets = options.get("thinkingBudgets").map(|budgets| {
        serde_json::from_value(budgets.clone()).unwrap_or_else(|e| panic!("thinkingBudgets: {e}"))
    });
    completions
}

fn simple_options(options: &Value) -> SimpleStreamOptions {
    let mut simple = SimpleStreamOptions::default();
    simple.base = stream_options(options);
    simple.reasoning = options.get("reasoning").map(|level| {
        serde_json::from_value(level.clone()).unwrap_or_else(|e| panic!("reasoning: {e}"))
    });
    simple.tool_choice = options.get("toolChoice").map(|choice| {
        serde_json::from_value(choice.clone()).unwrap_or_else(|e| panic!("toolChoice: {e}"))
    });
    simple.thinking_budgets = options.get("thinkingBudgets").map(|budgets| {
        serde_json::from_value(budgets.clone()).unwrap_or_else(|e| panic!("thinkingBudgets: {e}"))
    });
    simple
}
