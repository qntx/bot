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

#[test]
fn anthropic_request() {
    for case in vectors().as_array().unwrap_or(&Vec::new()) {
        let name = case["name"].as_str().unwrap_or("?");
        let model: ovo_ai::types::model::Model = serde_json::from_value(case["model"].clone())
            .unwrap_or_else(|e| panic!("{name}: model: {e}"));
        let context: ovo_ai::types::context::Context =
            serde_json::from_value(case["context"].clone())
                .unwrap_or_else(|e| panic!("{name}: context: {e}"));
        let mut options = ovo_ai::api::anthropic_messages::AnthropicOptions::default();
        if let Some(api_key) = case["options"]["apiKey"].as_str() {
            options.base.api_key = Some(api_key.to_owned());
        }
        if let Some(t) = case["options"]["temperature"].as_f64() {
            options.base.temperature = Some(t);
        }
        if let Some(r) = case["options"]["cacheRetention"].as_str() {
            options.base.cache_retention = Some(match r {
                "long" => ovo_ai::options::CacheRetention::Long,
                "none" => ovo_ai::options::CacheRetention::None,
                _ => ovo_ai::options::CacheRetention::Short,
            });
        }
        let request =
            ovo_ai::api::anthropic_messages::build_request(&model, &context, Some(&options))
                .unwrap_or_else(|e| panic!("build_request: {e}"));
        let (url, headers, body) = (request.url, request.headers, request.body);
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
