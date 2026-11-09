//! HTTP policy vectors.
//!
//! parity: AI-HTTP-RETRY — `provider_retry_decision` replays pi's
//! `retryProviderRequest` loop decisions; `anthropic`/`openai` cases replay
//! the real SDK clients' `shouldRetry`/`retryRequest` (recorded sleeps).
//! parity: AI-ERROR-BODY — `normalizeProviderError`/`formatProviderError`.
//! parity: AI-HTTP-PROXY — `resolveHttpProxyUrlForTarget` decisions.
//! parity: AI-RETRY-POLICY — `retry_assistant_call` unit tests in retry.rs.

use ovo_ai::http::headers::{HeaderRecord, provider_headers_to_record};
use ovo_ai::http::proxy::resolve_http_proxy_url_for_target;
use ovo_ai::http::retry::{
    RetryDecision, anthropic_retry_decision, openai_retry_decision, provider_retry_decision,
};
use serde_json::Value;

use crate::support::js_value_eq;

fn cases(path: &str) -> Vec<Value> {
    let text = std::fs::read_to_string(format!(
        "{}/../../vectors/{path}",
        env!("CARGO_MANIFEST_DIR")
    ))
    .unwrap_or_else(|e| panic!("read {path}: {e}"));
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("parse {path}: {e}"))
}

fn headers_of(case: &Value) -> HeaderRecord {
    let map = case["headers"].as_object().cloned().unwrap_or_default();
    HeaderRecord::from_record(&map)
}

#[test]
fn http_retry_provider() {
    // One recorded run of the upstream loop: `delays` are the sleeps it
    // actually took, `outcome` how the loop ended.
    for case in cases("ai/http-retry.json") {
        if case.get("family").is_some() {
            continue;
        }
        let name = case["name"].as_str().unwrap_or_default();
        let status = case["status"].as_u64().map(|s| s as u32);
        let headers = headers_of(&case);
        let max_retries = case["max_retries"].as_u64().unwrap_or_default() as u32;
        let no_retry: Vec<u32> = case["no_retry_statuses"]
            .as_array()
            .map(|l| {
                l.iter()
                    .filter_map(Value::as_u64)
                    .map(|v| v as u32)
                    .collect()
            })
            .unwrap_or_default();
        let cap = case["max_retry_delay_ms"].as_f64();
        let now_ms = case["now_ms"].as_f64().unwrap_or_default();
        let expected = &case["expected"];
        let delays = expected["delays"].as_array().cloned().unwrap_or_default();
        for (i, delay) in delays.iter().enumerate() {
            #[expect(
                clippy::cast_possible_truncation,
                reason = "attempt counts stay in u32 range"
            )]
            let remaining = max_retries.saturating_sub(i as u32);
            let got = provider_retry_decision(
                status,
                &headers,
                i as u32,
                remaining,
                &no_retry,
                cap,
                now_ms,
                &|| 0.5,
            );
            let expected_delay = delay.as_f64().unwrap_or_default();
            assert!(
                matches!(got, RetryDecision::Retry { delay_ms } if js_value_eq(&serde_json::json!(delay_ms), &serde_json::json!(expected_delay))),
                "{name}[{i}]: got {got:?}, expected delay {expected_delay}"
            );
        }
        #[expect(
            clippy::cast_possible_truncation,
            reason = "attempt counts stay in u32 range"
        )]
        let remaining = max_retries.saturating_sub(delays.len() as u32);
        let terminal = provider_retry_decision(
            status,
            &headers,
            delays.len() as u32,
            remaining,
            &no_retry,
            cap,
            now_ms,
            &|| 0.5,
        );
        match expected["outcome"].as_str().unwrap_or_default() {
            "gave_up" | "threw" => {
                assert_eq!(terminal, RetryDecision::GiveUp, "{name}: terminal");
            }
            "cap" => assert!(
                matches!(terminal, RetryDecision::FailOverCap { .. }),
                "{name}: terminal should be FailOverCap, got {terminal:?}"
            ),
            other => panic!("{name}: unknown outcome {other}"),
        }
    }
}

#[test]
fn http_retry_sdks() {
    for case in cases("ai/http-retry.json") {
        let Some(family) = case.get("family").and_then(Value::as_str) else {
            continue;
        };
        let name = case["name"].as_str().unwrap_or_default();
        let headers = headers_of(&case);
        match case["kind"].as_str().unwrap_or_default() {
            "should_retry" => {
                let status = case["status"].as_u64().map(|s| s as u32);
                let got = ovo_ai::http::retry::is_retryable_status(status, &headers);
                assert_eq!(
                    got,
                    case["expected"].as_bool().unwrap_or_default(),
                    "{name}"
                );
            }
            "delay" => {
                let remaining = case["retries_remaining"].as_u64().unwrap_or_default() as u32;
                let max_retries = case["max_retries"].as_u64().unwrap_or_default() as u32;
                let expected = case["expected"].as_f64().unwrap_or_default();
                let now_ms = case["now_ms"].as_f64().unwrap_or_default();
                // retryRequest runs after shouldRetry already passed — feed a
                // retryable status so the decision reaches the delay path.
                let got = match family {
                    "anthropic" => anthropic_retry_decision(
                        Some(503),
                        &headers,
                        remaining,
                        max_retries,
                        now_ms,
                        &|| 0.5,
                    ),
                    "openai" => openai_retry_decision(
                        Some(503),
                        &headers,
                        remaining,
                        max_retries,
                        now_ms,
                        &|| 0.5,
                    ),
                    other => panic!("{name}: unknown family {other}"),
                };
                assert!(
                    matches!(got, RetryDecision::Retry { delay_ms } if js_value_eq(&serde_json::json!(delay_ms), &serde_json::json!(expected))),
                    "{name}: got {got:?}, expected delay {expected}"
                );
            }
            other => panic!("{name}: unknown kind {other}"),
        }
    }
}

#[test]
fn http_error_body() {
    use ovo_ai::http::error_body::{format_provider_error, normalize_provider_error};
    for case in cases("ai/http-error-body.json") {
        let name = case["name"].as_str().unwrap_or_default();
        let norm = normalize_provider_error(
            case["message"].as_str().unwrap_or_default(),
            case.get("fields"),
        );
        let got = serde_json::json!({
            "status": norm.status,
            "body": norm.body,
            "message": norm.message,
            "messageCarriesBody": norm.message_carries_body,
        });
        // normalizeProviderError omits undefined fields — compare only the
        // keys the upstream record actually produced.
        let expected = &case["expected"];
        for (key, value) in expected.as_object().cloned().unwrap_or_default() {
            assert!(
                js_value_eq(&got[key.as_str()], &value),
                "{name}.{key}: got {:?}, expected {value:?}",
                got[key.as_str()]
            );
        }
        if let Some(expected_format) = case.get("expectedFormat") {
            let prefix = case["formatArgs"].get("prefix").and_then(Value::as_str);
            assert_eq!(
                format_provider_error(&norm, prefix),
                expected_format.as_str().unwrap_or_default(),
                "{name}"
            );
        }
    }
}

#[test]
fn http_headers() {
    for case in cases("ai/http-headers.json") {
        let name = case["name"].as_str().unwrap_or_default();
        let sources_value = case["sources"].as_array().cloned().unwrap_or_default();
        let sources: Vec<Option<&serde_json::Map<String, Value>>> =
            sources_value.iter().map(Value::as_object).collect();
        let got = provider_headers_to_record(&sources);
        assert_eq!(
            got.map_or(Value::Null, Value::Object),
            case["expected"],
            "{name}"
        );
    }
}

#[test]
fn http_proxy() {
    for case in cases("ai/http-proxy.json") {
        let name = case["name"].as_str().unwrap_or_default();
        let env = case["env"].as_object().cloned().unwrap_or_default();
        match resolve_http_proxy_url_for_target(case["target"].as_str().unwrap_or_default(), &env) {
            Ok(url) => {
                // url.href carries a trailing "/" — the Rust side compares
                // the authority+path without it.
                let got = url
                    .as_ref()
                    .map(|u| u.to_string().trim_end_matches('/').to_owned());
                assert_eq!(
                    got.map_or(Value::Null, Value::from),
                    if case["expected"].is_null() {
                        Value::Null
                    } else {
                        Value::from(
                            case["expected"]
                                .as_str()
                                .unwrap_or_default()
                                .trim_end_matches('/'),
                        )
                    },
                    "{name}"
                );
            }
            Err(e) => {
                assert_eq!(
                    e.message,
                    case["expectedError"].as_str().unwrap_or_default(),
                    "{name}"
                );
            }
        }
    }
}
