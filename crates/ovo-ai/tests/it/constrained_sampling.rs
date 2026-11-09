//! Replays the constrained-sampling vectors generated from the published
//! pi-ai `api/constrained-sampling.js`.

use ovo_ai::api::constrained_sampling::{
    make_strict_json_schema, resolve_json_schema_strict_sampling,
};
use ovo_ai::types::tool::Tool;
use serde_json::{Map, Value};

use crate::support::js_value_eq;

fn cases() -> Vec<Value> {
    let text = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../vectors/ai/constrained-sampling.json"
    ));
    let document: Value = serde_json::from_str(text).unwrap_or_else(|e| panic!("vector file: {e}"));
    document["cases"].as_array().cloned().unwrap_or_default()
}

/// The vector's `{result}` or `{error}` for an outcome.
fn recorded<T: serde::Serialize, E: std::fmt::Display>(outcome: Result<T, E>) -> Value {
    match outcome {
        Ok(value) => {
            serde_json::json!({ "result": serde_json::to_value(value).unwrap_or(Value::Null) })
        }
        Err(error) => serde_json::json!({ "error": error.to_string() }),
    }
}

/// parity: AI-CONSTRAINED-SAMPLING
#[test]
fn constrained_sampling_matches_upstream() {
    let cases = cases();
    assert!(!cases.is_empty(), "no cases");
    for case in &cases {
        let name = case["name"].as_str().unwrap_or("?");
        let actual = match case["kind"].as_str() {
            Some("makeStrict") => {
                let schema: Map<String, Value> = serde_json::from_value(case["schema"].clone())
                    .unwrap_or_else(|e| panic!("{name}: schema: {e}"));
                let outcome = make_strict_json_schema(&schema, None);
                let mut value = recorded(outcome);
                // Key order is part of the request bytes, so compare the
                // serialized text as well as the value.
                if let (Some(got), Some(want)) = (value.get("result"), case.get("result")) {
                    assert_eq!(
                        ovo_ai::json::to_string(got).ok(),
                        ovo_ai::json::to_string(want).ok(),
                        "{name}: key order"
                    );
                }
                value
                    .as_object_mut()
                    .map(std::mem::take)
                    .map_or(Value::Null, Value::Object)
            }
            Some("resolve") => {
                let tool: Tool = serde_json::from_value(case["tool"].clone())
                    .unwrap_or_else(|e| panic!("{name}: tool: {e}"));
                let supports = case["supportsStrictMode"].as_bool().unwrap_or(false);
                recorded(resolve_json_schema_strict_sampling(&tool, supports, None))
            }
            other => panic!("{name}: unknown kind {other:?}"),
        };
        let mut expected = Map::new();
        for key in ["result", "error"] {
            if let Some(value) = case.get(key) {
                expected.insert(key.to_owned(), value.clone());
            }
        }
        assert!(
            js_value_eq(&actual, &Value::Object(expected.clone())),
            "{name}\n  got:  {actual}\n  want: {}",
            Value::Object(expected)
        );
    }
}
