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
            Some("grammar-resolve") => {
                let tool: Tool = serde_json::from_value(case["tool"].clone())
                    .unwrap_or_else(|e| panic!("{name}: tool: {e}"));
                let supports = case["supports"].as_bool().unwrap_or(false);
                recorded(
                    ovo_ai::api::constrained_sampling::resolve_grammar_constrained_sampling(
                        &tool, supports,
                    ),
                )
            }
            Some("grammar-create-properties") => {
                // The case stores the tool list the generator used.
                let tools: Vec<Tool> = serde_json::from_value(case["tools"].clone())
                    .unwrap_or_else(|e| panic!("{name}: tools: {e}"));
                let result =
                    ovo_ai::api::constrained_sampling::create_grammar_tool_input_properties(
                        &tools, true,
                    );
                match result {
                    Ok(map) => {
                        serde_json::json!({"result": map.into_iter().map(|(k, v)| serde_json::json!([k, v])).collect::<Vec<_>>()})
                    }
                    Err(e) => serde_json::json!({"error": e.to_string()}),
                }
            }
            Some("grammar-input" | "grammar-input-error") => {
                let args: Map<String, Value> = serde_json::from_value(case["arguments"].clone())
                    .unwrap_or_else(|e| panic!("{name}: args: {e}"));
                recorded(ovo_ai::api::constrained_sampling::grammar_tool_input(
                    "t", &args, "code",
                ))
            }
            Some("grammar-buffer") => {
                let mut buffer =
                    ovo_ai::api::constrained_sampling::GrammarToolInputJsonBuffer::default();
                let ops = case["ops"].as_array().cloned().unwrap_or_default();
                for op in &ops {
                    let kind = op[0].as_str().unwrap_or_default();
                    let property = op[1].as_str().unwrap_or_default();
                    let outcome = if kind == "append" {
                        buffer.append_delta(property, op[2].as_str().unwrap_or_default())
                    } else {
                        buffer.close(property, op[3].as_str())
                    };
                    let actual = recorded(outcome);
                    let expected_op = op.get(4).cloned().unwrap_or(Value::Null);
                    assert!(
                        js_value_eq(&actual, &expected_op),
                        "{name}: op {kind} {property}: got {actual} want {expected_op}",
                    );
                }
                Value::Object(Map::new())
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
