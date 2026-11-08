//! Replays the `AI-TYPES-SERDE` parity vectors: decode each input into the
//! Rust type named by `kind`, re-encode, and compare by value.

use serde::Deserialize;
use serde_json::Value;

use ovo_ai::types::classifier::ClassifierResult;
use ovo_ai::types::context::Context;
use ovo_ai::types::deferred::DeferredHandle;
use ovo_ai::types::images::AssistantImages;
use ovo_ai::types::message::Message;
use ovo_ai::types::model::AnyModel;
use ovo_ai::types::tool::Tool;
use ovo_ai::types::usage::Usage;

use crate::support::js_value_eq;

#[derive(Debug, Deserialize)]
struct VectorFile {
    cases: Vec<VectorCase>,
}

#[derive(Debug, Deserialize)]
struct VectorCase {
    name: String,
    kind: String,
    input: Value,
}

fn round_trip<T>(input: &Value) -> Result<Value, String>
where
    T: for<'de> Deserialize<'de> + serde::Serialize,
{
    let text = json_text(input)?;
    let decoded: T = ovo_ai::json::from_str(&text).map_err(|error| format!("decode: {error}"))?;
    let output = ovo_ai::json::to_string(&decoded).map_err(|error| format!("encode: {error}"))?;
    serde_json::from_str(&output).map_err(|error| format!("reparse: {error}"))
}

fn json_text(value: &Value) -> Result<String, String> {
    ovo_ai::json::to_string(value).map_err(|error| format!("input encode: {error}"))
}

/// parity: AI-TYPES-SERDE
#[test]
fn types_round_trip_by_value() {
    let text = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../vectors/ai/types.json"
    ));
    let file: VectorFile = match serde_json::from_str(text) {
        Ok(file) => file,
        Err(error) => panic!("invalid vector file: {error}"),
    };
    assert!(!file.cases.is_empty(), "vector file has no cases");
    let mut names = std::collections::HashSet::new();
    let mut counts: std::collections::BTreeMap<&str, usize> = std::collections::BTreeMap::new();
    for case in &file.cases {
        assert!(
            names.insert(&case.name),
            "duplicate case name {}",
            case.name
        );
        *counts.entry(case.kind.as_str()).or_default() += 1;
        let actual = match case.kind.as_str() {
            "model" => round_trip::<AnyModel>(&case.input),
            "message" => round_trip::<Message>(&case.input),
            "usage" => round_trip::<Usage>(&case.input),
            "tool" => round_trip::<Tool>(&case.input),
            "context" => round_trip::<Context>(&case.input),
            "deferred" => round_trip::<DeferredHandle>(&case.input),
            "assistant-images" => round_trip::<AssistantImages>(&case.input),
            "classifier-result" => round_trip::<ClassifierResult>(&case.input),
            other => panic!("unknown case kind {other:?}"),
        };
        let output = match actual {
            Ok(output) => output,
            Err(error) => panic!("case {} ({}) failed: {error}", case.name, case.kind),
        };
        assert!(
            js_value_eq(&case.input, &output),
            "case {} differs\ninput:  {}\noutput: {}",
            case.name,
            ovo_ai::json::to_string(&case.input).unwrap_or_default(),
            ovo_ai::json::to_string(&output).unwrap_or_default(),
        );
        if case.kind == "model" {
            let (Some(input_obj), Some(output_obj)) = (case.input.as_object(), output.as_object())
            else {
                panic!("case {} is not an object", case.name);
            };
            for key in input_obj.keys() {
                assert!(
                    output_obj.contains_key(key),
                    "case {} dropped key {key:?}",
                    case.name
                );
            }
        }
    }
    println!("case counts: {counts:?}");
}
