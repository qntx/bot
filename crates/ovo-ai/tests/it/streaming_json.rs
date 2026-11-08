//! Replays the `AI-JSON-PARTIAL` parity vectors: streaming changes, repair,
//! and finished-argument parsing produced by `StreamingJsonParser`,
//! `ToolCallArgumentsStream`, `repairJson`, `parseJsonWithRepair` and
//! `parseToolCallArguments` upstream.

use serde::Deserialize;
use serde_json::Value;

use ovo_ai::json::repair::{parse_json_with_repair, parse_tool_call_arguments, repair_json};
use ovo_ai::json::streaming::StreamingJsonParser;

use crate::support::js_value_eq;

#[derive(Debug, Deserialize)]
struct VectorFile {
    cases: Vec<VectorCase>,
}

#[derive(Debug, Deserialize)]
struct VectorCase {
    name: String,
    kind: String,
    #[serde(default)]
    pieces: Vec<String>,
    #[serde(default)]
    changes: Vec<Vec<Value>>,
    #[serde(default)]
    finish: Option<Finish>,
    #[serde(default)]
    input: String,
    #[serde(default)]
    output: Option<Value>,
    #[serde(default)]
    classes: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct Finish {
    arguments: Value,
    #[serde(default)]
    #[serde(rename = "argumentsError")]
    arguments_error: Option<String>,
}

fn load_cases() -> Vec<VectorCase> {
    let text = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../vectors/ai/streaming-json.json"
    ));
    match ovo_ai::json::from_str::<VectorFile>(text) {
        Ok(file) => file.cases,
        Err(error) => panic!("invalid vector file: {error}"),
    }
}

fn replay_stream(case: &VectorCase) {
    let mut parser = StreamingJsonParser::new();
    let mut target = Value::Null;
    // `lone-surrogates`: upstream reports an escaped surrogate half in the
    // piece it appeared; ovo defers it pending the partner (UTF-8 cannot hold
    // half a pair), so only the accumulated value is compared for these.
    let per_piece = !case.classes.iter().any(|c| c == "lone-surrogates");
    for (index, piece) in case.pieces.iter().enumerate() {
        let changes = parser.append(piece);
        if per_piece {
            let recorded = case.changes.get(index);
            assert_eq!(
                changes.len(),
                recorded.map_or(0, Vec::len),
                "case {} piece {index}: {} changes vs recorded",
                case.name,
                changes.len(),
            );
            let Some(recorded) = recorded else {
                continue;
            };
            for (change, expected) in changes.iter().zip(recorded.iter()) {
                let encoded = match ovo_ai::json::to_string(change) {
                    Ok(text) => text,
                    Err(error) => panic!("case {} change encode failed: {error}", case.name),
                };
                let actual: Value = match ovo_ai::json::from_str(&encoded) {
                    Ok(value) => value,
                    Err(error) => panic!("case {} change decode failed: {error}", case.name),
                };
                assert!(
                    js_value_eq(&actual, expected),
                    "case {} piece {index}: change {actual} != expected {expected}",
                    case.name,
                );
            }
        }
        for change in &changes {
            change.apply(&mut target);
        }
    }
    if !per_piece {
        // The accumulated value equals the whole-text parse, repair
        // semantics included.
        if let Ok(expected) = parse_json_with_repair(&case.pieces.concat()) {
            assert!(
                js_value_eq(&target, &expected),
                "case {} applied value differs",
                case.name
            );
        }
    }
    let Some(finish) = &case.finish else {
        return;
    };
    let finished = parse_tool_call_arguments(&case.pieces.concat());
    assert!(
        js_value_eq(&Value::from(finished.arguments.clone()), &finish.arguments),
        "case {} arguments differ",
        case.name,
    );
    if finish.arguments_error.is_some() {
        let Some(actual) = &finished.arguments_error else {
            panic!("case {} missing argumentsError", case.name);
        };
        // engine-error-text: the V8 message part is masked; the template
        // tail must match verbatim and the message must be non-empty.
        let suffix = format!("\n\nReceived arguments:\n{}", case.pieces.concat());
        assert!(
            actual.ends_with(&suffix) && actual.len() > suffix.len(),
            "case {} argumentsError {actual:?} lacks expected suffix",
            case.name,
        );
    } else {
        assert!(
            finished.arguments_error.is_none(),
            "case {} unexpected argumentsError {:?}",
            case.name,
            finished.arguments_error,
        );
    }
}

fn replay_text(case: &VectorCase) {
    match case.kind.as_str() {
        "repair" => {
            let expected = case.output.as_ref().and_then(Value::as_str);
            let Some(expected) = expected else {
                panic!("case {} has no repair output", case.name);
            };
            assert_eq!(repair_json(&case.input), expected, "case {}", case.name);
        }
        "parse-repair" => match parse_json_with_repair(&case.input) {
            Ok(value) => {
                let Some(expected) = &case.output else {
                    panic!("case {} parsed but expected an error", case.name);
                };
                assert!(
                    js_value_eq(&value, expected),
                    "case {} value differs",
                    case.name
                );
            }
            Err(error) => {
                assert!(
                    case.classes.iter().any(|c| c == "engine-error-text"),
                    "case {} unexpected parse error: {error}",
                    case.name,
                );
            }
        },
        other => panic!("unknown text case kind {other:?}"),
    }
}

/// parity: AI-JSON-PARTIAL
#[test]
fn streaming_json_replays_pieces() {
    let cases = load_cases();
    assert!(!cases.is_empty(), "vector file has no cases");
    let mut names = std::collections::HashSet::new();
    for case in &cases {
        assert!(
            names.insert(&case.name),
            "duplicate case name {}",
            case.name
        );
        match case.kind.as_str() {
            "stream" => replay_stream(case),
            "repair" | "parse-repair" => replay_text(case),
            other => panic!("unknown case kind {other:?}"),
        }
    }
}
