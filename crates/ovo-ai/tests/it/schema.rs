/// parity: AI-SCHEMA-COERCE — `compile(..., {coerce:true}).coerce(value)`
/// replays `pi-schema`'s three-pass coercion.
/// parity: AI-SCHEMA-VALIDATE — `parse` (coerce + check + errors)
/// replays `validateToolArguments`' upstream result.
use serde_json::Value;

use crate::support::js_value_eq;

fn read_vectors<T: serde::de::DeserializeOwned>(path: &str) -> Vec<T> {
    let text = std::fs::read_to_string(format!(
        "{}/../../vectors/{path}",
        env!("CARGO_MANIFEST_DIR")
    ))
    .unwrap_or_else(|e| panic!("read {path}: {e}"));
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("parse {path}: {e}"))
}

#[test]
fn schema_coerce() {
    #[derive(serde::Deserialize)]
    struct Case {
        name: String,
        schema: Value,
        input: Value,
        expected: Value,
    }
    let cases: Vec<Case> = read_vectors("ai/schema-coerce.json");
    for case in cases {
        let validator = ovo_ai::schema::ToolSchemaValidator::compile(&case.schema)
            .unwrap_or_else(|e| panic!("{name}: compile: {e}", name = case.name));
        let got = validator.coerce(&case.input);
        assert!(
            js_value_eq(&got, &case.expected),
            "{name}: got {got:?}, expected {:?}",
            case.expected,
            name = case.name
        );
    }
}

#[test]
fn schema_validate() {
    #[derive(serde::Deserialize)]
    struct Case {
        name: String,
        tool: String,
        schema: Value,
        arguments: serde_json::Map<String, Value>,
        expected: Value,
    }
    let cases: Vec<Case> = read_vectors("ai/schema-validate.json");
    for case in cases {
        let validator = ovo_ai::schema::ToolSchemaValidator::compile(&case.schema)
            .unwrap_or_else(|e| panic!("{name}: compile: {e}", name = case.name));
        let args = Value::Object(case.arguments.clone());
        match case.expected.get("kind").and_then(Value::as_str) {
            Some("ok") => {
                let expected_args = case.expected.get("arguments").cloned().unwrap_or_default();
                let result = validator.parse(&args);
                match result {
                    Ok(got) => assert!(
                        js_value_eq(&got, &expected_args),
                        "{name}: got {got:?}, expected {expected_args:?}",
                        name = case.name
                    ),
                    Err(errors) => panic!(
                        "{} ({}): unexpected errors: {:?}",
                        case.name, case.tool, errors
                    ),
                }
            }
            Some("error") => {
                let result = validator.parse(&args);
                let Err(errors) = result else {
                    panic!("{} ({}): expected error, got ok", case.name, case.tool);
                };
                let expected_errors = case.expected.get("errors").cloned().unwrap_or_default();
                let got: Vec<Value> = errors
                    .iter()
                    .map(|e| serde_json::json!({"path": e.path, "keyword": e.keyword, "message": e.message}))
                    .collect();
                let got = Value::Array(got);
                assert!(
                    js_value_eq(&got, &expected_errors),
                    "{name}: got {got:?}, expected {expected_errors:?}",
                    name = case.name
                );
            }
            other => panic!("{}: bad expected kind {:?}", case.name, other),
        }
    }
}
