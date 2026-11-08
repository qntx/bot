// Arbitrary schema plus arbitrary JSON value: compile and coerce must
// never panic.
#![no_main]

use libfuzzer_sys::fuzz_target;

#[derive(Debug, arbitrary::Arbitrary)]
struct Input<'a> {
    schema: &'a str,
    value: &'a str,
}

fuzz_target!(|input: Input| {
    let Ok(schema) = ovo_ai::json::from_str::<serde_json::Value>(input.schema) else {
        return;
    };
    let Ok(value) = ovo_ai::json::from_str::<serde_json::Value>(input.value) else {
        return;
    };
    let Ok(validator) = ovo_ai::schema::ToolSchemaValidator::compile(&schema) else {
        return;
    };
    let coerced = validator.coerce(&value);
    let _ = validator.check(&coerced);
    let _ = validator.errors(&coerced);
});
