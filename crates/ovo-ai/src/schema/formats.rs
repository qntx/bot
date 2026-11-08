//! Format names pi-schema registers.
//!
//! Checks run through `jsonschema`'s built-in format validation
//! (class `schema-formats`: decisions may differ from `TypeBox`'s checks
//! on edge cases; messages match upstream's template).

use std::collections::HashMap;
use std::sync::LazyLock;

use serde_json::Value;

/// The format names upstream registers (`pi-schema/formats`, every
/// 2020-12 format plus `url` and `json-pointer-uri-fragment`).
const NAMES: &[&str] = &[
    "date",
    "date-time",
    "duration",
    "email",
    "hostname",
    "idn-email",
    "idn-hostname",
    "ipv4",
    "ipv6",
    "iri",
    "iri-reference",
    "json-pointer",
    "json-pointer-uri-fragment",
    "regex",
    "relative-json-pointer",
    "time",
    "uri",
    "uri-reference",
    "uri-template",
    "url",
    "uuid",
];

/// Whether `name` is a registered format.
pub(crate) fn is_format(name: &str) -> bool {
    NAMES.contains(&name)
}

/// Whether `text` passes format `name` under `jsonschema`'s built-in
/// checks (`None` when `name` cannot compile, treated as passing like
/// upstream's unknown formats).
pub(crate) fn format_is_valid(name: &str, text: &str) -> bool {
    static VALIDATORS: LazyLock<HashMap<&'static str, Option<jsonschema::Validator>>> =
        LazyLock::new(|| {
            NAMES
                .iter()
                .map(|name| {
                    let validator = jsonschema::Validator::options()
                        .with_draft(jsonschema::Draft::Draft202012)
                        .should_validate_formats(true)
                        .build(&serde_json::json!({"format": name}))
                        .ok();
                    (*name, validator)
                })
                .collect()
        });
    match VALIDATORS.get(name) {
        Some(Some(validator)) => validator.is_valid(&Value::from(text)),
        _ => true,
    }
}
