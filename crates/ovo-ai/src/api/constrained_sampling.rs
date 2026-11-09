//! JSON-schema constrained sampling for tool declarations: whether a tool is
//! sent in a provider's strict mode and the strict subset of its schema.
//!
//! Ported from earendil-works/pi packages/ai/src/api/constrained-sampling.ts
//! (MIT). The grammar variants arrive with the `OpenAI` adapters.

use serde_json::{Map, Value, json};

use crate::types::tool::{ConstrainedSampling, ConstrainedSamplingConfig, StrictMode, Tool};

/// Whether a provider's strict mode rejects this schema keyword with this value.
pub type UnsupportedKeyword = fn(&str, &Value) -> bool;

/// Keywords no provider's strict mode accepts.
const UNSUPPORTED_STRICT_SCHEMA_KEYS: [&str; 16] = [
    "$ref",
    "$defs",
    "definitions",
    "allOf",
    "oneOf",
    "patternProperties",
    "dependentSchemas",
    "dependencies",
    "unevaluatedProperties",
    "propertyNames",
    "contains",
    "prefixItems",
    "not",
    "if",
    "then",
    "else",
];

/// Messages Claude returns when a request's strict tools exceed its limits
/// (observed: 20 strict tools, 16 optional or union-typed parameters).
const STRICT_TOOL_LIMIT_ERROR_MESSAGES: [&str; 3] = [
    "The compiled grammar is too large",
    "Too many strict tools (",
    "Schemas contains too many parameters with union types (",
];

/// Why a schema has no strict form; the message is upstream's text.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[error("{0}")]
pub struct UnsupportedStrictSchema(String);

impl UnsupportedStrictSchema {
    fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

/// A tool that requires strict sampling the provider cannot give it.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[error("{0}")]
pub struct StrictSamplingRequired(String);

/// `makeStrictJsonSchema`: the strict subset of `schema` (every property
/// required, optional ones made nullable, no additional properties).
///
/// # Errors
/// Returns the reason when the schema uses a construct strict mode rejects.
pub fn make_strict_json_schema(
    schema: &Map<String, Value>,
    unsupported: Option<UnsupportedKeyword>,
) -> Result<Map<String, Value>, UnsupportedStrictSchema> {
    let mut root = Value::Object(schema.clone());
    make_node_strict(&mut root, unsupported)?;
    match root {
        Value::Object(map) if map.get("type").and_then(Value::as_str) == Some("object") => Ok(map),
        _ => Err(UnsupportedStrictSchema::new(
            "root schema must have type object",
        )),
    }
}

fn make_node_strict(
    schema: &mut Value,
    unsupported: Option<UnsupportedKeyword>,
) -> Result<(), UnsupportedStrictSchema> {
    let Value::Object(node) = schema else {
        return Err(UnsupportedStrictSchema::new(
            "boolean schemas are unsupported",
        ));
    };
    if let Some(key) = UNSUPPORTED_STRICT_SCHEMA_KEYS
        .iter()
        .find(|key| node.contains_key(**key))
    {
        return Err(UnsupportedStrictSchema::new(format!(
            "{key} schemas are unsupported"
        )));
    }
    if let Some(check) = unsupported
        && let Some((key, value)) = node.iter().find(|(key, value)| check(key, value))
    {
        let value = crate::json::to_string(value).unwrap_or_default();
        return Err(UnsupportedStrictSchema::new(format!(
            "{key}: {value} is unsupported"
        )));
    }
    if let Some(any_of) = node.get_mut("anyOf") {
        let variants = match any_of {
            Value::Array(variants) if !variants.is_empty() => variants,
            _ => {
                return Err(UnsupportedStrictSchema::new(
                    "anyOf must contain at least one schema",
                ));
            }
        };
        for variant in variants {
            if is_structured(variant) {
                return Err(UnsupportedStrictSchema::new(
                    "object and array unions are unsupported",
                ));
            }
            make_node_strict(variant, unsupported)?;
        }
    }
    if let Some(items) = node.get_mut("items") {
        if items.is_array() {
            return Err(UnsupportedStrictSchema::new(
                "tuple schemas are unsupported",
            ));
        }
        make_node_strict(items, unsupported)?;
    }
    let is_object = node.get("type").and_then(Value::as_str) == Some("object");
    if node.contains_key("properties") && !is_object {
        return Err(UnsupportedStrictSchema::new(
            "properties require type object",
        ));
    }
    if !is_object {
        return Ok(());
    }
    make_object_strict(node, unsupported)
}

/// The object part of `makeJsonSchemaNodeStrict`.
fn make_object_strict(
    node: &mut Map<String, Value>,
    unsupported: Option<UnsupportedKeyword>,
) -> Result<(), UnsupportedStrictSchema> {
    if node
        .get("additionalProperties")
        .is_some_and(|value| value != &Value::Bool(false))
    {
        return Err(UnsupportedStrictSchema::new(
            "schema-valued or true additionalProperties is unsupported",
        ));
    }
    if node
        .get("properties")
        .is_some_and(|value| !value.is_object())
    {
        return Err(UnsupportedStrictSchema::new(
            "object properties must be a schema map",
        ));
    }
    let required: Vec<String> = match node.get("required") {
        None => Vec::new(),
        Some(Value::Array(keys)) if keys.iter().all(Value::is_string) => keys
            .iter()
            .filter_map(Value::as_str)
            .map(str::to_owned)
            .collect(),
        Some(_) => {
            return Err(UnsupportedStrictSchema::new(
                "object required must be a string array",
            ));
        }
    };
    let names: Vec<String> = node
        .get("properties")
        .and_then(Value::as_object)
        .map(|properties| properties.keys().cloned().collect())
        .unwrap_or_default();
    if required.iter().any(|key| !names.contains(key)) {
        return Err(UnsupportedStrictSchema::new(
            "required contains an unknown property",
        ));
    }
    if let Some(Value::Object(properties)) = node.get_mut("properties") {
        for (key, property) in properties.iter_mut() {
            make_node_strict(property, unsupported)?;
            if !required.contains(key) && !allows_null(property) {
                let inner = std::mem::take(property);
                *property = json!({ "anyOf": [inner, { "type": "null" }] });
            }
        }
    }
    node.insert(
        String::from("required"),
        Value::Array(names.into_iter().map(Value::from).collect()),
    );
    node.insert(String::from("additionalProperties"), Value::Bool(false));
    Ok(())
}

/// `isStructuredSchema`: an object or array schema.
fn is_structured(schema: &Value) -> bool {
    let Value::Object(node) = schema else {
        return false;
    };
    let has_type = |name: &str| match node.get("type") {
        Some(Value::String(kind)) => kind == name,
        Some(Value::Array(kinds)) => kinds.iter().any(|kind| kind.as_str() == Some(name)),
        _ => false,
    };
    has_type("object")
        || has_type("array")
        || node.contains_key("properties")
        || node.contains_key("items")
}

/// `schemaAllowsNull`.
fn allows_null(schema: &Value) -> bool {
    let Value::Object(node) = schema else {
        return false;
    };
    let type_null = match node.get("type") {
        Some(Value::String(kind)) => kind == "null",
        Some(Value::Array(kinds)) => kinds.iter().any(|kind| kind.as_str() == Some("null")),
        _ => false,
    };
    type_null
        || node.get("const") == Some(&Value::Null)
        || node
            .get("enum")
            .and_then(Value::as_array)
            .is_some_and(|values| values.contains(&Value::Null))
        || node
            .get("anyOf")
            .and_then(Value::as_array)
            .is_some_and(|variants| variants.iter().any(allows_null))
}

/// `prefersJsonSchemaStrictSampling`: the tool asks for strict sampling only
/// when possible (`strict: "prefer"`).
#[must_use]
pub const fn prefers_json_schema_strict_sampling(tool: &Tool) -> bool {
    matches!(
        tool.constrained_sampling,
        Some(ConstrainedSampling::Config(
            ConstrainedSamplingConfig::JsonSchema {
                strict: StrictMode::Prefer
            }
        ))
    )
}

/// `isStrictToolLimitErrorMessage`.
#[must_use]
pub fn is_strict_tool_limit_error_message(message: &str) -> bool {
    STRICT_TOOL_LIMIT_ERROR_MESSAGES
        .iter()
        .any(|limit| message.contains(limit))
}

/// `getJsonSchemaToolParameters`: the strict schema for strict tools, the
/// declared one otherwise.
///
/// # Errors
/// Returns the reason when `strict` is set but the schema has no strict form.
pub fn json_schema_tool_parameters(
    tool: &Tool,
    strict: Option<bool>,
) -> Result<Map<String, Value>, UnsupportedStrictSchema> {
    if strict == Some(true) {
        make_strict_json_schema(&tool.parameters, None)
    } else {
        Ok(tool.parameters.clone())
    }
}

/// `resolveJsonSchemaStrictSampling`: `Some(true)` to send the tool in strict
/// mode, `None` to send it normally.
///
/// # Errors
/// Returns upstream's error text when the tool requires strict sampling and
/// the provider or its schema cannot provide it.
pub fn resolve_json_schema_strict_sampling(
    tool: &Tool,
    supports_strict_mode: bool,
    unsupported: Option<UnsupportedKeyword>,
) -> Result<Option<bool>, StrictSamplingRequired> {
    let Some(ConstrainedSampling::Config(ConstrainedSamplingConfig::JsonSchema { strict })) =
        &tool.constrained_sampling
    else {
        return Ok(None);
    };
    if supports_strict_mode {
        return match make_strict_json_schema(&tool.parameters, unsupported) {
            Ok(_) => Ok(Some(true)),
            Err(_) if *strict != StrictMode::Require => Ok(None),
            Err(error) => Err(StrictSamplingRequired(format!(
                "Tool \"{}\" requires JSON-schema constrained sampling, but {error}.",
                tool.name
            ))),
        };
    }
    if *strict == StrictMode::Require {
        return Err(StrictSamplingRequired(format!(
            "Tool \"{}\" requires JSON-schema constrained sampling, but strict tools are unsupported.",
            tool.name
        )));
    }
    Ok(None)
}
