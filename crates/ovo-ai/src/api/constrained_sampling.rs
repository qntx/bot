//! JSON-schema constrained sampling for tool declarations: whether a tool is
//! sent in a provider's strict mode and the strict subset of its schema.
//!
//! Ported from earendil-works/pi packages/ai/src/api/constrained-sampling.ts
//! (MIT), including the `OpenAI` grammar tool variants.

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

/// `GrammarConstrainedSampling`: the grammar variant a tool is sent with.
#[derive(Clone, Debug, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GrammarConstrainedSampling {
    /// `"lark"` | `"regex"`.
    pub format: GrammarFormat,
    /// The grammar text.
    pub definition: String,
    /// The single required string property carrying the grammar input.
    pub input_property: String,
}

/// The `OpenAI` grammar syntax of a variant.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub enum GrammarFormat {
    /// Lark grammar.
    Lark,
    /// Regular-expression grammar.
    Regex,
}

/// `GrammarToolInputJsonBuffer`: a streamed grammar tool input and how much
/// of its JSON arguments text was emitted.
#[derive(Clone, Debug, Default)]
pub struct GrammarToolInputJsonBuffer {
    /// The accumulated `input` text.
    pub input: String,
    /// `{"<property>":"` has been emitted.
    pub started: bool,
    /// The arguments text closed with `"}`.
    pub closed: bool,
}

/// A grammar tool that cannot be sent or replayed; the message is
/// upstream's text.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[error("{0}")]
pub struct GrammarSamplingError(String);

/// `getGrammarToolInput`: the grammar input of a replayed tool call.
///
/// # Errors
/// Returns upstream's message when the input property is not a string.
pub fn grammar_tool_input<'a>(
    tool_name: &str,
    arguments: &'a Map<String, Value>,
    input_property: &str,
) -> Result<&'a str, GrammarSamplingError> {
    let input = arguments.get(input_property);
    input.and_then(Value::as_str).ok_or_else(|| {
        GrammarSamplingError(format!(
            "Grammar tool call \"{tool_name}\" requires argument \"{input_property}\" to be a string."
        ))
    })
}

impl GrammarToolInputJsonBuffer {
    /// `appendGrammarToolInputJsonDelta`: append `input_delta` and return
    /// the matching piece of the JSON arguments text, or `None` for an empty
    /// delta. Costs time linear in the delta, never in the input so far.
    ///
    /// # Errors
    /// Returns upstream's message when the buffer is already closed.
    pub fn append_delta(
        &mut self,
        input_property: &str,
        input_delta: &str,
    ) -> Result<Option<String>, GrammarSamplingError> {
        if self.closed {
            return Err(GrammarSamplingError(format!(
                "grammar tool input for property \"{input_property}\" changed after it was closed"
            )));
        }
        if input_delta.is_empty() {
            return Ok(None);
        }
        self.input.push_str(input_delta);
        Ok(Some(format!(
            "{}{}",
            self.start(input_property),
            escape_json(input_delta)
        )))
    }

    /// `closeGrammarToolInputJson`: close the input and return the rest of
    /// the JSON arguments text. `final_input`, when the provider sends one,
    /// must continue the input streamed so far; closing again with the same
    /// input returns `None`.
    ///
    /// # Errors
    /// Returns upstream's message when the final input does not continue
    /// the streamed one, or changes after the buffer closed.
    pub fn close(
        &mut self,
        input_property: &str,
        final_input: Option<&str>,
    ) -> Result<Option<String>, GrammarSamplingError> {
        if self.closed {
            return match final_input {
                None => Ok(None),
                Some(final_input) if final_input == self.input => Ok(None),
                Some(_) => Err(GrammarSamplingError(format!(
                    "grammar tool input for property \"{input_property}\" changed after it was closed"
                ))),
            };
        }
        if let Some(final_input) = final_input
            && !final_input.starts_with(&self.input)
        {
            return Err(GrammarSamplingError(format!(
                "grammar tool input for property \"{input_property}\" changed non-monotonically"
            )));
        }
        let rest = final_input.map_or("", |input| &input[self.input.len()..]);
        if let Some(final_input) = final_input {
            final_input.clone_into(&mut self.input);
        }
        self.closed = true;
        Ok(Some(format!(
            "{}{}\"}}",
            self.start(input_property),
            escape_json(rest)
        )))
    }

    /// `grammarToolInputJsonStart`: the opening of the JSON arguments text,
    /// once.
    fn start(&mut self, input_property: &str) -> String {
        if self.started {
            return String::new();
        }
        self.started = true;
        format!("{{\"{}\":\"", escape_json(input_property))
    }
}

/// `JSON.stringify(text).slice(1, -1)`: the escaped string without quotes.
fn escape_json(text: &str) -> String {
    let mut quoted = crate::json::to_string(text).unwrap_or_default();
    quoted.pop();
    if !quoted.is_empty() {
        quoted.remove(0);
    }
    quoted
}

/// `inferGrammarInputProperty`: the single required string property of
/// the tool's `parameters` object.
fn infer_grammar_input_property(tool: &Tool) -> Result<String, GrammarSamplingError> {
    let schema = &tool.parameters;
    if schema.get("type").and_then(Value::as_str) != Some("object") {
        return Err(GrammarSamplingError(String::from(
            "grammar constrained sampling requires an object parameter schema",
        )));
    }
    let required = schema.get("required").and_then(Value::as_array);
    let input_property = required.and_then(|r| {
        (r.len() == 1)
            .then(|| r.first().and_then(Value::as_str))
            .flatten()
    });
    let Some(input_property) = input_property else {
        return Err(GrammarSamplingError(String::from(
            "grammar constrained sampling requires exactly one required string property",
        )));
    };
    let property = schema.get("properties").and_then(|p| p.get(input_property));
    let Some(property) = property else {
        return Err(GrammarSamplingError(format!(
            "grammar constrained sampling requires a properties entry for {input_property}"
        )));
    };
    if property.get("type").and_then(Value::as_str) != Some("string") {
        return Err(GrammarSamplingError(format!(
            "grammar constrained sampling property {input_property} must have type string"
        )));
    }
    Ok(input_property.to_owned())
}

/// `resolveGrammarConstrainedSampling`: the grammar variant to send a tool
/// with, or `None` when it has no grammar config or the provider has no
/// grammar tools.
///
/// # Errors
/// Returns upstream's message when the tool has no usable variant or its
/// schema has no single required string property.
pub fn resolve_grammar_constrained_sampling(
    tool: &Tool,
    supports_openai_grammar_tools: bool,
) -> Result<Option<GrammarConstrainedSampling>, GrammarSamplingError> {
    let Some(ConstrainedSampling::Config(ConstrainedSamplingConfig::Grammar { variants })) =
        &tool.constrained_sampling
    else {
        return Ok(None);
    };
    if !supports_openai_grammar_tools {
        return Ok(None);
    }
    let lark = variants.openai_lark.as_deref();
    let regex = variants.openai_regex.as_deref();
    let has_lark = lark.is_some_and(|text| !crate::text::js_trim(text).is_empty());
    let has_regex = regex.is_some_and(|text| !crate::text::js_trim(text).is_empty());
    if !has_lark && !has_regex {
        return Err(GrammarSamplingError(format!(
            "Tool \"{}\" cannot use grammar constrained sampling: no supported grammar variant was provided.",
            tool.name
        )));
    }
    let definition = if has_lark {
        lark.unwrap_or_default()
    } else {
        regex.unwrap_or_default()
    };
    infer_grammar_input_property(tool)
        .map(|input_property| {
            Some(GrammarConstrainedSampling {
                format: if has_lark {
                    GrammarFormat::Lark
                } else {
                    GrammarFormat::Regex
                },
                definition: definition.to_owned(),
                input_property,
            })
        })
        .map_err(|error| {
            GrammarSamplingError(format!(
                "Tool \"{}\" cannot use grammar constrained sampling: {error}.",
                tool.name
            ))
        })
}

/// `createGrammarToolInputProperties`: the input property of every tool
/// sent as a grammar tool, by tool name.
///
/// # Errors
/// Returns the first tool's grammar error.
pub fn create_grammar_tool_input_properties(
    tools: &[Tool],
    supports_openai_grammar_tools: bool,
) -> Result<std::collections::BTreeMap<String, String>, GrammarSamplingError> {
    let mut properties = std::collections::BTreeMap::new();
    for tool in tools {
        if let Some(grammar) =
            resolve_grammar_constrained_sampling(tool, supports_openai_grammar_tools)?
        {
            properties.insert(tool.name.clone(), grammar.input_property);
        }
    }
    Ok(properties)
}
