//! Coercion: upstream `schema/src/coerce.ts` — three passes
//! (drop optional nulls → TypeBox-style conversion → primitive
//! coercion), copy-on-write along changed paths only.
#![expect(
    clippy::excessive_nesting,
    clippy::collapsible_if,
    reason = "the port follows upstream's keyword-dispatch structure"
)]
use std::borrow::Cow;
use std::collections::HashMap;

use serde_json::{Map, Value};

use super::{ToolSchemaValidator, is_schema_object, js_object_is, js_strict_eq};
use crate::text::js_trim;

const CHARS_UNDEFINED: &str = "undefined";

/// `Coercer`: coerces values for one compiled schema.
pub(crate) struct Coercer<'v> {
    validator: &'v ToolSchemaValidator,
    /// `(node, value)` pairs pass 2 is converting through a re-entry
    /// point — a pair already on the stack leaves the value unchanged.
    entered: Vec<(usize, Value)>,
    /// How often a re-entry was cut short in this `coerce` call.
    cuts: u32,
    /// Results of `$ref` targets by (node ptr, value) within one call.
    memo: HashMap<(usize, u64), Value>,
}

impl<'v> Coercer<'v> {
    /// One coercer over `validator`'s schema.
    pub(crate) fn new(validator: &'v ToolSchemaValidator) -> Self {
        Self {
            validator,
            entered: Vec::new(),
            cuts: 0,
            memo: HashMap::new(),
        }
    }

    /// `coerce(value)`: all three passes in order.
    pub(crate) fn coerce(&mut self, value: &Value) -> Value {
        let base = self.validator.root_base().to_owned();
        let schema = self.validator.original().clone();
        let without_nulls = self.drop_optional_nulls(value, &schema, &base);
        let converted = self.convert_only(&without_nulls);
        self.coerce_types(&converted, &schema, &base).into_owned()
    }

    /// `convertOnly`: pass 2 alone (conversion rules).
    pub(crate) fn convert_only(&mut self, value: &Value) -> Value {
        let base = self.validator.root_base().to_owned();
        let schema = self.validator.original().clone();
        let out = self.convert(&schema, &base, value).into_owned();
        self.memo.clear();
        self.cuts = 0;
        out
    }

    fn passes(&self, node: &Value, base: &str, value: &Value) -> Option<bool> {
        let _ = base;
        self.validator.check_node(node, value).ok()
    }

    fn base_of(&self, node: &Value, base: &str) -> String {
        if is_schema_object(node) {
            self.validator.base_of(node, base)
        } else {
            base.to_owned()
        }
    }

    /// A `(node, value)` pair `convert` enters again only through a
    /// `$ref` or a scalar-to-array wrap; memoized per call.
    fn reenter<'a>(
        &mut self,
        node: &'a Value,
        base: &str,
        value: &'a Value,
        memoize: bool,
    ) -> Cow<'a, Value> {
        let key = std::ptr::from_ref(node) as usize;
        for (k, v) in &self.entered {
            if *k == key && js_object_is(v, value) {
                self.cuts += 1;
                return Cow::Borrowed(value);
            }
        }
        if memoize && let Some(result) = self.memo.get(&(key, value_key(value))) {
            return Cow::Owned(result.clone());
        }
        let cuts = self.cuts;
        self.entered.push((key, value.clone()));
        let result = self.convert(node, base, value).into_owned();
        self.entered.pop();
        if memoize && self.cuts == cuts {
            self.memo.insert((key, value_key(value)), result.clone());
        }
        if js_object_is(&result, value) {
            Cow::Borrowed(value)
        } else {
            Cow::Owned(result)
        }
    }

    // Pass 1: optional nulls -------------------------------------------------

    fn drop_optional_nulls<'a>(
        &mut self,
        value: &'a Value,
        schema: &Value,
        base: &str,
    ) -> Cow<'a, Value> {
        if !is_schema_object(schema) {
            return Cow::Borrowed(value);
        }
        let b = self.base_of(schema, base);
        if let Value::Array(items) = value {
            let mut out = items.clone();
            for (i, item) in items.iter().enumerate() {
                let Some(item_schema) = item_schema_at(schema, i) else {
                    continue;
                };
                let next = self.drop_optional_nulls(item, item_schema, &b);
                if !js_strict_eq(&next, item) {
                    if let Some(slot) = out.get_mut(i) {
                        *slot = next.into_owned();
                    }
                }
            }
            return if same_array(&out, items) {
                Cow::Borrowed(value)
            } else {
                Cow::Owned(Value::Array(out))
            };
        }
        let Value::Object(_) = value else {
            return Cow::Borrowed(value);
        };
        let properties = schema.get("properties").filter(|v| v.is_object());
        let patterns = pattern_entries(schema);
        let additional = schema.get("additionalProperties").filter(|v| v.is_object());
        if patterns.is_empty() && additional.is_none() && properties.is_none() {
            return Cow::Borrowed(value);
        }
        let required: Vec<&str> = schema
            .get("required")
            .and_then(Value::as_array)
            .map(|l| l.iter().filter_map(Value::as_str).collect())
            .unwrap_or_default();
        let mut out = value.as_object().cloned().unwrap_or_default();
        for (key, property) in properties.and_then(Value::as_object).into_iter().flatten() {
            let Some(old) = out.get(key).cloned() else {
                continue;
            };
            let is_ref = is_schema_object(property) && property.get("$ref").is_some();
            if old.is_null()
                && !required.contains(&key.as_str())
                && !is_ref
                && self.passes(property, &b, &Value::Null) == Some(false)
            {
                out.remove(key);
                continue;
            }
            let next = self.drop_optional_nulls(&old, property, &b);
            if !js_strict_eq(&next, &old) {
                out.insert(key.clone(), next.into_owned());
            }
        }
        if !patterns.is_empty() || additional.is_some() {
            let keys: Vec<String> = out.keys().cloned().collect();
            for key in keys {
                let old = out.get(&key).cloned().unwrap_or(Value::Null);
                let mut matched = properties
                    .and_then(Value::as_object)
                    .is_some_and(|m| m.contains_key(&key));
                let mut next = old.clone();
                for (pattern, item) in &patterns {
                    if pattern.is_match(&key) {
                        matched = true;
                        let coerced = self.drop_optional_nulls(&next, item, &b);
                        next = coerced.into_owned();
                    }
                }
                if !matched && let Some(add) = additional {
                    next = self.drop_optional_nulls(&old, add, &b).into_owned();
                }
                if !js_strict_eq(&next, &old) {
                    out.insert(key, next);
                }
            }
        }
        if same_object(&out, value) {
            Cow::Borrowed(value)
        } else {
            Cow::Owned(Value::Object(out))
        }
    }

    // Pass 2: TypeBox's conversion rules --------------------------------------

    fn convert<'a>(&mut self, node: &'a Value, base: &str, value: &'a Value) -> Cow<'a, Value> {
        if !is_schema_object(node) {
            return Cow::Borrowed(value);
        }
        let b = self.base_of(node, base);
        if let Some(reference) = node.get("$ref").and_then(Value::as_str) {
            let Ok(resolved) = self.validator.resolve(reference, &b) else {
                return Cow::Borrowed(value);
            };
            let Some(target) = self.validator.node_at_ptr(&resolved.ptr) else {
                return Cow::Borrowed(value);
            };
            return if is_schema_object(target) {
                // The target may live in `self.validator`'s schema; the
                // re-entered value's identity is the pointer's value.
                let out = self
                    .reenter(target, &resolved.base, value, true)
                    .into_owned();
                if js_object_is(&out, value) {
                    Cow::Borrowed(value)
                } else {
                    Cow::Owned(out)
                }
            } else {
                Cow::Borrowed(value)
            };
        }
        self.convert_node(node, &b, value)
    }

    fn convert_node<'a>(&mut self, node: &'a Value, b: &str, value: &'a Value) -> Cow<'a, Value> {
        if let Some(literal) = node.get("const") {
            return convert_literal(literal, value);
        }
        if let Some(values) = node.get("enum").and_then(Value::as_array) {
            return self.convert_enum(node, values, b, value);
        }
        if let Some(branches) = node.get("anyOf").and_then(Value::as_array) {
            return self.convert_union(node, branches, b, value);
        }
        if let Some(branches) = node.get("oneOf").and_then(Value::as_array) {
            return self.convert_union(node, branches, b, value);
        }
        if let Some(members) = node.get("allOf").and_then(Value::as_array) {
            let mut next: Cow<'_, Value> = Cow::Borrowed(value);
            for member in members {
                next = Cow::Owned(self.convert(member, b, &next).into_owned());
            }
            return Cow::Owned(self.convert_members(node, b, &next).into_owned());
        }
        match node.get("type").and_then(Value::as_str) {
            Some("string") => try_string(value),
            Some("number") => try_number(value),
            Some("integer") => match try_number(value) {
                Cow::Owned(n) => Cow::Owned(Value::from(n.as_f64().unwrap_or_default().trunc())),
                Cow::Borrowed(_) => Cow::Borrowed(value),
            },
            Some("boolean") => try_boolean(value),
            Some("null") => try_null(value),
            Some("array") => self.convert_array(node, b, value),
            Some("object") => self.convert_object(node, b, value),
            None => {
                if node.get("properties").is_some() || node.get("patternProperties").is_some() {
                    self.convert_object(node, b, value)
                } else if node.get("items").is_some() || node.get("prefixItems").is_some() {
                    self.convert_array(node, b, value)
                } else {
                    Cow::Borrowed(value)
                }
            }
            Some(_) => Cow::Borrowed(value),
        }
    }

    fn convert_enum<'a>(
        &self,
        node: &Value,
        values: &'a [Value],
        b: &str,
        value: &'a Value,
    ) -> Cow<'a, Value> {
        if values.iter().any(|member| js_strict_eq(member, value)) {
            return Cow::Borrowed(value);
        }
        for member in values {
            let candidate = convert_literal(member, value);
            if !js_strict_eq(&candidate, value) && self.passes(node, b, &candidate) == Some(true) {
                return candidate;
            }
        }
        Cow::Borrowed(value)
    }

    fn convert_union<'a>(
        &mut self,
        node: &'a Value,
        branches: &'a [Value],
        b: &str,
        value: &'a Value,
    ) -> Cow<'a, Value> {
        if branches
            .iter()
            .any(|branch| self.passes(branch, b, value) == Some(true))
        {
            return Cow::Owned(self.convert_members(node, b, value).into_owned());
        }
        for branch in branches {
            let converted = self.convert(branch, b, value);
            let candidate = self.convert_members(node, b, &converted).into_owned();
            if self.passes(node, b, &candidate) == Some(true) {
                return Cow::Owned(candidate);
            }
        }
        Cow::Owned(self.convert_members(node, b, value).into_owned())
    }

    fn convert_members<'a>(
        &mut self,
        node: &'a Value,
        b: &str,
        value: &'a Value,
    ) -> Cow<'a, Value> {
        let declares = node.get("properties").is_some()
            || node.get("patternProperties").is_some()
            || node.get("additionalProperties").is_some();
        if declares {
            self.convert_object(node, b, value)
        } else {
            Cow::Borrowed(value)
        }
    }

    fn convert_array<'a>(&mut self, node: &'a Value, b: &str, value: &'a Value) -> Cow<'a, Value> {
        let prefix = node
            .get("items")
            .and_then(Value::as_array)
            .or_else(|| node.get("prefixItems").and_then(Value::as_array));
        if let Some(prefix) = prefix {
            let Value::Array(items) = value else {
                return Cow::Borrowed(value);
            };
            let rest = if node.get("items").is_some_and(Value::is_array) {
                None
            } else {
                node.get("items")
            };
            let mut out = items.clone();
            for (i, item) in items.iter().enumerate() {
                let schema = prefix.get(i).or(rest);
                let Some(schema) = schema else { continue };
                let next = self.convert(schema, b, item);
                if !js_strict_eq(&next, item) {
                    if let Some(slot) = out.get_mut(i) {
                        *slot = next.into_owned();
                    }
                }
            }
            return if same_array(&out, items) {
                Cow::Borrowed(value)
            } else {
                Cow::Owned(Value::Array(out))
            };
        }
        if let Value::Array(items) = value {
            let Some(item_schema) = node.get("items") else {
                return Cow::Borrowed(value);
            };
            let mut out = items.clone();
            for (i, item) in items.iter().enumerate() {
                let next = self.convert(item_schema, b, item);
                if !js_strict_eq(&next, item) {
                    if let Some(slot) = out.get_mut(i) {
                        *slot = next.into_owned();
                    }
                }
            }
            return if same_array(&out, items) {
                Cow::Borrowed(value)
            } else {
                Cow::Owned(Value::Array(out))
            };
        }
        // Wrap a single value into a one-item array — a re-entry point.
        let item = node.get("items");
        let wrapped = match item {
            Some(item) if is_schema_object(item) => {
                self.reenter(item, b, value, false).into_owned()
            }
            _ => value.clone(),
        };
        Cow::Owned(Value::Array(vec![wrapped]))
    }

    fn convert_object<'a>(&mut self, node: &'a Value, b: &str, value: &'a Value) -> Cow<'a, Value> {
        let Value::Object(_) = value else {
            return Cow::Borrowed(value);
        };
        let properties = node.get("properties").and_then(Value::as_object);
        let patterns = pattern_entries(node);
        let additional = node.get("additionalProperties").filter(|v| v.is_object());
        let mut out = value.as_object().cloned().unwrap_or_default();
        for (key, old) in out.clone() {
            let mut next = old.clone();
            let mut matched = properties.is_some_and(|m| m.contains_key(&key));
            if let Some(property) = properties.and_then(|m| m.get(&key)) {
                next = self.convert(property, b, &old).into_owned();
            }
            for (pattern, schema) in &patterns {
                if pattern.is_match(&key) {
                    matched = true;
                    next = self.convert(schema, b, &next).into_owned();
                }
            }
            if !matched && let Some(add) = additional {
                next = self.convert(add, b, &next).into_owned();
            }
            if !js_strict_eq(&next, &old) {
                out.insert(key, next);
            }
        }
        if same_object(&out, value) {
            Cow::Borrowed(value)
        } else {
            Cow::Owned(Value::Object(out))
        }
    }

    // Pass 3: primitive coercion by type --------------------------------------

    fn coerce_types<'a>(&mut self, value: &'a Value, schema: &Value, base: &str) -> Cow<'a, Value> {
        if !is_schema_object(schema) {
            return Cow::Borrowed(value);
        }
        let b = self.base_of(schema, base);
        let mut next: Cow<'_, Value> = Cow::Borrowed(value);
        if let Some(members) = schema.get("allOf").and_then(Value::as_array) {
            for member in members {
                next = Cow::Owned(self.coerce_types(&next, member, &b).into_owned());
            }
        }
        if let Some(branches) = schema.get("anyOf").and_then(Value::as_array) {
            next = Cow::Owned(self.coerce_union(&next, branches, &b));
        }
        if let Some(branches) = schema.get("oneOf").and_then(Value::as_array) {
            next = Cow::Owned(self.coerce_union(&next, branches, &b));
        }
        let types = types_of(schema);
        let matches_member = types.len() > 1 && types.iter().any(|t| matches_type(&next, t));
        if !types.is_empty() && !matches_member {
            for t in &types {
                let candidate = coerce_primitive(&next, t);
                if !js_strict_eq(&candidate, &next) {
                    next = Cow::Owned(candidate.into_owned());
                    break;
                }
            }
        }
        if types.contains(&"object") && next.is_object() {
            next = Cow::Owned(self.coerce_properties(&next, schema, &b));
        }
        if types.contains(&"array") && next.is_array() {
            next = Cow::Owned(self.coerce_items(&next, schema, &b));
        }
        next
    }

    fn coerce_union(&mut self, value: &Value, branches: &[Value], base: &str) -> Value {
        for branch in branches {
            if self.passes(branch, base, value) == Some(true) {
                return value.clone();
            }
        }
        for branch in branches {
            let coerced = self.coerce_types(value, branch, base).into_owned();
            if self.passes(branch, base, &coerced) == Some(true) {
                return coerced;
            }
        }
        value.clone()
    }

    fn coerce_properties(&mut self, value: &Value, schema: &Value, base: &str) -> Value {
        let Some(map) = value.as_object() else {
            return value.clone();
        };
        let properties = schema.get("properties").and_then(Value::as_object);
        let patterns = pattern_entries(schema);
        let additional = schema.get("additionalProperties").filter(|v| v.is_object());
        let mut out = map.clone();
        for (key, property) in properties.into_iter().flatten() {
            if let Some(old) = out.get(key) {
                let next = self.coerce_types(old, property, base);
                if !js_strict_eq(&next, old) {
                    out.insert(key.clone(), next.into_owned());
                }
            }
        }
        for key in map.keys() {
            let mut matched = properties.is_some_and(|m| m.contains_key(key));
            let mut next = out.get(key).cloned().unwrap_or_default();
            for (pattern, item) in &patterns {
                if pattern.is_match(key) {
                    matched = true;
                    next = self.coerce_types(&next, item, base).into_owned();
                }
            }
            if !matched && let Some(add) = additional {
                next = self
                    .coerce_types(map.get(key).unwrap_or(&Value::Null), add, base)
                    .into_owned();
            }
            if !js_strict_eq(&next, map.get(key).unwrap_or(&Value::Null)) {
                out.insert(key.clone(), next);
            }
        }
        Value::Object(out)
    }

    fn coerce_items(&mut self, value: &Value, schema: &Value, base: &str) -> Value {
        let Value::Array(items) = value else {
            return value.clone();
        };
        let mut out = items.clone();
        for (i, item) in items.iter().enumerate() {
            if let Some(item_schema) = item_schema_at(schema, i) {
                if is_schema_object(item_schema) {
                    let next = self.coerce_types(item, item_schema, base);
                    if !js_strict_eq(&next, item) {
                        if let Some(slot) = out.get_mut(i) {
                            *slot = next.into_owned();
                        }
                    }
                }
            }
        }
        Value::Array(out)
    }
}

fn value_key(value: &Value) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut hasher = DefaultHasher::new();
    format!("{value:p}").hash(&mut hasher);
    hasher.finish()
}

fn same_array(out: &[Value], orig: &[Value]) -> bool {
    out.len() == orig.len() && out.iter().zip(orig).all(|(a, b)| js_strict_eq(a, b))
}

fn same_object(out: &Map<String, Value>, orig: &Value) -> bool {
    orig.as_object().is_some_and(|m| m == out)
}

/// `itemSchemaAt`: tuple prefix (`prefixItems` or draft-7 `items`
/// array) for its prefix, `items` otherwise.
fn item_schema_at(schema: &Value, index: usize) -> Option<&Value> {
    if let Some(items) = schema.get("items").and_then(Value::as_array) {
        return items.get(index);
    }
    if let Some(prefix) = schema.get("prefixItems").and_then(Value::as_array) {
        if index < prefix.len() {
            return prefix.get(index);
        }
    }
    schema.get("items").filter(|v| !v.is_array())
}

/// The `[regex, schema]` pairs of `patternProperties` (JS-`u` regexes;
/// the `regex` crate covers what tool schemas use).
fn pattern_entries(node: &Value) -> Vec<(regex::Regex, &Value)> {
    node.get("patternProperties")
        .and_then(Value::as_object)
        .map(|map| {
            map.iter()
                .filter_map(|(pattern, schema)| {
                    regex::Regex::new(pattern).ok().map(|re| (re, schema))
                })
                .collect()
        })
        .unwrap_or_default()
}

// TypeBox's conversions of one value --------------------------------------

fn convert_literal<'a>(literal: &Value, value: &'a Value) -> Cow<'a, Value> {
    if js_strict_eq(literal, value) {
        return Cow::Borrowed(value);
    }
    let result = match literal {
        Value::String(_) => try_string(value),
        Value::Number(_) => try_number(value),
        Value::Bool(_) => try_boolean(value),
        _ => return Cow::Borrowed(value),
    };
    if !js_strict_eq(&result, value) && js_strict_eq(&result, literal) {
        result
    } else {
        Cow::Borrowed(value)
    }
}

fn try_string(value: &Value) -> Cow<'_, Value> {
    match value {
        Value::Bool(b) => Cow::Owned(Value::from(if *b { "true" } else { "false" })),
        Value::Number(n) => Cow::Owned(Value::from(crate::text::js_number_to_string(
            n.as_f64().unwrap_or_default(),
        ))),
        Value::Null => Cow::Owned(Value::from("null")),
        _ => Cow::Borrowed(value),
    }
}

fn try_number(value: &Value) -> Cow<'_, Value> {
    match value {
        Value::Bool(b) => Cow::Owned(Value::from(u8::from(*b))),
        Value::Number(n) if n.as_f64().is_some_and(f64::is_finite) => Cow::Borrowed(value),
        Value::Null => Cow::Owned(Value::from(0)),
        Value::String(s) => {
            let coerced = crate::text::js_number_from_str(s);
            if coerced.is_finite() {
                return Cow::Owned(Value::from(coerced));
            }
            let lowercase = s.to_lowercase();
            if lowercase == "false" {
                return Cow::Owned(Value::from(0));
            }
            if lowercase == "true" {
                return Cow::Owned(Value::from(1));
            }
            big_int_of_string(s).map_or(Cow::Borrowed(value), |big| Cow::Owned(Value::from(big)))
        }
        _ => Cow::Borrowed(value),
    }
}

/// `TypeBox`'s `TryBigInt` of a string, as `f64` (JSON has no `BigInt`;
/// values beyond the safe-integer range keep JS `Number` precision).
fn big_int_of_string(value: &str) -> Option<f64> {
    let re_int = regex::Regex::new(r"^-?(0|[1-9]\d*)n$").ok()?;
    let re_frac = regex::Regex::new(r"^-?(0|[1-9]\d*)\.\d+$").ok()?;
    let re_plain = regex::Regex::new(r"^-?(0|[1-9]\d*)$").ok()?;
    let digits = if re_int.is_match(value) {
        &value[..value.len() - 1]
    } else if re_frac.is_match(value) {
        value.split('.').next()?
    } else if re_plain.is_match(value) {
        value
    } else {
        return None;
    };
    digits
        .parse::<f64>()
        .ok()
        .filter(|big| big.abs() <= 9_007_199_254_740_991.0)
}

fn try_boolean(value: &Value) -> Cow<'_, Value> {
    match value {
        Value::Number(n) => match n.as_f64().filter(|v| v.is_finite()) {
            Some(0.0) => Cow::Owned(Value::from(false)),
            Some(1.0) => Cow::Owned(Value::from(true)),
            _ => Cow::Borrowed(value),
        },
        Value::Null => Cow::Owned(Value::from(false)),
        Value::String(s) => {
            let lowercase = s.to_lowercase();
            if lowercase == "false" || s == "0" {
                Cow::Owned(Value::from(false))
            } else if lowercase == "true" || s == "1" {
                Cow::Owned(Value::from(true))
            } else {
                Cow::Borrowed(value)
            }
        }
        _ => Cow::Borrowed(value),
    }
}

fn try_null(value: &Value) -> Cow<'_, Value> {
    match value {
        Value::Bool(b) => {
            if *b {
                Cow::Borrowed(value)
            } else {
                Cow::Owned(Value::Null)
            }
        }
        Value::Number(n) => match n.as_f64().filter(|v| v.is_finite()) {
            Some(0.0) => Cow::Owned(Value::Null),
            _ => Cow::Borrowed(value),
        },
        Value::String(s) => {
            let lowercase = s.to_lowercase();
            if lowercase == CHARS_UNDEFINED || lowercase == "null" || s.is_empty() || s == "0" {
                Cow::Owned(Value::Null)
            } else {
                Cow::Borrowed(value)
            }
        }
        _ => Cow::Borrowed(value),
    }
}

// Primitive coercion by type ----------------------------------------------

fn types_of(schema: &Value) -> Vec<&str> {
    match schema.get("type") {
        Some(Value::String(t)) => vec![t.as_str()],
        Some(Value::Array(list)) => list.iter().filter_map(Value::as_str).collect(),
        _ => Vec::new(),
    }
}

fn matches_type(value: &Value, type_name: &str) -> bool {
    match type_name {
        "number" => value.is_number(),
        "integer" => value
            .as_f64()
            .is_some_and(|v| v.fract() == 0.0 && v.is_finite()),
        "boolean" => value.is_boolean(),
        "string" => value.is_string(),
        "null" => value.is_null(),
        "array" => value.is_array(),
        "object" => value.is_object(),
        _ => false,
    }
}

fn coerce_primitive<'a>(value: &'a Value, type_name: &str) -> Cow<'a, Value> {
    match type_name {
        "number" | "integer" => match value {
            Value::Null => Cow::Owned(Value::from(0)),
            Value::String(s) if !js_trim(s).is_empty() => {
                let parsed = crate::text::js_number_from_str(s);
                let ok = if type_name == "number" {
                    parsed.is_finite()
                } else {
                    parsed.is_finite() && parsed.fract() == 0.0
                };
                if ok {
                    Cow::Owned(Value::from(parsed))
                } else {
                    Cow::Borrowed(value)
                }
            }
            Value::Bool(b) => Cow::Owned(Value::from(u8::from(*b))),
            _ => Cow::Borrowed(value),
        },
        "boolean" => match value {
            Value::Null => Cow::Owned(Value::from(false)),
            Value::String(s) if s == "true" => Cow::Owned(Value::from(true)),
            Value::String(s) if s == "false" => Cow::Owned(Value::from(false)),
            Value::Number(n) if n.as_f64() == Some(1.0) => Cow::Owned(Value::from(true)),
            Value::Number(n) if n.as_f64() == Some(0.0) => Cow::Owned(Value::from(false)),
            _ => Cow::Borrowed(value),
        },
        "string" => match value {
            Value::Null => Cow::Owned(Value::from("")),
            Value::Number(n) => Cow::Owned(Value::from(crate::text::js_number_to_string(
                n.as_f64().unwrap_or_default(),
            ))),
            Value::Bool(b) => Cow::Owned(Value::from(if *b { "true" } else { "false" })),
            _ => Cow::Borrowed(value),
        },
        "null" => match value {
            Value::String(s) if s.is_empty() => Cow::Owned(Value::Null),
            Value::Number(n) if n.as_f64() == Some(0.0) => Cow::Owned(Value::Null),
            Value::Bool(false) => Cow::Owned(Value::Null),
            _ => Cow::Borrowed(value),
        },
        _ => Cow::Borrowed(value),
    }
}
