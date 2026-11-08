//! `annotations.ts`: evaluated-property/item tracking.
//!
//! Records which properties and items a schema evaluated (only when
//! the schema would pass — checks short-circuit). `walkUnevaluated`
//! uses this to decide `unevaluatedProperties` and
//! `unevaluatedItems`.
#![expect(
    clippy::excessive_nesting,
    clippy::collapsible_if,
    reason = "the port follows upstream's keyword-dispatch structure"
)]
use std::collections::HashSet;

use serde_json::Value;

use super::is_schema_object;

/// `Annotations`: `props`/`items`/`indexes` evaluated so far —
/// `allProps`/`allItems` mark "everything".
#[derive(Default)]
pub(crate) struct Annotations {
    /// Evaluated property names.
    pub props: Option<HashSet<String>>,
    /// All properties were evaluated.
    pub all_props: bool,
    /// The highest evaluated tuple prefix length.
    pub items: usize,
    /// All items were evaluated.
    pub all_items: bool,
    /// Item indexes `contains` matched.
    pub indexes: Option<HashSet<usize>>,
}

impl Annotations {
    /// `merge(a, b)`.
    pub(crate) fn merge(&mut self, b: &Self) {
        if let Some(props) = &b.props {
            let own = self.props.get_or_insert_with(HashSet::new);
            own.extend(props.iter().cloned());
        }
        self.all_props |= b.all_props;
        self.items = self.items.max(b.items);
        self.all_items |= b.all_items;
        if let Some(indexes) = &b.indexes {
            let own = self.indexes.get_or_insert_with(HashSet::new);
            own.extend(indexes.iter().copied());
        }
    }
}

/// `annotateNode`: runs `node`'s annotating keywords against `v`,
/// recording what passed — the explainer's stand-in for the compiled
/// annotating check. `ok` answers a node's boolean check (through
/// refs); `resolve` resolves `$ref`.
#[expect(
    clippy::cognitive_complexity,
    clippy::too_many_lines,
    reason = "the annotating walk mirrors upstream annotations.ts keyword order"
)]
pub(crate) fn annotate<'n, O, R>(
    node: &'n Value,
    v: &Value,
    a: &mut Annotations,
    ok: &O,
    resolve: &R,
) where
    O: Fn(&Value, &Value) -> bool,
    R: for<'r> Fn(&'r Value) -> Option<&'n Value>,
{
    if !is_schema_object(node) {
        return;
    }
    if let Some(r) = node.get("$ref")
        && r.is_string()
        && let Some(target) = resolve(r)
    {
        annotate(target, v, a, ok, resolve);
    }
    if let Some(props) = node.get("properties").and_then(Value::as_object) {
        if let Some(map) = v.as_object() {
            for (k, schema) in props {
                if let Some(val) = map.get(k) {
                    if ok(schema, val) {
                        a.props.get_or_insert_with(HashSet::new).insert(k.clone());
                    }
                }
            }
        }
    }
    if let Some(map) = v.as_object() {
        let declared: HashSet<&str> = node
            .get("properties")
            .and_then(Value::as_object)
            .map(|m| m.keys().map(String::as_str).collect())
            .unwrap_or_default();
        let patterns = pattern_schemas(node);
        let additional = node.get("additionalProperties");
        for (k, val) in map {
            let mut matched = declared.contains(k.as_str());
            for (re, schema) in &patterns {
                if re.is_match(k) {
                    if ok(schema, val) {
                        a.props.get_or_insert_with(HashSet::new).insert(k.clone());
                    }
                    matched = true;
                }
            }
            if !matched
                && let Some(add) = additional
                && ok(add, val)
            {
                a.props.get_or_insert_with(HashSet::new).insert(k.clone());
            }
        }
    }
    if let Some(items) = v.as_array() {
        let prefix = node
            .get("prefixItems")
            .or_else(|| node.get("items").filter(|i| i.is_array()))
            .and_then(Value::as_array);
        if let Some(prefix) = prefix {
            let m = prefix.len().min(items.len());
            for (i, schema) in prefix.iter().enumerate().take(m) {
                if !ok(schema, items.get(i).unwrap_or(&Value::Null)) {
                    return;
                }
            }
            a.items = a.items.max(m);
        }
        let rest = if prefix.is_some() {
            node.get("additionalItems")
        } else {
            node.get("items").filter(|i| !i.is_array())
        };
        if let Some(rest) = rest {
            for item in items.iter().skip(a.items) {
                if !ok(rest, item) {
                    return;
                }
            }
            a.all_items = true;
        }
        if let Some(contains) = node.get("contains") {
            let min = node
                .get("minContains")
                .and_then(Value::as_f64)
                .unwrap_or(1.0);
            let max = node
                .get("maxContains")
                .and_then(Value::as_f64)
                .unwrap_or(f64::INFINITY);
            let mut n = 0f64;
            for (i, item) in items.iter().enumerate() {
                if ok(contains, item) {
                    n += 1.0;
                    a.indexes.get_or_insert_with(HashSet::new).insert(i);
                }
            }
            if n < min || n > max {
                return;
            }
        }
    }
    // `allOf`: every member's annotations merge.
    if let Some(members) = node.get("allOf").and_then(Value::as_array) {
        for member in members {
            if !ok(member, v) {
                return;
            }
            annotate(member, v, a, ok, resolve);
        }
    }
    // `anyOf` / `oneOf`: every passing branch contributes.
    for key in ["anyOf", "oneOf"] {
        if let Some(branches) = node.get(key).and_then(Value::as_array) {
            let mut n = 0u32;
            for branch in branches {
                let mut ann = Annotations::default();
                if ok(branch, v) {
                    n += 1;
                    annotate(branch, v, &mut ann, ok, resolve);
                    a.merge(&ann);
                }
            }
            let passed = if key == "oneOf" { n == 1 } else { n > 0 };
            if !passed {
                return;
            }
        }
    }
    // `if`: a passing `if` contributes even without `then`.
    if let Some(cond) = node.get("if") {
        if ok(cond, v) {
            let mut ann = Annotations::default();
            annotate(cond, v, &mut ann, ok, resolve);
            a.merge(&ann);
            if let Some(then) = node.get("then") {
                if !ok(then, v) {
                    return;
                }
                annotate(then, v, a, ok, resolve);
            }
        } else if let Some(otherwise) = node.get("else") {
            if !ok(otherwise, v) {
                return;
            }
            annotate(otherwise, v, a, ok, resolve);
        }
    }
    // `dependentSchemas`: the triggered schema's annotations merge.
    if let Some(map) = v.as_object()
        && let Some(deps) = node.get("dependentSchemas").and_then(Value::as_object)
    {
        for (k, schema) in deps {
            if map.contains_key(k) {
                if !ok(schema, v) {
                    return;
                }
                annotate(schema, v, a, ok, resolve);
            }
        }
    }
}

fn pattern_schemas(node: &Value) -> Vec<(regex::Regex, &Value)> {
    node.get("patternProperties")
        .and_then(Value::as_object)
        .map(|m| {
            m.iter()
                .filter_map(|(p, s)| regex::Regex::new(p).ok().map(|re| (re, s)))
                .collect()
        })
        .unwrap_or_default()
}
