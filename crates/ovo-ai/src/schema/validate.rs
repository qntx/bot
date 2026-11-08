//! Error explanations: upstream `schema/src/explain.ts` — a schema walk
//! that reports each failing keyword at the value it failed on, with
//! pi-schema's exact messages, ordering and union blame.
#![expect(
    clippy::excessive_nesting,
    clippy::collapsible_if,
    reason = "the port follows upstream's keyword-dispatch structure"
)]
use std::collections::HashMap;

use serde_json::Value;

use super::{ToolSchemaValidator, ValidationError, annotations, deep_equal, is_schema_object};
use crate::text::js_len;

const LISTED_VALUES: usize = 20;

/// `Explainer`: explains failures of one compiled schema.
pub(crate) struct Explainer<'v> {
    validator: &'v ToolSchemaValidator,
    memo: HashMap<(usize, String), Vec<ValidationError>>,
}

impl<'v> Explainer<'v> {
    /// The errors of `value` — at least one when it fails the check.
    pub(crate) fn new(validator: &'v ToolSchemaValidator) -> Self {
        Self {
            validator,
            memo: HashMap::new(),
        }
    }

    /// `errors(value)`: every failing keyword's error.
    pub(crate) fn errors(mut self, value: &Value) -> Vec<ValidationError> {
        let mut out = Vec::new();
        let schema = self.validator.original().clone();
        let base = self.validator.root_base().to_owned();
        self.walk(&schema, &base, value, "", &mut out);
        self.memo.clear();
        if out.is_empty() {
            out.push(ValidationError {
                path: String::new(),
                keyword: "schema".into(),
                message: "invalid".into(),
            });
        }
        out
    }

    fn ok(&self, node: &Value, value: &Value) -> bool {
        self.validator.check_node(node, value).unwrap_or(false)
    }

    fn base_of(&self, node: &Value, base: &str) -> String {
        self.validator.base_of(node, base)
    }

    #[expect(
        clippy::cognitive_complexity,
        reason = "upstream walk dispatches on every keyword in one pass"
    )]
    fn walk(
        &mut self,
        node: &Value,
        base: &str,
        v: &Value,
        path: &str,
        out: &mut Vec<ValidationError>,
    ) {
        match node {
            Value::Bool(true) => return,
            Value::Bool(false) => {
                out.push(err(path, "false", "no value is allowed here"));
                return;
            }
            _ => {}
        }
        if !is_schema_object(node) {
            return;
        }
        let b = self.base_of(node, base);
        if self.ok(node, v) {
            return;
        }
        let start = out.len();
        macro_rules! push {
            ($kw:expr, $msg:expr) => {
                out.push(err(path, $kw, $msg))
            };
        }
        if let Some(types) = node.get("type") {
            let types: Vec<String> = match types {
                Value::Array(list) => list
                    .iter()
                    .filter_map(Value::as_str)
                    .map(str::to_owned)
                    .collect(),
                Value::String(t) => vec![t.clone()],
                _ => Vec::new(),
            };
            if !types.iter().any(|t| type_check(v, t)) {
                push!(
                    "type",
                    format!("expected {}, got {}", types.join(" or "), type_name(v))
                );
                return;
            }
        }
        if let Some(values) = node.get("enum").and_then(Value::as_array)
            && !values.iter().any(|x| deep_equal(x, v))
        {
            push!("enum", format!("expected one of {}", list_values(values)));
        }
        if node.get("const").is_some_and(|c| !deep_equal(c, v)) {
            let literal = crate::json::to_string(&node["const"]).unwrap_or_default();
            push!("const", format!("expected {literal}"));
        }
        if v.is_string() && node.get("type").and_then(Value::as_str) != Some("bytes") {
            Self::walk_string(node, v, &mut |kw, m| out.push(err(path, kw, m)));
        }
        if v.is_number() {
            Self::walk_number(node, v, &mut |kw, m| out.push(err(path, kw, m)));
        }
        if v.is_object() {
            self.walk_object(node, &b, v, path, out);
        }
        if v.is_array() {
            self.walk_array(node, &b, v, path, out);
        }
        if let Some(members) = node.get("allOf").and_then(Value::as_array) {
            for s in members {
                self.walk(s, &b, v, path, out);
            }
        }
        if let Some(branches) = node.get("anyOf").and_then(Value::as_array) {
            self.walk_union(node, branches, &b, v, path, out, false);
        }
        if let Some(branches) = node.get("oneOf").and_then(Value::as_array) {
            self.walk_union(node, branches, &b, v, path, out, true);
        }
        if node.get("not").is_some_and(|n| self.ok(n, v)) {
            out.push(err(path, "not", "must not match the schema in not"));
        }
        if let Some(cond) = node.get("if") {
            let branch = if self.ok(cond, v) {
                node.get("then")
            } else {
                node.get("else")
            };
            if let Some(branch) = branch {
                self.walk(branch, &b, v, path, out);
            }
        }
        if let Some(reference) = node.get("$ref").and_then(Value::as_str) {
            self.walk_ref(node, reference, &b, v, path, out);
        }
        if out.len() == start
            && (node.get("unevaluatedProperties").is_some()
                || node.get("unevaluatedItems").is_some())
        {
            self.walk_unevaluated(node, &b, v, path, out);
        }
        if out.len() == start && node.get("$dynamicRef").is_some() {
            out.push(err(
                path,
                "$dynamicRef",
                "does not match the dynamically referenced schema",
            ));
        }
    }

    fn walk_ref(
        &mut self,
        _node: &Value,
        reference: &str,
        base: &str,
        v: &Value,
        path: &str,
        out: &mut Vec<ValidationError>,
    ) {
        match self.validator.resolve(reference, base) {
            Err(()) => {
                out.push(err(
                    path,
                    "$ref",
                    format!("unresolved reference {reference}"),
                ));
            }
            Ok(resolved) => {
                let Some(target) = self.validator.node_at_ptr(&resolved.ptr) else {
                    out.push(err(
                        path,
                        "$ref",
                        format!("unresolved reference {reference}"),
                    ));
                    return;
                };
                if target == &Value::Bool(false) && self.validator.is_unresolved(reference, base) {
                    out.push(err(
                        path,
                        "$ref",
                        format!("unresolved reference {reference}"),
                    ));
                    return;
                }
                if !is_schema_object(target) || !(v.is_object() || v.is_array()) {
                    self.walk(target, &resolved.base, v, path, out);
                    return;
                }
                let key = (std::ptr::from_ref(target) as usize, path.to_owned());
                if let Some(errors) = self.memo.get(&key) {
                    out.extend(errors.iter().cloned());
                    return;
                }
                let mut errors = Vec::new();
                self.walk(target, &resolved.base, v, path, &mut errors);
                self.memo.insert(key, errors.clone());
                out.extend(errors);
            }
        }
    }

    fn walk_string(node: &Value, v: &Value, push: &mut dyn FnMut(&'static str, String)) {
        let text = v.as_str().unwrap_or_default();
        let n = js_len(text) as f64;
        if let Some(min) = node.get("minLength").and_then(Value::as_f64) {
            if n < min {
                push("minLength", format!("expected at least {min} characters"));
            }
        }
        if let Some(max) = node.get("maxLength").and_then(Value::as_f64) {
            if n > max {
                push("maxLength", format!("expected at most {max} characters"));
            }
        }
        if let Some(pattern) = node.get("pattern").and_then(Value::as_str)
            && regex::Regex::new(pattern).is_ok_and(|re| !re.is_match(text))
        {
            push("pattern", format!("expected to match {pattern}"));
        }
        if let Some(format_name) = node.get("format").and_then(Value::as_str)
            && super::formats::is_format(format_name)
            && !super::formats::format_is_valid(format_name, text)
        {
            push("format", format!("expected format {format_name}"));
        }
    }

    fn walk_number(node: &Value, v: &Value, push: &mut dyn FnMut(&'static str, String)) {
        let Some(n) = v.as_f64() else { return };
        if let Some(min) = node.get("minimum").and_then(Value::as_f64) {
            if n < min {
                push("minimum", format!("expected >= {min}"));
            }
        }
        if let Some(max) = node.get("maximum").and_then(Value::as_f64) {
            if n > max {
                push("maximum", format!("expected <= {max}"));
            }
        }
        if let Some(min) = node.get("exclusiveMinimum").and_then(Value::as_f64) {
            if n <= min {
                push("exclusiveMinimum", format!("expected > {min}"));
            }
        }
        if let Some(max) = node.get("exclusiveMaximum").and_then(Value::as_f64) {
            if n >= max {
                push("exclusiveMaximum", format!("expected < {max}"));
            }
        }
        if let Some(d) = node.get("multipleOf").and_then(Value::as_f64) {
            if !is_multiple(n, d) {
                push("multipleOf", format!("expected a multiple of {d}"));
            }
        }
    }

    #[expect(
        clippy::cognitive_complexity,
        reason = "upstream walkObject evaluates every object keyword in order"
    )]
    fn walk_object(
        &mut self,
        node: &Value,
        b: &str,
        v: &Value,
        path: &str,
        out: &mut Vec<ValidationError>,
    ) {
        let map = v.as_object().cloned().unwrap_or_default();
        let properties = node.get("properties").and_then(Value::as_object);
        if let Some(required) = node.get("required").and_then(Value::as_array) {
            for k in required.iter().filter_map(Value::as_str) {
                if !map.contains_key(k) {
                    out.push(err(&pointer(path, k), "required", "missing"));
                }
            }
        }
        for (k, property) in properties.into_iter().flatten() {
            if let Some(val) = map.get(k) {
                self.walk(property, b, val, &pointer(path, k), out);
            }
        }
        let patterns = pattern_entries(node);
        for (k, val) in &map {
            let mut matched = properties.is_some_and(|m| m.contains_key(k));
            for (re, schema) in &patterns {
                if re.is_match(k) {
                    matched = true;
                    self.walk(schema, b, val, &pointer(path, k), out);
                }
            }
            if !matched && node.get("additionalProperties").is_some() {
                match node.get("additionalProperties") {
                    Some(Value::Bool(false)) => {
                        out.push(err(
                            &pointer(path, k),
                            "additionalProperties",
                            "unknown field",
                        ));
                    }
                    Some(add) => self.walk(add, b, val, &pointer(path, k), out),
                    _ => {}
                }
            }
            if node
                .get("propertyNames")
                .is_some_and(|pn| !self.ok(pn, &Value::from(k.clone())))
            {
                out.push(err(
                    &pointer(path, k),
                    "propertyNames",
                    "invalid property name",
                ));
            }
        }
        let count = map.len() as f64;
        if let Some(min) = node.get("minProperties").and_then(Value::as_f64) {
            if count < min {
                out.push(err(
                    path,
                    "minProperties",
                    format!("expected at least {min} properties"),
                ));
            }
        }
        if let Some(max) = node.get("maxProperties").and_then(Value::as_f64) {
            if count > max {
                out.push(err(
                    path,
                    "maxProperties",
                    format!("expected at most {max} properties"),
                ));
            }
        }
        if let Some(deps) = node.get("dependentRequired").and_then(Value::as_object) {
            for (k, d) in deps {
                if map.contains_key(k) {
                    if let Some(list) = d.as_array() {
                        for r in list.iter().filter_map(Value::as_str) {
                            if !map.contains_key(r) {
                                out.push(err(
                                    &pointer(path, r),
                                    "dependentRequired",
                                    format!("required when {k} is present"),
                                ));
                            }
                        }
                    }
                }
            }
        }
        if let Some(deps) = node.get("dependentSchemas").and_then(Value::as_object) {
            for (k, s) in deps {
                if map.contains_key(k) {
                    self.walk(s, b, v, path, out);
                }
            }
        }
        if let Some(deps) = node.get("dependencies").and_then(Value::as_object) {
            for (k, d) in deps {
                if map.contains_key(k) {
                    if let Some(list) = d.as_array() {
                        for r in list.iter().filter_map(Value::as_str) {
                            if !map.contains_key(r) {
                                out.push(err(
                                    &pointer(path, r),
                                    "dependentRequired",
                                    format!("required when {k} is present"),
                                ));
                            }
                        }
                    } else {
                        self.walk(d, b, v, path, out);
                    }
                }
            }
        }
    }

    fn walk_array(
        &mut self,
        node: &Value,
        b: &str,
        v: &Value,
        path: &str,
        out: &mut Vec<ValidationError>,
    ) {
        let items = v.as_array().cloned().unwrap_or_default();
        let (prefix, rest): (Vec<&Value>, Option<&Value>) = match node.get("items") {
            Some(Value::Array(list)) => (list.iter().collect(), node.get("additionalItems")),
            _ => (
                node.get("prefixItems")
                    .and_then(Value::as_array)
                    .map(|l| l.iter().collect())
                    .unwrap_or_default(),
                node.get("items"),
            ),
        };
        for (i, item) in items.iter().enumerate() {
            let s = prefix.get(i).copied().or(rest);
            if let Some(s) = s {
                self.walk(s, b, item, &format!("{path}/{i}"), out);
            }
        }
        let len = items.len() as f64;
        if let Some(min) = node.get("minItems").and_then(Value::as_f64) {
            if len < min {
                out.push(err(
                    path,
                    "minItems",
                    format!("expected at least {min} items"),
                ));
            }
        }
        if let Some(max) = node.get("maxItems").and_then(Value::as_f64) {
            if len > max {
                out.push(err(
                    path,
                    "maxItems",
                    format!("expected at most {max} items"),
                ));
            }
        }
        if node.get("uniqueItems") == Some(&Value::Bool(true)) && !unique(&items) {
            out.push(err(path, "uniqueItems", "expected unique items"));
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
            let n = items.iter().filter(|item| self.ok(contains, item)).count() as f64;
            if n < min || n > max {
                out.push(err(path, "contains", contains_message(min, max)));
            }
        }
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "the union blame walk carries the full keyword context upstream passes"
    )]
    fn walk_union(
        &mut self,
        _node: &Value,
        branches: &[Value],
        b: &str,
        v: &Value,
        path: &str,
        out: &mut Vec<ValidationError>,
        exactly_one: bool,
    ) {
        let matches = branches.iter().filter(|s| self.ok(s, v)).count();
        if matches == 1 || (!exactly_one && matches > 0) {
            return;
        }
        let keyword = if exactly_one { "oneOf" } else { "anyOf" };
        if matches > 1 {
            out.push(err(path, keyword, "matches more than one alternative"));
            return;
        }
        // A tagged union reports the branch its tag selects; otherwise
        // the branch whose first error is deepest.
        if let Some((key, values)) = self.tag_of(branches, b) {
            if v.is_object() {
                let tag = v.get(&key);
                match values.iter().position(|x| Some(x) == tag) {
                    None => {
                        out.push(ValidationError {
                            path: pointer(path, &key),
                            keyword: keyword.to_owned(),
                            message: format!("expected one of {}", list_values(&values)),
                        });
                    }
                    Some(i) => {
                        if let Some(branch) = branches.get(i) {
                            let branch = branch.clone();
                            self.walk(&branch, b, v, path, out);
                        }
                    }
                }
                return;
            }
        }
        // An object is blamed on the branch declaring most of its keys,
        // then on the branch whose first error is deepest.
        let mut best: Option<(Vec<ValidationError>, usize)> = None;
        for s in branches {
            let mut errors = Vec::new();
            let s = s.clone();
            self.walk(&s, b, v, path, &mut errors);
            if errors.is_empty() {
                continue;
            }
            let declared = if v.is_object() {
                self.declared_keys(&s, b, v)
            } else {
                0
            };
            let first_len = errors.first().map_or(0, |e| e.path.len());
            let better = best.as_ref().is_none_or(|(errs, d)| {
                declared > *d
                    || (declared == *d && first_len > errs.first().map_or(0, |e| e.path.len()))
            });
            if better {
                best = Some((errors, declared));
            }
        }
        match best {
            Some((errors, _)) => out.extend(errors),
            None => out.push(err(path, keyword, "matches no alternative")),
        }
    }

    /// `tagOf`: a union of objects tagged by a `const` on a shared
    /// `properties` key — the key and its per-branch values.
    fn tag_of(&self, branches: &[Value], b: &str) -> Option<(String, Vec<Value>)> {
        let resolved: Vec<Value> = branches.iter().map(|s| self.resolve_branch(s, b)).collect();
        let first = resolved.first()?;
        let props = first.get("properties").and_then(Value::as_object)?;
        for key in props.keys() {
            let mut values = Vec::new();
            let mut primitive = true;
            for s in &resolved {
                let tag = s
                    .get("properties")
                    .and_then(Value::as_object)
                    .and_then(|m| m.get(key));
                let Some(tag) = tag else {
                    primitive = false;
                    break;
                };
                let Some(lit) = tag.get("const") else {
                    primitive = false;
                    break;
                };
                if lit.is_object() || lit.is_array() {
                    primitive = false;
                    break;
                }
                values.push(lit.clone());
            }
            if primitive {
                let mut seen = std::collections::HashSet::new();
                if values.iter().all(|v| seen.insert(format!("{v:?}")))
                    && values.len() == resolved.len()
                {
                    return Some((key.clone(), values));
                }
            }
        }
        None
    }

    fn resolve_branch(&self, s: &Value, b: &str) -> Value {
        if let Some(reference) = s.get("$ref").and_then(Value::as_str) {
            if let Ok(resolved) = self.validator.resolve(reference, b) {
                if let Some(node) = self.validator.node_at_ptr(&resolved.ptr) {
                    return node.clone();
                }
            }
        }
        s.clone()
    }

    /// `declaredKeys`: how many of `v`'s keys the branch declares in
    /// `properties` (through a `$ref`).
    fn declared_keys(&self, s: &Value, b: &str, v: &Value) -> usize {
        let node = self.resolve_branch(s, b);
        let properties = node.get("properties").and_then(Value::as_object);
        let Some(map) = v.as_object() else { return 0 };
        properties.map_or(0, |props| {
            map.keys()
                .filter(|k| props.contains_key(k.as_str()))
                .count()
        })
    }

    fn walk_unevaluated(
        &mut self,
        node: &Value,
        b: &str,
        v: &Value,
        path: &str,
        out: &mut Vec<ValidationError>,
    ) {
        // Simplified `walkUnevaluated`: collect evaluated props/items of
        // `node` minus the unevaluated keywords (upstream runs the
        // compiled annotating check).
        let mut rest = node.clone();
        if let Value::Object(m) = &mut rest {
            m.remove("unevaluatedProperties");
            m.remove("unevaluatedItems");
        }
        let mut ann = annotations::Annotations::default();
        self.annotate(&rest, b, v, &mut ann);
        if let (Some(map), Some(rule)) = (v.as_object(), node.get("unevaluatedProperties")) {
            if !ann.all_props {
                for (k, val) in map {
                    if ann.props.as_ref().is_some_and(|p| p.contains(k)) {
                        continue;
                    }
                    if rule == &Value::Bool(false) {
                        out.push(err(
                            &pointer(path, k),
                            "unevaluatedProperties",
                            "unknown field",
                        ));
                    } else {
                        self.walk(rule, b, val, &pointer(path, k), out);
                    }
                }
            }
        }
        if let (Some(items), Some(rule)) = (v.as_array(), node.get("unevaluatedItems")) {
            if !ann.all_items {
                for (i, item) in items.iter().enumerate().skip(ann.items) {
                    if ann.indexes.as_ref().is_some_and(|s| s.contains(&i)) {
                        continue;
                    }
                    if rule == &Value::Bool(false) {
                        out.push(err(
                            &format!("{path}/{i}"),
                            "unevaluatedItems",
                            "unexpected item",
                        ));
                    } else {
                        self.walk(rule, b, item, &format!("{path}/{i}"), out);
                    }
                }
            }
        }
    }

    /// `c.annotating(node, b)(v, a)`: which properties and items `node`
    /// evaluated — used only inside `walkUnevaluated`.
    fn annotate(&self, node: &Value, b: &str, v: &Value, a: &mut annotations::Annotations) {
        let b = b.to_owned();
        let ok =
            |schema: &Value, val: &Value| self.validator.check_node(schema, val).unwrap_or(false);
        let resolve = move |r: &Value| -> Option<&Value> {
            r.as_str().and_then(|reference| {
                self.validator
                    .resolve(reference, b.as_str())
                    .ok()
                    .and_then(|res| self.validator.node_at_ptr(&res.ptr))
            })
        };
        annotations::annotate(node, v, a, &ok, &resolve);
    }
}

fn err(path: &str, keyword: &str, message: impl Into<String>) -> ValidationError {
    ValidationError {
        path: path.to_owned(),
        keyword: keyword.to_owned(),
        message: message.into(),
    }
}

fn pointer(path: &str, key: &str) -> String {
    format!("{path}/{}", key.replace('~', "~0").replace('/', "~1"))
}

fn type_check(v: &Value, name: &str) -> bool {
    match name {
        "string" => v.is_string(),
        "number" => v.is_number(),
        "integer" => v
            .as_f64()
            .is_some_and(|n| n.fract() == 0.0 && n.is_finite()),
        "boolean" => v.is_boolean(),
        "null" => v.is_null(),
        "array" => v.is_array(),
        "object" => v.is_object(),
        _ => false,
    }
}

fn type_name(v: &Value) -> &'static str {
    match v {
        Value::Null => "null",
        Value::Array(_) => "array",
        Value::Number(n) => {
            if n.as_f64()
                .is_some_and(|x| x.fract() == 0.0 && x.is_finite())
            {
                "integer"
            } else {
                "number"
            }
        }
        Value::Object(_) => "object",
        Value::Bool(_) => "boolean",
        Value::String(_) => "string",
    }
}

fn list_values(values: &[Value]) -> String {
    if values.len() <= LISTED_VALUES {
        return crate::json::to_string(&values).unwrap_or_default();
    }
    let listed =
        crate::json::to_string(values.get(..LISTED_VALUES).unwrap_or(&[])).unwrap_or_default();
    let trimmed = listed.strip_suffix(']').unwrap_or(&listed);
    format!("{trimmed}, ... and {} more]", values.len() - LISTED_VALUES)
}

#[expect(
    clippy::float_cmp,
    reason = "upstream compares min/max against literal counts exactly"
)]
fn contains_message(min: f64, max: f64) -> String {
    if max.is_infinite() {
        return format!(
            "expected at least {min} matching {}",
            if min == 1.0 { "item" } else { "items" }
        );
    }
    if min == 0.0 {
        return format!(
            "expected at most {max} matching {}",
            if max == 1.0 { "item" } else { "items" }
        );
    }
    format!("expected between {min} and {max} matching items")
}

fn is_multiple(v: f64, d: f64) -> bool {
    if d == 0.0 {
        return false;
    }
    let r = v % d;
    r == 0.0 || (r.abs() - d.abs()).abs() < 1e-10
}

fn unique(items: &[Value]) -> bool {
    for (i, a) in items.iter().enumerate() {
        for b in items.get(i + 1..).unwrap_or(&[]) {
            if deep_equal(a, b) {
                return false;
            }
        }
    }
    true
}

fn pattern_entries(node: &Value) -> Vec<(regex::Regex, &Value)> {
    node.get("patternProperties")
        .and_then(Value::as_object)
        .map(|map| {
            map.iter()
                .filter_map(|(p, s)| regex::Regex::new(p).ok().map(|re| (re, s)))
                .collect()
        })
        .unwrap_or_default()
}
