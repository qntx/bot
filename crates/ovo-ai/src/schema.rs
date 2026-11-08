//! Tool-argument schema coercion and validation.
//!
//! Upstream `packages/schema` (`compile` with `coerce: true`,
//! `unresolvedRefs: "reject"`, `formats`) plus `ai/utils/validation.ts`
//! (`validateToolCall`, `validateToolArguments`).
//!
//! The boolean `check` uses the `jsonschema` crate (draft 2020-12);
//! coercion, error explanations and `$ref` bookkeeping are ports of the
//! upstream implementation, so messages, ordering and union blame match
//! pi-schema exactly.
//!
//! Upstream keys schema nodes by object identity; schema nodes here are
//! `&Value` references into the owned schema document, and every index
//! keys on the pointer address of a node instead.
#![expect(
    clippy::excessive_nesting,
    clippy::collapsible_if,
    reason = "the port follows upstream's keyword-dispatch structure"
)]
pub mod annotations;
pub mod coerce;
pub mod formats;
pub mod validate;

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::sync::Mutex;

use serde_json::{Map, Value};

use crate::types::content::ToolCall;
use crate::types::tool::Tool;

/// `DEFAULT_BASE`: the base URI of a root schema without `$id`.
pub const DEFAULT_BASE: &str = "https://json-schema.invalid/root.json";

const SCHEMA_KEYWORDS: &[&str] = &[
    "additionalProperties",
    "propertyNames",
    "contains",
    "not",
    "if",
    "then",
    "else",
    "unevaluatedProperties",
    "unevaluatedItems",
    "additionalItems",
    "contentSchema",
    "items",
];
const MAP_KEYWORDS: &[&str] = &[
    "properties",
    "patternProperties",
    "$defs",
    "definitions",
    "dependentSchemas",
    "dependencies",
];
const ARRAY_KEYWORDS: &[&str] = &["allOf", "anyOf", "oneOf", "prefixItems", "items"];

/// `isSchemaObject`: a plain object, not `true`/`false`, an array or a
/// primitive.
pub(crate) fn is_schema_object(value: &Value) -> bool {
    value.is_object()
}

fn node_key(node: &Value) -> usize {
    std::ptr::from_ref(node) as usize
}

/// `Object.is(a, b)`: `NaN` equal, `+0`/`-0` distinct, objects by
/// reference — the identity the coercer's re-entry guard uses.
pub(crate) fn js_object_is(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Number(x), Value::Number(y)) => x
            .as_f64()
            .is_some_and(|n| y.as_f64().is_some_and(|m| n.to_bits() == m.to_bits())),
        _ if a.is_array() || a.is_object() || b.is_array() || b.is_object() => std::ptr::eq(a, b),
        _ => a == b,
    }
}

/// `a === b` for JSON values: numbers by value (`NaN` never equal,
/// `+0 === -0`), objects by reference.
pub(crate) fn js_strict_eq(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Number(x), Value::Number(y)) => x
            .as_f64()
            .is_some_and(|n| y.as_f64().is_some_and(|m| n == m)),
        _ if a.is_array() || a.is_object() || b.is_array() || b.is_object() => std::ptr::eq(a, b),
        _ => a == b,
    }
}

/// JSON equality for `enum`/`const`/`uniqueItems` — upstream's
/// `deepEqual` on JS values (`0`/`-0`/`NaN` behave as `Object.is`).
pub(crate) fn deep_equal(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Number(x), Value::Number(y)) => x
            .as_f64()
            .is_some_and(|n| y.as_f64().is_some_and(|m| n.to_bits() == m.to_bits())),
        (Value::Array(xs), Value::Array(ys)) => {
            xs.len() == ys.len() && xs.iter().zip(ys).all(|(x, y)| deep_equal(x, y))
        }
        (Value::Object(xs), Value::Object(ys)) => {
            xs.len() == ys.len()
                && xs
                    .iter()
                    .all(|(k, x)| ys.get(k).is_some_and(|y| deep_equal(x, y)))
        }
        _ => a == b,
    }
}

/// `documentOf`: `uri` without its fragment.
fn document_of(uri: &str) -> &str {
    uri.find('#').map_or(uri, |hash| &uri[..hash])
}

/// `resolveUri`: `ref` resolved against `base`. `#frag` attaches to the
/// base document; an absolute `scheme:`-style ref stays absolute;
/// anything else joins under the base's last `/`.
fn resolve_uri(reference: &str, base: &str) -> String {
    if reference.starts_with('#') {
        return format!("{}{}", document_of(base), reference);
    }
    if reference.contains(':') {
        return reference.to_owned();
    }
    base.rfind('/').map_or_else(
        || reference.to_owned(),
        |at| format!("{}{reference}", &base[..=at]),
    )
}

/// `Resolved`: the document-pointer and base of a resolved `$ref`
/// target — the node itself is looked up by pointer on demand.
#[derive(Clone, Debug)]
pub(crate) struct Resolved {
    /// JSON-Pointer of the target node inside its document.
    pub ptr: String,
    /// The base URI of the target.
    pub base: String,
}

/// `Resolver`: upstream `resolver.ts` for the root schema only —
/// indexes `$id`, `$anchor`, `$dynamicAnchor`, every sub-schema's base
/// and document pointer; resolves references; tracks which refs were
/// unresolved under `unresolvedRefs: "reject"`. External `schemas`
/// documents are not supported (tool validation registers none).
#[derive(Default)]
pub(crate) struct Resolver {
    /// Node identity to its indexed base URI.
    pub base_of: HashMap<usize, String>,
    /// Node identity to its document JSON-Pointer.
    pub ptr_of: HashMap<usize, String>,
    /// `doc#name` anchors to the node's document-pointer.
    anchors: HashMap<String, String>,
    /// Document URI to the node pointer of that resource's root.
    resources: HashMap<String, String>,
    /// The root document, for verifying pointer fragments.
    root: Value,
    /// Absolute URIs of refs that resolved to `false`.
    unresolved: HashSet<String>,
}

impl Resolver {
    /// `baseFor`: `node`'s base given a parent `base` — `$id` changes it
    /// (draft 2020-12; draft-7's `$ref`-inert rule does not apply).
    pub(crate) fn base_for(&self, node: &Value, base: &str) -> String {
        if let Some(indexed) = self.base_of.get(&node_key(node)) {
            return indexed.clone();
        }
        node.get("$id").and_then(Value::as_str).map_or_else(
            || base.to_owned(),
            |id| document_of(&resolve_uri(id, base)).to_owned(),
        )
    }

    /// `ensureIndexed`: indexes the root schema — `$id` bases, anchors
    /// and node pointers.
    pub(crate) fn index(&mut self, root: &Value, root_base: &str) {
        self.root = root.clone();
        self.index_node(root, root_base);
        self.index_ptrs(root, "");
        self.resources.insert(root_base.to_owned(), String::new());
    }

    fn index_node(&mut self, node: &Value, base: &str) {
        if !is_schema_object(node) || self.base_of.contains_key(&node_key(node)) {
            return;
        }
        let mut b = base.to_owned();
        if let Some(id) = node.get("$id").and_then(Value::as_str) {
            let uri = resolve_uri(id, &b);
            b = document_of(&uri).into();
            if uri.len() > b.len() + 1 {
                self.anchors.insert(uri, String::new());
            }
        }
        self.base_of.insert(node_key(node), b.clone());
        if let Some(anchor) = node.get("$anchor").and_then(Value::as_str) {
            self.anchors.insert(
                format!("{b}#{anchor}"),
                self.ptr_of
                    .get(&node_key(node))
                    .cloned()
                    .unwrap_or_default(),
            );
        }
        if let Some(anchor) = node.get("$dynamicAnchor").and_then(Value::as_str) {
            self.anchors.insert(
                format!("{b}#{anchor}"),
                self.ptr_of
                    .get(&node_key(node))
                    .cloned()
                    .unwrap_or_default(),
            );
        }
        for key in SCHEMA_KEYWORDS {
            if let Some(child) = node.get(*key) {
                self.index_node(child, &b);
            }
        }
        for key in MAP_KEYWORDS {
            if let Some(map) = node.get(*key).and_then(Value::as_object) {
                for child in map.values() {
                    self.index_node(child, &b);
                }
            }
        }
        for key in ARRAY_KEYWORDS {
            if let Some(list) = node.get(*key).and_then(Value::as_array) {
                for child in list {
                    self.index_node(child, &b);
                }
            }
        }
    }

    /// Records the JSON-pointer position of every sub-schema node.
    fn index_ptrs(&mut self, node: &Value, pointer: &str) {
        self.ptr_of.insert(node_key(node), pointer.to_owned());
        let Some(map) = node.as_object() else { return };
        for key in SCHEMA_KEYWORDS {
            if let Some(child) = map.get(*key) {
                self.index_ptrs(child, &format!("{pointer}/{}", escape_segment(key)));
            }
        }
        for key in MAP_KEYWORDS {
            if let Some(sub) = map.get(*key).and_then(Value::as_object) {
                for (k, child) in sub {
                    self.index_ptrs(
                        child,
                        &format!("{pointer}/{}/{}", escape_segment(key), escape_segment(k)),
                    );
                }
            }
        }
        for key in ARRAY_KEYWORDS {
            if let Some(list) = map.get(*key).and_then(Value::as_array) {
                for (i, child) in list.iter().enumerate() {
                    self.index_ptrs(child, &format!("{pointer}/{}/{}", escape_segment(key), i));
                }
            }
        }
    }

    /// `resolve`: the document-pointer and base `ref` names under
    /// `base` — `Err(())` when it names nothing (rejected refs resolve
    /// to `false`).
    pub(crate) fn resolve(&mut self, reference: &str, base: &str) -> Result<Resolved, ()> {
        if let Some(resolved) = self.find(reference, base) {
            return Ok(resolved);
        }
        self.unresolved.insert(resolve_uri(reference, base));
        Err(())
    }

    /// `isUnresolved`: whether `ref` from `base` resolved to `false`.
    pub(crate) fn is_unresolved(&self, reference: &str, base: &str) -> bool {
        self.unresolved.contains(&resolve_uri(reference, base))
    }

    fn find(&self, reference: &str, base: &str) -> Option<Resolved> {
        let uri = resolve_uri(reference, base);
        let doc = document_of(&uri);
        let fragment = if uri.len() > doc.len() {
            &uri[doc.len() + 1..]
        } else {
            ""
        };
        // All resources live inside the root document for tool schemas:
        // the root itself and every `$id` subtree. A `doc` that is not
        // the root or a known `$id` names nothing.
        let root_ptr = self.resources.get(doc)?;
        if fragment.is_empty() {
            return Some(Resolved {
                ptr: root_ptr.clone(),
                base: doc.to_owned(),
            });
        }
        let fragment = percent_decode(fragment);
        if fragment.starts_with('/') {
            // Verify the pointer names a node; upstream resolves to
            // `false` when it does not.
            let root_node = self.resources.get(doc)?.clone();
            let doc_root = if root_node.is_empty() {
                &self.root
            } else {
                node_at(&self.root, &root_node)?
            };
            node_at(doc_root, &fragment)?;
            return Some(Resolved {
                ptr: fragment,
                base: doc.to_owned(),
            });
        }
        let target = self.anchors.get(&format!("{doc}#{fragment}"))?;
        Some(Resolved {
            ptr: target.clone(),
            base: doc.to_owned(),
        })
    }
}

fn escape_segment(key: &str) -> String {
    key.replace('~', "~0").replace('/', "~1")
}

fn percent_decode(text: &str) -> String {
    let mut out = String::new();
    let bytes = text.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes.get(i) == Some(&b'%')
            && i + 2 < bytes.len()
            && let Ok(hex) = u8::from_str_radix(text.get(i + 1..i + 3).unwrap_or(""), 16)
            && hex.is_ascii()
        {
            out.push(hex as char);
            i += 3;
            continue;
        }
        if let Some(&byte) = bytes.get(i) {
            out.push(byte as char);
        }
        i += 1;
    }
    out
}

/// The node at a document-pointer inside `root` (`""` is the root).
pub(crate) fn node_at<'v>(root: &'v Value, pointer: &str) -> Option<&'v Value> {
    if pointer.is_empty() {
        return Some(root);
    }
    let mut node = root;
    for raw in pointer.split('/').skip(1) {
        let key = raw.replace("~1", "/").replace("~0", "~");
        node = match node {
            Value::Object(map) => map.get(&key)?,
            Value::Array(list) => raw.parse::<usize>().ok().and_then(|i| list.get(i))?,
            _ => return None,
        };
    }
    Some(node)
}

/// A reusable validator for one tool's parameter schema.
///
/// Upstream's `CoercingValidator` (`compile` with `coerce: true` and
/// `unresolvedRefs: "reject"`). The caller owns it and may cache it per
/// tool, like upstream's `WeakMap` by schema object.
pub struct ToolSchemaValidator {
    /// The schema as written (coercion and explanations walk it).
    original: Value,
    /// The schema with unresolved `$ref`s replaced by `false`, for the
    /// compiled check (upstream resolves them to `false`).
    preprocessed: Value,
    /// The compiled root check.
    check: jsonschema::Validator,
    /// Compiled `{"$ref": "<root>#<ptr>"}` validators by pointer.
    compiled: Mutex<HashMap<String, jsonschema::Validator>>,
    /// The lazily built `referencing` registry for sub-schema compiles.
    registry_cache: RefCell<Option<referencing::Registry<'static>>>,
    /// Node identity to document pointer (for `ok(node, v)` and
    /// `passes(node, v)`); `RefCell` because `resolve` records
    /// unresolved refs.
    resolver: RefCell<Resolver>,
    /// The base URI `ptr_of` paths hang under (`$id` of the root, else
    /// [`DEFAULT_BASE`]).
    root_base: String,
}

impl std::fmt::Debug for ToolSchemaValidator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ToolSchemaValidator")
            .field("original", &self.original)
            .field("root_base", &self.root_base)
            .finish_non_exhaustive()
    }
}

impl ToolSchemaValidator {
    /// `compile(schema, { formats, coerce: true, unresolvedRefs:
    /// "reject" })` for a tool's `parameters` object.
    ///
    /// # Errors
    ///
    /// `Err` when `jsonschema` rejects the preprocessed schema.
    pub fn compile(schema: &Value) -> Result<Self, SchemaError> {
        let original = schema.clone();
        let root_base = original.get("$id").and_then(Value::as_str).map_or_else(
            || DEFAULT_BASE.to_owned(),
            |id| document_of(&resolve_uri(id, DEFAULT_BASE)).to_owned(),
        );
        let mut resolver = Resolver::default();
        resolver.index(&original, &root_base);
        // Index `$id`-registered resources so a `$ref` to a resource doc
        // resolves (tool schemas use intra-document refs).
        index_resources(&original, &mut resolver);
        let mut preprocessed = original.clone();
        mark_unresolved(&mut resolver, &original, &mut preprocessed, &root_base);
        let check = compile_root(&preprocessed)?;
        Ok(Self {
            original,
            preprocessed,
            check,
            compiled: Mutex::new(HashMap::new()),
            registry_cache: RefCell::new(None),
            resolver: RefCell::new(resolver),
            root_base,
        })
    }

    /// `check`: whether `value` is valid under the root schema.
    pub fn check(&self, value: &Value) -> bool {
        self.check.is_valid(value)
    }

    /// `coerce`: `value` coerced towards the schema (upstream's
    /// three-pass coercion). Copies only the changed paths.
    #[must_use]
    pub fn coerce(&self, value: &Value) -> Value {
        coerce::Coercer::new(self).coerce(value)
    }

    /// `errors`: the upstream `ValidationError` list of a failing value.
    pub fn errors(&self, value: &Value) -> Vec<ValidationError> {
        if self.check(value) {
            return Vec::new();
        }
        validate::Explainer::new(self).errors(value)
    }

    /// `parse`: coerce, then check — the coerced value when valid, else
    /// its errors.
    ///
    /// # Errors
    ///
    /// `Err(errors)` when the coerced value fails the schema.
    pub fn parse(&self, value: &Value) -> Result<Value, Vec<ValidationError>> {
        let coerced = self.coerce(value);
        if self.check(&coerced) {
            Ok(coerced)
        } else {
            Err(self.errors(&coerced))
        }
    }

    /// The original schema (resolver, coerce and explain walk it).
    pub(crate) const fn original(&self) -> &Value {
        &self.original
    }

    /// The base URI document pointers hang under.
    pub(crate) fn root_base(&self) -> &str {
        &self.root_base
    }

    /// `baseOf`/`baseFor` of a schema node through the resolver.
    pub(crate) fn base_of(&self, node: &Value, base: &str) -> String {
        self.resolver.borrow().base_for(node, base)
    }

    /// `resolve` through the resolver — `Err(())` when `ref` names
    /// nothing (it then counts as `false` upstream).
    pub(crate) fn resolve(&self, reference: &str, base: &str) -> Result<Resolved, ()> {
        self.resolver.borrow_mut().resolve(reference, base)
    }

    /// `isUnresolved` through the resolver.
    pub(crate) fn is_unresolved(&self, reference: &str, base: &str) -> bool {
        self.resolver.borrow().is_unresolved(reference, base)
    }

    /// The document-pointer of a schema node (identity by address), if
    /// it is a node of this schema.
    pub(crate) fn ptr_of(&self, node: &Value) -> Option<String> {
        self.resolver.borrow().ptr_of.get(&node_key(node)).cloned()
    }

    /// The node at `ptr` inside the original schema.
    pub(crate) fn node_at_ptr(&self, ptr: &str) -> Option<&Value> {
        node_at(&self.original, ptr)
    }

    /// The registry carrying the preprocessed root under `root_base`.
    fn registry(&self) -> Result<referencing::Registry<'static>, ()> {
        let mut cache = self.registry_cache.borrow_mut();
        if let Some(registry) = &*cache {
            return Ok(registry.clone());
        }
        let registry = referencing::Registry::new()
            .draft(referencing::Draft::Draft202012)
            .add(
                self.root_base.clone(),
                referencing::Resource::from_contents(self.preprocessed.clone()),
            )
            .map_err(|_| ())?
            .prepare()
            .map_err(|_| ())?;
        *cache = Some(registry.clone());
        Ok(registry)
    }

    /// `ok(node, v)` / `passes(node, v)`: the compiled check of a
    /// sub-schema in document context — `{"$ref": "<root>#<ptr>"}`
    /// compiled once per pointer. `Err(())` when the node is not part of
    /// the schema (compiled standalone) or cannot compile.
    pub(crate) fn check_node(&self, node: &Value, value: &Value) -> Result<bool, ()> {
        match node {
            Value::Bool(b) => return Ok(*b),
            Value::Object(_) => {}
            _ => return Ok(true),
        }
        let Some(ptr) = self.ptr_of(node) else {
            return jsonschema::Validator::options()
                .with_draft(jsonschema::Draft::Draft202012)
                .should_validate_formats(true)
                .build(node)
                .map(|v| v.is_valid(value))
                .map_err(|_| ());
        };
        let key = format!("{}#{}", self.root_base, ptr);
        if let Some(validator) = self
            .compiled
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get(&key)
        {
            return Ok(validator.is_valid(value));
        }
        let wrapper = serde_json::json!({ "$ref": key });
        let registry = self.registry()?;
        let validator = jsonschema::Validator::options()
            .with_draft(jsonschema::Draft::Draft202012)
            .should_validate_formats(true)
            .with_registry(&registry)
            .build(&wrapper)
            .map_err(|_| ())?;
        let valid = validator.is_valid(value);
        self.compiled
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .insert(key, validator);
        Ok(valid)
    }
}

fn compile_root(schema: &Value) -> Result<jsonschema::Validator, SchemaError> {
    jsonschema::Validator::options()
        .with_draft(jsonschema::Draft::Draft202012)
        .should_validate_formats(true)
        .build(schema)
        .map_err(|error| SchemaError {
            message: error.to_string(),
        })
}

/// Registers each `$id`-declared resource's document so a `$ref` to it
/// resolves to a node pointer (tool schemas stay in one document).
fn index_resources(node: &Value, resolver: &mut Resolver) {
    let Some(map) = node.as_object() else { return };
    if let Some(id) = map.get("$id").and_then(Value::as_str) {
        if let Some(ptr) = resolver.ptr_of.get(&node_key(node)) {
            resolver
                .resources
                .insert(document_of(id).to_owned(), ptr.clone());
        }
    }
    for child in map.values() {
        index_resources(child, resolver);
    }
}

/// Replaces each schema node whose `$ref` resolves to nothing with
/// `false` (upstream compiles it so with `unresolvedRefs: "reject"`).
fn mark_unresolved(
    resolver: &mut Resolver,
    original: &Value,
    preprocessed: &mut Value,
    root_base: &str,
) {
    if !is_schema_object(preprocessed) {
        return;
    }
    if let Some(reference) = preprocessed.get("$ref").and_then(Value::as_str) {
        let base = resolver.base_for(original, root_base);
        if resolver.resolve(reference, &base).is_err() {
            *preprocessed = Value::Bool(false);
            return;
        }
    }
    for key in SCHEMA_KEYWORDS {
        if let (Some(child), Some(orig)) = (preprocessed.get_mut(*key), original.get(*key)) {
            mark_unresolved(resolver, orig, child, root_base);
        }
    }
    for key in MAP_KEYWORDS {
        if let (Some(map), Some(orig_map)) = (
            preprocessed.get_mut(*key).and_then(Value::as_object_mut),
            original.get(*key).and_then(Value::as_object),
        ) {
            for (k, child) in map.iter_mut() {
                if let Some(orig) = orig_map.get(k) {
                    mark_unresolved(resolver, orig, child, root_base);
                }
            }
        }
    }
    for key in ARRAY_KEYWORDS {
        if let (Some(list), Some(orig_list)) = (
            preprocessed.get_mut(*key).and_then(Value::as_array_mut),
            original.get(*key).and_then(Value::as_array),
        ) {
            for (i, child) in list.iter_mut().enumerate() {
                if let Some(orig) = orig_list.get(i) {
                    mark_unresolved(resolver, orig, child, root_base);
                }
            }
        }
    }
}

/// `ValidationError`: one failure of a value — a JSON-Pointer path
/// (`""` for the value itself), the failing keyword and a message.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ValidationError {
    /// JSON-Pointer path to the failing value (`""` for the root).
    pub path: String,
    /// The failing JSON-Schema keyword.
    pub keyword: String,
    /// The upstream error message.
    pub message: String,
}

/// `SchemaError`: the tool's parameter schema failed to compile.
#[derive(Clone, Debug)]
pub struct SchemaError {
    /// The compile error.
    pub message: String,
}

impl std::fmt::Display for SchemaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for SchemaError {}

/// `ToolArgumentsError`: the failures `validateToolCall`/
/// `validateToolArguments` report — the `Display` text is exactly what
/// the model receives upstream.
#[derive(Clone, Debug)]
pub enum ToolArgumentsError {
    /// `Tool "<name>" not found`.
    ToolNotFound {
        /// The call's tool name.
        name: String,
    },
    /// `Invalid arguments for tool "<name>": <error>` — the call's
    /// `argumentsError` (malformed JSON upstream of validation).
    ArgumentsError {
        /// The call's tool name.
        name: String,
        /// The recorded arguments error.
        error: String,
    },
    /// `Validation failed for tool "<name>": …` — coerce-and-check
    /// failed; the message lists each error and echoes the arguments.
    ValidationFailed {
        /// The full model-facing message (upstream's thrown text).
        message: String,
        /// The errors, in upstream order.
        errors: Vec<ValidationError>,
    },
    /// The tool's parameter schema failed to compile.
    Schema {
        /// The compile error.
        error: SchemaError,
    },
}

impl std::fmt::Display for ToolArgumentsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ToolNotFound { name } => write!(f, "Tool \"{name}\" not found"),
            Self::ArgumentsError { name, error } => {
                write!(f, "Invalid arguments for tool \"{name}\": {error}")
            }
            Self::ValidationFailed { message, .. } => write!(f, "{message}"),
            Self::Schema { error } => write!(f, "{error}"),
        }
    }
}

impl std::error::Error for ToolArgumentsError {}

/// `formatValidationPath`: a JSON Pointer as the dotted path the model
/// sees — `/edits/0/oldText` becomes `edits.0.oldText`; `""` is `root`.
#[must_use]
pub fn format_validation_path(pointer: &str) -> String {
    if pointer.is_empty() {
        return "root".to_owned();
    }
    pointer
        .strip_prefix('/')
        .unwrap_or(pointer)
        .split('/')
        .map(|segment| segment.replace("~1", "/").replace("~0", "~"))
        .collect::<Vec<_>>()
        .join(".")
}

/// `validateToolCall`: find `toolCall.name` in `tools`, then
/// [`validate_tool_arguments`].
///
/// # Errors
///
/// `ToolNotFound` when no tool has the call's name, else the
/// `validate_tool_arguments` error.
pub fn validate_tool_call(
    tools: &[Tool],
    tool_call: &ToolCall,
) -> Result<Map<String, Value>, ToolArgumentsError> {
    let tool = tools
        .iter()
        .find(|tool| tool.name == tool_call.name)
        .ok_or_else(|| ToolArgumentsError::ToolNotFound {
            name: tool_call.name.clone(),
        })?;
    validate_tool_arguments(tool, tool_call)
}

/// `validateToolArguments`.
///
/// Coerces `tool_call.arguments` with `tool.parameters`, then
/// validates — the coerced arguments on success (a copy; tools never
/// mutate the transcript). Fails when `argumentsError` is set
/// (malformed JSON) or when coercion plus check reports errors; the
/// error text is upstream's thrown message.
///
/// # Errors
///
/// See [`ToolArgumentsError`].
pub fn validate_tool_arguments(
    tool: &Tool,
    tool_call: &ToolCall,
) -> Result<Map<String, Value>, ToolArgumentsError> {
    let validator = ToolSchemaValidator::compile(&Value::Object(tool.parameters.clone()))
        .map_err(|error| ToolArgumentsError::Schema { error })?;
    validate_tool_arguments_with(&validator, tool_call)
}

/// `validateToolArguments` against an already-compiled validator —
/// callers that run a tool repeatedly compile its schema once.
///
/// # Errors
///
/// See [`ToolArgumentsError`].
pub fn validate_tool_arguments_with(
    validator: &ToolSchemaValidator,
    tool_call: &ToolCall,
) -> Result<Map<String, Value>, ToolArgumentsError> {
    if let Some(error) = &tool_call.arguments_error {
        return Err(ToolArgumentsError::ArgumentsError {
            name: tool_call.name.clone(),
            error: error.clone(),
        });
    }
    // A copy, so tools that modify their arguments never modify the
    // tool call in the transcript.
    let arguments = tool_call.arguments.clone();
    match validator.parse(&Value::Object(arguments)) {
        Ok(Value::Object(map)) => Ok(map),
        Ok(_) => Ok(Map::new()),
        Err(errors) => {
            let details = errors
                .iter()
                .map(|error| {
                    format!(
                        "  - {}: {}",
                        format_validation_path(&error.path),
                        error.message
                    )
                })
                .collect::<Vec<_>>()
                .join("\n");
            let message = format!(
                "Validation failed for tool \"{}\":\n{details}\n\nReceived arguments:\n{}",
                tool_call.name,
                serde_json::to_string_pretty(&tool_call.arguments).unwrap_or_default(),
            );
            Err(ToolArgumentsError::ValidationFailed { message, errors })
        }
    }
}
