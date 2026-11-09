//! `headers.ts`: header records and `providerHeadersToRecord`'s
//! case-insensitive merge with `null` suppression.

use serde_json::{Map, Value};

/// `Record<string, string>` as upstream sees it (`Headers` already
/// folds case; lookup is case-insensitive).
#[derive(Clone, Debug, Default)]
pub struct HeaderRecord {
    /// `(display-name, value)` keyed by lowercase name.
    entries: Vec<(String, String)>,
}

impl HeaderRecord {
    /// `headers.get(name)` — case-insensitive.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&str> {
        let needle = name.to_lowercase();
        self.entries
            .iter()
            .find(|(k, _)| *k == needle)
            .map(|(_, v)| v.as_str())
    }

    /// `headers.entries()` in order.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &str)> {
        self.entries.iter().map(|(k, v)| (k.as_str(), v.as_str()))
    }

    /// `Object.entries` over a JSON `Record<string, string>`.
    #[must_use]
    pub fn from_record(record: &Map<String, Value>) -> Self {
        let mut entries = Vec::new();
        for (key, value) in record {
            if let Some(v) = value.as_str() {
                entries.push((key.clone(), v.to_owned()));
            }
        }
        Self { entries }
    }
}

/// `ProviderHeaders = Record<string, string | null>` — a `null` value
/// suppresses a provider default header of the same name.
pub type ProviderHeaders = Map<String, Value>;

/// `providerHeadersToRecord`: later sources override; a `null` removes
/// the name; the surviving entry keeps its last-written display case.
/// `undefined` when nothing is left.
#[must_use]
pub fn provider_headers_to_record(
    sources: &[Option<&ProviderHeaders>],
) -> Option<Map<String, Value>> {
    let mut merged: Vec<(String, (String, String))> = Vec::new();
    for source in sources.iter().copied().flatten() {
        for (name, value) in source {
            let normalized = name.to_lowercase();
            merged.retain(|(k, _)| k != &normalized);
            if let Some(v) = value.as_str() {
                merged.push((normalized, (name.clone(), v.to_owned())));
            }
        }
    }
    if merged.is_empty() {
        return None;
    }
    let mut out = Map::new();
    for (_, (display, value)) in merged {
        out.insert(display, Value::String(value));
    }
    Some(out)
}
