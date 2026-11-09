//! `AI-CATALOG-EMBED`: the embedded catalog against pi's built-in getters.

use ovo_ai::catalog;
use ovo_ai::provider::{ModelType, model_ids};
use ovo_ai::types::model::AnyModel;
use serde_json::{Map, Value};

use crate::support::js_value_eq;

const CATALOG_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/catalog");

fn vectors() -> Vec<Value> {
    let text = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../vectors/ai/catalog.json"
    ));
    let document: Value = serde_json::from_str(text).unwrap_or_else(|e| panic!("vector file: {e}"));
    document["cases"].as_array().cloned().unwrap_or_default()
}

fn ids(models: &[AnyModel]) -> Vec<Value> {
    models
        .iter()
        .map(|model| Value::from(model_ids(model).1))
        .collect()
}

/// parity: AI-CATALOG-EMBED
#[test]
fn catalog_model_order_matches_upstream() {
    let cases = vectors();
    assert_eq!(cases.len(), catalog::providers().count(), "provider count");
    for case in &cases {
        let provider = case["provider"].as_str().unwrap_or_default();
        for (kind, field) in [
            (ModelType::Chat, "chat"),
            (ModelType::Image, "image"),
            (ModelType::Classifier, "classifier"),
        ] {
            let models =
                catalog::models(provider, kind).unwrap_or_else(|e| panic!("{provider}: {e}"));
            assert_eq!(
                Value::Array(ids(&models)),
                case[field],
                "{provider}: {field} ids"
            );
        }
    }
}

/// Every embedded model reads without losing or adding fields.
#[test]
fn catalog_models_round_trip() {
    for provider in catalog::providers() {
        let text = std::fs::read_to_string(format!("{CATALOG_DIR}/{provider}.json"))
            .unwrap_or_else(|e| panic!("{provider}: {e}"));
        let groups: Map<String, Value> =
            serde_json::from_str(&text).unwrap_or_else(|e| panic!("{provider}: {e}"));
        let raw: Vec<&Value> = groups
            .values()
            .filter_map(Value::as_object)
            .flat_map(Map::values)
            .collect();
        let models = catalog::all_models(provider).unwrap_or_else(|e| panic!("{provider}: {e}"));
        assert_eq!(models.len(), raw.len(), "{provider}: model count");
        for model in &models {
            let id = model_ids(model).1;
            let written = serde_json::to_value(model).unwrap_or_default();
            // Ids are unique per type; the last entry of a repeated id wins.
            let original = raw
                .iter()
                .rev()
                .find(|value| value["id"] == id && value["type"] == written["type"])
                .unwrap_or_else(|| panic!("{provider}/{id}: not in the catalog file"));
            assert!(
                js_value_eq(&written, original),
                "{provider}/{id}\n  read: {written}\n  file: {original}"
            );
        }
    }
}

/// The embedded catalogs are exactly the files `sync-catalog.ts` maintains.
#[test]
fn catalog_embeds_every_synced_file() {
    let mut files: Vec<String> = std::fs::read_dir(CATALOG_DIR)
        .unwrap_or_else(|e| panic!("{CATALOG_DIR}: {e}"))
        .filter_map(Result::ok)
        .filter_map(|entry| {
            entry
                .file_name()
                .to_str()
                .and_then(|name| name.strip_suffix(".json"))
                .map(str::to_owned)
        })
        .collect();
    files.sort();
    let embedded: Vec<String> = catalog::providers().map(str::to_owned).collect();
    assert_eq!(embedded, files);
    let pin = std::fs::read_to_string(format!("{CATALOG_DIR}/PIN")).unwrap_or_default();
    assert_eq!(catalog::PIN, pin.trim());
    assert!(catalog::models("unknown", ModelType::Chat).is_err());
}
