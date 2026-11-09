//! The built-in model catalog (ADR-006): the pinned pi-ai package's
//! `dist/providers/data/<provider>.json` for every registered provider,
//! embedded and parsed on first use.
//!
//! `scripts/sync-catalog.ts` copies the files and `scripts/check-catalog.ts`
//! keeps them byte-identical to the package; [`PIN`] names its version.

use std::sync::LazyLock;

use serde_json::{Map, Value};

use crate::models_error::{ModelsError, ModelsErrorCode};
use crate::provider::{ModelType, model_ids, model_type};
use crate::types::model::AnyModel;

/// The `@earendil-works/pi-ai` version the catalog comes from.
pub const PIN: &str = include_str!("../catalog/PIN").trim_ascii();

/// The embedded catalog files, by provider id.
const CATALOGS: [(&str, &str); 6] = [
    ("anthropic", include_str!("../catalog/anthropic.json")),
    ("deepseek", include_str!("../catalog/deepseek.json")),
    ("groq", include_str!("../catalog/groq.json")),
    ("openai", include_str!("../catalog/openai.json")),
    ("openrouter", include_str!("../catalog/openrouter.json")),
    ("xai", include_str!("../catalog/xai.json")),
];

/// A catalog's chat, image and classifier models, each in pi's order.
#[derive(Debug)]
struct Flattened {
    chat: Vec<AnyModel>,
    image: Vec<AnyModel>,
    classifier: Vec<AnyModel>,
}

/// A provider's flattened catalog, or why it does not parse.
type Parsed = (&'static str, Result<Flattened, String>);

/// The parsed catalogs, in [`CATALOGS`] order.
static PARSED: LazyLock<Vec<Parsed>> = LazyLock::new(|| {
    CATALOGS
        .iter()
        .map(|(provider, text)| (*provider, parse(text)))
        .collect()
});

/// The providers with an embedded catalog.
pub fn providers() -> impl Iterator<Item = &'static str> {
    CATALOGS.iter().map(|(provider, _)| *provider)
}

/// The provider's built-in models of one type, in pi's order
/// (`getBuiltinModels`, `getBuiltinImageModels`,
/// `getBuiltinClassifierModels`).
///
/// # Errors
/// Returns a `model_source` error when the provider has no embedded catalog
/// or its catalog does not parse.
pub fn models(provider: &str, kind: ModelType) -> Result<Vec<AnyModel>, ModelsError> {
    let flattened = parsed(provider)?;
    Ok(match kind {
        ModelType::Chat => flattened.chat.clone(),
        ModelType::Image => flattened.image.clone(),
        ModelType::Classifier => flattened.classifier.clone(),
    })
}

/// `getAllBuiltinModels`: chat, then image, then classifier models.
///
/// # Errors
/// As [`models`].
pub fn all_models(provider: &str) -> Result<Vec<AnyModel>, ModelsError> {
    let flattened = parsed(provider)?;
    Ok(flattened
        .chat
        .iter()
        .chain(&flattened.image)
        .chain(&flattened.classifier)
        .cloned()
        .collect())
}

fn parsed(provider: &str) -> Result<&'static Flattened, ModelsError> {
    let error = |message: String| ModelsError::new(ModelsErrorCode::ModelSource, message);
    let Some((_, catalog)) = PARSED.iter().find(|(id, _)| *id == provider) else {
        return Err(error(format!(
            "No built-in catalog for provider: {provider}"
        )));
    };
    catalog.as_ref().map_err(|message| {
        error(format!(
            "Built-in catalog for {provider} is invalid: {message}"
        ))
    })
}

/// Read a catalog file (models grouped by API) and flatten each type.
fn parse(text: &str) -> Result<Flattened, String> {
    let groups: Map<String, Value> =
        crate::json::from_str(text).map_err(|error| error.to_string())?;
    let mut models: Vec<AnyModel> = Vec::new();
    for group in groups.values() {
        let Some(group) = group.as_object() else {
            return Err(String::from("an API group is not an object"));
        };
        for model in group.values() {
            models.push(serde_json::from_value(model.clone()).map_err(|error| error.to_string())?);
        }
    }
    Ok(Flattened {
        chat: flatten(&models, ModelType::Chat),
        image: flatten(&models, ModelType::Image),
        classifier: flatten(&models, ModelType::Classifier),
    })
}

/// `flattenModelCatalog`: the models of one type across the API groups,
/// keyed by id like `Object.fromEntries` (a repeated id keeps its first
/// position and takes the last value), in JavaScript property order:
/// canonical array-index ids first, ascending, then the rest in order.
fn flatten(models: &[AnyModel], kind: ModelType) -> Vec<AnyModel> {
    let mut by_id: Vec<(&str, &AnyModel)> = Vec::new();
    for model in models.iter().filter(|model| model_type(model) == kind) {
        let id = model_ids(model).1;
        match by_id.iter_mut().find(|(other, _)| *other == id) {
            Some(slot) => slot.1 = model,
            None => by_id.push((id, model)),
        }
    }
    let (mut indexed, named): (Vec<_>, Vec<_>) = by_id
        .into_iter()
        .map(|(id, model)| (array_index(id), model))
        .partition(|(index, _)| index.is_some());
    indexed.sort_by_key(|(index, _)| *index);
    indexed
        .into_iter()
        .chain(named)
        .map(|(_, model)| model.clone())
        .collect()
}

/// The numeric value of a canonical array-index key (`0` to `2^32 - 2`, no
/// leading zeros).
fn array_index(key: &str) -> Option<u32> {
    key.parse::<u32>()
        .ok()
        .filter(|index| *index != u32::MAX && index.to_string() == key)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recognizes_canonical_array_indices() {
        let indices: Vec<Option<u32>> = ["b", "10", "0", "02", "4294967294", "4294967295", "-1"]
            .iter()
            .map(|key| array_index(key))
            .collect();
        assert_eq!(
            indices,
            [
                None,
                Some(10),
                Some(0),
                None,
                Some(4_294_967_294),
                None,
                None
            ]
            .to_vec()
        );
    }
}
