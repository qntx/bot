//! Catalog model types from `providers/model-schema.ts` and `types.ts`.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use super::api::{Api, ClassifierApi, ImageApi, ProviderId};

/// Pi thinking level (`THINKING_LEVELS`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ThinkingLevel {
    /// Minimal thinking.
    #[serde(rename = "minimal")]
    Minimal,
    /// Low thinking.
    #[serde(rename = "low")]
    Low,
    /// Medium thinking.
    #[serde(rename = "medium")]
    Medium,
    /// High thinking.
    #[serde(rename = "high")]
    High,
    /// Extra-high thinking.
    #[serde(rename = "xhigh")]
    Xhigh,
    /// Maximum thinking.
    #[serde(rename = "max")]
    Max,
}

/// Thinking level including `"off"` (`MODEL_THINKING_LEVELS`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelThinkingLevel {
    /// Thinking disabled.
    #[serde(rename = "off")]
    Off,
    /// Minimal thinking.
    #[serde(rename = "minimal")]
    Minimal,
    /// Low thinking.
    #[serde(rename = "low")]
    Low,
    /// Medium thinking.
    #[serde(rename = "medium")]
    Medium,
    /// High thinking.
    #[serde(rename = "high")]
    High,
    /// Extra-high thinking.
    #[serde(rename = "xhigh")]
    Xhigh,
    /// Maximum thinking.
    #[serde(rename = "max")]
    Max,
}

/// Provider value for a thinking level; `null` marks it unsupported.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ThinkingLevelMapValue {
    /// Provider-native level string.
    Value(String),
    /// Level unsupported (`null`).
    Unsupported,
}

/// Maps pi thinking levels to provider/model-specific values.
///
/// Each key is tri-state: absent means the provider default, `null` marks the
/// level unsupported, a string is the provider-native value.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ThinkingLevelMap {
    /// Mapping for the `off` level.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "deserialize_some"
    )]
    pub off: Option<ThinkingLevelMapValue>,
    /// Mapping for the `minimal` level.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "deserialize_some"
    )]
    pub minimal: Option<ThinkingLevelMapValue>,
    /// Mapping for the `low` level.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "deserialize_some"
    )]
    pub low: Option<ThinkingLevelMapValue>,
    /// Mapping for the `medium` level.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "deserialize_some"
    )]
    pub medium: Option<ThinkingLevelMapValue>,
    /// Mapping for the `high` level.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "deserialize_some"
    )]
    pub high: Option<ThinkingLevelMapValue>,
    /// Mapping for the `xhigh` level.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "deserialize_some"
    )]
    pub xhigh: Option<ThinkingLevelMapValue>,
    /// Mapping for the `max` level.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "deserialize_some"
    )]
    pub max: Option<ThinkingLevelMapValue>,
}

/// Deserialize `T` then wrap in `Some`, so `Option<T>` fields can
/// distinguish an absent key (the `default` gives `None`) from an explicit
/// `null` (`Some(T)` where `T` decodes `null`).
fn deserialize_some<'de, D, T>(deserializer: D) -> Result<Option<T>, D::Error>
where
    D: serde::Deserializer<'de>,
    T: Deserialize<'de>,
{
    T::deserialize(deserializer).map(Some)
}

/// Input modality of a model.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelInputModality {
    /// Text input.
    #[serde(rename = "text")]
    Text,
    /// Image input.
    #[serde(rename = "image")]
    Image,
}

/// Prompt cache retention preference.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum CacheRetention {
    /// No caching.
    #[serde(rename = "none")]
    None,
    /// Short-lived cache entries.
    #[serde(rename = "short")]
    Short,
    /// Long-lived cache entries.
    #[serde(rename = "long")]
    Long,
}

/// Cost in USD per million tokens.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
#[non_exhaustive]
pub struct ModelCostRates {
    /// Input cost.
    pub input: f64,
    /// Output cost.
    pub output: f64,
    /// Cache-read cost.
    pub cache_read: f64,
    /// Cache-write cost.
    pub cache_write: f64,
}

/// A request-wide pricing tier; the highest matching threshold applies.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
#[non_exhaustive]
pub struct ModelCostTier {
    /// Applies when total request input exceeds this token count.
    pub input_tokens_above: f64,
    /// Input cost in this tier.
    pub input: f64,
    /// Output cost in this tier.
    pub output: f64,
    /// Cache-read cost in this tier.
    pub cache_read: f64,
    /// Cache-write cost in this tier.
    pub cache_write: f64,
}

/// Catalog cost of a model.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
#[non_exhaustive]
pub struct ModelCost {
    /// Input cost.
    pub input: f64,
    /// Output cost.
    pub output: f64,
    /// Cache-read cost.
    pub cache_read: f64,
    /// Cache-write cost.
    pub cache_write: f64,
    /// Request-wide pricing tiers.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tiers: Option<Vec<ModelCostTier>>,
}

/// Prompt cache lifetimes in seconds per retention tier.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ModelPromptCache {
    /// Lifetime of short-retention entries.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub short: Option<f64>,
    /// Lifetime of long-retention entries.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub long: Option<f64>,
}

/// Cache-safe image resize profile.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
#[non_exhaustive]
pub struct ModelImageResizeOptions {
    /// Maximum image width in pixels.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_width: Option<u64>,
    /// Maximum image height in pixels.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_height: Option<u64>,
    /// Maximum base64-encoded payload size in bytes.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_bytes: Option<u64>,
    /// JPEG re-encode quality, 1–100.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub jpeg_quality: Option<u64>,
}

/// Image input limits.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
#[non_exhaustive]
pub struct ModelImageInputLimits {
    /// Cache-safe resize profile applied before a new image enters history.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub resize: Option<ModelImageResizeOptions>,
    /// Maximum images accepted in one provider message.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_per_message: Option<u64>,
    /// Maximum images accepted across one provider request.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_per_request: Option<u64>,
}

/// Provider input limits and cache-safe preprocessing metadata.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
#[non_exhaustive]
pub struct ModelInputLimits {
    /// Maximum serialized provider request size in bytes.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_request_bytes: Option<u64>,
    /// Image-specific limits.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub images: Option<ModelImageInputLimits>,
}

/// The literal `"chat"` type tag (present or absent).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ChatType {
    /// Chat model.
    #[serde(rename = "chat")]
    Chat,
}

/// Chat model: usable with `stream()` and friends. `type` is optional —
/// absent means chat.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
#[non_exhaustive]
pub struct Model {
    /// Model identifier.
    pub id: String,
    /// Display name.
    pub name: String,
    /// Chat API family.
    pub api: Api,
    /// Provider offering the model.
    pub provider: ProviderId,
    /// API endpoint base URL.
    pub base_url: String,
    /// Accepted input modalities.
    pub input: Vec<ModelInputModality>,
    /// Provider input limits and cache-safe preprocessing metadata.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_limits: Option<ModelInputLimits>,
    /// Catalog cost.
    pub cost: ModelCost,
    /// Extra HTTP headers the provider requires.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub headers: Option<Map<String, Value>>,
    /// Model type tag; absent means chat.
    #[serde(skip_serializing_if = "Option::is_none", rename = "type")]
    pub kind: Option<ChatType>,
    /// Whether the model supports thinking.
    pub reasoning: bool,
    /// Maps pi thinking levels to provider/model-specific values.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking_level_map: Option<ThinkingLevelMap>,
    /// Prompt cache lifetimes per retention tier.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_cache: Option<ModelPromptCache>,
    /// Context window in tokens.
    pub context_window: u64,
    /// Maximum output tokens.
    pub max_tokens: u64,
    /// Default sampling parameters for this model.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sampling_params: Option<Map<String, Value>>,
    /// Sampling parameter overrides selected by the pi thinking level.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sampling_params_by_thinking_level: Option<Map<String, Value>>,
    /// Compatibility overrides; shape depends on `api`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub compat: Option<Map<String, Value>>,
    /// Unknown catalog fields preserved verbatim.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

/// The literal `"image"` type tag.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ImageModelType {
    /// Image model.
    #[serde(rename = "image")]
    Image,
}

/// Output modality of an image model.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ImageModelOutput {
    /// Text output.
    #[serde(rename = "text")]
    Text,
    /// Image output.
    #[serde(rename = "image")]
    Image,
}

/// Image-generation model.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
#[non_exhaustive]
pub struct ImageModel {
    /// Model identifier.
    pub id: String,
    /// Display name.
    pub name: String,
    /// Image API family.
    pub api: ImageApi,
    /// Provider offering the model.
    pub provider: ProviderId,
    /// API endpoint base URL.
    pub base_url: String,
    /// Accepted input modalities.
    pub input: Vec<ModelInputModality>,
    /// Provider input limits and cache-safe preprocessing metadata.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_limits: Option<ModelInputLimits>,
    /// Catalog cost.
    pub cost: ModelCost,
    /// Extra HTTP headers the provider requires.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub headers: Option<Map<String, Value>>,
    /// Model type tag.
    #[serde(rename = "type")]
    pub kind: ImageModelType,
    /// Output modalities; always includes `"image"`.
    pub output: Vec<ImageModelOutput>,
    /// Unknown catalog fields preserved verbatim.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

/// The literal `"classifier"` type tag.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ClassifierModelType {
    /// Classifier model.
    #[serde(rename = "classifier")]
    Classifier,
}

/// Structured classifier model.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
#[non_exhaustive]
pub struct ClassifierModel {
    /// Model identifier.
    pub id: String,
    /// Display name.
    pub name: String,
    /// Classifier API family.
    pub api: ClassifierApi,
    /// Provider offering the model.
    pub provider: ProviderId,
    /// API endpoint base URL.
    pub base_url: String,
    /// Accepted input modalities.
    pub input: Vec<ModelInputModality>,
    /// Provider input limits and cache-safe preprocessing metadata.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_limits: Option<ModelInputLimits>,
    /// Catalog cost.
    pub cost: ModelCost,
    /// Extra HTTP headers the provider requires.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub headers: Option<Map<String, Value>>,
    /// Model type tag.
    #[serde(rename = "type")]
    pub kind: ClassifierModelType,
    /// Context window in tokens.
    pub context_window: u64,
    /// Compatibility overrides for System One APIs.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub compat: Option<Map<String, Value>>,
    /// Unknown catalog fields preserved verbatim.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

/// Any catalog entry; `type` absent or `"chat"` means chat model.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(untagged)]
pub enum AnyModel {
    /// Chat model (`type` absent or `"chat"`).
    Chat(Box<Model>),
    /// Image-generation model.
    Image(ImageModel),
    /// Structured classifier model.
    Classifier(ClassifierModel),
}

impl<'de> Deserialize<'de> for AnyModel {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let value = Value::deserialize(deserializer)?;
        match value.get("type").and_then(Value::as_str) {
            None | Some("chat") => Model::deserialize(&value)
                .map(|model| Self::Chat(Box::new(model)))
                .map_err(serde::de::Error::custom),
            Some("image") => ImageModel::deserialize(&value)
                .map(Self::Image)
                .map_err(serde::de::Error::custom),
            Some("classifier") => ClassifierModel::deserialize(&value)
                .map(Self::Classifier)
                .map_err(serde::de::Error::custom),
            Some(other) => Err(serde::de::Error::custom(format!(
                "unknown model type {other:?}"
            ))),
        }
    }
}
