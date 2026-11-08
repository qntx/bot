//! Image-generation context and result types.

use serde::{Deserialize, Serialize};

use super::api::{ImageApi, ProviderId};
use super::content::UserContentBlock;
use super::usage::Usage;

/// Input to an image-generation request.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ImagesContext {
    /// Text and image input blocks.
    pub input: Vec<UserContentBlock>,
}

/// Why an image generation terminated.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ImagesStopReason {
    /// Natural stop.
    #[serde(rename = "stop")]
    Stop,
    /// The request failed.
    #[serde(rename = "error")]
    Error,
    /// The request was aborted.
    #[serde(rename = "aborted")]
    Aborted,
}

/// Result of an image-generation request.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
#[non_exhaustive]
pub struct AssistantImages {
    /// Image API family that produced the result.
    pub api: ImageApi,
    /// Provider that produced the result.
    pub provider: ProviderId,
    /// Model that produced the result.
    pub model: String,
    /// Generated text and image output blocks.
    pub output: Vec<UserContentBlock>,
    /// Provider-specific response identifier.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_id: Option<String>,
    /// Token usage, when the service reports it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
    /// Why the generation terminated.
    pub stop_reason: ImagesStopReason,
    /// Error text when `stopReason` is `"error"` or `"aborted"`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error_message: Option<String>,
    /// Unix timestamp in milliseconds.
    pub timestamp: i64,
}
