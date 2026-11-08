//! Durable handle for deferred (async) provider responses.

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Handle identifying a deferred response on the provider side.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
#[non_exhaustive]
pub struct DeferredHandle {
    /// Provider that owns the deferred response.
    pub provider: String,
    /// Model the deferred response is for.
    pub model_id: String,
    /// API family of the deferred response.
    pub api: String,
    /// Provider token, such as a response id or batch id plus row id.
    pub id: String,
    /// Unix timestamp in milliseconds after which the handle is invalid.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expires_at: Option<i64>,
    /// Suggested delay in milliseconds before polling.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub poll_after_ms: Option<u64>,
    /// Provider data needed to reconstruct the final message.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
}
