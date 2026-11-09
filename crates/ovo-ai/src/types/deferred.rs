//! Durable handle for deferred (async) provider responses.

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Handle identifying a deferred response on the provider side.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
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

impl DeferredHandle {
    /// `new` — a handle with the identity fields; optional fields default.
    #[must_use]
    pub fn new(
        provider: impl Into<String>,
        model_id: impl Into<String>,
        api: impl Into<String>,
        id: impl Into<String>,
    ) -> Self {
        Self {
            provider: provider.into(),
            model_id: model_id.into(),
            api: api.into(),
            id: id.into(),
            expires_at: None,
            poll_after_ms: None,
            data: None,
        }
    }
}
