//! Token usage and cost accounting attached to assistant messages.

use serde::{Deserialize, Serialize};

/// Token usage plus catalog- or provider-derived cost.
///
/// Base fields come from the upstream `Usage` literal; `cacheWrite1h` and
/// `reasoning` are appended by adapters when the provider reports them
/// (anthropic-messages.ts:723, :874) and serialize last.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
#[non_exhaustive]
pub struct Usage {
    /// Input (non-cached) tokens.
    pub input: u64,
    /// Output tokens.
    pub output: u64,
    /// Cache-read tokens.
    pub cache_read: u64,
    /// Cache-write tokens.
    pub cache_write: u64,
    /// Total tokens.
    pub total_tokens: u64,
    /// Cost breakdown for this usage record.
    pub cost: UsageCost,
    /// Subset of `cacheWrite` written with 1h retention (Anthropic only).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_write1h: Option<u64>,
    /// Reasoning/thinking tokens, a subset of `output` when reported.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<u64>,
}

/// Cost breakdown in USD for a usage record.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
#[non_exhaustive]
pub struct UsageCost {
    /// Catalog-derived input cost.
    pub input: f64,
    /// Catalog-derived output cost.
    pub output: f64,
    /// Catalog-derived cache-read cost.
    pub cache_read: f64,
    /// Catalog-derived cache-write cost.
    pub cache_write: f64,
    /// Provider-reported billed cost when available, else catalog estimate.
    pub total: f64,
    /// True when `total` is the provider-reported billed cost.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_from_provider: Option<bool>,
}
