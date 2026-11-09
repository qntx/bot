//! Catalog cost calculation and thinking-level support: upstream
//! `models.ts` `calculateCost`, `getSupportedThinkingLevels`,
//! `clampThinkingLevel`.

use serde_json::Value;

use crate::types::model::{Model, ModelThinkingLevel, ThinkingLevelMap, ThinkingLevelMapValue};
use crate::types::usage::{Usage, UsageCost};

const EXTENDED_THINKING_LEVELS: &[ModelThinkingLevel] = &[
    ModelThinkingLevel::Off,
    ModelThinkingLevel::Minimal,
    ModelThinkingLevel::Low,
    ModelThinkingLevel::Medium,
    ModelThinkingLevel::High,
    ModelThinkingLevel::Xhigh,
    ModelThinkingLevel::Max,
];

/// `calculateCost`: write the catalog cost of `usage` into
/// `usage.cost`, in upstream's exact operation order so results are
/// bit-identical f64 sums/products.
pub fn calculate_cost(model: &Model, usage: &mut Usage) -> UsageCost {
    let input_tokens = (usage.input + usage.cache_read + usage.cache_write) as f64;
    let mut rates = Rates {
        input: model.cost.input,
        output: model.cost.output,
        cache_read: model.cost.cache_read,
        cache_write: model.cost.cache_write,
    };
    let mut matched_threshold = -1f64;
    if let Some(tiers) = &model.cost.tiers {
        for tier in tiers {
            if input_tokens > tier.input_tokens_above && tier.input_tokens_above > matched_threshold
            {
                rates = Rates {
                    input: tier.input,
                    output: tier.output,
                    cache_read: tier.cache_read,
                    cache_write: tier.cache_write,
                };
                matched_threshold = tier.input_tokens_above;
            }
        }
    }
    // Anthropic charges 2x base input for 1h cache writes.
    let long_write = usage.cache_write1h.unwrap_or(0) as f64;
    let short_write = usage.cache_write as f64 - long_write;
    let input = (rates.input / 1_000_000.0) * usage.input as f64;
    let output = (rates.output / 1_000_000.0) * usage.output as f64;
    let cache_read = (rates.cache_read / 1_000_000.0) * usage.cache_read as f64;
    // Upstream's exact operation order is preserved bit-for-bit; fusing
    // into `mul_add` would change rounding.
    #[expect(
        clippy::suboptimal_flops,
        reason = "upstream operation order is preserved bit-for-bit"
    )]
    let cache_write =
        (rates.cache_write * short_write + rates.input * 2.0 * long_write) / 1_000_000.0;
    let total = input + output + cache_read + cache_write;
    usage.cost = UsageCost {
        input,
        output,
        cache_read,
        cache_write,
        total,
        total_from_provider: Some(false),
    };
    usage.cost
}

/// The four rate fields shared by `ModelCost` and `ModelCostTier`.
struct Rates {
    input: f64,
    output: f64,
    cache_read: f64,
    cache_write: f64,
}

/// `getSupportedThinkingLevels`: `"off"` for non-reasoning models, else
/// every level whose `thinkingLevelMap` entry is not `null`; `xhigh`/`max`
/// additionally require a mapped value.
#[must_use]
pub fn get_supported_thinking_levels(model: &Model) -> Vec<ModelThinkingLevel> {
    if !model.reasoning {
        return vec![ModelThinkingLevel::Off];
    }
    EXTENDED_THINKING_LEVELS
        .iter()
        .copied()
        .filter(|level| {
            let mapped = thinking_level_map_get(model.thinking_level_map.as_ref(), *level);
            if matches!(mapped, Some(ThinkingLevelMapValue::Unsupported)) {
                return false;
            }
            if matches!(level, ModelThinkingLevel::Xhigh | ModelThinkingLevel::Max) {
                return mapped.is_some();
            }
            true
        })
        .collect()
}

fn thinking_level_map_get(
    map: Option<&ThinkingLevelMap>,
    level: ModelThinkingLevel,
) -> Option<&ThinkingLevelMapValue> {
    let map = map?;
    match level {
        ModelThinkingLevel::Off => map.off.as_ref(),
        ModelThinkingLevel::Minimal => map.minimal.as_ref(),
        ModelThinkingLevel::Low => map.low.as_ref(),
        ModelThinkingLevel::Medium => map.medium.as_ref(),
        ModelThinkingLevel::High => map.high.as_ref(),
        ModelThinkingLevel::Xhigh => map.xhigh.as_ref(),
        ModelThinkingLevel::Max => map.max.as_ref(),
    }
}

/// `clampThinkingLevel`: the requested level when supported, else the
/// nearest higher supported level, else the nearest lower one, else
/// `"off"`.
#[must_use]
pub fn clamp_thinking_level(model: &Model, level: ModelThinkingLevel) -> ModelThinkingLevel {
    let available = get_supported_thinking_levels(model);
    if available.contains(&level) {
        return level;
    }
    let Some(requested_index) = EXTENDED_THINKING_LEVELS.iter().position(|l| l == &level) else {
        return available
            .first()
            .copied()
            .unwrap_or(ModelThinkingLevel::Off);
    };
    for candidate in EXTENDED_THINKING_LEVELS
        .get(requested_index..)
        .unwrap_or_default()
    {
        if available.contains(candidate) {
            return *candidate;
        }
    }
    for candidate in EXTENDED_THINKING_LEVELS
        .get(..requested_index)
        .unwrap_or_default()
        .iter()
        .rev()
    {
        if available.contains(candidate) {
            return *candidate;
        }
    }
    available
        .first()
        .copied()
        .unwrap_or(ModelThinkingLevel::Off)
}

/// `isOpenRouterModel`: the `openrouter` provider or an `openrouter.ai` base URL.
#[must_use]
pub fn is_openrouter_model(model: &Model) -> bool {
    model.provider.as_str() == "openrouter" || model.base_url.contains("openrouter.ai")
}

/// `applyOpenRouterReportedCost`: replace the catalog total with the cost
/// `OpenRouter` billed when `reported` (the streamed usage object) carries
/// complete, valid accounting.
pub fn apply_openrouter_reported_cost(usage: &mut Usage, reported: &Value) {
    let Some(cost) = valid_cost(reported.get("cost")) else {
        return;
    };
    let total = match reported.get("is_byok").and_then(Value::as_bool) {
        Some(false) => cost,
        Some(true) => {
            let Some(details) = reported.get("cost_details").filter(|d| !d.is_null()) else {
                return;
            };
            // Only an explicit `null` falls back to the per-phase sum; an
            // absent field is invalid like any other non-cost value.
            let upstream = match details.get("upstream_inference_cost") {
                Some(Value::Null) => {
                    let (Some(prompt), Some(completions)) = (
                        valid_cost(details.get("upstream_inference_prompt_cost")),
                        valid_cost(details.get("upstream_inference_completions_cost")),
                    ) else {
                        return;
                    };
                    prompt + completions
                }
                other => {
                    let Some(upstream) = valid_cost(other) else {
                        return;
                    };
                    upstream
                }
            };
            cost + upstream
        }
        None => return,
    };
    usage.cost.total = total;
    usage.cost.total_from_provider = Some(true);
}

/// `isValidCost`: a finite, non-negative number.
fn valid_cost(value: Option<&Value>) -> Option<f64> {
    value
        .and_then(Value::as_f64)
        .filter(|cost| cost.is_finite() && *cost >= 0.0)
}
