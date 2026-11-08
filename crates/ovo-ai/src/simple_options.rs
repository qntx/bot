//! Shared simple-option resolution: upstream `api/simple-options.ts`.
//!
//! Ports `clampMaxTokensToContext`, `resolveSamplingParams`,
//! `clampReasoning`, `thinkingBudgetForLevel`,
//! `clampThinkingBudgetToAnswerRoom`, and `adjustMaxTokensForThinking`.
//!
//! `buildBaseOptions` is not ported: it assembles `StreamOptions` (signal,
//! fetch, telemetry, headers, callbacks, env), which does not exist until
//! the HTTP layer (M1-08).

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::cost::clamp_thinking_level;
use crate::estimate::estimate_context_tokens;
use crate::transcript::TranscriptContext;
use crate::types::model::{Model, ModelThinkingLevel, ThinkingLevel};

/// `CONTEXT_SAFETY_TOKENS`: tokens reserved ahead of the model window when
/// clamping `maxTokens` to the estimated context size.
pub const CONTEXT_SAFETY_TOKENS: u64 = 4096;

/// `MIN_MAX_TOKENS`: the floor every clamped `maxTokens` keeps.
pub const MIN_MAX_TOKENS: u64 = 1;

/// `MIN_ANSWER_TOKENS`: tokens always left for the answer when a thinking
/// budget shares the response ceiling.
pub const MIN_ANSWER_TOKENS: u64 = 1024;

/// `SamplingParams`: arbitrary per-request sampling keys (upstream
/// `Record<string, unknown>`).
pub type SamplingParams = Map<String, Value>;

/// `ThinkingBudgets`: token budgets per thinking level; any absent field
/// falls back to [`DEFAULT_THINKING_BUDGETS`].
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ThinkingBudgets {
    /// Budget for the `minimal` level.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub minimal: Option<u64>,
    /// Budget for the `low` level.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub low: Option<u64>,
    /// Budget for the `medium` level.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub medium: Option<u64>,
    /// Budget for the `high` level.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub high: Option<u64>,
}

/// `DEFAULT_THINKING_BUDGETS`.
pub const DEFAULT_THINKING_BUDGETS: ThinkingBudgets = ThinkingBudgets {
    minimal: Some(1024),
    low: Some(2048),
    medium: Some(8192),
    high: Some(16384),
};

/// `clampMaxTokensToContext`: clamp `maxTokens` so the estimated context
/// plus the reserved margin fits the model window.
#[must_use]
pub fn clamp_max_tokens_to_context(
    model: &Model,
    context: &TranscriptContext,
    max_tokens: u64,
) -> u64 {
    if model.context_window == 0 {
        return max_tokens.max(MIN_MAX_TOKENS);
    }
    let used = estimate_context_tokens(context.messages()).tokens;
    let available =
        i128::from(model.context_window) - i128::from(used) - i128::from(CONTEXT_SAFETY_TOKENS);
    let floor = i128::from(MIN_MAX_TOKENS);
    let clamped = available.max(floor).min(i128::from(u64::MAX));
    max_tokens.min(u64::try_from(clamped).unwrap_or(MIN_MAX_TOKENS))
}

/// `resolveSamplingParams`: merge order model → thinking level → request.
///
/// `None` only when all three are absent (an empty map still counts, like a
/// truthy `{}` upstream).
#[must_use]
pub fn resolve_sampling_params(
    model: &Model,
    thinking_level: ModelThinkingLevel,
    request_params: Option<&SamplingParams>,
) -> Option<SamplingParams> {
    let effective_level = clamp_thinking_level(model, thinking_level);
    let level_params = model
        .sampling_params_by_thinking_level
        .as_ref()
        .and_then(|levels| levels.get(thinking_level_name(effective_level)))
        .and_then(Value::as_object);
    if model.sampling_params.is_none() && level_params.is_none() && request_params.is_none() {
        return None;
    }
    let mut merged = Map::new();
    for params in [model.sampling_params.as_ref(), level_params, request_params]
        .into_iter()
        .flatten()
    {
        merged.extend(
            params
                .iter()
                .map(|(key, value)| (key.clone(), value.clone())),
        );
    }
    Some(merged)
}

const fn thinking_level_name(level: ModelThinkingLevel) -> &'static str {
    match level {
        ModelThinkingLevel::Off => "off",
        ModelThinkingLevel::Minimal => "minimal",
        ModelThinkingLevel::Low => "low",
        ModelThinkingLevel::Medium => "medium",
        ModelThinkingLevel::High => "high",
        ModelThinkingLevel::Xhigh => "xhigh",
        ModelThinkingLevel::Max => "max",
    }
}

/// `clampReasoning`: `xhigh`/`max` map to `high`.
///
/// The result is never `xhigh`/`max` (upstream's
/// `Exclude<ThinkingLevel, "xhigh" | "max">`).
#[must_use]
pub const fn clamp_reasoning(effort: Option<ThinkingLevel>) -> Option<ThinkingLevel> {
    match effort {
        Some(ThinkingLevel::Xhigh | ThinkingLevel::Max) => Some(ThinkingLevel::High),
        other => other,
    }
}

/// `thinkingBudgetForLevel`: the budget of a level after
/// [`clamp_reasoning`], with `custom` fields overriding
/// [`DEFAULT_THINKING_BUDGETS`] per key.
#[must_use]
pub fn thinking_budget_for_level(
    reasoning_level: ThinkingLevel,
    custom_budgets: Option<&ThinkingBudgets>,
) -> u64 {
    let budget = |pick: fn(&ThinkingBudgets) -> Option<u64>| {
        custom_budgets
            .and_then(pick)
            .or_else(|| pick(&DEFAULT_THINKING_BUDGETS))
            .unwrap_or(0)
    };
    match clamp_reasoning(Some(reasoning_level)).unwrap_or(ThinkingLevel::High) {
        ThinkingLevel::Minimal => budget(|b| b.minimal),
        ThinkingLevel::Low => budget(|b| b.low),
        ThinkingLevel::Medium => budget(|b| b.medium),
        ThinkingLevel::High | ThinkingLevel::Xhigh | ThinkingLevel::Max => budget(|b| b.high),
    }
}

/// `clampThinkingBudgetToAnswerRoom`: cap a thinking budget so at least
/// [`MIN_ANSWER_TOKENS`] remain under a shared response ceiling.
#[must_use]
pub fn clamp_thinking_budget_to_answer_room(thinking_budget: u64, ceiling: u64) -> u64 {
    thinking_budget.min(ceiling.saturating_sub(MIN_ANSWER_TOKENS))
}

/// `adjustMaxTokensForThinking`: the `maxTokens`/`thinkingBudget` pair.
///
/// `None` for `base_max_tokens` means no explicit caller cap — the model
/// cap is used and thinking fits inside it.
#[must_use]
pub fn adjust_max_tokens_for_thinking(
    base_max_tokens: Option<u64>,
    model_max_tokens: u64,
    reasoning_level: ThinkingLevel,
    custom_budgets: Option<&ThinkingBudgets>,
) -> AdjustedMaxTokens {
    let mut thinking_budget = thinking_budget_for_level(reasoning_level, custom_budgets);
    let max_tokens = base_max_tokens.map_or(model_max_tokens, |base| {
        base.saturating_add(thinking_budget).min(model_max_tokens)
    });
    if max_tokens <= thinking_budget {
        thinking_budget = clamp_thinking_budget_to_answer_room(thinking_budget, max_tokens);
    }
    AdjustedMaxTokens {
        max_tokens,
        thinking_budget,
    }
}

/// `adjustMaxTokensForThinking` result.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AdjustedMaxTokens {
    /// The effective `maxTokens` of the request.
    pub max_tokens: u64,
    /// The effective thinking budget in tokens.
    pub thinking_budget: u64,
}
