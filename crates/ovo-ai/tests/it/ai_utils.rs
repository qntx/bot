//! Replays the M1-05B vectors: `AI-ESTIMATE`, `AI-OVERFLOW`,
//! `AI-RETRY-POLICY`, `AI-COST`, `AI-SIMPLE-OPTIONS`.

use serde_json::{Map, Value};

use ovo_ai::cost;
use ovo_ai::estimate;
use ovo_ai::overflow;
use ovo_ai::retry;
use ovo_ai::simple_options;
use ovo_ai::transcript;
use ovo_ai::types::content::TextOrBlocksWithImages;
use ovo_ai::types::context::Context;
use ovo_ai::types::message::{AssistantMessage, Message};
use ovo_ai::types::model::{Model, ModelThinkingLevel, ThinkingLevel};
use ovo_ai::types::usage::Usage;

use crate::support::js_value_eq;

fn cases(path: &str) -> Vec<Value> {
    let text = std::fs::read_to_string(format!("{}/../../{}", env!("CARGO_MANIFEST_DIR"), path))
        .unwrap_or_else(|error| panic!("{path}: {error}"));
    let file: Value = match ovo_ai::json::from_str(&text) {
        Ok(value) => value,
        Err(error) => panic!("{path}: invalid vector file: {error}"),
    };
    file.get("cases")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default()
}

fn decode<T: serde::de::DeserializeOwned>(name: &str, value: &Value) -> T {
    match serde_json::from_value(value.clone()) {
        Ok(v) => v,
        Err(error) => panic!("{name}: decode: {error}"),
    }
}

fn json_of<T: ?Sized + serde::Serialize>(value: &T) -> Value {
    serde_json::to_value(value).unwrap_or_default()
}

/// parity: AI-ESTIMATE
#[test]
fn estimate_cases_replay() {
    for case in cases("vectors/ai/estimate.json") {
        let name = case["name"].as_str().unwrap_or("?");
        let kind = case["kind"].as_str().unwrap_or("?");
        let args = &case["args"];
        let actual = match kind {
            "calculateContextTokens" => {
                let usage: Usage = decode(name, args);
                Value::from(estimate::calculate_context_tokens(&usage))
            }
            "estimateTextTokens" => Value::from(estimate::estimate_text_tokens(
                args.as_str().unwrap_or_default(),
            )),
            "estimateTextAndImageContentTokens" => {
                let content: TextOrBlocksWithImages = decode(name, args);
                Value::from(estimate::estimate_text_and_image_content_tokens(&content))
            }
            "estimateMessageTokens" => {
                let msg: Message = decode(name, args);
                Value::from(estimate::estimate_message_tokens(&msg))
            }
            "estimateContextTokens" => {
                let context: Context = decode(name, args);
                let folded = transcript::normalize_context(&context);
                let est = estimate::estimate_context_tokens_for_context(&folded);
                json_of(&serde_json::json!({
                    "tokens": est.tokens,
                    "usageTokens": est.usage_tokens,
                    "trailingTokens": est.trailing_tokens,
                    "lastUsageIndex": est.last_usage_index,
                }))
            }
            other => panic!("unknown kind {other}"),
        };
        assert!(
            js_value_eq(&actual, &case["result"]),
            "{name}: differs\n{actual}\n{}",
            case["result"]
        );
    }
}

/// parity: AI-OVERFLOW
#[test]
fn overflow_cases_replay() {
    for case in cases("vectors/ai/overflow.json") {
        let name = case["name"].as_str().unwrap_or("?");
        let kind = case["kind"].as_str().unwrap_or("?");
        let args = &case["args"];
        let actual = match kind {
            "isContextOverflow" => {
                let msg: AssistantMessage = decode(name, &args[0]);
                let cw = args[1].as_u64();
                Value::from(overflow::is_context_overflow(&msg, cw))
            }
            "isRecoverableLength" => {
                let msg: AssistantMessage = decode(name, &args[0]);
                let desired = args[1].as_u64().unwrap_or_default();
                Value::from(overflow::is_recoverable_length(&msg, desired))
            }
            other => panic!("unknown kind {other}"),
        };
        assert!(
            js_value_eq(&actual, &case["result"]),
            "{name}: differs\n{actual}\n{}",
            case["result"]
        );
    }
}

/// parity: AI-RETRY-POLICY
#[test]
fn retry_cases_replay() {
    for case in cases("vectors/ai/retry.json") {
        let name = case["name"].as_str().unwrap_or("?");
        let kind = case["kind"].as_str().unwrap_or("?");
        let args = &case["args"];
        let actual = match kind {
            "isRetryableAssistantError" => {
                let msg: AssistantMessage = decode(name, args);
                Value::from(retry::is_retryable_assistant_error(&msg))
            }
            "retryDelayMs" => {
                let policy: retry::RetryPolicy = decode(name, &args[0]);
                let attempt = args[1].as_u64().unwrap_or_default() as u32;
                Value::from(retry::retry_delay_ms(&policy, attempt))
            }
            other => panic!("unknown kind {other}"),
        };
        assert!(
            js_value_eq(&actual, &case["result"]),
            "{name}: differs\n{actual}\n{}",
            case["result"]
        );
    }
}

/// parity: AI-COST
#[test]
fn cost_cases_replay() {
    for case in cases("vectors/ai/cost.json") {
        let name = case["name"].as_str().unwrap_or("?");
        let kind = case["kind"].as_str().unwrap_or("?");
        let args = &case["args"];
        let actual = match kind {
            "calculateCost" => {
                let model: Model = decode(name, &args[0]);
                let mut usage: Usage = decode(name, &args[1]);
                json_of(&cost::calculate_cost(&model, &mut usage))
            }
            "getSupportedThinkingLevels" => {
                let model: Model = decode(name, args);
                json_of(&cost::get_supported_thinking_levels(&model))
            }
            "clampThinkingLevel" => {
                let model: Model = decode(name, &args[0]);
                let level: ModelThinkingLevel = decode(name, &args[1]);
                json_of(&cost::clamp_thinking_level(&model, level))
            }
            other => panic!("unknown kind {other}"),
        };
        assert!(
            js_value_eq(&actual, &case["result"]),
            "{name}: differs\n{actual}\n{}",
            case["result"]
        );
    }
}

/// parity: AI-SIMPLE-OPTIONS
#[test]
fn simple_options_cases_replay() {
    for case in cases("vectors/ai/simple-options.json") {
        let name = case["name"].as_str().unwrap_or("?");
        let kind = case["kind"].as_str().unwrap_or("?");
        let args = &case["args"];
        let actual = match kind {
            "clampMaxTokensToContext" => {
                let model: Model = decode(name, &args[0]);
                let context: Context = decode(name, &args[1]);
                let folded = transcript::normalize_context(&context);
                let max_tokens = args[2].as_u64().unwrap_or_default();
                Value::from(simple_options::clamp_max_tokens_to_context(
                    &model, &folded, max_tokens,
                ))
            }
            "resolveSamplingParams" => {
                let model: Model = decode(name, &args[0]);
                let level: ModelThinkingLevel = decode(name, &args[1]);
                let req: Option<Map<String, Value>> = args
                    .get(2)
                    .and_then(|v| serde_json::from_value(v.clone()).ok());
                simple_options::resolve_sampling_params(&model, level, req.as_ref())
                    .map_or_else(|| Value::Null, |params| json_of(&params))
            }
            "clampReasoning" => {
                let effort: Option<ThinkingLevel> = if args.is_null() {
                    None
                } else {
                    Some(decode(name, args))
                };
                simple_options::clamp_reasoning(effort)
                    .map_or_else(|| Value::Null, |level| json_of(&level))
            }
            "thinkingBudgetForLevel" => {
                let level: ThinkingLevel = decode(name, &args[0]);
                let custom: Option<simple_options::ThinkingBudgets> = args
                    .get(1)
                    .and_then(|v| serde_json::from_value(v.clone()).ok());
                Value::from(simple_options::thinking_budget_for_level(
                    level,
                    custom.as_ref(),
                ))
            }
            "clampThinkingBudgetToAnswerRoom" => {
                let budget = args[0].as_u64().unwrap_or_default();
                let ceiling = args[1].as_u64().unwrap_or_default();
                Value::from(simple_options::clamp_thinking_budget_to_answer_room(
                    budget, ceiling,
                ))
            }
            "adjustMaxTokensForThinking" => {
                let base = args[0].as_u64();
                let model_max = args[1].as_u64().unwrap_or_default();
                let level: ThinkingLevel = decode(name, &args[2]);
                let custom: Option<simple_options::ThinkingBudgets> = args
                    .get(3)
                    .and_then(|v| serde_json::from_value(v.clone()).ok());
                let adjusted = simple_options::adjust_max_tokens_for_thinking(
                    base,
                    model_max,
                    level,
                    custom.as_ref(),
                );
                json_of(&serde_json::json!({
                    "maxTokens": adjusted.max_tokens,
                    "thinkingBudget": adjusted.thinking_budget,
                }))
            }
            other => panic!("unknown kind {other}"),
        };
        assert!(
            js_value_eq(&actual, &case["result"]),
            "{name}: differs\n{actual}\n{}",
            case["result"]
        );
    }
}
