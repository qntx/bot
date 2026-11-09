//! Live smoke tests against real provider APIs — every test is
//! `#[ignore]`d and skips with `eprintln!` when its key is missing.
//! Run: `cargo test -p ovo-ai --test e2e -- --ignored`.

#![allow(
    unused_crate_dependencies,
    reason = "test binaries only use part of the dependency set"
)]
#![allow(clippy::panic, reason = "integration tests fail by panicking")]
#![allow(
    clippy::tests_outside_test_module,
    reason = "live smoke tests are plain async tests"
)]
#![allow(
    clippy::print_stderr,
    reason = "skips report through stderr like upstream test runners"
)]

use std::sync::Arc;

use ovo_ai::auth::env::ProcessEnv;
use ovo_ai::http::client::{HttpClient, HttpClientConfig};
use ovo_ai::models::{CreateModelsOptions, Models};
use ovo_ai::options::SimpleStreamOptions;
use ovo_ai::providers::builtin_models;
use ovo_ai::types::content::{AssistantContentBlock, StopReason, TextOrBlocksWithImages, ToolCall};
use ovo_ai::types::context::Context;
use ovo_ai::types::message::{Message, ToolResultMessage, UserMessage};
use ovo_ai::types::model::{Model, ThinkingLevel};
use ovo_ai::types::tool::Tool;

/// One provider's live smoke config — key env, model env, default id.
struct Live {
    /// `builtinModels` provider id.
    provider: &'static str,
    /// Env var holding the API key.
    key_env: &'static str,
    /// Env var overriding the model id.
    model_env: &'static str,
    /// Reasoning-capable default — must resolve in `builtin_models`.
    default_model: &'static str,
}

/// The resolved model, or `None` when the key is missing — the test
/// prints and returns.
fn live_model(live: &Live, models: &Models) -> Option<Model> {
    if std::env::var(live.key_env).is_err() {
        eprintln!("skipping: {} is not set", live.key_env);
        return None;
    }
    let id = std::env::var(live.model_env).unwrap_or_else(|_| live.default_model.to_owned());
    let model = models.get_model(live.provider, &id);
    if model.is_none() {
        eprintln!("skipping: model {id} not in {} catalog", live.provider);
    }
    model
}

/// `builtinModels` over the process env — keys are never printed.
fn models() -> Arc<Models> {
    let http = Arc::new(HttpClient::new(HttpClientConfig::default()));
    builtin_models(
        CreateModelsOptions {
            env: Some(Arc::new(ProcessEnv)),
            ..CreateModelsOptions::default()
        },
        &http,
    )
    .unwrap_or_else(|e| panic!("builtin_models: {e}"))
}

fn user(text: &str) -> Message {
    Message::User(UserMessage {
        content: TextOrBlocksWithImages::Text(text.to_owned()),
        timestamp: 0,
    })
}

fn get_time_tool() -> Tool {
    Tool {
        name: String::from("get_time"),
        description: String::from("Return the current time."),
        parameters: serde_json::Map::from_iter([(
            String::from("type"),
            serde_json::Value::String(String::from("object")),
        )]),
        constrained_sampling: None,
    }
}

/// The first `ToolCall` block in a message's content.
fn first_tool_call(message: &ovo_ai::types::message::AssistantMessage) -> Option<&ToolCall> {
    message.content.iter().find_map(|b| match b {
        AssistantContentBlock::ToolCall(c) => Some(c),
        _ => None,
    })
}

/// Concatenated text of a message's `Text` blocks.
fn text_of(message: &ovo_ai::types::message::AssistantMessage) -> String {
    message
        .content
        .iter()
        .filter_map(|b| match b {
            AssistantContentBlock::Text(t) => Some(t.text.as_str()),
            _ => None,
        })
        .collect()
}

/// Per-provider suites.
macro_rules! suite {
    ($mod_name:ident, $provider:literal, $key_env:literal, $model_env:literal, $default:literal) => {
        mod $mod_name {
            use super::*;

            const LIVE: Live = Live {
                provider: $provider,
                key_env: $key_env,
                model_env: $model_env,
                default_model: $default,
            };

            /// Plain-text completion: `stopReason` `stop` + non-empty text.
            #[tokio::test]
            #[ignore = "live provider smoke test"]
            async fn text() {
                let models = models();
                let Some(model) = live_model(&LIVE, &models) else {
                    return;
                };
                let message = models
                    .complete_simple(
                        &model,
                        &Context {
                            system_prompt: None,
                            messages: vec![user("Say hello in one word.")],
                            tools: None,
                        },
                        None,
                    )
                    .await;
                assert_eq!(message.stop_reason, StopReason::Stop, "{message:?}");
                assert!(!text_of(&message).is_empty(), "{message:?}");
            }

            /// A declared `get_time` tool yields `toolUse` + a parsed
            /// `ToolCall`; its result produces a final text answer.
            #[tokio::test]
            #[ignore = "live provider smoke test"]
            async fn tool_call() {
                let models = models();
                let Some(model) = live_model(&LIVE, &models) else {
                    return;
                };
                let first = models
                    .complete_simple(
                        &model,
                        &Context {
                            system_prompt: None,
                            messages: vec![user("What time is it? Use the get_time tool.")],
                            tools: Some(vec![get_time_tool()]),
                        },
                        None,
                    )
                    .await;
                assert_eq!(first.stop_reason, StopReason::ToolUse, "{first:?}");
                let call = first_tool_call(&first)
                    .cloned()
                    .unwrap_or_else(|| panic!("expected a ToolCall block: {first:?}"));
                let second = models
                    .complete_simple(
                        &model,
                        &Context {
                            system_prompt: None,
                            messages: vec![
                                user("What time is it? Use the get_time tool."),
                                Message::Assistant(Box::new(first)),
                                Message::ToolResult(ToolResultMessage {
                                    tool_call_id: call.id,
                                    tool_name: call.name,
                                    content: vec![ovo_ai::types::content::UserContentBlock::Text(
                                        ovo_ai::types::content::TextContent {
                                            text: String::from("12:00 UTC"),
                                            text_signature: None,
                                        },
                                    )],
                                    details: None,
                                    usage: None,
                                    nested_calls: None,
                                    is_error: false,
                                    timestamp: 0,
                                    duration_ms: None,
                                }),
                            ],
                            tools: Some(vec![get_time_tool()]),
                        },
                        None,
                    )
                    .await;
                assert_eq!(second.stop_reason, StopReason::Stop, "{second:?}");
                assert!(!text_of(&second).is_empty(), "{second:?}");
            }

            /// `reasoning` yields at least one thinking block.
            #[tokio::test]
            #[ignore = "live provider smoke test"]
            async fn thinking() {
                let models = models();
                let Some(model) = live_model(&LIVE, &models) else {
                    return;
                };
                let message = models
                    .complete_simple(
                        &model,
                        &Context {
                            system_prompt: None,
                            messages: vec![user("What is 17 * 23?")],
                            tools: None,
                        },
                        Some(SimpleStreamOptions {
                            reasoning: Some(ThinkingLevel::Low),
                            ..SimpleStreamOptions::default()
                        }),
                    )
                    .await;
                assert!(
                    message
                        .content
                        .iter()
                        .any(|b| matches!(b, AssistantContentBlock::Thinking(_))),
                    "expected a thinking block: {message:?}"
                );
            }
        }
    };
}

suite!(
    anthropic_messages,
    "anthropic",
    "ANTHROPIC_API_KEY",
    "OVO_E2E_ANTHROPIC_MODEL",
    "claude-sonnet-4-5"
);
suite!(
    openai_responses,
    "openai",
    "OPENAI_API_KEY",
    "OVO_E2E_OPENAI_MODEL",
    "gpt-5.1"
);
suite!(
    openai_completions,
    "openrouter",
    "OPENROUTER_API_KEY",
    "OVO_E2E_OPENROUTER_MODEL",
    "anthropic/claude-sonnet-4.5"
);

/// The default model ids must resolve in `builtin_models` — this runs
/// unconditionally (no key needed).
#[tokio::test]
async fn default_models_resolve_in_catalog() {
    let models = models();
    for (provider, id) in [
        ("anthropic", "claude-sonnet-4-5"),
        ("openai", "gpt-5.1"),
        ("openrouter", "anthropic/claude-sonnet-4.5"),
    ] {
        assert!(
            models.get_model(provider, id).is_some(),
            "{provider}:{id} must resolve in builtin_models"
        );
    }
}
