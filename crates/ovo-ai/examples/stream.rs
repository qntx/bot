//! Stream a prompt through a built-in provider and run one tool round.
//!
//! ```sh
//! cargo run -p ovo-ai --example stream -- anthropic claude-fable-5 "what time is it?"
//! ```

#![allow(
    unused_crate_dependencies,
    reason = "an example only needs ovo-ai, futures and tokio"
)]
#![allow(
    clippy::print_stdout,
    clippy::print_stderr,
    reason = "a CLI example reports through stdout/stderr"
)]

use std::process::ExitCode;
use std::sync::Arc;

use futures::StreamExt;
use ovo_ai::auth::ProcessEnv;
use ovo_ai::http::client::{HttpClient, HttpClientConfig};
use ovo_ai::models::CreateModelsOptions;
use ovo_ai::providers::builtin_models;
use ovo_ai::stream::{AssistantMessageEvent, AssistantStream};
use ovo_ai::types::content::{
    AssistantContentBlock, StopReason, TextContent, TextOrBlocksWithImages, UserContentBlock,
};
use ovo_ai::types::context::Context;
use ovo_ai::types::message::{AssistantMessage, Message, ToolResultMessage, UserMessage};
use ovo_ai::types::model::Model;
use ovo_ai::types::tool::Tool;
use serde_json::json;

fn get_time_tool() -> Tool {
    Tool {
        name: String::from("get_time"),
        description: String::from("Return the current wall-clock time in a timezone."),
        parameters: json!({
            "type": "object",
            "properties": {
                "timezone": {
                    "type": "string",
                    "description": "IANA timezone name, e.g. UTC or Europe/Berlin"
                }
            },
            "required": ["timezone"]
        })
        .as_object()
        .cloned()
        .unwrap_or_default(),
        constrained_sampling: None,
    }
}

/// `get_time` — a fixed answer keeps the example deterministic.
fn run_tool(arguments: &serde_json::Map<String, serde_json::Value>) -> String {
    let timezone = arguments
        .get("timezone")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("UTC");
    format!("12:00:00 {timezone}")
}

/// Print events as they arrive; returns the final message.
async fn pump(mut stream: AssistantStream) -> AssistantMessage {
    while let Some(event) = stream.next().await {
        match event {
            AssistantMessageEvent::TextDelta { delta, .. } => print!("{delta}"),
            AssistantMessageEvent::ThinkingDelta { delta, .. } => eprint!("{delta}"),
            AssistantMessageEvent::ToolcallEnd { tool_call, .. } => {
                println!(
                    "\n[tool call] {} {}",
                    tool_call.name,
                    json!(tool_call.arguments)
                );
            }
            AssistantMessageEvent::Error { error_message, .. } => {
                println!("\n[error] {error_message}");
            }
            _ => {}
        }
    }
    println!();
    stream.result().await
}

async fn run(provider: &str, model_id: &str, prompt: &str) -> Result<(), String> {
    let http = Arc::new(HttpClient::new(HttpClientConfig::default()));
    let models = builtin_models(
        CreateModelsOptions {
            env: Some(Arc::new(ProcessEnv)),
            ..CreateModelsOptions::default()
        },
        &http,
    )
    .map_err(|error| error.to_string())?;
    let model: Model = models
        .get_model(provider, model_id)
        .ok_or_else(|| format!("{provider}/{model_id} is not in the catalog"))?;
    let mut context = Context {
        messages: vec![Message::User(UserMessage {
            content: TextOrBlocksWithImages::Text(String::from(prompt)),
            timestamp: 0,
        })],
        tools: Some(vec![get_time_tool()]),
        ..Context::default()
    };
    let reply = pump(models.stream(&model, &context, None)).await;
    match reply.stop_reason {
        StopReason::Error | StopReason::Aborted => {
            let reason = reply.error_message.unwrap_or_default();
            Err(format!("stream failed: {reason}"))
        }
        StopReason::ToolUse => {
            let calls: Vec<_> = reply
                .content
                .iter()
                .filter_map(|block| match block {
                    AssistantContentBlock::ToolCall(call) => Some(call.clone()),
                    _ => None,
                })
                .collect();
            context.messages.push(Message::Assistant(Box::new(reply)));
            for call in &calls {
                let text = run_tool(&call.arguments);
                println!("[tool result] {text}");
                context
                    .messages
                    .push(Message::ToolResult(ToolResultMessage {
                        tool_call_id: call.id.clone(),
                        tool_name: call.name.clone(),
                        content: vec![UserContentBlock::Text(TextContent {
                            text,
                            text_signature: None,
                        })],
                        details: None,
                        usage: None,
                        nested_calls: None,
                        duration_ms: None,
                        is_error: false,
                        timestamp: 0,
                    }));
            }
            let follow_up = pump(models.stream(&model, &context, None)).await;
            if matches!(
                follow_up.stop_reason,
                StopReason::Error | StopReason::Aborted
            ) {
                let reason = follow_up.error_message.unwrap_or_default();
                return Err(format!("follow-up stream failed: {reason}"));
            }
            Ok(())
        }
        _ => Ok(()),
    }
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> ExitCode {
    let mut args = std::env::args().skip(1);
    let (Some(provider), Some(model)) = (args.next(), args.next()) else {
        eprintln!("usage: stream <provider> <model> [prompt]");
        return ExitCode::FAILURE;
    };
    let prompt = args
        .next()
        .unwrap_or_else(|| String::from("What time is it in UTC?"));
    match run(&provider, &model, &prompt).await {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{error}");
            ExitCode::FAILURE
        }
    }
}
