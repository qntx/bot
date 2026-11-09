//! Public wire types are constructible with struct literals — no
//! `serde_json::from_value` round-trip — and serialize to the pi JSON.

use ovo_ai::options::SimpleStreamOptions;
use ovo_ai::types::content::{TextContent, TextOrBlocksWithImages, UserContentBlock};
use ovo_ai::types::context::Context;
use ovo_ai::types::message::{AssistantMessage, Message, ToolResultMessage, UserMessage};
use ovo_ai::types::model::Model;
use ovo_ai::types::model::ThinkingLevel;
use ovo_ai::types::tool::Tool;
use serde_json::json;

#[test]
fn user_message_literal_serializes_to_pi_json() {
    let message = UserMessage {
        content: TextOrBlocksWithImages::Text(String::from("hi")),
        timestamp: 42,
    };
    assert_eq!(
        serde_json::to_value(&message).unwrap_or_default(),
        json!({ "content": "hi", "timestamp": 42 }),
    );
}

#[test]
fn tool_result_message_literal_serializes_to_pi_json() {
    let message = ToolResultMessage {
        tool_call_id: String::from("call-1"),
        tool_name: String::from("get_time"),
        content: vec![UserContentBlock::Text(TextContent {
            text: String::from("noon"),
            text_signature: None,
        })],
        details: None,
        usage: None,
        nested_calls: None,
        duration_ms: None,
        is_error: false,
        timestamp: 43,
    };
    assert_eq!(
        serde_json::to_value(&message).unwrap_or_default(),
        json!({
            "toolCallId": "call-1",
            "toolName": "get_time",
            "content": [{ "type": "text", "text": "noon" }],
            "isError": false,
            "timestamp": 43,
        }),
    );
}

#[test]
fn assistant_message_empty_serializes_to_pi_json() {
    let model: Model = serde_json::from_value(json!({
        "id": "faux-1", "name": "Faux", "api": "faux", "provider": "faux",
        "baseUrl": "", "reasoning": false, "input": ["text"],
        "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
        "contextWindow": 1, "maxTokens": 1
    }))
    .unwrap_or_else(|e| panic!("model: {e}"));
    let message = AssistantMessage::empty(&model, 7);
    assert_eq!(
        serde_json::to_value(&message).unwrap_or_default(),
        json!({
            "content": [],
            "api": "faux", "provider": "faux", "model": "faux-1",
            "usage": {
                "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0,
                "totalTokens": 0,
                "cost": {"input": 0.0, "output": 0.0, "cacheRead": 0.0,
                         "cacheWrite": 0.0, "total": 0.0},
            },
            "stopReason": "pending",
            "timestamp": 7,
        }),
    );
}

#[test]
fn context_and_tool_literals_serialize_to_pi_json() {
    let tool = Tool {
        name: String::from("get_time"),
        description: String::from("Current time"),
        parameters: json!({
            "type": "object",
            "properties": {},
        })
        .as_object()
        .cloned()
        .unwrap_or_default(),
        constrained_sampling: None,
    };
    let context = Context {
        system_prompt: Some(String::from("sys")),
        messages: vec![Message::User(UserMessage {
            content: TextOrBlocksWithImages::Text(String::from("hi")),
            timestamp: 1,
        })],
        tools: Some(vec![tool]),
    };
    assert_eq!(
        serde_json::to_value(&context).unwrap_or_default()["messages"][0]["role"],
        json!("user"),
    );
    assert_eq!(
        serde_json::to_value(&context).unwrap_or_default()["tools"][0]["name"],
        json!("get_time"),
    );
}

#[test]
fn simple_stream_options_default_constructs() {
    let options = SimpleStreamOptions {
        reasoning: Some(ThinkingLevel::High),
        ..SimpleStreamOptions::default()
    };
    assert_eq!(options.reasoning, Some(ThinkingLevel::High));
}
