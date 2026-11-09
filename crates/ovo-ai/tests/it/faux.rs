//! `faux-provider.test.ts` — the scripted provider's queue, usage
//! estimation, cache simulation, chunked streaming, pacing and aborts.

/// parity: AI-FAUX
use std::sync::Arc;

use futures::StreamExt;
use ovo_ai::faux::{
    FauxMessageOptions, FauxModelDefinition, FauxOptions, FauxProvider, FauxResponseStep,
    FauxTokenSize, faux_assistant_message, faux_assistant_message_with, faux_provider, faux_text,
    faux_thinking, faux_tool_call,
};
use ovo_ai::models::{CreateModelsOptions, Models, create_models};
use ovo_ai::options::{CacheRetention, StreamOptions};
use ovo_ai::stream::{AssistantMessageEvent, AssistantStream};
use ovo_ai::types::content::{AssistantContentBlock, StopReason};
use ovo_ai::types::context::Context;
use ovo_ai::types::message::AssistantMessage;
use serde_json::{Map, Value, json};
use tokio_util::sync::CancellationToken;

fn seeded() -> (Arc<Models>, FauxProvider) {
    seeded_with(FauxOptions::default())
}

fn seeded_with(options: FauxOptions) -> (Arc<Models>, FauxProvider) {
    let models = create_models(CreateModelsOptions::default());
    let faux = faux_provider(options).unwrap_or_else(|e| panic!("faux provider: {e}"));
    models.set_provider(Arc::clone(&faux.provider));
    (models, faux)
}

fn context(value: Value) -> Context {
    serde_json::from_value(value).unwrap_or_else(|error| panic!("context: {error}"))
}

fn user_context(text: &str) -> Context {
    context(json!({
        "messages": [{"role": "user", "content": text, "timestamp": 0}],
    }))
}

fn message_step(content: Vec<AssistantContentBlock>) -> FauxResponseStep {
    FauxResponseStep::Message(Box::new(faux_assistant_message(content)))
}

async fn collect_events(mut stream: AssistantStream) -> Vec<AssistantMessageEvent> {
    let mut events = Vec::new();
    while let Some(event) = stream.next().await {
        events.push(event);
    }
    events
}

const fn event_type(event: &AssistantMessageEvent) -> &'static str {
    match event {
        AssistantMessageEvent::Start => "start",
        AssistantMessageEvent::TextStart { .. } => "text_start",
        AssistantMessageEvent::TextDelta { .. } => "text_delta",
        AssistantMessageEvent::TextEnd { .. } => "text_end",
        AssistantMessageEvent::ThinkingStart { .. } => "thinking_start",
        AssistantMessageEvent::ThinkingDelta { .. } => "thinking_delta",
        AssistantMessageEvent::ThinkingEnd { .. } => "thinking_end",
        AssistantMessageEvent::ToolcallStart { .. } => "toolcall_start",
        AssistantMessageEvent::ToolcallDelta { .. } => "toolcall_delta",
        AssistantMessageEvent::ToolcallEnd { .. } => "toolcall_end",
        AssistantMessageEvent::Done { .. } => "done",
        AssistantMessageEvent::Error { .. } => "error",
    }
}

fn call_count(faux: &FauxProvider) -> usize {
    faux.state
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .call_count
}

fn args(value: Value) -> Map<String, Value> {
    match value {
        Value::Object(map) => map,
        other => panic!("arguments must be an object: {other}"),
    }
}

fn text_of(message: &AssistantMessage) -> &str {
    message
        .content
        .iter()
        .find_map(|block| match block {
            AssistantContentBlock::Text(text) => Some(text.text.as_str()),
            _ => None,
        })
        .unwrap_or_else(|| panic!("a text block"))
}

#[tokio::test]
async fn registers_a_custom_provider_and_estimates_usage() {
    let (models, faux) = seeded();
    faux.set_responses(vec![message_step(vec![faux_text("hello world")])]);
    let ctx = context(json!({
        "systemPrompt": "Be concise.",
        "messages": [{"role": "user", "content": "hi there", "timestamp": 0}],
    }));
    let response = models.complete(faux.get_model(), &ctx, None).await;
    assert_eq!(
        response.content,
        vec![faux_text("hello world")],
        "{response:?}"
    );
    assert!(response.usage.input > 0, "{:?}", response.usage);
    assert!(response.usage.output > 0, "{:?}", response.usage);
    assert_eq!(
        response.usage.total_tokens,
        response.usage.input + response.usage.output
    );
    assert_eq!(call_count(&faux), 1);
}

#[tokio::test]
async fn supports_helper_blocks_for_text_thinking_and_tool_calls() {
    let (models, faux) = seeded();
    faux.set_responses(vec![FauxResponseStep::Message(Box::new(
        faux_assistant_message_with(
            vec![
                faux_thinking("think"),
                faux_tool_call("echo", args(json!({"text": "hi"})), None),
                faux_text("done"),
            ],
            &FauxMessageOptions {
                stop_reason: Some(StopReason::ToolUse),
                ..FauxMessageOptions::default()
            },
        ),
    ))]);
    let response = models
        .complete(faux.get_model(), &user_context("hi"), None)
        .await;
    assert_eq!(response.stop_reason, StopReason::ToolUse);
    assert_eq!(response.content.len(), 3, "{:?}", response.content);
    let [thinking, call, text] = &response.content[..] else {
        panic!("three blocks: {:?}", response.content);
    };
    assert_eq!(thinking, &faux_thinking("think"), "thinking block");
    match call {
        AssistantContentBlock::ToolCall(call) => {
            assert!(!call.id.is_empty(), "tool call id");
            assert_eq!(call.name, "echo");
            assert_eq!(call.arguments, args(json!({"text": "hi"})));
        }
        other => panic!("tool call block: {other:?}"),
    }
    assert_eq!(text, &faux_text("done"), "text block");
}

#[tokio::test]
async fn supports_multiple_models_with_per_model_reasoning_and_model_aware_factories() {
    let (models, faux) = seeded_with(FauxOptions {
        models: vec![
            FauxModelDefinition {
                id: String::from("faux-fast"),
                name: Some(String::from("Faux Fast")),
                reasoning: false,
                context_window: 128_000,
                max_tokens: 16_384,
            },
            FauxModelDefinition {
                id: String::from("faux-thinker"),
                name: Some(String::from("Faux Thinker")),
                reasoning: true,
                context_window: 128_000,
                max_tokens: 16_384,
            },
        ],
        ..FauxOptions::default()
    });
    let factory = || {
        FauxResponseStep::Factory(Box::new(|_context, _options, _state, model| {
            let text = format!("{}:{}", model.id, model.reasoning);
            Box::pin(async move { Ok(faux_assistant_message(vec![faux_text(&text)])) })
        }))
    };
    faux.set_responses(vec![factory(), factory()]);
    assert_eq!(
        faux.models
            .iter()
            .map(|model| model.id.as_str())
            .collect::<Vec<_>>(),
        ["faux-fast", "faux-thinker"]
    );
    assert_eq!(faux.get_model().id, faux.models[0].id);
    assert!(
        !faux
            .model_by_id("faux-fast")
            .unwrap_or_else(|| panic!("faux-fast model"))
            .reasoning
    );
    assert!(
        faux.model_by_id("faux-thinker")
            .unwrap_or_else(|| panic!("faux-thinker model"))
            .reasoning
    );
    let fast = models
        .complete(
            faux.model_by_id("faux-fast")
                .unwrap_or_else(|| panic!("faux-fast model")),
            &user_context("hi"),
            None,
        )
        .await;
    let thinker = models
        .complete(
            faux.model_by_id("faux-thinker")
                .unwrap_or_else(|| panic!("faux-thinker model")),
            &user_context("hi"),
            None,
        )
        .await;
    assert_eq!(text_of(&fast), "faux-fast:false");
    assert_eq!(text_of(&thinker), "faux-thinker:true");
}

#[tokio::test]
async fn rewrites_api_provider_and_model_on_returned_messages() {
    let (models, faux) = seeded_with(FauxOptions {
        api: Some(String::from("faux:test")),
        provider: Some(String::from("faux-provider")),
        models: vec![FauxModelDefinition {
            id: String::from("faux-model"),
            name: None,
            reasoning: false,
            context_window: 128_000,
            max_tokens: 16_384,
        }],
        ..FauxOptions::default()
    });
    faux.set_responses(vec![message_step(vec![faux_text("hello")])]);
    let response = models
        .complete(faux.get_model(), &user_context("hi"), None)
        .await;
    assert_eq!(response.api.as_str(), "faux:test");
    assert_eq!(response.provider.as_str(), "faux-provider");
    assert_eq!(response.model, "faux-model");
}

#[tokio::test]
async fn consumes_queued_responses_in_order_and_errors_when_exhausted() {
    let (models, faux) = seeded();
    faux.set_responses(vec![
        message_step(vec![faux_text("first")]),
        message_step(vec![faux_text("second")]),
    ]);
    let ctx = user_context("hi");
    let first = models.complete(faux.get_model(), &ctx, None).await;
    let second = models.complete(faux.get_model(), &ctx, None).await;
    let exhausted = models.complete(faux.get_model(), &ctx, None).await;
    assert_eq!(text_of(&first), "first");
    assert_eq!(text_of(&second), "second");
    assert_eq!(exhausted.stop_reason, StopReason::Error);
    assert_eq!(
        exhausted.error_message.as_deref(),
        Some("No more faux responses queued")
    );
    assert_eq!(faux.pending_count(), 0);
    assert_eq!(call_count(&faux), 3);
}

#[tokio::test]
async fn can_replace_and_append_queued_responses() {
    let (models, faux) = seeded();
    faux.set_responses(vec![message_step(vec![faux_text("first")])]);
    let ctx = user_context("hi");
    let first = models.complete(faux.get_model(), &ctx, None).await;
    assert_eq!(text_of(&first), "first");
    assert_eq!(faux.pending_count(), 0);
    faux.set_responses(vec![message_step(vec![faux_text("second")])]);
    assert_eq!(faux.pending_count(), 1);
    let second = models.complete(faux.get_model(), &ctx, None).await;
    assert_eq!(text_of(&second), "second");
    faux.append_responses(vec![
        message_step(vec![faux_text("third")]),
        message_step(vec![faux_text("fourth")]),
    ]);
    assert_eq!(faux.pending_count(), 2);
    let third = models.complete(faux.get_model(), &ctx, None).await;
    let fourth = models.complete(faux.get_model(), &ctx, None).await;
    assert_eq!(text_of(&third), "third");
    assert_eq!(text_of(&fourth), "fourth");
    assert_eq!(faux.pending_count(), 0);
}

#[tokio::test]
async fn supports_async_response_factories() {
    let (models, faux) = seeded();
    faux.set_responses(vec![FauxResponseStep::Factory(Box::new(
        |context, _options, state, _model| {
            let text = format!("{}:{}", context.messages().len(), state.call_count);
            Box::pin(async move { Ok(faux_assistant_message(vec![faux_text(&text)])) })
        },
    ))]);
    let response = models
        .complete(faux.get_model(), &user_context("hi"), None)
        .await;
    assert_eq!(text_of(&response), "1:1");
}

#[tokio::test]
async fn emits_an_error_when_a_response_factory_throws() {
    let (models, faux) = seeded();
    faux.set_responses(vec![FauxResponseStep::Factory(Box::new(
        |_context, _options, _state, _model| Box::pin(async move { Err(String::from("boom")) }),
    ))]);
    let stream = models.stream(faux.get_model(), &user_context("hi"), None);
    let events = collect_events(stream).await;
    assert_eq!(events.len(), 1, "{events:?}");
    assert_eq!(event_type(&events[0]), "error");
    let AssistantMessageEvent::Error {
        reason,
        error_message,
    } = &events[0]
    else {
        panic!("error event: {events:?}");
    };
    assert_eq!(reason, &ovo_ai::stream::ErrorReason::Error);
    assert_eq!(error_message, "boom");
}

#[tokio::test]
async fn rejects_a_queued_response_without_a_terminal_stop_reason() {
    let (models, faux) = seeded();
    faux.set_responses(vec![FauxResponseStep::Message(Box::new(
        faux_assistant_message_with(
            vec![faux_text("partial")],
            &FauxMessageOptions {
                stop_reason: Some(StopReason::Pending),
                ..FauxMessageOptions::default()
            },
        ),
    ))]);
    let stream = models.stream(faux.get_model(), &user_context("hi"), None);
    let events = collect_events(stream).await;
    assert!(
        events.iter().all(|event| event_type(event) != "done"),
        "{events:?}"
    );
    let terminal = events.last().unwrap_or_else(|| panic!("a terminal event"));
    assert_eq!(event_type(terminal), "error");
    let AssistantMessageEvent::Error {
        reason,
        error_message,
    } = terminal
    else {
        panic!("error event: {terminal:?}");
    };
    assert_eq!(reason, &ovo_ai::stream::ErrorReason::Error);
    assert_eq!(error_message, "Faux response ended without a stop reason");
}

#[tokio::test]
async fn estimates_prompt_and_output_tokens_from_serialized_context() {
    let (models, faux) = seeded();
    faux.set_responses(vec![message_step(vec![faux_text("done")])]);
    let tool = json!({
        "name": "echo",
        "description": "Echo back text",
        "parameters": {"type": "object", "required": ["text"], "properties": {"text": {"type": "string"}}},
    });
    let prior = serde_json::to_value(ovo_ai::types::message::Message::Assistant(Box::new(
        faux_assistant_message(vec![faux_text("prior")]),
    )))
    .unwrap_or_else(|error| panic!("assistant message serializes: {error}"));
    let ctx = context(json!({
        "systemPrompt": "sys",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "hello"},
                    {"type": "image", "mimeType": "image/png", "data": "abcd"},
                ],
                "timestamp": 1,
            },
            prior,
            {
                "role": "toolResult",
                "toolCallId": "tool-1",
                "toolName": "echo",
                "content": [{"type": "text", "text": "tool out"}],
                "isError": false,
                "timestamp": 2,
            },
        ],
        "tools": [tool.clone()],
    }));
    let response = models.complete(faux.get_model(), &ctx, None).await;
    let tools_json = serde_json::to_string(&json!([tool]))
        .unwrap_or_else(|error| panic!("tools serialize: {error}"));
    let prompt_text = [
        "system:sys",
        "user:hello\n[image:image/png:4]",
        "assistant:prior",
        "toolResult:echo\ntool out",
        &format!("tools:{tools_json}"),
    ]
    .join("\n\n");
    let expected_prompt_tokens = ovo_ai::text::js_len(&prompt_text).div_ceil(4) as u64;
    let expected_output_tokens = "done".len().div_ceil(4) as u64;
    assert_eq!(response.usage.input, expected_prompt_tokens, "{response:?}");
    assert_eq!(response.usage.output, expected_output_tokens);
    assert_eq!(response.usage.cache_read, 0);
    assert_eq!(response.usage.cache_write, 0);
    assert_eq!(
        response.usage.total_tokens,
        expected_prompt_tokens + expected_output_tokens
    );
}

#[tokio::test]
async fn does_not_share_cache_across_sessions_or_requests_without_session_id() {
    let (models, faux) = seeded();
    faux.set_responses(vec![
        message_step(vec![faux_text("first")]),
        message_step(vec![faux_text("second")]),
        message_step(vec![faux_text("third")]),
    ]);
    let short = |session: &str| StreamOptions {
        session_id: Some(String::from(session)),
        cache_retention: Some(CacheRetention::Short),
        ..StreamOptions::default()
    };
    let mut messages = vec![json!({"role": "user", "content": "hello", "timestamp": 0})];
    let first = models
        .complete(
            faux.get_model(),
            &context(json!({"messages": messages.clone()})),
            Some(short("session-1")),
        )
        .await;
    assert!(first.usage.cache_write > 0, "{:?}", first.usage);
    messages.push(
        serde_json::to_value(ovo_ai::types::message::Message::Assistant(Box::new(
            first.clone(),
        )))
        .unwrap_or_else(|error| panic!("serialize: {error}")),
    );
    messages.push(json!({"role": "user", "content": "follow up", "timestamp": 1}));
    let ctx = context(json!({"messages": messages}));
    let second = models
        .complete(faux.get_model(), &ctx, Some(short("session-2")))
        .await;
    assert_eq!(second.usage.cache_read, 0, "{:?}", second.usage);
    assert!(second.usage.cache_write > 0, "{:?}", second.usage);
    let third = models.complete(faux.get_model(), &ctx, None).await;
    assert_eq!(third.usage.cache_read, 0, "{:?}", third.usage);
    assert_eq!(third.usage.cache_write, 0, "{:?}", third.usage);
}

#[tokio::test]
async fn simulates_prompt_caching_per_session_id() {
    let (models, faux) = seeded();
    faux.set_responses(vec![
        message_step(vec![faux_text("first")]),
        message_step(vec![faux_text("second")]),
    ]);
    let options = |()| StreamOptions {
        session_id: Some(String::from("session-1")),
        cache_retention: Some(CacheRetention::Short),
        ..StreamOptions::default()
    };
    let mut messages = vec![json!({"role": "user", "content": "hello", "timestamp": 0})];
    let first = models
        .complete(
            faux.get_model(),
            &context(json!({"systemPrompt": "Be concise.", "messages": messages.clone()})),
            Some(options(())),
        )
        .await;
    assert_eq!(first.usage.cache_read, 0, "{:?}", first.usage);
    assert!(first.usage.cache_write > 0, "{:?}", first.usage);
    messages.push(
        serde_json::to_value(ovo_ai::types::message::Message::Assistant(Box::new(
            first.clone(),
        )))
        .unwrap_or_else(|error| panic!("serialize: {error}")),
    );
    messages.push(json!({"role": "user", "content": "follow up", "timestamp": 1}));
    let second = models
        .complete(
            faux.get_model(),
            &context(json!({"systemPrompt": "Be concise.", "messages": messages})),
            Some(options(())),
        )
        .await;
    assert!(second.usage.cache_read > 0, "{:?}", second.usage);
    assert!(
        second.usage.input + second.usage.cache_read > second.usage.input,
        "{:?}",
        second.usage
    );
}

#[tokio::test]
async fn counts_cached_characters_up_to_the_first_difference_in_the_joined_prompt() {
    let (models, faux) = seeded();
    faux.set_responses(vec![
        message_step(vec![faux_text("a")]),
        message_step(vec![faux_text("b")]),
        message_step(vec![faux_text("c")]),
    ]);
    let options = |()| StreamOptions {
        session_id: Some(String::from("session-1")),
        cache_retention: Some(CacheRetention::Short),
        ..StreamOptions::default()
    };
    let user = |content: &str| json!({"role": "user", "content": content, "timestamp": 1});
    // Prompt texts: "user:hello world" (16 characters), then 16 + 2 +
    // "user:next" (9) = 27.
    let first = models
        .complete(
            faux.get_model(),
            &context(json!({"messages": [user("hello world")]})),
            Some(options(())),
        )
        .await;
    assert_eq!(first.usage.input, 0, "{:?}", first.usage);
    assert_eq!(first.usage.cache_read, 0, "{:?}", first.usage);
    assert_eq!(first.usage.cache_write, 4, "{:?}", first.usage);
    let extended = models
        .complete(
            faux.get_model(),
            &context(json!({"messages": [user("hello world"), user("next")]})),
            Some(options(())),
        )
        .await;
    assert_eq!(extended.usage.input, 0, "{:?}", extended.usage);
    assert_eq!(extended.usage.cache_read, 4, "{:?}", extended.usage);
    assert_eq!(extended.usage.cache_write, 3, "{:?}", extended.usage);
    // The first message now differs after "user:hello w" (12
    // characters): "user:hello wide" + 2 + 9 = 26.
    let edited = models
        .complete(
            faux.get_model(),
            &context(json!({"messages": [user("hello wide"), user("next")]})),
            Some(options(())),
        )
        .await;
    assert_eq!(edited.usage.input, 0, "{:?}", edited.usage);
    assert_eq!(edited.usage.cache_read, 3, "{:?}", edited.usage);
    assert_eq!(edited.usage.cache_write, 4, "{:?}", edited.usage);
}

#[tokio::test]
async fn does_not_simulate_caching_when_cache_retention_is_none() {
    let (models, faux) = seeded();
    faux.set_responses(vec![
        message_step(vec![faux_text("first")]),
        message_step(vec![faux_text("second")]),
    ]);
    let options = |()| StreamOptions {
        session_id: Some(String::from("session-1")),
        cache_retention: Some(CacheRetention::None),
        ..StreamOptions::default()
    };
    let mut messages = vec![json!({"role": "user", "content": "hello", "timestamp": 0})];
    let _ = models
        .complete(
            faux.get_model(),
            &context(json!({"messages": messages.clone()})),
            Some(options(())),
        )
        .await;
    messages.push(
        serde_json::to_value(ovo_ai::types::message::Message::Assistant(Box::new(
            faux_assistant_message(vec![faux_text("first")]),
        )))
        .unwrap_or_else(|error| panic!("serialize: {error}")),
    );
    messages.push(json!({"role": "user", "content": "follow up", "timestamp": 1}));
    let second = models
        .complete(
            faux.get_model(),
            &context(json!({"messages": messages})),
            Some(options(())),
        )
        .await;
    assert_eq!(second.usage.cache_read, 0, "{:?}", second.usage);
    assert_eq!(second.usage.cache_write, 0, "{:?}", second.usage);
}

#[tokio::test]
async fn streams_thinking_text_and_partial_tool_call_deltas() {
    let (models, faux) = seeded();
    faux.set_responses(vec![FauxResponseStep::Message(Box::new(
        faux_assistant_message_with(
            vec![
                faux_thinking("thinking text"),
                faux_text("answer text"),
                faux_tool_call(
                    "echo",
                    args(json!({"text": "hi", "count": 12})),
                    Some("tool-1"),
                ),
            ],
            &FauxMessageOptions {
                stop_reason: Some(StopReason::ToolUse),
                ..FauxMessageOptions::default()
            },
        ),
    ))]);
    let stream = models.stream(faux.get_model(), &user_context("hi"), None);
    let events = collect_events(stream).await;
    let types: Vec<&'static str> = events.iter().map(event_type).collect();
    for expected in [
        "thinking_start",
        "thinking_delta",
        "text_start",
        "text_delta",
        "toolcall_start",
        "toolcall_delta",
        "toolcall_end",
    ] {
        assert!(types.contains(&expected), "{expected} in {types:?}");
    }
    let deltas: String = events
        .iter()
        .filter_map(|event| match event {
            AssistantMessageEvent::ToolcallDelta { delta, .. } => Some(delta.as_str()),
            _ => None,
        })
        .collect();
    assert!(
        events
            .iter()
            .filter(|event| matches!(event, AssistantMessageEvent::ToolcallDelta { .. }))
            .count()
            > 1,
        "{events:?}"
    );
    assert_eq!(
        serde_json::from_str::<Value>(&deltas)
            .unwrap_or_else(|error| panic!("arguments parse: {error}")),
        json!({"text": "hi", "count": 12})
    );
}

#[tokio::test]
async fn streams_an_exact_event_order_for_fixed_size_chunks() {
    let (models, faux) = seeded_with(FauxOptions {
        token_size: Some(FauxTokenSize {
            min: Some(1),
            max: Some(1),
        }),
        ..FauxOptions::default()
    });
    faux.set_responses(vec![FauxResponseStep::Message(Box::new(
        faux_assistant_message_with(
            vec![
                faux_thinking("go"),
                faux_text("ok"),
                faux_tool_call("echo", Map::new(), Some("tool-1")),
            ],
            &FauxMessageOptions {
                stop_reason: Some(StopReason::ToolUse),
                ..FauxMessageOptions::default()
            },
        ),
    ))]);
    let mut stream = models.stream(faux.get_model(), &user_context("hi"), None);
    let mut types = Vec::new();
    while let Some(event) = stream.next().await {
        if types.is_empty() {
            assert_eq!(event_type(&event), "start");
            assert_eq!(stream.partial().stop_reason, StopReason::Pending);
        }
        types.push(event_type(&event));
    }
    assert_eq!(
        types,
        [
            "start",
            "thinking_start",
            "thinking_delta",
            "thinking_end",
            "text_start",
            "text_delta",
            "text_end",
            "toolcall_start",
            "toolcall_delta",
            "toolcall_end",
            "done",
        ]
    );
}

#[tokio::test]
async fn streams_multiple_tool_calls_in_one_message() {
    let (models, faux) = seeded();
    faux.set_responses(vec![FauxResponseStep::Message(Box::new(
        faux_assistant_message_with(
            vec![
                faux_tool_call("echo", args(json!({"text": "one"})), Some("tool-1")),
                faux_tool_call("echo", args(json!({"text": "two"})), Some("tool-2")),
            ],
            &FauxMessageOptions {
                stop_reason: Some(StopReason::ToolUse),
                ..FauxMessageOptions::default()
            },
        ),
    ))]);
    let stream = models.stream(faux.get_model(), &user_context("hi"), None);
    let events = collect_events(stream).await;
    let count = |kind: &str| {
        events
            .iter()
            .filter(|event| event_type(event) == kind)
            .count()
    };
    assert_eq!(count("toolcall_start"), 2, "{events:?}");
    assert_eq!(count("toolcall_end"), 2, "{events:?}");
}

#[tokio::test]
async fn streams_an_explicit_assistant_error_message_as_a_terminal_error() {
    let (models, faux) = seeded_with(FauxOptions {
        token_size: Some(FauxTokenSize {
            min: Some(2),
            max: Some(2),
        }),
        ..FauxOptions::default()
    });
    faux.set_responses(vec![FauxResponseStep::Message(Box::new(
        faux_assistant_message_with(
            vec![faux_text("partial")],
            &FauxMessageOptions {
                stop_reason: Some(StopReason::Error),
                error_message: Some(String::from("upstream failed")),
                ..FauxMessageOptions::default()
            },
        ),
    ))]);
    let stream = models.stream(faux.get_model(), &user_context("hi"), None);
    let events = collect_events(stream).await;
    assert_eq!(
        events.iter().map(event_type).collect::<Vec<_>>(),
        ["start", "text_start", "text_delta", "text_end", "error"],
        "{events:?}"
    );
    let terminal = events.last().unwrap_or_else(|| panic!("terminal"));
    let AssistantMessageEvent::Error {
        reason,
        error_message,
    } = terminal
    else {
        panic!("error event: {terminal:?}");
    };
    assert_eq!(reason, &ovo_ai::stream::ErrorReason::Error);
    assert_eq!(error_message, "upstream failed");
}

#[tokio::test]
async fn streams_an_explicit_assistant_aborted_message_as_a_terminal_error() {
    let (models, faux) = seeded_with(FauxOptions {
        token_size: Some(FauxTokenSize {
            min: Some(2),
            max: Some(2),
        }),
        ..FauxOptions::default()
    });
    faux.set_responses(vec![FauxResponseStep::Message(Box::new(
        faux_assistant_message_with(
            vec![faux_text("partial")],
            &FauxMessageOptions {
                stop_reason: Some(StopReason::Aborted),
                error_message: Some(String::from("Request was aborted")),
                ..FauxMessageOptions::default()
            },
        ),
    ))]);
    let stream = models.stream(faux.get_model(), &user_context("hi"), None);
    let events = collect_events(stream).await;
    assert_eq!(
        events.iter().map(event_type).collect::<Vec<_>>(),
        ["start", "text_start", "text_delta", "text_end", "error"],
        "{events:?}"
    );
    let terminal = events.last().unwrap_or_else(|| panic!("terminal"));
    let AssistantMessageEvent::Error {
        reason,
        error_message,
    } = terminal
    else {
        panic!("error event: {terminal:?}");
    };
    assert_eq!(reason, &ovo_ai::stream::ErrorReason::Aborted);
    assert_eq!(error_message, "Request was aborted");
}

#[tokio::test]
async fn supports_aborting_before_the_first_chunk() {
    let (models, faux) = seeded_with(FauxOptions {
        tokens_per_second: Some(50.0),
        token_size: Some(FauxTokenSize {
            min: Some(3),
            max: Some(3),
        }),
        ..FauxOptions::default()
    });
    faux.set_responses(vec![message_step(vec![faux_text(
        "abcdefghijklmnopqrstuvwxyz",
    )])]);
    let signal = CancellationToken::new();
    signal.cancel();
    let options = StreamOptions {
        signal: Some(signal),
        ..StreamOptions::default()
    };
    let stream = models.stream(faux.get_model(), &user_context("hi"), Some(options));
    let events = collect_events(stream).await;
    assert_eq!(events.len(), 1, "{events:?}");
    let AssistantMessageEvent::Error { reason, .. } = &events[0] else {
        panic!("error event: {events:?}");
    };
    assert_eq!(reason, &ovo_ai::stream::ErrorReason::Aborted);
}

#[tokio::test]
async fn supports_aborting_mid_text_stream_when_paced() {
    let (models, faux) = seeded_with(FauxOptions {
        tokens_per_second: Some(100.0),
        token_size: Some(FauxTokenSize {
            min: Some(3),
            max: Some(3),
        }),
        ..FauxOptions::default()
    });
    faux.set_responses(vec![message_step(vec![faux_text(
        "abcdefghijklmnopqrstuvwxyz",
    )])]);
    let signal = CancellationToken::new();
    let options = StreamOptions {
        signal: Some(signal.clone()),
        ..StreamOptions::default()
    };
    let mut stream = models.stream(faux.get_model(), &user_context("hi"), Some(options));
    let mut types = Vec::new();
    let mut text_delta_count = 0;
    while let Some(event) = stream.next().await {
        types.push(event_type(&event));
        if matches!(event, AssistantMessageEvent::TextDelta { .. }) {
            text_delta_count += 1;
            signal.cancel();
        }
    }
    assert_eq!(text_delta_count, 1, "{types:?}");
    assert!(types.contains(&"text_start"), "{types:?}");
    assert!(types.contains(&"text_delta"), "{types:?}");
    assert!(types.contains(&"error"), "{types:?}");
    assert!(!types.contains(&"text_end"), "{types:?}");
}

#[tokio::test]
async fn supports_aborting_mid_thinking_stream_when_paced() {
    let (models, faux) = seeded_with(FauxOptions {
        tokens_per_second: Some(100.0),
        token_size: Some(FauxTokenSize {
            min: Some(3),
            max: Some(3),
        }),
        ..FauxOptions::default()
    });
    faux.set_responses(vec![FauxResponseStep::Message(Box::new(
        faux_assistant_message(vec![faux_thinking("abcdefghijklmnopqrstuvwxyz")]),
    ))]);
    let signal = CancellationToken::new();
    let options = StreamOptions {
        signal: Some(signal.clone()),
        ..StreamOptions::default()
    };
    let mut stream = models.stream(faux.get_model(), &user_context("hi"), Some(options));
    let mut types = Vec::new();
    let mut thinking_delta_count = 0;
    while let Some(event) = stream.next().await {
        types.push(event_type(&event));
        if matches!(event, AssistantMessageEvent::ThinkingDelta { .. }) {
            thinking_delta_count += 1;
            signal.cancel();
        }
    }
    assert_eq!(thinking_delta_count, 1, "{types:?}");
    assert!(types.contains(&"thinking_start"), "{types:?}");
    assert!(types.contains(&"thinking_delta"), "{types:?}");
    assert!(types.contains(&"error"), "{types:?}");
    assert!(!types.contains(&"thinking_end"), "{types:?}");
}

#[tokio::test]
async fn supports_aborting_mid_toolcall_stream_when_paced() {
    let (models, faux) = seeded_with(FauxOptions {
        tokens_per_second: Some(100.0),
        token_size: Some(FauxTokenSize {
            min: Some(3),
            max: Some(3),
        }),
        ..FauxOptions::default()
    });
    faux.set_responses(vec![FauxResponseStep::Message(Box::new(
        faux_assistant_message_with(
            vec![faux_tool_call(
                "echo",
                args(json!({"text": "abcdefghijklmnopqrstuvwxyz", "count": 123_456_789})),
                Some("tool-1"),
            )],
            &FauxMessageOptions {
                stop_reason: Some(StopReason::ToolUse),
                ..FauxMessageOptions::default()
            },
        ),
    ))]);
    let signal = CancellationToken::new();
    let options = StreamOptions {
        signal: Some(signal.clone()),
        ..StreamOptions::default()
    };
    let mut stream = models.stream(faux.get_model(), &user_context("hi"), Some(options));
    let mut types = Vec::new();
    let mut tool_call_delta_count = 0;
    while let Some(event) = stream.next().await {
        types.push(event_type(&event));
        if matches!(event, AssistantMessageEvent::ToolcallDelta { .. }) {
            tool_call_delta_count += 1;
            signal.cancel();
        }
    }
    assert_eq!(tool_call_delta_count, 1, "{types:?}");
    assert!(types.contains(&"toolcall_start"), "{types:?}");
    assert!(types.contains(&"toolcall_delta"), "{types:?}");
    assert!(types.contains(&"error"), "{types:?}");
    assert!(!types.contains(&"toolcall_end"), "{types:?}");
}

#[tokio::test]
async fn unregisters_the_provider() {
    let (models, faux) = seeded();
    faux.set_responses(vec![message_step(vec![faux_text("hello")])]);
    models.delete_provider("faux");
    let result = models
        .complete(faux.get_model(), &user_context("hi"), None)
        .await;
    assert_eq!(result.stop_reason, StopReason::Error, "{result:?}");
    assert_eq!(
        result.error_message.as_deref(),
        Some("Unknown provider: faux")
    );
}
