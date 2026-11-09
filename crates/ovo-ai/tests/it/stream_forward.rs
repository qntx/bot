//! `Models::stream` must expose the same `partial()` as the provider's
//! direct stream after every event — the lazy-dispatch forward passes the
//! inner stream's items through unchanged.

use std::sync::Arc;

use futures::StreamExt;
use ovo_ai::faux::{
    FauxOptions, FauxProvider, FauxResponseStep, FauxTokenSize, faux_assistant_message,
    faux_provider, faux_text, faux_thinking, faux_tool_call,
};
use ovo_ai::models::{CreateModelsOptions, Models, create_models};
use ovo_ai::stream::AssistantStream;
use ovo_ai::types::content::AssistantContentBlock;
use ovo_ai::types::context::Context;
use serde_json::json;

fn seeded() -> (Arc<Models>, FauxProvider) {
    let models = create_models(CreateModelsOptions::default());
    let faux = faux_provider(FauxOptions {
        token_size: Some(FauxTokenSize {
            min: Some(1),
            max: Some(1),
        }),
        ..FauxOptions::default()
    })
    .unwrap_or_else(|e| panic!("faux: {e}"));
    models.set_provider(Arc::clone(&faux.provider));
    (models, faux)
}

fn context() -> Context {
    serde_json::from_value(json!({
        "messages": [{"role": "user", "content": "hi", "timestamp": 0}],
    }))
    .unwrap_or_else(|e| panic!("context: {e}"))
}

fn masked(message: &ovo_ai::types::message::AssistantMessage) -> serde_json::Value {
    let mut value = serde_json::to_value(message).unwrap_or_default();
    if let Some(object) = value.as_object_mut() {
        // Wall-clock fields — the vectors strip them the same way.
        object.remove("timestamp");
        object.remove("durationMs");
    }
    value
}

async fn collect(stream: AssistantStream) -> (Vec<serde_json::Value>, serde_json::Value) {
    let mut stream = stream;
    let mut partials = Vec::new();
    while let Some(_event) = stream.next().await {
        partials.push(masked(stream.partial()));
    }
    (partials, masked(&stream.result().await))
}

/// For every scripted content mix, `Models::stream`'s `partial()` after
/// each event equals the direct provider stream's.
#[tokio::test]
async fn models_stream_partial_matches_direct_stream() {
    for (label, content) in [
        ("text", vec![faux_text("hello")]),
        (
            "thinking-signature",
            vec![
                serde_json::from_value::<AssistantContentBlock>(json!({
                    "type": "thinking",
                    "thinking": "hm",
                    "thinkingSignature": "sig-1"
                }))
                .unwrap_or_else(|e| panic!("thinking block: {e}")),
            ],
        ),
        (
            "tool-call",
            vec![faux_tool_call(
                "read",
                json!({"path": "a.txt"})
                    .as_object()
                    .cloned()
                    .unwrap_or_default(),
                None,
            )],
        ),
        (
            "mixed",
            vec![
                faux_thinking("thinking"),
                faux_text("text"),
                faux_tool_call(
                    "write",
                    json!({"p": 1}).as_object().cloned().unwrap_or_default(),
                    None,
                ),
            ],
        ),
    ] {
        let (models, faux) = seeded();
        faux.set_responses(vec![
            FauxResponseStep::Message(Box::new(faux_assistant_message(content.clone()))),
            FauxResponseStep::Message(Box::new(faux_assistant_message(content))),
        ]);
        let model = faux.models[0].clone();
        // Direct provider stream.
        let transcript = ovo_ai::transcript::normalize_context(&context());
        let direct = collect(faux.provider.stream(
            &model,
            transcript,
            ovo_ai::options::StreamOptions::default(),
        ))
        .await;
        // Through `Models::stream`'s lazy forward.
        let via_models = collect(models.stream(&model, &context(), None)).await;
        assert_eq!(via_models, direct, "{label}: partial() per event + result");
    }
}
