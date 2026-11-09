//! `models.ts` + `faux.ts` — registry lookups, auth application,
//! header merging, and the scripted provider's streaming/deferred
//! paths.

/// parity: AI-MODELS-REGISTRY
/// parity: AI-FAUX
use std::sync::Arc;

use ovo_ai::cost::{clamp_thinking_level, get_supported_thinking_levels};
use ovo_ai::faux::{
    FauxOptions, FauxResponseStep, faux_assistant_message, faux_provider, faux_text,
};
use ovo_ai::models::{
    CreateModelsOptions, ModelsRefreshOptions, create_models, has_api, merge_headers,
    models_are_equal,
};
use ovo_ai::types::model::ModelThinkingLevel;

use ovo_ai::auth::AuthResolutionOverrides;
use ovo_ai::options::StreamOptions;
use ovo_ai::types::content::StopReason;
use ovo_ai::types::context::Context;

fn seeded_models() -> (Arc<ovo_ai::models::Models>, ovo_ai::faux::FauxProvider) {
    let models = create_models(CreateModelsOptions::default());
    let faux = faux_provider(FauxOptions::default());
    models.set_provider(Arc::clone(&faux.provider));
    models.set_provider(Arc::clone(&faux.provider));
    (models, faux)
}

fn text_context(text: &str) -> Context {
    serde_json::from_value(serde_json::json!({
        "messages": [{"role": "user", "content": text, "timestamp": 0}],
    }))
    .unwrap_or_else(|error| panic!("context: {error}"))
}

#[test]
fn registry_lookups_and_filters() {
    let (models, faux) = seeded_models();
    assert_eq!(models.get_providers().len(), 1);
    assert_eq!(models.get_models(None).len(), 1);
    let model = models
        .get_model("faux", "faux-1")
        .unwrap_or_else(|| panic!("model not found"));
    assert_eq!(model.api.as_str(), "faux");
    assert!(has_api(
        &ovo_ai::types::model::AnyModel::Chat(Box::new(model.clone())),
        "faux"
    ));
    assert!(models_are_equal(
        &ovo_ai::types::model::AnyModel::Chat(Box::new(model)),
        &ovo_ai::types::model::AnyModel::Chat(Box::new(faux.models[0].clone())),
    ));
    let levels = get_supported_thinking_levels(&faux.models[0]);
    assert_eq!(levels, [ModelThinkingLevel::Off]);
    assert_eq!(
        clamp_thinking_level(&faux.models[0], ModelThinkingLevel::High),
        ModelThinkingLevel::Off
    );
}

#[test]
fn merge_headers_case_insensitive() {
    let mut base = serde_json::Map::new();
    base.insert(String::from("Authorization"), serde_json::json!("a"));
    let mut override_ = serde_json::Map::new();
    override_.insert(String::from("authorization"), serde_json::json!("b"));
    let merged = merge_headers(Some(&base), Some(&override_)).expect("merged");
    assert_eq!(merged.len(), 1);
    assert_eq!(
        merged.get("authorization").and_then(|v| v.as_str()),
        Some("b")
    );
}

#[tokio::test]
async fn stream_queued_response() {
    let (models, faux) = seeded_models();
    faux.set_responses(vec![FauxResponseStep::Message(Box::new(
        faux_assistant_message(vec![faux_text("hello")]),
    ))]);
    let model = models
        .get_model("faux", "faux-1")
        .unwrap_or_else(|| panic!("model not found"));
    let context = text_context("hi");
    let result = models.complete(&model, &context, None).await;
    assert_eq!(result.stop_reason, StopReason::Stop);
    let text = result
        .content
        .iter()
        .find_map(|b| match b {
            ovo_ai::types::content::AssistantContentBlock::Text(t) => Some(t.text.clone()),
            _ => None,
        })
        .expect("text");
    assert_eq!(text, "hello");
    assert_eq!(
        faux.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .call_count,
        1
    );
}

#[tokio::test]
async fn stream_empty_queue_errors() {
    let (models, _faux) = seeded_models();
    let model = models
        .get_model("faux", "faux-1")
        .unwrap_or_else(|| panic!("model not found"));
    let context = text_context("hi");
    let result = models.complete(&model, &context, None).await;
    assert_eq!(result.stop_reason, StopReason::Error);
}

#[tokio::test]
async fn refresh_static_provider_no_op() {
    let (models, _faux) = seeded_models();
    let result = models.refresh(ModelsRefreshOptions::default()).await;
    assert!(!result.aborted);
    assert!(result.errors.is_empty());
}

/// `models-runtime`'s "streamDeferred → fetchDeferred" — a
/// `FauxResponseStep::Deferred` emits `done:deferred` with a handle;
/// `fetch_deferred` replays the eventual message.
#[tokio::test]
async fn stream_deferred_then_fetch() {
    let faux = faux_provider(FauxOptions::default());
    let models = create_models(CreateModelsOptions::default());
    models.set_provider(Arc::clone(&faux.provider));
    faux.set_responses(vec![FauxResponseStep::Deferred {
        id: String::from("d-1"),
        then: Box::new(faux_assistant_message(vec![faux_text("later")])),
    }]);
    let model = models
        .get_model("faux", "faux-1")
        .unwrap_or_else(|| panic!("model"));
    let stream = models.stream_deferred(&model, &text_context("hi"), None);
    let result = stream.result().await;
    assert_eq!(result.stop_reason, StopReason::Deferred, "{result:?}");
    let handle = ovo_ai::types::deferred::DeferredHandle::new("faux", "faux-1", "faux", "d-1");
    let fetched = models.fetch_deferred(&handle, None);
    let fetched = fetched.result().await;
    assert_eq!(fetched.stop_reason, StopReason::Stop, "{fetched:?}");
}

/// `faux-provider`'s `onResponse` — the callback fires on stream.
#[tokio::test]
async fn stream_invokes_on_response() {
    let calls = Arc::new(std::sync::Mutex::new(Vec::<u16>::new()));
    let faux = faux_provider(FauxOptions::default());
    let models = create_models(CreateModelsOptions::default());
    models.set_provider(Arc::clone(&faux.provider));
    faux.set_responses(vec![FauxResponseStep::Message(Box::new(
        faux_assistant_message(vec![faux_text("hi")]),
    ))]);
    let model = models
        .get_model("faux", "faux-1")
        .unwrap_or_else(|| panic!("model"));
    let calls2 = Arc::clone(&calls);
    let mut options = StreamOptions::default();
    options.on_response = Some(Arc::new(move |meta, _model| {
        let recorded = Arc::clone(&calls2);
        let status = meta.status;
        Box::pin(async move {
            recorded
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .push(status);
        })
    }));
    let stream = models.stream(&model, &text_context("x"), Some(options));
    let _ = stream.result().await;
    assert_eq!(
        *calls
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner),
        vec![200]
    );
}

/// `models-entry`'s `withClientIdentity` — request identity merges.
#[tokio::test]
async fn with_client_identity_carries() {
    let faux = faux_provider(FauxOptions::default());
    let models = create_models(CreateModelsOptions::default());
    models.set_provider(Arc::clone(&faux.provider));
    let identity = ovo_ai::identity::ClientIdentity {
        user_agent: None,
        version: Some(String::from("1.0")),
        attribution: None,
    };
    let wrapped = ovo_ai::models::with_client_identity(&models, identity);
    assert!(wrapped.get_provider("faux").is_some());
}

/// `models-runtime`'s `getAuth` — ambient env key resolves.
#[tokio::test]
async fn get_auth_resolves_env_key() {
    let faux = faux_provider(FauxOptions::default());
    let mut env = std::collections::HashMap::new();
    env.insert(String::from("FAUX_API_KEY"), String::from("k-1"));
    let models = create_models(CreateModelsOptions {
        env: Some(Arc::new(ovo_ai::auth::env::MapEnv::new(env))),
        ..CreateModelsOptions::default()
    });
    models.set_provider(Arc::clone(&faux.provider));
    let auth = models
        .get_auth(
            ovo_ai::models::ProviderOrModel::Id("faux"),
            &AuthResolutionOverrides::default(),
        )
        .await;
    assert!(auth.is_ok(), "{auth:?}");
}
