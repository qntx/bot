//! `models-runtime.test.ts` + `model-types.test.ts` +
//! `models-entry.test.ts` — the `Models` registry, refresh loop,
//! auth resolution and provider dispatch.

/// parity: AI-MODELS-REGISTRY
use std::pin::Pin;
use std::sync::{Arc, Mutex};

use futures::StreamExt;
use ovo_ai::auth::resolve::{ApiKeyAuth, AuthResolutionOverrides, OAuthAuth};
use ovo_ai::auth::store::{CredentialStore, InMemoryCredentialStore};
use ovo_ai::auth::types::{ApiKeyCredential, Credential};
use ovo_ai::auth::types::{AuthError, AuthResult, CredentialInfo, ModelAuth, OAuthCredential};
use ovo_ai::cost::{clamp_thinking_level, get_supported_thinking_levels};
use ovo_ai::faux::{
    FauxOptions, FauxResponseStep, faux_assistant_message, faux_provider, faux_text,
};
use ovo_ai::models::{
    CreateModelsOptions, InMemoryModelsStore, Models, ModelsRefreshOptions, ModelsStore,
    ProviderOrModel, create_models, has_api, merge_headers, models_are_equal,
};
use ovo_ai::models_error::{ModelsError, ModelsErrorCode};
use ovo_ai::options::{SimpleStreamOptions, StreamOptions};
use ovo_ai::provider::{
    BoxFuture, ModelType, ModelsPublication, ModelsStoreEntry, PersistAction, Provider,
    ProviderAuth, RefreshModelsContext, is_model_type,
};
use ovo_ai::stream::{AssistantStream, DoneReason};
use ovo_ai::transcript::TranscriptContext;
use ovo_ai::types::content::StopReason;
use ovo_ai::types::context::Context;
use ovo_ai::types::model::ModelThinkingLevel;
use ovo_ai::types::model::{AnyModel, Model};
use secrecy::ExposeSecret;
use serde_json::{Map, Value, json};

fn test_model(provider: &str, id: &str) -> Model {
    serde_json::from_value(json!({
        "id": id,
        "name": id,
        "api": "test-api",
        "provider": provider,
        "baseUrl": "https://example.test/v1",
        "reasoning": false,
        "input": ["text"],
        "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
        "contextWindow": 10000,
        "maxTokens": 1000,
    }))
    .unwrap_or_else(|error| panic!("test model: {error}"))
}

fn context() -> Context {
    serde_json::from_value(json!({
        "messages": [{"role": "user", "content": "hi", "timestamp": 0}],
    }))
    .unwrap_or_else(|error| panic!("context: {error}"))
}

type RefreshFn =
    Arc<dyn Fn(RefreshModelsContext) -> BoxFuture<'static, Result<(), ModelsError>> + Send + Sync>;

/// `testProvider` — ambient auth, scripted model list, recording
/// stream, optional `refreshModels`/`getModels` hooks.
struct TestProvider {
    id: String,
    base_url: Option<String>,
    auth: ProviderAuth,
    models: Arc<Mutex<Vec<Model>>>,
    all_models: Option<Vec<AnyModel>>,
    refresh: Option<RefreshFn>,
    calls: Arc<Mutex<Vec<(Model, StreamOptions)>>>,
}

impl TestProvider {
    fn new(id: &str) -> Self {
        Self {
            id: id.to_owned(),
            base_url: None,
            auth: ProviderAuth {
                api_key: Some(ApiKeyAuth::custom(
                    "Ambient",
                    Arc::new(|_, _| {
                        Ok(Some(AuthResult {
                            auth: ModelAuth::default(),
                            env: None,
                            source: None,
                        }))
                    }),
                )),
                oauth: None,
            },
            models: Arc::new(Mutex::new(vec![test_model(id, "model-a")])),
            all_models: None,
            refresh: None,
            calls: Arc::new(Mutex::new(Vec::new())),
        }
    }

    fn set(self, models: &Arc<Models>) {
        models.set_provider(Arc::new(self));
    }

    fn record(&self, model: &Model, options: &StreamOptions) {
        self.calls
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push((model.clone(), options.clone()));
    }
}

impl Provider for TestProvider {
    fn id(&self) -> &str {
        &self.id
    }
    fn name(&self) -> &str {
        &self.id
    }
    fn base_url(&self) -> Option<&str> {
        self.base_url.as_deref()
    }
    fn auth(&self) -> &ProviderAuth {
        &self.auth
    }
    fn get_models(&self) -> Vec<Model> {
        self.models
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }
    fn get_all_models(&self) -> Vec<AnyModel> {
        self.all_models.clone().unwrap_or_else(|| {
            self.get_models()
                .into_iter()
                .map(|m| AnyModel::Chat(Box::new(m)))
                .collect()
        })
    }
    fn refresh_models(
        &self,
        context: RefreshModelsContext,
    ) -> BoxFuture<'_, Result<(), ModelsError>> {
        match &self.refresh {
            Some(f) => f(context),
            None => Box::pin(async { Ok(()) }),
        }
    }
    fn stream(
        &self,
        model: &Model,
        _context: TranscriptContext,
        options: StreamOptions,
    ) -> AssistantStream {
        self.record(model, &options);
        let mut base = faux_assistant_message(Vec::new());
        base.stop_reason = StopReason::Pending;
        AssistantStream::spawn(base, 16, |mut sender| async move {
            sender.start().await.ok();
            sender.done(DoneReason::Stop).await.ok();
        })
    }
    fn stream_simple(
        &self,
        model: &Model,
        context: TranscriptContext,
        options: SimpleStreamOptions,
    ) -> AssistantStream {
        self.stream(model, context, options.base)
    }
}

fn models() -> Arc<Models> {
    create_models(CreateModelsOptions::default())
}

/// A provider that implements the trait surface by hand — the
/// `handwritten` provider from `model-types.test.ts`.
struct Handwritten {
    faux: ovo_ai::faux::FauxProvider,
}

impl Provider for Handwritten {
    fn id(&self) -> &'static str {
        "handwritten"
    }
    fn name(&self) -> &'static str {
        "Handwritten"
    }
    fn auth(&self) -> &ProviderAuth {
        self.faux.provider.auth()
    }
    fn get_models(&self) -> Vec<Model> {
        self.faux.models.clone()
    }
    fn stream(
        &self,
        model: &Model,
        context: TranscriptContext,
        options: StreamOptions,
    ) -> AssistantStream {
        self.faux.provider.stream(model, context, options)
    }
    fn stream_simple(
        &self,
        model: &Model,
        context: TranscriptContext,
        options: SimpleStreamOptions,
    ) -> AssistantStream {
        self.faux.provider.stream_simple(model, context, options)
    }
}

fn env_key_auth(key: Option<&str>) -> ApiKeyAuth {
    let key = key.map(str::to_owned);
    ApiKeyAuth::custom(
        "Test API key",
        Arc::new(move |credential, _env| {
            let resolved = credential
                .and_then(|c| c.key.as_ref())
                .map(|k| ExposeSecret::expose_secret(k).to_owned())
                .or_else(|| key.clone());
            Ok(resolved.map(|found| {
                let mut result = AuthResult {
                    auth: ModelAuth {
                        api_key: Some(secrecy::SecretString::from(found)),
                        ..ModelAuth::default()
                    },
                    env: None,
                    source: None,
                };
                result.source = Some(if credential.is_some() {
                    String::from("stored")
                } else {
                    String::from("env")
                });
                result
            }))
        }),
    )
}

#[tokio::test]
async fn enumerates_credential_metadata_without_exposing_secrets() {
    let credentials = InMemoryCredentialStore::new();
    credentials
        .modify(
            "api-provider",
            Box::new(|_| {
                Box::pin(async {
                    Some(Credential::ApiKey(ApiKeyCredential {
                        key: Some(secrecy::SecretString::from(String::from("secret"))),
                        env: None,
                    }))
                })
            }),
        )
        .await
        .unwrap_or_else(|error| panic!("modify: {error}"));
    let list = credentials
        .list()
        .await
        .unwrap_or_else(|error| panic!("list: {error}"));
    assert_eq!(
        list.iter()
            .map(|info| (info.provider_id.as_str(), info.type_))
            .collect::<Vec<_>>(),
        [("api-provider", "api_key")]
    );
}

#[test]
fn applies_request_wide_pricing_tiers_above_the_configured_input_threshold() {
    let mut model = test_model("openai", "gpt-5.6-sol");
    model.cost = serde_json::from_value(json!({
        "input": 5, "output": 30, "cacheRead": 0.5, "cacheWrite": 6.25,
        "tiers": [{"inputTokensAbove": 272_000, "input": 10, "output": 45, "cacheRead": 1, "cacheWrite": 12.5}],
    }))
    .unwrap_or_else(|error| panic!("cost: {error}"));
    let usage = |cache_write: u64| -> ovo_ai::types::usage::Usage {
        serde_json::from_value(json!({
            "input": 200_000, "output": 100_000, "cacheRead": 72_000,
            "cacheWrite": cache_write, "totalTokens": 372_000 + cache_write,
            "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "total": 0},
        }))
        .unwrap_or_else(|error| panic!("usage: {error}"))
    };
    let mut short = usage(0);
    let cost = ovo_ai::cost::calculate_cost(&model, &mut short);
    assert_eq!(cost.input, 1.0);
    assert_eq!(cost.output, 3.0);
    assert_eq!(cost.cache_read, 0.036);
    assert_eq!(cost.cache_write, 0.0);
    let mut long = usage(1);
    let long_cost = ovo_ai::cost::calculate_cost(&model, &mut long);
    assert_eq!(long_cost.input, 2.0);
    assert_eq!(long_cost.output, 4.5);
    assert_eq!(long_cost.cache_read, 0.072);
    assert_eq!(long_cost.cache_write, 0.000_012_5);
}

#[test]
fn marks_catalog_derived_totals_as_not_provider_reported() {
    let mut usage: ovo_ai::types::usage::Usage = serde_json::from_value(json!({
        "input": 1_000_000, "output": 0, "cacheRead": 0, "cacheWrite": 0,
        "totalTokens": 1_000_000,
        "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "total": 9, "totalFromProvider": true},
    }))
    .unwrap_or_else(|error| panic!("usage: {error}"));
    let _ = ovo_ai::cost::calculate_cost(&test_model("openai", "gpt-5.6-sol"), &mut usage);
    assert_eq!(usage.cost.total_from_provider, Some(false));
}

#[test]
fn registers_replaces_and_deletes_providers() {
    let models = models();
    TestProvider::new("p1").set(&models);
    TestProvider::new("p2").set(&models);
    assert_eq!(
        models
            .get_providers()
            .iter()
            .map(|p| p.id().to_owned())
            .collect::<Vec<_>>(),
        ["p1", "p2"]
    );
    models.set_provider(Arc::new(TestProvider::new("p1")));
    assert!(models.get_provider("p1").is_some());
    assert_eq!(models.get_providers().len(), 2);
    models.delete_provider("p1");
    assert!(models.get_provider("p1").is_none());
    models.clear_providers();
    assert!(models.get_providers().is_empty());
}

#[test]
fn lists_and_finds_models_per_provider() {
    let models = models();
    let mut p1 = TestProvider::new("p1");
    p1.models = Arc::new(Mutex::new(vec![
        test_model("p1", "m1"),
        test_model("p1", "m2"),
    ]));
    p1.set(&models);
    let mut p2 = TestProvider::new("p2");
    p2.models = Arc::new(Mutex::new(vec![test_model("p2", "m3")]));
    p2.set(&models);
    assert_eq!(
        models
            .get_models(None)
            .iter()
            .map(|m| m.id.as_str())
            .collect::<Vec<_>>(),
        ["m1", "m2", "m3"]
    );
    assert_eq!(
        models
            .get_models(Some("p1"))
            .iter()
            .map(|m| m.id.as_str())
            .collect::<Vec<_>>(),
        ["m1", "m2"]
    );
    assert_eq!(models.get_models(Some("nope")), Vec::new());
    assert_eq!(
        models.get_model("p2", "m3").map(|m| m.id),
        Some(String::from("m3"))
    );
    assert!(models.get_model("p2", "missing").is_none());
    let found = models.get_model("p2", "m3").unwrap_or_else(|| panic!("m3"));
    let found = AnyModel::Chat(Box::new(found));
    assert!(!has_api(&found, "openai-completions"));
    assert!(has_api(&found, "test-api"));
}

#[tokio::test]
async fn keeps_chat_reads_independent_from_the_all_model_catalog() {
    let models = models();
    let mut provider = TestProvider::new("chat-only");
    provider.all_models = Some(Vec::new());
    models.set_provider(Arc::new(provider));
    assert_eq!(
        models
            .get_models(Some("chat-only"))
            .iter()
            .map(|m| m.id.as_str())
            .collect::<Vec<_>>(),
        ["model-a"]
    );
    assert_eq!(
        models.get_model("chat-only", "model-a").map(|m| m.id),
        Some(String::from("model-a"))
    );
    assert_eq!(
        models
            .get_available(Some("chat-only"))
            .await
            .unwrap_or_else(|error| panic!("available: {error}"))
            .iter()
            .map(|m| m.id.as_str())
            .collect::<Vec<_>>(),
        ["model-a"]
    );
    assert_eq!(models.get_all_models(Some("chat-only")), Vec::new());
}

#[tokio::test]
async fn refresh_updates_every_configured_dynamic_provider_and_reports_failures() {
    let list = Arc::new(Mutex::new(vec![test_model("dyn", "before")]));
    let refreshes = Arc::new(Mutex::new(0usize));
    let models = models();
    let mut dyn_provider = TestProvider::new("dyn");
    {
        let list = Arc::clone(&list);
        let refreshes = Arc::clone(&refreshes);
        dyn_provider.models = Arc::clone(&list);
        dyn_provider.refresh = Some(Arc::new(move |context| {
            let list = Arc::clone(&list);
            let refreshes = Arc::clone(&refreshes);
            Box::pin(async move {
                if !context.allow_network {
                    return Ok(());
                }
                *refreshes
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner) += 1;
                let publish = Arc::clone(&context.publish);
                publish(ModelsPublication {
                    persist: PersistAction::Unchanged,
                    update: Some(Box::new(move || {
                        *list
                            .lock()
                            .unwrap_or_else(std::sync::PoisonError::into_inner) =
                            vec![test_model("dyn", "after")];
                    })),
                })
                .await;
                Ok(())
            })
        }));
    }
    models.set_provider(Arc::new(dyn_provider));
    TestProvider::new("static").set(&models);
    assert!(models.get_model("dyn", "before").is_some());
    let first = models.refresh(ModelsRefreshOptions::default()).await;
    assert!(first.errors.is_empty());
    assert_eq!(
        *refreshes
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner),
        1
    );
    assert!(models.get_model("dyn", "after").is_some());
    assert!(models.get_model("dyn", "before").is_none());

    let mut flaky = TestProvider::new("flaky");
    flaky.refresh = Some(Arc::new(|context| {
        Box::pin(async move {
            if context.allow_network {
                return Err(ModelsError::new(
                    ModelsErrorCode::ModelSource,
                    "fetch failed",
                ));
            }
            Ok(())
        })
    }));
    models.set_provider(Arc::new(flaky));
    let second = models.refresh(ModelsRefreshOptions::default()).await;
    assert_eq!(
        *refreshes
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner),
        2
    );
    assert_eq!(
        second
            .errors
            .get("flaky")
            .map(ToString::to_string)
            .as_deref(),
        Some("fetch failed"),
    );
}

#[tokio::test]
async fn restricts_refresh_work_to_selected_providers() {
    let calls = Arc::new(Mutex::new(Vec::<String>::new()));
    let models = models();
    for id in ["one", "two"] {
        let mut provider = TestProvider::new(id);
        let calls = Arc::clone(&calls);
        provider.refresh = Some(Arc::new(move |context| {
            let calls = Arc::clone(&calls);
            Box::pin(async move {
                calls
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .push(format!(
                        "{id}:{}",
                        if context.allow_network {
                            "network"
                        } else {
                            "cache"
                        }
                    ));
                Ok(())
            })
        }));
        models.set_provider(Arc::new(provider));
    }
    let result = models
        .refresh(ModelsRefreshOptions {
            providers: Some(vec![String::from("two"), String::from("unknown")]),
            ..ModelsRefreshOptions::default()
        })
        .await;
    assert!(result.errors.is_empty());
    assert_eq!(
        *calls
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner),
        ["two:cache", "two:network"]
    );
}

#[tokio::test]
async fn persists_dynamic_catalogs_and_restores_them_without_network_access() {
    let store: Arc<dyn ModelsStore> = Arc::new(InMemoryModelsStore::new());
    let models = create_models(CreateModelsOptions {
        models_store: Some(Arc::clone(&store)),
        ..CreateModelsOptions::default()
    });
    let mut provider = TestProvider::new("dynamic");
    provider.refresh = Some(Arc::new(|context| {
        Box::pin(async move {
            if !context.allow_network {
                return Ok(());
            }
            (context.publish)(ModelsPublication {
                persist: PersistAction::Write(ModelsStoreEntry {
                    models: vec![AnyModel::Chat(Box::new(test_model("dynamic", "fetched")))],
                    checked_at: 0.0,
                }),
                update: None,
            })
            .await;
            Ok(())
        })
    }));
    provider.models = Arc::new(Mutex::new(Vec::new()));
    provider.all_models = Some(Vec::new());
    models.set_provider(Arc::new(provider));
    let online = models.refresh(ModelsRefreshOptions::default()).await;
    assert!(online.errors.is_empty(), "{:?}", online.errors);
}

#[tokio::test]
async fn passes_the_registered_providers_base_url_to_refresh_models() {
    let base_urls = Arc::new(Mutex::new(Vec::<Option<String>>::new()));
    let models = models();
    let mut provider = TestProvider::new("dynamic");
    provider.base_url = Some(String::from("https://regional.example.test/v1"));
    let recorded_urls = Arc::clone(&base_urls);
    provider.refresh = Some(Arc::new(move |context| {
        let urls = Arc::clone(&recorded_urls);
        Box::pin(async move {
            urls.lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .push(context.base_url.clone());
            Ok(())
        })
    }));
    models.set_provider(Arc::new(provider));
    models.refresh(ModelsRefreshOptions::default()).await;
    assert_eq!(
        *base_urls
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner),
        [
            Some(String::from("https://regional.example.test/v1")),
            Some(String::from("https://regional.example.test/v1"))
        ]
    );
}

#[tokio::test]
async fn always_gives_providers_a_concrete_signal() {
    let received = Arc::new(Mutex::new(None::<bool>));
    let models = models();
    let mut provider = TestProvider::new("dynamic");
    let received_flag = Arc::clone(&received);
    provider.refresh = Some(Arc::new(move |context| {
        let flag = Arc::clone(&received_flag);
        Box::pin(async move {
            *flag
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner) =
                Some(context.signal.is_cancelled());
            Ok(())
        })
    }));
    models.set_provider(Arc::new(provider));
    let result = models.refresh(ModelsRefreshOptions::default()).await;
    assert!(!result.aborted);
    assert_eq!(
        *received
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner),
        Some(false)
    );
}

#[tokio::test]
async fn returns_aborted_state_without_reporting_cancellation_as_a_provider_error() {
    let controller = tokio_util::sync::CancellationToken::new();
    let models = models();
    let mut provider = TestProvider::new("dynamic");
    let abort_token = controller.clone();
    provider.refresh = Some(Arc::new(move |context| {
        let abort = abort_token.clone();
        Box::pin(async move {
            abort.cancel();
            let _ = context;
            Ok(())
        })
    }));
    models.set_provider(Arc::new(provider));
    let result = models
        .refresh(ModelsRefreshOptions {
            signal: Some(controller),
            ..ModelsRefreshOptions::default()
        })
        .await;
    assert!(result.aborted);
    assert!(result.errors.is_empty());
}

#[tokio::test]
async fn produces_an_error_stream_for_unknown_providers_instead_of_throwing() {
    let models = models();
    let result = models
        .complete_simple(&test_model("ghost", "model-a"), &context(), None)
        .await;
    assert_eq!(result.stop_reason, StopReason::Error);
    assert_eq!(
        result.error_message.as_deref(),
        Some("Unknown provider: ghost")
    );
}

#[tokio::test]
async fn streams_through_the_provider() {
    let models = models();
    TestProvider::new("p1").set(&models);
    let model = test_model("p1", "model-a");
    let mut stream = models.stream_simple(&model, &context(), None);
    let mut types = Vec::new();
    while let Some(event) = stream.next().await {
        types.push(match event {
            ovo_ai::stream::AssistantMessageEvent::Start => "start",
            ovo_ai::stream::AssistantMessageEvent::Done { .. } => "done",
            other => panic!("unexpected event: {other:?}"),
        });
    }
    assert_eq!(types, ["start", "done"]);
    let message = stream.result().await;
    assert_eq!(message.stop_reason, StopReason::Stop);
}

#[tokio::test]
async fn work_through_a_handwritten_provider_without_get_all_models() {
    let faux = faux_provider(FauxOptions {
        provider: Some(String::from("handwritten")),
        ..FauxOptions::default()
    })
    .unwrap_or_else(|error| panic!("faux provider: {error}"));
    let models = models();
    let handwritten = Handwritten { faux };
    let faux_models = handwritten.faux.models.clone();
    models.set_provider(Arc::new(handwritten));
    let model = models
        .get_model("handwritten", "faux-1")
        .unwrap_or_else(|| panic!("model"));
    let found = AnyModel::Chat(Box::new(model));
    assert!(has_api(&found, "faux"));
    // `modelsAreEqual` — same chat model true, image type false.
    let image: AnyModel = serde_json::from_value(json!({
        "type": "image", "id": "faux-1", "name": "faux-1",
        "api": "test-images", "provider": "handwritten",
        "baseUrl": "https://example.test/v1", "input": ["text"], "output": ["image"],
        "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
    }))
    .unwrap_or_else(|error| panic!("image model: {error}"));
    assert!(!models_are_equal(&found, &image));
    assert_eq!(
        models
            .get_models_of_type(ModelType::Chat, Some("handwritten"))
            .len(),
        1
    );
    assert_eq!(models.get_all_models(Some("handwritten")).len(), 1);
    assert_eq!(
        models.get_models_of_type(ModelType::Image, Some("handwritten")),
        Vec::new()
    );
    let _ = faux_models;
}

#[test]
fn narrow_mixed_lists_with_is_model_type() {
    let chat = AnyModel::Chat(Box::new(test_model("p", "c")));
    let image: AnyModel = serde_json::from_value(json!({
        "type": "image",
        "id": "i",
        "name": "i",
        "api": "test-images",
        "provider": "p",
        "baseUrl": "https://example.test/v1",
        "input": ["text"],
        "output": ["image"],
        "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
    }))
    .unwrap_or_else(|error| panic!("image model: {error}"));
    let mixed = [
        chat,
        AnyModel::Chat(Box::new(test_model("p", "typed"))),
        image,
    ];
    assert_eq!(
        mixed
            .iter()
            .filter(|m| is_model_type(m, ModelType::Chat))
            .count(),
        2
    );
    assert_eq!(
        mixed
            .iter()
            .filter(|m| is_model_type(m, ModelType::Image))
            .count(),
        1
    );
    assert_eq!(
        mixed
            .iter()
            .filter(|m| is_model_type(m, ModelType::Classifier))
            .count(),
        0
    );
}

#[tokio::test]
async fn lightweight_models_entry_runs_a_faux_completion() {
    let models = models();
    assert_eq!(models.get_models(None), Vec::new());
    assert!(matches!(models.get_providers().as_slice(), []));
    let faux = faux_provider(FauxOptions::default())
        .unwrap_or_else(|error| panic!("faux provider: {error}"));
    models.set_provider(Arc::clone(&faux.provider));
    faux.set_responses(vec![FauxResponseStep::Message(Box::new(
        faux_assistant_message(vec![faux_text("OK")]),
    ))]);
    let response = models
        .complete_simple(faux.get_model(), &context(), None)
        .await;
    assert_eq!(response.stop_reason, StopReason::Stop);
    assert_eq!(response.content, vec![faux_text("OK")]);
}

#[tokio::test]
async fn merges_resolved_auth_into_stream_options_explicit_options_win() {
    let models = models();
    let mut provider = TestProvider::new("p1");
    provider.auth = ProviderAuth {
        api_key: Some(ApiKeyAuth::custom(
            "Test",
            Arc::new(|_, _| {
                Ok(Some(AuthResult {
                    auth: ModelAuth {
                        api_key: Some(secrecy::SecretString::from(String::from("resolved-key"))),
                        headers: Some(
                            json!({"Authorization": "Bearer resolved-key", "x-a": "auth", "x-b": "auth"})
                                .as_object()
                                .cloned()
                                .unwrap_or_default(),
                        ),
                        base_url: Some(String::from("https://auth.test/v1")),
                    },
                    env: None,
                    source: None,
                }))
            }),
        )),
        oauth: None,
    };
    let calls = Arc::clone(&provider.calls);
    models.set_provider(Arc::new(provider));
    let model = test_model("p1", "model-a");
    let mut options = SimpleStreamOptions::default();
    options.base.api_key = Some(secrecy::SecretString::from(String::from("explicit-key")));
    options.base.headers = Some(
        json!({"authorization": "Explicit token", "x-b": "explicit"})
            .as_object()
            .cloned()
            .unwrap_or_default(),
    );
    let first_result = models
        .complete_simple(&model, &context(), Some(options))
        .await;
    assert_eq!(first_result.stop_reason, StopReason::Stop);
    let (applied_base_url, sent_api_key, sent_headers) = {
        let recorded = calls
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        (
            recorded[0].0.base_url.clone(),
            recorded[0].1.api_key.clone(),
            recorded[0].1.headers.clone(),
        )
    };
    assert_eq!(
        sent_api_key.as_ref().map(ExposeSecret::expose_secret),
        Some("explicit-key")
    );
    let headers = sent_headers.as_ref().expect("headers");
    assert_eq!(
        headers.get("authorization").and_then(Value::as_str),
        Some("Explicit token")
    );
    assert_eq!(headers.get("x-a").and_then(Value::as_str), Some("auth"));
    assert_eq!(headers.get("x-b").and_then(Value::as_str), Some("explicit"));
    assert_eq!(applied_base_url, "https://auth.test/v1");
    let second_result = models.complete_simple(&model, &context(), None).await;
    assert_eq!(second_result.stop_reason, StopReason::Stop);
    let recorded = calls
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    assert_eq!(
        recorded[1]
            .1
            .api_key
            .as_ref()
            .map(ExposeSecret::expose_secret),
        Some("resolved-key")
    );
}

#[tokio::test]
async fn uses_explicit_request_api_key_and_env_during_provider_auth_resolution() {
    let models = models();
    let mut provider = TestProvider::new("p1");
    provider.auth = ProviderAuth {
        api_key: Some(ApiKeyAuth::custom(
            "Scoped",
            Arc::new(|credential, env| {
                let account = credential
                    .and_then(|c| c.env.as_ref())
                    .and_then(|e| e.get("ACCOUNT_ID"))
                    .and_then(Value::as_str)
                    .map(str::to_owned)
                    .or_else(|| env.get("ACCOUNT_ID"));
                let key = credential
                    .and_then(|c| c.key.as_ref())
                    .map(|k| ExposeSecret::expose_secret(k).to_owned());
                Ok(match (key, account) {
                    (Some(key), Some(account)) => Some(AuthResult {
                        auth: ModelAuth {
                            api_key: Some(secrecy::SecretString::from(key)),
                            base_url: Some(format!("https://example.test/{account}")),
                            ..ModelAuth::default()
                        },
                        env: Some(
                            json!({"ACCOUNT_ID": account})
                                .as_object()
                                .cloned()
                                .unwrap_or_default(),
                        ),
                        source: None,
                    }),
                    _ => None,
                })
            }),
        )),
        oauth: None,
    };
    let calls = Arc::clone(&provider.calls);
    models.set_provider(Arc::new(provider));
    let model = test_model("p1", "model-a");
    let mut options = SimpleStreamOptions::default();
    options.base.api_key = Some(secrecy::SecretString::from(String::from("explicit-key")));
    options.base.env = Some(
        json!({"ACCOUNT_ID": "acct"})
            .as_object()
            .cloned()
            .unwrap_or_default(),
    );
    models
        .complete_simple(&model, &context(), Some(options))
        .await;
    let recorded = calls
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    assert_eq!(recorded[0].0.base_url, "https://example.test/acct");
    assert_eq!(
        recorded[0]
            .1
            .api_key
            .as_ref()
            .map(ExposeSecret::expose_secret),
        Some("explicit-key")
    );
    let env = recorded[0].1.env.as_ref().expect("env");
    assert_eq!(env.get("ACCOUNT_ID").and_then(Value::as_str), Some("acct"));
    drop(recorded);
}

/// A `ModelsStore` backed by a shared `Option<ModelsStoreEntry>` cell.
struct CellStore {
    entry: Arc<Mutex<Option<ModelsStoreEntry>>>,
}

impl ModelsStore for CellStore {
    fn read<'a>(
        &'a self,
        _id: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<Option<ModelsStoreEntry>, ModelsError>> + Send + 'a>>
    {
        let entry = Arc::clone(&self.entry);
        Box::pin(async move {
            Ok(entry
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .clone())
        })
    }
    fn write<'a>(
        &'a self,
        _id: &'a str,
        entry: &'a ModelsStoreEntry,
    ) -> Pin<Box<dyn Future<Output = Result<(), ModelsError>> + Send + 'a>> {
        let cell = Arc::clone(&self.entry);
        let entry = entry.clone();
        Box::pin(async move {
            *cell
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(entry);
            Ok(())
        })
    }
    fn delete<'a>(
        &'a self,
        _id: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<(), ModelsError>> + Send + 'a>> {
        let cell = Arc::clone(&self.entry);
        Box::pin(async move {
            *cell
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner) = None;
            Ok(())
        })
    }
}

#[tokio::test]
async fn restores_cached_models_before_waiting_for_network_auth() {
    let store: Arc<dyn ModelsStore> = Arc::new(InMemoryModelsStore::new());
    ModelsStore::write(
        store.as_ref(),
        "dynamic",
        &ModelsStoreEntry {
            models: vec![AnyModel::Chat(Box::new(test_model("dynamic", "cached")))],
            checked_at: 0.0,
        },
    )
    .await
    .unwrap_or_else(|error| panic!("seed store: {error}"));
    let (auth_started_tx, auth_started) = tokio::sync::oneshot::channel::<()>();
    let (finish_auth_tx, _finish_auth) = tokio::sync::oneshot::channel::<()>();
    let models = create_models(CreateModelsOptions {
        models_store: Some(Arc::clone(&store)),
        ..CreateModelsOptions::default()
    });
    let provider_models = Arc::new(Mutex::new(Vec::<Model>::new()));
    let mut provider = TestProvider::new("dynamic");
    provider.models = Arc::clone(&provider_models);
    provider.auth = ProviderAuth {
        api_key: Some(ApiKeyAuth::custom(
            "Blocked auth",
            Arc::new({
                move |_, _| {
                    Ok(Some(AuthResult {
                        auth: ModelAuth {
                            api_key: Some(secrecy::SecretString::from(String::from("key"))),
                            ..ModelAuth::default()
                        },
                        env: None,
                        source: None,
                    }))
                }
            }),
        )),
        oauth: None,
    };
    let auth_started2 = Arc::new(Mutex::new(Some(auth_started_tx)));
    provider.refresh = Some(Arc::new(move |context| {
        let provider_models = Arc::clone(&provider_models);
        let auth_started2 = Arc::clone(&auth_started2);
        Box::pin(async move {
            if context.allow_network {
                let mark = auth_started2
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .take();
                if let Some(tx) = mark {
                    let _sent = tx.send(());
                }
                // Blocked like upstream's `blockedAuth` — the abort
                // must end the wait.
                tokio::time::sleep(std::time::Duration::from_secs(30)).await;
                return Ok(());
            }
            // `createProvider`'s `refreshModels` republishes the
            // stored catalog during the offline phase.
            if let Some(stored) = &context.stored {
                let cached: Vec<Model> = stored
                    .models
                    .iter()
                    .filter_map(|m| match m {
                        AnyModel::Chat(m) => Some((**m).clone()),
                        _ => None,
                    })
                    .collect();
                (context.publish)(ModelsPublication {
                    persist: PersistAction::Unchanged,
                    update: Some(Box::new(move || {
                        *provider_models
                            .lock()
                            .unwrap_or_else(std::sync::PoisonError::into_inner) = cached;
                    })),
                })
                .await;
            }
            Ok(())
        })
    }));
    models.set_provider(Arc::new(provider));
    let controller = tokio_util::sync::CancellationToken::new();
    let models2 = Arc::clone(&models);
    let controller2 = controller.clone();
    let pending = tokio::spawn(async move {
        models2
            .refresh(ModelsRefreshOptions {
                providers: Some(vec![String::from("dynamic")]),
                signal: Some(controller2),
                ..ModelsRefreshOptions::default()
            })
            .await
    });
    auth_started
        .await
        .unwrap_or_else(|e| panic!("auth started: {e}"));
    // The stored catalog is visible before auth/network finishes.
    assert!(models.get_model("dynamic", "cached").is_some());
    controller.cancel();
    let result = pending.await.unwrap_or_else(|e| panic!("join: {e}"));
    assert!(result.aborted);
    let _sent = finish_auth_tx.send(());
}

#[tokio::test]
async fn lets_providers_choose_persistent_deletion_and_ephemeral_publication() {
    let entry = Arc::new(Mutex::new(Some(ModelsStoreEntry {
        models: vec![AnyModel::Chat(Box::new(test_model("dynamic", "stored")))],
        checked_at: 0.0,
    })));
    let store: Arc<dyn ModelsStore> = Arc::new(CellStore {
        entry: Arc::clone(&entry),
    });
    let state = Arc::new(Mutex::new(String::from("initial")));
    let models = create_models(CreateModelsOptions {
        models_store: Some(Arc::clone(&store)),
        ..CreateModelsOptions::default()
    });
    let mut provider = TestProvider::new("dynamic");
    let state2 = Arc::clone(&state);
    let entry2 = Arc::clone(&entry);
    provider.refresh = Some(Arc::new(move |context| {
        let shared_state = Arc::clone(&state2);
        let shared_entry = Arc::clone(&entry2);
        Box::pin(async move {
            assert_eq!(
                context
                    .stored
                    .as_ref()
                    .and_then(|e| e.models.first())
                    .map(|m| match m {
                        AnyModel::Chat(m) => m.id.clone(),
                        _ => String::new(),
                    })
                    .as_deref(),
                Some("stored")
            );
            let state3 = Arc::clone(&shared_state);
            let entry3 = Arc::clone(&shared_entry);
            (context.publish)(ModelsPublication {
                persist: PersistAction::Delete,
                update: Some(Box::new(move || {
                    assert!(
                        entry3
                            .lock()
                            .unwrap_or_else(std::sync::PoisonError::into_inner)
                            .is_none()
                    );
                    *state3
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner) =
                        String::from("deleted");
                })),
            })
            .await;
            let state4 = Arc::clone(&shared_state);
            (context.publish)(ModelsPublication {
                persist: PersistAction::Unchanged,
                update: Some(Box::new(move || {
                    *state4
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner) =
                        String::from("ephemeral");
                })),
            })
            .await;
            Ok(())
        })
    }));
    models.set_provider(Arc::new(provider));
    let result = models
        .refresh(ModelsRefreshOptions {
            allow_network: Some(false),
            ..ModelsRefreshOptions::default()
        })
        .await;
    assert!(result.errors.is_empty(), "{:?}", result.errors);
    assert!(
        entry
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .is_none()
    );
    assert_eq!(
        state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .as_str(),
        "ephemeral"
    );
}

#[tokio::test]
async fn rejects_late_publication_from_a_superseded_provider() {
    let store: Arc<dyn ModelsStore> = Arc::new(InMemoryModelsStore::new());
    let state = Arc::new(Mutex::new(String::from("initial")));
    let calls = Arc::new(Mutex::new(0usize));
    let (first_started_tx, first_started) = tokio::sync::oneshot::channel::<()>();
    let (finish_first_tx, finish_first) = tokio::sync::oneshot::channel::<()>();
    let models = create_models(CreateModelsOptions {
        models_store: Some(Arc::clone(&store)),
        ..CreateModelsOptions::default()
    });
    let mut provider = TestProvider::new("dynamic");
    {
        let state = Arc::clone(&state);
        let calls = Arc::clone(&calls);
        let started = Arc::new(Mutex::new(Some(first_started_tx)));
        let _finish_first = Mutex::new(Some(finish_first));
        provider.refresh = Some(Arc::new(move |context| {
            let state = Arc::clone(&state);
            let calls = Arc::clone(&calls);
            let first_mark = Arc::clone(&started);
            Box::pin(async move {
                if !context.allow_network {
                    return Ok(());
                }
                *calls
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner) += 1;
                let current = *calls
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                let maybe_tx = first_mark
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .take();
                if current == 1
                    && let Some(tx) = maybe_tx
                {
                    let _sent = tx.send(());
                    // Block the first refresh until the second
                    // supersedes it (the select! on the signal lets
                    // `refresh` return anyway).
                    tokio::time::sleep(std::time::Duration::from_secs(30)).await;
                }
                let value = format!("generation-{current}");
                let state = Arc::clone(&state);
                (context.publish)(ModelsPublication {
                    persist: PersistAction::Write(ModelsStoreEntry {
                        models: vec![AnyModel::Chat(Box::new(test_model("dynamic", &value)))],
                        checked_at: 0.0,
                    }),
                    update: Some(Box::new(move || {
                        *state
                            .lock()
                            .unwrap_or_else(std::sync::PoisonError::into_inner) = value;
                    })),
                })
                .await;
                Ok(())
            })
        }));
    }
    models.set_provider(Arc::new(provider));
    let m2 = Arc::clone(&models);
    let first = tokio::spawn(async move {
        m2.refresh(ModelsRefreshOptions {
            providers: Some(vec![String::from("dynamic")]),
            ..ModelsRefreshOptions::default()
        })
        .await
    });
    first_started.await.unwrap_or_else(|e| panic!("first: {e}"));
    let second = models
        .refresh(ModelsRefreshOptions {
            providers: Some(vec![String::from("dynamic")]),
            ..ModelsRefreshOptions::default()
        })
        .await;
    let _joined = first.await;
    let _sent = finish_first_tx.send(());
    assert!(second.errors.is_empty());
    assert_eq!(
        state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .as_str(),
        "generation-2"
    );
    let stored = ModelsStore::read(store.as_ref(), "dynamic")
        .await
        .unwrap_or_else(|e| panic!("store: {e}"));
    let id = stored.and_then(|e| {
        e.models.first().map(|m| match m {
            AnyModel::Chat(m) => m.id.clone(),
            _ => String::new(),
        })
    });
    assert_eq!(id.as_deref(), Some("generation-2"));
}

#[tokio::test]
async fn stops_waiting_on_abort_when_a_provider_ignores_its_signal() {
    let controller = tokio_util::sync::CancellationToken::new();
    let (started_tx, started) = tokio::sync::oneshot::channel::<()>();
    let models = models();
    let mut provider = TestProvider::new("dynamic");
    {
        let mark = Mutex::new(Some(started_tx));
        let calls = Arc::new(Mutex::new(0usize));
        provider.refresh = Some(Arc::new(move |_context| {
            let maybe_started = mark
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .take();
            let calls = Arc::clone(&calls);
            Box::pin(async move {
                *calls
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner) += 1;
                if *calls
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    != 1
                {
                    return Ok(());
                }
                if let Some(tx) = maybe_started {
                    let _sent = tx.send(());
                }
                // The provider never honours the signal.
                tokio::time::sleep(std::time::Duration::from_secs(30)).await;
                Ok(())
            })
        }));
    }
    models.set_provider(Arc::new(provider));
    let m2 = Arc::clone(&models);
    let c2 = controller.clone();
    let pending = tokio::spawn(async move {
        m2.refresh(ModelsRefreshOptions {
            signal: Some(c2),
            ..ModelsRefreshOptions::default()
        })
        .await
    });
    started.await.unwrap_or_else(|e| panic!("started: {e}"));
    controller.cancel();
    let result = pending.await.unwrap_or_else(|e| panic!("join: {e}"));
    assert!(result.aborted);
    assert!(result.errors.is_empty());
}

#[tokio::test]
async fn passes_effective_api_key_credentials_and_refresh_options() {
    let effective = Arc::new(Mutex::new(None::<Credential>));
    let force = Arc::new(Mutex::new(false));
    let unconfigured = Arc::new(Mutex::new(0usize));
    let models = models();
    let mut configured = TestProvider::new("configured");
    configured.auth = ProviderAuth {
        api_key: Some(env_key_auth(Some("ambient-key"))),
        oauth: None,
    };
    {
        let effective = Arc::clone(&effective);
        let force = Arc::clone(&force);
        configured.refresh = Some(Arc::new(move |context| {
            let effective = Arc::clone(&effective);
            let force = Arc::clone(&force);
            Box::pin(async move {
                if !context.allow_network {
                    return Ok(());
                }
                *effective
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner) =
                    context.credential.clone();
                *force
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner) =
                    context.force.unwrap_or(false);
                Ok(())
            })
        }));
    }
    models.set_provider(Arc::new(configured));
    let mut missing = TestProvider::new("unconfigured");
    missing.auth = ProviderAuth {
        api_key: Some(env_key_auth(None)),
        oauth: None,
    };
    {
        let unconfigured = Arc::clone(&unconfigured);
        missing.refresh = Some(Arc::new(move |context| {
            let unconfigured = Arc::clone(&unconfigured);
            Box::pin(async move {
                if context.allow_network {
                    *unconfigured
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner) += 1;
                }
                Ok(())
            })
        }));
    }
    models.set_provider(Arc::new(missing));
    models
        .refresh(ModelsRefreshOptions {
            force: Some(true),
            ..ModelsRefreshOptions::default()
        })
        .await;
    let credential = effective
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clone();
    match credential {
        Some(Credential::ApiKey(c)) => {
            assert_eq!(
                c.key
                    .as_ref()
                    .map(|k| ExposeSecret::expose_secret(k).to_owned())
                    .as_deref(),
                Some("ambient-key")
            );
            assert!(c.env.is_none());
        }
        other => panic!("credential: {other:?}"),
    }
    assert!(
        *force
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    );
    assert_eq!(
        *unconfigured
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner),
        0
    );
}

#[tokio::test]
async fn resolves_auth_stored_credential_owns_the_provider() {
    let credentials: Arc<dyn CredentialStore> = Arc::new(InMemoryCredentialStore::new());
    let models = create_models(CreateModelsOptions {
        credentials: Some(Arc::clone(&credentials)),
        ..CreateModelsOptions::default()
    });
    let mut provider = TestProvider::new("p1");
    provider.auth = ProviderAuth {
        api_key: Some(env_key_auth(Some("env-key"))),
        oauth: None,
    };
    models.set_provider(Arc::new(provider));
    let model = test_model("p1", "model-a");
    let any = AnyModel::Chat(Box::new(model));
    let resolution = models
        .get_auth(
            ProviderOrModel::Model(&any),
            &AuthResolutionOverrides::default(),
        )
        .await
        .unwrap_or_else(|e| panic!("get_auth: {e}"));
    assert_eq!(
        resolution
            .and_then(|r| r.auth.api_key)
            .as_ref()
            .map(ExposeSecret::expose_secret),
        Some("env-key")
    );
    let overrides = AuthResolutionOverrides {
        api_key: Some(secrecy::SecretString::from(String::from("explicit-key"))),
        ..AuthResolutionOverrides::default()
    };
    let explicit_resolution = models
        .get_auth(ProviderOrModel::Model(&any), &overrides)
        .await
        .unwrap_or_else(|e| panic!("get_auth: {e}"));
    assert_eq!(
        explicit_resolution
            .and_then(|r| r.auth.api_key)
            .as_ref()
            .map(ExposeSecret::expose_secret),
        Some("explicit-key")
    );
    credentials
        .modify(
            "p1",
            Box::new(|_| {
                Box::pin(async {
                    Some(Credential::ApiKey(ApiKeyCredential {
                        key: Some(secrecy::SecretString::from(String::from("stored-key"))),
                        env: None,
                    }))
                })
            }),
        )
        .await
        .unwrap_or_else(|e| panic!("modify: {e}"));
    let stored_resolution = models
        .get_auth(
            ProviderOrModel::Id("p1"),
            &AuthResolutionOverrides::default(),
        )
        .await
        .unwrap_or_else(|e| panic!("get_auth: {e}"));
    let stored_resolution = stored_resolution.unwrap_or_else(|| panic!("resolution"));
    assert_eq!(
        stored_resolution
            .auth
            .api_key
            .as_ref()
            .map(ExposeSecret::expose_secret),
        Some("stored-key")
    );
    assert_eq!(stored_resolution.source.as_deref(), Some("stored"));
}

#[tokio::test]
async fn checks_provider_auth_without_oauth_and_filters_available_models() {
    let credentials: Arc<dyn CredentialStore> = Arc::new(InMemoryCredentialStore::new());
    let models = create_models(CreateModelsOptions {
        credentials: Some(Arc::clone(&credentials)),
        ..CreateModelsOptions::default()
    });
    let mut ambient = TestProvider::new("ambient");
    ambient.auth = ProviderAuth {
        api_key: Some(env_key_auth(Some("env-key"))),
        oauth: None,
    };
    models.set_provider(Arc::new(ambient));
    let mut missing = TestProvider::new("missing");
    missing.auth = ProviderAuth {
        api_key: Some(env_key_auth(None)),
        oauth: None,
    };
    models.set_provider(Arc::new(missing));
    let mut oauth = TestProvider::new("oauth");
    oauth.auth = ProviderAuth {
        api_key: None,
        oauth: Some(OAuthAuth {
            name: String::from("Test OAuth"),
            to_auth: Arc::new(|credential| {
                let access = ExposeSecret::expose_secret(&credential.access).to_owned();
                Box::pin(async move {
                    ModelAuth {
                        api_key: Some(secrecy::SecretString::from(access)),
                        ..ModelAuth::default()
                    }
                })
            }),
        }),
    };
    models.set_provider(Arc::new(oauth));
    credentials
        .modify(
            "oauth",
            Box::new(|_| {
                Box::pin(async {
                    Some(Credential::OAuth(OAuthCredential {
                        access: secrecy::SecretString::from(String::from("expired")),
                        refresh: secrecy::SecretString::from(String::from("refresh")),
                        expires: 0.0,
                        extra: Map::new(),
                    }))
                })
            }),
        )
        .await
        .unwrap_or_else(|e| panic!("modify: {e}"));
    let ambient_check = models
        .check_auth("ambient")
        .await
        .unwrap_or_else(|e| panic!("check: {e}"));
    assert_eq!(
        ambient_check.map(|c| (c.source, c.type_)),
        Some((Some(String::from("env")), "api_key"))
    );
    assert!(
        models
            .check_auth("missing")
            .await
            .unwrap_or_else(|e| panic!("check: {e}"))
            .is_none()
    );
    let oauth_check = models
        .check_auth("oauth")
        .await
        .unwrap_or_else(|e| panic!("check: {e}"));
    assert_eq!(
        oauth_check.map(|c| (c.source, c.type_)),
        Some((Some(String::from("OAuth")), "oauth"))
    );
    let available = models
        .get_available(None)
        .await
        .unwrap_or_else(|e| panic!("available: {e}"));
    assert_eq!(
        available
            .iter()
            .map(|m| m.provider.as_str().to_owned())
            .collect::<Vec<_>>(),
        ["ambient", "oauth"]
    );
    assert_eq!(
        models
            .get_available(Some("ambient"))
            .await
            .unwrap_or_else(|e| panic!("available: {e}"))
            .iter()
            .map(|m| m.provider.as_str().to_owned())
            .collect::<Vec<_>>(),
        ["ambient"]
    );
}

#[tokio::test]
async fn runs_provider_login_and_logout_through_the_credential_store() {
    let credentials: Arc<dyn CredentialStore> = Arc::new(InMemoryCredentialStore::new());
    let models = create_models(CreateModelsOptions {
        credentials: Some(Arc::clone(&credentials)),
        ..CreateModelsOptions::default()
    });
    let mut provider = TestProvider::new("p1");
    provider.auth = ProviderAuth {
        api_key: Some(env_key_auth(None)),
        oauth: Some(OAuthAuth {
            name: String::from("Test OAuth"),
            to_auth: Arc::new(|credential| {
                let access = ExposeSecret::expose_secret(&credential.access).to_owned();
                Box::pin(async move {
                    ModelAuth {
                        api_key: Some(secrecy::SecretString::from(access)),
                        ..ModelAuth::default()
                    }
                })
            }),
        }),
    };
    models.set_provider(Arc::new(provider));
    let credential = Credential::ApiKey(ApiKeyCredential {
        key: Some(secrecy::SecretString::from(String::from("logged-in"))),
        env: None,
    });
    models
        .login("p1", credential.clone())
        .await
        .unwrap_or_else(|e| panic!("login: {e}"));
    let stored = credentials
        .read("p1")
        .await
        .unwrap_or_else(|e| panic!("read: {e}"));
    assert_eq!(stored, Some(credential));
    models
        .logout("p1")
        .await
        .unwrap_or_else(|e| panic!("logout: {e}"));
    assert!(
        credentials
            .read("p1")
            .await
            .unwrap_or_else(|e| panic!("read: {e}"))
            .is_none()
    );
}

#[tokio::test]
async fn a_stored_credential_without_a_matching_handler_blocks_ambient_fallback() {
    let credentials: Arc<dyn CredentialStore> = Arc::new(InMemoryCredentialStore::new());
    let models = create_models(CreateModelsOptions {
        credentials: Some(Arc::clone(&credentials)),
        ..CreateModelsOptions::default()
    });
    let mut provider = TestProvider::new("p1");
    provider.auth = ProviderAuth {
        api_key: Some(env_key_auth(Some("env-key"))),
        oauth: None,
    };
    models.set_provider(Arc::new(provider));
    credentials
        .modify(
            "p1",
            Box::new(|_| {
                Box::pin(async {
                    Some(Credential::OAuth(OAuthCredential {
                        access: secrecy::SecretString::from(String::from("a")),
                        refresh: secrecy::SecretString::from(String::from("r")),
                        expires: 0.0,
                        extra: Map::new(),
                    }))
                })
            }),
        )
        .await
        .unwrap_or_else(|e| panic!("modify: {e}"));
    let resolution = models
        .get_auth(
            ProviderOrModel::Id("p1"),
            &AuthResolutionOverrides::default(),
        )
        .await;
    // Upstream resolves `undefined`; the Rust overlay surfaces the M5
    // OAuth gap as `Err` — either way the ambient env key is not used.
    match resolution {
        Ok(r) => assert!(r.is_none()),
        Err(e) => assert_eq!(e.code, ModelsErrorCode::Oauth),
    }
}

/// A `CredentialStore` whose `read` always fails.
struct FailingStore;

impl CredentialStore for FailingStore {
    fn read<'a>(
        &'a self,
        _id: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<Option<Credential>, AuthError>> + Send + 'a>> {
        Box::pin(async { Err(AuthError::models(ModelsErrorCode::Auth, "disk on fire")) })
    }
    fn list(
        &self,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<CredentialInfo>, AuthError>> + Send + '_>> {
        Box::pin(async { Ok(Vec::new()) })
    }
    fn modify<'a>(
        &'a self,
        _id: &'a str,
        _f: ovo_ai::auth::store::ModifyFn<'a>,
    ) -> Pin<Box<dyn Future<Output = Result<Option<Credential>, AuthError>> + Send + 'a>> {
        Box::pin(async { Err(AuthError::models(ModelsErrorCode::Auth, "disk on fire")) })
    }
    fn delete<'a>(
        &'a self,
        _id: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<(), AuthError>> + Send + 'a>> {
        Box::pin(async { Ok(()) })
    }
}

#[tokio::test]
async fn wraps_credential_store_failures_in_models_error() {
    let models = create_models(CreateModelsOptions {
        credentials: Some(Arc::new(FailingStore)),
        ..CreateModelsOptions::default()
    });
    let mut provider = TestProvider::new("p1");
    provider.auth = ProviderAuth {
        api_key: Some(env_key_auth(Some("env-key"))),
        oauth: None,
    };
    models.set_provider(Arc::new(provider));
    let error = models
        .get_auth(
            ProviderOrModel::Id("p1"),
            &AuthResolutionOverrides::default(),
        )
        .await
        .err()
        .unwrap_or_else(|| panic!("expected error"));
    assert_eq!(error.code, ModelsErrorCode::Auth);
}

#[tokio::test]
async fn wraps_api_key_auth_failures_in_models_error() {
    let models = models();
    let mut provider = TestProvider::new("p1");
    provider.auth = ProviderAuth {
        api_key: Some(ApiKeyAuth::custom(
            "Failing",
            Arc::new(|_, _| Err(String::from("nope"))),
        )),
        oauth: None,
    };
    models.set_provider(Arc::new(provider));
    let error = models
        .get_auth(
            ProviderOrModel::Id("p1"),
            &AuthResolutionOverrides::default(),
        )
        .await
        .err()
        .unwrap_or_else(|| panic!("expected error"));
    assert_eq!(error.code, ModelsErrorCode::Auth);
}

/// parity: AI-FAUX
// Folded from `models_faux.rs` — ovo-specific registry/stream
// coverage that has no upstream `it` of its own.
fn seeded_models() -> (Arc<Models>, ovo_ai::faux::FauxProvider) {
    let models = create_models(CreateModelsOptions::default());
    let faux =
        faux_provider(FauxOptions::default()).unwrap_or_else(|e| panic!("faux provider: {e}"));
    models.set_provider(Arc::clone(&faux.provider));
    models.set_provider(Arc::clone(&faux.provider));
    (models, faux)
}

fn text_context(text: &str) -> Context {
    serde_json::from_value(json!({
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
    assert!(has_api(&AnyModel::Chat(Box::new(model.clone())), "faux"));
    assert!(models_are_equal(
        &AnyModel::Chat(Box::new(model)),
        &AnyModel::Chat(Box::new(faux.models[0].clone())),
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
    let mut base = Map::new();
    base.insert(String::from("Authorization"), json!("a"));
    let mut override_ = Map::new();
    override_.insert(String::from("authorization"), json!("b"));
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
    let faux =
        faux_provider(FauxOptions::default()).unwrap_or_else(|e| panic!("faux provider: {e}"));
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
    let calls = Arc::new(Mutex::new(Vec::<u16>::new()));
    let faux =
        faux_provider(FauxOptions::default()).unwrap_or_else(|e| panic!("faux provider: {e}"));
    let models = create_models(CreateModelsOptions::default());
    models.set_provider(Arc::clone(&faux.provider));
    faux.set_responses(vec![FauxResponseStep::Message(Box::new(
        faux_assistant_message(vec![faux_text("hi")]),
    ))]);
    let model = models
        .get_model("faux", "faux-1")
        .unwrap_or_else(|| panic!("model"));
    let calls2 = Arc::clone(&calls);
    let options = StreamOptions {
        on_response: Some(Arc::new(move |meta, _model| {
            let recorded = Arc::clone(&calls2);
            let status = meta.status;
            Box::pin(async move {
                recorded
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .push(status);
            })
        })),
        ..StreamOptions::default()
    };
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
    let faux =
        faux_provider(FauxOptions::default()).unwrap_or_else(|e| panic!("faux provider: {e}"));
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
    let faux =
        faux_provider(FauxOptions::default()).unwrap_or_else(|e| panic!("faux provider: {e}"));
    let mut env = std::collections::HashMap::new();
    env.insert(String::from("FAUX_API_KEY"), String::from("k-1"));
    let models = create_models(CreateModelsOptions {
        env: Some(Arc::new(ovo_ai::auth::env::MapEnv::new(env))),
        ..CreateModelsOptions::default()
    });
    models.set_provider(Arc::clone(&faux.provider));
    let auth = models
        .get_auth(
            ProviderOrModel::Id("faux"),
            &AuthResolutionOverrides::default(),
        )
        .await;
    assert!(auth.is_ok(), "{auth:?}");
}
