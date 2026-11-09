//! `Provider`/`ChatApi`/`ProviderSpec`/`create_provider` — `models.ts`.
//!
//! A provider is the runtime unit: metadata, auth, model lists, and
//! request behavior. `create_provider` assembles the declarative kind
//! every built-in provider uses; extension providers implement
//! [`Provider`] directly.

use std::collections::BTreeMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};
use tokio_util::sync::CancellationToken;

use crate::auth::resolve::{ApiKeyAuth, OAuthAuth};
use crate::auth::types::Credential;
use crate::models_error::ModelsError;
use crate::options::{
    ClassifierOptions, DeferredCancelOptions, DeferredFetchOptions, ImagesOptions,
    SimpleStreamOptions, StreamOptions,
};
use crate::stream::AssistantStream;
use crate::transcript::TranscriptContext;
use crate::types::classifier::{ClassifierContext, ClassifierResult};
use crate::types::deferred::DeferredHandle;
use crate::types::images::{AssistantImages, ImagesContext};
use crate::types::model::{AnyModel, Model};

/// `BoxFuture` per ADR-009 (object-safe async APIs).
pub type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// `ModelType`: `"chat"` | `"image"` | `"classifier"` — the `type` tag
/// upstream reads via `getModelType` (absent means chat).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ModelType {
    /// Chat model.
    Chat,
    /// Image-generation model.
    Image,
    /// Structured classifier model.
    Classifier,
}

/// `getModelType(model)` — absent `type` means chat.
#[must_use]
pub const fn model_type(model: &AnyModel) -> ModelType {
    match model {
        AnyModel::Chat(_) => ModelType::Chat,
        AnyModel::Image(_) => ModelType::Image,
        AnyModel::Classifier(_) => ModelType::Classifier,
    }
}

/// `isModelType`.
#[must_use]
pub fn is_model_type(model: &AnyModel, kind: ModelType) -> bool {
    model_type(model) == kind
}

/// `assertChatModel`: chat models only.
///
/// # Errors
/// [`crate::models_error::ModelsErrorCode::Provider`].
pub fn assert_chat_model(model: &AnyModel) -> Result<(), ModelsError> {
    if is_model_type(model, ModelType::Chat) {
        return Ok(());
    }
    let (provider, id) = model_ids(model);
    Err(ModelsError::new(
        crate::models_error::ModelsErrorCode::Provider,
        format!("Model {provider}/{id} is not a chat model"),
    ))
}

/// `assertImageModel`.
///
/// # Errors
/// [`crate::models_error::ModelsErrorCode::Provider`].
pub fn assert_image_model(model: &AnyModel) -> Result<(), ModelsError> {
    if is_model_type(model, ModelType::Image) {
        return Ok(());
    }
    let (provider, id) = model_ids(model);
    Err(ModelsError::new(
        crate::models_error::ModelsErrorCode::Provider,
        format!("Model {provider}/{id} is not an image model"),
    ))
}

/// `assertClassifierModel`.
///
/// # Errors
/// [`crate::models_error::ModelsErrorCode::Provider`].
pub fn assert_classifier_model(model: &AnyModel) -> Result<(), ModelsError> {
    if is_model_type(model, ModelType::Classifier) {
        return Ok(());
    }
    let (provider, id) = model_ids(model);
    Err(ModelsError::new(
        crate::models_error::ModelsErrorCode::Provider,
        format!("Model {provider}/{id} is not a classifier model"),
    ))
}

/// The `provider`/`id` pair shared by every model variant.
#[must_use]
pub fn model_ids(model: &AnyModel) -> (&str, &str) {
    match model {
        AnyModel::Chat(m) => (m.provider.as_str(), m.id.as_str()),
        AnyModel::Image(m) => (m.provider.as_str(), m.id.as_str()),
        AnyModel::Classifier(m) => (m.provider.as_str(), m.id.as_str()),
    }
}

/// `ProviderAuth`: `{apiKey?: ApiKeyAuth, oauth?: OAuthAuth}` — at
/// least one must be present.
#[derive(Clone)]
pub struct ProviderAuth {
    /// API-key auth (`envApiKeyAuth` or a custom resolver).
    pub api_key: Option<ApiKeyAuth>,
    /// OAuth auth — stubbed flows are M5; the shape is fixed.
    pub oauth: Option<OAuthAuth>,
}

impl std::fmt::Debug for ProviderAuth {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ProviderAuth")
            .field("api_key", &self.api_key.as_ref().map(|a| &a.name))
            .field("oauth", &self.oauth.as_ref().map(|o| &o.name))
            .finish()
    }
}

/// `Provider` — the concrete runtime unit (pi `Provider` interface).
///
/// Sync accessors mirror `getModels`/`getAllModels` (never fail).
/// Optional upstream methods have defaults that surface the same
/// "not supported" errors the dispatch would produce.
pub trait Provider: Send + Sync + 'static {
    /// `id`.
    fn id(&self) -> &str;
    /// `name`.
    fn name(&self) -> &str;
    /// `baseUrl` (optional upstream).
    fn base_url(&self) -> Option<&str> {
        None
    }
    /// `headers` (optional upstream provider-level headers).
    fn headers(&self) -> Option<&serde_json::Map<String, serde_json::Value>> {
        None
    }
    /// `auth` (required upstream).
    fn auth(&self) -> &ProviderAuth;
    /// `getModels()` — current chat catalog; never fails.
    fn get_models(&self) -> Vec<Model>;
    /// `getAllModels()` — every type; defaults to [`Self::get_models`] when
    /// the provider only has chat models.
    fn get_all_models(&self) -> Vec<AnyModel> {
        self.get_models()
            .into_iter()
            .map(|m| AnyModel::Chat(Box::new(m)))
            .collect()
    }
    /// `refreshModels(context)` — dynamic providers only. Restores
    /// `context.stored`, optionally fetches a newer list, and
    /// publishes through `context.publish`.
    fn refresh_models(
        &self,
        _context: RefreshModelsContext,
    ) -> BoxFuture<'_, Result<(), ModelsError>> {
        Box::pin(async { Ok(()) })
    }
    /// `filterModels` — credential-specific chat availability.
    fn filter_models(&self, models: &[Model], _credential: Option<&Credential>) -> Vec<Model> {
        models.to_vec()
    }
    /// `filterAllModels` — credential-specific availability across
    /// types; defaults match `Models.getAllAvailable`'s fallback.
    fn filter_all_models(
        &self,
        models: &[AnyModel],
        credential: Option<&Credential>,
    ) -> Vec<AnyModel> {
        let chat_ids: std::collections::HashSet<String> = self
            .filter_models(&self.get_models(), credential)
            .into_iter()
            .map(|m| m.id)
            .collect();
        models
            .iter()
            .filter(|m| !is_model_type(m, ModelType::Chat) || chat_ids.contains(model_ids(m).1))
            .cloned()
            .collect()
    }
    /// `stream` — normalized transcript → event stream (lazy; auth
    /// and dispatch errors surface as stream errors).
    fn stream(
        &self,
        model: &Model,
        context: TranscriptContext,
        options: StreamOptions,
    ) -> AssistantStream;
    /// `streamSimple` — simple-options variant.
    fn stream_simple(
        &self,
        model: &Model,
        context: TranscriptContext,
        options: SimpleStreamOptions,
    ) -> AssistantStream;
    /// `fetchDeferred` — providers without deferred support return
    /// the error stream upstream produces at dispatch.
    fn fetch_deferred(
        &self,
        model: &Model,
        handle: &DeferredHandle,
        options: DeferredFetchOptions,
    ) -> Option<AssistantStream> {
        let _ = (model, handle, options);
        None
    }
    /// `cancelDeferred`.
    fn cancel_deferred(
        &self,
        model: &Model,
        handle: &DeferredHandle,
        options: DeferredCancelOptions,
    ) -> Option<BoxFuture<'static, Result<(), ModelsError>>> {
        let _ = (model, handle, options);
        None
    }
    /// `generateImages`.
    fn generate_images(
        &self,
        model: &crate::types::model::ImageModel,
        context: ImagesContext,
        options: ImagesOptions,
    ) -> Option<BoxFuture<'static, AssistantImages>> {
        let _ = (model, context, options);
        None
    }
    /// `classify`.
    fn classify(
        &self,
        model: &crate::types::model::ClassifierModel,
        context: ClassifierContext,
        options: ClassifierOptions,
    ) -> Option<BoxFuture<'static, ClassifierResult>> {
        let _ = (model, context, options);
        None
    }
}

/// `ChatApi` (`ProviderStreams`): one wire-protocol implementation.
pub trait ChatApi: Send + Sync + 'static {
    /// `stream(model, context, options)`.
    fn stream(
        &self,
        model: &Model,
        context: TranscriptContext,
        options: StreamOptions,
    ) -> AssistantStream;
    /// `streamSimple(model, context, options)` — defaults to
    /// [`Self::stream`] with the base options (upstream adapters override).
    fn stream_simple(
        &self,
        model: &Model,
        context: TranscriptContext,
        options: SimpleStreamOptions,
    ) -> AssistantStream {
        self.stream(model, context, options.base)
    }
    /// `fetchDeferred` — `None` = not supported.
    fn fetch_deferred(
        &self,
        model: &Model,
        handle: &DeferredHandle,
        options: DeferredFetchOptions,
    ) -> Option<AssistantStream> {
        let _ = (model, handle, options);
        None
    }
    /// `cancelDeferred` — `None` = not supported.
    fn cancel_deferred(
        &self,
        model: &Model,
        handle: &DeferredHandle,
        options: DeferredCancelOptions,
    ) -> Option<BoxFuture<'static, Result<(), ModelsError>>> {
        let _ = (model, handle, options);
        None
    }
}

/// `ProviderImages`: image-generation implementation.
pub trait ImageApiImpl: Send + Sync + 'static {
    /// `generateImages(model, context, options)`.
    fn generate_images(
        &self,
        model: &crate::types::model::ImageModel,
        context: ImagesContext,
        options: ImagesOptions,
    ) -> BoxFuture<'static, AssistantImages>;
}

/// `ProviderClassifier`: classifier implementation.
pub trait ClassifierApiImpl: Send + Sync + 'static {
    /// `classify(model, context, options)`.
    fn classify(
        &self,
        model: &crate::types::model::ClassifierModel,
        context: ClassifierContext,
        options: ClassifierOptions,
    ) -> BoxFuture<'static, ClassifierResult>;
}

/// `api` dispatch: one implementation for every chat model, or a
/// map keyed by `model.api` (upstream `api | Partial<Record<Api>>`).
#[derive(Clone)]
pub enum ApiDispatch {
    /// `api` as a single `ProviderStreams`.
    Single(Arc<dyn ChatApi>),
    /// `api` as a `Partial<Record<Api, ProviderStreams>>` map.
    ByApi(BTreeMap<String, Arc<dyn ChatApi>>),
}

impl std::fmt::Debug for ApiDispatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Single(_) => f.debug_tuple("Single").finish_non_exhaustive(),
            Self::ByApi(map) => f
                .debug_tuple("ByApi")
                .field(&map.keys().collect::<Vec<_>>())
                .finish(),
        }
    }
}

impl ApiDispatch {
    /// `apiFor(model)` — `None` when the map lacks the api entry.
    fn for_model(&self, model: &Model) -> Option<&Arc<dyn ChatApi>> {
        match self {
            Self::Single(api) => Some(api),
            Self::ByApi(map) => map.get(model.api.as_str()),
        }
    }
}

/// `fetchModels` callback (`RefreshModelsContext → fetched list`).
pub type FetchModels =
    Arc<dyn for<'a> Fn(&'a RefreshModelsContext) -> BoxFuture<'a, Vec<AnyModel>> + Send + Sync>;

/// `filterModels`/`filterAllModels` callbacks.
pub type FilterModels =
    Arc<dyn Fn(&[AnyModel], Option<&Credential>) -> Vec<AnyModel> + Send + Sync>;

/// `ProviderSpec` (`CreateProviderOptions`).
pub struct ProviderSpec {
    /// `id`.
    pub id: String,
    /// `name` (default `id`).
    pub name: Option<String>,
    /// `baseUrl`.
    pub base_url: Option<String>,
    /// `headers`.
    pub headers: Option<serde_json::Map<String, serde_json::Value>>,
    /// `auth` (required).
    pub auth: ProviderAuth,
    /// `models` — static baseline of every type (empty for dynamic).
    pub models: Vec<AnyModel>,
    /// `api` — single or per-api chat implementations.
    pub api: Option<ApiDispatch>,
    /// `images` — image-generation implementations keyed by api.
    pub images: BTreeMap<String, Arc<dyn ImageApiImpl>>,
    /// `classifiers` — classifier implementations keyed by api.
    pub classifiers: BTreeMap<String, Arc<dyn ClassifierApiImpl>>,
    /// `fetchModels` — dynamic overlay fetch.
    pub fetch_models: Option<FetchModels>,
    /// `filterModels`/`filterAllModels` — credential-specific
    /// availability over every type.
    pub filter_models: Option<FilterModels>,
    /// `filterAllModels` — override `filter_models`'s default.
    pub filter_all_models: Option<FilterModels>,
}

impl std::fmt::Debug for ProviderSpec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ProviderSpec")
            .field("id", &self.id)
            .finish_non_exhaustive()
    }
}

/// `ModelsStoreEntry`: the persisted provider catalog entry.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelsStoreEntry {
    /// Persisted model list.
    pub models: Vec<AnyModel>,
    /// `checkedAt` (epoch ms).
    pub checked_at: f64,
}

/// `ModelsPublication.persist` tri-state.
#[derive(Clone, Debug)]
pub enum PersistAction {
    /// Leave the stored entry unchanged.
    Unchanged,
    /// Delete the entry.
    Delete,
    /// Write `entry`.
    Write(ModelsStoreEntry),
}

/// `ModelsPublication`: `persist` + `update`.
pub struct ModelsPublication {
    /// `persist` (default `Unchanged`).
    pub persist: PersistAction,
    /// `update` — synchronous in-memory change after persistence.
    pub update: Option<Box<dyn FnOnce() + Send>>,
}

impl std::fmt::Debug for ModelsPublication {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ModelsPublication")
            .field("persist", &self.persist)
            .finish_non_exhaustive()
    }
}

/// `RefreshModelsContext` (pi) — credential, stored snapshot,
/// `publish`, `allowNetwork`, `force`, `signal`, `baseUrl`.
pub struct RefreshModelsContext {
    /// Effective configured credential (OAuth refreshed first).
    pub credential: Option<Credential>,
    /// Immutable stored catalog snapshot.
    pub stored: Option<ModelsStoreEntry>,
    /// `publish(publication)` — generation-checked.
    pub publish: Arc<dyn Fn(ModelsPublication) -> BoxFuture<'static, bool> + Send + Sync>,
    /// `allowNetwork` — `false` during offline init.
    pub allow_network: bool,
    /// `force` — bypass freshness checks.
    pub force: Option<bool>,
    /// `signal` — upstream `AbortSignal`.
    pub signal: CancellationToken,
    /// `baseUrl` of the registered provider.
    pub base_url: Option<String>,
}

impl std::fmt::Debug for RefreshModelsContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RefreshModelsContext")
            .field("allow_network", &self.allow_network)
            .finish_non_exhaustive()
    }
}

/// `SpecProvider`: the declarative provider `create_provider` builds.
pub struct SpecProvider {
    spec: ProviderSpec,
    dynamic: Arc<Mutex<Vec<AnyModel>>>,
}

impl std::fmt::Debug for SpecProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SpecProvider")
            .field("id", &self.spec.id)
            .finish_non_exhaustive()
    }
}

/// `createProvider` — `Err` verbatim when no implementation exists.
///
/// # Errors
/// `"Provider {id}: at least one of \"api\", \"images\", or
/// \"classifiers\" is required."`
pub fn create_provider(spec: ProviderSpec) -> Result<SpecProvider, ModelsError> {
    let has_streams = match &spec.api {
        Some(ApiDispatch::Single(_)) => true,
        Some(ApiDispatch::ByApi(map)) => !map.is_empty(),
        None => false,
    };
    if !has_streams && spec.images.is_empty() && spec.classifiers.is_empty() {
        return Err(ModelsError::new(
            crate::models_error::ModelsErrorCode::Provider,
            format!(
                "Provider {}: at least one of \"api\", \"images\", or \"classifiers\" is required.",
                spec.id
            ),
        ));
    }
    Ok(SpecProvider {
        spec,
        dynamic: Arc::new(Mutex::new(Vec::new())),
    })
}

impl SpecProvider {
    /// `currentModels()` — baseline plus the dynamic overlay merged
    /// by `(type, id)` (upstream `currentModels`).
    fn current_models(&self) -> Vec<AnyModel> {
        let mut merged = self.spec.models.clone();
        {
            let overlay = self
                .dynamic
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            for model in overlay.iter() {
                merge_one(&mut merged, model.clone());
            }
        }
        merged
    }

    /// `apiFor(model)` — `None` when no implementation covers the api.
    fn api_for(&self, model: &Model) -> Option<Arc<dyn ChatApi>> {
        self.spec
            .api
            .as_ref()
            .and_then(|d| d.for_model(model).cloned())
    }

    /// `dispatch` — a missing api surfaces an in-stream error (pi's
    /// `lazyStream` throwing `ModelsError("stream", ...)`).
    fn dispatch(
        &self,
        model: &Model,
        run: impl FnOnce(Arc<dyn ChatApi>) -> AssistantStream + Send + 'static,
    ) -> AssistantStream {
        self.api_for(model).map_or_else(
            || {
                let id = self.spec.id.clone();
                let api = model.api.as_str().to_owned();
                crate::models::lazy_stream_error(
                    model,
                    format!("Provider {id} has no API implementation for \"{api}\""),
                )
            },
            run,
        )
    }
}

impl SpecProvider {
    /// `refreshModels`'s offline-restore `publish` — `update` applies
    /// the restored list.
    async fn publish_restore(
        &self,
        context: &RefreshModelsContext,
        restored: Vec<AnyModel>,
    ) -> bool {
        let dynamic = Arc::clone(&self.dynamic);
        (context.publish)(ModelsPublication {
            persist: PersistAction::Unchanged,
            update: Some(Box::new(move || {
                *dynamic
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner) = restored;
            })),
        })
        .await
    }

    /// `refreshModels`'s network `publish` — persists the fetched
    /// entry and applies `update`.
    async fn publish_refresh(&self, context: &RefreshModelsContext, refreshed: Vec<AnyModel>) {
        let dynamic = Arc::clone(&self.dynamic);
        let for_store = refreshed.clone();
        let _ = (context.publish)(ModelsPublication {
            persist: PersistAction::Write(ModelsStoreEntry {
                models: refreshed,
                checked_at: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map_or(0.0, |d| d.as_millis() as f64),
            }),
            update: Some(Box::new(move || {
                *dynamic
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner) = for_store;
            })),
        })
        .await;
    }
}

impl Provider for SpecProvider {
    fn id(&self) -> &str {
        &self.spec.id
    }

    fn name(&self) -> &str {
        self.spec.name.as_deref().unwrap_or(&self.spec.id)
    }

    fn base_url(&self) -> Option<&str> {
        self.spec.base_url.as_deref()
    }

    fn headers(&self) -> Option<&serde_json::Map<String, serde_json::Value>> {
        self.spec.headers.as_ref()
    }

    fn auth(&self) -> &ProviderAuth {
        &self.spec.auth
    }

    fn get_models(&self) -> Vec<Model> {
        self.current_models()
            .into_iter()
            .filter_map(|m| match m {
                AnyModel::Chat(c) => Some(*c),
                _ => None,
            })
            .collect()
    }

    fn get_all_models(&self) -> Vec<AnyModel> {
        self.current_models()
    }

    #[expect(
        clippy::excessive_nesting,
        reason = "mirrors upstream `refreshModels`'s restore/publish phases"
    )]
    fn refresh_models(
        &self,
        context: RefreshModelsContext,
    ) -> BoxFuture<'_, Result<(), ModelsError>> {
        Box::pin(async move {
            let Some(fetch) = &self.spec.fetch_models else {
                return Ok(());
            };
            if let Some(stored) = &context.stored {
                let restored: Vec<AnyModel> = stored
                    .models
                    .iter()
                    .filter(|m| model_ids(m).0 == self.spec.id)
                    .cloned()
                    .collect();
                if !self.publish_restore(&context, restored).await {
                    return Ok(());
                }
            }
            if !context.allow_network || context.signal.is_cancelled() {
                return Ok(());
            }
            let fetched = fetch(&context).await;
            if context.signal.is_cancelled() {
                return Ok(());
            }
            let refreshed: Vec<AnyModel> =
                fetched.into_iter().filter(has_known_model_type).collect();
            self.publish_refresh(&context, refreshed).await;
            Ok(())
        })
    }

    fn stream(
        &self,
        model: &Model,
        context: TranscriptContext,
        options: StreamOptions,
    ) -> AssistantStream {
        let request_model = model.clone();
        self.dispatch(model, move |api| {
            api.stream(&request_model, context, options)
        })
    }

    fn stream_simple(
        &self,
        model: &Model,
        context: TranscriptContext,
        options: SimpleStreamOptions,
    ) -> AssistantStream {
        let request_model = model.clone();
        self.dispatch(model, move |api| {
            api.stream_simple(&request_model, context, options)
        })
    }

    fn fetch_deferred(
        &self,
        model: &Model,
        handle: &DeferredHandle,
        options: DeferredFetchOptions,
    ) -> Option<AssistantStream> {
        let api = self.api_for(model)?;
        api.fetch_deferred(&model.clone(), &handle.clone(), options)
    }

    fn cancel_deferred(
        &self,
        model: &Model,
        handle: &DeferredHandle,
        options: DeferredCancelOptions,
    ) -> Option<BoxFuture<'static, Result<(), ModelsError>>> {
        let api = self.api_for(model)?;
        api.cancel_deferred(&model.clone(), &handle.clone(), options)
    }

    fn generate_images(
        &self,
        model: &crate::types::model::ImageModel,
        context: ImagesContext,
        options: ImagesOptions,
    ) -> Option<BoxFuture<'static, AssistantImages>> {
        let implementation = Arc::clone(self.spec.images.get(model.api.as_str())?);
        Some(implementation.generate_images(&model.clone(), context, options))
    }

    fn classify(
        &self,
        model: &crate::types::model::ClassifierModel,
        context: ClassifierContext,
        options: ClassifierOptions,
    ) -> Option<BoxFuture<'static, ClassifierResult>> {
        let implementation = Arc::clone(self.spec.classifiers.get(model.api.as_str())?);
        Some(implementation.classify(&model.clone(), context, options))
    }
}

/// `currentModels`'s one merge step — `(type, id)` replace-or-append.
fn merge_one(merged: &mut Vec<AnyModel>, model: AnyModel) {
    let kind = model_type(&model);
    let (_, id) = model_ids(&model);
    let index = merged
        .iter()
        .position(|entry| model_type(entry) == kind && model_ids(entry).1 == id);
    match index {
        Some(i) => {
            if let Some(slot) = merged.get_mut(i) {
                *slot = model;
            }
        }
        None => merged.push(model),
    }
}

/// `hasKnownModelType` — stored catalogs may carry types newer
/// versions know and this build does not.
const fn has_known_model_type(model: &AnyModel) -> bool {
    matches!(
        model,
        AnyModel::Chat(_) | AnyModel::Image(_) | AnyModel::Classifier(_)
    )
}
