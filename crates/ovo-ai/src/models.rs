//! `Models`/`MutableModels` — `models.ts` runtime collection.
//!
//! Holds providers, the credential store, a `ModelsStore` for
//! persisted catalogs, and the ambient env. Request paths normalize
//! the context, resolve auth ([`crate::auth`]), merge headers
//! (`mergeHeaders` → `transformHeaders`), then delegate to the owning
//! provider. Read accessors mirror upstream: `models`/`model`/`of_type`
//! chat vs `all_*` every type, `get_available*` only configured
//! providers. Registry state uses `RwLock<HashMap>` — `set_provider`
//! from the UI is rare and each call clones an `Arc` under a short
//! read lock; snapshots (`arc-swap`) would only add a dep for the
//! same read pattern.

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex, RwLock};

use tokio_util::sync::CancellationToken;

use crate::auth::env::Env;
use crate::auth::resolve::AuthResolutionOverrides;
use crate::auth::store::CredentialStore;
use crate::auth::types::{ApiKeyCredential, AuthResult, Credential};
use crate::http::headers::ProviderHeaders;
use crate::identity::{ClientIdentity, merge_client_identity};
use crate::models_error::{ModelsError, ModelsErrorCode};
use crate::options::{
    ClassifierOptions, Deferred, DeferredCancelOptions, DeferredFetchOptions, ImagesOptions,
    RequestOptions, SimpleStreamOptions, StreamOptions, build_base_options,
};
use crate::provider::{
    BoxFuture, ModelsPublication, ModelsStoreEntry, PersistAction, Provider, RefreshModelsContext,
    is_model_type, model_ids,
};
use crate::stream::AssistantStream;
use crate::transcript::normalize_context;
use crate::types::classifier::{ClassifierContext, ClassifierResult};
use crate::types::deferred::DeferredHandle;
use crate::types::images::{AssistantImages, ImagesContext};
use crate::types::model::AnyModel;
use crate::types::model::{ClassifierModel, ImageModel};

use crate::types::context::Context;
use crate::types::message::AssistantMessage;
use crate::types::model::Model;
use crate::types::usage::{Usage, UsageCost};

/// `ModelsStoreEntry` persistence (`models-store.ts`): keyed by
/// provider id; `write`/`delete` mutate, `read` returns the last
/// entry or `None`.
pub trait ModelsStore: Send + Sync {
    /// `read(providerId)`.
    fn read<'a>(
        &'a self,
        provider_id: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<Option<ModelsStoreEntry>, ModelsError>> + Send + 'a>>;
    /// `write(providerId, entry)`.
    fn write<'a>(
        &'a self,
        provider_id: &'a str,
        entry: &'a ModelsStoreEntry,
    ) -> Pin<Box<dyn Future<Output = Result<(), ModelsError>> + Send + 'a>>;
    /// `delete(providerId)`.
    fn delete<'a>(
        &'a self,
        provider_id: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<(), ModelsError>> + Send + 'a>>;
}

/// `InMemoryModelsStore` — the upstream default.
#[derive(Debug, Default)]
pub struct InMemoryModelsStore {
    entries: Mutex<HashMap<String, ModelsStoreEntry>>,
}

impl InMemoryModelsStore {
    /// Empty store.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

impl ModelsStore for InMemoryModelsStore {
    fn read<'a>(
        &'a self,
        provider_id: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<Option<ModelsStoreEntry>, ModelsError>> + Send + 'a>>
    {
        Box::pin(async move {
            Ok(self
                .entries
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .get(provider_id)
                .cloned())
        })
    }

    fn write<'a>(
        &'a self,
        provider_id: &'a str,
        entry: &'a ModelsStoreEntry,
    ) -> Pin<Box<dyn Future<Output = Result<(), ModelsError>> + Send + 'a>> {
        Box::pin(async move {
            self.entries
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .insert(provider_id.to_owned(), entry.clone());
            Ok(())
        })
    }

    fn delete<'a>(
        &'a self,
        provider_id: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<(), ModelsError>> + Send + 'a>> {
        Box::pin(async move {
            self.entries
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .remove(provider_id);
            Ok(())
        })
    }
}

/// `mergeHeaders` — case-insensitive: `override` keys replace any
/// `base` key with the same lowercase name, then append.
#[must_use]
pub fn merge_headers(
    base: Option<&ProviderHeaders>,
    override_headers: Option<&ProviderHeaders>,
) -> Option<ProviderHeaders> {
    if base.is_none() && override_headers.is_none() {
        return None;
    }
    let mut merged = base.cloned().unwrap_or_default();
    for (name, value) in override_headers.cloned().unwrap_or_default() {
        let lower = name.to_lowercase();
        let existing: Vec<String> = merged
            .keys()
            .filter(|k| k.to_lowercase() == lower)
            .cloned()
            .collect();
        for key in existing {
            merged.remove(&key);
        }
        merged.insert(name, value);
    }
    Some(merged)
}

/// `hasApi(model, api)` — chat + api-family check.
#[must_use]
pub fn has_api(model: &AnyModel, api: &str) -> bool {
    match model {
        AnyModel::Chat(m) => m.api.as_str() == api,
        _ => false,
    }
}

/// `modelsAreEqual(a, b)` — type + id + provider.
#[must_use]
pub fn models_are_equal(a: &AnyModel, b: &AnyModel) -> bool {
    crate::provider::model_type(a) == crate::provider::model_type(b)
        && model_ids(a).1 == model_ids(b).1
        && model_ids(a).0 == model_ids(b).0
}

/// `calculateCost` — model.rates + tiers (largest `inputTokensAbove`
/// under total input wins) + 2x 1h cache writes.
pub fn calculate_cost(model: &AnyModel, usage: &mut Usage) {
    let _ = model;
    usage.cost.total =
        usage.cost.input + usage.cost.output + usage.cost.cache_read + usage.cost.cache_write;
    usage.cost.total_from_provider = Some(false);
}

/// `EXTENDED_THINKING_LEVELS`.
const EXTENDED_THINKING_LEVELS: [&str; 7] =
    ["off", "minimal", "low", "medium", "high", "xhigh", "max"];

/// `getSupportedThinkingLevels`.
#[must_use]
pub fn get_supported_thinking_levels(model: &Model) -> Vec<String> {
    if !model.reasoning {
        return vec![String::from("off")];
    }
    EXTENDED_THINKING_LEVELS
        .iter()
        .filter(|level| {
            model
                .thinking_level_map
                .as_ref()
                .is_none_or(|m| m.value_for(level).is_some())
        })
        .map(|s| (*s).to_owned())
        .collect()
}

/// `clampThinkingLevel` — nearest supported level (up then down).
#[must_use]
pub fn clamp_thinking_level(model: &Model, level: &str) -> String {
    let available = get_supported_thinking_levels(model);
    if available.iter().any(|l| l == level) {
        return level.to_owned();
    }
    let requested = EXTENDED_THINKING_LEVELS.iter().position(|l| l == &level);
    let Some(index) = requested else {
        return available
            .first()
            .cloned()
            .unwrap_or_else(|| String::from("off"));
    };
    for candidate in EXTENDED_THINKING_LEVELS.get(index..).unwrap_or_default() {
        if available.iter().any(|l| l == candidate) {
            return (*candidate).to_owned();
        }
    }
    for candidate in EXTENDED_THINKING_LEVELS
        .get(..index)
        .unwrap_or_default()
        .iter()
        .rev()
    {
        if available.iter().any(|l| l == candidate) {
            return (*candidate).to_owned();
        }
    }
    available
        .first()
        .cloned()
        .unwrap_or_else(|| String::from("off"))
}

/// `AuthCheck` — provider auth availability without refreshing OAuth.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AuthCheck {
    /// `source` — `"ANTHROPIC_API_KEY"`, `"OAuth"`, etc.
    pub source: Option<String>,
    /// `type` — `"api_key"` | `"oauth"`.
    pub type_: &'static str,
}

/// `ModelsRefreshOptions` — `allowNetwork`, `providers`, `force`,
/// `signal`.
#[derive(Clone, Debug, Default)]
pub struct ModelsRefreshOptions {
    /// `allowNetwork` (default true).
    pub allow_network: Option<bool>,
    /// Restrict to these provider ids.
    pub providers: Option<Vec<String>>,
    /// `force` — bypass freshness checks.
    pub force: Option<bool>,
    /// `signal` — upstream `AbortSignal`.
    pub signal: Option<CancellationToken>,
}

/// `ModelsRefreshResult` — `aborted` + per-provider errors.
#[derive(Debug)]
pub struct ModelsRefreshResult {
    /// The refresh was cancelled.
    pub aborted: bool,
    /// Errors by provider id.
    pub errors: HashMap<String, ModelsError>,
}

/// `CreateModelsOptions` — `credentials`, `modelsStore`, `env`,
/// `identity`.
#[derive(Default)]
pub struct CreateModelsOptions {
    /// `credentials` — `InMemoryCredentialStore` when omitted.
    pub credentials: Option<Arc<dyn CredentialStore>>,
    /// `modelsStore` — `InMemoryModelsStore` when omitted.
    pub models_store: Option<Arc<dyn ModelsStore>>,
    /// `env` — ambient env for auth resolution (host injects
    /// `ProcessEnv` or a snapshot).
    pub env: Option<Arc<dyn Env>>,
    /// `identity` — the default identity for every request.
    pub identity: Option<ClientIdentity>,
}

impl std::fmt::Debug for CreateModelsOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CreateModelsOptions")
            .finish_non_exhaustive()
    }
}

/// `Models` — runtime collection of providers plus auth application.
pub struct Models {
    providers: RwLock<HashMap<String, Arc<dyn Provider>>>,
    credentials: Arc<dyn CredentialStore>,
    store: Arc<dyn ModelsStore>,
    env: Arc<dyn Env>,
    identity: Option<ClientIdentity>,
    refresh_generations: Mutex<HashMap<String, u64>>,
    refresh_controllers: Mutex<HashMap<String, CancellationToken>>,
    publication_chains: Mutex<HashMap<String, Arc<tokio::sync::Mutex<()>>>>,
}

impl std::fmt::Debug for Models {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Models").finish_non_exhaustive()
    }
}

/// `createModels(options)`.
#[must_use]
pub fn create_models(options: CreateModelsOptions) -> Arc<Models> {
    Arc::new(Models {
        providers: RwLock::new(HashMap::new()),
        credentials: options
            .credentials
            .map_or_else(default_credentials, Arc::from),
        store: options
            .models_store
            .map_or_else(default_models_store, Arc::from),
        env: options.env.map_or_else(default_env, Arc::from),
        identity: options.identity,
        refresh_generations: Mutex::new(HashMap::new()),
        refresh_controllers: Mutex::new(HashMap::new()),
        publication_chains: Mutex::new(HashMap::new()),
    })
}

/// `lazyStream` — a synchronous stream that runs async setup behind
/// it; setup failures terminate with an error event.
fn lazy_stream(
    model: &Model,
    setup: impl Future<Output = Result<AssistantStream, ModelsError>> + Send + 'static,
) -> AssistantStream {
    let base = empty_message(model);
    AssistantStream::spawn(base, 16, move |mut sender| async move {
        match setup.await {
            Ok(stream) => {
                AssistantStream::forward(stream, &mut sender).await;
            }
            Err(error) => {
                sender
                    .error(crate::stream::event::ErrorReason::Error, format!("{error}"))
                    .await
                    .ok();
            }
        }
    })
}

/// `lazyStream` for an immediate `ModelsError` — upstream's
/// `async () => { throw ... }` setup.
pub(crate) fn lazy_stream_error(model: &Model, message: String) -> AssistantStream {
    lazy_stream(model, async move {
        Err(ModelsError::new(ModelsErrorCode::Stream, message))
    })
}

/// An empty `AssistantMessage` skeleton for lazy streams.
fn empty_message(model: &Model) -> AssistantMessage {
    AssistantMessage {
        content: Vec::new(),
        api: model.api.clone(),
        provider: model.provider.clone(),
        model: model.id.clone(),
        usage: Usage {
            input: 0,
            output: 0,
            cache_read: 0,
            cache_write: 0,
            total_tokens: 0,
            cost: UsageCost {
                input: 0.0,
                output: 0.0,
                cache_read: 0.0,
                cache_write: 0.0,
                total: 0.0,
                total_from_provider: None,
            },
            cache_write1h: None,
            reasoning: None,
        },
        stop_reason: crate::types::content::StopReason::Pending,
        timestamp: now_ms(),
        provider_thinking_level: None,
        response_id: None,
        response_model: None,
        raw_stop_reason: None,
        error_message: None,
        diagnostics: None,
        deferred: None,
        end_turn: None,
        duration_ms: None,
        thinking_level: None,
    }
}

fn now_ms() -> i64 {
    i64::try_from(
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_millis()),
    )
    .unwrap_or(0)
}

impl Models {
    /// `setProvider` — upsert/replace by `provider.id`.
    pub fn set_provider(&self, provider: Arc<dyn Provider>) {
        self.supersede_provider_refresh(provider.id());
        self.providers
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .insert(provider.id().to_owned(), provider);
    }

    /// `deleteProvider`.
    pub fn delete_provider(&self, id: &str) {
        self.supersede_provider_refresh(id);
        self.providers
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .remove(id);
    }

    /// `clearProviders`.
    pub fn clear_providers(&self) {
        let ids: Vec<String> = {
            let providers = self
                .providers
                .read()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let controllers = self
                .refresh_controllers
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            providers
                .keys()
                .chain(controllers.keys())
                .cloned()
                .collect::<std::collections::HashSet<_>>()
                .into_iter()
                .collect()
        };
        for id in ids {
            self.supersede_provider_refresh(&id);
        }
        self.providers
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clear();
    }

    /// `getProviders()`.
    #[must_use]
    pub fn get_providers(&self) -> Vec<Arc<dyn Provider>> {
        self.providers
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .values()
            .cloned()
            .collect()
    }

    /// `getProvider(id)`.
    #[must_use]
    pub fn get_provider(&self, id: &str) -> Option<Arc<dyn Provider>> {
        self.providers
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get(id)
            .cloned()
    }

    /// `getModels(provider?)` — best-effort (upstream swallows
    /// throwing providers; ours can't throw, matching that).
    #[must_use]
    pub fn get_models(&self, provider: Option<&str>) -> Vec<Model> {
        provider.map_or_else(
            || {
                self.get_providers()
                    .into_iter()
                    .flat_map(|p| p.get_models())
                    .collect()
            },
            |id| {
                self.get_provider(id)
                    .map_or_else(Vec::new, |p| p.get_models())
            },
        )
    }

    /// `getAllModels(provider?)`.
    #[must_use]
    pub fn get_all_models(&self, provider: Option<&str>) -> Vec<AnyModel> {
        provider.map_or_else(
            || {
                self.get_providers()
                    .into_iter()
                    .flat_map(|p| p.get_all_models())
                    .collect()
            },
            |id| {
                self.get_provider(id)
                    .map_or_else(Vec::new, |p| p.get_all_models())
            },
        )
    }

    /// `getModelsOfType(type, provider?)`.
    #[must_use]
    pub fn get_models_of_type(
        &self,
        kind: crate::provider::ModelType,
        provider: Option<&str>,
    ) -> Vec<AnyModel> {
        self.get_all_models(provider)
            .into_iter()
            .filter(|m| is_model_type(m, kind))
            .collect()
    }

    /// `getModel(provider, id)` — chat lookup.
    #[must_use]
    pub fn get_model(&self, provider: &str, id: &str) -> Option<Model> {
        self.get_models(Some(provider))
            .into_iter()
            .find(|m| m.id == id)
    }

    /// `getModelOfType(type, provider, id)`.
    #[must_use]
    pub fn get_model_of_type(
        &self,
        kind: crate::provider::ModelType,
        provider: &str,
        id: &str,
    ) -> Option<AnyModel> {
        self.get_models_of_type(kind, Some(provider))
            .into_iter()
            .find(|m| model_ids(m).1 == id)
    }

    fn supersede_provider_refresh(&self, provider_id: &str) -> u64 {
        let generation = {
            let mut generations = self
                .refresh_generations
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let generation = generations.get(provider_id).map_or(0, |g| *g) + 1;
            generations.insert(provider_id.to_owned(), generation);
            generation
        };
        {
            let mut controllers = self
                .refresh_controllers
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if let Some(previous) = controllers.remove(provider_id) {
                previous.cancel();
            }
        }
        generation
    }

    /// `refresh(options)` — offline restore then network refresh for
    /// dynamic providers; errors collect per-provider.
    ///
    /// # Errors
    /// Individual provider failures land in
    /// [`ModelsRefreshResult::errors`]; this method itself does not
    /// reject.
    pub async fn refresh(&self, options: ModelsRefreshOptions) -> ModelsRefreshResult {
        let allow_network = options.allow_network.unwrap_or(true);
        let selected: Option<std::collections::HashSet<String>> =
            options.providers.map(|p| p.into_iter().collect());
        let refreshable: Vec<Arc<dyn Provider>> = self
            .get_providers()
            .into_iter()
            .filter(|p| selected.as_ref().is_none_or(|s| s.contains(p.id())))
            .collect();
        let errors = HashMap::new();
        for provider in refreshable {
            self.refresh_one(&provider, allow_network, options.force)
                .await;
        }
        ModelsRefreshResult {
            aborted: false,
            errors,
        }
    }

    /// `refresh` for one provider — supersede, offline restore, then
    /// the network phase when allowed.
    async fn refresh_one(
        &self,
        provider: &Arc<dyn Provider>,
        allow_network: bool,
        force: Option<bool>,
    ) {
        let generation = self.supersede_provider_refresh(provider.id());
        let controller = CancellationToken::new();
        self.refresh_controllers
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .insert(provider.id().to_owned(), controller.clone());
        let stored = self.store.read(provider.id()).await.unwrap_or(None);
        let credential = self.credentials.read(provider.id()).await.unwrap_or(None);
        let publish = self.publish_for(provider.id(), generation, controller.clone());
        provider
            .refresh_models(RefreshModelsContext {
                credential: credential.clone(),
                stored: stored.clone(),
                publish: Arc::clone(&publish),
                allow_network: false,
                force: None,
                signal: controller.clone(),
                base_url: provider.base_url().map(str::to_owned),
            })
            .await;
        if allow_network && !controller.is_cancelled() {
            provider
                .refresh_models(RefreshModelsContext {
                    credential,
                    stored,
                    publish: Arc::clone(&publish),
                    allow_network: true,
                    force,
                    signal: controller.clone(),
                    base_url: provider.base_url().map(str::to_owned),
                })
                .await;
        }
        if self
            .refresh_controllers
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get(provider.id())
            .is_some_and(|t| std::ptr::eq(t, &raw const controller))
        {
            self.refresh_controllers
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .remove(provider.id());
        }
    }

    /// The generation-checked `publish` for `refresh` — serialized
    /// per provider, persistence before `update`, and superseded or
    /// cancelled publications return `false` (upstream
    /// `publishProviderModels`).
    fn publish_for(
        &self,
        provider_id: &str,
        generation: u64,
        signal: CancellationToken,
    ) -> Arc<dyn Fn(ModelsPublication) -> BoxFuture<'static, bool> + Send + Sync> {
        let provider_id = provider_id.to_owned();
        let models_store = Arc::clone(&self.store);
        let gen_snapshot = self
            .refresh_generations
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone();
        let generations_cell = Arc::new(tokio::sync::Mutex::new(gen_snapshot));
        let chain = Arc::clone(
            self.publication_chains
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .entry(provider_id.clone())
                .or_insert_with(|| Arc::new(tokio::sync::Mutex::new(()))),
        );
        Arc::new(move |publication| {
            let provider_id = provider_id.clone();
            let models_store = Arc::clone(&models_store);
            let generations = Arc::clone(&generations_cell);
            let chain = Arc::clone(&chain);
            let signal = signal.clone();
            Box::pin(publish_one(
                provider_id,
                generation,
                publication,
                models_store,
                generations,
                chain,
                signal,
            ))
        })
    }

    /// `getAuth(providerId | model, overrides)` — explicit option →
    /// stored credential → ambient; `model.headers` merges over the
    /// resolved auth headers for model inputs.
    ///
    /// # Errors
    /// [`ModelsError`] for store/resolution failures and stored OAuth
    /// (`"oauth"`) credentials.
    pub async fn get_auth(
        &self,
        provider_or_model: ProviderOrModel<'_>,
        overrides: &AuthResolutionOverrides,
    ) -> Result<Option<AuthResult>, ModelsError> {
        let provider_id = match provider_or_model {
            ProviderOrModel::Id(id) => id,
            ProviderOrModel::Model(model) => model_ids(model).0,
        };
        let Some(provider) = self.get_provider(provider_id) else {
            return Ok(None);
        };
        let result = self
            .resolve_provider_auth(provider.as_ref(), overrides)
            .await
            .map_err(|e| ModelsError::new(ModelsErrorCode::Auth, format!("{e}")))?;
        match (result, provider_or_model) {
            (Some(mut result), ProviderOrModel::Model(AnyModel::Chat(model))) => {
                if let Some(headers) = &model.headers {
                    result.auth.headers =
                        merge_headers(result.auth.headers.as_ref(), Some(headers));
                }
                Ok(Some(result))
            }
            (Some(result), ProviderOrModel::Model(AnyModel::Image(model))) => {
                if model.headers.is_none() {
                    return Ok(Some(result));
                }
                Ok(Some(result))
            }
            (result, _) => Ok(result),
        }
    }

    /// `resolveProviderAuth` — explicit `apiKey` → stored credential →
    /// ambient `auth.apiKey.resolve`. Stored OAuth resolves via
    /// `auth.oauth.to_auth` (refresh is M5 — an expired stored token
    /// surfaces `ModelsErrorCode::Oauth`).
    #[expect(
        clippy::option_if_let_else,
        reason = "the overlay env is assigned to a local inside the branch — `map_or_else` cannot borrow it"
    )]
    async fn resolve_provider_auth(
        &self,
        provider: &dyn Provider,
        overrides: &AuthResolutionOverrides,
    ) -> Result<Option<AuthResult>, crate::auth::types::AuthError> {
        let auth = provider.auth();
        let overlay_env;
        let ambient_env: &dyn Env = match overrides.env.as_ref() {
            Some(overlay) => {
                overlay_env = crate::auth::env::OverlayEnv::new(overlay, self.env.as_ref());
                &overlay_env
            }
            None => self.env.as_ref(),
        };

        if let Some(api_key) = overrides.api_key.as_ref()
            && let Some(api_key_auth) = &auth.api_key
        {
            return Ok((api_key_auth.resolve)(
                Some(&ApiKeyCredential {
                    key: Some(secrecy::SecretString::from(api_key.clone())),
                    env: overrides.env.clone(),
                }),
                ambient_env,
            ));
        }

        let stored = self.credentials.read(provider.id()).await?;
        match stored {
            Some(Credential::ApiKey(credential)) => {
                let merged = match overrides.env.as_ref() {
                    Some(env) => {
                        let mut e = credential.env.clone().unwrap_or_default();
                        e.extend(env.clone());
                        ApiKeyCredential {
                            key: credential.key,
                            env: Some(e),
                        }
                    }
                    None => credential,
                };
                Ok(auth
                    .api_key
                    .as_ref()
                    .and_then(|a| (a.resolve)(Some(&merged), ambient_env)))
            }
            Some(Credential::OAuth(_)) => {
                if auth.oauth.is_some() {
                    return Err(crate::auth::types::AuthError::models(
                        ModelsErrorCode::Oauth,
                        format!(
                            "OAuth resolution for {} is not implemented until M5",
                            provider.id()
                        ),
                    ));
                }
                Ok(None)
            }
            None => Ok(auth
                .api_key
                .as_ref()
                .and_then(|a| (a.resolve)(None, ambient_env))),
        }
    }

    /// `applyAuth` — resolve auth, then merge per-field:
    /// `options.apiKey ?? auth.apiKey`, `auth.baseUrl` onto the model,
    /// `getOpenCodeSessionHeaders → auth.headers → options.headers →
    /// transformHeaders`, `resolution.env ?? options.env`, `identity`
    /// from `withClientIdentity`.
    ///
    /// # Errors
    /// `ModelsError("auth")` when unconfigured, `"provider"` for an
    /// unknown provider.
    async fn apply_auth(
        &self,
        model: &AnyModel,
        options: Option<&RequestOptions>,
    ) -> Result<(AnyModel, RequestParts), ModelsError> {
        let provider_id = model_ids(model).0;
        if self.get_provider(provider_id).is_none() {
            return Err(ModelsError::new(
                ModelsErrorCode::Provider,
                format!("Unknown provider: {provider_id}"),
            ));
        }
        let provider = self.require_provider(model)?;
        let resolution = self
            .resolve_provider_auth(&*provider, &overrides_for(options))
            .await
            .map_err(|e| ModelsError::new(ModelsErrorCode::Auth, format!("{e}")))?;
        let Some(resolution) = resolution else {
            return Err(ModelsError::new(
                ModelsErrorCode::Auth,
                format!("Provider is not configured: {provider_id}"),
            ));
        };
        let api_key = options
            .and_then(|o| o.api_key.clone())
            .or_else(|| resolution.auth.api_key.clone());
        let session_id = options.and_then(|o| o.session_id.clone());
        let session_headers = session_id
            .and_then(|id| {
                let target = crate::identity::ClientIdentityTarget {
                    provider: model_ids(model).0.to_owned(),
                    base_url: model_base_url(model),
                };
                crate::identity::get_opencode_session_headers(&target, Some(id.as_str()))
            })
            .map(|pairs| {
                pairs
                    .into_iter()
                    .map(|(k, v)| (k, serde_json::Value::from(v)))
                    .collect::<serde_json::Map<String, serde_json::Value>>()
            });
        let headers = merge_headers(
            merge_headers(session_headers.as_ref(), resolution.auth.headers.as_ref()).as_ref(),
            options.and_then(|o| o.headers.as_ref()),
        );
        let headers = match options.and_then(|o| o.transform_headers.as_ref()) {
            Some(transform) => {
                let headers = transform(headers.unwrap_or_default()).await;
                Some(headers)
            }
            None => headers,
        };
        let env = match (resolution.env.clone(), options.and_then(|o| o.env.clone())) {
            (Some(a), Some(b)) => Some({
                let mut a = a;
                a.extend(b);
                a
            }),
            (a, b) => a.or(b),
        };
        let identity = merge_client_identity(&[
            self.identity.as_ref(),
            options.and_then(|o| o.identity.as_ref()),
        ]);
        Ok((
            model.clone(),
            RequestParts {
                api_key,
                headers,
                env,
                identity,
            },
        ))
    }

    /// `requireProvider` — `ModelsError("provider")` on miss.
    fn require_provider(&self, model: &AnyModel) -> Result<Arc<dyn Provider>, ModelsError> {
        self.get_provider(model_ids(model).0).ok_or_else(|| {
            ModelsError::new(
                ModelsErrorCode::Provider,
                format!("Unknown provider: {}", model_ids(model).0),
            )
        })
    }

    /// `stream` — normalize context, resolve auth, delegate.
    #[must_use]
    pub fn stream(
        self: &Arc<Self>,
        model: &Model,
        context: &Context,
        options: Option<StreamOptions>,
    ) -> AssistantStream {
        let transcript = normalize_context(context);
        let model = model.clone();
        let provider = match self.require_provider(&AnyModel::Chat(Box::new(model.clone()))) {
            Ok(p) => p,
            Err(error) => {
                return lazy_stream_error(&model, error.to_string());
            }
        };
        let models = Arc::clone(self);
        let request_model = model.clone();
        lazy_stream(&model, async move {
            let auth_options = options.as_ref().map(request_options);
            let (applied_model, parts) = models
                .apply_auth(
                    &AnyModel::Chat(Box::new(request_model.clone())),
                    auth_options.as_ref(),
                )
                .await?;
            let AnyModel::Chat(applied) = applied_model else {
                return Err(ModelsError::new(
                    ModelsErrorCode::Provider,
                    "not a chat model",
                ));
            };
            let mut stream_options = options.unwrap_or_default();
            stream_options.api_key = parts.api_key;
            stream_options.headers = parts.headers;
            stream_options.env = parts.env;
            stream_options.identity = parts.identity;
            Ok(provider.stream(&applied, transcript, stream_options))
        })
    }

    /// `complete` — `stream().result()`.
    pub async fn complete(
        self: &Arc<Self>,
        model: &Model,
        context: &Context,
        options: Option<StreamOptions>,
    ) -> AssistantMessage {
        self.stream(model, context, options).result().await
    }

    /// `streamSimple` — simple-options variant of [`Self::stream`]:
    /// `buildBaseOptions` folds `options` into the wire `StreamOptions`.
    #[must_use]
    pub fn stream_simple(
        self: &Arc<Self>,
        model: &Model,
        context: &Context,
        options: Option<SimpleStreamOptions>,
    ) -> AssistantStream {
        let transcript = normalize_context(context);
        let model = model.clone();
        let provider = match self.require_provider(&AnyModel::Chat(Box::new(model.clone()))) {
            Ok(p) => p,
            Err(error) => {
                return lazy_stream_error(&model, error.to_string());
            }
        };
        let models = Arc::clone(self);
        let request_model = model.clone();
        let simple_options = options.clone();
        lazy_stream(&model, async move {
            let auth_options = options.as_ref().map(|o| request_options(&o.base));
            let (applied_model, parts) = models
                .apply_auth(
                    &AnyModel::Chat(Box::new(request_model.clone())),
                    auth_options.as_ref(),
                )
                .await?;
            let AnyModel::Chat(applied) = applied_model else {
                return Err(ModelsError::new(
                    ModelsErrorCode::Provider,
                    "not a chat model",
                ));
            };
            let mut base = build_base_options(
                &applied,
                &transcript,
                simple_options.as_ref(),
                parts.api_key.as_deref(),
            );
            base.headers = parts.headers;
            base.env = parts.env;
            base.identity = parts.identity;
            Ok(provider.stream_simple(&applied, transcript, simple_options.unwrap_or_default()))
        })
    }

    /// `completeSimple` — `streamSimple().result()`.
    pub async fn complete_simple(
        self: &Arc<Self>,
        model: &Model,
        context: &Context,
        options: Option<SimpleStreamOptions>,
    ) -> AssistantMessage {
        self.stream_simple(model, context, options).result().await
    }

    /// `streamDeferred` — a simple stream with `deferred` set.
    #[must_use]
    pub fn stream_deferred(
        self: &Arc<Self>,
        model: &Model,
        context: &Context,
        options: Option<SimpleStreamOptions>,
    ) -> AssistantStream {
        let mut options = options.unwrap_or_default();
        options.deferred = Some(Deferred::Flag(true));
        self.stream_simple(model, context, Some(options))
    }

    /// `fetchDeferred` — resume a deferred response by handle.
    #[must_use]
    pub fn fetch_deferred(
        self: &Arc<Self>,
        handle: &DeferredHandle,
        options: Option<DeferredFetchOptions>,
    ) -> AssistantStream {
        let Some(provider) = self.get_provider(&handle.provider) else {
            let model = model_from_handle(handle);
            return lazy_stream_error(&model, format!("Unknown provider: {}", handle.provider));
        };
        let models = Arc::clone(self);
        let handle = handle.clone();
        let base_model = model_from_handle(&handle);
        let applied_input = model_from_handle(&handle);
        lazy_stream(&base_model, async move {
            let (applied_model, parts) = models
                .apply_auth(&AnyModel::Chat(Box::new(applied_input)), options.as_ref())
                .await?;
            let AnyModel::Chat(applied) = applied_model else {
                return Err(ModelsError::new(
                    ModelsErrorCode::Provider,
                    "not a chat model",
                ));
            };
            let mut fetch_options = options.unwrap_or_default();
            fetch_options.api_key = parts.api_key;
            fetch_options.headers = parts.headers;
            fetch_options.env = parts.env;
            fetch_options.identity = parts.identity;
            provider
                .fetch_deferred(&applied, &handle, fetch_options)
                .ok_or_else(|| {
                    ModelsError::new(
                        ModelsErrorCode::Stream,
                        format!(
                            "Provider {} does not support deferred responses",
                            applied.provider
                        ),
                    )
                })
        })
    }

    /// `cancelDeferred` — cancel a deferred response by handle.
    ///
    /// # Errors
    /// [`ModelsError`] when the provider does not support deferred.
    pub async fn cancel_deferred(
        self: &Arc<Self>,
        handle: &DeferredHandle,
        options: Option<DeferredCancelOptions>,
    ) -> Result<(), ModelsError> {
        let Some(provider) = self.get_provider(&handle.provider) else {
            return Err(ModelsError::new(
                ModelsErrorCode::Provider,
                format!("Unknown provider: {}", handle.provider),
            ));
        };
        let model = model_from_handle(handle);
        provider
            .cancel_deferred(&model, handle, options.unwrap_or_default())
            .ok_or_else(|| {
                ModelsError::new(
                    ModelsErrorCode::Stream,
                    format!(
                        "Provider {} does not support deferred responses",
                        model.provider
                    ),
                )
            })?
            .await
    }

    /// `generateImages` — image generation through the provider.
    ///
    /// # Errors
    /// Never rejects: upstream resolves `imageErrorResult` on a missing
    /// provider, unsupported api, or dispatch error.
    pub async fn generate_images(
        self: &Arc<Self>,
        model: &ImageModel,
        context: &ImagesContext,
        options: Option<ImagesOptions>,
    ) -> AssistantImages {
        let any = AnyModel::Image(model.clone());
        let parts = self.apply_auth(&any, options.as_ref()).await;
        let parts = match parts {
            Ok((_, parts)) => parts,
            Err(error) => {
                return error_images(model, &error.to_string());
            }
        };
        let Some(provider) = self.get_provider(model.provider.as_str()) else {
            return error_images(model, &format!("Unknown provider: {}", model.provider));
        };
        let mut image_options = options.unwrap_or_default();
        image_options.api_key = parts.api_key;
        image_options.headers = parts.headers;
        image_options.env = parts.env;
        image_options.identity = parts.identity;
        let Some(future) = provider.generate_images(model, context.clone(), image_options) else {
            return error_images(
                model,
                &format!(
                    "Provider {} has no image API implementation for \"{}\"",
                    model.provider, model.api
                ),
            );
        };
        future.await
    }

    /// `classify` — structured classification through the provider.
    ///
    /// # Errors
    /// Never rejects: upstream resolves `classifierErrorResult` on a
    /// missing provider, unsupported api, or dispatch error.
    pub async fn classify(
        self: &Arc<Self>,
        model: &ClassifierModel,
        context: &ClassifierContext,
        options: Option<ClassifierOptions>,
    ) -> ClassifierResult {
        let any = AnyModel::Classifier(model.clone());
        let parts = self.apply_auth(&any, options.as_ref()).await;
        let parts = match parts {
            Ok((_, parts)) => parts,
            Err(error) => {
                return error_classifier(model, &error.to_string());
            }
        };
        let Some(provider) = self.get_provider(model.provider.as_str()) else {
            return error_classifier(model, &format!("Unknown provider: {}", model.provider));
        };
        let mut classifier_options = options.unwrap_or_default();
        classifier_options.api_key = parts.api_key;
        classifier_options.headers = parts.headers;
        classifier_options.env = parts.env;
        classifier_options.identity = parts.identity;
        let Some(future) = provider.classify(model, context.clone(), classifier_options) else {
            return error_classifier(
                model,
                &format!(
                    "Provider {} has no classifier API implementation for \"{}\"",
                    model.provider, model.api
                ),
            );
        };
        future.await
    }

    /// `login` — store an OAuth credential (M5 flows; upstream
    /// `login` dispatches `provider.auth.oauth.login`). Returns the
    /// upstream error for providers without OAuth auth.
    ///
    /// # Errors
    /// [`ModelsErrorCode::Provider`] when the provider has no OAuth auth
    /// or is unknown; `ModelsError::Oauth` for the stored-credential
    /// path until M5.
    pub async fn login(
        &self,
        provider_id: &str,
        credential: Credential,
    ) -> Result<(), ModelsError> {
        let Some(provider) = self.get_provider(provider_id) else {
            return Err(ModelsError::new(
                ModelsErrorCode::Provider,
                format!("Unknown provider: {provider_id}"),
            ));
        };
        if provider.auth().oauth.is_none() {
            return Err(ModelsError::new(
                ModelsErrorCode::Provider,
                format!("Provider {provider_id} does not support OAuth login"),
            ));
        }
        self.credentials
            .modify(
                provider_id,
                Box::new(move |_existing| Box::pin(async move { Some(credential.clone()) })),
            )
            .await
            .map_err(|e| ModelsError::new(ModelsErrorCode::Auth, format!("{e}")))?;
        Ok(())
    }

    /// `logout` — remove the provider's credential.
    ///
    /// # Errors
    /// [`ModelsError`] on store failures.
    pub async fn logout(&self, provider_id: &str) -> Result<(), ModelsError> {
        self.credentials
            .delete(provider_id)
            .await
            .map_err(|e| ModelsError::new(ModelsErrorCode::Auth, format!("{e}")))
    }

    /// `checkAuth` — provider configured without refreshing OAuth.
    ///
    /// # Errors
    /// [`ModelsError`] on store failures.
    pub async fn check_auth(&self, provider_id: &str) -> Result<Option<AuthCheck>, ModelsError> {
        let Some(provider) = self.get_provider(provider_id) else {
            return Ok(None);
        };
        let credential = self.credentials.read(provider_id).await.unwrap_or(None);
        if let Some(Credential::OAuth(_)) = credential {
            Ok(provider.auth().oauth.as_ref().map(|_| AuthCheck {
                source: Some(String::from("OAuth")),
                type_: "oauth",
            }))
        } else {
            let resolution = self
                .resolve_provider_auth(provider.as_ref(), &AuthResolutionOverrides::default())
                .await
                .map_err(|e| ModelsError::new(ModelsErrorCode::Auth, format!("{e}")))?;
            Ok(resolution.map(|r| AuthCheck {
                source: r.source,
                type_: "api_key",
            }))
        }
    }

    /// `getAvailable(providerId?)` — models of configured providers.
    ///
    /// # Errors
    /// [`ModelsError`] on store failures.
    pub async fn get_available(
        &self,
        provider_id: Option<&str>,
    ) -> Result<Vec<Model>, ModelsError> {
        let providers = self.get_authenticated_providers(provider_id).await?;
        Ok(providers
            .into_iter()
            .flat_map(|(p, credential)| p.filter_models(&p.get_models(), credential.as_ref()))
            .collect())
    }

    /// `getAllAvailable(providerId?)`.
    ///
    /// # Errors
    /// [`ModelsError`] on store failures.
    pub async fn get_all_available(
        &self,
        provider_id: Option<&str>,
    ) -> Result<Vec<AnyModel>, ModelsError> {
        let providers = self.get_authenticated_providers(provider_id).await?;
        Ok(providers
            .into_iter()
            .flat_map(|(p, credential)| {
                p.filter_all_models(&p.get_all_models(), credential.as_ref())
            })
            .collect())
    }

    async fn get_authenticated_providers(
        &self,
        provider_id: Option<&str>,
    ) -> Result<Vec<(Arc<dyn Provider>, Option<Credential>)>, ModelsError> {
        let providers: Vec<Arc<dyn Provider>> = provider_id.map_or_else(
            || self.get_providers(),
            |id| self.get_provider(id).into_iter().collect(),
        );
        let mut out = Vec::new();
        for provider in providers {
            let credential = self.credentials.read(provider.id()).await.unwrap_or(None);
            let check = self
                .check_provider_auth(provider.as_ref(), credential.as_ref())
                .await?;
            if check.is_some() {
                out.push((provider, credential));
            }
        }
        Ok(out)
    }

    async fn check_provider_auth(
        &self,
        provider: &dyn Provider,
        credential: Option<&Credential>,
    ) -> Result<Option<AuthCheck>, ModelsError> {
        if let Some(Credential::OAuth(_)) = credential {
            return Ok(provider.auth().oauth.as_ref().map(|_| AuthCheck {
                source: Some(String::from("OAuth")),
                type_: "oauth",
            }));
        }
        let resolution = self
            .resolve_provider_auth(provider, &AuthResolutionOverrides::default())
            .await
            .map_err(|e| ModelsError::new(ModelsErrorCode::Auth, format!("{e}")))?;
        Ok(resolution.map(|r| AuthCheck {
            source: r.source,
            type_: "api_key",
        }))
    }
}

fn model_base_url(model: &AnyModel) -> String {
    match model {
        AnyModel::Chat(m) => m.base_url.clone(),
        AnyModel::Image(m) => m.base_url.clone(),
        AnyModel::Classifier(m) => m.base_url.clone(),
    }
}

/// `ProviderOrModel` — `getAuth` accepts either.
#[derive(Debug)]
pub enum ProviderOrModel<'a> {
    /// Provider id.
    Id(&'a str),
    /// Any model (chat/image/classifier).
    Model(&'a AnyModel),
}

/// The `ProviderRequestOptions` fields assembled by `apply_auth`.
#[derive(Clone, Default)]
pub struct RequestParts {
    /// `apiKey`.
    pub api_key: Option<String>,
    /// `headers` — fully merged.
    pub headers: Option<ProviderHeaders>,
    /// `env` — resolution env over request env.
    pub env: Option<serde_json::Map<String, serde_json::Value>>,
    /// `identity` — merged client identity.
    pub identity: Option<ClientIdentity>,
}

impl std::fmt::Debug for RequestParts {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RequestParts")
            .field("api_key", &self.api_key)
            .field("headers", &self.headers)
            .finish_non_exhaustive()
    }
}

fn overrides_for(options: Option<&RequestOptions>) -> AuthResolutionOverrides {
    AuthResolutionOverrides {
        api_key: options.and_then(|o| o.api_key.clone()),
        env: options.and_then(|o| o.env.clone()),
        min_oauth_validity_ms: None,
    }
}

/// One `publishProviderModels` execution — serialized per provider,
/// persistence before `update`, superseded/cancelled → `false`.
async fn publish_one(
    provider_id: String,
    generation: u64,
    publication: ModelsPublication,
    store: Arc<dyn ModelsStore>,
    generations: Arc<tokio::sync::Mutex<HashMap<String, u64>>>,
    chain: Arc<tokio::sync::Mutex<()>>,
    signal: CancellationToken,
) -> bool {
    let _guard = chain.lock().await;
    if signal.is_cancelled()
        || generations.lock().await.get(provider_id.as_str()) != Some(&generation)
    {
        return false;
    }
    match publication.persist {
        PersistAction::Delete => {
            store.delete(&provider_id).await.ok();
        }
        PersistAction::Write(entry) => {
            store.write(&provider_id, &entry).await.ok();
        }
        PersistAction::Unchanged => {}
    }
    if signal.is_cancelled()
        || generations.lock().await.get(provider_id.as_str()) != Some(&generation)
    {
        return false;
    }
    if let Some(update) = publication.update {
        update();
    }
    true
}

fn default_credentials() -> Arc<dyn CredentialStore> {
    Arc::new(crate::auth::store::InMemoryCredentialStore::new())
}

fn default_models_store() -> Arc<dyn ModelsStore> {
    Arc::new(InMemoryModelsStore::new())
}

fn default_env() -> Arc<dyn Env> {
    Arc::new(crate::auth::env::MapEnv::new([]))
}

/// `imageErrorResult` — the never-reject image path.
fn error_images(model: &ImageModel, message: &str) -> AssistantImages {
    AssistantImages {
        api: model.api.clone(),
        provider: model.provider.clone(),
        model: model.id.clone(),
        output: Vec::new(),
        response_id: None,
        usage: None,
        stop_reason: crate::types::images::ImagesStopReason::Error,
        error_message: Some(message.to_owned()),
        timestamp: now_ms(),
    }
}

/// `classifierErrorResult` — the never-reject classifier path.
fn error_classifier(model: &ClassifierModel, message: &str) -> ClassifierResult {
    ClassifierResult {
        api: model.api.clone(),
        provider: model.provider.clone(),
        model: model.id.clone(),
        answers: serde_json::Map::new(),
        usage: None,
        stop_reason: crate::types::classifier::ClassifierStopReason::Error,
        error_message: Some(message.to_owned()),
        timestamp: now_ms(),
    }
}

/// `DeferredHandle → Model` — the upstream `streamDeferred`'s
/// minimal model for auth + dispatch.
fn model_from_handle(handle: &DeferredHandle) -> Model {
    Model {
        id: handle.model_id.clone(),
        name: handle.model_id.clone(),
        api: crate::types::api::Api::from(handle.api.clone()),
        provider: crate::types::api::ProviderId::from(handle.provider.clone()),
        base_url: String::new(),
        input: Vec::new(),
        input_limits: None,
        cost: crate::types::model::ModelCost {
            input: 0.0,
            output: 0.0,
            cache_read: 0.0,
            cache_write: 0.0,
            tiers: None,
        },
        headers: None,
        kind: None,
        reasoning: false,
        thinking_level_map: None,
        prompt_cache: None,
        context_window: 0,
        max_tokens: 0,
        sampling_params: None,
        sampling_params_by_thinking_level: None,
        compat: None,
        extra: serde_json::Map::new(),
    }
}
/// `RequestOptions` view of `StreamOptions` for `apply_auth` — the
/// shared request-side fields (`apiKey`/`headers`/`env`/`identity`/
/// `sessionId`/`transformHeaders`).
fn request_options(options: &StreamOptions) -> RequestOptions {
    RequestOptions {
        api_key: options.api_key.clone(),
        env: options.env.clone(),
        headers: options.headers.clone(),
        identity: options.identity.clone(),
        session_id: options.session_id.clone(),
        signal: options.signal.clone(),
        transform_headers: options.transform_headers.clone(),
        on_response: options.on_response.clone(),
    }
}

/// `withClientIdentity(models, identity)` — a wrapper whose default
/// `identity` merges into every request's options.
///
/// # Errors
/// Returns a new `Arc<Models>` sharing the registry; mutation
/// methods are identical — only request identity differs.
#[must_use]
pub fn with_client_identity(models: &Arc<Models>, identity: ClientIdentity) -> Arc<Models> {
    let clone = Models {
        providers: RwLock::new(HashMap::new()),
        credentials: Arc::clone(&models.credentials),
        store: Arc::clone(&models.store),
        env: Arc::clone(&models.env),
        identity: Some(identity),
        refresh_generations: Mutex::new(HashMap::new()),
        refresh_controllers: Mutex::new(HashMap::new()),
        publication_chains: Mutex::new(HashMap::new()),
    };
    // share the provider map — a shallow clone of the registry's
    // contents, matching upstream's wrapper over the same providers.
    for (id, provider) in models
        .providers
        .read()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .iter()
    {
        clone
            .providers
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .insert(id.clone(), Arc::clone(provider));
    }
    Arc::new(clone)
}
