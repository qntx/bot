//! `Models`/`MutableModels` — `models.ts` runtime collection.
//!
//! Holds providers, the credential store, a `ModelsStore` for
//! persisted catalogs, and the ambient env. Request paths normalize
//! the context, resolve auth ([`crate::auth`]), merge headers
//! (`mergeHeaders` → `transformHeaders`), then delegate to the owning
//! provider. Read accessors mirror upstream: `models`/`model`/`of_type`
//! chat vs `all_*` every type, `get_available*` only configured
//! providers. Registry state uses `RwLock<Vec>` — `set_provider`
//! from the UI is rare and each call clones an `Arc` under a short
//! read lock; snapshots (`arc-swap`) would only add a dep for the
//! same read pattern.

use std::collections::HashMap;
use std::future::Future;
use std::sync::{Arc, Mutex, RwLock, Weak};

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

/// `ModelsStoreEntry` persistence (`models-store.ts`): keyed by
/// provider id; `write`/`delete` mutate, `read` returns the last
/// entry or `None`.
pub trait ModelsStore: Send + Sync {
    /// `read(providerId)`.
    fn read<'a>(
        &'a self,
        provider_id: &'a str,
    ) -> BoxFuture<'a, Result<Option<ModelsStoreEntry>, ModelsError>>;
    /// `write(providerId, entry)`.
    fn write<'a>(
        &'a self,
        provider_id: &'a str,
        entry: &'a ModelsStoreEntry,
    ) -> BoxFuture<'a, Result<(), ModelsError>>;
    /// `delete(providerId)`.
    fn delete<'a>(&'a self, provider_id: &'a str) -> BoxFuture<'a, Result<(), ModelsError>>;
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
    ) -> BoxFuture<'a, Result<Option<ModelsStoreEntry>, ModelsError>> {
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
    ) -> BoxFuture<'a, Result<(), ModelsError>> {
        Box::pin(async move {
            self.entries
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .insert(provider_id.to_owned(), entry.clone());
            Ok(())
        })
    }

    fn delete<'a>(&'a self, provider_id: &'a str) -> BoxFuture<'a, Result<(), ModelsError>> {
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

/// `AuthType` — `"api_key"` | `"oauth"` like upstream's auth
/// resolution kind.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AuthType {
    /// API key auth.
    ApiKey,
    /// OAuth auth.
    OAuth,
}

impl AuthType {
    /// Upstream's `type` string.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ApiKey => "api_key",
            Self::OAuth => "oauth",
        }
    }
}

/// `AuthCheck` — provider auth availability without refreshing OAuth.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AuthCheck {
    /// `source` — `"ANTHROPIC_API_KEY"`, `"OAuth"`, etc.
    pub source: Option<String>,
    /// `type` — `"api_key"` | `"oauth"`.
    pub kind: AuthType,
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

/// The pieces `refresh_phase` threads into
/// `runProviderRefreshPhase`.
struct RefreshPhase<'a> {
    credential: Option<Credential>,
    stored: Option<ModelsStoreEntry>,
    allow_network: bool,
    force: Option<bool>,
    generation: u64,
    signal: &'a CancellationToken,
}

/// `Registry` — the shared state every `Models` handle reads and
/// mutates; `withClientIdentity` wrappers alias it.
struct Registry {
    /// `providers` — insertion-ordered like upstream's `Map`.
    providers: RwLock<Vec<Arc<dyn Provider>>>,
    /// `credentials`.
    credentials: Arc<dyn CredentialStore>,
    /// `modelsStore`.
    store: Arc<dyn ModelsStore>,
    /// `env`.
    env: Arc<dyn Env>,
    /// Per-provider refresh generation — a supersede bumps it.
    refresh_generations: Mutex<HashMap<String, u64>>,
    /// Per-provider linked `AbortSignal.any` tokens.
    refresh_controllers: Mutex<HashMap<String, CancellationToken>>,
    /// Per-provider publication serialization chain.
    publication_chains: Mutex<HashMap<String, Arc<tokio::sync::Mutex<()>>>>,
}

/// `Models` — a handle over a shared `Registry` plus this handle's
/// default `identity`. Mutations through any handle are visible to
/// the others, like upstream's shared `ModelsRegistry`.
pub struct Models {
    /// The shared registry.
    registry: Arc<Registry>,
    /// This handle's default request identity.
    identity: Option<ClientIdentity>,
}

impl std::fmt::Debug for Registry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Registry").finish_non_exhaustive()
    }
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
        registry: Arc::new(Registry {
            providers: RwLock::new(Vec::new()),
            credentials: options
                .credentials
                .map_or_else(default_credentials, Arc::from),
            store: options
                .models_store
                .map_or_else(default_models_store, Arc::from),
            env: options.env.map_or_else(default_env, Arc::from),
            refresh_generations: Mutex::new(HashMap::new()),
            refresh_controllers: Mutex::new(HashMap::new()),
            publication_chains: Mutex::new(HashMap::new()),
        }),
        identity: options.identity,
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
    AssistantMessage::empty(model, crate::stream::accumulator::wall_ms())
}

impl Models {
    /// `setProvider` — upsert/replace by `provider.id`.
    pub fn set_provider(&self, provider: Arc<dyn Provider>) {
        self.supersede_provider_refresh(provider.id());
        // JS `Map.set` keeps the entry's position on replace.
        let mut providers = self
            .registry
            .providers
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        match providers.iter().position(|p| p.id() == provider.id()) {
            Some(index) => {
                if let Some(slot) = providers.get_mut(index) {
                    *slot = provider;
                }
            }
            None => providers.push(provider),
        }
    }

    /// `deleteProvider`.
    pub fn delete_provider(&self, id: &str) {
        self.supersede_provider_refresh(id);
        self.registry
            .providers
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .retain(|p| p.id() != id);
    }

    /// `clearProviders`.
    pub fn clear_providers(&self) {
        let ids: Vec<String> = {
            let providers = self
                .registry
                .providers
                .read()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let controllers = self
                .registry
                .refresh_controllers
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            providers
                .iter()
                .map(|p| p.id().to_owned())
                .chain(controllers.keys().cloned())
                .collect::<std::collections::HashSet<_>>()
                .into_iter()
                .collect()
        };
        for id in ids {
            self.supersede_provider_refresh(&id);
        }
        self.registry
            .providers
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clear();
    }

    /// `getProviders()`.
    #[must_use]
    pub fn get_providers(&self) -> Vec<Arc<dyn Provider>> {
        self.registry
            .providers
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }

    /// `getProvider(id)`.
    #[must_use]
    pub fn get_provider(&self, id: &str) -> Option<Arc<dyn Provider>> {
        self.registry
            .providers
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .iter()
            .find(|p| p.id() == id)
            .cloned()
    }

    /// `getModels(provider?)` — best-effort (upstream swallows
    /// throwing providers; ours can't throw, matching that).
    #[must_use]
    pub fn get_models(&self, provider: Option<&str>) -> Vec<Model> {
        // Upstream swallows provider `getModels` failures at listing —
        // the throwing analogue of a sync method is a panic.
        provider.map_or_else(
            || {
                self.get_providers()
                    .into_iter()
                    .flat_map(|p| p.get_models().iter().cloned().collect::<Vec<_>>())
                    .collect()
            },
            |id| {
                self.get_provider(id)
                    .map_or_else(Vec::new, |p| p.get_models().to_vec())
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
                    .flat_map(|p| p.get_all_models().iter().cloned().collect::<Vec<_>>())
                    .collect()
            },
            |id| {
                self.get_provider(id)
                    .map_or_else(Vec::new, |p| p.get_all_models().to_vec())
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
        self.get_provider(provider)
            .and_then(|p| p.get_models().iter().find(|m| m.id == id).cloned())
    }

    /// `getModelOfType(type, provider, id)`.
    #[must_use]
    pub fn get_model_of_type(
        &self,
        kind: crate::provider::ModelType,
        provider: &str,
        id: &str,
    ) -> Option<AnyModel> {
        self.get_provider(provider).and_then(|p| {
            p.get_all_models()
                .iter()
                .find(|m| is_model_type(m, kind) && model_ids(m).1 == id)
                .cloned()
        })
    }

    fn supersede_provider_refresh(&self, provider_id: &str) -> u64 {
        let generation = {
            let mut generations = self
                .registry
                .refresh_generations
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let generation = generations.get(provider_id).map_or(0, |g| *g) + 1;
            generations.insert(provider_id.to_owned(), generation);
            generation
        };
        {
            let mut controllers = self
                .registry
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
        // Upstream `refresh` — `raceWithAbortSignal` per provider op,
        // `{aborted: callerSignal.aborted, errors}` (models.ts:567-624).
        let allow_network = options.allow_network.unwrap_or(true);
        let mut errors = HashMap::new();
        let caller = options.signal.clone().unwrap_or_default();
        if caller.is_cancelled() {
            return ModelsRefreshResult {
                aborted: true,
                errors,
            };
        }
        let selected: Option<std::collections::HashSet<String>> =
            options.providers.map(|p| p.into_iter().collect());
        // Only providers with a `refreshModels` hook run — static
        // providers get no generation bump, controller, or store
        // read (upstream `fetchModels ? … : undefined`).
        let refreshable: Vec<Arc<dyn Provider>> = self
            .get_providers()
            .into_iter()
            .filter(|p| selected.as_ref().is_none_or(|s| s.contains(p.id())))
            .filter(|p| p.refresh_models().is_some())
            .collect();
        // `AbortSignal.any([callerSignal, controller.signal])` — a
        // child token is cancelled by either. Generations and
        // controllers are set up before the concurrent run.
        let mut signals = Vec::with_capacity(refreshable.len());
        for provider in &refreshable {
            let signal = caller.child_token();
            let generation = self.supersede_provider_refresh(provider.id());
            self.registry
                .refresh_controllers
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .insert(provider.id().to_owned(), signal.clone());
            signals.push((generation, signal));
        }
        let operations =
            refreshable
                .iter()
                .zip(&signals)
                .map(|(provider, (generation, signal))| {
                    self.refresh_operation(
                        provider,
                        allow_network,
                        options.force,
                        *generation,
                        signal,
                    )
                });
        // Upstream races the whole join against the caller signal.
        let join = futures::future::join_all(operations);
        tokio::select! {
            outcomes = join => {
                for ((provider, (generation, signal)), outcome) in
                    refreshable.iter().zip(&signals).zip(outcomes)
                {
                    // Cleanup is generation-checked — a supersede
                    // bumps it and owns the new controller slot.
                    let still_mine = self
                        .registry
                        .refresh_generations
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner)
                        .get(provider.id())
                        .copied()
                        == Some(*generation);
                    if still_mine {
                        self.registry
                            .refresh_controllers
                            .lock()
                            .unwrap_or_else(std::sync::PoisonError::into_inner)
                            .remove(provider.id());
                    }
                    // A cancellation is not a provider error.
                    if let Err(error) = outcome
                        && !signal.is_cancelled()
                    {
                        errors.insert(provider.id().to_owned(), error);
                    }
                }
            }
            () = caller.cancelled() => {}
        }
        ModelsRefreshResult {
            aborted: caller.is_cancelled(),
            errors,
        }
    }
    /// One provider's refresh operation (models.ts:577-601): stored
    /// credential → offline restore phase → `resolveRefreshCredential`
    /// → network phase. `raceWithAbortSignal` is a `tokio::select!` on
    /// the linked signal — a stuck provider does not block refresh.
    async fn refresh_operation(
        &self,
        provider: &Arc<dyn Provider>,
        allow_network: bool,
        force: Option<bool>,
        generation: u64,
        signal: &CancellationToken,
    ) -> Result<(), ModelsError> {
        // Upstream races the *whole* operation against the linked
        // signal — a cancel resolves `Ok(())` and cleanup always runs.
        let body = async {
            self.refresh_operation_body(provider, allow_network, force, generation, signal)
                .await
        };
        tokio::select! {
            result = body => result,
            () = signal.cancelled() => Ok(()),
        }
    }

    /// The refresh body — credential read, offline phase, credential
    /// resolution, network phase.
    async fn refresh_operation_body(
        &self,
        provider: &Arc<dyn Provider>,
        allow_network: bool,
        force: Option<bool>,
        generation: u64,
        signal: &CancellationToken,
    ) -> Result<(), ModelsError> {
        let stored_credential = self.registry.credentials.read(provider.id()).await;
        let credential_error = stored_credential.as_ref().err().map(|e| {
            ModelsError::new(
                ModelsErrorCode::Auth,
                format!("Credential store read failed for {}: {e}", provider.id()),
            )
        });
        // Store failures propagate unchanged into `errors` — the
        // provider's whole refresh operation fails.
        let stored = self.registry.store.read(provider.id()).await?;
        self.refresh_phase(
            provider,
            RefreshPhase {
                credential: stored_credential.as_ref().ok().cloned().flatten(),
                stored: stored.clone(),
                allow_network: false,
                force: None,
                generation,
                signal,
            },
        )
        .await?;
        // Upstream reports the credential-read failure only after
        // the offline phase ran (models.ts:582-591).
        if let Some(error) = credential_error {
            return Err(error);
        }
        if !allow_network || signal.is_cancelled() {
            return Ok(());
        }
        let credential = self
            .resolve_refresh_credential(
                provider.as_ref(),
                stored_credential.ok().flatten().as_ref(),
                signal,
            )
            .await?;
        let Some(credential) = credential else {
            return Ok(());
        };
        self.refresh_phase(
            provider,
            RefreshPhase {
                credential: Some(credential),
                stored,
                allow_network: true,
                force,
                generation,
                signal,
            },
        )
        .await
    }

    /// `runProviderRefreshPhase` raced against the linked signal
    /// (`raceWithAbortSignal`).
    async fn refresh_phase(
        &self,
        provider: &Arc<dyn Provider>,
        phase: RefreshPhase<'_>,
    ) -> Result<(), ModelsError> {
        let RefreshPhase {
            credential,
            stored,
            allow_network,
            force,
            generation,
            signal,
        } = phase;
        let publish = self.publish_for(provider.id(), generation, signal.clone());
        let Some(hook) = provider.refresh_models() else {
            return Ok(());
        };
        let run = hook.refresh(RefreshModelsContext {
            credential,
            stored,
            publish,
            allow_network,
            force: if allow_network { force } else { None },
            signal: signal.clone(),
            base_url: provider.base_url().map(str::to_owned),
        });
        tokio::select! {
            result = run => result,
            () = signal.cancelled() => Ok(()),
        }
    }

    /// `resolveRefreshCredential` (models.ts:626-652): an unexpired
    /// stored OAuth credential is used as-is; OAuth refresh flows are
    /// M5 so an expired one skips the provider. Otherwise the apiKey
    /// handler resolves the effective credential.
    async fn resolve_refresh_credential(
        &self,
        provider: &dyn Provider,
        stored: Option<&Credential>,
        signal: &CancellationToken,
    ) -> Result<Option<Credential>, ModelsError> {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0.0, |d| d.as_millis() as f64);
        if let Some(Credential::OAuth(credential)) = stored {
            if provider.auth().oauth.is_none() || now >= credential.expires {
                return Ok(None);
            }
            return Ok(stored.cloned());
        }
        let Some(api_key) = &provider.auth().api_key else {
            return Ok(None);
        };
        let credential = match stored {
            Some(Credential::ApiKey(c)) => Some(c.clone()),
            _ => None,
        };
        let result = (api_key.resolve)(crate::auth::resolve::ApiKeyResolveInput {
            credential: credential.as_ref(),
            env: self.registry.env.as_ref(),
            signal,
        })
        .await
        .map_err(|e| ModelsError::new(ModelsErrorCode::Auth, e))?;
        Ok(result.map(|r| {
            Credential::ApiKey(ApiKeyCredential {
                key: r.auth.api_key,
                env: r.env,
            })
        }))
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
    ) -> Arc<dyn Fn(ModelsPublication) -> BoxFuture<'static, Result<bool, ModelsError>> + Send + Sync>
    {
        let provider_id = provider_id.to_owned();
        let registry = Arc::downgrade(&self.registry);
        Arc::new(move |publication| {
            let provider_id = provider_id.clone();
            let registry = Weak::clone(&registry);
            let signal = signal.clone();
            Box::pin(publish_one(
                provider_id,
                generation,
                publication,
                registry,
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
        let fresh = CancellationToken::new();
        let signal = overrides.signal.as_ref().unwrap_or(&fresh);
        let result = self
            .resolve_provider_auth(provider.as_ref(), overrides, signal)
            .await
            .map_err(Self::auth_wrap)?;
        // `model.headers` merges over the resolved auth headers for
        // any model that carries them (Chat/Image/Classifier).
        let model_headers = match provider_or_model {
            ProviderOrModel::Model(AnyModel::Chat(m)) => m.headers.as_ref(),
            ProviderOrModel::Model(AnyModel::Image(m)) => m.headers.as_ref(),
            ProviderOrModel::Model(AnyModel::Classifier(m)) => m.headers.as_ref(),
            ProviderOrModel::Id(_) => None,
        };
        Ok(match (result, model_headers) {
            (Some(mut result), Some(headers)) => {
                result.auth.headers = merge_headers(result.auth.headers.as_ref(), Some(headers));
                Some(result)
            }
            (result, _) => result,
        })
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
        signal: &CancellationToken,
    ) -> Result<Option<AuthResult>, crate::auth::types::AuthError> {
        let auth = provider.auth();
        let overlay_env;
        let ambient_env: &dyn Env = match overrides.env.as_ref() {
            Some(overlay) => {
                overlay_env =
                    crate::auth::env::OverlayEnv::new(overlay, self.registry.env.as_ref());
                &overlay_env
            }
            None => self.registry.env.as_ref(),
        };

        if let Some(api_key) = overrides.api_key.as_ref()
            && let Some(api_key_auth) = &auth.api_key
        {
            // Upstream `apiKey.resolve` may throw — `getAuth` wraps it
            // into `ModelsError("auth")`.
            return (api_key_auth.resolve)(crate::auth::resolve::ApiKeyResolveInput {
                credential: Some(&ApiKeyCredential {
                    key: Some(secrecy::SecretString::from(api_key.clone())),
                    env: overrides.env.clone(),
                }),
                env: ambient_env,
                signal,
            })
            .await
            .map_err(|e| crate::auth::types::AuthError::models(ModelsErrorCode::Auth, e));
        }

        let stored = self.registry.credentials.read(provider.id()).await?;
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
                match &auth.api_key {
                    Some(a) => (a.resolve)(crate::auth::resolve::ApiKeyResolveInput {
                        credential: Some(&merged),
                        env: ambient_env,
                        signal,
                    })
                    .await
                    .map_err(|e| crate::auth::types::AuthError::models(ModelsErrorCode::Auth, e)),
                    None => Ok(None),
                }
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
            None => match &auth.api_key {
                Some(a) => (a.resolve)(crate::auth::resolve::ApiKeyResolveInput {
                    credential: None,
                    env: ambient_env,
                    signal,
                })
                .await
                .map_err(|e| crate::auth::types::AuthError::models(ModelsErrorCode::Auth, e)),
                None => Ok(None),
            },
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
        let provider = self.require_provider(model)?;
        let fresh = CancellationToken::new();
        let resolution = self
            .resolve_provider_auth(
                &*provider,
                &overrides_for(options),
                options.and_then(|o| o.signal.as_ref()).unwrap_or(&fresh),
            )
            .await
            .map_err(Self::auth_wrap)?;
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
        // `requestModel` — `auth.baseUrl` overrides the catalog URL.
        let request_model = match (model.clone(), resolution.auth.base_url.clone()) {
            (AnyModel::Chat(mut m), Some(base_url)) => {
                m.base_url = base_url;
                AnyModel::Chat(m)
            }
            (AnyModel::Image(mut m), Some(base_url)) => {
                m.base_url = base_url;
                AnyModel::Image(m)
            }
            (AnyModel::Classifier(mut m), Some(base_url)) => {
                m.base_url = base_url;
                AnyModel::Classifier(m)
            }
            (m, _) => m,
        };
        Ok((
            request_model,
            RequestParts {
                api_key,
                headers,
                env,
                identity,
            },
        ))
    }

    /// `getAuth`'s error wrap: an `AuthError::Models` already carries
    /// the upstream code ("oauth", "auth"); anything else is an
    /// `auth` failure.
    fn auth_wrap(error: crate::auth::types::AuthError) -> ModelsError {
        match error {
            crate::auth::types::AuthError::Models(e) => e,
            e => ModelsError::new(ModelsErrorCode::Auth, e.to_string()),
        }
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

    /// Shared `stream`/`streamSimple` scaffold — transcript
    /// normalization, provider lookup, `applyAuth`, and the lazy
    /// stream; `call` assembles the per-flavor request options.
    fn stream_chat(
        self: &Arc<Self>,
        model: &Model,
        context: &Context,
        auth_options: Option<RequestOptions>,
        call: impl FnOnce(
            Arc<dyn Provider>,
            Box<Model>,
            crate::transcript::TranscriptContext,
            RequestParts,
        ) -> AssistantStream
        + Send
        + 'static,
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
            let (applied_model, parts) = models
                .apply_auth(
                    &AnyModel::Chat(Box::new(request_model)),
                    auth_options.as_ref(),
                )
                .await?;
            let AnyModel::Chat(applied) = applied_model else {
                return Err(ModelsError::new(
                    ModelsErrorCode::Provider,
                    "not a chat model",
                ));
            };
            Ok(call(provider, applied, transcript, parts))
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
        self.stream_chat(
            model,
            context,
            options.as_ref().map(request_options),
            move |provider, applied, transcript, parts| {
                let mut stream_options = options.unwrap_or_default();
                stream_options.transform_headers = None;
                stream_options.api_key = parts.api_key;
                stream_options.headers = parts.headers;
                stream_options.env = parts.env;
                stream_options.identity = parts.identity;
                provider.stream(&applied, transcript, stream_options)
            },
        )
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
        let auth_options = options.as_ref().map(|o| request_options(&o.base));
        self.stream_chat(
            model,
            context,
            auth_options,
            move |provider, applied, transcript, parts| {
                let mut base = build_base_options(
                    &applied,
                    &transcript,
                    options.as_ref(),
                    parts
                        .api_key
                        .as_ref()
                        .map(secrecy::ExposeSecret::expose_secret),
                );
                base.headers = parts.headers;
                base.env = parts.env;
                base.identity = parts.identity;
                let mut request_options = options.unwrap_or_default();
                request_options.base = base;
                provider.stream_simple(&applied, transcript, request_options)
            },
        )
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
        self.registry
            .credentials
            .modify(
                provider_id,
                Box::new(move |_existing| Box::pin(async move { Some(credential.clone()) })),
            )
            .await
            .map_err(Self::auth_wrap)?;
        Ok(())
    }

    /// `logout` — remove the provider's credential.
    ///
    /// # Errors
    /// [`ModelsError`] on store failures.
    pub async fn logout(&self, provider_id: &str) -> Result<(), ModelsError> {
        self.registry
            .credentials
            .delete(provider_id)
            .await
            .map_err(Self::auth_wrap)
    }

    /// `checkAuth` — provider configured without refreshing OAuth.
    ///
    /// # Errors
    /// [`ModelsError`] on store failures.
    pub async fn check_auth(&self, provider_id: &str) -> Result<Option<AuthCheck>, ModelsError> {
        let Some(provider) = self.get_provider(provider_id) else {
            return Ok(None);
        };
        let credential = self
            .registry
            .credentials
            .read(provider_id)
            .await
            .map_err(|e| {
                ModelsError::new(
                    ModelsErrorCode::Auth,
                    format!("Credential store read failed for {provider_id}: {e}"),
                )
            })?;
        self.check_provider_auth(provider.as_ref(), credential.as_ref())
            .await
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
        // Credential read + auth check run concurrently per
        // provider; `join_all` keeps provider order (upstream
        // `getAuthenticatedProviders`).
        let checks = providers.into_iter().map(|provider| async move {
            let credential = self
                .registry
                .credentials
                .read(provider.id())
                .await
                .map_err(|e| {
                    ModelsError::new(
                        ModelsErrorCode::Auth,
                        format!("Credential store read failed for {}: {e}", provider.id()),
                    )
                })?;
            let check = self
                .check_provider_auth(provider.as_ref(), credential.as_ref())
                .await?;
            Ok::<_, ModelsError>((provider, check.is_some().then_some(credential)))
        });
        let mut out = Vec::new();
        for result in futures::future::join_all(checks).await {
            let (provider, credential) = result?;
            if let Some(credential) = credential {
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
                kind: AuthType::OAuth,
            }));
        }
        let resolution = self
            .resolve_provider_auth(
                provider,
                &AuthResolutionOverrides::default(),
                &CancellationToken::new(),
            )
            .await
            .map_err(Self::auth_wrap)?;
        Ok(resolution.map(|r| AuthCheck {
            source: r.source,
            kind: AuthType::ApiKey,
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
    /// `apiKey` — `Debug` redacts it.
    pub api_key: Option<secrecy::SecretString>,
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
            .field("api_key", &self.api_key.as_ref().map(|_| "<redacted>"))
            .field(
                "headers",
                &self.headers.as_ref().map(|h| h.keys().collect::<Vec<_>>()),
            )
            .finish_non_exhaustive()
    }
}

fn overrides_for(options: Option<&RequestOptions>) -> AuthResolutionOverrides {
    AuthResolutionOverrides {
        api_key: options.and_then(|o| o.api_key.clone()),
        env: options.and_then(|o| o.env.clone()),
        signal: options.and_then(|o| o.signal.clone()),
    }
}

/// One `publishProviderModels` execution — serialized per provider,
/// persistence before `update`, superseded/cancelled → `false`.
/// One `publish` call — serialized per provider through the chain
/// mutex fetched at publish time, persistence before `update`. The
/// generation and signal are checked live against the registry
/// before persisting and again before `update` (upstream L525/L533):
/// a superseded or cancelled publication returns `false` without
/// writing.
async fn publish_one(
    provider_id: String,
    generation: u64,
    publication: ModelsPublication,
    registry: Weak<Registry>,
    signal: CancellationToken,
) -> Result<bool, ModelsError> {
    let Some(registry) = registry.upgrade() else {
        return Ok(false);
    };
    // Fetch (or create) this provider's chain at publish time — the
    // map entry is the single ordering point for all publications.
    let chain = Arc::clone(
        registry
            .publication_chains
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .entry(provider_id.clone())
            .or_insert_with(|| Arc::new(tokio::sync::Mutex::new(()))),
    );
    let body = async {
        let _guard = chain.lock().await;
        let live = || {
            !signal.is_cancelled()
                && registry
                    .refresh_generations
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .get(provider_id.as_str())
                    .copied()
                    == Some(generation)
        };
        if !live() {
            return Ok(false);
        }
        // Store failures reject the publish — the provider's refresh
        // fails with the store's `model_source` error.
        match publication.persist {
            PersistAction::Delete => registry.store.delete(&provider_id).await?,
            PersistAction::Write(entry) => registry.store.write(&provider_id, &entry).await?,
            PersistAction::Unchanged => {}
        }
        if !live() {
            return Ok(false);
        }
        if let Some(update) = publication.update {
            update();
        }
        Ok(true)
    };
    let result = body.await;
    // Settled tail cleanup — remove the entry only when it is still
    // ours and nobody else holds a clone (map + ours = refcount 2).
    let mut chains = registry
        .publication_chains
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if chains
        .get(provider_id.as_str())
        .is_some_and(|c| Arc::ptr_eq(c, &chain) && Arc::strong_count(c) == 2)
    {
        chains.remove(provider_id.as_str());
    }
    drop(chains);
    result
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
        timestamp: crate::stream::accumulator::wall_ms(),
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
        timestamp: crate::stream::accumulator::wall_ms(),
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

/// `withClientIdentity(models, identity)` — a wrapper sharing the
/// same `Registry`.
///
/// Its `identity` merges over the base handle's for every request,
/// field by field (base → wrapper → options); mutators on either
/// handle are visible to the other.
#[must_use]
#[expect(
    clippy::needless_pass_by_value,
    reason = "the upstream  signature takes ownership"
)]
pub fn with_client_identity(models: &Models, identity: ClientIdentity) -> Arc<Models> {
    Arc::new(Models {
        registry: Arc::clone(&models.registry),
        identity: merge_client_identity(&[models.identity.as_ref(), Some(&identity)]),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::options::{SimpleStreamOptions, StreamOptions};
    use crate::provider::{Provider, ProviderAuth, RefreshModels, RefreshModelsContext};
    use crate::stream::AssistantStream;
    use crate::transcript::TranscriptContext;
    use crate::types::model::Model;

    /// B3 — a publication from a superseded generation returns
    /// `false` and does not write, even when its token was not
    /// cancelled. `publish_for` is exercised directly — through
    /// `refresh` a supersede drops the in-flight hook future in the
    /// `select`, so the publish never executes.
    #[tokio::test]
    async fn superseded_publication_returns_false_without_writing() {
        let models = create_models(CreateModelsOptions::default());
        let signal = CancellationToken::new();
        let publish = models.publish_for("p", 1, signal);
        // Bump the generation without touching the publish's token.
        models
            .registry
            .refresh_generations
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .insert(String::from("p"), 2);
        let result = publish(ModelsPublication {
            persist: PersistAction::Write(ModelsStoreEntry {
                models: Vec::new(),
                checked_at: 0,
            }),
            update: None,
        })
        .await;
        assert!(result.is_ok(), "{result:?}");
        assert!(
            !result.unwrap_or(true),
            "superseded publish must return false"
        );
        let entry = models.registry.store.read("p").await;
        assert!(entry.is_ok(), "{entry:?}");
        assert!(
            entry.ok().flatten().is_none(),
            "superseded publish must not write"
        );
    }

    /// A provider with a `refreshModels` hook that publishes once.
    struct Refreshing {
        auth: ProviderAuth,
    }

    impl Provider for Refreshing {
        fn id(&self) -> &'static str {
            "static"
        }
        fn name(&self) -> &'static str {
            "Static"
        }
        fn auth(&self) -> &ProviderAuth {
            &self.auth
        }
        fn get_models(&self) -> Arc<[Model]> {
            Vec::new().into()
        }
        fn stream(
            &self,
            model: &Model,
            _context: TranscriptContext,
            _options: StreamOptions,
        ) -> AssistantStream {
            lazy_stream_error(model, String::from("no streams in this test"))
        }
        fn stream_simple(
            &self,
            model: &Model,
            _context: TranscriptContext,
            _options: SimpleStreamOptions,
        ) -> AssistantStream {
            lazy_stream_error(model, String::from("no streams in this test"))
        }
        fn refresh_models(&self) -> Option<&dyn RefreshModels> {
            Some(self)
        }
    }

    impl RefreshModels for Refreshing {
        fn refresh(&self, context: RefreshModelsContext) -> BoxFuture<'_, Result<(), ModelsError>> {
            Box::pin(async move {
                (context.publish)(ModelsPublication {
                    persist: PersistAction::Unchanged,
                    update: None,
                })
                .await?;
                Ok(())
            })
        }
    }

    fn fail_once(failed: &Mutex<bool>) -> Result<(), ModelsError> {
        let mut failed = failed
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if std::mem::replace(&mut *failed, false) {
            Err(ModelsError::new(
                ModelsErrorCode::ModelSource,
                "write failed",
            ))
        } else {
            Ok(())
        }
    }

    /// A store that fails `write` once — exercises the error path's
    /// chain cleanup.
    struct FailOnce {
        failed: Mutex<bool>,
    }

    impl ModelsStore for FailOnce {
        fn read<'a>(
            &'a self,
            _provider_id: &'a str,
        ) -> BoxFuture<'a, Result<Option<ModelsStoreEntry>, ModelsError>> {
            Box::pin(async { Ok(None) })
        }
        fn write<'a>(
            &'a self,
            _provider_id: &'a str,
            _entry: &'a ModelsStoreEntry,
        ) -> BoxFuture<'a, Result<(), ModelsError>> {
            Box::pin(async move { fail_once(&self.failed) })
        }
        fn delete<'a>(&'a self, _provider_id: &'a str) -> BoxFuture<'a, Result<(), ModelsError>> {
            Box::pin(async { Ok(()) })
        }
    }

    fn push_tag(order: Arc<Mutex<Vec<u8>>>, tag: u8) -> impl Fn() + Send + 'static {
        move || {
            order
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .push(tag);
        }
    }

    /// B3 — queued publications run strictly in order and the map
    /// empties afterwards, including after a store-error publish.
    #[tokio::test]
    async fn publication_chains_serialize_and_clear() {
        let order = Arc::new(Mutex::new(Vec::<u8>::new()));
        let models = create_models(CreateModelsOptions {
            models_store: Some(Arc::new(FailOnce {
                failed: Mutex::new(true),
            })),
            ..CreateModelsOptions::default()
        });
        models
            .registry
            .refresh_generations
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .insert(String::from("p"), 1);
        let publish = |tag: u8| {
            models.publish_for("p", 1, CancellationToken::new())(ModelsPublication {
                persist: PersistAction::Unchanged,
                update: Some(Box::new(push_tag(Arc::clone(&order), tag))),
            })
        };
        // A runs, B and C wait on the same chain — updates must be
        // strictly ordered.
        let (a, b, c) = futures::future::join3(publish(1), publish(2), publish(3)).await;
        assert!(a.is_ok(), "{a:?}");
        assert!(b.is_ok(), "{b:?}");
        assert!(c.is_ok(), "{c:?}");
        assert_eq!(
            *order
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner),
            vec![1, 2, 3]
        );
        // A store-error publication also clears its tail.
        let failing = models.publish_for("p", 1, CancellationToken::new())(ModelsPublication {
            persist: PersistAction::Write(ModelsStoreEntry {
                models: Vec::new(),
                checked_at: 0,
            }),
            update: None,
        });
        let result = failing.await;
        assert!(result.is_err(), "{result:?}");
        assert!(
            models
                .registry
                .publication_chains
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .is_empty(),
            "chains must be empty after all publications settle"
        );
    }

    /// B3 — `publication_chains` empties once the last tail settles.
    #[tokio::test]
    async fn publication_chains_clear_after_refresh_settles() {
        let models = create_models(CreateModelsOptions::default());
        models.set_provider(Arc::new(Refreshing {
            auth: ProviderAuth {
                api_key: None,
                oauth: None,
            },
        }));
        let result = models.refresh(ModelsRefreshOptions::default()).await;
        assert!(result.errors.is_empty(), "{:?}", result.errors);
        assert!(
            models
                .registry
                .publication_chains
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .is_empty(),
            "publication_chains must be empty after refresh settles"
        );
    }
}
