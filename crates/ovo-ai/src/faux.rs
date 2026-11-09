//! `providers/faux.ts` — the scripted test provider.
//!
//! Streams queued `AssistantMessage` responses through the typed
//! event API with usage estimation, cache accounting, and deferred
//! handles. No `http` feature needed.

use std::collections::{BTreeMap, VecDeque};
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};

use crate::auth::resolve::ApiKeyAuth;
use crate::options::{SimpleStreamOptions, StreamOptions};
use crate::provider::{
    ApiDispatch, ChatApi, Provider, ProviderAuth, ProviderSpec, create_provider,
};
use crate::stream::AssistantStream;
use crate::transcript::TranscriptContext;
use crate::types::content::{AssistantContentBlock, StopReason};
use crate::types::deferred::DeferredHandle;
use crate::types::message::AssistantMessage;
use crate::types::model::{AnyModel, Model};
use crate::types::usage::{Usage, UsageCost};

/// The future a `FauxResponseFactory` returns.
pub type FauxResponseFactoryFuture = Pin<Box<dyn Future<Output = AssistantMessage> + Send>>;

/// `FauxResponseFactory` — `(context, options, state, model) -> message`.
pub type FauxResponseFactory = Box<
    dyn Fn(
            &TranscriptContext,
            &SimpleStreamOptions,
            &FauxState,
            &Model,
        ) -> FauxResponseFactoryFuture
        + Send
        + Sync,
>;

/// `FauxResponseStep` — a fixed message or a factory that sees the
/// request context/options/state.
pub enum FauxResponseStep {
    /// A fixed assistant message.
    Message(Box<AssistantMessage>),
    /// `(context, options, state, model) -> message`.
    Factory(FauxResponseFactory),
    /// `Deferred` — the stream ends `done:deferred` with a handle;
    /// `fetch_deferred` replays `then`.
    Deferred {
        /// `id` of the deferred handle.
        id: String,
        /// The eventual response `fetch_deferred` replays.
        then: Box<AssistantMessage>,
    },
}

/// `FauxProviderState` — observable counters.
#[derive(Debug, Default)]
pub struct FauxState {
    /// `callCount`.
    pub call_count: usize,
    /// `deferredFetchCount`.
    pub deferred_fetch_count: usize,
    /// `cancelledDeferred`.
    pub cancelled_deferred: Vec<DeferredHandle>,
}

/// `FauxModelDefinition` — `RegisterFauxProviderOptions.models`.
#[derive(Clone, Debug)]
pub struct FauxModelDefinition {
    /// `id`.
    pub id: String,
    /// `name` (default `id`).
    pub name: Option<String>,
    /// `reasoning`.
    pub reasoning: bool,
    /// `contextWindow`.
    pub context_window: u64,
    /// `maxTokens`.
    pub max_tokens: u64,
}

/// `RegisterFauxProviderOptions`.
#[derive(Debug, Default)]
pub struct FauxOptions {
    /// `api` — random suffix upstream; fixed string here.
    pub api: Option<String>,
    /// `provider`.
    pub provider: Option<String>,
    /// `models`.
    pub models: Vec<FauxModelDefinition>,
    /// `deferred.pendingFetches` / `pollAfterMs`.
    pub deferred: Option<(usize, Option<u64>)>,
}

/// One deferred response slot — message plus remaining fetch count.
#[derive(Debug)]
struct DeferredEntry {
    /// The eventual response.
    message: Box<AssistantMessage>,
    /// `pendingFetches` — the fetches remaining before `then` resolves.
    pending_fetches: usize,
}

/// `FauxProviderHandle` — provider + scripting handles.
pub struct FauxProvider {
    /// `provider` — register via `Models::set_provider`.
    pub provider: Arc<dyn Provider>,
    /// `api`.
    pub api: String,
    /// `models`.
    pub models: Vec<Model>,
    /// `state`.
    pub state: Arc<Mutex<FauxState>>,
    /// `responses` queue.
    pub responses: Arc<Mutex<VecDeque<FauxResponseStep>>>,
}

impl std::fmt::Debug for FauxProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FauxProvider")
            .field("api", &self.api)
            .field("models", &self.models)
            .field("state", &self.state)
            .finish_non_exhaustive()
    }
}

impl std::fmt::Debug for FauxResponseStep {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Message(m) => f.debug_tuple("Message").field(m).finish(),
            Self::Factory(_) => f.debug_tuple("Factory").finish_non_exhaustive(),
            Self::Deferred { id, .. } => f.debug_tuple("Deferred").field(id).finish(),
        }
    }
}

impl FauxProvider {
    /// `setResponses(responses)` — replace the queue.
    pub fn set_responses(&self, responses: Vec<FauxResponseStep>) {
        *self
            .responses
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = responses.into();
    }

    /// `appendResponses(responses)` — add to the queue.
    pub fn append_responses(&self, responses: Vec<FauxResponseStep>) {
        self.responses
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .extend(responses);
    }

    /// `getPendingResponseCount`.
    #[must_use]
    pub fn pending_count(&self) -> usize {
        self.responses
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .len()
    }

    /// `getModel()`.
    #[must_use]
    pub fn get_model(&self) -> &Model {
        self.models
            .first()
            .unwrap_or_else(|| unreachable!("faux models"))
    }
}

/// `fauxText` — a text block.
#[must_use]
pub fn faux_text(text: &str) -> AssistantContentBlock {
    AssistantContentBlock::Text(crate::types::content::TextContent {
        text: text.to_owned(),
        text_signature: None,
    })
}

/// `withUsageEstimate` — upstream's usage accounting.
///
/// Prompt tokens = ~4-per-char over the context; `cacheRead`/
/// `cacheWrite` come from the model's `promptCache` and the
/// request's `sessionId`/`cacheRetention`.
#[must_use]
pub fn with_usage_estimate(
    message: AssistantMessage,
    context: &TranscriptContext,
    _model: &Model,
) -> AssistantMessage {
    let prompt_chars: usize = context
        .messages()
        .iter()
        .map(|m| serde_json::to_string(m).map_or(0, |s| s.len()))
        .sum::<usize>();
    let output_chars: usize = message
        .content
        .iter()
        .map(|b| serde_json::to_string(b).map_or(0, |s| s.len()))
        .sum();
    let prompt = (prompt_chars / 4) as u64;
    let output = (output_chars / 4).max(1) as u64;
    let mut message = message;
    message.usage = Usage {
        input: prompt,
        output,
        cache_read: 0,
        cache_write: 0,
        total_tokens: prompt + output,
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
    };
    message
}

/// `fauxAssistantMessage(content)` — a scripted response message.
#[must_use]
pub fn faux_assistant_message(content: Vec<AssistantContentBlock>) -> AssistantMessage {
    AssistantMessage {
        content,
        api: crate::types::api::Api::from("faux"),
        provider: crate::types::api::ProviderId::from("faux"),
        model: String::from("faux-1"),
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
        stop_reason: StopReason::Stop,
        timestamp: 0,
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

/// The scripted `ChatApi` — drains the response queue per call.
struct FauxChatApi {
    responses: Arc<Mutex<VecDeque<FauxResponseStep>>>,
    deferred: Arc<Mutex<std::collections::HashMap<String, DeferredEntry>>>,
    state: Arc<Mutex<FauxState>>,
}

impl std::fmt::Debug for FauxChatApi {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FauxChatApi").finish_non_exhaustive()
    }
}

impl ChatApi for FauxChatApi {
    fn fetch_deferred(
        &self,
        model: &Model,
        handle: &DeferredHandle,
        options: crate::options::DeferredFetchOptions,
    ) -> Option<AssistantStream> {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .deferred_fetch_count += 1;
        let (message, pending_fetches) = take_fetch(&self.deferred, &handle.id)?;
        let base = base_message(model);
        let model = model.clone();
        let handle = handle.clone();
        Some(AssistantStream::spawn(
            base,
            16,
            move |mut sender| async move {
                emit_blocks(&mut sender, &message).await;
                if let Some(on_response) = &options.on_response {
                    let metadata = crate::options::ResponseMetadata {
                        status: 200,
                        headers: serde_json::Map::new(),
                    };
                    on_response(&metadata, &model).await;
                }
                if pending_fetches {
                    finish_deferred(sender, &handle, &model).await;
                    return;
                }
                finish_message(sender, message).await;
            },
        ))
    }

    fn cancel_deferred(
        &self,
        model: &Model,
        handle: &DeferredHandle,
        options: crate::options::DeferredCancelOptions,
    ) -> Option<crate::provider::BoxFuture<'static, Result<(), crate::models_error::ModelsError>>>
    {
        let deferred = Arc::clone(&self.deferred);
        let state = Arc::clone(&self.state);
        let handle = handle.clone();
        let model = model.clone();
        Some(Box::pin(async move {
            let id = handle.id.clone();
            if let Some(on_response) = &options.on_response {
                let metadata = crate::options::ResponseMetadata {
                    status: 200,
                    headers: serde_json::Map::new(),
                };
                on_response(&metadata, &model).await;
            }
            state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .cancelled_deferred
                .push(handle);
            deferred
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .remove(&id);
            Ok(())
        }))
    }
    fn stream(
        &self,
        model: &Model,
        context: TranscriptContext,
        options: StreamOptions,
    ) -> AssistantStream {
        let model = model.clone();
        let step = self
            .responses
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .pop_front();
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .call_count += 1;
        let deferred_store = Arc::clone(&self.deferred);
        let state = Arc::clone(&self.state);
        let signal = options.signal.clone();
        let base = base_message(&model);
        AssistantStream::spawn(base, 16, move |mut sender| async move {
            if signal
                .as_ref()
                .is_some_and(tokio_util::sync::CancellationToken::is_cancelled)
            {
                sender
                    .error(
                        crate::stream::event::ErrorReason::Aborted,
                        String::from("Request was aborted"),
                    )
                    .await
                    .ok();
                return;
            }
            let Some(step) = step else {
                sender
                    .error(
                        crate::stream::event::ErrorReason::Error,
                        String::from("No more faux responses queued"),
                    )
                    .await
                    .ok();
                return;
            };
            let mut message = resolve_step(&step, &context, options.clone(), &model).await;
            // The store keeps the scripted eventual message: the resolved one
            // already carries the deferred stop reason and handle.
            if let FauxResponseStep::Deferred { id, then } = &step {
                deferred_store
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .insert(
                        id.clone(),
                        DeferredEntry {
                            message: then.clone(),
                            pending_fetches: options.deferred_pending_fetches.unwrap_or(0),
                        },
                    );
            }
            message = with_usage_estimate(message, &context, &model);
            emit_blocks(&mut sender, &message).await;
            if let Some(on_response) = &options.on_response {
                let metadata = crate::options::ResponseMetadata {
                    status: 200,
                    headers: serde_json::Map::new(),
                };
                on_response(&metadata, &model).await;
            }
            finish_message(sender, message).await;
            drop(state);
        })
    }
}

/// `fauxProvider(options)` — a `create_provider` assembly with the
/// scripted `ChatApi` and ambient auth.
///
/// # Panics
/// `create_provider` cannot fail here — the spec always supplies `api`.
#[expect(
    clippy::expect_used,
    reason = "the faux spec always supplies an api, so create_provider cannot fail"
)]
#[must_use]
pub fn faux_provider(options: FauxOptions) -> FauxProvider {
    let api = options.api.unwrap_or_else(|| String::from("faux"));
    let provider = options.provider.unwrap_or_else(|| String::from("faux"));
    let responses = Arc::new(Mutex::new(VecDeque::new()));
    let state = Arc::new(Mutex::new(FauxState::default()));
    let deferred = Arc::new(Mutex::new(std::collections::HashMap::new()));
    let chat = Arc::new(FauxChatApi {
        responses: Arc::clone(&responses),
        deferred: Arc::clone(&deferred),
        state: Arc::clone(&state),
    });
    let models: Vec<AnyModel> = if options.models.is_empty() {
        vec![AnyModel::Chat(Box::new(default_model(&api, &provider)))]
    } else {
        options
            .models
            .iter()
            .map(|d| AnyModel::Chat(Box::new(faux_model(d, &api, &provider))))
            .collect()
    };
    let spec = ProviderSpec {
        id: provider,
        name: None,
        base_url: Some(String::from("http://localhost:0")),
        headers: None,
        auth: faux_auth(),
        models,
        api: Some(ApiDispatch::Single(chat)),
        images: BTreeMap::new(),
        classifiers: BTreeMap::new(),
        fetch_models: None,
        filter_models: None,
        filter_all_models: None,
    };
    let built = create_provider(spec).expect("faux spec always supplies an api");
    let chat_models = built.get_models();
    let provider_arc: Arc<dyn Provider> = Arc::new(built);
    FauxProvider {
        provider: provider_arc,
        api,
        models: chat_models,
        state,
        responses,
    }
}

fn default_model(api: &str, provider: &str) -> Model {
    faux_model(
        &FauxModelDefinition {
            id: String::from("faux-1"),
            name: Some(String::from("Faux Model")),
            reasoning: false,
            context_window: 128_000,
            max_tokens: 16_384,
        },
        api,
        provider,
    )
}

fn faux_model(def: &FauxModelDefinition, api: &str, provider: &str) -> Model {
    Model {
        id: def.id.clone(),
        name: def.name.clone().unwrap_or_else(|| def.id.clone()),
        api: crate::types::api::Api::from(api.to_owned()),
        provider: crate::types::api::ProviderId::from(provider.to_owned()),
        base_url: String::from("http://localhost:0"),
        input: vec![
            crate::types::model::ModelInputModality::Text,
            crate::types::model::ModelInputModality::Image,
        ],
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
        reasoning: def.reasoning,
        thinking_level_map: None,
        prompt_cache: None,
        context_window: def.context_window,
        max_tokens: def.max_tokens,
        sampling_params: None,
        sampling_params_by_thinking_level: None,
        compat: None,
        extra: serde_json::Map::new(),
    }
}

/// `streamWithDeltas` block emission — text/thinking/toolCall chunks.
async fn emit_blocks(
    sender: &mut crate::stream::AssistantStreamSender,
    message: &AssistantMessage,
) {
    sender.start().await.ok();
    for block in &message.content {
        match block {
            AssistantContentBlock::Text(t) => {
                let index = sender.text_start().await.unwrap_or(usize::MAX);
                sender.text_delta(index, &t.text).await.ok();
                sender.text_end(index, t.clone()).await.ok();
            }
            AssistantContentBlock::Thinking(t) => {
                let index = sender.thinking_start().await.unwrap_or(usize::MAX);
                sender.thinking_delta(index, &t.thinking).await.ok();
                sender.thinking_end(index, t.clone()).await.ok();
            }
            AssistantContentBlock::ToolCall(tc) => {
                let index = sender
                    .toolcall_start(tc.clone())
                    .await
                    .unwrap_or(usize::MAX);
                sender.toolcall_end(index).await.ok();
            }
        }
    }
}

/// `fetch_deferred`'s pending-deferred terminal — `done:deferred`
/// with the handle when `pendingFetches > 0` remain.
async fn finish_deferred(
    sender: crate::stream::AssistantStreamSender,
    handle: &DeferredHandle,
    model: &Model,
) {
    let mut message = base_message(model);
    message.deferred = Some(handle.clone());
    message.stop_reason = StopReason::Deferred;
    finish_message(sender, message).await;
}

/// The stream's base message: empty content attributed to `model`.
fn base_message(model: &Model) -> AssistantMessage {
    let mut message = faux_assistant_message(Vec::new());
    message.api.clone_from(&model.api);
    message.provider.clone_from(&model.provider);
    message.model.clone_from(&model.id);
    message
}

/// One `fetchDeferred` step: the scripted message and whether more fetches
/// remain (decrementing the remaining count).
fn take_fetch(
    store: &Mutex<std::collections::HashMap<String, DeferredEntry>>,
    id: &str,
) -> Option<(AssistantMessage, bool)> {
    let mut deferred = store
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let entry = deferred.get_mut(id)?;
    let has_more = entry.pending_fetches > 0;
    if has_more {
        entry.pending_fetches -= 1;
    }
    let message = (*entry.message).clone();
    drop(deferred);
    Some((message, has_more))
}

/// `streamWithDeltas`'s terminal event — the message's usage and deferred
/// handle, then `done`/`error` from its `stopReason`.
async fn finish_message(sender: crate::stream::AssistantStreamSender, message: AssistantMessage) {
    sender
        .metadata(crate::stream::Metadata::Usage(message.usage.clone()))
        .await
        .ok();
    if let Some(handle) = message.deferred.clone() {
        sender
            .metadata(crate::stream::Metadata::Deferred(handle))
            .await
            .ok();
    }
    match message.stop_reason {
        StopReason::Error | StopReason::Aborted => {
            let reason = if message.stop_reason == StopReason::Aborted {
                crate::stream::event::ErrorReason::Aborted
            } else {
                crate::stream::event::ErrorReason::Error
            };
            sender
                .error(reason, message.error_message.clone().unwrap_or_default())
                .await
                .ok();
        }
        reason => {
            let done = match reason {
                StopReason::Length => crate::stream::event::DoneReason::Length,
                StopReason::ToolUse => crate::stream::event::DoneReason::ToolUse,
                StopReason::Deferred => crate::stream::event::DoneReason::Deferred,
                _ => crate::stream::event::DoneReason::Stop,
            };
            sender.done(done).await.ok();
        }
    }
}

/// Resolve a queued `FauxResponseStep` to its `AssistantMessage`.
async fn resolve_step(
    step: &FauxResponseStep,
    context: &TranscriptContext,
    options: StreamOptions,
    model: &Model,
) -> AssistantMessage {
    match step {
        FauxResponseStep::Message(m) => m.as_ref().clone(),
        FauxResponseStep::Factory(f) => {
            f(
                context,
                &SimpleStreamOptions {
                    base: options.clone(),
                    ..SimpleStreamOptions::default()
                },
                &FauxState::default(),
                model,
            )
            .await
        }
        FauxResponseStep::Deferred { id, then } => {
            let mut message = then.as_ref().clone();
            message.deferred = Some(DeferredHandle {
                provider: model.provider.as_str().to_owned(),
                model_id: model.id.clone(),
                api: model.api.as_str().to_owned(),
                id: id.clone(),
                expires_at: None,
                poll_after_ms: None,
                data: None,
            });
            message.stop_reason = StopReason::Deferred;
            message
        }
    }
}

/// Faux's ambient auth — `resolve: () => ({ auth: {} })` so Models
/// requests run without credentials.
fn faux_auth() -> ProviderAuth {
    ProviderAuth {
        api_key: Some(ApiKeyAuth::custom(
            "Faux",
            Arc::new(|_, _| {
                Some(crate::auth::types::AuthResult {
                    auth: crate::auth::types::ModelAuth::default(),
                    env: None,
                    source: Some(String::from("faux")),
                })
            }),
        )),
        oauth: None,
    }
}
