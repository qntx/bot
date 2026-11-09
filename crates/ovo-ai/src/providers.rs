//! The built-in providers ovo registers (upstream `providers/<id>.ts` and
//! `builtinProviders` in `providers/all.ts`): the providers whose chat
//! protocols ovo implements, each with its embedded catalog.
//!
//! Only chat models are registered; image and classifier models arrive with
//! their protocols. OAuth sign-in arrives with the OAuth flows.

use std::sync::Arc;

use serde_json::{Map, Value};

use crate::api::anthropic_messages::AnthropicMessagesApi;
use crate::api::openai_completions::stream::OpenAiCompletionsApi;
use crate::api::openai_responses::stream::OpenAiResponsesApi;
use crate::auth::env::{Env, env_get};
use crate::auth::resolve::ApiKeyAuth;
use crate::auth::types::{ApiKeyCredential, AuthResult, ModelAuth};
use crate::http::client::HttpClient;
use crate::models::{CreateModelsOptions, Models, create_models};
use crate::models_error::ModelsError;
use crate::provider::{
    ApiDispatch, ChatApi, ModelType, ProviderAuth, ProviderSpec, SpecProvider, create_provider,
};

/// Every built-in provider, in upstream's order.
///
/// # Errors
/// Returns a `model_source` error when an embedded catalog does not parse.
pub fn builtin_providers(http: &Arc<HttpClient>) -> Result<Vec<SpecProvider>, ModelsError> {
    Ok(vec![
        anthropic(http)?,
        deepseek(http)?,
        groq(http)?,
        openai(http)?,
        openrouter(http)?,
        xai(http)?,
    ])
}

/// `builtinModels`: a [`Models`] with every built-in provider registered.
///
/// # Errors
/// As [`builtin_providers`].
pub fn builtin_models(
    options: CreateModelsOptions,
    http: &Arc<HttpClient>,
) -> Result<Arc<Models>, ModelsError> {
    let models = create_models(options);
    for provider in builtin_providers(http)? {
        models.set_provider(Arc::new(provider));
    }
    Ok(models)
}

/// `anthropicProvider`.
///
/// # Errors
/// As [`builtin_providers`].
pub fn anthropic(http: &Arc<HttpClient>) -> Result<SpecProvider, ModelsError> {
    let api_key = ApiKeyAuth::custom("Anthropic API key", Arc::new(resolve_anthropic_auth));
    provider(
        "anthropic",
        "Anthropic",
        "https://api.anthropic.com",
        api_key,
        ApiDispatch::Single(Arc::new(AnthropicMessagesApi::new(Arc::clone(http)))),
    )
}

/// `deepseekProvider`.
///
/// # Errors
/// As [`builtin_providers`].
pub fn deepseek(http: &Arc<HttpClient>) -> Result<SpecProvider, ModelsError> {
    provider(
        "deepseek",
        "DeepSeek",
        "https://api.deepseek.com",
        ApiKeyAuth::env_api_key("DeepSeek API key", &["DEEPSEEK_API_KEY"]),
        completions(http),
    )
}

/// `groqProvider`.
///
/// # Errors
/// As [`builtin_providers`].
pub fn groq(http: &Arc<HttpClient>) -> Result<SpecProvider, ModelsError> {
    provider(
        "groq",
        "Groq",
        "https://api.groq.com/openai/v1",
        ApiKeyAuth::env_api_key("Groq API key", &["GROQ_API_KEY"]),
        completions(http),
    )
}

/// `openaiProvider` (chat models; the Decisions classifier arrives with
/// its protocol).
///
/// # Errors
/// As [`builtin_providers`].
pub fn openai(http: &Arc<HttpClient>) -> Result<SpecProvider, ModelsError> {
    provider(
        "openai",
        "OpenAI",
        "https://api.openai.com/v1",
        ApiKeyAuth::env_api_key("OpenAI API key", &["OPENAI_API_KEY"]),
        responses(http),
    )
}

/// `openrouterProvider` (chat models over both protocols; images and
/// classifiers arrive with their protocols).
///
/// # Errors
/// As [`builtin_providers`].
pub fn openrouter(http: &Arc<HttpClient>) -> Result<SpecProvider, ModelsError> {
    let anthropic: Arc<dyn ChatApi> = Arc::new(AnthropicMessagesApi::new(Arc::clone(http)));
    let completions: Arc<dyn ChatApi> = Arc::new(OpenAiCompletionsApi::new(Arc::clone(http)));
    provider(
        "openrouter",
        "OpenRouter",
        "https://openrouter.ai/api/v1",
        ApiKeyAuth::env_api_key("OpenRouter API key", &["OPENROUTER_API_KEY"]),
        ApiDispatch::ByApi(
            [
                (String::from("anthropic-messages"), anthropic),
                (String::from("openai-completions"), completions),
            ]
            .into(),
        ),
    )
}

/// `xaiProvider`.
///
/// # Errors
/// As [`builtin_providers`].
pub fn xai(http: &Arc<HttpClient>) -> Result<SpecProvider, ModelsError> {
    provider(
        "xai",
        "xAI",
        "https://api.x.ai/v1",
        ApiKeyAuth::env_api_key("xAI API key", &["XAI_API_KEY"]),
        responses(http),
    )
}

fn completions(http: &Arc<HttpClient>) -> ApiDispatch {
    ApiDispatch::Single(Arc::new(OpenAiCompletionsApi::new(Arc::clone(http))))
}

fn responses(http: &Arc<HttpClient>) -> ApiDispatch {
    ApiDispatch::Single(Arc::new(OpenAiResponsesApi::new(Arc::clone(http))))
}

/// `createProvider` for a built-in provider and its embedded chat models.
fn provider(
    id: &str,
    name: &str,
    base_url: &str,
    api_key: ApiKeyAuth,
    api: ApiDispatch,
) -> Result<SpecProvider, ModelsError> {
    create_provider(ProviderSpec {
        id: id.to_owned(),
        name: Some(name.to_owned()),
        base_url: Some(base_url.to_owned()),
        headers: None,
        auth: ProviderAuth {
            api_key: Some(api_key),
            oauth: None,
        },
        models: crate::catalog::models(id, ModelType::Chat)?,
        api: Some(api),
        images: std::collections::BTreeMap::new(),
        classifiers: std::collections::BTreeMap::new(),
        fetch_models: None,
        filter_models: None,
        filter_all_models: None,
    })
}

/// `anthropicApiKeyAuth().resolve`: a stored key, else
/// `ANTHROPIC_AUTH_TOKEN` as a bearer header, else the OAuth token or API
/// key, else workload identity federation. Federation comes last so keys
/// keep winning, as in the Anthropic SDK; the identity-token exchange is
/// M5, so federation variables alone resolve to `None`.
fn resolve_anthropic_auth(
    input: crate::auth::resolve::ApiKeyResolveInput<'_>,
) -> crate::provider::BoxFuture<'_, Result<Option<AuthResult>, String>> {
    Box::pin(async move { Ok(resolve_anthropic_auth_inner(input.credential, input.env)) })
}

fn resolve_anthropic_auth_inner(
    credential: Option<&ApiKeyCredential>,
    env: &dyn Env,
) -> Option<AuthResult> {
    if let Some(credential) = credential.filter(|credential| credential.key.is_some()) {
        return crate::auth::resolve::resolve_api_key_env(&[], Some(credential), env);
    }
    if let Some(token) = env_get(env, "ANTHROPIC_AUTH_TOKEN") {
        let mut headers = Map::new();
        headers.insert(
            "Authorization".into(),
            Value::from(format!("Bearer {token}")),
        );
        let auth = ModelAuth {
            headers: Some(headers),
            ..ModelAuth::default()
        };
        return Some(AuthResult {
            auth,
            env: None,
            source: Some(String::from("ANTHROPIC_AUTH_TOKEN")),
        });
    }
    if let Some(found) = crate::auth::resolve::resolve_api_key_env(
        &["ANTHROPIC_OAUTH_TOKEN", "ANTHROPIC_API_KEY"],
        None,
        env,
    ) {
        return Some(found);
    }
    // Workload identity federation: upstream exchanges the identity
    // token for a credential, which is M5 — with only federation
    // variables present the provider resolves to no auth.
    None
}
