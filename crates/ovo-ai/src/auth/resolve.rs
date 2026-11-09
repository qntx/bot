//! API-key resolution — `env-api-keys.ts`/`helpers.ts`/`resolve.ts`
//! for the M1 provider scope. OAuth flows are M5; the stored-OAuth
//! branch surfaces a `ModelsError` here.

use std::future::Future;
use std::pin::Pin;

use crate::auth::env::{Env, env_get};
use std::sync::Arc;

use crate::auth::types::{ApiKeyCredential, AuthResult, ModelAuth, OAuthCredential};

/// `ProcessCommandRunner` for `!command` values: `sh -c` on unix,
/// `cmd /C` on Windows, 10 s timeout like upstream.
///
/// Only exists with the `host` feature — without it there is no
/// default command runner and `!command` values resolve exactly as a
/// failed command does.
#[cfg(feature = "host")]
#[derive(Clone, Debug)]
pub struct ProcessCommandRunner {
    /// Shell program (`getShellConfig` is host-owned upstream; the
    /// host injects its choice here).
    pub shell: String,
    /// Args pattern: the command is appended as the final arg.
    pub args_prefix: Vec<String>,
    /// Timeout in ms (upstream `timeout: 10000`).
    pub timeout_ms: u64,
}

#[cfg(feature = "host")]
impl Default for ProcessCommandRunner {
    fn default() -> Self {
        Self {
            shell: String::from(if cfg!(windows) { "cmd" } else { "sh" }),
            args_prefix: if cfg!(windows) {
                vec![String::from("/C")]
            } else {
                vec![String::from("-c")]
            },
            timeout_ms: 10_000,
        }
    }
}

#[cfg(feature = "host")]
impl crate::auth::config_value::CommandRunner for ProcessCommandRunner {
    fn run(&self, command: &str) -> Pin<Box<dyn Future<Output = Option<String>> + Send + '_>> {
        let shell = self.shell.clone();
        let args_prefix = self.args_prefix.clone();
        let timeout_ms = self.timeout_ms;
        let command = command.to_owned();
        Box::pin(async move {
            let args = args_prefix.iter().cloned().chain([command]);
            let output = tokio::time::timeout(
                std::time::Duration::from_millis(timeout_ms),
                tokio::process::Command::new(&shell).args(args).output(),
            )
            .await;
            let Ok(Ok(output)) = output else {
                return None;
            };
            if !output.status.success() {
                return None;
            }
            let value = String::from_utf8_lossy(&output.stdout).trim().to_owned();
            if value.is_empty() { None } else { Some(value) }
        })
    }
}

/// `getApiKeyEnvVars` for the M1 provider scope — the full table
/// lands in M5.
#[must_use]
pub fn api_key_env_vars(provider: &str) -> &'static [&'static str] {
    match provider {
        // ANTHROPIC_AUTH_TOKEN participates in env discovery but
        // `getEnvApiKey` skips it (it goes as Authorization: Bearer,
        // not x-api-key).
        "anthropic" => &[
            "ANTHROPIC_AUTH_TOKEN",
            "ANTHROPIC_OAUTH_TOKEN",
            "ANTHROPIC_API_KEY",
        ],
        "openai" => &["OPENAI_API_KEY"],
        "xai" => &["XAI_API_KEY"],
        "deepseek" => &["DEEPSEEK_API_KEY"],
        "groq" => &["GROQ_API_KEY"],
        "openrouter" => &["OPENROUTER_API_KEY"],
        _ => &[],
    }
}

/// `findEnvKeys` — env vars from the provider's list that are set.
#[must_use]
pub fn find_env_keys(provider: &str, env: &dyn Env) -> Vec<String> {
    api_key_env_vars(provider)
        .iter()
        .filter(|name| env_get(env, name).is_some())
        .map(|s| (*s).to_owned())
        .collect()
}

/// `getEnvApiKey` — first found env var; `anthropic` skips
/// `ANTHROPIC_AUTH_TOKEN` (it must not be passed as the API key).
#[must_use]
pub fn get_env_api_key(provider: &str, env: &dyn Env) -> Option<String> {
    let keys = find_env_keys(provider, env);
    if provider == "anthropic" {
        return keys
            .iter()
            .find(|k| k.as_str() != "ANTHROPIC_AUTH_TOKEN")
            .and_then(|k| env_get(env, k));
    }
    keys.first().and_then(|k| env_get(env, k))
}

/// `envApiKeyAuth` resolve step — a stored key wins; else the first
/// set env var in order; `None` = not configured.
#[must_use]
pub fn resolve_api_key_env(
    env_vars: &[&str],
    credential: Option<&ApiKeyCredential>,
    env: &dyn Env,
) -> Option<AuthResult> {
    if let Some(key) = credential.and_then(|c| c.key.as_ref()) {
        return Some(AuthResult {
            auth: ModelAuth {
                api_key: Some(key.clone()),
                ..ModelAuth::default()
            },
            env: credential.and_then(|c| c.env.clone()),
            source: Some(String::from("stored credential")),
        });
    }
    for name in env_vars {
        if let Some(value) = env_get(env, name) {
            return Some(AuthResult {
                auth: ModelAuth {
                    api_key: Some(secrecy::SecretString::from(value)),
                    ..ModelAuth::default()
                },
                env: None,
                source: Some((*name).to_owned()),
            });
        }
    }
    None
}

/// `AuthResolutionOverrides`: `resolveProviderAuth`'s `overrides`.
#[derive(Clone, Debug, Default)]
pub struct AuthResolutionOverrides {
    /// Explicit request-level key — wins over stored/ambient.
    pub api_key: Option<secrecy::SecretString>,
    /// Provider env overlay merged over `Env`.
    pub env: Option<serde_json::Map<String, serde_json::Value>>,
    /// `overrides.signal` — aborted resolutions stop waiting.
    pub signal: Option<tokio_util::sync::CancellationToken>,
}

/// `ApiKeyAuth.resolve` hook.
///
/// A stored key wins, else ambient sources (env vars, custom
/// resolution). `envApiKeyAuth` builds the standard resolver;
/// providers with custom auth (faux, IAM, ADC) supply their own.
/// The input every `ApiKeyResolve` sees — upstream's
impl std::fmt::Debug for ApiKeyResolveInput<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ApiKeyResolveInput")
            .field("credential", &self.credential)
            .finish_non_exhaustive()
    }
}

/// The input every `ApiKeyResolve` sees — upstream's
/// `(credential, env)` plus the caller's `AbortSignal`.
#[derive(Clone, Copy)]
pub struct ApiKeyResolveInput<'a> {
    /// The stored credential, when one exists.
    pub credential: Option<&'a ApiKeyCredential>,
    /// Ambient environment for key lookup.
    pub env: &'a dyn Env,
    /// The caller's `AbortSignal` (linked refresh token, request
    /// signal, or a fresh token for checks).
    pub signal: &'a tokio_util::sync::CancellationToken,
}

/// `ApiKeyAuth.resolve` — async so callers can abort it.
pub type ApiKeyResolve = Arc<
    dyn for<'a> Fn(
            ApiKeyResolveInput<'a>,
        ) -> crate::provider::BoxFuture<'a, Result<Option<AuthResult>, String>>
        + Send
        + Sync,
>;

/// `ApiKeyAuth`-shaped provider hook (`name` + `resolve`), the M1
/// subset of upstream's `ApiKeyAuth` (`login`/`check` are app-side).
pub struct ApiKeyAuth {
    /// Display name ("Anthropic API key").
    pub name: String,
    /// `resolve` — ambient/stored resolution.
    pub resolve: ApiKeyResolve,
}

impl Clone for ApiKeyAuth {
    fn clone(&self) -> Self {
        Self {
            name: self.name.clone(),
            resolve: Arc::clone(&self.resolve),
        }
    }
}

impl std::fmt::Debug for ApiKeyAuth {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ApiKeyAuth")
            .field("name", &self.name)
            .finish_non_exhaustive()
    }
}

impl ApiKeyAuth {
    /// `envApiKeyAuth(name, envVars)` — stored key → env vars in order.
    #[must_use]
    pub fn env_api_key(name: &str, env_vars: &[&str]) -> Self {
        let vars: Vec<String> = env_vars.iter().map(|s| (*s).to_owned()).collect();
        Self {
            name: name.to_owned(),
            resolve: Arc::new(move |input| {
                let vars = vars.clone();
                Box::pin(async move {
                    Ok(resolve_api_key_env(
                        &vars.iter().map(String::as_str).collect::<Vec<_>>(),
                        input.credential,
                        input.env,
                    ))
                })
            }),
        }
    }

    /// A custom resolver (faux's `resolve: async () => ({ auth: {} })`
    /// and friends).
    #[must_use]
    pub fn custom(name: &str, resolve: ApiKeyResolve) -> Self {
        Self {
            name: name.to_owned(),
            resolve,
        }
    }
}

/// The future `toAuth` returns.
pub type OauthToAuthFuture = Pin<Box<dyn Future<Output = ModelAuth> + Send>>;

/// `toAuth(credential)` — request auth from a valid credential.
pub type OauthToAuth = Arc<dyn Fn(&OAuthCredential) -> OauthToAuthFuture + Send + Sync>;

/// `OAuthAuth` — the M1 stub: `name` plus `toAuth`/`refresh` hooks so
/// the stored-OAuth branch in `resolve_provider_auth` is modeled.
/// Flows land in M5.
pub struct OAuthAuth {
    /// Display name ("Anthropic (Claude Pro/Max)").
    pub name: String,
    /// `toAuth(credential)` — request auth from a valid credential.
    pub to_auth: OauthToAuth,
}

impl Clone for OAuthAuth {
    fn clone(&self) -> Self {
        Self {
            name: self.name.clone(),
            to_auth: Arc::clone(&self.to_auth),
        }
    }
}

impl std::fmt::Debug for OAuthAuth {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OAuthAuth")
            .field("name", &self.name)
            .finish_non_exhaustive()
    }
}
