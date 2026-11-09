//! API-key resolution — `env-api-keys.ts`/`helpers.ts`/`resolve.ts`
//! for the M1 provider scope. OAuth flows are M5; the stored-OAuth
//! branch surfaces a `ModelsError` here.

use Future;

use crate::auth::config_value::CommandRunner;
use crate::auth::env::{Env, env_get};
use crate::auth::store::CredentialStore;
use crate::auth::types::{ApiKeyCredential, AuthError, AuthResult, Credential, ModelAuth};
use crate::models_error::ModelsErrorCode;

/// `ProcessCommandRunner` for `!command` values: `sh -c` on unix,
/// `cmd /C` on Windows, 10 s timeout like upstream.
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

impl CommandRunner for ProcessCommandRunner {
    fn run(
        &self,
        command: &str,
    ) -> std::pin::Pin<Box<dyn Future<Output = Option<String>> + Send + '_>> {
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
        use secrecy::ExposeSecret;
        return Some(AuthResult {
            auth: ModelAuth {
                api_key: Some(key.expose_secret().to_owned()),
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
                    api_key: Some(value),
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
    pub api_key: Option<String>,
    /// Provider env overlay merged over `Env`.
    pub env: Option<serde_json::Map<String, serde_json::Value>>,
    /// `minOAuthValidityMs` (defaults to five minutes).
    pub min_oauth_validity_ms: Option<f64>,
}

/// `resolveProviderAuth` (M1 scope): `overrides.apiKey` → stored
/// credential → ambient env vars.
///
/// A stored `api_key` credential owns the provider (no silent env
/// fallback); a stored `oauth` credential surfaces the OAuth refresh
/// path as a `ModelsError` (`"oauth"`) — flows land in M5.
///
/// # Errors
/// [`AuthError::Models`] for store failures (`"auth"`) or a stored
/// OAuth credential (`"oauth"`); [`AuthError::Io`]/`InvalidData`
/// propagate from the store.
pub async fn resolve_provider_auth(
    provider_id: &str,
    store: &dyn CredentialStore,
    env: &dyn Env,
    overrides: Option<&AuthResolutionOverrides>,
) -> Result<Option<AuthResult>, AuthError> {
    // Overlay env when provided — the injected `Env` stays the base.
    let overlay_env = overrides
        .and_then(|o| o.env.as_ref())
        .map(|overlay| crate::auth::env::OverlayEnv::new(overlay, env));
    let env: &dyn Env = if let Some(o) = &overlay_env { o } else { env };
    let env_vars = api_key_env_vars(provider_id);

    if let Some(api_key) = overrides.and_then(|o| o.api_key.as_ref()) {
        let credential = ApiKeyCredential {
            key: Some(secrecy::SecretString::from(api_key.clone())),
            env: overrides.and_then(|o| o.env.clone()),
        };
        return Ok(resolve_api_key_env(env_vars, Some(&credential), env));
    }

    let stored = store.read(provider_id).await.map_err(|e| {
        AuthError::models(
            ModelsErrorCode::Auth,
            format!("Credential store read failed for {provider_id}: {e}"),
        )
    })?;
    match stored {
        Some(Credential::ApiKey(credential)) => {
            Ok(resolve_api_key_env(env_vars, Some(&credential), env))
        }
        Some(Credential::OAuth(_)) => Err(AuthError::models(
            ModelsErrorCode::Oauth,
            format!("OAuth resolution for {provider_id} is not implemented until M5"),
        )),
        None => Ok(resolve_api_key_env(env_vars, None, env)),
    }
}

/// `ApiKeyAuth`-shaped provider hook (`name` + `resolve`), the M1
/// subset of upstream's `ApiKeyAuth` (`login`/`check` are app-side).
#[derive(Debug)]
pub struct ApiKeyAuth {
    /// Display name ("Anthropic API key").
    pub name: String,
    /// Env vars checked in order (the provider's
    /// [`api_key_env_vars`] list or a custom set).
    pub env_vars: Vec<String>,
}
