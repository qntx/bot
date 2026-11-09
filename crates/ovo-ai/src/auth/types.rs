//! Auth data types — `auth/types.ts` port.

use secrecy::SecretString;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::models_error::{ModelsError, ModelsErrorCode};

/// `ModelAuth`: request auth for a single model request.
#[expect(
    clippy::derive_partial_eq_without_eq,
    reason = "Map<String, Value> holds f64 — Eq is invalid"
)]
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ModelAuth {
    /// `apiKey`.
    pub api_key: Option<String>,
    /// `headers`.
    pub headers: Option<Map<String, Value>>,
    /// `baseUrl`.
    pub base_url: Option<String>,
}

/// `ApiKeyCredential`: `{type: "api_key", key?, env?}`. `key` is a
/// [`SecretString`] — `Debug` redacts it; plaintext serialization
/// happens only at the store boundary.
#[derive(Clone)]
pub struct ApiKeyCredential {
    /// The stored key — a literal, `$ENV`/`${ENV}` template, or a
    /// `!command` (unresolved until `resolve_config_value` runs).
    pub key: Option<SecretString>,
    /// Provider-scoped environment/config values.
    pub env: Option<Map<String, Value>>,
}

impl PartialEq for ApiKeyCredential {
    fn eq(&self, other: &Self) -> bool {
        use secrecy::ExposeSecret;
        self.key.as_ref().map(|k| k.expose_secret().to_owned())
            == other.key.as_ref().map(|k| k.expose_secret().to_owned())
            && self.env == other.env
    }
}

impl std::fmt::Debug for ApiKeyCredential {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ApiKeyCredential")
            .field("key", &self.key.as_ref().map(|_| "<redacted>"))
            .field("env", &self.env)
            .finish()
    }
}

/// `OAuthCredential`: `{type: "oauth", access, refresh, expires, ...}`
/// with arbitrary extra fields preserved (upstream `[key: string]:
/// unknown`). Secrets are redacted in `Debug`.
#[derive(Clone)]
pub struct OAuthCredential {
    /// Bearer token for the current window.
    pub access: SecretString,
    /// Token exchanged for new credentials.
    pub refresh: SecretString,
    /// Absolute expiry timestamp (epoch ms, JS `number`).
    pub expires: f64,
    /// Extra provider-specific fields (`[key]: unknown`).
    pub extra: Map<String, Value>,
}

impl PartialEq for OAuthCredential {
    fn eq(&self, other: &Self) -> bool {
        use secrecy::ExposeSecret;
        self.access.expose_secret() == other.access.expose_secret()
            && self.refresh.expose_secret() == other.refresh.expose_secret()
            && self.expires == other.expires
            && self.extra == other.extra
    }
}

impl std::fmt::Debug for OAuthCredential {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OAuthCredential")
            .field("access", &"<redacted>")
            .field("refresh", &"<redacted>")
            .field("expires", &self.expires)
            .field("extra", &self.extra)
            .finish()
    }
}

/// `Credential`: one type-tagged credential per provider — the
/// auth.json shape.
#[derive(Clone, Debug, PartialEq)]
pub enum Credential {
    /// `{type: "api_key"}`.
    ApiKey(ApiKeyCredential),
    /// `{type: "oauth"}`.
    OAuth(OAuthCredential),
}

impl Credential {
    /// The `"type"` tag.
    #[must_use]
    pub const fn type_tag(&self) -> &'static str {
        match self {
            Self::ApiKey(_) => "api_key",
            Self::OAuth(_) => "oauth",
        }
    }
}

/// Serialize as the auth.json object; secrets are written in
/// plaintext (this impl exists for the store boundary only).
impl Serialize for Credential {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use secrecy::ExposeSecret;
        match self {
            Self::ApiKey(c) => {
                let mut map = Map::new();
                map.insert(String::from("type"), Value::from("api_key"));
                if let Some(key) = &c.key {
                    map.insert(
                        String::from("key"),
                        Value::from(key.expose_secret().to_owned()),
                    );
                }
                if let Some(env) = &c.env {
                    map.insert(String::from("env"), Value::Object(env.clone()));
                }
                map.serialize(serializer)
            }
            Self::OAuth(c) => {
                let mut map = c.extra.clone();
                map.insert(String::from("type"), Value::from("oauth"));
                map.insert(
                    String::from("access"),
                    Value::from(c.access.expose_secret().to_owned()),
                );
                map.insert(
                    String::from("refresh"),
                    Value::from(c.refresh.expose_secret().to_owned()),
                );
                map.insert(String::from("expires"), Value::from(c.expires));
                map.serialize(serializer)
            }
        }
    }
}

impl<'de> Deserialize<'de> for Credential {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        use secrecy::SecretString as S;
        let value = Map::deserialize(deserializer)?;
        match value.get("type").and_then(Value::as_str) {
            Some("api_key") => Ok(Self::ApiKey(ApiKeyCredential {
                key: value
                    .get("key")
                    .and_then(Value::as_str)
                    .map(|k| S::from(k.to_owned())),
                env: value.get("env").and_then(|e| e.as_object().cloned()),
            })),
            Some("oauth") => {
                let mut extra = value;
                extra.shift_remove("type");
                let access = extra
                    .shift_remove("access")
                    .and_then(|v| v.as_str().map(str::to_owned))
                    .map(S::from);
                let refresh = extra
                    .shift_remove("refresh")
                    .and_then(|v| v.as_str().map(str::to_owned))
                    .map(S::from);
                let expires = extra.shift_remove("expires").and_then(|v| v.as_f64());
                match (access, refresh, expires) {
                    (Some(access), Some(refresh), Some(expires)) => {
                        Ok(Self::OAuth(OAuthCredential {
                            access,
                            refresh,
                            expires,
                            extra,
                        }))
                    }
                    _ => Err(serde::de::Error::custom(
                        "invalid oauth credential: access/refresh/expires required",
                    )),
                }
            }
            _ => Err(serde::de::Error::custom(
                "invalid credential: type must be api_key or oauth",
            )),
        }
    }
}

/// `CredentialInfo`: non-secret metadata for account enumeration.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CredentialInfo {
    /// The provider id the credential is stored under.
    pub provider_id: String,
    /// `"api_key"` | `"oauth"`.
    pub type_: &'static str,
}

/// `AuthResult`: resolved request auth.
#[derive(Clone, Debug, PartialEq)]
pub struct AuthResult {
    /// The resolved request auth.
    pub auth: ModelAuth,
    /// Provider-scoped env resolved from credentials/ambient context.
    pub env: Option<Map<String, Value>>,
    /// Human-readable label: `"ANTHROPIC_API_KEY"`, `"OAuth"`,
    /// `"stored credential"`.
    pub source: Option<String>,
}

/// `AuthError`: auth-path failures wrap storage/resolution errors in
/// `ModelsError` codes like upstream (`"auth"`, `"oauth"`).
#[derive(Debug, thiserror::Error)]
pub enum AuthError {
    /// Store or resolution failure.
    #[error("{0}")]
    Models(#[from] ModelsError),
    /// IO failure in the file store.
    #[error("auth storage io failed: {0}")]
    Io(#[from] std::io::Error),
    /// auth.json failed validation.
    #[error("invalid auth.json: {0}")]
    InvalidData(String),
}

impl AuthError {
    /// Wrap as upstream `resolve.ts` does.
    #[must_use]
    pub fn models(code: ModelsErrorCode, message: impl Into<String>) -> Self {
        Self::Models(ModelsError::new(code, message.into()))
    }
}
