//! `models-error.ts`: `ModelsError` — the provider/model error type
//! (also the auth error type).

/// `ModelsErrorCode`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelsErrorCode {
    /// `model_source`.
    ModelSource,
    /// `model_validation`.
    ModelValidation,
    /// `provider`.
    Provider,
    /// `stream`.
    Stream,
    /// `auth`.
    Auth,
    /// `oauth`.
    Oauth,
}

impl ModelsErrorCode {
    /// The upstream code string.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ModelSource => "model_source",
            Self::ModelValidation => "model_validation",
            Self::Provider => "provider",
            Self::Stream => "stream",
            Self::Auth => "auth",
            Self::Oauth => "oauth",
        }
    }
}

/// `ModelsError`: `new ModelsError(code, message, {cause})`.
#[derive(Debug)]
pub struct ModelsError {
    /// The upstream code.
    pub code: ModelsErrorCode,
    /// `message`, with the cause detail appended like upstream.
    message: String,
}

impl ModelsError {
    /// `new ModelsError(code, message)` — `withCauseDetail` appends
    /// `formatThrownValue(cause)` when it is not already in the message.
    #[must_use]
    pub fn new(code: ModelsErrorCode, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }

    /// `new ModelsError(code, message, {cause})`.
    #[must_use]
    pub fn with_cause(code: ModelsErrorCode, message: impl Into<String>, cause: &str) -> Self {
        let message = with_cause_detail(message.into(), cause);
        Self { code, message }
    }
}

fn with_cause_detail(message: String, cause: &str) -> String {
    let detail = cause.trim();
    if detail.is_empty() || message.contains(detail) {
        return message;
    }
    format!("{message}: {detail}")
}

impl std::fmt::Display for ModelsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for ModelsError {}
