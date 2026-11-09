//! Redacted provider/runtime diagnostics attached to assistant messages.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

/// Serialized error shape stored inside a diagnostic entry.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DiagnosticErrorInfo {
    /// Error class name, e.g. `"Error"`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Human-readable error message.
    pub message: String,
    /// Stack trace, when available.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stack: Option<String>,
    /// Error code, when the error carries one.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub code: Option<DiagnosticCode>,
}

/// `string | number` diagnostic error code.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum DiagnosticCode {
    /// Numeric error code.
    Number(f64),
    /// String error code.
    Text(String),
}

/// A diagnostic entry for failures and recoveries during a response.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AssistantMessageDiagnostic {
    /// Diagnostic kind, e.g. `"api_retry"`.
    #[serde(rename = "type")]
    pub kind: String,
    /// Unix timestamp in milliseconds when the diagnostic was recorded.
    pub timestamp: i64,
    /// The associated error, when the diagnostic carries one.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<DiagnosticErrorInfo>,
    /// Structured context details.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<Map<String, Value>>,
}
