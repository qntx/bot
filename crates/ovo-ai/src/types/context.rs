//! Request context accepted by the public stream entry points.

use serde::{Deserialize, Serialize};

use super::message::Message;
use super::tool::Tool;

/// Request input: `systemPrompt` and `tools` are shorthand for a leading
/// system message.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
#[non_exhaustive]
pub struct Context {
    /// System prompt for the request.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_prompt: Option<String>,
    /// The transcript messages.
    pub messages: Vec<Message>,
    /// Tools available to the model.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,
}
