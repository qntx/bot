//! Structured classifier context and result types.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use super::api::{ClassifierApi, ProviderId};
use super::content::ImageContent;
use super::usage::Usage;

/// A choice question: pick one named option.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClassifierChoiceQuestion {
    /// What the classifier should evaluate.
    pub instructions: String,
    /// Candidate labels mapped to their descriptions.
    pub criteria: Map<String, Value>,
}

/// A score question: rate on the provider's scale.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClassifierScoreQuestion {
    /// What the classifier should evaluate.
    pub instructions: String,
    /// Ordered scale descriptions.
    pub criteria: Vec<String>,
}

/// A yes/no question.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClassifierBoolQuestion {
    /// What the classifier should evaluate.
    pub instructions: String,
    /// The true/false labels.
    pub criteria: ClassifierBoolCriteria,
}

/// The true/false labels of a bool question.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClassifierBoolCriteria {
    /// Label for the true outcome.
    #[serde(rename = "true")]
    pub true_: String,
    /// Label for the false outcome.
    #[serde(rename = "false")]
    pub false_: String,
}

/// One classifier question.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ClassifierQuestion {
    /// Pick one named option.
    #[serde(rename = "choice")]
    Choice(ClassifierChoiceQuestion),
    /// Rate on the provider's scale.
    #[serde(rename = "score")]
    Score(ClassifierScoreQuestion),
    /// Yes/no question.
    #[serde(rename = "bool")]
    Bool(ClassifierBoolQuestion),
}

/// Input to a classifier request.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClassifierContext {
    /// The structured state to judge.
    pub state: Map<String, Value>,
    /// Images judged together with `state`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub images: Option<Vec<ImageContent>>,
    /// Named questions to answer.
    pub questions: Map<String, Value>,
}

/// Choice answer.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ClassifierChoiceAnswer {
    /// The chosen label.
    pub choice: String,
    /// Probability per candidate label.
    pub probabilities: Map<String, Value>,
    /// Confidence in the answer.
    pub confidence: f64,
}

/// Score answer.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ClassifierScoreAnswer {
    /// The score.
    pub score: f64,
    /// Confidence in the answer.
    pub confidence: f64,
}

/// Bool answer.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ClassifierBoolAnswer {
    /// Probability of the true outcome.
    pub probability: f64,
}

/// One classifier answer.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ClassifierAnswer {
    /// Chosen label with probabilities.
    #[serde(rename = "choice")]
    Choice(ClassifierChoiceAnswer),
    /// Score with confidence.
    #[serde(rename = "score")]
    Score(ClassifierScoreAnswer),
    /// Boolean probability.
    #[serde(rename = "bool")]
    Bool(ClassifierBoolAnswer),
}

/// Why a classify call terminated.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ClassifierStopReason {
    /// Natural stop.
    #[serde(rename = "stop")]
    Stop,
    /// The request failed.
    #[serde(rename = "error")]
    Error,
    /// The request was aborted.
    #[serde(rename = "aborted")]
    Aborted,
}

/// Result of a classify request.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ClassifierResult {
    /// Classifier API family that produced the result.
    pub api: ClassifierApi,
    /// Provider that produced the result.
    pub provider: ProviderId,
    /// Model that produced the result.
    pub model: String,
    /// Answers keyed by question name.
    pub answers: Map<String, Value>,
    /// Token usage at the model's catalog price, when reported.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
    /// Why the classify call terminated.
    pub stop_reason: ClassifierStopReason,
    /// Error text when `stopReason` is `"error"` or `"aborted"`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error_message: Option<String>,
    /// Unix timestamp in milliseconds.
    pub timestamp: i64,
}
