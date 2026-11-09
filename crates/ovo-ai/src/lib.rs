//! `ovo-ai` is the Rust port of `pi-ai` 1.5.0, the unified multi-provider
//! LLM API used by the ovo agent harness. Behaviour is verified against the
//! published TypeScript package with replay vectors (`parity.json`).
//!
//! # Entry points
//!
//! - [`providers::builtin_models`] builds a [`models::Models`] registry with
//!   the six built-in providers wired to the embedded [`catalog`].
//!   [`models::Models::stream`], [`models::Models::stream_simple`] and
//!   [`models::Models::complete`] run requests; [`models::Models::get_model`]
//!   looks a model up by provider and id.
//! - [`stream::AssistantStream`] yields [`stream::AssistantMessageEvent`]s and
//!   resolves to the final message through `result()`.
//! - [`types`] holds the wire shapes: [`types::context::Context`],
//!   [`types::tool::Tool`] and the [`types::message`] transcript types.
//! - [`auth`] resolves credentials; [`auth::ProcessEnv`] is the ambient
//!   environment the host injects and the only place the process environment
//!   is read.
//! - [`faux`] is a deterministic provider for tests.
//! - [`json`] is the JSON layer every wire format goes through, so emitted
//!   bytes match `JSON.stringify` exactly.
//!
//! # Features
//!
//! - `http` (default) — the HTTP adapters and built-in providers
//!   (`anthropic-messages`, `openai-completions`, `openai-responses`).
//!
//! # Example
//!
//! ```no_run
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! use std::sync::Arc;
//!
//! use futures::StreamExt;
//! use ovo_ai::auth::ProcessEnv;
//! use ovo_ai::http::client::{HttpClient, HttpClientConfig};
//! use ovo_ai::models::CreateModelsOptions;
//! use ovo_ai::providers::builtin_models;
//! use ovo_ai::stream::AssistantMessageEvent;
//! use ovo_ai::types::context::Context;
//!
//! let http = Arc::new(HttpClient::new(HttpClientConfig::default()));
//! let models = builtin_models(
//!     CreateModelsOptions {
//!         env: Some(Arc::new(ProcessEnv)),
//!         ..CreateModelsOptions::default()
//!     },
//!     &http,
//! )?;
//! let model = models
//!     .get_model("anthropic", "claude-fable-5")
//!     .ok_or("anthropic/claude-fable-5 not in catalog")?;
//! let context: Context = serde_json::from_value(serde_json::json!({
//!     "messages": [{ "role": "user", "content": "Say hi in one word.", "timestamp": 0 }]
//! }))?;
//! let mut stream = models.stream(&model, &context, None);
//! while let Some(event) = stream.next().await {
//!     if let AssistantMessageEvent::TextDelta { delta, .. } = event {
//!         print!("{delta}");
//!     }
//! }
//! let message = stream.result().await;
//! println!("{:?}", message.stop_reason);
//! #     Ok(())
//! # }
//! ```
#![allow(
    unused_crate_dependencies,
    reason = "wiremock is a dev-dep used only by integration tests"
)]

pub mod api;
pub mod auth;
pub mod catalog;
pub mod cost;
pub mod estimate;
pub mod faux;
pub mod hash;
pub mod http;
pub mod identity;
pub mod json;
pub mod models;
pub mod models_error;
pub mod options;
pub mod overflow;
pub mod provider;
#[cfg(feature = "http")]
pub mod providers;
pub mod retry;
pub mod schema;
pub mod simple_options;
pub mod stream;
pub mod text;
pub mod transcript;
pub mod transform;
pub mod types;
