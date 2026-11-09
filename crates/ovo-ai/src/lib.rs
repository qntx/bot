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
//! - `host` (default) — `auth::file` (file credential store) and
//!   `ProcessCommandRunner` for `!command` config values. Without it
//!   `!command` values resolve exactly as a failed command does.
//!
/// Compiles the README's quick start as a doctest under the http
/// feature — the example imports `http`/`providers`.
#[cfg(all(doctest, feature = "http"))]
#[doc = include_str!("../README.md")]
struct ReadmeDoctest;

#[cfg(test)]
use tempfile as _;
#[cfg(test)]
use wiremock as _;

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
