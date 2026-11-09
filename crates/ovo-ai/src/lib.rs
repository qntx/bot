//! `ovo-ai` is the Rust port of `pi-ai`, the unified multi-provider LLM API
#![allow(
    unused_crate_dependencies,
    reason = "wiremock is a dev-dep used only by integration tests"
)]
//! used by the ovo agent harness.
//!
//! [`json`] is the JSON layer every wire format (session files, RPC records,
//! model catalogs) goes through, so the emitted bytes match `JSON.stringify`
//! exactly.

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
pub mod retry;
pub mod schema;
pub mod simple_options;
pub mod stream;
pub mod text;
pub mod transcript;
pub mod transform;
pub mod types;
