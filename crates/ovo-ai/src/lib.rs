//! `ovo-ai` is the Rust port of `pi-ai`, the unified multi-provider LLM API
//! used by the ovo agent harness.
//!
//! [`json`] is the JSON layer every wire format (session files, RPC records,
//! model catalogs) goes through, so the emitted bytes match `JSON.stringify`
//! exactly.

pub mod cost;
pub mod estimate;
pub mod identity;
pub mod json;
pub mod overflow;
pub mod retry;
pub mod simple_options;
pub mod stream;
pub mod text;
pub mod transcript;
pub mod transform;
pub mod types;
