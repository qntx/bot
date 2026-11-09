//! Integration tests for `ovo-ai`, kept in a single binary.

// Test binaries do not reference every dependency of the library they
// exercise.
#![allow(
    unused_crate_dependencies,
    reason = "test binaries only use part of the dependency set"
)]
#![allow(clippy::panic, reason = "integration tests fail by panicking")]
#![allow(
    clippy::indexing_slicing,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::unnecessary_cast,
    clippy::option_if_let_else,
    clippy::excessive_nesting,
    reason = "vector replays index fixed-shape case tables"
)]
#![allow(
    clippy::tests_outside_test_module,
    reason = "integration test files are test modules"
)]

mod ai_utils;
mod anthropic;
#[cfg(feature = "http")]
mod anthropic_stream;
mod auth;
mod constrained_sampling;
mod http_client;
mod http_policy;
mod http_sse;
mod json;
mod models_faux;
mod openai_completions;
#[cfg(feature = "http")]
mod openai_completions_stream;
mod openai_responses;
mod schema;
mod stream;
mod streaming_json;
mod support;
mod transcript;
mod types;
