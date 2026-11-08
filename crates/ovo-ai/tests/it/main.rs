//! Integration tests for `ovo-ai`, kept in a single binary.

// Test binaries do not reference every dependency of the library they
// exercise.
#![allow(
    unused_crate_dependencies,
    reason = "test binaries only use part of the dependency set"
)]
#![allow(clippy::panic, reason = "integration tests fail by panicking")]
#![allow(
    clippy::tests_outside_test_module,
    reason = "integration test files are test modules"
)]

mod json;
mod stream;
mod streaming_json;
mod support;
#[expect(
    clippy::indexing_slicing,
    clippy::cast_possible_truncation,
    clippy::option_if_let_else,
    clippy::excessive_nesting,
    reason = "vector case args are indexed/cast freely in tests"
)]
mod transcript;
mod types;
