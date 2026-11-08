//! Integration tests for `ovo-ai`, kept in a single binary.

// Test binaries do not reference every dependency of the library they
// exercise.
#![allow(
    unused_crate_dependencies,
    reason = "test binaries only use part of the dependency set"
)]

mod json;
