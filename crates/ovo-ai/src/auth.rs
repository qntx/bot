//! Credential store and API-key resolution (`auth/` port).
//!
//! The store boundary owns secrets: `Credential` keys serialize to
//! auth.json plaintext only through the file store; `Debug` redacts
//! via `secrecy`. Library code never reads `std::env` directly —
//! env access goes through the injected [`env::Env`] (only
//! [`env::ProcessEnv`] touches the process environment, and the host
//! constructs it).

pub mod config_value;
pub mod env;
pub mod file;
pub mod resolve;
pub mod store;
pub mod types;

pub use config_value::*;
pub use env::*;
pub use file::*;
pub use resolve::*;
pub use store::*;
pub use types::*;
