//! `provider-env.ts`: `getProviderEnvValue` against the injected
//! `ProviderEnv` snapshot. The process-env fallback lives with the
//! caller (library code does not read `std::env`).

use serde_json::{Map, Value};

/// `getProviderEnvValue(name, env)` — the map is the upstream
/// `ProviderEnv` plus whatever the host merged from `process.env`.
#[must_use]
pub fn get_provider_env_value<'m>(name: &str, env: &'m Map<String, Value>) -> Option<&'m str> {
    env.get(name).and_then(Value::as_str)
}
