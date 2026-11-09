//! `Env`: injected environment access (`AuthContext.env` in upstream
//! terms). Library code never touches `std::env`; the host constructs
//! [`ProcessEnv`] explicitly or injects a [`MapEnv`] snapshot.

use std::collections::HashMap;

/// Environment access for auth resolution. Synchronous because the
/// ambient env is a process snapshot; async file checks stay with
/// the store/credential types that need them.
pub trait Env: Send + Sync + std::fmt::Debug {
    /// `env(name)` — `Some` when the variable is set (upstream checks
    /// truthiness — an empty string does not resolve).
    fn get(&self, name: &str) -> Option<String>;
}

/// `MapEnv`: an env snapshot (merged `ProviderEnv` over process env,
/// or a test fixture). Provider-injected values take precedence via
/// [`MapEnv::with_overlay`].
#[derive(Clone, Default)]
pub struct MapEnv {
    values: HashMap<String, String>,
}

impl std::fmt::Debug for MapEnv {
    /// Values are secrets — print variable names only.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MapEnv")
            .field("values", &self.values.keys().collect::<Vec<_>>())
            .finish()
    }
}

impl MapEnv {
    /// Snapshot from an iterator of pairs.
    pub fn new(values: impl IntoIterator<Item = (String, String)>) -> Self {
        Self {
            values: values.into_iter().collect(),
        }
    }

    /// `overlay`: a view where `overlay` wins then falls back to
    /// `self` (upstream `overlayEnvAuthContext`).
    #[must_use]
    pub fn with_overlay(self, overlay: &serde_json::Map<String, serde_json::Value>) -> Self {
        let mut values = self.values;
        for (name, value) in overlay {
            if let Some(v) = value.as_str().filter(|v| !v.is_empty()) {
                values.insert(name.clone(), v.to_owned());
            }
        }
        Self { values }
    }
}

impl Env for MapEnv {
    fn get(&self, name: &str) -> Option<String> {
        self.values.get(name).filter(|v| !v.is_empty()).cloned()
    }
}

/// `ProcessEnv`: reads the real process environment. The only place
/// `std::env` is touched inside the library — the host constructs
/// this explicitly.
#[derive(Clone, Copy, Debug, Default)]
pub struct ProcessEnv;

impl Env for ProcessEnv {
    fn get(&self, name: &str) -> Option<String> {
        std::env::var(name).ok().filter(|v| !v.is_empty())
    }
}

/// `env(name)` on an `&dyn Env` — truthiness rule applied once.
pub fn env_get(env: &dyn Env, name: &str) -> Option<String> {
    env.get(name).filter(|v| !v.is_empty())
}

/// `overlayEnvAuthContext`: provider-injected env wins over the
/// ambient `Env` — without reading the process environment.
#[derive(Debug)]
pub struct OverlayEnv<'a> {
    overlay: &'a serde_json::Map<String, serde_json::Value>,
    base: &'a dyn Env,
}

impl<'a> OverlayEnv<'a> {
    /// `overlay` wins over `base` per key.
    #[must_use]
    pub const fn new(
        overlay: &'a serde_json::Map<String, serde_json::Value>,
        base: &'a dyn Env,
    ) -> Self {
        Self { overlay, base }
    }
}

impl Env for OverlayEnv<'_> {
    fn get(&self, name: &str) -> Option<String> {
        self.overlay
            .get(name)
            .and_then(|v| v.as_str())
            .filter(|v| !v.is_empty())
            .map(str::to_owned)
            .or_else(|| self.base.get(name))
    }
}
