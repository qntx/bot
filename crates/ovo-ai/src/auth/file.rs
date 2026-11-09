//! `FileCredentialStore` — `auth-storage.ts` port over auth.json.
//!
//! Format: flat `{providerId: Credential}` JSON, `{}` initialized on
//! first write when the file is missing, UTF-8 BOM stripped on read.
//! Per ADR-016, writes go through a 0600 temp file + atomic rename
//! under an exclusive `File::lock` on the `<path>.lock` sidecar
//! (upstream uses proper-lockfile + in-place write; the atomic form
//! is unobservable hardening). The parent dir is created 0700 on
//! unix.

use std::collections::HashMap;
use std::fs;
use std::future::Future;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use tokio::sync::Mutex;

use serde_json::Value;

use crate::auth::config_value::{
    CommandCache, CommandRunner, is_command_config_value, resolve_config_value,
};
use crate::auth::env::Env;
use crate::auth::store::CredentialStore;
use crate::auth::types::{AuthError, Credential, CredentialInfo};

/// `FileCredentialStore`: `auth.json`-backed `CredentialStore` with
/// atomic writes, an exclusive sidecar lock, and a per-store
/// `!command` result cache.
pub struct FileCredentialStore {
    path: PathBuf,
    env: Box<dyn Env>,
    runner: Box<dyn CommandRunner>,
    cache: CommandCache,
    modify_lock: Mutex<()>,
}

impl std::fmt::Debug for FileCredentialStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FileCredentialStore")
            .field("path", &self.path)
            .finish_non_exhaustive()
    }
}

impl FileCredentialStore {
    /// `path` is the auth.json location; `env`/`runner` feed
    /// `resolve_config_value` for stored `$NAME`/`!command` keys.
    #[must_use]
    pub fn new(path: PathBuf, env: Box<dyn Env>, runner: Box<dyn CommandRunner>) -> Self {
        Self {
            path,
            env,
            runner,
            cache: CommandCache::default(),
            modify_lock: Mutex::new(()),
        }
    }

    /// The sidecar lock path (`auth.json.lock`).
    fn lock_path(&self) -> PathBuf {
        self.path.with_extension(self.path.extension().map_or_else(
            || String::from("lock"),
            |e| format!("{}.lock", e.to_string_lossy()),
        ))
    }

    /// Read + parse auth.json — missing file is `{}`, BOM stripped,
    /// credentials validated like `parseStorageData`.
    fn load(&self) -> Result<HashMap<String, Credential>, AuthError> {
        let content = match fs::read_to_string(&self.path) {
            Ok(c) => c,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(HashMap::new()),
            Err(e) => return Err(AuthError::Io(e)),
        };
        let content = content.strip_prefix('\u{feff}').unwrap_or(&content);
        let value: Value =
            serde_json::from_str(content).map_err(|e| AuthError::InvalidData(format!("{e}")))?;
        let Value::Object(map) = value else {
            return Err(AuthError::InvalidData(String::from("expected an object")));
        };
        let mut out = HashMap::new();
        for (provider_id, credential) in map {
            validate_credential(&provider_id, &credential)?;
            out.insert(
                provider_id,
                serde_json::from_value(credential)
                    .map_err(|e| AuthError::InvalidData(format!("{e}")))?,
            );
        }
        Ok(out)
    }

    /// Atomic write: temp file in the same dir (0600 on unix), fsync,
    /// rename over `path`.
    fn write_all(&self, data: &HashMap<String, Credential>) -> Result<(), AuthError> {
        let dir = self
            .path
            .parent()
            .map_or_else(|| PathBuf::from("."), Path::to_path_buf);
        #[cfg(unix)]
        {
            use std::os::unix::fs::DirBuilderExt;
            fs::DirBuilder::new()
                .recursive(true)
                .mode(0o700)
                .create(&dir)?;
        }
        #[cfg(not(unix))]
        fs::create_dir_all(&dir)?;
        let file_name = self.path.file_name().map_or_else(
            || String::from("auth.json"),
            |n| n.to_string_lossy().into_owned(),
        );
        let temp = dir.join(format!(".{file_name}.tmp-{}", std::process::id()));
        let mut options = fs::OpenOptions::new();
        options.write(true).truncate(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        let mut file = options.open(&temp)?;
        let body = serde_json::to_string_pretty(data)
            .map_err(|e| AuthError::InvalidData(format!("{e}")))?;
        file.write_all(body.as_bytes())?;
        file.sync_all()?;
        drop(file);
        fs::rename(&temp, &self.path)?;
        Ok(())
    }

    /// Take the exclusive sidecar lock (blocking `File::lock`).
    fn take_lock(&self) -> Result<fs::File, AuthError> {
        let lock = fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .open(self.lock_path())?;
        lock.lock()?;
        Ok(lock)
    }
}

/// `auth-storage.ts` per-credential validation.
fn validate_credential(provider_id: &str, credential: &Value) -> Result<(), AuthError> {
    let invalid = || {
        AuthError::InvalidData(format!(
            "Invalid auth.json credential for provider \"{provider_id}\""
        ))
    };
    let Value::Object(map) = credential else {
        return Err(invalid());
    };
    match map.get("type").and_then(Value::as_str) {
        Some("api_key") => {
            let valid_key = map.get("key").is_none_or(Value::is_string);
            let valid_env = map.get("env").is_none_or(|e| {
                e.as_object()
                    .is_some_and(|o| o.values().all(Value::is_string))
            });
            if valid_key && valid_env {
                return Ok(());
            }
        }
        Some("oauth") => {
            let valid = map.get("access").is_some_and(Value::is_string)
                && map.get("refresh").is_some_and(Value::is_string)
                && map
                    .get("expires")
                    .and_then(Value::as_f64)
                    .is_some_and(f64::is_finite);
            if valid {
                return Ok(());
            }
        }
        _ => {}
    }
    Err(invalid())
}

impl CredentialStore for FileCredentialStore {
    fn read<'a>(
        &'a self,
        provider_id: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<Option<Credential>, AuthError>> + Send + 'a>> {
        Box::pin(async move {
            let credential = self.load()?.get(provider_id).cloned();
            let inner = match &credential {
                Some(Credential::ApiKey(inner)) => inner.clone(),
                _ => return Ok(credential),
            };
            let Some(key) = &inner.key else {
                return Ok(credential);
            };
            // Command values are NOT executed by read (upstream leaves
            // them raw — executing is a resolve-time side effect).
            let raw = secrecy::ExposeSecret::expose_secret(key);
            if is_command_config_value(raw) {
                return Ok(credential);
            }
            let resolved = resolve_config_value(raw, self.env.as_ref(), &*self.runner, &self.cache)
                .await
                .map(secrecy::SecretString::from);
            Ok(Some(Credential::ApiKey(
                crate::auth::types::ApiKeyCredential {
                    key: resolved,
                    env: inner.env,
                },
            )))
        })
    }

    fn list(
        &self,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<CredentialInfo>, AuthError>> + Send + '_>> {
        Box::pin(async move {
            Ok(self
                .load()?
                .into_iter()
                .map(|(provider_id, credential)| CredentialInfo {
                    provider_id,
                    type_: credential.type_tag(),
                })
                .collect())
        })
    }

    fn modify<'a>(
        &'a self,
        provider_id: &'a str,
        f: Box<
            dyn FnOnce(
                    Option<Credential>,
                ) -> Pin<Box<dyn Future<Output = Option<Credential>> + Send>>
                + Send
                + 'a,
        >,
    ) -> Pin<Box<dyn Future<Output = Result<Option<Credential>, AuthError>> + Send + 'a>> {
        Box::pin(async move {
            let _serialize = self.modify_lock.lock().await;
            let _file_lock = self.take_lock()?;
            let mut data = self.load()?;
            let current = data.get(provider_id).cloned();
            let next = f(current.clone()).await;
            match next {
                Some(credential) => {
                    data.insert(provider_id.to_owned(), credential.clone());
                    self.write_all(&data)?;
                    Ok(Some(credential))
                }
                None => Ok(current),
            }
        })
    }

    fn delete<'a>(
        &'a self,
        provider_id: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<(), AuthError>> + Send + 'a>> {
        Box::pin(async move {
            let _serialize = self.modify_lock.lock().await;
            let _file_lock = self.take_lock()?;
            let mut data = self.load()?;
            data.remove(provider_id);
            self.write_all(&data)?;
            Ok(())
        })
    }
}
