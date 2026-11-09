//! `CredentialStore` + `InMemoryCredentialStore` + `RuntimeCredentials`.

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use tokio::sync::Mutex;

use crate::auth::types::{AuthError, Credential, CredentialInfo};

/// `ModifyFn`: the `modify` callback — sees the current credential,
/// returns the next (`None` leaves it unchanged).
pub type ModifyFn<'a> = Box<
    dyn FnOnce(Option<Credential>) -> Pin<Box<dyn Future<Output = Option<Credential>> + Send>>
        + Send
        + 'a,
>;

/// `CredentialStore` (`auth/types.ts`): app-owned credential storage.
///
/// Keyed by `Provider.id`, one credential per provider. `modify` is
/// the only write path — a serialized read-modify-write per provider
/// so OAuth refresh cannot double-rotate a token.
pub trait CredentialStore: Send + Sync {
    /// `read`: the stored credential (possibly expired), `None` for
    /// missing entries; errors only on storage failure.
    fn read<'a>(
        &'a self,
        provider_id: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<Option<Credential>, AuthError>> + Send + 'a>>;

    /// `list`: non-secret metadata; must not execute configured
    /// API-key commands.
    fn list(
        &self,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<CredentialInfo>, AuthError>> + Send + '_>>;

    /// `modify`: serialized write — `fn` sees the current credential,
    /// returns the new one or `None` to leave it unchanged; resolves
    /// with the post-write credential.
    fn modify<'a>(
        &'a self,
        provider_id: &'a str,
        f: ModifyFn<'a>,
    ) -> Pin<Box<dyn Future<Output = Result<Option<Credential>, AuthError>> + Send + 'a>>;

    /// `delete`: remove a credential; serialized against `modify`.
    fn delete<'a>(
        &'a self,
        provider_id: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<(), AuthError>> + Send + 'a>>;
}

/// `InMemoryCredentialStore`: upstream default — a map with a
/// per-provider async mutex serializing `modify`/`delete`.
#[derive(Default)]
pub struct InMemoryCredentialStore {
    credentials: Mutex<HashMap<String, Credential>>,
    locks: Mutex<HashMap<String, Arc<Mutex<()>>>>,
}

impl std::fmt::Debug for InMemoryCredentialStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InMemoryCredentialStore")
            .finish_non_exhaustive()
    }
}

impl InMemoryCredentialStore {
    /// Empty store.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    async fn provider_lock(&self, provider_id: &str) -> Arc<Mutex<()>> {
        Arc::clone(
            self.locks
                .lock()
                .await
                .entry(provider_id.to_owned())
                .or_insert_with(|| Arc::new(Mutex::new(()))),
        )
    }
}

impl CredentialStore for InMemoryCredentialStore {
    fn read<'a>(
        &'a self,
        provider_id: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<Option<Credential>, AuthError>> + Send + 'a>> {
        Box::pin(async move { Ok(self.credentials.lock().await.get(provider_id).cloned()) })
    }

    fn list(
        &self,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<CredentialInfo>, AuthError>> + Send + '_>> {
        Box::pin(async move {
            Ok(self
                .credentials
                .lock()
                .await
                .iter()
                .map(|(provider_id, credential)| CredentialInfo {
                    provider_id: provider_id.clone(),
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
            let lock = self.provider_lock(provider_id).await;
            let _guard = lock.lock().await;
            let current = self.credentials.lock().await.get(provider_id).cloned();
            let next = f(current.clone()).await;
            match next {
                Some(credential) => {
                    self.credentials
                        .lock()
                        .await
                        .insert(provider_id.to_owned(), credential.clone());
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
            let lock = self.provider_lock(provider_id).await;
            let _guard = lock.lock().await;
            self.credentials.lock().await.remove(provider_id);
            Ok(())
        })
    }
}

/// `RuntimeCredentials` (`runtime-credentials.ts`): non-persistent
/// per-run api-key overlay over any store.
pub struct RuntimeCredentials {
    store: Box<dyn CredentialStore>,
    overrides: Mutex<HashMap<String, String>>,
}

impl std::fmt::Debug for RuntimeCredentials {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RuntimeCredentials").finish_non_exhaustive()
    }
}

impl RuntimeCredentials {
    /// Wrap `store` with a runtime-key overlay.
    #[must_use]
    pub fn new(store: Box<dyn CredentialStore>) -> Self {
        Self {
            store,
            overrides: Mutex::new(HashMap::new()),
        }
    }

    /// `setRuntimeApiKey` — non-persistent override.
    pub async fn set_runtime_api_key(&self, provider_id: &str, api_key: &str) {
        self.overrides
            .lock()
            .await
            .insert(provider_id.to_owned(), api_key.to_owned());
    }

    /// `removeRuntimeApiKey`.
    pub async fn remove_runtime_api_key(&self, provider_id: &str) {
        self.overrides.lock().await.remove(provider_id);
    }

    /// `hasRuntimeApiKey`.
    pub async fn has_runtime_api_key(&self, provider_id: &str) -> bool {
        self.overrides.lock().await.contains_key(provider_id)
    }
}

impl CredentialStore for RuntimeCredentials {
    fn read<'a>(
        &'a self,
        provider_id: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<Option<Credential>, AuthError>> + Send + 'a>> {
        Box::pin(async move {
            let key = self.overrides.lock().await.get(provider_id).cloned();
            match key {
                Some(key) => Ok(Some(Credential::ApiKey(
                    crate::auth::types::ApiKeyCredential {
                        key: Some(secrecy::SecretString::from(key)),
                        env: None,
                    },
                ))),
                None => self.store.read(provider_id).await,
            }
        })
    }

    fn list(
        &self,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<CredentialInfo>, AuthError>> + Send + '_>> {
        Box::pin(async move {
            let mut entries: HashMap<String, CredentialInfo> = self
                .store
                .list()
                .await?
                .into_iter()
                .map(|e| (e.provider_id.clone(), e))
                .collect();
            for provider_id in self.overrides.lock().await.keys() {
                entries.insert(
                    provider_id.clone(),
                    CredentialInfo {
                        provider_id: provider_id.clone(),
                        type_: "api_key",
                    },
                );
            }
            Ok(entries.into_values().collect())
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
        self.store.modify(provider_id, f)
    }

    fn delete<'a>(
        &'a self,
        provider_id: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<(), AuthError>> + Send + 'a>> {
        Box::pin(async move {
            self.store.delete(provider_id).await?;
            self.overrides.lock().await.remove(provider_id);
            Ok(())
        })
    }
}
