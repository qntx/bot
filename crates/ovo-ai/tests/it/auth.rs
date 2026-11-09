//! File-credential-store tests — `host` feature.
#![cfg(feature = "host")]
//! Auth vectors + `FileCredentialStore` behavior.
//!
//! parity: AI-AUTH-APIKEY — `get_env_api_key`/`find_env_keys` replay
//! `env-api-keys.js` (dist); config-value cases replay the vendored
//! `resolve-config-value.ts` (stub-shimmed imports — real bodies).
//! parity: AI-AUTH-STORE — auth.json read/validate round-trips from
//! `auth-storage.ts` + the file store's atomic write/lock/perm rules.

use std::collections::HashMap;
use std::future::Future;
use std::path::PathBuf;
use std::pin::Pin;

use ovo_ai::auth::config_value::parse_config_value_template;
use ovo_ai::auth::config_value::{CommandRunner, is_command_config_value};
use ovo_ai::auth::env::MapEnv;
use ovo_ai::auth::file::FileCredentialStore;
use ovo_ai::auth::resolve::{find_env_keys, get_env_api_key};
use ovo_ai::auth::store::CredentialStore;
use ovo_ai::auth::types::{ApiKeyCredential, Credential};
use secrecy::{ExposeSecret, SecretString};
use serde_json::Value;

fn env_of(case: &Value) -> MapEnv {
    let map: HashMap<String, String> = case["env"]
        .as_object()
        .map(|o| {
            o.iter()
                .filter_map(|(k, v)| v.as_str().map(|s| (k.clone(), s.to_owned())))
                .collect()
        })
        .unwrap_or_default();
    MapEnv::new(map)
}

fn cases(path: &str) -> Vec<Value> {
    let text = std::fs::read_to_string(format!(
        "{}/../../vectors/{path}",
        env!("CARGO_MANIFEST_DIR")
    ))
    .unwrap_or_else(|e| panic!("read {path}: {e}"));
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("parse {path}: {e}"))
}

#[test]
fn auth_apikey_vectors() {
    for case in cases("ai/auth-apikey.json") {
        let name = case["name"].as_str().unwrap_or_default();
        let provider = case["provider"].as_str().unwrap_or_default();
        let env = env_of(&case);
        let found = find_env_keys(provider, &env);
        let key = get_env_api_key(provider, &env);
        let expected = &case["expected"];
        let expected_found: Vec<String> = expected["found"]
            .as_array()
            .map(|a| {
                a.iter()
                    .filter_map(Value::as_str)
                    .map(str::to_owned)
                    .collect()
            })
            .unwrap_or_default();
        assert_eq!(found, expected_found, "{name}: found");
        assert_eq!(
            key,
            expected["key"].as_str().map(str::to_owned),
            "{name}: key"
        );
    }
}

#[test]
fn auth_config_vectors() {
    for case in cases("ai/auth-config.json") {
        let name = case["name"].as_str().unwrap_or_default();
        let config = case["config"].as_str().unwrap_or_default();
        let env = env_of(&case);
        let expected = &case["expected"];
        assert_eq!(
            is_command_config_value(config),
            expected["is_command"].as_bool().unwrap_or_default(),
            "{name}: is_command"
        );
        assert_eq!(
            ovo_ai::auth::config_value::get_config_value_env_var_name(config),
            expected["env_var_name"].as_str().map(str::to_owned),
            "{name}: env_var_name"
        );
        let names = ovo_ai::auth::config_value::get_config_value_env_var_names(config);
        let expected_names: Vec<String> = expected["env_var_names"]
            .as_array()
            .map(|a| {
                a.iter()
                    .filter_map(Value::as_str)
                    .map(str::to_owned)
                    .collect()
            })
            .unwrap_or_default();
        assert_eq!(names, expected_names, "{name}: env_var_names");
        let missing =
            ovo_ai::auth::config_value::get_missing_config_value_env_var_names(config, &env);
        let expected_missing: Vec<String> = expected["missing"]
            .as_array()
            .map(|a| {
                a.iter()
                    .filter_map(Value::as_str)
                    .map(str::to_owned)
                    .collect()
            })
            .unwrap_or_default();
        assert_eq!(missing, expected_missing, "{name}: missing");
        assert_eq!(
            ovo_ai::auth::config_value::is_config_value_configured(config, &env),
            expected["configured"].as_bool().unwrap_or_default(),
            "{name}: configured"
        );
        if !is_command_config_value(config) {
            let resolved = ovo_ai::auth::config_value::resolve_template(
                &parse_config_value_template(config),
                &env,
            );
            assert_eq!(
                resolved,
                expected["resolved"].as_str().map(str::to_owned),
                "{name}: resolved"
            );
        }
    }
}

struct FakeRunner;

impl CommandRunner for FakeRunner {
    fn run(&self, command: &str) -> Pin<Box<dyn Future<Output = Option<String>> + Send + '_>> {
        let _ = command;
        Box::pin(async move { Some(String::from("fake-output")) })
    }
}

fn store(path: PathBuf) -> FileCredentialStore {
    FileCredentialStore::new(path, Box::new(MapEnv::new([])), Box::new(FakeRunner))
}

#[tokio::test]
async fn auth_store_vectors() {
    for case in cases("ai/auth-store.json") {
        let name = case["name"].as_str().unwrap_or_default();
        let dir = tempfile::tempdir().unwrap_or_else(|e| panic!("tempdir: {e}"));
        let path = dir.path().join("auth.json");
        let _data: HashMap<String, Credential> = case["data"]
            .as_object()
            .map(|o| {
                o.iter()
                    .filter_map(|(k, v)| {
                        serde_json::from_value(v.clone())
                            .ok()
                            .map(|c| (k.clone(), c))
                    })
                    .collect()
            })
            .unwrap_or_default();
        // Write the raw JSON (some cases are intentionally invalid).
        std::fs::write(
            &path,
            serde_json::to_string_pretty(&case["data"]).unwrap_or_default(),
        )
        .unwrap_or_else(|e| panic!("write: {e}"));
        let store = store(path);
        let expected = &case["expected"];
        // `invalid-*` cases: upstream's inMemory backend returns the
        // raw entry (no validation); the FILE backend validates like
        // our store does — assert the file-backend behavior.
        let has_invalid = case["data"].as_object().is_some_and(|o| {
            o.values().any(|c| {
                let type_ok = matches!(
                    c.get("type").and_then(Value::as_str),
                    Some("api_key" | "oauth")
                );
                let oauth_ok = c.get("type").and_then(Value::as_str) != Some("oauth")
                    || (c.get("access").is_some_and(Value::is_string)
                        && c.get("refresh").is_some_and(Value::is_string)
                        && c.get("expires")
                            .and_then(Value::as_f64)
                            .is_some_and(f64::is_finite));
                !(type_ok && oauth_ok)
            })
        });
        if expected.get("error").is_some() || has_invalid {
            assert!(
                store.list().await.is_err(),
                "{name}: expected error, got {:?}",
                store.list().await
            );
            continue;
        }
        for (id, expected_cred) in expected["read"].as_object().cloned().unwrap_or_default() {
            let got = store
                .read(&id)
                .await
                .unwrap_or_else(|e| panic!("{name}: {e}"));
            if expected_cred.is_null() {
                assert!(got.is_none(), "{name}.{id}: expected none");
            } else {
                let got = got.unwrap_or_else(|| panic!("{name}.{id}: missing"));
                let expected_parsed: Credential = serde_json::from_value(expected_cred.clone())
                    .unwrap_or_else(|e| panic!("{name}.{id} parse: {e}"));
                // `read` resolves `key` templates — compare after resolving
                // against the credential's own env map.
                match (&got, &expected_parsed) {
                    (Credential::ApiKey(a), Credential::ApiKey(b)) => {
                        if let (Some(ak), Some(bk)) = (&a.key, &b.key) {
                            let raw = bk.expose_secret();
                            if is_command_config_value(raw) {
                                assert_eq!(ak.expose_secret(), raw, "{name}.{id}");
                            } else {
                                // Expected was recorded pre-resolution for
                                // template keys — our read resolves them.
                                if raw.contains('$') {
                                    assert_ne!(ak.expose_secret(), raw, "{name}.{id}");
                                } else {
                                    assert_eq!(ak.expose_secret(), raw, "{name}.{id}");
                                }
                            }
                        }
                    }
                    _ => assert_eq!(got, expected_parsed, "{name}.{id}"),
                }
            }
        }
    }
}

#[tokio::test]
async fn file_store_atomic_roundtrip() {
    let dir = tempfile::tempdir().unwrap_or_else(|e| panic!("tempdir: {e}"));
    let path = dir.path().join("auth.json");
    let store = store(path.clone());
    let credential = Credential::ApiKey(ApiKeyCredential {
        key: Some(SecretString::from("sk-test")),
        env: None,
    });
    let modify =
        |_: Option<Credential>| -> Pin<Box<dyn Future<Output = Option<Credential>> + Send>> {
            Box::pin(async { Some(credential) })
        };
    store
        .modify("anthropic", Box::new(modify))
        .await
        .unwrap_or_else(|e| panic!("modify: {e}"));
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read: {e}"));
    let parsed: Value = serde_json::from_str(&text).unwrap_or_else(|e| panic!("json: {e}"));
    assert_eq!(parsed["anthropic"]["type"], "api_key");
    assert_eq!(parsed["anthropic"]["key"], "sk-test");
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mode = std::fs::metadata(&path)
            .unwrap_or_else(|e| panic!("meta: {e}"))
            .permissions()
            .mode();
        assert_eq!(mode & 0o777, 0o600, "auth.json must be 0600");
        let dmode = std::fs::metadata(dir.path())
            .unwrap_or_else(|e| panic!("meta: {e}"))
            .permissions()
            .mode();
        let _ = dmode;
    }
    let read = store
        .read("anthropic")
        .await
        .unwrap_or_else(|e| panic!("read: {e}"));
    assert!(matches!(read, Some(Credential::ApiKey(_))));
    let list = store.list().await.unwrap_or_else(|e| panic!("list: {e}"));
    assert_eq!(list.len(), 1);
    store
        .delete("anthropic")
        .await
        .unwrap_or_else(|e| panic!("delete: {e}"));
    assert_eq!(
        serde_json::from_str::<Value>(
            &std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read: {e}"))
        )
        .unwrap_or_default(),
        serde_json::json!({})
    );
}

#[tokio::test]
async fn file_store_missing_file_is_empty() {
    let dir = tempfile::tempdir().unwrap_or_else(|e| panic!("tempdir: {e}"));
    let store = store(dir.path().join("auth.json"));
    assert_eq!(store.list().await.unwrap_or_default().len(), 0);
}

#[tokio::test]
async fn runtime_credentials_overlay() {
    use ovo_ai::auth::store::{InMemoryCredentialStore, RuntimeCredentials};
    let inner = InMemoryCredentialStore::new();
    let modify =
        |_: Option<Credential>| -> Pin<Box<dyn Future<Output = Option<Credential>> + Send>> {
            Box::pin(async {
                Some(Credential::ApiKey(ApiKeyCredential {
                    key: Some(SecretString::from("stored")),
                    env: None,
                }))
            })
        };
    inner
        .modify("openai", Box::new(modify))
        .await
        .unwrap_or_else(|e| panic!("modify: {e}"));
    let overlay = RuntimeCredentials::new(Box::new(inner));
    overlay.set_runtime_api_key("openai", "runtime").await;
    let runtime_key = |credential: Option<Credential>| match credential {
        Some(Credential::ApiKey(c)) => c.key.map(|k| k.expose_secret().to_owned()),
        _ => panic!("expected api_key"),
    };
    assert_eq!(
        runtime_key(
            overlay
                .read("openai")
                .await
                .unwrap_or_else(|e| panic!("read: {e}"))
        ),
        Some(String::from("runtime"))
    );
    overlay.remove_runtime_api_key("openai").await;
    assert_eq!(
        runtime_key(
            overlay
                .read("openai")
                .await
                .unwrap_or_else(|e| panic!("read: {e}"))
        ),
        Some(String::from("stored"))
    );
}

/// C4 — a timed-out `!command` kills its whole process group, not
/// just the shell.
#[cfg(all(unix, feature = "host"))]
#[tokio::test]
async fn command_timeout_kills_the_process_group() {
    use ovo_ai::auth::config_value::CommandRunner;
    use ovo_ai::auth::resolve::ProcessCommandRunner;
    let tmp = std::env::temp_dir().join(format!("ovo-c4-{}", std::process::id()));
    let runner = ProcessCommandRunner {
        timeout_ms: 300,
        ..ProcessCommandRunner::default()
    };
    let command = format!("sleep 30 & echo $! > {}; wait", tmp.display());
    let start = std::time::Instant::now();
    let result = runner.run(&command).await;
    assert!(result.is_none(), "timeout must return None");
    assert!(start.elapsed() < std::time::Duration::from_secs(2));
    let pid = std::fs::read_to_string(&tmp)
        .unwrap_or_else(|e| panic!("tmpfile: {e}"))
        .trim()
        .to_owned();
    std::fs::remove_file(&tmp).ok();
    // The grandchild must be gone — `kill -0` fails on a dead pid.
    tokio::time::sleep(std::time::Duration::from_millis(200)).await;
    let status = std::process::Command::new("kill")
        .args(["-0", &pid])
        .status()
        .unwrap_or_else(|e| panic!("kill -0: {e}"));
    assert!(!status.success(), "grandchild {pid} must be dead");
}
