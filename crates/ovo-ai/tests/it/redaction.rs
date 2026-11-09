//! Secret-bearing values never appear in `Debug` output — `api_key`
//! fields, `AuthResult`, env snapshots and headers print names only.

use ovo_ai::auth::env::MapEnv;
use ovo_ai::auth::types::{AuthResult, ModelAuth};
use ovo_ai::options::{RequestOptions, StreamOptions};
use serde_json::json;

#[test]
fn secrets_are_redacted_in_debug_output() {
    let key = "sk-secret-123";
    let token = "tok-456";

    let stream = StreamOptions {
        api_key: Some(secrecy::SecretString::from(key.to_owned())),
        headers: Some(
            json!({"authorization": format!("Bearer {token}")})
                .as_object()
                .cloned()
                .unwrap_or_default(),
        ),
        ..StreamOptions::default()
    };
    let request = RequestOptions {
        api_key: stream.api_key.clone(),
        headers: stream.headers.clone(),
        ..RequestOptions::default()
    };
    let auth = AuthResult {
        auth: ModelAuth {
            api_key: stream.api_key.clone(),
            headers: stream.headers.clone(),
            ..ModelAuth::default()
        },
        env: None,
        source: Some(String::from("test")),
    };
    let env = MapEnv::new([
        (String::from("SECRET"), String::from(key)),
        (String::from("TOKEN"), String::from(token)),
    ]);
    let parts = ovo_ai::models::RequestParts {
        api_key: stream.api_key.clone(),
        headers: stream.headers.clone(),
        env: None,
        identity: None,
    };

    for (label, debug) in [
        ("StreamOptions", format!("{stream:?}")),
        ("RequestOptions", format!("{request:?}")),
        ("AuthResult", format!("{auth:?}")),
        ("MapEnv", format!("{env:?}")),
        ("RequestParts", format!("{parts:?}")),
    ] {
        assert!(!debug.contains(key), "{label} leaked the key: {debug}");
        assert!(!debug.contains(token), "{label} leaked the token: {debug}");
    }
    // The assertions above could pass vacuously if Debug dropped the
    // fields entirely — the key sites must show a redaction marker.
    for (label, debug) in [
        ("StreamOptions", format!("{stream:?}")),
        ("RequestOptions", format!("{request:?}")),
        ("AuthResult", format!("{auth:?}")),
        ("RequestParts", format!("{parts:?}")),
    ] {
        assert!(
            debug.contains("<redacted>") || debug.contains("[REDACTED]"),
            "{label} shows no redaction marker: {debug}"
        );
    }
}
