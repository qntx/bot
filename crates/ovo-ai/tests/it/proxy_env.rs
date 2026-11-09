#![cfg(feature = "http")]
//! C1 — `.no_proxy()` when no proxy is configured: reqwest must not
//! pick up `HTTPS_PROXY`/`HTTP_PROXY`/`ALL_PROXY` from the process
//! env. The outer test re-execs this binary with the vars pointing at
//! a wiremock "proxy"; the inner `#[ignore = "runs as a re-exec child"]` test posts to its own
//! plain-http wiremock target.

use ovo_ai::http::client::{HttpClient, HttpClientConfig, RequestPolicy};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

const INNER: &str = "proxy_env::posts_directly_despite_process_proxy_vars";

#[tokio::test]
async fn process_env_proxy_is_not_honored() {
    // The "proxy" server — a correct client never contacts it.
    let proxy = MockServer::start().await;
    let exe = std::env::current_exe().unwrap_or_else(|e| panic!("exe: {e}"));
    let status = std::process::Command::new(exe)
        .args(["--exact", INNER, "--ignored", "--nocapture"])
        .env("HTTPS_PROXY", proxy.uri())
        .env("HTTP_PROXY", proxy.uri())
        .env("ALL_PROXY", proxy.uri())
        .env("https_proxy", proxy.uri())
        .env("http_proxy", proxy.uri())
        .env("all_proxy", proxy.uri())
        .status()
        .unwrap_or_else(|e| panic!("spawn: {e}"));
    assert!(status.success(), "inner test failed: {status}");
    assert_eq!(
        proxy.received_requests().await.unwrap_or_default().len(),
        0,
        "proxy must not be contacted"
    );
}

#[tokio::test]
#[ignore = "runs as a re-exec child"]
async fn posts_directly_despite_process_proxy_vars() {
    // This process runs with HTTPS_PROXY et al. set by the parent.
    let target = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1"))
        .respond_with(ResponseTemplate::new(200).set_body_string("{}"))
        .mount(&target)
        .await;
    // Injected env snapshot carries no proxy — `no_proxy` must keep
    // reqwest from reading the process env anyway.
    let client = HttpClient::new(HttpClientConfig {
        env: Some(serde_json::Map::new()),
        ..HttpClientConfig::default()
    });
    let result = client
        .post_json(
            &format!("{}/v1", target.uri()),
            &bytes::Bytes::from_static(b"{}"),
            &[],
            &RequestPolicy::default(),
            &tokio_util::sync::CancellationToken::new(),
            &|| 0.5,
        )
        .await;
    assert!(result.is_ok(), "post failed: {result:?}");
    assert_eq!(
        target.received_requests().await.unwrap_or_default().len(),
        1,
        "target must receive the request"
    );
}
