#![cfg(feature = "http")]
//! `HttpClient` against a `wiremock` server.
//!
//! parity: AI-HTTP-RETRY — exercises `provider_retry_decision` inside
//! `post_json` over a real socket (429+retry-after, give-up, cap).

use ovo_ai::http::client::{HttpClient, HttpClientConfig, RequestPolicy, SendError};
use tokio_util::sync::CancellationToken;
use wiremock::matchers::{body_bytes, header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn client() -> HttpClient {
    HttpClient::new(HttpClientConfig::default())
}

const fn policy(max_retries: u32, cap: Option<f64>) -> RequestPolicy {
    RequestPolicy {
        timeout: std::time::Duration::from_secs(5),
        max_retries,
        max_retry_delay_ms: cap,
    }
}

fn json_body() -> bytes::Bytes {
    ovo_ai::json::to_vec(&serde_json::json!({"model": "m", "n": 1}))
        .unwrap_or_else(|e| panic!("to_vec: {e:?}"))
        .into()
}

#[tokio::test]
async fn posts_headers_and_exact_body() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1"))
        .and(header("x-auth", "k"))
        .and(header("content-type", "application/json"))
        .and(body_bytes(&*json_body()))
        .respond_with(ResponseTemplate::new(200).set_body_string("ok"))
        .expect(1)
        .mount(&server)
        .await;
    let response = client()
        .post_json(
            &format!("{}/v1", server.uri()),
            &json_body(),
            &[("x-auth", "k")],
            &policy(0, None),
            &CancellationToken::new(),
            &|| 0.5,
        )
        .await;
    assert!(response.is_ok(), "{response:?}");
}

#[tokio::test]
async fn retries_429_then_succeeds() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1"))
        .respond_with(ResponseTemplate::new(429).insert_header("retry-after-ms", "1"))
        .up_to_n_times(1)
        .expect(1)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/v1"))
        .respond_with(ResponseTemplate::new(200).set_body_string("ok"))
        .expect(1)
        .mount(&server)
        .await;
    let response = client()
        .post_json(
            &format!("{}/v1", server.uri()),
            &json_body(),
            &[],
            &policy(2, None),
            &CancellationToken::new(),
            &|| 0.5,
        )
        .await;
    assert!(response.is_ok(), "{response:?}");
}

#[tokio::test]
async fn gives_up_after_max_retries() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1"))
        .respond_with(ResponseTemplate::new(500))
        .expect(2)
        .mount(&server)
        .await;
    let err = client()
        .post_json(
            &format!("{}/v1", server.uri()),
            &json_body(),
            &[],
            &policy(1, None),
            &CancellationToken::new(),
            &|| 0.5,
        )
        .await
        .err()
        .unwrap_or_else(|| panic!("expected error"));
    assert!(matches!(err, SendError::Provider(_)), "{err:?}");
}

#[tokio::test]
async fn cap_exceeded_is_provider_error() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1"))
        .respond_with(ResponseTemplate::new(429).insert_header("retry-after", "120"))
        .expect(1)
        .mount(&server)
        .await;
    let err = client()
        .post_json(
            &format!("{}/v1", server.uri()),
            &json_body(),
            &[],
            &policy(2, Some(60_000.0)),
            &CancellationToken::new(),
            &|| 0.5,
        )
        .await
        .err()
        .unwrap_or_else(|| panic!("expected error"));
    assert!(matches!(err, SendError::Provider(_)), "{err:?}");
}

#[tokio::test]
async fn cancellation_during_backoff_aborts() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1"))
        .respond_with(ResponseTemplate::new(429).insert_header("retry-after", "600"))
        .mount(&server)
        .await;
    let token = CancellationToken::new();
    let token2 = token.clone();
    let handle = tokio::spawn(async move {
        client()
            .post_json(
                &format!("{}/v1", server.uri()),
                &json_body(),
                &[],
                &policy(5, Some(0.0)),
                &token2,
                &|| 0.5,
            )
            .await
    });
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    token.cancel();
    let err = handle
        .await
        .unwrap_or_else(|e| panic!("join: {e:?}"))
        .err()
        .unwrap_or_else(|| panic!("expected error"));
    assert!(matches!(err, SendError::Aborted), "{err:?}");
}

#[tokio::test]
async fn error_body_is_normalized() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1"))
        .respond_with(ResponseTemplate::new(400).set_body_json(serde_json::json!({
            "error": {"message": "bad request", "type": "invalid"}
        })))
        .expect(1)
        .mount(&server)
        .await;
    let err = client()
        .post_json(
            &format!("{}/v1", server.uri()),
            &json_body(),
            &[],
            &policy(0, None),
            &CancellationToken::new(),
            &|| 0.5,
        )
        .await
        .err()
        .unwrap_or_else(|| panic!("expected error"));
    let SendError::Provider(e) = err else {
        panic!("expected Provider, got {err:?}")
    };
    assert!(e.message.contains("bad request"), "{}", e.message);
    assert_eq!(e.status, Some(400.0));
}
