//! `AI-PROVIDERS-CORE`: the built-in providers against pi's
//! `builtinProviders()`.

use std::sync::Arc;

use futures::StreamExt;
use ovo_ai::auth::env::MapEnv;
use ovo_ai::auth::types::{ApiKeyCredential, AuthResult};
use ovo_ai::http::client::{HttpClient, HttpClientConfig};
use ovo_ai::models::CreateModelsOptions;
use ovo_ai::options::StreamOptions;
use ovo_ai::provider::{ModelType, Provider, model_ids};
use ovo_ai::providers::{builtin_models, builtin_providers};
use ovo_ai::stream::AssistantMessageEvent;
use ovo_ai::transcript::normalize_context;
use ovo_ai::types::context::Context;
use serde_json::{Map, Value, json};

use crate::support::js_value_eq;

fn vectors() -> Vec<Value> {
    let text = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../vectors/ai/providers.json"
    ));
    let document: Value = serde_json::from_str(text).unwrap_or_else(|e| panic!("vector file: {e}"));
    document["cases"].as_array().cloned().unwrap_or_default()
}

fn http() -> Arc<HttpClient> {
    Arc::new(HttpClient::new(HttpClientConfig::default()))
}

fn credential(value: &Value) -> Option<ApiKeyCredential> {
    value.as_object().map(|object| ApiKeyCredential {
        key: object
            .get("key")
            .and_then(Value::as_str)
            .map(|key| secrecy::SecretString::from(key.to_owned())),
        env: object.get("env").and_then(Value::as_object).cloned(),
    })
}

/// An `AuthResult` as pi's resolver returns it.
fn recorded(result: Option<&AuthResult>) -> Value {
    let Some(result) = result else {
        return Value::Null;
    };
    let mut auth = Map::new();
    if let Some(key) = &result.auth.api_key {
        auth.insert(
            "apiKey".into(),
            Value::from(secrecy::ExposeSecret::expose_secret(key).to_owned()),
        );
    }
    if let Some(headers) = &result.auth.headers {
        auth.insert("headers".into(), Value::Object(headers.clone()));
    }
    let mut recorded = Map::new();
    recorded.insert("auth".into(), Value::Object(auth));
    if let Some(env) = &result.env {
        recorded.insert("env".into(), Value::Object(env.clone()));
    }
    if let Some(source) = &result.source {
        recorded.insert("source".into(), Value::from(source.clone()));
    }
    Value::Object(recorded)
}

/// parity: AI-PROVIDERS-CORE
#[test]
fn builtin_providers_match_upstream() {
    let providers = builtin_providers(&http()).unwrap_or_else(|e| panic!("providers: {e}"));
    let cases = vectors();
    let ids: Vec<&str> = providers.iter().map(Provider::id).collect();
    let wanted: Vec<&str> = cases
        .iter()
        .filter_map(|case| case["id"].as_str())
        .collect();
    assert_eq!(ids, wanted, "provider order");
    for (provider, case) in providers.iter().zip(&cases) {
        let id = provider.id();
        assert_eq!(provider.name(), case["name"], "{id}: name");
        assert_eq!(
            provider.base_url(),
            case["baseUrl"].as_str(),
            "{id}: baseUrl"
        );
        assert_eq!(
            provider.headers().is_some(),
            !case["headers"].is_null(),
            "{id}: headers"
        );
        let auth = provider.auth();
        let api_key = auth
            .api_key
            .as_ref()
            .unwrap_or_else(|| panic!("{id}: no API key auth"));
        assert_eq!(api_key.name, case["apiKeyAuth"], "{id}: API key auth name");
        // OAuth sign-in arrives with the OAuth flows (M5).
        assert!(auth.oauth.is_none(), "{id}: oauth");
        for resolution in case["resolutions"].as_array().into_iter().flatten() {
            let name = resolution["name"].as_str().unwrap_or("?");
            let env =
                MapEnv::new(resolution["env"].as_object().into_iter().flatten().map(
                    |(key, value)| (key.clone(), value.as_str().unwrap_or_default().to_owned()),
                ));
            let stored = credential(&resolution["credential"]);
            let result = (api_key.resolve)(stored.as_ref(), &env);
            let got = recorded(result.as_ref().ok().and_then(Option::as_ref));
            assert!(
                js_value_eq(&got, &resolution["result"]),
                "{id}/{name}\n  got:  {got}\n  want: {}",
                resolution["result"]
            );
        }
        let models = provider.get_models();
        let catalog = ovo_ai::catalog::models(id, ModelType::Chat)
            .unwrap_or_else(|e| panic!("{id}: catalog: {e}"));
        let catalog_ids: Vec<&str> = catalog.iter().map(|model| model_ids(model).1).collect();
        let model_ids: Vec<&str> = models.iter().map(|model| model.id.as_str()).collect();
        assert_eq!(model_ids, catalog_ids, "{id}: chat models");
    }
}

/// Every protocol a built-in provider serves dispatches to its adapter: a
/// request without a key fails in the adapter, not in dispatch.
#[tokio::test]
async fn builtin_providers_dispatch_every_api() {
    let context: Context = serde_json::from_value(json!({
        "messages": [{ "role": "user", "content": "hi", "timestamp": 0 }]
    }))
    .unwrap_or_else(|e| panic!("context: {e}"));
    for provider in builtin_providers(&http()).unwrap_or_else(|e| panic!("providers: {e}")) {
        let id = provider.id().to_owned();
        let mut apis: Vec<String> = Vec::new();
        for model in provider.get_models() {
            let api = model.api.as_str().to_owned();
            if apis.contains(&api) {
                continue;
            }
            let mut stream = provider.stream(
                &model,
                normalize_context(&context),
                StreamOptions::default(),
            );
            let mut error = None;
            while let Some(event) = stream.next().await {
                if let AssistantMessageEvent::Error { error_message, .. } = event {
                    error = Some(error_message);
                }
            }
            assert_eq!(
                error.as_deref(),
                Some(format!("No API key for provider: {id}").as_str()),
                "{id}/{api}"
            );
            apis.push(api);
        }
        let expected: &[&str] = match id.as_str() {
            "anthropic" => &["anthropic-messages"],
            "openai" | "xai" => &["openai-responses"],
            "openrouter" => &["openai-completions", "anthropic-messages"],
            _ => &["openai-completions"],
        };
        let mut sorted = apis.clone();
        sorted.sort();
        let mut wanted: Vec<String> = expected.iter().map(|api| (*api).to_owned()).collect();
        wanted.sort();
        assert_eq!(sorted, wanted, "{id}: protocols");
    }
}

/// `builtinModels` registers every built-in provider.
#[test]
fn builtin_models_registers_every_provider() {
    let models = builtin_models(CreateModelsOptions::default(), &http())
        .unwrap_or_else(|e| panic!("models: {e}"));
    let mut ids: Vec<String> = models
        .get_providers()
        .iter()
        .map(|provider| provider.id().to_owned())
        .collect();
    ids.sort();
    assert_eq!(
        ids,
        [
            "anthropic",
            "deepseek",
            "groq",
            "openai",
            "openrouter",
            "xai"
        ]
    );
    assert!(models.get_model("openai", "gpt-5").is_some());
}
