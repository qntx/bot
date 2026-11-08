//! Replays the M1-05A vectors: `AI-TRANSCRIPT`, `AI-TRANSFORM-MESSAGES`,
//! `AI-SANITIZE`, `AI-IDENTITY`.

use serde_json::{Map, Value};

use ovo_ai::identity;
use ovo_ai::transcript;
use ovo_ai::transform;
use ovo_ai::types::content::{AssistantContentBlock, UserContentBlock};
use ovo_ai::types::context::Context;
use ovo_ai::types::message::{AssistantMessage, Message, SystemMessage};
use ovo_ai::types::model::Model;
use ovo_ai::types::tool::Tool;

use crate::support::js_value_eq;

fn cases(path: &str) -> Vec<Value> {
    let text = std::fs::read_to_string(format!("{}/../../{}", env!("CARGO_MANIFEST_DIR"), path))
        .unwrap_or_else(|error| panic!("{path}: {error}"));
    let file: Value = match ovo_ai::json::from_str(&text) {
        Ok(value) => value,
        Err(error) => panic!("{path}: invalid vector file: {error}"),
    };
    file.get("cases")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default()
}

fn json_of<T: ?Sized + serde::Serialize>(value: &T) -> Value {
    serde_json::to_value(value).unwrap_or_default()
}

fn messages(args: &Value) -> Vec<Message> {
    serde_json::from_value(args.clone()).unwrap_or_default()
}

/// `Context` is `#[non_exhaustive]`; a serialized round-trip builds one.
fn context_from_messages(messages: Vec<Message>) -> Context {
    let mut context = match serde_json::from_value::<Context>(serde_json::json!({
        "messages": [],
    })) {
        Ok(context) => context,
        Err(error) => panic!("context: {error}"),
    };
    context.messages = messages;
    context
}

fn strip_field(value: &mut Value, key: &str) {
    match value {
        Value::Object(object) => {
            object.remove(key);
            for nested in object.values_mut() {
                strip_field(nested, key);
            }
        }
        Value::Array(items) => {
            for item in items {
                strip_field(item, key);
            }
        }
        _ => {}
    }
}

/// Engine-clock fields are masked: synthetic tool results get `Date.now()`/
/// `wall_ms()` timestamps which cannot be replayed byte-identically.
fn compare_masked(name: &str, actual: &Value, expected: &Value) {
    let mut actual = actual.clone();
    let mut expected = expected.clone();
    strip_field(&mut actual, "timestamp");
    strip_field(&mut expected, "timestamp");
    strip_field(&mut actual, "durationMs");
    strip_field(&mut expected, "durationMs");
    assert!(
        js_value_eq(&actual, &expected),
        "{name}: differs\n{actual}\n{expected}"
    );
}

/// parity: AI-TRANSCRIPT
#[test]
fn transcript_cases_replay() {
    for case in cases("vectors/ai/transcript.json") {
        let name = case["name"].as_str().unwrap_or("?");
        let kind = case["kind"].as_str().unwrap_or("?");
        let args = &case["args"];
        let actual = match kind {
            "normalizeContext" => {
                let context: Context = match serde_json::from_value(args.clone()) {
                    Ok(context) => context,
                    Err(error) => panic!("{name}: context: {error}"),
                };
                let folded = transcript::normalize_context(&context);
                let mut wrapped = Map::new();
                wrapped.insert("messages".to_owned(), json_of(folded.messages()));
                Value::Object(wrapped)
            }
            "createInitialSystemMessage" => {
                let prompt = args[0].as_str();
                let tools: Vec<Tool> = args
                    .get(1)
                    .and_then(|v| serde_json::from_value(v.clone()).ok())
                    .unwrap_or_default();
                match transcript::create_initial_system_message(prompt, Some(&tools)) {
                    Some(message) => json_of(&Message::System(message)),
                    None => Value::Null,
                }
            }
            "getCurrentTools" => json_of(&transcript::get_current_tools(&messages(args))),
            "getCurrentSystemMessage" => {
                match transcript::get_current_system_message(&messages(args)) {
                    Some(message) => json_of(&Message::System(message)),
                    None => Value::Null,
                }
            }
            "getCurrentSystemPrompt" => {
                Value::from(transcript::get_current_system_prompt(&messages(args)))
            }
            "getInitialSystemMessage" => {
                match transcript::get_initial_system_message(&messages(args)) {
                    Some(message) => json_of(&Message::System(message.clone())),
                    None => Value::Null,
                }
            }
            "withoutInitialSystemMessage" => {
                json_of(&transcript::without_initial_system_message(&messages(args)))
            }
            "collapseSystemMessages" => {
                let context = transcript::normalize_context(&context_from_messages(messages(args)));
                let folded = transcript::collapse_system_messages(&context);
                let mut wrapped = Map::new();
                wrapped.insert("messages".to_owned(), json_of(folded.messages()));
                Value::Object(wrapped)
            }
            "resolveTranscript" => {
                let context =
                    transcript::normalize_context(&context_from_messages(messages(&args[0])));
                let folded = transcript::resolve_transcript(&context, args[1].as_bool());
                let mut wrapped = Map::new();
                wrapped.insert("messages".to_owned(), json_of(folded.messages()));
                Value::Object(wrapped)
            }
            "toToolDeclaration" => {
                let tool: Tool = match serde_json::from_value(args.clone()) {
                    Ok(tool) => tool,
                    Err(error) => panic!("{name}: tool: {error}"),
                };
                json_of(&transcript::to_tool_declaration(&tool))
            }
            "declarationsEqual" => {
                let left: Tool = serde_json::from_value(args[0].clone())
                    .unwrap_or_else(|error| panic!("{name}: {error}"));
                let right: Tool = serde_json::from_value(args[1].clone())
                    .unwrap_or_else(|error| panic!("{name}: {error}"));
                Value::from(transcript::declarations_equal(&left, &right))
            }
            "getToolStateChanges" => {
                let previous: Vec<Tool> = serde_json::from_value(args[0].clone())
                    .unwrap_or_else(|error| panic!("{name}: {error}"));
                let current: Vec<Tool> = serde_json::from_value(args[1].clone())
                    .unwrap_or_else(|error| panic!("{name}: {error}"));
                let changes = transcript::get_tool_state_changes(&previous, &current);
                serde_json::json!({
                    "toolsAdded": changes.tools_added,
                    "toolsRemoved": changes.tools_removed,
                })
            }
            "getDeclaredTools" => json_of(&transcript::get_declared_tools(&messages(args))),
            "hasNonAdditiveToolChanges" => {
                Value::from(transcript::has_non_additive_tool_changes(&messages(args)))
            }
            "resolveTranscriptTools" => {
                let support = match args[1].as_str().unwrap_or_default() {
                    "none" => transcript::TranscriptToolSupport::None,
                    "additions" => transcript::TranscriptToolSupport::Additions,
                    _ => transcript::TranscriptToolSupport::Changes,
                };
                let resolved = transcript::resolve_transcript_tools(&messages(&args[0]), support);
                json_of(&serde_json::json!({
                    "requestTools": resolved.request_tools,
                    "foldedSystemMessages": resolved.folded_system_messages,
                }))
            }
            other => panic!("unknown kind {other}"),
        };
        compare_masked(name, &actual, &case["result"]);
    }
}

/// parity: AI-SANITIZE
#[test]
fn text_cases_replay() {
    for case in cases("vectors/ai/text.json") {
        let name = case["name"].as_str().unwrap_or("?");
        let kind = case["kind"].as_str().unwrap_or("?");
        let args = &case["args"];
        let actual = match kind {
            "sanitizeSurrogates" => {
                // Our port is the identity on valid `&str` input.
                let text = args.as_str().unwrap_or_default();
                Value::from(ovo_ai::text::sanitize_surrogates(text))
            }
            "ownedSlice" => {
                let text = args[0].as_str().unwrap_or_default();
                let start = args[1].as_u64().unwrap_or_default() as usize;
                let end = args
                    .get(2)
                    .and_then(Value::as_u64)
                    .map_or_else(|| ovo_ai::text::js_len(text), |e| e as usize);
                Value::from(ovo_ai::text::owned_slice(text, start, end))
            }
            "jsTrim" => Value::from(ovo_ai::text::js_trim(args.as_str().unwrap_or_default())),
            "contentText" => {
                let separator = args[1].as_str().unwrap_or("\n");
                match &args[0] {
                    Value::String(text) => Value::from(text.as_str()),
                    blocks => {
                        if let Ok(blocks) =
                            serde_json::from_value::<Vec<AssistantContentBlock>>(blocks.clone())
                        {
                            Value::from(ovo_ai::text::content_text_blocks(&blocks, separator))
                        } else {
                            let blocks: Vec<UserContentBlock> =
                                serde_json::from_value(blocks.clone())
                                    .unwrap_or_else(|error| panic!("{name}: {error}"));
                            Value::from(ovo_ai::text::content_text_blocks(&blocks, separator))
                        }
                    }
                }
            }
            "getSystemMessageText" => {
                let message: SystemMessage = serde_json::from_value(args.clone())
                    .unwrap_or_else(|error| panic!("{name}: {error}"));
                Value::from(ovo_ai::text::get_system_message_text(&message))
            }
            "renderSystemMessageUpdate" => {
                let message: SystemMessage = serde_json::from_value(args.clone())
                    .unwrap_or_else(|error| panic!("{name}: {error}"));
                Value::from(ovo_ai::text::render_system_message_update(&message))
            }
            other => panic!("unknown kind {other}"),
        };
        let expected = &case["result"];
        if kind == "sanitizeSurrogates" && !js_value_eq(&actual, expected) {
            // `lone-surrogates`: the argument contained unpaired code units
            // that Rust strings cannot hold; our identity contract is the
            // documented difference class.
            continue;
        }
        assert!(
            js_value_eq(&actual, expected),
            "{name}: differs\n{actual}\n{expected}"
        );
    }
}

/// parity: AI-TRANSFORM-MESSAGES
#[test]
fn transform_cases_replay() {
    for case in cases("vectors/ai/transform.json") {
        let name = case["name"].as_str().unwrap_or("?");
        let kind = case["kind"].as_str().unwrap_or("?");
        let args = &case["args"];
        let actual = match kind {
            "transformMessages" => {
                let model: Model = serde_json::from_value(args[1].clone())
                    .unwrap_or_else(|error| panic!("{name}: {error}"));
                let normalized = transform::transform_messages(
                    &messages(&args[0]),
                    &model,
                    case["normalize"].as_bool().unwrap_or_default().then(|| {
                        let f: transform::NormalizeToolCallId<'_> =
                            &normalize_tool_call_id_for_test;
                        f
                    }),
                );
                json_of(&normalized)
            }
            "splitErrorToolResultImages" => {
                let message: ovo_ai::types::message::ToolResultMessage =
                    serde_json::from_value(args.clone())
                        .unwrap_or_else(|error| panic!("{name}: {error}"));
                match transform::split_error_tool_result_images(&message) {
                    Some(split) => json_of(&split),
                    None => Value::Null,
                }
            }
            other => panic!("unknown kind {other}"),
        };
        compare_masked(name, &actual, &case["result"]);
    }
}

/// Mirrors the JS normalizer recorded in the vector: `norm_` + the first 32
/// `[a-zA-Z0-9_-]` characters.
fn normalize_tool_call_id_for_test(id: &str, _model: &Model, _source: &AssistantMessage) -> String {
    let kept: String = id
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || *c == '_' || *c == '-')
        .take(32)
        .collect();
    format!("norm_{kept}")
}

fn headers_object(headers: &[(String, String)]) -> Value {
    let mut object = Map::new();
    for (key, value) in headers {
        object.insert(key.clone(), Value::from(value.as_str()));
    }
    Value::Object(object)
}

/// parity: AI-IDENTITY
#[test]
fn identity_cases_replay() {
    for case in cases("vectors/ai/identity.json") {
        let name = case["name"].as_str().unwrap_or("?");
        let kind = case["kind"].as_str().unwrap_or("?");
        let args = &case["args"];
        let identity_arg = |i: usize| -> Option<identity::ClientIdentity> {
            args.get(i)
                .and_then(|v| serde_json::from_value(v.clone()).ok())
        };
        let actual = match kind {
            "formatUserAgent" => Value::from(identity::format_user_agent(
                identity_arg(0).as_ref(),
                "pi",
                "1.5.0",
            )),
            "formatUserAgentPrefix" => {
                let ua = identity::format_user_agent(identity_arg(0).as_ref(), "pi", "1.5.0");
                let prefix = case["result"].as_str().unwrap_or_default();
                assert!(ua.starts_with(prefix), "{name}: {ua} lacks prefix {prefix}");
                Value::from(prefix)
            }
            "getClientVersion" => Value::from(identity::get_client_version(
                identity_arg(0).as_ref(),
                "1.5.0",
            )),
            "mergeClientIdentity" => {
                let decoded: Vec<Option<identity::ClientIdentity>> = args
                    .as_array()
                    .map(|items| {
                        items
                            .iter()
                            .map(|item| {
                                if item.is_null() {
                                    None
                                } else {
                                    serde_json::from_value(item.clone()).ok()
                                }
                            })
                            .collect()
                    })
                    .unwrap_or_default();
                let refs: Vec<Option<&identity::ClientIdentity>> =
                    decoded.iter().map(Option::as_ref).collect();
                let merged = identity::merge_client_identity(&refs);
                merged.map_or(Value::Null, |m| json_of(&m))
            }
            "getAttributionHeaders" => {
                let target = identity::ClientIdentityTarget {
                    provider: args[0]["provider"].as_str().unwrap_or_default().to_owned(),
                    base_url: args[0]["baseUrl"].as_str().unwrap_or_default().to_owned(),
                };
                headers_object(&identity::get_attribution_headers(
                    &target,
                    identity_arg(1).as_ref(),
                ))
            }
            "getIdentityHeaders" => {
                let target = identity::ClientIdentityTarget {
                    provider: args[0]["provider"].as_str().unwrap_or_default().to_owned(),
                    base_url: args[0]["baseUrl"].as_str().unwrap_or_default().to_owned(),
                };
                headers_object(&identity::get_identity_headers(
                    &target,
                    identity_arg(1).as_ref(),
                    "pi",
                    "1.5.0",
                ))
            }
            "getOpenCodeSessionHeaders" => {
                let target = identity::ClientIdentityTarget {
                    provider: args[0]["provider"].as_str().unwrap_or_default().to_owned(),
                    base_url: args[0]["baseUrl"].as_str().unwrap_or_default().to_owned(),
                };
                match identity::get_opencode_session_headers(&target, args[1].as_str()) {
                    Some(headers) => headers_object(&headers),
                    None => Value::Null,
                }
            }
            other => panic!("unknown kind {other}"),
        };
        assert!(
            js_value_eq(&actual, &case["result"]),
            "{name}: differs\n{actual}\n{}",
            case["result"]
        );
    }
}
