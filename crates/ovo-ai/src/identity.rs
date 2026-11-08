//! Client identity: upstream `utils/identity.ts` — `User-Agent`
//! formatting, provider attribution headers, and identity merging.
//!
//! The product name and version are injected as parameters (`product`,
//! `fallback_version`); the ovo defaults are [`OVO_PRODUCT`]/
//! [`OVO_VERSION`]. Upstream substitutes `pi`/`PI_AI_VERSION` and a
//! `node|bun/<ver>` runtime segment — ovo reports `rust` (no runtime
//! version exists). Upstream ships `PI_ATTRIBUTION` (`pi`, `https://pi.dev`,
//! `["cli-agent"]`) only when its install telemetry records an origin;
//! ovo has no such telemetry and sends no attribution by default, while
//! keeping the header logic for hosts that supply their own. Vectors
//! exercise the logic with upstream's `pi` inputs injected.
//! `with_client_identity` wraps the upstream `Models` registry, which does
//! not exist yet (M1-08).

/// The ovo product name in `User-Agent` and attribution contexts.
pub const OVO_PRODUCT: &str = "ovo";

/// The ovo default version: this crate's package version.
pub const OVO_VERSION: &str = env!("CARGO_PKG_VERSION");

use serde::{Deserialize, Serialize};

/// `UserAgentMode`: what the `User-Agent` header carries.
///
/// Wire: `"short"` / `"detailed"` / `{"custom": "acme/1.0"}`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum UserAgentMode {
    /// `pi` alone (or the injected product name).
    Short,
    /// `product/version (os; rust; arch)`.
    Detailed,
    /// A caller-provided value used verbatim (`{"custom": "acme/1.0"}`).
    Custom(String),
}

impl Serialize for UserAgentMode {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            Self::Short => serializer.serialize_str("short"),
            Self::Detailed => serializer.serialize_str("detailed"),
            Self::Custom(custom) => {
                let mut state = serializer.serialize_map(Some(1))?;
                serde::ser::SerializeMap::serialize_entry(&mut state, "custom", custom)?;
                serde::ser::SerializeMap::end(state)
            }
        }
    }
}

impl<'de> Deserialize<'de> for UserAgentMode {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let value = serde_json::Value::deserialize(deserializer)?;
        match value {
            serde_json::Value::String(mode) if mode == "short" => Ok(Self::Short),
            serde_json::Value::String(mode) if mode == "detailed" => Ok(Self::Detailed),
            serde_json::Value::Object(mut object) => object
                .remove("custom")
                .and_then(|v| v.as_str().map(str::to_owned))
                .map_or_else(
                    || {
                        Err(serde::de::Error::custom(
                            "expected a custom user agent object",
                        ))
                    },
                    |custom| Ok(Self::Custom(custom)),
                ),
            _ => Err(serde::de::Error::custom("expected user agent mode")),
        }
    }
}

/// `ClientAttribution`: who requests are attributed to.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ClientAttribution {
    /// App name: `OpenRouter` app title, `NVIDIA` invoke origin (first letter
    /// upper-cased), `OpenCode` client.
    pub name: String,
    /// App URL: `OpenRouter` `HTTP-Referer`.
    pub url: String,
    /// App categories, e.g. `["cli-agent"]`: `OpenRouter`
    /// `X-OpenRouter-Categories`, comma-separated.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub categories: Option<Vec<String>>,
}

/// `ClientIdentity`: how the calling app is identified to providers.
///
/// Every field is optional; absent fields use the defaults (a detailed
/// `User-Agent`, no attribution). Plain JSON so it can be persisted.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ClientIdentity {
    /// Default: `Detailed`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user_agent: Option<UserAgentMode>,
    /// Tri-state matching the upstream `attribution?: ClientAttribution
    /// | null`: not set (merge keeps earlier values), `null` (off), or a
    /// value.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "deserialize_attribution"
    )]
    #[expect(
        clippy::option_option,
        reason = "the upstream field has three states: absent, null, value"
    )]
    pub attribution: Option<Option<ClientAttribution>>,
    /// Version in the detailed `User-Agent`. Default: the injected
    /// `fallback_version`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,
}

/// `null` deserializes to `Some(None)` (explicit off); absent stays
/// `None` (merge skips). `Option<Option<T>>` derive would collapse null to
/// the outer `None`.
#[expect(clippy::option_option, reason = "tri-state: absent, null, value")]
fn deserialize_attribution<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> Result<Option<Option<ClientAttribution>>, D::Error> {
    let value = serde_json::Value::deserialize(deserializer)?;
    match value {
        serde_json::Value::Null => Ok(Some(None)),
        other => Ok(Some(Some(
            ClientAttribution::deserialize(other).map_err(serde::de::Error::custom)?,
        ))),
    }
}

/// The provider and endpoint a request goes to.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ClientIdentityTarget {
    /// The provider id the request targets.
    pub provider: String,
    /// The request's base URL.
    pub base_url: String,
}

const NVIDIA_NIM_HOST: &str = "integrate.api.nvidia.com";
const OPENCODE_HOST: &str = "opencode.ai";

/// `mergeClientIdentity`: merge field by field.
///
/// Later sources win and absent fields are skipped. Returns `None` for no
/// sources.
#[must_use]
pub fn merge_client_identity(sources: &[Option<&ClientIdentity>]) -> Option<ClientIdentity> {
    let mut merged: Option<ClientIdentity> = None;
    for source in sources.iter().flatten() {
        let target = merged.get_or_insert_with(ClientIdentity::default);
        if source.user_agent.is_some() {
            target.user_agent.clone_from(&source.user_agent);
        }
        if source.attribution.is_some() {
            target.attribution.clone_from(&source.attribution);
        }
        if source.version.is_some() {
            target.version.clone_from(&source.version);
        }
    }
    merged
}

/// `getClientVersion`: the version in the detailed `User-Agent`.
#[must_use]
pub fn get_client_version<'a>(identity: Option<&'a ClientIdentity>, fallback: &'a str) -> &'a str {
    identity
        .and_then(|identity| identity.version.as_deref())
        .unwrap_or(fallback)
}

/// `formatUserAgent`: the `User-Agent` value of an identity.
///
/// `product` and `fallback_version` replace upstream's `pi`/`PI_AI_VERSION`;
/// `os`/`arch` come from the compiled target. Upstream's runtime segment is
/// `bun/<ver>` or `node/<ver>`; ovo's is `rust` (no runtime version exists
/// to report).
#[must_use]
pub fn format_user_agent(
    identity: Option<&ClientIdentity>,
    product: &str,
    fallback_version: &str,
) -> String {
    match identity.and_then(|identity| identity.user_agent.as_ref()) {
        Some(UserAgentMode::Custom(custom)) => custom.clone(),
        Some(UserAgentMode::Short) => product.to_owned(),
        Some(UserAgentMode::Detailed) | None => {
            let product = format!(
                "{product}/{}",
                get_client_version(identity, fallback_version)
            );
            format!(
                "{product} ({}; rust; {})",
                std::env::consts::OS,
                std::env::consts::ARCH
            )
        }
    }
}

fn host_of(base_url: &str) -> Option<&str> {
    let after_scheme = base_url.split("://").nth(1)?;
    let authority = after_scheme.split(['/', '?', '#']).next()?;
    authority.split(':').next()
}

fn matches_host(base_url: &str, expected_host: &str) -> bool {
    host_of(base_url) == Some(expected_host)
}

fn is_openrouter(target: &ClientIdentityTarget) -> bool {
    target.provider == "openrouter" || target.base_url.contains("openrouter.ai")
}

fn is_nvidia_nim(target: &ClientIdentityTarget) -> bool {
    target.provider == "nvidia" || matches_host(&target.base_url, NVIDIA_NIM_HOST)
}

fn is_opencode(target: &ClientIdentityTarget) -> bool {
    target.provider == "opencode"
        || target.provider == "opencode-go"
        || matches_host(&target.base_url, OPENCODE_HOST)
}

/// `getAttributionHeaders`: the provider attribution headers of an identity.
///
/// Headers come in upstream insertion order and are empty when attribution
/// is off.
#[must_use]
pub fn get_attribution_headers(
    target: &ClientIdentityTarget,
    identity: Option<&ClientIdentity>,
) -> Vec<(String, String)> {
    let Some(attribution) = identity
        .and_then(|identity| identity.attribution.as_ref())
        .and_then(Option::as_ref)
    else {
        return Vec::new();
    };
    let ClientAttribution {
        name,
        url,
        categories,
    } = attribution;
    if is_openrouter(target) {
        let mut headers = vec![
            ("HTTP-Referer".to_owned(), url.clone()),
            ("X-OpenRouter-Title".to_owned(), name.clone()),
        ];
        if let Some(categories) = categories
            && !categories.is_empty()
        {
            headers.push(("X-OpenRouter-Categories".to_owned(), categories.join(",")));
        }
        return headers;
    }
    if is_nvidia_nim(target) {
        // NVIDIA expects a capitalized origin, such as `Pi`.
        let capitalized = name
            .chars()
            .next()
            .map(|first| first.to_uppercase().collect::<String>() + &name[first.len_utf8()..])
            .unwrap_or_default();
        return vec![("X-BILLING-INVOKE-ORIGIN".to_owned(), capitalized)];
    }
    if is_opencode(target) {
        return vec![("x-opencode-client".to_owned(), name.clone())];
    }
    Vec::new()
}

/// `getOpenCodeSessionHeaders`: `OpenCode`'s per-conversation routing
/// header. `None` without a session or for other targets.
#[must_use]
pub fn get_opencode_session_headers(
    target: &ClientIdentityTarget,
    session_id: Option<&str>,
) -> Option<Vec<(String, String)>> {
    if session_id.is_some() && is_opencode(target) {
        Some(vec![(
            "x-opencode-session".to_owned(),
            session_id?.to_owned(),
        )])
    } else {
        None
    }
}

/// `getIdentityHeaders`: `User-Agent` plus provider attribution.
///
/// APIs apply these below model, provider, and request headers, so
/// configured headers override them and `null` request headers remove
/// them.
#[must_use]
pub fn get_identity_headers(
    target: &ClientIdentityTarget,
    identity: Option<&ClientIdentity>,
    product: &str,
    fallback_version: &str,
) -> Vec<(String, String)> {
    let mut headers = vec![(
        "User-Agent".to_owned(),
        format_user_agent(identity, product, fallback_version),
    )];
    headers.extend(get_attribution_headers(target, identity));
    headers
}
