//! `node-http-proxy.ts`: `NO_PROXY` matching and proxy-URL selection —
//! pure decisions; reqwest wiring is part B.

use serde_json::{Map, Value};

/// `UNSUPPORTED_PROXY_PROTOCOL_MESSAGE`.
pub const UNSUPPORTED_PROXY_PROTOCOL_MESSAGE: &str = "Unsupported proxy protocol. SOCKS and PAC proxy URLs are not supported; use an HTTP or HTTPS proxy URL.";

fn default_port(protocol: &str) -> u16 {
    match protocol {
        "ftp" => 21,
        "gopher" => 70,
        "https" | "wss" => 443,
        _ => 80,
    }
}

/// `getProxyEnv`: lowercase then uppercase key in the injected map
/// (the caller's map already folds `process.env` in).
fn get_proxy_env<'m>(key: &str, env: &'m Map<String, Value>) -> Option<&'m str> {
    let lower = key.to_lowercase();
    let upper = key.to_uppercase();
    env.get(&lower)
        .and_then(Value::as_str)
        .or_else(|| env.get(&upper).and_then(Value::as_str))
}

fn strip_brackets(host: &str) -> &str {
    host.strip_prefix('[')
        .and_then(|h| h.strip_suffix(']'))
        .unwrap_or(host)
}

fn parse_no_proxy_entry(entry: &str) -> Option<(String, u16)> {
    let trimmed = entry.trim().to_lowercase();
    if trimmed.is_empty() {
        return None;
    }
    if let Some(rest) = trimmed.strip_prefix('[')
        && let Some(close) = rest.find(']')
    {
        let host = &rest[..close];
        let tail = &rest[close + 1..];
        if let Some(port_str) = tail.strip_prefix(':') {
            let port = port_str.parse::<u16>().unwrap_or(0);
            return Some((host.to_owned(), port));
        }
        return Some((host.to_owned(), 0));
    }
    if trimmed.matches(':').count() > 1 {
        return Some((trimmed, 0));
    }
    if let Some(colon) = trimmed.rfind(':')
        && trimmed[..colon].find(':').is_none()
        && let Ok(port) = trimmed[colon + 1..].parse::<u16>()
    {
        return Some((trimmed[..colon].to_owned(), port));
    }
    Some((trimmed, 0))
}

/// `shouldProxyHostname`: `NO_PROXY` `*` never proxies; entries match
/// exactly or as a `.` suffix after stripping `*.`, `.`, `*` prefixes;
/// a port mismatch keeps proxying.
fn should_proxy_hostname(hostname: &str, port: u16, env: &Map<String, Value>) -> bool {
    let no_proxy = get_proxy_env("no_proxy", env)
        .unwrap_or_default()
        .to_lowercase();
    if no_proxy.is_empty() {
        return true;
    }
    if no_proxy == "*" {
        return false;
    }
    let lowered = hostname.to_lowercase();
    let target = strip_brackets(&lowered);
    no_proxy.split([',', ' ']).all(|entry| {
        let Some((raw_host, parsed_port)) = parse_no_proxy_entry(entry) else {
            return true;
        };
        if parsed_port != 0 && parsed_port != port {
            return true;
        }
        let mut domain = strip_brackets(&raw_host).to_owned();
        if let Some(rest) = domain.strip_prefix("*.") {
            domain = rest.to_owned();
        } else if domain.starts_with('.') || domain.starts_with('*') {
            domain = domain[1..].to_owned();
        }
        if domain.is_empty() {
            return true;
        }
        if target == domain {
            return false;
        }
        if target.ends_with(&format!(".{domain}")) {
            return false;
        }
        true
    })
}

/// The proxy error `resolveHttpProxyUrlForTarget` throws.
#[derive(Debug)]
pub struct ProxyError {
    /// The upstream message.
    pub message: String,
}

impl std::fmt::Display for ProxyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for ProxyError {}

/// `getProxyForUrl`: `<protocol>_proxy` else `all_proxy`, adding the
/// `scheme://` prefix when missing; `""` when `NO_PROXY` covers the
/// target or no proxy is set.
#[must_use]
pub fn proxy_for_url(target_url: &str, env: &Map<String, Value>) -> String {
    let Ok(parsed) = url::Url::parse(target_url) else {
        return String::new();
    };
    if parsed.host_str().is_none_or(str::is_empty) {
        return String::new();
    }
    let protocol = parsed.scheme();
    let hostname = strip_brackets(parsed.host_str().unwrap_or_default()).to_owned();
    let port = parsed.port().unwrap_or_else(|| default_port(protocol));
    if !should_proxy_hostname(&hostname, port, env) {
        return String::new();
    }
    let mut proxy = get_proxy_env(&format!("{protocol}_proxy"), env)
        .or_else(|| get_proxy_env("all_proxy", env))
        .unwrap_or_default()
        .to_owned();
    if !proxy.is_empty() && !proxy.contains("://") {
        proxy = format!("{protocol}://{proxy}");
    }
    proxy
}

/// `resolveHttpProxyUrlForTarget`: the proxy URL for a target — `Err`
/// on an invalid or non-HTTP(S) proxy URL, `None` when unproxied.
///
/// # Errors
///
/// `Invalid proxy URL <json>: <reason>` and
/// [`UNSUPPORTED_PROXY_PROTOCOL_MESSAGE`] `Got <scheme>`.
pub fn resolve_http_proxy_url_for_target(
    target_url: &str,
    env: &Map<String, Value>,
) -> Result<Option<url::Url>, ProxyError> {
    let proxy = proxy_for_url(target_url, env);
    if proxy.is_empty() {
        return Ok(None);
    }
    let proxy_url = url::Url::parse(&proxy).map_err(|error| ProxyError {
        message: format!(
            "Invalid proxy URL {}: {error}",
            crate::json::to_string(&proxy).unwrap_or_default()
        ),
    })?;
    match proxy_url.scheme() {
        "http" | "https" => Ok(Some(proxy_url)),
        scheme => Err(ProxyError {
            message: format!("{UNSUPPORTED_PROXY_PROTOCOL_MESSAGE} Got {scheme}:"),
        }),
    }
}
