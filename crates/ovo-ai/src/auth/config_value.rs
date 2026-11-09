//! `resolve-config-value.ts` port: `!command`, `$NAME`/`${NAME}`
//! templates, `$$`/`$!` escapes, per-store command cache.

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Mutex;

use crate::auth::env::{Env, env_get};

const fn is_name_start(c: char) -> bool {
    c.is_ascii_alphabetic() || c == '_'
}

const fn is_name_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_'
}

fn valid_env_name(name: &str) -> bool {
    !name.is_empty()
        && name.chars().all(is_name_char)
        && name.chars().next().is_some_and(is_name_start)
}

/// `TemplatePart`: literal text or an `$NAME`/`${NAME}` env reference.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TemplatePart {
    /// Literal text (`$$`/`$!` already unescaped).
    Literal(String),
    /// `$NAME`/`${NAME}` environment reference.
    Env(String),
}

/// `parseConfigValueTemplate`: `$` handling —
/// `$$`/`$!` → literal `$`/`!`; `${NAME}` valid name → env, else
/// literal `${...}`; `$NAME` longest-valid-prefix → env; trailing
/// lone `$`/`${` → literal.
#[expect(
    clippy::excessive_nesting,
    reason = "upstream's sequential parse loop over char positions maps naturally to match arms"
)]
#[must_use]
pub fn parse_config_value_template(config: &str) -> Vec<TemplatePart> {
    let mut parts: Vec<TemplatePart> = Vec::new();
    let mut literal = String::new();
    let chars: Vec<char> = config.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        let Some(&c) = chars.get(i) else {
            break;
        };
        if c != '$' {
            literal.push(c);
            i += 1;
            continue;
        }
        let Some(&next) = chars.get(i + 1) else {
            literal.push('$');
            i += 1;
            continue;
        };
        match next {
            '$' | '!' => {
                literal.push(next);
                i += 2;
            }
            '{' => {
                // Find the closing `}`.
                let mut name = String::new();
                let mut j = i + 2;
                let mut closed = false;
                while let Some(&ch) = chars.get(j) {
                    if ch == '}' {
                        closed = true;
                        break;
                    }
                    name.push(ch);
                    j += 1;
                }
                if !closed {
                    literal.push('$');
                    i += 1;
                    continue;
                }
                if valid_env_name(&name) {
                    if !literal.is_empty() {
                        parts.push(TemplatePart::Literal(std::mem::take(&mut literal)));
                    }
                    parts.push(TemplatePart::Env(name));
                } else {
                    // Invalid name → literal `${...}`.
                    literal.push('$');
                    literal.push('{');
                    literal.push_str(&name);
                    literal.push('}');
                }
                i = j + 1;
            }
            _ if is_name_start(next) => {
                let mut name = String::new();
                let mut j = i + 1;
                while let Some(&ch) = chars.get(j) {
                    if !is_name_char(ch) {
                        break;
                    }
                    name.push(ch);
                    j += 1;
                }
                if !literal.is_empty() {
                    parts.push(TemplatePart::Literal(std::mem::take(&mut literal)));
                }
                parts.push(TemplatePart::Env(name));
                i = j;
            }
            _ => {
                literal.push('$');
                i += 1;
            }
        }
    }
    if !literal.is_empty() {
        parts.push(TemplatePart::Literal(literal));
    }
    parts
}

/// `isCommandConfigValue` — `!`-prefixed values execute as shell.
#[must_use]
pub fn is_command_config_value(config: &str) -> bool {
    config.starts_with('!')
}

/// `getConfigValueEnvVarName`: when the whole value is one `$NAME`.
#[must_use]
pub fn get_config_value_env_var_name(config: &str) -> Option<String> {
    if is_command_config_value(config) {
        return None;
    }
    match parse_config_value_template(config).as_slice() {
        [TemplatePart::Env(name)] => Some(name.clone()),
        _ => None,
    }
}

/// `getConfigValueEnvVarNames`: distinct `$NAME`s in template order.
#[must_use]
pub fn get_config_value_env_var_names(config: &str) -> Vec<String> {
    if is_command_config_value(config) {
        return Vec::new();
    }
    let mut names = Vec::new();
    for part in parse_config_value_template(config) {
        if let TemplatePart::Env(name) = part
            && !names.contains(&name)
        {
            names.push(name);
        }
    }
    names
}

/// `getMissingConfigValueEnvVarNames`: `$NAME`s with no value.
#[must_use]
pub fn get_missing_config_value_env_var_names(config: &str, env: &dyn Env) -> Vec<String> {
    get_config_value_env_var_names(config)
        .into_iter()
        .filter(|name| env_get(env, name).is_none())
        .collect()
}

/// `isConfigValueConfigured`: every `$NAME` resolves.
#[must_use]
pub fn is_config_value_configured(config: &str, env: &dyn Env) -> bool {
    get_missing_config_value_env_var_names(config, env).is_empty()
}

/// `resolveTemplate`: `undefined` when any `$NAME` is unset.
#[must_use]
pub fn resolve_template(parts: &[TemplatePart], env: &dyn Env) -> Option<String> {
    let mut resolved = String::new();
    for part in parts {
        match part {
            TemplatePart::Literal(v) => resolved.push_str(v),
            TemplatePart::Env(name) => resolved.push_str(&env_get(env, name)?),
        }
    }
    Some(resolved)
}

/// `CommandRunner`: executes a `!command`, returns trimmed stdout.
///
/// `None` on failure/empty (upstream `value || undefined`). Injected
/// because the shell decision (`getShellConfig`) is host-owned; see
/// [`crate::auth::resolve::ProcessCommandRunner`].
pub trait CommandRunner: Send + Sync {
    /// Run `command` (the text after `!`), return trimmed stdout.
    fn run(&self, command: &str) -> Pin<Box<dyn Future<Output = Option<String>> + Send + '_>>;
}

/// `CommandCache`: process-lifetime result memo.
///
/// *Owned by the store* (ADR-020) — never a global.
#[derive(Debug, Default)]
pub struct CommandCache {
    inner: Mutex<HashMap<String, Option<String>>>,
}

impl CommandCache {
    /// `executeCommand` with memoization per command string.
    pub async fn run(&self, runner: &dyn CommandRunner, command: &str) -> Option<String> {
        if let Some(cached) = self
            .inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get(command)
        {
            return cached.clone();
        }
        let value = runner.run(command).await;
        self.inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .insert(command.to_owned(), value.clone());
        value
    }
}

/// `resolveConfigValue`: `!` → command via cache; else template.
pub async fn resolve_config_value(
    config: &str,
    env: &dyn Env,
    runner: &dyn CommandRunner,
    cache: &CommandCache,
) -> Option<String> {
    if is_command_config_value(config) {
        return cache.run(runner, config).await;
    }
    resolve_template(&parse_config_value_template(config), env)
}
