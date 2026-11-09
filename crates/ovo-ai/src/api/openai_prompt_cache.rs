//! The `OpenAI` prompt cache key limit (upstream `api/openai-prompt-cache.ts`).

/// `OPENAI_PROMPT_CACHE_KEY_MAX_LENGTH`.
pub const OPENAI_PROMPT_CACHE_KEY_MAX_LENGTH: usize = 64;

/// `clampOpenAIPromptCacheKey` — truncate to 64 Unicode code points
/// (`Array.from` semantics; a surrogate pair counts once).
#[must_use]
pub fn clamp_openai_prompt_cache_key(key: Option<&str>) -> Option<String> {
    let key = key?;
    if key.chars().count() <= OPENAI_PROMPT_CACHE_KEY_MAX_LENGTH {
        return Some(key.to_owned());
    }
    Some(
        key.chars()
            .take(OPENAI_PROMPT_CACHE_KEY_MAX_LENGTH)
            .collect(),
    )
}
