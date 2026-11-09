//! Protocol adapter layer — one module per upstream `api/*.ts`
//! implementation. Each adapter implements `ChatApi` (and image/classifier
//! traits where upstream does) so `create_provider` can dispatch by api.

/// `anthropic-messages` — the Anthropic Messages API adapter.
pub mod anthropic_messages;
pub mod constrained_sampling;
/// `openai-completions` — the OpenAI Chat Completions API adapter.
pub mod openai_completions;
pub mod openai_prompt_cache;
