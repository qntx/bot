//! HTTP policy layer — upstream `utils/provider-retry.ts`,
//! `utils/error-body.ts`, `utils/headers.ts`, `utils/provider-env.ts`
//! and `utils/node-http-proxy.ts`.
//!
//! The retry/delay decisions, error normalization, header merging, proxy
//! selection, SSE decoding and (feature `http`) the pooled transport.
//! Provider env injection: upstream reads `process.env` behind the
//! `ProviderEnv` map; library code here takes the injected snapshot
//! only — callers resolve process env and pass it in.

#[cfg(feature = "http")]
pub mod client;
pub mod env;
pub mod error_body;
pub mod headers;
pub mod proxy;
pub mod retry;
pub mod sse;
