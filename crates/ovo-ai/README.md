# ovo-ai

`ovo-ai` is the Rust port of `pi-ai` 1.5.0, the unified multi-provider LLM
API used by the ovo agent harness. It speaks the provider wire protocols,
streams normalized assistant-message events, and resolves credentials —
byte-for-byte parity with the TypeScript package is checked by replay
vectors.

## Features

| Feature | Default | Effect                                                                                                                                                                                                                            |
| ------- | ------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `http`  | yes     | HTTP adapters and the built-in providers (`reqwest`). Without it the crate keeps the non-network surface: models registry, auth, streaming fold, JSON, catalogs, faux.                                                            |
| `host`  | yes     | Host-side capabilities: `auth::file` (file credential store) and `ProcessCommandRunner` for `!command` config values. Without it there is no default command runner — `!command` values resolve exactly as a failed command does. |

## Providers and protocols

M1 ships six built-in providers — `anthropic`, `openai`, `xai`, `deepseek`,
`groq`, `openrouter` — over three API families: `anthropic-messages`,
`openai-completions`, `openai-responses`. Model catalogs for every provider
are embedded at `catalog/<provider>.json`.

## Quick start

```rust,no_run
use std::sync::Arc;

use futures::StreamExt;
use ovo_ai::auth::ProcessEnv;
use ovo_ai::http::client::{HttpClient, HttpClientConfig};
use ovo_ai::models::CreateModelsOptions;
use ovo_ai::providers::builtin_models;
use ovo_ai::stream::AssistantMessageEvent;
use ovo_ai::types::content::TextOrBlocksWithImages;
use ovo_ai::types::context::Context;
use ovo_ai::types::message::{Message, UserMessage};

# async fn run() -> Result<(), Box<dyn std::error::Error>> {
let http = Arc::new(HttpClient::new(HttpClientConfig::default()));
let models = builtin_models(
    CreateModelsOptions {
        env: Some(Arc::new(ProcessEnv)),
        ..CreateModelsOptions::default()
    },
    &http,
)?;
let model = models
    .get_model("anthropic", "claude-fable-5")
    .ok_or("anthropic/claude-fable-5 not in catalog")?;
let context = Context {
    messages: vec![Message::User(UserMessage {
        content: TextOrBlocksWithImages::Text(String::from("Say hi in one word.")),
        timestamp: 0,
    })],
    ..Context::default()
};
let mut stream = models.stream(&model, &context, None);
while let Some(event) = stream.next().await {
    if let AssistantMessageEvent::TextDelta { delta, .. } = event {
        print!("{delta}");
    }
}
let message = stream.result().await;
#     Ok(())
# }
```

`ProcessEnv` reads API keys from the process environment
(`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, …); it is the only place the crate
touches `std::env`.

## Tool calling

Declare tools on the context:

```rust,ignore
context.tools = Some(vec![Tool {
    name: "get_time".into(),
    description: "Return the current time.".into(),
    parameters: json_schema_map,
    constrained_sampling: None,
}]);
```

When the reply's `stop_reason` is `toolUse`, each
`AssistantContentBlock::ToolCall` in `message.content` is one call: read
`arguments`, run the tool, append a `Message::ToolResult` and stream the
follow-up. `examples/stream.rs` runs a complete single-round loop.

## Parity

Every adapter, the models registry, auth and the stream fold are verified
against the published `@earendil-works/pi-ai` dist: `bun run vectors`
regenerates `vectors/ai/*.json` under Node (pi's own runtime semantics),
`parity.json` maps each capability to its vector file and the Rust replay
test, and `bun run lint` validates the mapping. Vectors are byte-frozen;
regenerate only when upstream or Node changes.

## License

MIT OR Apache-2.0
