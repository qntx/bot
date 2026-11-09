/**
 * `generate-anthropic.ts` — vectors for `anthropic-messages`'s request shape. Each case drives the
 * real dist `streamAnthropic` with a custom `fetch` that records the outgoing request
 * (method/url/headers/body) and answers with a minimal valid SSE (`message_start`/`message_stop`).
 *
 * Pi's adapter builds `headers` via `mergeHeaders(model, provider, request)` — the vector captures
 * what pi sets explicitly; the SDK's `x-stainless-*` headers are ignored (a `sdk-headers` class in
 * parity.json notes the divergence).
 */
import { writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const piAiDir = dirname(dirname(fileURLToPath(import.meta.resolve("@earendil-works/pi-ai"))));
type StreamFn = (
  model: unknown,
  context: unknown,
  options: Record<string, unknown>,
) => { result: () => Promise<unknown> };

function isStreamModule(value: unknown): value is { stream: StreamFn; streamSimple: StreamFn } {
  return (
    typeof value === "object" &&
    value !== null &&
    "stream" in value &&
    typeof value.stream === "function" &&
    "streamSimple" in value &&
    typeof value.streamSimple === "function"
  );
}

const anthropicModule: unknown = await import(
  pathToFileURL(join(piAiDir, "dist/api/anthropic-messages.js")).href
);
if (!isStreamModule(anthropicModule)) {
  throw new Error("anthropic-messages dist module has no stream export");
}
const streamAnthropic = anthropicModule.stream;
const streamAnthropicSimple = anthropicModule.streamSimple;

const SSE = [
  "event: message_start",
  'data: {"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude-test","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":10,"output_tokens":1}}}',
  "",
  "event: message_stop",
  'data: {"type":"message_stop"}',
  "",
  "",
].join("\n");

type CapturedRequest = {
  url: string;
  method: string;
  headers: Record<string, string>;
  body: string;
};

function captureFetch(recorded: CapturedRequest[]): typeof globalThis.fetch {
  return async (url: string | URL | Request, init?: RequestInit) => {
    const headers: Record<string, string> = {};
    for (const [key, value] of new Headers(init?.headers)) {
      // `x-stainless-*` is SDK telemetry; pi's default `user-agent` embeds the host platform and
      // Node version, so freezing it would make the vectors machine-dependent.
      if (!key.startsWith("x-stainless-") && !(key === "user-agent" && value.startsWith("pi/"))) {
        headers[key] = value;
      }
    }
    recorded.push({
      url: url instanceof URL ? url.href : typeof url === "string" ? url : url.url,
      method: init?.method ?? "POST",
      headers,
      body: typeof init?.body === "string" ? init.body : await new Response(init?.body).text(),
    });
    return new Response(SSE, { status: 200, headers: { "content-type": "text/event-stream" } });
  };
}

const BASE_MODEL = {
  id: "claude-test",
  name: "Test",
  api: "anthropic-messages",
  provider: "anthropic",
  baseUrl: "https://api.anthropic.com",
  input: ["text", "image"],
  cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
  reasoning: true,
  contextWindow: 200_000,
  maxTokens: 8192,
};

const BASE_OPTIONS = {
  apiKey: "sk-ant-api03-test",
  env: {},
  headers: {},
  identity: null,
  sessionId: null,
  signal: null,
  maxRetries: 0,
};

const user = (text: string) => ({ role: "user", content: text, timestamp: 0 });
const ctx = (messages: unknown[], systemPrompt?: string, tools?: unknown[]) => ({
  systemPrompt,
  messages,
  tools: tools ?? [],
});

const tool = (name: string, constrainedSampling?: unknown) => ({
  name,
  description: `${name} tool`,
  parameters: {
    type: "object",
    properties: { path: { type: "string" }, limit: { type: "number", minimum: 1 } },
    required: ["path"],
  },
  ...(constrainedSampling === undefined ? {} : { constrainedSampling }),
});
const withCompat = (compat: Record<string, unknown>, extra: Record<string, unknown> = {}) => ({
  ...BASE_MODEL,
  ...extra,
  compat,
});
const system = (toolsAdded?: unknown[], toolsRemoved?: unknown[]) => ({
  role: "system",
  content: "You are helpful.",
  ...(toolsAdded === undefined ? {} : { toolsAdded }),
  ...(toolsRemoved === undefined ? {} : { toolsRemoved }),
  timestamp: 0,
});
const assistantWithLevel = (level: string) => ({
  role: "assistant",
  content: [{ type: "text", text: "earlier" }],
  api: "anthropic-messages",
  provider: "anthropic",
  model: "claude-test",
  providerThinkingLevel: level,
  usage: {
    input: 0,
    output: 0,
    cacheRead: 0,
    cacheWrite: 0,
    totalTokens: 0,
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
  },
  stopReason: "stop",
  timestamp: 0,
});

const THINKING_CASES = [
  {
    name: "thinking-budget",
    model: BASE_MODEL,
    context: ctx([user("think")]),
    options: {
      ...BASE_OPTIONS,
      thinkingEnabled: true,
      thinkingBudgetTokens: 2048,
      temperature: 0.3,
    },
  },
  {
    name: "thinking-budget-default",
    model: BASE_MODEL,
    context: ctx([user("think")]),
    options: { ...BASE_OPTIONS, thinkingEnabled: true, interleavedThinking: false },
  },
  {
    name: "thinking-adaptive-effort",
    model: withCompat({ forceAdaptiveThinking: true }),
    context: ctx([user("think")]),
    options: {
      ...BASE_OPTIONS,
      thinkingEnabled: true,
      effort: "medium",
      thinkingDisplay: "omitted",
    },
  },
  {
    name: "thinking-disabled",
    model: BASE_MODEL,
    context: ctx([user("no thinking")]),
    options: { ...BASE_OPTIONS, thinkingEnabled: false },
  },
  {
    name: "managed-effort",
    model: withCompat({ supportsMidConvoEffort: true }),
    context: ctx([user("one"), assistantWithLevel("medium"), user("two")]),
    options: { ...BASE_OPTIONS, effort: "xhigh", temperature: 0.5 },
  },
  {
    name: "native-tool-changes",
    model: withCompat({ supportsMidConvoSystemMessages: true, supportsMidConvoToolChanges: true }),
    context: ctx([
      system([tool("Read")]),
      user("one"),
      system([tool("Grep")], [{ name: "Read" }]),
      user("two"),
    ]),
    options: BASE_OPTIONS,
  },
  {
    name: "strict-tools-prefer",
    model: withCompat({ supportsStrictTools: true }),
    context: ctx([
      system([
        tool("Read", { type: "json_schema", strict: "prefer" }),
        {
          name: "Write",
          description: "write",
          parameters: { type: "object", properties: { path: { type: "string" } } },
          constrainedSampling: { type: "json_schema", strict: "prefer" },
        },
      ]),
      user("strict"),
    ]),
    options: BASE_OPTIONS,
  },
  {
    name: "strict-tools-require-unsupported",
    model: BASE_MODEL,
    context: ctx([
      system([tool("Read", { type: "json_schema", strict: "require" })]),
      user("strict"),
    ]),
    options: BASE_OPTIONS,
  },
  {
    name: "metadata-tool-choice-fallbacks",
    model: withCompat({
      allowedFallbackModels: [{ provider: "anthropic", model: "claude-fallback" }],
    }),
    context: ctx([system([tool("Read")]), user("pick")]),
    options: {
      ...BASE_OPTIONS,
      metadata: { user_id: "u-1", other: 1 },
      toolChoice: { type: "tool", name: "Read" },
    },
  },
  {
    name: "configured-beta-header",
    model: BASE_MODEL,
    context: ctx([user("beta")]),
    options: {
      ...BASE_OPTIONS,
      thinkingEnabled: true,
      headers: { "Anthropic-Beta": "a-1, b-2,a-1", "x-extra": "1" },
    },
  },
  {
    name: "configured-beta-null",
    model: BASE_MODEL,
    context: ctx([user("beta")]),
    options: { ...BASE_OPTIONS, thinkingEnabled: true, headers: { "anthropic-beta": null } },
  },
  {
    name: "simple-no-reasoning",
    simple: true,
    model: BASE_MODEL,
    context: ctx([user("simple")]),
    options: { ...BASE_OPTIONS, toolChoice: "auto" },
  },
  {
    name: "simple-reasoning-budget",
    simple: true,
    model: BASE_MODEL,
    context: ctx([user("simple")]),
    options: { ...BASE_OPTIONS, reasoning: "high", maxTokens: 4000 },
  },
  {
    name: "simple-reasoning-adaptive",
    simple: true,
    model: withCompat(
      { forceAdaptiveThinking: true },
      { thinkingLevelMap: { minimal: "low", xhigh: "xhigh" } },
    ),
    context: ctx([user("simple")]),
    options: { ...BASE_OPTIONS, reasoning: "xhigh" },
  },
];

const CASES = [
  {
    name: "plain-text",
    model: BASE_MODEL,
    context: ctx([user("hello")]),
    options: BASE_OPTIONS,
  },
  {
    name: "system-prompt",
    model: BASE_MODEL,
    context: ctx([user("hello")], "you are helpful"),
    options: BASE_OPTIONS,
  },
  {
    name: "image-only",
    model: BASE_MODEL,
    context: ctx([
      {
        role: "user",
        content: [{ type: "image", data: "aW1hZ2U=", mimeType: "image/png" }],
        timestamp: 0,
      },
    ]),
    options: BASE_OPTIONS,
  },
  {
    name: "mixed-text-image",
    model: BASE_MODEL,
    context: ctx([
      {
        role: "user",
        content: [
          { type: "text", text: "see this" },
          { type: "image", data: "aW1hZ2U=", mimeType: "image/jpeg" },
        ],
        timestamp: 0,
      },
    ]),
    options: BASE_OPTIONS,
  },
  {
    name: "assistant-thinking-signature",
    model: BASE_MODEL,
    context: ctx([
      user("q"),
      {
        role: "assistant",
        content: [
          { type: "thinking", thinking: "hmm", thinkingSignature: "sig-1" },
          { type: "text", text: "answer" },
        ],
        api: "anthropic-messages",
        provider: "anthropic",
        model: "claude-test",
        usage: {
          input: 0,
          output: 0,
          cacheRead: 0,
          cacheWrite: 0,
          totalTokens: 0,
          cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
        },
        stopReason: "stop",
        timestamp: 0,
      },
      user("followup"),
    ]),
    options: BASE_OPTIONS,
  },
  {
    name: "redacted-thinking",
    model: BASE_MODEL,
    context: ctx([
      user("q"),
      {
        role: "assistant",
        content: [
          { type: "thinking", thinking: "", thinkingSignature: "redacted-blob", redacted: true },
        ],
        api: "anthropic-messages",
        provider: "anthropic",
        model: "claude-test",
        usage: {
          input: 0,
          output: 0,
          cacheRead: 0,
          cacheWrite: 0,
          totalTokens: 0,
          cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
        },
        stopReason: "stop",
        timestamp: 0,
      },
      user("next"),
    ]),
    options: BASE_OPTIONS,
  },
  {
    name: "tool-use-and-result",
    model: BASE_MODEL,
    context: ctx(
      [
        user("check"),
        {
          role: "assistant",
          content: [
            { type: "toolCall", id: "call_1", name: "Read", arguments: { path: "/tmp/a" } },
          ],
          api: "anthropic-messages",
          provider: "anthropic",
          model: "claude-test",
          usage: {
            input: 0,
            output: 0,
            cacheRead: 0,
            cacheWrite: 0,
            totalTokens: 0,
            cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
          },
          stopReason: "toolUse",
          timestamp: 0,
        },
        {
          role: "toolResult",
          toolCallId: "call_1",
          toolName: "Read",
          content: [{ type: "text", text: "file contents" }],
          isError: false,
          timestamp: 0,
        },
      ],
      undefined,
    ),
    options: BASE_OPTIONS,
  },
  {
    name: "tool-result-error-with-image",
    model: BASE_MODEL,
    context: ctx([
      user("check"),
      {
        role: "assistant",
        content: [{ type: "toolCall", id: "call_1", name: "Screenshot", arguments: {} }],
        api: "anthropic-messages",
        provider: "anthropic",
        model: "claude-test",
        usage: {
          input: 0,
          output: 0,
          cacheRead: 0,
          cacheWrite: 0,
          totalTokens: 0,
          cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
        },
        stopReason: "toolUse",
        timestamp: 0,
      },
      {
        role: "toolResult",
        toolCallId: "call_1",
        toolName: "Screenshot",
        content: [
          { type: "text", text: "failed" },
          { type: "image", data: "aW1hZ2U=", mimeType: "image/png" },
        ],
        isError: true,
        timestamp: 0,
      },
    ]),
    options: BASE_OPTIONS,
  },
  {
    name: "consecutive-tool-results",
    model: BASE_MODEL,
    context: ctx([
      user("check"),
      {
        role: "assistant",
        content: [
          { type: "toolCall", id: "c1", name: "Read", arguments: {} },
          { type: "toolCall", id: "c2", name: "Read", arguments: {} },
        ],
        api: "anthropic-messages",
        provider: "anthropic",
        model: "claude-test",
        usage: {
          input: 0,
          output: 0,
          cacheRead: 0,
          cacheWrite: 0,
          totalTokens: 0,
          cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
        },
        stopReason: "toolUse",
        timestamp: 0,
      },
      {
        role: "toolResult",
        toolCallId: "c1",
        toolName: "Read",
        content: [{ type: "text", text: "r1" }],
        isError: false,
        timestamp: 0,
      },
      {
        role: "toolResult",
        toolCallId: "c2",
        toolName: "Read",
        content: [{ type: "text", text: "r2" }],
        isError: false,
        timestamp: 0,
      },
    ]),
    options: BASE_OPTIONS,
  },
  {
    name: "oauth-token",
    model: BASE_MODEL,
    context: ctx([user("hi")], "system"),
    options: { ...BASE_OPTIONS, apiKey: "sk-ant-oat-oauth-token" },
  },
  {
    name: "cache-retention-long",
    model: BASE_MODEL,
    context: ctx([user("hi")], "system"),
    options: { ...BASE_OPTIONS, cacheRetention: "long" },
  },
  {
    name: "cache-retention-none",
    model: BASE_MODEL,
    context: ctx([user("hi")], "system"),
    options: { ...BASE_OPTIONS, cacheRetention: "none" },
  },
  {
    name: "temperature-and-tools",
    model: BASE_MODEL,
    context: ctx([user("hi")], "system", [
      {
        name: "Read",
        description: "read a file",
        parameters: {
          type: "object",
          properties: { path: { type: "string" } },
          required: ["path"],
        },
      },
    ]),
    options: { ...BASE_OPTIONS, temperature: 0.5 },
  },
  ...THINKING_CASES,
];

const results: Array<{
  name: string;
  simple?: boolean;
  model: unknown;
  context: unknown;
  options: unknown;
  request: CapturedRequest | { error: string };
}> = [];
for (const testCase of CASES) {
  const recorded: CapturedRequest[] = [];
  let failure: string | undefined;
  const fetch = captureFetch(recorded);
  const origFetch = globalThis.fetch;
  globalThis.fetch = fetch;
  try {
    const run = "simple" in testCase && testCase.simple ? streamAnthropicSimple : streamAnthropic;
    const stream = run(testCase.model, testCase.context, { ...testCase.options, fetch });
    // Cases run one at a time: each swaps the global fetch.
    // oxlint-disable-next-line no-await-in-loop -- sequential by design (global fetch swap)
    const message = await stream.result();
    if (typeof message === "object" && message !== null && "errorMessage" in message) {
      failure = String(message.errorMessage);
    }
  } catch (error) {
    failure = error instanceof Error ? error.message : String(error);
  } finally {
    globalThis.fetch = origFetch;
  }
  results.push({
    name: testCase.name,
    ...("simple" in testCase && testCase.simple ? { simple: true } : {}),
    model: testCase.model,
    context: testCase.context,
    options: testCase.options,
    request: recorded[0] ?? { error: failure ?? "no request was sent" },
  });
}
writeFileSync(
  join(import.meta.dirname, "../../vectors/ai/anthropic-request.json"),
  `${JSON.stringify(results, null, "\t")}\n`,
);
console.log(`wrote ${results.length} cases`);
