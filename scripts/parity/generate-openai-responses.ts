/**
 * `generate-openai-responses.ts` — vectors for `openai-responses`'s request shape. Each case drives
 * the real dist `stream`/`streamSimple` with a recording `fetch` that answers a minimal completed
 * stream, and records the outgoing url, the headers pi sets (the SDK's `x-stainless-*` headers are
 * the `sdk-headers` class; pi's default `user-agent` embeds the host platform), and the body; a
 * case that fails before sending records the terminal `errorMessage`.
 *
 * The dist adapter reads a transcript context: tools arrive as `toolsAdded` on system messages. The
 * OpenAI SDK reads `OPENAI_ORG_ID`, `OPENAI_PROJECT_ID` and `OPENAI_BASE_URL` from the process
 * environment; they are removed before import because ovo never reads them.
 */
import { writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

for (const key of ["OPENAI_ORG_ID", "OPENAI_PROJECT_ID", "OPENAI_BASE_URL"]) {
  Reflect.deleteProperty(process.env, key);
}

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

const responses: unknown = await import(
  pathToFileURL(join(piAiDir, "dist/api/openai-responses.js")).href
);
if (!isStreamModule(responses)) {
  throw new Error("openai-responses dist module has no stream export");
}

const SSE = [
  "event: response.completed",
  'data: {"type":"response.completed","response":{"id":"resp_1","status":"completed","output":[]}}',
  "",
  "",
].join("\n");

type CapturedRequest = { url: string; headers: Record<string, string>; body: string };

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
      headers,
      body: typeof init?.body === "string" ? init.body : await new Response(init?.body).text(),
    });
    return new Response(SSE, { status: 200, headers: { "content-type": "text/event-stream" } });
  };
}

type Json = Record<string, unknown>;

const OPENAI = { provider: "openai", baseUrl: "https://api.openai.com/v1" };
const OPENROUTER = { provider: "openrouter", baseUrl: "https://openrouter.ai/api/v1" };
const CUSTOM = { provider: "custom", baseUrl: "https://llm.example.com/v1" };

const model = (overrides: Json = {}): Json => ({
  id: "m",
  name: "M",
  api: "openai-responses",
  ...OPENAI,
  input: ["text", "image"],
  cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
  reasoning: false,
  contextWindow: 128_000,
  maxTokens: 4096,
  ...overrides,
});
const reasoningModel = (overrides: Json = {}) => model({ reasoning: true, ...overrides });

const options = (extra: Json = {}): Json => ({ apiKey: "sk-test", maxRetries: 0, ...extra });

const USAGE = {
  input: 0,
  output: 0,
  cacheRead: 0,
  cacheWrite: 0,
  totalTokens: 0,
  cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
};
const user = (content: unknown) => ({ role: "user", content, timestamp: 0 });
const system = (content: string, extra: Json = {}) => ({
  role: "system",
  content,
  ...extra,
  timestamp: 0,
});
const assistant = (content: unknown[], extra: Json = {}) => ({
  role: "assistant",
  content,
  api: "openai-responses",
  provider: "openai",
  model: "m",
  usage: USAGE,
  stopReason: content.some(
    (block) =>
      typeof block === "object" && block !== null && "type" in block && block.type === "toolCall",
  )
    ? "toolUse"
    : "stop",
  timestamp: 0,
  ...extra,
});
const toolResult = (toolCallId: string, content: unknown[], extra: Json = {}) => ({
  role: "toolResult",
  toolCallId,
  toolName: "read",
  content,
  isError: false,
  timestamp: 0,
  ...extra,
});
const text = (value: string, textSignature?: string) => ({
  type: "text",
  text: value,
  ...(textSignature === undefined ? {} : { textSignature }),
});
const image = { type: "image", data: "aGk=", mimeType: "image/png" };
const thinking = (value: string, thinkingSignature?: string) => ({
  type: "thinking",
  thinking: value,
  ...(thinkingSignature === undefined ? {} : { thinkingSignature }),
});
const READ_ARGS: Json = { path: "a" };
const call = (id: string, name = "read", args = READ_ARGS, extra: Json = {}) => ({
  type: "toolCall",
  id,
  name,
  arguments: args,
  ...extra,
});
const ctx = (...messages: unknown[]) => ({ messages });

const tool = (name = "read", constrainedSampling?: unknown, parameters?: Json) => ({
  name,
  description: `${name} tool`,
  parameters: parameters ?? {
    type: "object",
    properties: { path: { type: "string" } },
    required: ["path"],
  },
  ...(constrainedSampling === undefined ? {} : { constrainedSampling }),
});
const grammarTool = (variants: Json) =>
  tool(
    "patch",
    { type: "grammar", variants },
    { type: "object", properties: { input: { type: "string" } }, required: ["input"] },
  );
const withTools = (...tools: unknown[]) => system("You are helpful.", { toolsAdded: tools });
const conversation = ctx(system("You are helpful."), user("hi"));
const REASONING_ITEM = JSON.stringify({
  type: "reasoning",
  id: "rs_1",
  summary: [{ type: "summary_text", text: "thought" }],
  encrypted_content: "enc",
});

type Case = { name: string; model: Json; context: unknown; options: Json; simple?: boolean };

const BASIC: Case[] = [
  { name: "basic", model: model(), context: conversation, options: options() },
  { name: "user-string-only", model: model(), context: ctx(user("hi")), options: options() },
  {
    name: "user-blocks",
    model: model(),
    context: ctx(user([text("look"), image, text("")])),
    options: options(),
  },
  {
    name: "user-empty-blocks",
    model: model(),
    context: ctx(user([]), user("hi")),
    options: options(),
  },
  { name: "developer-role", model: reasoningModel(), context: conversation, options: options() },
  {
    name: "developer-role-disabled",
    model: reasoningModel({ compat: { supportsDeveloperRole: false } }),
    context: conversation,
    options: options(),
  },
  {
    name: "system-mid-convo-folded",
    model: model(),
    context: ctx(system("Base."), user("a"), system("Later rule."), user("b")),
    options: options(),
  },
  {
    name: "system-mid-convo-in-place",
    model: model({ compat: { supportsMidConvoSystemMessages: true } }),
    context: ctx(system("Base."), user("a"), system("Later rule."), user("b")),
    options: options(),
  },
  {
    name: "base-url-trailing-slash",
    model: model({ baseUrl: "https://llm.example.com/v1/", provider: "custom" }),
    context: conversation,
    options: options(),
  },
];

const REASONING: Case[] = [
  {
    name: "reasoning-effort",
    model: reasoningModel(),
    context: conversation,
    options: options({ reasoningEffort: "high" }),
  },
  {
    name: "reasoning-effort-mapped",
    model: reasoningModel({ thinkingLevelMap: { high: "xhigh", low: null } }),
    context: conversation,
    options: options({ reasoningEffort: "high" }),
  },
  {
    name: "reasoning-effort-null-mapped",
    model: reasoningModel({ thinkingLevelMap: { low: null } }),
    context: conversation,
    options: options({ reasoningEffort: "low", reasoningSummary: "concise" }),
  },
  {
    name: "reasoning-off-default",
    model: reasoningModel(),
    context: conversation,
    options: options(),
  },
  {
    name: "reasoning-off-mapped",
    model: reasoningModel({ thinkingLevelMap: { off: "minimal" } }),
    context: conversation,
    options: options(),
  },
  {
    name: "reasoning-off-unsupported",
    model: reasoningModel({ thinkingLevelMap: { off: null } }),
    context: conversation,
    options: options(),
  },
  {
    name: "reasoning-summary-only",
    model: reasoningModel(),
    context: conversation,
    options: options({ reasoningSummary: "detailed" }),
  },
  {
    name: "reasoning-summary-null",
    model: reasoningModel(),
    context: conversation,
    options: options({ reasoningEffort: "low", reasoningSummary: null }),
  },
  {
    name: "reasoning-github-copilot-off",
    model: reasoningModel({ provider: "github-copilot", baseUrl: "https://api.githubcopilot.com" }),
    context: conversation,
    options: options(),
  },
  {
    name: "reasoning-xai-off",
    model: reasoningModel({ provider: "xai", baseUrl: "https://api.x.ai/v1" }),
    context: conversation,
    options: options(),
  },
  {
    name: "reasoning-xai-effort",
    model: reasoningModel({ provider: "xai", baseUrl: "https://api.x.ai/v1" }),
    context: conversation,
    options: options({ reasoningEffort: "medium" }),
  },
  {
    name: "reasoning-non-reasoning-model",
    model: model(),
    context: conversation,
    options: options({ reasoningEffort: "high" }),
  },
  {
    name: "configuration-update",
    model: reasoningModel({
      compat: { supportsMidConvoEffort: true },
      thinkingLevelMap: { high: "x" },
    }),
    context: ctx(
      system("Base."),
      user("a"),
      assistant([text("one")], { providerThinkingLevel: "low" }),
      user("b"),
      assistant([text("two")], { model: "other", providerThinkingLevel: "high" }),
      user("c"),
      assistant([text("three")]),
      user("d"),
    ),
    options: options({ reasoningEffort: "high" }),
  },
  {
    name: "configuration-update-without-effort",
    model: reasoningModel({ compat: { supportsMidConvoEffort: true } }),
    context: ctx(user("a"), assistant([text("one")], { providerThinkingLevel: "low" }), user("b")),
    options: options(),
  },
  {
    name: "chatgpt-sign-in",
    model: reasoningModel({ compat: { supportsMidConvoEffort: true } }),
    context: conversation,
    options: options({
      apiKey: "eyJ-access-token",
      reasoningEffort: "high",
      maxTokens: 500,
      temperature: 0.5,
      sessionId: "s1",
      cacheRetention: "long",
    }),
  },
];

const CACHE: Case[] = [
  {
    name: "cache-short",
    model: model(),
    context: conversation,
    options: options({ sessionId: "s1" }),
  },
  {
    name: "cache-long",
    model: model(),
    context: conversation,
    options: options({ sessionId: "s1", cacheRetention: "long" }),
  },
  {
    name: "cache-long-unsupported",
    model: model({ compat: { supportsLongCacheRetention: false } }),
    context: conversation,
    options: options({ sessionId: "s1", cacheRetention: "long" }),
  },
  {
    name: "cache-none",
    model: model(),
    context: conversation,
    options: options({ sessionId: "s1", cacheRetention: "none" }),
  },
  {
    name: "cache-env-long",
    model: model(),
    context: conversation,
    options: options({ sessionId: "s1", env: { PI_CACHE_RETENTION: "long" } }),
  },
  {
    name: "cache-explicit-none",
    model: model({ compat: { supportsExplicitPromptCacheMode: true } }),
    context: conversation,
    options: options({ sessionId: "s1", cacheRetention: "none" }),
  },
  {
    name: "cache-explicit-long",
    model: model({ compat: { supportsExplicitPromptCacheMode: true } }),
    context: conversation,
    options: options({ sessionId: "s1", cacheRetention: "long" }),
  },
  {
    name: "cache-explicit-short",
    model: model({ compat: { supportsExplicitPromptCacheMode: true } }),
    context: conversation,
    options: options({ sessionId: "s1" }),
  },
  {
    name: "cache-key-clamped",
    model: model(),
    context: conversation,
    options: options({ sessionId: "k".repeat(80) }),
  },
  {
    name: "cache-empty-session",
    model: model(),
    context: conversation,
    options: options({ sessionId: "" }),
  },
];

const FIELDS: Case[] = [
  {
    name: "max-output-tokens-floor",
    model: model(),
    context: conversation,
    options: options({ maxTokens: 8 }),
  },
  {
    name: "max-output-tokens",
    model: model(),
    context: conversation,
    options: options({ maxTokens: 900 }),
  },
  {
    name: "max-output-tokens-unsupported",
    model: model({ compat: { supportsMaxOutputTokens: false } }),
    context: conversation,
    options: options({ maxTokens: 900 }),
  },
  {
    name: "max-output-tokens-zero",
    model: model(),
    context: conversation,
    options: options({ maxTokens: 0 }),
  },
  {
    name: "temperature",
    model: model(),
    context: conversation,
    options: options({ temperature: 0 }),
  },
  {
    name: "service-tier",
    model: model(),
    context: conversation,
    options: options({ serviceTier: "flex" }),
  },
  {
    name: "tool-choice-object",
    model: model(),
    context: ctx(withTools(tool()), user("q")),
    options: options({ toolChoice: { type: "function", name: "read" } }),
  },
  {
    name: "sampling-override",
    model: reasoningModel({ samplingParams: { top_p: 0.9 } }),
    context: conversation,
    options: options({
      reasoningEffort: "low",
      temperature: 0.2,
      samplingParams: { temperature: 0.7, seed: 1 },
    }),
  },
];

const HEADERS: Case[] = [
  {
    name: "affinity-openai",
    model: model(),
    context: conversation,
    options: options({ sessionId: "s1" }),
  },
  {
    name: "affinity-openrouter",
    model: model({ ...OPENROUTER }),
    context: conversation,
    options: options({ sessionId: "s1" }),
  },
  {
    name: "affinity-nosession",
    model: model({ ...CUSTOM, compat: { sessionAffinityFormat: "openai-nosession" } }),
    context: conversation,
    options: options({ sessionId: "s1" }),
  },
  {
    name: "affinity-cache-none",
    model: model(),
    context: conversation,
    options: options({ sessionId: "s1", cacheRetention: "none" }),
  },
  {
    name: "headers-merge-order",
    model: model({ headers: { "X-Model": "m", "x-shared": "model" } }),
    context: conversation,
    options: options({
      sessionId: "s1",
      headers: { "x-shared": "request", "x-client-request-id": "r" },
    }),
  },
  {
    name: "headers-null-removal",
    model: model({ headers: { "x-model": "m" } }),
    context: conversation,
    options: options({ headers: { "x-model": null, accept: null } }),
  },
  {
    name: "identity-openrouter",
    model: model({ ...OPENROUTER }),
    context: conversation,
    options: options({
      identity: {
        attribution: { name: "ovo", url: "https://ovo.example", categories: ["cli-agent"] },
      },
    }),
  },
  {
    name: "github-copilot-agent-vision",
    model: model({ provider: "github-copilot", baseUrl: "https://api.githubcopilot.com" }),
    context: ctx(
      withTools(tool()),
      user([text("look"), image]),
      assistant([call("c1|fc_1")], { provider: "github-copilot" }),
      toolResult("c1|fc_1", [text("ok")]),
    ),
    options: options(),
  },
  { name: "auth-no-key", model: model(), context: conversation, options: { maxRetries: 0 } },
  {
    name: "auth-header-only",
    model: model({ ...CUSTOM }),
    context: conversation,
    options: { maxRetries: 0, headers: { Authorization: "Bearer from-header" } },
  },
];

const MESSAGES: Case[] = [
  {
    name: "assistant-text-signatures",
    model: model(),
    context: ctx(
      user("a"),
      assistant([
        text("v1", JSON.stringify({ v: 1, id: "msg_abc", phase: "final_answer" })),
        text("legacy", "msg_legacy"),
        text("long", `msg_${"x".repeat(70)}`),
        text("plain"),
        text("bad-phase", JSON.stringify({ v: 1, id: "msg_p", phase: "other" })),
        text("v2", JSON.stringify({ v: 2, id: "msg_v2" })),
      ]),
      user("b"),
    ),
    options: options(),
  },
  {
    name: "assistant-fallback-ids-skip-empty",
    model: model(),
    context: ctx(
      system("Base."),
      user("a"),
      assistant([]),
      user([]),
      assistant([text("first"), text("second")]),
      user("b"),
    ),
    options: options(),
  },
  {
    name: "assistant-thinking-replay",
    model: reasoningModel(),
    context: ctx(
      user("a"),
      assistant([thinking("thought", REASONING_ITEM), thinking("unsigned"), text("answer")]),
      user("b"),
    ),
    options: options({ reasoningEffort: "low" }),
  },
  {
    name: "assistant-thinking-invalid-signature",
    model: reasoningModel(),
    context: ctx(user("a"), assistant([thinking("t", "{not json"), text("x")]), user("b")),
    options: options(),
  },
  {
    name: "tool-call-ids-same-model",
    model: model(),
    context: ctx(
      withTools(tool()),
      user("a"),
      assistant([call("call_1|fc_1"), call("call_2|item_2"), call("call_3")]),
      toolResult("call_1|fc_1", [text("one")]),
      toolResult("call_2|item_2", [text("two")]),
      toolResult("call_3", [text("three")]),
      user("b"),
    ),
    options: options(),
  },
  {
    name: "tool-call-ids-different-model",
    model: model(),
    context: ctx(
      withTools(tool()),
      user("a"),
      assistant([call("call_1|fc_1", "read", { path: "a" }, { namespace: "ns" })], {
        model: "other",
      }),
      toolResult("call_1|fc_1", [text("one")]),
      user("b"),
    ),
    options: options(),
  },
  {
    name: "tool-call-namespace-same-model",
    model: model(),
    context: ctx(
      withTools(tool()),
      user("a"),
      assistant([call("call_1|fc_1", "read", { path: "a" }, { namespace: "ns" })]),
      toolResult("call_1|fc_1", [text("one")]),
    ),
    options: options(),
  },
  {
    name: "tool-call-ids-foreign-provider",
    model: model(),
    context: ctx(
      withTools(tool()),
      user("a"),
      assistant([call("toolu_1|item.with.dots"), call("call ä|x")], {
        provider: "anthropic",
        api: "anthropic-messages",
        model: "claude",
      }),
      toolResult("toolu_1|item.with.dots", [text("one")]),
      toolResult("call ä|x", [text("two")]),
    ),
    options: options(),
  },
  {
    name: "tool-call-ids-other-target-provider",
    model: model({ ...CUSTOM }),
    context: ctx(
      withTools(tool()),
      user("a"),
      assistant([call("call_1|fc_1")], { provider: "custom" }),
      toolResult("call_1|fc_1", [text("one")]),
    ),
    options: options(),
  },
  {
    name: "tool-results-content",
    model: model(),
    context: ctx(
      withTools(tool()),
      user("a"),
      assistant([call("c1|fc_1"), call("c2|fc_2"), call("c3|fc_3"), call("c4|fc_4")]),
      toolResult("c1|fc_1", [text("text"), image]),
      toolResult("c2|fc_2", [image]),
      toolResult("c3|fc_3", []),
      toolResult("c4|fc_4", [text("  ")]),
    ),
    options: options(),
  },
  {
    name: "tool-results-no-image-model",
    model: model({ input: ["text"] }),
    context: ctx(
      withTools(tool()),
      user("a"),
      assistant([call("c1|fc_1"), call("c2|fc_2")]),
      toolResult("c1|fc_1", [text("text"), image]),
      toolResult("c2|fc_2", [image]),
    ),
    options: options(),
  },
  {
    name: "grammar-call-replay",
    model: model({ compat: { supportsOpenAIGrammarTools: true } }),
    context: ctx(
      withTools(grammarTool({ openai_lark: "start: WORD" })),
      user("a"),
      assistant([
        call("c1|ctc_1", "patch", { input: "abc" }),
        call("c2|fc_2", "patch", { input: "d" }),
      ]),
      toolResult("c1|ctc_1", [text("ok")], { toolName: "patch" }),
      toolResult("c2|fc_2", [text("ok")], { toolName: "patch" }),
    ),
    options: options(),
  },
  {
    name: "grammar-call-bad-input",
    model: model({ compat: { supportsOpenAIGrammarTools: true } }),
    context: ctx(
      withTools(grammarTool({ openai_lark: "start: WORD" })),
      user("a"),
      assistant([call("c1|ctc_1", "patch", { input: 1 })]),
    ),
    options: options(),
  },
];

const TOOLS: Case[] = [
  {
    name: "tools-function",
    model: model(),
    context: ctx(withTools(tool()), user("q")),
    options: options(),
  },
  {
    name: "tools-strict-supported",
    model: model({ compat: { supportsStrictMode: true } }),
    context: ctx(
      withTools(tool(), tool("strict", { type: "json_schema", strict: "prefer" })),
      user("q"),
    ),
    options: options(),
  },
  {
    name: "tools-strict-require-unsupported",
    model: model(),
    context: ctx(withTools(tool("strict", { type: "json_schema", strict: "require" })), user("q")),
    options: options(),
  },
  {
    name: "tools-grammar",
    model: model({ compat: { supportsOpenAIGrammarTools: true } }),
    context: ctx(
      withTools(grammarTool({ openai_lark: "start: WORD", openai_regex: "a+" })),
      user("q"),
    ),
    options: options(),
  },
  {
    name: "tools-grammar-unsupported",
    model: model(),
    context: ctx(withTools(grammarTool({ openai_lark: "start: WORD" })), user("q")),
    options: options(),
  },
  {
    name: "tools-additional",
    model: model({
      compat: { supportsMidConvoSystemMessages: true, supportsAdditionalTools: true },
    }),
    context: ctx(
      withTools(tool("read"), tool("grep")),
      user("a"),
      system("", { toolsRemoved: [{ name: "grep" }] }),
      user("b"),
      system("", {
        toolsAdded: [tool("write")],
        toolsRemoved: [{ name: "read" }, { name: "grep" }],
      }),
      user("c"),
      system("", { toolsAdded: [tool("grep", undefined, { type: "object", properties: {} })] }),
      user("d"),
    ),
    options: options(),
  },
  {
    name: "tools-additional-without-mid-convo",
    model: model({ compat: { supportsAdditionalTools: true } }),
    context: ctx(
      withTools(tool()),
      user("a"),
      system("", { toolsAdded: [tool("write")] }),
      user("b"),
    ),
    options: options(),
  },
  {
    name: "tools-tool-search",
    model: model({
      compat: {
        supportsMidConvoSystemMessages: true,
        supportsToolSearch: true,
        supportsStrictMode: true,
      },
    }),
    context: ctx(
      withTools(tool()),
      user("a"),
      system("New tools.", { toolsAdded: [tool("write"), tool("grep")] }),
      user("b"),
      system("", { toolsAdded: [tool("edit")] }),
      user("c"),
    ),
    options: options(),
  },
  {
    name: "tools-tool-search-folded",
    model: model({ compat: { supportsToolSearch: true, supportsStrictMode: true } }),
    context: ctx(
      withTools(tool()),
      user("a"),
      system("New tools.", { toolsAdded: [tool("write"), tool("grep")] }),
      user("b"),
    ),
    options: options(),
  },
  {
    name: "tools-history-without-tools",
    model: model(),
    context: ctx(user("a"), assistant([call("c1|fc_1")]), toolResult("c1|fc_1", [text("ok")])),
    options: options(),
  },
];

const SIMPLE: Case[] = [
  {
    name: "simple-reasoning",
    model: reasoningModel(),
    context: conversation,
    options: options({ reasoning: "high", toolChoice: "none" }),
    simple: true,
  },
  {
    name: "simple-reasoning-clamped",
    model: reasoningModel({ thinkingLevelMap: { xhigh: null, high: "high" } }),
    context: conversation,
    options: options({ reasoning: "xhigh" }),
    simple: true,
  },
  {
    name: "simple-no-reasoning",
    model: reasoningModel(),
    context: conversation,
    options: options({ maxTokens: 300 }),
    simple: true,
  },
  {
    name: "simple-no-key",
    model: model(),
    context: conversation,
    options: { maxRetries: 0 },
    simple: true,
  },
];

const CASES: Case[] = [
  ...BASIC,
  ...REASONING,
  ...CACHE,
  ...FIELDS,
  ...HEADERS,
  ...MESSAGES,
  ...TOOLS,
  ...SIMPLE,
];

function errorOf(message: unknown): string | undefined {
  return typeof message === "object" && message !== null && "errorMessage" in message
    ? String(message.errorMessage)
    : undefined;
}

const results: Array<Case & { request: CapturedRequest | { error: string } }> = [];
for (const testCase of CASES) {
  const recorded: CapturedRequest[] = [];
  const run = testCase.simple === true ? responses.streamSimple : responses.stream;
  let failure: string | undefined;
  try {
    failure = errorOf(
      // oxlint-disable-next-line no-await-in-loop -- one recorded request per case, in order
      await run(testCase.model, testCase.context, {
        ...testCase.options,
        fetch: captureFetch(recorded),
      }).result(),
    );
  } catch (error) {
    failure = error instanceof Error ? error.message : String(error);
  }
  const [request] = recorded;
  if (request === undefined && failure === undefined) {
    throw new Error(`${testCase.name}: no request and no error`);
  }
  results.push({ ...testCase, request: request ?? { error: failure ?? "" } });
}
writeFileSync(
  join(import.meta.dirname, "../../vectors/ai/openai-responses-request.json"),
  `${JSON.stringify(results, null, "\t")}\n`,
);
console.log(`wrote ${results.length} cases`);
