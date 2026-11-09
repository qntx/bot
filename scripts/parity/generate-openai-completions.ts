/**
 * `generate-openai-completions.ts` — vectors for `openai-completions`'s request shape. Each case
 * drives the real dist `stream`/`streamSimple` with a recording `fetch` that answers a minimal
 * valid stream, and records the outgoing url, the headers pi sets (the SDK's `x-stainless-*`
 * headers are the `sdk-headers` class), and the body; a case that fails before sending records the
 * terminal `errorMessage`.
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

const completions: unknown = await import(
  pathToFileURL(join(piAiDir, "dist/api/openai-completions.js")).href
);
if (!isStreamModule(completions)) {
  throw new Error("openai-completions dist module has no stream export");
}

const SSE = [
  'data: {"id":"c1","object":"chat.completion.chunk","created":0,"model":"m","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}',
  "",
  "data: [DONE]",
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
  api: "openai-completions",
  ...CUSTOM,
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
const system = (content: string, toolsAdded?: unknown[]) => ({
  role: "system",
  content,
  ...(toolsAdded === undefined ? {} : { toolsAdded }),
  timestamp: 0,
});
const assistant = (content: unknown[], extra: Json = {}) => ({
  role: "assistant",
  content,
  api: "openai-completions",
  provider: "custom",
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
const text = (value: string) => ({ type: "text", text: value });
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
    properties: { path: { type: "string" }, limit: { type: "number" } },
    required: ["path"],
  },
  ...(constrainedSampling === undefined ? {} : { constrainedSampling }),
});
const grammarTool = (variants: Json, parameters?: Json) =>
  tool(
    "patch",
    { type: "grammar", variants },
    parameters ?? {
      type: "object",
      properties: { input: { type: "string" } },
      required: ["input"],
    },
  );
const withTools = (...tools: unknown[]) => system("You are helpful.", tools);
const conversation = ctx(system("You are helpful."), user("hi"));

type Case = {
  name: string;
  model: Json;
  context: unknown;
  options: Json;
  simple?: boolean;
  payloadPatch?: { set?: Record<string, unknown>; delete?: string[] };
  payloadUndefined?: boolean;
};

const DETECTION_TARGETS: Array<[string, Json]> = [
  ["openai", OPENAI],
  ["openrouter", OPENROUTER],
  ["openrouter-anthropic", { ...OPENROUTER, id: "anthropic/claude-sonnet" }],
  ["openrouter-openai", { ...OPENROUTER, id: "openai/gpt-5" }],
  ["deepseek", { provider: "deepseek", baseUrl: "https://api.deepseek.com" }],
  ["deepseek-url", { baseUrl: "https://API.DeepSeek.com/v1" }],
  ["zai", { provider: "zai", baseUrl: "https://api.z.ai/api/paas/v4" }],
  ["zai-url", { baseUrl: "https://open.bigmodel.cn/api/paas/v4" }],
  ["together", { provider: "together", baseUrl: "https://api.together.xyz/v1" }],
  ["moonshot", { provider: "moonshotai", baseUrl: "https://api.moonshot.ai/v1" }],
  [
    "cloudflare-workers",
    {
      provider: "cloudflare-workers-ai",
      baseUrl: "https://api.cloudflare.com/client/v4/accounts/a/ai/v1",
    },
  ],
  [
    "cloudflare-gateway",
    {
      provider: "cloudflare-ai-gateway",
      baseUrl: "https://gateway.ai.cloudflare.com/v1/a/g/compat",
    },
  ],
  ["nvidia", { provider: "nvidia", baseUrl: "https://integrate.api.nvidia.com/v1" }],
  ["ant-ling", { provider: "ant-ling", baseUrl: "https://api.ant-ling.com/v1" }],
  ["cerebras", { provider: "cerebras", baseUrl: "https://api.cerebras.ai/v1" }],
  ["xai", { provider: "xai", baseUrl: "https://api.x.ai/v1" }],
  ["chutes", { baseUrl: "https://llm.chutes.ai/v1" }],
  ["opencode", { provider: "opencode", baseUrl: "https://opencode.ai/zen/v1" }],
  ["groq", { provider: "groq", baseUrl: "https://api.groq.com/openai/v1" }],
  ["custom", CUSTOM],
];
const DETECTION: Case[] = DETECTION_TARGETS.map(([name, overrides]) => ({
  name: `detect-${name}`,
  model: reasoningModel(overrides),
  context: conversation,
  options: options({ maxTokens: 1000, reasoningEffort: "medium", sessionId: "session-1" }),
}));

const OVERRIDES: Case[] = [
  {
    name: "compat-overrides",
    model: reasoningModel({
      ...OPENAI,
      compat: {
        supportsStore: false,
        supportsDeveloperRole: false,
        supportsReasoningEffort: false,
        supportsUsageInStreaming: false,
        maxTokensField: "max_tokens",
      },
    }),
    context: conversation,
    options: options({ maxTokens: 1000, reasoningEffort: "high" }),
  },
  {
    name: "compat-unknown-thinking-format",
    model: reasoningModel({ compat: { thinkingFormat: "future-format" } }),
    context: conversation,
    options: options({ reasoningEffort: "high" }),
  },
];

const FORMATS = [
  "openai",
  "openrouter",
  "deepseek",
  "together",
  "baseten",
  "zai",
  "qwen",
  "chat-template",
  "qwen-chat-template",
  "string-thinking",
  "ant-ling",
];
const LEVEL_MAPS: Array<[string, Json | undefined]> = [
  ["unmapped", undefined],
  ["mapped", { off: "OFF", high: "HIGH" }],
  ["null", { off: null, high: null }],
];
const TEMPLATE_VALUES = {
  literal: "x",
  count: 2,
  flag: false,
  nothing: null,
  enabled: { $var: "thinking.enabled" },
  effort: { $var: "thinking.effort" },
  budget: { $var: "thinking.budget" },
  effortWhenOn: { $var: "thinking.effort", omitWhenOff: true },
};
const THINKING: Case[] = FORMATS.flatMap((format) =>
  LEVEL_MAPS.flatMap(([mapName, thinkingLevelMap]) =>
    ["high", undefined].map((effort) => ({
      name: `thinking-${format}-${mapName}-${effort ?? "off"}`,
      model: reasoningModel({
        ...(thinkingLevelMap === undefined ? {} : { thinkingLevelMap }),
        compat: {
          thinkingFormat: format,
          supportsReasoningEffort: true,
          chatTemplateKwargs: TEMPLATE_VALUES,
          chatTemplateArgs: TEMPLATE_VALUES,
        },
      }),
      context: conversation,
      options: options({
        maxTokens: 20_000,
        thinkingBudgets: { high: 8000 },
        ...(effort === undefined ? {} : { reasoningEffort: effort }),
      }),
    })),
  ),
);
const THINKING_EXTRA: Case[] = [
  {
    name: "thinking-zai-no-effort-support",
    model: reasoningModel({ compat: { thinkingFormat: "zai", supportsReasoningEffort: false } }),
    context: conversation,
    options: options({ reasoningEffort: "low" }),
  },
  {
    name: "thinking-baseten-no-effort-support",
    model: reasoningModel({
      compat: {
        thinkingFormat: "baseten",
        supportsReasoningEffort: false,
        chatTemplateArgs: TEMPLATE_VALUES,
      },
    }),
    context: conversation,
    options: options({ reasoningEffort: "low" }),
  },
  {
    name: "thinking-non-reasoning-model",
    model: model({ compat: { thinkingFormat: "zai" } }),
    context: conversation,
    options: options({ reasoningEffort: "high" }),
  },
  {
    name: "thinking-chat-template-empty",
    model: reasoningModel({ compat: { thinkingFormat: "chat-template" } }),
    context: conversation,
    options: options({ reasoningEffort: "high" }),
  },
];

const BUDGETS: Case[] = [
  ...["thinking_token_budget", "thinking_budget", "thinking_budget_tokens"].map((field) => ({
    name: `budget-${field}`,
    model: reasoningModel({ compat: { thinkingTokenBudgetField: field } }),
    context: conversation,
    options: options({ reasoningEffort: "medium", thinkingBudgets: { medium: 3000 } }),
  })),
  {
    name: "budget-alias",
    model: reasoningModel({ compat: { supportsThinkingTokenBudget: true } }),
    context: conversation,
    options: options({ reasoningEffort: "low" }),
  },
  {
    name: "budget-clamped-by-max-tokens",
    model: reasoningModel({
      compat: { thinkingTokenBudgetField: "thinking_budget", maxTokensField: "max_tokens" },
    }),
    context: conversation,
    options: options({ reasoningEffort: "high", maxTokens: 3000 }),
  },
  {
    name: "budget-zero",
    model: reasoningModel({
      maxTokens: 500,
      compat: { thinkingTokenBudgetField: "thinking_budget" },
    }),
    context: conversation,
    options: options({ reasoningEffort: "high" }),
  },
  {
    name: "budget-no-effort",
    model: reasoningModel({ compat: { thinkingTokenBudgetField: "thinking_budget" } }),
    context: conversation,
    options: options({}),
  },
];

const LONG_SESSION = `${"s".repeat(62)}😀😀tail`;
const ANTHROPIC_VIA_OPENROUTER = reasoningModel({ ...OPENROUTER, id: "anthropic/claude-sonnet" });
const CACHE: Case[] = [
  {
    name: "cache-none",
    model: model(OPENAI),
    context: conversation,
    options: options({ cacheRetention: "none", sessionId: "s1" }),
  },
  {
    name: "cache-short",
    model: model(OPENAI),
    context: conversation,
    options: options({ cacheRetention: "short", sessionId: "s1" }),
  },
  {
    name: "cache-long-clamped-key",
    model: model(OPENAI),
    context: conversation,
    options: options({ cacheRetention: "long", sessionId: LONG_SESSION }),
  },
  {
    name: "cache-env-long",
    model: model(CUSTOM),
    context: conversation,
    options: options({ env: { PI_CACHE_RETENTION: "long" }, sessionId: "s1" }),
  },
  {
    name: "cache-long-unsupported",
    model: model({ ...CUSTOM, compat: { supportsLongCacheRetention: false } }),
    context: conversation,
    options: options({ cacheRetention: "long", sessionId: "s1" }),
  },
  {
    name: "cache-custom-short",
    model: model(CUSTOM),
    context: conversation,
    options: options({ sessionId: "s1" }),
  },
  {
    name: "cache-control-tools-and-tool-result",
    model: ANTHROPIC_VIA_OPENROUTER,
    context: ctx(
      withTools(tool("read"), tool("grep")),
      user("hi"),
      assistant([call("c1")]),
      toolResult("c1", [text("file")]),
    ),
    options: options({}),
  },
  {
    name: "cache-control-assistant-text",
    model: ANTHROPIC_VIA_OPENROUTER,
    context: ctx(system("sys"), user("hi"), assistant([text("answer")])),
    options: options({ cacheRetention: "long" }),
  },
  {
    name: "cache-control-user-blocks",
    model: ANTHROPIC_VIA_OPENROUTER,
    context: ctx(user([text("look"), image])),
    options: options({}),
  },
  {
    name: "cache-control-user-image-only",
    model: ANTHROPIC_VIA_OPENROUTER,
    context: ctx(system("sys"), user("first"), user([image])),
    options: options({}),
  },
  {
    name: "cache-control-format-override",
    model: model({ compat: { cacheControlFormat: "anthropic" } }),
    context: ctx(system("sys"), user("hi")),
    options: options({ cacheRetention: "long" }),
  },
  {
    name: "cache-control-none-retention",
    model: ANTHROPIC_VIA_OPENROUTER,
    context: conversation,
    options: options({ cacheRetention: "none" }),
  },
];

const REASONING_DETAILS = JSON.stringify([
  { type: "reasoning.summary", summary: "s", id: "r1", format: "f", index: 0 },
  { type: "reasoning.encrypted", data: "enc", id: "r2" },
  { type: "reasoning.text", text: "t", signature: null },
]);
const LEGACY_DETAIL = JSON.stringify({ type: "reasoning.encrypted", id: "e1", data: "blob" });
const MESSAGES: Case[] = [
  {
    name: "user-blocks-filtered",
    model: model(),
    context: ctx(user([text(""), text("a"), image]), user([text("")]), user("after")),
    options: options(),
  },
  ...["reasoning_content", "reasoning", "reasoning_text", "unknown_field"].map((signature) => ({
    name: `assistant-thinking-${signature}`,
    model: reasoningModel(),
    context: ctx(
      user("q"),
      assistant([thinking("first", signature), thinking("  "), thinking("second"), text("answer")]),
      user("next"),
    ),
    options: options(),
  })),
  {
    name: "assistant-thinking-opencode-go",
    model: reasoningModel({ provider: "opencode-go", baseUrl: "https://opencode.ai/zen/go/v1" }),
    context: ctx(user("q"), assistant([thinking("why", "reasoning"), text("a")]), user("next")),
    options: options(),
  },
  {
    name: "assistant-reasoning-details",
    model: reasoningModel(),
    context: ctx(
      user("q"),
      assistant([thinking("why", REASONING_DETAILS), text("a")]),
      user("next"),
    ),
    options: options(),
  },
  {
    name: "assistant-reasoning-details-invalid",
    model: reasoningModel(),
    context: ctx(
      user("q"),
      assistant([thinking("why", '[{"type":"reasoning.other"}]'), text("a")]),
      user("next"),
    ),
    options: options(),
  },
  {
    name: "assistant-legacy-encrypted-detail",
    model: reasoningModel(),
    context: ctx(
      user("q"),
      assistant([
        thinking("why", "reasoning_content"),
        call("c1", "read", { path: "a" }, { thoughtSignature: LEGACY_DETAIL }),
      ]),
      toolResult("c1", [text("ok")]),
    ),
    options: options(),
  },
  {
    name: "assistant-thinking-as-text",
    model: reasoningModel({ compat: { requiresThinkingAsText: true } }),
    context: ctx(
      user("q"),
      assistant([thinking("one"), thinking("two"), text("answer")]),
      user("next"),
    ),
    options: options(),
  },
  {
    name: "assistant-reasoning-content-required",
    model: reasoningModel({ provider: "deepseek", baseUrl: "https://api.deepseek.com" }),
    context: ctx(
      user("q"),
      assistant([text("plain")]),
      user("q2"),
      assistant([thinking("why", "reasoning_content"), text("b")]),
      user("next"),
    ),
    options: options(),
  },
  {
    name: "assistant-empty-skipped",
    model: model(),
    context: ctx(
      user("q"),
      assistant([], { stopReason: "aborted" }),
      assistant([text("   ")]),
      user("again"),
    ),
    options: options(),
  },
  {
    name: "assistant-after-tool-result-bridge",
    model: model({
      compat: { requiresAssistantAfterToolResult: true, requiresToolResultName: true },
    }),
    context: ctx(
      withTools(tool()),
      user("q"),
      assistant([call("c1")]),
      toolResult("c1", [text("ok")]),
      user("next"),
    ),
    options: options(),
  },
  {
    name: "tool-results-grouped",
    model: model(),
    context: ctx(
      withTools(tool()),
      user("q"),
      assistant([text("calling"), call("c1"), call("c2")]),
      toolResult("c1", [text("line1"), text("line2")]),
      toolResult("c2", [image]),
      toolResult("c3", []),
      user("next"),
    ),
    options: options(),
  },
  {
    name: "tool-result-images-bridge",
    model: model({ compat: { requiresAssistantAfterToolResult: true } }),
    context: ctx(
      withTools(tool()),
      user("q"),
      assistant([call("c1")]),
      toolResult("c1", [text("see"), image]),
      user("next"),
    ),
    options: options(),
  },
  {
    name: "tool-result-image-text-only-model",
    model: model({ input: ["text"] }),
    context: ctx(withTools(tool()), user("q"), assistant([call("c1")]), toolResult("c1", [image])),
    options: options(),
  },
  {
    name: "tool-result-error-image",
    model: model(),
    context: ctx(
      withTools(tool()),
      user("q"),
      assistant([call("c1")]),
      toolResult("c1", [text("failed"), image], { isError: true }),
    ),
    options: options(),
  },
  {
    name: "tool-call-ids",
    model: model(OPENAI),
    context: ctx(
      withTools(tool()),
      user("q"),
      assistant([
        call("call_1|item+/=x"),
        call(`call_${"a".repeat(30)}|${"b".repeat(30)}+/=`),
        call("x".repeat(50)),
        call("|only-item"),
      ]),
      toolResult("call_1|item+/=x", [text("1")]),
      toolResult(`call_${"a".repeat(30)}|${"b".repeat(30)}+/=`, [text("2")]),
      toolResult("x".repeat(50), [text("3")]),
      toolResult("|only-item", [text("4")]),
    ),
    options: options(),
  },
  {
    name: "tool-call-ids-other-provider",
    model: model({ provider: "groq", baseUrl: "https://api.groq.com/openai/v1" }),
    context: ctx(
      withTools(tool()),
      user("q"),
      assistant([call("y".repeat(50))]),
      toolResult("y".repeat(50), [text("ok")]),
    ),
    options: options(),
  },
  {
    name: "tool-history-without-tools",
    model: model(),
    context: ctx(user("q"), assistant([call("c1")]), toolResult("c1", [text("ok")])),
    options: options(),
  },
  {
    name: "system-folded",
    model: model(OPENAI),
    context: ctx(system("first"), user("q"), system("later", [tool("grep")]), user("next")),
    options: options(),
  },
  {
    name: "system-mid-convo",
    model: reasoningModel({ ...OPENAI, compat: { supportsMidConvoSystemMessages: true } }),
    context: ctx(
      system("first", [tool("read")]),
      user("q"),
      system("later", [tool("grep")]),
      user("next"),
    ),
    options: options(),
  },
  {
    name: "system-mid-convo-tool-additions",
    model: model({
      compat: { supportsMidConvoSystemMessages: true, supportsMidConvoToolAdditions: true },
    }),
    context: ctx(
      system("first", [tool("read")]),
      user("q"),
      system("later", [tool("grep")]),
      user("next"),
    ),
    options: options(),
  },
  {
    name: "unicode-text",
    model: model(),
    context: ctx(
      system("系统 ✓"),
      user("héllo 😀 \u2028 end"),
      assistant([text("答")]),
      user("ok"),
    ),
    options: options(),
  },
];

const LARK = { openai_lark: "start: WORD" };
const TOOLS: Case[] = [
  {
    name: "tools-function",
    model: model(),
    context: ctx(withTools(tool("read"), tool("grep")), user("q")),
    options: options({ toolChoice: "required" }),
  },
  {
    name: "tools-strict-prefer-supported",
    model: model({ compat: { supportsStrictMode: true } }),
    context: ctx(
      withTools(tool("read", { type: "json_schema", strict: "prefer" }), tool("grep")),
      user("q"),
    ),
    options: options({ toolChoice: { type: "function", function: { name: "read" } } }),
  },
  {
    name: "tools-strict-prefer-unsupported-schema",
    model: model({ compat: { supportsStrictMode: true } }),
    context: ctx(
      withTools(
        tool(
          "read",
          { type: "json_schema", strict: "prefer" },
          { type: "object", properties: { a: { oneOf: [{ type: "string" }] } } },
        ),
      ),
      user("q"),
    ),
    options: options(),
  },
  {
    name: "tools-strict-require-supported",
    model: model({ compat: { supportsStrictMode: true } }),
    context: ctx(withTools(tool("read", { type: "json_schema", strict: "require" })), user("q")),
    options: options(),
  },
  {
    name: "tools-strict-require-unsupported",
    model: model(),
    context: ctx(withTools(tool("read", { type: "json_schema", strict: "require" })), user("q")),
    options: options(),
  },
  {
    name: "tools-grammar-lark",
    model: model({ compat: { supportsOpenAIGrammarTools: true } }),
    context: ctx(withTools(grammarTool({ ...LARK, openai_regex: "a+" })), user("q")),
    options: options(),
  },
  {
    name: "tools-grammar-regex",
    model: model({ compat: { supportsOpenAIGrammarTools: true } }),
    context: ctx(withTools(grammarTool({ openai_lark: "  ", openai_regex: "a+" })), user("q")),
    options: options(),
  },
  {
    name: "tools-grammar-unsupported-provider",
    model: model(),
    context: ctx(withTools(grammarTool(LARK)), user("q")),
    options: options(),
  },
  {
    name: "tools-grammar-no-variant",
    model: model({ compat: { supportsOpenAIGrammarTools: true } }),
    context: ctx(withTools(grammarTool({})), user("q")),
    options: options(),
  },
  {
    name: "tools-grammar-bad-schema",
    model: model({ compat: { supportsOpenAIGrammarTools: true } }),
    context: ctx(
      withTools(
        grammarTool(LARK, {
          type: "object",
          properties: { input: { type: "number" } },
          required: ["input"],
        }),
      ),
      user("q"),
    ),
    options: options(),
  },
  {
    name: "tools-grammar-call-replay",
    model: model({ compat: { supportsOpenAIGrammarTools: true } }),
    context: ctx(
      withTools(grammarTool(LARK)),
      user("q"),
      assistant([call("c1", "patch", { input: 'say "hi"\n' })]),
      toolResult("c1", [text("ok")]),
    ),
    options: options(),
  },
  {
    name: "tools-grammar-call-bad-input",
    model: model({ compat: { supportsOpenAIGrammarTools: true } }),
    context: ctx(
      withTools(grammarTool(LARK)),
      user("q"),
      assistant([call("c1", "patch", { input: 3 })]),
      toolResult("c1", [text("ok")]),
    ),
    options: options(),
  },
  {
    name: "tools-zai-tool-stream",
    model: model({
      provider: "zai",
      baseUrl: "https://api.z.ai/api/paas/v4",
      compat: { zaiToolStream: true },
    }),
    context: ctx(withTools(tool()), user("q")),
    options: options(),
  },
];

const ROUTING: Case[] = [
  {
    name: "routing-openrouter-empty",
    model: model({ ...OPENROUTER, compat: { openRouterRouting: {} } }),
    context: conversation,
    options: options(),
  },
  {
    name: "routing-openrouter",
    model: model({
      ...OPENROUTER,
      compat: { openRouterRouting: { only: ["a"], allow_fallbacks: false } },
    }),
    context: conversation,
    options: options(),
  },
  {
    name: "routing-vercel-only",
    model: model({ compat: { vercelGatewayRouting: { only: ["bedrock"] } } }),
    context: conversation,
    options: options(),
  },
  {
    name: "routing-vercel-order",
    model: model({ compat: { vercelGatewayRouting: { order: ["a", "b"], only: ["a"] } } }),
    context: conversation,
    options: options(),
  },
  {
    name: "routing-vercel-neither",
    model: model({ compat: { vercelGatewayRouting: {} } }),
    context: conversation,
    options: options(),
  },
  {
    name: "vllm-priority",
    model: model({ compat: { vllmPriority: -1 } }),
    context: conversation,
    options: options(),
  },
];

const SAMPLING: Case[] = [
  {
    name: "sampling-override-keeps-position",
    model: model({ samplingParams: { top_p: 0.9, temperature: 0.2 } }),
    context: conversation,
    options: options({
      temperature: 0.7,
      maxTokens: 100,
      samplingParams: { temperature: 0.1, seed: 7 },
    }),
  },
  {
    name: "sampling-by-thinking-level",
    model: reasoningModel({
      samplingParams: { top_k: 20 },
      samplingParamsByThinkingLevel: { high: { top_k: 40, min_p: 0.1 }, off: { top_k: 1 } },
    }),
    context: conversation,
    options: options({ reasoningEffort: "high" }),
  },
  {
    name: "sampling-off-level",
    model: reasoningModel({ samplingParamsByThinkingLevel: { off: { top_k: 1 } } }),
    context: conversation,
    options: options(),
  },
  {
    name: "temperature-zero",
    model: model(),
    context: conversation,
    options: options({ temperature: 0, maxTokens: 0 }),
  },
];

const ATTRIBUTION = {
  attribution: { name: "ovo", url: "https://ovo.example", categories: ["cli-agent", "desktop"] },
};
const HEADERS: Case[] = [
  { name: "auth-no-key", model: model(), context: conversation, options: { maxRetries: 0 } },
  {
    name: "auth-header-only",
    model: model(),
    context: conversation,
    options: { maxRetries: 0, headers: { Authorization: "Bearer custom" } },
  },
  {
    name: "auth-blank-header",
    model: model(),
    context: conversation,
    options: { maxRetries: 0, headers: { authorization: "  " } },
  },
  {
    name: "auth-cf-aig-only",
    model: model(),
    context: conversation,
    options: { maxRetries: 0, headers: { "cf-aig-authorization": "Bearer gw" } },
  },
  {
    name: "headers-merge-order",
    model: model({ headers: { "X-Model": "m", "X-Shared": "model", Accept: "text/plain" } }),
    context: conversation,
    options: options({ headers: { "x-shared": "request", "X-Request": "r" } }),
  },
  {
    name: "headers-null-removal",
    model: model({ headers: { "X-Model": "m" } }),
    context: conversation,
    options: options({ headers: { "x-model": null, authorization: null } }),
  },
  {
    name: "identity-openrouter",
    model: model(OPENROUTER),
    context: conversation,
    options: options({ identity: ATTRIBUTION, sessionId: "s1" }),
  },
  {
    name: "github-copilot-user",
    model: model({ provider: "github-copilot", baseUrl: "https://api.githubcopilot.com" }),
    context: conversation,
    options: options(),
  },
  {
    name: "github-copilot-agent-vision",
    model: model({ provider: "github-copilot", baseUrl: "https://api.githubcopilot.com" }),
    context: ctx(
      withTools(tool()),
      user("look"),
      assistant([call("c1")]),
      toolResult("c1", [text("shot"), image]),
    ),
    options: options({ headers: { "X-Initiator": "user" } }),
  },
  {
    name: "identity-nvidia",
    model: model({ provider: "nvidia", baseUrl: "https://integrate.api.nvidia.com/v1" }),
    context: conversation,
    options: options({ identity: ATTRIBUTION }),
  },
  {
    name: "affinity-openai",
    model: model({ compat: { sendSessionAffinityHeaders: true } }),
    context: conversation,
    options: options({ sessionId: "s1" }),
  },
  {
    name: "affinity-nosession",
    model: model({
      compat: { sendSessionAffinityHeaders: true, sessionAffinityFormat: "openai-nosession" },
    }),
    context: conversation,
    options: options({ sessionId: "s1" }),
  },
  {
    name: "affinity-openrouter-disabled",
    model: model({ ...OPENROUTER, compat: { sendSessionAffinityHeaders: false } }),
    context: conversation,
    options: options({ sessionId: "s1" }),
  },
  {
    name: "affinity-cache-none",
    model: model(OPENROUTER),
    context: conversation,
    options: options({ sessionId: "s1", cacheRetention: "none" }),
  },
  {
    name: "base-url-trailing-slash",
    model: model({ baseUrl: "https://llm.example.com/v1/" }),
    context: conversation,
    options: options(),
  },
];

const SIMPLE: Case[] = [
  {
    name: "simple-no-reasoning",
    model: reasoningModel(OPENAI),
    context: conversation,
    options: options(),
    simple: true,
  },
  {
    name: "simple-reasoning",
    model: reasoningModel(OPENAI),
    context: conversation,
    options: options({ reasoning: "high", toolChoice: "none" }),
    simple: true,
  },
  {
    name: "simple-reasoning-clamped",
    model: reasoningModel({ ...OPENAI, thinkingLevelMap: { xhigh: null, high: "high" } }),
    context: conversation,
    options: options({ reasoning: "xhigh" }),
    simple: true,
  },
  {
    name: "simple-thinking-budgets",
    model: reasoningModel({ compat: { thinkingTokenBudgetField: "thinking_budget" } }),
    context: conversation,
    options: options({ reasoning: "medium", thinkingBudgets: { medium: 2500 }, maxTokens: 9000 }),
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
  ...DETECTION,
  ...OVERRIDES,
  ...THINKING,
  ...THINKING_EXTRA,
  ...BUDGETS,
  ...CACHE,
  ...MESSAGES,
  ...TOOLS,
  ...ROUTING,
  ...SAMPLING,
  ...HEADERS,
  ...SIMPLE,
  {
    name: "onPayload-patch",
    model: model(),
    context: conversation,
    options: options(),
    payloadPatch: { set: { "x-ovo-probe": "completions" }, delete: ["store"] },
  },
  {
    name: "onPayload-undefined",
    model: model(),
    context: conversation,
    options: options(),
    payloadUndefined: true,
  },
];

function errorOf(message: unknown): string | undefined {
  return typeof message === "object" && message !== null && "errorMessage" in message
    ? String(message.errorMessage)
    : undefined;
}

/** `payloadPatch`/`payloadUndefined` become an `onPayload` the adapter applies. */
function onPayloadOption(testCase: Case) {
  const patch = testCase.payloadPatch;
  if (patch) {
    return {
      onPayload: (params: Record<string, unknown>) => {
        const drop = new Set(patch.delete);
        const next = Object.fromEntries(Object.entries(params).filter(([key]) => !drop.has(key)));
        Object.assign(next, patch.set ?? {});
        return next;
      },
    };
  }
  return testCase.payloadUndefined === true ? { onPayload: () => undefined } : {};
}

const results: Array<Case & { request: CapturedRequest | { error: string } }> = [];
for (const testCase of CASES) {
  const recorded: CapturedRequest[] = [];
  const run = testCase.simple === true ? completions.streamSimple : completions.stream;
  let failure: string | undefined;
  try {
    failure = errorOf(
      // oxlint-disable-next-line no-await-in-loop -- one recorded request per case, in order
      await run(testCase.model, testCase.context, {
        ...testCase.options,
        fetch: captureFetch(recorded),
        ...onPayloadOption(testCase),
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
  join(import.meta.dirname, "../../vectors/ai/openai-completions-request.json"),
  `${JSON.stringify(results, null, "\t")}\n`,
);
console.log(`wrote ${results.length} cases`);
