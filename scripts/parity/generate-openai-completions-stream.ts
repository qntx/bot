// Parity vectors for the openai-completions response side
// (AI-API-OPENAI-COMPLETIONS-STREAM): scripted SSE bodies and error responses
// run through the published adapter's `stream`/`streamSimple`; each case
// records the events (without `partial`) and the final message. Run with
// `node scripts/parity/generate-openai-completions-stream.ts`.
import { writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

type StreamResult = AsyncIterable<Record<string, unknown>> & {
  result: () => Promise<Record<string, unknown>>;
};
type StreamFn = (
  model: unknown,
  context: unknown,
  options: Record<string, unknown>,
) => StreamResult;

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

const piAiDir = dirname(dirname(fileURLToPath(import.meta.resolve("@earendil-works/pi-ai"))));
const completions: unknown = await import(
  pathToFileURL(join(piAiDir, "dist/api/openai-completions.js")).href
);
if (!isStreamModule(completions)) {
  throw new Error("openai-completions dist module has no stream export");
}

type Json = Record<string, unknown>;

const model = (overrides: Json = {}): Json => ({
  id: "m",
  name: "M",
  api: "openai-completions",
  provider: "openai",
  baseUrl: "https://api.openai.com/v1",
  input: ["text"],
  cost: { input: 1, output: 2, cacheRead: 0.5, cacheWrite: 1.5 },
  reasoning: true,
  contextWindow: 128_000,
  maxTokens: 4096,
  ...overrides,
});

const user = { role: "user", content: "hi", timestamp: 0 };
const CONTEXT = { messages: [user] };
const GRAMMAR_CONTEXT = {
  messages: [
    {
      role: "system",
      content: "You are helpful.",
      toolsAdded: [
        {
          name: "patch",
          description: "patch tool",
          parameters: {
            type: "object",
            properties: { patch: { type: "string" } },
            required: ["patch"],
          },
          constrainedSampling: { type: "grammar", variants: { openai_lark: "start: WORD" } },
        },
      ],
      timestamp: 0,
    },
    user,
  ],
};
const GRAMMAR_MODEL = model({ compat: { supportsOpenAIGrammarTools: true } });

/** One SSE record carrying a chunk. */
const record = (data: unknown, event?: string): string =>
  `${event === undefined ? "" : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
const chunk = (delta: Json, extra: Json = {}, choiceExtra: Json = {}) =>
  record({
    id: "chatcmpl-1",
    object: "chat.completion.chunk",
    created: 0,
    model: "m",
    choices: [{ index: 0, delta, finish_reason: null, ...choiceExtra }],
    ...extra,
  });
const finish = (reason: string, extra: Json = {}) =>
  record({
    id: "chatcmpl-1",
    object: "chat.completion.chunk",
    created: 0,
    model: "m",
    choices: [{ index: 0, delta: {}, finish_reason: reason }],
    ...extra,
  });
const usageChunk = (usage: Json) =>
  record({
    id: "chatcmpl-1",
    object: "chat.completion.chunk",
    created: 0,
    model: "m",
    choices: [],
    usage,
  });
const DONE = "data: [DONE]\n\n";
const USAGE = { prompt_tokens: 100, completion_tokens: 20, total_tokens: 120 };
const toolDelta = (entry: Json) => chunk({ tool_calls: [entry] });

type Case = {
  name: string;
  model?: Json;
  context?: Json;
  sse?: string;
  status?: number;
  body?: string;
  simple?: boolean;
  options?: Json;
  abort?: "before" | "on-response";
};

const sse = (...records: string[]) => records.join("");

const CASES: Case[] = [
  {
    name: "text",
    sse: sse(
      chunk({ role: "assistant", content: "Hel" }),
      chunk({ content: "lo" }),
      finish("stop"),
      DONE,
    ),
  },
  {
    name: "text-usage",
    sse: sse(chunk({ content: "hi" }), finish("stop"), usageChunk(USAGE), DONE),
  },
  {
    name: "empty-content-skipped",
    sse: sse(chunk({ content: "" }), chunk({ content: null }), finish("stop"), DONE),
  },
  {
    name: "reasoning-content",
    sse: sse(chunk({ reasoning_content: "think" }), chunk({ content: "ok" }), finish("stop"), DONE),
  },
  { name: "reasoning", sse: sse(chunk({ reasoning: "think" }), finish("stop"), DONE) },
  { name: "reasoning-text", sse: sse(chunk({ reasoning_text: "think" }), finish("stop"), DONE) },
  {
    name: "reasoning-duplicate-fields",
    sse: sse(chunk({ reasoning_content: "same", reasoning: "same" }), finish("stop"), DONE),
  },
  {
    name: "reasoning-opencode-go",
    model: model({ provider: "opencode-go" }),
    sse: sse(chunk({ reasoning: "think" }), finish("stop"), DONE),
  },
  {
    name: "reasoning-and-content-one-delta",
    sse: sse(chunk({ reasoning_content: "think", content: "answer" }), finish("stop"), DONE),
  },
  {
    name: "reasoning-details-merge",
    model: model({ provider: "openrouter", baseUrl: "https://openrouter.ai/api/v1" }),
    sse: sse(
      chunk({
        reasoning: "a",
        reasoning_details: [{ type: "reasoning.text", text: "a", format: "unknown", index: 0 }],
      }),
      chunk({
        reasoning: "b",
        reasoning_details: [{ type: "reasoning.text", text: "b", signature: "sig", id: "r1" }],
      }),
      chunk({
        reasoning_details: [
          { type: "reasoning.encrypted", data: "enc1", id: "e1" },
          { type: "reasoning.encrypted", data: "enc2", id: "e2" },
          { type: "reasoning.summary", summary: "s1" },
          { type: "reasoning.summary", summary: "s2", format: "f", index: 3 },
        ],
      }),
      chunk({ content: "ok" }),
      finish("stop"),
      DONE,
    ),
  },
  {
    name: "reasoning-details-null-fields",
    sse: sse(
      chunk({
        reasoning_details: [
          { type: "reasoning.text", text: "a", id: null, signature: null, format: "" },
        ],
      }),
      chunk({ reasoning_details: [{ type: "reasoning.text", text: "b" }] }),
      chunk({
        reasoning_details: [
          { type: "reasoning.text", text: "c", id: "r2", signature: "s", format: "g" },
        ],
      }),
      finish("stop"),
      DONE,
    ),
  },
  {
    name: "reasoning-details-invalid-skipped",
    sse: sse(
      chunk({
        reasoning_details: [
          { type: "reasoning.text", text: 1 },
          { type: "reasoning.unknown", text: "x" },
          { type: "reasoning.summary", summary: "s", id: 5 },
          "not-an-object",
          [1],
        ],
      }),
      chunk({ content: "ok" }),
      finish("stop"),
      DONE,
    ),
  },
  {
    name: "reasoning-details-only",
    sse: sse(
      chunk({ reasoning_details: [{ type: "reasoning.encrypted", data: "enc" }] }),
      chunk({ content: "ok" }),
      finish("stop"),
      DONE,
    ),
  },
  {
    name: "text-tool-text-single-text-block",
    sse: sse(
      chunk({ content: "Reading. " }),
      toolDelta({
        index: 0,
        id: "call_1",
        type: "function",
        function: { name: "read", arguments: "" },
      }),
      toolDelta({ index: 0, function: { arguments: '{"path":' } }),
      chunk({ content: "More." }),
      toolDelta({ index: 0, function: { arguments: '"a"}' } }),
      finish("tool_calls"),
      DONE,
    ),
  },
  {
    name: "parallel-tool-calls",
    sse: sse(
      toolDelta({
        index: 0,
        id: "call_1",
        type: "function",
        function: { name: "read", arguments: '{"path":"a"}' },
      }),
      toolDelta({
        index: 1,
        id: "call_2",
        type: "function",
        function: { name: "read", arguments: '{"pa' },
      }),
      toolDelta({ index: 1, function: { arguments: 'th":"b"}' } }),
      finish("tool_calls"),
      DONE,
    ),
  },
  {
    name: "tool-call-by-id",
    sse: sse(
      toolDelta({
        id: "call_1",
        type: "function",
        function: { name: "read", arguments: '{"path"' },
      }),
      toolDelta({ id: "call_1", function: { arguments: ':"a"}' } }),
      finish("tool_calls"),
      DONE,
    ),
  },
  {
    name: "tool-call-late-id-and-name",
    sse: sse(
      toolDelta({ index: 0, function: { arguments: '{"path":' } }),
      toolDelta({ index: 0, id: "call_9", function: { name: "read", arguments: '"a"}' } }),
      finish("tool_calls"),
      DONE,
    ),
  },
  {
    name: "tool-call-empty-arguments",
    sse: sse(
      toolDelta({ index: 0, id: "call_1", type: "function", function: { name: "now" } }),
      finish("tool_calls"),
      DONE,
    ),
  },
  {
    name: "tool-call-malformed-arguments",
    sse: sse(
      toolDelta({ index: 0, id: "call_1", function: { name: "read", arguments: '{"path": ' } }),
      finish("tool_calls"),
      DONE,
    ),
  },
  {
    name: "grammar-tool-input",
    model: GRAMMAR_MODEL,
    context: GRAMMAR_CONTEXT,
    sse: sse(
      toolDelta({ index: 0, id: "call_1", type: "custom", custom: { name: "patch" } }),
      toolDelta({ index: 0, custom: { input: 'line "1"\n' } }),
      toolDelta({ index: 0, custom: { input: "line 2" } }),
      finish("tool_calls"),
      DONE,
    ),
  },
  {
    name: "grammar-tool-no-input",
    model: GRAMMAR_MODEL,
    context: GRAMMAR_CONTEXT,
    sse: sse(
      toolDelta({ index: 0, id: "call_1", type: "custom", custom: { name: "patch" } }),
      finish("tool_calls"),
      DONE,
    ),
  },
  {
    name: "grammar-tool-unknown-name",
    model: GRAMMAR_MODEL,
    context: GRAMMAR_CONTEXT,
    sse: sse(
      toolDelta({
        index: 0,
        id: "call_1",
        type: "custom",
        custom: { name: "invented", input: "x" },
      }),
      finish("tool_calls"),
      DONE,
    ),
  },
  {
    name: "grammar-tool-json-to-custom-switch",
    model: GRAMMAR_MODEL,
    context: GRAMMAR_CONTEXT,
    sse: sse(
      toolDelta({ index: 0, id: "call_1" }),
      toolDelta({ index: 0, custom: { name: "patch", input: "abc" } }),
      finish("tool_calls"),
      DONE,
    ),
  },
  {
    name: "grammar-tool-error-mid-input",
    model: GRAMMAR_MODEL,
    context: GRAMMAR_CONTEXT,
    sse: sse(
      toolDelta({ index: 0, id: "call_1", type: "custom", custom: { name: "patch", input: "ab" } }),
      record({ error: { message: "boom" } }),
    ),
  },
  {
    name: "usage-openai-cached",
    sse: sse(
      chunk({ content: "hi" }),
      finish("stop"),
      usageChunk({
        prompt_tokens: 100,
        completion_tokens: 20,
        prompt_tokens_details: { cached_tokens: 60 },
        completion_tokens_details: { reasoning_tokens: 5 },
      }),
      DONE,
    ),
  },
  {
    name: "usage-deepseek-cache-hit",
    sse: sse(
      chunk({ content: "hi" }),
      finish("stop", {
        usage: { prompt_tokens: 100, completion_tokens: 20, prompt_cache_hit_tokens: 30 },
      }),
      DONE,
    ),
  },
  {
    name: "usage-kimi-top-level-cached",
    sse: sse(
      chunk({ content: "hi" }),
      finish("stop", { usage: { prompt_tokens: 100, completion_tokens: 20, cached_tokens: 40 } }),
      DONE,
    ),
  },
  {
    name: "usage-cache-write",
    sse: sse(
      chunk({ content: "hi" }),
      finish("stop", {
        usage: {
          prompt_tokens: 100,
          completion_tokens: 20,
          prompt_tokens_details: { cached_tokens: 10, cache_write_tokens: 200 },
        },
      }),
      DONE,
    ),
  },
  {
    name: "usage-null-cached-falls-through",
    sse: sse(
      chunk({ content: "hi" }),
      finish("stop", {
        usage: {
          prompt_tokens: 100,
          completion_tokens: 20,
          prompt_tokens_details: { cached_tokens: null },
          cached_tokens: 7,
        },
      }),
      DONE,
    ),
  },
  {
    name: "usage-moonshot-choice",
    sse: sse(
      chunk({ content: "hi" }),
      record({
        id: "chatcmpl-1",
        object: "chat.completion.chunk",
        created: 0,
        model: "m",
        choices: [
          {
            index: 0,
            delta: {},
            finish_reason: "stop",
            usage: { prompt_tokens: 50, completion_tokens: 5 },
          },
        ],
      }),
      DONE,
    ),
  },
  {
    name: "openrouter-billed-cost",
    model: model({ provider: "openrouter", baseUrl: "https://openrouter.ai/api/v1" }),
    sse: sse(
      chunk({ content: "hi" }),
      finish("stop"),
      usageChunk({ ...USAGE, cost: 0.0042, is_byok: false }),
      DONE,
    ),
  },
  {
    name: "openrouter-byok-cost",
    model: model({ provider: "openrouter", baseUrl: "https://openrouter.ai/api/v1" }),
    sse: sse(
      chunk({ content: "hi" }),
      finish("stop"),
      usageChunk({
        ...USAGE,
        cost: 0.0001,
        is_byok: true,
        cost_details: { upstream_inference_cost: 0.003 },
      }),
      DONE,
    ),
  },
  {
    name: "billed-cost-ignored-off-openrouter",
    sse: sse(
      chunk({ content: "hi" }),
      finish("stop"),
      usageChunk({ ...USAGE, cost: 0.0042, is_byok: false }),
      DONE,
    ),
  },
  {
    name: "response-model-differs",
    sse: sse(
      chunk({ content: "hi" }, { model: "m-2026" }),
      chunk({ content: "!" }, { model: "m-other" }),
      finish("stop"),
      DONE,
    ),
  },
  {
    name: "response-id-later-chunk",
    sse: sse(
      chunk({ content: "hi" }, { id: "" }),
      chunk({ content: "!" }, { id: "late" }),
      finish("stop"),
      DONE,
    ),
  },
  { name: "finish-end", sse: sse(chunk({ content: "hi" }), finish("end"), DONE) },
  { name: "finish-length", sse: sse(chunk({ content: "hi" }), finish("length"), DONE) },
  {
    name: "finish-function-call",
    sse: sse(
      toolDelta({ index: 0, id: "call_1", function: { name: "read", arguments: "{}" } }),
      finish("function_call"),
      DONE,
    ),
  },
  {
    name: "finish-content-filter",
    sse: sse(chunk({ content: "hi" }), finish("content_filter"), DONE),
  },
  {
    name: "finish-network-error",
    sse: sse(chunk({ content: "hi" }), finish("network_error"), DONE),
  },
  { name: "finish-unknown", sse: sse(chunk({ content: "hi" }), finish("weird"), DONE) },
  { name: "finish-missing", sse: sse(chunk({ content: "hi" }), DONE) },
  {
    name: "finish-unsupported-stop",
    model: model({ compat: { supportsFinishReason: false } }),
    sse: sse(chunk({ content: "hi" }), DONE),
  },
  {
    name: "finish-unsupported-tool-use",
    model: model({ compat: { supportsFinishReason: false } }),
    sse: sse(
      toolDelta({ index: 0, id: "call_1", function: { name: "read", arguments: "{}" } }),
      DONE,
    ),
  },
  {
    name: "events-after-done-ignored",
    sse: sse(chunk({ content: "hi" }), finish("stop"), DONE, chunk({ content: "late" })),
  },
  { name: "no-done-sentinel", sse: sse(chunk({ content: "hi" }), finish("stop")) },
  {
    name: "trailing-record-without-blank-line",
    sse: `${chunk({ content: "hi" })}data: ${JSON.stringify({ id: "chatcmpl-1", choices: [{ index: 0, delta: {}, finish_reason: "stop" }] })}`,
  },
  { name: "malformed-json-chunk", sse: sse(chunk({ content: "hi" }), "data: {not json\n\n") },
  {
    name: "stream-error-object",
    sse: sse(
      chunk({ content: "partial" }),
      record({
        error: { message: "Upstream failed", code: 502, metadata: { raw: "provider said no" } },
      }),
    ),
  },
  {
    name: "stream-error-event",
    sse: sse(
      chunk({ reasoning_content: "hm" }),
      record({ error: { type: "server_error", message: "Overloaded" } }, "error"),
    ),
  },
  {
    name: "stream-error-thinking-details",
    sse: sse(
      chunk({ reasoning_details: [{ type: "reasoning.encrypted", data: "enc", id: "e1" }] }),
      record({ error: { message: "boom" } }),
    ),
  },
  {
    name: "chunk-not-object-skipped",
    sse: sse(record([1, 2]), record("text"), chunk({ content: "hi" }), finish("stop"), DONE),
  },
  {
    name: "chunk-without-choices",
    sse: sse(record({ id: "c" }), chunk({ content: "hi" }), finish("stop"), DONE),
  },
  {
    name: "http-400-json",
    status: 400,
    body: JSON.stringify({
      error: { message: "Invalid request", type: "invalid_request_error", param: null, code: null },
    }),
  },
  {
    name: "http-401-message-only",
    status: 401,
    body: JSON.stringify({ error: { message: "Incorrect API key" } }),
  },
  {
    name: "http-429-openrouter-raw",
    status: 429,
    body: JSON.stringify({
      error: { message: "Rate limited", code: 429, metadata: { raw: 'upstream "quota" hit' } },
    }),
  },
  { name: "http-500-error-string", status: 500, body: JSON.stringify({ error: "internal" }) },
  {
    name: "http-502-no-error-field",
    status: 502,
    body: JSON.stringify({ message: "bad gateway" }),
  },
  { name: "http-503-text", status: 503, body: "Service Unavailable" },
  { name: "http-504-empty", status: 504, body: "" },
  {
    name: "simple-reasoning",
    simple: true,
    options: { reasoning: "high" },
    sse: sse(chunk({ reasoning_content: "t" }), chunk({ content: "a" }), finish("stop"), DONE),
  },
  {
    name: "abort-before-request",
    abort: "before",
    sse: sse(chunk({ content: "hi" }), finish("stop"), DONE),
  },
  {
    name: "abort-on-response",
    abort: "on-response",
    sse: sse(chunk({ content: "hi" }), finish("stop"), DONE),
  },
];

function respond(testCase: Case): Response {
  if (testCase.status !== undefined) {
    return new Response(testCase.body ?? "", {
      status: testCase.status,
      headers: { "content-type": "application/json" },
    });
  }
  return new Response(testCase.sse ?? "", {
    status: 200,
    headers: { "content-type": "text/event-stream" },
  });
}

/** Events without `partial`; terminal events keep only their type and reason. */
function recordEvent(event: Record<string, unknown>): Record<string, unknown> {
  if (event["type"] === "done" || event["type"] === "error") {
    return { type: event["type"], reason: event["reason"] };
  }
  const { partial: _partial, ...rest } = event;
  return rest;
}

// The SDK reads these at construction; ovo never reads the process environment.
delete process.env["OPENAI_ORG_ID"];
delete process.env["OPENAI_PROJECT_ID"];

const results: Array<Record<string, unknown>> = [];
for (const testCase of CASES) {
  const controller = new AbortController();
  if (testCase.abort === "before") {
    controller.abort();
  }
  // oxlint-disable-next-line require-await -- fetch must return a promise; the response is ready
  const fetch: typeof globalThis.fetch = async () => respond(testCase);
  const caseModel = testCase.model ?? model();
  const context = testCase.context ?? CONTEXT;
  const options = {
    apiKey: "sk-test",
    maxRetries: 0,
    fetch,
    signal: controller.signal,
    ...(testCase.abort === "on-response" ? { onResponse: () => controller.abort() } : {}),
    ...testCase.options,
  };
  const stream =
    testCase.simple === true
      ? completions.streamSimple(caseModel, context, options)
      : completions.stream(caseModel, context, options);
  const events: Array<Record<string, unknown>> = [];
  // Cases run one at a time so each records only its own events.
  // oxlint-disable-next-line no-await-in-loop -- sequential by design
  for await (const event of stream) {
    events.push(recordEvent(event));
  }
  // oxlint-disable-next-line no-await-in-loop -- sequential by design
  const { timestamp: _timestamp, durationMs: _durationMs, ...message } = await stream.result();
  results.push({ ...testCase, model: caseModel, context, events, message });
}

const document = {
  generator: {
    script: "scripts/parity/generate-openai-completions-stream.ts",
    node: process.version,
  },
  cases: results,
};
const target = "vectors/ai/openai-completions-stream.json";
writeFileSync(target, `${JSON.stringify(document, null, 2)}\n`);
console.log(`${target}: ${results.length} cases`);
