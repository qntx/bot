// Parity vectors for the anthropic-messages response side (AI-API-ANTHROPIC):
// scripted SSE bodies and error responses run through the published adapter's
// `stream`; each case records the events (without `partial`) and the final
// message. Run with `node scripts/parity/generate-anthropic-stream.ts`.
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

function isStreamModule(value: unknown): value is { stream: StreamFn } {
  return (
    typeof value === "object" &&
    value !== null &&
    "stream" in value &&
    typeof value.stream === "function"
  );
}

const piAiDir = dirname(dirname(fileURLToPath(import.meta.resolve("@earendil-works/pi-ai"))));
const anthropicModule: unknown = await import(
  pathToFileURL(join(piAiDir, "dist/api/anthropic-messages.js")).href
);
if (!isStreamModule(anthropicModule)) {
  throw new Error("anthropic-messages dist module has no stream export");
}
const streamAnthropic = anthropicModule.stream;

const MODEL = {
  id: "claude-test",
  name: "Claude Test",
  api: "anthropic-messages",
  provider: "anthropic",
  baseUrl: "https://api.anthropic.com",
  reasoning: true,
  input: ["text", "image"],
  cost: { input: 3, output: 15, cacheRead: 0.3, cacheWrite: 3.75 },
  contextWindow: 200_000,
  maxTokens: 8192,
};

const CONTEXT = { messages: [{ role: "user", content: "hi", timestamp: 0 }] };

/** One SSE record. */
const record = (event: string, data: unknown): string =>
  `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;

const BASE_USAGE = { input_tokens: 10, output_tokens: 1 };
const DELTA_USAGE = { output_tokens: 5 };
const start = (usage: Record<string, unknown> = BASE_USAGE, model = "claude-test") =>
  record("message_start", {
    type: "message_start",
    message: { id: "msg_1", type: "message", role: "assistant", model, content: [], usage },
  });
const blockStart = (index: number, block: Record<string, unknown>) =>
  record("content_block_start", { type: "content_block_start", index, content_block: block });
const delta = (index: number, value: Record<string, unknown>) =>
  record("content_block_delta", { type: "content_block_delta", index, delta: value });
const blockStop = (index: number) =>
  record("content_block_stop", { type: "content_block_stop", index });
const messageDelta = (
  stopReason: string,
  usage: Record<string, unknown> = DELTA_USAGE,
  extra = {},
) =>
  record("message_delta", {
    type: "message_delta",
    delta: { stop_reason: stopReason, stop_sequence: null, ...extra },
    usage,
  });
const stop = record("message_stop", { type: "message_stop" });
const ping = record("ping", { type: "ping" });

const text = (index: number, parts: string[]) => [
  blockStart(index, { type: "text", text: "" }),
  ...parts.map((part) => delta(index, { type: "text_delta", text: part })),
  blockStop(index),
];
const toolUse = (index: number, id: string, name: string, parts: string[]) => [
  blockStart(index, { type: "tool_use", id, name, input: {} }),
  ...parts.map((part) => delta(index, { type: "input_json_delta", partial_json: part })),
  blockStop(index),
];

type Case = {
  name: string;
  provider?: string;
  sse?: string;
  status?: number;
  body?: string;
  abort?: "before";
};

const CASES: Case[] = [
  {
    name: "text",
    sse: [start(), ...text(0, ["Hel", "lo"]), messageDelta("end_turn"), stop].join(""),
  },
  {
    name: "thinking-signature",
    sse: [
      start(),
      blockStart(0, { type: "thinking", thinking: "", signature: "" }),
      delta(0, { type: "thinking_delta", thinking: "Let me " }),
      delta(0, { type: "thinking_delta", thinking: "think." }),
      delta(0, { type: "signature_delta", signature: "sig-" }),
      delta(0, { type: "signature_delta", signature: "abc" }),
      blockStop(0),
      ...text(1, ["Done."]),
      messageDelta("end_turn"),
      stop,
    ].join(""),
  },
  {
    name: "redacted-thinking",
    sse: [
      start(),
      blockStart(0, { type: "redacted_thinking", data: "opaque-data" }),
      blockStop(0),
      ...text(1, ["ok"]),
      messageDelta("end_turn"),
      stop,
    ].join(""),
  },
  {
    name: "tool-use-split-json",
    sse: [
      start(),
      ...toolUse(0, "toolu_1", "read", ['{"pa', 'th": "a.', 'txt"}']),
      messageDelta("tool_use"),
      stop,
    ].join(""),
  },
  {
    name: "two-tool-uses",
    sse: [
      start(),
      ...text(0, ["Reading."]),
      ...toolUse(1, "toolu_1", "read", ['{"path":"a"}']),
      ...toolUse(2, "toolu_2", "read", ['{"path":', '"b"}']),
      messageDelta("tool_use"),
      stop,
    ].join(""),
  },
  {
    name: "max-tokens",
    sse: [start(), ...text(0, ["cut"]), messageDelta("max_tokens"), stop].join(""),
  },
  {
    name: "refusal",
    sse: [
      start(),
      ...text(0, ["no"]),
      messageDelta(
        "refusal",
        { output_tokens: 2 },
        { stop_details: { explanation: "Declined for safety." } },
      ),
      stop,
    ].join(""),
  },
  {
    name: "error-event",
    sse: [
      start(),
      ...text(0, ["partial"]),
      `event: error\ndata: {"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}\n\n`,
    ].join(""),
  },
  {
    name: "http-400",
    status: 400,
    body: JSON.stringify({
      type: "error",
      error: { type: "invalid_request_error", message: "max_tokens: too large" },
    }),
  },
  { name: "truncated", sse: [start(), ...text(0, ["half"])].join("") },
  {
    name: "pings-interleaved",
    sse: [ping, start(), ping, ...text(0, ["a"]), ping, messageDelta("end_turn"), stop].join(""),
  },
  {
    name: "cache-usage",
    sse: [
      start({
        input_tokens: 100,
        output_tokens: 1,
        cache_read_input_tokens: 2000,
        cache_creation_input_tokens: 500,
        cache_creation: { ephemeral_1h_input_tokens: 200 },
      }),
      ...text(0, ["cached"]),
      messageDelta("end_turn", {
        output_tokens: 40,
        output_tokens_details: { thinking_tokens: 12 },
      }),
      stop,
    ].join(""),
  },
  {
    name: "response-model-differs",
    sse: [
      start(undefined, "claude-test-20260101"),
      ...text(0, ["hi"]),
      messageDelta("end_turn"),
      stop,
    ].join(""),
  },
  {
    name: "openrouter-billed-cost",
    provider: "openrouter",
    sse: [
      start(),
      ...text(0, ["hi"]),
      messageDelta("end_turn", { output_tokens: 5, cost: 0.0042, is_byok: false }),
      stop,
    ].join(""),
  },
  {
    name: "openrouter-byok-cost",
    provider: "openrouter",
    sse: [
      start(),
      ...text(0, ["hi"]),
      messageDelta("end_turn", {
        output_tokens: 5,
        cost: 0.0001,
        is_byok: true,
        cost_details: {
          upstream_inference_cost: null,
          upstream_inference_prompt_cost: 0.002,
          upstream_inference_completions_cost: 0.003,
        },
      }),
      stop,
    ].join(""),
  },
  {
    name: "abort-before-request",
    abort: "before",
    sse: [start(), ...text(0, ["hi"]), messageDelta("end_turn"), stop].join(""),
  },
  {
    name: "billed-cost-ignored-off-openrouter",
    sse: [
      start(),
      ...text(0, ["hi"]),
      messageDelta("end_turn", { output_tokens: 5, cost: 0.0042, is_byok: false }),
      stop,
    ].join(""),
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

const results: Array<Record<string, unknown>> = [];
for (const testCase of CASES) {
  // oxlint-disable-next-line require-await -- fetch must return a promise; the response is ready
  const fetch: typeof globalThis.fetch = async () => respond(testCase);
  const model = testCase.provider === undefined ? MODEL : { ...MODEL, provider: testCase.provider };
  const controller = new AbortController();
  if (testCase.abort === "before") {
    controller.abort();
  }
  const providerEvents: Array<Record<string, unknown>> = [];
  const stream = streamAnthropic(model, CONTEXT, {
    apiKey: "sk-ant-api03-test",
    maxRetries: 0,
    fetch,
    signal: controller.signal,
    onProviderStreamEvent: (data: Record<string, unknown>) => {
      providerEvents.push(data);
    },
  });
  const events: Array<Record<string, unknown>> = [];
  // Cases run one at a time so each records only its own events.
  // oxlint-disable-next-line no-await-in-loop -- sequential by design
  for await (const event of stream) {
    events.push(recordEvent(event));
  }
  // oxlint-disable-next-line no-await-in-loop -- sequential by design
  const { timestamp: _timestamp, durationMs: _durationMs, ...message } = await stream.result();
  results.push({ ...testCase, events, providerEvents, message });
}

const document = {
  generator: { script: "scripts/parity/generate-anthropic-stream.ts", node: process.version },
  model: MODEL,
  context: CONTEXT,
  cases: results,
};
const target = "vectors/ai/anthropic-stream.json";
writeFileSync(target, `${JSON.stringify(document, null, 2)}\n`);
console.log(`${target}: ${results.length} cases`);
