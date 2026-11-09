// Parity vectors for the openai-responses response side (AI-API-OPENAI-RESPONSES):
// scripted SSE bodies and error responses run through the published adapter's
// `stream`/`streamSimple`; each case records the events (without `partial`) and
// the final message. Run with `node scripts/parity/generate-openai-responses-stream.ts`.
import { writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

for (const key of ["OPENAI_ORG_ID", "OPENAI_PROJECT_ID", "OPENAI_BASE_URL"]) {
  Reflect.deleteProperty(process.env, key);
}

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
const responses: unknown = await import(
  pathToFileURL(join(piAiDir, "dist/api/openai-responses.js")).href
);
if (!isStreamModule(responses)) {
  throw new Error("openai-responses dist module has no stream export");
}

type Json = Record<string, unknown>;

const model = (overrides: Json = {}): Json => ({
  id: "m",
  name: "M",
  api: "openai-responses",
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

/** One SSE record named after its event type. */
const record = (data: Json): string =>
  `event: ${String(data["type"])}\ndata: ${JSON.stringify(data)}\n\n`;
/** One SSE record without an event name. */
const unnamed = (data: Json): string => `data: ${JSON.stringify(data)}\n\n`;
const sse = (...records: string[]) => records.join("");

const created = record({
  type: "response.created",
  response: { id: "resp_1", status: "in_progress" },
});
const added = (outputIndex: number, item: Json) =>
  record({ type: "response.output_item.added", output_index: outputIndex, item });
const itemDone = (outputIndex: number, item: Json) =>
  record({ type: "response.output_item.done", output_index: outputIndex, item });
const delta = (type: string, outputIndex: number, value: string) =>
  record({ type, output_index: outputIndex, delta: value });
const USAGE = { input_tokens: 100, output_tokens: 20, total_tokens: 120 };
const completed = (response: Json = {}) =>
  record({
    type: "response.completed",
    response: { id: "resp_1", status: "completed", output: [], usage: USAGE, ...response },
  });
const incomplete = (reason: string | undefined) =>
  record({
    type: "response.incomplete",
    response: {
      id: "resp_1",
      status: "incomplete",
      output: [],
      usage: USAGE,
      incomplete_details: reason === undefined ? null : { reason },
    },
  });

const message = (id: string, text: string, extra: Json = {}) => ({
  type: "message",
  id,
  role: "assistant",
  status: "completed",
  content: [{ type: "output_text", text, annotations: [] }],
  ...extra,
});
const emptyMessage = (id: string) => ({
  type: "message",
  id,
  role: "assistant",
  status: "in_progress",
  content: [],
});
const textReply = (text: string) =>
  sse(
    added(0, emptyMessage("msg_1")),
    delta("response.output_text.delta", 0, text),
    itemDone(0, message("msg_1", text)),
  );
const reasoning = (extra: Json = {}) => ({ type: "reasoning", id: "rs_1", summary: [], ...extra });
const functionCall = (extra: Json = {}) => ({
  type: "function_call",
  id: "fc_1",
  call_id: "call_1",
  name: "read",
  arguments: "",
  status: "in_progress",
  ...extra,
});
const customCall = (extra: Json = {}) => ({
  type: "custom_tool_call",
  id: "ctc_1",
  call_id: "call_1",
  name: "patch",
  input: "",
  ...extra,
});

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

const CASES: Case[] = [
  { name: "text", sse: sse(created, textReply("Hello"), completed()) },
  {
    name: "text-split-deltas",
    sse: sse(
      created,
      added(0, emptyMessage("msg_1")),
      delta("response.output_text.delta", 0, "Hel"),
      delta("response.output_text.delta", 0, "lo"),
      itemDone(0, message("msg_1", "Hello")),
      completed(),
    ),
  },
  {
    name: "text-phase",
    sse: sse(
      added(0, { ...emptyMessage("msg_1"), phase: "commentary" }),
      delta("response.output_text.delta", 0, "a"),
      itemDone(0, message("msg_1", "a", { phase: "commentary" })),
      added(1, { ...emptyMessage("msg_2"), phase: "final_answer" }),
      itemDone(1, message("msg_2", "b", { phase: "final_answer" })),
      completed(),
    ),
  },
  {
    name: "refusal",
    sse: sse(
      added(0, emptyMessage("msg_1")),
      delta("response.refusal.delta", 0, "No."),
      itemDone(0, { ...emptyMessage("msg_1"), content: [{ type: "refusal", refusal: "No." }] }),
      completed(),
    ),
  },
  {
    name: "reasoning-summary",
    sse: sse(
      added(0, reasoning()),
      delta("response.reasoning_summary_text.delta", 0, "First."),
      record({ type: "response.reasoning_summary_part.done", output_index: 0 }),
      delta("response.reasoning_summary_text.delta", 0, "Second."),
      itemDone(
        0,
        reasoning({
          summary: [
            { type: "summary_text", text: "First." },
            { type: "summary_text", text: "Second." },
          ],
          encrypted_content: "enc",
        }),
      ),
      added(1, emptyMessage("msg_1")),
      itemDone(1, message("msg_1", "ok")),
      completed(),
    ),
  },
  {
    name: "reasoning-text",
    sse: sse(
      added(0, reasoning()),
      delta("response.reasoning_text.delta", 0, "raw"),
      itemDone(0, reasoning({ content: [{ type: "reasoning_text", text: "raw thought" }] })),
      completed(),
    ),
  },
  {
    name: "reasoning-done-keeps-streamed",
    sse: sse(
      added(0, reasoning()),
      delta("response.reasoning_summary_text.delta", 0, "streamed"),
      itemDone(0, reasoning()),
      completed(),
    ),
  },
  {
    name: "reasoning-encrypted-backfill",
    sse: sse(
      added(0, reasoning()),
      itemDone(0, reasoning({ summary: [{ type: "summary_text", text: "t" }] })),
      completed({ output: [reasoning({ encrypted_content: "late-enc" })] }),
    ),
  },
  {
    name: "function-call",
    sse: sse(
      added(0, functionCall()),
      delta("response.function_call_arguments.delta", 0, '{"path":'),
      delta("response.function_call_arguments.delta", 0, '"a"}'),
      record({
        type: "response.function_call_arguments.done",
        output_index: 0,
        arguments: '{"path":"a"}',
      }),
      itemDone(0, functionCall({ arguments: '{"path":"a"}', status: "completed" })),
      completed(),
    ),
  },
  {
    name: "function-call-arguments-done-continues",
    sse: sse(
      added(0, functionCall()),
      delta("response.function_call_arguments.delta", 0, '{"pa'),
      record({
        type: "response.function_call_arguments.done",
        output_index: 0,
        arguments: '{"path":"b"}',
      }),
      itemDone(0, functionCall({ arguments: '{"path":"b"}' })),
      completed(),
    ),
  },
  {
    name: "function-call-arguments-done-restarts",
    sse: sse(
      added(0, functionCall()),
      delta("response.function_call_arguments.delta", 0, '{"x":1'),
      record({
        type: "response.function_call_arguments.done",
        output_index: 0,
        arguments: '{"path":"c"}',
      }),
      itemDone(0, functionCall({ arguments: '{"path":"c"}' })),
      completed(),
    ),
  },
  {
    name: "function-call-initial-arguments-and-namespace",
    sse: sse(
      added(0, functionCall({ arguments: '{"path":' })),
      delta("response.function_call_arguments.delta", 0, '"a"}'),
      itemDone(0, functionCall({ arguments: '{"path":"a"}', namespace: "ns" })),
      completed(),
    ),
  },
  {
    name: "function-call-done-only",
    sse: sse(itemDone(0, functionCall({ arguments: '{"path":"z"}' })), completed()),
  },
  {
    name: "function-call-malformed-arguments",
    sse: sse(
      added(0, functionCall()),
      itemDone(0, functionCall({ arguments: '{"path": ' })),
      completed(),
    ),
  },
  {
    name: "text-and-two-function-calls",
    sse: sse(
      textReply("Reading."),
      added(1, functionCall()),
      added(2, functionCall({ id: "fc_2", call_id: "call_2" })),
      delta("response.function_call_arguments.delta", 2, '{"path":"b"}'),
      delta("response.function_call_arguments.delta", 1, '{"path":"a"}'),
      itemDone(1, functionCall({ arguments: '{"path":"a"}' })),
      itemDone(2, functionCall({ id: "fc_2", call_id: "call_2", arguments: '{"path":"b"}' })),
      completed(),
    ),
  },
  {
    name: "custom-tool-call",
    model: GRAMMAR_MODEL,
    context: GRAMMAR_CONTEXT,
    sse: sse(
      added(0, customCall()),
      delta("response.custom_tool_call_input.delta", 0, 'line "1"\n'),
      delta("response.custom_tool_call_input.delta", 0, "line 2"),
      record({
        type: "response.custom_tool_call_input.done",
        output_index: 0,
        input: 'line "1"\nline 2',
      }),
      itemDone(0, customCall({ input: 'line "1"\nline 2' })),
      completed(),
    ),
  },
  {
    name: "custom-tool-call-initial-input",
    model: GRAMMAR_MODEL,
    context: GRAMMAR_CONTEXT,
    sse: sse(
      added(0, customCall({ input: "ab" })),
      itemDone(0, customCall({ input: "abc", namespace: "ns" })),
      completed(),
    ),
  },
  {
    name: "custom-tool-call-initial-input-then-delta",
    model: GRAMMAR_MODEL,
    context: GRAMMAR_CONTEXT,
    sse: sse(
      added(0, customCall({ input: "ab" })),
      delta("response.custom_tool_call_input.delta", 0, "c"),
      itemDone(0, customCall({ input: "abc" })),
      completed(),
    ),
  },
  {
    name: "custom-tool-call-unknown-name",
    model: GRAMMAR_MODEL,
    context: GRAMMAR_CONTEXT,
    sse: sse(
      added(0, customCall({ name: "invented" })),
      itemDone(0, customCall({ name: "invented", input: "x" })),
      completed(),
    ),
  },
  {
    name: "custom-tool-call-non-monotonic",
    model: GRAMMAR_MODEL,
    context: GRAMMAR_CONTEXT,
    sse: sse(
      added(0, customCall()),
      delta("response.custom_tool_call_input.delta", 0, "abc"),
      itemDone(0, customCall({ input: "xyz" })),
      completed(),
    ),
  },
  {
    name: "custom-tool-call-error-before-piece",
    model: GRAMMAR_MODEL,
    context: GRAMMAR_CONTEXT,
    sse: sse(
      added(0, customCall({ input: "ab" })),
      unnamed({ type: "error", code: "server_error", message: "boom" }),
    ),
  },
  {
    name: "events-for-unknown-slots-skipped",
    sse: sse(
      added(0, emptyMessage("msg_1")),
      delta("response.output_text.delta", 5, "lost"),
      delta("response.reasoning_summary_text.delta", 0, "wrong kind"),
      added(1, { type: "web_search_call", id: "ws_1" }),
      itemDone(0, message("msg_1", "kept")),
      completed(),
    ),
  },
  {
    name: "missing-output-index",
    sse: sse(
      record({ type: "response.output_item.added", item: emptyMessage("msg_1") }),
      record({ type: "response.output_text.delta", delta: "no index" }),
      record({ type: "response.output_item.done", item: message("msg_1", "no index") }),
      completed(),
    ),
  },
  {
    name: "usage-cached-and-reasoning",
    sse: sse(
      textReply("hi"),
      completed({
        usage: {
          input_tokens: 1000,
          output_tokens: 50,
          total_tokens: 1050,
          input_tokens_details: { cached_tokens: 600, cache_write_tokens: 100 },
          output_tokens_details: { reasoning_tokens: 30 },
        },
      }),
    ),
  },
  { name: "usage-missing", sse: sse(textReply("hi"), completed({ usage: undefined })) },
  {
    name: "service-tier-response-flex",
    sse: sse(textReply("hi"), completed({ service_tier: "flex" })),
  },
  {
    name: "service-tier-request-priority",
    options: { serviceTier: "priority" },
    sse: sse(textReply("hi"), completed()),
  },
  {
    name: "service-tier-response-overrides-request",
    options: { serviceTier: "flex" },
    sse: sse(textReply("hi"), completed({ service_tier: "default" })),
  },
  {
    name: "service-tier-fast-gpt-5.5",
    model: model({ id: "gpt-5.5" }),
    sse: sse(textReply("hi"), completed({ service_tier: "fast" })),
  },
  {
    name: "response-ids",
    sse: sse(created, textReply("hi"), completed({ id: "resp_final" })),
  },
  {
    name: "incomplete-max-output-tokens",
    sse: sse(textReply("cut"), incomplete("max_output_tokens")),
  },
  { name: "incomplete-content-filter", sse: sse(textReply("cut"), incomplete("content_filter")) },
  { name: "incomplete-without-reason", sse: sse(textReply("cut"), incomplete(undefined)) },
  {
    name: "completed-status-cancelled",
    sse: sse(textReply("hi"), completed({ status: "cancelled" })),
  },
  { name: "completed-status-unknown", sse: sse(textReply("hi"), completed({ status: "weird" })) },
  { name: "completed-status-missing", sse: sse(textReply("hi"), completed({ status: undefined })) },
  {
    name: "failed-with-error",
    sse: sse(
      textReply("partial"),
      record({
        type: "response.failed",
        response: {
          id: "resp_1",
          status: "failed",
          error: { code: "server_error", message: "It broke" },
        },
      }),
    ),
  },
  {
    name: "failed-with-incomplete-details",
    sse: sse(
      record({
        type: "response.failed",
        response: {
          status: "failed",
          error: null,
          incomplete_details: { reason: "max_output_tokens" },
        },
      }),
    ),
  },
  {
    name: "failed-without-details",
    sse: sse(
      record({ type: "response.failed", response: { status: "failed", error: { message: "" } } }),
    ),
  },
  {
    name: "failed-no-error",
    sse: sse(record({ type: "response.failed", response: { status: "failed" } })),
  },
  {
    name: "error-event-unnamed",
    sse: sse(
      textReply("partial"),
      unnamed({ type: "error", code: "rate_limit", message: "Slow down" }),
    ),
  },
  {
    name: "error-event-named",
    sse: sse(
      textReply("partial"),
      record({ type: "error", code: "server_error", message: "Overloaded" }),
    ),
  },
  { name: "no-terminal-event", sse: sse(textReply("hi")) },
  {
    name: "unfinished-tool-call",
    sse: sse(
      added(0, functionCall()),
      delta("response.function_call_arguments.delta", 0, "{"),
      completed(),
    ),
  },
  {
    name: "malformed-json-record",
    sse: sse(textReply("hi"), "event: response.completed\ndata: {oops\n\n"),
  },
  {
    name: "trailing-record-without-blank-line",
    sse: `${textReply("hi")}event: response.completed\ndata: ${JSON.stringify({ type: "response.completed", response: { id: "r", status: "completed" } })}`,
  },
  {
    name: "http-400",
    status: 400,
    body: JSON.stringify({
      error: {
        message: "Invalid input",
        type: "invalid_request_error",
        param: "input",
        code: null,
      },
    }),
  },
  {
    name: "http-401-other-provider",
    model: model({ provider: "custom", baseUrl: "https://llm.example.com/v1" }),
    status: 401,
    body: JSON.stringify({ error: { message: "Bad key" } }),
  },
  {
    name: "http-429-usage-limit",
    status: 429,
    body: JSON.stringify({
      error: { message: "Limit", code: "subscription_sharing_usage_limit_exceeded" },
    }),
  },
  {
    name: "http-429-workspace-limit",
    status: 429,
    body: JSON.stringify({
      error: {
        message: "Limit",
        code: "subscription_sharing_workplace_individual_usage_limit_exceeded",
      },
    }),
  },
  { name: "http-500-text", status: 500, body: "Internal Server Error" },
  {
    name: "configuration-update-thinking-level",
    model: model({
      compat: { supportsMidConvoEffort: true },
      thinkingLevelMap: { high: "x-high" },
    }),
    options: { reasoningEffort: "high" },
    sse: sse(textReply("hi"), completed()),
  },
  {
    name: "configuration-update-thinking-level-on-error",
    model: model({ compat: { supportsMidConvoEffort: true } }),
    options: { reasoningEffort: "low" },
    status: 400,
    body: JSON.stringify({ error: { message: "nope" } }),
  },
  {
    name: "simple-reasoning",
    simple: true,
    options: { reasoning: "high" },
    sse: sse(added(0, reasoning()), itemDone(0, reasoning()), textReply("a"), completed()),
  },
  { name: "abort-before-request", abort: "before", sse: sse(textReply("hi"), completed()) },
  { name: "abort-on-response", abort: "on-response", sse: sse(textReply("hi"), completed()) },
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
  const controller = new AbortController();
  if (testCase.abort === "before") {
    controller.abort();
  }
  // oxlint-disable-next-line require-await -- fetch must return a promise; the response is ready
  const fetch: typeof globalThis.fetch = async () => respond(testCase);
  const caseModel = testCase.model ?? model();
  const context = testCase.context ?? CONTEXT;
  const providerEvents: Array<Record<string, unknown>> = [];
  const options = {
    apiKey: "sk-test",
    maxRetries: 0,
    fetch,
    signal: controller.signal,
    ...(testCase.abort === "on-response" ? { onResponse: () => controller.abort() } : {}),
    onProviderStreamEvent: (data: Record<string, unknown>) => {
      providerEvents.push(data);
    },
    ...testCase.options,
  };
  const stream =
    testCase.simple === true
      ? responses.streamSimple(caseModel, context, options)
      : responses.stream(caseModel, context, options);
  const events: Array<Record<string, unknown>> = [];
  // Cases run one at a time so each records only its own events.
  // oxlint-disable-next-line no-await-in-loop -- sequential by design
  for await (const event of stream) {
    events.push(recordEvent(event));
  }
  // oxlint-disable-next-line no-await-in-loop -- sequential by design
  const { timestamp: _timestamp, durationMs: _durationMs, ...message } = await stream.result();
  results.push({ ...testCase, model: caseModel, context, events, providerEvents, message });
}

const document = {
  generator: {
    script: "scripts/parity/generate-openai-responses-stream.ts",
    node: process.version,
  },
  cases: results,
};
const target = "vectors/ai/openai-responses-stream.json";
writeFileSync(target, `${JSON.stringify(document, null, 2)}\n`);
console.log(`${target}: ${results.length} cases`);
