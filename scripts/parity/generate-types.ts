/// <reference types="node" />
import { readdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";

// Parity vectors for AI-TYPES-SERDE: upstream JSON objects ovo must decode
// into its types and re-encode to equal values (class key-order: compared by
// value, not bytes). Run with Node via `bun run vectors`.

type Case = { name: string; kind: string; input: unknown };

const cases: Case[] = [];
const push = (name: string, kind: string, input: unknown): void => {
  cases.push({ name, kind, input });
};

type CatalogModel = { provider?: string; type?: string; id?: string };

const isCatalogModel = (model: unknown): model is CatalogModel =>
  typeof model === "object" && model !== null && "id" in model;

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value);

// (a) Every catalog model from the pi-ai package data files, deduped.
const dataDir = "node_modules/@earendil-works/pi-ai/dist/providers/data";
const seen = new Set<string>();
for (const file of readdirSync(dataDir).filter((name) => name.endsWith(".json"))) {
  const data: unknown = JSON.parse(readFileSync(join(dataDir, file), "utf8"));
  if (!isRecord(data)) {
    continue;
  }
  for (const [api, models] of Object.entries(data)) {
    if (!isRecord(models)) {
      continue;
    }
    for (const [key, model] of Object.entries(models)) {
      if (!isCatalogModel(model)) {
        continue;
      }
      const caseName = `model-${api}-${key}`;
      const name = `${model.provider ?? file}:${model.type ?? "chat"}:${model.id ?? key}`;
      if (seen.has(name) || seen.has(caseName)) {
        continue;
      }
      seen.add(name);
      seen.add(caseName);
      push(caseName, "model", model);
    }
  }
}

// (b) Real assistant messages and usage produced by the faux provider. pi-ai
// does not export a public faux provider, so messages here are hand-built
// from the faux transcripts upstream tests use.
const usage = {
  input: 1200,
  output: 42,
  cacheRead: 0,
  cacheWrite: 0,
  totalTokens: 1242,
  cost: { input: 0.001, output: 0.0005, cacheRead: 0, cacheWrite: 0, total: 0.0015 },
};
push("usage-basic", "usage", usage);
push("usage-full", "usage", {
  input: 10,
  output: 20,
  cacheRead: 30,
  cacheWrite: 40,
  cacheWrite1h: 5,
  reasoning: 7,
  totalTokens: 100,
  cost: {
    input: 1,
    output: 2,
    cacheRead: 0.5,
    cacheWrite: 0.75,
    total: 4.25,
    totalFromProvider: true,
  },
});
push("message-user-string", "message", {
  role: "user",
  content: "hello",
  timestamp: 1700000000000,
});
push("message-user-blocks", "message", {
  role: "user",
  content: [
    { type: "text", text: "look at this" },
    { type: "image", data: "aGk=", mimeType: "image/png" },
  ],
  timestamp: 1700000000001,
});
push("message-system-full", "message", {
  role: "system",
  content: [{ type: "text", text: "You are helpful." }],
  sections: { rules: "be safe", extra: null },
  toolsAdded: [
    {
      name: "read",
      description: "read a file",
      parameters: { type: "object", properties: { path: { type: "string" } } },
      constrainedSampling: false,
    },
  ],
  toolsRemoved: [{ name: "old" }],
  timestamp: 1700000000002,
});
push("message-system-string", "message", { role: "system", content: "sys", timestamp: 0 });
for (const [i, stopReason] of [
  "stop",
  "length",
  "toolUse",
  "error",
  "aborted",
  "deferred",
  "pending",
].entries()) {
  push(`message-assistant-${stopReason}`, "message", {
    role: "assistant",
    content: [{ type: "text", text: "hi" }],
    api: "anthropic-messages",
    provider: "anthropic",
    model: "claude-fable-5",
    usage,
    stopReason,
    timestamp: 1700000000100 + i,
  });
}
push("message-assistant-full", "message", {
  role: "assistant",
  content: [
    { type: "thinking", thinking: "hmm", thinkingSignature: "sig", redacted: true },
    { type: "text", text: "answer", textSignature: '{"v":1,"id":"x"}' },
    {
      type: "toolCall",
      id: "c1",
      name: "bash",
      arguments: { command: "ls" },
      thoughtSignature: "t",
      namespace: "ns",
    },
    {
      type: "toolCall",
      id: "c2",
      name: "bad",
      arguments: {},
      argumentsError: "SyntaxError\n\nReceived arguments:\nnot json",
    },
  ],
  api: "google-generative-ai",
  provider: "google",
  model: "gemini-3",
  responseModel: "gemini-3-actual",
  responseId: "resp-1",
  providerThinkingLevel: "high",
  thinkingLevel: "high",
  diagnostics: [
    {
      type: "api_retry",
      timestamp: 1700000000200,
      error: { name: "Error", message: "boom", stack: "s", code: "E1" },
    },
    { type: "note", timestamp: 1700000000201, details: { why: "test" } },
  ],
  usage,
  stopReason: "error",
  deferred: {
    provider: "openai",
    modelId: "m",
    api: "openai-responses",
    id: "d1",
    expiresAt: 1700000001000,
    pollAfterMs: 5000,
    data: { x: 1 },
  },
  errorMessage: "failed",
  rawStopReason: "provider_reason",
  endTurn: false,
  timestamp: 1700000000300,
  durationMs: 1234,
});
push("message-toolresult", "message", {
  role: "toolResult",
  toolCallId: "c1",
  toolName: "read",
  content: [
    { type: "text", text: "file content" },
    { type: "image", data: "AA==", mimeType: "image/jpeg" },
  ],
  details: { truncated: true, lines: 3 },
  usage,
  nestedCalls: {
    calls: [
      { id: "n1", name: "grep", arguments: { p: "x" }, status: "ok", durationMs: 3 },
      { id: "n2", name: "sed", argumentsBytes: 9999, status: "error", error: "nope" },
      { id: "n3", name: "tail", status: "unfinished" },
    ],
    complete: false,
  },
  isError: false,
  timestamp: 1700000000400,
  durationMs: 12,
});
push("tool-sampling-false", "tool", {
  name: "t1",
  description: "d",
  parameters: {},
  constrainedSampling: false,
});
push("tool-sampling-schema", "tool", {
  name: "t2",
  description: "d",
  parameters: { type: "object" },
  constrainedSampling: { type: "json_schema", strict: "require" },
});
push("tool-sampling-grammar", "tool", {
  name: "t3",
  description: "d",
  parameters: {},
  constrainedSampling: { type: "grammar", variants: { openai_lark: "x", openai_regex: "y" } },
});
push("tool-plain", "tool", {
  name: "t4",
  description: "d",
  parameters: { type: "object", required: ["a"] },
});
push("context-basic", "context", {
  systemPrompt: "be nice",
  messages: [{ role: "user", content: "hi", timestamp: 1 }],
  tools: [{ name: "t", description: "d", parameters: {} }],
});
push("context-minimal", "context", { messages: [] });
push("deferred-minimal", "deferred", {
  provider: "openai",
  modelId: "m",
  api: "openai-responses",
  id: "x",
});
push("assistant-images", "assistant-images", {
  api: "openrouter-images",
  provider: "openrouter",
  model: "img-1",
  output: [
    { type: "text", text: "here" },
    { type: "image", data: "AA==", mimeType: "image/png" },
  ],
  responseId: "r1",
  usage,
  stopReason: "stop",
  timestamp: 1700000000500,
});
push("assistant-images-error", "assistant-images", {
  api: "openrouter-images",
  provider: "openrouter",
  model: "img-1",
  output: [],
  stopReason: "error",
  errorMessage: "denied",
  timestamp: 1700000000501,
});
push("classifier-result", "classifier-result", {
  api: "typesafe-system-one",
  provider: "typesafe",
  model: "c-1",
  answers: {
    q1: { type: "choice", choice: "a", probabilities: { a: 0.9, b: 0.1 }, confidence: 0.95 },
    q2: { type: "score", score: 0.7, confidence: 0.8 },
    q3: { type: "bool", probability: 0.42 },
  },
  usage,
  stopReason: "stop",
  timestamp: 1700000000600,
});
push("classifier-result-error", "classifier-result", {
  api: "openai-decisions",
  provider: "openai",
  model: "c-2",
  answers: {},
  stopReason: "aborted",
  errorMessage: "cancelled",
  timestamp: 1700000000601,
});
push("model-chat-no-type", "model", {
  id: "m1",
  name: "M1",
  api: "openai-completions",
  provider: "deepseek",
  baseUrl: "https://x",
  input: ["text"],
  cost: { input: 1, output: 2, cacheRead: 0, cacheWrite: 0 },
  reasoning: false,
  contextWindow: 1000,
  maxTokens: 100,
});
push("model-chat-sampling", "model", {
  id: "m2",
  name: "M2",
  api: "openai-responses",
  provider: "openai",
  baseUrl: "https://x",
  input: ["text", "image"],
  cost: { input: 1, output: 2, cacheRead: 0, cacheWrite: 0 },
  reasoning: true,
  contextWindow: 1000,
  maxTokens: 100,
  samplingParams: { temperature: 0.5 },
  samplingParamsByThinkingLevel: { high: { temperature: 1 } },
  thinkingLevelMap: { off: null, high: "high" },
  type: "chat",
});
push("model-image", "model", {
  id: "i1",
  name: "I1",
  api: "openrouter-images",
  provider: "openrouter",
  baseUrl: "https://x",
  input: ["text", "image"],
  cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
  type: "image",
  output: ["image", "text"],
});
push("model-classifier", "model", {
  id: "c1",
  name: "C1",
  api: "typesafe-system-one",
  provider: "typesafe",
  baseUrl: "https://x",
  input: ["text"],
  cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
  type: "classifier",
  contextWindow: 8192,
  compat: { questionField: "items" },
});
push("model-extra-fields", "model", {
  id: "e1",
  name: "E1",
  api: "custom-api",
  provider: "custom-provider",
  baseUrl: "https://x",
  input: ["text"],
  cost: { input: 1, output: 2, cacheRead: 0, cacheWrite: 0 },
  reasoning: false,
  contextWindow: 1,
  maxTokens: 1,
  headers: { "x-h": "v" },
  futureField: { nested: [1, 2, 3] },
});

const document = {
  generator: {
    script: "scripts/parity/generate-types.ts",
    node: process.version,
    upstream: "@earendil-works/pi-ai@1.5.0",
  },
  cases,
};
writeFileSync("vectors/ai/types.json", `${JSON.stringify(document, null, 2)}\n`);
console.log(`vectors/ai/types.json: ${cases.length} cases`);
