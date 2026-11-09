/// <reference types="node" />
import { writeFileSync } from "node:fs";

import {
  calculateCost,
  clampThinkingLevel,
  getSupportedThinkingLevels,
  isContextOverflow,
  isRecoverableLength,
  isRetryableAssistantError,
  normalizeContext,
  retryDelayMs,
} from "@earendil-works/pi-ai";
import type {
  AssistantMessage,
  ImageContent,
  TextContent,
  Context,
  Message,
  Model,
  ModelThinkingLevel,
  Usage,
} from "@earendil-works/pi-ai";

import {
  adjustMaxTokensForThinking,
  calculateContextTokens,
  clampMaxTokensToContext,
  clampReasoning,
  clampThinkingBudgetToAnswerRoom,
  estimateContextTokens,
  estimateMessageTokens,
  estimateTextAndImageContentTokens,
  estimateTextTokens,
  resolveSamplingParams,
  thinkingBudgetForLevel,
} from "./pi-internal.js";

type Case = { name: string; kind: string; args: unknown; result: unknown };
const push = (cases: Case[], name: string, kind: string, args: unknown, result: unknown) =>
  cases.push({ name, kind, args, result });

const usage = (over: Partial<Usage> = {}): Usage => ({
  input: 0,
  output: 0,
  cacheRead: 0,
  cacheWrite: 0,
  totalTokens: 0,
  cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
  ...over,
});

const assistant = (over: Partial<AssistantMessage> = {}): AssistantMessage => ({
  role: "assistant",
  content: [],
  api: "openai-completions",
  provider: "test",
  model: "m",
  usage: usage(),
  stopReason: "stop",
  timestamp: 1,
  ...over,
});

const user = (content: string | Array<TextContent | ImageContent>, timestamp = 1): Message => ({
  role: "user",
  content,
  timestamp,
});

const modelBase: Model<"openai-completions"> = {
  id: "m",
  name: "M",
  api: "openai-completions",
  provider: "test",
  baseUrl: "https://x.test/v1",
  reasoning: true,
  input: ["text" as const],
  cost: { input: 1, output: 5, cacheRead: 0.2, cacheWrite: 1.25 },
  contextWindow: 10000,
  maxTokens: 8000,
};

// ---------------- AI-ESTIMATE ----------------
const estimateCases: Case[] = [];
push(
  estimateCases,
  "calculate-total",
  "calculateContextTokens",
  usage({ totalTokens: 5, input: 1 }),
  calculateContextTokens(usage({ totalTokens: 5, input: 1 })),
);
push(
  estimateCases,
  "calculate-sum",
  "calculateContextTokens",
  usage({ input: 3, output: 2, cacheRead: 4, cacheWrite: 1 }),
  calculateContextTokens(usage({ input: 3, output: 2, cacheRead: 4, cacheWrite: 1 })),
);
for (const [name, text] of [
  ["empty", ""],
  ["three", "abc"],
  ["seven", "a".repeat(7)],
  ["astral", "a🙈b"],
] as Array<[string, string]>) {
  push(estimateCases, `text-${name}`, "estimateTextTokens", text, estimateTextTokens(text));
}
push(
  estimateCases,
  "blocks-text-image",
  "estimateTextAndImageContentTokens",
  [
    { type: "text", text: "ab" },
    { type: "image", data: "x", mimeType: "image/png" },
  ],
  estimateTextAndImageContentTokens([
    { type: "text", text: "ab" },
    { type: "image", data: "x", mimeType: "image/png" },
  ]),
);
push(
  estimateCases,
  "string-content",
  "estimateTextAndImageContentTokens",
  "12345",
  estimateTextAndImageContentTokens("12345"),
);
const tool = { name: "t", description: "d", parameters: { type: "object", properties: {} } };
const msgs: Array<[string, Message]> = [
  [
    "system-sections-tools",
    {
      role: "system",
      content: "prompt",
      sections: { a: "A" },
      toolsAdded: [tool],
      toolsRemoved: [{ name: "gone" }],
      timestamp: 0,
    },
  ],
  ["user-text", user("hello world")],
  [
    "user-image",
    user([
      { type: "text", text: "see" },
      { type: "image", data: "x", mimeType: "image/png" },
    ]),
  ],
  [
    "tool-result",
    {
      role: "toolResult",
      toolCallId: "1",
      toolName: "t",
      content: [{ type: "text", text: "out" }],
      isError: false,
      timestamp: 2,
    },
  ],
  [
    "assistant-all",
    assistant({
      content: [
        { type: "thinking", thinking: "think" },
        { type: "text", text: "say" },
        { type: "toolCall", id: "1", name: "run", arguments: { a: 1, b: "x" } },
      ],
    }),
  ],
];
for (const [name, msg] of msgs) {
  push(estimateCases, `message-${name}`, "estimateMessageTokens", msg, estimateMessageTokens(msg));
}
const estUsageAssistant = (
  timestamp: number,
  totalTokens: number,
  stopReason: AssistantMessage["stopReason"] = "stop",
): AssistantMessage =>
  assistant({ timestamp, usage: usage({ input: totalTokens, totalTokens }), stopReason });
const estCases: Array<[string, Context]> = [
  ["upstream-reserve", { messages: [estUsageAssistant(100, 2000), user("x".repeat(3500), 200)] }],
  [
    "upstream-stale",
    {
      systemPrompt: "system",
      messages: [user("summary", 200), estUsageAssistant(100, 9500), user("x".repeat(4000), 300)],
    },
  ],
  [
    "upstream-responded",
    {
      messages: [
        user("summary", 200),
        estUsageAssistant(100, 9500),
        user("new prompt", 300),
        estUsageAssistant(400, 2000),
        user("tail", 500),
      ],
    },
  ],
  [
    "no-assistant",
    {
      messages: [
        user("a"),
        {
          role: "toolResult",
          toolCallId: "1",
          toolName: "t",
          content: [{ type: "text", text: "r" }],
          isError: false,
          timestamp: 2,
        },
      ],
    },
  ],
  [
    "aborted-skipped",
    {
      messages: [
        estUsageAssistant(1, 500, "aborted"),
        estUsageAssistant(2, 900, "error"),
        user("x".repeat(7), 3),
      ],
    },
  ],
  ["zero-usage-skipped", { messages: [estUsageAssistant(1, 0), user("x".repeat(14), 2)] }],
];
for (const [name, context] of estCases) {
  push(
    estimateCases,
    `context-${name}`,
    "estimateContextTokens",
    context,
    estimateContextTokens(normalizeContext(context)),
  );
}

// ---------------- AI-OVERFLOW ----------------
const overflowCases: Case[] = [];
const errMsg = (
  errorMessage: string,
  provider = "ollama",
  stopReason: AssistantMessage["stopReason"] = "error",
): AssistantMessage => assistant({ errorMessage, provider, stopReason });
const ofCases: Array<[string, AssistantMessage, number | undefined]> = [
  [
    "copilot-413",
    errMsg('github-copilot API error (413): {"message":"failed to parse request","code":""}'),
    1000000,
  ],
  [
    "copilot-413-nocw",
    errMsg(
      'github-copilot API error (413): {"message":"failed to parse request","code":""}',
      "github-copilot",
    ),
    undefined,
  ],
  ["provider-413", errMsg("custom-provider API error (413): Payload Too Large"), undefined],
  ["bare-413", errMsg('413 {"error":{"message":"failed to parse request","code":""}}'), undefined],
  ["413-eos", errMsg("413"), undefined],
  [
    "400-not-413",
    errMsg('github-copilot API error (400): {"message":"failed to parse request","code":""}'),
    undefined,
  ],
  ["429-too-many", errMsg("custom-provider API error (429): Too many requests"), undefined],
  [
    "413-in-body",
    errMsg(
      'github-copilot API error (400): {"message":"failed to parse request: invalid value 413"}',
    ),
    undefined,
  ],
  [
    "413-mid",
    errMsg(
      "custom-provider API error (400): upstream mentioned OpenAI API error (413): invalid input",
    ),
    undefined,
  ],
  ["4130", errMsg("custom-provider API error (4130): invalid input"), undefined],
  ["4130-bare", errMsg("4130 failed to parse request"), undefined],
  [
    "413-stop-not-error",
    errMsg('OpenAI API error (413): {"message":"failed to parse request"}', "openai", "stop"),
    undefined,
  ],
  [
    "413-aborted",
    errMsg('OpenAI API error (413): {"message":"failed to parse request"}', "openai", "aborted"),
    undefined,
  ],
  [
    "ollama-long",
    errMsg("400 `prompt too long; exceeded max context length by 100918 tokens`"),
    32768,
  ],
  ["zai-1261", errMsg('400 {"code":"1261","message":"Prompt too long"}', "zai"), 1048576],
  ["zai-cn", errMsg('400 {"code":"1261","message":"Prompt exceeds max length"}', "zai"), 1048576],
  [
    "together",
    errMsg(
      "400 The input (516368 tokens) is longer than the model's context length (262144 tokens).",
    ),
    262144,
  ],
  [
    "litellm",
    errMsg(
      "Error: 503 litellm.ServiceUnavailableError: litellm.MidStreamFallbackError: litellm.APIConnectionError: APIConnectionError: OpenAIException - Requested token count exceeds the model's maximum context length of 131072 tokens.",
    ),
    131072,
  ],
  [
    "litellm-precall",
    errMsg(
      String.raw`400: {"message":"litellm.ContextWindowExceededError: litellm.BadRequestError: litellm._pre_call_checks: Context Window exceeded for given call. No models have context window large enough for this call.\nModel=Qwen3.8-Flash-Next, Max Input Tokens=262144, Got=265262\nmodel=Qwen3.8-Flash-Next. context_window_fallbacks=None. fallbacks=[]."}`,
    ),
    262144,
  ],
  [
    "paren-context",
    errMsg("Error: 400 Input length (265330) exceeds model's maximum context length (262144)."),
    262144,
  ],
  [
    "poolside",
    errMsg(
      "Provider returned error: Input length 131393 exceeds the maximum allowed input length of 131040 tokens.",
    ),
    131072,
  ],
  [
    "ds4",
    errMsg("400 Prompt has 256468 tokens, but the configured context size is 256000 tokens"),
    256000,
  ],
  [
    "ds4-comma",
    errMsg("Prompt has 5,958,968 tokens, but the configured context size is 256,000 tokens"),
    256000,
  ],
  ["ollama-not", errMsg("500 `model runner crashed unexpectedly`"), 32768],
  ["cerebras-400", errMsg("400 status code (no body)", "cerebras"), 131072],
  ["cerebras-other", errMsg("400 status code (no body)", "opencode-go"), 1000000],
  [
    "bedrock-throttle",
    errMsg("Throttling error: Too many tokens, please wait before trying again."),
    200000,
  ],
  [
    "bedrock-unavail",
    errMsg("Service unavailable: The service is temporarily unavailable."),
    200000,
  ],
  ["rate-limit", errMsg("Rate limit exceeded, please retry after 30 seconds."), 200000],
  ["too-many-req", errMsg("Too many requests. Please slow down."), 200000],
  [
    "zai-silent",
    assistant({
      provider: "zai",
      usage: usage({ input: 1048600, cacheRead: 100 }),
      stopReason: "stop",
    }),
    1048576,
  ],
  [
    "zai-silent-under",
    assistant({ provider: "zai", usage: usage({ input: 100, cacheRead: 10 }), stopReason: "stop" }),
    1048576,
  ],
  [
    "xiaomi-length",
    assistant({
      provider: "xiaomi",
      model: "mimo",
      usage: usage({ input: 58, cacheRead: 1048512, output: 0 }),
      stopReason: "length",
    }),
    1048576,
  ],
  [
    "length-normal",
    assistant({ usage: usage({ input: 1000, output: 4096 }), stopReason: "length" }),
    200000,
  ],
  [
    "length-zero-under",
    assistant({ usage: usage({ input: 100, output: 0 }), stopReason: "length" }),
    200000,
  ],
];
for (const [name, msg, cw] of ofCases) {
  push(
    overflowCases,
    `overflow-${name}`,
    "isContextOverflow",
    [msg, cw ?? null],
    isContextOverflow(msg, cw),
  );
}
const rlCases: Array<[string, AssistantMessage, number]> = [
  [
    "recoverable-below",
    assistant({
      usage: usage({ input: 3, cacheRead: 253584, cacheWrite: 25554, output: 16 }),
      stopReason: "length",
    }),
    128000,
  ],
  [
    "recoverable-at-limit",
    assistant({ usage: usage({ input: 4062, output: 1024 }), stopReason: "length" }),
    1024,
  ],
  [
    "recoverable-zero",
    assistant({ usage: usage({ input: 100, output: 0 }), stopReason: "length" }),
    128000,
  ],
  [
    "recoverable-stop",
    assistant({ usage: usage({ input: 10, output: 5 }), stopReason: "stop" }),
    100,
  ],
  [
    "recoverable-zero-desired",
    assistant({ usage: usage({ input: 100, output: 0 }), stopReason: "length" }),
    0,
  ],
];
for (const [name, msg, desired] of rlCases) {
  push(
    overflowCases,
    `length-${name}`,
    "isRecoverableLength",
    [msg, desired],
    isRecoverableLength(msg, desired),
  );
}

// ---------------- AI-RETRY-POLICY ----------------
const retryCases: Case[] = [];
const retryable: Array<[string, string]> = [
  [
    "openai-explicit",
    "An error occurred while processing your request. You can retry your request, or contact us through our help center at help.openai.com if the error persists. Please include the request ID req_******** in your message.",
  ],
  [
    "bedrock-explicit",
    '{"message":"The system encountered an unexpected error during processing. Try your request again."}',
  ],
  ["nvidia-nim", "ResourceExhausted: Worker local total request limit reached (288/48)"],
  [
    "bun-socket",
    "The socket connection was closed unexpectedly. For more information, pass `verbose: true` in the second argument to fetch()",
  ],
  ["upstream-buffer", "Error: exceeded request buffer limit while retrying upstream"],
  [
    "dns-wrapped",
    "The pending stream has been canceled (caused by: getaddrinfo ENOTFOUND bedrock-runtime.us-east-1.amazonaws.com)",
  ],
  ["dns-connect", "connect ENOTFOUND api.example.com"],
  ["dns-eai", "EAI_AGAIN api.example.com"],
  ["dns-lookup", "getaddrinfo failed for api.example.com"],
  ["pending-cancel", "The pending stream has been canceled"],
  ["pending-cancel-cause", "The pending stream has been canceled (caused by: socket closed)"],
  ["svc-unavail-10965", "The service is temporarily unavailable. Please retry later."],
  ["svc-unavail-bedrock", "Service unavailable: The service is temporarily unavailable."],
  ["node-fetch-reason", "request to https://oauth2.googleapis.com/token failed, reason:"],
  [
    "node-fetch-conn",
    "request to https://oauth2.googleapis.com/token failed, reason: connect ECONNREFUSED 127.0.0.1:9",
  ],
  ["forward-upstream", "unknown: Unable to forward upstream stream"],
  ["upstream-req-fail", "upstream_error: Upstream request failed"],
  ["upstream-read", "upstream_stream_read_error: Upstream response stream was interrupted"],
  ["http2-stream", "Upstream HTTP/2 stream failed"],
  ["early-eof", "OpenAI Responses stream ended before a terminal response event"],
  [
    "azure-peak",
    "The system is currently experiencing high demand and cannot process your request. Your request exceeds the maximum usage size allowed during peak load. For improved capacity reliability, consider switching to Provisioned Throughput.",
  ],
  ["sub-sharing-usage", "subscription_sharing_usage_unavailable: Usage cannot be checked."],
  ["sub-sharing-user", "subscription_sharing_user_unavailable: User cannot be loaded."],
  ["overloaded", "overloaded_error"],
  ["520-bodyless", "520 status code (no body)"],
  ["524-bodyless", "524 status code (no body)"],
  ["429-bare", "HTTP 429 too many requests"],
];
const notRetryable: Array<[string, string]> = [
  ["quota-429", "429 quota exceeded"],
  [
    "chatgpt-limit",
    'OpenAI API error (429): {"code":"subscription_sharing_usage_limit_exceeded","message":"Usage limit reached."}',
  ],
  [
    "chatgpt-workspace",
    'OpenAI API error (429): {"code":"subscription_sharing_workplace_individual_usage_limit_exceeded","message":"Usage limit reached."}',
  ],
  ["zai-1308", 'OpenAI API error (429): {"code":"1308","message":"Usage limit reached."}'],
  ["go-usage-limit", '{"type":"error","error":{"type":"GoUsageLimitError","message":"x"}}'],
  ["free-usage-limit", '{"type":"error","error":{"type":"FreeUsageLimitError","message":"x"}}'],
  ["monthly-limit", "Monthly usage limit reached. Please enable available balance"],
  ["insufficient-quota", "insufficient_quota"],
  ["out-of-budget", "out of budget"],
  ["billing", "billing required"],
  [
    "401-id",
    '401 {"type":"error","error":{"type":"authentication_error","message":"invalid x-api-key"},"request_id":"req_011CU503xyz"}',
  ],
  ["img-bytes", "400 image exceeds 5 MB maximum: 5316784 bytes > 5242880 bytes"],
  ["zai-1308-longer", '{"code":"13080","message":"x"}'],
];
for (const [name, errorMessage] of retryable) {
  const msg = errMsg(errorMessage);
  push(
    retryCases,
    `retryable-${name}`,
    "isRetryableAssistantError",
    msg,
    isRetryableAssistantError(msg),
  );
}
for (const [name, errorMessage] of notRetryable) {
  const msg = errMsg(errorMessage);
  push(
    retryCases,
    `not-retryable-${name}`,
    "isRetryableAssistantError",
    msg,
    isRetryableAssistantError(msg),
  );
}
push(
  retryCases,
  "not-error-stop",
  "isRetryableAssistantError",
  assistant({ stopReason: "stop" }),
  isRetryableAssistantError(assistant({ stopReason: "stop" })),
);
push(
  retryCases,
  "error-no-message",
  "isRetryableAssistantError",
  assistant({ stopReason: "error" }),
  isRetryableAssistantError(assistant({ stopReason: "error" })),
);
for (const [name, policy, attempt] of [
  ["cap-default", { baseDelayMs: 2000 }, 6],
  ["cap-custom", { baseDelayMs: 2000, maxAgentDelayMs: 5000 }, 5],
  ["cap-zero", { baseDelayMs: 2000, maxAgentDelayMs: 0 }, 5],
  ["attempt-1", { baseDelayMs: 2000 }, 1],
  ["attempt-2", { baseDelayMs: 2000 }, 2],
  ["attempt-3", { baseDelayMs: 2000 }, 3],
  ["attempt-0", { baseDelayMs: 2000 }, 0],
  ["huge-base", { baseDelayMs: 9_007_199_254_740_000 }, 60],
] as const) {
  push(
    retryCases,
    `delay-${name}`,
    "retryDelayMs",
    [policy, attempt],
    retryDelayMs(policy, attempt),
  );
}

// ---------------- AI-COST ----------------
const costCases: Case[] = [];
const costModel = (
  cost: Model<"openai-completions">["cost"],
  extra: Partial<Model<"openai-completions">> = {},
): Model<"openai-completions"> => ({ ...modelBase, cost, ...extra });
for (const [name, model, u] of [
  [
    "base",
    costModel({ input: 1, output: 5, cacheRead: 0.2, cacheWrite: 1.25 }),
    usage({ input: 1000, output: 500, cacheRead: 100, cacheWrite: 50 }),
  ],
  [
    "tier-below",
    costModel({
      input: 1,
      output: 5,
      cacheRead: 0.2,
      cacheWrite: 1.25,
      tiers: [{ inputTokensAbove: 1000, input: 2, output: 10, cacheRead: 0.5, cacheWrite: 2.5 }],
    }),
    usage({ input: 500, cacheRead: 200, cacheWrite: 300 }),
  ],
  [
    "tier-at-boundary",
    costModel({
      input: 1,
      output: 5,
      cacheRead: 0.2,
      cacheWrite: 1.25,
      tiers: [{ inputTokensAbove: 1000, input: 2, output: 10, cacheRead: 0.5, cacheWrite: 2.5 }],
    }),
    usage({ input: 600, cacheRead: 400 }),
  ],
  [
    "tier-over",
    costModel({
      input: 1,
      output: 5,
      cacheRead: 0.2,
      cacheWrite: 1.25,
      tiers: [{ inputTokensAbove: 1000, input: 2, output: 10, cacheRead: 0.5, cacheWrite: 2.5 }],
    }),
    usage({ input: 1001 }),
  ],
  [
    "tier-highest-wins",
    costModel({
      input: 1,
      output: 5,
      cacheRead: 0,
      cacheWrite: 0,
      tiers: [
        { inputTokensAbove: 500, input: 2, output: 3, cacheRead: 1, cacheWrite: 1 },
        { inputTokensAbove: 100, input: 9, output: 9, cacheRead: 9, cacheWrite: 9 },
      ],
    }),
    usage({ input: 600 }),
  ],
  [
    "cache-1h",
    costModel({ input: 1, output: 5, cacheRead: 0.2, cacheWrite: 1.25 }),
    usage({ input: 100, output: 10, cacheWrite: 80, cacheWrite1h: 30 }),
  ],
  [
    "zero-rates",
    costModel({ input: 0, output: 0, cacheRead: 0, cacheWrite: 0 }),
    usage({ input: 999 }),
  ],
] as const) {
  push(costCases, `cost-${name}`, "calculateCost", [model, u], calculateCost(model, u));
}
const thinkingModels: Array<[string, Model<"openai-completions">]> = [
  ["no-reasoning", { ...modelBase, reasoning: false }],
  ["all-levels", modelBase],
  ["xhigh-null", { ...modelBase, thinkingLevelMap: { xhigh: null } }],
  ["xhigh-mapped", { ...modelBase, thinkingLevelMap: { xhigh: "e", max: "m" } }],
  ["off-null", { ...modelBase, thinkingLevelMap: { off: null } }],
];
for (const [name, model] of thinkingModels) {
  push(
    costCases,
    `supported-${name}`,
    "getSupportedThinkingLevels",
    model,
    getSupportedThinkingLevels(model),
  );
}
const clampCases: Array<[string, Model<"openai-completions">, ModelThinkingLevel]> = [
  ["clamp-supported", modelBase, "low"],
  ["clamp-unsupported-xhigh", { ...modelBase, thinkingLevelMap: { xhigh: null } }, "xhigh"],
  ["clamp-max-only-high", { ...modelBase, thinkingLevelMap: { xhigh: null, max: null } }, "max"],
  ["clamp-higher", { ...modelBase, thinkingLevelMap: { minimal: null, low: null } }, "minimal"],
  ["clamp-no-reasoning", { ...modelBase, reasoning: false }, "high"],
  ["clamp-off-null", { ...modelBase, thinkingLevelMap: { off: null } }, "off"],
];
for (const [name, model, level] of clampCases) {
  push(
    costCases,
    `clamp-${name}`,
    "clampThinkingLevel",
    [model, level],
    clampThinkingLevel(model, level),
  );
}

// ---------------- AI-SIMPLE-OPTIONS ----------------
const simpleCases: Case[] = [];
const ctxArg = (messages: Message[]) => normalizeContext({ messages });
push(
  simpleCases,
  "clamp-reserve",
  "clampMaxTokensToContext",
  [modelBase, { messages: [estUsageAssistant(100, 2000), user("x".repeat(3500), 200)] }, 8000],
  clampMaxTokensToContext(
    modelBase,
    ctxArg([estUsageAssistant(100, 2000), user("x".repeat(3500), 200)]),
    8000,
  ),
);
const staleContext = {
  systemPrompt: "system",
  messages: [user("summary", 200), estUsageAssistant(100, 9500), user("x".repeat(4000), 300)],
};
push(
  simpleCases,
  "clamp-stale",
  "clampMaxTokensToContext",
  [modelBase, staleContext, 8000],
  clampMaxTokensToContext(modelBase, normalizeContext(staleContext), 8000),
);
push(
  simpleCases,
  "clamp-explicit",
  "clampMaxTokensToContext",
  [modelBase, { messages: [user("hi")] }, 100],
  clampMaxTokensToContext(modelBase, ctxArg([user("hi")]), 100),
);
push(
  simpleCases,
  "clamp-zero-window",
  "clampMaxTokensToContext",
  [{ ...modelBase, contextWindow: 0 }, { messages: [user("hi")] }, 500],
  clampMaxTokensToContext({ ...modelBase, contextWindow: 0 }, ctxArg([user("hi")]), 500),
);
push(
  simpleCases,
  "clamp-underflow",
  "clampMaxTokensToContext",
  [modelBase, { messages: [user("x".repeat(100000))] }, 500],
  clampMaxTokensToContext(modelBase, ctxArg([user("x".repeat(100000))]), 500),
);
for (const [name, model, level, req] of [
  ["merge-none", modelBase, "off", undefined],
  ["merge-model-only", { ...modelBase, samplingParams: { a: 1 } }, "off", undefined],
  [
    "merge-level",
    { ...modelBase, samplingParamsByThinkingLevel: { high: { b: 2 } } },
    "high",
    undefined,
  ],
  [
    "merge-order",
    {
      ...modelBase,
      samplingParams: { a: 1, k: "m" },
      samplingParamsByThinkingLevel: { low: { k: "l", b: 2 } },
    },
    "low",
    { k: "r", c: 3 },
  ],
  ["merge-request-only", modelBase, "off", { q: 1 }],
  ["merge-empty-map", { ...modelBase, samplingParams: {} }, "off", undefined],
  [
    "merge-clamped-level",
    {
      ...modelBase,
      thinkingLevelMap: { xhigh: null },
      samplingParamsByThinkingLevel: { high: { h: 1 }, xhigh: { x: 2 } },
    },
    "xhigh",
    undefined,
  ],
] as const) {
  push(
    simpleCases,
    `sampling-${name}`,
    "resolveSamplingParams",
    [model, level, req ?? null],
    resolveSamplingParams(model, level, req),
  );
}
for (const [name, effort] of [
  ["off", undefined],
  ["xhigh", "xhigh"],
  ["max", "max"],
  ["low", "low"],
] as const) {
  push(simpleCases, `reasoning-${name}`, "clampReasoning", effort ?? null, clampReasoning(effort));
}
for (const [name, level, custom] of [
  ["minimal", "minimal", undefined],
  ["low", "low", undefined],
  ["medium", "medium", undefined],
  ["high", "high", undefined],
  ["xhigh-to-high", "xhigh", undefined],
  ["max-to-high", "max", undefined],
  ["custom", "low", { low: 4096 }],
  ["custom-partial", "high", { minimal: 512 }],
] as const) {
  push(
    simpleCases,
    `budget-${name}`,
    "thinkingBudgetForLevel",
    [level, custom ?? null],
    thinkingBudgetForLevel(level, custom),
  );
}
for (const [name, budget, ceiling] of [
  ["fits", 100, 2000],
  ["clamped", 2000, 2000],
  ["tiny-ceiling", 500, 100],
] as const) {
  push(
    simpleCases,
    `room-${name}`,
    "clampThinkingBudgetToAnswerRoom",
    [budget, ceiling],
    clampThinkingBudgetToAnswerRoom(budget, ceiling),
  );
}
for (const [name, base, modelMax, level, custom] of [
  ["no-cap", undefined, 8000, "low", undefined],
  ["explicit", 1000, 8000, "medium", undefined],
  ["explicit-over", 7000, 8000, "high", undefined],
  ["budget-over-max", undefined, 2000, "high", undefined],
  ["custom", 1000, 8000, "low", { low: 3000 }],
] as const) {
  push(
    simpleCases,
    `adjust-${name}`,
    "adjustMaxTokensForThinking",
    [base ?? null, modelMax, level, custom ?? null],
    adjustMaxTokensForThinking(base, modelMax, level, custom),
  );
}

type HashModule = {
  shortHash: (s: string) => string;
};
type PromptCacheModule = {
  clampOpenAIPromptCacheKey: (key: string | undefined) => string | undefined;
};
function isHashModule(value: unknown): value is HashModule {
  return (
    typeof value === "object" &&
    value !== null &&
    "shortHash" in value &&
    typeof value.shortHash === "function"
  );
}
function isPromptCacheModule(value: unknown): value is PromptCacheModule {
  return (
    typeof value === "object" &&
    value !== null &&
    "clampOpenAIPromptCacheKey" in value &&
    typeof value.clampOpenAIPromptCacheKey === "function"
  );
}
const hashModule: unknown = await import(import.meta.resolve("@earendil-works/pi-ai/utils/hash"));
const promptCacheModule: unknown = await import(
  import.meta.resolve("@earendil-works/pi-ai/api/openai-prompt-cache")
);
if (!isHashModule(hashModule) || !isPromptCacheModule(promptCacheModule)) {
  throw new Error("hash/prompt-cache dist modules lack the expected exports");
}

const hashCases: Case[] = [];
const HASH_INPUTS = [
  "",
  "a",
  "call|item",
  "call_xyz|item_abc123",
  `call_${"x".repeat(50)}|item_${"y".repeat(50)}`,
  `call|${"i".repeat(80)}`,
  "sk-ant-oat-1234567890",
  "\u4F60\u597D\u4E16\u754C",
  `emoji ${String.fromCodePoint(0x1f600)} test`,
  "a+b=c/d",
  "the quick brown fox jumps over the lazy dog",
];
for (const input of HASH_INPUTS) {
  push(
    hashCases,
    `hash-${input.slice(0, 20) || "empty"}`,
    "shortHash",
    input,
    hashModule.shortHash(input),
  );
}
const PROMPT_CACHE_INPUTS: Array<string | undefined> = [
  undefined,
  "",
  "key",
  "k".repeat(64),
  "k".repeat(65),
  "k".repeat(100),
  `${"e".repeat(63)}${String.fromCodePoint(0x1f600)}`,
  `${String.fromCodePoint(0x1f600)}${"x".repeat(63)}`,
];
for (const input of PROMPT_CACHE_INPUTS) {
  push(
    hashCases,
    `prompt-cache-${input === undefined ? "undefined" : String(input.length)}`,
    "clampPromptCacheKey",
    input,
    promptCacheModule.clampOpenAIPromptCacheKey(input),
  );
}

type OpenRouterCostModule = {
  applyOpenRouterReportedCost: (usage: Usage, reported: unknown) => void;
  isOpenRouterModel: (model: { provider: string; baseUrl: string }) => boolean;
};
function isOpenRouterCostModule(value: unknown): value is OpenRouterCostModule {
  return (
    typeof value === "object" &&
    value !== null &&
    "applyOpenRouterReportedCost" in value &&
    typeof value.applyOpenRouterReportedCost === "function" &&
    "isOpenRouterModel" in value &&
    typeof value.isOpenRouterModel === "function"
  );
}
const openRouterCostModule: unknown = await import(
  import.meta.resolve("@earendil-works/pi-ai/utils/openrouter-cost")
);
if (!isOpenRouterCostModule(openRouterCostModule)) {
  throw new Error("openrouter-cost dist module lacks the expected exports");
}

const openRouterCases: Case[] = [];
const catalogUsage = (): Usage =>
  usage({
    input: 100,
    output: 20,
    totalTokens: 120,
    cost: { input: 0.0003, output: 0.0003, cacheRead: 0, cacheWrite: 0, total: 0.0006 },
  });
const byok = (details: unknown) => ({ cost: 0.001, is_byok: true, cost_details: details });
const parts = (prompt: unknown, completions: unknown) => ({
  upstream_inference_cost: null,
  upstream_inference_prompt_cost: prompt,
  upstream_inference_completions_cost: completions,
});
for (const [name, reported] of [
  ["cost-null", { cost: null, is_byok: false }],
  ["cost-negative", { cost: -1, is_byok: false }],
  ["cost-string", { cost: "0.1", is_byok: false }],
  ["cost-missing", { is_byok: false }],
  ["cost-zero", { cost: 0, is_byok: false }],
  ["byok-false", { cost: 0.0042, is_byok: false, cost_details: null }],
  ["byok-missing", { cost: 0.0042 }],
  ["byok-string", { cost: 0.0042, is_byok: "false" }],
  ["byok-details-null", byok(null)],
  ["byok-details-missing", { cost: 0.001, is_byok: true }],
  ["byok-upstream-number", byok({ upstream_inference_cost: 0.004 })],
  ["byok-upstream-zero", byok({ upstream_inference_cost: 0 })],
  ["byok-upstream-negative", byok({ upstream_inference_cost: -0.004 })],
  [
    "byok-upstream-missing",
    byok({ upstream_inference_prompt_cost: 0.002, upstream_inference_completions_cost: 0.003 }),
  ],
  ["byok-upstream-null-parts", byok(parts(0.002, 0.003))],
  [
    "byok-upstream-null-part-missing",
    byok({ upstream_inference_cost: null, upstream_inference_prompt_cost: 0.002 }),
  ],
  ["byok-upstream-null-part-negative", byok(parts(0.002, -0.003))],
] as const) {
  const target = catalogUsage();
  openRouterCostModule.applyOpenRouterReportedCost(target, reported);
  push(
    openRouterCases,
    name,
    "applyOpenRouterReportedCost",
    { usage: catalogUsage(), reported },
    target,
  );
}
for (const [name, provider, baseUrl] of [
  ["provider-openrouter", "openrouter", "https://example.com/v1"],
  ["url-openrouter", "custom", "https://openrouter.ai/api/v1"],
  ["neither", "openai", "https://api.openai.com/v1"],
] as const) {
  push(
    openRouterCases,
    name,
    "isOpenRouterModel",
    { provider, baseUrl },
    openRouterCostModule.isOpenRouterModel({ provider, baseUrl }),
  );
}

const generator = {
  script: "scripts/parity/generate-ai-utils.ts",
  node: process.version,
  upstream: "@earendil-works/pi-ai@1.5.0",
};
for (const [file, cases] of [
  ["vectors/ai/estimate.json", estimateCases],
  ["vectors/ai/overflow.json", overflowCases],
  ["vectors/ai/retry.json", retryCases],
  ["vectors/ai/cost.json", costCases],
  ["vectors/ai/simple-options.json", simpleCases],
  ["vectors/ai/hash.json", hashCases],
  ["vectors/ai/openrouter-cost.json", openRouterCases],
] as const) {
  writeFileSync(file, `${JSON.stringify({ generator, cases }, null, 2)}\n`);
  console.log(`${file}: ${cases.length} cases`);
}
