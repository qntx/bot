/// <reference types="node" />
import { writeFileSync } from "node:fs";

import {
  collapseSystemMessages,
  contentText,
  createInitialSystemMessage,
  declarationsEqual,
  formatUserAgent,
  getAttributionHeaders,
  getClientVersion,
  getCurrentSystemMessage,
  getCurrentSystemPrompt,
  getCurrentTools,
  getDeclaredTools,
  getIdentityHeaders,
  getInitialSystemMessage,
  getOpenCodeSessionHeaders,
  getSystemMessageText,
  getToolStateChanges,
  hasNonAdditiveToolChanges,
  mergeClientIdentity,
  normalizeContext,
  renderSystemMessageUpdate,
  resolveTranscript,
  resolveTranscriptTools,
  toToolDeclaration,
  withoutInitialSystemMessage,
} from "@earendil-works/pi-ai";
import type {
  AssistantMessage,
  Context,
  ImageContent,
  Message,
  Model,
  SystemMessage,
  TextContent,
  ThinkingContent,
  Tool,
  ToolCall,
  ToolResultMessage,
} from "@earendil-works/pi-ai";

// Not exported from the npm package; re-exported from the pinned
// checkout through a .js shim so typecheck never enters 3rdparty.
import {
  ownedSlice,
  sanitizeSurrogates,
  splitErrorToolResultImages,
  transformMessages,
} from "./pi-internal.js";

type AssistantBlock = TextContent | ThinkingContent | ToolCall;

type Case = { name: string; kind: string; args: unknown; result: unknown; normalize?: boolean };

const tool = (name: string, description = `${name} tool`): Tool => ({
  name,
  description,
  parameters: { type: "object", properties: {} },
});

const sys = (m: Partial<SystemMessage>): SystemMessage => ({
  role: "system",
  content: "",
  timestamp: 0,
  ...m,
});
const user = (text: string): Message => ({ role: "user", content: text, timestamp: 1 });
const assistant = (
  model: string,
  content: AssistantBlock[],
  stopReason: "stop" | "error" | "aborted" | "toolUse" = "stop",
): AssistantMessage => ({
  role: "assistant",
  content,
  api: "openai-completions",
  provider: "sference",
  model,
  usage: {
    input: 0,
    output: 0,
    cacheRead: 0,
    cacheWrite: 0,
    totalTokens: 0,
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
  },
  stopReason,
  timestamp: 1,
});

const modelBase: Model<"openai-completions"> = {
  id: "m",
  name: "M",
  api: "openai-completions",
  provider: "sference",
  baseUrl: "https://api.sference.com/v1",
  reasoning: true,
  input: ["text"],
  cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
  contextWindow: 100000,
  maxTokens: 1000,
};
const visionModel: Model<"openai-completions"> = { ...modelBase, input: ["text", "image"] };

const thinking = (t: string, extra: Partial<ThinkingContent> = {}): ThinkingContent => ({
  type: "thinking",
  thinking: t,
  ...extra,
});
const text = (t: string): TextContent => ({ type: "text", text: t });
const call = (
  id: string,
  name: string,
  args: ToolCall["arguments"],
  extra: Partial<ToolCall> = {},
): ToolCall => ({ type: "toolCall", id, name, arguments: args, ...extra });
const image = (): ImageContent => ({ type: "image", data: "AAAA", mimeType: "image/png" });
const toolResult = (
  id: string,
  name: string,
  content: ToolResultMessage["content"],
  isError = false,
): ToolResultMessage => ({
  role: "toolResult",
  toolCallId: id,
  toolName: name,
  content,
  isError,
  timestamp: 1,
});

const transcriptCases: Case[] = [];
const push = (cases: Case[], name: string, kind: string, args: unknown, result: unknown) =>
  cases.push({ name, kind, args, result });

// --- AI-TRANSCRIPT ---
const normalizeArgs: Context = {
  messages: [user("hi")],
  systemPrompt: "sys",
  tools: [tool("a")],
};
push(
  transcriptCases,
  "normalize-empty",
  "normalizeContext",
  { messages: [] },
  normalizeContext({ messages: [] }),
);
push(
  transcriptCases,
  "normalize-prompt-tools",
  "normalizeContext",
  normalizeArgs,
  normalizeContext(normalizeArgs),
);
push(
  transcriptCases,
  "create-initial-none",
  "createInitialSystemMessage",
  [undefined, undefined],
  createInitialSystemMessage(undefined, undefined),
);
push(
  transcriptCases,
  "create-initial-prompt",
  "createInitialSystemMessage",
  ["p", [tool("a")]],
  createInitialSystemMessage("p", [tool("a")]),
);
const deltaMessages: Message[] = [
  sys({ content: "base", toolsAdded: [tool("a"), tool("b"), tool("drop")], timestamp: 0 }),
  user("u1"),
  sys({ content: "", toolsRemoved: [{ name: "drop" }], timestamp: 2 }),
  sys({
    content: "extra",
    sections: { env: "prod", note: "n" },
    toolsAdded: [tool("late")],
    timestamp: 3,
  }),
  sys({ sections: { env: null, extra: "x" }, timestamp: 4 }),
];
push(
  transcriptCases,
  "current-tools",
  "getCurrentTools",
  deltaMessages,
  getCurrentTools(deltaMessages),
);
push(
  transcriptCases,
  "current-system-message",
  "getCurrentSystemMessage",
  deltaMessages,
  getCurrentSystemMessage(deltaMessages),
);
push(
  transcriptCases,
  "current-system-prompt",
  "getCurrentSystemPrompt",
  deltaMessages,
  getCurrentSystemPrompt(deltaMessages),
);
push(
  transcriptCases,
  "initial-system-message",
  "getInitialSystemMessage",
  deltaMessages,
  getInitialSystemMessage(deltaMessages),
);
push(
  transcriptCases,
  "initial-none",
  "getInitialSystemMessage",
  [user("u")],
  getInitialSystemMessage([user("u")]),
);
push(
  transcriptCases,
  "without-initial",
  "withoutInitialSystemMessage",
  deltaMessages,
  withoutInitialSystemMessage(deltaMessages),
);
push(
  transcriptCases,
  "collapse",
  "collapseSystemMessages",
  deltaMessages,
  collapseSystemMessages(normalizeContext({ messages: deltaMessages })),
);
push(
  transcriptCases,
  "resolve-mid-true",
  "resolveTranscript",
  [deltaMessages, true],
  resolveTranscript(normalizeContext({ messages: deltaMessages }), true),
);
push(
  transcriptCases,
  "resolve-mid-false",
  "resolveTranscript",
  [deltaMessages, false],
  resolveTranscript(normalizeContext({ messages: deltaMessages }), false),
);
push(
  transcriptCases,
  "tool-declaration",
  "toToolDeclaration",
  tool("x"),
  toToolDeclaration(tool("x")),
);
push(
  transcriptCases,
  "declarations-equal",
  "declarationsEqual",
  [tool("x"), tool("x")],
  declarationsEqual(tool("x"), tool("x")),
);
push(
  transcriptCases,
  "declarations-unequal",
  "declarationsEqual",
  [tool("x"), tool("x", "y")],
  declarationsEqual(tool("x"), tool("x", "y")),
);
push(
  transcriptCases,
  "tool-state-changes",
  "getToolStateChanges",
  [
    [tool("a"), tool("b"), tool("c")],
    [tool("b"), tool("c", "c2"), tool("d")],
  ],
  getToolStateChanges([tool("a"), tool("b"), tool("c")], [tool("b"), tool("c", "c2"), tool("d")]),
);
push(
  transcriptCases,
  "declared-tools",
  "getDeclaredTools",
  deltaMessages,
  getDeclaredTools(deltaMessages),
);
push(
  transcriptCases,
  "non-additive-yes",
  "hasNonAdditiveToolChanges",
  deltaMessages,
  hasNonAdditiveToolChanges(deltaMessages),
);
push(
  transcriptCases,
  "non-additive-no",
  "hasNonAdditiveToolChanges",
  [sys({ toolsAdded: [tool("a")] })],
  hasNonAdditiveToolChanges([sys({ toolsAdded: [tool("a")] })]),
);
const redefine: Message[] = [
  sys({ toolsAdded: [tool("r", "v1")] }),
  sys({ toolsAdded: [tool("r", "v2")] }),
];
for (const support of ["none", "additions", "changes"] as const) {
  push(
    transcriptCases,
    `resolve-tools-${support}`,
    "resolveTranscriptTools",
    [deltaMessages, support],
    resolveTranscriptTools(deltaMessages, support),
  );
}
push(
  transcriptCases,
  "resolve-tools-additions-redeclare",
  "resolveTranscriptTools",
  [redefine, "additions"],
  resolveTranscriptTools(redefine, "additions"),
);

// --- AI-SANITIZE / text ---
const textCases: Case[] = [
  {
    name: "sanitize-clean",
    kind: "sanitizeSurrogates",
    args: "Hello 🙈 World",
    result: sanitizeSurrogates("Hello 🙈 World"),
  },
  {
    name: "sanitize-lone-high",
    kind: "sanitizeSurrogates",
    args: "a\uD800b",
    result: sanitizeSurrogates("a\uD800b"),
  },
  {
    name: "sanitize-lone-low",
    kind: "sanitizeSurrogates",
    args: "a\uDC00b",
    result: sanitizeSurrogates("a\uDC00b"),
  },
  {
    name: "owned-slice",
    kind: "ownedSlice",
    args: ["hello 🙈 world", 4, 9],
    result: ownedSlice("hello 🙈 world", 4, 9),
  },
  {
    name: "owned-slice-split-pair",
    kind: "ownedSlice",
    args: ["a🙈b", 1, 3],
    result: ownedSlice("a🙈b", 1, 3),
  },
  {
    name: "owned-slice-end",
    kind: "ownedSlice",
    args: ["abcdef", 2],
    result: ownedSlice("abcdef", 2),
  },
  {
    name: "trim-whitespace",
    kind: "jsTrim",
    args: " \u00A0\uFEFFx\u1680 y\u2028 ",
    result: "x\u1680 y",
  },
];
for (const [content, sep, result] of [
  [
    [thinking("t"), text("first"), call("1", "r", {}), text("second")],
    "\n",
    contentText([thinking("t"), text("first"), call("1", "r", {}), text("second")]),
  ],
  [[text("a"), image(), text("b")], "", contentText([text("a"), image(), text("b")], "")],
  ["plain string", "\n", contentText("plain string")],
] as const) {
  push(textCases, `content-text-${textCases.length}`, "contentText", [content, sep], result);
}
const sysMsg = sys({ content: "prompt", sections: { a: "A", b: null, c: "C" }, timestamp: 1 });
push(
  textCases,
  "system-message-text",
  "getSystemMessageText",
  sysMsg,
  getSystemMessageText(sysMsg),
);
push(
  textCases,
  "render-update",
  "renderSystemMessageUpdate",
  sysMsg,
  renderSystemMessageUpdate(sysMsg),
);

// --- AI-TRANSFORM-MESSAGES ---
const transformCases: Case[] = [];
push(
  transformCases,
  "cross-model-thinking-dropped",
  "transformMessages",
  [
    [
      user("q"),
      assistant("zai-org/GLM-5.3", [
        thinking("reasoning", { thinkingSignature: "sig" }),
        text("visible"),
        call("c1", "bash", { cmd: "x" }),
      ]),
      toolResult("c1", "bash", [text("r")]),
    ],
    modelBase,
  ],
  transformMessages(
    [
      user("q"),
      assistant("zai-org/GLM-5.3", [
        thinking("reasoning", { thinkingSignature: "sig" }),
        text("visible"),
        call("c1", "bash", { cmd: "x" }),
      ]),
      toolResult("c1", "bash", [text("r")]),
    ],
    modelBase,
  ),
);
push(
  transformCases,
  "same-model-thinking-kept",
  "transformMessages",
  [[assistant("m", [thinking("kept", { thinkingSignature: "sig" }), text("a")])], modelBase],
  transformMessages(
    [assistant("m", [thinking("kept", { thinkingSignature: "sig" }), text("a")])],
    modelBase,
  ),
);
push(
  transformCases,
  "empty-thinking-dropped-signed-kept",
  "transformMessages",
  [
    [
      assistant("m", [
        thinking("  "),
        thinking("", { thinkingSignature: "s" }),
        thinking("", { redacted: true }),
      ]),
    ],
    modelBase,
  ],
  transformMessages(
    [
      assistant("m", [
        thinking("  "),
        thinking("", { thinkingSignature: "s" }),
        thinking("", { redacted: true }),
      ]),
    ],
    modelBase,
  ),
);
push(
  transformCases,
  "thought-signature-removed-cross-model",
  "transformMessages",
  [[assistant("other", [call("c", "t", {}, { thoughtSignature: "tsig" })])], modelBase],
  transformMessages(
    [assistant("other", [call("c", "t", {}, { thoughtSignature: "tsig" })])],
    modelBase,
  ),
);
transformCases.push({
  name: "tool-call-id-normalized",
  kind: "transformMessages",
  args: [
    [
      assistant("other", [
        call("weird|id|with:chars-and-more-than-64-chars-to-force-normalization-xx", "bash", {}),
      ]),
      toolResult("weird|id|with:chars-and-more-than-64-chars-to-force-normalization-xx", "bash", [
        text("o"),
      ]),
    ],
    modelBase,
  ],
  normalize: true,
  result: transformMessages(
    [
      assistant("other", [
        call("weird|id|with:chars-and-more-than-64-chars-to-force-normalization-xx", "bash", {}),
      ]),
      toolResult("weird|id|with:chars-and-more-than-64-chars-to-force-normalization-xx", "bash", [
        text("o"),
      ]),
    ],
    modelBase,
    (id: string) => `norm_${id.replaceAll(/[^a-zA-Z0-9_-]/g, "").slice(0, 32)}`,
  ),
});
push(
  transformCases,
  "image-downgrade",
  "transformMessages",
  [
    [
      { role: "user", content: [text("see"), image()], timestamp: 1 },
      toolResult("c", "t", [text("r"), image()]),
    ],
    modelBase,
  ],
  transformMessages(
    [
      { role: "user", content: [text("see"), image()], timestamp: 1 },
      toolResult("c", "t", [text("r"), image()]),
    ],
    modelBase,
  ),
);
push(
  transformCases,
  "vision-keeps-images",
  "transformMessages",
  [[{ role: "user", content: [text("see"), image()], timestamp: 1 }], visionModel],
  transformMessages([{ role: "user", content: [text("see"), image()], timestamp: 1 }], visionModel),
);
push(
  transformCases,
  "orphan-call-synthetic-result",
  "transformMessages",
  [[assistant("m", [call("orphan", "t", {})]), user("next")], modelBase],
  transformMessages([assistant("m", [call("orphan", "t", {})]), user("next")], modelBase),
);
push(
  transformCases,
  "system-held-between-call-result",
  "transformMessages",
  [
    [
      assistant("m", [call("c", "t", {})]),
      sys({ content: "mid", timestamp: 2 }),
      toolResult("c", "t", [text("r")]),
    ],
    modelBase,
  ],
  transformMessages(
    [
      assistant("m", [call("c", "t", {})]),
      sys({ content: "mid", timestamp: 2 }),
      toolResult("c", "t", [text("r")]),
    ],
    modelBase,
  ),
);
push(
  transformCases,
  "error-assistant-skipped",
  "transformMessages",
  [
    [
      user("q"),
      assistant("m", [text("partial")], "error"),
      assistant("m", [text("partial2")], "aborted"),
      assistant("m", [text("ok")]),
    ],
    modelBase,
  ],
  transformMessages(
    [
      user("q"),
      assistant("m", [text("partial")], "error"),
      assistant("m", [text("partial2")], "aborted"),
      assistant("m", [text("ok")]),
    ],
    modelBase,
  ),
);
const errorResult = toolResult(
  "err",
  "img_tool",
  [text("fail"), image(), image(), text("tail")],
  true,
);
push(
  transformCases,
  "split-error-images",
  "splitErrorToolResultImages",
  errorResult,
  splitErrorToolResultImages(errorResult),
);
push(
  transformCases,
  "split-non-error-none",
  "splitErrorToolResultImages",
  toolResult("ok", "t", [text("x")]),
  splitErrorToolResultImages(toolResult("ok", "t", [text("x")])),
);

// --- AI-IDENTITY ---
const PI_ATTRIBUTION = { name: "pi", url: "https://pi.dev", categories: ["cli-agent"] };
const ACME = { name: "acme", url: "https://acme.example" };
const identityCases: Case[] = [];
const idPush = (name: string, kind: string, args: unknown, result: unknown) =>
  push(identityCases, name, kind, args, result);
idPush(
  "ua-short",
  "formatUserAgent",
  [{ userAgent: "short" }],
  formatUserAgent({ userAgent: "short" }),
);
idPush(
  "ua-custom",
  "formatUserAgent",
  [{ userAgent: { custom: "acme/1.0" } }],
  formatUserAgent({ userAgent: { custom: "acme/1.0" } }),
);
// `detailed` embeds runtime/platform strings — record the parts our port
// owns (product/version prefix; runtime segment differs by engine).
idPush(
  "ua-detailed-prefix",
  "formatUserAgentPrefix",
  [{ version: "9.9.9" }],
  `pi/${getClientVersion({ version: "9.9.9" })} (`,
);
idPush("version-default", "getClientVersion", [undefined], getClientVersion());
idPush(
  "version-override",
  "getClientVersion",
  [{ version: "2.0" }],
  getClientVersion({ version: "2.0" }),
);
idPush(
  "merge-none",
  "mergeClientIdentity",
  [undefined, undefined],
  mergeClientIdentity(undefined, undefined),
);
idPush(
  "merge-fields",
  "mergeClientIdentity",
  [
    { attribution: PI_ATTRIBUTION, userAgent: "short" },
    { attribution: undefined, version: "1.0.0" },
  ],
  mergeClientIdentity({ attribution: PI_ATTRIBUTION, userAgent: "short" }, { version: "1.0.0" }),
);
idPush(
  "merge-null-attribution",
  "mergeClientIdentity",
  [{ attribution: PI_ATTRIBUTION }, { attribution: null }],
  mergeClientIdentity({ attribution: PI_ATTRIBUTION }, { attribution: null }),
);
const openrouter = { provider: "openrouter", baseUrl: "https://openrouter.ai/api/v1" };
idPush(
  "attr-none",
  "getAttributionHeaders",
  [openrouter, undefined],
  getAttributionHeaders(openrouter),
);
idPush(
  "attr-openrouter",
  "getAttributionHeaders",
  [openrouter, { attribution: PI_ATTRIBUTION }],
  getAttributionHeaders(openrouter, { attribution: PI_ATTRIBUTION }),
);
idPush(
  "attr-openrouter-custom",
  "getAttributionHeaders",
  [{ provider: "custom", baseUrl: "not-a-url-openrouter.ai" }, { attribution: PI_ATTRIBUTION }],
  getAttributionHeaders(
    { provider: "custom", baseUrl: "not-a-url-openrouter.ai" },
    { attribution: PI_ATTRIBUTION },
  ),
);
idPush(
  "attr-openrouter-no-categories",
  "getAttributionHeaders",
  [openrouter, { attribution: { ...ACME, categories: [] } }],
  getAttributionHeaders(openrouter, { attribution: { ...ACME, categories: [] } }),
);
idPush(
  "attr-nvidia",
  "getAttributionHeaders",
  [{ provider: "nvidia", baseUrl: "https://example.test/v1" }, { attribution: PI_ATTRIBUTION }],
  getAttributionHeaders(
    { provider: "nvidia", baseUrl: "https://example.test/v1" },
    { attribution: PI_ATTRIBUTION },
  ),
);
idPush(
  "attr-nvidia-host",
  "getAttributionHeaders",
  [
    { provider: "custom-nim", baseUrl: "https://integrate.api.nvidia.com/v1" },
    { attribution: ACME },
  ],
  getAttributionHeaders(
    { provider: "custom-nim", baseUrl: "https://integrate.api.nvidia.com/v1" },
    { attribution: ACME },
  ),
);
idPush(
  "attr-opencode",
  "getAttributionHeaders",
  [
    { provider: "opencode", baseUrl: "https://opencode.ai/zen/v1" },
    { attribution: PI_ATTRIBUTION },
  ],
  getAttributionHeaders(
    { provider: "opencode", baseUrl: "https://opencode.ai/zen/v1" },
    { attribution: PI_ATTRIBUTION },
  ),
);
idPush(
  "attr-opencode-host",
  "getAttributionHeaders",
  [{ provider: "custom", baseUrl: "https://opencode.ai/zen/v1" }, { attribution: ACME }],
  getAttributionHeaders(
    { provider: "custom", baseUrl: "https://opencode.ai/zen/v1" },
    { attribution: ACME },
  ),
);
idPush(
  "attr-gateway-none",
  "getAttributionHeaders",
  [
    { provider: "vercel-ai-gateway", baseUrl: "https://ai-gateway.vercel.sh/v1" },
    { attribution: PI_ATTRIBUTION },
  ],
  getAttributionHeaders(
    { provider: "vercel-ai-gateway", baseUrl: "https://ai-gateway.vercel.sh/v1" },
    { attribution: PI_ATTRIBUTION },
  ),
);
idPush(
  "identity-headers",
  "getIdentityHeaders",
  [openrouter, { attribution: PI_ATTRIBUTION, userAgent: "short" }],
  getIdentityHeaders(openrouter, { attribution: PI_ATTRIBUTION, userAgent: "short" }),
);
idPush(
  "opencode-session",
  "getOpenCodeSessionHeaders",
  [{ provider: "custom", baseUrl: "https://opencode.ai/zen/v1" }, "session-1"],
  getOpenCodeSessionHeaders(
    { provider: "custom", baseUrl: "https://opencode.ai/zen/v1" },
    "session-1",
  ),
);
idPush(
  "opencode-session-none",
  "getOpenCodeSessionHeaders",
  [openrouter, "session-1"],
  getOpenCodeSessionHeaders(openrouter, "session-1"),
);

// Synthetic tool results carry `Date.now()` — mask them for determinism.
const isPlainObject = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value);

const maskTimestamps = (value: unknown): unknown => {
  if (Array.isArray(value)) {
    return value.map(maskTimestamps);
  }
  if (isPlainObject(value)) {
    const out: Record<string, unknown> = {};
    for (const [key, nested] of Object.entries(value)) {
      out[key] =
        key === "timestamp" && typeof nested === "number" && nested > 1_000_000_000_000
          ? 0
          : maskTimestamps(nested);
    }
    return out;
  }
  return value;
};

const generator = {
  script: "scripts/parity/generate-transcript.ts",
  node: process.version,
  upstream: "@earendil-works/pi-ai@1.5.0",
};
for (const [file, cases] of [
  ["vectors/ai/transcript.json", transcriptCases],
  ["vectors/ai/text.json", textCases],
  ["vectors/ai/transform.json", transformCases],
  ["vectors/ai/identity.json", identityCases],
] as const) {
  writeFileSync(
    file,
    `${JSON.stringify({ generator, cases: cases.map((c) => ({ ...c, result: maskTimestamps(c.result) })) }, null, 2)}\n`,
  );
  console.log(`${file}: ${cases.length} cases`);
}
