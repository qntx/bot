/// <reference types="node" />
import { writeFileSync } from "node:fs";

import { createAssistantMessageEventStream, parseToolCallArguments } from "@earendil-works/pi-ai";
import type { AssistantMessage, DeferredHandle, ToolCall, Usage } from "@earendil-works/pi-ai";

// AI-STREAM-PROTOCOL vectors: sender-side op scripts replayed through
// upstream's AssistantMessageEventStream the way adapters drive it. Records
// the ops, the yielded events (minus `partial`), and the final message
// (minus `durationMs`).

type Op =
  | { op: "start" }
  | { op: "text_start" }
  | { op: "thinking_start" }
  | { op: "toolcall_start"; call: ToolCall }
  | { op: "text_delta" | "thinking_delta" | "toolcall_delta"; index?: number; delta: string }
  | { op: "toolcall_replace"; index?: number; text: string }
  | { op: "text_end"; index?: number; block: { text: string; textSignature?: string } }
  | {
      op: "thinking_end";
      index?: number;
      block: { thinking: string; thinkingSignature?: string; redacted?: boolean };
    }
  | { op: "toolcall_end"; index?: number }
  | { op: "metadata"; field: "usage"; value: Usage }
  | { op: "metadata"; field: "responseId" | "responseModel"; value: string }
  | { op: "metadata"; field: "deferred"; value: DeferredHandle }
  | { op: "done"; reason: "stop" | "length" | "toolUse" | "deferred" }
  | { op: "error"; reason: "aborted" | "error"; errorMessage: string };

type RecordedCase = { name: string; ops: Op[]; events: unknown[]; message: unknown };

const zeroUsage = (): Usage => ({
  input: 0,
  output: 0,
  cacheRead: 0,
  cacheWrite: 0,
  totalTokens: 0,
  cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
});

const toolCallOf = (id: string, name: string, text: string): ToolCall => {
  const { arguments: args, argumentsError } = parseToolCallArguments(text);
  const call: ToolCall = { type: "toolCall", id, name, arguments: args };
  if (argumentsError !== undefined) {
    call.argumentsError = argumentsError;
  }
  return call;
};

const run = async (name: string, ops: Op[]): Promise<RecordedCase> => {
  const events: unknown[] = [];
  const stream = createAssistantMessageEventStream();
  const output: AssistantMessage = {
    role: "assistant",
    content: [],
    api: "anthropic-messages",
    provider: "anthropic",
    model: "m",
    usage: zeroUsage(),
    stopReason: "stop",
    timestamp: Date.now(),
  };
  let nextIndex = 0;
  const lastOf = { text: -1, thinking: -1, toolcall: -1 };
  const argsStreams = new Map<number, string>();
  const kindOf = (kind: Op["op"]): keyof typeof lastOf =>
    kind.startsWith("toolcall") ? "toolcall" : kind.startsWith("thinking") ? "thinking" : "text";
  const at = (index: number | undefined, kind: Op["op"]): number => index ?? lastOf[kindOf(kind)];
  for (const op of ops) {
    switch (op.op) {
      case "start":
        stream.push({ type: "start", partial: output });
        break;

      case "text_start":
      case "thinking_start": {
        const i = nextIndex;
        nextIndex += 1;
        lastOf[kindOf(op.op)] = i;
        if (op.op === "text_start") {
          output.content.push({ type: "text", text: "" });
        } else {
          output.content.push({ type: "thinking", thinking: "" });
        }
        stream.push({ type: op.op, contentIndex: i, partial: output });
        break;
      }
      case "toolcall_start": {
        const i = nextIndex;
        nextIndex += 1;
        lastOf.toolcall = i;
        const { call } = op;
        output.content.push({ ...call, type: "toolCall" });
        argsStreams.set(
          i,
          Object.keys(call.arguments).length > 0 ? JSON.stringify(call.arguments) : "",
        );
        stream.push({ type: "toolcall_start", contentIndex: i, partial: output });
        break;
      }
      case "text_delta":
      case "thinking_delta":
      case "toolcall_delta": {
        const i = at(op.index, op.op);
        if (op.op === "toolcall_delta") {
          argsStreams.set(i, `${argsStreams.get(i) ?? ""}${op.delta}`);
        } else {
          const block = output.content[i];
          if (block?.type === "text") {
            block.text += op.delta;
          } else if (block?.type === "thinking") {
            block.thinking += op.delta;
          }
        }
        stream.push({ type: op.op, contentIndex: i, delta: op.delta, partial: output });
        break;
      }
      case "toolcall_replace": {
        const i = at(op.index, op.op);
        const soFar = argsStreams.get(i) ?? "";
        if (op.text.startsWith(soFar)) {
          const delta = op.text.slice(soFar.length);
          argsStreams.set(i, op.text);
          stream.push({ type: "toolcall_delta", contentIndex: i, delta, partial: output });
        } else {
          argsStreams.set(i, op.text);
        }
        break;
      }
      case "text_end":
      case "thinking_end": {
        const i = at(op.index, op.op);
        const content =
          op.op === "text_end"
            ? ({ type: "text", ...op.block } as const)
            : ({ type: "thinking", ...op.block } as const);
        output.content[i] = content;
        stream.push({
          type: op.op,
          contentIndex: i,
          content: content.type === "text" ? content.text : content.thinking,
          partial: output,
        });
        break;
      }
      case "toolcall_end": {
        const i = at(op.index, op.op);
        const target = output.content[i];
        if (!target || target.type !== "toolCall") {
          continue;
        }
        const call = toolCallOf(target.id, target.name, argsStreams.get(i) ?? "");
        Object.assign(target, call);
        stream.push({ type: "toolcall_end", contentIndex: i, toolCall: call, partial: output });
        break;
      }
      case "metadata":
        if (op.field === "usage") {
          output.usage = op.value;
        } else if (op.field === "responseId") {
          output.responseId = op.value;
        } else if (op.field === "responseModel") {
          output.responseModel = op.value;
        } else if (op.field === "deferred") {
          output.deferred = op.value;
        }
        break;

      case "done":
        output.stopReason = op.reason;
        stream.push({ type: "done", reason: op.reason, message: output });
        break;

      case "error":
        output.stopReason = op.reason;
        output.errorMessage = op.errorMessage;
        stream.push({ type: "error", reason: op.reason, error: output });
        break;
    }
  }
  for await (const event of stream) {
    // Upstream events carry `partial`; strip it, and mask wall-clock
    // timestamps inside the terminal `message`/`error` payloads.
    if (event.type === "done") {
      event.message.timestamp = 0;
    } else if (event.type === "error") {
      event.error.timestamp = 0;
    }
    events.push({ ...event, partial: undefined });
  }
  const message = await stream.result();
  const { durationMs: _durationMs, ...rest } = message;
  // `timestamp` is wall-clock; a sentinel keeps the file deterministic.
  return { name, ops, events, message: { ...rest, timestamp: 0 } };
};

const cases: RecordedCase[] = [
  await run("text", [
    { op: "start" },
    { op: "text_start" },
    { op: "text_delta", delta: "he" },
    { op: "text_delta", delta: "llo" },
    { op: "text_end", block: { text: "hello" } },
    { op: "done", reason: "stop" },
  ]),
  await run("thinking-signature", [
    { op: "start" },
    { op: "thinking_start" },
    { op: "thinking_delta", delta: "let me" },
    { op: "thinking_delta", delta: " think" },
    { op: "thinking_end", block: { thinking: "let me think", thinkingSignature: "sig-1" } },
    { op: "done", reason: "stop" },
  ]),
  await run("toolcall-split-replace", [
    { op: "start" },
    {
      op: "toolcall_start",
      call: { type: "toolCall", id: "call-1", name: "write", arguments: {} },
    },
    { op: "toolcall_delta", delta: '{"path":"a' },
    { op: "toolcall_replace", text: String.raw`{"path":"ab","content":"x\n"}` },
    { op: "toolcall_end" },
    { op: "done", reason: "toolUse" },
  ]),
  await run("metadata-then-done-length", [
    { op: "start" },
    {
      op: "metadata",
      field: "usage",
      value: { ...zeroUsage(), input: 10, output: 3, totalTokens: 13 },
    },
    { op: "metadata", field: "responseId", value: "resp-1" },
    { op: "metadata", field: "responseModel", value: "m-actual" },
    { op: "text_start" },
    { op: "text_end", block: { text: "hi" } },
    { op: "done", reason: "length" },
  ]),
  await run("done-deferred", [
    { op: "start" },
    {
      op: "metadata",
      field: "deferred",
      value: { provider: "anthropic", modelId: "m", api: "anthropic-messages", id: "h-1" },
    },
    { op: "done", reason: "deferred" },
  ]),
  await run("error-before-start", [
    { op: "error", reason: "error", errorMessage: "provider unavailable" },
  ]),
  await run("error-after-start", [
    { op: "start" },
    { op: "error", reason: "error", errorMessage: "stream reset" },
  ]),
  await run("aborted", [{ op: "start" }, { op: "error", reason: "aborted", errorMessage: "" }]),
  await run("redacted-thinking", [
    { op: "start" },
    { op: "thinking_start" },
    { op: "thinking_end", block: { thinking: "", redacted: true, thinkingSignature: "rsig" } },
    { op: "done", reason: "stop" },
  ]),
  await run("text-signature", [
    { op: "start" },
    { op: "text_start" },
    { op: "text_delta", delta: "hi" },
    { op: "text_end", block: { text: "hi", textSignature: "tsig" } },
    { op: "done", reason: "stop" },
  ]),
  await run("two-toolcalls", [
    { op: "start" },
    {
      op: "toolcall_start",
      call: { type: "toolCall", id: "c1", name: "read", arguments: {} },
    },
    {
      op: "toolcall_start",
      call: { type: "toolCall", id: "c2", name: "write", arguments: {} },
    },
    { op: "toolcall_delta", index: 0, delta: '{"file":"a.txt"}' },
    { op: "toolcall_delta", index: 1, delta: '{"file":"b.txt"' },
    { op: "toolcall_delta", index: 1, delta: ',"n":1}' },
    { op: "toolcall_end", index: 0 },
    { op: "toolcall_end", index: 1 },
    { op: "done", reason: "toolUse" },
  ]),
  await run("toolcall-replace-restart", [
    { op: "start" },
    {
      op: "toolcall_start",
      call: { type: "toolCall", id: "call-9", name: "t", arguments: {} },
    },
    { op: "toolcall_delta", delta: '{"a":1,' },
    { op: "toolcall_replace", text: '{"b":2}' },
    { op: "toolcall_end" },
    { op: "done", reason: "toolUse" },
  ]),
  await run("mixed-blocks", [
    { op: "start" },
    { op: "text_start" },
    { op: "text_delta", delta: "let me" },
    { op: "text_end", block: { text: "let me" } },
    { op: "thinking_start" },
    { op: "thinking_delta", delta: "think" },
    { op: "thinking_end", block: { thinking: "think", thinkingSignature: "s" } },
    {
      op: "toolcall_start",
      call: { type: "toolCall", id: "c3", name: "m", arguments: {} },
    },
    { op: "toolcall_delta", delta: '{"k":true}' },
    { op: "toolcall_end" },
    { op: "done", reason: "toolUse" },
  ]),
  await run("error-after-partial", [
    { op: "start" },
    { op: "text_start" },
    { op: "text_delta", delta: "partial out" },
    { op: "error", reason: "error", errorMessage: "stream reset" },
  ]),
  await run("multiple-blocks", [
    { op: "start" },
    { op: "text_start" },
    {
      op: "toolcall_start",
      call: { type: "toolCall", id: "c2", name: "read", arguments: {} },
    },
    { op: "text_delta", delta: "ok" },
    { op: "text_end", block: { text: "ok" } },
    { op: "toolcall_delta", index: 1, delta: '{"file":"f.txt"}' },
    { op: "toolcall_end", index: 1 },
    { op: "done", reason: "toolUse" },
  ]),
];

writeFileSync(
  "vectors/ai/stream.json",
  `${JSON.stringify(
    {
      generator: {
        script: "scripts/parity/generate-stream.ts",
        node: process.version,
        upstream: "@earendil-works/pi-ai@1.5.0",
      },
      cases,
    },
    null,
    2,
  )}\n`,
);
console.log(`vectors/ai/stream.json: ${cases.length} cases`);
