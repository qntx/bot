import { writeFileSync } from "node:fs";
// Generates vectors/ai/sse.json: real Stainless `_iterSSEMessages`
// (vendored @anthropic-ai/sdk + openai) fed byte streams split at seeded
// boundaries — the exported generator runs each SDK's own LineDecoder +
// SSEDecoder + EOF flush end to end. Per-SDK expectations are recorded
// when outputs differ.
/* oxlint-disable no-bitwise -- seeded 32-bit PRNG requires int ops */
import { createRequire } from "node:module";
import { dirname, join } from "node:path";
import { pathToFileURL } from "node:url";

const require_ = createRequire(import.meta.url);

// Resolve the vendored streaming modules through the installed
// packages (works with any package-manager layout). The SDKs are
// deps of `@earendil-works/pi-ai`, so resolution starts there.
const piAiDir = dirname(require_.resolve("@earendil-works/pi-ai/package.json"));
function sdkFile(package_: string, file: string): string {
  const packageJson = require_.resolve(`${package_}/package.json`, {
    paths: [piAiDir],
  });
  return join(dirname(packageJson), file);
}

const anthropic: unknown = await import(
  pathToFileURL(sdkFile("@anthropic-ai/sdk", "core/streaming.js")).href
);
const openai: unknown = await import(pathToFileURL(sdkFile("openai", "core/streaming.js")).href);

type SseRecord = {
  // oxlint-disable-next-line no-restricted-types -- SDK emits null for absent event
  event: string | null;
  data: string;
  raw: string[];
};

type IterFn = (r: Response, c: AbortController) => AsyncGenerator<SseRecord, unknown, unknown>;

function isStreamingModule(m: unknown): m is { _iterSSEMessages: IterFn } {
  return (
    typeof m === "object" &&
    m !== null &&
    "_iterSSEMessages" in m &&
    typeof m._iterSSEMessages === "function"
  );
}

function iter(m: unknown): IterFn {
  if (!isStreamingModule(m)) {
    throw new Error("module lacks _iterSSEMessages");
  }
  return m._iterSSEMessages;
}

function mulberry32(seed0: number) {
  let seed = seed0;
  return () => {
    seed = Math.trunc(seed + 0x6d2b79f5);
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t ^= t + Math.imul(t ^ (t >>> 7), 61 | t);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

async function run(iterFn: ReturnType<typeof iter>, bytes: Uint8Array, splits: number[]) {
  const body = new ReadableStream<Uint8Array>({
    start(c) {
      let at = 0;
      for (const end of [...splits, bytes.length]) {
        if (end > at) {
          c.enqueue(bytes.subarray(at, end));
        }
        at = end;
      }
      c.close();
    },
  });
  const out: SseRecord[] = [];
  for await (const sse of iterFn(new Response(body), new AbortController())) {
    out.push(sse);
  }
  return out;
}

const cases: Array<Record<string, unknown>> = [];
const rand = mulberry32(42);
function add(name: string, bytes: Uint8Array, chunkCount: number) {
  const splits: number[] = [];
  for (let i = 0; i < chunkCount; i++) {
    splits.push(Math.floor(rand() * bytes.length));
  }
  splits.sort((a, b) => a - b);
  return {
    name,
    bytes_hex: [...bytes].map((b) => b.toString(16).padStart(2, "0")).join(""),
    splits,
  };
}
const enc = new TextEncoder();
const u8 = (s: string) => enc.encode(s);
const inputs: Array<[string, Uint8Array, number]> = [
  ["simple-crlf", u8("data: hello\r\n\r\n"), 3],
  ["lf-only", u8("data: a\ndata: b\n\nevent: e\ndata: c\n\n"), 4],
  ["cr-only", u8("data: x\rdata: y\r\r"), 2],
  ["lone-cr-at-chunk-end", u8("data: a\rdata: b\rdata: c\r\r"), 5],
  ["comments", u8(": hi\ndata: a\n\n: bye\n\n"), 3],
  ["comment-only-block", u8(": only\n: more\n\ndata: real\n\n"), 2],
  ["multi-line-data", u8("data: one\ndata: two\ndata: three\n\n"), 3],
  ["missing-final-blank", u8("event: last\ndata: tail"), 2],
  ["utf8-split", enc.encode("data: héllo wörld ✓\n\n"), 5],
  ["colon-in-data", u8("data: a:b:c\n\n"), 2],
  ["no-space-after-colon", u8("data:value\n\n"), 2],
  ["double-space", u8("data:  two\n\n"), 1],
  ["done-marker", u8("data: [DONE]\n\n"), 1],
  ["event-and-data", u8("event: ping\ndata: pong\n\n"), 2],
  ["unknown-field", u8("id: 42\nretry: 100\ndata: x\n\n"), 2],
  ["empty-event-line", u8("event:\ndata: x\n\n"), 1],
  ["leading-blank-lines", u8("\n\n\ndata: x\n\n"), 2],
  ["trailing-cr", u8("data: x\r\n\r\n"), 3],
  ["no-final-newline", u8("data: partial"), 1],
  ["empty-stream", u8(""), 0],
  ["blank-stream", u8("\n\n\n"), 1],
  ["utf8-char-across-chunk", enc.encode("data: a✓b\n\n"), 0],
  ["raw-cr-in-line", u8("data: a\rmid\n\n"), 3],
  ["multi-events", u8("event: a\ndata: 1\n\ndata: 2\n\n"), 3],
];
for (const [name, bytes, n] of inputs) {
  const base = add(name, bytes, n);
  // oxlint-disable-next-line no-await-in-loop -- deterministic case order
  const a = await run(iter(anthropic), bytes, base.splits);
  // oxlint-disable-next-line no-await-in-loop -- deterministic case order
  const o = await run(iter(openai), bytes, base.splits);
  const same = JSON.stringify(a) === JSON.stringify(o);
  cases.push(
    same ? { ...base, expected: a } : { ...base, expected_anthropic: a, expected_openai: o },
  );
}
writeFileSync("vectors/ai/sse.json", `${JSON.stringify(cases, null, 2)}\n`);
console.log(
  `cases=${cases.length} differ=${cases.filter((c) => "expected_anthropic" in c).length}`,
);
