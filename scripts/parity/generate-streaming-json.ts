/// <reference types="node" />
import { writeFileSync } from "node:fs";

import {
  parseJsonWithRepair,
  repairJson,
  StreamingJsonParser,
  ToolCallArgumentsStream,
} from "@earendil-works/pi-ai";
import type { JsonChange } from "@earendil-works/pi-ai";

// Parity vectors for AI-JSON-PARTIAL: incremental parsing changes, repair,
// and finished-argument parsing. `argumentsError` embeds the V8 message, so
// cases with one carry class `engine-error-text`.

const PRNG_SEED = 20260303;

type StreamCase = {
  name: string;
  kind: "stream";
  pieces: string[];
  changes: JsonChange[][];
  finish: { arguments: unknown; argumentsError?: string };
  classes?: string[];
};
type TextCase = {
  name: string;
  kind: "repair" | "parse-repair";
  input: string;
  output: unknown;
  outputError?: string;
  classes?: string[];
};

const SURROGATE_ESCAPE = /\\u[dD][89a-fA-F][0-9a-fA-F]{2}/;

const streamCase = (name: string, pieces: string[]): StreamCase => {
  const parser = new StreamingJsonParser();
  const changes = pieces.map((piece) => [...parser.append(piece)]);
  const stream = new ToolCallArgumentsStream();
  for (const piece of pieces) {
    stream.append(piece);
  }
  const { arguments: args, argumentsError } = stream.finish();
  const finish: StreamCase["finish"] = { arguments: args };
  const entry: StreamCase = { name, kind: "stream", pieces, changes, finish };
  const classes: string[] = [];
  if (argumentsError !== undefined) {
    finish.argumentsError = argumentsError;
    classes.push("engine-error-text");
  }
  // Pieces containing surrogate-range escapes: upstream reports the half
  // immediately; ovo defers it pending the partner, so per-piece texts differ.
  if (pieces.some((piece) => SURROGATE_ESCAPE.test(piece))) {
    classes.push("lone-surrogates");
  }
  if (classes.length > 0) {
    entry.classes = classes;
  }
  return entry;
};

const repairCase = (name: string, input: string): TextCase => ({
  name,
  kind: "repair",
  input,
  output: repairJson(input),
});

const parseRepairCase = (name: string, input: string): TextCase => {
  const entry: TextCase = { name, kind: "parse-repair", input, output: null };
  try {
    entry.output = parseJsonWithRepair<unknown>(input);
  } catch (error) {
    entry.outputError = error instanceof Error ? error.message : String(error);
    entry.classes = ["engine-error-text"];
  }
  return entry;
};

// Seeded PRNG (upstream streaming-json.test.ts LCG) for reproducible docs.
const random = (seed: number): (() => number) => {
  let state = seed;
  return () => {
    // `x % 2^31` is identical to `x & 0x7fffffff` for every x >= 0.
    state = (state * 1103515245 + 12345) % 2147483648;
    return state / 0x7fffffff;
  };
};
const rand = random(PRNG_SEED);
const pick = <T>(items: ReadonlyArray<T>): T => {
  const item = items[Math.floor(rand() * items.length)];
  if (item === undefined) {
    throw new Error("pick from empty array");
  }
  return item;
};

const STRING_PIECES = [
  "a",
  "b",
  "x y",
  " ",
  "\n",
  "\t",
  "\\",
  '"',
  "/",
  "é",
  "😀",
  "",
  "",
  " ",
  "{",
  "]",
  ",",
] as const;
const randomString = (): string =>
  Array.from({ length: Math.floor(rand() * 6) }, () => pick(STRING_PIECES)).join("");
const randomScalar = (): unknown =>
  pick([
    () => randomString(),
    () => Math.floor(rand() * 2000) - 1000,
    () => rand() * 100,
    () => 1e21,
    () => -5e-8,
    () => -0,
    () => 0,
    () => true,
    () => false,
    () => null,
  ])();
const randomValue = (depth: number): unknown => {
  const roll = rand();
  if (depth > 4 || roll < 0.3) {
    return randomScalar();
  }
  if (roll < 0.65) {
    const members: Record<string, unknown> = {};
    for (let index = 0; index < Math.floor(rand() * 4); index++) {
      members[randomString()] = randomValue(depth + 1);
    }
    return members;
  }
  return Array.from({ length: Math.floor(rand() * 4) }, () => randomValue(depth + 1));
};
const randomChunks = (text: string, count: number): string[] => {
  const cuts = new Set<number>();
  while (cuts.size < count && cuts.size < text.length) {
    cuts.add(Math.floor(rand() * (text.length + 1)));
  }
  const sorted = [...cuts].toSorted((a, b) => a - b);
  const pieces: string[] = [];
  let start = 0;
  for (const cut of sorted) {
    pieces.push(text.slice(start, cut));
    start = cut;
  }
  pieces.push(text.slice(start));
  return pieces.filter((piece) => piece !== "");
};
const corrupt = (text: string): string => {
  const roll = rand();
  if (roll < 0.25) {
    return text.slice(0, Math.floor(rand() * text.length));
  }
  if (roll < 0.4) {
    return text.replaceAll(",", pick(["", " ", " ,", ":"]));
  }
  if (roll < 0.55) {
    return text.replaceAll(":", pick(["", " ", "::"]));
  }
  if (roll < 0.7) {
    return text.replaceAll('"', () => (rand() < 0.5 ? "" : '"'));
  }
  if (roll < 0.8) {
    return `${text}\u0000\u001F`;
  }
  if (roll < 0.9) {
    return text.replaceAll(/[a-z]+/g, (word) => (rand() < 0.3 ? "NaN" : word));
  }
  return `${text}${pick([" garbage", "]", "}", "1e", String.raw`\q`])}`;
};

const cases: Array<StreamCase | TextCase> = [];

// Literal inputs from the upstream test files.
const upstreamInputs = [
  String.raw`{"a":"\ud83d`,
  '{"code":"x\t\t  ',
  '{"a":"x\ny \\d \\u12"',
  String.raw`{"text":"ab\u00`,
  '{"n":1',
  '{"a":[1,{"b":["x"',
  '{"a":1 2, "b" "x"}',
  '{"c":3}',
  '{"path":"a.ts","content":"x',
  '{"a":"x',
  String.raw`"\\ud800"`,
  String.raw`{"a": "\u0061\u0062"}`,
  '{"big": 12345678901234567890}',
];
for (const input of upstreamInputs) {
  cases.push(
    streamCase(
      `upstream-${input.length > 20 ? input.slice(0, 20) : input}-${cases.length}`,
      randomChunks(input, 2),
    ),
  );
}
cases.push(
  streamCase("upstream-split-escape", ['{"a":"x', String.raw`\u00`, 'e9"}']),
  streamCase("upstream-split-pair", [String.raw`{"a":"\ud83d`, String.raw`\ude00"}`]),
  streamCase("upstream-split-basic", ['{"n":1', '2,"s":"ab', 'c","t":tru', "e}"]),
  streamCase("upstream-second-doc", ['{"a":1}', ' {"ignored":1}']),
  streamCase("upstream-empty-and-ws", ["", " ", '{"a":1}']),
);

// Seeded valid documents with random chunking.
for (let index = 0; index < 30; index++) {
  const doc = rand() < 0.7 ? { path: randomString(), args: randomValue(0) } : randomValue(0);
  const text = JSON.stringify(doc);
  cases.push(streamCase(`seeded-valid-${index}`, randomChunks(text, 4)));
}
// Seeded corrupted documents.
for (let index = 0; index < 30; index++) {
  const text = corrupt(JSON.stringify(randomValue(0)));
  cases.push(streamCase(`seeded-corrupt-${index}`, randomChunks(text, 4)));
}
// Deep nesting past the 32-level report limit.
const deep = `[${"[".repeat(50)}1${"]".repeat(50)}]`;
cases.push(streamCase("deep-nesting-50", randomChunks(deep, 3)));
const deepObj = `${Array.from({ length: 45 }, () => '{"k":').join("")}1${"}".repeat(45)}`;
cases.push(streamCase("deep-object-45", randomChunks(deepObj, 3)));
// Large array in moderate chunks.
const large = JSON.stringify({ items: Array.from({ length: 2000 }, (_, i) => `item-${i}`) });
// Astral text raw and escaped.
cases.push(
  streamCase("large-array-2000", randomChunks(large, 12)),
  streamCase("astral-raw", ['{"a":"😀𝄞"}']),
  streamCase("astral-escaped", [String.raw`{"a":"\ud83d\ude00\ud834\udd1e"}`]),
);

// repairJson / parseJsonWithRepair inputs (upstream json-parse.test.ts).
const repairInputs = [
  '{"a":"x\ny"}',
  String.raw`{"a":"x\dy"}`,
  String.raw`{"a":"x\u12"y"}`,
  String.raw`{"a":"x\"}`,
  String.raw`{"a":1,"b":"\\"}`,
  String.raw`{"a":"\t\n\r"}`,
  String.raw`{"a":"\u00e9\uD800"}`,
  "not json at all",
  '{"a":"ok"}',
  "[1,2,3]",
  "",
  "   ",
  '{"k":"a"}',
  '{"k":"a\u0001b"}',
  '{"k":"a\u0000b"}',
  '{"a":',
  '{"a":"unterminated',
  '{"a": "v" "b": 2}',
  '{"bad": tru}',
  "NaN",
  "Infinity",
  String.raw`{"a": "\q"}`,
];
for (const input of repairInputs) {
  cases.push(
    repairCase(`repair-${cases.length}`, input),
    parseRepairCase(`parse-${cases.length}`, input),
  );
}
// parseToolCallArguments-only inputs: empty/blank and non-object JSON.
const argInputs = ["", "   ", "null", "[1,2]", '"text"', "42", "true", '{"ok":true}'];
for (const input of argInputs) {
  cases.push(
    streamCase(`args-${input === "" ? "empty" : input.slice(0, 12)}-${cases.length}`, [input]),
  );
}

const document = {
  generator: {
    script: "scripts/parity/generate-streaming-json.ts",
    node: process.version,
    upstream: "@earendil-works/pi-ai@1.5.0",
    seed: PRNG_SEED,
  },
  cases,
};
writeFileSync("vectors/ai/streaming-json.json", `${JSON.stringify(document, null, 2)}\n`);
console.log(`vectors/ai/streaming-json.json: ${cases.length} cases`);
