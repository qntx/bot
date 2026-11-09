// Parity vectors for AI-CONSTRAINED-SAMPLING: the strict subset of tool schemas
// (`makeStrictJsonSchema`) and the strict-mode decision
// (`resolveJsonSchemaStrictSampling`) from the published pi-ai package.
// Run with `node scripts/parity/generate-constrained-sampling.ts`.
import { writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

type GrammarFn = (...args: unknown[]) => unknown;
type Module = {
  makeStrictJsonSchema: GrammarFn;
  resolveJsonSchemaStrictSampling: GrammarFn;
  resolveGrammarConstrainedSampling: GrammarFn;
  createGrammarToolInputProperties: GrammarFn;
  getGrammarToolInput: GrammarFn;
  appendGrammarToolInputJsonDelta: GrammarFn;
  closeGrammarToolInputJson: GrammarFn;
};
function isModule(value: unknown): value is Module {
  return (
    typeof value === "object" &&
    value !== null &&
    "makeStrictJsonSchema" in value &&
    typeof value.makeStrictJsonSchema === "function" &&
    "resolveJsonSchemaStrictSampling" in value &&
    typeof value.resolveJsonSchemaStrictSampling === "function" &&
    "resolveGrammarConstrainedSampling" in value &&
    typeof value.resolveGrammarConstrainedSampling === "function" &&
    "createGrammarToolInputProperties" in value &&
    typeof value.createGrammarToolInputProperties === "function" &&
    "getGrammarToolInput" in value &&
    typeof value.getGrammarToolInput === "function" &&
    "appendGrammarToolInputJsonDelta" in value &&
    typeof value.appendGrammarToolInputJsonDelta === "function" &&
    "closeGrammarToolInputJson" in value &&
    typeof value.closeGrammarToolInputJson === "function"
  );
}

const piAiDir = dirname(dirname(fileURLToPath(import.meta.resolve("@earendil-works/pi-ai"))));
const loaded: unknown = await import(
  pathToFileURL(join(piAiDir, "dist/api/constrained-sampling.js")).href
);
if (!isModule(loaded)) {
  throw new Error("constrained-sampling dist module lacks the expected exports");
}
const {
  makeStrictJsonSchema,
  resolveJsonSchemaStrictSampling,
  resolveGrammarConstrainedSampling,
  createGrammarToolInputProperties,
  getGrammarToolInput,
  appendGrammarToolInputJsonDelta,
  closeGrammarToolInputJson,
} = loaded;

/** The thrown message, or the returned value. */
function outcome(run: () => unknown): Record<string, unknown> {
  try {
    const value = run();
    return value === undefined ? { result: null } : { result: value };
  } catch (error) {
    return { error: error instanceof Error ? error.message : String(error) };
  }
}

const object = (properties: Record<string, unknown>, extra: Record<string, unknown> = {}) => ({
  type: "object",
  properties,
  ...extra,
});

const SCHEMAS: Array<{ name: string; schema: unknown }> = [
  { name: "flat-required", schema: object({ path: { type: "string" } }, { required: ["path"] }) },
  {
    name: "flat-optional",
    schema: object({ path: { type: "string" }, limit: { type: "number" } }, { required: ["path"] }),
  },
  {
    name: "optional-nullable",
    schema: object({
      a: { type: ["string", "null"] },
      b: { enum: ["x", null] },
      c: { const: null },
    }),
  },
  {
    name: "nested-object",
    schema: object({ inner: object({ x: { type: "integer" } }) }, { required: ["inner"] }),
  },
  {
    name: "array-items",
    schema: object({ list: { type: "array", items: object({ k: { type: "string" } }) } }),
  },
  {
    name: "any-of-scalars",
    schema: object({ v: { anyOf: [{ type: "string" }, { type: "number" }] } }),
  },
  { name: "any-of-object", schema: object({ v: { anyOf: [object({}), { type: "string" }] } }) },
  { name: "any-of-empty", schema: object({ v: { anyOf: [] } }) },
  { name: "tuple", schema: object({ t: { type: "array", items: [{ type: "string" }] } }) },
  {
    name: "ref",
    schema: object({ r: { $ref: "#/$defs/x" } }, { $defs: { x: { type: "string" } } }),
  },
  { name: "one-of", schema: object({ v: { oneOf: [{ type: "string" }] } }) },
  { name: "additional-true", schema: object({}, { additionalProperties: true }) },
  {
    name: "additional-false",
    schema: object({ a: { type: "string" } }, { additionalProperties: false, required: [] }),
  },
  { name: "properties-without-object", schema: object({ inner: { properties: {} } }) },
  { name: "required-unknown", schema: object({ a: { type: "string" } }, { required: ["b"] }) },
  { name: "required-not-strings", schema: object({ a: { type: "string" } }, { required: [1] }) },
  { name: "root-not-object", schema: { type: "string" } },
  { name: "boolean-property", schema: object({ a: true }) },
  {
    name: "key-order-kept",
    schema: {
      description: "d",
      type: "object",
      required: [],
      properties: { b: { type: "string" }, a: { type: "string" } },
      title: "t",
    },
  },
];

const tool = (schema: unknown, constrainedSampling?: unknown) => ({
  name: "probe",
  description: "probe",
  parameters: schema,
  ...(constrainedSampling === undefined ? {} : { constrainedSampling }),
});

const cases: Array<Record<string, unknown>> = [];
for (const { name, schema } of SCHEMAS) {
  cases.push({
    name: `strict-${name}`,
    kind: "makeStrict",
    schema,
    ...outcome(() => makeStrictJsonSchema(schema)),
  });
}
const RESOLVE: Array<{ name: string; schema: unknown; sampling?: unknown }> = [
  { name: "none", schema: SCHEMAS[0]?.schema },
  { name: "disabled", schema: SCHEMAS[0]?.schema, sampling: false },
  { name: "grammar", schema: SCHEMAS[0]?.schema, sampling: { type: "grammar", variants: {} } },
  {
    name: "prefer-ok",
    schema: SCHEMAS[1]?.schema,
    sampling: { type: "json_schema", strict: "prefer" },
  },
  {
    name: "prefer-unsupported",
    schema: SCHEMAS[9]?.schema,
    sampling: { type: "json_schema", strict: "prefer" },
  },
  {
    name: "require-ok",
    schema: SCHEMAS[1]?.schema,
    sampling: { type: "json_schema", strict: "require" },
  },
  {
    name: "require-unsupported",
    schema: SCHEMAS[9]?.schema,
    sampling: { type: "json_schema", strict: "require" },
  },
];
for (const { name, schema, sampling } of RESOLVE) {
  for (const supports of [true, false]) {
    const probe = tool(schema, sampling);
    cases.push({
      name: `resolve-${name}-${supports ? "supported" : "unsupported"}`,
      kind: "resolve",
      tool: probe,
      supportsStrictMode: supports,
      ...outcome(() => resolveJsonSchemaStrictSampling(probe, supports)),
    });
  }
}

const document = {
  generator: { script: "scripts/parity/generate-constrained-sampling.ts", node: process.version },
  cases,
};
const target = "vectors/ai/constrained-sampling.json";
// Grammar tools: resolution, input properties, replayed input, and the streamed input buffer.
function isParametersObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
type GrammarTool = {
  name: string;
  description: string;
  parameters: Record<string, unknown>;
  constrainedSampling?: unknown;
};
const grammarTool = (_name: string, sampling: unknown, parameters?: unknown): GrammarTool => {
  const built = tool(
    parameters ?? { type: "object", required: ["code"], properties: { code: { type: "string" } } },
    sampling,
  );
  if (typeof built !== "object" || built === null || !("parameters" in built)) {
    throw new Error("tool() returned an unexpected shape");
  }
  return {
    name: "check",
    description: "check input",
    parameters: isParametersObject(built.parameters) ? built.parameters : {},
    constrainedSampling: sampling,
  };
};

const GRAMMAR_RESOLVE: Array<{
  name: string;
  supports: boolean;
  sampling?: unknown;
  parameters?: unknown;
}> = [
  { name: "no-constrained-sampling", supports: true },
  {
    name: "grammar-unsupported",
    supports: false,
    sampling: { type: "grammar", variants: { openai_lark: 'root ::= "x"' } },
  },
  {
    name: "grammar-lark",
    supports: true,
    sampling: { type: "grammar", variants: { openai_lark: 'root ::= "x"' } },
  },
  {
    name: "grammar-regex-when-lark-blank",
    supports: true,
    sampling: { type: "grammar", variants: { openai_lark: "   ", openai_regex: "[a-z]+" } },
  },
  {
    name: "grammar-both-blank",
    supports: true,
    sampling: { type: "grammar", variants: { openai_lark: " ", openai_regex: "" } },
  },
  {
    name: "grammar-schema-not-object",
    supports: true,
    sampling: { type: "grammar", variants: { openai_regex: "x" } },
    parameters: { type: "string" },
  },
  {
    name: "grammar-required-multi",
    supports: true,
    sampling: { type: "grammar", variants: { openai_regex: "x" } },
    parameters: {
      type: "object",
      required: ["a", "b"],
      properties: { a: { type: "string" }, b: { type: "string" } },
    },
  },
  {
    name: "grammar-required-missing",
    supports: true,
    sampling: { type: "grammar", variants: { openai_regex: "x" } },
    parameters: { type: "object", properties: { code: { type: "string" } } },
  },
  {
    name: "grammar-property-missing",
    supports: true,
    sampling: { type: "grammar", variants: { openai_regex: "x" } },
    parameters: { type: "object", required: ["code"], properties: {} },
  },
  {
    name: "grammar-property-not-string",
    supports: true,
    sampling: { type: "grammar", variants: { openai_regex: "x" } },
    parameters: { type: "object", required: ["code"], properties: { code: { type: "number" } } },
  },
];
for (const entry of GRAMMAR_RESOLVE) {
  const { parameters } = entry;
  const t = grammarTool("check", entry.sampling, parameters);
  cases.push({
    kind: "grammar-resolve",
    name: entry.name,
    tool: t,
    supports: entry.supports,
    ...outcome(() => resolveGrammarConstrainedSampling(t, entry.supports)),
  });
}
const grammarToolsForCreate = [
  grammarTool("a", { type: "grammar", variants: { openai_lark: "r ::= 1" } }),
  grammarTool("b", { type: "grammar", variants: { openai_regex: "[0-9]" } }),
  grammarTool("c", undefined),
];
cases.push(
  {
    kind: "grammar-create-properties",
    tools: grammarToolsForCreate,
    ...outcome(() => {
      const map = createGrammarToolInputProperties(grammarToolsForCreate, true);
      if (!(map instanceof Map)) {
        throw new TypeError("createGrammarToolInputProperties returned a non-Map");
      }
      return [...(map.entries() as Iterable<[string, string]>)];
    }),
  },
  {
    kind: "grammar-input",
    arguments: { code: "src" },
    property: "code",
    ...outcome(() => getGrammarToolInput("t", { code: "src" }, "code")),
  },
  {
    kind: "grammar-input-error",
    arguments: { code: 3 },
    property: "code",
    ...outcome(() => getGrammarToolInput("t", { code: 3 }, "code")),
  },
);

type BufferOp = [string, string, string?, string?];
const GRAMMAR_BUFFER: Array<{ name: string; ops: BufferOp[] }> = [
  {
    name: "append-basic",
    ops: [
      ["append", "in", "he"],
      ["append", "in", "llo"],
      ["close", "in"],
    ],
  },
  {
    name: "append-empty",
    ops: [
      ["append", "in", ""],
      ["close", "in"],
    ],
  },
  {
    name: "close-no-final",
    ops: [
      ["append", "in", "x"],
      ["close", "in"],
    ],
  },
  {
    name: "close-with-final",
    ops: [
      ["append", "in", "he"],
      ["close", "in", "hello"],
    ],
  },
  {
    name: "close-twice-same",
    ops: [
      ["append", "in", "x"],
      ["close", "in", "xy"],
      ["close", "in", "xy"],
    ],
  },
  {
    name: "close-twice-diff",
    ops: [
      ["append", "in", "x"],
      ["close", "in", "xy"],
      ["close", "in", "xz"],
    ],
  },
  {
    name: "close-non-monotonic",
    ops: [
      ["append", "in", "he"],
      ["close", "in", "xx"],
    ],
  },
  {
    name: "append-after-close",
    ops: [
      ["append", "in", "x"],
      ["close", "in"],
      ["append", "in", "y"],
    ],
  },
  {
    name: "escape-quote",
    ops: [
      ["append", "in", 'say "hi"'],
      ["close", "in"],
    ],
  },
  {
    name: "escape-backslash",
    ops: [
      ["append", "in", String.raw`a\b`],
      ["close", "in"],
    ],
  },
  {
    name: "escape-newline",
    ops: [
      ["append", "in", "a\nb"],
      ["close", "in"],
    ],
  },
  {
    name: "escape-tab",
    ops: [
      ["append", "in", "a\tb"],
      ["close", "in"],
    ],
  },
  {
    name: "escape-control",
    ops: [
      ["append", "in", "a\u0001b"],
      ["close", "in"],
    ],
  },
  {
    name: "escape-u2028",
    ops: [
      ["append", "in", "a\u2028b"],
      ["close", "in"],
    ],
  },
  {
    name: "escape-non-bmp",
    ops: [
      ["append", "in", "x\u{1F600}y"],
      ["close", "in"],
    ],
  },
  {
    name: "escape-property-name",
    ops: [
      ["append", 'weird"prop', "v"],
      ["close", 'weird"prop'],
    ],
  },
];
for (const entry of GRAMMAR_BUFFER) {
  const buffer = { input: "", started: false, closed: false };
  const ops = entry.ops.map((op) => {
    const [kind, property, a, b] = op;
    const value =
      kind === "append"
        ? outcome(() => appendGrammarToolInputJsonDelta(buffer, property, a ?? ""))
        : outcome(() => closeGrammarToolInputJson(buffer, property, b));
    return [kind, property, a, b, value] as const;
  });
  cases.push({ kind: "grammar-buffer", name: entry.name, ops, buffer });
}
writeFileSync(target, `${JSON.stringify(document, null, 2)}\n`);
console.log(`${target}: ${cases.length} cases`);
