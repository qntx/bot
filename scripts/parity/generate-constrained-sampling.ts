// Parity vectors for AI-CONSTRAINED-SAMPLING: the strict subset of tool schemas
// (`makeStrictJsonSchema`) and the strict-mode decision
// (`resolveJsonSchemaStrictSampling`) from the published pi-ai package.
// Run with `node scripts/parity/generate-constrained-sampling.ts`.
import { writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

type MakeStrict = (schema: unknown) => unknown;
type Resolve = (tool: unknown, supportsStrictMode: boolean) => unknown;
type Module = { makeStrictJsonSchema: MakeStrict; resolveJsonSchemaStrictSampling: Resolve };

function isModule(value: unknown): value is Module {
  return (
    typeof value === "object" &&
    value !== null &&
    "makeStrictJsonSchema" in value &&
    typeof value.makeStrictJsonSchema === "function" &&
    "resolveJsonSchemaStrictSampling" in value &&
    typeof value.resolveJsonSchemaStrictSampling === "function"
  );
}

const piAiDir = dirname(dirname(fileURLToPath(import.meta.resolve("@earendil-works/pi-ai"))));
const loaded: unknown = await import(
  pathToFileURL(join(piAiDir, "dist/api/constrained-sampling.js")).href
);
if (!isModule(loaded)) {
  throw new Error("constrained-sampling dist module lacks the expected exports");
}
const { makeStrictJsonSchema, resolveJsonSchemaStrictSampling } = loaded;

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
writeFileSync(target, `${JSON.stringify(document, null, 2)}\n`);
console.log(`${target}: ${cases.length} cases`);
