// Generates vectors/ai/schema-coerce.json and schema-validate.json:
// pi-schema coercion (compile + coerce) and tool-argument validation
// (pi-ai validateToolArguments shape) cases computed by upstream.

import { writeFileSync } from "node:fs";

import { compile } from "@earendil-works/pi-schema";
import type { JsonSchema } from "@earendil-works/pi-schema";

import { formats } from "../../node_modules/@earendil-works/pi-schema/dist/formats/index.js";

type CoerceCase = {
  name: string;
  schema: JsonSchema;
  input: unknown;
  expected: unknown;
};

type ValidateCase = {
  name: string;
  tool: string;
  schema: JsonSchema;
  arguments: Record<string, unknown>;
  expected:
    | { kind: "ok"; arguments: Record<string, unknown> }
    | { kind: "error"; errors: Array<{ path: string; keyword: string; message: string }> };
};

function args_record(value: unknown): Record<string, unknown> {
  if (typeof value === "object" && value !== null && !Array.isArray(value)) {
    return Object.fromEntries(Object.entries(value));
  }
  return {};
}

const compiled = (schema: JsonSchema) =>
  compile(schema, {
    formats,
    coerce: true,
    unresolvedRefs: "reject",
  });

const coerceCases: CoerceCase[] = [];
const validateCases: ValidateCase[] = [];

function coerce(name: string, schema: JsonSchema, input: unknown): void {
  const validator = compiled(schema);
  coerceCases.push({ name, schema, input, expected: validator.coerce(input) });
}

function validate(
  name: string,
  schema: JsonSchema,
  tool: string,
  args: Record<string, unknown>,
): void {
  const validator = compiled(schema);
  const coerced = validator.coerce(args);
  const ok = validator.check(coerced);
  if (ok) {
    validateCases.push({
      name,
      tool,
      schema,
      arguments: args,
      expected: { kind: "ok", arguments: args_record(coerced) },
    });
    return;
  }
  const errors = validator.errors(coerced).map((e) => ({
    path: e.path,
    keyword: e.keyword,
    message: e.message,
  }));
  validateCases.push({ name, tool, schema, arguments: args, expected: { kind: "error", errors } });
}

// Coercion: every primitive conversion -----------------------------------
coerce("string-from-number", { type: "string" }, 42);
coerce("string-from-bool", { type: "string" }, true);
coerce("string-from-null", { type: "string" }, null);
coerce("string-passthrough", { type: "string" }, "x");
coerce("number-from-string", { type: "number" }, "42");
coerce("number-from-float-string", { type: "number" }, "3.5");
coerce("number-from-bool", { type: "number" }, true);
coerce("number-from-null", { type: "number" }, null);
coerce("number-from-empty-string", { type: "number" }, "");
coerce("number-invalid-string", { type: "number" }, "abc");
coerce("integer-truncate", { type: "integer" }, "3.9");
coerce("integer-neg-truncate", { type: "integer" }, "-3.9");
coerce("integer-passthrough", { type: "integer" }, 3);
coerce("boolean-true-string", { type: "boolean" }, "true");
coerce("boolean-TRUE-string", { type: "boolean" }, "TRUE");
coerce("boolean-1-string", { type: "boolean" }, "1");
coerce("boolean-0-number", { type: "boolean" }, 0);
coerce("boolean-from-null", { type: "boolean" }, null);
coerce("null-from-empty-string", { type: "null" }, "");
coerce("null-from-undefined-string", { type: "null" }, "undefined");
coerce("null-from-zero", { type: "null" }, 0);
coerce("null-from-false", { type: "null" }, false);
coerce("null-passthrough", { type: "null" }, null);
// Scalar to array
coerce("scalar-to-array", { type: "array", items: { type: "number" } }, "7");
coerce("number-to-array", { type: "array", items: { type: "number" } }, 7);
coerce("array-passthrough", { type: "array", items: { type: "number" } }, [1, "2"]);
// Optional null drop
coerce(
  "optional-null-drop",
  { type: "object", properties: { a: { type: "string" } } },
  { a: null },
);
coerce(
  "required-null-kept",
  { type: "object", properties: { a: { type: "string" } }, required: ["a"] },
  { a: null },
);
coerce(
  "nullable-null-kept",
  { type: "object", properties: { a: { type: ["string", "null"] } } },
  { a: null },
);
coerce(
  "nested-null-drop",
  { type: "object", properties: { o: { type: "object", properties: { n: { type: "number" } } } } },
  { o: { n: null } },
);
// Nested objects/arrays
coerce(
  "nested-object",
  { type: "object", properties: { o: { type: "object", properties: { n: { type: "number" } } } } },
  { o: { n: "5" } },
);
coerce(
  "nested-array",
  { type: "object", properties: { l: { type: "array", items: { type: "number" } } } },
  { l: ["1", 2] },
);
coerce("tuple", { type: "array", prefixItems: [{ type: "string" }, { type: "number" }] }, [
  "a",
  "2",
]);
// Unions in both orders
coerce("union-first-match", { anyOf: [{ type: "string" }, { type: "number" }] }, "42");
coerce("union-second-match", { anyOf: [{ type: "number" }, { type: "string" }] }, "42");
coerce("union-preserve", { anyOf: [{ type: "number" }, { type: "string" }] }, "abc");
coerce("oneof", { oneOf: [{ type: "number" }, { type: "string" }] }, "42");
coerce("allof", { allOf: [{ type: "number" }, { minimum: 1 }] }, "5");
// $ref local and dangling
coerce(
  "ref-local",
  { $defs: { n: { type: "number" } }, properties: { x: { $ref: "#/$defs/n" } }, type: "object" },
  { x: "5" },
);
coerce(
  "ref-dangling",
  { type: "object", properties: { x: { $ref: "#/missing" }, y: { type: "number" } } },
  { x: 1, y: "5" },
);
coerce(
  "ref-recursive",
  { type: "object", properties: { next: { $ref: "#" }, v: { type: "number" } } },
  { v: "1", next: { v: "2" } },
);
// enum/const
coerce("const-string", { const: "ok" }, "ok");
coerce("const-number", { const: 1 }, "1");
coerce("const-mismatch", { const: 1 }, "2");
coerce("enum-member", { enum: [1, "two", null] }, "two");
coerce("enum-coerce-to", { enum: [1, "two"] }, "1");
// additionalProperties
coerce(
  "additional-coerced",
  { type: "object", additionalProperties: { type: "number" } },
  { a: "1" },
);
coerce(
  "pattern-properties",
  { type: "object", patternProperties: { "^n_": { type: "number" } } },
  { n_a: "1" },
);
// TypeBox common model mistakes
coerce("json-string-array", { type: "array", items: { type: "string" } }, '["a","b"]');
coerce(
  "number-string-number",
  { type: "object", properties: { timeout: { type: "number" } } },
  { timeout: "30" },
);

// Validation: built-in tool schemas + model mistakes ----------------------
const readSchema = {
  type: "object",
  properties: {
    path: { type: "string", description: "Path to the file to read" },
    offset: { type: "number", description: "Line offset to start reading" },
    limit: { type: "number", description: "Number of lines to read" },
  },
  required: ["path"],
};
const bashSchema = {
  type: "object",
  properties: {
    command: { type: "string" },
    timeout: { type: "number" },
  },
  required: ["command"],
};
const editSchema = {
  type: "object",
  properties: {
    path: { type: "string" },
    edits: {
      type: "array",
      items: {
        type: "object",
        properties: { oldText: { type: "string" }, newText: { type: "string" } },
        required: ["oldText", "newText"],
      },
    },
  },
  required: ["path", "edits"],
};
const writeSchema = {
  type: "object",
  properties: { path: { type: "string" }, content: { type: "string" } },
  required: ["path", "content"],
};
const grepSchema = {
  type: "object",
  properties: {
    pattern: { type: "string" },
    path: { type: "string" },
    glob: { type: "string" },
    ignoreCase: { type: "boolean" },
    literal: { type: "boolean" },
    context: { type: "number" },
    limit: { type: "number" },
  },
  required: ["pattern"],
};
const findSchema = {
  type: "object",
  properties: { pattern: { type: "string" }, path: { type: "string" }, limit: { type: "number" } },
  required: ["pattern"],
};
const lsSchema = {
  type: "object",
  properties: { path: { type: "string" }, limit: { type: "number" } },
  required: [],
};

validate("read-ok", readSchema, "read", { path: "x.ts" });
validate("read-coerced-limit", readSchema, "read", { path: "x.ts", limit: "50" });
validate("read-missing-path", readSchema, "read", { offset: 3 });
validate("read-limit-object", readSchema, "read", { path: "x", limit: { deep: 1 } });
validate("bash-ok", bashSchema, "bash", { command: "ls" });
validate("bash-timeout-string", bashSchema, "bash", { command: "ls", timeout: "30" });
validate("edit-ok", editSchema, "edit", { path: "f", edits: [{ oldText: "a", newText: "b" }] });
validate("edit-edits-jsonstring", editSchema, "edit", {
  path: "f",
  edits: '[{"oldText":"a","newText":"b"}]',
});
validate("edit-edits-scalar", editSchema, "edit", { path: "f", edits: "x" });
validate("edit-missing-newtext", editSchema, "edit", { path: "f", edits: [{ oldText: "a" }] });
validate("write-ok", writeSchema, "write", { path: "f", content: "c" });
validate("write-missing-content", writeSchema, "write", { path: "f" });
validate("grep-coerced", grepSchema, "grep", { pattern: "x", ignoreCase: "true", limit: "10" });
validate("find-ok", findSchema, "find", { pattern: "*.ts" });
validate("ls-ok", lsSchema, "ls", {});
validate(
  "dangling-ref",
  { type: "object", properties: { x: { $ref: "#/nope" }, y: { type: "number" } }, required: ["y"] },
  "tool",
  { x: 1, y: 2 },
);
validate(
  "union-error-dotted",
  {
    type: "object",
    properties: {
      edits: {
        type: "array",
        items: {
          type: "object",
          properties: { oldText: { type: "string" } },
          required: ["oldText"],
        },
      },
    },
  },
  "tool",
  { edits: [{}] },
);
validate(
  "enum-error",
  { type: "object", properties: { mode: { enum: ["a", "b"] } }, required: ["mode"] },
  "tool",
  { mode: "c" },
);
validate(
  "additional-false",
  { type: "object", properties: { x: { type: "number" } }, additionalProperties: false },
  "tool",
  { x: 1, z: 2 },
);
validate(
  "format-date",
  { type: "object", properties: { d: { type: "string", format: "date" } } },
  "tool",
  { d: "not-a-date" },
);
validate(
  "ref-local-invalid",
  { $defs: { n: { type: "number" } }, type: "object", properties: { x: { $ref: "#/$defs/n" } } },
  "tool",
  { x: {} },
);
validate(
  "required-missing-multi",
  {
    type: "object",
    properties: { a: { type: "string" }, b: { type: "number" } },
    required: ["a", "b"],
  },
  "tool",
  {},
);

writeFileSync("vectors/ai/schema-coerce.json", `${JSON.stringify(coerceCases, null, 2)}\n`);
writeFileSync("vectors/ai/schema-validate.json", `${JSON.stringify(validateCases, null, 2)}\n`);
console.log(`coerce=${coerceCases.length} validate=${validateCases.length}`);
