/// <reference types="node" />
import { writeFileSync } from "node:fs";

// Parity vectors for AI-JSON-JS: ovo's JSON layer must round-trip these
// inputs to the exact bytes `JSON.stringify(JSON.parse(input))` produces.
// Run with `node scripts/parity/generate-js-json.ts` (Node strips the type
// annotations natively); regenerate only when upstream or Node changes.

type Case = { name: string; input: string; expected?: string; classes?: string[] };

const inputs: Array<{ name: string; input: string; classes?: string[] }> = [
  // Integers around the f64 safe-integer boundary.
  { name: "int-zero", input: "0" },
  { name: "int-neg-zero", input: "-0" },
  { name: "int-one", input: "1" },
  { name: "int-max-safe", input: "9007199254740991" },
  { name: "int-over-safe", input: "9007199254740992" },
  { name: "int-over-safe-plus-one", input: "9007199254740993" },
  { name: "int-big", input: "12345678901234567890" },
  { name: "int-big-neg", input: "-12345678901234567890" },
  { name: "int-u64-max", input: "18446744073709551615" },
  { name: "int-i64-min", input: "-9223372036854775808" },
  // Floats.
  { name: "float-one", input: "1.0" },
  { name: "float-neg-zero", input: "-0.0" },
  { name: "float-tenth", input: "0.1" },
  { name: "float-long", input: "0.30000000000000004" },
  { name: "float-1e21", input: "1e21" },
  { name: "float-1e-7", input: "1e-7" },
  { name: "float-small-exp", input: "123e-20" },
  { name: "float-min-subnormal", input: "5e-324" },
  { name: "float-max", input: "1.7976931348623157e308" },
  { name: "float-exp-plus", input: "1e+21" },
  { name: "float-trailing-zeros", input: "1.50000" },
  // Array-index-shaped object keys.
  { name: "keys-index-basic", input: '{"b":1,"2":3,"1":4}' },
  { name: "keys-index-zero", input: '{"a":1,"0":2}' },
  { name: "keys-index-ten", input: '{"10":1,"2":2,"a":3}' },
  { name: "keys-index-max", input: '{"x":1,"4294967294":2}' },
  { name: "keys-index-over", input: '{"x":1,"4294967295":2}' },
  { name: "keys-index-over-plus", input: '{"x":1,"4294967296":2}' },
  { name: "keys-leading-zero", input: '{"01":1,"1":2}' },
  { name: "keys-negative", input: '{"-1":1,"1":2}' },
  { name: "keys-decimal", input: '{"1.5":1,"1":2}' },
  { name: "keys-empty", input: '{"":1,"0":2}' },
  { name: "keys-index-then-string", input: '{"5":1,"b":2,"2":3,"a":4,"1":5}' },
  { name: "keys-nested-object", input: '{"a":{"b":1,"1":2},"2":{"z":0,"0":9}}' },
  { name: "keys-in-array", input: '[{"b":1,"0":5},{"9":1,"a":2}]' },
  { name: "keys-deep-nest", input: '{"0":{"1":{"10":{"2":"x"},"a":1},"b":2},"c":3}' },
  { name: "keys-index-value-object", input: '{"10":{"z":1},"2":{"a":2},"b":3}' },
  // Duplicate keys: last value wins, first position kept.
  { name: "keys-duplicate", input: '{"a":1,"b":2,"a":3}' },
  { name: "keys-duplicate-index", input: '{"2":1,"b":2,"2":3}' },
  // Escapes and control characters.
  { name: "str-escapes", input: String.raw`"\b\t\n\f\r\"\\\/"` },
  { name: "str-control-escape", input: String.raw`"\u0000\u001f"` },
  {
    name: "str-control-all",
    input: `"${Array.from({ length: 0x20 }, (_, i) => `\\u${i.toString(16).padStart(4, "0")}`).join("")}"`,
  },
  { name: "str-del", input: String.raw`"\u007f"` },
  { name: "str-line-separators", input: '"\u2028\u2029"' },
  { name: "str-slash-escape", input: String.raw`"a\/b"` },
  { name: "str-non-ascii", input: '"héllo wörld"' },
  { name: "str-astral-raw", input: '"𝄞😀"' },
  { name: "str-astral-escaped", input: String.raw`"\ud834\udd1e"` },
  { name: "str-non-ascii-escaped", input: String.raw`"\u00e9\u4e2d"` },
  { name: "str-mixed", input: String.raw`["\u0041A\u0000",{"k\u0022y":"v\u005cv"}]` },
  // Lone surrogate escapes (JSON.parse accepts; ovo repairs to U+FFFD).
  { name: "surrogate-lone-high", input: String.raw`"\ud800x"`, classes: ["lone-surrogates"] },
  { name: "surrogate-lone-low", input: String.raw`"\udc00"`, classes: ["lone-surrogates"] },
  { name: "surrogate-high-then-text", input: String.raw`"\ud800\n"`, classes: ["lone-surrogates"] },
  { name: "surrogate-two-highs", input: String.raw`"\ud800\ud800"`, classes: ["lone-surrogates"] },
  {
    name: "surrogate-lone-then-pair",
    input: String.raw`"\ud800\ud800\udc00"`,
    classes: ["lone-surrogates"],
  },
  { name: "surrogate-high-max", input: String.raw`"\udbff"`, classes: ["lone-surrogates"] },
  { name: "surrogate-low-min", input: String.raw`"\udc00\udc01"`, classes: ["lone-surrogates"] },
  {
    name: "surrogate-in-object",
    input: String.raw`{"key":"a\udfff","n":1}`,
    classes: ["lone-surrogates"],
  },
  // Escaped backslash before what looks like a surrogate escape.
  { name: "surrogate-after-backslash", input: String.raw`"\\ud800"` },
  // Structures and whitespace variants.
  { name: "struct-empty-object", input: "{}" },
  { name: "struct-empty-array", input: "[]" },
  { name: "struct-nested-empty", input: '{"a":[],"b":{}}' },
  { name: "struct-whitespace", input: '  {\n\t"a" : [ 1 , 2 ]\r\n}  ' },
  { name: "struct-literals", input: "[true,false,null]" },
  { name: "struct-string-doc", input: '"hello"' },
  { name: "struct-mixed", input: '{"a":[1,2.5,"x",{"0":"i","z":"s"}],"b":null}' },
  // input-limits: JS accepts, ovo rejects.
  { name: "limit-inf-positive", input: "1e400", classes: ["input-limits"] },
  { name: "limit-inf-negative", input: "-1e400", classes: ["input-limits"] },
  {
    name: "limit-depth-129",
    input: "[".repeat(129) + "]".repeat(129),
    classes: ["input-limits"],
  },
  {
    name: "limit-depth-200",
    input: "[".repeat(200) + "]".repeat(200),
    classes: ["input-limits"],
  },
];

const cases: Case[] = inputs.map(({ name, input, classes }) => {
  const caseEntry: Case = {
    name,
    input,
    expected: JSON.stringify(JSON.parse(input)),
  };
  if (classes !== undefined) {
    caseEntry.classes = classes;
  }
  return caseEntry;
});

const document = {
  generator: { script: "scripts/parity/generate-js-json.ts", node: process.version },
  cases,
};

const output = `${JSON.stringify(document, null, 2)}\n`;
const target = "vectors/ai/js-json.json";
writeFileSync(target, output);
console.log(`${target}: ${cases.length} cases`);
