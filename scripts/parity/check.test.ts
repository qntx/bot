import { describe, expect, test } from "vite-plus/test";

import { checkParity, checkUpstreamTests } from "./check.ts";
import type { ParityFs } from "./check.ts";

function fsWith(files: Record<string, string>): ParityFs {
  return {
    fileExists: (path) => path in files,
    fileText: (path) => files[path],
  };
}

const CAPABILITY = {
  id: "AI-JSON-JS",
  upstream: ["packages/coding-agent/src/core/session-manager.ts"],
  vectors: ["vectors/ai/js-json.json"],
  rust: ["crates/ovo-ai/tests/it/json.rs"],
  classes: ["lone-surrogates", "input-limits"],
};

const FILES: Record<string, string> = {
  "vectors/ai/js-json.json": JSON.stringify({ cases: [{ name: "x", input: "1", expected: "1" }] }),
  "crates/ovo-ai/tests/it/json.rs": "/// parity: AI-JSON-JS\n#[test] fn t() {}",
  "3rdparty/pi/packages/coding-agent/src/core/session-manager.ts": "export {}",
};

function parity(overrides: Record<string, unknown> = {}) {
  return {
    classes: { "lone-surrogates": "a", "input-limits": "b" },
    capabilities: [CAPABILITY],
    ...overrides,
  };
}

describe("checkParity", () => {
  test("accepts a valid parity.json", () => {
    expect(checkParity(parity(), fsWith(FILES))).toStrictEqual([]);
  });

  test("rejects bad and duplicate ids", () => {
    const bad = { ...CAPABILITY, id: "ai-json-js" };
    expect(checkParity(parity({ capabilities: [bad] }), fsWith(FILES))).toStrictEqual([
      expect.stringContaining("ai-json-js: id must match"),
      expect.stringContaining('ai-json-js: rust file lacks "parity: ai-json-js" marker'),
    ]);
    const dup = { capabilities: [CAPABILITY, CAPABILITY] };
    expect(checkParity(parity(dup), fsWith(FILES))).toStrictEqual([
      expect.stringContaining("duplicate capability id"),
    ]);
  });

  test("rejects a missing vector file", () => {
    const files = { ...FILES };
    delete files["vectors/ai/js-json.json"];
    expect(checkParity(parity(), fsWith(files))).toStrictEqual([
      "AI-JSON-JS: vector file missing: vectors/ai/js-json.json",
    ]);
  });

  test("rejects a missing rust file and a missing parity marker", () => {
    const files = { ...FILES };
    delete files["crates/ovo-ai/tests/it/json.rs"];
    expect(checkParity(parity(), fsWith(files))).toStrictEqual([
      "AI-JSON-JS: rust file missing: crates/ovo-ai/tests/it/json.rs",
    ]);
    const noMarker = { ...FILES, "crates/ovo-ai/tests/it/json.rs": "// no marker here" };
    expect(checkParity(parity(), fsWith(noMarker))).toStrictEqual([
      'AI-JSON-JS: rust file lacks "parity: AI-JSON-JS" marker: crates/ovo-ai/tests/it/json.rs',
    ]);
  });

  test("rejects undefined capability classes and unused defined classes", () => {
    const bad = { ...CAPABILITY, classes: [...CAPABILITY.classes, "bogus"] };
    const errors = checkParity(parity({ capabilities: [bad] }), fsWith(FILES));
    expect(errors).toStrictEqual([
      'AI-JSON-JS: class "bogus" is not defined in parity.json classes',
    ]);

    const definedOnly = parity({
      classes: { "lone-surrogates": "a", "input-limits": "b", extra: "c" },
    });
    expect(checkParity(definedOnly, fsWith(FILES))).toStrictEqual([
      'class "extra" is defined but never used',
    ]);
  });

  test("rejects vector case classes outside the capability classes", () => {
    const files = {
      ...FILES,
      "vectors/ai/js-json.json": JSON.stringify({
        cases: [{ name: "x", input: "1", expected: "1", classes: ["unordered-paths"] }],
      }),
    };
    expect(checkParity(parity(), fsWith(files))).toStrictEqual([
      'AI-JSON-JS: vector vectors/ai/js-json.json case uses undeclared class "unordered-paths"',
    ]);
  });

  test("--upstream checks upstream paths", () => {
    const files = { ...FILES };
    delete files["3rdparty/pi/packages/coding-agent/src/core/session-manager.ts"];
    expect(checkParity(parity(), fsWith(files))).toStrictEqual([]);
    expect(checkParity(parity(), fsWith(files), true)).toStrictEqual([
      "AI-JSON-JS: upstream file missing: packages/coding-agent/src/core/session-manager.ts",
    ]);
  });
});

describe("checkUpstreamTests", () => {
  const inventory = {
    commit: "abc",
    files: {
      "packages/ai/test/a.test.ts": [],
      "packages/agent/test/b.test.ts": [],
      "packages/agent/test/c.test.ts": [],
    },
  };
  const owned = (upstreamTests: Record<string, unknown>) => ({
    upstream: { commit: "abc" },
    capabilities: [CAPABILITY],
    upstreamTests,
  });

  test("accepts file and directory owners", () => {
    const owners = {
      "packages/ai/test/a.test.ts": ["AI-JSON-JS"],
      "packages/agent/test/": { milestone: "M2" },
    };
    expect(checkUpstreamTests(owned(owners), inventory)).toStrictEqual([]);
  });

  test("rejects unowned files, stale entries and a commit mismatch", () => {
    const owners = { "packages/ai/test/gone.test.ts": { e2e: "live" } };
    expect(
      checkUpstreamTests({ ...owned(owners), upstream: { commit: "def" } }, inventory),
    ).toStrictEqual([
      "upstream test inventory is of abc, parity.json pins def",
      "upstream test packages/ai/test/a.test.ts has no owner in parity.json upstreamTests",
      "upstream test packages/agent/test/b.test.ts has no owner in parity.json upstreamTests",
      "upstream test packages/agent/test/c.test.ts has no owner in parity.json upstreamTests",
      "upstreamTests entry packages/ai/test/gone.test.ts owns no listed upstream test",
    ]);
  });

  test("rejects invalid owners", () => {
    const owners = {
      "packages/ai/test/a.test.ts": ["AI-NOPE"],
      "packages/agent/test/b.test.ts": { milestone: "later" },
      "packages/agent/test/c.test.ts": { excluded: "why", milestone: "M2" },
    };
    expect(checkUpstreamTests(owned(owners), inventory)).toStrictEqual([
      "upstreamTests entry packages/ai/test/a.test.ts names capabilities not in parity.json: AI-NOPE",
      'upstreamTests entry packages/agent/test/b.test.ts needs "e2e", "excluded" or a "milestone" like "M5"',
      "upstreamTests entry packages/agent/test/c.test.ts takes no other fields",
    ]);
  });
});
