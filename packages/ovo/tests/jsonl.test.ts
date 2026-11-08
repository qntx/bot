import { describe, expect, test } from "vite-plus/test";

import { JsonlDecoder, JsonlError, encodeJsonl } from "../src/index.ts";

describe("encodeJsonl", () => {
  test("appends a single LF", () => {
    expect(encodeJsonl({ a: 1 })).toBe('{"a":1}\n');
    expect(encodeJsonl("x")).toBe('"x"\n');
  });
});

describe("JsonlDecoder", () => {
  test("does not split on U+2028/U+2029 inside a string", () => {
    const value = { text: "a\u2028b\u2029c" };
    const decoder = new JsonlDecoder();
    expect(decoder.push(encodeJsonl(value))).toStrictEqual([value]);
  });

  test("strips one CR from CRLF records", () => {
    const decoder = new JsonlDecoder();
    expect(decoder.push('{"a":1}\r\n{"b":2}\r\n')).toStrictEqual([{ a: 1 }, { b: 2 }]);
  });

  test("assembles a record spread over three chunks", () => {
    const decoder = new JsonlDecoder();
    expect(decoder.push('{"he')).toStrictEqual([]);
    expect(decoder.push('llo":"wor')).toStrictEqual([]);
    expect(decoder.push('ld"}\n')).toStrictEqual([{ hello: "world" }]);
  });

  test("assembles a 4-byte emoji split across byte chunks", () => {
    const bytes = new TextEncoder().encode('{"emoji":"😀"}\n');
    const decoder = new JsonlDecoder();
    const out: unknown[] = [];
    for (const byte of bytes) {
      out.push(...decoder.push(Uint8Array.of(byte)));
    }
    expect(out).toStrictEqual([{ emoji: "😀" }]);
  });

  test("parses multiple records from one chunk", () => {
    const decoder = new JsonlDecoder();
    expect(decoder.push('{"a":1}\n{"b":2}\n')).toStrictEqual([{ a: 1 }, { b: 2 }]);
  });

  test("flushes an unterminated tail on end", () => {
    const decoder = new JsonlDecoder();
    expect(decoder.push('{"a":1}')).toStrictEqual([]);
    expect(decoder.end()).toStrictEqual([{ a: 1 }]);
  });

  test("end ignores an empty or whitespace-only tail", () => {
    expect(new JsonlDecoder().end()).toStrictEqual([]);
    const decoder = new JsonlDecoder();
    decoder.push("  \t");
    expect(decoder.end()).toStrictEqual([]);
  });

  test("skips empty lines between records", () => {
    const decoder = new JsonlDecoder();
    expect(decoder.push('\n{"a":1}\n\n\n{"b":2}\n')).toStrictEqual([{ a: 1 }, { b: 2 }]);
  });

  test("throws JsonlError carrying the offending line on invalid JSON", () => {
    const decoder = new JsonlDecoder();
    let thrown: unknown;
    try {
      decoder.push("{oops}\n");
    } catch (error) {
      thrown = error;
    }
    expect(thrown).toBeInstanceOf(JsonlError);
    expect((thrown as JsonlError).line).toBe("{oops}");
  });
});
