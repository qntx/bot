/**
 * Strict JSON Lines framing for the ovo RPC stream.
 *
 * Records are split on LF (`\n`) only — U+2028/U+2029 inside a JSON string do not terminate a
 * record. One trailing CR before the LF is stripped (CRLF peers); any other `\r` is left for
 * `JSON.parse` to reject. Records that are empty or whitespace-only are skipped.
 */

/** A line that was framed correctly but did not parse as JSON. */
export class JsonlError extends Error {
  /** The offending record text, without its line terminator. */
  readonly line: string;

  constructor(line: string, options?: { cause?: unknown }) {
    super(`invalid JSONL record: ${JSON.stringify(truncate(line))}`, options);
    this.name = "JsonlError";
    this.line = line;
  }
}

const MAX_LINE_PREVIEW = 80;

function truncate(line: string): string {
  return line.length <= MAX_LINE_PREVIEW ? line : `${line.slice(0, MAX_LINE_PREVIEW)}…`;
}

function parseRecord(raw: string): unknown {
  const line = raw.endsWith("\r") ? raw.slice(0, -1) : raw;
  try {
    return JSON.parse(line) as unknown;
  } catch (error) {
    throw new JsonlError(line, { cause: error });
  }
}

/** Encodes one value as a JSONL record: `JSON.stringify(value)` plus LF. */
export function encodeJsonl(value: unknown): string {
  return `${JSON.stringify(value)}\n`;
}

/**
 * Incremental JSONL decoder. `push` accepts text or UTF-8 bytes — multibyte sequences may span
 * chunks — and returns the records completed so far. `end` flushes the decoder and parses a final
 * unterminated record.
 */
export class JsonlDecoder {
  readonly #decoder = new TextDecoder();
  #buffer = "";

  /** Feeds a chunk and returns every record completed by it. */
  push(chunk: string | Uint8Array): unknown[] {
    this.#buffer +=
      typeof chunk === "string" ? chunk : this.#decoder.decode(chunk, { stream: true });
    return this.#drain(false);
  }

  /** Flushes pending bytes and returns remaining records, including an unterminated tail. */
  end(): unknown[] {
    this.#buffer += this.#decoder.decode();
    return this.#drain(true);
  }

  #drain(flush: boolean): unknown[] {
    const records: unknown[] = [];
    let start = 0;
    for (;;) {
      const nl = this.#buffer.indexOf("\n", start);
      if (nl === -1) {
        break;
      }
      const raw = this.#buffer.slice(start, nl);
      if (raw.trim() !== "") {
        records.push(parseRecord(raw));
      }
      start = nl + 1;
    }
    this.#buffer = this.#buffer.slice(start);
    if (flush && this.#buffer.trim() !== "") {
      records.push(parseRecord(this.#buffer));
      this.#buffer = "";
    }
    return records;
  }
}
