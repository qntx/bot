// Typed facades for the upstream functions re-exported by pi-internal.js.
// Arguments are `unknown` so call sites pass package-typed values without
// widening; results land in `Case.result`, which is `unknown` already.
export declare function transformMessages(
  messages: unknown,
  model: unknown,
  normalizeToolCallId?: unknown,
): unknown;

export declare function splitErrorToolResultImages(message: unknown): unknown;

export declare function sanitizeSurrogates(text: string): string;

export declare function ownedSlice(text: string, start: number, end?: number): string;
