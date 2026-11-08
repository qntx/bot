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

export declare function calculateContextTokens(usage: unknown): number;

export declare function estimateContextTokens(context: unknown): unknown;

export declare function estimateMessageTokens(message: unknown): number;

export declare function estimateTextAndImageContentTokens(content: unknown): number;

export declare function estimateTextTokens(text: string): number;

export declare function clampMaxTokensToContext(
  model: unknown,
  context: unknown,
  maxTokens: number,
): number;

export declare function resolveSamplingParams(
  model: unknown,
  thinkingLevel: unknown,
  requestParams?: unknown,
): unknown;

export declare function clampReasoning(effort: unknown): unknown;

export declare function thinkingBudgetForLevel(
  reasoningLevel: unknown,
  customBudgets?: unknown,
): number;

export declare function clampThinkingBudgetToAnswerRoom(
  thinkingBudget: number,
  ceiling: number,
): number;

export declare function adjustMaxTokensForThinking(
  baseMaxTokens: unknown,
  modelMaxTokens: number,
  reasoningLevel: unknown,
  customBudgets?: unknown,
): unknown;

export declare const DEFAULT_THINKING_BUDGETS: unknown;
