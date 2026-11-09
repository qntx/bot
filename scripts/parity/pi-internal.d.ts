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

export declare function retryProviderRequest(
  request: () => Promise<unknown>,
  options?: {
    maxRetries?: number | undefined;
    maxRetryDelayMs?: number | undefined;
    signal?: AbortSignal | undefined;
    noRetryStatuses?: ReadonlyArray<number> | undefined;
  },
): Promise<unknown>;

export declare function normalizeProviderError(value: unknown): {
  status?: number;
  body?: string;
  message: string;
  messageCarriesBody: boolean;
};

export declare function formatProviderError(
  norm: {
    status?: number;
    body?: string;
    message: string;
    messageCarriesBody: boolean;
  },
  prefix?: string,
): string;

export declare function truncateErrorText(text: string, maxChars: number): string;

export declare function providerHeadersToRecord(
  ...headerSources: Array<Record<string, unknown> | undefined>
): Record<string, string> | undefined;

export declare function resolveHttpProxyUrlForTarget(
  targetUrl: string,
  env?: Record<string, string>,
): { href: string } | undefined;

export declare const UNSUPPORTED_PROXY_PROTOCOL_MESSAGE: string;

export declare function findEnvKeys(
  provider: string,
  env?: Record<string, string>,
): string[] | undefined;

export declare function getEnvApiKey(
  provider: string,
  env?: Record<string, string>,
): string | undefined;

export declare function getConfigValueEnvVarName(config: string): string | undefined;

export declare function getConfigValueEnvVarNames(config: string): string[];

export declare function getMissingConfigValueEnvVarNames(
  config: string,
  env?: Record<string, string>,
): string[];

export declare function isCommandConfigValue(config: string): boolean;

export declare function isConfigValueConfigured(
  config: string,
  env?: Record<string, string>,
): boolean;

export declare function resolveConfigValue(
  config: string,
  env?: Record<string, string>,
): string | undefined;

interface AuthCredentialLike {
  type: string;
  [key: string]: unknown;
}

export declare class ReadOnlyAuthStorage {
  constructor(authPath?: string);
  read(providerId: string, options?: unknown): Promise<AuthCredentialLike | undefined>;
  list(options?: unknown): Promise<ReadonlyArray<{ providerId: string; type: string }>>;
  modify(
    providerId: string,
    fn: (current: unknown) => Promise<unknown>,
    options?: unknown,
  ): Promise<unknown>;
  delete(providerId: string, options?: unknown): Promise<void>;
}

export declare class AuthStorage {
  static create(authPath?: string): AuthStorage;
  static inMemory(data?: Record<string, unknown>): AuthStorage;
  read(providerId: string, options?: unknown): Promise<AuthCredentialLike | undefined>;
  list(options?: unknown): Promise<ReadonlyArray<{ providerId: string; type: string }>>;
  modify(
    providerId: string,
    fn: (current: unknown) => Promise<unknown>,
    options?: unknown,
  ): Promise<unknown>;
  delete(providerId: string, options?: unknown): Promise<void>;
}
