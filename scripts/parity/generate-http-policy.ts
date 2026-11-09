// Generates vectors/ai/http-*.json from the REAL upstream code:
// - pi-ai dist functions (provider-retry, error-body, headers, node-http-proxy)
//   via pi-internal.js
// - @anthropic-ai/sdk 0.129.0 and openai 7.19.0 client instances (shouldRetry,
//   retryRequest driven end-to-end with stubbed timers) — located inside bun's
//   isolated node_modules/.bun layout because they are transitive deps.
// Determinism: Math.random() = 0.5 (jitter 0.875), Date.now() = NOW, process
// proxy env vars removed.
import { readdirSync, writeFileSync } from "node:fs";
import { pathToFileURL } from "node:url";

import {
  formatProviderError,
  normalizeProviderError,
  providerHeadersToRecord,
  resolveHttpProxyUrlForTarget,
  retryProviderRequest,
} from "./pi-internal.js";

const NOW = 1_700_000_000_000;
const realRandom = Math.random;
const realNow = Date.now;
const realSetTimeout = globalThis.setTimeout;
Math.random = () => 0.5;
Date.now = () => NOW;
// Capture the delay a sleep asks for and resolve instantly.
const sleeps: number[] = [];
// oxlint-disable-next-line no-unsafe-type-assertion -- stub over the global timer type
globalThis.setTimeout = ((fn: () => void, ms?: number) => {
  sleeps.push(ms ?? 0);
  fn();
  return 0;
}) as unknown as typeof setTimeout;
for (const key of Object.keys(process.env)) {
  if (/(?:^|_)(?:https?_proxy|all_proxy|no_proxy)$/i.test(key)) {
    Reflect.deleteProperty(process.env, key);
  }
}

// Locate a package inside bun's isolated store.
function findPackage(name: string): string {
  const dir = `node_modules/.bun`;
  for (const entry of readdirSync(dir)) {
    if (!entry.startsWith(`${name.replace("/", "+")}@`)) {
      continue;
    }
    return `${dir}/${entry}/node_modules/${name}/client.js`;
  }
  throw new Error(`${name} not found in .bun`);
}

// --- pi provider-retry: drive retryProviderRequest end to end ---

type ProviderCase = {
  name: string;
  status: number | undefined;
  headers: Record<string, string>;
  max_retries: number;
  no_retry_statuses: number[];
  max_retry_delay_ms: number | undefined;
  now_ms: number;
  expected: unknown;
};

async function providerCase(
  name: string,
  status: number | undefined,
  headers: Record<string, string>,
  maxRetries: number,
  noRetry: number[],
  cap: number | undefined,
): Promise<ProviderCase> {
  sleeps.length = 0;
  const request = () => {
    const error = new Error(`${status ?? "conn"} failure`);
    Object.assign(error, { status, headers: new Headers(headers) });
    throw error;
  };
  let expected: Record<string, unknown>;
  try {
    await retryProviderRequest(request, {
      maxRetries,
      maxRetryDelayMs: cap,
      noRetryStatuses: noRetry,
    });
    expected = { outcome: "threw" };
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    expected = message.startsWith("Server requested")
      ? { outcome: "cap", message }
      : { outcome: "gave_up", message };
  }
  return {
    name,
    status,
    headers,
    max_retries: maxRetries,
    no_retry_statuses: noRetry,
    max_retry_delay_ms: cap,
    now_ms: NOW,
    expected: { ...expected, delays: [...sleeps] },
  };
}

// --- Stainless SDKs: real client instances ---

type RetryEntry = Record<string, unknown>;

async function sdkShouldRetry(
  client: RetryEntry,
  status: number,
  headers: Record<string, string>,
): Promise<boolean> {
  const response = new Response("x", {
    status,
    headers,
  });
  // oxlint-disable-next-line no-unsafe-type-assertion -- SDK method signature from vendored source
  const shouldRetry = client["shouldRetry"] as (
    r: Response,
    o: Record<string, unknown>,
  ) => Promise<boolean>;
  return shouldRetry.call(client, response, {});
}

// Drives the real retryRequest once: the SDK parses retry-after(-ms), checks
// its validity window, sleeps via the (stubbed) timer and calls makeRequest —
// which we stub to observe the follow-up attempt.
async function sdkDelay(
  client: RetryEntry,
  headers: Record<string, string>,
  retriesRemaining: number,
  maxRetries: number,
): Promise<number> {
  sleeps.length = 0;
  let sawRequest = false;
  const original = client["makeRequest"];
  client["makeRequest"] = () => {
    sawRequest = true;
    return null;
  };
  // oxlint-disable-next-line no-unsafe-type-assertion -- SDK method signature from vendored source
  const retryRequest = client["retryRequest"] as (
    o: Record<string, unknown>,
    r: number,
    l: string,
    h: Headers,
  ) => Promise<unknown>;
  try {
    await retryRequest.call(client, { maxRetries }, retriesRemaining, "log", new Headers(headers));
  } finally {
    client["makeRequest"] = original;
  }
  if (!sawRequest) {
    throw new Error("retryRequest did not call makeRequest");
  }
  return sleeps[0] ?? -1;
}

const retryCases: unknown[] = [];
// oxlint-disable-next-line no-unsafe-assignment -- dynamic import of the vendored SDK returns any
const anthropicClient = await import(pathToFileURL(findPackage("@anthropic-ai/sdk")).href);
const anthropic =
  new // oxlint-disable-next-line no-unsafe-type-assertion -- SDK class from vendored module
  (anthropicClient as { Anthropic: new (o: { apiKey: string }) => RetryEntry }).Anthropic({
    apiKey: "x",
  });
// oxlint-disable-next-line no-unsafe-assignment -- dynamic import of the vendored SDK returns any
const openaiClient = await import(pathToFileURL(findPackage("openai")).href);
const openai =
  new // oxlint-disable-next-line no-unsafe-type-assertion -- SDK class from vendored module
  (openaiClient as { OpenAI: new (o: { apiKey: string }) => RetryEntry }).OpenAI({
    apiKey: "x",
  });
const sdkClients: Record<string, RetryEntry> = { anthropic, openai };

async function buildRetryCases() {
  for (const status of [undefined, 200, 400, 408, 409, 429, 500, 503] as const) {
    // oxlint-disable-next-line no-await-in-loop -- cases share the stubbed timer
    retryCases.push(await providerCase(`status-${status ?? "none"}`, status, {}, 1, [], undefined));
  }
  retryCases.push(
    await providerCase("should-retry-true", 400, { "x-should-retry": "true" }, 1, [], undefined),
    await providerCase("should-retry-false", 429, { "x-should-retry": "false" }, 1, [], undefined),
    await providerCase("retry-after-ms", 429, { "retry-after-ms": "2500" }, 1, [], undefined),
    await providerCase("retry-after-seconds", 429, { "retry-after": "3" }, 1, [], undefined),
    await providerCase(
      "retry-after-date",
      429,
      { "retry-after": new Date(NOW + 5000).toUTCString() },
      1,
      [],
      undefined,
    ),
    await providerCase("retry-after-cap", 429, { "retry-after": "120" }, 1, [], undefined),
    await providerCase("retry-after-cap-disabled", 429, { "retry-after": "120" }, 1, [], 0),
    await providerCase("retry-after-custom-cap", 429, { "retry-after": "30" }, 1, [], 20_000),
    await providerCase("no-retry-statuses", 429, {}, 1, [429], undefined),
    await providerCase("max-retries-0", 429, {}, 0, [], undefined),
    await providerCase("backoff-sequence", 500, {}, 4, [], undefined),
    await providerCase("backoff-exhausts", 500, {}, 2, [], undefined),
  );

  for (const [family, client] of Object.entries(sdkClients)) {
    for (const [name, status, headers] of [
      ["should-retry-true", 400, { "x-should-retry": "true" }],
      ["should-retry-false", 429, { "x-should-retry": "false" }],
      ["400", 400, {}],
      ["408", 408, {}],
      ["409", 409, {}],
      ["429", 429, {}],
      ["500", 500, {}],
      ["503", 503, {}],
    ] as const) {
      retryCases.push({
        name: `${family}-retry-${name}`,
        family,
        kind: "should_retry",
        status,
        headers,
        // oxlint-disable-next-line no-await-in-loop -- cases share the stubbed timer
        expected: await sdkShouldRetry(client, status, headers),
      });
    }
    for (const [name, headers, remaining, maxRetries] of [
      ["after-ms", { "retry-after-ms": "1500" }, 2, 2],
      ["after-seconds", { "retry-after": "4" }, 2, 2],
      ["after-huge", { "retry-after": "999999" }, 2, 2],
      ["after-zero", { "retry-after": "0" }, 2, 2],
      ["after-past", { "retry-after": new Date(NOW - 1000).toUTCString() }, 2, 2],
      ["after-date", { "retry-after": new Date(NOW + 5000).toUTCString() }, 2, 2],
      ["backoff-first", {}, 2, 2],
      ["backoff-second", {}, 1, 2],
      ["backoff-more", {}, 1, 8],
    ] as const) {
      retryCases.push({
        name: `${family}-delay-${name}`,
        family,
        kind: "delay",
        headers,
        retries_remaining: remaining,
        max_retries: maxRetries,
        now_ms: NOW,
        // oxlint-disable-next-line no-await-in-loop -- cases share the stubbed timer
        expected: await sdkDelay(client, headers, remaining, maxRetries),
      });
    }
  }
}
await buildRetryCases();

// --- Error bodies: real normalizeProviderError/formatProviderError ---
// normalizeProviderError needs `error instanceof Error` — fields are own
// enumerable props (matching how SDKs attach them).
function errorBodyCase(
  name: string,
  message: string,
  fields: Record<string, unknown>,
  prefix?: string,
) {
  const error = Object.assign(new Error(message), fields);
  const norm = normalizeProviderError(error) as Record<string, unknown>;
  const c: Record<string, unknown> = { name, message, fields, expected: norm };
  if (prefix !== undefined) {
    c["expectedFormat"] = formatProviderError(normalizeProviderError(error), prefix);
    c["formatArgs"] = { prefix };
  }
  return c;
}
const errorCases = [
  errorBodyCase("anthropic-message", "429 rate limited", { status: 429 }),
  errorBodyCase("mistral-body-string", "403 status code (no body)", {
    statusCode: 403,
    body: "forbidden by gateway",
  }),
  errorBodyCase("openai-error-object", "Error", { status: 400, error: { code: "bad", type: "x" } }),
  errorBodyCase("bedrock-metadata", "throttled", { $metadata: { httpStatusCode: 503 } }),
  errorBodyCase("response-body-object", "err", {
    status: 500,
    $response: { statusCode: 502, body: "gateway down" },
  }),
  errorBodyCase("empty-body", "e", { status: 500, body: "   " }),
  errorBodyCase("long-body", "e", { status: 500, body: "x".repeat(4100) }),
  errorBodyCase("format-prefix", "m", { status: 429, body: "too many" }, "Anthropic"),
  errorBodyCase("format-carries", "429 too many", { status: 429, body: "too many" }, "Anthropic"),
  errorBodyCase("format-no-prefix", "m", { status: 503, body: "down" }),
];

// --- Headers: real providerHeadersToRecord ---
const headerSources: Array<[string, Array<Record<string, unknown> | undefined>]> = [
  ["simple-merge", [{ A: "1" }, { B: "2" }]],
  ["override-case", [{ "X-Auth": "a" }, { "x-auth": "b" }]],
  ["null-suppresses", [{ "X-A": "1", "X-B": "2" }, { "X-B": null }]],
  ["null-then-set", [{ "X-A": null }, { "x-a": "v" }]],
  ["empty-returns-undefined", [{ "X-A": null }]],
  ["undefined-sources", [undefined, { A: "1" }, undefined]],
  ["null-then-restore", [{ A: "1" }, { A: null }, { a: "v" }]],
];
const headerCases = headerSources.map(([name, sources]) => ({
  name,
  sources: sources.map((s) => s ?? null),
  expected: providerHeadersToRecord(...sources) ?? null,
}));

// --- Proxy: real resolveHttpProxyUrlForTarget (env map is the caller's
// ProviderEnv merge; process env proxy vars deleted above) ---
function proxyCase(name: string, target: string, env: Record<string, string>) {
  const c: Record<string, unknown> = { name, target, env };
  try {
    const url = resolveHttpProxyUrlForTarget(target, env);
    c["expected"] = url ? url.href : null;
  } catch (error) {
    c["expectedError"] = error instanceof Error ? error.message : String(error);
  }
  return c;
}
const proxyCases = [
  proxyCase("no-proxy-set", "https://api.x.com", {}),
  proxyCase("https-proxy", "https://api.x.com", { https_proxy: "proxy.local:8080" }),
  proxyCase("http-target", "http://api.x.com", { http_proxy: "http://p:3128" }),
  proxyCase("all-proxy", "https://a.b", { all_proxy: "http://p:1" }),
  proxyCase("no-proxy-star", "https://a.b", { https_proxy: "p:1", no_proxy: "*" }),
  proxyCase("no-proxy-exact", "https://api.x.com", { https_proxy: "p:1", no_proxy: "api.x.com" }),
  proxyCase("no-proxy-suffix", "https://api.x.com", { https_proxy: "p:1", no_proxy: ".x.com" }),
  proxyCase("no-proxy-suffix-root", "https://x.com", { https_proxy: "p:1", no_proxy: ".x.com" }),
  proxyCase("no-proxy-suffix-miss", "https://other.com", {
    https_proxy: "p:1",
    no_proxy: ".x.com",
  }),
  proxyCase("no-proxy-wildcard", "https://a.x.com", { https_proxy: "p:1", no_proxy: "*.x.com" }),
  proxyCase("no-proxy-port-match", "https://api.x.com:443", {
    https_proxy: "p:1",
    no_proxy: "api.x.com:443",
  }),
  proxyCase("no-proxy-port-miss", "https://api.x.com:8443", {
    https_proxy: "p:1",
    no_proxy: "api.x.com:443",
  }),
  proxyCase("no-proxy-list", "https://a.b", { https_proxy: "p:1", no_proxy: "c.d, a.b" }),
  proxyCase("no-proxy-case", "https://API.X.COM", { https_proxy: "p:1", no_proxy: "api.x.com" }),
  proxyCase("socks-unsupported", "https://a.b", { https_proxy: "socks5://p:1" }),
  proxyCase("scheme-added", "https://a.b", { https_proxy: "p:8080" }),
  proxyCase("uppercase-env", "https://a.b", { HTTPS_PROXY: "http://p:1" }),
  proxyCase("ws-scheme", "ws://a.b", { ws_proxy: "http://p:1" }),
];

Math.random = realRandom;
Date.now = realNow;
globalThis.setTimeout = realSetTimeout;

writeFileSync("vectors/ai/http-retry.json", `${JSON.stringify(retryCases, null, 2)}\n`);
writeFileSync("vectors/ai/http-error-body.json", `${JSON.stringify(errorCases, null, 2)}\n`);
writeFileSync("vectors/ai/http-headers.json", `${JSON.stringify(headerCases, null, 2)}\n`);
writeFileSync("vectors/ai/http-proxy.json", `${JSON.stringify(proxyCases, null, 2)}\n`);
console.log(
  `retry=${retryCases.length} errors=${errorCases.length} headers=${headerCases.length} proxy=${proxyCases.length}`,
);
