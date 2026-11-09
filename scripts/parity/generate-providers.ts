/**
 * `generate-providers.ts` — vectors for the built-in providers ovo registers (AI-PROVIDERS-CORE):
 * from pi's real `builtinProviders()`, each provider's id, name, base URL, headers, auth names, and
 * the result of its API-key `resolve` for a set of environments and stored credentials. Run with
 * `node scripts/parity/generate-providers.ts`.
 */
import { readdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";

type Json = Record<string, unknown>;
type Resolve = (input: {
  ctx: { env: (name: string) => Promise<string | undefined> };
  credential: Json | undefined;
  signal: AbortSignal;
}) => Promise<unknown>;
type Provider = {
  id: string;
  name: string;
  baseUrl?: string;
  headers?: unknown;
  auth: { apiKey?: { name: string; resolve: Resolve }; oauth?: { name: string } };
};

function isProviderList(value: unknown): value is Provider[] {
  return (
    Array.isArray(value) &&
    value.every((entry) => typeof entry === "object" && entry !== null && "id" in entry)
  );
}

function isAllModule(value: unknown): value is { builtinProviders: () => unknown } {
  return (
    typeof value === "object" &&
    value !== null &&
    "builtinProviders" in value &&
    typeof value.builtinProviders === "function"
  );
}

const all: unknown = await import(import.meta.resolve("@earendil-works/pi-ai/providers/all"));
const builtin = isAllModule(all) ? all.builtinProviders() : undefined;
if (!isProviderList(builtin)) {
  throw new Error("providers/all dist module has no builtinProviders");
}

const root = join(import.meta.dirname, "../..");
const registered = new Set(
  readdirSync(join(root, "crates/ovo-ai/catalog"))
    .filter((file) => file.endsWith(".json"))
    .map((file) => file.slice(0, -".json".length)),
);

type EnvCase = [string, Json, Json | undefined];

/** Cases for every provider; `KEY` stands for the provider's API key variable. */
const COMMON_CASES: EnvCase[] = [
  ["none", {}, undefined],
  ["empty-value", { KEY: "" }, undefined],
  ["env-key", { KEY: "from-env" }, undefined],
  ["stored-key", { KEY: "from-env" }, { type: "api_key", key: "stored", env: { REGION: "eu" } }],
  ["stored-without-key", { KEY: "from-env" }, { type: "api_key" }],
];

/** Anthropic's own resolution branches. */
const ANTHROPIC_CASES: EnvCase[] = [
  ["auth-token", { ANTHROPIC_AUTH_TOKEN: "tok", ANTHROPIC_API_KEY: "key" }, undefined],
  [
    "oauth-token-before-api-key",
    { ANTHROPIC_OAUTH_TOKEN: "oat", ANTHROPIC_API_KEY: "key" },
    undefined,
  ],
  ["api-key", { ANTHROPIC_API_KEY: "key" }, undefined],
  ["empty-auth-token", { ANTHROPIC_AUTH_TOKEN: "", ANTHROPIC_API_KEY: "key" }, undefined],
  [
    "federation",
    {
      ANTHROPIC_FEDERATION_RULE_ID: "rule",
      ANTHROPIC_ORGANIZATION_ID: "org",
      ANTHROPIC_IDENTITY_TOKEN_FILE: "/token",
    },
    undefined,
  ],
  [
    "federation-with-optional-ids",
    {
      ANTHROPIC_FEDERATION_RULE_ID: "rule",
      ANTHROPIC_ORGANIZATION_ID: "org",
      ANTHROPIC_IDENTITY_TOKEN_FILE: "/token",
      ANTHROPIC_SERVICE_ACCOUNT_ID: "sa",
      ANTHROPIC_WORKSPACE_ID: "",
    },
    undefined,
  ],
  [
    "federation-incomplete",
    { ANTHROPIC_FEDERATION_RULE_ID: "rule", ANTHROPIC_IDENTITY_TOKEN_FILE: "/t" },
    undefined,
  ],
  [
    "api-key-before-federation",
    {
      ANTHROPIC_API_KEY: "key",
      ANTHROPIC_FEDERATION_RULE_ID: "rule",
      ANTHROPIC_ORGANIZATION_ID: "org",
      ANTHROPIC_IDENTITY_TOKEN_FILE: "/token",
    },
    undefined,
  ],
];

const PRIMARY_KEY: Record<string, string> = {
  anthropic: "ANTHROPIC_API_KEY",
  deepseek: "DEEPSEEK_API_KEY",
  groq: "GROQ_API_KEY",
  openai: "OPENAI_API_KEY",
  openrouter: "OPENROUTER_API_KEY",
  xai: "XAI_API_KEY",
};

/** Replace the placeholder `KEY` with the provider's API key variable. */
function forProvider(env: Json, provider: string): Json {
  const key = PRIMARY_KEY[provider];
  if (key === undefined) {
    throw new Error(`no API key variable recorded for ${provider}`);
  }
  return Object.fromEntries(
    Object.entries(env).map(([name, value]) => [name === "KEY" ? key : name, value]),
  );
}

const cases: Json[] = [];
for (const provider of builtin.filter((entry) => registered.has(entry.id))) {
  const { apiKey } = provider.auth;
  const resolutions: Json[] = [];
  for (const [name, rawEnv, credential] of [
    ...COMMON_CASES,
    ...(provider.id === "anthropic" ? ANTHROPIC_CASES : []),
  ]) {
    const env = forProvider(rawEnv, provider.id);
    // oxlint-disable-next-line no-await-in-loop -- one resolution per case, in order
    const result = await apiKey?.resolve({
      ctx: {
        // oxlint-disable-next-line require-await -- the context reads an in-memory snapshot
        env: async (variable) => {
          const value = env[variable];
          return typeof value === "string" ? value : undefined;
        },
      },
      credential,
      signal: new AbortController().signal,
    });
    resolutions.push({ name, env, credential: credential ?? null, result: result ?? null });
  }
  cases.push({
    id: provider.id,
    name: provider.name,
    baseUrl: provider.baseUrl ?? null,
    headers: provider.headers ?? null,
    apiKeyAuth: apiKey?.name ?? null,
    oauth: provider.auth.oauth?.name ?? null,
    resolutions,
  });
}
if (cases.length !== registered.size) {
  throw new Error(`builtinProviders() lacks some of ${[...registered].join(", ")}`);
}

const target = "vectors/ai/providers.json";
writeFileSync(
  join(root, target),
  `${JSON.stringify({ generator: { script: "scripts/parity/generate-providers.ts", node: process.version }, cases }, null, 2)}\n`,
);
console.log(`${target}: ${cases.length} providers`);
