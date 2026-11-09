// Generates vectors/ai/auth-*.json from the REAL upstream modules:
// pi-ai's dist env-api-keys.js and pi-coding-agent's dist
// resolve-config-value.js / auth-storage.js (re-exported via
// pi-internal.js so types resolve to the published .d.ts). Store
// cases run ReadOnlyAuthStorage against real temp dirs — the strict
// read path our FileCredentialStore mirrors (AuthStorage is the
// best-effort writer; it masks corrupt files by design).
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import {
  AuthStorage,
  findEnvKeys,
  getConfigValueEnvVarName,
  getConfigValueEnvVarNames,
  getEnvApiKey,
  getMissingConfigValueEnvVarNames,
  isCommandConfigValue,
  isConfigValueConfigured,
  resolveConfigValue,
  ReadOnlyAuthStorage,
} from "./pi-internal.js";

const apiKeyCases: unknown[] = [];
const envFixture: Record<string, string> = {
  ANTHROPIC_API_KEY: "sk-ant",
  ANTHROPIC_AUTH_TOKEN: "oauth-token",
  ANTHROPIC_OAUTH_TOKEN: "oauth-t",
  OPENAI_API_KEY: "sk-oai",
  XAI_API_KEY: "sk-xai",
  DEEPSEEK_API_KEY: "sk-ds",
  GROQ_API_KEY: "sk-groq",
  OPENROUTER_API_KEY: "sk-or",
};
for (const provider of [
  "anthropic",
  "openai",
  "xai",
  "deepseek",
  "groq",
  "openrouter",
  "unknown",
]) {
  apiKeyCases.push({
    name: `env-full-${provider}`,
    provider,
    env: envFixture,
    expected: {
      found: findEnvKeys(provider, envFixture) ?? null,
      key: getEnvApiKey(provider, envFixture) ?? null,
    },
  });
}
for (const [name, provider, env] of [
  ["anthropic-only-auth-token", "anthropic", { ANTHROPIC_AUTH_TOKEN: "tok" }],
  [
    "anthropic-no-api-key",
    "anthropic",
    { ANTHROPIC_AUTH_TOKEN: "tok", ANTHROPIC_OAUTH_TOKEN: "o" },
  ],
  ["openai-empty", "openai", { OPENAI_API_KEY: "" }],
  ["openai-missing", "openai", {}],
] as const) {
  apiKeyCases.push({
    name,
    provider,
    env,
    expected: {
      found: findEnvKeys(provider, env) ?? null,
      key: getEnvApiKey(provider, env) ?? null,
    },
  });
}

const envForTemplates: Record<string, string> = {
  A: "va",
  B: "vb",
  KEY: "vkey",
  N1: "1",
  N2: "2",
};
const configCases: unknown[] = [];
for (const config of [
  "$A",
  `\${A}`,
  "pre$A post",
  "$A$B",
  `\${A}-\${B}`,
  "$$",
  "$!",
  "$$A",
  "literal",
  "$MISSING",
  "pre$MISSING post",
  `\${}`,
  `\${1BAD}`,
  `\${_OK}`,
  "$A_extra",
  "$",
  "$-",
  "x$",
  "!echo hi",
  "!!",
]) {
  configCases.push({
    name: config.replaceAll(/[^a-zA-Z0-9]+/g, "_") || "literal",
    config,
    env: envForTemplates,
    expected: {
      is_command: isCommandConfigValue(config),
      env_var_name: getConfigValueEnvVarName(config) ?? null,
      env_var_names: getConfigValueEnvVarNames(config),
      missing: getMissingConfigValueEnvVarNames(config, envForTemplates),
      configured: isConfigValueConfigured(config, envForTemplates),
      // `!command` cases are parse-decision only — vectors never
      // execute commands.
      resolved: isCommandConfigValue(config)
        ? "<command>"
        : (resolveConfigValue(config, envForTemplates) ?? null),
    },
  });
}

const storeCases: unknown[] = [];
const authDocs: Array<[string, Record<string, unknown>]> = [
  ["empty", {}],
  ["api-key", { anthropic: { type: "api_key", key: "sk-ant" } }],
  [
    "api-key-env-template",
    { openai: { type: "api_key", key: `\${KEY}`, env: { KEY: "resolved" } } },
  ],
  ["api-key-command", { xai: { type: "api_key", key: "!pass show xai" } }],
  [
    "oauth",
    {
      anthropic: {
        type: "oauth",
        access: "a",
        refresh: "r",
        expires: 1_900_000_000_000,
        extra: "x",
      },
    },
  ],
  ["invalid-type", { p: { type: "weird" } }],
  ["missing-expires", { p: { type: "oauth", access: "a", refresh: "r" } }],
];
for (const [name, data] of authDocs) {
  const dir = mkdtempSync(join(tmpdir(), "ovo-auth-"));
  const authPath = join(dir, "auth.json");
  writeFileSync(authPath, `${JSON.stringify(data, null, 2)}\n`, {
    encoding: "utf8",
    mode: 0o600,
  });
  try {
    const storage = new ReadOnlyAuthStorage(authPath);
    const read: Record<string, unknown> = {};
    // oxlint-disable-next-line no-await-in-loop -- cases run sequentially
    const list = await storage.list();
    for (const id of Object.keys(data)) {
      // oxlint-disable-next-line no-await-in-loop -- deterministic order
      read[id] = (await storage.read(id)) ?? null;
    }
    storeCases.push({ name, data, expected: { read, list } });
  } catch (error) {
    storeCases.push({
      name,
      data,
      expected: { error: error instanceof Error ? error.message : String(error) },
    });
  }
  rmSync(dir, { recursive: true, force: true });
}

// upstream modify() expects a Promise-returning callback.
const setAnthropic = async (): Promise<unknown> => {
  await Promise.resolve();
  return { type: "api_key", key: "sk-ant" };
};

// File-backend round-trip: the real FileAuthStorageBackend + AuthStorage
// writes `{}`, reads back credentials, and deletes to `{}`.
const writeCases: unknown[] = [];
{
  const dir = mkdtempSync(join(tmpdir(), "ovo-auth-write-"));
  const authPath = join(dir, "auth.json");
  const storage = AuthStorage.create(authPath);
  const written = await storage.modify("anthropic", setAnthropic);
  const readBack = await storage.read("anthropic");
  const fileText = readFileSync(authPath, "utf8");
  writeCases.push({
    name: "modify-roundtrip",
    expected: { written, readBack, file: fileText },
  });
  rmSync(dir, { recursive: true, force: true });
}

writeFileSync("vectors/ai/auth-apikey.json", `${JSON.stringify(apiKeyCases, null, 2)}\n`);
writeFileSync("vectors/ai/auth-config.json", `${JSON.stringify(configCases, null, 2)}\n`);
writeFileSync("vectors/ai/auth-store.json", `${JSON.stringify(storeCases, null, 2)}\n`);
writeFileSync("vectors/ai/auth-write.json", `${JSON.stringify(writeCases, null, 2)}\n`);
console.log(
  `apikey=${apiKeyCases.length} config=${configCases.length} store=${storeCases.length} write=${writeCases.length}`,
);
