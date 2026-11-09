/// <reference types="node" />
/**
 * `check-catalog.ts` — the embedded model catalog matches the pinned upstream (ADR-006): `PIN`
 * equals the `@earendil-works/pi-ai` version in `package.json` and the installed one, and every
 * `crates/ovo-ai/catalog/<id>.json` is byte-identical to the package's
 * `dist/providers/data/<id>.json`. Refresh with `node scripts/sync-catalog.ts`.
 */
import { existsSync, readdirSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const PACKAGE = "@earendil-works/pi-ai";

type JsonObject = Record<string, unknown>;

function isRecord(value: unknown): value is JsonObject {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function readJson(path: string): JsonObject {
  const value: unknown = JSON.parse(readFileSync(path, "utf8"));
  return isRecord(value) ? value : {};
}

const root = join(import.meta.dirname, "..");
const catalogDir = join(root, "crates/ovo-ai/catalog");
const piAiDir = dirname(dirname(fileURLToPath(import.meta.resolve(PACKAGE))));
const errors: string[] = [];

const pin = readFileSync(join(catalogDir, "PIN"), "utf8").trim();
const { devDependencies } = readJson(join(root, "package.json"));
const declared = isRecord(devDependencies) ? devDependencies[PACKAGE] : undefined;
const installed = readJson(join(piAiDir, "package.json"))["version"];
if (pin !== declared) {
  errors.push(
    `crates/ovo-ai/catalog/PIN is ${pin}, package.json pins ${PACKAGE}@${String(declared)}`,
  );
}
if (pin !== installed) {
  errors.push(
    `crates/ovo-ai/catalog/PIN is ${pin}, the installed ${PACKAGE} is ${String(installed)}`,
  );
}

const files = readdirSync(catalogDir).filter((file) => file.endsWith(".json"));
if (files.length === 0) {
  errors.push("crates/ovo-ai/catalog has no provider catalogs");
}
for (const file of files) {
  const upstream = join(piAiDir, "dist/providers/data", file);
  if (!existsSync(upstream)) {
    errors.push(
      `crates/ovo-ai/catalog/${file}: ${PACKAGE}@${String(installed)} has no such catalog`,
    );
  } else if (!readFileSync(join(catalogDir, file)).equals(readFileSync(upstream))) {
    errors.push(
      `crates/ovo-ai/catalog/${file} differs from ${PACKAGE}; run node scripts/sync-catalog.ts`,
    );
  }
}

if (errors.length > 0) {
  throw new Error(`catalog check failed:\n${errors.join("\n")}`);
}
console.log(`catalog: ${files.length} providers match ${PACKAGE}@${pin}`);
