/// <reference types="node" />
/**
 * `sync-catalog.ts` — copy the built-in model catalog of the registered providers from the pinned
 * `@earendil-works/pi-ai` package into `crates/ovo-ai/catalog/` (ADR-006) and record the version in
 * `PIN`.
 *
 * The registered providers are the `<id>.json` files already in the catalog directory; ids passed
 * as arguments are added. Files are copied byte for byte: `check-catalog.ts` compares them with the
 * package on every lint. Run with `node scripts/sync-catalog.ts [provider ...]`.
 */
import { copyFileSync, existsSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(import.meta.dirname, "..");
const catalogDir = join(root, "crates/ovo-ai/catalog");
const piAiDir = dirname(dirname(fileURLToPath(import.meta.resolve("@earendil-works/pi-ai"))));
const dataDir = join(piAiDir, "dist/providers/data");

const manifest: unknown = JSON.parse(readFileSync(join(piAiDir, "package.json"), "utf8"));
const version =
  typeof manifest === "object" && manifest !== null && "version" in manifest
    ? manifest.version
    : undefined;
if (typeof version !== "string") {
  throw new TypeError("@earendil-works/pi-ai package.json has no version");
}

const registered = readdirSync(catalogDir)
  .filter((file) => file.endsWith(".json"))
  .map((file) => file.slice(0, -".json".length));
const providers = [...new Set([...registered, ...process.argv.slice(2)])].toSorted();
for (const provider of providers) {
  const source = join(dataDir, `${provider}.json`);
  if (!existsSync(source)) {
    throw new Error(`@earendil-works/pi-ai@${version} has no catalog for provider "${provider}"`);
  }
  copyFileSync(source, join(catalogDir, `${provider}.json`));
}
writeFileSync(join(catalogDir, "PIN"), `${version}\n`);
console.log(
  `crates/ovo-ai/catalog: ${providers.length} providers from @earendil-works/pi-ai@${version}`,
);
