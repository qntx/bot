/**
 * `generate-catalog.ts` — vectors for the embedded model catalog (AI-CATALOG-EMBED): for every
 * provider ovo embeds, the model ids pi's `getBuiltinModels`, `getBuiltinImageModels` and
 * `getBuiltinClassifierModels` return, in order. The model values themselves are the catalog files,
 * which `check-catalog.ts` keeps byte-identical to the package. Run with `node
 * scripts/parity/generate-catalog.ts`.
 */
import { readdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";

type ModelList = (provider: string) => Array<{ id: string }>;
type AllModule = {
  getBuiltinModels: ModelList;
  getBuiltinImageModels: ModelList;
  getBuiltinClassifierModels: ModelList;
};

function isAllModule(value: unknown): value is AllModule {
  return (
    typeof value === "object" &&
    value !== null &&
    "getBuiltinModels" in value &&
    typeof value.getBuiltinModels === "function" &&
    "getBuiltinImageModels" in value &&
    typeof value.getBuiltinImageModels === "function" &&
    "getBuiltinClassifierModels" in value &&
    typeof value.getBuiltinClassifierModels === "function"
  );
}

const all: unknown = await import(import.meta.resolve("@earendil-works/pi-ai/providers/all"));
if (!isAllModule(all)) {
  throw new Error("providers/all dist module lacks the built-in catalog getters");
}

const root = join(import.meta.dirname, "../..");
const providers = readdirSync(join(root, "crates/ovo-ai/catalog"))
  .filter((file) => file.endsWith(".json"))
  .map((file) => file.slice(0, -".json".length))
  .toSorted();
const ids = (models: Array<{ id: string }>) => models.map((model) => model.id);
const cases = providers.map((provider) => ({
  provider,
  chat: ids(all.getBuiltinModels(provider)),
  image: ids(all.getBuiltinImageModels(provider)),
  classifier: ids(all.getBuiltinClassifierModels(provider)),
}));

const target = "vectors/ai/catalog.json";
writeFileSync(
  join(root, target),
  `${JSON.stringify({ generator: { script: "scripts/parity/generate-catalog.ts", node: process.version }, cases }, null, 2)}\n`,
);
console.log(`${target}: ${cases.length} providers`);
