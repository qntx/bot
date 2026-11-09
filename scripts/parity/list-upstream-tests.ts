/**
 * `list-upstream-tests.ts` — the inventory of upstream tests (M1-03): every `*.test.ts` under
 * `packages/{ai,agent,coding-agent,schema}/test` at the commit pinned in `parity.json`, with the
 * titles of its literal-titled cases, written to `vectors/upstream/tests.json`. `check.ts` then
 * requires an owner for every listed file in `parity.json` `upstreamTests`.
 *
 * Files are read from the pinned commit with git, not from the working tree. Run with `node
 * scripts/parity/list-upstream-tests.ts`; `--check` fails when the committed inventory differs from
 * the pinned commit.
 */
import { execFileSync } from "node:child_process";
import { readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";

const PACKAGES = ["ai", "agent", "coding-agent", "schema"];
const TARGET = "vectors/upstream/tests.json";

const root = join(import.meta.dirname, "../..");
const checkout = join(root, "3rdparty/pi");
const git = (...args: string[]) =>
  execFileSync("git", ["-C", checkout, ...args], {
    encoding: "utf8",
    maxBuffer: 256 * 1024 * 1024,
  });

const parity: unknown = JSON.parse(readFileSync(join(root, "parity.json"), "utf8"));
const upstream =
  typeof parity === "object" && parity !== null && "upstream" in parity
    ? parity.upstream
    : undefined;
const commit =
  typeof upstream === "object" && upstream !== null && "commit" in upstream
    ? upstream.commit
    : undefined;
if (typeof commit !== "string") {
  throw new TypeError("parity.json has no upstream.commit");
}

// A literal title right after `it(` / `test(` and their modifiers (`.skip`, `.only`, ...).
const CASE =
  /\b(?:it|test)(?:\.(?:skip|only|todo|concurrent|fails|sequential))*\(\s*(["'`])((?:\\.|(?!\1)[^\\])*)\1/g;

const paths = git(
  "ls-tree",
  "-r",
  "--name-only",
  commit,
  "--",
  ...PACKAGES.map((name) => `packages/${name}/test`),
)
  .split("\n")
  .filter((path) => path.endsWith(".test.ts"))
  .toSorted();
const files: Record<string, string[]> = {};
for (const path of paths) {
  const text = git("show", `${commit}:${path}`);
  files[path] = [...text.matchAll(CASE)].map((match) => match[2] ?? "");
}

const inventory = `${JSON.stringify({ commit, files }, null, 2)}\n`;
if (process.argv.includes("--check")) {
  if (readFileSync(join(root, TARGET), "utf8") !== inventory) {
    throw new Error(
      `${TARGET} differs from upstream ${commit}; run node scripts/parity/list-upstream-tests.ts`,
    );
  }
  console.log(`${TARGET}: matches ${commit}`);
} else {
  writeFileSync(join(root, TARGET), inventory);
  const cases = Object.values(files).reduce((sum, names) => sum + names.length, 0);
  console.log(`${TARGET}: ${paths.length} files, ${cases} cases at ${commit}`);
}
