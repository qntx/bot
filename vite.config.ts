import { builtinModules } from "node:module";

import { defineConfig } from "vite-plus";
import type { UserConfig } from "vite-plus";

import { fmt } from "@qntx/oxfmt";
import { merge, react } from "@qntx/oxlint";

// The library runs on Node, browsers, and embedded runtimes: Node builtins
// must not be imported from src.
const platformNeutralImports = {
  paths: builtinModules.map((name) => ({
    name,
    message: "Node builtin — the library must stay platform-neutral",
  })),
  patterns: [
    {
      group: ["node:*"],
      message: "Node builtin — the library must stay platform-neutral",
    },
  ],
};

const config: UserConfig = defineConfig({
  defaultPackage: {
    dev: "./apps/desktop",
    build: "./apps/desktop",
    preview: "./apps/desktop",
    pack: "./packages/ovo",
  },
  staged: { "*": "vp check --fix" },
  fmt: {
    ...fmt,
    ignorePatterns: [
      ...fmt.ignorePatterns,
      "target/**",
      "bun.lock",
      // TOML is owned by taplo (.taplo.toml, align_entries).
      "**/*.toml",
      // tauri-specta bindings and generated Tauri schemas are byte-frozen.
      "**/*.gen.ts",
      "apps/desktop/src-tauri/gen/**",
    ],
  },
  lint: merge(react, {
    // merge() concatenates arrays onto the preset's own ignorePatterns.
    ignorePatterns: ["target/**", "**/*.gen.ts", "apps/desktop/src-tauri/gen/**"],
    jsPlugins: [{ name: "vite-plus", specifier: "vite-plus/oxlint-plugin" }],
    rules: {
      "vite-plus/prefer-vite-plus-imports": "error",
      // Vite idiom: CSS is imported for side effects.
      "import/no-unassigned-import": ["error", { allow: ["**/*.css"] }],
      // oxfmt canonicalizes hex digits to lowercase; this rule demands uppercase, so fmt always wins.
      "unicorn/number-literal-case": "off",
    },
    overrides: [
      {
        files: ["scripts/**"],
        rules: {
          // Maintenance scripts print their results.
          "eslint/no-console": "off",
        },
      },
      {
        files: ["packages/*/src/**"],
        rules: {
          // @types/node is in the program via tests and scripts, so Node-only
          // globals would typecheck silently; browser-only globals are equally
          // absent on other runtimes. Both sets are banned — library code
          // must use globalThis lookups instead.
          "eslint/no-restricted-globals": [
            "error",
            { name: "Buffer", message: "Node-only global — use Uint8Array" },
            { name: "process", message: "Node-only global — not available in browsers" },
            { name: "global", message: "Node-only global — use globalThis" },
            { name: "require", message: "CJS-only — use import" },
            { name: "module", message: "CJS-only global" },
            { name: "__dirname", message: "CJS-only global" },
            { name: "__filename", message: "CJS-only global" },
            { name: "setImmediate", message: "Node-only global — use setTimeout" },
            { name: "clearImmediate", message: "Node-only global — use clearTimeout" },
            { name: "window", message: "browser-only global — use globalThis" },
            { name: "document", message: "browser-only global — use globalThis" },
            { name: "navigator", message: "browser-only global — use globalThis" },
            { name: "location", message: "browser-only global — use globalThis" },
            { name: "localStorage", message: "browser-only global — inject a store" },
            { name: "sessionStorage", message: "browser-only global — inject a store" },
          ],
          "eslint/no-restricted-imports": ["error", platformNeutralImports],
        },
      },
    ],
  }),
  run: { cache: process.env["CI"] === undefined || process.env["CI"] === "" },
  test: {
    include: ["scripts/**/*.test.ts"],
  },
});

export default config;
