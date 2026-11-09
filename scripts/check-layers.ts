// Layer gate: `ovo-ai --no-default-features` must not pull the HTTP
// transport stack (reqwest/hyper/rustls/h2) — part-B deps are optional.
import { execSync } from "node:child_process";

const out = execSync("cargo tree -p ovo-ai --no-default-features -e normal", {
  encoding: "utf8",
});
const banned = /\b(reqwest|hyper|rustls|h2)\b/.exec(out);
if (banned) {
  console.error(`no-default-features build pulls ${banned[0]}:\n${out}`);
  process.exitCode = 1;
} else {
  console.log("layers: no-default-features tree is clean");
}

// `host` gates process/fs: the default-free tree must not enable
// tokio's process/fs/net either.
// Inverted tree: every feature edge that activates `tokio`, so the
// gate sees tokio's enabled features rather than the forward list.
const features = execSync(
  "cargo tree -p ovo-ai --no-default-features -e features,no-dev -i tokio --prefix none",
  { encoding: "utf8" },
);
const tokioBlock = /tokio feature "([a-z_-]+)"/g;
const enabled = [...features.matchAll(tokioBlock)].map((m) => m[1] ?? "");
const hostLeaks = enabled.filter((f) => ["process", "fs", "net"].includes(f));
if (hostLeaks.length > 0) {
  console.error(`no-default-features enables tokio ${hostLeaks.join(", ")}:\n${features}`);
  process.exitCode = 1;
}
