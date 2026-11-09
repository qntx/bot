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
