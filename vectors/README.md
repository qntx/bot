# Vectors

Parity vectors are byte-frozen JSON documents generated from upstream
(Node `JSON.parse`/`JSON.stringify`, and later recorded upstream outputs).
`cargo test` replays them and compares bytes exactly; allowed differences
are the classes declared in `parity.json`.

- Layout: `vectors/<domain>/<topic>.json` (session-style inputs live in a
  same-named directory as `.jsonl` files).
- Each file is an object with `generator` (script path, Node version) and
  `cases` (`name`, `input`, `expected`, optional `classes`).
- Regenerate only when upstream or Node changes: `bun run vectors`. Review
  the diff like any other code change; the Node version is recorded inside
  each file.
- `vectors/**` is ignored by oxfmt; files are edited only by their
  generator.
Generators may import functions the package does not export from the installed artifact's `dist/` via `scripts/parity/pi-internal.js` (vectors must be computed by upstream 1.5.0 code, not the `3rdparty/pi` checkout).
