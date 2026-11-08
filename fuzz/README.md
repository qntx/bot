# Fuzz targets

cargo-fuzz workspace for the ovo crates; excluded from the root workspace.

```bash
cargo +nightly fuzz run streaming_json   # build + run
cargo +nightly fuzz build                # build only
```

`streaming_json` feeds arbitrary bytes (lossy UTF-8, random chunking) to
`StreamingJsonParser` and asserts the applied value equals the whole-text
result without panicking.
