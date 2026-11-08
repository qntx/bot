# Changelog

All notable changes to this project are documented in this file. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Changed

- Rebuilt the repository from scratch as a Tauri 2 + Rust workspace + vite-plus bun monorepo. **BREAKING**: the previous `ovo` 0.9 crates are removed; the workspace version restarts at 0.10.0 with `apps/desktop` (Tauri shell) and `packages/ovo` (`@qntx/ovo` TypeScript SDK).
