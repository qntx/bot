# Security Policy

## Supported versions

| Version | Supported |
| ------- | --------- |
| 0.10.x  | Yes       |
| < 0.10  | No        |

## Reporting a vulnerability

Report security issues privately via GitHub Security Advisories on
[qntx/bot](https://github.com/qntx/bot/security/advisories/new), or email
`security@qntx.org` if that channel is unavailable.

Please include:

- Affected version and platform (macOS/Linux/Windows, Tauri shell or SDK)
- Minimal reproduction
- Impact assessment

Do not open public issues for unfixed vulnerabilities.

## Threat model (summary)

The desktop app runs a local agent harness: it spawns processes, reads and
writes files under user-chosen directories, and talks to model providers over
the network. The frontend is confined to typed tauri-specta commands with a
restrictive CSP; agent capabilities (filesystem, shell, network) are gated by
the harness, not by the webview.

| Threat                                | Severity | Mitigation                                                           |
| ------------------------------------- | -------- | -------------------------------------------------------------------- |
| Webview XSS reaching the host         | High     | CSP (`default-src 'self'`), no remote content, typed IPC only        |
| Untrusted model/agent output executed | High     | Agent actions go through the ovo harness policy, not raw eval/shell  |
| JSONL record injection over RPC       | Medium   | Strict LF framing; one record per line; `JsonlError` on invalid JSON |
| Secrets in logs or committed files    | High     | `.env*` gitignored; never log provider keys                          |

Treat all data returned from provider APIs and all content the agent reads as
untrusted.
