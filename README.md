# agent-video

Video understanding toolkit for AI agents: one Rust binary that turns videos
into VLM-readable grid images and exposes them to agents as a CLI and an MCP
`video` tool, with URL downloads and whisper transcription built in.

## Layout

- `crates/core` — media kernel: decode, sample, score, compose grids.
- `crates/harness` — policy layer over the kernel: budget ladder, consent
  tokens, URL sources, disk cache, whisper transcription; `mcp` module owns
  the stdio server.
- `crates/cli` — the `agent-video` binary: `process`, `extract`, `mcp`,
  `install-whisper`. See [crates/README.md](crates/README.md).
- `packages/harness` — npm distribution shell (`@agent-video/cli`): resolves
  the platform binary and passes through; no logic of its own.
- `surfaces` — discovery artifacts derived from the Rust tool schema: skill,
  Codex plugin, and the pi extension (`@agent-video/pi`, a thin MCP client
  over the binary).
- `scripts` — release and maintenance helpers.
