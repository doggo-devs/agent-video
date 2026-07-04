# agent-video

Video understanding toolkit for AI agents: a Rust core that turns videos
into VLM-readable grid images, and a TypeScript layer that exposes it to
agents as CLI / MCP tools.

## Layout

- `crates/core` — Rust kernel: decode, sample, score, compose grids.
  Standalone binary, no Node required. See [crates/README.md](crates/README.md).
- `packages/harness` — TypeScript strategy layer: CLI wrapper, MCP server,
  tool schemas, pluggable video sources (work in progress).
- `scripts` — release and maintenance helpers.
