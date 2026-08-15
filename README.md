# agent-video

Video understanding toolkit for AI agents: one Rust binary that turns videos
into VLM-readable grid images and exposes them to agents as a CLI and an MCP
`video` tool, with URL downloads and whisper transcription built in.

Instead of feeding a model one frame at a time, agent-video samples frames,
drops blurry and near-duplicate ones, scores the rest, and packs them into grid
images sized to a single pixel budget — key frames get larger cells so
important moments keep their detail. The harness adds a policy layer: a budget
ladder that bounds what a render emits, consent tokens for expensive actions, URL
sources (direct links and yt-dlp), a shared disk cache, and whisper
transcription that rides along with every grid.

## Layout

- `crates/core` — media kernel: decode, sample, score, compose grids.
- `crates/harness` — policy layer over the kernel: budget ladder, consent
  tokens, URL sources, disk cache, whisper transcription; `mcp` module owns
  the stdio server.
- `crates/cli` — the `agent-video` binary: `process`, `extract`, `mcp`,
  `install-whisper`, `cookie`. See [crates/README.md](crates/README.md).
- `packages/harness` — npm distribution shell (`@agent-video/cli`): resolves
  the platform binary and passes through; no logic of its own.
- `surfaces` — discovery artifacts derived from the Rust tool schema: skill,
  Codex plugin, and the pi extension (`@agent-video/pi`, a thin MCP client
  over the binary).
- `scripts` — release and maintenance helpers.

## Install

### npm (prebuilt binary)

The npm package `@agent-video/cli` is a thin shell that resolves an
exact-pinned `@agent-video/<platform>` binary (fully-static musl on Linux,
dynamic elsewhere) and hands over. Supported platforms: `linux-x64`,
`linux-arm64`, `darwin-arm64`, `win32-x64`.

```bash
npm install -g @agent-video/cli
agent-video --help
```

Binary resolution order is `$AGENT_VIDEO_BIN`, then the platform package — no
`PATH` fallback.

### Prebuilt binaries

Release tarballs for every platform (dynamic without a suffix, `-static` for
the self-contained variant) are attached to each tag at
<https://github.com/durvia/agent-video/releases>. The static Linux build is a
single file with no dynamic dependencies; the dynamic builds are GPU-capable
and expect a matching system FFmpeg.

### Build from source

Requires Rust (edition 2024) and FFmpeg shared libraries + headers. See
[crates/README.md](crates/README.md) for the full build matrix (dynamic,
static, musl, GPU features).

```bash
cargo build --release
# binary at target/release/agent-video
```

## CLI quickstart

```bash
# Smart frame selection + grid composition into a directory
agent-video process video.mp4 -o out/

# Straight from a URL (any protocol the build's FFmpeg can open)
agent-video process https://example.com/talk.mp4 -o out/

# High-density sampling of one time range into a single grid
agent-video extract video.mp4 --range 01:23-01:35 -o out/

# JSON with base64 images to stdout, no files written
agent-video process video.mp4 --json
```

Output is one or more `grid_*.jpg` images plus `metadata.json` describing each
grid, its time range, and its key frames. `--from`/`--until`/`--range` seek by
timestamp. Run `agent-video process --help` for the full override set.

## MCP quickstart

The `mcp` subcommand serves the `video` tool over stdio. Point any MCP client
at the binary:

```json
{
  "mcpServers": {
    "agent-video": {
      "command": "agent-video",
      "args": ["mcp"]
    }
  }
}
```

If you installed via npm, use the npm shim as the command instead:

```json
{
  "mcpServers": {
    "agent-video": {
      "command": "npx",
      "args": ["-y", "@agent-video/cli", "mcp"]
    }
  }
}
```

Call `video` with just a `source` first: short videos run immediately, longer
ones answer with a plan and copyable follow-up calls. Set `transcribe: true`
(alone) to get a whisper transcript back as timestamped lines; it is cached
and rides along with every grid afterwards.
Transcription needs weights (`agent-video install-whisper`, GPU build) or a
`WHISPER_API_KEY` for the API route.

For login-gated sources, manage cookies with `agent-video cookie import`,
`cookie status`, and `cookie clear`.

## Documentation

- [docs/config.md](docs/config.md) — `config.toml` reference: every key, its
  default, validation rule, and what it controls.
- [docs/protocol.md](docs/protocol.md) — the `video` tool wire contract:
  response states, consent-token lifecycle, and fault classification.
- [docs/shared-cache.md](docs/shared-cache.md) — the three legal cross-process
  sharing patterns for cache and cookie-jar artifacts.

## License

Apache-2.0. See [crates/README.md](crates/README.md) for embedded-font and
FFmpeg licensing details.
