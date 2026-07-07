#!/usr/bin/env node
/**
 * npm distribution shell for the agent-video binary (esbuild pattern).
 *
 * All logic — CLI, MCP server (`mcp` / `--mcp`), whisper — lives in the Rust
 * binary shipped by the exact-pinned `@agent-video/<platform>` optional
 * dependency. This stub only resolves that binary and hands over: args pass
 * through verbatim, stdio is inherited, signals are forwarded.
 *
 * Resolution order: `AGENT_VIDEO_BIN` env override, then the platform
 * package; no PATH fallback by design.
 */

import { spawn } from "node:child_process";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);

const RELEASES_URL = "https://github.com/durvia/agent-video/releases";
const PLATFORM_KEY = `${process.platform}-${process.arch}`;

function resolveBinary() {
  const override = process.env.AGENT_VIDEO_BIN;
  if (override) return override;

  const pkg = `@agent-video/${PLATFORM_KEY}`;
  const bin = process.platform === "win32" ? "agent-video.exe" : "agent-video";
  try {
    return require.resolve(`${pkg}/bin/${bin}`);
  } catch {
    process.stderr.write(
      `agent-video: core binary not found: ${pkg} is not installed.\n` +
        `Your platform (${PLATFORM_KEY}) may be unsupported by the npm ` +
        `distribution, or optional dependencies were skipped at install ` +
        `time (--no-optional / --omit=optional).\n` +
        `Prebuilt binaries for all supported platforms: ${RELEASES_URL}\n`,
    );
    process.exit(1);
  }
}

const child = spawn(resolveBinary(), process.argv.slice(2), {
  stdio: "inherit",
});

for (const signal of ["SIGINT", "SIGTERM", "SIGHUP"]) {
  process.on(signal, () => child.kill(signal));
}

child.on("error", (err) => {
  process.stderr.write(`agent-video: ${err.message}\n`);
  process.exit(1);
});

child.on("close", (code, signal) => {
  process.exit(code ?? (signal ? 1 : 0));
});
