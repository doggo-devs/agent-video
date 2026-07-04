#!/usr/bin/env bun
//! CLI entry point. Thin wrapper around the core binary: everything is
//! passed through verbatim (zero flag drift), except two intercepts:
//! `--mcp` switches to the MCP stdio server, `--version`/`-V` reports the
//! harness and core versions together.
//!
//! Runs as TypeScript directly under bun; there is no build step.

import { createRequire } from "node:module";
import { BinaryNotFoundError, coreOutput, runCore } from "./core/binary.ts";

const require = createRequire(import.meta.url);

const args = process.argv.slice(2);

function fail(err: unknown): never {
  const msg = err instanceof Error ? err.message : String(err);
  process.stderr.write(`agent-video: ${msg}\n`);
  process.exit(1);
}

if (args.includes("--mcp")) {
  // Lazy import: keeps plain CLI passthrough free of the MCP dependency
  // graph. The process stays alive on the stdio transport until the
  // client closes stdin.
  const { serveStdio } = await import("./mcp/server.ts");
  try {
    await serveStdio();
  } catch (err) {
    fail(err);
  }
} else if (args.includes("--version") || args.includes("-V")) {

  const { name, version } = require("../package.json") as {
    name: string;
    version: string;
  };
  process.stdout.write(`${name} ${version}\n`);
  try {
    process.stdout.write(`${await coreOutput(["--version"])} (core)\n`);
  } catch (err) {
    // Version reporting should still tell the harness half of the story
    // when the core binary is missing or broken.
    const msg = err instanceof Error ? err.message : String(err);
    process.stderr.write(`core unavailable: ${msg}\n`);
    process.exit(1);
  }
  process.exit(0);
} else {
  try {
    process.exit(await runCore(args));
  } catch (err) {
    if (err instanceof BinaryNotFoundError) fail(err);
    fail(err);
  }
}
