#!/usr/bin/env bun
//! CLI entry point. Thin wrapper around the core binary: everything is
//! passed through verbatim (zero flag drift), except two intercepts:
//! `--mcp` switches to the MCP stdio server, `--version`/`-V` reports the
//! harness and core versions together.
//!
//! Runs as TypeScript directly under bun; there is no build step.

import { createRequire } from "node:module";
import { BinaryNotFoundError, coreOutput, runCore } from "./binary.ts";

const require = createRequire(import.meta.url);

const args = process.argv.slice(2);

function fail(err: unknown): never {
  const msg = err instanceof Error ? err.message : String(err);
  process.stderr.write(`agent-video: ${msg}\n`);
  process.exit(1);
}

if (args.includes("--mcp")) {
  process.stderr.write(
    "agent-video: MCP server is not implemented yet; track progress at " +
      "https://github.com/durvia/agent-video\n",
  );
  process.exit(2);
}

if (args.includes("--version") || args.includes("-V")) {
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
}

try {
  process.exit(await runCore(args));
} catch (err) {
  if (err instanceof BinaryNotFoundError) fail(err);
  fail(err);
}
