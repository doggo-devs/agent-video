#!/usr/bin/env bun
/**
 * CLI entry point, a thin wrapper around the core binary.
 * Passes args through verbatim except --mcp and --version/-V intercepts.
 */

import { createRequire } from 'node:module'
import { BinaryNotFoundError, coreOutput, runCore } from './core/binary.ts'

const require = createRequire(import.meta.url)

const args = process.argv.slice(2)

function fail(err: unknown): never {
	const msg = err instanceof Error ? err.message : String(err)
	process.stderr.write(`agent-video: ${msg}\n`)
	process.exit(1)
}

if (args.includes('--mcp')) {
	// Lazy-load so plain CLI passthrough avoids the MCP dependency graph; the process stays alive on stdio until stdin closes
	const { serveStdio } = await import('./mcp/server.ts')
	try {
		await serveStdio()
	} catch (err) {
		fail(err)
	}
} else if (args.includes('--version') || args.includes('-V')) {
	const { name, version } = require('../package.json') as { name: string; version: string }
	process.stdout.write(`${name} ${version}\n`)
	try {
		process.stdout.write(`${await coreOutput(['--version'])} (core)\n`)
	} catch (err) {
		// When core is missing or broken, still report the harness half of the version
		const msg = err instanceof Error ? err.message : String(err)
		process.stderr.write(`core unavailable: ${msg}\n`)
		process.exit(1)
	}
	process.exit(0)
} else {
	try {
		process.exit(await runCore(args))
	} catch (err) {
		if (err instanceof BinaryNotFoundError) fail(err)
		fail(err)
	}
}
