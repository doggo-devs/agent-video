/**
 * Resolve the platform-specific core binary and run it.
 * Resolution order: `AGENT_VIDEO_BIN` env override, then the exact-pinned
 * `@agent-video/<platform>` optional dependency; no PATH fallback by design.
 */

import { spawn } from 'node:child_process'
import { createRequire } from 'node:module'

const require = createRequire(import.meta.url)

export const RELEASES_URL = 'https://github.com/durvia/agent-video/releases'

// npm platform key; platform package names match it verbatim
export const PLATFORM_KEY = `${process.platform}-${process.arch}`

// the core binary could not be located; `message` is user-presentable
export class BinaryNotFoundError extends Error {
	constructor(readonly pkg: string) {
		super(
			`agent-video core binary not found: ${pkg} is not installed.\n` +
				`Your platform (${PLATFORM_KEY}) may be unsupported by the npm ` +
				`distribution, or optional dependencies were skipped at install ` +
				`time (--no-optional / --omit=optional).\n` +
				`Prebuilt binaries for all supported platforms: ${RELEASES_URL}`
		)
		this.name = 'BinaryNotFoundError'
	}
}

// absolute path to the core binary; throws BinaryNotFoundError when unresolved and no override set
export function resolveBinary(): string {
	const override = process.env['AGENT_VIDEO_BIN']
	if (override) return override

	const pkg = `@agent-video/${PLATFORM_KEY}`
	const bin = process.platform === 'win32' ? 'agent-video.exe' : 'agent-video'
	try {
		return require.resolve(`${pkg}/bin/${bin}`)
	} catch {
		throw new BinaryNotFoundError(pkg)
	}
}

// run the core with inherited stdio, forwarding termination signals; resolves to the exit code
export function runCore(args: readonly string[]): Promise<number> {
	const child = spawn(resolveBinary(), args, { stdio: 'inherit' })

	const forward = (signal: NodeJS.Signals) => () => child.kill(signal)
	const onInt = forward('SIGINT')
	const onTerm = forward('SIGTERM')
	process.on('SIGINT', onInt)
	process.on('SIGTERM', onTerm)

	return new Promise((resolve, reject) => {
		child.once('error', reject)
		child.once('close', (code, signal) => {
			process.off('SIGINT', onInt)
			process.off('SIGTERM', onTerm)
			resolve(code ?? (signal === 'SIGINT' ? 130 : 143))
		})
	})
}

// captured result of a core run; `stdout` may be tens of MB in --json mode
export interface CoreRun {
	code: number
	stdout: string
	stderr: string
}

// run the core capturing both streams; never rejects on non-zero exit, only on spawn/resolve failures
export function captureCore(args: readonly string[]): Promise<CoreRun> {
	return new Promise((resolve, reject) => {
		const child = spawn(resolveBinary(), args, { stdio: ['ignore', 'pipe', 'pipe'] })
		let stdout = ''
		let stderr = ''
		child.stdout.on('data', (chunk: Buffer) => (stdout += chunk))
		child.stderr.on('data', (chunk: Buffer) => (stderr += chunk))
		child.once('error', reject)
		child.once('close', code => resolve({ code: code ?? 1, stdout, stderr }))
	})
}

// run the core and capture trimmed stdout for small invocations like `--version`; rejects on non-zero exit
export function coreOutput(args: readonly string[]): Promise<string> {
	return new Promise((resolve, reject) => {
		const child = spawn(resolveBinary(), args, { stdio: ['ignore', 'pipe', 'inherit'] })
		let out = ''
		child.stdout.on('data', (chunk: Buffer) => (out += chunk))
		child.once('error', reject)
		child.once('close', code =>
			code === 0 ? resolve(out.trim()) : reject(new Error(`core exited with ${code}: ${args.join(' ')}`))
		)
	})
}
