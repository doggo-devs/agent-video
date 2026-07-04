/**
 * `raw` escape hatch parsing: a strategy bypass, not a safety bypass.
 * Enforces the subcommand and plumbing ban, extracts what budgeting needs.
 */

import { ArgError } from './errors.ts'
import { DEFAULT_CHUNK_SECONDS } from './look.ts'

// Server-owned flags; rejected if they appear in raw
const FORBIDDEN = new Set(['-o', '--output', '--json'])

// A parsed raw invocation
export interface RawInvocation {
	subcommand: 'process' | 'extract'
	// everything after the subcommand, tokenized
	flags: string[]
	// chunk length implied by the raw args, i.e. the denominator of the grid count
	chunkSeconds: number
}

// Tokenize a command string: split on whitespace, quotes group
export function tokenize(input: string): string[] {
	const tokens: string[] = []
	const re = /"([^"]*)"|'([^']*)'|(\S+)/g
	for (const m of input.matchAll(re)) {
		tokens.push(m[1] ?? m[2] ?? m[3] ?? '')
	}
	return tokens
}

// chunk length per profile, mirrors crates/core/src/config.rs
const PROFILE_CHUNKS: Record<string, number> = { fast: 90, quality: 45 }

// Parse and validate a raw string; throws ArgError on invalid subcommand or plumbing flag
export function parseRaw(raw: string): RawInvocation {
	const tokens = tokenize(raw.trim())
	const subcommand = tokens[0]
	if (subcommand !== 'process' && subcommand !== 'extract') {
		throw new ArgError(
			`raw must start with a core subcommand ("process" or "extract"), e.g. "process --strategy keyframes" — got "${raw}"`
		)
	}
	const flags = tokens.slice(1)

	for (const t of flags) {
		if (FORBIDDEN.has(t)) {
			throw new ArgError(
				`raw must not contain ${t}: input/output plumbing is injected by the server. Pass only strategy flags.`
			)
		}
	}

	let chunkSeconds = DEFAULT_CHUNK_SECONDS
	const profileIdx = flags.indexOf('--profile')
	if (profileIdx !== -1) {
		const profile = flags[profileIdx + 1]
		if (profile !== undefined) chunkSeconds = PROFILE_CHUNKS[profile] ?? chunkSeconds
	}
	const chunkIdx = flags.indexOf('--chunk-seconds')
	if (chunkIdx !== -1) {
		const v = Number(flags[chunkIdx + 1])
		if (Number.isFinite(v) && v > 0) chunkSeconds = v
	}

	return { subcommand, flags, chunkSeconds }
}
