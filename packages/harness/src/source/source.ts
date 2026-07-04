/**
 * URL source interface
 * Sources turn URLs into download offers and execute the chosen recipe;
 * policy owns tokens and caps, the server owns paths and cache.
 */

import { formatBytes } from '../policy/plan.ts'
import type { DownloadPlan } from '../policy/tokens.ts'

// Byte-level progress callback; total set when the server reveals it.
export type ProgressFn = (downloadedBytes: number, totalBytes?: number) => void

// One download candidate offered to the model behind a consent token
export interface DownloadOption {
	token: string
	dest: 'cache' | 'keep'
	// short human label, used when the option is withheld for size ("720p (1.2 GiB)")
	label: string
	// sentence for the offered-token list
	effect: string
	// best-effort size; undefined = unknown, never withheld
	estimatedBytes?: number
	// source-private download recipe, round-trips through DownloadPlan
	recipe: unknown
}

// Probe outcome of one source: summary line plus download candidates
export interface SourceInfo {
	headline: string
	options: DownloadOption[]
}

// Filesystem surface handed to a download: cache entries are keyed, keep files are free-form
export type DownloadIO = { dest: 'cache'; dir: string; key: string } | { dest: 'keep'; dir: string }

// A pluggable URL handler; probe passes (undefined) when the URL is not its kind
export interface UrlSource {
	name: string
	probe(url: string): Promise<SourceInfo | undefined>
	download(plan: DownloadPlan, io: DownloadIO, onProgress?: ProgressFn): Promise<{ filepath: string }>
}

// Probe outcome tagged with the source that claimed the URL
export interface ProbedUrl {
	source: string
	info: SourceInfo
}

// Walk the chain; first source that claims the URL wins, terminal failures throw
export async function probeUrl(sources: readonly UrlSource[], url: string): Promise<ProbedUrl> {
	for (const s of sources) {
		const info = await s.probe(url)
		if (info !== undefined) return { source: s.name, info }
	}
	throw new Error(`no source could handle ${url}`)
}

// "~1.2 GiB" or "size unknown", shared by offer effects
export function estimateLabel(bytes: number | undefined): string {
	return bytes !== undefined ? `~${formatBytes(bytes)}` : 'size unknown'
}
