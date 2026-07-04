/**
 * Direct http(s) source
 * Probes direct media links via ffprobe-over-http and streams downloads to
 * `<dest>.part` against a hard cap, so a file at the final path is always whole.
 */

import * as fs from 'node:fs/promises'
import * as path from 'node:path'

import { formatBytes } from '../policy/plan.ts'
import { formatTimestamp } from '../policy/time.ts'
import { ProbeError, probeMediaUrl } from '../probe/ffprobe.ts'
import type { Config } from '../store/config.ts'
import type { ProbeSummary } from '../tool/schema.ts'
import { estimateLabel, type ProgressFn, type SourceInfo, type UrlSource } from './source.ts'

// A download failed or was refused; message is model-presentable.
export class DownloadError extends Error {
	constructor(message: string) {
		super(message)
		this.name = 'DownloadError'
	}
}

// Minimum interval between progress callbacks.
const PROGRESS_INTERVAL_MS = 1000

// Recipe carried by direct-link download plans
export interface DirectRecipe {
	// file extension for the cache path
	ext: string
	// human-readable filename used when dest is keep
	filename: string
}

// The pluggable direct-link source: ffprobe-over-http probe, streaming fetch download
export function createDirectSource(cfg: Config): UrlSource {
	return {
		name: 'direct',
		async probe(url) {
			let probe: ProbeSummary
			try {
				probe = await probeMediaUrl(url)
			} catch (err) {
				// not a direct media link (HTML pages, missing ffprobe): pass down the chain
				if (err instanceof ProbeError) return undefined
				throw err
			}
			if (probe.sizeBytes === 0) {
				const len = await headContentLength(url)
				if (len !== undefined) probe = { ...probe, sizeBytes: len }
			}
			return directInfo(url, probe, cfg)
		},
		async download(plan, io, onProgress) {
			const recipe = plan.recipe as DirectRecipe
			const dest =
				io.dest === 'cache'
					? path.join(io.dir, `${io.key}.${recipe.ext}`)
					: await availablePath(path.join(io.dir, recipe.filename))
			await downloadDirect(plan.url, dest, cfg.download.maxBytes, onProgress)
			return { filepath: dest }
		},
	}
}

// Build the source info of a direct media link: one file, one keep tier
export function directInfo(url: string, probe: ProbeSummary, cfg: Config): SourceInfo {
	const { ext, filename } = directIdentity(url, probe)
	const size = probe.sizeBytes > 0 ? probe.sizeBytes : undefined
	const label = `the file (${size !== undefined ? formatBytes(size) : 'size unknown'})`
	const recipe: DirectRecipe = { ext, filename }
	const sizeText = size !== undefined ? formatBytes(size) : 'size unknown'
	return {
		headline:
			`Direct media link: ${probe.width}x${probe.height}, ` +
			`${formatTimestamp(probe.duration)}, ${sizeText}. Nothing is downloaded yet.`,
		options: [
			{
				token: 'dl-analysis',
				dest: 'cache',
				label,
				effect: `download the file (${estimateLabel(size)}) into the analysis cache (evictable) and analyze from there`,
				estimatedBytes: size,
				recipe,
			},
			{
				token: `dl-keep-${probe.height}`,
				dest: 'keep',
				label,
				effect: `save the file (${estimateLabel(size)}) to ${path.join(cfg.download.keepDir, filename)} for the user`,
				estimatedBytes: size,
				recipe,
			},
		],
	}
}

// ── link identity ──

// Container-name → file extension for URLs whose path carries none
const CONTAINER_EXT: Record<string, string> = {
	matroska: 'mkv',
	webm: 'webm',
	mov: 'mp4',
	mp4: 'mp4',
	mpegts: 'ts',
	avi: 'avi',
	flv: 'flv',
	asf: 'wmv',
	ogg: 'ogg',
}

// Best-effort extension from an ffprobe format_name (comma-list)
export function extFromContainer(container: string | undefined): string {
	const first = container?.split(',')[0]?.trim() ?? ''
	return CONTAINER_EXT[first] ?? (/^[a-z0-9]{2,5}$/.test(first) ? first : 'mp4')
}

// Extension plus a keep-dir filename for a direct media link
export function directIdentity(url: string, probe: ProbeSummary): DirectRecipe {
	let base = ''
	try {
		base = decodeURIComponent(path.posix.basename(new URL(url).pathname))
	} catch {
		// unparseable URL — fall through to defaults
	}
	base = base
		.replace(/[/\\<>:"|?*]/g, '')
		.replace(/\p{Cc}/gu, '')
		.trim()
	const urlExt = path.posix.extname(base).slice(1).toLowerCase()
	const ext = /^[a-z0-9]{2,5}$/.test(urlExt) ? urlExt : extFromContainer(probe.container)
	const filename = base === '' || base === `.${urlExt}` ? `download.${ext}` : urlExt === '' ? `${base}.${ext}` : base
	return { ext, filename }
}

// First non-colliding variant of `p`: "name.ext", "name (1).ext", …
export async function availablePath(p: string): Promise<string> {
	const dir = path.dirname(p)
	const ext = path.extname(p)
	const stem = path.basename(p, ext)
	for (let n = 0; n < 100; n += 1) {
		const candidate = n === 0 ? p : path.join(dir, `${stem} (${n})${ext}`)
		try {
			await fs.access(candidate)
		} catch {
			return candidate
		}
	}
	throw new DownloadError(`could not find a free filename for ${p} in ${dir}`)
}

// ── plumbing ──

// Content-Length via HEAD, or undefined when the server hides it.
export async function headContentLength(url: string): Promise<number | undefined> {
	try {
		const res = await fetch(url, { method: 'HEAD', redirect: 'follow' })
		const len = Number(res.headers.get('content-length'))
		return res.ok && Number.isFinite(len) && len > 0 ? len : undefined
	} catch {
		return undefined
	}
}

// Download url to dest, refusing to exceed maxBytes: known oversize is refused
// before the first byte, unknown-length streams are cut off mid-flight.
// The .part file never survives a failure; throws DownloadError on http error or cap.
export async function downloadDirect(
	url: string,
	dest: string,
	maxBytes: number,
	onProgress?: ProgressFn
): Promise<{ bytes: number }> {
	const res = await fetch(url, { redirect: 'follow' })
	if (!res.ok || res.body === null) {
		throw new DownloadError(`download failed: GET ${url} → HTTP ${res.status}`)
	}
	const lenHeader = Number(res.headers.get('content-length'))
	const total = Number.isFinite(lenHeader) && lenHeader > 0 ? lenHeader : undefined
	if (total !== undefined && total > maxBytes) {
		await res.body.cancel()
		throw new DownloadError(
			`the file is ${formatBytes(total)}, past the ${formatBytes(maxBytes)} ceiling ` +
				'(download.max_bytes) — download it manually and pass the local path as source'
		)
	}

	await fs.mkdir(path.dirname(dest), { recursive: true })
	const part = `${dest}.part`
	const handle = await fs.open(part, 'w')
	// Explicit reader, not for await: Bun's fetch body lacks a working async
	// iterator (TypeError as of 1.3.14); the error path must cancel the stream.
	const reader = res.body.getReader()
	let bytes = 0
	let lastReport = 0
	try {
		for (;;) {
			const { done, value } = await reader.read()
			if (done) break
			bytes += value.byteLength
			if (bytes > maxBytes) {
				throw new DownloadError(
					`download aborted at ${formatBytes(bytes)}: the stream exceeded the ` +
						`${formatBytes(maxBytes)} ceiling (download.max_bytes) — download it manually`
				)
			}
			await handle.write(value)
			const now = Date.now()
			if (now - lastReport >= PROGRESS_INTERVAL_MS) {
				lastReport = now
				onProgress?.(bytes, total)
			}
		}
	} catch (err) {
		await reader.cancel().catch(() => {})
		await handle.close()
		await fs.rm(part, { force: true })
		throw err
	}
	await handle.close()
	await fs.rename(part, dest)
	onProgress?.(bytes, total ?? bytes)
	return { bytes }
}
