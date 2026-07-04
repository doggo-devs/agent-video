/**
 * yt-dlp source adapter
 * Probes and downloads 1800+ sites via an optional yt-dlp on PATH; its absence
 * becomes an install hint while direct media links keep working without it.
 */

// Machine interfaces only: -J for probing, --progress-template for progress,
// --print after_move:filepath for the final path; the human bar is never parsed.

import { spawn } from 'node:child_process'

import type { KeepTier, SiteUrlInfo } from '../policy/url.ts'
import type { ProgressFn } from './direct.ts'

// yt-dlp failed or is not installed; message is model-presentable.
export class YtdlpError extends Error {
	constructor(
		message: string,
		readonly missingBinary = false
	) {
		super(message)
		this.name = 'YtdlpError'
	}
}

const INSTALL_HINT =
	'yt-dlp not found on PATH — install it (https://github.com/yt-dlp/yt-dlp) to handle ' +
	'site URLs, or download the video manually and pass the local file path as source'

// How many keep tiers (distinct heights) to offer at most.
export const KEEP_TIER_LIMIT = 3

// ── -J document ──

export interface YtdlpFormat {
	height?: number | null
	vcodec?: string | null
	acodec?: string | null
	filesize?: number | null
	filesize_approx?: number | null
}

export interface YtdlpJson {
	title?: string
	id?: string
	duration?: number
	subtitles?: Record<string, unknown>
	formats?: YtdlpFormat[]
}

function sizeOf(f: YtdlpFormat): number | undefined {
	const n = f.filesize ?? f.filesize_approx
	return typeof n === 'number' && n > 0 ? n : undefined
}

function isVideo(f: YtdlpFormat): boolean {
	return typeof f.vcodec === 'string' && f.vcodec !== 'none' && (f.height ?? 0) > 0
}

function isAudioOnly(f: YtdlpFormat): boolean {
	return typeof f.acodec === 'string' && f.acodec !== 'none' && !isVideo(f)
}

// Largest known size among fmts, or undefined when none reports one.
function maxSize(fmts: YtdlpFormat[]): number | undefined {
	const sizes = fmts.map(sizeOf).filter((n): n is number => n !== undefined)
	return sizes.length > 0 ? Math.max(...sizes) : undefined
}

// Reduce a -J document to the policy-facing SiteUrlInfo; pure. Size estimates
// mirror download selectors: keep = best video at tier + best audio, analysis =
// best video-only rendition at or under analysisMaxHeight.
export function siteInfo(doc: YtdlpJson, analysisMaxHeight: number): SiteUrlInfo {
	const formats = doc.formats ?? []
	const video = formats.filter(isVideo)
	const audioSize = maxSize(formats.filter(isAudioOnly))

	const heights = [...new Set(video.map(f => f.height as number))].sort((a, b) => b - a).slice(0, KEEP_TIER_LIMIT)

	const keepTiers: KeepTier[] = heights.map(height => {
		const atHeight = video.filter(f => f.height === height)
		const videoSize = maxSize(atHeight)
		const muxed = atHeight.some(f => typeof f.acodec === 'string' && f.acodec !== 'none')
		return { height, estimatedBytes: videoSize === undefined ? undefined : videoSize + (muxed ? 0 : (audioSize ?? 0)) }
	})

	const analysisCandidates = video.filter(f => (f.height as number) <= analysisMaxHeight && f.acodec === 'none')
	const pool =
		analysisCandidates.length > 0 ? analysisCandidates : video.filter(f => (f.height as number) <= analysisMaxHeight)
	const bestHeight = pool.length > 0 ? Math.max(...pool.map(f => f.height as number)) : undefined

	return {
		kind: 'site',
		title: doc.title ?? doc.id ?? 'untitled',
		duration: typeof doc.duration === 'number' && doc.duration > 0 ? doc.duration : undefined,
		subtitleLangs: Object.keys(doc.subtitles ?? {}),
		keepTiers,
		analysisEstimatedBytes: bestHeight === undefined ? undefined : maxSize(pool.filter(f => f.height === bestHeight)),
	}
}

// ── spawning ──

function stderrTail(stderr: string): string {
	return stderr.trim().split('\n').slice(-5).join('\n') || 'no error output'
}

function run(
	args: readonly string[],
	onLine?: (line: string, stream: 'stdout' | 'stderr') => void
): Promise<{ code: number; stdout: string; stderr: string }> {
	return new Promise((resolve, reject) => {
		const child = spawn('yt-dlp', args, { stdio: ['ignore', 'pipe', 'pipe'] })
		let stdout = ''
		let stderr = ''
		const feed = (stream: 'stdout' | 'stderr') => {
			let buffer = ''
			return (chunk: Buffer) => {
				if (stream === 'stdout') stdout += chunk
				else stderr += chunk
				buffer += chunk
				const lines = buffer.split('\n')
				buffer = lines.pop() ?? ''
				for (const line of lines) onLine?.(line, stream)
			}
		}
		child.stdout.on('data', feed('stdout'))
		child.stderr.on('data', feed('stderr'))
		child.once('error', (e: NodeJS.ErrnoException) => {
			reject(e.code === 'ENOENT' ? new YtdlpError(INSTALL_HINT, true) : e)
		})
		child.once('close', code => resolve({ code: code ?? 1, stdout, stderr }))
	})
}

// Probe a site page: yt-dlp -J reduced through siteInfo.
// Throws YtdlpError when yt-dlp is absent, exits non-zero, or emits non-JSON.
export async function probeSite(url: string, analysisMaxHeight: number): Promise<SiteUrlInfo> {
	const { code, stdout, stderr } = await run(['-J', '--no-playlist', url])
	if (code !== 0) {
		throw new YtdlpError(`yt-dlp could not read ${url} (exit ${code}): ${stderrTail(stderr)}`)
	}
	let doc: unknown
	try {
		doc = JSON.parse(stdout)
	} catch {
		throw new YtdlpError('yt-dlp -J emitted invalid JSON — version too old?')
	}
	if (typeof doc !== 'object' || doc === null) {
		throw new YtdlpError('yt-dlp -J emitted no metadata object')
	}
	return siteInfo(doc as YtdlpJson, analysisMaxHeight)
}

// ── downloading ──

// One download, fully specified by the redeemed plan.
export interface SiteDownloadRequest {
	url: string
	// yt-dlp -f selector.
	format: string
	// Hard cap; yt-dlp aborts oversized downloads itself.
	maxBytes: number
	// Output template for the media file (absolute).
	output: string
	// Subtitle sidecar template; set only for analysis rips.
	subtitleOutput?: string
}

const PROGRESS_PREFIX = 'AVDL '
const PROGRESS_TEMPLATE =
	'download:AVDL %(progress.downloaded_bytes)s %(progress.total_bytes)s %(progress.total_bytes_estimate)s'

// Parse one --progress-template line; undefined for foreign lines.
export function parseProgress(line: string): { downloaded: number; total?: number } | undefined {
	if (!line.startsWith(PROGRESS_PREFIX)) return undefined
	const [downloaded, total, estimate] = line.slice(PROGRESS_PREFIX.length).trim().split(/\s+/)
	const num = (s: string | undefined): number | undefined => {
		const n = Number(s)
		return s !== undefined && s !== 'NA' && Number.isFinite(n) && n > 0 ? n : undefined
	}
	const d = num(downloaded)
	if (d === undefined) return undefined
	return { downloaded: d, total: num(total) ?? num(estimate) }
}

// Download via yt-dlp, blocking until the file is at its final path; yt-dlp owns
// .part handling and atomic renames. Throws YtdlpError when yt-dlp is absent,
// fails, or finishes without a file (e.g. the size cap aborted the download).
export async function downloadSite(req: SiteDownloadRequest, onProgress?: ProgressFn): Promise<{ filepath: string }> {
	const args = [
		'--no-playlist',
		'--no-simulate',
		'--quiet',
		'--progress',
		'--newline',
		'--progress-delta',
		'1',
		'--progress-template',
		PROGRESS_TEMPLATE,
		'--print',
		'after_move:filepath',
		'-f',
		req.format,
		'--max-filesize',
		String(req.maxBytes),
		'-o',
		req.output,
	]
	if (req.subtitleOutput !== undefined) {
		args.push(
			'--write-subs',
			'--sub-langs',
			'all,-live_chat',
			'--convert-subs',
			'srt',
			'-o',
			`subtitle:${req.subtitleOutput}`
		)
	}
	args.push(req.url)

	let filepath: string | undefined
	const { code, stderr } = await run(args, (line, stream) => {
		const progress = parseProgress(line)
		if (progress !== undefined) {
			onProgress?.(progress.downloaded, progress.total)
			return
		}
		const trimmed = line.trim()
		if (stream === 'stdout' && trimmed.length > 0) filepath = trimmed
	})

	if (code !== 0) {
		throw new YtdlpError(`yt-dlp download failed (exit ${code}): ${stderrTail(stderr)}`)
	}
	if (filepath === undefined) {
		throw new YtdlpError(
			'yt-dlp finished without producing a file — the size cap (download.max_bytes) ' +
				`may have aborted it: ${stderrTail(stderr)}`
		)
	}
	return { filepath }
}
