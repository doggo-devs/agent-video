/**
 * MCP video tool server
 * Stdio-transport MCP server exposing a single `video` tool that couriers
 * requests from argument coercion through cached probe, policy, and core.
 */

import * as fs from 'node:fs/promises'
import { createRequire } from 'node:module'
import * as os from 'node:os'
import * as path from 'node:path'

import { Server } from '@modelcontextprotocol/sdk/server/index.js'
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js'
import { CallToolRequestSchema, type CallToolResult, ListToolsRequestSchema } from '@modelcontextprotocol/sdk/types.js'

import { type CoreRun, captureCore, resolveBinary } from '../core/binary.ts'
import type { GridMeta, Metadata } from '../core/contract.ts'
import { type CoreInvocation, type Decision, decide, formatBytes, planResponse, redeem } from '../policy/plan.ts'
import { formatTimestamp } from '../policy/time.ts'
import { type DownloadPlan, type PendingStore, TokenRegistry } from '../policy/tokens.ts'
import { decideUrl, directInfo, type SiteUrlInfo, type UrlInfo } from '../policy/url.ts'
import { probeViaCore } from '../probe/core.ts'
import { ProbeError, probeLocalFile, probeMediaUrl } from '../probe/ffprobe.ts'
import { DownloadError, downloadDirect, headContentLength, type ProgressFn } from '../source/direct.ts'
import { downloadSite, probeSite, type SiteDownloadRequest, YtdlpError } from '../source/ytdlp.ts'
import { Cache, urlKey } from '../store/cache.ts'
import { type Config, loadConfig } from '../store/config.ts'
import { fingerprintFile } from '../store/identity.ts'
import {
	type ConfirmRequired,
	type Failed,
	LOOKS,
	type Look,
	type Plan,
	type ProbeSummary,
	VIDEO_INPUT_SCHEMA,
	VIDEO_TOOL_DESCRIPTION,
	type VideoArgs,
} from '../tool/schema.ts'

const require = createRequire(import.meta.url)

// Forwarded to MCP progress notifications when the client asked
export type ProgressReporter = (progress: number, total?: number, message?: string) => void

// Everything the tool handler needs; injectable for tests
export interface ServerDeps {
	config: Config
	cache: Cache
	tokens: PendingStore
	probeFile: (file: string) => Promise<ProbeSummary>
	// Direct media link over http(s); throws ProbeError for site pages
	probeMediaUrl: (url: string) => Promise<ProbeSummary>
	// Site page via yt-dlp -J; throws YtdlpError when yt-dlp is absent
	probeSite: (url: string) => Promise<SiteUrlInfo>
	downloadDirect: (url: string, dest: string, maxBytes: number, onProgress?: ProgressFn) => Promise<{ bytes: number }>
	downloadSite: (req: SiteDownloadRequest, onProgress?: ProgressFn) => Promise<{ filepath: string }>
	runCore: (args: readonly string[]) => Promise<CoreRun>
}

// Production wiring: real config, disk cache, ffprobe, yt-dlp, core
export function defaultDeps(): ServerDeps {
	const config = loadConfig()
	// local probing goes through `core probe`; direct ffprobe only remains for
	// binary-less dev setups (AGENT_VIDEO_BIN unset, no platform package)
	let hasCore = true
	try {
		resolveBinary()
	} catch {
		hasCore = false
	}
	return {
		config,
		cache: new Cache(config.cache),
		tokens: new TokenRegistry(),
		probeFile: hasCore ? file => probeViaCore(captureCore, file) : probeLocalFile,
		probeMediaUrl: async url => {
			const probe = await probeMediaUrl(url)
			if (probe.sizeBytes > 0) return probe
			const len = await headContentLength(url)
			return len === undefined ? probe : { ...probe, sizeBytes: len }
		},
		probeSite: url => probeSite(url, config.download.analysisMaxHeight),
		downloadDirect,
		downloadSite,
		runCore: captureCore,
	}
}

function isUrl(source: string): boolean {
	return /^https?:\/\//i.test(source)
}

// ── argument coercion ──
// clients do not reliably enforce additionalProperties, so coerce defensively

function coerceArgs(raw: Record<string, unknown>): VideoArgs | Failed {
	const fail = (error: string): Failed => ({ status: 'failed', error })

	for (const key of Object.keys(raw)) {
		if (!['source', 'from', 'until', 'look', 'raw', 'confirm'].includes(key)) {
			return fail(`unknown argument "${key}"; valid: source, from, until, look, raw, confirm`)
		}
	}
	const { source, from, until, look, raw: rawArg, confirm } = raw
	if (typeof source !== 'string' || source.length === 0) {
		return fail('source must be a non-empty string, e.g. {"source": "/videos/talk.mp4"}')
	}
	const isTime = (v: unknown): v is string | number => typeof v === 'string' || typeof v === 'number'
	if (from !== undefined && !isTime(from)) {
		return fail('from must be seconds or a timestamp string, e.g. {"from": "1:23"}')
	}
	if (until !== undefined && !isTime(until)) {
		return fail('until must be seconds or a timestamp string, e.g. {"until": 300}')
	}
	if (look !== undefined && !LOOKS.includes(look as Look)) {
		return fail(`look must be one of ${LOOKS.join(', ')}, e.g. {"look": "overview"}`)
	}
	if (rawArg !== undefined && typeof rawArg !== 'string') {
		return fail('raw must be a string of core CLI arguments, e.g. {"raw": "process --strategy keyframes"}')
	}
	if (confirm !== undefined && typeof confirm !== 'string') {
		return fail('confirm must be a string token echoed from a previous response')
	}
	return {
		source,
		from: from as VideoArgs['from'],
		until: until as VideoArgs['until'],
		look: look as Look | undefined,
		raw: rawArg,
		confirm,
	}
}

// ── probe with cache ──

function isProbeSummary(v: unknown): v is ProbeSummary {
	if (typeof v !== 'object' || v === null) return false
	const p = v as Record<string, unknown>
	return (
		typeof p['duration'] === 'number' &&
		typeof p['width'] === 'number' &&
		typeof p['height'] === 'number' &&
		typeof p['sizeBytes'] === 'number' &&
		Array.isArray(p['subtitles'])
	)
}

// Probe via content-fingerprint cache; unstable files are probed fresh, never cached
async function cachedProbe(source: string, deps: ServerDeps): Promise<ProbeSummary> {
	const fp = await fingerprintFile(source)
	if (fp.stable) {
		const hit = await deps.cache.getProbe(fp.hex)
		if (isProbeSummary(hit)) return hit
	}
	const probe = await deps.probeFile(source)
	if (fp.stable) await deps.cache.putProbe(fp.hex, probe)
	return probe
}

// ── core invocation ──

// `-o` is mandatory in the core CLI but never touched in --json mode
const UNUSED_OUT_DIR = path.join(os.tmpdir(), 'agent-video-json-unused')

async function runInvocation(
	mediaPath: string,
	invocation: CoreInvocation,
	notices: string[],
	deps: ServerDeps
): Promise<CallToolResult> {
	const window =
		invocation.window === undefined
			? []
			: ['--from', String(invocation.window.start), '--until', String(invocation.window.end)]
	const args = [invocation.subcommand, mediaPath, '--output', UNUSED_OUT_DIR, ...window, ...invocation.flags, '--json']
	const run = await deps.runCore(args)
	if (run.code !== 0) {
		const detail = run.stderr.trim().split('\n').slice(-5).join('\n')
		return failed(`core ${invocation.subcommand} failed (exit ${run.code}):\n${detail}`)
	}

	let metadata: Metadata
	try {
		metadata = JSON.parse(run.stdout) as Metadata
	} catch {
		return failed('core emitted invalid JSON — version mismatch between harness and core?')
	}
	return readyResult(metadata, notices)
}

// ── result rendering ──

function failed(error: string): CallToolResult {
	const body: Failed = { status: 'failed', error }
	return { content: [{ type: 'text', text: JSON.stringify(body) }], isError: true }
}

function jsonResult(body: Plan | ConfirmRequired): CallToolResult {
	return { content: [{ type: 'text', text: JSON.stringify(body, null, 1) }] }
}

// One text block for all grids: ranges, key frames, subtitles, notices
export function describeGrids(metadata: Metadata, notices: readonly string[]): string {
	const lines: string[] = []
	for (const notice of notices) lines.push(`note: ${notice}`)
	lines.push(
		`${metadata.source}: ${formatTimestamp(metadata.duration)}, ` +
			`${metadata.grids.length} grid(s) below, read top-left to bottom-right.`
	)
	metadata.grids.forEach((grid: GridMeta, i: number) => {
		lines.push('')
		lines.push(
			`[grid ${i + 1}/${metadata.grids.length}] ${grid.time_range} · layout ${grid.layout} · ${grid.total_frames} frames`
		)
		if (grid.key_frames.length > 0) {
			const keys = grid.key_frames.map(k => `${k.timestamp} @${k.grid_position}`).join(', ')
			lines.push(`  key frames (bordered): ${keys}`)
		}
		if (grid.subtitles !== undefined && grid.subtitles.length > 0) {
			lines.push(`  subtitles:`)
			for (const line of grid.subtitles.split('\n')) lines.push(`    ${line}`)
		}
	})
	return lines.join('\n')
}

function readyResult(metadata: Metadata, notices: string[]): CallToolResult {
	const content: CallToolResult['content'] = [{ type: 'text', text: describeGrids(metadata, notices) }]
	for (const grid of metadata.grids) {
		if ('base64' in grid) {
			content.push({ type: 'image', data: grid.base64, mimeType: 'image/jpeg' })
		}
	}
	return { content }
}

// ── tool handler ──

// Handle one `video` call end to end; never throws, errors become Failed
export async function handleVideoCall(
	rawArgs: Record<string, unknown>,
	deps: ServerDeps,
	onProgress?: ProgressReporter
): Promise<CallToolResult> {
	const args = coerceArgs(rawArgs)
	if ('status' in args) {
		return { content: [{ type: 'text', text: JSON.stringify(args) }], isError: true }
	}

	// consent first: a URL source may not exist locally until its download runs
	if (args.confirm !== undefined) {
		return renderDecision(args, redeem(args, deps.tokens), undefined, [], deps, onProgress)
	}

	if (isUrl(args.source)) {
		const cached = await deps.cache.findDownload(urlKey(args.source))
		if (cached === undefined) return urlFirstContact(args, deps, onProgress)
		return localFlow(args, cached, ['serving from the cached analysis rip'], deps, onProgress)
	}
	return localFlow(args, args.source, [], deps, onProgress)
}

// Probe a local media file and run the local decision engine
async function localFlow(
	args: VideoArgs,
	mediaPath: string,
	extraNotices: string[],
	deps: ServerDeps,
	onProgress?: ProgressReporter
): Promise<CallToolResult> {
	let probe: ProbeSummary
	try {
		probe = await cachedProbe(mediaPath, deps)
	} catch (err) {
		if ((err as NodeJS.ErrnoException).code === 'ENOENT') {
			return failed(`file not found: ${mediaPath}`)
		}
		if (err instanceof ProbeError) return failed(err.message)
		throw err
	}
	const decision = decide(args, probe, deps.config, deps.tokens)
	return renderDecision(args, decision, mediaPath, extraNotices, deps, onProgress)
}

// First contact with a URL that has no cached rip: offer downloads
async function urlFirstContact(
	args: VideoArgs,
	deps: ServerDeps,
	onProgress?: ProgressReporter
): Promise<CallToolResult> {
	let info: UrlInfo
	try {
		info = directInfo(args.source, await deps.probeMediaUrl(args.source))
	} catch (err) {
		if (!(err instanceof ProbeError)) throw err
		try {
			info = await deps.probeSite(args.source)
		} catch (siteErr) {
			if (siteErr instanceof YtdlpError) return failed(siteErr.message)
			throw siteErr
		}
	}
	const decision = decideUrl(args, info, deps.config, deps.tokens)
	return renderDecision(args, decision, undefined, [], deps, onProgress)
}

// Turn a policy decision into the wire result, executing side effects
async function renderDecision(
	args: VideoArgs,
	decision: Decision,
	mediaPath: string | undefined,
	extraNotices: string[],
	deps: ServerDeps,
	onProgress?: ProgressReporter
): Promise<CallToolResult> {
	switch (decision.kind) {
		case 'reject':
			return failed(decision.message)
		case 'plan':
			return jsonResult({
				status: 'plan',
				probe: decision.probe,
				guidance: decision.guidance,
				recommended_calls: decision.recommendedCalls,
			})
		case 'confirm':
			return jsonResult({ status: 'confirm_required', consequence: decision.consequence, tokens: decision.tokens })
		case 'run': {
			const resolved = mediaPath ?? (await resolveMedia(args.source, deps))
			if (resolved === undefined) {
				return failed('the cached rip for this URL is gone (evicted). Call again with just the URL to re-download.')
			}
			return runInvocation(resolved, decision.invocation, [...extraNotices, ...decision.notices], deps)
		}
		case 'download':
			return executeDownload(args.source, decision.plan, deps, onProgress)
	}
}

// Local paths resolve to themselves; URLs to their cached rip, if any
async function resolveMedia(source: string, deps: ServerDeps): Promise<string | undefined> {
	if (!isUrl(source)) return source
	return deps.cache.findDownload(urlKey(source))
}

// ── download execution ──

async function executeDownload(
	source: string,
	plan: DownloadPlan,
	deps: ServerDeps,
	onProgress?: ProgressReporter
): Promise<CallToolResult> {
	const progress: ProgressFn = (bytes, total) =>
		onProgress?.(
			bytes,
			total,
			`downloading: ${formatBytes(bytes)}${total !== undefined ? ` / ${formatBytes(total)}` : ''}`
		)
	try {
		const result =
			plan.dest === 'cache'
				? await downloadAnalysis(source, plan, deps, progress)
				: await downloadKeep(plan, deps, progress)
		// eviction sweep after every download
		await deps.cache.evict()
		return result
	} catch (err) {
		if (err instanceof DownloadError || err instanceof YtdlpError || err instanceof ProbeError) {
			return failed(err.message)
		}
		throw err
	}
}

// Analysis rip into the evictable cache; answer with the next plan
async function downloadAnalysis(
	source: string,
	plan: DownloadPlan,
	deps: ServerDeps,
	progress: ProgressFn
): Promise<CallToolResult> {
	const dir = await deps.cache.ensureDownloadsDir()
	const key = urlKey(plan.url)
	let filePath: string
	if (plan.method.type === 'direct') {
		filePath = path.join(dir, `${key}.${plan.method.ext}`)
		await deps.downloadDirect(plan.url, filePath, deps.config.download.maxBytes, progress)
	} else {
		;({ filepath: filePath } = await deps.downloadSite(
			{
				url: plan.url,
				format: plan.method.format,
				maxBytes: deps.config.download.maxBytes,
				output: path.join(dir, `${key}.%(ext)s`),
				subtitleOutput: plan.method.subs ? path.join(dir, `${key}.%(lang)s.%(ext)s`) : undefined,
			},
			progress
		))
		if (plan.method.subs) await promoteSubtitle(dir, key)
	}

	const probe = await cachedProbe(filePath, deps)
	const next = planResponse(source, probe, deps.config)
	return jsonResult({
		status: 'plan',
		probe: next.probe,
		guidance:
			`Analysis rip downloaded (${formatBytes(probe.sizeBytes)}, cached, evictable). ` +
			`The URL now resolves to it automatically. ${next.guidance}`,
		recommended_calls: next.recommendedCalls,
	})
}

// Full-quality download into keep_dir; the file belongs to the user
async function downloadKeep(plan: DownloadPlan, deps: ServerDeps, progress: ProgressFn): Promise<CallToolResult> {
	const keepDir = deps.config.download.keepDir
	await fs.mkdir(keepDir, { recursive: true })
	let filePath: string
	if (plan.method.type === 'direct') {
		filePath = await availablePath(path.join(keepDir, plan.method.filename))
		await deps.downloadDirect(plan.url, filePath, deps.config.download.maxBytes, progress)
	} else {
		;({ filepath: filePath } = await deps.downloadSite(
			{
				url: plan.url,
				format: plan.method.format,
				maxBytes: deps.config.download.maxBytes,
				output: path.join(keepDir, '%(title)s [%(id)s].%(ext)s'),
			},
			progress
		))
	}

	const probe = await cachedProbe(filePath, deps)
	return jsonResult({
		status: 'plan',
		probe,
		guidance:
			`Saved to ${filePath} (${formatBytes(probe.sizeBytes)}). This file belongs to the user: ` +
			'hand the path over if they want to watch it, and they may move or rename it freely. ' +
			'To analyze it, call again with the (possibly moved) path as source — probe results ' +
			'are keyed by content, so a move stays a cache hit.',
		recommended_calls: [{ call: { source: filePath }, note: 'analyze the saved file' }],
	})
}

// First non-colliding variant of `p`: "name.ext", "name (1).ext", …
async function availablePath(p: string): Promise<string> {
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

// Copy the preferred-language subtitle to `<key>.srt` so it rides along with every grid
async function promoteSubtitle(dir: string, key: string): Promise<void> {
	let names: string[]
	try {
		names = await fs.readdir(dir)
	} catch {
		return
	}
	const re = new RegExp(`^${key}\\.([^.]+)\\.(srt|vtt|ass)$`)
	const candidates = names.map(n => re.exec(n)).filter((m): m is RegExpExecArray => m !== null)
	if (candidates.length === 0) return
	const preferred =
		candidates.find(m => m[1]?.startsWith('en')) ?? candidates.find(m => m[1]?.startsWith('zh')) ?? candidates[0]
	if (!preferred) return
	const target = path.join(dir, `${key}.${preferred[2]}`)
	try {
		await fs.access(target)
	} catch {
		await fs.copyFile(path.join(dir, preferred[0]), target)
	}
}

// ── server assembly ──

// Build the MCP server around injected deps (tests use fakes here)
export function createServer(deps: ServerDeps): Server {
	const { version } = require('../../package.json') as { version: string }
	const server = new Server({ name: 'agent-video', version }, { capabilities: { tools: {} } })

	server.setRequestHandler(ListToolsRequestSchema, () => ({
		tools: [
			{
				name: 'video',
				description: VIDEO_TOOL_DESCRIPTION,
				inputSchema: VIDEO_INPUT_SCHEMA as unknown as { type: 'object'; [k: string]: unknown },
			},
		],
	}))

	server.setRequestHandler(CallToolRequestSchema, async (request, extra) => {
		if (request.params.name !== 'video') {
			return failed(`unknown tool "${request.params.name}"; only "video" exists`)
		}
		const progressToken = request.params._meta?.progressToken
		const onProgress: ProgressReporter | undefined =
			progressToken === undefined
				? undefined
				: (progress, total, message) => {
						// progress is best-effort; a dead transport must not kill the call
						void extra
							.sendNotification({
								method: 'notifications/progress',
								params: { progressToken, progress, total, message },
							})
							.catch(() => {})
					}
		return handleVideoCall(request.params.arguments ?? {}, deps, onProgress)
	})

	return server
}

// Boot the stdio server with production deps; resolves after connect
export async function serveStdio(): Promise<void> {
	const deps = defaultDeps()
	// eviction sweep on boot; never blocks or fails startup
	void deps.cache.evict().catch(() => {})
	const server = createServer(deps)
	await server.connect(new StdioServerTransport())
}
