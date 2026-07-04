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
import {
	type CoreInvocation,
	type Decision,
	decide,
	formatBytes,
	type PlanDecision,
	planResponse,
	redeem,
} from '../policy/plan.ts'
import { formatTimestamp } from '../policy/time.ts'
import { type DownloadPlan, type PendingStore, TokenRegistry, type TranscribePlan } from '../policy/tokens.ts'
import { decideUrl } from '../policy/url.ts'
import { probeViaCore } from '../probe/core.ts'
import { ProbeError, probeLocalFile } from '../probe/ffprobe.ts'
import { createDirectSource, DownloadError } from '../source/direct.ts'
import { type ProbedUrl, type ProgressFn, probeUrl, type UrlSource } from '../source/source.ts'
import { createYtdlpSource, YtdlpError } from '../source/ytdlp.ts'
import { Cache, urlKey } from '../store/cache.ts'
import { type Config, loadConfig } from '../store/config.ts'
import { fingerprintFile } from '../store/identity.ts'
import {
	type ConfirmRequired,
	type Failed,
	LOOKS,
	type Look,
	type Pending,
	type Plan,
	type ProbeSummary,
	VIDEO_INPUT_SCHEMA,
	VIDEO_TOOL_DESCRIPTION,
	type VideoArgs,
} from '../tool/schema.ts'
import { createWhisper, TranscribeError, type WhisperFacade } from '../transcribe/whisper.ts'
import { DownloadTracker, type JobStatus } from './tracker.ts'

const require = createRequire(import.meta.url)

// Forwarded to MCP progress notifications when the client asked
export type ProgressReporter = (progress: number, total?: number, message?: string) => void

// Everything the tool handler needs; injectable for tests
export interface ServerDeps {
	config: Config
	cache: Cache
	tokens: PendingStore
	probeFile: (file: string) => Promise<ProbeSummary>
	// URL handlers, walked in order; first probe claim wins
	sources: UrlSource[]
	// in-flight background downloads, keyed by source URL
	downloads: DownloadTracker
	// optional transcription capability; absence degrades to a hint
	whisper: WhisperFacade
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
		sources: [createDirectSource(config), createYtdlpSource(config)],
		downloads: new DownloadTracker(),
		whisper: createWhisper(config.transcribe),
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

// Probe via content-fingerprint cache; unstable files are probed fresh, never
// cached. A cached whisper transcript decorates the summary (never the cache).
async function cachedProbe(source: string, deps: ServerDeps): Promise<{ probe: ProbeSummary; fingerprint: string }> {
	const fp = await fingerprintFile(source)
	let probe: ProbeSummary | undefined
	if (fp.stable) {
		const hit = await deps.cache.getProbe(fp.hex)
		if (isProbeSummary(hit)) probe = hit
	}
	if (probe === undefined) {
		probe = await deps.probeFile(source)
		if (fp.stable) await deps.cache.putProbe(fp.hex, probe)
	}
	if ((await deps.cache.findTranscript(fp.hex)) !== undefined) {
		probe = { ...probe, subtitles: [...probe.subtitles, 'transcript (whisper)'] }
	}
	return { probe, fingerprint: fp.hex }
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
	// a cached whisper transcript rides along unless the caller picked a subtitle
	const fp = await fingerprintFile(mediaPath)
	const transcript = await deps.cache.findTranscript(fp.hex)
	const subtitle =
		transcript !== undefined && !invocation.flags.includes('--subtitle') ? ['--subtitle', transcript] : []
	const args = [
		invocation.subcommand,
		mediaPath,
		'--output',
		UNUSED_OUT_DIR,
		...window,
		...subtitle,
		...invocation.flags,
		'--json',
	]
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

function jsonResult(body: Plan | ConfirmRequired | Pending): CallToolResult {
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

	// background downloads first: hand out results, report progress on polls
	if (isUrl(args.source)) {
		const key = urlKey(args.source)
		const settled = deps.downloads.take(key)
		if (settled !== undefined) return settled
		const inflight = deps.downloads.peek(key)
		if (inflight !== undefined) return pendingResult(inflight)
	}

	// consent next: a URL source may not exist locally until its download runs
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
	let fingerprint: string
	try {
		;({ probe, fingerprint } = await cachedProbe(mediaPath, deps))
	} catch (err) {
		if ((err as NodeJS.ErrnoException).code === 'ENOENT') {
			return failed(`file not found: ${mediaPath}`)
		}
		if (err instanceof ProbeError) return failed(err.message)
		throw err
	}
	const decision = decide(args, probe, deps.config, deps.tokens)
	if (decision.kind === 'plan') {
		await offerTranscription(args.source, mediaPath, fingerprint, decision, deps)
	}
	return renderDecision(args, decision, mediaPath, extraNotices, deps, onProgress)
}

// Subtitle-less plans grow a transcribe offer (or an install hint)
async function offerTranscription(
	source: string,
	mediaPath: string,
	fingerprint: string,
	plan: PlanDecision,
	deps: ServerDeps
): Promise<void> {
	if (plan.probe.subtitles.length > 0) return
	if (await deps.whisper.available()) {
		deps.tokens.offer(source, { transcribe: { kind: 'transcribe', plan: { file: mediaPath, fingerprint } } })
		plan.recommendedCalls.push({
			call: { source, confirm: 'transcribe' },
			note: 'no subtitles found — generate a whisper transcript locally (CPU-bound, can take minutes); it then rides along with every grid',
		})
	} else {
		plan.guidance +=
			' No subtitles found and no whisper tool detected — install openai-whisper or whisper.cpp (plus transcribe.model) to enable transcription.'
	}
}

// First contact with a URL that has no cached rip: offer downloads
async function urlFirstContact(
	args: VideoArgs,
	deps: ServerDeps,
	onProgress?: ProgressReporter
): Promise<CallToolResult> {
	let probed: ProbedUrl
	try {
		probed = await probeUrl(deps.sources, args.source)
	} catch (err) {
		if (err instanceof YtdlpError || err instanceof ProbeError) return failed(err.message)
		throw err
	}
	const decision = decideUrl(args, probed, deps.config, deps.tokens)
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
		case 'transcribe':
			return executeTranscribe(args.source, decision.plan, deps, onProgress)
	}
}

// Run whisper against local media; answer with the refreshed plan
async function executeTranscribe(
	source: string,
	plan: TranscribePlan,
	deps: ServerDeps,
	onProgress?: ProgressReporter
): Promise<CallToolResult> {
	try {
		let step = 0
		await deps.whisper.transcribe(plan.file, deps.cache.transcriptPath(plan.fingerprint), message => {
			step += 1
			onProgress?.(step, undefined, message)
		})
	} catch (err) {
		if (err instanceof TranscribeError) return failed(err.message)
		throw err
	}
	const { probe } = await cachedProbe(plan.file, deps)
	const next = planResponse(source, probe, deps.config)
	return jsonResult({
		status: 'plan',
		probe: next.probe,
		guidance: `Transcript generated and cached — it rides along with every grid from now on. ${next.guidance}`,
		recommended_calls: next.recommendedCalls,
	})
}

// Local paths resolve to themselves; URLs to their cached rip, if any
async function resolveMedia(source: string, deps: ServerDeps): Promise<string | undefined> {
	if (!isUrl(source)) return source
	return deps.cache.findDownload(urlKey(source))
}

// ── download execution ──

// Start the download, then race a sync grace window: fast downloads answer
// inline, slow ones detach into the tracker and are polled by later calls
async function executeDownload(
	source: string,
	plan: DownloadPlan,
	deps: ServerDeps,
	onProgress?: ProgressReporter
): Promise<CallToolResult> {
	const key = urlKey(source)
	const job = deps.downloads.start(key)
	const progress: ProgressFn = (bytes, total) => {
		job.downloaded = bytes
		if (total !== undefined) job.total = total
		onProgress?.(
			bytes,
			total,
			`downloading: ${formatBytes(bytes)}${total !== undefined ? ` / ${formatBytes(total)}` : ''}`
		)
	}
	const run = performDownload(source, plan, deps, progress).then(
		result => {
			job.settled = result
			return result
		},
		err => {
			// unexpected crash: keep it observable, settle as a poll-able failure
			console.error('[agent-video] background download crashed:', err)
			const result = failed(`download failed unexpectedly: ${err instanceof Error ? err.message : String(err)}`)
			job.settled = result
			return result
		}
	)
	const winner = await Promise.race([run, delay(deps.config.download.syncWaitSeconds * 1000)])
	if (winner !== undefined) {
		deps.downloads.clear(key)
		return winner
	}
	return pendingResult(job)
}

function delay(ms: number): Promise<undefined> {
	return new Promise(resolve => setTimeout(() => resolve(undefined), ms).unref())
}

// Progress snapshot for a download that went background
function pendingResult(job: JobStatus): CallToolResult {
	const total = job.total !== undefined ? ` / ${formatBytes(job.total)}` : ''
	return jsonResult({
		status: 'pending',
		message:
			`Download in progress: ${formatBytes(job.downloaded)}${total} so far. It continues in the background — ` +
			'call again with just the same source to check on it; the result is served once it finishes. ' +
			'Repeat any from/until/look after that.',
	})
}

// The blocking download flow: fetch, respond, sweep; known errors become Failed
async function performDownload(
	source: string,
	plan: DownloadPlan,
	deps: ServerDeps,
	progress: ProgressFn
): Promise<CallToolResult> {
	try {
		const src = deps.sources.find(s => s.name === plan.source)
		if (src === undefined) {
			return failed(`source "${plan.source}" is not available — call again with just the URL for a fresh plan`)
		}
		let filepath: string
		if (plan.dest === 'cache') {
			const dir = await deps.cache.ensureDownloadsDir()
			;({ filepath } = await src.download(plan, { dest: 'cache', dir, key: urlKey(plan.url) }, progress))
		} else {
			await fs.mkdir(deps.config.download.keepDir, { recursive: true })
			;({ filepath } = await src.download(plan, { dest: 'keep', dir: deps.config.download.keepDir }, progress))
		}
		const result =
			plan.dest === 'cache' ? await analysisResponse(source, filepath, deps) : await keepResponse(filepath, deps)
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

// Analysis rip landed in the evictable cache; answer with the next plan
async function analysisResponse(source: string, filepath: string, deps: ServerDeps): Promise<CallToolResult> {
	const { probe, fingerprint } = await cachedProbe(filepath, deps)
	const next = planResponse(source, probe, deps.config)
	next.guidance =
		`Analysis rip downloaded (${formatBytes(probe.sizeBytes)}, cached, evictable). ` +
		`The URL now resolves to it automatically. ${next.guidance}`
	await offerTranscription(source, filepath, fingerprint, next, deps)
	return jsonResult({
		status: 'plan',
		probe: next.probe,
		guidance: next.guidance,
		recommended_calls: next.recommendedCalls,
	})
}

// Full-quality download landed in keep_dir; the file belongs to the user
async function keepResponse(filepath: string, deps: ServerDeps): Promise<CallToolResult> {
	const { probe } = await cachedProbe(filepath, deps)
	return jsonResult({
		status: 'plan',
		probe,
		guidance:
			`Saved to ${filepath} (${formatBytes(probe.sizeBytes)}). This file belongs to the user: ` +
			'hand the path over if they want to watch it, and they may move or rename it freely. ' +
			'To analyze it, call again with the (possibly moved) path as source — probe results ' +
			'are keyed by content, so a move stays a cache hit.',
		recommended_calls: [{ call: { source: filepath }, note: 'analyze the saved file' }],
	})
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
