/**
 * Decision engine
 * Pure (args, probe, config) → action; consent state via the injected PendingStore.
 * Grid count picks extract vs process; budget ladder gates on stretch and consent.
 */

import type { Config } from '../store/config.ts'
import type { OfferedToken, ProbeSummary, RecommendedCall, VideoArgs } from '../tool/schema.ts'
import { ArgError } from './errors.ts'
import { presetFor } from './look.ts'
import { parseRaw } from './raw.ts'
import { formatTimestamp, resolveWindow, type Window } from './time.ts'
import type { DownloadPlan, PendingStore, TranscribePlan } from './tokens.ts'

// A fully-decided core run; the server injects source and I/O flags
export interface CoreInvocation {
	subcommand: 'process' | 'extract'
	// Strategy flags only; never contains input path, -o, or --json
	flags: string[]
	// Attention window for process (core clamp); extract encodes --range in flags
	window?: { start: number; end: number }
	predictedGrids: number
}

// Probe-only decision; also reused verbatim after analysis downloads
export interface PlanDecision {
	kind: 'plan'
	probe: ProbeSummary
	guidance: string
	recommendedCalls: RecommendedCall[]
}

export type Decision =
	| { kind: 'run'; invocation: CoreInvocation; notices: string[] }
	| { kind: 'download'; plan: DownloadPlan; notices: string[] }
	| { kind: 'transcribe'; plan: TranscribePlan; notices: string[] }
	| PlanDecision
	| { kind: 'confirm'; consequence: string; tokens: OfferedToken[] }
	| { kind: 'reject'; message: string }

// Degradation floor: max seconds one grid may cover before consent is required
export const MAX_CHUNK_STRETCH_S = 600

// Decide what a video call does; never throws, invalid input → reject
export function decide(args: VideoArgs, probe: ProbeSummary, cfg: Config, store: PendingStore): Decision {
	try {
		return decideInner(args, probe, cfg, store)
	} catch (err) {
		if (err instanceof ArgError) return { kind: 'reject', message: err.message }
		throw err
	}
}

// Redeem a consent token: pure lookup against the pending store, no probe needed
export function redeem(args: VideoArgs, store: PendingStore): Decision {
	const hasIntent = args.from !== undefined || args.until !== undefined || args.look !== undefined
	if (args.raw !== undefined || hasIntent) {
		return {
			kind: 'reject',
			message: 'confirm must be passed alone (with source): it authorizes exactly the pending plan, nothing else',
		}
	}
	const token = args.confirm ?? ''
	const action = store.take(args.source, token)
	if (action === undefined) {
		return {
			kind: 'reject',
			message: `token "${token}" is unknown or stale for this source. Tokens are single-use and bound to the latest plan; call again without confirm to get a fresh one.`,
		}
	}
	const notices = [`confirmed: ${token}`]
	switch (action.kind) {
		case 'core':
			return { kind: 'run', invocation: action.invocation, notices }
		case 'download':
			return { kind: 'download', plan: action.plan, notices }
		case 'transcribe':
			return { kind: 'transcribe', plan: action.plan, notices }
	}
}

function decideInner(args: VideoArgs, probe: ProbeSummary, cfg: Config, store: PendingStore): Decision {
	const hasIntent = args.from !== undefined || args.until !== undefined || args.look !== undefined

	if (args.confirm !== undefined) return redeem(args, store)

	if (args.raw !== undefined) {
		if (hasIntent) {
			throw new ArgError('raw is mutually exclusive with from/until/look: raw callers own the strategy entirely')
		}
		return decideRaw(args, probe, cfg, store)
	}

	const window = resolveWindow(args.from, args.until, probe.duration)
	const preset = presetFor(args.look)
	const span = window.end - window.start
	const natural = Math.max(1, Math.ceil(span / preset.chunkSeconds))

	// Bare source on anything non-trivial: plan first, decode later
	if (!hasIntent && natural > cfg.budget.autoRun) {
		return planResponse(args.source, probe, cfg)
	}

	if (natural === 1) {
		const range = `${Math.floor(window.start)}-${Math.ceil(window.end)}`
		return {
			kind: 'run',
			invocation: { subcommand: 'extract', flags: [...preset.flags, '--range', range], predictedGrids: 1 },
			notices: window.notices,
		}
	}

	if (natural <= cfg.budget.max) {
		return { kind: 'run', invocation: processInvocation([...preset.flags], window, natural), notices: window.notices }
	}

	// Over budget: stretch chunks so the window fits in budget.max grids
	const stretched = Math.ceil(span / cfg.budget.max)
	if (stretched <= MAX_CHUNK_STRETCH_S) {
		const flags = [...preset.flags, '--chunk-seconds', String(stretched)]
		return {
			kind: 'run',
			invocation: processInvocation(flags, window, cfg.budget.max),
			notices: [
				...window.notices,
				`budget: ${natural} grids at requested density exceeds the ${cfg.budget.max}-grid budget; each grid now covers ~${stretched}s (degraded temporal resolution). Zoom into ranges with from/until for full density.`,
			],
		}
	}

	// Even stretched, grids would be absurdly thin: require consent
	const token = `force-stretch-${cfg.budget.max}-grids`
	const invocation = processInvocation([...preset.flags, '--chunk-seconds', String(stretched)], window, cfg.budget.max)
	store.offer(args.source, { [token]: { kind: 'core', invocation } })
	return {
		kind: 'confirm',
		consequence: `covering ${formatTimestamp(span)} in ${cfg.budget.max} grids means each grid spans ~${Math.round(stretched / 60)} minutes — likely too thin to be useful. Better: survey sub-windows with from/until (look=overview), or rely on subtitles for long talks.`,
		tokens: [{ token, effect: `run anyway: ${cfg.budget.max} grids, ~${stretched}s of video per grid` }],
	}
}

function decideRaw(args: VideoArgs, probe: ProbeSummary, cfg: Config, store: PendingStore): Decision {
	const raw = parseRaw(args.raw ?? '')
	if (raw.subcommand === 'extract') {
		return { kind: 'run', invocation: { subcommand: 'extract', flags: raw.flags, predictedGrids: 1 }, notices: [] }
	}
	const predicted = Math.max(1, Math.ceil(probe.duration / raw.chunkSeconds))
	const invocation: CoreInvocation = { subcommand: 'process', flags: raw.flags, predictedGrids: predicted }
	if (predicted <= cfg.budget.max) {
		return { kind: 'run', invocation, notices: [] }
	}
	const token = `raw-full-${predicted}-grids`
	store.offer(args.source, { [token]: { kind: 'core', invocation } })
	return {
		kind: 'confirm',
		consequence: `this raw plan emits ~${predicted} grids over ${formatTimestamp(probe.duration)} — far past the ${cfg.budget.max}-grid budget and likely to flood your context.`,
		tokens: [{ token, effect: `run anyway, emitting ~${predicted} grids` }],
	}
}

function processInvocation(flags: string[], window: Window, predictedGrids: number): CoreInvocation {
	return {
		subcommand: 'process',
		flags,
		window: window.full ? undefined : { start: window.start, end: window.end },
		predictedGrids,
	}
}

// Plan response for a bare source: probe summary plus verbatim next calls
export function planResponse(source: string, probe: ProbeSummary, cfg: Config): PlanDecision {
	const overviewGrids = Math.ceil(probe.duration / presetFor('overview').chunkSeconds)
	const capped = Math.min(overviewGrids, cfg.budget.max)
	const mid = probe.duration / 2
	const zoomFrom = formatTimestamp(Math.max(0, mid - 30))
	const zoomUntil = formatTimestamp(Math.min(probe.duration, mid + 30))

	const subs =
		probe.subtitles.length > 0
			? ` Subtitles available (${probe.subtitles.join('; ')}) — they ride along with every grid.`
			: ''

	const recommendedCalls: RecommendedCall[] = [
		{
			call: { source, look: 'overview' },
			note:
				overviewGrids > cfg.budget.max
					? `full pass, degraded to ${capped} grids (~${Math.ceil(probe.duration / cfg.budget.max)}s each)`
					: `full pass, ~${overviewGrids} grids`,
		},
		{
			call: { source, from: zoomFrom, until: zoomUntil, look: 'detailed' },
			note: 'zoom pattern: any ~1-minute window at full density — adjust from/until',
		},
	]

	return {
		kind: 'plan',
		probe,
		guidance: `${formatTimestamp(probe.duration)} video, ${probe.width}x${probe.height}, ${formatBytes(probe.sizeBytes)}.${subs} Pick a call below (copy the args verbatim), or answer from metadata alone if that already satisfies the user.`,
		recommendedCalls,
	}
}

// Human-readable byte sizes for consequence text
export function formatBytes(n: number): string {
	if (n >= 1024 ** 3) return `${(n / 1024 ** 3).toFixed(1)} GiB`
	if (n >= 1024 ** 2) return `${(n / 1024 ** 2).toFixed(0)} MiB`
	return `${Math.ceil(n / 1024)} KiB`
}
