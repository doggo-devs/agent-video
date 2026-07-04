/**
 * URL-source decisions
 * Turns source facts into download tiers behind consent tokens, or refusals.
 * Nothing is downloaded without an echoed token; tiers over max_bytes are dropped.
 */

import type { ProbedUrl } from '../source/source.ts'
import type { Config } from '../store/config.ts'
import type { OfferedToken, VideoArgs } from '../tool/schema.ts'
import { type Decision, formatBytes } from './plan.ts'
import type { PendingAction, PendingStore } from './tokens.ts'

// Model-facing guidance attached to every download offer
const CHOICE_GUIDANCE =
	'dl-analysis fetches a low-res rip plus subtitles into an evictable cache — ' +
	'enough for you to watch and report on the video. If the user wants the video ' +
	'for themselves, pick a dl-keep tier instead: never serve them the analysis rip. ' +
	'Echo exactly one token via confirm.'

// First contact with a not-yet-downloaded URL: offer tiers behind tokens, or refuse
export function decideUrl(args: VideoArgs, probed: ProbedUrl, cfg: Config, store: PendingStore): Decision {
	if (args.raw !== undefined) {
		return {
			kind: 'reject',
			message: 'raw needs a local file. Call with just the URL first, confirm a download, then repeat the raw call.',
		}
	}

	const offers: Record<string, PendingAction> = {}
	const tokens: OfferedToken[] = []
	const dropped: string[] = []
	for (const opt of probed.info.options) {
		if (opt.estimatedBytes !== undefined && opt.estimatedBytes > cfg.download.maxBytes) {
			if (!dropped.includes(opt.label)) dropped.push(opt.label)
			continue
		}
		offers[opt.token] = {
			kind: 'download',
			plan: {
				url: args.source,
				dest: opt.dest,
				source: probed.source,
				recipe: opt.recipe,
				estimatedBytes: opt.estimatedBytes,
			},
		}
		tokens.push({ token: opt.token, effect: opt.effect })
	}

	if (tokens.length === 0) {
		return {
			kind: 'reject',
			message:
				`${probed.info.headline} Every available tier exceeds the ${formatBytes(cfg.download.maxBytes)} ` +
				'download ceiling (download.max_bytes). Download the file manually and pass the local path as source.',
		}
	}

	store.offer(args.source, offers)

	const notes: string[] = [probed.info.headline]
	if (dropped.length > 0) {
		notes.push(`Tiers over the ${formatBytes(cfg.download.maxBytes)} ceiling were not offered: ${dropped.join(', ')}.`)
	}
	if (args.from !== undefined || args.until !== undefined || args.look !== undefined) {
		notes.push(
			'Your from/until/look cannot apply yet — the download comes first. Repeat them in the call after it completes.'
		)
	}
	notes.push(CHOICE_GUIDANCE)

	return { kind: 'confirm', consequence: notes.join(' '), tokens }
}
