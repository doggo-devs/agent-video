/**
 * Timestamp parsing and window resolution.
 * Lenient on format, strict on validity within the video bounds.
 */

import { ArgError } from './errors.ts'

const SECONDS = /^\d+(\.\d+)?$/
const MM_SS = /^(\d+):([0-5]?\d(\.\d+)?)$/
const H_MM_SS = /^(\d+):([0-5]?\d):([0-5]?\d(\.\d+)?)$/

// Parse a user-provided timestamp into seconds; throws ArgError with param name and examples when invalid
export function parseTimestamp(value: string | number, param: string): number {
	if (typeof value === 'number') {
		if (!Number.isFinite(value) || value < 0) {
			throw new ArgError(`${param} must be a non-negative number of seconds, got ${value}`)
		}
		return value
	}
	const s = value.trim()
	if (SECONDS.test(s)) return Number(s)
	let m = MM_SS.exec(s)
	if (m) return Number(m[1]) * 60 + Number(m[2])
	m = H_MM_SS.exec(s)
	if (m) return Number(m[1]) * 3600 + Number(m[2]) * 60 + Number(m[3])
	throw new ArgError(`${param} must be seconds or a timestamp — e.g. 83, "1:23", or "1:02:03" — got "${value}"`)
}

// Format seconds as "MM:SS" ("H:MM:SS" past one hour), matching core output
export function formatTimestamp(seconds: number): string {
	const total = Math.floor(seconds)
	const h = Math.floor(total / 3600)
	const m = Math.floor((total % 3600) / 60)
	const s = total % 60
	const mm = String(m).padStart(2, '0')
	const ss = String(s).padStart(2, '0')
	return h > 0 ? `${h}:${mm}:${ss}` : `${mm}:${ss}`
}

// A resolved window of interest, always a valid sub-range of the video
export interface Window {
	start: number
	end: number
	// true when the window covers the entire video
	full: boolean
	notices: string[]
}

// Resolve [from ?? 0, until ?? duration] against the video duration; out-of-bounds until is clamped with a notice, out-of-bounds from errors
export function resolveWindow(
	from: string | number | undefined,
	until: string | number | undefined,
	duration: number
): Window {
	const notices: string[] = []
	const start = from === undefined ? 0 : parseTimestamp(from, 'from')
	let end = until === undefined ? duration : parseTimestamp(until, 'until')

	if (start >= duration) {
		throw new ArgError(`from (${formatTimestamp(start)}) is past the end of the video (${formatTimestamp(duration)})`)
	}
	if (end > duration) {
		notices.push(`until clamped to video end (${formatTimestamp(duration)})`)
		end = duration
	}
	if (end <= start) {
		throw new ArgError(`until (${formatTimestamp(end)}) must be after from (${formatTimestamp(start)})`)
	}
	return { start, end, full: start === 0 && end === duration, notices }
}
