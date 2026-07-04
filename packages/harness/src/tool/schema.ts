/**
 * Single source of truth for the `video` tool.
 * Defines input schema, argument types, and the result family.
 */

import type { Metadata } from '../core/contract.ts'

// Intent-level presets; mapped to core parameters in policy/look.ts
export type Look = 'overview' | 'detailed' | 'read_text' | 'track_motion'

export const LOOKS: readonly Look[] = ['overview', 'detailed', 'read_text', 'track_motion']

// Arguments for the video tool
export interface VideoArgs {
	// Local file path or URL
	source: string
	// Window start: seconds or "MM:SS" / "H:MM:SS"; omit = start of video
	from?: string | number
	// Window end, same format; omit = end of video
	until?: string | number
	// How to look at the window; omit = balanced default
	look?: Look
	// Expert escape hatch: verbatim core CLI arguments, e.g. "process --strategy keyframes"
	raw?: string
	// Echo the consent token issued by the server in the previous response
	confirm?: string
}

// JSON Schema for MCP tools/list, kept in sync with VideoArgs
export const VIDEO_INPUT_SCHEMA = {
	type: 'object',
	properties: {
		source: {
			type: 'string',
			description:
				'Local video file path or URL. Call with only `source` first; the response tells you what to do next.',
		},
		from: {
			anyOf: [{ type: 'string' }, { type: 'number' }],
			description: 'Window start: seconds (83) or timestamp ("1:23", "1:02:03"). Omit to start from the beginning.',
		},
		until: {
			anyOf: [{ type: 'string' }, { type: 'number' }],
			description: 'Window end, same formats. Omit to go to the end of the video.',
		},
		look: {
			type: 'string',
			enum: LOOKS as Look[],
			description:
				'How to look: overview (fast triage), detailed (miss nothing), read_text (few large frames to read on-screen text/UI), track_motion (dense small frames for fast action). Subtitle text always rides along in metadata regardless; this only affects frame size/density. Omit for a balanced default.',
		},
		raw: {
			type: 'string',
			description:
				'Expert escape hatch: verbatim agent-video core CLI arguments starting with the subcommand, e.g. "process --strategy keyframes --base-fps 0.2". Mutually exclusive with from/until/look. Input/output plumbing is injected by the server.',
		},
		confirm: {
			type: 'string',
			description:
				'Echo a consent token issued by a previous response to authorize the pending action. Tokens are single-use and bound to the latest plan; pass confirm alone with source.',
		},
	},
	required: ['source'],
	additionalProperties: false,
} as const

// Single-line tool description shown in the tool list
export const VIDEO_TOOL_DESCRIPTION =
	'Watch a video (local path or URL) as VLM-readable frame grids with timestamps and subtitles. Call with just `source` first: short videos run immediately, everything else answers with metadata and copyable next calls.'

// ── result family ──

// Probe summary shown to the model in a Plan response
export interface ProbeSummary {
	// Seconds
	duration: number
	width: number
	height: number
	sizeBytes: number
	container?: string
	// Human-readable descriptions, e.g. "embedded #0 (eng, subrip)", "sidecar movie.srt"
	subtitles: string[]
}

// Argument object the model can copy verbatim, with a rationale
export interface RecommendedCall {
	call: Partial<VideoArgs>
	note: string
}

// Consent token sent to the model, spelling out its effect
export interface OfferedToken {
	token: string
	effect: string
}

// Delivered grids; notices carry degradation and confirmation info
export interface Ready {
	status: 'ready'
	metadata: Metadata
	notices: string[]
}

// Probe-only response: summary plus ready-to-paste follow-up calls
export interface Plan {
	status: 'plan'
	probe: ProbeSummary
	guidance: string
	recommended_calls: RecommendedCall[]
}

// Request requires explicit consent; retry with confirm
export interface ConfirmRequired {
	status: 'confirm_required'
	consequence: string
	tokens: OfferedToken[]
}

// Invalid input or environment failure; message can be shown to the model
export interface Failed {
	status: 'failed'
	error: string
}

export type VideoResult = Ready | Plan | ConfirmRequired | Failed
