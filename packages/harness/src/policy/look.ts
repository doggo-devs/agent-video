/**
 * `look` presets: intent-level names mapped to core knobs.
 * Numbers mirror crates/core/src/config.rs; the shape stays data-only.
 */

import type { Look } from '../tool/schema.ts'

// The core knobs behind a single look value
export interface LookPreset {
	// core flags ready to append, no I/O flags involved
	flags: readonly string[]
	// effective chunk seconds, i.e. the denominator of the grid count
	chunkSeconds: number
}

// core default when no --chunk-seconds / profile override is given
export const DEFAULT_CHUNK_SECONDS = 60

// preset applied when look is omitted
export const BALANCED: LookPreset = {
	flags: ['--detail', 'standard', '--quality', 'low'],
	chunkSeconds: DEFAULT_CHUNK_SECONDS,
}

export const LOOK_PRESETS: Record<Look, LookPreset> = {
	overview: { flags: ['--detail', 'coarse', '--quality', 'low'], chunkSeconds: DEFAULT_CHUNK_SECONDS },
	detailed: { flags: ['--detail', 'fine', '--quality', 'medium'], chunkSeconds: DEFAULT_CHUNK_SECONDS },
	read_text: { flags: ['--detail', 'standard', '--quality', 'high'], chunkSeconds: DEFAULT_CHUNK_SECONDS },
	track_motion: {
		// action profile favors scene changes and tightens min-gap; quality low makes cells small and dense
		flags: ['--profile', 'action', '--quality', 'low'],
		chunkSeconds: DEFAULT_CHUNK_SECONDS,
	},
}

export function presetFor(look: Look | undefined): LookPreset {
	return look === undefined ? BALANCED : LOOK_PRESETS[look]
}
