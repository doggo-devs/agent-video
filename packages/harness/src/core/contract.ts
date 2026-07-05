/**
 * TypeScript mirror of the core's machine interface.
 * The `Metadata` document emitted with `--json` (mirrors
 * `crates/core/src/types.rs`), field-for-field.
 */

// cell legibility tier; lower packs more, smaller frames per canvas
export type Quality = 'low' | 'medium' | 'high'

// detail tier; controls sampling fps
export type Detail = 'coarse' | 'standard' | 'fine'

// a grid's image payload: written file path (file mode) or inline base64 JPEG (`--json` mode)
export type GridImage = { file: string } | { base64: string }

// one key (hero) frame inside a grid
export interface KeyFrameMeta {
	// "MM:SS", or "H:MM:SS" past an hour
	timestamp: string
	score: number
	// human-readable cell position, e.g. "r2c3"
	grid_position: string
}

// metadata for one rendered grid image
export type GridMeta = GridImage & {
	// "MM:SS-MM:SS" ("H:MM:SS" past an hour)
	time_range: string
	total_frames: number
	key_frames: KeyFrameMeta[]
	layout: string
	subtitles?: string
}

// the resolved settings echoed back by the core
export interface Settings {
	max_edge: number
	quality: Quality
	detail: Detail
}

// top-level structured output of `process` / `extract`
export interface Metadata {
	source: string
	// video duration in seconds
	duration: number
	settings: Settings
	grids: GridMeta[]
}
