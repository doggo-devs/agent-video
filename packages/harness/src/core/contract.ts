//! TypeScript mirror of the core's machine interface: the `Metadata`
//! document emitted on stdout with `--json`, or written to `metadata.json`
//! in file mode. Shapes mirror `crates/core/src/types.rs` field-for-field.
//!
//! No runtime validation on purpose: platform packages are exact-pinned to
//! the harness version, so the binary producing this JSON is always the
//! same release as this mirror.

/** Cell legibility tier; lower packs more, smaller frames per canvas. */
export type Quality = "low" | "medium" | "high";

/** Detail tier; controls sampling fps. */
export type Detail = "coarse" | "standard" | "fine";

/**
 * A grid's image payload: exactly one of a written file path (file mode)
 * or an inline base64 JPEG (`--json` mode).
 */
export type GridImage = { file: string } | { base64: string };

/** One key (hero) frame inside a grid. */
export interface KeyFrameMeta {
  /** "MM:SS", or "H:MM:SS" past an hour. */
  timestamp: string;
  score: number;
  /** Human-readable cell position, e.g. "r2c3". */
  grid_position: string;
}

/** Metadata for one rendered grid image. */
export type GridMeta = GridImage & {
  /** "MM:SS-MM:SS" ("H:MM:SS" past an hour). */
  time_range: string;
  total_frames: number;
  key_frames: KeyFrameMeta[];
  layout: string;
  subtitles?: string;
};

/** The resolved settings echoed back by the core. */
export interface Settings {
  max_edge: number;
  quality: Quality;
  detail: Detail;
}

/** Top-level structured output of `process` / `extract`. */
export interface Metadata {
  source: string;
  /** Video duration in seconds. */
  duration: number;
  settings: Settings;
  grids: GridMeta[];
}
