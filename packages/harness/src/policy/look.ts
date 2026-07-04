//! `look` presets: intent-level names mapped to core knobs.
//!
//! Numbers mirror crates/core/src/config.rs — chunk_seconds drives grid
//! count (one chunk composes exactly one grid), detail drives sampling
//! fps, quality drives cell size. Config may override per-preset knobs
//! later; keep the shape data-only.

import type { Look } from "../tool/schema.ts";

/** Core knobs behind one `look` value. */
export interface LookPreset {
  /** Core flags, ready to append (never touches I/O flags). */
  flags: readonly string[];
  /** Effective chunk length in seconds — the grid-count denominator. */
  chunkSeconds: number;
}

/** Core default when no `--chunk-seconds` / profile override applies. */
export const DEFAULT_CHUNK_SECONDS = 60;

/** Preset applied when `look` is omitted. */
export const BALANCED: LookPreset = {
  flags: ["--detail", "standard", "--quality", "low"],
  chunkSeconds: DEFAULT_CHUNK_SECONDS,
};

export const LOOK_PRESETS: Record<Look, LookPreset> = {
  overview: {
    flags: ["--detail", "coarse", "--quality", "low"],
    chunkSeconds: DEFAULT_CHUNK_SECONDS,
  },
  detailed: {
    flags: ["--detail", "fine", "--quality", "medium"],
    chunkSeconds: DEFAULT_CHUNK_SECONDS,
  },
  read_text: {
    flags: ["--detail", "standard", "--quality", "high"],
    chunkSeconds: DEFAULT_CHUNK_SECONDS,
  },
  track_motion: {
    // The action profile reweights scoring toward scene change and
    // tightens min-gap; quality low keeps cells small and dense.
    flags: ["--profile", "action", "--quality", "low"],
    chunkSeconds: DEFAULT_CHUNK_SECONDS,
  },
};

export function presetFor(look: Look | undefined): LookPreset {
  return look === undefined ? BALANCED : LOOK_PRESETS[look];
}
