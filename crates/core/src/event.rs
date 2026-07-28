// SPDX-License-Identifier: Apache-2.0
//! Diagnostic events the kernel emits, and the sink they flow into.
//!
//! Events are data, not prose: the consumer decides wording, destination,
//! or silence. The kernel never writes to stderr itself.

use std::path::Path;

/// One diagnostic event. Non-exhaustive: consumers must have a catch-all.
///
/// Every variant carries the raw numbers behind a diagnostic, never a
/// pre-rendered string — the exact wording, float precision, and timestamp
/// format are the renderer's choice, so the kernel stays free of presentation
/// policy. The `'a` borrow lets a variant reference a path or file name that
/// outlives the call without allocating.
#[non_exhaustive]
#[derive(Debug)]
pub enum Event<'a> {
    /// A seek to `start` seconds failed; decoding falls back to a sequential
    /// read from the start. `error` is the decoder's message.
    SeekFallback { start: f32, error: String },
    /// Fixed-fps sampling announced over a bounded window. `end` is `None`
    /// when the window runs to EOF.
    FixedFpsWindow {
        fps: f32,
        start: f32,
        end: Option<f32>,
    },
    /// Fixed-fps sampling announced over the whole video.
    FixedFps { fps: f32 },
    /// Keyframe sampling announced: `count` I-frames spanning `duration`
    /// seconds, with the given mean and max inter-keyframe gaps.
    KeyframesMode {
        count: usize,
        duration: f32,
        mean_gap: f32,
        max_gap: f32,
    },
    /// Adaptive sampling announced: `keyframes` I-frames (largest gap
    /// `max_gap`), plus `sparse_regions` fill regions past `threshold`.
    AdaptiveMode {
        keyframes: usize,
        max_gap: f32,
        sparse_regions: usize,
        threshold: f32,
    },
    /// Adaptive sampling degraded to a single fixed-fps pass because sparse
    /// regions cover `coverage_percent` of the video (already computed).
    AdaptiveFallbackFixedFps { coverage_percent: f32 },
    /// One sparse region being filled at fixed fps, in seconds.
    FillingSparseRegion { start: f32, end: f32 },
    /// A defaulted keyframe/adaptive request degraded to fixed-fps: only
    /// `count` keyframes, below the `minimum` required.
    KeyframePoorDegrade { count: usize, minimum: usize },
    /// A static stretch `[start, end]` seconds was collapsed into `file`.
    StaticCollapsed { start: f32, end: f32, file: &'a str },
    /// A decoded frame's timestamp could not be placed on the stream's
    /// timeline and the frame was dropped (first occurrence in a decode
    /// pass; the pass total arrives as [`Event::TimestampsDropped`]).
    /// `last` is the last released timestamp, when any.
    UnplaceableTimestamp { ts: f32, last: Option<f32> },
    /// End-of-decode accounting: `count` decoded frames were dropped off
    /// the timeline over the whole pass. Emitted only when `count > 0`.
    TimestampsDropped { count: u64 },
    /// A run finished writing `count` grids to `dir`.
    GridsWritten { count: usize, dir: &'a Path },
    /// An embedded subtitle stream held no text cues; the auto path continues
    /// without subtitles.
    EmbeddedSubtitlesEmpty,
    /// A directory could not be listed while scanning for subtitle sidecars.
    SidecarScanFailed { dir: &'a Path, error: String },
    /// A directory entry was unreadable during the sidecar scan.
    SidecarEntryUnreadable { error: String },
}

/// Where the kernel reports diagnostics. `Sync` because emission sites sit
/// on both the orchestrator and decode-open paths.
pub trait EventSink: Sync {
    /// Handle one diagnostic event. Implementations decide the wording,
    /// destination, or silence; the call must not fail the run.
    fn event(&self, event: Event<'_>);
}

/// Discards every event; the default for library consumers who want silence.
#[derive(Debug, Clone, Copy)]
pub struct Discard;

impl EventSink for Discard {
    fn event(&self, _: Event<'_>) {}
}
