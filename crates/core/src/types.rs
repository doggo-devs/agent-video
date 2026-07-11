// SPDX-License-Identifier: Apache-2.0
//! Public data types shared across the pipeline.
//!
//! The central design constraint lives here: neither frame type holds a
//! full-resolution decoded frame. Right after a frame is decoded, `extract`
//! derives a small grayscale image (for scoring) and a small RGB image (for
//! composition), then drops the original buffer. Frames travel as
//! [`CandidateFrame`] until the scoring boundary, where per-competition-set
//! rank normalization turns them into [`ScoredFrame`]; both share their RGB
//! pixels through an [`Arc`], so retaining a representative is a pointer clone,
//! not a deep copy. This keeps peak memory in the hundreds-of-MB range instead
//! of GBs for long videos.

use std::sync::Arc;

use image::{GrayImage, RgbImage};
use serde::{Deserialize, Serialize};

/// A sampled frame before scoring: what extract, dedup, and chunking carry.
///
/// Holds only downscaled derivatives, never the source-resolution buffer. The
/// RGB is shared via [`Arc`] so retaining a static-stretch representative is a
/// pointer clone rather than a ~1.4 MB deep copy.
#[derive(Debug, Clone)]
pub struct CandidateFrame {
    /// Presentation timestamp in seconds.
    pub ts: f32,
    /// Small grayscale image used for scoring convolutions.
    pub gray: GrayImage,
    /// Small RGB image pasted onto the grid during composition.
    pub rgb: Arc<RgbImage>,
}

/// A frame after per-competition-set rank normalization.
///
/// Produced by `crate::score::score_frames` at the single scoring boundary.
#[derive(Debug, Clone)]
pub struct ScoredFrame {
    /// Presentation timestamp in seconds.
    pub ts: f32,
    /// Small grayscale image, retained because `extract` re-scores globally.
    pub gray: GrayImage,
    /// Small RGB image pasted onto the grid during composition.
    pub rgb: Arc<RgbImage>,
    /// Final weighted score.
    pub score: Score,
    /// Raw scene difference from the frame scored immediately before this
    /// one. Drives hero detection, which needs absolute magnitude where
    /// `score` only preserves rank, and travels with the frame into global
    /// re-ranks (`extract`'s compaction), where recomputing it between
    /// non-consecutive frames would be noise.
    pub scene_diff: SceneDiff,
}

/// Rank-normalized frame score: finite and in `0..=100` by construction.
///
/// Only the scoring boundary (`crate::score`) mints values; everything
/// downstream (selection, eviction, layout promotion) reads them via
/// [`Score::get`], so a `-999` or `NaN` score is unrepresentable past that
/// boundary.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Score(f32);

impl Score {
    /// Wrap a score, or `None` unless it is a finite value in `0..=100`.
    pub fn new(v: f32) -> Option<Self> {
        (v.is_finite() && (0.0..=100.0).contains(&v)).then_some(Self(v))
    }

    pub fn get(self) -> f32 {
        self.0
    }
}

/// Raw scene difference: mean absolute grayscale difference from the frame's
/// true temporal predecessor, finite and in `0..=255` by construction (the
/// range a mean of `u8` differences can span).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SceneDiff(f32);

impl SceneDiff {
    /// Wrap a difference, or `None` unless it is a finite value in `0..=255`.
    pub fn new(v: f32) -> Option<Self> {
        (v.is_finite() && (0.0..=255.0).contains(&v)).then_some(Self(v))
    }

    pub fn get(self) -> f32 {
        self.0
    }
}

/// An axis-aligned rectangle on the output canvas, in pixels.
///
/// Non-degenerate by construction: `w` and `h` are at least 1 and the far
/// corner (`x + w`, `y + h`) fits in `u32`, so pasting and border drawing
/// can add coordinates without overflow checks. [`Rect::new`] is the only
/// door.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct Rect {
    x: u32,
    y: u32,
    w: u32,
    h: u32,
}

impl Rect {
    /// Construct a rectangle, or `None` when the area would be zero or the
    /// far corner would leave the `u32` coordinate space.
    pub fn new(x: u32, y: u32, w: u32, h: u32) -> Option<Self> {
        (w > 0 && h > 0 && x.checked_add(w).is_some() && y.checked_add(h).is_some())
            .then_some(Self { x, y, w, h })
    }

    pub fn x(&self) -> u32 {
        self.x
    }

    pub fn y(&self) -> u32 {
        self.y
    }

    pub fn w(&self) -> u32 {
        self.w
    }

    pub fn h(&self) -> u32 {
        self.h
    }

    /// The rectangle mirrored across the main diagonal — the portrait
    /// transpose layout uses to turn hero rows into hero columns. Validity is
    /// preserved: swapping the axes of a valid rect cannot invalidate it.
    pub fn transposed(self) -> Self {
        Self {
            x: self.y,
            y: self.x,
            w: self.h,
            h: self.w,
        }
    }

    /// Returns true if this rectangle shares any interior area with `other`.
    /// Used by layout tests to assert non-overlapping placement.
    #[cfg(test)]
    pub fn overlaps(&self, other: &Rect) -> bool {
        let ax2 = self.x + self.w;
        let ay2 = self.y + self.h;
        let bx2 = other.x + other.w;
        let by2 = other.y + other.h;
        self.x < bx2 && other.x < ax2 && self.y < by2 && other.y < ay2
    }
}

/// Grid layout template. The hero-count names (UNIFORM / HERO_LEFT /
/// DUAL_HERO / TOP_STRIP) label composed grids for metadata; STATIC marks a
/// static stretch collapsed into a single representative image.
///
/// Serializes as the SCREAMING_SNAKE_CASE name, so the JSON contract is
/// byte-identical to the old string constants.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum LayoutKind {
    Uniform,
    HeroLeft,
    DualHero,
    TopStrip,
    Static,
}

impl LayoutKind {
    /// The wire name, identical to the serialized form (locked by test).
    pub fn as_str(self) -> &'static str {
        match self {
            LayoutKind::Uniform => "UNIFORM",
            LayoutKind::HeroLeft => "HERO_LEFT",
            LayoutKind::DualHero => "DUAL_HERO",
            LayoutKind::TopStrip => "TOP_STRIP",
            LayoutKind::Static => "STATIC",
        }
    }
}

/// Role of a key frame inside its grid. Serializes lowercase (`"hero"`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum GridPosition {
    Hero,
}

impl GridPosition {
    /// The wire name, identical to the serialized form (locked by test).
    pub fn as_str(self) -> &'static str {
        match self {
            GridPosition::Hero => "hero",
        }
    }
}

/// Cell legibility tier: sets the minimum readable cell edge. The canvas edge
/// is fixed by `max_edge`, so lower quality packs more (smaller) frames onto
/// the same canvas.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "clap", derive(clap::ValueEnum))]
#[serde(rename_all = "lowercase")]
pub enum Quality {
    Low,
    Medium,
    High,
}

/// Detail tier: controls sampling fps.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "clap", derive(clap::ValueEnum))]
#[serde(rename_all = "lowercase")]
pub enum Detail {
    Coarse,
    Standard,
    Fine,
}

/// JPEG caps image dimensions at 65535 px; a larger canvas would run the whole
/// pipeline only to fail at encode time (after a multi-GB allocation).
const MAX_CANVAS_EDGE: u32 = 65_535;

/// Resolved canvas geometry for layout: `1 <= min_cell_edge <= edge <=`
/// `MAX_CANVAS_EDGE` by construction.
#[derive(Debug, Clone, Copy)]
pub struct CanvasSpec {
    /// Longer canvas edge in pixels.
    edge: u32,
    /// Target shorter edge of a single cell, in pixels. A soft lower bound:
    /// capacity planning treats the canvas as square while the packed canvas is
    /// only roughly square, so cells can land slightly below it.
    min_cell_edge: u32,
}

impl CanvasSpec {
    pub fn new(edge: u32, min_cell_edge: u32) -> crate::error::Result<Self> {
        use crate::error::{Error, ErrorCode};
        // Reject a canvas too small for one legible cell (would "succeed" as a
        // degenerate, unreadable image).
        if edge < min_cell_edge.max(1) {
            return Err(Error::new(
                ErrorCode::InvalidConfig,
                format!(
                    "invalid max_edge: {edge} (canvas must fit one cell at the quality floor, \
                     >= {min_cell_edge}px)"
                ),
            ));
        }
        // Cap before the pipeline allocates from this value; JPEG encoding
        // would reject the dimensions only at the very end.
        if edge > MAX_CANVAS_EDGE {
            return Err(Error::new(
                ErrorCode::InvalidConfig,
                format!(
                    "invalid max_edge: {edge} (exceeds JPEG's {MAX_CANVAS_EDGE}px dimension limit)"
                ),
            ));
        }
        Ok(Self {
            edge,
            min_cell_edge: min_cell_edge.max(1),
        })
    }

    pub fn edge(&self) -> u32 {
        self.edge
    }

    pub fn min_cell_edge(&self) -> u32 {
        self.min_cell_edge
    }
}

/// A closed time range in seconds; `start <= end` and both finite by
/// construction. Displays as `"MM:SS-MM:SS"` (`H:MM:SS` past an hour) —
/// the form the wire contract carries.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TimeRange {
    start: f32,
    end: f32,
}

impl TimeRange {
    /// Construct a range, or `None` if either bound is non-finite or `start >
    /// end`.
    pub fn new(start: f32, end: f32) -> Option<Self> {
        (start.is_finite() && end.is_finite() && start <= end).then_some(Self { start, end })
    }

    pub fn start(&self) -> f32 {
        self.start
    }

    pub fn end(&self) -> f32 {
        self.end
    }

    /// Length of the range in seconds; non-negative by construction.
    pub fn duration(&self) -> f32 {
        self.end - self.start
    }
}

impl std::fmt::Display for TimeRange {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}-{}", format_ts(self.start), format_ts(self.end))
    }
}

/// A grid's image payload: exactly one of a written file path or an inline
/// base64 blob, by construction.
///
/// Serialized externally tagged, so a single-variant value becomes a
/// single-entry map (`{"file": …}` / `{"base64": …}`) that flattens into
/// [`crate::wire::Grid`] under the original key.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum GridImage {
    File(String),
    Base64(String),
}

/// Per-grid subtitle outcome.
///
/// The states a bare `Option<String>` collapsed — "resolution never ran",
/// "the run has no subtitle source", and "a source exists but no cue overlaps
/// this range" — are distinct constructors here, so downstream code can tell
/// "not found" from "not looked for" by type. On the wire only
/// [`GridSubtitles::Text`] survives, as [`crate::wire::Grid::subtitles`].
#[derive(Debug, Clone, Default, PartialEq)]
pub enum GridSubtitles {
    /// Subtitle resolution has not run for this grid yet (worker drafts,
    /// static entries before the output pass).
    #[default]
    Unresolved,
    /// The run resolved no subtitle source at all.
    NoSource,
    /// A source exists, but no cue overlaps this grid's time range.
    NoCueInRange,
    /// Concatenated text of the cues overlapping this grid's time range.
    Text(String),
}

/// Metadata for one rendered grid image; converts into [`crate::wire::Grid`]
/// at the output boundary.
#[derive(Debug, Clone)]
pub struct GridMeta {
    pub image: GridImage,
    pub time_range: TimeRange,
    pub total_frames: usize,
    pub key_frames: Vec<KeyFrameMeta>,
    pub layout: LayoutKind,
    pub subtitles: GridSubtitles,
}

/// A grid's metadata before its image payload exists.
///
/// Produced by chunk workers, which render JPEG bytes but deliberately cannot
/// construct a [`GridImage`]: whether those bytes become a written file or an
/// inline base64 blob is the output session's decision alone
/// ([`crate::pipeline`]'s `OutputSession::emit_image`). The draft is completed
/// via [`GridDraft::into_meta`] at emission.
#[derive(Debug, Clone)]
pub struct GridDraft {
    pub time_range: TimeRange,
    pub total_frames: usize,
    pub key_frames: Vec<KeyFrameMeta>,
    pub layout: LayoutKind,
}

impl GridDraft {
    /// Complete the draft with the image payload the session produced.
    pub fn into_meta(self, image: GridImage) -> GridMeta {
        GridMeta {
            image,
            time_range: self.time_range,
            total_frames: self.total_frames,
            key_frames: self.key_frames,
            layout: self.layout,
            subtitles: GridSubtitles::Unresolved,
        }
    }
}

/// Format a timestamp as `MM:SS`, or `H:MM:SS` from one hour up (keyframe
/// strategies support videos past the hour), clamping negatives to zero and
/// rounding to the nearest second.
pub fn format_ts(ts: f32) -> String {
    let total = ts.max(0.0).round() as u32;
    let (h, m, s) = (total / 3600, total % 3600 / 60, total % 60);
    if h > 0 {
        format!("{h}:{m:02}:{s:02}")
    } else {
        format!("{m:02}:{s:02}")
    }
}

/// Metadata for a single key (hero) frame inside a grid.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyFrameMeta {
    pub timestamp: String,
    pub score: u32,
    pub grid_position: GridPosition,
}

/// The resolved settings echoed back into the output document.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Settings {
    pub max_edge: u32,
    pub quality: Quality,
    pub detail: Detail,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn formats_timestamp_as_mmss() {
        assert_eq!(format_ts(0.0), "00:00");
        assert_eq!(format_ts(83.0), "01:23");
        assert_eq!(format_ts(605.4), "10:05");
    }

    #[test]
    fn formats_hour_plus_timestamps_with_hours() {
        assert_eq!(format_ts(3600.0), "1:00:00");
        assert_eq!(format_ts(5025.0), "1:23:45");
        assert_eq!(format_ts(3599.6), "1:00:00", "rounds up into the hour");
        assert_eq!(format_ts(3599.4), "59:59", "stays MM:SS below the hour");
    }

    #[test]
    fn rejects_out_of_range_scores_and_diffs() {
        for bad in [-0.001, 100.001, f32::NAN, f32::INFINITY] {
            assert!(Score::new(bad).is_none(), "score {bad} must be rejected");
        }
        assert!(Score::new(0.0).is_some());
        assert!(Score::new(100.0).is_some());
        for bad in [-1.0, 255.5, f32::NAN, f32::NEG_INFINITY] {
            assert!(SceneDiff::new(bad).is_none(), "diff {bad} must be rejected");
        }
        assert!(SceneDiff::new(255.0).is_some());
    }

    #[test]
    fn rejects_degenerate_rects() {
        assert!(Rect::new(0, 0, 0, 5).is_none(), "zero width");
        assert!(Rect::new(0, 0, 5, 0).is_none(), "zero height");
        assert!(
            Rect::new(u32::MAX, 0, 1, 1).is_none(),
            "far corner overflows"
        );
        assert!(Rect::new(1, 2, 3, 4).is_some());
    }

    #[test]
    fn transposes_rect_across_the_diagonal() {
        let r = Rect::new(1, 2, 3, 4).expect("valid rect").transposed();
        assert_eq!((r.x(), r.y(), r.w(), r.h()), (2, 1, 4, 3));
    }

    #[test]
    fn serializes_layout_and_position_as_legacy_strings() {
        // JSON payload must not move from the legacy string constants.
        for (kind, expected) in [
            (LayoutKind::Uniform, "UNIFORM"),
            (LayoutKind::HeroLeft, "HERO_LEFT"),
            (LayoutKind::DualHero, "DUAL_HERO"),
            (LayoutKind::TopStrip, "TOP_STRIP"),
            (LayoutKind::Static, "STATIC"),
        ] {
            let json = serde_json::to_value(kind).expect("serialize layout kind");
            assert_eq!(json, expected);
            assert_eq!(kind.as_str(), expected, "as_str mirrors the wire name");
        }
        let json = serde_json::to_value(GridPosition::Hero).expect("serialize grid position");
        assert_eq!(json, "hero");
        assert_eq!(GridPosition::Hero.as_str(), "hero");
    }

    #[test]
    fn displays_time_range_as_wire_string() {
        let range = TimeRange::new(0.0, 90.0).expect("valid range");
        assert_eq!(range.to_string(), "00:00-01:30");
        let long = TimeRange::new(3600.0, 5025.0).expect("valid range");
        assert_eq!(long.to_string(), "1:00:00-1:23:45");
    }
}
