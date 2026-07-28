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
//!
//! Timestamps travel as [`Ts`] — `f32` seconds, finite by construction,
//! minted once at the decode boundary. The ulp is under 1 ms out to ~4.5
//! hours and under 8 ms out to 24 hours — far finer than grid timestamps
//! (whole seconds on the wire) ever need. The harness computes windows in
//! `f64`, but the hand-off is CLI text re-parsed here, so no silent
//! in-process narrowing exists. Revisit `f64` before supporting multi-day
//! streams, where dense sampling could collide within one ulp.

use std::sync::Arc;

use image::{GrayImage, RgbImage};
use serde::{Deserialize, Serialize};

/// Presentation timestamp in seconds, 0-based at the stream start: finite by
/// construction.
///
/// Minted once where timestamps are born — the decode boundary, from the
/// frame's own pts — so a non-finite pts conversion can never enter the
/// pipeline as a frame timestamp, and everything downstream orders and
/// buckets timestamps without runtime finiteness checks. Negative values are
/// valid: edit lists and decoder pre-roll place frames before the stream
/// start. Non-decreasing order is a stream property, not a value property,
/// and stays the chunker's to enforce.
#[derive(Debug, Clone, Copy)]
pub struct Ts(f32);

impl Ts {
    /// Wrap a timestamp, or `None` unless it is finite.
    pub fn new(v: f32) -> Option<Self> {
        v.is_finite().then_some(Self(v))
    }

    /// The wrapped timestamp in seconds, finite.
    pub fn get(self) -> f32 {
        self.0
    }
}

// Equality and ordering delegate to `total_cmp`, which agrees with the IEEE
// comparison on the values `Ts` admits (finite, `-0.0` vs `0.0` excepted),
// so the manual `Eq`/`Ord` are mutually consistent by construction.
impl PartialEq for Ts {
    fn eq(&self, other: &Self) -> bool {
        self.0.total_cmp(&other.0).is_eq()
    }
}

impl Eq for Ts {}

impl PartialOrd for Ts {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Ts {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&other.0)
    }
}

/// A sampled frame before scoring: what extract, dedup, and chunking carry.
///
/// Holds only downscaled derivatives, never the source-resolution buffer.
/// Both derivatives are shared via [`Arc`], so retaining a representative —
/// a static-stretch RGB or a chunk-boundary grayscale tail — is a pointer
/// clone rather than a deep copy.
#[derive(Debug, Clone)]
pub struct CandidateFrame {
    /// Presentation timestamp.
    pub ts: Ts,
    /// Small grayscale image used for scoring convolutions.
    pub gray: Arc<GrayImage>,
    /// Small RGB image pasted onto the grid during composition.
    pub rgb: Arc<RgbImage>,
}

/// A frame after per-competition-set rank normalization.
///
/// Produced by `crate::score::score_frames` at the single scoring boundary.
#[derive(Debug, Clone)]
pub struct ScoredFrame {
    /// Presentation timestamp.
    pub ts: Ts,
    /// Small grayscale image, retained because `extract` re-scores globally.
    pub gray: Arc<GrayImage>,
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

    /// The wrapped score, finite and in `0..=100`.
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

    /// The wrapped difference, finite and in `0..=255`.
    pub fn get(self) -> f32 {
        self.0
    }
}

/// Source frame aspect ratio (`w/h`): finite and strictly positive by
/// construction.
///
/// Minted where aspect is born — the decode boundary, from a decoded frame's
/// own dimensions ([`Aspect::from_dims`]) — so layout geometry never has to
/// re-check for the 0/NaN/inf that would round every cell to nothing.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Aspect(f32);

impl Aspect {
    /// The square ratio — the capacity-planning default before any frame
    /// has been decoded.
    pub const SQUARE: Aspect = Aspect(1.0);

    /// Wrap a ratio, or `None` unless it is finite and strictly positive.
    pub fn new(v: f32) -> Option<Self> {
        (v.is_finite() && v > 0.0).then_some(Self(v))
    }

    /// The ratio of decoded frame dimensions, or `None` when either edge is
    /// zero. Non-zero `u32` edges always divide into a finite positive `f32`.
    pub fn from_dims(w: u32, h: u32) -> Option<Self> {
        (w > 0 && h > 0).then(|| Self(w as f32 / h as f32))
    }

    /// The wrapped ratio, finite and strictly positive.
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

    /// Left edge in pixels.
    pub fn x(&self) -> u32 {
        self.x
    }

    /// Top edge in pixels.
    pub fn y(&self) -> u32 {
        self.y
    }

    /// Width in pixels; at least 1.
    pub fn w(&self) -> u32 {
        self.w
    }

    /// Height in pixels; at least 1.
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
/// Serializes as the SCREAMING_SNAKE_CASE name; the wire strings are a
/// compatibility contract and must not move (locked by test).
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

/// JPEG caps image dimensions at 65535 px; a larger edge would run the whole
/// pipeline only to fail at encode time (after a multi-GB allocation). The
/// one ceiling every pixel-edge config knob validates against.
pub(crate) const MAX_IMAGE_EDGE: u32 = 65_535;

/// Validate a pixel-edge config value into `1..=`[`MAX_IMAGE_EDGE`].
///
/// The single upper-bound gate for every edge-shaped knob (canvas
/// `max_edge`, sampling `score_edge`): an edge past the ceiling feeds
/// decode-side pre-scaling and could only fail later as a capacity
/// overflow, after arbitrarily large allocations. Rejects with
/// [`crate::error::ErrorCode::InvalidConfig`], naming `field`.
pub(crate) fn checked_edge(field: &str, edge: u32) -> crate::error::Result<u32> {
    use crate::error::{Error, ErrorCode};
    if edge == 0 {
        return Err(Error::new(
            ErrorCode::InvalidConfig,
            format!("invalid {field}: 0 (must be >= 1)"),
        ));
    }
    if edge > MAX_IMAGE_EDGE {
        return Err(Error::new(
            ErrorCode::InvalidConfig,
            format!("invalid {field}: {edge} (exceeds JPEG's {MAX_IMAGE_EDGE}px dimension limit)"),
        ));
    }
    Ok(edge)
}

/// Resolved canvas geometry for layout: `1 <= min_cell_edge <= edge <=`
/// `MAX_IMAGE_EDGE` by construction.
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
    /// Validate canvas geometry: the canvas must fit one legible cell and
    /// stay within the JPEG dimension limit.
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
        let edge = checked_edge("max_edge", edge)?;
        Ok(Self {
            edge,
            min_cell_edge: min_cell_edge.max(1),
        })
    }

    /// Longer canvas edge in pixels.
    pub fn edge(&self) -> u32 {
        self.edge
    }

    /// Target shorter edge of a single cell, in pixels.
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

    /// Start of the range in seconds.
    pub fn start(&self) -> f32 {
        self.start
    }

    /// End of the range in seconds; at least `start`.
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

/// Per-grid subtitle outcome — the *result* of subtitle resolution.
///
/// The states a bare `Option<String>` collapsed — "the run has no subtitle
/// source" and "a source exists but no cue overlaps this range" — are
/// distinct constructors here, so downstream code can tell them apart by
/// type. "Resolution never ran" is not a state of this type at all: a
/// [`GridMeta`] carries no subtitle slot, and the wire shape is only
/// constructible by pairing one with a resolved outcome
/// ([`crate::wire::Grid::new`]). On the wire only [`GridSubtitles::Text`]
/// survives, as [`crate::wire::Grid::subtitles`].
#[derive(Debug, Clone, PartialEq)]
pub enum GridSubtitles {
    /// The run resolved no subtitle source at all.
    NoSource,
    /// A source exists, but no cue overlaps this grid's time range.
    NoCueInRange,
    /// Concatenated text of the cues overlapping this grid's time range.
    Text(String),
}

/// Metadata for one rendered grid image.
///
/// Deliberately has no subtitle slot: subtitle resolution is a separate
/// output-boundary step, and the wire shape ([`crate::wire::Grid`]) can only
/// be built by pairing this with its resolved [`GridSubtitles`] — a grid
/// whose resolution was forgotten does not typecheck, instead of silently
/// serializing as "no subtitles".
#[derive(Debug, Clone)]
pub struct GridMeta {
    /// Image payload, produced by the output session.
    pub image: GridImage,
    /// Span of the frames on this grid.
    pub time_range: TimeRange,
    /// Number of frames composed onto the grid.
    pub total_frames: usize,
    /// Hero frames, empty for uniform grids.
    pub key_frames: Vec<KeyFrameMeta>,
    /// Layout template label.
    pub layout: LayoutKind,
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
    /// Span of the frames on this grid.
    pub time_range: TimeRange,
    /// Number of frames composed onto the grid.
    pub total_frames: usize,
    /// Hero frames, empty for uniform grids.
    pub key_frames: Vec<KeyFrameMeta>,
    /// Layout template label.
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

/// Parse a text clock into seconds — the crate's single text clock grammar
/// and [`format_ts`]'s parsing counterpart: `SS`, `MM:SS`, or `HH:MM:SS`,
/// with `.` or `,` fractions allowed in any component.
///
/// Returns `None` unless every component parses finite and non-negative and
/// the weighted total stays finite (finite components can still overflow
/// f32 when scaled to seconds). Both textual clock consumers — `--range`
/// bounds and SRT/WebVTT cue timestamps — delegate here, so their accepted
/// syntax cannot drift and the finiteness discipline cannot be forgotten at
/// one entry point but not the other.
pub(crate) fn parse_clock(s: &str) -> Option<f32> {
    let normalized = s.replace(',', ".");
    let component = |p: &str| {
        let v: f32 = p.parse().ok()?;
        (v.is_finite() && v >= 0.0).then_some(v)
    };
    let parts: Vec<&str> = normalized.split(':').collect();
    let total = match parts.as_slice() {
        [sec] => component(sec)?,
        [m, sec] => component(m)? * 60.0 + component(sec)?,
        [h, m, sec] => component(h)? * 3600.0 + component(m)? * 60.0 + component(sec)?,
        _ => return None,
    };
    total.is_finite().then_some(total)
}

/// Metadata for a single key (hero) frame inside a grid. Rides the wire
/// unchanged inside [`crate::wire::Grid`], so both value fields are frozen
/// presentation, not pipeline data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyFrameMeta {
    /// Presentation timestamp, frozen at grid-build time by [`format_ts`]
    /// (`MM:SS`, `H:MM:SS` past an hour) — wire-frozen display text, not a
    /// number to compute with.
    pub timestamp: String,
    /// [`Score`] rounded to the nearest integer, so `0..=100` by
    /// provenance.
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
    fn parses_clock_with_comma_fraction() {
        let secs = parse_clock("1:23,500").expect("comma fraction is valid");
        assert!((secs - 83.5).abs() < 0.01);
    }

    #[test]
    fn parses_bare_seconds_clock() {
        assert_eq!(parse_clock("83"), Some(83.0));
        assert_eq!(parse_clock("83.5"), Some(83.5));
    }

    #[test]
    fn rejects_nonfinite_clock_component() {
        for bad in ["inf:00", "nan:30", "0:infinity", "NaN"] {
            assert_eq!(parse_clock(bad), None, "clock '{bad}' must be rejected");
        }
    }

    #[test]
    fn rejects_negative_clock_component() {
        assert_eq!(parse_clock("0:-30"), None);
        assert_eq!(parse_clock("-5"), None);
    }

    #[test]
    fn rejects_negative_component_with_positive_total() {
        // 2*60 - 30 = 90s: the old range parser checked only the total, so
        // this slipped through. Each component must be non-negative.
        assert_eq!(parse_clock("2:-30"), None);
    }

    #[test]
    fn rejects_overflowing_clock_total() {
        // Finite components can still overflow f32 when scaled to seconds.
        assert_eq!(parse_clock("1e38:0:0"), None);
    }

    #[test]
    fn rejects_extra_clock_groups() {
        assert_eq!(parse_clock("1:2:3:4"), None);
    }

    #[test]
    fn rejects_empty_clock_token() {
        assert_eq!(parse_clock(""), None);
        assert_eq!(parse_clock("1::5"), None);
    }

    #[test]
    fn rejects_non_finite_timestamps() {
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(Ts::new(bad).is_none(), "ts {bad} must be rejected");
        }
        assert!(Ts::new(-0.5).is_some(), "pre-roll timestamps are valid");
    }

    #[test]
    fn orders_timestamps_ascending() {
        let ts = |v| Ts::new(v).expect("finite test ts");
        let mut v = [ts(9.0), ts(-0.5), ts(3.0)];
        v.sort();
        assert_eq!(v, [ts(-0.5), ts(3.0), ts(9.0)]);
    }

    #[test]
    fn rejects_degenerate_aspect() {
        for bad in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            assert!(Aspect::new(bad).is_none(), "aspect {bad} must be rejected");
        }
        assert!(Aspect::new(16.0 / 9.0).is_some());
        assert!(Aspect::from_dims(0, 9).is_none(), "zero width");
        assert!(Aspect::from_dims(16, 0).is_none(), "zero height");
        let a = Aspect::from_dims(1920, 1080).expect("real dims form an aspect");
        assert!((a.get() - 16.0 / 9.0).abs() < 1e-6);
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
    fn serializes_layout_and_position_as_wire_strings() {
        // Wire compatibility: the serialized strings must not move.
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
