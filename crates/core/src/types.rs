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
use serde::Serialize;

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

/// A frame after per-competition-set rank normalization; `score` is `0..=100`.
///
/// Produced by [`crate::score::score_frames`] at the single scoring boundary.
#[derive(Debug, Clone)]
pub struct ScoredFrame {
    /// Presentation timestamp in seconds.
    pub ts: f32,
    /// Small grayscale image, retained because `extract` re-scores globally.
    pub gray: GrayImage,
    /// Small RGB image pasted onto the grid during composition.
    pub rgb: Arc<RgbImage>,
    /// Final weighted score in `0..=100`.
    pub score: f32,
    /// Raw scene difference from the frame scored immediately before this one
    /// (mean absolute grayscale difference, `0..=255`). Drives hero detection,
    /// which needs absolute magnitude where `score` only preserves rank, and
    /// travels with the frame into global re-ranks (`extract`'s compaction),
    /// where recomputing it between non-consecutive frames would be noise.
    pub scene_diff: f32,
}

/// An axis-aligned rectangle on the output canvas, in pixels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct Rect {
    pub x: u32,
    pub y: u32,
    pub w: u32,
    pub h: u32,
}

impl Rect {
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

/// Cell legibility tier: sets the minimum readable cell edge. The canvas edge
/// is fixed by `max_edge`, so lower quality packs more (smaller) frames onto
/// the same canvas.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, clap::ValueEnum)]
#[serde(rename_all = "lowercase")]
pub enum Quality {
    Low,
    Medium,
    High,
}

/// Detail tier: controls sampling fps.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, clap::ValueEnum)]
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
/// [`MAX_CANVAS_EDGE`] by construction.
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
        // The canvas must hold at least one cell at the quality tier's
        // legibility floor; anything smaller (down to --max-edge 0) would
        // still "succeed" as a degenerate, unreadable image.
        if edge < min_cell_edge.max(1) {
            return Err(Error::new(
                ErrorCode::InvalidConfig,
                format!(
                    "invalid max_edge: {edge} (canvas must fit one cell at the quality floor, \
                     >= {min_cell_edge}px)"
                ),
            ));
        }
        // The canvas allocation is sized directly from this value; without a
        // cap, an oversized flag burns the full pipeline before JPEG encoding
        // rejects the dimensions (or the allocation aborts first).
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
/// construction. Serializes as `"MM:SS-MM:SS"` (`H:MM:SS` past an hour).
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

impl Serialize for TimeRange {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(&format!(
            "{}-{}",
            format_ts(self.start),
            format_ts(self.end)
        ))
    }
}

/// A grid's image payload: exactly one of a written file path or an inline
/// base64 blob, by construction.
///
/// Serialized externally tagged, so a single-variant value becomes a
/// single-entry map (`{"file": …}` / `{"base64": …}`) that flattens into
/// [`GridMeta`] under the original key.
#[derive(Debug, Clone, Serialize)]
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
/// "not found" from "not looked for" by type. The JSON contract is unchanged:
/// only [`GridSubtitles::Text`] serializes, under the same `subtitles` key.
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

impl GridSubtitles {
    /// Whether serialization omits the field — every state except `Text`.
    pub fn is_absent(&self) -> bool {
        !matches!(self, Self::Text(_))
    }
}

impl Serialize for GridSubtitles {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        match self {
            Self::Text(t) => s.serialize_str(t),
            // Reached only without `skip_serializing_if`; the absent states
            // have no textual payload to expose.
            _ => s.serialize_none(),
        }
    }
}

/// Metadata for one rendered grid image.
#[derive(Debug, Clone, Serialize)]
pub struct GridMeta {
    #[serde(flatten)]
    pub image: GridImage,
    /// Time range as "MM:SS-MM:SS" ("H:MM:SS" past an hour) for JSON output.
    pub time_range: TimeRange,
    pub total_frames: usize,
    pub key_frames: Vec<KeyFrameMeta>,
    pub layout: &'static str,
    #[serde(skip_serializing_if = "GridSubtitles::is_absent")]
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
    pub layout: &'static str,
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
#[derive(Debug, Clone, Serialize)]
pub struct KeyFrameMeta {
    pub timestamp: String,
    pub score: u32,
    pub grid_position: &'static str,
}

/// Top-level structured output written to `metadata.json`.
#[derive(Debug, Clone, Serialize)]
pub struct Metadata {
    pub source: String,
    pub duration: f32,
    pub settings: Settings,
    pub grids: Vec<GridMeta>,
}

/// The resolved settings echoed back into `metadata.json`.
#[derive(Debug, Clone, Copy, Serialize)]
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
    fn serializes_grid_image_under_original_key() {
        // The flattened enum must keep the exact `"file"` / `"base64"` JSON
        // keys, with only one present at a time.
        let file_meta = GridMeta {
            image: GridImage::File("grid_001.jpg".into()),
            time_range: TimeRange::new(0.0, 5.0).expect("valid range"),
            total_frames: 3,
            key_frames: Vec::new(),
            layout: "UNIFORM",
            subtitles: GridSubtitles::default(),
        };
        let json = serde_json::to_value(&file_meta).expect("serialize file meta");
        assert_eq!(json["file"], "grid_001.jpg");
        assert!(json.get("base64").is_none(), "file variant omits base64");

        let b64_meta = GridMeta {
            image: GridImage::Base64("QUJD".into()),
            ..file_meta
        };
        let json = serde_json::to_value(&b64_meta).expect("serialize base64 meta");
        assert_eq!(json["base64"], "QUJD");
        assert!(json.get("file").is_none(), "base64 variant omits file");
    }
}
