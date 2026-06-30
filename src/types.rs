//! Public data types shared across the pipeline.
//!
//! The central design constraint lives here: [`ScoredFrame`] never holds a
//! full-resolution decoded frame. Right after a frame is decoded, `extract`
//! derives a small grayscale image (for scoring) and a small RGB image (for
//! composition), then drops the original buffer. This keeps peak memory in the
//! hundreds-of-MB range instead of GBs for long videos.

use image::{GrayImage, RgbImage};
use serde::Serialize;

/// Raw per-dimension scores, in their natural (incomparable) units.
///
/// Only meaningful after per-chunk rank normalization — see [`crate::score`].
/// `clarity` is a Laplacian variance, `edge_density` a Sobel energy mean,
/// `scene_diff` a mean absolute pixel difference (0-255).
#[derive(Debug, Clone, Copy, Default)]
pub struct RawScores {
    pub clarity: f32,
    pub edge_density: f32,
    /// Mean absolute grayscale difference from the previous kept frame.
    pub scene_diff: f32,
}

/// A single sampled frame carrying everything the pipeline needs downstream.
/// Holds only downscaled derivatives, never the source-resolution buffer.
#[derive(Debug, Clone)]
pub struct ScoredFrame {
    /// Presentation timestamp in seconds.
    pub ts: f32,
    /// Small grayscale image used for scoring convolutions.
    pub gray: GrayImage,
    /// Small RGB image pasted onto the grid during composition.
    pub rgb: RgbImage,
    pub raw: RawScores,
    /// Final weighted score in `0..=100`, filled after rank normalization.
    pub score: f32,
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

/// Resolution tier: controls minimum readable frame size and canvas scale.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Quality {
    Low,
    Medium,
    High,
}

/// Detail tier: controls sampling fps, dedup strictness, and clarity filtering.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Detail {
    Coarse,
    Standard,
    Fine,
}

/// Resolved canvas geometry for layout; the quality scale is already applied.
#[derive(Debug, Clone, Copy)]
pub struct CanvasSpec {
    /// Longer canvas edge in pixels.
    pub edge: u32,
    /// Target shorter edge of a single cell, in pixels. A soft lower bound:
    /// capacity planning treats the canvas as square while the packed canvas is
    /// only roughly square, so cells can land slightly below it.
    pub min_cell_edge: u32,
}

/// Metadata for one rendered grid image.
#[derive(Debug, Clone, Serialize)]
pub struct GridMeta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub file: Option<String>,
    /// Time range as "MM:SS-MM:SS" for JSON output.
    #[serde(serialize_with = "serialize_time_range")]
    pub time_range: (f32, f32),
    pub total_frames: usize,
    pub key_frames: Vec<KeyFrameMeta>,
    pub layout: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub base64: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub subtitles: Option<String>,
}

fn serialize_time_range<S: serde::Serializer>(range: &(f32, f32), s: S) -> Result<S::Ok, S::Error> {
    s.serialize_str(&format!("{}-{}", format_ts(range.0), format_ts(range.1)))
}

/// Format a timestamp as `MM:SS`, clamping negatives to zero and rounding to
/// the nearest second.
pub fn format_ts(ts: f32) -> String {
    let total = ts.max(0.0).round() as u32;
    format!("{:02}:{:02}", total / 60, total % 60)
}

/// Metadata for a single key (hero) frame inside a grid.
#[derive(Debug, Clone, Serialize)]
pub struct KeyFrameMeta {
    pub timestamp: String,
    pub score: u32,
    pub grid_position: String,
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
}
