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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Quality {
    Low,
    Medium,
    High,
}

impl Quality {
    /// Lowercase tier name as written into `metadata.json`.
    pub fn as_str(self) -> &'static str {
        match self {
            Quality::Low => "low",
            Quality::Medium => "medium",
            Quality::High => "high",
        }
    }
}

/// Detail tier: controls sampling fps, dedup strictness, and clarity filtering.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Detail {
    Coarse,
    Standard,
    Fine,
}

impl Detail {
    /// Lowercase tier name as written into `metadata.json`.
    pub fn as_str(self) -> &'static str {
        match self {
            Detail::Coarse => "coarse",
            Detail::Standard => "standard",
            Detail::Fine => "fine",
        }
    }
}

/// Fully resolved processing parameters, merged from model preset + quality +
/// detail tiers. `aspect` is supplied separately at layout time.
#[derive(Debug, Clone, Copy)]
pub struct Preset {
    /// Canvas area budget (pixel edge length, not area).
    pub max_pixels: u32,
    /// Minimum readable edge length for a single frame cell, in pixels.
    pub min_frame_size: u32,
    /// Canvas scale multiplier from the quality tier (0.5 / 0.75 / 1.0).
    pub canvas_scale: f32,
    /// Base sampling fps before any dynamic governor adjustments.
    pub base_fps: f32,
    /// pHash Hamming distance threshold; frames closer than this are dropped.
    pub hash_threshold: u32,
}

/// Metadata for one rendered grid image.
#[derive(Debug, Clone, Serialize)]
pub struct GridMeta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub file: Option<String>,
    pub time_range: String,
    pub total_frames: usize,
    pub key_frames: Vec<KeyFrameMeta>,
    pub layout: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub base64: Option<String>,
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
#[derive(Debug, Clone, Serialize)]
pub struct Settings {
    pub max_pixels: u32,
    pub quality: String,
    pub detail: String,
}
