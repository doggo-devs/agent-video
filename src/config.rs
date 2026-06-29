//! Unified runtime configuration.
//!
//! [`Config`] merges preset (sampling/layout), profile (scoring/output), and
//! metadata labels into a single source of truth. Pipeline functions take
//! `&Config` instead of multiple independent parameters.

use crate::profile::Profile;
use crate::types::{Detail, Preset, Quality};

/// All runtime parameters needed by the pipeline.
#[derive(Debug, Clone)]
pub struct Config {
    /// Canvas area budget in pixels.
    pub max_pixels: u32,
    /// Minimum readable frame edge length in pixels.
    pub min_frame_size: u32,
    /// Canvas scale multiplier from quality tier.
    pub canvas_scale: f32,
    /// Base sampling fps.
    pub base_fps: f32,
    /// pHash Hamming distance threshold for dedup.
    pub hash_threshold: u32,
    /// Dedup sliding window size in frames.
    pub dedup_window: usize,
    /// Edge length of grayscale image for scoring convolutions.
    pub score_edge: u32,

    /// Scene-change weight.
    pub w_scene: f32,
    /// Clarity weight.
    pub w_clarity: f32,
    /// Edge-density weight.
    pub w_edge: f32,

    /// Seconds per grid chunk.
    pub chunk_seconds: f32,
    /// Floor on temporal NMS gap.
    pub min_gap: f32,

    /// JPEG quality 0-100.
    pub jpeg_quality: u8,
    /// Hero frame border thickness in pixels.
    pub hero_border_px: u32,

    /// Quality tier name for metadata.json.
    pub quality_label: &'static str,
    /// Detail tier name for metadata.json.
    pub detail_label: &'static str,
}

impl Config {
    /// Merge a resolved preset, profile, and tier labels into a unified config.
    ///
    /// Profile values for `base_fps` and `hash_threshold` override preset defaults
    /// when explicitly set (non-default). This allows `--detail` to set baseline
    /// sampling while `--base-fps` / `--hash-threshold` CLI args fine-tune.
    pub fn new(preset: Preset, profile: Profile, quality: Quality, detail: Detail) -> Self {
        let default_profile = Profile::default();

        // Profile overrides preset when the profile value differs from its default.
        let base_fps = if (profile.base_fps - default_profile.base_fps).abs() > f32::EPSILON {
            profile.base_fps
        } else {
            preset.base_fps
        };
        let hash_threshold = if profile.hash_threshold != default_profile.hash_threshold {
            profile.hash_threshold
        } else {
            preset.hash_threshold
        };

        Self {
            max_pixels: preset.max_pixels,
            min_frame_size: preset.min_frame_size,
            canvas_scale: preset.canvas_scale,
            base_fps,
            hash_threshold,
            dedup_window: profile.dedup_window,
            score_edge: profile.score_edge,

            w_scene: profile.w_scene,
            w_clarity: profile.w_clarity,
            w_edge: profile.w_edge,

            chunk_seconds: profile.chunk_seconds,
            min_gap: profile.min_gap,

            jpeg_quality: profile.jpeg_quality,
            hero_border_px: profile.hero_border_px,

            quality_label: quality.as_str(),
            detail_label: detail.as_str(),
        }
    }

    pub fn as_preset(&self) -> Preset {
        Preset {
            max_pixels: self.max_pixels,
            min_frame_size: self.min_frame_size,
            canvas_scale: self.canvas_scale,
            base_fps: self.base_fps,
            hash_threshold: self.hash_threshold,
        }
    }

    pub fn weights(&self) -> crate::score::Weights {
        crate::score::Weights {
            scene: self.w_scene,
            clarity: self.w_clarity,
            edge: self.w_edge,
        }
    }
}
