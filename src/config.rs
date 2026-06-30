//! Unified runtime configuration.
//!
//! [`Config`] merges preset (sampling/layout), profile (scoring/output), and
//! metadata labels into a single source of truth. Pipeline functions take
//! `&Config` instead of multiple independent parameters.

use crate::probe::SampleStrategy;
use crate::profile::Profile;
use crate::types::{Detail, Preset, Quality};

/// Strategy-related options passed from CLI.
#[derive(Debug, Clone, Copy, Default)]
pub struct StrategyOpts {
    pub strategy: SampleStrategy,
    pub gap_threshold: f32,
    pub min_keyframes: usize,
    pub force: bool,
}

impl StrategyOpts {
    /// Default gap threshold for adaptive sparse region fill (seconds).
    pub const DEFAULT_GAP_THRESHOLD: f32 = 5.0;
    /// Default minimum keyframes required.
    pub const DEFAULT_MIN_KEYFRAMES: usize = 10;
    /// Default maximum duration (seconds) before requiring --force.
    pub const MAX_DURATION_DEFAULT: f32 = 3600.0;
}

/// All runtime parameters needed by the pipeline.
#[derive(Debug, Clone)]
pub struct Config {
    /// Canvas edge length in pixels.
    pub max_edge: u32,
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

    /// Frame sampling strategy.
    pub strategy: SampleStrategy,
    /// Gap threshold for adaptive sparse region fill (seconds).
    pub gap_threshold: f32,
    /// Minimum keyframes required for keyframes/adaptive strategies.
    pub min_keyframes: usize,
    /// Allow fixed-fps on videos exceeding duration limit.
    pub force_long_video: bool,
}

impl Config {
    /// Merge a resolved preset, profile, and tier labels into a unified config.
    ///
    /// Profile values for `base_fps` and `hash_threshold` override preset defaults
    /// when explicitly set (non-default). This allows `--detail` to set baseline
    /// sampling while `--base-fps` / `--hash-threshold` CLI args fine-tune.
    pub fn new(
        preset: Preset,
        profile: Profile,
        quality: Quality,
        detail: Detail,
        strategy_opts: StrategyOpts,
    ) -> Self {
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
            max_edge: preset.max_edge,
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

            strategy: strategy_opts.strategy,
            gap_threshold: strategy_opts.gap_threshold,
            min_keyframes: strategy_opts.min_keyframes,
            force_long_video: strategy_opts.force,
        }
    }

    pub fn as_preset(&self) -> Preset {
        Preset {
            max_edge: self.max_edge,
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
