//! Unified runtime configuration.
//!
//! All parameters live in [`Config`]. Construction resolves named profiles,
//! quality/detail tiers, and CLI overrides in one place — no layered merging.

use crate::error::{Error, ErrorCode};
use crate::probe::SampleStrategy;
use crate::types::{Detail, Quality};

/// CLI overrides — all optional, highest priority.
#[derive(Debug, Clone, Default)]
pub struct Overrides {
    pub base_fps: Option<f32>,
    pub hash_threshold: Option<u32>,
    pub dedup_window: Option<usize>,
    pub score_edge: Option<u32>,
    pub w_scene: Option<f32>,
    pub w_clarity: Option<f32>,
    pub w_edge: Option<f32>,
    pub chunk_seconds: Option<f32>,
    pub min_gap: Option<f32>,
    pub jpeg_quality: Option<u8>,
    pub hero_border_px: Option<u32>,
}

/// Strategy-related options.
#[derive(Debug, Clone, Copy, Default)]
pub struct StrategyOpts {
    pub strategy: SampleStrategy,
    pub gap_threshold: f32,
    pub min_keyframes: usize,
    pub force: bool,
}

impl StrategyOpts {
    pub const DEFAULT_GAP_THRESHOLD: f32 = 5.0;
    pub const DEFAULT_MIN_KEYFRAMES: usize = 10;
    pub const MAX_DURATION_DEFAULT: f32 = 3600.0;
}

/// All runtime parameters needed by the pipeline.
#[derive(Debug, Clone)]
pub struct Config {
    pub max_edge: u32,
    pub min_frame_size: u32,
    pub canvas_scale: f32,
    pub base_fps: f32,
    pub hash_threshold: u32,
    pub dedup_window: usize,
    pub score_edge: u32,

    pub w_scene: f32,
    pub w_clarity: f32,
    pub w_edge: f32,

    pub chunk_seconds: f32,
    pub min_gap: f32,

    pub jpeg_quality: u8,
    pub hero_border_px: u32,

    pub quality: Quality,
    pub detail: Detail,

    pub strategy: SampleStrategy,
    pub gap_threshold: f32,
    pub min_keyframes: usize,
    pub force_long_video: bool,
}

/// Named profile — partial overrides.
#[derive(Debug, Clone, Default)]
struct ProfileDef {
    base_fps: Option<f32>,
    hash_threshold: Option<u32>,
    dedup_window: Option<usize>,
    score_edge: Option<u32>,
    w_scene: Option<f32>,
    w_clarity: Option<f32>,
    w_edge: Option<f32>,
    chunk_seconds: Option<f32>,
    min_gap: Option<f32>,
    jpeg_quality: Option<u8>,
    hero_border_px: Option<u32>,
}

fn builtin_profile(name: &str) -> Option<ProfileDef> {
    Some(match name {
        "balanced" => ProfileDef::default(),
        "fast" => ProfileDef {
            chunk_seconds: Some(90.0),
            min_gap: Some(1.0),
            jpeg_quality: Some(85),
            ..Default::default()
        },
        "quality" => ProfileDef {
            chunk_seconds: Some(45.0),
            min_gap: Some(0.3),
            jpeg_quality: Some(95),
            ..Default::default()
        },
        "action" => ProfileDef {
            w_scene: Some(0.55),
            w_clarity: Some(0.25),
            w_edge: Some(0.20),
            min_gap: Some(0.3),
            ..Default::default()
        },
        _ => return None,
    })
}

impl Config {
    /// Construct config from CLI arguments.
    ///
    /// Priority: CLI overrides > named profile > detail tier > defaults.
    pub fn new(
        max_edge: u32,
        quality: Quality,
        detail: Detail,
        profile: Option<&str>,
        cli: &Overrides,
        strategy_opts: StrategyOpts,
    ) -> crate::error::Result<Self> {
        let prof = match profile {
            Some(name) => match builtin_profile(name) {
                Some(p) => Some(p),
                None => {
                    return Err(Error::new(
                        ErrorCode::InvalidProfile,
                        format!("unknown profile '{name}'; available: balanced, fast, quality, action"),
                    ))
                }
            },
            None => None,
        };

        // Quality tier → layout
        let (min_frame_size, canvas_scale) = match quality {
            Quality::Low => (256, 0.5),
            Quality::Medium => (384, 0.75),
            Quality::High => (512, 1.0),
        };

        // Detail tier → sampling defaults
        let (detail_fps, detail_hash) = match detail {
            Detail::Coarse => (0.5, 4),
            Detail::Standard => (1.0, 8),
            Detail::Fine => (3.0, 12),
        };

        // Resolve each field: CLI > profile > detail/default
        macro_rules! resolve {
            ($field:ident, $default:expr) => {
                cli.$field
                    .or(prof.as_ref().and_then(|p| p.$field))
                    .unwrap_or($default)
            };
        }

        Ok(Self {
            max_edge,
            min_frame_size,
            canvas_scale,

            base_fps: resolve!(base_fps, detail_fps),
            hash_threshold: resolve!(hash_threshold, detail_hash),
            dedup_window: resolve!(dedup_window, 8),
            score_edge: resolve!(score_edge, 256).max(1),

            w_scene: resolve!(w_scene, 0.40),
            w_clarity: resolve!(w_clarity, 0.35),
            w_edge: resolve!(w_edge, 0.25),

            chunk_seconds: resolve!(chunk_seconds, 60.0),
            min_gap: resolve!(min_gap, 0.5),

            jpeg_quality: resolve!(jpeg_quality, 92),
            hero_border_px: resolve!(hero_border_px, 4),

            quality,
            detail,

            strategy: strategy_opts.strategy,
            gap_threshold: strategy_opts.gap_threshold,
            min_keyframes: strategy_opts.min_keyframes,
            force_long_video: strategy_opts.force,
        })
    }

    /// Canvas geometry for layout, with the quality scale applied to the edge.
    pub fn canvas_spec(&self) -> crate::types::CanvasSpec {
        crate::types::CanvasSpec {
            edge: (self.max_edge as f32 * self.canvas_scale).round() as u32,
            min_cell_edge: self.min_frame_size,
        }
    }

    pub fn settings(&self) -> crate::types::Settings {
        crate::types::Settings {
            max_edge: self.max_edge,
            quality: self.quality,
            detail: self.detail,
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
