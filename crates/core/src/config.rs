//! Unified runtime configuration.
//!
//! All parameters live in [`Config`]. Construction resolves named profiles,
//! quality/detail tiers, and CLI overrides in one place — no layered merging.

use std::num::NonZeroUsize;

use crate::error::{Error, ErrorCode};
use crate::probe::SampleStrategy;
use crate::types::{Detail, Quality};

/// JPEG caps image dimensions at 65535 px; a larger canvas would run the whole
/// pipeline only to fail at encode time (after a multi-GB allocation).
const MAX_CANVAS_EDGE: u32 = 65_535;

/// Default dedup sliding window, in kept frames.
const DEFAULT_DEDUP_WINDOW: NonZeroUsize = NonZeroUsize::new(8).unwrap();

/// A partial parameter set: CLI overrides (highest priority) and named
/// profiles both resolve through this shape.
#[derive(Debug, Clone, Default)]
pub struct Overrides {
    pub base_fps: Option<f32>,
    pub hash_threshold: Option<u32>,
    pub dedup_window: Option<NonZeroUsize>,
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
#[derive(Debug, Clone, Copy)]
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

impl Default for StrategyOpts {
    /// Mirrors the CLI defaults so a programmatically-built config validates;
    /// the derived all-zero default would make `gap_threshold` invalid.
    fn default() -> Self {
        Self {
            strategy: SampleStrategy::default(),
            gap_threshold: Self::DEFAULT_GAP_THRESHOLD,
            min_keyframes: Self::DEFAULT_MIN_KEYFRAMES,
            force: false,
        }
    }
}

/// All runtime parameters needed by the pipeline.
#[derive(Debug, Clone)]
pub struct Config {
    pub max_edge: u32,
    pub min_frame_size: u32,
    pub base_fps: f32,
    /// Sampling fps for the `extract` subcommand. Defaults to three times the
    /// detail tier's fps (extract exists for high-density passes over short
    /// ranges), but an explicit `--base-fps` or profile value wins as-is, so
    /// the flag means the same density in both subcommands.
    pub extract_fps: f32,
    pub hash_threshold: u32,
    pub dedup_window: NonZeroUsize,
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

/// Named profiles are partial parameter sets — the same shape as CLI
/// overrides, one priority level below them.
fn builtin_profile(name: &str) -> Option<Overrides> {
    Some(match name {
        "balanced" => Overrides::default(),
        "fast" => Overrides {
            chunk_seconds: Some(90.0),
            min_gap: Some(1.0),
            jpeg_quality: Some(85),
            ..Default::default()
        },
        "quality" => Overrides {
            chunk_seconds: Some(45.0),
            min_gap: Some(0.3),
            jpeg_quality: Some(95),
            ..Default::default()
        },
        "action" => Overrides {
            w_scene: Some(0.75),
            w_edge: Some(0.25),
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
                        format!(
                            "unknown profile '{name}'; available: balanced, fast, quality, action"
                        ),
                    ));
                }
            },
            None => None,
        };

        // Quality tier → cell legibility floor. The canvas edge stays at
        // max_edge regardless: quality trades per-cell size for grid capacity
        // (capacity ∝ (edge / min_frame_size)²), so lower quality packs more
        // frames onto the same canvas instead of shrinking the output image.
        // Extraction pre-scales to this same edge, so cells at the floor are
        // rendered 1:1 with no upscaling.
        let min_frame_size = match quality {
            Quality::Low => 256,
            Quality::Medium => 384,
            Quality::High => 512,
        };

        // Detail tier → sampling density only. The dedup threshold is a
        // candidate-pruning knob, not a density knob: tying it to detail made
        // fine mode (3 fps, meant to catch subtle changes) dedup the most
        // aggressively, silently merging small text/UI diffs that no later
        // stage can recover. Density is handled by fps + NMS + capacity.
        let detail_fps = match detail {
            Detail::Coarse => 0.5,
            Detail::Standard => 1.0,
            Detail::Fine => 3.0,
        };

        // Resolve each field: CLI > profile > detail/default
        macro_rules! resolve {
            ($field:ident, $default:expr) => {
                cli.$field
                    .or(prof.as_ref().and_then(|p| p.$field))
                    .unwrap_or($default)
            };
        }

        let cfg = Self {
            max_edge,
            min_frame_size,

            base_fps: resolve!(base_fps, detail_fps),
            extract_fps: resolve!(base_fps, detail_fps * 3.0),
            hash_threshold: resolve!(hash_threshold, 8),
            dedup_window: resolve!(dedup_window, DEFAULT_DEDUP_WINDOW),
            score_edge: resolve!(score_edge, 256),

            // Clarity defaults to 0: Laplacian variance and Sobel energy are
            // both high-frequency measures, so a nonzero clarity weight
            // double-counts texture. It stays available as an override knob.
            w_scene: resolve!(w_scene, 0.60),
            w_clarity: resolve!(w_clarity, 0.0),
            w_edge: resolve!(w_edge, 0.40),

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
        };
        cfg.validate()?;
        Ok(cfg)
    }

    /// Reject resolved values that would corrupt scoring or sampling arithmetic
    /// before they reach the pipeline. Names the offending field and value.
    fn validate(&self) -> crate::error::Result<()> {
        let invalid = |field: &str, detail: String| -> crate::error::Result<()> {
            Err(Error::new(
                ErrorCode::InvalidConfig,
                format!("invalid {field}: {detail}"),
            ))
        };

        // The canvas must hold at least one cell at the quality tier's
        // legibility floor; anything smaller (down to --max-edge 0) would
        // still "succeed" as a degenerate, unreadable image.
        if self.max_edge < self.min_frame_size {
            return invalid(
                "max_edge",
                format!(
                    "{} (canvas must fit one cell at the quality floor, >= {}px)",
                    self.max_edge, self.min_frame_size
                ),
            );
        }
        // The canvas allocation is sized directly from this value; without a
        // cap, an oversized flag burns the full pipeline before JPEG encoding
        // rejects the dimensions (or the allocation aborts first).
        if self.max_edge > MAX_CANVAS_EDGE {
            return invalid(
                "max_edge",
                format!(
                    "{} (exceeds JPEG's {MAX_CANVAS_EDGE}px dimension limit)",
                    self.max_edge
                ),
            );
        }
        for (name, w) in [
            ("w_scene", self.w_scene),
            ("w_clarity", self.w_clarity),
            ("w_edge", self.w_edge),
        ] {
            if !w.is_finite() || w < 0.0 {
                return invalid(name, format!("{w} (weights must be finite and >= 0)"));
            }
        }
        if self.w_scene + self.w_clarity + self.w_edge <= 0.0 {
            return invalid("weights", "all zero (at least one must be > 0)".to_string());
        }
        if !self.base_fps.is_finite() || self.base_fps <= 0.0 {
            return invalid(
                "base_fps",
                format!("{} (must be finite and > 0)", self.base_fps),
            );
        }
        if !self.chunk_seconds.is_finite() || self.chunk_seconds <= 0.0 {
            return invalid(
                "chunk_seconds",
                format!("{} (must be finite and > 0)", self.chunk_seconds),
            );
        }
        if !self.min_gap.is_finite() || self.min_gap < 0.0 {
            return invalid(
                "min_gap",
                format!("{} (must be finite and >= 0)", self.min_gap),
            );
        }
        if !self.gap_threshold.is_finite() || self.gap_threshold <= 0.0 {
            return invalid(
                "gap_threshold",
                format!("{} (must be finite and > 0)", self.gap_threshold),
            );
        }
        if !(1..=100).contains(&self.jpeg_quality) {
            return invalid(
                "jpeg_quality",
                format!("{} (must be in 1..=100)", self.jpeg_quality),
            );
        }
        if self.hash_threshold > 64 {
            return invalid(
                "hash_threshold",
                format!("{} (must be <= 64)", self.hash_threshold),
            );
        }
        if self.score_edge == 0 {
            return invalid("score_edge", "0 (must be >= 1)".to_string());
        }
        Ok(())
    }

    /// Canvas geometry for layout: the full `max_edge` budget with the
    /// quality tier's cell floor.
    pub fn canvas_spec(&self) -> crate::types::CanvasSpec {
        crate::types::CanvasSpec {
            edge: self.max_edge,
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

#[cfg(test)]
mod tests {
    use super::*;

    fn config_with(overrides: Overrides) -> crate::error::Result<Config> {
        Config::new(
            2000,
            Quality::High,
            Detail::Standard,
            None,
            &overrides,
            StrategyOpts::default(),
        )
    }

    #[test]
    fn accepts_default_overrides() {
        assert!(config_with(Overrides::default()).is_ok());
    }

    #[test]
    fn extract_fps_defaults_to_triple_detail_fps() {
        let cfg = config_with(Overrides::default()).expect("default config builds");
        assert_eq!(cfg.base_fps, 1.0, "standard detail samples at 1 fps");
        assert_eq!(cfg.extract_fps, 3.0, "extract densifies the detail tier");
    }

    #[test]
    fn explicit_base_fps_sets_extract_fps_verbatim() {
        let ov = Overrides {
            base_fps: Some(2.0),
            ..Default::default()
        };
        let cfg = config_with(ov).expect("config builds");
        assert_eq!(
            cfg.extract_fps, 2.0,
            "an explicit --base-fps means the same density in both subcommands"
        );
    }

    #[test]
    fn rejects_canvas_below_cell_floor() {
        // quality high → 512px floor; a 0 or sub-floor canvas cannot hold a
        // single legible cell.
        for max_edge in [0, 511] {
            let cfg = Config::new(
                max_edge,
                Quality::High,
                Detail::Standard,
                None,
                &Overrides::default(),
                StrategyOpts::default(),
            );
            assert_eq!(
                cfg.expect_err("sub-floor canvas rejected").code,
                ErrorCode::InvalidConfig
            );
        }
    }

    #[test]
    fn rejects_negative_weight() {
        let ov = Overrides {
            w_scene: Some(-0.1),
            ..Default::default()
        };
        assert_eq!(
            config_with(ov).expect_err("negative weight rejected").code,
            ErrorCode::InvalidConfig
        );
    }

    #[test]
    fn rejects_all_zero_weights() {
        let ov = Overrides {
            w_scene: Some(0.0),
            w_clarity: Some(0.0),
            w_edge: Some(0.0),
            ..Default::default()
        };
        assert_eq!(
            config_with(ov).expect_err("all-zero weights rejected").code,
            ErrorCode::InvalidConfig
        );
    }

    #[test]
    fn rejects_zero_fps() {
        let ov = Overrides {
            base_fps: Some(0.0),
            ..Default::default()
        };
        assert_eq!(
            config_with(ov).expect_err("zero fps rejected").code,
            ErrorCode::InvalidConfig
        );
    }

    #[test]
    fn rejects_out_of_range_jpeg_quality() {
        let ov = Overrides {
            jpeg_quality: Some(0),
            ..Default::default()
        };
        assert_eq!(
            config_with(ov).expect_err("quality 0 rejected").code,
            ErrorCode::InvalidConfig
        );
    }

    #[test]
    fn rejects_oversized_hash_threshold() {
        let ov = Overrides {
            hash_threshold: Some(65),
            ..Default::default()
        };
        assert_eq!(
            config_with(ov).expect_err("threshold > 64 rejected").code,
            ErrorCode::InvalidConfig
        );
    }

    #[test]
    fn rejects_zero_score_edge() {
        let ov = Overrides {
            score_edge: Some(0),
            ..Default::default()
        };
        assert_eq!(
            config_with(ov).expect_err("score_edge 0 rejected").code,
            ErrorCode::InvalidConfig
        );
    }

    #[test]
    fn rejects_canvas_above_jpeg_dimension_limit() {
        let cfg = Config::new(
            65_536,
            Quality::High,
            Detail::Standard,
            None,
            &Overrides::default(),
            StrategyOpts::default(),
        );
        assert_eq!(
            cfg.expect_err("over-limit canvas rejected").code,
            ErrorCode::InvalidConfig
        );
    }
}
