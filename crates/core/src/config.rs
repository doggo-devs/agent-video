//! Unified runtime configuration.
//!
//! Parse, don't validate: [`Config`] is a bundle of domain groups whose
//! invariants live in their constructors — holding a `Config` means every
//! parameter has already been checked, and there is no post-hoc `validate()`
//! step to forget or bypass. Fields are private; the pipeline reads through
//! getters. Construction resolves named profiles, quality/detail tiers, and
//! CLI overrides in one place — no layered merging.

use std::num::NonZeroUsize;

use crate::error::{Error, ErrorCode};
use crate::probe::SampleStrategy;
use crate::score::Weights;
use crate::types::{CanvasSpec, Detail, Quality};

/// Default dedup sliding window, in kept frames.
const DEFAULT_DEDUP_WINDOW: NonZeroUsize = NonZeroUsize::new(8).unwrap();

/// Reject a resolved value that would corrupt scoring or sampling arithmetic,
/// naming the offending field and value.
fn invalid<T>(field: &str, detail: impl std::fmt::Display) -> crate::error::Result<T> {
    Err(Error::new(
        ErrorCode::InvalidConfig,
        format!("invalid {field}: {detail}"),
    ))
}

/// A partial parameter set: CLI overrides (highest priority) and named
/// profiles both resolve through this shape.
///
/// Derives [`clap::Args`] directly, so the CLI flags and the override fields
/// are the same declaration: adding an override is one field here, with no
/// hand-copied mirror to drift in `main.rs`.
#[derive(Debug, Clone, Default, clap::Args)]
pub struct Overrides {
    /// Override base sampling fps.
    #[arg(long)]
    pub base_fps: Option<f32>,
    /// Override pHash Hamming distance threshold for dedup.
    #[arg(long)]
    pub hash_threshold: Option<u32>,
    /// Override dedup sliding window size in frames (must be >= 1).
    #[arg(long)]
    pub dedup_window: Option<NonZeroUsize>,
    /// Override edge length of grayscale image for scoring.
    #[arg(long)]
    pub score_edge: Option<u32>,

    /// Override scene-change weight in scoring.
    #[arg(long)]
    pub w_scene: Option<f32>,
    /// Override clarity weight in scoring.
    #[arg(long)]
    pub w_clarity: Option<f32>,
    /// Override edge-density weight in scoring.
    #[arg(long)]
    pub w_edge: Option<f32>,

    /// Override chunk size in seconds.
    #[arg(long)]
    pub chunk_seconds: Option<f32>,
    /// Override minimum temporal gap (seconds) between selected frames.
    #[arg(long)]
    pub min_gap: Option<f32>,

    /// Override JPEG output quality (1-100).
    #[arg(long)]
    pub jpeg_quality: Option<u8>,
    /// Override hero frame border thickness in pixels.
    #[arg(long)]
    pub hero_border_px: Option<u32>,
}

/// Sampling-strategy options, `process` only: `extract` always samples its
/// explicit range at fixed fps, so it uses the [`Default`] values.
///
/// The clap defaults and [`Default`] share the `DEFAULT_*` consts, so the CLI
/// and a programmatically-built config cannot disagree.
#[derive(Debug, Clone, Copy, clap::Args)]
pub struct StrategyOpts {
    /// Frame sampling strategy.
    #[arg(long, value_enum, default_value_t = SampleStrategy::FixedFps)]
    pub strategy: SampleStrategy,
    /// Gap threshold (seconds) for adaptive sparse region fill.
    #[arg(long, default_value_t = Self::DEFAULT_GAP_THRESHOLD)]
    pub gap_threshold: f32,
    /// Minimum keyframes required for keyframes/adaptive strategies.
    #[arg(long, default_value_t = Self::DEFAULT_MIN_KEYFRAMES)]
    pub min_keyframes: usize,
    /// Max duration (seconds) allowed for fixed-fps sampling; longer videos
    /// need --force or a keyframe strategy.
    #[arg(long, default_value_t = Self::DEFAULT_MAX_DURATION)]
    pub max_duration: f32,
    /// Allow fixed-fps on videos exceeding the max-duration limit.
    #[arg(long)]
    pub force: bool,
}

impl StrategyOpts {
    pub const DEFAULT_GAP_THRESHOLD: f32 = 5.0;
    pub const DEFAULT_MIN_KEYFRAMES: usize = 10;
    pub const DEFAULT_MAX_DURATION: f32 = 3600.0;
}

impl Default for StrategyOpts {
    fn default() -> Self {
        Self {
            strategy: SampleStrategy::default(),
            gap_threshold: Self::DEFAULT_GAP_THRESHOLD,
            min_keyframes: Self::DEFAULT_MIN_KEYFRAMES,
            max_duration: Self::DEFAULT_MAX_DURATION,
            force: false,
        }
    }
}

/// Sampling density and chunking cadence. All values finite by construction;
/// fps and chunk length strictly positive, `min_gap` non-negative.
#[derive(Debug, Clone, Copy)]
struct Sampling {
    base_fps: f32,
    /// Sampling fps for the `extract` subcommand. Defaults to three times the
    /// detail tier's fps (extract exists for high-density passes over short
    /// ranges), but an explicit `--base-fps` or profile value wins as-is, so
    /// the flag means the same density in both subcommands.
    extract_fps: f32,
    chunk_seconds: f32,
    min_gap: f32,
    /// Shorter edge of the grayscale image scoring convolutions run on; >= 1.
    score_edge: u32,
}

impl Sampling {
    fn new(
        base_fps: f32,
        extract_fps: f32,
        chunk_seconds: f32,
        min_gap: f32,
        score_edge: u32,
    ) -> crate::error::Result<Self> {
        if !base_fps.is_finite() || base_fps <= 0.0 {
            return invalid("base_fps", format!("{base_fps} (must be finite and > 0)"));
        }
        if !extract_fps.is_finite() || extract_fps <= 0.0 {
            return invalid(
                "extract_fps",
                format!("{extract_fps} (must be finite and > 0)"),
            );
        }
        if !chunk_seconds.is_finite() || chunk_seconds <= 0.0 {
            return invalid(
                "chunk_seconds",
                format!("{chunk_seconds} (must be finite and > 0)"),
            );
        }
        if !min_gap.is_finite() || min_gap < 0.0 {
            return invalid("min_gap", format!("{min_gap} (must be finite and >= 0)"));
        }
        if score_edge == 0 {
            return invalid("score_edge", "0 (must be >= 1)");
        }
        Ok(Self {
            base_fps,
            extract_fps,
            chunk_seconds,
            min_gap,
            score_edge,
        })
    }
}

/// Perceptual-dedup parameters. The pHash is 64 bits, so a Hamming threshold
/// past 64 would silently mean "drop everything after the first frame".
#[derive(Debug, Clone, Copy)]
struct DedupCfg {
    hash_threshold: u32,
    window: NonZeroUsize,
}

impl DedupCfg {
    fn new(hash_threshold: u32, window: NonZeroUsize) -> crate::error::Result<Self> {
        if hash_threshold > 64 {
            return invalid(
                "hash_threshold",
                format!("{hash_threshold} (must be <= 64)"),
            );
        }
        Ok(Self {
            hash_threshold,
            window,
        })
    }
}

/// JPEG/compose parameters. Quality 0 is rejected by the encoder at the end
/// of the run; catching it here fails before any frame work.
#[derive(Debug, Clone, Copy)]
struct RenderCfg {
    jpeg_quality: u8,
    hero_border_px: u32,
}

impl RenderCfg {
    fn new(jpeg_quality: u8, hero_border_px: u32) -> crate::error::Result<Self> {
        if !(1..=100).contains(&jpeg_quality) {
            return invalid(
                "jpeg_quality",
                format!("{jpeg_quality} (must be in 1..=100)"),
            );
        }
        Ok(Self {
            jpeg_quality,
            hero_border_px,
        })
    }
}

/// Resolved strategy parameters; thresholds finite and positive by
/// construction.
#[derive(Debug, Clone, Copy)]
struct StrategyCfg {
    strategy: SampleStrategy,
    gap_threshold: f32,
    min_keyframes: usize,
    max_duration: f32,
    force_long_video: bool,
}

impl StrategyCfg {
    fn new(opts: StrategyOpts) -> crate::error::Result<Self> {
        if !opts.gap_threshold.is_finite() || opts.gap_threshold <= 0.0 {
            return invalid(
                "gap_threshold",
                format!("{} (must be finite and > 0)", opts.gap_threshold),
            );
        }
        if !opts.max_duration.is_finite() || opts.max_duration <= 0.0 {
            return invalid(
                "max_duration",
                format!("{} (must be finite and > 0)", opts.max_duration),
            );
        }
        Ok(Self {
            strategy: opts.strategy,
            gap_threshold: opts.gap_threshold,
            min_keyframes: opts.min_keyframes,
            max_duration: opts.max_duration,
            force_long_video: opts.force,
        })
    }
}

/// All runtime parameters needed by the pipeline, grouped by domain.
///
/// Every group is built by a validating constructor, so a held `Config` is
/// valid by construction; fields are private so nothing — production code or
/// tests — can route around those constructors. Tests wanting non-default
/// values go through [`Overrides`], the same door the CLI uses.
#[derive(Debug, Clone)]
pub struct Config {
    canvas: CanvasSpec,
    sampling: Sampling,
    dedup: DedupCfg,
    weights: Weights,
    render: RenderCfg,
    strategy: StrategyCfg,
    quality: Quality,
    detail: Detail,
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

        Ok(Self {
            canvas: CanvasSpec::new(max_edge, min_frame_size)?,
            sampling: Sampling::new(
                resolve!(base_fps, detail_fps),
                resolve!(base_fps, detail_fps * 3.0),
                resolve!(chunk_seconds, 60.0),
                resolve!(min_gap, 0.5),
                resolve!(score_edge, 256),
            )?,
            dedup: DedupCfg::new(
                resolve!(hash_threshold, 8),
                resolve!(dedup_window, DEFAULT_DEDUP_WINDOW),
            )?,
            // Clarity defaults to 0: Laplacian variance and Sobel energy are
            // both high-frequency measures, so a nonzero clarity weight
            // double-counts texture. It stays available as an override knob.
            weights: Weights::new(
                resolve!(w_scene, 0.60),
                resolve!(w_clarity, 0.0),
                resolve!(w_edge, 0.40),
            )?,
            render: RenderCfg::new(resolve!(jpeg_quality, 92), resolve!(hero_border_px, 4))?,
            strategy: StrategyCfg::new(strategy_opts)?,
            quality,
            detail,
        })
    }

    pub fn max_edge(&self) -> u32 {
        self.canvas.edge()
    }

    /// The quality tier's minimum readable cell edge; extraction pre-scales
    /// frames to this size.
    pub fn min_frame_size(&self) -> u32 {
        self.canvas.min_cell_edge()
    }

    /// Canvas geometry for layout: the full `max_edge` budget with the
    /// quality tier's cell floor.
    pub fn canvas_spec(&self) -> CanvasSpec {
        self.canvas
    }

    pub fn base_fps(&self) -> f32 {
        self.sampling.base_fps
    }

    pub fn extract_fps(&self) -> f32 {
        self.sampling.extract_fps
    }

    pub fn chunk_seconds(&self) -> f32 {
        self.sampling.chunk_seconds
    }

    pub fn min_gap(&self) -> f32 {
        self.sampling.min_gap
    }

    pub fn score_edge(&self) -> u32 {
        self.sampling.score_edge
    }

    pub fn hash_threshold(&self) -> u32 {
        self.dedup.hash_threshold
    }

    pub fn dedup_window(&self) -> NonZeroUsize {
        self.dedup.window
    }

    pub fn weights(&self) -> Weights {
        self.weights
    }

    pub fn jpeg_quality(&self) -> u8 {
        self.render.jpeg_quality
    }

    pub fn hero_border_px(&self) -> u32 {
        self.render.hero_border_px
    }

    pub fn strategy(&self) -> SampleStrategy {
        self.strategy.strategy
    }

    pub fn gap_threshold(&self) -> f32 {
        self.strategy.gap_threshold
    }

    pub fn min_keyframes(&self) -> usize {
        self.strategy.min_keyframes
    }

    pub fn max_duration(&self) -> f32 {
        self.strategy.max_duration
    }

    pub fn force_long_video(&self) -> bool {
        self.strategy.force_long_video
    }

    pub fn settings(&self) -> crate::types::Settings {
        crate::types::Settings {
            max_edge: self.max_edge(),
            quality: self.quality,
            detail: self.detail,
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
        assert_eq!(cfg.base_fps(), 1.0, "standard detail samples at 1 fps");
        assert_eq!(cfg.extract_fps(), 3.0, "extract densifies the detail tier");
    }

    #[test]
    fn explicit_base_fps_sets_extract_fps_verbatim() {
        let ov = Overrides {
            base_fps: Some(2.0),
            ..Default::default()
        };
        let cfg = config_with(ov).expect("config builds");
        assert_eq!(
            cfg.extract_fps(),
            2.0,
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

    #[test]
    fn rejects_invalid_strategy_thresholds() {
        for opts in [
            StrategyOpts {
                gap_threshold: 0.0,
                ..Default::default()
            },
            StrategyOpts {
                max_duration: f32::NAN,
                ..Default::default()
            },
        ] {
            let cfg = Config::new(
                2000,
                Quality::High,
                Detail::Standard,
                None,
                &Overrides::default(),
                opts,
            );
            assert_eq!(
                cfg.expect_err("invalid strategy opts rejected").code,
                ErrorCode::InvalidConfig
            );
        }
    }
}
