// SPDX-License-Identifier: Apache-2.0
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

/// Default pHash Hamming distance threshold for dedup: distances at or below
/// it mean "same shot". 8 of 64 bits catches compression jitter and slow
/// drift without merging genuinely distinct frames.
const DEFAULT_HASH_THRESHOLD: u32 = 8;

/// Default quality tier when neither `--quality` nor a profile sets one.
const DEFAULT_QUALITY: Quality = Quality::Low;

/// Default detail tier when neither `--detail` nor a profile sets one.
const DEFAULT_DETAIL: Detail = Detail::Standard;

/// Default chunk length in seconds when neither `--chunk-seconds` nor a
/// profile provides one.
pub const DEFAULT_CHUNK_SECONDS: f32 = 60.0;

/// Default minimum temporal gap (seconds) between selected frames — the
/// floor under the span-derived gap `process_chunk` computes per chunk.
const DEFAULT_MIN_GAP: f32 = 0.5;

/// Default shorter edge (px) of the grayscale image scoring convolutions
/// run on: small enough to keep the hottest CPU stage cheap, large enough
/// that Laplacian/Sobel statistics stay stable.
const DEFAULT_SCORE_EDGE: u32 = 256;

/// Default scoring weights (scene, clarity, edge). Clarity defaults to 0: it
/// double-counts the texture Sobel edge energy already measures. Available
/// as an override knob.
const DEFAULT_W_SCENE: f32 = 0.60;
const DEFAULT_W_CLARITY: f32 = 0.0;
const DEFAULT_W_EDGE: f32 = 0.40;

/// Default JPEG output quality: high enough that grid text stays legible
/// after the cells' downscale, without the size cliff past ~95.
const DEFAULT_JPEG_QUALITY: u8 = 92;

/// Default hero frame border thickness in pixels.
const DEFAULT_HERO_BORDER_PX: u32 = 4;

/// Default cap on hero frames per grid. A hero occupies ~4 cells, so three
/// heroes already claim most of a typical grid's emphasis budget; the
/// effective cap is further bounded by `capacity / 4` at the call site.
const DEFAULT_MAX_HEROES: usize = 3;

/// `extract` samples at this multiple of the detail tier's fps: the
/// subcommand exists for high-density passes over short ranges. An explicit
/// `--base-fps` bypasses the factor so the flag means the same density in
/// both subcommands.
const EXTRACT_FPS_FACTOR: f32 = 3.0;

/// Minimum readable cell edge (px) per quality tier. Quality trades per-cell
/// size for grid capacity at a fixed canvas edge, not output size; extraction
/// pre-scales frames to this floor, so cells render 1:1.
const fn quality_cell_edge(quality: Quality) -> u32 {
    match quality {
        Quality::Low => 256,
        Quality::Medium => 384,
        Quality::High => 512,
    }
}

/// Sampling fps per detail tier. Detail drives sampling density only; dedup
/// stays detail-independent so fine mode doesn't merge the subtle diffs it
/// exists to catch.
const fn detail_fps(detail: Detail) -> f32 {
    match detail {
        Detail::Coarse => 0.5,
        Detail::Standard => 1.0,
        Detail::Fine => 3.0,
    }
}

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
/// With the `clap` feature this derives `clap::Args` directly, so the CLI
/// flags and the override fields are the same declaration: adding an
/// override is one field here, with no hand-copied mirror to drift in
/// `main.rs`. Without the feature it is a plain struct — the programmatic
/// door library consumers and tests use.
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "clap", derive(clap::Args))]
pub struct Overrides {
    /// Resolution tier: minimum readable cell edge (default: low).
    #[cfg_attr(feature = "clap", arg(long, value_enum))]
    pub quality: Option<Quality>,
    /// Detail / sampling-density tier: sampling fps (default: standard).
    #[cfg_attr(feature = "clap", arg(long, value_enum))]
    pub detail: Option<Detail>,

    /// Override base sampling fps (finer than --detail; wins over its tier).
    #[cfg_attr(feature = "clap", arg(long))]
    pub base_fps: Option<f32>,
    /// Override pHash Hamming distance threshold for dedup.
    #[cfg_attr(feature = "clap", arg(long))]
    pub hash_threshold: Option<u32>,
    /// Override dedup sliding window size in frames (must be >= 1).
    #[cfg_attr(feature = "clap", arg(long))]
    pub dedup_window: Option<NonZeroUsize>,
    /// Override edge length of grayscale image for scoring.
    #[cfg_attr(feature = "clap", arg(long))]
    pub score_edge: Option<u32>,

    /// Override scene-change weight in scoring.
    #[cfg_attr(feature = "clap", arg(long))]
    pub w_scene: Option<f32>,
    /// Override clarity weight in scoring.
    #[cfg_attr(feature = "clap", arg(long))]
    pub w_clarity: Option<f32>,
    /// Override edge-density weight in scoring.
    #[cfg_attr(feature = "clap", arg(long))]
    pub w_edge: Option<f32>,

    /// Override chunk size in seconds.
    #[cfg_attr(feature = "clap", arg(long))]
    pub chunk_seconds: Option<f32>,
    /// Override minimum temporal gap (seconds) between selected frames.
    #[cfg_attr(feature = "clap", arg(long))]
    pub min_gap: Option<f32>,

    /// Override JPEG output quality (1-100).
    #[cfg_attr(feature = "clap", arg(long))]
    pub jpeg_quality: Option<u8>,
    /// Override hero frame border thickness in pixels.
    #[cfg_attr(feature = "clap", arg(long))]
    pub hero_border_px: Option<u32>,

    /// Override minimum raw scene difference (0-255) for a selected frame to
    /// qualify as a hero; lower catches softer cuts.
    #[cfg_attr(feature = "clap", arg(long))]
    pub hero_scene_diff: Option<f32>,
    /// Override max hero frames per grid (0 disables hero emphasis).
    #[cfg_attr(feature = "clap", arg(long))]
    pub max_heroes: Option<usize>,
}

/// Sampling-strategy options, `process` only: `extract` always samples its
/// explicit range at fixed fps, so it uses the [`Default`] values.
///
/// The clap defaults and [`Default`] share the `DEFAULT_*` consts, so the CLI
/// and a programmatically-built config cannot disagree.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "clap", derive(clap::Args))]
pub struct StrategyOpts {
    /// Frame sampling strategy [default: adaptive]. When omitted, the default
    /// is applied leniently: it degrades to fixed-fps on sources with too few
    /// keyframes. An explicit value is honored strictly — too few keyframes is
    /// then an error rather than a fall back.
    #[cfg_attr(feature = "clap", arg(long, value_enum))]
    pub strategy: Option<SampleStrategy>,
    /// Gap threshold (seconds) for adaptive sparse region fill.
    #[cfg_attr(feature = "clap", arg(long, default_value_t = Self::DEFAULT_GAP_THRESHOLD))]
    pub gap_threshold: f32,
    /// Minimum keyframes required for keyframes/adaptive strategies.
    #[cfg_attr(feature = "clap", arg(long, default_value_t = Self::DEFAULT_MIN_KEYFRAMES))]
    pub min_keyframes: usize,
    /// Max duration (seconds) allowed for fixed-fps sampling; longer videos
    /// need --force or a keyframe strategy.
    #[cfg_attr(feature = "clap", arg(long, default_value_t = Self::DEFAULT_MAX_DURATION))]
    pub max_duration: f32,
    /// Allow fixed-fps on videos exceeding the max-duration limit.
    #[cfg_attr(feature = "clap", arg(long))]
    pub force: bool,
}

impl StrategyOpts {
    /// Strategy used when `--strategy` is omitted. Adaptive samples keyframes
    /// (dense, deduplicated coverage) and fills only sparse regions at
    /// fixed-fps, so a bare `process` gets high information density on typical
    /// edited video while still degrading gracefully on keyframe-poor sources.
    pub const DEFAULT_STRATEGY: SampleStrategy = SampleStrategy::Adaptive;
    pub const DEFAULT_GAP_THRESHOLD: f32 = 5.0;
    pub const DEFAULT_MIN_KEYFRAMES: usize = 10;
    pub const DEFAULT_MAX_DURATION: f32 = 3600.0;
}

impl Default for StrategyOpts {
    fn default() -> Self {
        Self {
            strategy: None,
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
    /// Sampling fps for the `extract` subcommand. Defaults to the detail
    /// tier's fps × [`EXTRACT_FPS_FACTOR`], but an explicit `--base-fps` or
    /// profile value wins as-is, so the flag means the same density in both
    /// subcommands.
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
        // Shares the canvas edge's ceiling (`types::checked_edge`): an
        // unbounded score_edge drives decode pre-scaling just like an
        // unbounded canvas would.
        let score_edge = crate::types::checked_edge("score_edge", score_edge)?;
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
    /// Sanity cap on the hero border thickness. Deliberately not derived
    /// from cell geometry: the justified layout sizes cells from the canvas
    /// edge and row counts (a cell may land below any fixed floor), and
    /// `draw_border` already clamps the drawn thickness per cell. The cap's
    /// one job is to keep the `rect + inset` pixel arithmetic in compose
    /// overflow-free by construction — canvas coordinates are bounded by
    /// the JPEG dimension limit, so any small constant suffices; 256 merely
    /// leaves generous headroom over every plausible border.
    const MAX_HERO_BORDER_PX: u32 = 256;

    fn new(jpeg_quality: u8, hero_border_px: u32) -> crate::error::Result<Self> {
        if !(1..=100).contains(&jpeg_quality) {
            return invalid(
                "jpeg_quality",
                format!("{jpeg_quality} (must be in 1..=100)"),
            );
        }
        if hero_border_px > Self::MAX_HERO_BORDER_PX {
            return invalid(
                "hero_border_px",
                format!("{hero_border_px} (must be <= {})", Self::MAX_HERO_BORDER_PX),
            );
        }
        Ok(Self {
            jpeg_quality,
            hero_border_px,
        })
    }
}

/// Hero-selection parameters. `scene_diff` is a mean absolute grayscale
/// difference in `0..=255`; a threshold outside that range would silently
/// disable heroes (`--max-heroes 0` is the explicit way to do that).
#[derive(Debug, Clone, Copy)]
struct HeroCfg {
    scene_diff_min: f32,
    max_per_grid: usize,
}

impl HeroCfg {
    fn new(scene_diff_min: f32, max_per_grid: usize) -> crate::error::Result<Self> {
        if !scene_diff_min.is_finite() || !(0.0..=255.0).contains(&scene_diff_min) {
            return invalid(
                "hero_scene_diff",
                format!("{scene_diff_min} (must be finite and in 0..=255)"),
            );
        }
        Ok(Self {
            scene_diff_min,
            max_per_grid,
        })
    }
}

/// Resolved strategy parameters; thresholds finite and positive by
/// construction.
#[derive(Debug, Clone, Copy)]
struct StrategyCfg {
    strategy: SampleStrategy,
    /// True when `--strategy` was given explicitly. Governs the
    /// insufficient-keyframe response: explicit requests error, defaulted
    /// ones fall back to fixed-fps.
    explicit: bool,
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
        // `min_keyframes` needs no floor: 0 merely disables the
        // keyframe-poor degrade/reject checks, and the downstream math is
        // total — `mean_gap`/`max_gap` return 0 below two keyframes, and a
        // run whose keyframe selection comes up empty fails with
        // `NoFramesExtracted` (`pipeline::reject_empty_run`) instead of
        // rendering an empty grid.
        Ok(Self {
            strategy: opts.strategy.unwrap_or(StrategyOpts::DEFAULT_STRATEGY),
            explicit: opts.strategy.is_some(),
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
    hero: HeroCfg,
    strategy: StrategyCfg,
    quality: Quality,
    detail: Detail,
}

/// A named tuning profile: a partial parameter set one priority level
/// below CLI overrides. The enum is the profile's identity everywhere —
/// CLI flag values, chunk-length prediction, override resolution — so an
/// unknown name can only exist as a parse error, never as a value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Profile {
    Balanced,
    Fast,
    Quality,
    Action,
}

impl Profile {
    /// Variant count, witnessed by an exhaustive match: adding a profile
    /// fails to compile here until it is counted, and [`Profile::ALL`]'s
    /// array type then refuses to build until the new variant joins it.
    const COUNT: usize = match Profile::Balanced {
        Profile::Balanced | Profile::Fast | Profile::Quality | Profile::Action => 4,
    };

    /// Every profile, in the order help text and error messages list them.
    pub const ALL: [Profile; Self::COUNT] = [
        Profile::Balanced,
        Profile::Fast,
        Profile::Quality,
        Profile::Action,
    ];

    /// The one CLI spelling of this profile (`--profile <name>`).
    pub const fn cli_name(self) -> &'static str {
        match self {
            Profile::Balanced => "balanced",
            Profile::Fast => "fast",
            Profile::Quality => "quality",
            Profile::Action => "action",
        }
    }
}

/// Parse a CLI spelling; an unknown name is an [`ErrorCode::InvalidProfile`]
/// error listing the valid options (derived from [`Profile::ALL`]).
impl std::str::FromStr for Profile {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Profile::ALL
            .into_iter()
            .find(|p| p.cli_name() == s)
            .ok_or_else(|| {
                Error::new(
                    ErrorCode::InvalidProfile,
                    format!(
                        "unknown profile '{s}'; available: {}",
                        Profile::ALL.map(Profile::cli_name).join(", ")
                    ),
                )
            })
    }
}

/// Named profiles are partial parameter sets — the same shape as CLI
/// overrides, one priority level below them. Total: every [`Profile`] has
/// a parameter set, so "unknown profile" cannot reach this table.
fn builtin_profile(profile: Profile) -> Overrides {
    match profile {
        Profile::Balanced => Overrides::default(),
        // Coarse tier (not raw base_fps) so extract keeps its 3x-densify semantics.
        Profile::Fast => Overrides {
            detail: Some(Detail::Coarse),
            chunk_seconds: Some(90.0),
            min_gap: Some(1.0),
            jpeg_quality: Some(85),
            ..Default::default()
        },
        Profile::Quality => Overrides {
            quality: Some(Quality::High),
            chunk_seconds: Some(45.0),
            min_gap: Some(0.3),
            jpeg_quality: Some(95),
            ..Default::default()
        },
        // Fine tier to catch fast cuts; hero threshold at the bottom of the
        // calibrated cut/motion overlap (~67..=70) to flag softer cuts.
        Profile::Action => Overrides {
            detail: Some(Detail::Fine),
            w_scene: Some(0.75),
            w_edge: Some(0.25),
            min_gap: Some(0.3),
            hero_scene_diff: Some(65.0),
            ..Default::default()
        },
    }
}

/// The chunk length (seconds) a run with `--profile <name>` resolves to,
/// absent an explicit `--chunk-seconds` override.
///
/// Routed through the same profile table [`Config::new`] resolves against,
/// so consumers predicting grid counts (the harness budget gate divides the
/// covered span by this value) cannot drift from what a run will actually
/// do.
pub fn default_chunk_seconds(profile: Option<Profile>) -> f32 {
    profile
        .map(builtin_profile)
        .and_then(|p| p.chunk_seconds)
        .unwrap_or(DEFAULT_CHUNK_SECONDS)
}

impl Config {
    /// Construct config from CLI arguments.
    ///
    /// Priority: CLI overrides > named profile > defaults. The tiers resolve
    /// through the same chain first; tier-derived values (sampling fps, cell
    /// floor) then serve as defaults for the finer per-field knobs.
    pub fn new(
        max_edge: u32,
        profile: Option<Profile>,
        cli: &Overrides,
        strategy_opts: StrategyOpts,
    ) -> crate::error::Result<Self> {
        let prof = profile.map(builtin_profile);

        macro_rules! resolve {
            ($field:ident, $default:expr) => {
                cli.$field
                    .or(prof.as_ref().and_then(|p| p.$field))
                    .unwrap_or($default)
            };
        }

        let quality = resolve!(quality, DEFAULT_QUALITY);
        let detail = resolve!(detail, DEFAULT_DETAIL);
        let tier_fps = detail_fps(detail);

        Ok(Self {
            canvas: CanvasSpec::new(max_edge, quality_cell_edge(quality))?,
            sampling: Sampling::new(
                resolve!(base_fps, tier_fps),
                resolve!(base_fps, tier_fps * EXTRACT_FPS_FACTOR),
                resolve!(chunk_seconds, DEFAULT_CHUNK_SECONDS),
                resolve!(min_gap, DEFAULT_MIN_GAP),
                resolve!(score_edge, DEFAULT_SCORE_EDGE),
            )?,
            dedup: DedupCfg::new(
                resolve!(hash_threshold, DEFAULT_HASH_THRESHOLD),
                resolve!(dedup_window, DEFAULT_DEDUP_WINDOW),
            )?,
            weights: Weights::new(
                resolve!(w_scene, DEFAULT_W_SCENE),
                resolve!(w_clarity, DEFAULT_W_CLARITY),
                resolve!(w_edge, DEFAULT_W_EDGE),
            )?,
            render: RenderCfg::new(
                resolve!(jpeg_quality, DEFAULT_JPEG_QUALITY),
                resolve!(hero_border_px, DEFAULT_HERO_BORDER_PX),
            )?,
            hero: HeroCfg::new(
                resolve!(hero_scene_diff, crate::score::DEFAULT_HERO_SCENE_DIFF),
                resolve!(max_heroes, DEFAULT_MAX_HEROES),
            )?,
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

    /// Minimum raw `scene_diff` for a frame to qualify as a hero.
    pub fn hero_scene_diff(&self) -> f32 {
        self.hero.scene_diff_min
    }

    /// Cap on hero frames per grid; `0` disables hero emphasis entirely.
    pub fn max_heroes(&self) -> usize {
        self.hero.max_per_grid
    }

    pub fn strategy(&self) -> SampleStrategy {
        self.strategy.strategy
    }

    /// Whether the strategy was requested explicitly (`--strategy`) rather than
    /// defaulted. An explicit keyframe/adaptive request errors on too few
    /// keyframes; a defaulted one degrades to fixed-fps instead.
    pub fn strategy_explicit(&self) -> bool {
        self.strategy.explicit
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
        Config::new(2000, None, &overrides, StrategyOpts::default())
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
        // quality high → 512px floor.
        for max_edge in [0, 511] {
            let ov = Overrides {
                quality: Some(Quality::High),
                ..Default::default()
            };
            let cfg = Config::new(max_edge, None, &ov, StrategyOpts::default());
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
    fn rejects_oversized_hero_border() {
        let ov = Overrides {
            hero_border_px: Some(u32::MAX),
            ..Default::default()
        };
        assert_eq!(
            config_with(ov)
                .expect_err("oversized hero border rejected")
                .code,
            ErrorCode::InvalidConfig
        );
    }

    #[test]
    fn rejects_out_of_range_hero_scene_diff() {
        for bad in [f32::NAN, -1.0, 255.1] {
            let ov = Overrides {
                hero_scene_diff: Some(bad),
                ..Default::default()
            };
            assert_eq!(
                config_with(ov)
                    .expect_err("out-of-range hero_scene_diff rejected")
                    .code,
                ErrorCode::InvalidConfig
            );
        }
    }

    #[test]
    fn accepts_zero_max_heroes_as_disable() {
        let ov = Overrides {
            max_heroes: Some(0),
            ..Default::default()
        };
        let cfg = config_with(ov).expect("max_heroes 0 is a valid disable switch");
        assert_eq!(cfg.max_heroes(), 0);
    }

    #[test]
    fn cli_override_beats_profile_value() {
        let ov = Overrides {
            hero_scene_diff: Some(80.0),
            ..Default::default()
        };
        let cfg = Config::new(2000, Some(Profile::Action), &ov, StrategyOpts::default())
            .expect("config builds");
        assert_eq!(cfg.hero_scene_diff(), 80.0, "CLI outranks the profile");
        assert_eq!(
            cfg.base_fps(),
            3.0,
            "untouched profile fields still apply (action sets the fine tier)"
        );
    }

    #[test]
    fn profile_sets_tiers() {
        let cfg = Config::new(
            2000,
            Some(Profile::Quality),
            &Overrides::default(),
            StrategyOpts::default(),
        )
        .expect("config builds");
        assert_eq!(cfg.min_frame_size(), 512, "quality profile → high tier");

        let cfg = Config::new(
            2000,
            Some(Profile::Fast),
            &Overrides::default(),
            StrategyOpts::default(),
        )
        .expect("config builds");
        assert_eq!(cfg.base_fps(), 0.5, "fast profile → coarse tier");
        assert_eq!(
            cfg.extract_fps(),
            1.5,
            "tier-based profile keeps extract's 3x densify semantics"
        );
    }

    #[test]
    fn cli_tier_overrides_profile_tier() {
        let ov = Overrides {
            quality: Some(Quality::Low),
            ..Default::default()
        };
        let cfg = Config::new(2000, Some(Profile::Quality), &ov, StrategyOpts::default())
            .expect("config builds");
        assert_eq!(cfg.min_frame_size(), 256, "explicit --quality low wins");
        assert_eq!(cfg.jpeg_quality(), 95, "other profile fields still apply");
    }

    #[test]
    fn default_chunk_seconds_matches_every_profile_resolution() {
        for profile in std::iter::once(None).chain(Profile::ALL.into_iter().map(Some)) {
            let cfg = Config::new(
                2000,
                profile,
                &Overrides::default(),
                StrategyOpts::default(),
            )
            .expect("config builds");
            assert_eq!(
                cfg.chunk_seconds(),
                default_chunk_seconds(profile),
                "profile {profile:?} prediction drifted from resolution"
            );
        }
    }

    #[test]
    fn parses_every_cli_name_back_to_its_profile() {
        for profile in Profile::ALL {
            assert_eq!(
                profile.cli_name().parse::<Profile>().expect("round-trips"),
                profile
            );
        }
    }

    #[test]
    fn rejects_unknown_profile_name_listing_the_options() {
        let err = "cinematic"
            .parse::<Profile>()
            .expect_err("unknown profile rejected");
        assert_eq!(err.code, ErrorCode::InvalidProfile);
        let listed = Profile::ALL.map(Profile::cli_name).join(", ");
        assert!(err.to_string().contains(&listed), "{err}");
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
    fn rejects_score_edge_above_the_image_edge_limit() {
        let ov = Overrides {
            score_edge: Some(65_536),
            ..Default::default()
        };
        assert_eq!(
            config_with(ov)
                .expect_err("score_edge above the JPEG edge limit rejected")
                .code,
            ErrorCode::InvalidConfig
        );
    }

    #[test]
    fn rejects_canvas_above_jpeg_dimension_limit() {
        let cfg = Config::new(65_536, None, &Overrides::default(), StrategyOpts::default());
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
            let cfg = Config::new(2000, None, &Overrides::default(), opts);
            assert_eq!(
                cfg.expect_err("invalid strategy opts rejected").code,
                ErrorCode::InvalidConfig
            );
        }
    }
}
