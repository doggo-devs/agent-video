//! agent-video CLI entry point.
//!
//! Two subcommands: `process` (smart frame selection + grid composition for
//! short/medium videos) and `extract` (high-density sampling of a time range).

mod child;
mod compose;
mod config;
mod dedup;
mod error;
mod extract;
mod layout;
mod pipeline;
mod probe;
mod score;
mod subtitle;
mod types;

use std::num::NonZeroUsize;
use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, Subcommand, ValueEnum, ValueHint};

use crate::config::{Config, Overrides, StrategyOpts};
use crate::error::Error;
use crate::pipeline::parse_range;
use crate::probe::SampleStrategy;
use crate::types::{Detail, Quality, TimeRange};

#[derive(Parser)]
#[command(name = "agent-video", about = "Video → VLM-readable grid images")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Smart frame selection + grid composition.
    Process {
        #[command(flatten)]
        io: IoArgs,
        #[command(flatten)]
        opts: CommonOpts,
        #[command(flatten)]
        strategy: StrategyArgs,
        /// Rayon worker threads (default: CPU cores).
        #[arg(long, short = 'j')]
        threads: Option<usize>,
        /// Max in-flight chunk jobs; bounds memory (default: 2, must be >= 1).
        #[arg(long)]
        in_flight: Option<NonZeroUsize>,
    },
    /// High-density sampling of a specific time range into a single grid.
    Extract {
        #[command(flatten)]
        io: IoArgs,
        /// Time range, e.g. `01:23-01:35`, validated at parse time.
        #[arg(long, value_parser = parse_range)]
        range: TimeRange,
        #[command(flatten)]
        opts: CommonOpts,
    },
}

/// Input file and output directory, shared by both subcommands.
#[derive(clap::Args)]
struct IoArgs {
    /// Input video file.
    #[arg(value_hint = ValueHint::FilePath)]
    input: PathBuf,
    /// Output directory.
    #[arg(short, long, value_hint = ValueHint::DirPath)]
    output: PathBuf,
}

/// Options shared by both subcommands.
#[derive(clap::Args)]
struct CommonOpts {
    /// Canvas edge length in pixels.
    #[arg(long, default_value_t = 2048)]
    max_edge: u32,
    /// Resolution tier.
    #[arg(long, value_enum, default_value_t = QualityArg::Low)]
    quality: QualityArg,
    /// Detail / sampling-density tier.
    #[arg(long, value_enum, default_value_t = DetailArg::Standard)]
    detail: DetailArg,
    /// Named profile (balanced, fast, quality, action).
    #[arg(long)]
    profile: Option<String>,

    // --- Sampling overrides ---
    /// Override base sampling fps.
    #[arg(long)]
    base_fps: Option<f32>,
    /// Override pHash Hamming distance threshold for dedup.
    #[arg(long)]
    hash_threshold: Option<u32>,
    /// Override dedup sliding window size in frames (must be >= 1).
    #[arg(long)]
    dedup_window: Option<NonZeroUsize>,
    /// Override edge length of grayscale image for scoring.
    #[arg(long)]
    score_edge: Option<u32>,

    // --- Scoring overrides ---
    /// Override scene-change weight in scoring.
    #[arg(long)]
    w_scene: Option<f32>,
    /// Override clarity weight in scoring.
    #[arg(long)]
    w_clarity: Option<f32>,
    /// Override edge-density weight in scoring.
    #[arg(long)]
    w_edge: Option<f32>,

    // --- Timing overrides ---
    /// Override chunk size in seconds.
    #[arg(long)]
    chunk_seconds: Option<f32>,
    /// Override minimum temporal gap (seconds) between selected frames.
    #[arg(long)]
    min_gap: Option<f32>,

    // --- Output overrides ---
    /// Override JPEG output quality (0-100).
    #[arg(long)]
    jpeg_quality: Option<u8>,
    /// Override hero frame border thickness in pixels.
    #[arg(long)]
    hero_border_px: Option<u32>,

    /// Output JSON with base64 images to stdout instead of writing files.
    /// Holds all grids in memory until done; prefer files for long videos.
    #[arg(long)]
    json: bool,

    /// Explicit subtitle file (overrides embedded and sidecar detection).
    #[arg(long, value_hint = ValueHint::FilePath)]
    subtitle: Option<PathBuf>,

    /// Embedded subtitle track index (0-based, use when multiple tracks exist).
    #[arg(long, conflicts_with = "subtitle")]
    subtitle_track: Option<usize>,
}

/// Sampling-strategy options, `process` only: `extract` always samples its
/// explicit range at fixed fps, so accepting these there would be a silent
/// no-op.
#[derive(clap::Args)]
struct StrategyArgs {
    /// Frame sampling strategy.
    #[arg(long, value_enum, default_value_t = StrategyArg::FixedFps)]
    strategy: StrategyArg,
    /// Allow fixed-fps on videos exceeding 1h duration limit.
    #[arg(long)]
    force: bool,
    /// Minimum keyframes required for keyframes/adaptive strategies.
    #[arg(long, default_value_t = StrategyOpts::DEFAULT_MIN_KEYFRAMES)]
    min_keyframes: usize,
    /// Gap threshold (seconds) for adaptive sparse region fill.
    #[arg(long, default_value_t = StrategyOpts::DEFAULT_GAP_THRESHOLD)]
    gap_threshold: f32,
}

#[derive(Copy, Clone, Debug, ValueEnum)]
enum QualityArg {
    Low,
    Medium,
    High,
}

#[derive(Copy, Clone, Debug, ValueEnum)]
enum DetailArg {
    Coarse,
    Standard,
    Fine,
}

#[derive(Copy, Clone, Debug, Default, ValueEnum)]
enum StrategyArg {
    #[default]
    FixedFps,
    Keyframes,
    Adaptive,
}

impl From<QualityArg> for Quality {
    fn from(q: QualityArg) -> Self {
        match q {
            QualityArg::Low => Quality::Low,
            QualityArg::Medium => Quality::Medium,
            QualityArg::High => Quality::High,
        }
    }
}

impl From<DetailArg> for Detail {
    fn from(d: DetailArg) -> Self {
        match d {
            DetailArg::Coarse => Detail::Coarse,
            DetailArg::Standard => Detail::Standard,
            DetailArg::Fine => Detail::Fine,
        }
    }
}

impl From<StrategyArg> for crate::probe::SampleStrategy {
    fn from(s: StrategyArg) -> Self {
        match s {
            StrategyArg::FixedFps => Self::FixedFps,
            StrategyArg::Keyframes => Self::Keyframes,
            StrategyArg::Adaptive => Self::Adaptive,
        }
    }
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            let code = e
                .downcast_ref::<Error>()
                .map(|e| e.code.exit_code())
                .unwrap_or(1);
            eprintln!("error: {e:#}");
            ExitCode::from(code)
        }
    }
}

fn run() -> anyhow::Result<()> {
    let cli = Cli::parse();

    let (io, opts) = match &cli.command {
        Command::Process { io, opts, .. } => (io, opts),
        Command::Extract { io, opts, .. } => (io, opts),
    };

    let quality = Quality::from(opts.quality);
    let detail = Detail::from(opts.detail);

    let overrides = Overrides {
        w_scene: opts.w_scene,
        w_clarity: opts.w_clarity,
        w_edge: opts.w_edge,
        base_fps: opts.base_fps,
        hash_threshold: opts.hash_threshold,
        dedup_window: opts.dedup_window,
        score_edge: opts.score_edge,
        chunk_seconds: opts.chunk_seconds,
        min_gap: opts.min_gap,
        jpeg_quality: opts.jpeg_quality,
        hero_border_px: opts.hero_border_px,
    };

    let strategy_opts = match &cli.command {
        Command::Process { strategy, .. } => StrategyOpts {
            strategy: SampleStrategy::from(strategy.strategy),
            gap_threshold: strategy.gap_threshold,
            min_keyframes: strategy.min_keyframes,
            force: strategy.force,
        },
        Command::Extract { .. } => StrategyOpts::default(),
    };

    let cfg = Config::new(
        opts.max_edge,
        quality,
        detail,
        opts.profile.as_deref(),
        &overrides,
        strategy_opts,
    )?;

    let mode = if opts.json {
        pipeline::OutputMode::Json
    } else {
        pipeline::OutputMode::Files
    };

    // clap's `conflicts_with` guarantees the flags are mutually exclusive.
    let subs = match (&opts.subtitle, opts.subtitle_track) {
        (Some(path), _) => pipeline::SubtitleSource::File(path),
        (None, Some(track)) => pipeline::SubtitleSource::Track(track),
        (None, None) => pipeline::SubtitleSource::Auto,
    };

    match &cli.command {
        Command::Process {
            threads, in_flight, ..
        } => {
            let threads = threads.unwrap_or(0);
            let in_flight = in_flight.unwrap_or(pipeline::DEFAULT_IN_FLIGHT);
            pipeline::process(&io.input, &io.output, &cfg, mode, threads, in_flight, subs)
        }
        Command::Extract { range, .. } => {
            pipeline::extract_range(&io.input, &io.output, &cfg, *range, mode, subs)
        }
    }
}
