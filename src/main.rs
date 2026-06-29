//! vidgrid CLI entry point.
//!
//! Two subcommands: `process` (smart frame selection + grid composition for
//! short/medium videos) and `extract` (high-density sampling of a time range).
//! This module only parses arguments, resolves a [`Preset`], and dispatches.

mod compose;
mod config;
mod dedup;
mod extract;
mod layout;
mod pipeline;
mod preset;
mod profile;
mod score;
mod subtitle;
mod types;

use std::path::PathBuf;

use clap::{Parser, Subcommand, ValueEnum, ValueHint};

use crate::config::Config;
use crate::pipeline::parse_range;
use crate::preset::{model_max_pixels, resolve_preset};
use crate::profile::ProfileOverrides;
use crate::types::{Detail, Quality};

#[derive(Parser)]
#[command(name = "vidgrid", about = "Video → VLM-readable grid images")]
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
        /// Rayon worker threads (default: CPU cores).
        #[arg(long, short = 'j')]
        threads: Option<usize>,
        /// Max in-flight chunk jobs; bounds memory (default: 2).
        #[arg(long)]
        in_flight: Option<usize>,
    },
    /// High-density sampling of a specific time range into a single grid.
    Extract {
        #[command(flatten)]
        io: IoArgs,
        /// Time range, e.g. `01:23-01:35`, validated at parse time.
        #[arg(long, value_parser = parse_range)]
        range: (f32, f32),
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
    /// Model preset seeding `max_pixels` (qwen2.5vl, gpt4o, claude, internvl).
    #[arg(long, default_value = "qwen2.5vl")]
    model: String,
    /// Override the canvas area budget in pixels.
    #[arg(long)]
    max_pixels: Option<u32>,
    /// Resolution tier.
    #[arg(long, value_enum, default_value_t = QualityArg::High)]
    quality: QualityArg,
    /// Detail / sampling-density tier.
    #[arg(long, value_enum, default_value_t = DetailArg::Standard)]
    detail: DetailArg,
    /// Named profile (balanced, fast, quality, action, or custom).
    #[arg(long)]
    profile: Option<String>,

    // --- Sampling overrides ---
    /// Override base sampling fps.
    #[arg(long)]
    base_fps: Option<f32>,
    /// Override pHash Hamming distance threshold for dedup.
    #[arg(long)]
    hash_threshold: Option<u32>,
    /// Override dedup sliding window size in frames.
    #[arg(long)]
    dedup_window: Option<usize>,
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
    #[arg(long)]
    json: bool,
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

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    let (io, opts) = match &cli.command {
        Command::Process { io, opts, .. } => (io, opts),
        Command::Extract { io, opts, .. } => (io, opts),
    };

    let max_pixels = match opts.max_pixels {
        Some(p) => p,
        None => model_max_pixels(&opts.model).ok_or_else(|| {
            anyhow::anyhow!(
                "unknown model preset '{}'; use --max-pixels or one of: \
                 qwen2.5vl, gpt4o, claude, internvl",
                opts.model
            )
        })?,
    };

    let quality = Quality::from(opts.quality);
    let detail = Detail::from(opts.detail);
    let preset = resolve_preset(max_pixels, quality, detail);

    let overrides = ProfileOverrides {
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
    let prof = profile::resolve(opts.profile.as_deref(), &overrides)?;
    let cfg = Config::new(preset, prof, quality, detail);

    let mode = if opts.json {
        pipeline::OutputMode::Json
    } else {
        pipeline::OutputMode::Files
    };

    match &cli.command {
        Command::Process {
            threads, in_flight, ..
        } => {
            let threads = threads.unwrap_or(0); // 0 = rayon default (CPU cores)
            let in_flight = in_flight.unwrap_or(pipeline::DEFAULT_IN_FLIGHT);
            pipeline::process(&io.input, &io.output, &cfg, mode, threads, in_flight)
        }
        Command::Extract { range, .. } => {
            let (start, end) = *range;
            pipeline::extract_range(&io.input, &io.output, &cfg, start, end, mode)
        }
    }
}
