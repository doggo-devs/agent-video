//! vidgrid CLI entry point.
//!
//! Two subcommands: `process` (smart frame selection + grid composition for
//! short/medium videos) and `extract` (high-density sampling of a time range).
//! This module only parses arguments, resolves a [`Preset`], and dispatches.

mod compose;
mod dedup;
mod extract;
mod layout;
mod pipeline;
mod preset;
mod score;
mod subtitle;
mod types;

use std::path::PathBuf;

use clap::{Parser, Subcommand, ValueEnum};

use crate::preset::{model_max_pixels, resolve_preset};
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
        /// Input video file.
        input: PathBuf,
        /// Output directory.
        #[arg(short, long)]
        output: PathBuf,
        #[command(flatten)]
        opts: CommonOpts,
    },
    /// High-density sampling of a specific time range into a single grid.
    Extract {
        /// Input video file.
        input: PathBuf,
        /// Output directory.
        #[arg(short, long)]
        output: PathBuf,
        /// Time range, e.g. `01:23-01:35`.
        #[arg(long)]
        range: String,
        #[command(flatten)]
        opts: CommonOpts,
    },
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

    let (input, output, opts) = match &cli.command {
        Command::Process { input, output, opts } => (input, output, opts),
        Command::Extract { input, output, opts, .. } => (input, output, opts),
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

    let quality_label = format!("{:?}", opts.quality).to_lowercase();
    let detail_label = format!("{:?}", opts.detail).to_lowercase();
    let preset = resolve_preset(max_pixels, opts.quality.into(), opts.detail.into());

    match &cli.command {
        Command::Process { .. } => {
            pipeline::process(input, output, &preset, &quality_label, &detail_label)
        }
        Command::Extract { range, .. } => {
            let (start, end) = pipeline::parse_range(range)?;
            pipeline::extract_range(
                input,
                output,
                &preset,
                &quality_label,
                &detail_label,
                start,
                end,
            )
        }
    }
}
