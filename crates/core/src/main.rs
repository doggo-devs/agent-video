//! agent-video CLI entry point.
//!
//! Three subcommands: `process` (smart frame selection + grid composition),
//! `extract` (high-density sampling of a time range), and `probe` (media
//! inspection to machine-readable JSON).

mod compose;
mod config;
mod dedup;
mod error;
mod extract;
mod layout;
mod media;
mod pipeline;
mod probe;
mod report;
mod score;
mod subtitle;
mod types;

use std::num::NonZeroUsize;
use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, Subcommand, ValueHint};

use crate::config::{Config, Overrides, StrategyOpts};
use crate::error::Error;
use crate::pipeline::parse_range;
use crate::types::TimeRange;

#[derive(Parser)]
#[command(
    name = "agent-video",
    version,
    about = "Video → VLM-readable grid images"
)]
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
        /// Sampling-strategy options, `process` only: `extract` always samples
        /// its explicit range at fixed fps, so accepting these there would be
        /// a silent no-op.
        #[command(flatten)]
        strategy: StrategyOpts,
        /// Only decode from this timestamp in seconds; grid timestamps stay
        /// absolute to the file.
        #[arg(long)]
        from: Option<f32>,
        /// Only decode until this timestamp in seconds.
        #[arg(long)]
        until: Option<f32>,
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
    /// Inspect a video: machine-readable JSON metadata on stdout.
    Probe {
        /// Input video file.
        #[arg(value_hint = ValueHint::FilePath)]
        input: PathBuf,
        /// Add keyframe statistics (slower: scans every video packet).
        #[arg(long)]
        keyframes: bool,
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

/// Options shared by both subcommands. The parameter overrides live directly
/// on [`Overrides`] (flattened), so the CLI surface and the config override
/// set are one declaration.
#[derive(clap::Args)]
struct CommonOpts {
    /// Canvas edge length in pixels.
    #[arg(long, default_value_t = 2048)]
    max_edge: u32,
    /// Named profile (balanced, fast, quality, action); presets tiers and
    /// tuning knobs, explicit flags still win.
    #[arg(long)]
    profile: Option<String>,

    #[command(flatten)]
    overrides: Overrides,

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

fn main() -> ExitCode {
    reset_sigpipe();
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

/// Restore the default `SIGPIPE` disposition on Unix.
///
/// The Rust runtime installs `SIG_IGN` for `SIGPIPE` before `main` runs. That
/// makes a write to a closed downstream pipe (e.g. `agent-video ... | head`)
/// fail with `EPIPE` instead of terminating the process; the failing
/// `println!`/`eprintln!` then panics, which under this crate's
/// `panic = "abort"` profile aborts with `SIGABRT` (exit 134) and a core dump.
/// Resetting to `SIG_DFL` restores the conventional CLI behavior: a closed
/// pipe kills the process quietly via `SIGPIPE` (exit 141). No-op off Unix.
fn reset_sigpipe() {
    #[cfg(unix)]
    // SAFETY: `signal` with `SIG_DFL` merely restores the kernel's default
    // disposition for a standard signal. It touches no memory we own and is
    // called once, before any threads are spawned, so there is no data race.
    unsafe {
        libc::signal(libc::SIGPIPE, libc::SIG_DFL);
    }
}

fn run() -> anyhow::Result<()> {
    let cli = Cli::parse();

    let (io, opts) = match &cli.command {
        Command::Process { io, opts, .. } => (io, opts),
        Command::Extract { io, opts, .. } => (io, opts),
        Command::Probe { input, keyframes } => return report::run(input, *keyframes),
    };

    let strategy_opts = match &cli.command {
        Command::Process { strategy, .. } => *strategy,
        Command::Extract { .. } | Command::Probe { .. } => StrategyOpts::default(),
    };

    let cfg = Config::new(
        opts.max_edge,
        opts.profile.as_deref(),
        &opts.overrides,
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
            threads,
            in_flight,
            from,
            until,
            ..
        } => {
            let window = pipeline::Window::from_args(*from, *until)?;
            let threads = threads.unwrap_or(0);
            let in_flight = in_flight.unwrap_or(pipeline::DEFAULT_IN_FLIGHT);
            pipeline::process(
                &io.input, &io.output, &cfg, mode, threads, in_flight, subs, window,
            )
        }
        Command::Extract { range, .. } => {
            pipeline::extract_range(&io.input, &io.output, &cfg, *range, mode, subs)
        }
        Command::Probe { .. } => unreachable!("probe returns before config resolution"),
    }
}
