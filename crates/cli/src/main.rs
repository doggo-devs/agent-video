// SPDX-License-Identifier: Apache-2.0
//! agent-video CLI entry point.
//!
//! Five subcommands: `process` (smart frame selection + grid composition),
//! `extract` (high-density sampling of a time range), `mcp` (the stdio MCP
//! server exposing the `video` tool), `install-whisper` (provision
//! transcription weights), and `cookie` (manage the jar for login-gated
//! sites). `process`/`extract` accept a local file path or any URL the
//! linked FFmpeg can open.

use std::num::NonZeroUsize;
use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, Subcommand, ValueHint};

use agent_video_core::config::{Config, Overrides, StrategyOpts};
use agent_video_core::error::Error;
use agent_video_core::event::{Event, EventSink};
use agent_video_core::pipeline::{self, parse_range};
use agent_video_core::source::Source;
use agent_video_core::types::{TimeRange, format_ts};
use agent_video_harness as harness;

/// The CLI's diagnostic renderer: turns the kernel's typed [`Event`]s back
/// into the exact stderr lines the tool has always printed.
///
/// This is the one place that decides wording, float precision, and timestamp
/// format; the kernel only reports data. Every arm reproduces its historical
/// string byte-for-byte, indentation included.
struct StderrEvents;

impl EventSink for StderrEvents {
    fn event(&self, event: Event<'_>) {
        match event {
            Event::SeekFallback { start, error } => eprintln!(
                "warning: cannot seek to {start:.2}s ({error}); \
                 reading sequentially from the start"
            ),
            Event::FixedFpsWindow { fps, start, end } => {
                let end = end.map_or("end".to_string(), |e| format!("{e:.1}s"));
                eprintln!("fixed-fps mode: sampling at {fps:.1} fps over {start:.1}s-{end}");
            }
            Event::FixedFps { fps } => {
                eprintln!("fixed-fps mode: sampling at {fps:.1} fps");
            }
            Event::KeyframesMode {
                count,
                duration,
                mean_gap,
                max_gap,
            } => eprintln!(
                "keyframes mode: {count} I-frames over {duration:.1}s \
                 (mean gap {mean_gap:.1}s, max {max_gap:.1}s)"
            ),
            Event::AdaptiveMode {
                keyframes,
                max_gap,
                sparse_regions,
                threshold,
            } => eprintln!(
                "adaptive mode: {keyframes} I-frames (max gap {max_gap:.1}s), \
                 {sparse_regions} sparse regions (threshold {threshold:.1}s)"
            ),
            Event::AdaptiveFallbackFixedFps { coverage_percent } => eprintln!(
                "adaptive mode: sparse regions cover {coverage_percent:.0}% of the video; \
                 falling back to a single fixed-fps pass"
            ),
            Event::FillingSparseRegion { start, end } => {
                eprintln!("  filling sparse region {start:.1}s-{end:.1}s");
            }
            Event::KeyframePoorDegrade { count, minimum } => eprintln!(
                "warning: only {count} keyframes (<{minimum} required); \
                 falling back to fixed-fps sampling"
            ),
            Event::StaticCollapsed { start, end, file } => eprintln!(
                "  static stretch {}-{} collapsed into {file}",
                format_ts(start),
                format_ts(end)
            ),
            Event::UnplaceableTimestamp { ts, last } => eprintln!(
                "  warning: skipping frame with unplaceable timestamp {ts} (last {last:?})"
            ),
            Event::GridsWritten { count, dir } => {
                eprintln!("wrote {count} grid(s) to {}", dir.display());
            }
            Event::EmbeddedSubtitlesEmpty => eprintln!(
                "warning: the embedded subtitle stream contains no text cues; \
                 continuing without subtitles"
            ),
            Event::SidecarScanFailed { dir, error } => eprintln!(
                "warning: cannot list directory {} for subtitle sidecars: {error}; \
                 pass --subtitle to use one explicitly",
                dir.display()
            ),
            Event::SidecarEntryUnreadable { error } => {
                eprintln!("warning: skipping unreadable directory entry during sidecar scan: {error}")
            }
            // `Event` is non_exhaustive: a kernel that learns a new diagnostic
            // stays silent here until this renderer teaches it a string.
            _ => {}
        }
    }
}

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
    /// Serve the MCP `video` tool over stdio.
    Mcp,
    /// Download whisper weights (model + Silero VAD) into the models
    /// directory, sha256-verified. Never runs implicitly.
    InstallWhisper {
        /// Pinned model to provision (medium, large-v3); defaults to
        /// `whisper.model` from config.toml.
        #[arg(long)]
        model: Option<String>,
    },
    /// Manage the cookie jar used for login-gated sites (yt-dlp sources).
    Cookie {
        #[command(subcommand)]
        action: CookieAction,
    },
    /// Hidden child side of the crash-isolated GPU probe: enumerate the
    /// named backend's devices and answer via the exit status (0 = a
    /// usable device exists). Runs in a disposable process because ggml's
    /// failure paths can abort; the parent reads only the exit code.
    #[command(hide = true)]
    ProbeGpu { backend: String },
}

#[derive(Subcommand)]
enum CookieAction {
    /// Merge a Netscape-format cookie export into the jar. Tip: export
    /// from a private browsing session and close it afterwards, or the
    /// live browser keeps rotating the cookies out from under the export.
    Import {
        /// Path to a `cookies.txt` exported from a logged-in browser.
        #[arg(value_hint = ValueHint::FilePath)]
        file: PathBuf,
    },
    /// Show the active cookie mode and the jar's domains.
    Status,
    /// Remove all cookies, or one domain's (including its subdomains).
    Clear {
        #[arg(long)]
        domain: Option<String>,
    },
}

/// Run one `cookie` action against the loaded config.
fn run_cookie(action: &CookieAction) -> anyhow::Result<()> {
    let dirs = harness::config::BaseDirs::detect();
    let cfg = harness::config::load_config(&dirs)?;
    match action {
        CookieAction::Import { file } => {
            let report = harness::cookies::import(&cfg.cookies, file)?;
            println!(
                "imported {} cookie(s); the jar now holds {} across all sites",
                report.imported, report.total
            );
            if cfg.cookies.from_browser.is_some() {
                println!(
                    "note: cookies.from_browser is set in config.toml and takes \
                     precedence over the jar"
                );
            }
        }
        CookieAction::Status => {
            match harness::cookies::mode(&cfg.cookies) {
                harness::cookies::CookieMode::Browser(b) => {
                    println!("mode: --cookies-from-browser {b} (live read at every spawn)");
                }
                harness::cookies::CookieMode::Jar(path) => {
                    println!("mode: jar at {}", path.display());
                }
                harness::cookies::CookieMode::Anonymous => {
                    println!(
                        "mode: anonymous — import a jar with `agent-video cookie import` \
                         or set cookies.from_browser in config.toml"
                    );
                }
            }
            for (domain, count) in harness::cookies::status(&cfg.cookies)? {
                println!("  {domain}: {count} cookie(s)");
            }
        }
        CookieAction::Clear { domain } => {
            let removed = harness::cookies::clear(&cfg.cookies, domain.as_deref())?;
            println!("removed {removed} cookie(s)");
        }
    }
    Ok(())
}

/// Input source and output directory, shared by both subcommands.
#[derive(clap::Args)]
struct IoArgs {
    /// Input video: a local file path, or any URL the linked FFmpeg can open
    /// (`https://`, `http://`, ...). The bytes are handed verbatim to the
    /// demuxer; a protocol this build lacks fails with a coded error.
    input: String,
    /// Output directory. Required unless `--json` selects stdout — the two
    /// destinations are mutually exclusive, so a run names exactly one sink.
    #[arg(
        short,
        long,
        value_hint = ValueHint::DirPath,
        required_unless_present = "json",
        conflicts_with = "json"
    )]
    output: Option<PathBuf>,
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
            eprintln!("error: {e:#}");
            ExitCode::from(exit_code_of(&e))
        }
    }
}

/// Every coded error type maps to its decade; exit 1 is reserved for what
/// core's contract defines it as — uncoded errors: bugs and panics. A bad
/// config.toml or a failed weight download is neither.
fn exit_code_of(e: &anyhow::Error) -> u8 {
    if let Some(err) = e.downcast_ref::<Error>() {
        return err.code.exit_code();
    }
    if let Some(err) = e.downcast_ref::<harness::config::ConfigError>() {
        return err.exit_code();
    }
    if let Some(err) = e.downcast_ref::<harness::install::InstallError>() {
        return err.exit_code();
    }
    if let Some(err) = e.downcast_ref::<harness::cookies::CookieError>() {
        return err.exit_code();
    }
    1
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
    // `--mcp` compat: MCP client configs written for the TS harness passed a
    // flag, not a subcommand. Rewriting the first argument keeps them valid.
    let args = std::env::args_os().enumerate().map(|(i, a)| {
        if i == 1 && a == "--mcp" {
            "mcp".into()
        } else {
            a
        }
    });
    let cli = Cli::parse_from(args);

    let (io, opts) = match &cli.command {
        Command::Process { io, opts, .. } => (io, opts),
        Command::Extract { io, opts, .. } => (io, opts),
        Command::Mcp => return harness::mcp::server::serve_stdio(),
        Command::InstallWhisper { model } => {
            let dirs = harness::config::BaseDirs::detect();
            let cfg = harness::config::load_config(&dirs)?;
            harness::install::install_whisper(&cfg, model.as_deref())?;
            return Ok(());
        }
        Command::Cookie { action } => return run_cookie(action),
        Command::ProbeGpu { backend } => {
            // The parent reads only the exit status; bypass the error
            // plumbing and its exit-code conventions entirely.
            std::process::exit(harness::gpu::probe_child(backend));
        }
    };

    let strategy_opts = match &cli.command {
        Command::Process { strategy, .. } => *strategy,
        _ => StrategyOpts::default(),
    };

    let cfg = Config::new(
        opts.max_edge,
        opts.profile.as_deref(),
        &opts.overrides,
        strategy_opts,
    )?;

    // clap guarantees exactly one destination: `-o` xor `--json`.
    let sink = match &io.output {
        Some(dir) => pipeline::OutputSink::Files(dir),
        None => pipeline::OutputSink::JsonStdout,
    };

    // clap's `conflicts_with` guarantees the flags are mutually exclusive.
    let subs = match (&opts.subtitle, opts.subtitle_track) {
        (Some(path), _) => pipeline::SubtitleSource::File(path),
        (None, Some(track)) => pipeline::SubtitleSource::Track(track),
        (None, None) => pipeline::SubtitleSource::Auto,
    };

    let source = Source::classify(&io.input);

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
                &source,
                sink,
                &cfg,
                threads,
                in_flight,
                subs,
                window,
                &StderrEvents,
            )
        }
        Command::Extract { range, .. } => {
            pipeline::extract_range(&source, sink, &cfg, *range, subs, &StderrEvents)
        }
        Command::Mcp
        | Command::InstallWhisper { .. }
        | Command::Cookie { .. }
        | Command::ProbeGpu { .. } => {
            unreachable!("handled above")
        }
    }
}
