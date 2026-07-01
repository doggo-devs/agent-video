//! Pipeline orchestration: extract → dedup → chunk → (per-chunk) score → select
//! → layout → compose, then write grids, `metadata.json`, and subtitles.
//!
//! Rank normalization and selection are deliberately per-chunk: scores are only
//! meaningful relative to the frames they compete with. A chunk is at least
//! `chunk_seconds` of video, and stretches further when sparse chunks merge
//! forward (see [`state`]) — the unit of competition is "whatever fills one
//! grid".

mod chunk;
mod extract;
mod output;
mod range;
mod state;

use std::fs;
use std::path::Path;

use anyhow::{Context, Result};

use crate::config::Config;
use crate::error::{Error, ErrorCode, ErrorContext};
use crate::probe::{KeyframeInfo, SampleStrategy};
use crate::{extract as extract_frames, probe};

use state::ProcessState;

pub use extract::extract_range;
pub use range::parse_range;

/// Subtitle extraction options.
#[derive(Debug, Clone, Copy, Default)]
pub struct SubtitleOpts<'a> {
    /// Explicit subtitle file path (highest priority).
    pub path: Option<&'a Path>,
    /// Embedded track index (used when path is None).
    pub track: Option<usize>,
}

/// Default max in-flight chunks (running + queued results awaiting emit).
pub const DEFAULT_IN_FLIGHT: usize = 2;

/// Shared context for one `process` run, threaded through the strategy paths.
struct RunCtx<'a> {
    input: &'a Path,
    out_dir: &'a Path,
    cfg: &'a Config,
    mode: OutputMode,
    max_in_flight: usize,
    pool: &'a rayon::ThreadPool,
    subs: SubtitleOpts<'a>,
    /// Container duration from ffprobe, when a probe ran. Authoritative over
    /// the frame-derived estimate: keyframe sampling can stop well short of
    /// the tail, and its `+1/base_fps` padding only fits fixed-fps sampling.
    probed_duration: Option<f32>,
}

/// Output mode: write files or JSON to stdout.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputMode {
    /// Write grid images and metadata.json to disk.
    Files,
    /// Output JSON with embedded base64 images to stdout.
    Json,
}

/// Run the full `process` pipeline.
pub fn process(
    input: &Path,
    out_dir: &Path,
    cfg: &Config,
    mode: OutputMode,
    threads: usize,
    max_in_flight: usize,
    subs: SubtitleOpts<'_>,
) -> Result<()> {
    validate_input(input)?;
    let keyframe_info = probe_if_needed(input, cfg)?;
    validate_strategy(cfg, keyframe_info.as_ref())?;

    let mut builder = rayon::ThreadPoolBuilder::new();
    if threads > 0 {
        builder = builder.num_threads(threads);
    }
    let pool = builder
        .thread_name(|i| format!("vidgrid-{i}"))
        .build()
        .context("building rayon thread pool")?;

    // The orchestrator stays on the calling thread and blocks on the result
    // channel; only worker jobs run on the pool. Running the orchestrator
    // inside the pool would deadlock a single-thread pool: the drain loop
    // would block the one thread the spawned chunk jobs need.
    let ctx = RunCtx {
        input,
        out_dir,
        cfg,
        mode,
        max_in_flight,
        pool: &pool,
        subs,
        probed_duration: keyframe_info.as_ref().map(|i| i.duration),
    };
    match cfg.strategy {
        SampleStrategy::FixedFps => {
            ensure_output_dir(out_dir, mode)?;
            eprintln!("fixed-fps mode: sampling at {:.1} fps", cfg.base_fps);
            let stream = extract_frames::stream_frames(
                input,
                cfg.base_fps,
                cfg.min_frame_size,
                cfg.score_edge,
            )?;
            process_stream(&ctx, stream)
        }
        SampleStrategy::Keyframes => {
            ensure_output_dir(out_dir, mode)?;
            let info = keyframe_info.expect("validated");
            eprintln!(
                "keyframes mode: {} I-frames over {:.1}s (mean gap {:.1}s, max {:.1}s)",
                info.count(),
                info.duration,
                info.mean_gap(),
                info.max_gap()
            );
            let stream = extract_frames::stream_keyframes(
                input,
                info.timestamps,
                cfg.min_frame_size,
                cfg.score_edge,
            )?;
            process_stream(&ctx, stream)
        }
        SampleStrategy::Adaptive => {
            let info = keyframe_info.expect("validated");
            process_adaptive(&ctx, info)
        }
    }
}

fn probe_if_needed(input: &Path, cfg: &Config) -> Result<Option<KeyframeInfo>> {
    match cfg.strategy {
        SampleStrategy::FixedFps if cfg.force_long_video => Ok(None),
        SampleStrategy::FixedFps => match probe::probe_duration(input) {
            Ok(duration) => Ok(Some(KeyframeInfo {
                timestamps: Vec::new(),
                duration,
            })),
            Err(e) => Err(Error::new(
                ErrorCode::FfprobeFailed,
                format!("{e}; cannot verify duration safety"),
            )
            .with_context(ErrorContext::default().with_input(input))
            .into()),
        },
        SampleStrategy::Keyframes | SampleStrategy::Adaptive => {
            if !probe::ffprobe_available() {
                return Err(Error::new(
                    ErrorCode::FfprobeNotFound,
                    format!(
                        "--strategy {:?} requires ffprobe (not found on PATH)",
                        cfg.strategy
                    ),
                )
                .into());
            }
            Ok(Some(probe::probe_keyframes(input)?))
        }
    }
}

fn validate_strategy(cfg: &Config, info: Option<&KeyframeInfo>) -> Result<()> {
    use crate::config::StrategyOpts;

    match cfg.strategy {
        SampleStrategy::FixedFps => {
            if let Some(i) = info.filter(|i| i.duration > StrategyOpts::MAX_DURATION_DEFAULT)
                && !cfg.force_long_video
            {
                let hours = i.duration / 3600.0;
                return Err(Error::new(
                    ErrorCode::InputTooLong,
                    format!(
                        "video is {hours:.1}h (>{:.0}s limit for fixed-fps); \
                         use --force or --strategy keyframes/adaptive",
                        StrategyOpts::MAX_DURATION_DEFAULT
                    ),
                )
                .into());
            }
        }
        SampleStrategy::Keyframes | SampleStrategy::Adaptive => {
            let i = info.expect("probe should succeed for keyframes/adaptive");
            if i.count() < cfg.min_keyframes {
                return Err(Error::new(
                    ErrorCode::InsufficientKeyframes,
                    format!(
                        "video has only {} keyframes (<{} required); \
                         lower --min-keyframes or use --strategy fixed-fps",
                        i.count(),
                        cfg.min_keyframes
                    ),
                )
                .into());
            }
        }
    }
    Ok(())
}

/// Reject a missing, unreadable, or non-file input before any external tool
/// runs, so the harness gets a precise 2xx code instead of a generic ffprobe
/// failure wrapping the real cause.
fn validate_input(input: &Path) -> Result<()> {
    let meta = fs::metadata(input).map_err(|e| {
        let code = match e.kind() {
            std::io::ErrorKind::NotFound => ErrorCode::InputNotFound,
            _ => ErrorCode::InputUnreadable,
        };
        Error::new(code, format!("cannot open input {}", input.display()))
            .with_context(ErrorContext::default().with_input(input))
            .with_source(e)
    })?;
    if meta.is_dir() {
        return Err(Error::new(
            ErrorCode::InputUnreadable,
            format!("input {} is a directory, not a video file", input.display()),
        )
        .with_context(ErrorContext::default().with_input(input))
        .into());
    }
    Ok(())
}

fn ensure_output_dir(out_dir: &Path, mode: OutputMode) -> Result<()> {
    if mode == OutputMode::Files {
        fs::create_dir_all(out_dir).map_err(|e| {
            Error::new(
                ErrorCode::OutputDirCreate,
                format!("creating output dir {}: {e}", out_dir.display()),
            )
            .with_context(ErrorContext::default().with_output(out_dir))
            .with_source(e)
        })?;
    }
    Ok(())
}

/// Drain a single frame stream into ProcessState.
fn drain_source(
    stream: &mut extract_frames::FrameStream,
    state: &mut ProcessState<'_>,
    out_dir: &Path,
) -> Result<()> {
    while let Some(frame) = stream.next_frame()? {
        if let Some(aspect) = stream.aspect() {
            state.set_aspect(aspect);
        }
        state.push_rgb(frame, out_dir)?;
    }
    Ok(())
}

/// Drive a single-stream pipeline: drain into state, seal by time chunk, write.
///
/// The caller ensures the output dir, prints the mode-specific progress line,
/// and constructs `stream`; this shared tail runs for both fixed-fps and
/// keyframe modes, which differ only in those caller-side steps.
fn process_stream(ctx: &RunCtx<'_>, mut stream: extract_frames::FrameStream) -> Result<()> {
    let write_files = ctx.mode == OutputMode::Files;
    let mut state = ProcessState::new(ctx.cfg, ctx.pool, write_files, ctx.max_in_flight);

    drain_source(&mut stream, &mut state, ctx.out_dir)?;
    let (duration, grid_metas) = state.finish(ctx.out_dir)?;
    let duration = final_duration(duration, ctx.probed_duration);

    output::write_outputs(
        ctx.input,
        ctx.out_dir,
        ctx.cfg,
        ctx.mode,
        duration,
        grid_metas,
        ctx.subs,
    )
}

fn process_adaptive(ctx: &RunCtx<'_>, info: KeyframeInfo) -> Result<()> {
    let cfg = ctx.cfg;
    ensure_output_dir(ctx.out_dir, ctx.mode)?;

    let sparse_regions = info.sparse_regions(cfg.gap_threshold);
    eprintln!(
        "adaptive mode: {} I-frames (max gap {:.1}s), {} sparse regions (threshold {:.1}s)",
        info.count(),
        info.max_gap(),
        sparse_regions.len(),
        cfg.gap_threshold
    );

    let write_files = ctx.mode == OutputMode::Files;
    let mut state = ProcessState::new(cfg, ctx.pool, write_files, ctx.max_in_flight);

    let mut stream = extract_frames::stream_keyframes(
        ctx.input,
        info.timestamps,
        cfg.min_frame_size,
        cfg.score_edge,
    )?;

    // Interleave sparse-region fills into the keyframe pass in time order:
    // before pushing a keyframe at `ts`, drain every region that ends at or
    // before `ts`. Regions ascending + non-overlapping and fill timestamps
    // strictly below their region end make the merged arrival monotonic, so
    // ProcessState's high-water-mark sealing and static detection stay valid.
    let mut regions = sparse_regions.into_iter().peekable();
    while let Some(frame) = stream.next_frame()? {
        if let Some(aspect) = stream.aspect() {
            state.set_aspect(aspect);
        }
        for (start, end) in due_regions(&mut regions, frame.ts) {
            fill_region(ctx, &mut state, start, end)?;
        }
        state.push_rgb(frame, ctx.out_dir)?;
    }
    for (start, end) in regions {
        fill_region(ctx, &mut state, start, end)?;
    }

    let (duration, grid_metas) = state.finish(ctx.out_dir)?;
    let duration = final_duration(duration, ctx.probed_duration);

    output::write_outputs(
        ctx.input,
        ctx.out_dir,
        cfg,
        ctx.mode,
        duration,
        grid_metas,
        ctx.subs,
    )
}

/// Duration reported in `metadata.json`: the probed container duration when
/// available, floored by the frame-derived estimate in case the container
/// header under-reports what was actually decoded.
fn final_duration(observed: f32, probed: Option<f32>) -> f32 {
    observed.max(probed.unwrap_or(0.0))
}

/// Pop every region whose `end <= ts` off the front of `regions`, in order.
///
/// The regions are ascending and non-overlapping, so this yields exactly the
/// fills that must be drained before a main-stream frame at `ts` to keep
/// arrival order monotonic.
fn due_regions(
    regions: &mut std::iter::Peekable<std::vec::IntoIter<(f32, f32)>>,
    ts: f32,
) -> Vec<(f32, f32)> {
    let mut due = Vec::new();
    while regions.peek().is_some_and(|&(_, end)| ts >= end) {
        due.push(regions.next().expect("peeked region exists"));
    }
    due
}

/// Drain one sparse region into `state` via a fixed-fps sub-stream.
fn fill_region(ctx: &RunCtx<'_>, state: &mut ProcessState<'_>, start: f32, end: f32) -> Result<()> {
    eprintln!("  filling sparse region {start:.1}s-{end:.1}s");
    let mut fill_stream = extract_frames::stream_range(
        ctx.input,
        ctx.cfg.base_fps,
        ctx.cfg.min_frame_size,
        ctx.cfg.score_edge,
        start,
        end,
    )?;
    drain_source(&mut fill_stream, state, ctx.out_dir)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_missing_input_with_not_found_code() {
        let err = validate_input(Path::new("definitely_missing_video.mp4"))
            .expect_err("missing input must fail");
        let code = err.downcast_ref::<Error>().map(|e| e.code);
        assert_eq!(code, Some(ErrorCode::InputNotFound));
    }

    #[test]
    fn rejects_directory_input_as_unreadable() {
        let err = validate_input(Path::new(".")).expect_err("directory input must fail");
        let code = err.downcast_ref::<Error>().map(|e| e.code);
        assert_eq!(code, Some(ErrorCode::InputUnreadable));
    }

    #[test]
    fn prefers_probed_duration_over_frame_estimate() {
        assert_eq!(
            final_duration(3500.5, Some(3600.0)),
            3600.0,
            "probe extends past the last sampled frame"
        );
        assert_eq!(
            final_duration(3600.0, Some(3500.0)),
            3600.0,
            "observed frames floor an under-reporting header"
        );
        assert_eq!(final_duration(120.0, None), 120.0, "no probe, no change");
    }

    #[test]
    fn fills_regions_before_first_frame_past_their_end() {
        // A frame at ts=4 is at the boundary of the head region (0,4): that
        // region is due; the next region (6,12) is not until a later frame.
        let mut regions = vec![(0.0, 4.0), (6.0, 12.0)].into_iter().peekable();
        assert_eq!(due_regions(&mut regions, 4.0), vec![(0.0, 4.0)]);
        assert_eq!(due_regions(&mut regions, 6.0), Vec::new(), "end 12 > 6");
        assert_eq!(due_regions(&mut regions, 20.0), vec![(6.0, 12.0)]);
    }

    #[test]
    fn fills_trailing_regions_after_eof() {
        // Frames only reach ts=10, so a tail region ending at 30 is never due
        // during the pass and remains to be drained after EOF.
        let mut regions = vec![(0.0, 4.0), (20.0, 30.0)].into_iter().peekable();
        assert_eq!(due_regions(&mut regions, 10.0), vec![(0.0, 4.0)]);
        let trailing: Vec<_> = regions.collect();
        assert_eq!(trailing, vec![(20.0, 30.0)]);
    }
}
