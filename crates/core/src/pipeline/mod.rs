//! Pipeline orchestration: extract → dedup → chunk → (per-chunk) score → select
//! → layout → compose, then write grids, `metadata.json`, and subtitles.
//!
//! Rank normalization and selection are deliberately per-chunk: scores are only
//! meaningful relative to the frames they compete with. A chunk is at least
//! `chunk_seconds` of video, and stretches further when sparse chunks merge
//! forward (see [`state`]) — the unit of competition is "whatever fills one
//! grid".

mod candidates;
mod chunk;
mod extract;
mod jobs;
mod output;
mod range;
mod state;
mod stretch;

use std::fs;
use std::num::NonZeroUsize;
use std::path::Path;

use anyhow::{Context, Result};

use crate::config::Config;
use crate::error::{Error, ErrorCode, ErrorContext};
use crate::probe::{KeyframeInfo, MonotonicTimestamps, SampleStrategy};
use crate::subtitle::Subtitles;
use crate::types::{GridMeta, TimeRange};
use crate::{dedup, extract as extract_frames, probe};

use candidates::CandidateStream;
use output::OutputSession;
use state::{ProcessState, extract_batch_frames};

pub use crate::subtitle::SubtitleSource;
pub use extract::extract_range;
pub use range::{Window, parse_range};

/// Default max in-flight chunks (running + queued results awaiting emit).
pub const DEFAULT_IN_FLIGHT: NonZeroUsize = NonZeroUsize::new(2).unwrap();

/// Shared context for one `process` run, threaded through the strategy paths.
struct RunCtx<'a> {
    input: &'a Path,
    cfg: &'a Config,
    max_in_flight: NonZeroUsize,
    pool: &'a rayon::ThreadPool,
    /// Decoder thread cap (`None` = auto-detect). Derived from `--threads`,
    /// which otherwise only bounds the rayon pool while the video decoder
    /// schedules its own threads outside Rust.
    ffmpeg_threads: Option<NonZeroUsize>,
    /// Subtitles resolved before any frame work: subtitle errors (ambiguous
    /// sources, unreadable files) must fail the run before grids are written,
    /// never after.
    subs: Option<Subtitles>,
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
#[allow(clippy::too_many_arguments)]
pub fn process(
    input: &Path,
    out_dir: &Path,
    cfg: &Config,
    mode: OutputMode,
    threads: usize,
    max_in_flight: NonZeroUsize,
    subs: SubtitleSource<'_>,
    window: Option<Window>,
) -> Result<()> {
    validate_input(input)?;
    // Keyframe strategies pair decoded frames with probed timestamps by
    // index; a window seek would decode a different frame set and silently
    // shift every pairing, so the clamp is fixed-fps only.
    if window.is_some() && cfg.strategy() != SampleStrategy::FixedFps {
        return Err(Error::new(
            ErrorCode::InvalidRange,
            "--from/--until require the fixed-fps strategy; drop --strategy or the window",
        )
        .into());
    }
    let keyframe_info = probe_if_needed(input, cfg)?;
    validate_strategy(cfg, keyframe_info.as_ref(), window)?;
    // Resolve subtitles before any extraction or output: whether the final
    // outputs can be assembled must be settled before side effects begin, so
    // a subtitle failure cannot leave grids on disk without metadata.
    let subs = Subtitles::resolve(input, subs)?;

    let mut builder = rayon::ThreadPoolBuilder::new();
    if threads > 0 {
        builder = builder.num_threads(threads);
    }
    let pool = builder
        .thread_name(|i| format!("agent-video-{i}"))
        .build()
        .context("building rayon thread pool")?;

    // The orchestrator stays on the calling thread and blocks on the result
    // channel; only worker jobs run on the pool. Running the orchestrator
    // inside the pool would deadlock a single-thread pool: the drain loop
    // would block the one thread the spawned chunk jobs need.
    let ctx = RunCtx {
        input,
        cfg,
        max_in_flight,
        pool: &pool,
        ffmpeg_threads: NonZeroUsize::new(threads),
        subs,
        probed_duration: keyframe_info.as_ref().map(|i| i.duration),
    };
    // Every strategy reduces to a sampling plan executed by one decode pass;
    // the pipeline tail is strategy-agnostic.
    let session = OutputSession::open(out_dir, mode)?;
    let stream = match cfg.strategy() {
        SampleStrategy::FixedFps => match window {
            Some(w) => {
                eprintln!(
                    "fixed-fps mode: sampling at {:.1} fps over {:.1}s-{}",
                    cfg.base_fps(),
                    w.start,
                    w.end.map_or("end".into(), |e| format!("{e:.1}s"))
                );
                extract_frames::stream_window(
                    input,
                    cfg.base_fps(),
                    cfg.min_frame_size(),
                    cfg.score_edge(),
                    w.start,
                    w.end,
                    ctx.ffmpeg_threads,
                )?
            }
            None => {
                eprintln!("fixed-fps mode: sampling at {:.1} fps", cfg.base_fps());
                extract_frames::stream_frames(
                    input,
                    cfg.base_fps(),
                    cfg.min_frame_size(),
                    cfg.score_edge(),
                    ctx.ffmpeg_threads,
                )?
            }
        },
        SampleStrategy::Keyframes => {
            let info = keyframe_info.expect("validated");
            eprintln!(
                "keyframes mode: {} I-frames over {:.1}s (mean gap {:.1}s, max {:.1}s)",
                info.count(),
                info.duration,
                info.mean_gap(),
                info.max_gap()
            );
            extract_frames::stream_keyframes(
                input,
                cfg.min_frame_size(),
                cfg.score_edge(),
                ctx.ffmpeg_threads,
            )?
        }
        SampleStrategy::Adaptive => {
            let info = keyframe_info.expect("validated");
            adaptive_stream(&ctx, &info)?
        }
    };
    process_stream(&ctx, session, stream)
}

/// Build the adaptive sampling stream: keyframes everywhere, fixed-fps fill
/// inside sparse regions, all from a single decode pass ([`SampleSpec::Mixed`]
/// executes the plan; no per-region sub-streams to interleave).
///
/// [`SampleSpec::Mixed`]: crate::media::decode::SampleSpec::Mixed
fn adaptive_stream(ctx: &RunCtx<'_>, info: &KeyframeInfo) -> Result<extract_frames::FrameStream> {
    let cfg = ctx.cfg;
    let fill = merge_regions(
        info.sparse_regions(cfg.gap_threshold()),
        cfg.gap_threshold(),
    );
    eprintln!(
        "adaptive mode: {} I-frames (max gap {:.1}s), {} sparse regions (threshold {:.1}s)",
        info.count(),
        info.max_gap(),
        fill.len(),
        cfg.gap_threshold()
    );

    // When sparse regions dominate the timeline, the mixed plan decodes
    // nearly every packet anyway and its keyframe pass contributes little; a
    // plain fixed-fps stream covers the same ground with uniform sampling.
    let covered: f32 = fill.iter().map(TimeRange::duration).sum();
    if covered > 0.5 * info.duration {
        eprintln!(
            "adaptive mode: sparse regions cover {:.0}% of the video; \
             falling back to a single fixed-fps pass",
            100.0 * covered / info.duration
        );
        return Ok(extract_frames::stream_frames(
            ctx.input,
            cfg.base_fps(),
            cfg.min_frame_size(),
            cfg.score_edge(),
            ctx.ffmpeg_threads,
        )?);
    }

    for region in &fill {
        eprintln!(
            "  filling sparse region {:.1}s-{:.1}s",
            region.start(),
            region.end()
        );
    }
    Ok(extract_frames::stream_mixed(
        ctx.input,
        cfg.base_fps(),
        fill,
        cfg.min_frame_size(),
        cfg.score_edge(),
        ctx.ffmpeg_threads,
    )?)
}

fn probe_if_needed(input: &Path, cfg: &Config) -> Result<Option<KeyframeInfo>> {
    match cfg.strategy() {
        SampleStrategy::FixedFps if cfg.force_long_video() => Ok(None),
        // `.context` keeps the probe's own error code downcastable, so the
        // exit code names the real cause rather than a generic wrapper.
        SampleStrategy::FixedFps => {
            let duration = probe::probe_duration(input)
                .context("cannot verify duration safety for fixed-fps sampling (use --force to skip this check)")?;
            Ok(Some(KeyframeInfo {
                timestamps: MonotonicTimestamps::default(),
                duration,
            }))
        }
        SampleStrategy::Keyframes | SampleStrategy::Adaptive => {
            Ok(Some(probe::probe_keyframes(input)?))
        }
    }
}

fn validate_strategy(
    cfg: &Config,
    info: Option<&KeyframeInfo>,
    window: Option<Window>,
) -> Result<()> {
    match cfg.strategy() {
        SampleStrategy::FixedFps => {
            let Some(i) = info else { return Ok(()) };
            if let Some(w) = window
                && w.start >= i.duration
            {
                return Err(Error::new(
                    ErrorCode::InvalidRange,
                    format!(
                        "--from ({:.1}s) is at or past the end of the video ({:.1}s)",
                        w.start, i.duration
                    ),
                )
                .into());
            }
            // The decode-cost guard applies to what is actually decoded: the
            // window span when clamped, the whole video otherwise.
            let span = window.map_or(i.duration, |w| w.effective_span(i.duration));
            if span > cfg.max_duration() && !cfg.force_long_video() {
                let hours = span / 3600.0;
                let what = if window.is_some() {
                    "window covers"
                } else {
                    "video is"
                };
                return Err(Error::new(
                    ErrorCode::InputTooLong,
                    format!(
                        "{what} {hours:.1}h (>{:.0}s limit for fixed-fps); \
                         use --force, --max-duration, --strategy keyframes/adaptive, \
                         or a narrower --from/--until",
                        cfg.max_duration()
                    ),
                )
                .into());
            }
        }
        SampleStrategy::Keyframes | SampleStrategy::Adaptive => {
            let i = info.expect("probe should succeed for keyframes/adaptive");
            if i.count() < cfg.min_keyframes() {
                return Err(Error::new(
                    ErrorCode::InsufficientKeyframes,
                    format!(
                        "video has only {} keyframes (<{} required); \
                         lower --min-keyframes or use --strategy fixed-fps",
                        i.count(),
                        cfg.min_keyframes()
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
pub(crate) fn validate_input(input: &Path) -> Result<()> {
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

/// Drive a single-stream pipeline: judge candidates in batches, drain them
/// into state, seal by time chunk, write.
///
/// The caller opens the session, prints the mode-specific progress line, and
/// constructs `stream`; this shared tail runs for every strategy — they
/// differ only in those caller-side steps.
fn process_stream(
    ctx: &RunCtx<'_>,
    session: OutputSession<'_>,
    stream: extract_frames::FrameStream,
) -> Result<()> {
    let cfg = ctx.cfg;
    let mut stream = CandidateStream::new(
        stream,
        dedup::Deduper::new(cfg.hash_threshold(), cfg.dedup_window()),
        extract_batch_frames(cfg.base_fps(), cfg.chunk_seconds()),
        cfg.score_edge(),
        Some(ctx.pool),
    );
    let mut state = ProcessState::new(cfg, ctx.pool, &session, ctx.max_in_flight);

    while let Some(batch) = stream.next_batch()? {
        if let Some(aspect) = stream.aspect() {
            state.set_aspect(aspect);
        }
        for (frame, kept) in batch {
            state.push_judged(frame, kept)?;
        }
    }
    let (duration, grid_metas) = state.finish()?;
    reject_empty_run(ctx.input, &grid_metas)?;
    let duration = final_duration(duration, ctx.probed_duration);

    output::write_outputs(
        session,
        ctx.input,
        ctx.cfg,
        duration,
        grid_metas,
        ctx.subs.as_ref(),
    )
}

/// Reject a run that produced no grids before any metadata is written.
///
/// `extract_range` already fails an empty selection with
/// [`ErrorCode::NoFramesExtracted`]; `process` must honor the same contract —
/// README promises "one or more grid images", and a successful exit with
/// `grids: []` (e.g. ffmpeg exited 0 without emitting a single PPM frame)
/// would hand callers a fabricated success.
fn reject_empty_run(input: &Path, metas: &[GridMeta]) -> Result<()> {
    if metas.is_empty() {
        return Err(Error::new(
            ErrorCode::NoFramesExtracted,
            "no frames decoded from the video stream; is the input empty or corrupt?",
        )
        .with_context(ErrorContext::default().with_input(input))
        .into());
    }
    Ok(())
}

/// Duration reported in `metadata.json`: the probed container duration when
/// available, floored by the frame-derived estimate in case the container
/// header under-reports what was actually decoded.
fn final_duration(observed: f32, probed: Option<f32>) -> f32 {
    observed.max(probed.unwrap_or(0.0))
}

/// Merge sparse regions separated by a dense run shorter than `bridge`.
///
/// Absorbing a short dense run costs at most `bridge` seconds of extra decode
/// at base_fps and buys a simpler fill plan with fewer tick restarts. Input
/// regions are ascending and non-overlapping (see
/// [`KeyframeInfo::sparse_regions`]); output preserves both invariants.
fn merge_regions(regions: Vec<TimeRange>, bridge: f32) -> Vec<TimeRange> {
    let mut merged: Vec<TimeRange> = Vec::with_capacity(regions.len());
    for region in regions {
        match merged.last_mut() {
            Some(last) if region.start() - last.end() <= bridge => {
                *last = TimeRange::new(last.start(), region.end())
                    .expect("ascending regions extend forward");
            }
            _ => merged.push(region),
        }
    }
    merged
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

    fn tr(start: f32, end: f32) -> TimeRange {
        TimeRange::new(start, end).expect("valid test range")
    }

    #[test]
    fn rejects_empty_run_with_no_frames_code() {
        let err = reject_empty_run(Path::new("video.mp4"), &[])
            .expect_err("zero grids must fail, not write grids: []");
        let code = err.downcast_ref::<Error>().map(|e| e.code);
        assert_eq!(code, Some(ErrorCode::NoFramesExtracted));
    }

    #[test]
    fn merges_regions_bridged_by_short_dense_runs() {
        // (0,10) and (12,20) are 2s apart (<= bridge 5): one ffmpeg spawn
        // covering both is cheaper than two. (40,50) stays separate.
        let regions = vec![tr(0.0, 10.0), tr(12.0, 20.0), tr(40.0, 50.0)];
        assert_eq!(
            merge_regions(regions, 5.0),
            vec![tr(0.0, 20.0), tr(40.0, 50.0)]
        );
    }

    #[test]
    fn keeps_regions_separated_by_long_dense_runs() {
        let regions = vec![tr(0.0, 10.0), tr(30.0, 40.0)];
        assert_eq!(merge_regions(regions.clone(), 5.0), regions);
    }
}
