// SPDX-License-Identifier: Apache-2.0
//! Pipeline orchestration: sample → dedup → chunk → (per-chunk) score → select
//! → layout → compose, then write grids, `metadata.json`, and subtitles.
//!
//! Rank normalization and selection are deliberately per-chunk: scores are only
//! meaningful relative to the frames they compete with. A chunk is at least
//! `chunk_seconds` of video, and stretches further when sparse chunks merge
//! forward (see `state`) — the unit of competition is "whatever fills one
//! grid".

mod candidates;
mod chunk;
mod chunker;
mod extract;
mod jobs;
mod output;
mod range;
mod state;
mod stretch;

use std::fs;
use std::num::NonZeroUsize;
use std::path::Path;

use crate::config::Config;
use crate::error::{Context as _, Error, ErrorCode, ErrorContext, Result};
use crate::event::{Event, EventSink};
use crate::media;
use crate::probe::{KeyframeInfo, SampleStrategy};
use crate::source::Source;
use crate::subtitle::Subtitles;
use crate::types::{GridMeta, TimeRange};
use crate::{dedup, probe, sample};

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
    input: &'a Source,
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
    /// Diagnostic sink for the run: mode announcements, sparse-region fill
    /// lines, and decode-open warnings flow here as typed events.
    events: &'a dyn EventSink,
}

impl<'a> RunCtx<'a> {
    /// The extractor knobs every sampling shape shares, derived once from
    /// the run's config and thread cap.
    fn stream_params(&self) -> sample::StreamParams<'a> {
        sample::StreamParams {
            cell_min_edge: self.cfg.min_frame_size(),
            score_edge: self.cfg.score_edge(),
            threads: self.ffmpeg_threads,
            events: self.events,
        }
    }
}

/// Where a run's artifacts land. The output directory exists only in the
/// files variant, so "JSON mode plus a dummy directory" is unrepresentable
/// — no caller ever has to invent a path it promises not to use.
#[derive(Debug, Clone, Copy)]
pub enum OutputSink<'a> {
    /// Grid images and `metadata.json` under this directory.
    Files(&'a Path),
    /// One JSON document with base64-inline images on stdout; the
    /// filesystem is never touched.
    JsonStdout,
}

/// Strategy resolved and validated against the probed input; keyframe
/// data rides inside the variant that needs it, so "keyframe strategy
/// without probe data" is unrepresentable. Every value is already valid:
/// [`resolve_plan`] is the only constructor, and it folds probing and
/// validation in before returning, so no downstream code has to re-check
/// the pairing invariant.
enum ResolvedPlan {
    /// `probed_duration` is `None` exactly when `--force` skipped the
    /// duration probe (see the fixed-fps arm of [`resolve_plan`]).
    FixedFps {
        probed_duration: Option<f32>,
    },
    Keyframes(KeyframeInfo),
    Adaptive(KeyframeInfo),
}

impl ResolvedPlan {
    /// Container duration from the probe, when one ran. `None` only for a
    /// forced fixed-fps run that skipped the probe.
    fn probed_duration(&self) -> Option<f32> {
        match self {
            ResolvedPlan::FixedFps { probed_duration } => *probed_duration,
            ResolvedPlan::Keyframes(info) | ResolvedPlan::Adaptive(info) => Some(info.duration),
        }
    }
}

/// Run knobs for [`process`] beyond the input, the output sink, and the
/// frame config: parallelism, subtitles, the attention window, and the
/// diagnostic sink. The subset of the run's internal context a caller
/// decides.
pub struct ProcessOpts<'a> {
    /// Rayon pool size and decoder thread cap; `0` = auto-detect.
    pub threads: usize,
    /// Max in-flight chunks (running + queued results awaiting emit).
    pub max_in_flight: NonZeroUsize,
    /// Subtitle selection.
    pub subs: SubtitleSource<'a>,
    /// Attention window (`--from`/`--until`); `None` covers the whole video.
    pub window: Option<Window>,
    /// Stall guard for the run's probe and subtitle-extraction reads: a
    /// stalled source aborts with an error instead of hanging when the
    /// token trips. The decode thread self-mints its own token (its Drop
    /// owns the trip right — see [`crate::media::decode::FrameSource::open`]);
    /// a caller wiring ctrl-C must bridge the two.
    pub stop: media::Stop,
    /// Receives every diagnostic the run emits (mode announcements,
    /// sparse-region fill lines, decode-open and subtitle warnings) as
    /// typed [`Event`]s; pass [`crate::event::Discard`] for silence.
    pub events: &'a dyn EventSink,
}

/// Run the full `process` pipeline.
///
/// Every failure is a coded [`Error`]: the exit-code contract is stated by
/// the signature, not recovered by downcasting.
pub fn process(
    input: &Source,
    sink: OutputSink<'_>,
    cfg: &Config,
    opts: ProcessOpts<'_>,
) -> Result<()> {
    let ProcessOpts {
        threads,
        max_in_flight,
        subs,
        window,
        stop,
        events,
    } = opts;
    validate_input(input)?;
    // One door: window resolution, probe, keyframe-poor degradation, and
    // validation fold into a single plan. Two adjustments can move a
    // defaulted keyframe/adaptive request onto fixed-fps: a window (pairing
    // can't survive a seek) and a keyframe-poor source. An explicit
    // --strategy is honored and validated instead.
    let plan = resolve_plan(input, cfg, window, &stop, events)?;
    // Before any side effect: a subtitle failure must not leave grids on
    // disk without metadata.
    let subs = Subtitles::resolve(input, subs, &stop, events)?;

    let mut builder = rayon::ThreadPoolBuilder::new();
    if threads > 0 {
        builder = builder.num_threads(threads);
    }
    let pool = builder
        .thread_name(|i| format!("agent-video-{i}"))
        .build()
        .map_err(|e| {
            // The pool size is configuration (`--threads`); a pool the OS
            // cannot provide is a config problem, not a bug.
            Error::new(
                ErrorCode::InvalidConfig,
                format!("building rayon thread pool: {e}"),
            )
            .with_source(e)
        })?;

    // The orchestrator stays on the calling thread; only worker jobs run on
    // the pool. Inside the pool it would deadlock a single-thread pool — the
    // drain loop would block the one thread the chunk jobs need.
    let ctx = RunCtx {
        input,
        cfg,
        max_in_flight,
        pool: &pool,
        ffmpeg_threads: NonZeroUsize::new(threads),
        subs,
        probed_duration: plan.probed_duration(),
        events,
    };
    // Every strategy reduces to one decode pass; the tail is shared.
    let session = OutputSession::open(sink)?;
    let stream = match plan {
        ResolvedPlan::FixedFps { .. } => match window {
            Some(w) => {
                events.event(Event::FixedFpsWindow {
                    fps: cfg.base_fps(),
                    start: w.start(),
                    end: w.end(),
                });
                sample::stream_window(
                    input,
                    cfg.base_fps(),
                    w.start(),
                    w.end(),
                    ctx.stream_params(),
                )?
            }
            None => {
                events.event(Event::FixedFps {
                    fps: cfg.base_fps(),
                });
                sample::stream_frames(input, cfg.base_fps(), ctx.stream_params())?
            }
        },
        ResolvedPlan::Keyframes(info) => {
            events.event(Event::KeyframesMode {
                count: info.count(),
                duration: info.duration,
                mean_gap: info.mean_gap(),
                max_gap: info.max_gap(),
            });
            sample::stream_keyframes(input, ctx.stream_params())?
        }
        ResolvedPlan::Adaptive(info) => adaptive_stream(&ctx, &info)?,
    };
    process_stream(&ctx, session, stream, window)
}

/// Build the adaptive sampling stream: keyframes everywhere, fixed-fps fill
/// inside sparse regions, all from a single decode pass ([`SampleSpec::Mixed`]
/// executes the plan; no per-region sub-streams to interleave).
///
/// [`SampleSpec::Mixed`]: crate::media::decode::SampleSpec::Mixed
fn adaptive_stream(ctx: &RunCtx<'_>, info: &KeyframeInfo) -> Result<sample::FrameStream> {
    let cfg = ctx.cfg;
    let fill = merge_regions(
        info.sparse_regions(cfg.gap_threshold()),
        cfg.gap_threshold(),
    );
    ctx.events.event(Event::AdaptiveMode {
        keyframes: info.count(),
        max_gap: info.max_gap(),
        sparse_regions: fill.len(),
        threshold: cfg.gap_threshold(),
    });

    // When sparse regions dominate, the mixed plan decodes nearly every
    // packet anyway; a plain fixed-fps pass covers the same ground.
    let covered: f32 = fill.iter().map(TimeRange::duration).sum();
    if covered > 0.5 * info.duration {
        ctx.events.event(Event::AdaptiveFallbackFixedFps {
            coverage_percent: 100.0 * covered / info.duration,
        });
        return sample::stream_frames(ctx.input, cfg.base_fps(), ctx.stream_params());
    }

    for region in &fill {
        ctx.events.event(Event::FillingSparseRegion {
            start: region.start(),
            end: region.end(),
        });
    }
    sample::stream_mixed(ctx.input, cfg.base_fps(), fill, ctx.stream_params())
}

/// Resolve the strategy against a window (`--from/--until`).
///
/// A window is fixed-fps only: keyframe strategies pair decoded frames with
/// probed timestamps by index, and a seek shifts every pairing. An explicit
/// keyframe/adaptive request with a window is a user error; a defaulted one
/// silently uses fixed-fps.
fn resolve_windowed_strategy(cfg: &Config, window: Option<Window>) -> Result<SampleStrategy> {
    let requested = cfg.strategy();
    if window.is_none() || requested == SampleStrategy::FixedFps {
        return Ok(requested);
    }
    if cfg.strategy_explicit() {
        return Err(Error::new(
            ErrorCode::InvalidRange,
            "--from/--until require the fixed-fps strategy; drop --strategy or the window",
        ));
    }
    Ok(SampleStrategy::FixedFps)
}

/// The single door onto a validated [`ResolvedPlan`]: resolve the strategy
/// against the window, probe the input, degrade a keyframe-poor default to
/// fixed-fps, and validate — folded so every returned plan is already valid
/// and no caller has to re-establish the "keyframe strategy implies probe
/// data" invariant.
fn resolve_plan(
    input: &Source,
    cfg: &Config,
    window: Option<Window>,
    stop: &media::Stop,
    events: &dyn EventSink,
) -> Result<ResolvedPlan> {
    let strategy = resolve_windowed_strategy(cfg, window)?;
    match strategy {
        SampleStrategy::FixedFps if cfg.force_long_video() => Ok(ResolvedPlan::FixedFps {
            probed_duration: None,
        }),
        // The typed `.context` decorates the reason and keeps the code, so
        // the exit code names the real cause rather than a generic wrapper.
        SampleStrategy::FixedFps => {
            let duration = probe::probe_duration(input, stop)
                .context("cannot verify duration safety for fixed-fps sampling (use --force to skip this check)")?;
            let probed_duration = Some(duration);
            validate_fixed_fps(cfg, probed_duration, window)?;
            Ok(ResolvedPlan::FixedFps { probed_duration })
        }
        SampleStrategy::Keyframes | SampleStrategy::Adaptive => {
            let info = probe::probe_keyframes(input, stop)?;
            // A defaulted keyframe/adaptive request degrades to fixed-fps for a
            // keyframe-poor source; its probed duration still carries forward.
            if degrade_if_keyframe_poor(cfg, strategy, &info, events) == SampleStrategy::FixedFps {
                let probed_duration = Some(info.duration);
                validate_fixed_fps(cfg, probed_duration, window)?;
                return Ok(ResolvedPlan::FixedFps { probed_duration });
            }
            validate_keyframe_sufficiency(cfg, &info)?;
            Ok(match strategy {
                SampleStrategy::Keyframes => ResolvedPlan::Keyframes(info),
                _ => ResolvedPlan::Adaptive(info),
            })
        }
    }
}

/// Degrade a *defaulted* keyframe/adaptive strategy to fixed-fps when the
/// source has fewer than `min_keyframes`, so a bare `process` always produces
/// output regardless of keyframe density. An explicit request is left intact
/// for [`validate_keyframe_sufficiency`] to reject with `InsufficientKeyframes`.
/// `events` receives a [`Event::KeyframePoorDegrade`] on the degrade path.
fn degrade_if_keyframe_poor(
    cfg: &Config,
    strategy: SampleStrategy,
    info: &KeyframeInfo,
    events: &dyn EventSink,
) -> SampleStrategy {
    if cfg.strategy_explicit()
        || !matches!(
            strategy,
            SampleStrategy::Keyframes | SampleStrategy::Adaptive
        )
    {
        return strategy;
    }
    let count = info.count();
    if count >= cfg.min_keyframes() {
        return strategy;
    }
    events.event(Event::KeyframePoorDegrade {
        count,
        minimum: cfg.min_keyframes(),
    });
    SampleStrategy::FixedFps
}

/// Validate a fixed-fps plan against its probed duration and any window.
///
/// `probed_duration` is `None` only for a forced run that skipped the probe,
/// in which case there is nothing to check.
fn validate_fixed_fps(
    cfg: &Config,
    probed_duration: Option<f32>,
    window: Option<Window>,
) -> Result<()> {
    let Some(duration) = probed_duration else {
        return Ok(());
    };
    if let Some(w) = window
        && w.start() >= duration
    {
        return Err(Error::new(
            ErrorCode::InvalidRange,
            format!(
                "--from ({:.1}s) is at or past the end of the video ({:.1}s)",
                w.start(),
                duration
            ),
        ));
    }
    // The decode-cost guard applies to what is actually decoded: the
    // window span when clamped, the whole video otherwise.
    let span = window.map_or(duration, |w| w.effective_span(duration));
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
        ));
    }
    Ok(())
}

/// Reject a keyframe/adaptive plan whose source has fewer than
/// `min_keyframes`. Reached only for an explicit request; a defaulted one has
/// already degraded to fixed-fps by this point.
fn validate_keyframe_sufficiency(cfg: &Config, info: &KeyframeInfo) -> Result<()> {
    if info.count() < cfg.min_keyframes() {
        return Err(Error::new(
            ErrorCode::InsufficientKeyframes,
            format!(
                "video has only {} keyframes (<{} required); \
                 lower --min-keyframes or use --strategy fixed-fps",
                info.count(),
                cfg.min_keyframes()
            ),
        ));
    }
    Ok(())
}

/// Reject a missing, unreadable, or non-file local input before any media
/// work runs, so the harness gets a precise 2xx code. A remote source skips
/// this entirely: it has no filesystem entry to stat, and
/// [`crate::media::open_input`] reports an unreachable host, HTTP error, or
/// unsupported protocol as its own coded error.
pub(crate) fn validate_input(input: &Source) -> Result<()> {
    let Source::Local(path) = input else {
        return Ok(());
    };
    let meta = fs::metadata(path).map_err(|e| {
        let code = match e.kind() {
            std::io::ErrorKind::NotFound => ErrorCode::InputNotFound,
            _ => ErrorCode::InputUnreadable,
        };
        Error::new(code, format!("cannot open input {}", path.display()))
            .with_context(ErrorContext::default().with_input(path))
            .with_source(e)
    })?;
    if meta.is_dir() {
        return Err(Error::new(
            ErrorCode::InputUnreadable,
            format!("input {} is a directory, not a video file", path.display()),
        )
        .with_context(ErrorContext::default().with_input(path)));
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
    stream: sample::FrameStream,
    window: Option<Window>,
) -> Result<()> {
    let cfg = ctx.cfg;
    let mut stream = CandidateStream::new(
        stream,
        dedup::Deduper::new(cfg.hash_threshold(), cfg.dedup_window()),
        extract_batch_frames(cfg.base_fps(), cfg.chunk_seconds()),
        cfg.score_edge(),
        Some(ctx.pool),
    );
    let mut state = ProcessState::new(cfg, ctx.pool, &session, ctx.max_in_flight, ctx.events);

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
        output::Timeline {
            duration,
            processed: processed_range(window, duration),
        },
        grid_metas,
        ctx.subs.as_ref(),
        ctx.events,
    )
}

/// The range this run attended to — the subtitle-ownership boundary for
/// [`output::write_outputs`]: the window when one was given (an open end
/// reads to the video's end), else the whole video. The end floors at the
/// start so a window reaching past a short video still forms a range.
fn processed_range(window: Option<Window>, duration: f32) -> TimeRange {
    let (start, end) = match window {
        Some(w) => (w.start(), w.end().unwrap_or(duration).max(w.start())),
        None => (0.0, duration.max(0.0)),
    };
    TimeRange::new(start, end).expect("window bounds are finite and ordered by construction")
}

/// Reject a run that produced no grids before any metadata is written.
///
/// `extract_range` already fails an empty selection with
/// [`ErrorCode::NoFramesExtracted`]; `process` must honor the same contract —
/// README promises "one or more grid images", and a successful exit with
/// `grids: []` (e.g. ffmpeg exited 0 without emitting a single PPM frame)
/// would hand callers a fabricated success.
fn reject_empty_run(input: &Source, metas: &[GridMeta]) -> Result<()> {
    if metas.is_empty() {
        return Err(Error::new(
            ErrorCode::NoFramesExtracted,
            "no frames decoded from the video stream; is the input empty or corrupt?",
        )
        .with_context(ErrorContext::default().with_input(input.as_path())));
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
    use crate::event::Discard;
    use crate::probe::MonotonicTimestamps;

    #[test]
    fn rejects_missing_input_with_not_found_code() {
        let err = validate_input(&Source::classify("definitely_missing_video.mp4"))
            .expect_err("missing input must fail");
        assert_eq!(err.code, ErrorCode::InputNotFound);
    }

    #[test]
    fn rejects_directory_input_as_unreadable() {
        let err = validate_input(&Source::classify(".")).expect_err("directory input must fail");
        assert_eq!(err.code, ErrorCode::InputUnreadable);
    }

    #[test]
    fn skips_filesystem_validation_for_remote_source() {
        // Validation waves it through; open reports protocol/network failures.
        validate_input(&Source::classify("https://example.com/missing.mp4"))
            .expect("remote input skips the filesystem check");
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
        let err = reject_empty_run(&Source::classify("video.mp4"), &[])
            .expect_err("zero grids must fail, not write grids: []");
        assert_eq!(err.code, ErrorCode::NoFramesExtracted);
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

    fn cfg_with_strategy(strategy: Option<SampleStrategy>) -> Config {
        Config::new(
            2048,
            None,
            &crate::config::Overrides::default(),
            crate::config::StrategyOpts {
                strategy,
                ..Default::default()
            },
        )
        .expect("config builds")
    }

    fn keyframes(count: usize) -> KeyframeInfo {
        let timestamps = (0..count).map(|i| i as f32).collect();
        KeyframeInfo {
            timestamps: MonotonicTimestamps::new(timestamps).expect("monotonic"),
            duration: count.max(1) as f32,
        }
    }

    fn window() -> Option<Window> {
        Window::from_args(Some(10.0), None).expect("valid window")
    }

    #[test]
    fn defaults_to_adaptive_leniently() {
        let cfg = cfg_with_strategy(None);
        assert_eq!(cfg.strategy(), SampleStrategy::Adaptive);
        assert!(
            !cfg.strategy_explicit(),
            "omitted --strategy is not explicit"
        );
    }

    #[test]
    fn degrades_defaulted_adaptive_when_keyframe_poor() {
        let cfg = cfg_with_strategy(None);
        // 2 < default min_keyframes (10): a bare `process` must still produce
        // output, so the plan degrades to fixed-fps.
        let resolved =
            degrade_if_keyframe_poor(&cfg, SampleStrategy::Adaptive, &keyframes(2), &Discard);
        assert_eq!(resolved, SampleStrategy::FixedFps);
    }

    #[test]
    fn keeps_defaulted_adaptive_when_keyframes_sufficient() {
        let cfg = cfg_with_strategy(None);
        let resolved =
            degrade_if_keyframe_poor(&cfg, SampleStrategy::Adaptive, &keyframes(20), &Discard);
        assert_eq!(resolved, SampleStrategy::Adaptive);
    }

    #[test]
    fn keeps_explicit_adaptive_when_keyframe_poor() {
        let cfg = cfg_with_strategy(Some(SampleStrategy::Adaptive));
        let resolved =
            degrade_if_keyframe_poor(&cfg, SampleStrategy::Adaptive, &keyframes(2), &Discard);
        assert_eq!(resolved, SampleStrategy::Adaptive);
    }

    #[test]
    fn degrades_windowed_default_to_fixed_fps() {
        let cfg = cfg_with_strategy(None);
        let resolved = resolve_windowed_strategy(&cfg, window()).expect("window resolves");
        assert_eq!(resolved, SampleStrategy::FixedFps);
    }

    #[test]
    fn rejects_windowed_explicit_keyframe_strategy() {
        let cfg = cfg_with_strategy(Some(SampleStrategy::Adaptive));
        let err = resolve_windowed_strategy(&cfg, window())
            .expect_err("explicit adaptive + window must fail");
        assert_eq!(err.code, ErrorCode::InvalidRange);
    }

    #[test]
    fn keeps_strategy_without_window() {
        let cfg = cfg_with_strategy(None);
        let resolved = resolve_windowed_strategy(&cfg, None).expect("no window resolves");
        assert_eq!(resolved, SampleStrategy::Adaptive);
    }
}
