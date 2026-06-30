//! Pipeline orchestration: extract → dedup → chunk → (per-chunk) score → select
//! → layout → compose, then write grids, `metadata.json`, and subtitles.
//!
//! Rank normalization and selection are deliberately per-chunk: scores are only
//! meaningful relative to the frames they compete with, and a "minute" is the
//! v1 unit of competition.

mod chunk;
mod output;
mod state;

use std::fs;
use std::mem;
use std::path::Path;

use anyhow::{Context, Result, bail};

use crate::config::Config;
use crate::error::{Error, ErrorCode, ErrorContext};
use crate::probe::{KeyframeInfo, SampleStrategy};
use crate::types::Metadata;
use crate::{compose, dedup, extract, layout, probe, score, subtitle};

use chunk::{build_grid_meta, extract_selected};
use output::atomic_write;
use state::{extract_batch_frames, ProcessState};

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
    };
    match cfg.strategy {
        SampleStrategy::FixedFps => {
            ensure_output_dir(out_dir, mode)?;
            eprintln!("fixed-fps mode: sampling at {:.1} fps", cfg.base_fps);
            let stream =
                extract::stream_frames(input, cfg.base_fps, cfg.min_frame_size, cfg.score_edge)?;
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
            let stream = extract::stream_keyframes(
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
        SampleStrategy::FixedFps => {
            match probe::probe_duration(input) {
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
            }
        }
        SampleStrategy::Keyframes | SampleStrategy::Adaptive => {
            if !probe::ffprobe_available() {
                return Err(Error::new(
                    ErrorCode::FfprobeNotFound,
                    format!("--strategy {:?} requires ffprobe (not found on PATH)", cfg.strategy),
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
            if let Some(i) =
                info.filter(|i| i.duration > StrategyOpts::MAX_DURATION_DEFAULT)
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
    stream: &mut extract::FrameStream,
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
fn process_stream(ctx: &RunCtx<'_>, mut stream: extract::FrameStream) -> Result<()> {
    let write_files = ctx.mode == OutputMode::Files;
    let mut state = ProcessState::new(ctx.cfg, ctx.pool, write_files, ctx.max_in_flight, true);

    drain_source(&mut stream, &mut state, ctx.out_dir)?;
    let (duration, grid_metas) = state.finish(ctx.out_dir)?;

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
    let mut state = ProcessState::new(cfg, ctx.pool, write_files, ctx.max_in_flight, false);

    let mut stream =
        extract::stream_keyframes(ctx.input, info.timestamps, cfg.min_frame_size, cfg.score_edge)?;
    drain_source(&mut stream, &mut state, ctx.out_dir)?;

    for (start, end) in sparse_regions {
        eprintln!("  filling sparse region {:.1}s-{:.1}s", start, end);
        let mut fill_stream = extract::stream_range(
            ctx.input,
            cfg.base_fps,
            cfg.min_frame_size,
            cfg.score_edge,
            start,
            end,
        )?;
        drain_source(&mut fill_stream, &mut state, ctx.out_dir)?;
    }

    let (duration, grid_metas) = state.finish(ctx.out_dir)?;

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

/// Run the `extract` pipeline: high-density sampling of `[start, end)` into a
/// single equal-cell grid, frames in time order (no hero emphasis).
pub fn extract_range(
    input: &Path,
    out_dir: &Path,
    cfg: &Config,
    start: f32,
    end: f32,
    mode: OutputMode,
    subs: SubtitleOpts<'_>,
) -> Result<()> {
    let write_files = mode == OutputMode::Files;
    if write_files {
        fs::create_dir_all(out_dir)
            .with_context(|| format!("creating output dir {}", out_dir.display()))?;
    }

    let fps = cfg.base_fps * 3.0;
    let mut stream =
        extract::stream_range(input, fps, cfg.min_frame_size, cfg.score_edge, start, end)?;

    let batch_size = extract_batch_frames(cfg);
    let mut deduper = dedup::Deduper::new(cfg.hash_threshold, cfg.dedup_window);
    let weights = cfg.weights();
    let spec = cfg.canvas_spec();

    let mut rgb_batch = Vec::with_capacity(batch_size);
    let mut scored_batch: Vec<crate::types::ScoredFrame> = Vec::new();
    let mut candidates: Vec<crate::types::ScoredFrame> = Vec::new();
    let mut aspect = 1.0_f32;

    while let Some(rgb) = stream.next_frame()? {
        if let Some(a) = stream.aspect() {
            aspect = a;
        }
        rgb_batch.push(rgb);
        if rgb_batch.len() >= batch_size {
            let scored =
                extract::scored_frames_from_rgbs(mem::take(&mut rgb_batch), cfg.score_edge);
            scored_batch.extend(deduper.filter(scored));
        }

        if scored_batch.len() >= batch_size {
            flush_batch_to_candidates(
                &mut scored_batch,
                &mut candidates,
                &weights,
                &spec,
                aspect,
                cfg.min_gap,
            );
        }
    }

    if !rgb_batch.is_empty() {
        let scored = extract::scored_frames_from_rgbs(rgb_batch, cfg.score_edge);
        scored_batch.extend(deduper.filter(scored));
    }
    if !scored_batch.is_empty() {
        flush_batch_to_candidates(
            &mut scored_batch,
            &mut candidates,
            &weights,
            &spec,
            aspect,
            cfg.min_gap,
        );
    }

    let capacity = layout::grid_capacity(&spec, aspect);
    let picked = if candidates.len() <= capacity {
        candidates
    } else {
        score::compute_raw(&mut candidates);
        score::rank_normalize_and_score(&mut candidates, &weights);
        let selected = score::select_top_k(&candidates, capacity, cfg.min_gap);
        extract_selected(&mut candidates, selected)
    };

    let lay = layout::compute(picked.len(), &[], aspect, &spec);
    let jpeg = compose::render(&picked, &lay, &[], cfg.jpeg_quality, cfg.hero_border_px)?;

    let file = "grid_extract.jpg";
    let mut grid_meta = build_grid_meta(
        if write_files { Some(file) } else { None },
        if write_files { None } else { Some(&jpeg) },
        &picked,
        &[],
        lay.template,
    );

    let srt_text = subtitle::resolve(input, subs.path, subs.track)?;
    if let Some(ref text) = srt_text {
        let cues = subtitle::parse_srt(text);
        grid_meta.subtitles = subtitle::slice_for_range(&cues, start, end);
    }

    let meta = Metadata {
        source: input.display().to_string(),
        duration: end - start,
        settings: cfg.settings(),
        grids: vec![grid_meta],
    };

    if write_files {
        atomic_write(out_dir, file, &jpeg)?;
        let json = serde_json::to_string_pretty(&meta).context("serializing metadata")?;
        atomic_write(out_dir, "metadata.json", json.as_bytes())?;
        eprintln!("wrote {} to {}", file, out_dir.display());
    } else {
        let json = serde_json::to_string_pretty(&meta).context("serializing metadata")?;
        println!("{json}");
    }

    Ok(())
}

fn flush_batch_to_candidates(
    batch: &mut Vec<crate::types::ScoredFrame>,
    candidates: &mut Vec<crate::types::ScoredFrame>,
    weights: &score::Weights,
    spec: &crate::types::CanvasSpec,
    aspect: f32,
    min_gap: f32,
) {
    if batch.is_empty() {
        return;
    }
    score::compute_raw(batch);
    score::rank_normalize_and_score(batch, weights);
    let capacity = layout::grid_capacity(spec, aspect);
    let selected = score::select_top_k(batch, capacity, min_gap);
    candidates.extend(extract_selected(batch, selected));
    batch.clear();
}

/// Parse a `MM:SS-MM:SS` (or `SS-SS`) range into `(start, end)` seconds.
pub fn parse_range(s: &str) -> Result<(f32, f32)> {
    let (a, b) = s.split_once('-').ok_or_else(|| {
        Error::new(ErrorCode::InvalidRange, format!("range '{s}' must be START-END"))
    })?;
    let start = parse_clock(a)?;
    let end = parse_clock(b)?;
    if end <= start {
        return Err(Error::new(
            ErrorCode::InvalidRange,
            format!("range end ({end}s) must be after start ({start}s)"),
        )
        .into());
    }
    Ok((start, end))
}

fn parse_clock(s: &str) -> Result<f32> {
    let parts: Vec<&str> = s.split(':').collect();
    let parse = |p: &str| p.parse::<f32>().with_context(|| format!("invalid time '{s}'"));
    match parts.as_slice() {
        [sec] => parse(sec),
        [m, sec] => Ok(parse(m)? * 60.0 + parse(sec)?),
        [h, m, sec] => Ok(parse(h)? * 3600.0 + parse(m)? * 60.0 + parse(sec)?),
        _ => bail!("expected MM:SS or seconds, got '{s}'"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_mmss_range() {
        let (s, e) = parse_range("01:23-01:35").expect("valid range");
        assert_eq!(s, 83.0);
        assert_eq!(e, 95.0);
    }

    #[test]
    fn parses_plain_seconds_range() {
        let (s, e) = parse_range("5-12").expect("valid range");
        assert_eq!((s, e), (5.0, 12.0));
    }

    #[test]
    fn rejects_inverted_range() {
        assert!(parse_range("01:35-01:23").is_err());
    }
}
