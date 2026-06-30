//! Pipeline orchestration: extract → dedup → chunk → (per-chunk) score → select
//! → layout → compose, then write grids, `metadata.json`, and subtitles.
//!
//! Rank normalization and selection are deliberately per-chunk: scores are only
//! meaningful relative to the frames they compete with, and a "minute" is the
//! v1 unit of competition.

use std::collections::BTreeMap;
use std::fs;
use std::mem;
use std::panic::{self, AssertUnwindSafe};
use std::path::Path;
use std::sync::mpsc::{self, Receiver, Sender, TryRecvError};

use anyhow::{Context, Result, anyhow, bail};

use crate::config::Config;
use crate::probe::{KeyframeInfo, SampleStrategy};
use crate::types::{GridMeta, KeyFrameMeta, Metadata, ScoredFrame, Settings};
use crate::{compose, dedup, extract, layout, probe, score, subtitle};

/// Subtitle extraction options.
#[derive(Debug, Clone, Copy, Default)]
pub struct SubtitleOpts<'a> {
    /// Explicit subtitle file path (highest priority).
    pub path: Option<&'a Path>,
    /// Embedded track index (used when path is None).
    pub track: Option<usize>,
}

/// Lower bound for an extraction batch.
///
/// Very low sampling rates can produce tiny chunks. Keeping at least this many
/// frames per batch amortizes rayon scheduling and pHash setup without moving
/// the memory bound materially.
const MIN_EXTRACT_BATCH_FRAMES: usize = 32;

/// Upper bound for an extraction batch.
///
/// This caps transient RGB+gray memory for high-fps profiles. The exact chunk
/// boundary is still timestamp-driven after dedup, so this bound only controls
/// conversion/dedup batching.
const MAX_EXTRACT_BATCH_FRAMES: usize = 240;

/// Default max in-flight chunks (running + queued results awaiting emit).
///
/// Two is enough to overlap ffmpeg extraction with chunk scoring/encoding.
/// Users can raise this with `-j` for more throughput at the cost of memory.
pub const DEFAULT_IN_FLIGHT: usize = 2;

/// Output mode: write files or JSON to stdout.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputMode {
    /// Write grid images and metadata.json to disk.
    Files,
    /// Output JSON with embedded base64 images to stdout.
    Json,
}

/// Run the full `process` pipeline.
///
/// `threads` controls Rayon worker count (0 = Rayon default, i.e. CPU cores).
/// `max_in_flight` bounds how many chunks can be processing/queued at once.
pub fn process(
    input: &Path,
    out_dir: &Path,
    cfg: &Config,
    mode: OutputMode,
    threads: usize,
    max_in_flight: usize,
    subs: SubtitleOpts<'_>,
) -> Result<()> {
    // Probe if needed for strategy validation
    let keyframe_info = probe_if_needed(input, cfg)?;
    validate_strategy(cfg, keyframe_info.as_ref())?;

    // Build a dedicated thread pool so -j controls actual parallelism.
    let mut builder = rayon::ThreadPoolBuilder::new();
    if threads > 0 {
        builder = builder.num_threads(threads);
    }
    let pool = builder
        .thread_name(|i| format!("vidgrid-{i}"))
        .build()
        .context("building rayon thread pool")?;

    pool.install(|| match cfg.strategy {
        SampleStrategy::FixedFps => {
            process_inner(input, out_dir, cfg, mode, max_in_flight, subs)
        }
        SampleStrategy::Keyframes => {
            let info = keyframe_info.expect("validated");
            process_keyframes(input, out_dir, cfg, mode, max_in_flight, info, subs)
        }
        SampleStrategy::Adaptive => {
            let info = keyframe_info.expect("validated");
            process_adaptive(input, out_dir, cfg, mode, max_in_flight, info, subs)
        }
    })
}

/// Probe video metadata when the strategy requires it.
fn probe_if_needed(input: &Path, cfg: &Config) -> Result<Option<KeyframeInfo>> {
    match cfg.strategy {
        SampleStrategy::FixedFps if cfg.force_long_video => Ok(None),
        SampleStrategy::FixedFps => {
            // Need duration check for the 1-hour limit
            match probe::probe_duration(input) {
                Ok(duration) => Ok(Some(KeyframeInfo {
                    timestamps: Vec::new(),
                    duration,
                })),
                Err(e) => {
                    eprintln!("warning: ffprobe failed ({e}); duration check skipped");
                    Ok(None)
                }
            }
        }
        SampleStrategy::Keyframes | SampleStrategy::Adaptive => {
            if !probe::ffprobe_available() {
                bail!(
                    "--strategy {:?} requires ffprobe (not found on PATH)",
                    cfg.strategy
                );
            }
            Ok(Some(probe::probe_keyframes(input)?))
        }
    }
}

/// Validate strategy constraints against probed video info.
fn validate_strategy(cfg: &Config, info: Option<&KeyframeInfo>) -> Result<()> {
    use crate::config::StrategyOpts;

    match cfg.strategy {
        SampleStrategy::FixedFps => {
            let dominated = info.is_some_and(|i| i.duration > StrategyOpts::MAX_DURATION_DEFAULT);
            if dominated && !cfg.force_long_video {
                let hours = info.unwrap().duration / 3600.0;
                bail!(
                    "video is {hours:.1}h (>{:.0}s limit for fixed-fps); \
                     use --force or --strategy keyframes/adaptive",
                    StrategyOpts::MAX_DURATION_DEFAULT
                );
            }
        }
        SampleStrategy::Keyframes | SampleStrategy::Adaptive => {
            let i = info.expect("probe should succeed for keyframes/adaptive");
            if i.count() < cfg.min_keyframes {
                bail!(
                    "video has only {} keyframes (<{} required); \
                     lower --min-keyframes or use --strategy fixed-fps",
                    i.count(),
                    cfg.min_keyframes
                );
            }
        }
    }
    Ok(())
}

fn process_inner(
    input: &Path,
    out_dir: &Path,
    cfg: &Config,
    mode: OutputMode,
    max_in_flight: usize,
    subs: SubtitleOpts<'_>,
) -> Result<()> {
    if mode == OutputMode::Files {
        fs::create_dir_all(out_dir)
            .with_context(|| format!("creating output dir {}", out_dir.display()))?;
    }

    let write_files = mode == OutputMode::Files;
    let mut stream =
        extract::stream_frames(input, cfg.base_fps, cfg.min_frame_size, cfg.score_edge)?;
    let mut state = ProcessState::new(cfg, write_files, max_in_flight);

    while let Some(frame) = stream.next_frame()? {
        if let Some(aspect) = stream.aspect() {
            state.set_aspect(aspect);
        }
        state.push_rgb(frame, out_dir)?;
    }
    let (duration, grid_metas) = state.finish(out_dir)?;

    write_outputs(input, out_dir, cfg, mode, duration, grid_metas, subs)
}

/// Process video using keyframes-only strategy.
fn process_keyframes(
    input: &Path,
    out_dir: &Path,
    cfg: &Config,
    mode: OutputMode,
    max_in_flight: usize,
    info: KeyframeInfo,
    subs: SubtitleOpts<'_>,
) -> Result<()> {
    if mode == OutputMode::Files {
        fs::create_dir_all(out_dir)
            .with_context(|| format!("creating output dir {}", out_dir.display()))?;
    }

    eprintln!(
        "keyframes mode: {} I-frames over {:.1}s (mean gap {:.1}s, max {:.1}s)",
        info.count(),
        info.duration,
        info.mean_gap(),
        info.max_gap()
    );

    let write_files = mode == OutputMode::Files;
    let mut stream =
        extract::stream_keyframes(input, info.timestamps, cfg.min_frame_size, cfg.score_edge)?;
    let mut state = ProcessState::new(cfg, write_files, max_in_flight);

    while let Some(frame) = stream.next_frame()? {
        if let Some(aspect) = stream.aspect() {
            state.set_aspect(aspect);
        }
        state.push_rgb(frame, out_dir)?;
    }
    let (duration, grid_metas) = state.finish(out_dir)?;

    write_outputs(input, out_dir, cfg, mode, duration, grid_metas, subs)
}

/// Process video using adaptive strategy: keyframes + sparse region fill.
fn process_adaptive(
    input: &Path,
    out_dir: &Path,
    cfg: &Config,
    mode: OutputMode,
    max_in_flight: usize,
    info: KeyframeInfo,
    subs: SubtitleOpts<'_>,
) -> Result<()> {
    if mode == OutputMode::Files {
        fs::create_dir_all(out_dir)
            .with_context(|| format!("creating output dir {}", out_dir.display()))?;
    }

    let sparse_regions = info.sparse_regions(cfg.gap_threshold);
    eprintln!(
        "adaptive mode: {} I-frames (max gap {:.1}s), {} sparse regions (threshold {:.1}s)",
        info.count(),
        info.max_gap(),
        sparse_regions.len(),
        cfg.gap_threshold
    );

    let write_files = mode == OutputMode::Files;
    let mut state = ProcessState::new(cfg, write_files, max_in_flight);

    // Phase 1: extract keyframes
    let mut stream =
        extract::stream_keyframes(input, info.timestamps, cfg.min_frame_size, cfg.score_edge)?;
    while let Some(frame) = stream.next_frame()? {
        if let Some(aspect) = stream.aspect() {
            state.set_aspect(aspect);
        }
        state.push_rgb(frame, out_dir)?;
    }

    // Phase 2: fill sparse regions with fixed-fps samples
    for (start, end) in sparse_regions {
        eprintln!("  filling sparse region {:.1}s-{:.1}s", start, end);
        let mut fill_stream = extract::stream_range(
            input,
            cfg.base_fps,
            cfg.min_frame_size,
            cfg.score_edge,
            start,
            end,
        )?;
        while let Some(frame) = fill_stream.next_frame()? {
            if let Some(aspect) = fill_stream.aspect() {
                state.set_aspect(aspect);
            }
            state.push_rgb(frame, out_dir)?;
        }
    }

    let (duration, grid_metas) = state.finish(out_dir)?;

    write_outputs(input, out_dir, cfg, mode, duration, grid_metas, subs)
}

/// Shared output writing for all strategies.
fn write_outputs(
    input: &Path,
    out_dir: &Path,
    cfg: &Config,
    mode: OutputMode,
    duration: f32,
    mut grid_metas: Vec<GridMeta>,
    subs: SubtitleOpts<'_>,
) -> Result<()> {
    let write_files = mode == OutputMode::Files;

    // Extract subtitles: explicit path > explicit track > auto-detection
    let srt_text = match (subs.path, subs.track) {
        (Some(path), _) => Some(subtitle::extract_from_file(path)?),
        (None, Some(track)) => Some(subtitle::extract_track(input, track)?),
        (None, None) => subtitle::extract(input)?,
    };
    if let Some(ref text) = srt_text {
        let cues = subtitle::parse_srt(text);
        for grid in &mut grid_metas {
            if let Ok((start, end)) = parse_range(&grid.time_range) {
                grid.subtitles = subtitle::slice_for_range(&cues, start, end);
            }
        }
        if write_files {
            fs::write(out_dir.join("subtitles.txt"), text).context("writing subtitles.txt")?;
        }
    }

    let meta = Metadata {
        source: input.display().to_string(),
        duration,
        settings: Settings {
            max_edge: cfg.max_edge,
            quality: cfg.quality_label.to_string(),
            detail: cfg.detail_label.to_string(),
        },
        grids: grid_metas,
    };

    if write_files {
        let json = serde_json::to_string_pretty(&meta).context("serializing metadata")?;
        fs::write(out_dir.join("metadata.json"), json).context("writing metadata.json")?;
        eprintln!(
            "wrote {} grid(s) to {}",
            meta.grids.len(),
            out_dir.display()
        );
    } else {
        let json = serde_json::to_string_pretty(&meta).context("serializing metadata")?;
        println!("{json}");
    }

    Ok(())
}

#[derive(Debug)]
struct GridResult {
    ci: usize,
    jpeg: Vec<u8>,
    meta: GridMeta,
}

#[derive(Clone, Copy)]
struct ProcessChunkEnv {
    preset: crate::types::Preset,
    aspect: f32,
    capacity: usize,
    weights: score::Weights,
    write_files: bool,
    min_gap_floor: f32,
    jpeg_quality: u8,
    hero_border_px: u32,
}

struct ProcessState<'a> {
    cfg: &'a Config,
    env: ProcessChunkEnv,
    deduper: dedup::Deduper,
    rgb_batch: Vec<extract::RgbFrame>,
    batch_frames: usize,
    max_in_flight: usize,
    current_chunk: Vec<ScoredFrame>,
    current_key: Option<usize>,
    next_chunk_ci: usize,
    tx: Option<Sender<ChunkMsg>>,
    rx: Receiver<ChunkMsg>,
    ready: BTreeMap<usize, Result<GridResult>>,
    outstanding: usize,
    next_emit: usize,
    duration: f32,
    grid_metas: Vec<GridMeta>,
}

impl<'a> ProcessState<'a> {
    fn new(cfg: &'a Config, write_files: bool, parallelism: usize) -> Self {
        let preset = cfg.as_preset();
        let (tx, rx) = mpsc::channel();
        Self {
            cfg,
            env: ProcessChunkEnv {
                preset,
                aspect: 1.0,
                capacity: layout::grid_capacity(&preset, 1.0),
                weights: cfg.weights(),
                write_files,
                min_gap_floor: cfg.min_gap,
                jpeg_quality: cfg.jpeg_quality,
                hero_border_px: cfg.hero_border_px,
            },
            deduper: dedup::Deduper::new(cfg.hash_threshold, cfg.dedup_window),
            rgb_batch: Vec::with_capacity(extract_batch_frames(cfg)),
            batch_frames: extract_batch_frames(cfg),
            max_in_flight: parallelism.max(1),
            current_chunk: Vec::new(),
            current_key: None,
            next_chunk_ci: 0,
            tx: Some(tx),
            rx,
            ready: BTreeMap::new(),
            outstanding: 0,
            next_emit: 0,
            duration: 0.0,
            grid_metas: Vec::new(),
        }
    }

    fn set_aspect(&mut self, aspect: f32) {
        self.env.aspect = aspect;
        self.env.capacity = layout::grid_capacity(&self.env.preset, aspect);
    }

    fn push_rgb(&mut self, frame: extract::RgbFrame, out_dir: &Path) -> Result<()> {
        self.rgb_batch.push(frame);
        if self.rgb_batch.len() == self.batch_frames {
            self.flush_extract_batch();
            self.drain_results(out_dir, Some(self.max_in_flight))?;
        }
        Ok(())
    }

    fn finish(mut self, out_dir: &Path) -> Result<(f32, Vec<GridMeta>)> {
        self.flush_extract_batch();
        self.flush_current_chunk();
        self.tx = None;
        self.drain_results(out_dir, Some(0))?;
        Ok((self.duration, self.grid_metas))
    }

    fn flush_extract_batch(&mut self) {
        if self.rgb_batch.is_empty() {
            return;
        }

        let frames =
            extract::scored_frames_from_rgbs(mem::take(&mut self.rgb_batch), self.cfg.score_edge);
        for frame in self.deduper.filter(frames) {
            self.duration = frame.ts + 1.0 / self.cfg.base_fps;
            self.push_chunked_frame(frame);
        }
    }

    fn push_chunked_frame(&mut self, frame: ScoredFrame) {
        let key = (frame.ts / self.cfg.chunk_seconds).floor() as usize;
        match self.current_key {
            None => self.current_key = Some(key),
            Some(cur) if cur != key => {
                self.flush_current_chunk();
                self.current_key = Some(key);
            }
            Some(_) => {}
        }
        self.current_chunk.push(frame);
    }

    fn flush_current_chunk(&mut self) {
        if self.current_chunk.is_empty() {
            return;
        }
        let ci = self.next_chunk_ci;
        self.next_chunk_ci += 1;
        let chunk = mem::take(&mut self.current_chunk);
        self.spawn_chunk(ci, chunk);
    }

    fn spawn_chunk(&mut self, ci: usize, chunk: Vec<ScoredFrame>) {
        self.outstanding += 1;
        let tx = self.tx.as_ref().expect("sender lives until finish").clone();
        let env = self.env;
        rayon::spawn_fifo(move || {
            // Catch panics so we always send a result — otherwise outstanding
            // never decrements and the main thread blocks forever on recv().
            let result = panic::catch_unwind(AssertUnwindSafe(|| process_chunk(ci, chunk, env)))
                .unwrap_or_else(|panic| Err(panic_to_anyhow(panic)));
            let _ = tx.send((ci, result));
        });
    }

    fn drain_results(&mut self, out_dir: &Path, max_in_flight: Option<usize>) -> Result<()> {
        loop {
            // Backpressure: count both outstanding jobs AND queued results awaiting emit.
            let buffered = self.outstanding + self.ready.len();
            if max_in_flight.is_some_and(|max| buffered > max) {
                let (ci, result) = self.rx.recv().context("receiving grid result")?;
                self.outstanding -= 1;
                self.ready.insert(ci, result);
            } else {
                loop {
                    match self.rx.try_recv() {
                        Ok((ci, result)) => {
                            self.outstanding -= 1;
                            self.ready.insert(ci, result);
                        }
                        Err(TryRecvError::Empty) => break,
                        Err(TryRecvError::Disconnected) => break,
                    }
                }
            }

            while let Some(result) = self.ready.remove(&self.next_emit) {
                let result = result?;
                debug_assert_eq!(result.ci, self.next_emit);
                if self.env.write_files {
                    let file = format!("grid_{:03}.jpg", result.ci + 1);
                    fs::write(out_dir.join(&file), &result.jpeg)
                        .with_context(|| format!("writing {file}"))?;
                }
                self.grid_metas.push(result.meta);
                self.next_emit += 1;
            }

            let buffered = self.outstanding + self.ready.len();
            if max_in_flight.is_none_or(|max| buffered <= max) {
                break;
            }
        }
        Ok(())
    }
}

fn extract_batch_frames(cfg: &Config) -> usize {
    ((cfg.base_fps * cfg.chunk_seconds).round().max(1.0) as usize)
        .clamp(MIN_EXTRACT_BATCH_FRAMES, MAX_EXTRACT_BATCH_FRAMES)
}

type ChunkMsg = (usize, Result<GridResult>);

fn process_chunk(
    ci: usize,
    mut chunk: Vec<ScoredFrame>,
    env: ProcessChunkEnv,
) -> Result<GridResult> {
    score::compute_raw(&mut chunk);
    score::rank_normalize_and_score(&mut chunk, &env.weights);

    let span = chunk_span(&chunk);
    let min_gap = (0.5 * span / env.capacity.max(1) as f32).max(env.min_gap_floor);
    let selected = score::select_top_k(&chunk, env.capacity, min_gap);
    let picked = extract_selected(&mut chunk, selected);

    let heroes = score::pick_heroes(&picked);
    let lay = layout::compute(picked.len(), &heroes, env.aspect, &env.preset);

    let jpeg = compose::render(&picked, &lay, &heroes, env.jpeg_quality, env.hero_border_px)?;
    let file = format!("grid_{:03}.jpg", ci + 1);

    let meta = build_grid_meta(
        if env.write_files {
            Some(file.as_str())
        } else {
            None
        },
        if env.write_files { None } else { Some(&jpeg) },
        &picked,
        &heroes,
        lay.template,
    );

    Ok(GridResult { ci, jpeg, meta })
}

/// Run the `extract` pipeline: high-density sampling of `[start, end)` into a
/// single equal-cell grid, frames in time order (no hero emphasis).
///
/// Uses streaming dedup + batched selection to bound memory: each batch is
/// scored and reduced to `capacity` candidates before the next batch loads.
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

    let extract_batch = extract_batch_frames(cfg);
    let select_batch = (cfg.base_fps * cfg.chunk_seconds).round().max(1.0) as usize;
    let mut deduper = dedup::Deduper::new(cfg.hash_threshold, cfg.dedup_window);
    let weights = cfg.weights();
    let preset = cfg.as_preset();

    let mut rgb_batch = Vec::with_capacity(extract_batch);
    let mut scored_batch: Vec<ScoredFrame> = Vec::new();
    let mut candidates: Vec<ScoredFrame> = Vec::new();
    let mut aspect = 1.0_f32;

    while let Some(rgb) = stream.next_frame()? {
        if let Some(a) = stream.aspect() {
            aspect = a;
        }
        rgb_batch.push(rgb);
        if rgb_batch.len() >= extract_batch {
            let scored =
                extract::scored_frames_from_rgbs(mem::take(&mut rgb_batch), cfg.score_edge);
            scored_batch.extend(deduper.filter(scored));
        }

        if scored_batch.len() >= select_batch {
            flush_batch_to_candidates(
                &mut scored_batch,
                &mut candidates,
                &weights,
                &preset,
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
            &preset,
            aspect,
            cfg.min_gap,
        );
    }

    let capacity = layout::grid_capacity(&preset, aspect);
    let picked = if candidates.len() <= capacity {
        candidates
    } else {
        score::compute_raw(&mut candidates);
        score::rank_normalize_and_score(&mut candidates, &weights);
        let selected = score::select_top_k(&candidates, capacity, cfg.min_gap);
        extract_selected(&mut candidates, selected)
    };

    let lay = layout::compute(picked.len(), &[], aspect, &preset);
    let jpeg = compose::render(&picked, &lay, &[], cfg.jpeg_quality, cfg.hero_border_px)?;

    let file = "grid_extract.jpg";
    let mut grid_meta = build_grid_meta(
        if write_files { Some(file) } else { None },
        if write_files { None } else { Some(&jpeg) },
        &picked,
        &[],
        lay.template,
    );

    // Extract subtitles for this range
    let srt_text = match (subs.path, subs.track) {
        (Some(path), _) => Some(subtitle::extract_from_file(path)?),
        (None, Some(track)) => Some(subtitle::extract_track(input, track)?),
        (None, None) => subtitle::extract(input)?,
    };
    if let Some(ref text) = srt_text {
        let cues = subtitle::parse_srt(text);
        grid_meta.subtitles = subtitle::slice_for_range(&cues, start, end);
    }

    let meta = Metadata {
        source: input.display().to_string(),
        duration: end - start,
        settings: Settings {
            max_edge: cfg.max_edge,
            quality: cfg.quality_label.to_string(),
            detail: cfg.detail_label.to_string(),
        },
        grids: vec![grid_meta],
    };

    if write_files {
        fs::write(out_dir.join(file), &jpeg).context("writing grid_extract.jpg")?;
        let json = serde_json::to_string_pretty(&meta).context("serializing metadata")?;
        fs::write(out_dir.join("metadata.json"), json).context("writing metadata.json")?;
        eprintln!("wrote {} to {}", file, out_dir.display());
    } else {
        let json = serde_json::to_string_pretty(&meta).context("serializing metadata")?;
        println!("{json}");
    }

    Ok(())
}

fn flush_batch_to_candidates(
    batch: &mut Vec<ScoredFrame>,
    candidates: &mut Vec<ScoredFrame>,
    weights: &score::Weights,
    preset: &crate::types::Preset,
    aspect: f32,
    min_gap: f32,
) {
    if batch.is_empty() {
        return;
    }
    score::compute_raw(batch);
    score::rank_normalize_and_score(batch, weights);
    let capacity = layout::grid_capacity(preset, aspect);
    let selected = score::select_top_k(batch, capacity, min_gap);
    candidates.extend(extract_selected(batch, selected));
    batch.clear();
}

/// Parse a `MM:SS-MM:SS` (or `SS-SS`) range into `(start, end)` seconds.
pub fn parse_range(s: &str) -> Result<(f32, f32)> {
    let (a, b) = s
        .split_once('-')
        .with_context(|| format!("range '{s}' must be START-END"))?;
    let start = parse_clock(a)?;
    let end = parse_clock(b)?;
    if end <= start {
        anyhow::bail!("range end ({end}s) must be after start ({start}s)");
    }
    Ok((start, end))
}

/// Parse `MM:SS`, `HH:MM:SS`, or plain seconds into seconds.
fn parse_clock(s: &str) -> Result<f32> {
    let parts: Vec<&str> = s.split(':').collect();
    let secs = match parts.as_slice() {
        [sec] => sec.parse::<f32>(),
        [m, sec] => Ok(m.parse::<f32>()? * 60.0 + sec.parse::<f32>()?),
        [h, m, sec] => {
            Ok(h.parse::<f32>()? * 3600.0 + m.parse::<f32>()? * 60.0 + sec.parse::<f32>()?)
        }
        _ => anyhow::bail!("expected MM:SS or seconds, got '{s}'"),
    };
    secs.with_context(|| format!("invalid time '{s}'"))
}

/// Extract frames at `indices` from `frames` without cloning.
///
/// Uses `swap_remove` so indices must be processed in descending order to avoid
/// invalidation. The returned frames are sorted by timestamp.
fn extract_selected(frames: &mut Vec<ScoredFrame>, mut indices: Vec<usize>) -> Vec<ScoredFrame> {
    indices.sort_unstable_by(|a, b| b.cmp(a));
    let mut picked = Vec::with_capacity(indices.len());
    for i in indices {
        picked.push(frames.swap_remove(i));
    }
    picked.sort_by(|a, b| a.ts.total_cmp(&b.ts));
    picked
}

/// Build the metadata record for one rendered grid.
fn build_grid_meta(
    file: Option<&str>,
    jpeg: Option<&[u8]>,
    picked: &[ScoredFrame],
    heroes: &[usize],
    template: &'static str,
) -> GridMeta {
    use base64::{Engine, engine::general_purpose::STANDARD};

    let (first, last) = (
        picked.first().map(|f| f.ts).unwrap_or(0.0),
        picked.last().map(|f| f.ts).unwrap_or(0.0),
    );
    let key_frames = heroes
        .iter()
        .map(|&i| KeyFrameMeta {
            timestamp: compose::format_ts(picked[i].ts),
            score: picked[i].score.round() as u32,
            grid_position: "hero".to_string(),
        })
        .collect();
    GridMeta {
        file: file.map(str::to_string),
        time_range: format!("{}-{}", compose::format_ts(first), compose::format_ts(last)),
        total_frames: picked.len(),
        key_frames,
        layout: template.to_string(),
        base64: jpeg.map(|b| STANDARD.encode(b)),
        subtitles: None,
    }
}

/// Largest timestamp span within a chunk.
fn chunk_span(chunk: &[ScoredFrame]) -> f32 {
    match (chunk.first(), chunk.last()) {
        (Some(a), Some(b)) => (b.ts - a.ts).max(0.0),
        _ => 0.0,
    }
}

/// Partition frames (assumed time-ordered) into chunks of `seconds` each, keyed
/// by `floor(ts / seconds)`. Empty chunks are not emitted.
#[cfg(test)]
fn chunk_by_seconds(frames: Vec<ScoredFrame>, seconds: f32) -> Vec<Vec<ScoredFrame>> {
    let mut chunks: Vec<Vec<ScoredFrame>> = Vec::new();
    let mut cur_key = usize::MAX;
    for f in frames {
        let key = (f.ts / seconds).floor() as usize;
        if key != cur_key {
            chunks.push(Vec::new());
            cur_key = key;
        }
        chunks.last_mut().expect("just pushed").push(f);
    }
    chunks
}

/// Convert a panic payload to an anyhow error.
fn panic_to_anyhow(payload: Box<dyn std::any::Any + Send>) -> anyhow::Error {
    if let Some(s) = payload.downcast_ref::<&str>() {
        anyhow!("worker panic: {s}")
    } else if let Some(s) = payload.downcast_ref::<String>() {
        anyhow!("worker panic: {s}")
    } else {
        anyhow!("worker panic (unknown payload)")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::RawScores;
    use image::{GrayImage, RgbImage};

    fn frame(ts: f32) -> ScoredFrame {
        ScoredFrame {
            ts,
            gray: GrayImage::new(1, 1),
            rgb: RgbImage::new(1, 1),
            raw: RawScores::default(),
            score: 0.0,
        }
    }

    #[test]
    fn splits_frames_at_minute_boundaries() {
        let frames = vec![
            frame(0.0),
            frame(30.0),
            frame(59.9),
            frame(60.1),
            frame(125.0),
        ];
        let chunks = chunk_by_seconds(frames, 60.0);
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].len(), 3);
        assert_eq!(chunks[1].len(), 1);
        assert_eq!(chunks[2].len(), 1);
    }

    #[test]
    fn keeps_single_chunk_for_short_clip() {
        let frames = vec![frame(0.0), frame(5.0), frame(10.0)];
        let chunks = chunk_by_seconds(frames, 60.0);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].len(), 3);
    }

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
