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

use anyhow::{Context, Result, anyhow};

use crate::config::Config;
use crate::types::{GridMeta, KeyFrameMeta, Metadata, ScoredFrame, Settings};
use crate::{compose, dedup, extract, layout, score, subtitle};

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
) -> Result<()> {
    // Build a dedicated thread pool so -j controls actual parallelism.
    let mut builder = rayon::ThreadPoolBuilder::new();
    if threads > 0 {
        builder = builder.num_threads(threads);
    }
    let pool = builder
        .thread_name(|i| format!("vidgrid-{i}"))
        .build()
        .context("building rayon thread pool")?;

    pool.install(|| process_inner(input, out_dir, cfg, mode, max_in_flight))
}

fn process_inner(
    input: &Path,
    out_dir: &Path,
    cfg: &Config,
    mode: OutputMode,
    max_in_flight: usize,
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

    if write_files && let Some(text) = subtitle::extract(input).ok().flatten() {
        fs::write(out_dir.join("subtitles.txt"), text).context("writing subtitles.txt")?;
    }

    let meta = Metadata {
        source: input.display().to_string(),
        duration,
        settings: Settings {
            max_pixels: cfg.max_pixels,
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
    let picked: Vec<ScoredFrame> = selected.iter().map(|&i| chunk[i].clone()).collect();

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
pub fn extract_range(
    input: &Path,
    out_dir: &Path,
    cfg: &Config,
    start: f32,
    end: f32,
    mode: OutputMode,
) -> Result<()> {
    let write_files = mode == OutputMode::Files;
    if write_files {
        fs::create_dir_all(out_dir)
            .with_context(|| format!("creating output dir {}", out_dir.display()))?;
    }

    // Triple the base fps for dense sampling within the window.
    let fps = cfg.base_fps * 3.0;
    let mut stream =
        extract::stream_range(input, fps, cfg.min_frame_size, cfg.score_edge, start, end)?;

    let batch_size = extract_batch_frames(cfg);
    let mut deduper = dedup::Deduper::new(cfg.hash_threshold, cfg.dedup_window);
    let mut rgb_batch = Vec::with_capacity(batch_size);
    let mut frames = Vec::new();
    let mut aspect = 1.0_f32;

    while let Some(rgb) = stream.next_frame()? {
        if let Some(a) = stream.aspect() {
            aspect = a;
        }
        rgb_batch.push(rgb);
        if rgb_batch.len() == batch_size {
            let scored =
                extract::scored_frames_from_rgbs(mem::take(&mut rgb_batch), cfg.score_edge);
            frames.extend(deduper.filter(scored));
        }
    }
    if !rgb_batch.is_empty() {
        let scored = extract::scored_frames_from_rgbs(rgb_batch, cfg.score_edge);
        frames.extend(deduper.filter(scored));
    }

    let weights = cfg.weights();
    score::compute_raw(&mut frames);
    score::rank_normalize_and_score(&mut frames, &weights);

    let preset = cfg.as_preset();
    let capacity = layout::grid_capacity(&preset, aspect);
    let selected = score::select_top_k(&frames, capacity, cfg.min_gap);
    let picked: Vec<ScoredFrame> = selected.iter().map(|&i| frames[i].clone()).collect();

    let lay = layout::compute(picked.len(), &[], aspect, &preset);
    let jpeg = compose::render(&picked, &lay, &[], cfg.jpeg_quality, cfg.hero_border_px)?;

    let file = "grid_extract.jpg";
    let grid_meta = build_grid_meta(
        if write_files { Some(file) } else { None },
        if write_files { None } else { Some(&jpeg) },
        &picked,
        &[],
        lay.template,
    );

    let meta = Metadata {
        source: input.display().to_string(),
        duration: end - start,
        settings: Settings {
            max_pixels: cfg.max_pixels,
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
        [s] => s.parse::<f32>()?,
        [m, s] => m.parse::<f32>()? * 60.0 + s.parse::<f32>()?,
        [h, m, s] => h.parse::<f32>()? * 3600.0 + m.parse::<f32>()? * 60.0 + s.parse::<f32>()?,
        _ => anyhow::bail!("bad time '{s}'"),
    };
    Ok(secs)
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
