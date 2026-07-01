//! ProcessState: the streaming state machine that buffers RGB frames, deduplicates,
//! chunks by time, and spawns parallel chunk-processing jobs.
//!
//! Time chunks are a throttle, not a floor: a chunk whose surviving frames
//! cannot fill a grid is merged forward into the next chunk(s), so sparse
//! content produces fewer, fuller grids instead of one half-empty grid per
//! chunk.
//!
//! It also detects static stretches. Dedup collapses an unchanging scene into a
//! long gap between kept frames; when such a gap spans at least a chunk and
//! enough sampled frames were dropped inside it as duplicates (the evidence
//! that the stretch was watched, not just unsampled), the state emits a single
//! representative image plus the covered time range instead of empty grids.

use std::collections::BTreeMap;
use std::mem;
use std::panic::{self, AssertUnwindSafe};
use std::path::Path;
use std::sync::Arc;
use std::sync::mpsc::{self, Receiver, Sender, TryRecvError};

use anyhow::{Context, Result};
use image::{GrayImage, RgbImage};

use crate::config::Config;
use crate::types::{CandidateFrame, GridMeta, TimeRange, format_ts};
use crate::{compose, dedup, extract, layout};

use super::chunk::{ChunkMsg, GridResult, ProcessChunkEnv, process_chunk};
use super::output::atomic_write;

/// Lower bound for an extraction batch.
const MIN_EXTRACT_BATCH_FRAMES: usize = 32;

/// Upper bound for an extraction batch.
const MAX_EXTRACT_BATCH_FRAMES: usize = 240;

/// Minimum number of frames dropped as duplicates inside a kept-frame gap for
/// the gap to count as a static stretch. One drop could be a flicker; a few
/// spanning drops are evidence that the scene genuinely held still. Gaps
/// without drops (sparse keyframe sampling) are left alone.
const MIN_STATIC_DROPS: usize = 3;

pub struct ProcessState<'a> {
    cfg: &'a Config,
    /// Worker pool for chunk jobs and per-frame batch work. The state's own
    /// methods run on the caller's thread and may block on `rx`, so they must
    /// never execute on a pool thread — a single-thread pool would deadlock.
    pool: &'a rayon::ThreadPool,
    pub(super) env: ProcessChunkEnv,
    deduper: dedup::Deduper,
    rgb_batch: Vec<extract::RgbFrame>,
    batch_frames: usize,
    max_in_flight: usize,
    pending: BTreeMap<usize, Vec<CandidateFrame>>,
    /// Frames from sealed time chunks that have not yet filled a grid. The
    /// time chunk is a throttle (at most one grid per `chunk_seconds`), not a
    /// floor: sparse chunks merge forward here until the pool can fill a grid.
    carry: Vec<CandidateFrame>,
    /// Grayscale of the previously spawned chunk's last frame — scene-diff
    /// boundary context, so the next chunk's first frame is scored against its
    /// true predecessor instead of a hardcoded zero.
    prev_chunk_tail: Option<GrayImage>,
    high_water_mark: Option<usize>,
    next_chunk_ci: usize,
    tx: Option<Sender<ChunkMsg>>,
    rx: Receiver<ChunkMsg>,
    ready: BTreeMap<usize, Result<GridResult>>,
    outstanding: usize,
    next_emit: usize,
    pub(super) duration: f32,
    pub(super) grid_metas: Vec<GridMeta>,
    /// Timestamp and shared RGB of the most recent kept frame — the candidate
    /// representative should a static gap follow it. The `Arc` makes retaining
    /// it a pointer clone rather than a deep copy of the ~1.4 MB image.
    last_kept: Option<(f32, Arc<RgbImage>)>,
    /// Frames dropped as duplicates since the last kept frame.
    drops_since_kept: usize,
    /// Timestamp of the most recent dropped frame (bounds a trailing stretch).
    last_dropped_ts: f32,
    static_seq: usize,
    static_metas: Vec<GridMeta>,
}

impl<'a> ProcessState<'a> {
    pub fn new(
        cfg: &'a Config,
        pool: &'a rayon::ThreadPool,
        write_files: bool,
        parallelism: usize,
    ) -> Self {
        let (tx, rx) = mpsc::channel();
        Self {
            cfg,
            pool,
            env: ProcessChunkEnv {
                spec: cfg.canvas_spec(),
                aspect: 1.0,
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
            pending: BTreeMap::new(),
            carry: Vec::new(),
            prev_chunk_tail: None,
            high_water_mark: None,
            next_chunk_ci: 0,
            tx: Some(tx),
            rx,
            ready: BTreeMap::new(),
            outstanding: 0,
            next_emit: 0,
            duration: 0.0,
            grid_metas: Vec::new(),
            last_kept: None,
            drops_since_kept: 0,
            last_dropped_ts: 0.0,
            static_seq: 0,
            static_metas: Vec::new(),
        }
    }

    pub fn set_aspect(&mut self, aspect: f32) {
        self.env.aspect = aspect;
    }

    /// Buffer one sampled frame. Frames must arrive in non-decreasing `ts`
    /// order: chunk keying, high-water-mark sealing, and static detection all
    /// assume a monotonic stream (adaptive mode interleaves fills to preserve
    /// this).
    pub fn push_rgb(&mut self, frame: extract::RgbFrame, out_dir: &Path) -> Result<()> {
        self.rgb_batch.push(frame);
        if self.rgb_batch.len() == self.batch_frames {
            self.flush_extract_batch(out_dir)?;
            self.drain_results(out_dir, Some(self.max_in_flight))?;
        }
        Ok(())
    }

    pub fn finish(mut self, out_dir: &Path) -> Result<(f32, Vec<GridMeta>)> {
        self.flush_extract_batch(out_dir)?;
        self.finish_trailing_static(out_dir)?;
        self.seal_all();
        self.tx = None;
        self.drain_results(out_dir, Some(0))?;

        // Interleave static entries with grids by start time. Both lists are
        // already time-ascending; a static range starts inside the grid that
        // holds its representative, so it sorts right after that grid.
        let mut metas = mem::take(&mut self.grid_metas);
        metas.append(&mut self.static_metas);
        metas.sort_by(|a, b| a.time_range.start().total_cmp(&b.time_range.start()));
        Ok((self.duration, metas))
    }

    fn flush_extract_batch(&mut self, out_dir: &Path) -> Result<()> {
        if self.rgb_batch.is_empty() {
            return Ok(());
        }

        // Reuse the buffer's capacity: mem::take would leave an empty Vec that
        // reallocates as the next batch fills.
        let batch = mem::replace(&mut self.rgb_batch, Vec::with_capacity(self.batch_frames));
        let score_edge = self.cfg.score_edge;
        let deduper = &mut self.deduper;
        // Run the per-frame parallel work (grayscale, pHash) on the configured
        // pool so `--threads` caps it too.
        let judged = self
            .pool
            .install(|| deduper.judge(extract::candidate_frames_from_rgbs(batch, score_edge)));

        for (frame, kept) in judged {
            self.duration = self.duration.max(frame.ts + 1.0 / self.cfg.base_fps);
            if kept {
                self.note_kept(&frame, out_dir)?;
                self.push_chunked_frame(frame);
            } else {
                self.note_dropped(frame.ts);
            }
        }
        Ok(())
    }

    /// Record a kept frame for static detection: close out a static stretch if
    /// the gap since the previous kept frame qualifies, then make this frame
    /// the new representative candidate.
    fn note_kept(&mut self, frame: &CandidateFrame, out_dir: &Path) -> Result<()> {
        let stretch = match &self.last_kept {
            Some((prev_ts, prev_rgb))
                if frame.ts - prev_ts >= self.cfg.chunk_seconds
                    && self.drops_since_kept >= MIN_STATIC_DROPS =>
            {
                Some((*prev_ts, prev_rgb.clone()))
            }
            _ => None,
        };
        if let Some((start, rgb)) = stretch {
            self.emit_static(start, frame.ts, &rgb, out_dir)?;
        }
        self.last_kept = Some((frame.ts, frame.rgb.clone()));
        self.drops_since_kept = 0;
        Ok(())
    }

    fn note_dropped(&mut self, ts: f32) {
        self.drops_since_kept += 1;
        self.last_dropped_ts = ts;
    }

    /// Close out a static stretch that runs to the end of the stream, bounded
    /// by the last dropped duplicate rather than a following kept frame.
    fn finish_trailing_static(&mut self, out_dir: &Path) -> Result<()> {
        if self.drops_since_kept < MIN_STATIC_DROPS {
            return Ok(());
        }
        let Some((prev_ts, prev_rgb)) = self.last_kept.take() else {
            return Ok(());
        };
        if self.last_dropped_ts - prev_ts >= self.cfg.chunk_seconds {
            self.emit_static(prev_ts, self.last_dropped_ts, &prev_rgb, out_dir)?;
        }
        Ok(())
    }

    /// Render and record one static-stretch entry: the last visually-distinct
    /// frame stands in for the whole unchanged `[start, end]` range.
    fn emit_static(&mut self, start: f32, end: f32, rgb: &RgbImage, out_dir: &Path) -> Result<()> {
        self.static_seq += 1;
        let jpeg = compose::render_single(rgb, start, self.cfg.jpeg_quality)?;
        let file = format!("static_{:03}.jpg", self.static_seq);
        if self.env.write_files {
            atomic_write(out_dir, &file, &jpeg)?;
        }
        eprintln!(
            "  static stretch {}-{} collapsed into {file}",
            format_ts(start),
            format_ts(end)
        );
        self.static_metas.push(GridMeta {
            image: super::chunk::grid_image(self.env.write_files, file, &jpeg),
            time_range: TimeRange::new(start, end).expect("static stretch has start <= end"),
            total_frames: 1,
            key_frames: Vec::new(),
            layout: "STATIC",
            subtitles: None,
        });
        Ok(())
    }

    fn push_chunked_frame(&mut self, frame: CandidateFrame) {
        let key = (frame.ts / self.cfg.chunk_seconds).floor() as usize;
        self.pending.entry(key).or_default().push(frame);

        if let Some(hwm) = self.high_water_mark
            && key > hwm
        {
            self.seal_up_to(hwm);
        }
        self.high_water_mark = Some(self.high_water_mark.unwrap_or(key).max(key));
    }

    fn seal_up_to(&mut self, seal_key: usize) {
        let keys: Vec<usize> = self
            .pending
            .keys()
            .copied()
            .take_while(|&k| k <= seal_key)
            .collect();
        for key in keys {
            if let Some(chunk) = self.pending.remove(&key) {
                self.carry.extend(chunk);
            }
            // Lazy sealing: only spawn once the pool can fill a grid, so a
            // sparse chunk merges into the following ones instead of wasting a
            // half-empty grid. Dense chunks reach capacity alone and behave as
            // before.
            if self.carry.len() >= layout::grid_capacity(&self.env.spec, self.env.aspect) {
                self.spawn_carry();
            }
        }
    }

    fn seal_all(&mut self) {
        self.seal_up_to(usize::MAX);
        // Whatever is left becomes the final (possibly under-full) grid.
        self.spawn_carry();
    }

    fn spawn_carry(&mut self) {
        if self.carry.is_empty() {
            return;
        }
        let chunk = mem::take(&mut self.carry);
        let tail = chunk.last().map(|f| f.gray.clone());
        let prev_tail = mem::replace(&mut self.prev_chunk_tail, tail);
        let ci = self.next_chunk_ci;
        self.next_chunk_ci += 1;
        self.spawn_chunk(ci, chunk, prev_tail);
    }

    fn spawn_chunk(&mut self, ci: usize, chunk: Vec<CandidateFrame>, prev_tail: Option<GrayImage>) {
        self.outstanding += 1;
        let tx = self.tx.as_ref().expect("sender lives until finish").clone();
        let env = self.env;
        self.pool.spawn_fifo(move || {
            let result =
                panic::catch_unwind(AssertUnwindSafe(|| process_chunk(ci, chunk, prev_tail, env)))
                    .unwrap_or_else(|panic| Err(panic_to_anyhow(panic)));
            let _ = tx.send((ci, result));
        });
    }

    fn drain_results(&mut self, out_dir: &Path, max_in_flight: Option<usize>) -> Result<()> {
        loop {
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
                    atomic_write(out_dir, &file, &result.jpeg)?;
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

pub fn extract_batch_frames(cfg: &Config) -> usize {
    ((cfg.base_fps * cfg.chunk_seconds).round().max(1.0) as usize)
        .clamp(MIN_EXTRACT_BATCH_FRAMES, MAX_EXTRACT_BATCH_FRAMES)
}

fn panic_to_anyhow(payload: Box<dyn std::any::Any + Send>) -> anyhow::Error {
    use anyhow::anyhow;
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
    use crate::config::{Overrides, StrategyOpts};
    use crate::types::{Detail, GridImage, GridMeta, Quality};
    use image::{Rgb, RgbImage};

    fn base_config() -> Config {
        Config::new(
            2000,
            Quality::High,
            Detail::Standard,
            None,
            &Overrides::default(),
            StrategyOpts::default(),
        )
        .expect("default config builds")
    }

    /// Config whose grid capacity is 4 for the square 64x64 test frames
    /// (1024² / 512² = 4), so four distinct frames fill a grid exactly.
    fn small_grid_config() -> Config {
        let mut cfg = Config::new(
            1024,
            Quality::High,
            Detail::Standard,
            None,
            &Overrides::default(),
            StrategyOpts::default(),
        )
        .expect("config builds");
        cfg.chunk_seconds = 60.0;
        cfg.base_fps = 1.0;
        cfg
    }

    /// Four distinct frames inside chunk `key` — exactly one grid's worth.
    fn full_chunk(key: u32) -> Vec<(f32, u32)> {
        (0..4)
            .map(|i| ((key * 60 + i * 10) as f32, key * 10 + i))
            .collect()
    }

    /// Build a 64x64 frame whose pixels form a seed-dependent diagonal ramp, so
    /// distinct seeds hash far enough apart to survive perceptual dedup.
    fn distinct_frame(ts: f32, seed: u32) -> extract::RgbFrame {
        let rgb = RgbImage::from_fn(64, 64, |x, y| {
            let v = ((x * (3 + seed) + y * (7 + 2 * seed)) % 256) as u8;
            Rgb([v, v, v])
        });
        extract::RgbFrame { ts, rgb }
    }

    fn pool(threads: usize) -> rayon::ThreadPool {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("test pool builds")
    }

    /// Drive a full push→finish cycle with `write_files = false` and return the
    /// grid metadata in emission order.
    fn run(cfg: &Config, parallelism: usize, frames: &[(f32, u32)]) -> Vec<GridMeta> {
        run_on(cfg, &pool(2), parallelism, frames)
    }

    fn run_on(
        cfg: &Config,
        pool: &rayon::ThreadPool,
        parallelism: usize,
        frames: &[(f32, u32)],
    ) -> Vec<GridMeta> {
        let out_dir = Path::new("out");
        let mut state = ProcessState::new(cfg, pool, false, parallelism);
        for &(ts, seed) in frames {
            state
                .push_rgb(distinct_frame(ts, seed), out_dir)
                .expect("push frame");
        }
        let (_duration, grid_metas) = state.finish(out_dir).expect("finish");
        grid_metas
    }

    #[test]
    fn seals_full_chunks_by_time() {
        let cfg = small_grid_config();
        let frames: Vec<_> = full_chunk(0).into_iter().chain(full_chunk(1)).collect();

        let metas = run(&cfg, 2, &frames);

        assert_eq!(metas.len(), 2, "each full chunk becomes its own grid");
        let counts: Vec<usize> = metas.iter().map(|m| m.total_frames).collect();
        assert_eq!(counts, vec![4, 4], "frames per grid");
        let starts: Vec<f32> = metas.iter().map(|m| m.time_range.start()).collect();
        assert!(
            starts.windows(2).all(|w| w[0] < w[1]),
            "grid time ranges ascending: {starts:?}"
        );
    }

    #[test]
    fn merges_sparse_chunks_until_capacity() {
        let cfg = small_grid_config();

        // One surviving frame per chunk (capacity 4): the time chunk is a
        // throttle, not a floor — sparse chunks merge into one full grid.
        let metas = run(&cfg, 2, &[(0.0, 0), (65.0, 1), (130.0, 2), (195.0, 3)]);

        assert_eq!(metas.len(), 1, "sparse chunks merge into a single grid");
        assert_eq!(metas[0].total_frames, 4);
        assert_eq!(
            (metas[0].time_range.start(), metas[0].time_range.end()),
            (0.0, 195.0),
            "range spans merged chunks"
        );
    }

    #[test]
    fn emits_grids_in_time_order_with_single_slot() {
        let cfg = small_grid_config();
        let frames: Vec<_> = (0..4).flat_map(full_chunk).collect();

        let metas = run(&cfg, 1, &frames);

        assert_eq!(metas.len(), 4);
        let starts: Vec<f32> = metas.iter().map(|m| m.time_range.start()).collect();
        assert!(
            starts.windows(2).all(|w| w[0] < w[1]),
            "grids emitted in strictly ascending time order: {starts:?}"
        );
    }

    #[test]
    fn completes_on_a_single_thread_pool() {
        // Regression: the orchestrator once ran inside the pool and blocked its
        // only thread on rx.recv(), deadlocking the queued chunk jobs. On a
        // regression this test hangs rather than fails.
        let cfg = small_grid_config();
        let frames: Vec<_> = (0..4).flat_map(full_chunk).collect();

        let metas = run_on(&cfg, &pool(1), 1, &frames);

        assert_eq!(metas.len(), 4, "all grids complete without deadlock");
        let starts: Vec<f32> = metas.iter().map(|m| m.time_range.start()).collect();
        assert!(
            starts.windows(2).all(|w| w[0] < w[1]),
            "grids emitted in strictly ascending time order: {starts:?}"
        );
    }

    #[test]
    fn clamps_extract_batch_to_bounds() {
        let mut cfg = base_config();

        cfg.base_fps = 0.1;
        cfg.chunk_seconds = 1.0;
        assert_eq!(extract_batch_frames(&cfg), MIN_EXTRACT_BATCH_FRAMES);

        cfg.base_fps = 100.0;
        cfg.chunk_seconds = 100.0;
        assert_eq!(extract_batch_frames(&cfg), MAX_EXTRACT_BATCH_FRAMES);

        cfg.base_fps = 1.0;
        cfg.chunk_seconds = 100.0;
        assert_eq!(extract_batch_frames(&cfg), 100);
    }

    #[test]
    fn collapses_static_stretch_into_single_entry() {
        let mut cfg = base_config();
        cfg.chunk_seconds = 60.0;
        cfg.base_fps = 1.0;

        // Two distinct frames, then 69 duplicates of the second (dedup drops
        // them), then a distinct frame at 75s: a 74s static stretch.
        let mut frames = vec![(0.0, 0), (1.0, 1)];
        frames.extend((2..=70).map(|t| (t as f32, 1)));
        frames.push((75.0, 2));

        let metas = run(&cfg, 2, &frames);

        // The three kept frames merge into one (sparse) grid; the static
        // stretch gets its own entry, sorted by start time.
        assert_eq!(metas.len(), 2, "merged grid plus static entry");
        let stat = &metas[1];
        assert_eq!(stat.layout, "STATIC");
        assert_eq!(
            (stat.time_range.start(), stat.time_range.end()),
            (1.0, 75.0),
            "stretch spans kept-frame gap"
        );
        assert_eq!(stat.total_frames, 1, "one representative frame");
        assert!(
            matches!(stat.image, GridImage::Base64(_)),
            "json mode embeds the representative"
        );
    }

    #[test]
    fn closes_trailing_static_stretch_at_finish() {
        let mut cfg = base_config();
        cfg.chunk_seconds = 60.0;
        cfg.base_fps = 1.0;

        // The video ends inside the static stretch: no following kept frame,
        // so the last dropped duplicate bounds the range.
        let mut frames = vec![(0.0, 0), (1.0, 1)];
        frames.extend((2..=70).map(|t| (t as f32, 1)));

        let metas = run(&cfg, 2, &frames);

        assert_eq!(metas.len(), 2, "grid plus trailing static entry");
        assert_eq!(metas[1].layout, "STATIC");
        assert_eq!(
            (metas[1].time_range.start(), metas[1].time_range.end()),
            (1.0, 70.0)
        );
    }

    #[test]
    fn sparse_sampling_gap_is_not_static() {
        let mut cfg = base_config();
        cfg.chunk_seconds = 60.0;
        cfg.base_fps = 1.0;

        // Big gaps between kept frames but zero dropped duplicates inside them
        // (sparse keyframe sampling): no evidence of stillness, no static entry.
        let metas = run(&cfg, 2, &[(0.0, 0), (65.0, 1), (130.0, 2)]);

        assert_eq!(metas.len(), 1, "sparse frames merge into one grid");
        assert!(
            metas.iter().all(|m| m.layout != "STATIC"),
            "no static entries without dropped-duplicate evidence"
        );
    }

    #[test]
    fn maps_worker_panic_payloads_to_messages() {
        let from_str = panic_to_anyhow(Box::new("boom"));
        let from_string = panic_to_anyhow(Box::new(String::from("boom")));
        let from_other = panic_to_anyhow(Box::new(42u8));

        assert!(from_str.to_string().contains("worker panic"));
        assert!(from_string.to_string().contains("worker panic"));
        assert!(from_other.to_string().contains("worker panic"));
    }
}
