//! ProcessState: the streaming state machine that buffers RGB frames, deduplicates,
//! chunks by time, and spawns parallel chunk-processing jobs.

use std::collections::BTreeMap;
use std::mem;
use std::panic::{self, AssertUnwindSafe};
use std::path::Path;
use std::sync::mpsc::{self, Receiver, Sender, TryRecvError};

use anyhow::{Context, Result};

use crate::config::Config;
use crate::types::{GridMeta, ScoredFrame};
use crate::{dedup, extract};

use super::chunk::{process_chunk, ChunkMsg, GridResult, ProcessChunkEnv};
use super::output::atomic_write;

/// Lower bound for an extraction batch.
const MIN_EXTRACT_BATCH_FRAMES: usize = 32;

/// Upper bound for an extraction batch.
const MAX_EXTRACT_BATCH_FRAMES: usize = 240;

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
    pending: BTreeMap<usize, Vec<ScoredFrame>>,
    high_water_mark: Option<usize>,
    auto_seal: bool,
    next_chunk_ci: usize,
    tx: Option<Sender<ChunkMsg>>,
    rx: Receiver<ChunkMsg>,
    ready: BTreeMap<usize, Result<GridResult>>,
    outstanding: usize,
    next_emit: usize,
    pub(super) duration: f32,
    pub(super) grid_metas: Vec<GridMeta>,
}

impl<'a> ProcessState<'a> {
    pub fn new(
        cfg: &'a Config,
        pool: &'a rayon::ThreadPool,
        write_files: bool,
        parallelism: usize,
        auto_seal: bool,
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
            high_water_mark: None,
            auto_seal,
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

    pub fn set_aspect(&mut self, aspect: f32) {
        self.env.aspect = aspect;
    }

    pub fn push_rgb(&mut self, frame: extract::RgbFrame, out_dir: &Path) -> Result<()> {
        self.rgb_batch.push(frame);
        if self.rgb_batch.len() == self.batch_frames {
            self.flush_extract_batch();
            self.drain_results(out_dir, Some(self.max_in_flight))?;
        }
        Ok(())
    }

    pub fn finish(mut self, out_dir: &Path) -> Result<(f32, Vec<GridMeta>)> {
        self.flush_extract_batch();
        self.seal_all();
        self.tx = None;
        self.drain_results(out_dir, Some(0))?;
        Ok((self.duration, self.grid_metas))
    }

    fn flush_extract_batch(&mut self) {
        if self.rgb_batch.is_empty() {
            return;
        }

        let batch = mem::take(&mut self.rgb_batch);
        let score_edge = self.cfg.score_edge;
        let deduper = &mut self.deduper;
        // Run the per-frame parallel work (grayscale, pHash) on the configured
        // pool so `--threads` caps it too.
        let kept = self
            .pool
            .install(|| deduper.filter(extract::scored_frames_from_rgbs(batch, score_edge)));
        for frame in kept {
            self.duration = frame.ts + 1.0 / self.cfg.base_fps;
            self.push_chunked_frame(frame);
        }
    }

    fn push_chunked_frame(&mut self, frame: ScoredFrame) {
        let key = (frame.ts / self.cfg.chunk_seconds).floor() as usize;
        self.pending.entry(key).or_default().push(frame);

        if self.auto_seal
            && let Some(hwm) = self.high_water_mark
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
                let ci = self.next_chunk_ci;
                self.next_chunk_ci += 1;
                self.spawn_chunk(ci, chunk);
            }
        }
    }

    fn seal_all(&mut self) {
        self.seal_up_to(usize::MAX);
    }

    fn spawn_chunk(&mut self, ci: usize, chunk: Vec<ScoredFrame>) {
        self.outstanding += 1;
        let tx = self.tx.as_ref().expect("sender lives until finish").clone();
        let env = self.env;
        self.pool.spawn_fifo(move || {
            let result = panic::catch_unwind(AssertUnwindSafe(|| process_chunk(ci, chunk, env)))
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
    use crate::types::{Detail, GridMeta, Quality};
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
    fn run(cfg: &Config, auto_seal: bool, parallelism: usize, frames: &[(f32, u32)]) -> Vec<GridMeta> {
        run_on(cfg, &pool(2), auto_seal, parallelism, frames)
    }

    fn run_on(
        cfg: &Config,
        pool: &rayon::ThreadPool,
        auto_seal: bool,
        parallelism: usize,
        frames: &[(f32, u32)],
    ) -> Vec<GridMeta> {
        let out_dir = Path::new("out");
        let mut state = ProcessState::new(cfg, pool, false, parallelism, auto_seal);
        for &(ts, seed) in frames {
            state
                .push_rgb(distinct_frame(ts, seed), out_dir)
                .expect("push frame");
        }
        let (_duration, grid_metas) = state.finish(out_dir).expect("finish");
        grid_metas
    }

    #[test]
    fn splits_frames_into_chunks_by_time() {
        let mut cfg = base_config();
        cfg.chunk_seconds = 60.0;
        cfg.base_fps = 1.0;

        let metas = run(&cfg, true, 2, &[(0.0, 0), (30.0, 1), (65.0, 2), (125.0, 3)]);

        assert_eq!(metas.len(), 3, "three time chunks");
        let counts: Vec<usize> = metas.iter().map(|m| m.total_frames).collect();
        assert_eq!(counts, vec![2, 1, 1], "frames per chunk");
        let starts: Vec<f32> = metas.iter().map(|m| m.time_range.0).collect();
        assert!(
            starts.windows(2).all(|w| w[0] < w[1]),
            "chunk time ranges ascending: {starts:?}"
        );
    }

    #[test]
    fn emits_grids_in_time_order_with_single_slot() {
        let mut cfg = base_config();
        cfg.chunk_seconds = 60.0;
        cfg.base_fps = 1.0;

        let metas = run(&cfg, true, 1, &[(0.0, 0), (65.0, 1), (130.0, 2), (195.0, 3)]);

        assert_eq!(metas.len(), 4);
        let starts: Vec<f32> = metas.iter().map(|m| m.time_range.0).collect();
        assert!(
            starts.windows(2).all(|w| w[0] < w[1]),
            "grids emitted in strictly ascending time order: {starts:?}"
        );
    }

    #[test]
    fn merges_late_frame_into_its_chunk_without_auto_seal() {
        let mut cfg = base_config();
        cfg.chunk_seconds = 60.0;
        cfg.base_fps = 1.0;

        // ts=65 (chunk 1) arrives before the late ts=30 (chunk 0), mimicking an
        // adaptive fill stream that backfills an earlier time chunk.
        let metas = run(&cfg, false, 2, &[(65.0, 0), (30.0, 1)]);

        assert_eq!(metas.len(), 2, "no duplicate chunk for the late frame");
        assert_eq!(
            metas[0].time_range.0, 30.0,
            "late frame landed in the first time chunk"
        );
    }

    #[test]
    fn completes_on_a_single_thread_pool() {
        // Regression: the orchestrator once ran inside the pool and blocked its
        // only thread on rx.recv(), deadlocking the queued chunk jobs. On a
        // regression this test hangs rather than fails.
        let mut cfg = base_config();
        cfg.chunk_seconds = 60.0;
        cfg.base_fps = 1.0;

        let metas = run_on(
            &cfg,
            &pool(1),
            true,
            1,
            &[(0.0, 0), (65.0, 1), (130.0, 2), (195.0, 3)],
        );

        assert_eq!(metas.len(), 4, "all chunks complete without deadlock");
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
    fn maps_worker_panic_payloads_to_messages() {
        let from_str = panic_to_anyhow(Box::new("boom"));
        let from_string = panic_to_anyhow(Box::new(String::from("boom")));
        let from_other = panic_to_anyhow(Box::new(42u8));

        assert!(from_str.to_string().contains("worker panic"));
        assert!(from_string.to_string().contains("worker panic"));
        assert!(from_other.to_string().contains("worker panic"));
    }
}
