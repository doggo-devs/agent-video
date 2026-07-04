//! ProcessState: the streaming state machine that consumes judged candidate
//! frames, chunks them by time, and spawns parallel chunk-processing jobs.
//!
//! Time chunks are a throttle, not a floor: a chunk whose surviving frames
//! cannot fill a grid is merged forward into the next chunk(s), so sparse
//! content produces fewer, fuller grids instead of one half-empty grid per
//! chunk.
//!
//! Static-stretch detection lives in [`super::stretch`]; this module feeds it
//! the dedup verdicts and renders qualifying stretches as a single
//! representative image plus the covered time range instead of empty grids.

use std::collections::BTreeMap;
use std::mem;
use std::num::NonZeroUsize;
use std::panic::{self, AssertUnwindSafe};

use anyhow::Result;
use image::GrayImage;

use crate::config::Config;
use crate::types::{CandidateFrame, GridMeta, TimeRange, format_ts};
use crate::{compose, layout};

use super::chunk::{GridResult, ProcessChunkEnv, process_chunk};
use super::jobs::OrderedJobs;
use super::output::OutputSession;
use super::stretch::{Stretch, StretchDetector};

/// Lower bound for an extraction batch.
const MIN_EXTRACT_BATCH_FRAMES: usize = 32;

/// Upper bound for an extraction batch.
const MAX_EXTRACT_BATCH_FRAMES: usize = 240;

pub struct ProcessState<'a> {
    cfg: &'a Config,
    /// The run's output session: incremental grid/static writes go through it,
    /// so file-vs-JSON dispatch lives in one place.
    session: &'a OutputSession<'a>,
    /// Worker pool for chunk jobs. The state's own methods run on the
    /// caller's thread and may block on job completions, so they must never
    /// execute on a pool thread — a single-thread pool would deadlock.
    pool: &'a rayon::ThreadPool,
    pub(super) env: ProcessChunkEnv,
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
    /// Timestamp of the last pushed frame, enforcing the monotonic-arrival
    /// contract of [`Self::push_rgb`].
    last_ts: Option<f32>,
    /// Chunk jobs in flight: assigns chunk indexes, caps buffered chunks at
    /// `max_in_flight`, and hands results back in chunk order.
    jobs: OrderedJobs<Result<GridResult>>,
    /// Static-stretch detector, fed every dedup verdict in timestamp order.
    stretches: StretchDetector,
    pub(super) duration: f32,
    pub(super) grid_metas: Vec<GridMeta>,
    static_seq: usize,
    static_metas: Vec<GridMeta>,
}

impl<'a> ProcessState<'a> {
    pub fn new(
        cfg: &'a Config,
        pool: &'a rayon::ThreadPool,
        session: &'a OutputSession<'a>,
        parallelism: NonZeroUsize,
    ) -> Self {
        Self {
            cfg,
            session,
            pool,
            env: ProcessChunkEnv {
                spec: cfg.canvas_spec(),
                aspect: 1.0,
                weights: cfg.weights(),
                min_gap_floor: cfg.min_gap(),
                jpeg_quality: cfg.jpeg_quality(),
                hero_border_px: cfg.hero_border_px(),
            },
            pending: BTreeMap::new(),
            carry: Vec::new(),
            prev_chunk_tail: None,
            high_water_mark: None,
            last_ts: None,
            jobs: OrderedJobs::new(parallelism),
            stretches: StretchDetector::new(cfg.chunk_seconds()),
            duration: 0.0,
            grid_metas: Vec::new(),
            static_seq: 0,
            static_metas: Vec::new(),
        }
    }

    pub fn set_aspect(&mut self, aspect: f32) {
        self.env.aspect = aspect;
    }

    /// Record one judged frame: a kept frame joins its time chunk, a dropped
    /// one feeds static-stretch evidence; both extend the observed duration.
    ///
    /// Timestamps must be finite and arrive in non-decreasing order: chunk
    /// keying, high-water-mark sealing, and static detection all assume a
    /// monotonic stream (the single decode pass emits frames in presentation
    /// order, so this holds structurally). A violation is rejected here
    /// rather than silently mixing old frames into the wrong chunk.
    pub fn push_judged(&mut self, frame: CandidateFrame, kept: bool) -> Result<()> {
        if !frame.ts.is_finite() {
            anyhow::bail!("frame timestamp must be finite: got {}", frame.ts);
        }
        if let Some(last) = self.last_ts
            && frame.ts < last
        {
            anyhow::bail!(
                "frame timestamps must be non-decreasing: got {} after {last}",
                frame.ts
            );
        }
        self.last_ts = Some(frame.ts);
        self.duration = self.duration.max(frame.ts + 1.0 / self.cfg.base_fps());
        if kept {
            if let Some(stretch) = self.stretches.note_kept(frame.ts, &frame.rgb) {
                self.emit_static(stretch)?;
            }
            self.push_chunked_frame(frame)
        } else {
            self.stretches.note_dropped(frame.ts);
            Ok(())
        }
    }

    pub fn finish(mut self) -> Result<(f32, Vec<GridMeta>)> {
        if let Some(stretch) = self.stretches.finish() {
            self.emit_static(stretch)?;
        }
        self.seal_all()?;
        for result in self.jobs.finish()? {
            self.emit_grid(result?)?;
        }

        // Interleave static entries with grids by start time. Both lists are
        // already time-ascending; a static range starts inside the grid that
        // holds its representative, so it sorts right after that grid.
        let mut metas = mem::take(&mut self.grid_metas);
        metas.append(&mut self.static_metas);
        metas.sort_by(|a, b| a.time_range.start().total_cmp(&b.time_range.start()));
        Ok((self.duration, metas))
    }

    /// Render and record one static-stretch entry: the last visually-distinct
    /// frame stands in for the whole unchanged range.
    fn emit_static(&mut self, stretch: Stretch) -> Result<()> {
        let Stretch { start, end, rgb } = stretch;
        self.static_seq += 1;
        let jpeg = compose::render_single(&rgb, start, self.cfg.jpeg_quality())?;
        let file = format!("static_{:03}.jpg", self.static_seq);
        let image = self.session.emit_image(&file, &jpeg)?;
        eprintln!(
            "  static stretch {}-{} collapsed into {file}",
            format_ts(start),
            format_ts(end)
        );
        self.static_metas.push(GridMeta {
            image,
            time_range: TimeRange::new(start, end).expect("static stretch has start <= end"),
            total_frames: 1,
            key_frames: Vec::new(),
            layout: "STATIC",
            subtitles: crate::types::GridSubtitles::Unresolved,
        });
        Ok(())
    }

    fn push_chunked_frame(&mut self, frame: CandidateFrame) -> Result<()> {
        let key = (frame.ts / self.cfg.chunk_seconds()).floor() as usize;
        self.pending.entry(key).or_default().push(frame);

        if let Some(hwm) = self.high_water_mark
            && key > hwm
        {
            self.seal_up_to(hwm)?;
        }
        self.high_water_mark = Some(self.high_water_mark.unwrap_or(key).max(key));
        Ok(())
    }

    fn seal_up_to(&mut self, seal_key: usize) -> Result<()> {
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
                self.spawn_carry()?;
            }
        }
        Ok(())
    }

    fn seal_all(&mut self) -> Result<()> {
        self.seal_up_to(usize::MAX)?;
        // Whatever is left becomes the final (possibly under-full) grid.
        self.spawn_carry()
    }

    fn spawn_carry(&mut self) -> Result<()> {
        if self.carry.is_empty() {
            return Ok(());
        }
        let chunk = mem::take(&mut self.carry);
        let tail = chunk.last().map(|f| f.gray.clone());
        let prev_tail = mem::replace(&mut self.prev_chunk_tail, tail);
        self.spawn_chunk(chunk, prev_tail)
    }

    fn spawn_chunk(
        &mut self,
        chunk: Vec<CandidateFrame>,
        prev_tail: Option<GrayImage>,
    ) -> Result<()> {
        // `jobs` enforces the in-flight cap at this spawn boundary: one
        // extraction batch can seal several chunks in a row, so waiting for
        // room here (rather than once per pushed frame) is what actually
        // bounds buffered chunks.
        let env = self.env;
        let results = self.jobs.spawn(self.pool, move |ci| {
            panic::catch_unwind(AssertUnwindSafe(|| {
                process_chunk(ci, chunk, prev_tail, env)
            }))
            .unwrap_or_else(|panic| Err(panic_to_anyhow(panic)))
        })?;
        for result in results {
            self.emit_grid(result?)?;
        }
        Ok(())
    }

    /// Emit one finished grid and record its metadata. Grids arrive from
    /// `jobs` in chunk order, so emission order is chunk order. The grid's
    /// file name is assigned here, at the single emission point, rather than
    /// inside the worker.
    fn emit_grid(&mut self, result: GridResult) -> Result<()> {
        let file = format!("grid_{:03}.jpg", result.ci + 1);
        let image = self.session.emit_image(&file, &result.jpeg)?;
        self.grid_metas.push(result.draft.into_meta(image));
        Ok(())
    }
}

/// Frames per extraction batch for a stream sampled at `fps` and chunked every
/// `chunk_seconds`, clamped to the batch bounds. Parameterized on fps because
/// `process` and `extract` sample at different densities.
pub fn extract_batch_frames(fps: f32, chunk_seconds: f32) -> usize {
    ((fps * chunk_seconds).round().max(1.0) as usize)
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
    use crate::{dedup, extract};
    use image::{Rgb, RgbImage};

    /// Config with explicit 60s chunks at 1 fps — the values the tests'
    /// timestamp math assumes — set through [`Overrides`], the same door the
    /// CLI uses (Config fields are private by design).
    fn config_with_edge(max_edge: u32) -> Config {
        let ov = Overrides {
            chunk_seconds: Some(60.0),
            base_fps: Some(1.0),
            ..Default::default()
        };
        Config::new(
            max_edge,
            Quality::High,
            Detail::Standard,
            None,
            &ov,
            StrategyOpts::default(),
        )
        .expect("test config builds")
    }

    fn base_config() -> Config {
        config_with_edge(2000)
    }

    /// Config whose grid capacity is 4 for the square 64x64 test frames
    /// (1024² / 512² = 4), so four distinct frames fill a grid exactly.
    fn small_grid_config() -> Config {
        config_with_edge(1024)
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

    /// Judge synthetic frames through the real dedup machinery, exactly as
    /// the candidate stream does in production.
    fn judge(cfg: &Config, frames: &[(f32, u32)]) -> Vec<(CandidateFrame, bool)> {
        let rgbs = frames
            .iter()
            .map(|&(ts, seed)| distinct_frame(ts, seed))
            .collect();
        let candidates = extract::candidate_frames_from_rgbs(rgbs, cfg.score_edge());
        dedup::Deduper::new(cfg.hash_threshold(), cfg.dedup_window()).judge(candidates)
    }

    /// Drive a full push→finish cycle in JSON mode and return the grid
    /// metadata in emission order.
    fn run(cfg: &Config, parallelism: usize, frames: &[(f32, u32)]) -> Vec<GridMeta> {
        run_on(cfg, &pool(2), parallelism, frames)
    }

    /// A JSON-mode session for tests: opening it touches no files.
    fn json_session() -> OutputSession<'static> {
        OutputSession::open(std::path::Path::new("out"), super::super::OutputMode::Json)
            .expect("json session opens without touching disk")
    }

    fn run_on(
        cfg: &Config,
        pool: &rayon::ThreadPool,
        parallelism: usize,
        frames: &[(f32, u32)],
    ) -> Vec<GridMeta> {
        let session = json_session();
        let parallelism = NonZeroUsize::new(parallelism).expect("test parallelism is nonzero");
        let mut state = ProcessState::new(cfg, pool, &session, parallelism);
        for (frame, kept) in judge(cfg, frames) {
            state.push_judged(frame, kept).expect("push frame");
        }
        let (_duration, grid_metas) = state.finish().expect("finish");
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
        assert_eq!(extract_batch_frames(0.1, 1.0), MIN_EXTRACT_BATCH_FRAMES);
        assert_eq!(extract_batch_frames(100.0, 100.0), MAX_EXTRACT_BATCH_FRAMES);
        assert_eq!(extract_batch_frames(1.0, 100.0), 100);
    }

    #[test]
    fn bounds_buffered_chunks_within_one_batch_flush() {
        // 16 chunks' worth of frames seal one full chunk after another. With
        // --in-flight 1 the spawn boundary must wait for room each time;
        // buffered chunks may never exceed 1.
        let cfg = small_grid_config();
        let frames: Vec<_> = (0..16).flat_map(full_chunk).collect();

        let pool = pool(2);
        let session = json_session();
        let one = NonZeroUsize::new(1).expect("1 is nonzero");
        let mut state = ProcessState::new(&cfg, &pool, &session, one);
        for (frame, kept) in judge(&cfg, &frames) {
            state.push_judged(frame, kept).expect("push frame");
        }
        assert!(
            state.jobs.peak_buffered() <= 1,
            "in-flight cap violated during batch flush: peak {}",
            state.jobs.peak_buffered()
        );

        let (_duration, metas) = state.finish().expect("finish");
        assert_eq!(metas.len(), 16, "every sealed chunk still becomes a grid");
    }

    #[test]
    fn collapses_static_stretch_into_single_entry() {
        let cfg = base_config();

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
        let cfg = base_config();

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
        let cfg = base_config();

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
    fn rejects_backwards_timestamps() {
        let cfg = small_grid_config();
        let pool = pool(2);
        let session = json_session();
        let one = NonZeroUsize::new(1).expect("1 is nonzero");
        let mut state = ProcessState::new(&cfg, &pool, &session, one);

        let mut frames = judge(&cfg, &[(10.0, 0), (5.0, 1)]).into_iter();
        let (first, kept) = frames.next().expect("two frames judged");
        state.push_judged(first, kept).expect("first push");
        let (backwards, kept) = frames.next().expect("two frames judged");
        state
            .push_judged(backwards, kept)
            .expect_err("a timestamp going backwards must be rejected");
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
