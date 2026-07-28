// SPDX-License-Identifier: Apache-2.0
//! ProcessState: the shell around the pure [`Chunker`] core — it consumes
//! judged candidate frames, lets the chunker turn them into chunk/static
//! events, and interprets those events against the effectful world: parallel
//! chunk jobs on the rayon pool and incremental writes through the output
//! session.
//!
//! The division of labor is deliberate: everything about *where frames go on
//! the timeline* (chunk keying, sparse-chunk merging, static stretches,
//! broken-timeline accounting) lives in [`super::chunker`] and is unit-tested
//! there without a pool or a session; this module owns only the effects.

use std::mem;
use std::num::NonZeroUsize;
use std::panic::{self, AssertUnwindSafe};
use std::sync::Arc;

use image::GrayImage;

use crate::config::Config;
use crate::error::Result;
use crate::event::EventSink;
use crate::types::{Aspect, CandidateFrame, GridMeta, LayoutKind};
use crate::{compose, layout};

use super::chunk::{GridResult, ProcessChunkEnv, process_chunk};
use super::chunker::{Chunker, Event};
use super::jobs::OrderedJobs;
use super::output::OutputSession;
use super::stretch::Stretch;

/// Lower bound for an extraction batch.
const MIN_EXTRACT_BATCH_FRAMES: usize = 32;

/// Upper bound for an extraction batch.
const MAX_EXTRACT_BATCH_FRAMES: usize = 240;

/// One chunk worker's outcome: the caught unwind of a panicking worker, or
/// its typed result. The panic is a value only for the trip back to the
/// orchestrator thread, where [`worker_result`] resumes it — a bug never
/// masquerades as a coded, recoverable error.
type ChunkOutcome = std::thread::Result<Result<GridResult>>;

/// The effectful `process` orchestrator: judged frames in, grids and static
/// entries out through the run's output session.
pub struct ProcessState<'a> {
    cfg: &'a Config,
    /// The run's output session: incremental grid/static writes go through it,
    /// so file-vs-JSON dispatch lives in one place.
    session: &'a OutputSession<'a>,
    /// Worker pool for chunk jobs. The state's own methods run on the
    /// caller's thread and may block on job completions, so they must never
    /// execute on a pool thread — a single-thread pool would deadlock.
    pool: &'a rayon::ThreadPool,
    env: ProcessChunkEnv,
    /// The pure chunking core; this shell only interprets its events.
    chunker: Chunker,
    /// Chunk jobs in flight: assigns chunk indexes, caps buffered chunks at
    /// `max_in_flight`, and hands results back in chunk order.
    jobs: OrderedJobs<ChunkOutcome>,
    grid_metas: Vec<GridMeta>,
    static_seq: usize,
    static_metas: Vec<GridMeta>,
    /// Diagnostic sink: static-stretch collapses report through it as typed
    /// events, so this shell never writes to stderr itself.
    events: &'a dyn EventSink,
}

impl<'a> ProcessState<'a> {
    /// Wire the shell to its run: config, worker pool, output session,
    /// in-flight cap, and diagnostic sink.
    pub fn new(
        cfg: &'a Config,
        pool: &'a rayon::ThreadPool,
        session: &'a OutputSession<'a>,
        parallelism: NonZeroUsize,
        events: &'a dyn EventSink,
    ) -> Self {
        Self {
            cfg,
            session,
            pool,
            env: ProcessChunkEnv {
                spec: cfg.canvas_spec(),
                aspect: Aspect::SQUARE,
                weights: cfg.weights(),
                min_gap_floor: cfg.min_gap(),
                jpeg_quality: cfg.jpeg_quality(),
                hero_border_px: cfg.hero_border_px(),
                hero_scene_diff: cfg.hero_scene_diff(),
                max_heroes: cfg.max_heroes(),
            },
            chunker: Chunker::new(cfg.chunk_seconds(), cfg.base_fps()),
            jobs: OrderedJobs::new(parallelism),
            grid_metas: Vec::new(),
            static_seq: 0,
            static_metas: Vec::new(),
            events,
        }
    }

    /// Update the source aspect ratio once the decoder reports it; grid
    /// capacity is re-derived from it on demand.
    pub fn set_aspect(&mut self, aspect: Aspect) {
        self.env.aspect = aspect;
    }

    /// Record one judged frame, interpreting whatever events it seals off.
    /// Timeline placement lives in [`Chunker::push`]; its monotone-input
    /// precondition is guaranteed by the decode pass's segmenter.
    pub fn push_judged(&mut self, frame: CandidateFrame, kept: bool) -> Result<()> {
        let events = self.chunker.push(frame, kept, self.capacity());
        self.handle_events(events)
    }

    pub fn finish(mut self) -> Result<(f32, Vec<GridMeta>)> {
        let events = self.chunker.finish(self.capacity());
        self.handle_events(events)?;
        for outcome in self
            .jobs
            .finish()
            .expect("chunk jobs catch unwinds and always deliver an outcome")
        {
            self.emit_grid(worker_result(outcome)?)?;
        }

        // Interleave static entries with grids by start time. Both lists are
        // time-ascending; a static range starts inside its grid, so the
        // stable sort keeps it right after that grid.
        let mut metas = mem::take(&mut self.grid_metas);
        metas.append(&mut self.static_metas);
        metas.sort_by(|a, b| a.time_range.start().total_cmp(&b.time_range.start()));
        Ok((self.chunker.duration(), metas))
    }

    /// Current grid capacity: how many carried frames fill one grid. Cheap
    /// and derived on demand because `set_aspect` can change it mid-run.
    fn capacity(&self) -> usize {
        layout::grid_capacity(&self.env.spec, self.env.aspect)
    }

    fn handle_events(&mut self, events: Vec<Event>) -> Result<()> {
        for event in events {
            match event {
                Event::Chunk { frames, prev_tail } => self.spawn_chunk(frames, prev_tail)?,
                Event::Static(stretch) => self.emit_static(stretch)?,
            }
        }
        Ok(())
    }

    /// Render and record one static-stretch entry: the last visually-distinct
    /// frame stands in for the whole unchanged range.
    fn emit_static(&mut self, stretch: Stretch) -> Result<()> {
        let Stretch { range, rgb } = stretch;
        self.static_seq += 1;
        let jpeg = compose::render_single(&rgb, range.start(), self.cfg.jpeg_quality())?;
        let file = format!("static_{:03}.jpg", self.static_seq);
        let image = self.session.emit_image(&file, &jpeg)?;
        self.events.event(crate::event::Event::StaticCollapsed {
            start: range.start(),
            end: range.end(),
            file: &file,
        });
        self.static_metas.push(GridMeta {
            image,
            time_range: range,
            total_frames: 1,
            key_frames: Vec::new(),
            layout: LayoutKind::Static,
        });
        Ok(())
    }

    fn spawn_chunk(
        &mut self,
        chunk: Vec<CandidateFrame>,
        prev_tail: Option<Arc<GrayImage>>,
    ) -> Result<()> {
        // `jobs` enforces the in-flight cap here: one batch can seal several
        // chunks, so waiting for room at the spawn boundary is what bounds
        // buffered chunks.
        let env = self.env;
        // The catch serves unwind builds (tests, downstream consumers on
        // the default profile), where a chunk-worker panic must reach the
        // orchestrator instead of aborting inside rayon; `worker_result`
        // resumes it there. This workspace's release binaries build with
        // `panic = "abort"` and terminate before reaching it.
        let results = self
            .jobs
            .spawn(self.pool, move |ci| {
                panic::catch_unwind(AssertUnwindSafe(|| {
                    process_chunk(ci, chunk, prev_tail, env)
                }))
            })
            .expect("chunk jobs catch unwinds and always deliver an outcome");
        for outcome in results {
            self.emit_grid(worker_result(outcome)?)?;
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

/// Unwrap one worker outcome: a typed error fails the run; a caught panic is
/// a bug and resumes unwinding on the orchestrator thread with its original
/// payload — the typed error channel carries codes only, never bugs.
fn worker_result(outcome: ChunkOutcome) -> Result<GridResult> {
    outcome.unwrap_or_else(|panic| std::panic::resume_unwind(panic))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{Overrides, StrategyOpts};
    use crate::event::Discard;
    use crate::types::{Detail, GridImage, GridMeta, Quality};
    use crate::{dedup, sample};
    use image::{Rgb, RgbImage};

    /// Config with explicit 60s chunks at 1 fps and the high quality tier
    /// (512px cell floor) — the values the tests' timestamp and capacity
    /// math assume — set through [`Overrides`], the same door the CLI uses
    /// (Config fields are private by design).
    fn config_with_edge(max_edge: u32) -> Config {
        let ov = Overrides {
            quality: Some(Quality::High),
            detail: Some(Detail::Standard),
            chunk_seconds: Some(60.0),
            base_fps: Some(1.0),
            ..Default::default()
        };
        Config::new(max_edge, None, &ov, StrategyOpts::default()).expect("test config builds")
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
    fn distinct_frame(ts: f32, seed: u32) -> sample::RgbFrame {
        let rgb = RgbImage::from_fn(64, 64, |x, y| {
            let v = ((x * (3 + seed) + y * (7 + 2 * seed)) % 256) as u8;
            Rgb([v, v, v])
        });
        sample::RgbFrame {
            ts: crate::types::Ts::new(ts).expect("finite test ts"),
            rgb,
        }
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
        let candidates = sample::candidate_frames_from_rgbs(rgbs, cfg.score_edge());
        dedup::Deduper::new(cfg.hash_threshold(), cfg.dedup_window()).judge(candidates)
    }

    /// Drive a full push→finish cycle in JSON mode and return the grid
    /// metadata in emission order.
    fn run(cfg: &Config, parallelism: usize, frames: &[(f32, u32)]) -> Vec<GridMeta> {
        run_on(cfg, &pool(2), parallelism, frames)
    }

    /// A JSON-mode session for tests: opening it touches no files.
    fn json_session() -> OutputSession<'static> {
        OutputSession::open(super::super::OutputSink::JsonStdout)
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
        let mut state = ProcessState::new(cfg, pool, &session, parallelism, &Discard);
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
        let mut state = ProcessState::new(&cfg, &pool, &session, one, &Discard);
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
        assert_eq!(stat.layout, LayoutKind::Static);
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
        assert_eq!(metas[1].layout, LayoutKind::Static);
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
            metas.iter().all(|m| m.layout != LayoutKind::Static),
            "no static entries without dropped-duplicate evidence"
        );
    }

    #[test]
    #[should_panic(expected = "boom")]
    fn resumes_worker_panic_on_the_orchestrator() {
        // A worker panic is a bug: it must re-raise with its original
        // payload, never flatten into a coded, recoverable error.
        let _ = worker_result(Err(Box::new(String::from("boom"))));
    }
}
