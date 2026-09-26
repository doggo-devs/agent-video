// SPDX-License-Identifier: Apache-2.0
//! The `extract` CLI subcommand: high-density sampling of a single time range
//! into one equal-cell grid, frames in time order (no hero emphasis).
//!
//! This is a one-shot batch pipeline, distinct from `process`'s streaming,
//! multi-chunk orchestration in the parent module — it scores and selects
//! within a single competition set instead of per-chunk.

use std::mem;
use std::sync::Arc;

use image::GrayImage;

use crate::config::Config;
use crate::error::{Error, ErrorCode, ErrorContext, Result};
use crate::event::EventSink;
use crate::source::Source;
use crate::subtitle::Subtitles;
use crate::types::{
    Aspect, CandidateFrame, CanvasSpec, GridSubtitles, ScoredFrame, TimeRange, format_ts,
};
use crate::{compose, dedup, layout, sample, score, wire};

use super::candidates::CandidateStream;
use super::chunk::{build_grid_draft, extract_selected};
use super::output::OutputSession;
use super::state::extract_batch_frames;
use super::{OutputSink, SubtitleSource};

/// Run the `extract` pipeline: high-density sampling of `range` into a single
/// equal-cell grid, frames in time order (no hero emphasis).
///
/// Output contract: `grid_extract.jpg` and `metadata.json` only. Subtitles for
/// the requested range travel inside the metadata's `subtitles` field; no
/// `subtitles.txt` is written — the full-video text is not an artifact of a
/// range extraction.
///
/// `events` receives every diagnostic the run emits (decode-open and subtitle
/// warnings) as typed events; pass [`crate::event::Discard`] for silence.
/// `stop` bounds the subtitle-extraction reads on a stalled source; the
/// sampling pass decodes behind its own self-minted token (see
/// [`crate::media::decode::FrameSource::open`]).
///
/// Every failure is a coded [`Error`]: the exit-code contract is stated by
/// the signature, not recovered by downcasting.
pub fn extract_range(
    input: &Source,
    sink: OutputSink<'_>,
    cfg: &Config,
    range: TimeRange,
    subs: SubtitleSource<'_>,
    stop: &crate::media::Stop,
    events: &dyn EventSink,
) -> Result<()> {
    super::validate_input(input)?;
    // Before the expensive sampling pass: subtitle errors must fail fast,
    // not after all the CPU and IO have been spent.
    let subs = Subtitles::resolve(input, subs, stop, events)?;
    run(input, sink, cfg, range, subs.as_ref(), events)
}

/// The decode-and-compose pass of [`extract_range`].
fn run(
    input: &Source,
    sink: OutputSink<'_>,
    cfg: &Config,
    range: TimeRange,
    subs: Option<&Subtitles>,
    events: &dyn EventSink,
) -> Result<()> {
    let session = OutputSession::open(sink)?;

    let fps = cfg.extract_fps();
    // `extract` exposes no --threads flag; the decoder auto-sizes its pool.
    let stream = sample::stream_range(
        input,
        fps,
        range,
        sample::StreamParams {
            cell_min_edge: cfg.min_frame_size(),
            score_edge: cfg.score_edge(),
            threads: None,
            events,
        },
    )?;

    let batch_size = extract_batch_frames(fps, cfg.chunk_seconds());
    let mut stream = CandidateStream::new(stream, batch_size, cfg.score_edge(), None);
    let mut deduper = dedup::Deduper::new(cfg.hash_threshold(), cfg.dedup_window());
    let weights = cfg.weights();
    let spec = cfg.canvas_spec();

    let mut candidate_batch: Vec<CandidateFrame> = Vec::new();
    let mut candidates: Vec<ScoredFrame> = Vec::new();
    // Tail grayscale of the previously scored batch: batches are consecutive
    // in time, so it is the true predecessor of the next batch's first frame.
    let mut prev_tail: Option<Arc<GrayImage>> = None;
    let mut aspect = Aspect::SQUARE;

    while let Some(batch) = stream.next_batch()? {
        if let Some(a) = stream.aspect() {
            aspect = a;
        }
        candidate_batch.extend(
            batch
                .into_iter()
                .filter_map(|sample| deduper.accepts(&sample.hash).then_some(sample.frame)),
        );

        if candidate_batch.len() >= batch_size {
            flush_batch_to_candidates(
                &mut candidate_batch,
                &mut candidates,
                &mut prev_tail,
                &weights,
                &spec,
                aspect,
                cfg.min_gap(),
            );
            // Bound memory over long ranges: once winners pile up to several
            // grids' worth, re-rank as one set and keep the top `capacity`.
            // Rank normalization is per-set, consistent with the final pass.
            let capacity = layout::grid_capacity(&spec, aspect);
            if candidates.len() >= 4 * capacity {
                candidates = select_capacity(candidates, capacity, &weights, cfg.min_gap());
            }
        }
    }

    if !candidate_batch.is_empty() {
        flush_batch_to_candidates(
            &mut candidate_batch,
            &mut candidates,
            &mut prev_tail,
            &weights,
            &spec,
            aspect,
            cfg.min_gap(),
        );
    }

    let capacity = layout::grid_capacity(&spec, aspect);
    let picked = if candidates.len() <= capacity {
        candidates
    } else {
        select_capacity(candidates, capacity, &weights, cfg.min_gap())
    };

    // An empty selection (range past the end of the video) must not render
    // an empty grid and report success — fail with a code instead.
    if picked.is_empty() {
        return Err(Error::new(
            ErrorCode::NoFramesExtracted,
            format!(
                "no frames extracted in range {}-{}; does the range lie within the video?",
                format_ts(range.start()),
                format_ts(range.end())
            ),
        )
        .with_context(ErrorContext::default().with_input(input.as_path())));
    }

    let scores: Vec<f32> = picked.iter().map(|f| f.score.get()).collect();
    let lay = layout::compute(&scores, &[], aspect, &spec);
    let jpeg = compose::render(&picked, &lay, &[], cfg.jpeg_quality(), cfg.hero_border_px())?;

    let file = "grid_extract.jpg";
    let image = session.emit_image(file, &jpeg)?;
    let grid_meta = build_grid_draft(&picked, &[], lay.template).into_meta(image);

    let subtitles = match subs {
        // One span, one assignment: every cue overlapping the requested
        // range lands on the single grid, even when the picked frames
        // cover a narrower stretch than the range itself.
        Some(subs) => subs
            .assign(range, &[grid_meta.time_range])
            .pop()
            .expect("one span yields exactly one assignment"),
        None => GridSubtitles::NoSource,
    };

    // `range` marks the document as an extract; the constructor derives the
    // wire `duration` from the segment, so consumers cannot mistake the
    // harvested span for a total duration.
    let meta = wire::Metadata::extract(
        input.to_string(),
        wire::Segment::from(range),
        cfg.settings(),
        vec![wire::Grid::new(grid_meta, subtitles)],
    );

    // No subtitles.txt: the range-sliced cues already travel in the metadata.
    session.commit(&meta, None, events)?;

    Ok(())
}

fn flush_batch_to_candidates(
    batch: &mut Vec<CandidateFrame>,
    candidates: &mut Vec<ScoredFrame>,
    prev_tail: &mut Option<Arc<GrayImage>>,
    weights: &score::Weights,
    spec: &CanvasSpec,
    aspect: Aspect,
    min_gap: f32,
) {
    if batch.is_empty() {
        return;
    }
    let tail = batch.last().map(|f| f.gray.clone());
    // Keep the buffer's capacity across flushes: mem::take would leave a
    // zero-capacity Vec that reallocates as the next batch refills it.
    let drained = mem::replace(batch, Vec::with_capacity(batch.capacity()));
    let mut scored = score::score_frames(drained, weights, prev_tail.as_deref());
    *prev_tail = tail;
    let capacity = layout::grid_capacity(spec, aspect);
    let selected = score::select_top_k(&scored, capacity, min_gap);
    candidates.extend(extract_selected(&mut scored, selected));
}

/// Re-rank per-batch winners as one global competition set and keep the top
/// `capacity` by score.
///
/// The survivors are non-consecutive in time, so their `scene_diff` is NOT
/// recomputed against slice neighbors (frames possibly minutes apart — that
/// difference is noise, not a transition). [`score::rescore_frames`] keeps
/// each frame's carried `scene_diff` from its original consecutive batch and
/// rank-normalizes within this set, consistent with the final pass.
fn select_capacity(
    candidates: Vec<ScoredFrame>,
    capacity: usize,
    weights: &score::Weights,
    min_gap: f32,
) -> Vec<ScoredFrame> {
    let mut rescored = score::rescore_frames(candidates, weights);
    let selected = score::select_top_k(&rescored, capacity, min_gap);
    extract_selected(&mut rescored, selected)
}
