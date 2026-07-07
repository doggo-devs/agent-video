// SPDX-License-Identifier: Apache-2.0
//! The `extract` CLI subcommand: high-density sampling of a single time range
//! into one equal-cell grid, frames in time order (no hero emphasis).
//!
//! This is a one-shot batch pipeline, distinct from `process`'s streaming,
//! multi-chunk orchestration in the parent module — it scores and selects
//! within a single competition set instead of per-chunk.

use std::mem;
use std::path::Path;

use anyhow::Result;
use image::GrayImage;

use crate::config::Config;
use crate::error::{Error, ErrorCode, ErrorContext};
use crate::source::Source;
use crate::subtitle::Subtitles;
use crate::types::{CandidateFrame, CanvasSpec, ScoredFrame, TimeRange, format_ts};
use crate::{compose, dedup, extract, layout, score, wire};

use super::candidates::CandidateStream;
use super::chunk::{build_grid_draft, extract_selected};
use super::output::OutputSession;
use super::state::extract_batch_frames;
use super::{OutputMode, SubtitleSource};

/// Run the `extract` pipeline: high-density sampling of `range` into a single
/// equal-cell grid, frames in time order (no hero emphasis).
///
/// Output contract: `grid_extract.jpg` and `metadata.json` only. Subtitles for
/// the requested range travel inside the metadata's `subtitles` field; no
/// `subtitles.txt` is written — the full-video text is not an artifact of a
/// range extraction.
pub fn extract_range(
    input: &Source,
    out_dir: &Path,
    cfg: &Config,
    range: TimeRange,
    mode: OutputMode,
    subs: SubtitleSource<'_>,
) -> Result<()> {
    super::validate_input(input)?;
    // Before the expensive sampling pass: subtitle errors must fail fast,
    // not after all the CPU and IO have been spent.
    let subs = Subtitles::resolve(input, subs)?;
    let session = OutputSession::open(out_dir, mode)?;

    let fps = cfg.extract_fps();
    // `extract` exposes no --threads flag; the decoder auto-sizes its pool.
    let stream = extract::stream_range(
        input,
        fps,
        cfg.min_frame_size(),
        cfg.score_edge(),
        range,
        None,
    )?;

    let batch_size = extract_batch_frames(fps, cfg.chunk_seconds());
    // The same judging machinery `process` uses — the two subcommands must
    // select surviving frames identically for the same input.
    let mut stream = CandidateStream::new(
        stream,
        dedup::Deduper::new(cfg.hash_threshold(), cfg.dedup_window()),
        batch_size,
        cfg.score_edge(),
        None,
    );
    let weights = cfg.weights();
    let spec = cfg.canvas_spec();

    let mut candidate_batch: Vec<CandidateFrame> = Vec::new();
    let mut candidates: Vec<ScoredFrame> = Vec::new();
    // Tail grayscale of the previously scored batch: batches are consecutive
    // in time, so it is the true predecessor of the next batch's first frame.
    let mut prev_tail: Option<GrayImage> = None;
    let mut aspect = 1.0_f32;

    while let Some(judged) = stream.next_batch()? {
        if let Some(a) = stream.aspect() {
            aspect = a;
        }
        candidate_batch.extend(
            judged
                .into_iter()
                .filter_map(|(frame, kept)| kept.then_some(frame)),
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
        .with_context(ErrorContext::default().with_input(input.as_path()))
        .into());
    }

    let scores: Vec<f32> = picked.iter().map(|f| f.score.get()).collect();
    let lay = layout::compute(&scores, &[], aspect, &spec);
    let jpeg = compose::render(&picked, &lay, &[], cfg.jpeg_quality(), cfg.hero_border_px())?;

    let file = "grid_extract.jpg";
    let image = session.emit_image(file, &jpeg)?;
    let mut grid_meta = build_grid_draft(&picked, &[], lay.template).into_meta(image);

    grid_meta.subtitles = match subs {
        Some(ref subs) => subs.slice(range),
        None => crate::types::GridSubtitles::NoSource,
    };

    // `duration` carries the harvested span; `range` marks the document as
    // an extract, so consumers cannot mistake the span for a total duration.
    let meta = wire::Metadata {
        source: input.to_string(),
        duration: range.duration(),
        range: Some((range.start(), range.end())),
        settings: cfg.settings(),
        grids: vec![grid_meta.into()],
    };

    // No subtitles.txt: the range-sliced cues already travel in the metadata.
    session.commit(&meta, None)?;

    Ok(())
}

fn flush_batch_to_candidates(
    batch: &mut Vec<CandidateFrame>,
    candidates: &mut Vec<ScoredFrame>,
    prev_tail: &mut Option<GrayImage>,
    weights: &score::Weights,
    spec: &CanvasSpec,
    aspect: f32,
    min_gap: f32,
) {
    if batch.is_empty() {
        return;
    }
    let tail = batch.last().map(|f| f.gray.clone());
    // Keep the buffer's capacity across flushes: mem::take would leave a
    // zero-capacity Vec that reallocates as the next batch refills it.
    let drained = mem::replace(batch, Vec::with_capacity(batch.capacity()));
    let mut scored = score::score_frames(drained, weights, prev_tail.as_ref());
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
