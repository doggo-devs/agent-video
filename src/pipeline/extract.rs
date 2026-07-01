//! The `extract` CLI subcommand: high-density sampling of a single time range
//! into one equal-cell grid, frames in time order (no hero emphasis).
//!
//! This is a one-shot batch pipeline, distinct from `process`'s streaming,
//! multi-chunk orchestration in the parent module — it scores and selects
//! within a single competition set instead of per-chunk.

use std::fs;
use std::mem;
use std::path::Path;

use anyhow::{Context, Result};
use image::GrayImage;

use crate::config::Config;
use crate::types::{CandidateFrame, CanvasSpec, Metadata, ScoredFrame};
use crate::{compose, dedup, extract, layout, score, subtitle};

use super::chunk::{build_grid_meta, extract_selected, grid_image};
use super::output::atomic_write;
use super::state::extract_batch_frames;
use super::{OutputMode, SubtitleOpts};

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
    super::validate_input(input)?;
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
    let mut candidate_batch: Vec<CandidateFrame> = Vec::new();
    let mut candidates: Vec<ScoredFrame> = Vec::new();
    // Tail grayscale of the previously scored batch: batches are consecutive
    // in time, so it is the true predecessor of the next batch's first frame.
    let mut prev_tail: Option<GrayImage> = None;
    let mut aspect = 1.0_f32;

    while let Some(rgb) = stream.next_frame()? {
        if let Some(a) = stream.aspect() {
            aspect = a;
        }
        rgb_batch.push(rgb);
        if rgb_batch.len() >= batch_size {
            let deduped = deduper.filter(extract::candidate_frames_from_rgbs(
                mem::replace(&mut rgb_batch, Vec::with_capacity(batch_size)),
                cfg.score_edge,
            ));
            candidate_batch.extend(deduped);
        }

        if candidate_batch.len() >= batch_size {
            flush_batch_to_candidates(
                &mut candidate_batch,
                &mut candidates,
                &mut prev_tail,
                &weights,
                &spec,
                aspect,
                cfg.min_gap,
            );
            // Bound memory over arbitrarily long ranges: once per-batch winners
            // pile up to several grids' worth, re-rank them as one set and keep
            // the top `capacity`. Rank normalization is per-competition-set, so
            // this periodic compaction is consistent with the final pass.
            let capacity = layout::grid_capacity(&spec, aspect);
            if candidates.len() >= 4 * capacity {
                candidates = select_capacity(candidates, capacity, &weights, cfg.min_gap);
            }
        }
    }

    if !rgb_batch.is_empty() {
        let deduped = deduper.filter(extract::candidate_frames_from_rgbs(
            rgb_batch,
            cfg.score_edge,
        ));
        candidate_batch.extend(deduped);
    }
    if !candidate_batch.is_empty() {
        flush_batch_to_candidates(
            &mut candidate_batch,
            &mut candidates,
            &mut prev_tail,
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
        select_capacity(candidates, capacity, &weights, cfg.min_gap)
    };

    let scores: Vec<f32> = picked.iter().map(|f| f.score).collect();
    let lay = layout::compute(&scores, &[], aspect, &spec);
    let jpeg = compose::render(&picked, &lay, &[], cfg.jpeg_quality, cfg.hero_border_px)?;

    let file = "grid_extract.jpg";
    let image = grid_image(write_files, file.to_string(), &jpeg);
    let mut grid_meta = build_grid_meta(image, &picked, &[], lay.template);

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
    let mut scored = score::score_frames(mem::take(batch), weights, prev_tail.as_ref());
    *prev_tail = tail;
    let capacity = layout::grid_capacity(spec, aspect);
    let selected = score::select_top_k(&scored, capacity, min_gap);
    candidates.extend(extract_selected(&mut scored, selected));
}

/// Re-rank per-batch winners as one global competition set and keep the top
/// `capacity` by score. Rank normalization is per-competition-set, so dropping
/// the stale per-batch scores (`ScoredFrame` → `CandidateFrame`) and re-scoring
/// the survivors together is the consistent choice.
fn select_capacity(
    candidates: Vec<ScoredFrame>,
    capacity: usize,
    weights: &score::Weights,
    min_gap: f32,
) -> Vec<ScoredFrame> {
    let recandidates: Vec<CandidateFrame> =
        candidates.into_iter().map(CandidateFrame::from).collect();
    // Survivors are non-consecutive winners; there is no boundary predecessor.
    let mut rescored = score::score_frames(recandidates, weights, None);
    let selected = score::select_top_k(&rescored, capacity, min_gap);
    extract_selected(&mut rescored, selected)
}
