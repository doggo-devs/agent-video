//! Per-chunk processing: score → select → layout → compose.

use anyhow::Result;
use image::GrayImage;

use crate::types::{CandidateFrame, GridDraft, KeyFrameMeta, ScoredFrame, TimeRange};
use crate::{compose, layout, score};

#[derive(Debug)]
pub struct GridResult {
    pub ci: usize,
    pub jpeg: Vec<u8>,
    pub draft: GridDraft,
}

#[derive(Clone, Copy)]
pub struct ProcessChunkEnv {
    pub spec: crate::types::CanvasSpec,
    pub aspect: f32,
    pub weights: score::Weights,
    pub min_gap_floor: f32,
    pub jpeg_quality: u8,
    pub hero_border_px: u32,
}

/// Score, select, lay out, and compose one chunk into a grid.
///
/// `prev_tail` is the grayscale of the frame immediately before `chunk` (the
/// previous chunk's last frame), so a scene change landing exactly on the
/// chunk boundary still scores; `None` for the first chunk.
pub fn process_chunk(
    ci: usize,
    chunk: Vec<CandidateFrame>,
    prev_tail: Option<GrayImage>,
    env: ProcessChunkEnv,
) -> Result<GridResult> {
    let mut chunk = score::score_frames(chunk, &env.weights, prev_tail.as_ref());

    // grid_capacity is a cheap pure function; deriving it here keeps the env
    // free of values that could drift from their inputs.
    let capacity = layout::grid_capacity(&env.spec, env.aspect);
    let span = chunk_span(&chunk);
    let min_gap = (0.5 * span / capacity as f32).max(env.min_gap_floor);
    let selected = score::select_top_k(&chunk, capacity, min_gap);
    let mut picked = extract_selected(&mut chunk, selected);

    // A hero occupies ~4 cells of area, so only capacity/4 heroes ever fit.
    let mut heroes = score::pick_heroes(&picked, (capacity / 4).min(3));
    let lay = evict_for_heroes(&mut picked, &mut heroes, env.aspect, &env.spec);

    let jpeg = compose::render(&picked, &lay, &heroes, env.jpeg_quality, env.hero_border_px)?;
    let draft = build_grid_draft(&picked, &heroes, lay.template);

    Ok(GridResult { ci, jpeg, draft })
}

/// Evict the lowest-scored normal frames until hero enlargement no longer
/// squeezes cells below the readable floor, and return the final layout.
///
/// The floor is ground truth, not cell arithmetic: the justified layout adapts
/// row shapes to the frame mix, so the only reliable check is to compute the
/// layout and measure its smallest cell. The bar is what the same frames would
/// get with no heroes at all (capped at `min_cell_edge`): heroes must pay for
/// their size by displacing the weakest frames, never by shrinking everyone
/// below that baseline. Hero indices are shifted to stay aligned with
/// `picked`.
fn evict_for_heroes(
    picked: &mut Vec<ScoredFrame>,
    heroes: &mut [usize],
    aspect: f32,
    spec: &crate::types::CanvasSpec,
) -> layout::Layout {
    let scores: Vec<f32> = picked.iter().map(|f| f.score).collect();
    if heroes.is_empty() {
        return layout::compute(&scores, heroes, aspect, spec);
    }
    // Fixed bar from the pre-eviction frame set: evicting frames must not
    // raise the bar, or the loop would chase its own tail.
    let uniform = layout::compute(&scores, &[], aspect, spec);
    let baseline = min_short_edge(&uniform).min(spec.min_cell_edge());

    loop {
        let scores: Vec<f32> = picked.iter().map(|f| f.score).collect();
        let lay = layout::compute(&scores, heroes, aspect, spec);
        if min_short_edge(&lay) >= baseline {
            return lay;
        }
        let Some(victim) = (0..picked.len())
            .filter(|i| !heroes.contains(i))
            .min_by(|&a, &b| picked[a].score.total_cmp(&picked[b].score))
        else {
            // Only heroes left: nothing to trade away.
            return lay;
        };
        picked.remove(victim);
        for h in heroes.iter_mut() {
            if *h > victim {
                *h -= 1;
            }
        }
    }
}

/// Shortest cell edge in a layout, `u32::MAX` for an empty one.
fn min_short_edge(lay: &layout::Layout) -> u32 {
    lay.rects
        .iter()
        .map(|r| r.w.min(r.h))
        .min()
        .unwrap_or(u32::MAX)
}

/// Extract frames at `indices` from `frames` without cloning.
pub fn extract_selected(
    frames: &mut Vec<ScoredFrame>,
    mut indices: Vec<usize>,
) -> Vec<ScoredFrame> {
    indices.sort_unstable_by(|a, b| b.cmp(a));
    let mut picked = Vec::with_capacity(indices.len());
    for i in indices {
        picked.push(frames.swap_remove(i));
    }
    picked.sort_by(|a, b| a.ts.total_cmp(&b.ts));
    picked
}

pub fn build_grid_draft(
    picked: &[ScoredFrame],
    heroes: &[usize],
    template: &'static str,
) -> GridDraft {
    let (first, last) = (
        picked.first().map(|f| f.ts).unwrap_or(0.0),
        picked.last().map(|f| f.ts).unwrap_or(0.0),
    );
    let key_frames = heroes
        .iter()
        .map(|&i| KeyFrameMeta {
            timestamp: crate::types::format_ts(picked[i].ts),
            score: picked[i].score.round() as u32,
            grid_position: "hero",
        })
        .collect();
    GridDraft {
        time_range: TimeRange::new(first, last).expect("picked frames are ts-sorted ascending"),
        total_frames: picked.len(),
        key_frames,
        layout: template,
    }
}

fn chunk_span(chunk: &[ScoredFrame]) -> f32 {
    match (chunk.first(), chunk.last()) {
        (Some(a), Some(b)) => (b.ts - a.ts).max(0.0),
        _ => 0.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{GrayImage, RgbImage};
    use std::sync::Arc;

    fn frame(ts: f32) -> ScoredFrame {
        ScoredFrame {
            ts,
            gray: GrayImage::new(1, 1),
            rgb: Arc::new(RgbImage::new(1, 1)),
            score: 0.0,
            scene_diff: 0.0,
        }
    }

    #[test]
    fn extracts_selected_frames_sorted_by_ascending_ts() {
        let mut frames = vec![frame(5.0), frame(1.0), frame(9.0)];

        let picked = extract_selected(&mut frames, vec![0, 2]);

        let picked_ts: Vec<f32> = picked.iter().map(|f| f.ts).collect();
        assert_eq!(picked_ts, vec![5.0, 9.0], "chosen frames sorted by ts");
        let remaining_ts: Vec<f32> = frames.iter().map(|f| f.ts).collect();
        assert_eq!(
            remaining_ts,
            vec![1.0],
            "source keeps only the unselected frame"
        );
    }

    fn scored(ts: f32, score: f32) -> ScoredFrame {
        let mut f = frame(ts);
        f.score = score;
        f
    }

    #[test]
    fn evicts_weakest_frames_until_cells_stay_readable() {
        // 16 frames tile a 2048/512 square canvas at exactly the readable
        // floor; a hero must displace weak frames, not shrink everyone.
        let spec = crate::types::CanvasSpec::new(2048, 512).expect("valid test spec");
        let mut picked: Vec<ScoredFrame> = (0..16).map(|i| scored(i as f32, i as f32)).collect();
        let mut heroes = vec![15];

        let lay = evict_for_heroes(&mut picked, &mut heroes, 1.0, &spec);

        assert!(picked.len() < 16, "hero displaced some frames");
        assert!(
            min_short_edge(&lay) >= 512,
            "no cell below the readable floor: {}",
            min_short_edge(&lay)
        );
        let hero = heroes[0];
        assert!(hero < picked.len(), "hero index stays aligned");
        assert_eq!(picked[hero].ts, 15.0, "hero frame survived eviction");
        let min_kept = picked.iter().map(|f| f.score).fold(f32::INFINITY, f32::min);
        assert!(min_kept > 0.0, "the weakest frames were the ones evicted");
    }

    #[test]
    fn keeps_all_frames_when_hero_fits_free() {
        // An under-full grid has slack: the hero enlarges into free space and
        // nothing needs to go.
        let spec = crate::types::CanvasSpec::new(1024, 512).expect("valid test spec");
        let mut picked: Vec<ScoredFrame> =
            (0..4).map(|i| scored(i as f32, 50.0 + i as f32)).collect();
        let mut heroes = vec![3];

        let lay = evict_for_heroes(&mut picked, &mut heroes, 1.0, &spec);

        assert_eq!(picked.len(), 4, "no frame evicted");
        assert_eq!(heroes, vec![3]);
        assert!(min_short_edge(&lay) >= 512);
    }
}
