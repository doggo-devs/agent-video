//! Per-chunk processing: score → select → layout → compose.

use anyhow::Result;

use crate::types::{GridMeta, KeyFrameMeta, ScoredFrame};
use crate::{compose, layout, score};

#[derive(Debug)]
pub struct GridResult {
    pub ci: usize,
    pub jpeg: Vec<u8>,
    pub meta: GridMeta,
}

#[derive(Clone, Copy)]
pub struct ProcessChunkEnv {
    pub spec: crate::types::CanvasSpec,
    pub aspect: f32,
    pub weights: score::Weights,
    pub write_files: bool,
    pub min_gap_floor: f32,
    pub jpeg_quality: u8,
    pub hero_border_px: u32,
}

pub type ChunkMsg = (usize, Result<GridResult>);

pub fn process_chunk(ci: usize, mut chunk: Vec<ScoredFrame>, env: ProcessChunkEnv) -> Result<GridResult> {
    score::compute_raw(&mut chunk);
    score::rank_normalize_and_score(&mut chunk, &env.weights);

    // grid_capacity is a cheap pure function; deriving it here keeps the env
    // free of values that could drift from their inputs.
    let capacity = layout::grid_capacity(&env.spec, env.aspect);
    let span = chunk_span(&chunk);
    let min_gap = (0.5 * span / capacity as f32).max(env.min_gap_floor);
    let selected = score::select_top_k(&chunk, capacity, min_gap);
    let picked = extract_selected(&mut chunk, selected);

    let heroes = score::pick_heroes(&picked);
    let lay = layout::compute(picked.len(), &heroes, env.aspect, &env.spec);

    let jpeg = compose::render(&picked, &lay, &heroes, env.jpeg_quality, env.hero_border_px)?;
    let file = format!("grid_{:03}.jpg", ci + 1);

    let meta = build_grid_meta(
        if env.write_files { Some(file.as_str()) } else { None },
        if env.write_files { None } else { Some(&jpeg) },
        &picked,
        &heroes,
        lay.template,
    );

    Ok(GridResult { ci, jpeg, meta })
}

/// Extract frames at `indices` from `frames` without cloning.
pub fn extract_selected(frames: &mut Vec<ScoredFrame>, mut indices: Vec<usize>) -> Vec<ScoredFrame> {
    indices.sort_unstable_by(|a, b| b.cmp(a));
    let mut picked = Vec::with_capacity(indices.len());
    for i in indices {
        picked.push(frames.swap_remove(i));
    }
    picked.sort_by(|a, b| a.ts.total_cmp(&b.ts));
    picked
}

pub fn build_grid_meta(
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
            timestamp: crate::types::format_ts(picked[i].ts),
            score: picked[i].score.round() as u32,
            grid_position: "hero".to_string(),
        })
        .collect();
    GridMeta {
        file: file.map(str::to_string),
        time_range: (first, last),
        total_frames: picked.len(),
        key_frames,
        layout: template.to_string(),
        base64: jpeg.map(|b| STANDARD.encode(b)),
        subtitles: None,
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
    fn extracts_selected_frames_sorted_by_ascending_ts() {
        let mut frames = vec![frame(5.0), frame(1.0), frame(9.0)];

        let picked = extract_selected(&mut frames, vec![0, 2]);

        let picked_ts: Vec<f32> = picked.iter().map(|f| f.ts).collect();
        assert_eq!(picked_ts, vec![5.0, 9.0], "chosen frames sorted by ts");
        let remaining_ts: Vec<f32> = frames.iter().map(|f| f.ts).collect();
        assert_eq!(remaining_ts, vec![1.0], "source keeps only the unselected frame");
    }
}
