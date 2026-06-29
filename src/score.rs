//! Frame scoring, rank normalization, and selection.
//!
//! Three raw dimensions are computed per frame in their natural (incomparable)
//! units, then **rank-normalized within a chunk** to `0..=100` before being
//! weighted into a final score. Ranking is scale-free: it removes the unit
//! mismatch between a Laplacian variance, a Sobel energy, and a pixel
//! difference without any magic normalization constant. The cost is that only
//! order is preserved, not magnitude — acceptable for what is fundamentally a
//! ranking task.
//!
//! Selection takes the top-k by score with a temporal non-maximum suppression
//! pass so chosen frames spread across time rather than clustering on one
//! high-action moment. Hero detection (which drives layout template choice) is
//! a separate, statistics-based step.

use image::GrayImage;
use rayon::prelude::*;

use crate::types::{RawScores, ScoredFrame};

/// Weight of the scene-change dimension in the final score.
const W_SCENE: f32 = 0.40;
/// Weight of the clarity dimension.
const W_CLARITY: f32 = 0.35;
/// Weight of the edge-density dimension.
const W_EDGE: f32 = 0.25;

/// Laplacian variance: a focus/sharpness measure. Higher is sharper.
///
/// Convolves with the 3×3 kernel `[[0,1,0],[1,-4,1],[0,1,0]]` (edges clamped)
/// and returns the variance of the response. A blurred frame has a low-variance
/// (flat) Laplacian; a sharp one has a spiky response.
pub fn clarity(gray: &GrayImage) -> f32 {
    let (w, h) = gray.dimensions();
    if w == 0 || h == 0 {
        return 0.0;
    }
    let at = |x: i64, y: i64| -> f32 {
        let xc = x.clamp(0, w as i64 - 1) as u32;
        let yc = y.clamp(0, h as i64 - 1) as u32;
        gray.get_pixel(xc, yc).0[0] as f32
    };

    let mut sum = 0.0_f64;
    let mut sum_sq = 0.0_f64;
    let n = (w as f64) * (h as f64);
    for y in 0..h as i64 {
        for x in 0..w as i64 {
            let lap = at(x, y - 1)
                + at(x - 1, y)
                + at(x + 1, y)
                + at(x, y + 1)
                - 4.0 * at(x, y);
            sum += lap as f64;
            sum_sq += (lap as f64) * (lap as f64);
        }
    }
    let mean = sum / n;
    (sum_sq / n - mean * mean) as f32
}

/// Sobel energy mean: a content-richness measure.
///
/// Applies horizontal and vertical Sobel kernels, takes the gradient magnitude
/// `sqrt(gx² + gy²)` per pixel, and returns the mean (edges clamped).
pub fn edge_density(gray: &GrayImage) -> f32 {
    let (w, h) = gray.dimensions();
    if w == 0 || h == 0 {
        return 0.0;
    }
    let at = |x: i64, y: i64| -> f32 {
        let xc = x.clamp(0, w as i64 - 1) as u32;
        let yc = y.clamp(0, h as i64 - 1) as u32;
        gray.get_pixel(xc, yc).0[0] as f32
    };

    let mut total = 0.0_f64;
    for y in 0..h as i64 {
        for x in 0..w as i64 {
            let gx = -at(x - 1, y - 1) + at(x + 1, y - 1) - 2.0 * at(x - 1, y)
                + 2.0 * at(x + 1, y)
                - at(x - 1, y + 1)
                + at(x + 1, y + 1);
            let gy = -at(x - 1, y - 1) - 2.0 * at(x, y - 1) - at(x + 1, y - 1)
                + at(x - 1, y + 1)
                + 2.0 * at(x, y + 1)
                + at(x + 1, y + 1);
            total += (gx * gx + gy * gy).sqrt() as f64;
        }
    }
    (total / ((w as f64) * (h as f64))) as f32
}

/// Mean absolute grayscale difference between two equally-sized frames.
///
/// Returns 0.0 if dimensions differ (cannot compare). Used as the scene-change
/// signal: how much this frame differs from the previous kept frame.
pub fn frame_diff(a: &GrayImage, b: &GrayImage) -> f32 {
    if a.dimensions() != b.dimensions() {
        return 0.0;
    }
    let n = a.as_raw().len();
    if n == 0 {
        return 0.0;
    }
    let total: u64 = a
        .as_raw()
        .iter()
        .zip(b.as_raw().iter())
        .map(|(&x, &y)| (x as i32 - y as i32).unsigned_abs() as u64)
        .sum();
    total as f32 / n as f32
}

/// Fill raw scores (clarity, edge density, scene diff) for a chunk in place.
///
/// `scene_diff` of the first frame is 0 (no predecessor); each subsequent
/// frame's is its difference from the previous frame in the slice.
///
/// Each frame's three dimensions are independent reads (scene_diff reads the
/// previous frame's grayscale, never mutates it), so the per-frame work runs in
/// parallel — this is the pipeline's hottest CPU stage.
pub fn compute_raw(frames: &mut [ScoredFrame]) {
    let raws: Vec<RawScores> = (0..frames.len())
        .into_par_iter()
        .map(|i| RawScores {
            clarity: clarity(&frames[i].gray),
            edge_density: edge_density(&frames[i].gray),
            scene_diff: if i == 0 {
                0.0
            } else {
                frame_diff(&frames[i].gray, &frames[i - 1].gray)
            },
        })
        .collect();
    for (frame, raw) in frames.iter_mut().zip(raws) {
        frame.raw = raw;
    }
}

/// Rank-normalize each dimension to `0..=100` within the slice, then write the
/// weighted final score into each frame.
///
/// For a dimension, the frame with the smallest raw value gets 0, the largest
/// gets 100, and the rest interpolate linearly by rank position. Ties share the
/// rank of their first occurrence. With fewer than two frames every score is
/// 100 (a single frame is trivially the best of its chunk).
pub fn rank_normalize_and_score(frames: &mut [ScoredFrame]) {
    let n = frames.len();
    if n == 0 {
        return;
    }
    if n == 1 {
        frames[0].score = 100.0;
        return;
    }

    let clarity_rank = rank_to_percentile(frames, |f| f.raw.clarity);
    let edge_rank = rank_to_percentile(frames, |f| f.raw.edge_density);
    let scene_rank = rank_to_percentile(frames, |f| f.raw.scene_diff);

    for i in 0..n {
        frames[i].score =
            W_SCENE * scene_rank[i] + W_CLARITY * clarity_rank[i] + W_EDGE * edge_rank[i];
    }
}

/// Map each frame's value (via `key`) to a rank percentile in `0..=100`.
fn rank_to_percentile(frames: &[ScoredFrame], key: impl Fn(&ScoredFrame) -> f32) -> Vec<f32> {
    let n = frames.len();
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| {
        key(&frames[a])
            .partial_cmp(&key(&frames[b]))
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut out = vec![0.0_f32; n];
    let denom = (n - 1) as f32;
    // `idx` is ascending: position 0 is the smallest (percentile 0).
    for (rank, &frame_i) in idx.iter().enumerate() {
        out[frame_i] = 100.0 * rank as f32 / denom;
    }
    out
}

/// Select up to `k` frames by descending score, enforcing a minimum temporal
/// gap (`min_gap`, seconds) between any two selected frames.
///
/// Greedy: walk frames best-first, accept one only if it is at least `min_gap`
/// from every already-accepted frame. Returns indices into `frames`, sorted by
/// timestamp ascending.
pub fn select_top_k(frames: &[ScoredFrame], k: usize, min_gap: f32) -> Vec<usize> {
    if k == 0 || frames.is_empty() {
        return Vec::new();
    }
    let mut order: Vec<usize> = (0..frames.len()).collect();
    order.sort_by(|&a, &b| {
        frames[b]
            .score
            .partial_cmp(&frames[a].score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut chosen: Vec<usize> = Vec::with_capacity(k);
    for &i in &order {
        if chosen.len() == k {
            break;
        }
        let too_close = chosen
            .iter()
            .any(|&c| (frames[c].ts - frames[i].ts).abs() < min_gap);
        if !too_close {
            chosen.push(i);
        }
    }
    chosen.sort_by(|&a, &b| {
        frames[a]
            .ts
            .partial_cmp(&frames[b].ts)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    chosen
}

/// Pick hero frames from an already-selected set: those whose score is at least
/// one standard deviation above the mean. Returns at most 3 indices (into the
/// passed slice), highest score first.
///
/// These statistically-prominent frames drive layout template choice. An empty
/// result means a uniform grid.
pub fn pick_heroes(selected: &[ScoredFrame]) -> Vec<usize> {
    let n = selected.len();
    if n < 2 {
        return Vec::new();
    }
    let mean: f32 = selected.iter().map(|f| f.score).sum::<f32>() / n as f32;
    let var: f32 =
        selected.iter().map(|f| (f.score - mean).powi(2)).sum::<f32>() / n as f32;
    let std = var.sqrt();
    // No spread means no frame stands out — a uniform grid.
    if std < f32::EPSILON {
        return Vec::new();
    }
    let threshold = mean + std;

    let mut heroes: Vec<usize> = (0..n).filter(|&i| selected[i].score >= threshold).collect();
    heroes.sort_by(|&a, &b| {
        selected[b]
            .score
            .partial_cmp(&selected[a].score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    heroes.truncate(3);
    heroes
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::RawScores;
    use image::{Luma, RgbImage};

    fn frame(ts: f32, clarity: f32, edge: f32, scene: f32) -> ScoredFrame {
        ScoredFrame {
            ts,
            gray: GrayImage::new(1, 1),
            rgb: RgbImage::new(1, 1),
            raw: RawScores { clarity, edge_density: edge, scene_diff: scene },
            score: 0.0,
        }
    }

    #[test]
    fn flat_image_has_zero_clarity() {
        let g = GrayImage::from_pixel(16, 16, Luma([120]));
        assert_eq!(clarity(&g), 0.0);
    }

    #[test]
    fn sharp_edge_beats_blur_in_clarity() {
        let mut sharp = GrayImage::new(16, 16);
        for y in 0..16 {
            for x in 0..16 {
                sharp.put_pixel(x, y, Luma([if x < 8 { 0 } else { 255 }]));
            }
        }
        let blur = GrayImage::from_pixel(16, 16, Luma([128]));
        assert!(clarity(&sharp) > clarity(&blur));
    }

    #[test]
    fn identical_frames_have_zero_diff() {
        let g = GrayImage::from_pixel(8, 8, Luma([50]));
        assert_eq!(frame_diff(&g, &g), 0.0);
    }

    #[test]
    fn opposite_frames_have_full_diff() {
        let black = GrayImage::from_pixel(8, 8, Luma([0]));
        let white = GrayImage::from_pixel(8, 8, Luma([255]));
        assert_eq!(frame_diff(&black, &white), 255.0);
    }

    #[test]
    fn normalizes_ranks_to_full_range() {
        // clarity ascending 1,2,3 → percentiles 0,50,100; same for others so
        // the weighted score collapses to those percentiles.
        let mut fs = vec![
            frame(0.0, 1.0, 1.0, 1.0),
            frame(1.0, 2.0, 2.0, 2.0),
            frame(2.0, 3.0, 3.0, 3.0),
        ];
        rank_normalize_and_score(&mut fs);
        assert!((fs[0].score - 0.0).abs() < 1e-3);
        assert!((fs[1].score - 50.0).abs() < 1e-3);
        assert!((fs[2].score - 100.0).abs() < 1e-3);
    }

    #[test]
    fn single_frame_scores_full() {
        let mut fs = vec![frame(0.0, 9.0, 9.0, 9.0)];
        rank_normalize_and_score(&mut fs);
        assert_eq!(fs[0].score, 100.0);
    }

    #[test]
    fn enforces_minimum_temporal_gap() {
        // Two high scorers 0.5s apart; NMS with gap 2.0 keeps only the better.
        let mut fs = vec![
            frame(0.0, 0.0, 0.0, 0.0),
            frame(0.5, 0.0, 0.0, 0.0),
            frame(5.0, 0.0, 0.0, 0.0),
        ];
        fs[0].score = 90.0;
        fs[1].score = 95.0;
        fs[2].score = 80.0;
        let sel = select_top_k(&fs, 3, 2.0);
        assert_eq!(sel, vec![1, 2], "0.5s neighbor of the best is suppressed");
    }

    #[test]
    fn returns_indices_sorted_by_time() {
        let mut fs = vec![frame(9.0, 0.0, 0.0, 0.0), frame(1.0, 0.0, 0.0, 0.0)];
        fs[0].score = 50.0;
        fs[1].score = 99.0;
        let sel = select_top_k(&fs, 2, 0.0);
        assert_eq!(sel, vec![1, 0], "output ordered by timestamp, not score");
    }

    #[test]
    fn picks_outlier_as_hero() {
        let mut fs = vec![
            frame(0.0, 0.0, 0.0, 0.0),
            frame(1.0, 0.0, 0.0, 0.0),
            frame(2.0, 0.0, 0.0, 0.0),
            frame(3.0, 0.0, 0.0, 0.0),
        ];
        fs[0].score = 50.0;
        fs[1].score = 50.0;
        fs[2].score = 50.0;
        fs[3].score = 99.0;
        let heroes = pick_heroes(&fs);
        assert_eq!(heroes, vec![3]);
    }

    #[test]
    fn no_hero_when_scores_uniform() {
        let mut fs = vec![frame(0.0, 0.0, 0.0, 0.0), frame(1.0, 0.0, 0.0, 0.0)];
        fs[0].score = 70.0;
        fs[1].score = 70.0;
        let heroes = pick_heroes(&fs);
        assert!(heroes.is_empty(), "uniform scores yield a uniform grid");
    }
}
