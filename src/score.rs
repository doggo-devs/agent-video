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

use crate::types::{CandidateFrame, ScoredFrame};

/// Raw per-dimension scores, in their natural (incomparable) units — a scoring
/// transient, never stored on a frame. `clarity` is a Laplacian variance,
/// `edge_density` a Sobel energy mean, `scene_diff` a mean absolute grayscale
/// difference (0-255) from the previous frame.
#[derive(Debug, Clone, Copy, Default)]
struct RawScores {
    clarity: f32,
    edge_density: f32,
    scene_diff: f32,
}

/// Scoring dimension weights, passed from the profile.
#[derive(Debug, Clone, Copy)]
pub struct Weights {
    pub scene: f32,
    pub clarity: f32,
    pub edge: f32,
}

/// Laplacian variance: a focus/sharpness measure. Higher is sharper.
///
/// Convolves with `[[0,1,0],[1,-4,1],[0,1,0]]` (edges clamped) and returns the
/// response variance. The interior uses plain slice indexing so LLVM can
/// vectorize it; the one-pixel border falls back to clamped reads, keeping the
/// result identical to a naive clamped convolution.
pub fn clarity(gray: &GrayImage) -> f32 {
    let (w, h) = gray.dimensions();
    if w == 0 || h == 0 {
        return 0.0;
    }
    let (wu, hu) = (w as usize, h as usize);
    let buf = gray.as_raw();
    let n = (w as f64) * (h as f64);
    let mut sum = 0.0_f64;
    let mut sum_sq = 0.0_f64;

    let mut accumulate = |lap: f32| {
        sum += lap as f64;
        sum_sq += (lap as f64) * (lap as f64);
    };

    // Interior: all neighbours in bounds — plain indexing.
    if wu >= 3 && hu >= 3 {
        for y in 1..hu - 1 {
            let row = y * wu;
            let up = row - wu;
            let down = row + wu;
            for x in 1..wu - 1 {
                let c = buf[row + x] as f32;
                let lap = buf[up + x] as f32
                    + buf[row + x - 1] as f32
                    + buf[row + x + 1] as f32
                    + buf[down + x] as f32
                    - 4.0 * c;
                accumulate(lap);
            }
        }
    }

    // Border: clamped reads.
    let at = |x: i64, y: i64| -> f32 {
        let xc = x.clamp(0, w as i64 - 1) as usize;
        let yc = y.clamp(0, h as i64 - 1) as usize;
        buf[yc * wu + xc] as f32
    };
    let is_interior = |x: usize, y: usize| x >= 1 && y >= 1 && x + 1 < wu && y + 1 < hu;
    for y in 0..hu {
        for x in 0..wu {
            if wu >= 3 && hu >= 3 && is_interior(x, y) {
                continue;
            }
            let (xi, yi) = (x as i64, y as i64);
            let lap = at(xi, yi - 1) + at(xi - 1, yi) + at(xi + 1, yi) + at(xi, yi + 1)
                - 4.0 * at(xi, yi);
            accumulate(lap);
        }
    }

    let mean = sum / n;
    (sum_sq / n - mean * mean) as f32
}

/// Sobel energy mean: a content-richness measure.
///
/// Applies horizontal and vertical Sobel kernels, takes `sqrt(gx² + gy²)` per
/// pixel, and returns the mean. Same interior/border split as [`clarity`].
pub fn edge_density(gray: &GrayImage) -> f32 {
    let (w, h) = gray.dimensions();
    if w == 0 || h == 0 {
        return 0.0;
    }
    let (wu, hu) = (w as usize, h as usize);
    let buf = gray.as_raw();
    let mut total = 0.0_f64;

    if wu >= 3 && hu >= 3 {
        for y in 1..hu - 1 {
            let row = y * wu;
            let up = row - wu;
            let down = row + wu;
            for x in 1..wu - 1 {
                let (ul, uc, ur) = (
                    buf[up + x - 1] as f32,
                    buf[up + x] as f32,
                    buf[up + x + 1] as f32,
                );
                let (l, r) = (buf[row + x - 1] as f32, buf[row + x + 1] as f32);
                let (dl, dc, dr) = (
                    buf[down + x - 1] as f32,
                    buf[down + x] as f32,
                    buf[down + x + 1] as f32,
                );
                let gx = -ul + ur - 2.0 * l + 2.0 * r - dl + dr;
                let gy = -ul - 2.0 * uc - ur + dl + 2.0 * dc + dr;
                total += (gx * gx + gy * gy).sqrt() as f64;
            }
        }
    }

    let at = |x: i64, y: i64| -> f32 {
        let xc = x.clamp(0, w as i64 - 1) as usize;
        let yc = y.clamp(0, h as i64 - 1) as usize;
        buf[yc * wu + xc] as f32
    };
    let is_interior = |x: usize, y: usize| x >= 1 && y >= 1 && x + 1 < wu && y + 1 < hu;
    for y in 0..hu {
        for x in 0..wu {
            if wu >= 3 && hu >= 3 && is_interior(x, y) {
                continue;
            }
            let (xi, yi) = (x as i64, y as i64);
            let gx = -at(xi - 1, yi - 1) + at(xi + 1, yi - 1) - 2.0 * at(xi - 1, yi)
                + 2.0 * at(xi + 1, yi)
                - at(xi - 1, yi + 1)
                + at(xi + 1, yi + 1);
            let gy = -at(xi - 1, yi - 1) - 2.0 * at(xi, yi - 1) - at(xi + 1, yi - 1)
                + at(xi - 1, yi + 1)
                + 2.0 * at(xi, yi + 1)
                + at(xi + 1, yi + 1);
            total += (gx * gx + gy * gy).sqrt() as f64;
        }
    }

    (total / ((w as f64) * (h as f64))) as f32
}

/// Mean absolute grayscale difference between two equally-sized frames.
///
/// Used as the scene-change signal: how much this frame differs from the
/// previous kept frame. Dimensions must match (debug-asserted).
pub fn frame_diff(a: &GrayImage, b: &GrayImage) -> f32 {
    debug_assert_eq!(
        a.dimensions(),
        b.dimensions(),
        "frame_diff: dimension mismatch"
    );
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
        .map(|(&x, &y)| x.abs_diff(y) as u64)
        .sum();
    total as f32 / n as f32
}

/// Score one competition set: compute raw dimensions, rank-normalize each to
/// `0..=100` within the set, and attach the weighted final score. Consumes the
/// candidates and returns [`ScoredFrame`]s in the same order.
///
/// `prev` is the grayscale of the frame immediately preceding this set in time
/// (e.g. the previous chunk's tail): with it, the first frame's `scene_diff`
/// measures a real transition instead of defaulting to zero. Pass `None` when
/// the set has no temporal predecessor or its frames are not consecutive.
///
/// This is the pipeline's single scoring boundary — everything upstream carries
/// unscored [`CandidateFrame`]s.
pub fn score_frames(
    frames: Vec<CandidateFrame>,
    weights: &Weights,
    prev: Option<&GrayImage>,
) -> Vec<ScoredFrame> {
    let raws = compute_raw(&frames, prev);
    let scores = rank_normalize(&raws, weights);
    frames
        .into_iter()
        .zip(scores)
        .map(|(f, score)| ScoredFrame {
            ts: f.ts,
            gray: f.gray,
            rgb: f.rgb,
            score,
        })
        .collect()
}

/// Compute raw scores (clarity, edge density, scene diff) for a competition set.
///
/// `scene_diff` of the first frame is its difference from `prev` when given
/// (the frame just before this set), else 0; each subsequent frame's is its
/// difference from the previous frame in the slice.
///
/// Each frame's three dimensions are independent reads (scene_diff reads the
/// previous frame's grayscale, never mutates it), so the per-frame work runs in
/// parallel — this is the pipeline's hottest CPU stage.
fn compute_raw(frames: &[CandidateFrame], prev: Option<&GrayImage>) -> Vec<RawScores> {
    (0..frames.len())
        .into_par_iter()
        .map(|i| RawScores {
            clarity: clarity(&frames[i].gray),
            edge_density: edge_density(&frames[i].gray),
            scene_diff: match i.checked_sub(1) {
                Some(p) => frame_diff(&frames[i].gray, &frames[p].gray),
                None => prev.map_or(0.0, |p| frame_diff(&frames[0].gray, p)),
            },
        })
        .collect()
}

/// Rank-normalize each raw dimension to `0..=100` within the set, then return
/// the weighted final score per frame (index-aligned with `raws`).
///
/// For a dimension, the frame with the smallest raw value gets 0, the largest
/// gets 100, and the rest interpolate linearly by rank position. Ties share the
/// rank of their first occurrence. With fewer than two frames every score is
/// 100 (a single frame is trivially the best of its set).
fn rank_normalize(raws: &[RawScores], weights: &Weights) -> Vec<f32> {
    let n = raws.len();
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![100.0];
    }

    let clarity_rank = rank_to_percentile(raws, |r| r.clarity);
    let edge_rank = rank_to_percentile(raws, |r| r.edge_density);
    let scene_rank = rank_to_percentile(raws, |r| r.scene_diff);

    (0..n)
        .map(|i| {
            weights.scene * scene_rank[i]
                + weights.clarity * clarity_rank[i]
                + weights.edge * edge_rank[i]
        })
        .collect()
}

/// Map each raw value (via `key`) to a rank percentile in `0..=100`.
fn rank_to_percentile(raws: &[RawScores], key: impl Fn(&RawScores) -> f32) -> Vec<f32> {
    let n = raws.len();
    let mut idx: Vec<usize> = (0..n).collect();
    // Stable sort is load-bearing: tied raw values must map to deterministic
    // percentiles, or selection stops being reproducible across runs. total_cmp
    // matches partial_cmp for finite values but is faster and NaN-safe.
    idx.sort_by(|&a, &b| key(&raws[a]).total_cmp(&key(&raws[b])));

    let mut out = vec![0.0_f32; n];
    let denom = (n - 1) as f32;
    for (rank, &raw_i) in idx.iter().enumerate() {
        out[raw_i] = 100.0 * rank as f32 / denom;
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
    order.sort_by(|&a, &b| frames[b].score.total_cmp(&frames[a].score));

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
    chosen.sort_by(|&a, &b| frames[a].ts.total_cmp(&frames[b].ts));
    chosen
}

/// Pick hero frames from an already-selected set: those whose score is at least
/// one standard deviation above the mean. Returns at most `max_heroes` indices
/// (into the passed slice), highest score first.
///
/// These statistically-prominent frames drive layout template choice. An empty
/// result means a uniform grid. Callers derive `max_heroes` from grid capacity
/// (a hero occupies 4 cells, so at most `capacity / 4` heroes are placeable).
pub fn pick_heroes(selected: &[ScoredFrame], max_heroes: usize) -> Vec<usize> {
    let n = selected.len();
    if n < 2 || max_heroes == 0 {
        return Vec::new();
    }
    let mean: f32 = selected.iter().map(|f| f.score).sum::<f32>() / n as f32;
    let var: f32 = selected
        .iter()
        .map(|f| (f.score - mean).powi(2))
        .sum::<f32>()
        / n as f32;
    let std = var.sqrt();
    // No spread means no frame stands out — a uniform grid.
    if std < f32::EPSILON {
        return Vec::new();
    }
    let threshold = mean + std;

    let mut heroes: Vec<usize> = (0..n).filter(|&i| selected[i].score >= threshold).collect();
    heroes.sort_by(|&a, &b| selected[b].score.total_cmp(&selected[a].score));
    heroes.truncate(max_heroes);
    heroes
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{Luma, RgbImage};
    use std::sync::Arc;

    /// A scored frame carrying only the fields selection and hero-picking read.
    fn scored(ts: f32, score: f32) -> ScoredFrame {
        ScoredFrame {
            ts,
            gray: GrayImage::new(1, 1),
            rgb: Arc::new(RgbImage::new(1, 1)),
            score,
        }
    }

    fn raw(clarity: f32, edge: f32, scene: f32) -> RawScores {
        RawScores {
            clarity,
            edge_density: edge,
            scene_diff: scene,
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

    /// Naive fully-clamped Laplacian variance — the reference the optimized
    /// interior/border split must match exactly.
    fn clarity_naive(g: &GrayImage) -> f32 {
        let (w, h) = g.dimensions();
        let at = |x: i64, y: i64| {
            let xc = x.clamp(0, w as i64 - 1) as u32;
            let yc = y.clamp(0, h as i64 - 1) as u32;
            g.get_pixel(xc, yc).0[0] as f32
        };
        let (mut sum, mut sq) = (0.0_f64, 0.0_f64);
        for y in 0..h as i64 {
            for x in 0..w as i64 {
                let lap =
                    at(x, y - 1) + at(x - 1, y) + at(x + 1, y) + at(x, y + 1) - 4.0 * at(x, y);
                sum += lap as f64;
                sq += (lap as f64) * (lap as f64);
            }
        }
        let n = (w as f64) * (h as f64);
        let mean = sum / n;
        (sq / n - mean * mean) as f32
    }

    #[test]
    fn fast_clarity_matches_naive_clamped() {
        // A varied gradient+checker pattern exercises interior and border alike.
        let mut g = GrayImage::new(13, 9);
        for y in 0..9u32 {
            for x in 0..13u32 {
                let v = ((x * 17 + y * 31 + (x ^ y) * 5) % 256) as u8;
                g.put_pixel(x, y, Luma([v]));
            }
        }
        assert_eq!(clarity(&g), clarity_naive(&g), "interior path must match");
    }

    #[test]
    fn diffs_first_frame_against_boundary_context() {
        let candidate = |gray: GrayImage| CandidateFrame {
            ts: 0.0,
            gray,
            rgb: Arc::new(RgbImage::new(1, 1)),
        };
        let black = GrayImage::from_pixel(8, 8, Luma([0]));
        let white = GrayImage::from_pixel(8, 8, Luma([255]));
        let frames = vec![candidate(white.clone()), candidate(white)];

        let no_ctx = compute_raw(&frames, None);
        let with_ctx = compute_raw(&frames, Some(&black));

        assert_eq!(no_ctx[0].scene_diff, 0.0, "no predecessor, no diff");
        assert_eq!(
            with_ctx[0].scene_diff, 255.0,
            "boundary transition is scored, not zeroed"
        );
        assert_eq!(with_ctx[1].scene_diff, 0.0, "interior diffs unchanged");
    }

    #[test]
    fn normalizes_ranks_to_full_range() {
        // clarity ascending 1,2,3 → percentiles 0,50,100; same for others so
        // the weighted score collapses to those percentiles.
        let raws = vec![raw(1.0, 1.0, 1.0), raw(2.0, 2.0, 2.0), raw(3.0, 3.0, 3.0)];
        let weights = Weights {
            scene: 0.40,
            clarity: 0.35,
            edge: 0.25,
        };
        let scores = rank_normalize(&raws, &weights);
        assert!((scores[0] - 0.0).abs() < 1e-3);
        assert!((scores[1] - 50.0).abs() < 1e-3);
        assert!((scores[2] - 100.0).abs() < 1e-3);
    }

    #[test]
    fn single_frame_scores_full() {
        let weights = Weights {
            scene: 0.40,
            clarity: 0.35,
            edge: 0.25,
        };
        assert_eq!(rank_normalize(&[raw(9.0, 9.0, 9.0)], &weights), vec![100.0]);
    }

    #[test]
    fn enforces_minimum_temporal_gap() {
        // Two high scorers 0.5s apart; NMS with gap 2.0 keeps only the better.
        let fs = vec![scored(0.0, 90.0), scored(0.5, 95.0), scored(5.0, 80.0)];
        let sel = select_top_k(&fs, 3, 2.0);
        assert_eq!(sel, vec![1, 2], "0.5s neighbor of the best is suppressed");
    }

    #[test]
    fn returns_indices_sorted_by_time() {
        let fs = vec![scored(9.0, 50.0), scored(1.0, 99.0)];
        let sel = select_top_k(&fs, 2, 0.0);
        assert_eq!(sel, vec![1, 0], "output ordered by timestamp, not score");
    }

    #[test]
    fn picks_outlier_as_hero() {
        let fs = vec![
            scored(0.0, 50.0),
            scored(1.0, 50.0),
            scored(2.0, 50.0),
            scored(3.0, 99.0),
        ];
        let heroes = pick_heroes(&fs, 3);
        assert_eq!(heroes, vec![3]);
    }

    #[test]
    fn no_hero_when_scores_uniform() {
        let fs = vec![scored(0.0, 70.0), scored(1.0, 70.0)];
        let heroes = pick_heroes(&fs, 3);
        assert!(heroes.is_empty(), "uniform scores yield a uniform grid");
    }

    #[test]
    fn caps_hero_count_at_max() {
        // Two clear outliers, but a capacity that only affords one hero.
        let fs = vec![
            scored(0.0, 10.0),
            scored(1.0, 10.0),
            scored(2.0, 10.0),
            scored(3.0, 95.0),
            scored(4.0, 99.0),
        ];
        let heroes = pick_heroes(&fs, 1);
        assert_eq!(heroes, vec![4], "only the single best outlier is kept");
        assert!(
            pick_heroes(&fs, 0).is_empty(),
            "zero budget means no heroes"
        );
    }
}
