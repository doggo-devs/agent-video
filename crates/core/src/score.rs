// SPDX-License-Identifier: Apache-2.0
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
//! a separate step that deliberately bypasses rank normalization: it tests the
//! raw `scene_diff` magnitude against an absolute threshold, because ranks are
//! spread over the full range by construction and carry no evidence that
//! anything actually stood out.

use std::ops::{Index, IndexMut};

use image::GrayImage;
use rayon::prelude::*;

use crate::types::{CandidateFrame, SceneDiff, Score, ScoredFrame};

/// One scoring dimension. Doubles as the index into [`RawScores`] and the
/// key of [`Weights::active`], so "which dimensions exist" is stated once.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dim {
    Scene,
    Clarity,
    Edge,
}

impl Dim {
    const ALL: [Dim; 3] = [Dim::Scene, Dim::Clarity, Dim::Edge];
}

/// Raw per-dimension scores, in their natural (incomparable) units — a scoring
/// transient, never stored on a frame. Indexed by [`Dim`]: `Scene` is a mean
/// absolute grayscale difference (0-255) from the previous frame, `Clarity` a
/// Laplacian variance, `Edge` a Sobel energy mean. Slots of inactive
/// dimensions stay at their zero default and are never read.
#[derive(Debug, Clone, Copy, Default)]
struct RawScores([f32; 3]);

impl Index<Dim> for RawScores {
    type Output = f32;
    fn index(&self, d: Dim) -> &f32 {
        &self.0[d as usize]
    }
}

impl IndexMut<Dim> for RawScores {
    fn index_mut(&mut self, d: Dim) -> &mut f32 {
        &mut self.0[d as usize]
    }
}

/// Scoring dimension weights, passed from the profile. Weights are relative:
/// only ratios matter — [`Weights::new`] normalizes them to sum to 1, so the
/// weighted rank average stays inside [`Score`]'s `0..=100` contract at any
/// input scale.
///
/// A zero input weight is not a value — it is absence: the dimension is
/// simply not in the active set, so no consumer can forget to skip it (the
/// old three-named-fields shape required every call site to re-derive
/// "is this dimension on?" from `w > 0.0`, and one of them didn't).
/// Invariants — at least one active dimension, every stored weight finite
/// and strictly positive, actives summing to 1 — live here, not in a
/// caller-side check.
#[derive(Debug, Clone, Copy)]
pub struct Weights {
    w: [Option<f32>; 3],
}

impl Weights {
    /// Validate and normalize raw profile weights (in `Dim` order:
    /// scene, clarity, edge). Zero weights become inactive dimensions.
    pub fn new(scene: f32, clarity: f32, edge: f32) -> crate::error::Result<Self> {
        use crate::error::{Error, ErrorCode};
        for (name, w) in [("w_scene", scene), ("w_clarity", clarity), ("w_edge", edge)] {
            if !w.is_finite() || w < 0.0 {
                return Err(Error::new(
                    ErrorCode::InvalidConfig,
                    format!("invalid {name}: {w} (weights must be finite and >= 0)"),
                ));
            }
        }
        let total = scene + clarity + edge;
        if total <= 0.0 {
            return Err(Error::new(
                ErrorCode::InvalidConfig,
                "invalid weights: all zero (at least one must be > 0)",
            ));
        }
        // Individually finite weights can still overflow their sum; a
        // non-finite total would turn every score into NaN downstream.
        if !total.is_finite() {
            return Err(Error::new(
                ErrorCode::InvalidConfig,
                format!("invalid weights: sum overflows ({scene} + {clarity} + {edge})"),
            ));
        }
        // Normalize once here so products in rank_normalize cannot overflow
        // and the weighted average is bounded by the rank range.
        Ok(Self {
            w: [scene, clarity, edge].map(|w| (w > 0.0).then(|| w / total)),
        })
    }

    /// The dimensions that participate in scoring, with their normalized
    /// weights — the single source of truth every scoring stage iterates.
    fn active(&self) -> impl Iterator<Item = (Dim, f32)> + '_ {
        Dim::ALL
            .into_iter()
            .zip(self.w)
            .filter_map(|(d, w)| w.map(|w| (d, w)))
    }
}

/// Clamped pixel read for border convolutions: out-of-range coordinates
/// replicate the nearest edge pixel, matching a naive clamped convolution.
fn at_clamped(buf: &[u8], w: usize, h: usize, x: i64, y: i64) -> f32 {
    let xc = x.clamp(0, w as i64 - 1) as usize;
    let yc = y.clamp(0, h as i64 - 1) as usize;
    buf[yc * w + xc] as f32
}

/// Visit every pixel of a `w`×`h` image that is NOT a 3x3-interior pixel,
/// exactly once: top and bottom rows, then the side columns between them.
/// For `w < 3` or `h < 3` that is every pixel — no pixel has a full 3x3
/// neighbourhood.
fn for_each_border(w: usize, h: usize, mut f: impl FnMut(usize, usize)) {
    if w == 0 || h == 0 {
        return;
    }
    for x in 0..w {
        f(x, 0);
    }
    if h > 1 {
        for x in 0..w {
            f(x, h - 1);
        }
    }
    for y in 1..h.saturating_sub(1) {
        f(0, y);
        if w > 1 {
            f(w - 1, y);
        }
    }
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
    let at = |x: i64, y: i64| at_clamped(buf, wu, hu, x, y);
    for_each_border(wu, hu, |x, y| {
        let (xi, yi) = (x as i64, y as i64);
        let lap =
            at(xi, yi - 1) + at(xi - 1, yi) + at(xi + 1, yi) + at(xi, yi + 1) - 4.0 * at(xi, yi);
        accumulate(lap);
    });

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

    let at = |x: i64, y: i64| at_clamped(buf, wu, hu, x, y);
    for_each_border(wu, hu, |x, y| {
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
    });

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
    let raws = compute_raw(&frames, prev, weights);
    let scores = rank_normalize(&raws, weights);
    frames
        .into_iter()
        .zip(scores.into_iter().zip(raws))
        .map(|(f, (score, raw))| ScoredFrame {
            ts: f.ts,
            gray: f.gray,
            rgb: f.rgb,
            score: mint_score(score),
            scene_diff: SceneDiff::new(raw[Dim::Scene])
                .expect("mean absolute u8 difference lies in 0..=255"),
        })
        .collect()
}

/// Wrap a weighted rank average into a [`Score`].
///
/// The average is mathematically bounded by `0..=100` (normalized weights ×
/// percentiles), but float rounding of the products can overshoot the bound
/// by an ulp; the clamp removes exactly that noise, nothing more.
fn mint_score(score: f32) -> Score {
    Score::new(score.clamp(0.0, 100.0)).expect("clamped weighted rank average is finite")
}

/// Re-rank frames that are NOT temporally consecutive — e.g. per-batch
/// winners compacted into one global competition set.
///
/// Recomputing `scene_diff` between such frames would measure the difference
/// of frames minutes apart, which is noise, not a scene transition. Each
/// frame instead keeps the `scene_diff` it earned against its true temporal
/// predecessor at first scoring; clarity and edge density are intrinsic to a
/// frame, so recomputing them is exact. Scores are rank-normalized within
/// this set, consistent with [`score_frames`]'s per-set contract.
pub fn rescore_frames(mut frames: Vec<ScoredFrame>, weights: &Weights) -> Vec<ScoredFrame> {
    let raws: Vec<RawScores> = frames
        .par_iter()
        .map(|f| {
            let mut raw = RawScores::default();
            raw[Dim::Scene] = f.scene_diff.get();
            for (dim, _) in weights.active() {
                match dim {
                    Dim::Clarity => raw[Dim::Clarity] = clarity(&f.gray),
                    Dim::Edge => raw[Dim::Edge] = edge_density(&f.gray),
                    // Carried, never recomputed — filled above.
                    Dim::Scene => {}
                }
            }
            raw
        })
        .collect();
    let scores = rank_normalize(&raws, weights);
    for (f, score) in frames.iter_mut().zip(scores) {
        f.score = mint_score(score);
    }
    frames
}

/// Compute raw scores (clarity, edge density, scene diff) for a competition set.
///
/// `scene_diff` of the first frame is its difference from `prev` when given
/// (the frame just before this set), else 0; each subsequent frame's is its
/// difference from the previous frame in the slice.
///
/// Each frame's dimensions are independent reads (scene_diff reads the
/// previous frame's grayscale, never mutates it), so the per-frame work runs
/// in parallel — this is the pipeline's hottest CPU stage. Inactive
/// convolution dimensions are structurally absent from [`Weights::active`],
/// so their cost is never paid. `scene_diff` is computed unconditionally: it
/// is a cheap linear pass, and hero detection reads its raw magnitude even
/// when the dimension is inactive.
fn compute_raw(
    frames: &[CandidateFrame],
    prev: Option<&GrayImage>,
    weights: &Weights,
) -> Vec<RawScores> {
    (0..frames.len())
        .into_par_iter()
        .map(|i| {
            let mut raw = RawScores::default();
            raw[Dim::Scene] = match i.checked_sub(1) {
                Some(p) => frame_diff(&frames[i].gray, &frames[p].gray),
                None => prev.map_or(0.0, |p| frame_diff(&frames[0].gray, p)),
            };
            for (dim, _) in weights.active() {
                match dim {
                    Dim::Clarity => raw[Dim::Clarity] = clarity(&frames[i].gray),
                    Dim::Edge => raw[Dim::Edge] = edge_density(&frames[i].gray),
                    // Unconditional — filled above.
                    Dim::Scene => {}
                }
            }
            raw
        })
        .collect()
}

/// Rank-normalize each active dimension to `0..=100` within the set, then
/// return the weighted final score per frame (index-aligned with `raws`).
///
/// For a dimension, the frame with the smallest raw value gets 0, the largest
/// gets 100, and the rest interpolate linearly by rank position. Ties do not
/// share a rank: equal raw values receive distinct adjacent ranks in input
/// order (stable sort), so the contract is determinism, not tie fairness — a
/// set of identical raw values still spreads across `0..=100`. With fewer than
/// two frames every score is 100 (a single frame is trivially the best of its
/// set).
///
/// Only [`Weights::active`] dimensions are ranked — an inactive dimension
/// costs neither the sort nor the allocation. Active weights sum to 1 by
/// [`Weights::new`]'s contract, so the accumulated score stays inside the
/// rank range without further normalization.
fn rank_normalize(raws: &[RawScores], weights: &Weights) -> Vec<f32> {
    let n = raws.len();
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![100.0];
    }

    let mut scores = vec![0.0_f32; n];
    for (dim, w) in weights.active() {
        let ranks = rank_to_percentile(raws, dim);
        for (score, rank) in scores.iter_mut().zip(ranks) {
            *score += w * rank;
        }
    }
    scores
}

/// Map each raw value of `dim` to a rank percentile in `0..=100`.
fn rank_to_percentile(raws: &[RawScores], dim: Dim) -> Vec<f32> {
    let n = raws.len();
    let mut idx: Vec<usize> = (0..n).collect();
    // Stable sort is load-bearing: tied raw values must map to deterministic
    // percentiles, or selection stops being reproducible across runs. total_cmp
    // matches partial_cmp for finite values but is faster and NaN-safe.
    idx.sort_by(|&a, &b| raws[a][dim].total_cmp(&raws[b][dim]));

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
    order.sort_by(|&a, &b| frames[b].score.get().total_cmp(&frames[a].score.get()));

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

/// Default minimum raw `scene_diff` (mean absolute grayscale difference,
/// `0..=255`) for a selected frame to qualify as a hero.
///
/// Calibrated visually on 1 fps samples of fast-edited real footage: one-second
/// gaps within the same scene (camera pans, subject motion, animation) stayed
/// below ~67 while hard cuts clustered above ~70, with a narrow overlap. The
/// default sits at the top of that overlap because the two errors are
/// asymmetric: a false hero spends ~4 cells of canvas on an ordinary frame,
/// while a missed cut still shows at normal cell size. Content with a
/// different `scene_diff` distribution tunes this via `--hero-scene-diff`.
pub const DEFAULT_HERO_SCENE_DIFF: f32 = 70.0;

/// Pick hero frames from an already-selected set: those whose raw scene
/// difference (at least `scene_diff_min`, in `0..=255` units) marks a hard
/// scene change. Returns at most `max_heroes` indices (into the passed
/// slice), strongest change first.
///
/// The criterion is an absolute magnitude, not a within-set statistic: rank
/// normalization spreads scores over `0..=100` by construction, so any
/// statistical outlier test on them fires at a content-independent rate. Raw
/// `scene_diff` has fixed units, which makes "no cuts, no heroes" hold across
/// chunks and videos. An empty result means a uniform grid. Callers derive
/// `max_heroes` from grid capacity (a hero occupies 4 cells, so at most
/// `capacity / 4` heroes are placeable).
pub fn pick_heroes(selected: &[ScoredFrame], max_heroes: usize, scene_diff_min: f32) -> Vec<usize> {
    let mut heroes: Vec<usize> = (0..selected.len())
        .filter(|&i| selected[i].scene_diff.get() >= scene_diff_min)
        .collect();
    heroes.sort_by(|&a, &b| {
        selected[b]
            .scene_diff
            .get()
            .total_cmp(&selected[a].scene_diff.get())
    });
    heroes.truncate(max_heroes);
    heroes
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{Luma, RgbImage};
    use std::sync::Arc;

    #[test]
    fn border_visits_each_non_interior_pixel_once() {
        // The border set must be exactly the complement of the 3x3-interior
        // set, with no pixel visited twice — including degenerate shapes
        // where every pixel is border.
        for (w, h) in [(5, 4), (3, 3), (2, 5), (5, 2), (1, 1), (1, 6), (7, 1)] {
            let mut counts = vec![0u8; w * h];
            for_each_border(w, h, |x, y| counts[y * w + x] += 1);
            for y in 0..h {
                for x in 0..w {
                    let interior = x >= 1 && y >= 1 && x + 1 < w && y + 1 < h;
                    let expected = if interior { 0 } else { 1 };
                    assert_eq!(
                        counts[y * w + x],
                        expected,
                        "pixel ({x},{y}) in {w}x{h} visited {} times",
                        counts[y * w + x]
                    );
                }
            }
        }
    }

    /// A scored frame carrying only the fields selection reads.
    fn scored(ts: f32, score: f32) -> ScoredFrame {
        ScoredFrame {
            ts,
            gray: GrayImage::new(1, 1),
            rgb: Arc::new(RgbImage::new(1, 1)),
            score: Score::new(score).expect("test score in 0..=100"),
            scene_diff: SceneDiff::new(0.0).expect("zero diff is valid"),
        }
    }

    /// A scored frame carrying only the field hero-picking reads.
    fn with_diff(ts: f32, scene_diff: f32) -> ScoredFrame {
        ScoredFrame {
            scene_diff: SceneDiff::new(scene_diff).expect("test diff in 0..=255"),
            ..scored(ts, 50.0)
        }
    }

    fn raw(clarity: f32, edge: f32, scene: f32) -> RawScores {
        let mut r = RawScores::default();
        r[Dim::Clarity] = clarity;
        r[Dim::Edge] = edge;
        r[Dim::Scene] = scene;
        r
    }

    fn weights(scene: f32, clarity: f32, edge: f32) -> Weights {
        Weights::new(scene, clarity, edge).expect("valid test weights")
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
        let weights = weights(1.0, 0.0, 0.0);

        let no_ctx = compute_raw(&frames, None, &weights);
        let with_ctx = compute_raw(&frames, Some(&black), &weights);

        assert_eq!(no_ctx[0][Dim::Scene], 0.0, "no predecessor, no diff");
        assert_eq!(
            with_ctx[0][Dim::Scene],
            255.0,
            "boundary transition is scored, not zeroed"
        );
        assert_eq!(with_ctx[1][Dim::Scene], 0.0, "interior diffs unchanged");
    }

    #[test]
    fn normalizes_ranks_to_full_range() {
        // clarity ascending 1,2,3 → percentiles 0,50,100; same for others so
        // the weighted score collapses to those percentiles.
        let raws = vec![raw(1.0, 1.0, 1.0), raw(2.0, 2.0, 2.0), raw(3.0, 3.0, 3.0)];
        let scores = rank_normalize(&raws, &weights(0.40, 0.35, 0.25));
        assert!((scores[0] - 0.0).abs() < 1e-3);
        assert!((scores[1] - 50.0).abs() < 1e-3);
        assert!((scores[2] - 100.0).abs() < 1e-3);
    }

    #[test]
    fn keeps_scores_in_range_for_unnormalized_weights() {
        // CLI weights need not sum to 1 (e.g. --w-scene 100 --w-edge 100);
        // scores must still land in 0..=100 per ScoredFrame's contract, and
        // scaling all weights uniformly must not change any score.
        let raws = vec![raw(1.0, 1.0, 1.0), raw(2.0, 2.0, 2.0), raw(3.0, 3.0, 3.0)];
        let unit = weights(0.5, 0.0, 0.5);
        let scaled = weights(100.0, 0.0, 100.0);
        let unit_scores = rank_normalize(&raws, &unit);
        let scaled_scores = rank_normalize(&raws, &scaled);
        assert_eq!(unit_scores, scaled_scores, "only weight ratios matter");
        assert!(
            scaled_scores.iter().all(|&s| (0.0..=100.0).contains(&s)),
            "scores stay in 0..=100: {scaled_scores:?}"
        );
    }

    #[test]
    fn rejects_weights_whose_sum_overflows() {
        // Finite weights, non-finite sum — would NaN every downstream score.
        Weights::new(f32::MAX, f32::MAX, 0.0).expect_err("overflowing sum must be rejected");
    }

    #[test]
    fn single_frame_scores_full() {
        let w = weights(0.40, 0.35, 0.25);
        assert_eq!(rank_normalize(&[raw(9.0, 9.0, 9.0)], &w), vec![100.0]);
    }

    #[test]
    fn excludes_zero_weight_dimensions_from_the_active_set() {
        // A zero weight is absence: the dimension must not appear in the
        // iteration every scoring stage drives.
        let w = weights(1.0, 0.0, 3.0);
        let active: Vec<Dim> = w.active().map(|(d, _)| d).collect();
        assert_eq!(active, vec![Dim::Scene, Dim::Edge]);
    }

    #[test]
    fn normalizes_active_weights_to_unit_sum() {
        let w = weights(1.0, 0.0, 3.0);
        let total: f32 = w.active().map(|(_, w)| w).sum();
        assert!((total - 1.0).abs() < 1e-6, "active weights sum to 1");
    }

    #[test]
    fn rescore_ranks_by_carried_scene_diff_not_slice_neighbors() {
        // All frames share identical pixels: recomputing scene_diff between
        // slice neighbors would yield 0.0 for every frame and the scene
        // dimension would collapse to input-order noise. The carried values
        // (earned against true temporal predecessors) must drive the rank.
        let weights = weights(1.0, 0.0, 0.0);
        let frames = vec![
            with_diff(0.0, 10.0),
            with_diff(60.0, 90.0), // strongest real transition
            with_diff(120.0, 40.0),
        ];
        let rescored = rescore_frames(frames, &weights);
        let best = rescored
            .iter()
            .max_by(|a, b| a.score.get().total_cmp(&b.score.get()))
            .expect("non-empty set");
        assert_eq!(best.ts, 60.0, "carried scene_diff decides the winner");
        assert_eq!(
            best.scene_diff.get(),
            90.0,
            "carried raw scene_diff survives the re-rank"
        );
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
    fn picks_hard_cut_as_hero() {
        let fs = vec![
            with_diff(0.0, 15.0),
            with_diff(1.0, 40.0),
            with_diff(2.0, 20.0),
            with_diff(3.0, 90.0),
        ];
        let heroes = pick_heroes(&fs, 3, DEFAULT_HERO_SCENE_DIFF);
        assert_eq!(heroes, vec![3]);
    }

    #[test]
    fn no_hero_without_hard_cut() {
        // Plenty of score spread, but every scene_diff is same-scene motion:
        // the hero test must read raw magnitude, not relative prominence.
        let mut fs = vec![
            with_diff(0.0, 5.0),
            with_diff(1.0, 30.0),
            with_diff(2.0, 60.0),
        ];
        fs[2].score = Score::new(99.0).expect("test score in 0..=100");
        let heroes = pick_heroes(&fs, 3, DEFAULT_HERO_SCENE_DIFF);
        assert!(heroes.is_empty(), "no cut means a uniform grid");
    }

    #[test]
    fn lowered_threshold_admits_softer_cuts() {
        let fs = vec![with_diff(0.0, 15.0), with_diff(1.0, 65.0)];
        assert!(
            pick_heroes(&fs, 3, DEFAULT_HERO_SCENE_DIFF).is_empty(),
            "65 is same-scene motion at the default threshold"
        );
        assert_eq!(pick_heroes(&fs, 3, 60.0), vec![1]);
    }

    #[test]
    fn orders_heroes_by_cut_strength_and_caps_at_max() {
        // Two hard cuts, but a capacity that only affords one hero.
        let fs = vec![
            with_diff(0.0, 10.0),
            with_diff(1.0, 95.0),
            with_diff(2.0, 10.0),
            with_diff(3.0, 140.0),
        ];
        let t = DEFAULT_HERO_SCENE_DIFF;
        assert_eq!(pick_heroes(&fs, 2, t), vec![3, 1], "strongest change first");
        assert_eq!(
            pick_heroes(&fs, 1, t),
            vec![3],
            "budget keeps the strongest"
        );
        assert!(
            pick_heroes(&fs, 0, t).is_empty(),
            "zero budget means no heroes"
        );
    }
}
