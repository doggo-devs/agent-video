// SPDX-License-Identifier: Apache-2.0
//! Grid geometry. Pure calculation — no image operations.
//!
//! The layout is justified: frames are partitioned into rows, every row spans
//! the full canvas width, and each cell keeps the source aspect ratio exactly.
//! Cell counts may differ between rows (so cell sizes vary slightly row to
//! row), but rows always fill their width — the canvas is tiled completely,
//! with no filler cells, by construction.
//!
//! Hero rows come first and hold half as many cells as normal rows, making a
//! hero ~2× larger on each edge. A short hero row is topped up with the
//! highest-scored normal frames instead of leaving holes. Portrait sources
//! transpose the whole scheme: justified columns, heroes on the left.
//! [`CanvasSpec::edge`] caps the longer canvas edge; row counts are chosen so
//! the canvas stays roughly square, which maximizes the pixel budget a VLM
//! actually sees. The template ([`LayoutKind`]) simply labels the hero count
//! for metadata.

use std::mem;

use crate::types::{CanvasSpec, LayoutKind, Rect};

/// A computed grid: one rectangle per frame (in input order) plus canvas size
/// and the template name driven by hero count.
#[derive(Debug, Clone)]
pub struct Layout {
    /// Pixel cell per frame, indexed as the input frames were. `None` when
    /// rounding on an over-packed canvas left the frame no non-degenerate
    /// cell — composition skips it, and hero eviction treats it as below any
    /// readable floor. Capacity-driven callers never see it in practice.
    pub rects: Vec<Option<Rect>>,
    pub canvas_w: u32,
    pub canvas_h: u32,
    /// Template for metadata, named by hero count.
    pub template: LayoutKind,
}

/// How many cells fit on the canvas at the minimum readable cell size.
///
/// An estimate: treats the canvas as square with edge [`CanvasSpec::edge`],
/// while [`compute`] only keeps it roughly square. A cell's shorter edge is
/// `min_cell_edge`, so its area is `min_cell_edge² × max(aspect, 1/aspect)`.
/// Returns at least 1.
pub fn grid_capacity(spec: &CanvasSpec, aspect: f32) -> usize {
    let aspect = sane_aspect(aspect);
    let edge = spec.edge() as f32;
    let long_ratio = aspect.max(1.0 / aspect);
    let cell_area = (spec.min_cell_edge() as f32).powi(2) * long_ratio;
    ((edge * edge) / cell_area).floor().max(1.0) as usize
}

/// Guard against a degenerate aspect (0, negative, NaN, inf), which would
/// round the whole geometry into zero-sized rects or a NaN canvas. Frames
/// decoded by [`crate::extract`] always yield a finite positive aspect, so the
/// square fallback is defense in depth for other callers, not a normal path.
fn sane_aspect(aspect: f32) -> f32 {
    if aspect.is_finite() && aspect > 0.0 {
        aspect
    } else {
        1.0
    }
}

fn template_kind(hero_count: usize) -> LayoutKind {
    match hero_count {
        0 => LayoutKind::Uniform,
        1 => LayoutKind::HeroLeft,
        2 => LayoutKind::DualHero,
        _ => LayoutKind::TopStrip,
    }
}

/// Compute the layout for the frames whose scores are given (one per frame,
/// in time order), where `heroes` lists hero frame indices. `aspect` is the
/// source frame `w/h`.
///
/// Scores rank normal frames for promotion into a short hero row; they play
/// no other role. The returned `rects` are aligned with the input frame order.
pub fn compute(scores: &[f32], heroes: &[usize], aspect: f32, spec: &CanvasSpec) -> Layout {
    let aspect = sane_aspect(aspect);
    if scores.is_empty() {
        return Layout {
            rects: Vec::new(),
            canvas_w: 0,
            canvas_h: 0,
            template: LayoutKind::Uniform,
        };
    }
    if aspect < 1.0 {
        // Portrait: lay out transposed, then swap axes so hero rows become
        // hero columns on the left.
        let mut layout = compute_wide(scores, heroes, 1.0 / aspect, spec);
        for r in &mut layout.rects {
            *r = r.map(Rect::transposed);
        }
        mem::swap(&mut layout.canvas_w, &mut layout.canvas_h);
        return layout;
    }
    compute_wide(scores, heroes, aspect, spec)
}

/// Justified layout for cell aspect `a >= 1` (landscape orientation).
fn compute_wide(scores: &[f32], heroes: &[usize], a: f32, spec: &CanvasSpec) -> Layout {
    let n = scores.len();

    // Dedup: a repeated index must not occupy two hero slots.
    let mut is_hero = vec![false; n];
    let mut hero_list: Vec<usize> = Vec::new();
    for &h in heroes {
        if h < n && !is_hero[h] {
            is_hero[h] = true;
            hero_list.push(h);
        }
    }
    let hero_count = hero_list.len();

    // Cells per row for a roughly-square canvas; a hero is worth ~4 normal
    // cells of area (2× per edge), hence the weighted demand.
    let weighted = (n - hero_count) + 4 * hero_count;
    let c = ((weighted as f32 / a).sqrt().round() as usize).max(1);
    let k_hero = c.div_ceil(2);

    // Hero rows first, in groups of k_hero; a short last group is topped up
    // with the best-scored normals to span the full width.
    hero_list.sort_unstable();
    let mut rows: Vec<Vec<usize>> = hero_list.chunks(k_hero).map(<[usize]>::to_vec).collect();
    let mut in_hero_row = is_hero;
    if let Some(last) = rows.last_mut() {
        let deficit = k_hero - last.len();
        if deficit > 0 {
            let mut pool: Vec<usize> = (0..n).filter(|&i| !in_hero_row[i]).collect();
            pool.sort_by(|&x, &y| scores[y].total_cmp(&scores[x]));
            for i in pool.into_iter().take(deficit) {
                in_hero_row[i] = true;
                last.push(i);
            }
            last.sort_unstable();
        }
    }

    // Remaining frames split into evenly-sized rows (differ by at most one).
    let normals: Vec<usize> = (0..n).filter(|&i| !in_hero_row[i]).collect();
    if !normals.is_empty() {
        let row_count = normals.len().div_ceil(c);
        let base = normals.len() / row_count;
        let extra = normals.len() % row_count;
        let mut it = normals.into_iter();
        for i in 0..row_count {
            let take = base + usize::from(i < extra);
            rows.push(it.by_ref().take(take).collect());
        }
    }

    // Pixel sizing: row height = width / (cells × a), longer edge capped at spec.edge.
    let height_ratio: f32 = rows.iter().map(|row| 1.0 / (row.len() as f32 * a)).sum();
    let width = spec.edge() as f32 / height_ratio.max(1.0);

    // Cell edges derive from shared rounded boundaries so rows tile without
    // gaps; a zero-extent boundary pair yields `None` (no degenerate Rect).
    let mut rects: Vec<Option<Rect>> = vec![None; n];
    let mut y = 0.0_f32;
    for row in &rows {
        let k = row.len() as f32;
        let row_h = width / (k * a);
        let y0 = y.round() as u32;
        let y1 = (y + row_h).round() as u32;
        for (j, &idx) in row.iter().enumerate() {
            let x0 = (j as f32 * width / k).round() as u32;
            let x1 = ((j + 1) as f32 * width / k).round() as u32;
            rects[idx] = Rect::new(x0, y0, x1 - x0, y1 - y0);
        }
        y += row_h;
    }

    Layout {
        rects,
        canvas_w: width.round() as u32,
        canvas_h: y.round() as u32,
        template: template_kind(hero_count),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec() -> CanvasSpec {
        crate::types::CanvasSpec::new(8000, 512).expect("valid test spec")
    }

    /// Uniform scores for `n` frames (promotion order irrelevant).
    fn flat(n: usize) -> Vec<f32> {
        vec![50.0; n]
    }

    /// Rects are in bounds, non-overlapping, all placed, and tile the canvas
    /// completely — justified layout leaves no filler area by construction.
    fn assert_tiles(layout: &Layout) {
        let mut area = 0_u64;
        let rects: Vec<Rect> = layout
            .rects
            .iter()
            .map(|r| r.expect("every frame gets a cell"))
            .collect();
        for r in &rects {
            assert!(
                r.x() + r.w() <= layout.canvas_w && r.y() + r.h() <= layout.canvas_h,
                "rect {r:?} exceeds canvas {}x{}",
                layout.canvas_w,
                layout.canvas_h
            );
            area += u64::from(r.w()) * u64::from(r.h());
        }
        for i in 0..rects.len() {
            for j in (i + 1)..rects.len() {
                assert!(
                    !rects[i].overlaps(&rects[j]),
                    "overlap: {:?} vs {:?}",
                    rects[i],
                    rects[j]
                );
            }
        }
        assert_eq!(
            area,
            u64::from(layout.canvas_w) * u64::from(layout.canvas_h),
            "cells must tile the canvas with no holes"
        );
    }

    #[test]
    fn lays_out_uniform_landscape() {
        let l = compute(&flat(12), &[], 16.0 / 9.0, &spec());
        assert_eq!(l.template, LayoutKind::Uniform);
        assert_eq!(l.rects.len(), 12);
        assert_tiles(&l);
    }

    #[test]
    fn lays_out_uniform_portrait() {
        let l = compute(&flat(12), &[], 9.0 / 16.0, &spec());
        assert_tiles(&l);
        // Portrait cells are taller than wide.
        let r = l.rects[0].expect("cell placed");
        assert!(
            r.h() > r.w(),
            "portrait cell should be taller than wide: {r:?}"
        );
    }

    #[test]
    fn landscape_cell_is_wider_than_tall() {
        let l = compute(&flat(12), &[], 16.0 / 9.0, &spec());
        let r = l.rects[0].expect("cell placed");
        assert!(
            r.w() > r.h(),
            "landscape cell should be wider than tall: {r:?}"
        );
    }

    #[test]
    fn awkward_frame_count_still_tiles() {
        // 7 is prime: cannot tile a rectangle exactly.
        for n in [3, 5, 7, 11, 13] {
            let l = compute(&flat(n), &[], 16.0 / 9.0, &spec());
            assert_tiles(&l);
        }
    }

    #[test]
    fn hero_cell_is_larger_than_normal() {
        // Frame 9 has the top normal score, so it gets promoted beside the
        // hero; frame 1 stays in a normal row for the size comparison.
        let mut scores = vec![10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 90.0];
        scores[0] = 95.0;
        let l = compute(&scores, &[0], 16.0 / 9.0, &spec());
        assert_eq!(l.template, LayoutKind::HeroLeft);
        assert_tiles(&l);
        let hero = l.rects[0].expect("hero placed");
        let normal = l.rects[1].expect("normal placed");
        assert!(
            hero.w() > normal.w() && hero.h() > normal.h(),
            "hero must be bigger: {hero:?} vs {normal:?}"
        );
    }

    #[test]
    fn deduplicates_repeated_hero_index() {
        // A repeated index must occupy one hero slot, not two. HERO_LEFT
        // (single hero) is the correct template.
        let l = compute(&flat(10), &[0, 0], 16.0 / 9.0, &spec());
        assert_eq!(l.template, LayoutKind::HeroLeft);
        assert_tiles(&l);
    }

    #[test]
    fn dual_hero_names_template() {
        let l = compute(&flat(10), &[0, 1], 16.0 / 9.0, &spec());
        assert_eq!(l.template, LayoutKind::DualHero);
        assert_tiles(&l);
    }

    #[test]
    fn three_heroes_make_top_strip() {
        let l = compute(&flat(12), &[0, 1, 2], 16.0 / 9.0, &spec());
        assert_eq!(l.template, LayoutKind::TopStrip);
        assert_tiles(&l);
    }

    #[test]
    fn portrait_hero_lays_out_left_column() {
        let l = compute(&flat(10), &[0], 9.0 / 16.0, &spec());
        assert_tiles(&l);
        // Hero anchors at the top-left: hero rows transpose to left columns.
        let hero = l.rects[0].expect("hero placed");
        assert_eq!((hero.x(), hero.y()), (0, 0));
    }

    #[test]
    fn hero_only_chunk_tiles_without_normals() {
        // Eviction can leave nothing but heroes; rows shrink to fit them.
        let l = compute(&flat(2), &[0, 1], 16.0 / 9.0, &spec());
        assert_eq!(l.template, LayoutKind::DualHero);
        assert_tiles(&l);
    }

    #[test]
    fn lower_quality_packs_more_frames_on_same_canvas() {
        // Quality trades per-cell size for capacity at a fixed canvas edge:
        // halving the cell floor quadruples how many frames fit.
        let hi = grid_capacity(
            &crate::types::CanvasSpec::new(8000, 512).expect("valid test spec"),
            1.0,
        );
        let lo = grid_capacity(
            &crate::types::CanvasSpec::new(8000, 256).expect("valid test spec"),
            1.0,
        );
        assert_eq!(lo, hi * 4);
    }

    #[test]
    fn degenerate_aspect_falls_back_to_square() {
        // 0, NaN, and inf would otherwise produce zero-sized rects or a NaN
        // canvas; the layout must stay a valid tiling.
        for aspect in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            let l = compute(&flat(6), &[], aspect, &spec());
            assert_tiles(&l);
            assert!(
                grid_capacity(&spec(), aspect) >= 1,
                "capacity stays positive for aspect {aspect}"
            );
        }
    }

    #[test]
    fn single_frame_fills_whole_canvas() {
        let l = compute(&flat(1), &[], 1.0, &spec());
        assert_eq!(l.rects.len(), 1);
        assert_tiles(&l);
        let r = l.rects[0].expect("cell placed");
        assert_eq!(r.w(), l.canvas_w);
        assert_eq!(r.h(), l.canvas_h);
    }
}
