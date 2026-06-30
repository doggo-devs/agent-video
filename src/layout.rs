//! Grid geometry. Pure calculation — no image operations.
//!
//! The layout is aspect-aware: cell aspect ratio follows the source frame, so
//! landscape and portrait videos tile differently. [`CanvasSpec::edge`] caps
//! the canvas **edge** (the longer side, in pixels); the grid is sized to keep
//! the canvas roughly square, which maximizes the pixel budget a VLM actually
//! sees.
//!
//! Hero frames occupy 2×2 blocks; normal frames a single cell. Placement uses
//! an occupancy packer rather than fixed-coordinate templates: it generalizes
//! to any frame count and guarantees non-overlapping, in-bounds rectangles. The
//! four template names (UNIFORM / HERO_LEFT / DUAL_HERO / TOP_STRIP) simply
//! label the hero count for metadata. Scan order is orientation-aware so heroes
//! land top-left and read naturally (a top strip for landscape, a left column
//! for portrait).

use crate::types::{CanvasSpec, Rect};

/// A computed grid: one rectangle per frame (in input order) plus canvas size
/// and the template name driven by hero count.
#[derive(Debug, Clone)]
pub struct Layout {
    /// Pixel rectangle for each frame, indexed as the input frames were.
    pub rects: Vec<Rect>,
    pub canvas_w: u32,
    pub canvas_h: u32,
    /// Template name for metadata: UNIFORM / HERO_LEFT / DUAL_HERO / TOP_STRIP.
    pub template: &'static str,
}

/// How many cells fit on the canvas at the minimum readable cell size.
///
/// An estimate: treats the canvas as square with edge [`CanvasSpec::edge`],
/// while [`compute`] only keeps it roughly square. A cell's shorter edge is
/// `min_cell_edge`, so its area is `min_cell_edge² × max(aspect, 1/aspect)`.
/// Returns at least 1.
pub fn grid_capacity(spec: &CanvasSpec, aspect: f32) -> usize {
    let edge = spec.edge as f32;
    let long_ratio = aspect.max(1.0 / aspect);
    let cell_area = (spec.min_cell_edge as f32).powi(2) * long_ratio;
    ((edge * edge) / cell_area).floor().max(1.0) as usize
}

fn template_name(hero_count: usize) -> &'static str {
    match hero_count {
        0 => "UNIFORM",
        1 => "HERO_LEFT",
        2 => "DUAL_HERO",
        _ => "TOP_STRIP",
    }
}

/// Compute the layout for `n` frames, where `heroes` lists which frame indices
/// (into `0..n`) are heroes. `aspect` is the source frame `w/h`.
///
/// Heroes occupy 2×2 cell blocks; all others a single cell. The returned
/// `rects` are aligned with the input frame order.
pub fn compute(n: usize, heroes: &[usize], aspect: f32, spec: &CanvasSpec) -> Layout {
    if n == 0 {
        return Layout {
            rects: Vec::new(),
            canvas_w: 0,
            canvas_h: 0,
            template: "UNIFORM",
        };
    }

    // Mark hero frames, deduplicating and counting in a single pass: a caller
    // may pass the same index twice, which must not place two 2×2 blocks.
    let mut hero_set = vec![false; n];
    let mut hero_count = 0;
    for &h in heroes {
        if h < n && !hero_set[h] {
            hero_set[h] = true;
            hero_count += 1;
        }
    }

    // Weighted cell demand: a hero needs 4 cells, a normal 1.
    let weighted = (n - hero_count) + 4 * hero_count;

    // Grid dimensions sized to keep the canvas roughly square given cell aspect.
    let (cols, rows) = grid_dims(weighted, aspect, hero_count > 0);

    // Occupancy grid; place heroes (score order) first, then normals (time order).
    let landscape = aspect >= 1.0;
    let mut occupied = vec![false; cols * rows];
    let mut grid_cells: Vec<Option<(usize, usize, usize, usize)>> = vec![None; n];

    for &h in heroes {
        if h >= n || grid_cells[h].is_some() {
            continue;
        }
        if let Some((gx, gy)) = find_block(&occupied, cols, rows, 2, 2, landscape) {
            mark(&mut occupied, cols, gx, gy, 2, 2);
            grid_cells[h] = Some((gx, gy, 2, 2));
        }
    }
    for i in 0..n {
        if hero_set[i] {
            continue;
        }
        if let Some((gx, gy)) = find_block(&occupied, cols, rows, 1, 1, landscape) {
            mark(&mut occupied, cols, gx, gy, 1, 1);
            grid_cells[i] = Some((gx, gy, 1, 1));
        }
    }

    // Pixel sizing: keep cell aspect = source aspect, cap the longer canvas edge.
    let edge = spec.edge as f32;
    let cell_h = edge / (cols as f32 * aspect).max(rows as f32);
    let cell_w = cell_h * aspect;

    let rects = grid_cells
        .iter()
        .map(|cell| match cell {
            Some((gx, gy, gw, gh)) => {
                // Derive edges from rounded cell boundaries so adjacent cells
                // share an exact boundary — independent rounding would overlap.
                let x0 = (*gx as f32 * cell_w).round() as u32;
                let y0 = (*gy as f32 * cell_h).round() as u32;
                let x1 = ((*gx + *gw) as f32 * cell_w).round() as u32;
                let y1 = ((*gy + *gh) as f32 * cell_h).round() as u32;
                Rect {
                    x: x0,
                    y: y0,
                    w: x1 - x0,
                    h: y1 - y0,
                }
            }
            // Unplaced (packer ran out of room) — zero-size, skipped by compose.
            None => Rect {
                x: 0,
                y: 0,
                w: 0,
                h: 0,
            },
        })
        .collect();

    let canvas_w = (cols as f32 * cell_w).round() as u32;
    let canvas_h = (rows as f32 * cell_h).round() as u32;

    Layout {
        rects,
        canvas_w,
        canvas_h,
        template: template_name(hero_count),
    }
}

/// Choose `(cols, rows)` so `cols × rows ≥ weighted` and the canvas (cells
/// scaled by `aspect`) is as square as possible. When heroes are present both
/// dimensions are at least 2 so a 2×2 block fits.
fn grid_dims(weighted: usize, aspect: f32, has_hero: bool) -> (usize, usize) {
    let min_dim = if has_hero { 2 } else { 1 };
    // round(sqrt(weighted / aspect)) makes the *canvas* (not the grid) square.
    let mut cols = ((weighted as f32 / aspect).sqrt().round() as usize).max(min_dim);
    let mut rows = weighted.div_ceil(cols).max(min_dim);
    // Grow to guarantee capacity after the min_dim clamps.
    while cols * rows < weighted {
        if (cols as f32 * aspect) <= rows as f32 {
            cols += 1;
        } else {
            rows += 1;
        }
    }
    (cols, rows)
}

/// Find the top-left of the first free `bw × bh` block, scanning in an
/// orientation-aware order (row-major for landscape, column-major for portrait)
/// so heroes cluster top-left.
fn find_block(
    occupied: &[bool],
    cols: usize,
    rows: usize,
    bw: usize,
    bh: usize,
    landscape: bool,
) -> Option<(usize, usize)> {
    if bw > cols || bh > rows {
        return None;
    }
    let fits = |gx: usize, gy: usize| -> bool {
        for dy in 0..bh {
            for dx in 0..bw {
                if occupied[(gy + dy) * cols + (gx + dx)] {
                    return false;
                }
            }
        }
        true
    };
    if landscape {
        for gy in 0..=rows - bh {
            for gx in 0..=cols - bw {
                if fits(gx, gy) {
                    return Some((gx, gy));
                }
            }
        }
    } else {
        for gx in 0..=cols - bw {
            for gy in 0..=rows - bh {
                if fits(gx, gy) {
                    return Some((gx, gy));
                }
            }
        }
    }
    None
}

fn mark(occupied: &mut [bool], cols: usize, gx: usize, gy: usize, bw: usize, bh: usize) {
    for dy in 0..bh {
        for dx in 0..bw {
            occupied[(gy + dy) * cols + (gx + dx)] = true;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec() -> CanvasSpec {
        CanvasSpec {
            edge: 8000,
            min_cell_edge: 512,
        }
    }

    fn assert_valid(layout: &Layout) {
        let rs: Vec<_> = layout.rects.iter().filter(|r| r.w > 0 && r.h > 0).collect();
        for r in &rs {
            assert!(
                r.x + r.w <= layout.canvas_w && r.y + r.h <= layout.canvas_h,
                "rect {r:?} exceeds canvas {}x{}",
                layout.canvas_w,
                layout.canvas_h
            );
        }
        for i in 0..rs.len() {
            for j in (i + 1)..rs.len() {
                assert!(
                    !rs[i].overlaps(rs[j]),
                    "overlap: {:?} vs {:?}",
                    rs[i],
                    rs[j]
                );
            }
        }
    }

    #[test]
    fn lays_out_uniform_landscape() {
        let l = compute(12, &[], 16.0 / 9.0, &spec());
        assert_eq!(l.template, "UNIFORM");
        assert_eq!(l.rects.len(), 12);
        assert_valid(&l);
    }

    #[test]
    fn lays_out_uniform_portrait() {
        let l = compute(12, &[], 9.0 / 16.0, &spec());
        assert_valid(&l);
        // Portrait cells are taller than wide.
        let r = l.rects.iter().find(|r| r.w > 0).expect("a cell");
        assert!(r.h > r.w, "portrait cell should be taller than wide: {r:?}");
    }

    #[test]
    fn landscape_cell_is_wider_than_tall() {
        let l = compute(12, &[], 16.0 / 9.0, &spec());
        let r = l.rects.iter().find(|r| r.w > 0).expect("a cell");
        assert!(r.w > r.h, "landscape cell should be wider than tall: {r:?}");
    }

    #[test]
    fn hero_block_is_larger_than_normal() {
        let l = compute(10, &[0], 16.0 / 9.0, &spec());
        assert_eq!(l.template, "HERO_LEFT");
        assert_valid(&l);
        let hero = l.rects[0];
        let normal = l.rects[1];
        assert!(
            hero.w > normal.w && hero.h > normal.h,
            "hero must be bigger"
        );
    }

    #[test]
    fn deduplicates_repeated_hero_index() {
        // A repeated index must place one 2×2 block, not two, and leave a valid
        // grid. HERO_LEFT (single hero) is the correct template.
        let l = compute(10, &[0, 0], 16.0 / 9.0, &spec());
        assert_eq!(l.template, "HERO_LEFT");
        assert_valid(&l);
        let placed = l.rects.iter().filter(|r| r.w > 0 && r.h > 0).count();
        assert_eq!(placed, 10, "all frames placed exactly once");
    }

    #[test]
    fn dual_hero_names_template() {
        let l = compute(10, &[0, 1], 16.0 / 9.0, &spec());
        assert_eq!(l.template, "DUAL_HERO");
        assert_valid(&l);
    }

    #[test]
    fn three_heroes_make_top_strip() {
        let l = compute(12, &[0, 1, 2], 16.0 / 9.0, &spec());
        assert_eq!(l.template, "TOP_STRIP");
        assert_valid(&l);
    }

    #[test]
    fn portrait_hero_lays_out_valid() {
        let l = compute(10, &[0], 9.0 / 16.0, &spec());
        assert_valid(&l);
        let hero = l.rects[0];
        // Hero anchors at the top-left for portrait (column-major scan).
        assert_eq!((hero.x, hero.y), (0, 0));
    }

    #[test]
    fn quality_tiers_preserve_capacity() {
        // Quality changes pixel density, not layout structure: the canvas edge
        // and min cell edge scale together, so capacity stays identical.
        let hi = grid_capacity(&CanvasSpec { edge: 8000, min_cell_edge: 512 }, 1.0);
        let lo = grid_capacity(&CanvasSpec { edge: 4000, min_cell_edge: 256 }, 1.0);
        assert_eq!(hi, lo);
    }

    #[test]
    fn single_frame_fills_one_cell() {
        let l = compute(1, &[], 1.0, &spec());
        assert_eq!(l.rects.len(), 1);
        assert_valid(&l);
    }
}
