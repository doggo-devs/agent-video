//! Grid composition: paste resized frames onto a canvas, draw timestamp labels
//! and hero highlight borders, encode as JPEG.

use std::sync::LazyLock;

use ab_glyph::{FontRef, PxScale};
use anyhow::{Context, Result};
use image::imageops::FilterType;
use image::{Rgb, RgbImage};
use imageproc::drawing::{self, draw_filled_rect_mut};
use imageproc::rect::Rect as IpRect;
use rayon::prelude::*;

use crate::layout::Layout;
use crate::types::ScoredFrame;

/// Embedded font for timestamp labels — keeps the build a single binary.
static FONT: LazyLock<FontRef<'static>> = LazyLock::new(|| {
    FontRef::try_from_slice(include_bytes!("../assets/font.ttf"))
        .expect("embedded font is a valid TTF")
});

const HERO_BORDER: Rgb<u8> = Rgb([255, 210, 0]);

/// Render the selected frames onto a single grid image, returning JPEG bytes.
///
/// `frames` and `layout.rects` are index-aligned; rectangles with zero area are
/// skipped (a frame the packer could not place). `hero_idx` marks which of
/// `frames` get a highlight border. Each cell shows a `MM:SS` timestamp.
pub fn render(
    frames: &[ScoredFrame],
    layout: &Layout,
    hero_idx: &[usize],
    jpeg_quality: u8,
    hero_border_px: u32,
) -> Result<Vec<u8>> {
    let mut canvas = RgbImage::from_pixel(
        layout.canvas_w.max(1),
        layout.canvas_h.max(1),
        Rgb([16, 16, 16]),
    );

    // Resizing dominates render time and each cell is independent, so it runs
    // in parallel; pasting mutates the shared canvas and stays sequential.
    // The resized cells together cover at most one canvas, so the extra peak
    // memory is bounded by the canvas itself.
    let cells: Vec<Option<RgbImage>> = frames
        .par_iter()
        .zip(layout.rects.par_iter())
        .map(|(frame, rect)| {
            (rect.w > 0 && rect.h > 0)
                .then(|| image::imageops::resize(&*frame.rgb, rect.w, rect.h, FilterType::Triangle))
        })
        .collect();

    for (i, frame) in frames.iter().enumerate() {
        let rect = layout.rects[i];
        let Some(cell) = &cells[i] else {
            continue;
        };
        image::imageops::overlay(&mut canvas, cell, rect.x as i64, rect.y as i64);

        // hero_idx holds at most 3 entries, so a linear check is cheaper than
        // allocating an n-sized lookup table.
        let is_hero = hero_idx.contains(&i);
        if is_hero {
            draw_border(
                &mut canvas,
                rect.x,
                rect.y,
                rect.w,
                rect.h,
                HERO_BORDER,
                hero_border_px,
            );
        }
        // Inset the label past the hero border: the label's translucent strip
        // darkens pixels in place and would dim the border's top-left corner.
        // Inset by the *drawn* thickness, not the raw config value — a border
        // wider than the cell is clamped by draw_border, and the label must
        // stay inside its own cell.
        let inset = if is_hero {
            border_thickness(rect.w, rect.h, hero_border_px)
        } else {
            0
        };
        draw_timestamp(&mut canvas, &FONT, rect.x + inset, rect.y + inset, frame.ts);
    }

    encode_jpeg(&canvas, jpeg_quality)
}

/// Render a single representative frame at its stored size with a timestamp
/// label. Used for static stretches, where one frame stands in for a whole
/// unchanged time range.
pub fn render_single(rgb: &RgbImage, ts: f32, jpeg_quality: u8) -> Result<Vec<u8>> {
    let mut canvas = rgb.clone();
    draw_timestamp(&mut canvas, &FONT, 0, 0, ts);
    encode_jpeg(&canvas, jpeg_quality)
}

/// Border thickness actually drawn inside a `w`×`h` cell: capped at half of
/// each edge, so the four border rects never cross the cell's midlines.
fn border_thickness(w: u32, h: u32, thickness: u32) -> u32 {
    thickness.min(h.div_ceil(2)).min(w.div_ceil(2))
}

/// Draw a `thickness`-px border just inside the given rectangle using four
/// filled rects. `draw_filled_rect_mut` clips to image bounds automatically.
fn draw_border(img: &mut RgbImage, x: u32, y: u32, w: u32, h: u32, color: Rgb<u8>, thickness: u32) {
    if thickness == 0 || w == 0 || h == 0 {
        return;
    }
    let t = border_thickness(w, h, thickness);
    let (xi, yi) = (x as i32, y as i32);
    let mid_h = h.saturating_sub(2 * t);

    draw_filled_rect_mut(img, IpRect::at(xi, yi).of_size(w, t), color);
    draw_filled_rect_mut(img, IpRect::at(xi, (y + h - t) as i32).of_size(w, t), color);
    if mid_h > 0 {
        draw_filled_rect_mut(img, IpRect::at(xi, (y + t) as i32).of_size(t, mid_h), color);
        draw_filled_rect_mut(
            img,
            IpRect::at((x + w - t) as i32, (y + t) as i32).of_size(t, mid_h),
            color,
        );
    }
}

/// Draw an `MM:SS` timestamp at the cell's top-left over a translucent strip.
fn draw_timestamp(img: &mut RgbImage, font: &FontRef, x: u32, y: u32, ts: f32) {
    let label = crate::types::format_ts(ts);
    let scale = PxScale::from(20.0);

    // get_pixel_mut darkens in place — one bounds check per pixel instead of two.
    let strip_w = (label.len() as u32 * 12 + 8).min(img.width().saturating_sub(x));
    let strip_h = 24.min(img.height().saturating_sub(y));
    for py in y..(y + strip_h) {
        for px in x..(x + strip_w) {
            let p = img.get_pixel_mut(px, py);
            p.0[0] /= 3;
            p.0[1] /= 3;
            p.0[2] /= 3;
        }
    }
    drawing::draw_text_mut(
        img,
        Rgb([255, 255, 255]),
        x as i32 + 4,
        y as i32 + 2,
        scale,
        font,
        &label,
    );
}

fn encode_jpeg(img: &RgbImage, quality: u8) -> Result<Vec<u8>> {
    let mut buf = Vec::new();
    let mut enc = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut buf, quality);
    enc.encode_image(img).context("JPEG encoding failed")?;
    Ok(buf)
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::GrayImage;
    use std::sync::Arc;

    fn frame(ts: f32) -> ScoredFrame {
        ScoredFrame {
            ts,
            gray: GrayImage::new(1, 1),
            rgb: Arc::new(RgbImage::from_pixel(32, 18, Rgb([100, 120, 140]))),
            score: 0.0,
            scene_diff: 0.0,
        }
    }

    #[test]
    fn clamps_border_thickness_to_half_cell() {
        assert_eq!(border_thickness(100, 60, 4), 4, "thin border unclamped");
        assert_eq!(border_thickness(100, 60, 256), 30, "capped at half height");
        assert_eq!(border_thickness(9, 100, 256), 5, "div_ceil on odd edges");
    }

    #[test]
    fn renders_nonempty_jpeg() {
        let spec = crate::types::CanvasSpec::new(1024, 256).expect("valid test spec");
        let frames: Vec<_> = (0..4).map(|i| frame(i as f32)).collect();
        let layout = crate::layout::compute(&[0.0; 4], &[], 16.0 / 9.0, &spec);
        let bytes = render(&frames, &layout, &[], 92, 4).expect("render ok");
        assert!(bytes.len() > 100, "JPEG should have real content");
        assert_eq!(&bytes[0..2], &[0xff, 0xd8]);
    }
}
