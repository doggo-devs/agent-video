//! Grid composition: paste resized frames onto a canvas, draw timestamp labels
//! and hero highlight borders, encode as JPEG.

use ab_glyph::{FontRef, PxScale};
use anyhow::{Context, Result};
use image::imageops::FilterType;
use image::{Rgb, RgbImage};
use imageproc::drawing::{self, draw_filled_rect_mut};
use imageproc::rect::Rect as IpRect;

use crate::layout::Layout;
use crate::types::ScoredFrame;

/// Embedded font for timestamp labels — keeps the build a single binary.
const FONT: &[u8] = include_bytes!("../assets/font.ttf");

/// JPEG quality for output grids.
const JPEG_QUALITY: u8 = 92;

/// Hero highlight border color and thickness (px).
const HERO_BORDER: Rgb<u8> = Rgb([255, 210, 0]);
const HERO_BORDER_PX: u32 = 4;

/// Render the selected frames onto a single grid image, returning JPEG bytes.
///
/// `frames` and `layout.rects` are index-aligned; rectangles with zero area are
/// skipped (a frame the packer could not place). `hero_idx` marks which of
/// `frames` get a highlight border. Each cell shows a `MM:SS` timestamp.
pub fn render(frames: &[ScoredFrame], layout: &Layout, hero_idx: &[usize]) -> Result<Vec<u8>> {
    let font = FontRef::try_from_slice(FONT).context("embedded font failed to load")?;
    let mut canvas = RgbImage::from_pixel(
        layout.canvas_w.max(1),
        layout.canvas_h.max(1),
        Rgb([16, 16, 16]),
    );

    let is_hero: Vec<bool> = {
        let mut v = vec![false; frames.len()];
        for &h in hero_idx {
            if h < frames.len() {
                v[h] = true;
            }
        }
        v
    };

    for (i, frame) in frames.iter().enumerate() {
        let rect = layout.rects[i];
        if rect.w == 0 || rect.h == 0 {
            continue;
        }
        let cell = image::imageops::resize(&frame.rgb, rect.w, rect.h, FilterType::Triangle);
        image::imageops::overlay(&mut canvas, &cell, rect.x as i64, rect.y as i64);

        if is_hero[i] {
            draw_border(&mut canvas, rect.x, rect.y, rect.w, rect.h, HERO_BORDER, HERO_BORDER_PX);
        }
        draw_timestamp(&mut canvas, &font, rect.x, rect.y, frame.ts);
    }

    encode_jpeg(&canvas)
}

/// Draw a `thickness`-px border just inside the given rectangle.
///
/// Composed of four filled rectangles (top/bottom span the full width, left/
/// right fill the gap between them). `draw_filled_rect_mut` clips to the image
/// bounds, so no manual edge checks are needed.
fn draw_border(img: &mut RgbImage, x: u32, y: u32, w: u32, h: u32, color: Rgb<u8>, thickness: u32) {
    if thickness == 0 || w == 0 || h == 0 {
        return;
    }
    let t = thickness.min(h.div_ceil(2)).min(w.div_ceil(2));
    let (xi, yi) = (x as i32, y as i32);
    let mid_h = h.saturating_sub(2 * t);

    // Top and bottom edges span the full width.
    draw_filled_rect_mut(img, IpRect::at(xi, yi).of_size(w, t), color);
    draw_filled_rect_mut(img, IpRect::at(xi, (y + h - t) as i32).of_size(w, t), color);
    // Left and right edges fill the remaining height between them.
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
    let label = format_ts(ts);
    let scale = PxScale::from(20.0);

    // Translucent dark backing strip for legibility.
    let strip_w = (label.len() as u32 * 12 + 8).min(img.width().saturating_sub(x));
    let strip_h = 24.min(img.height().saturating_sub(y));
    for py in y..(y + strip_h) {
        for px in x..(x + strip_w) {
            let p = img.get_pixel(px, py);
            img.put_pixel(
                px,
                py,
                Rgb([p.0[0] / 3, p.0[1] / 3, p.0[2] / 3]),
            );
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

/// Format seconds as `MM:SS`.
pub fn format_ts(ts: f32) -> String {
    let total = ts.max(0.0).round() as u32;
    format!("{:02}:{:02}", total / 60, total % 60)
}

/// Encode an RGB image as JPEG bytes at [`JPEG_QUALITY`].
fn encode_jpeg(img: &RgbImage) -> Result<Vec<u8>> {
    let mut buf = Vec::new();
    let mut enc = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut buf, JPEG_QUALITY);
    enc.encode_image(img).context("JPEG encoding failed")?;
    Ok(buf)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::RawScores;
    use image::GrayImage;

    fn frame(ts: f32) -> ScoredFrame {
        ScoredFrame {
            ts,
            gray: GrayImage::new(1, 1),
            rgb: RgbImage::from_pixel(32, 18, Rgb([100, 120, 140])),
            raw: RawScores::default(),
            score: 0.0,
        }
    }

    #[test]
    fn formats_timestamp_as_mmss() {
        assert_eq!(format_ts(0.0), "00:00");
        assert_eq!(format_ts(83.0), "01:23");
        assert_eq!(format_ts(605.4), "10:05");
    }

    #[test]
    fn renders_nonempty_jpeg() {
        let preset = crate::preset::resolve_preset(2048, crate::types::Quality::Low, crate::types::Detail::Standard);
        let frames: Vec<_> = (0..4).map(|i| frame(i as f32)).collect();
        let layout = crate::layout::compute(4, &[], 16.0 / 9.0, &preset);
        let bytes = render(&frames, &layout, &[]).expect("render ok");
        assert!(bytes.len() > 100, "JPEG should have real content");
        // JPEG SOI marker.
        assert_eq!(&bytes[0..2], &[0xff, 0xd8]);
    }
}
