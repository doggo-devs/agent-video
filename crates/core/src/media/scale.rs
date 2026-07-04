//! Frame rescaling via libswscale: decoded native-format frames to small RGB
//! images sized for grid composition.
//!
//! The conversion runs on the decode thread, immediately after each frame is
//! received, so the full-resolution frame is dropped before the next one is
//! decoded — peak memory stays at one native frame regardless of video size.

use ffmpeg::format::Pixel;
use ffmpeg::software::scaling;
use ffmpeg_next as ffmpeg;
use image::RgbImage;

use crate::error::{Error, ErrorCode, Result};

/// Cached swscale conversion to RGB24 with the shorter edge pinned to
/// `min_edge` (aspect preserved, upscaling allowed) — the same geometry the
/// old `scale=E:E:force_original_aspect_ratio=increase` filter produced.
/// The context is rebuilt only when the source geometry or pixel format
/// changes (mid-stream resolution switches are rare but legal).
pub struct Rescaler {
    min_edge: u32,
    ctx: Option<scaling::Context>,
    src_key: Option<(Pixel, u32, u32)>,
    dst: (u32, u32),
}

impl Rescaler {
    pub fn new(min_edge: u32) -> Self {
        Self {
            min_edge: min_edge.max(1),
            ctx: None,
            src_key: None,
            dst: (0, 0),
        }
    }

    /// Convert one decoded frame to a small RGB image.
    pub fn rgb(&mut self, frame: &ffmpeg::frame::Video) -> Result<RgbImage> {
        let key = (frame.format(), frame.width(), frame.height());
        if self.src_key != Some(key) {
            let (dw, dh) = fit_min_edge(frame.width(), frame.height(), self.min_edge);
            // BICUBIC matches the ffmpeg scale filter's default algorithm.
            let ctx = scaling::Context::get(
                key.0,
                key.1,
                key.2,
                Pixel::RGB24,
                dw,
                dh,
                scaling::Flags::BICUBIC,
            )
            .map_err(|e| {
                Error::new(
                    ErrorCode::DecodeFailed,
                    format!(
                        "cannot build scaler for {}x{} {:?}: {e}",
                        key.1, key.2, key.0
                    ),
                )
            })?;
            self.ctx = Some(ctx);
            self.src_key = Some(key);
            self.dst = (dw, dh);
        }

        let mut out = ffmpeg::frame::Video::empty();
        self.ctx
            .as_mut()
            .expect("scaler built above")
            .run(frame, &mut out)
            .map_err(|e| Error::new(ErrorCode::DecodeFailed, format!("scaling frame: {e}")))?;

        rgb_image_from_frame(&out, self.dst)
    }
}

/// Copy an RGB24 frame into an [`RgbImage`], honoring the row stride.
///
/// swscale pads each row to an alignment boundary; `data(0)` is
/// `stride × height` bytes with `width × 3` payload per row. A whole-buffer
/// `from_raw` would shear the image whenever `stride != width * 3`, so rows
/// are copied individually.
fn rgb_image_from_frame(frame: &ffmpeg::frame::Video, (w, h): (u32, u32)) -> Result<RgbImage> {
    let stride = frame.stride(0);
    let data = frame.data(0);
    let row = w as usize * 3;
    let mut buf = Vec::with_capacity(row * h as usize);
    for y in 0..h as usize {
        let start = y * stride;
        buf.extend_from_slice(&data[start..start + row]);
    }
    RgbImage::from_raw(w, h, buf).ok_or_else(|| {
        Error::new(
            ErrorCode::DecodeFailed,
            format!("scaled frame buffer does not match {w}x{h}"),
        )
    })
}

/// Aspect-preserving dimensions with the shorter edge exactly `edge`.
/// Upscales small sources, matching `force_original_aspect_ratio=increase`.
fn fit_min_edge(w: u32, h: u32, edge: u32) -> (u32, u32) {
    if w <= h {
        let scaled = (h as f64 * edge as f64 / w.max(1) as f64).round() as u32;
        (edge, scaled.max(edge))
    } else {
        let scaled = (w as f64 * edge as f64 / h.max(1) as f64).round() as u32;
        (scaled.max(edge), edge)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pins_shorter_edge_preserving_aspect() {
        assert_eq!(fit_min_edge(1920, 1080, 256), (455, 256));
        assert_eq!(fit_min_edge(1080, 1920, 256), (256, 455));
        assert_eq!(fit_min_edge(640, 640, 256), (256, 256));
    }

    #[test]
    fn upscales_small_sources_like_increase_mode() {
        // force_original_aspect_ratio=increase scales the shorter edge UP to
        // the target as well; small sources must not stay small.
        assert_eq!(fit_min_edge(100, 50, 256), (512, 256));
    }
}
