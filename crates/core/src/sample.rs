// SPDX-License-Identifier: Apache-2.0
//! Frame sampling: sampled, pre-scaled RGB frames from the in-process
//! decoder.
//!
//! Decoding and scaling run on a dedicated thread inside [`crate::media`];
//! this module chooses the sampling spec for each pipeline mode and derives
//! per-frame scoring grayscales in parallel over caller-sized batches, so
//! memory stays bounded by the batch, not the video.

use std::num::NonZeroUsize;
use std::sync::Arc;

use image::{GrayImage, RgbImage};
use rayon::prelude::*;

use crate::event::EventSink;
use crate::media::decode::{FrameSource, SampleSpec};
use crate::source::Source;
use crate::types::{CandidateFrame, TimeRange};

pub use crate::media::decode::RgbFrame;

/// Streaming source of sampled frames — the pipeline-facing name for
/// [`FrameSource`].
pub type FrameStream = FrameSource;

/// Shared knobs of every streaming-extractor entry point: what varies per
/// call is only the sampling shape (fps, window, fill ranges), so those
/// stay positional while these travel together.
#[derive(Clone, Copy)]
pub struct StreamParams<'a> {
    /// Shorter-edge target for the composition RGB; the decode pre-scale
    /// takes the larger of the two edges (see `pipe_edge`).
    pub cell_min_edge: u32,
    /// Shorter-edge target for the scoring grayscale.
    pub score_edge: u32,
    /// Decoder thread cap; `None` = auto-detect.
    pub threads: Option<NonZeroUsize>,
    /// Receives any decode-open diagnostics.
    pub events: &'a dyn EventSink,
}

impl StreamParams<'_> {
    fn open(&self, input: &Source, spec: SampleSpec) -> crate::error::Result<FrameStream> {
        FrameSource::open(
            input,
            spec,
            pipe_edge(self.cell_min_edge, self.score_edge),
            self.threads,
            self.events,
        )
    }
}

/// Open a streaming extractor for the whole video at a fixed fps.
pub fn stream_frames(
    input: &Source,
    fps: f32,
    params: StreamParams<'_>,
) -> crate::error::Result<FrameStream> {
    params.open(
        input,
        SampleSpec::Fps {
            fps,
            start: 0.0,
            end: None,
        },
    )
}

/// Open a streaming extractor for a time range at a fixed fps.
///
/// Timestamps produced by the stream are absolute (0-based at the stream
/// start), not range-relative.
pub fn stream_range(
    input: &Source,
    fps: f32,
    range: TimeRange,
    params: StreamParams<'_>,
) -> crate::error::Result<FrameStream> {
    params.open(
        input,
        SampleSpec::Fps {
            fps,
            start: range.start(),
            end: Some(range.end()),
        },
    )
}

/// Open a streaming extractor from `start` to `end`, or to EOF when `end` is
/// `None` — the `process --from/--until` window. Timestamps stay absolute.
/// Decode-open diagnostics include a seek fallback, if any.
pub fn stream_window(
    input: &Source,
    fps: f32,
    start: f32,
    end: Option<f32>,
    params: StreamParams<'_>,
) -> crate::error::Result<FrameStream> {
    params.open(input, SampleSpec::Fps { fps, start, end })
}

/// Open the adaptive-plan extractor: keyframes everywhere, plus fixed-fps
/// ticks inside each `fill` range, from a single decode pass.
///
/// `fill` must be ascending and non-overlapping — the invariant
/// [`crate::probe::KeyframeInfo::sparse_regions`] provides and region merging
/// preserves. One decoder emits the merged stream in presentation order, so
/// downstream monotonicity needs no interleaving choreography.
pub fn stream_mixed(
    input: &Source,
    fps: f32,
    fill: Vec<TimeRange>,
    params: StreamParams<'_>,
) -> crate::error::Result<FrameStream> {
    params.open(input, SampleSpec::Mixed { fps, fill })
}

/// Open a keyframe-only streaming extractor.
///
/// Only key packets reach the decoder, so this decodes I-frames exclusively.
/// Each frame carries its own pts.
pub fn stream_keyframes(
    input: &Source,
    params: StreamParams<'_>,
) -> crate::error::Result<FrameStream> {
    params.open(input, SampleSpec::Keyframes)
}

/// The decoded frame doubles as the composition RGB, so pre-scale to the
/// larger of the two consumers' edges; only the grayscale downscales further.
fn pipe_edge(cell_min_edge: u32, score_edge: u32) -> u32 {
    cell_min_edge.max(1).max(score_edge)
}

/// Convert a bounded RGB batch to [`CandidateFrame`]s, deriving grayscale in parallel.
pub fn candidate_frames_from_rgbs(frames: Vec<RgbFrame>, score_edge: u32) -> Vec<CandidateFrame> {
    frames
        .into_par_iter()
        .map(|frame| candidate_frame_from_rgb(frame, score_edge))
        .collect()
}

/// Convert a single RGB frame to a [`CandidateFrame`] with grayscale for scoring.
fn candidate_frame_from_rgb(frame: RgbFrame, score_edge: u32) -> CandidateFrame {
    CandidateFrame {
        ts: frame.ts,
        gray: Arc::new(to_gray(&frame.rgb, score_edge)),
        rgb: Arc::new(frame.rgb),
    }
}

/// Downscale RGB to grayscale with shorter edge `min_edge`; no-op resize if
/// already smaller.
fn to_gray(img: &RgbImage, min_edge: u32) -> GrayImage {
    let (w, h) = img.dimensions();
    let (nw, nh) = target_dims(w, h, min_edge);
    if (nw, nh) == (w, h) {
        image::imageops::grayscale(img)
    } else {
        downscale_to_gray(img, nw, nh)
    }
}

/// sRGB luma weights per mille-myriad — the same coefficients `image`'s
/// `grayscale` uses, so the fused path and the no-resize path measure the
/// same quantity.
const SRGB_LUMA: [u64; 3] = [2126, 7152, 722];
const SRGB_LUMA_DIV: u64 = 10_000;

/// Fused box-average luma downscale: one pass over the source, one (small)
/// allocation for the output.
///
/// The scoring convolutions need stable pixel statistics, not resampling
/// quality, so area averaging is sufficient — and it avoids the full-size
/// grayscale allocation + full-frame conversion a grayscale-then-resize
/// pipeline pays on every decoded frame. Requires `nw <= w` and `nh <= h`
/// with all dimensions non-zero (the strict-downscale path of [`to_gray`]).
fn downscale_to_gray(img: &RgbImage, nw: u32, nh: u32) -> GrayImage {
    let (w, h) = img.dimensions();
    debug_assert!(0 < nw && nw <= w && 0 < nh && nh <= h);
    let src = img.as_raw();
    let w = w as u64;
    let h = h as u64;

    // Floor boundaries partition the source exactly: box (ox, oy) covers
    // [ox*w/nw, (ox+1)*w/nw) × [oy*h/nh, (oy+1)*h/nh), never empty when
    // downscaling.
    let bound = |o: u32, n: u32, len: u64| (o as u64 * len / n as u64) as usize;
    GrayImage::from_fn(nw, nh, |ox, oy| {
        let (x0, x1) = (bound(ox, nw, w), bound(ox + 1, nw, w));
        let (y0, y1) = (bound(oy, nh, h), bound(oy + 1, nh, h));
        let mut sum = 0u64;
        for y in y0..y1 {
            let row = (y * w as usize + x0) * 3;
            for px in src[row..row + (x1 - x0) * 3].chunks_exact(3) {
                sum += SRGB_LUMA[0] * px[0] as u64
                    + SRGB_LUMA[1] * px[1] as u64
                    + SRGB_LUMA[2] * px[2] as u64;
            }
        }
        let count = ((x1 - x0) * (y1 - y0)) as u64;
        image::Luma([(sum / (SRGB_LUMA_DIV * count)) as u8])
    })
}

/// Aspect-preserving target dimensions for a shorter-edge of `min_edge`.
/// Returns the original dimensions unchanged if it would mean upscaling.
fn target_dims(w: u32, h: u32, min_edge: u32) -> (u32, u32) {
    let cur_min = w.min(h);
    if cur_min <= min_edge {
        return (w, h);
    }
    let scale = min_edge as f32 / cur_min as f32;
    let nw = ((w as f32 * scale).round() as u32).max(1);
    let nh = ((h as f32 * scale).round() as u32).max(1);
    (nw, nh)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn does_not_upscale_small_source() {
        assert_eq!(target_dims(100, 50, 256), (100, 50));
    }

    #[test]
    fn downscales_preserving_aspect() {
        assert_eq!(target_dims(1920, 1080, 270), (480, 270));
    }

    #[test]
    fn pipe_edge_covers_both_consumers() {
        assert_eq!(pipe_edge(256, 512), 512);
        assert_eq!(pipe_edge(512, 256), 512);
        assert_eq!(pipe_edge(0, 0), 1);
    }

    #[test]
    fn fused_downscale_matches_grayscale_on_flat_color() {
        // A constant-color source has one exact luma; the fused path must
        // agree with `image::imageops::grayscale` on it at any target size.
        let img = RgbImage::from_pixel(64, 48, image::Rgb([200, 100, 50]));
        let reference = image::imageops::grayscale(&img).get_pixel(0, 0).0[0];

        let small = downscale_to_gray(&img, 16, 12);

        assert_eq!(small.dimensions(), (16, 12));
        assert!(small.pixels().all(|p| p.0[0] == reference));
    }

    #[test]
    fn fused_downscale_averages_over_the_source_box() {
        // Left half black, right half white: each output pixel's box lies
        // entirely in one half, so the averages are exact.
        let img = RgbImage::from_fn(8, 8, |x, _| {
            image::Rgb(if x < 4 { [0, 0, 0] } else { [255, 255, 255] })
        });

        let small = downscale_to_gray(&img, 2, 2);

        assert_eq!(small.get_pixel(0, 0).0[0], 0);
        assert_eq!(small.get_pixel(1, 1).0[0], 255);
    }

    #[test]
    fn fused_downscale_handles_non_divisible_dimensions() {
        // 7×5 → 3×2: boxes have unequal sizes; boundaries must still
        // partition the source without slicing out of bounds.
        let img = RgbImage::from_fn(7, 5, |x, y| {
            let v = ((x * 37 + y * 91) % 256) as u8;
            image::Rgb([v, v, v])
        });

        let small = downscale_to_gray(&img, 3, 2);

        assert_eq!(small.dimensions(), (3, 2));
    }
}
