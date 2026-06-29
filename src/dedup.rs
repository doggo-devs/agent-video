//! Perceptual-hash deduplication.
//!
//! Frames sampled at a fixed fps often repeat near-identically (static shots,
//! slow pans). This stage drops a frame when it is within `hash_threshold`
//! Hamming distance of any frame in a small sliding window of recently *kept*
//! frames. It is a cheap candidate-set shrink run right after extraction — it
//! is distinct from the temporal NMS in [`crate::score`], which spreads the
//! finally-selected frames across the timeline.

use image_hasher::{HasherConfig, ImageHash};

use crate::types::ScoredFrame;

/// How many recently-kept hashes to compare against. A short window catches
/// repeats across brief dips without letting slow drift accumulate into a
/// missed duplicate.
const WINDOW: usize = 8;

/// Drop frames that are perceptually within `hash_threshold` of a recent frame.
///
/// Consumes the input and returns the kept frames in order. The first frame is
/// always kept. Comparison uses pHash (DCT-based) Hamming distance.
pub fn deduplicate(frames: Vec<ScoredFrame>, hash_threshold: u32) -> Vec<ScoredFrame> {
    let hasher = HasherConfig::new().to_hasher();
    let mut window: Vec<ImageHash> = Vec::with_capacity(WINDOW);
    let mut kept: Vec<ScoredFrame> = Vec::with_capacity(frames.len());

    for frame in frames {
        // The scoring grayscale is what we hash — small and already uniform.
        let dyn_img = image::DynamicImage::ImageLuma8(frame.gray.clone());
        let hash = hasher.hash_image(&dyn_img);

        let is_dup = window
            .iter()
            .any(|h| h.dist(&hash) <= hash_threshold);
        if is_dup {
            continue;
        }

        if window.len() == WINDOW {
            window.remove(0);
        }
        window.push(hash);
        kept.push(frame);
    }

    kept
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{GrayImage, RgbImage};
    use crate::types::RawScores;

    /// Build a frame whose grayscale is a flat fill of `level`.
    fn flat_frame(ts: f32, level: u8) -> ScoredFrame {
        ScoredFrame {
            ts,
            gray: GrayImage::from_pixel(64, 64, image::Luma([level])),
            rgb: RgbImage::new(1, 1),
            raw: RawScores::default(),
            score: 0.0,
        }
    }

    #[test]
    fn keeps_first_frame_always() {
        let kept = deduplicate(vec![flat_frame(0.0, 128)], 8);
        assert_eq!(kept.len(), 1);
    }

    #[test]
    fn drops_identical_consecutive_frames() {
        let frames = vec![
            flat_frame(0.0, 100),
            flat_frame(1.0, 100),
            flat_frame(2.0, 100),
        ];
        let kept = deduplicate(frames, 8);
        assert_eq!(kept.len(), 1, "identical frames collapse to one");
    }

    #[test]
    fn keeps_visually_distinct_frames() {
        // Half-black / half-white pattern vs its inverse hashes very differently.
        let mut a = GrayImage::new(64, 64);
        let mut b = GrayImage::new(64, 64);
        for y in 0..64 {
            for x in 0..64 {
                let left = x < 32;
                a.put_pixel(x, y, image::Luma([if left { 0 } else { 255 }]));
                b.put_pixel(x, y, image::Luma([if left { 255 } else { 0 }]));
            }
        }
        let mk = |ts, g| ScoredFrame {
            ts,
            gray: g,
            rgb: RgbImage::new(1, 1),
            raw: RawScores::default(),
            score: 0.0,
        };
        let kept = deduplicate(vec![mk(0.0, a), mk(1.0, b)], 8);
        assert_eq!(kept.len(), 2, "distinct frames are both kept");
    }
}
