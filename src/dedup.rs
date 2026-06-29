//! Perceptual-hash deduplication.
//!
//! Frames sampled at a fixed fps often repeat near-identically (static shots,
//! slow pans). This stage drops a frame when it is within `hash_threshold`
//! Hamming distance of any frame in a small sliding window of recently *kept*
//! frames. It is a cheap candidate-set shrink run right after extraction — it
//! is distinct from the temporal NMS in [`crate::score`], which spreads the
//! finally-selected frames across the timeline.

use std::collections::VecDeque;

use image_hasher::{HasherConfig, ImageHash};
use rayon::prelude::*;

use crate::types::ScoredFrame;

/// How many recently-kept hashes to compare against. A short window catches
/// repeats across brief dips without letting slow drift accumulate into a
/// missed duplicate.
const DEFAULT_WINDOW: usize = 8;

/// Streaming perceptual-hash deduplicator.
///
/// Retains only the hashes of recently-kept frames so duplicates are dropped
/// before they reach the scoring buffer.
pub struct Deduper {
    hasher: image_hasher::Hasher,
    recent: VecDeque<ImageHash>,
    hash_threshold: u32,
    window_size: usize,
}

impl Deduper {
    pub fn new(hash_threshold: u32, window: usize) -> Self {
        let window_size = if window == 0 { DEFAULT_WINDOW } else { window };
        Self {
            hasher: HasherConfig::new().to_hasher(),
            recent: VecDeque::with_capacity(window_size),
            hash_threshold,
            window_size,
        }
    }

    /// Drop duplicates from a bounded batch while preserving frame order.
    ///
    /// Hashing within the batch runs in parallel; the keep/drop decisions remain
    /// sequential so results match the streaming single-frame path.
    pub fn filter(&mut self, frames: Vec<ScoredFrame>) -> Vec<ScoredFrame> {
        let hashes: Vec<ImageHash> = frames
            .par_iter()
            .map(|frame| self.hasher.hash_image(&frame.gray))
            .collect();

        let mut kept = Vec::with_capacity(frames.len());
        for (frame, hash) in frames.into_iter().zip(hashes) {
            if self.accepts_hash(hash) {
                kept.push(frame);
            }
        }
        kept
    }

    fn accepts_hash(&mut self, hash: ImageHash) -> bool {
        let is_dup = self
            .recent
            .iter()
            .any(|recent| recent.dist(&hash) <= self.hash_threshold);
        if is_dup {
            return false;
        }

        if self.recent.len() == self.window_size {
            self.recent.pop_front();
        }
        self.recent.push_back(hash);
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::RawScores;
    use image::{GrayImage, RgbImage};

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

    fn dedup(frames: Vec<ScoredFrame>, threshold: u32, window: usize) -> Vec<ScoredFrame> {
        Deduper::new(threshold, window).filter(frames)
    }

    #[test]
    fn keeps_first_frame_always() {
        let kept = dedup(vec![flat_frame(0.0, 128)], 8, 8);
        assert_eq!(kept.len(), 1);
    }

    #[test]
    fn drops_identical_consecutive_frames() {
        let frames = vec![
            flat_frame(0.0, 100),
            flat_frame(1.0, 100),
            flat_frame(2.0, 100),
        ];
        let kept = dedup(frames, 8, 8);
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
        let kept = dedup(vec![mk(0.0, a), mk(1.0, b)], 8, 8);
        assert_eq!(kept.len(), 2, "distinct frames are both kept");
    }
}
