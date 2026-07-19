// SPDX-License-Identifier: Apache-2.0
//! Perceptual-hash deduplication.
//!
//! Frames sampled at a fixed fps often repeat near-identically (static shots,
//! slow pans). This stage drops a frame when it is within `hash_threshold`
//! Hamming distance of any frame in a small sliding window of recently *kept*
//! frames. It is a cheap candidate-set shrink run right after extraction — it
//! is distinct from the temporal NMS in [`crate::score`], which spreads the
//! finally-selected frames across the timeline.

use std::collections::VecDeque;
use std::num::NonZeroUsize;

use image_hasher::{HasherConfig, ImageHash};
use rayon::prelude::*;

use crate::types::CandidateFrame;

/// Streaming perceptual-hash deduplicator.
///
/// Retains only the hashes of recently-kept frames so duplicates are dropped
/// before they reach the scoring buffer. A short window catches repeats across
/// brief dips without letting slow drift accumulate into a missed duplicate.
pub struct Deduper {
    hasher: image_hasher::Hasher,
    recent: VecDeque<ImageHash>,
    hash_threshold: u32,
    window_size: usize,
}

impl Deduper {
    /// `window` is the number of recently-kept hashes to compare against;
    /// zero would grow `recent` without bound.
    pub fn new(hash_threshold: u32, window: NonZeroUsize) -> Self {
        Self {
            hasher: HasherConfig::new().to_hasher(),
            recent: VecDeque::with_capacity(window.get()),
            hash_threshold,
            window_size: window.get(),
        }
    }

    /// Tag every frame of a bounded batch with its keep/drop verdict,
    /// preserving frame order.
    ///
    /// Hashing within the batch runs in parallel; the keep/drop decisions remain
    /// sequential so results match the streaming single-frame path. Callers that
    /// track dropped frames (static-stretch detection) get the verdict attached
    /// to the frame itself instead of reconstructing it from timestamps.
    pub fn judge(&mut self, frames: Vec<CandidateFrame>) -> Vec<(CandidateFrame, bool)> {
        let hashes: Vec<ImageHash> = frames
            .par_iter()
            .map(|frame| self.hasher.hash_image(&frame.gray))
            .collect();

        frames
            .into_iter()
            .zip(hashes)
            .map(|(frame, hash)| {
                let kept = self.accepts_hash(hash);
                (frame, kept)
            })
            .collect()
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
    use image::{GrayImage, RgbImage};
    use std::sync::Arc;

    /// Build a frame whose grayscale is a flat fill of `level`.
    fn flat_frame(ts: f32, level: u8) -> CandidateFrame {
        CandidateFrame {
            ts,
            gray: GrayImage::from_pixel(64, 64, image::Luma([level])),
            rgb: Arc::new(RgbImage::new(1, 1)),
        }
    }

    fn win(n: usize) -> NonZeroUsize {
        NonZeroUsize::new(n).expect("test window is nonzero")
    }

    fn dedup(frames: Vec<CandidateFrame>, threshold: u32, window: usize) -> Vec<CandidateFrame> {
        Deduper::new(threshold, win(window))
            .judge(frames)
            .into_iter()
            .filter_map(|(frame, kept)| kept.then_some(frame))
            .collect()
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
    fn tags_every_frame_with_its_verdict_in_order() {
        let frames = vec![
            flat_frame(0.0, 100),
            flat_frame(1.0, 100),
            flat_frame(2.0, 100),
        ];
        let judged = Deduper::new(8, win(8)).judge(frames);
        let verdicts: Vec<(f32, bool)> = judged.iter().map(|(f, kept)| (f.ts, *kept)).collect();
        assert_eq!(
            verdicts,
            vec![(0.0, true), (1.0, false), (2.0, false)],
            "duplicates come back tagged dropped, in input order"
        );
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
        let mk = |ts, g| CandidateFrame {
            ts,
            gray: g,
            rgb: Arc::new(RgbImage::new(1, 1)),
        };
        let kept = dedup(vec![mk(0.0, a), mk(1.0, b)], 8, 8);
        assert_eq!(kept.len(), 2, "distinct frames are both kept");
    }
}
