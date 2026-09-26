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

use image_hasher::ImageHash;

/// Streaming perceptual-hash deduplicator.
///
/// Retains only the hashes of recently-kept frames so duplicates are dropped
/// before they reach the scoring buffer. A short window catches repeats across
/// brief dips without letting slow drift accumulate into a missed duplicate.
pub struct Deduper {
    recent: VecDeque<ImageHash>,
    hash_threshold: u32,
    window_size: usize,
}

impl Deduper {
    /// `window` is the number of recently-kept hashes to compare against;
    /// zero would grow `recent` without bound.
    pub fn new(hash_threshold: u32, window: NonZeroUsize) -> Self {
        Self {
            recent: VecDeque::with_capacity(window.get()),
            hash_threshold,
            window_size: window.get(),
        }
    }

    /// Admit a fingerprint unless it matches one of the recently admitted frames.
    /// The caller computes fingerprints once and may share them with other analyses.
    pub fn accepts(&mut self, hash: &ImageHash) -> bool {
        let is_dup = self
            .recent
            .iter()
            .any(|recent| recent.dist(hash) <= self.hash_threshold);
        if is_dup {
            return false;
        }

        if self.recent.len() == self.window_size {
            self.recent.pop_front();
        }
        self.recent.push_back(hash.clone());
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hash(byte: u8) -> ImageHash {
        ImageHash::from_bytes(&[byte; 8]).expect("64-bit fingerprint")
    }

    #[test]
    fn remembers_nonconsecutive_duplicates_until_evicted() {
        let mut dedup = Deduper::new(0, NonZeroUsize::new(2).expect("nonzero window"));
        assert!(dedup.accepts(&hash(0)));
        assert!(dedup.accepts(&hash(1)));
        assert!(!dedup.accepts(&hash(0)), "a revisit is still a duplicate");
        assert!(dedup.accepts(&hash(2)));
        assert!(
            dedup.accepts(&hash(0)),
            "the oldest admitted hash was evicted"
        );
    }
}
