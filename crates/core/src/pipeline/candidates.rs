// SPDX-License-Identifier: Apache-2.0
//! Bounded batches of sampled frames and their shared perceptual fingerprints.
//! Fingerprints are computed in parallel; temporal consumers retain input order.

use image_hasher::{HasherConfig, ImageHash};
use rayon::prelude::*;

use crate::error::Result;
use crate::sample::{self, RgbFrame};
use crate::types::{Aspect, CandidateFrame};

/// A sampled frame with a fingerprint shared by dedup and static detection.
pub struct HashedFrame {
    /// Image and timestamp of this observation, including repeated content.
    pub frame: CandidateFrame,
    /// Perceptual fingerprint of the scoring grayscale.
    pub hash: ImageHash,
    /// Mean grayscale intensity; gradient hashes alone ignore uniform flashes.
    pub mean_luma: f32,
}

/// Hash a bounded batch without reordering or filtering its observations.
pub(super) fn fingerprint(
    frames: Vec<CandidateFrame>,
    hasher: &image_hasher::Hasher,
) -> Vec<HashedFrame> {
    frames
        .into_par_iter()
        .map(|frame| {
            let hash = hasher.hash_image(frame.gray.as_ref());
            let gray = frame.gray.as_raw();
            let mean_luma =
                gray.iter().map(|&v| u64::from(v)).sum::<u64>() as f32 / gray.len() as f32;
            HashedFrame {
                frame,
                hash,
                mean_luma,
            }
        })
        .collect()
}

/// A pull source of decoded RGB frames.
///
/// [`sample::FrameStream`] is the production implementation; tests feed
/// synthetic frames through a hand-written fake.
pub trait RgbSource {
    /// The next decoded frame, or `None` after a clean end of stream.
    fn next_frame(&mut self) -> Result<Option<RgbFrame>>;
    /// Source frame aspect ratio, known once at least one frame was read.
    fn aspect(&self) -> Option<Aspect>;
}

impl RgbSource for sample::FrameStream<'_> {
    fn next_frame(&mut self) -> Result<Option<RgbFrame>> {
        sample::FrameStream::next_frame(self)
    }
    fn aspect(&self) -> Option<Aspect> {
        // Minted here, at the pipeline's edge of the decoder: a degenerate
        // ratio reads as "not known yet", so layout keeps its square default
        // instead of ever seeing a 0/NaN aspect.
        sample::FrameStream::aspect(self).and_then(Aspect::new)
    }
}

/// Streaming candidate producer over any [`RgbSource`].
///
/// Batches bound image memory to `batch_frames`; their boundaries carry no
/// temporal meaning for consumers.
pub struct CandidateStream<'a, S> {
    source: S,
    hasher: image_hasher::Hasher,
    batch_frames: usize,
    score_edge: u32,
    /// Pool for the per-frame parallel work (grayscale, fingerprint) so `--threads`
    /// caps it too; `None` uses the global rayon pool (`extract` exposes no
    /// thread flag).
    pool: Option<&'a rayon::ThreadPool>,
}

impl<'a, S: RgbSource> CandidateStream<'a, S> {
    /// Wire a source to bounded parallel grayscale and fingerprint computation.
    pub fn new(
        source: S,
        batch_frames: usize,
        score_edge: u32,
        pool: Option<&'a rayon::ThreadPool>,
    ) -> Self {
        Self {
            source,
            hasher: HasherConfig::new().to_hasher(),
            batch_frames: batch_frames.max(1),
            score_edge,
            pool,
        }
    }

    /// Source frame aspect ratio, known once at least one frame was read.
    pub fn aspect(&self) -> Option<Aspect> {
        self.source.aspect()
    }

    /// Pull the next batch of fingerprinted samples, in timestamp order; `None`
    /// after a clean end of stream. The final batch may be short.
    pub fn next_batch(&mut self) -> Result<Option<Vec<HashedFrame>>> {
        let mut rgbs = Vec::with_capacity(self.batch_frames);
        while rgbs.len() < self.batch_frames {
            match self.source.next_frame()? {
                Some(frame) => rgbs.push(frame),
                None => break,
            }
        }
        if rgbs.is_empty() {
            return Ok(None);
        }

        let score_edge = self.score_edge;
        let hasher = &self.hasher;
        let analyze =
            move || fingerprint(sample::candidate_frames_from_rgbs(rgbs, score_edge), hasher);
        Ok(Some(match self.pool {
            Some(pool) => pool.install(analyze),
            None => analyze(),
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dedup::Deduper;
    use image::{Rgb, RgbImage};
    use std::num::NonZeroUsize;

    /// Hand-written fake source yielding pre-built frames.
    struct FakeSource {
        frames: std::vec::IntoIter<RgbFrame>,
    }

    impl RgbSource for FakeSource {
        fn next_frame(&mut self) -> Result<Option<RgbFrame>> {
            Ok(self.frames.next())
        }
        fn aspect(&self) -> Option<Aspect> {
            Some(Aspect::SQUARE)
        }
    }

    fn frame(ts: f32, seed: u32) -> RgbFrame {
        let rgb = RgbImage::from_fn(64, 64, |x, y| {
            let v = ((x * (3 + seed) + y * (7 + 2 * seed)) % 256) as u8;
            Rgb([v, v, v])
        });
        RgbFrame {
            ts: crate::types::Ts::new(ts).expect("finite test ts"),
            rgb,
        }
    }

    /// Ten frames where every third is a near-duplicate of its predecessor,
    /// so dedup verdicts are a mix of kept and dropped.
    fn mixed_frames() -> Vec<RgbFrame> {
        (0..10)
            .map(|i| frame(i as f32, if i % 3 == 2 { (i - 1) as u32 } else { i as u32 }))
            .collect()
    }

    fn verdicts(batch_frames: usize) -> Vec<(f32, bool)> {
        let source = FakeSource {
            frames: mixed_frames().into_iter(),
        };
        let mut deduper = Deduper::new(8, NonZeroUsize::new(4).expect("nonzero window"));
        let mut stream = CandidateStream::new(source, batch_frames, 64, None);
        let mut out = Vec::new();
        while let Some(batch) = stream.next_batch().expect("fake source never fails") {
            out.extend(
                batch
                    .into_iter()
                    .map(|sample| (sample.frame.ts.get(), deduper.accepts(&sample.hash))),
            );
        }
        out
    }

    /// The anti-drift contract: `process` and `extract` size their batches
    /// differently (base_fps vs extract_fps), so identical inputs must yield
    /// identical verdicts regardless of batch size — otherwise the two
    /// subcommands would select different frames from the same video.
    #[test]
    fn judges_identically_across_batch_sizes() {
        let baseline = verdicts(100);
        assert_eq!(baseline.len(), 10, "every frame gets a verdict");
        assert!(
            baseline.iter().any(|&(_, kept)| !kept),
            "fixture must exercise dropped frames"
        );
        for batch_frames in [1, 3, 4, 7] {
            assert_eq!(
                verdicts(batch_frames),
                baseline,
                "batch size {batch_frames} changed dedup verdicts"
            );
        }
    }

    #[test]
    fn preserves_static_evidence_across_batch_boundaries() {
        use super::super::stretch::StretchDetector;
        for batch_size in [1, 3, 60, 240] {
            let frames: Vec<_> = (0..80)
                .map(|t| frame(t as f32, u32::from(t != 0 && t != 2)))
                .collect();
            let source = FakeSource {
                frames: frames.into_iter(),
            };
            let mut stream = CandidateStream::new(source, batch_size, 64, None);
            let mut detector = StretchDetector::new(60.0, 8, 5.0);
            let mut stretches = Vec::new();
            while let Some(batch) = stream.next_batch().expect("batch") {
                for sample in batch {
                    stretches.extend(detector.observe(&sample));
                }
            }
            stretches.extend(detector.finish());
            let ranges: Vec<_> = stretches
                .iter()
                .map(|s| (s.range.start(), s.range.end()))
                .collect();
            assert_eq!(ranges, vec![(3.0, 79.0)], "batch size {batch_size}");
        }
    }

    #[test]
    fn ends_cleanly_after_a_short_final_batch() {
        let source = FakeSource {
            frames: mixed_frames().into_iter(),
        };
        let mut stream = CandidateStream::new(source, 4, 64, None);

        let sizes: Vec<usize> = std::iter::from_fn(|| {
            stream
                .next_batch()
                .expect("fake source never fails")
                .map(|b| b.len())
        })
        .collect();
        assert_eq!(sizes, vec![4, 4, 2], "full batches then the remainder");
        assert!(
            stream.next_batch().expect("still clean").is_none(),
            "stream stays exhausted"
        );
    }
}
