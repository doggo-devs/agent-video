// SPDX-License-Identifier: Apache-2.0
//! The shared sampling front half of both pipelines: decoded frames in,
//! judged candidates out.
//!
//! [`CandidateStream`] drains a frame source in bounded batches, derives
//! per-frame scoring grayscales in parallel, and judges every frame against
//! the perceptual deduper. `process` and `extract` both consume it, so the
//! two subcommands select surviving frames with the same machinery and
//! cannot drift apart — the property the tests below pin down.

use anyhow::Result;

use crate::dedup::Deduper;
use crate::extract::{self, RgbFrame};
use crate::types::CandidateFrame;

/// One judged frame: the candidate and whether dedup kept it.
pub type Judged = (CandidateFrame, bool);

/// A pull source of decoded RGB frames.
///
/// [`extract::FrameStream`] is the production implementation; tests feed
/// synthetic frames through a hand-written fake.
pub trait RgbSource {
    /// The next decoded frame, or `None` after a clean end of stream.
    fn next_frame(&mut self) -> Result<Option<RgbFrame>>;
    /// Source frame aspect ratio, known once at least one frame was read.
    fn aspect(&self) -> Option<f32>;
}

impl RgbSource for extract::FrameStream {
    fn next_frame(&mut self) -> Result<Option<RgbFrame>> {
        // `?` lifts the typed `error::Error` into `anyhow::Error`, keeping it
        // downcastable so the CLI can still recover the exit code.
        Ok(extract::FrameStream::next_frame(self)?)
    }
    fn aspect(&self) -> Option<f32> {
        extract::FrameStream::aspect(self)
    }
}

/// Streaming candidate producer over any [`RgbSource`].
///
/// Batches bound memory to `batch_frames` decoded frames; the deduper's
/// sliding window carries across batches, so verdicts are independent of the
/// batch size (see `judges_identically_across_batch_sizes`).
pub struct CandidateStream<'a, S> {
    source: S,
    deduper: Deduper,
    batch_frames: usize,
    score_edge: u32,
    /// Pool for the per-frame parallel work (grayscale, pHash) so `--threads`
    /// caps it too; `None` uses the global rayon pool (`extract` exposes no
    /// thread flag).
    pool: Option<&'a rayon::ThreadPool>,
}

impl<'a, S: RgbSource> CandidateStream<'a, S> {
    pub fn new(
        source: S,
        deduper: Deduper,
        batch_frames: usize,
        score_edge: u32,
        pool: Option<&'a rayon::ThreadPool>,
    ) -> Self {
        Self {
            source,
            deduper,
            batch_frames: batch_frames.max(1),
            score_edge,
            pool,
        }
    }

    /// Source frame aspect ratio, known once at least one frame was read.
    pub fn aspect(&self) -> Option<f32> {
        self.source.aspect()
    }

    /// Pull the next batch of judged candidates, in timestamp order; `None`
    /// after a clean end of stream. The final batch may be short.
    pub fn next_batch(&mut self) -> Result<Option<Vec<Judged>>> {
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
        let deduper = &mut self.deduper;
        let judge = move || deduper.judge(extract::candidate_frames_from_rgbs(rgbs, score_edge));
        Ok(Some(match self.pool {
            Some(pool) => pool.install(judge),
            None => judge(),
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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
        fn aspect(&self) -> Option<f32> {
            Some(1.0)
        }
    }

    fn frame(ts: f32, seed: u32) -> RgbFrame {
        let rgb = RgbImage::from_fn(64, 64, |x, y| {
            let v = ((x * (3 + seed) + y * (7 + 2 * seed)) % 256) as u8;
            Rgb([v, v, v])
        });
        RgbFrame { ts, rgb }
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
        let deduper = Deduper::new(8, NonZeroUsize::new(4).expect("nonzero window"));
        let mut stream = CandidateStream::new(source, deduper, batch_frames, 64, None);
        let mut out = Vec::new();
        while let Some(batch) = stream.next_batch().expect("fake source never fails") {
            out.extend(batch.into_iter().map(|(f, kept)| (f.ts, kept)));
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
    fn ends_cleanly_after_a_short_final_batch() {
        let source = FakeSource {
            frames: mixed_frames().into_iter(),
        };
        let deduper = Deduper::new(8, NonZeroUsize::new(4).expect("nonzero window"));
        let mut stream = CandidateStream::new(source, deduper, 4, 64, None);

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
