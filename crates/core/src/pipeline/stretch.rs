// SPDX-License-Identifier: Apache-2.0
//! Continuous static evidence over every sampled frame, independent of dedup.
//! One fixed anchor bounds both visual drift and retained image memory.

use std::sync::Arc;

use image::RgbImage;
use image_hasher::ImageHash;

use crate::types::{TimeRange, Ts};

use super::candidates::HashedFrame;

const MIN_STATIC_SAMPLES: usize = 4;
// Gradient hashes discard absolute brightness. Allow small decode variations,
// but do not join a uniform flash to the preceding scene.
const MAX_LUMA_DELTA: f32 = 8.0;

/// A continuous sampled run and its first frame, ready for rendering.
#[derive(Debug)]
pub struct Stretch {
    /// From the first observation to the last matching observation.
    pub range: TimeRange,
    /// The fixed representative of this run, including revisited scenes.
    pub rgb: Arc<RgbImage>,
}

struct Run {
    hash: ImageHash,
    mean_luma: f32,
    start: Ts,
    last: Ts,
    samples: usize,
    rgb: Arc<RgbImage>,
}

impl Run {
    fn new(sample: &HashedFrame) -> Self {
        Self {
            hash: sample.hash.clone(),
            mean_luma: sample.mean_luma,
            start: sample.frame.ts,
            last: sample.frame.ts,
            samples: 1,
            rgb: sample.frame.rgb.clone(),
        }
    }

    fn finish(self, min_span: f32) -> Option<Stretch> {
        let range = TimeRange::new(self.start.get().max(0.0), self.last.get())?;
        (self.samples >= MIN_STATIC_SAMPLES && range.duration() >= min_span).then_some(Stretch {
            range,
            rgb: self.rgb,
        })
    }
}

/// A bounded detector: one candidate, one representative, no completed-run list.
/// Short runs are discarded. A change or sampling hole seals the old candidate
/// at its last matching sample, then starts afresh with the current observation.
pub struct StretchDetector {
    min_span: f32,
    hash_threshold: u32,
    max_sample_gap: f32,
    run: Option<Run>,
}

impl StretchDetector {
    /// Require both `min_span` seconds and four distinct sample timestamps.
    /// `max_sample_gap` bounds the unobserved interval admitted as continuous
    /// evidence, including sparse keyframe sampling; it is not learned from gaps.
    pub fn new(min_span: f32, hash_threshold: u32, max_sample_gap: f32) -> Self {
        Self {
            min_span,
            hash_threshold,
            max_sample_gap,
            run: None,
        }
    }

    /// Observe every sample in timestamp order, including dedup rejections.
    pub fn observe(&mut self, sample: &HashedFrame) -> Option<Stretch> {
        if let Some(run) = &mut self.run {
            let gap = sample.frame.ts.get() - run.last.get();
            if gap <= self.max_sample_gap
                && run.hash.dist(&sample.hash) <= self.hash_threshold
                && (run.mean_luma - sample.mean_luma).abs() <= MAX_LUMA_DELTA
            {
                if sample.frame.ts > run.last {
                    run.samples = (run.samples + 1).min(MIN_STATIC_SAMPLES);
                    run.last = sample.frame.ts;
                }
                return None;
            }
        }
        let previous = self.run.replace(Run::new(sample));
        previous.and_then(|run| run.finish(self.min_span))
    }

    /// Seal the trailing candidate once, ending at its last observed sample.
    pub fn finish(&mut self) -> Option<Stretch> {
        self.run.take().and_then(|run| run.finish(self.min_span))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::CandidateFrame;
    use image::GrayImage;

    fn sample(ts: f32, hash: u8, mean_luma: f32) -> HashedFrame {
        HashedFrame {
            frame: CandidateFrame {
                ts: Ts::new(ts).expect("finite timestamp"),
                gray: Arc::new(GrayImage::new(1, 1)),
                rgb: Arc::new(RgbImage::new(1, 1)),
            },
            hash: ImageHash::from_bytes(&[hash; 8]).expect("64-bit hash"),
            mean_luma,
        }
    }

    #[test]
    fn requires_duration_and_distinct_observations() {
        let mut detector = StretchDetector::new(3.0, 0, 5.0);
        for ts in [0.0, 0.0, 1.0, 3.0] {
            assert!(detector.observe(&sample(ts, 0, 128.0)).is_none());
        }
        assert!(
            detector.finish().is_none(),
            "three timestamps do not suffice"
        );
        for ts in [0.0, 0.5, 1.0, 2.0] {
            detector.observe(&sample(ts, 0, 128.0));
        }
        assert!(
            detector.finish().is_none(),
            "samples cannot replace duration"
        );
        for ts in [0.0, 1.0, 2.0, 3.0] {
            detector.observe(&sample(ts, 0, 128.0));
        }
        assert_eq!(
            detector
                .finish()
                .expect("both thresholds met")
                .range
                .duration(),
            3.0
        );
        assert!(detector.finish().is_none(), "finish does not repeat output");
    }

    #[test]
    fn compares_to_the_anchor_instead_of_chaining_similarity() {
        let mut detector = StretchDetector::new(3.0, 8, 5.0);
        // A~B and B~C, but A!~C. C must begin a separate run.
        for (ts, hash) in [(0.0, 0), (1.0, 1), (2.0, 3), (3.0, 3), (4.0, 3), (5.0, 3)] {
            assert!(detector.observe(&sample(ts, hash, 128.0)).is_none());
        }
        let stretch = detector.finish().expect("C settles");
        assert_eq!((stretch.range.start(), stretch.range.end()), (2.0, 5.0));
    }

    #[test]
    fn closes_before_a_sampling_hole_and_restarts_after_it() {
        let mut detector = StretchDetector::new(3.0, 0, 5.0);
        for ts in 0..=3 {
            detector.observe(&sample(ts as f32, 0, 128.0));
        }
        let stretch = detector
            .observe(&sample(80.0, 0, 128.0))
            .expect("earlier run qualified");
        assert_eq!((stretch.range.start(), stretch.range.end()), (0.0, 3.0));
        assert!(
            detector.finish().is_none(),
            "one sample after a hole is not a run"
        );

        let mut detector = StretchDetector::new(60.0, 0, 5.0);
        for ts in [0.0, 1.0, 2.0, 3.0, 80.0] {
            assert!(detector.observe(&sample(ts, 0, 128.0)).is_none());
        }
        assert!(
            detector.finish().is_none(),
            "the hole cannot satisfy duration"
        );
    }

    #[test]
    fn rejects_brightness_flicker_and_recovers_without_retaining_old_images() {
        let mut detector = StretchDetector::new(3.0, 8, 5.0);
        for ts in 0..10_000 {
            let observation = sample(ts as f32, 0, if ts % 2 == 0 { 0.0 } else { 255.0 });
            let weak = Arc::downgrade(&observation.frame.rgb);
            assert!(detector.observe(&observation).is_none());
            drop(observation);
            if ts % 2 == 1 {
                let next = sample(ts as f32 + 0.5, 0, 128.0);
                assert!(detector.observe(&next).is_none());
                assert!(weak.upgrade().is_none(), "replaced anchor was released");
            }
        }
        let first = sample(10_000.0, 0, 128.0);
        detector.observe(&first);
        for ts in 10_001..=10_004 {
            detector.observe(&sample(ts as f32, 0, 128.0));
        }
        let stretch = detector
            .finish()
            .expect("flicker does not poison future evidence");
        assert_eq!(stretch.range.end(), 10_004.0);
        assert!(stretch.range.start() >= 9_999.5);
    }
}
