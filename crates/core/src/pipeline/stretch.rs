// SPDX-License-Identifier: Apache-2.0
//! Static-stretch detection over the dedup verdict stream.
//!
//! Dedup collapses an unchanging scene into a long gap between kept frames.
//! [`StretchDetector`] watches that stream: when a gap spans at least
//! `min_span` seconds and enough sampled frames were dropped inside it as
//! duplicates (the evidence that the stretch was watched, not just unsampled),
//! it reports the gap as a [`Stretch`] whose representative is the last kept
//! frame before the scene froze.

use std::sync::Arc;

use image::RgbImage;

use crate::types::TimeRange;

/// Minimum number of frames dropped as duplicates inside a kept-frame gap for
/// the gap to count as a static stretch. One drop could be a flicker; a few
/// spanning drops are evidence that the scene genuinely held still. Gaps
/// without drops (sparse keyframe sampling) are left alone.
const MIN_STATIC_DROPS: usize = 3;

/// A detected static stretch: the scene held still over `range` and `rgb`
/// (the kept frame at the range's start) stands in for the whole span.
/// Carrying a [`TimeRange`] makes `start <= end` a construction fact, not a
/// consumer-side re-proof.
#[derive(Debug)]
pub struct Stretch {
    /// The still span on the timeline.
    pub range: TimeRange,
    /// The representative frame, shared with the pipeline via [`Arc`].
    pub rgb: Arc<RgbImage>,
}

/// The run of frames dropped as duplicates since the last kept frame: how
/// many, and when the last one was seen.
struct DropRun {
    count: usize,
    last_ts: f32,
}

/// Streaming detector for static stretches.
///
/// Feed every dedup verdict in timestamp order via [`note_kept`] /
/// [`note_dropped`], then call [`finish`] to close a stretch that runs to the
/// end of the stream.
///
/// [`note_kept`]: StretchDetector::note_kept
/// [`note_dropped`]: StretchDetector::note_dropped
/// [`finish`]: StretchDetector::finish
pub struct StretchDetector {
    /// Minimum kept-frame gap, in seconds, for a stretch to qualify.
    min_span: f32,
    /// Timestamp and shared RGB of the most recent kept frame — the candidate
    /// representative should a static gap follow it. The `Arc` makes retaining
    /// it a pointer clone rather than a deep copy of the ~1.4 MB image.
    last_kept: Option<(f32, Arc<RgbImage>)>,
    drops: Option<DropRun>,
}

impl StretchDetector {
    /// A detector requiring kept-frame gaps of at least `min_span` seconds.
    pub fn new(min_span: f32) -> Self {
        Self {
            min_span,
            last_kept: None,
            drops: None,
        }
    }

    fn drop_count(&self) -> usize {
        self.drops.as_ref().map_or(0, |d| d.count)
    }

    /// Record a kept frame: returns the closed stretch if the gap since the
    /// previous kept frame qualifies, then makes this frame the new
    /// representative candidate.
    pub fn note_kept(&mut self, ts: f32, rgb: &Arc<RgbImage>) -> Option<Stretch> {
        let stretch = self.last_kept.as_ref().and_then(|(prev_ts, prev_rgb)| {
            // The range doubles as the gap proof: timestamps are fed in
            // order, so `new` only fails on a backwards pair — which is no
            // stretch — and the span check subsumes the ordering.
            let range = TimeRange::new(*prev_ts, ts)?;
            (range.duration() >= self.min_span && self.drop_count() >= MIN_STATIC_DROPS).then(
                || Stretch {
                    range,
                    rgb: prev_rgb.clone(),
                },
            )
        });
        self.last_kept = Some((ts, rgb.clone()));
        self.drops = None;
        stretch
    }

    /// Record a frame dropped as a duplicate of the last kept frame.
    pub fn note_dropped(&mut self, ts: f32) {
        let count = self.drop_count() + 1;
        self.drops = Some(DropRun { count, last_ts: ts });
    }

    /// Close out a stretch that runs to the end of the stream, bounded by the
    /// last dropped duplicate rather than a following kept frame.
    pub fn finish(&mut self) -> Option<Stretch> {
        let drops = self.drops.take()?;
        if drops.count < MIN_STATIC_DROPS {
            return None;
        }
        let (prev_ts, rgb) = self.last_kept.take()?;
        let range = TimeRange::new(prev_ts, drops.last_ts)?;
        (range.duration() >= self.min_span).then_some(Stretch { range, rgb })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rgb() -> Arc<RgbImage> {
        Arc::new(RgbImage::new(1, 1))
    }

    /// Feed `n` drops at one-second intervals starting after `from`.
    fn drop_range(det: &mut StretchDetector, from: f32, n: usize) {
        for i in 1..=n {
            det.note_dropped(from + i as f32);
        }
    }

    #[test]
    fn closes_stretch_on_qualifying_gap() {
        let mut det = StretchDetector::new(60.0);

        assert!(det.note_kept(0.0, &rgb()).is_none());
        drop_range(&mut det, 0.0, 5);
        let stretch = det.note_kept(75.0, &rgb()).expect("gap qualifies");

        assert_eq!((stretch.range.start(), stretch.range.end()), (0.0, 75.0));
    }

    #[test]
    fn ignores_gap_with_too_few_drops() {
        let mut det = StretchDetector::new(60.0);

        det.note_kept(0.0, &rgb());
        drop_range(&mut det, 0.0, MIN_STATIC_DROPS - 1);

        assert!(
            det.note_kept(75.0, &rgb()).is_none(),
            "sparse sampling gap is not static"
        );
    }

    #[test]
    fn ignores_gap_shorter_than_min_span() {
        let mut det = StretchDetector::new(60.0);

        det.note_kept(0.0, &rgb());
        drop_range(&mut det, 0.0, 5);

        assert!(det.note_kept(30.0, &rgb()).is_none(), "gap below min span");
    }

    #[test]
    fn closes_trailing_stretch_at_last_drop() {
        let mut det = StretchDetector::new(60.0);

        det.note_kept(1.0, &rgb());
        drop_range(&mut det, 1.0, 69);
        let stretch = det.finish().expect("trailing stretch qualifies");

        assert_eq!((stretch.range.start(), stretch.range.end()), (1.0, 70.0));
    }

    #[test]
    fn finish_without_drops_yields_nothing() {
        let mut det = StretchDetector::new(60.0);

        det.note_kept(0.0, &rgb());

        assert!(det.finish().is_none());
    }

    #[test]
    fn drops_reset_after_each_kept_frame() {
        let mut det = StretchDetector::new(60.0);

        det.note_kept(0.0, &rgb());
        drop_range(&mut det, 0.0, 5);
        det.note_kept(30.0, &rgb());
        drop_range(&mut det, 30.0, 2);

        assert!(
            det.note_kept(95.0, &rgb()).is_none(),
            "drops before an intervening kept frame do not carry over"
        );
    }
}
