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

/// Minimum number of frames dropped as duplicates inside a kept-frame gap for
/// the gap to count as a static stretch. One drop could be a flicker; a few
/// spanning drops are evidence that the scene genuinely held still. Gaps
/// without drops (sparse keyframe sampling) are left alone.
const MIN_STATIC_DROPS: usize = 3;

/// A detected static stretch: the scene held still over `[start, end]` and
/// `rgb` (the kept frame at `start`) stands in for the whole range.
pub struct Stretch {
    pub start: f32,
    pub end: f32,
    pub rgb: Arc<RgbImage>,
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
    /// Frames dropped as duplicates since the last kept frame.
    drops_since_kept: usize,
    /// Timestamp of the most recent dropped frame (bounds a trailing stretch).
    last_dropped_ts: f32,
}

impl StretchDetector {
    pub fn new(min_span: f32) -> Self {
        Self {
            min_span,
            last_kept: None,
            drops_since_kept: 0,
            last_dropped_ts: 0.0,
        }
    }

    /// Record a kept frame: returns the closed stretch if the gap since the
    /// previous kept frame qualifies, then makes this frame the new
    /// representative candidate.
    pub fn note_kept(&mut self, ts: f32, rgb: &Arc<RgbImage>) -> Option<Stretch> {
        let stretch = match &self.last_kept {
            Some((prev_ts, prev_rgb))
                if ts - prev_ts >= self.min_span && self.drops_since_kept >= MIN_STATIC_DROPS =>
            {
                Some(Stretch {
                    start: *prev_ts,
                    end: ts,
                    rgb: prev_rgb.clone(),
                })
            }
            _ => None,
        };
        self.last_kept = Some((ts, rgb.clone()));
        self.drops_since_kept = 0;
        stretch
    }

    /// Record a frame dropped as a duplicate of the last kept frame.
    pub fn note_dropped(&mut self, ts: f32) {
        self.drops_since_kept += 1;
        self.last_dropped_ts = ts;
    }

    /// Close out a stretch that runs to the end of the stream, bounded by the
    /// last dropped duplicate rather than a following kept frame.
    pub fn finish(&mut self) -> Option<Stretch> {
        if self.drops_since_kept < MIN_STATIC_DROPS {
            return None;
        }
        let (prev_ts, rgb) = self.last_kept.take()?;
        (self.last_dropped_ts - prev_ts >= self.min_span).then_some(Stretch {
            start: prev_ts,
            end: self.last_dropped_ts,
            rgb,
        })
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

        assert_eq!((stretch.start, stretch.end), (0.0, 75.0));
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

        assert_eq!((stretch.start, stretch.end), (1.0, 70.0));
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
