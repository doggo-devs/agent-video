// SPDX-License-Identifier: Apache-2.0
//! Video metadata probing via the in-process demuxer.
//!
//! Keyframe timestamps come from a packet-level scan (`packet.is_key()`)
//! that decodes no pixels — strictly cheaper than the old ffprobe
//! `-skip_frame nokey` pass, which had to run the video decoder. Duration
//! comes from the container header.

use crate::error::{Error, ErrorCode, ErrorContext, Result};
use crate::media;
use crate::source::Source;
use crate::types::TimeRange;

/// Keyframe pts in demux order, verified non-decreasing.
///
/// The gap/sparse-region math below assumes an ordered timeline. Key packets
/// arrive in stream order and I-frames are not reordered, so a violation
/// means a spliced or timestamp-reset stream; sorting would silently hide
/// that the timeline is untrustworthy, so construction fails instead.
#[derive(Debug, Clone, Default)]
pub struct MonotonicTimestamps(Vec<f32>);

/// The first adjacent pair that breaks non-decreasing order (or does not
/// compare, e.g. NaN), reported by [`MonotonicTimestamps::new`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NonMonotonic {
    /// Index of the offending (second) element.
    pub index: usize,
    pub prev: f32,
    pub next: f32,
}

impl MonotonicTimestamps {
    /// Verify `ts` is non-decreasing (NaN, being incomparable, never is) and
    /// wrap it.
    pub fn new(ts: Vec<f32>) -> std::result::Result<Self, NonMonotonic> {
        for (i, w) in ts.windows(2).enumerate() {
            if w[0].partial_cmp(&w[1]).is_none_or(|o| o.is_gt()) {
                return Err(NonMonotonic {
                    index: i + 1,
                    prev: w[0],
                    next: w[1],
                });
            }
        }
        Ok(Self(ts))
    }
}

impl std::ops::Deref for MonotonicTimestamps {
    type Target = [f32];
    fn deref(&self) -> &[f32] {
        &self.0
    }
}

/// Sampling strategy for frame extraction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, clap::ValueEnum)]
pub enum SampleStrategy {
    /// Fixed-rate sampling at base_fps.
    #[default]
    FixedFps,
    /// Sample only I-frames (keyframes).
    Keyframes,
    /// I-frames with fixed-fps fill in sparse regions.
    Adaptive,
}

/// Keyframe timeline and duration from a demux-level scan.
#[derive(Debug, Clone)]
pub struct KeyframeInfo {
    /// Timestamps of I-frames in seconds, in demux order (verified
    /// non-decreasing, which the gap/sparse-region math below relies on).
    pub timestamps: MonotonicTimestamps,
    /// Total video duration in seconds.
    pub duration: f32,
}

impl KeyframeInfo {
    /// Number of keyframes detected.
    pub fn count(&self) -> usize {
        self.timestamps.len()
    }

    /// Iterator over gaps between consecutive keyframes.
    pub fn gaps(&self) -> impl Iterator<Item = f32> + '_ {
        self.timestamps.windows(2).map(|w| w[1] - w[0])
    }

    /// Mean gap between keyframes, or 0 if fewer than 2 keyframes.
    pub fn mean_gap(&self) -> f32 {
        if self.timestamps.len() < 2 {
            return 0.0;
        }
        let total: f32 = self.gaps().sum();
        total / (self.timestamps.len() - 1) as f32
    }

    /// Maximum gap between keyframes, or 0 if fewer than 2 keyframes.
    pub fn max_gap(&self) -> f32 {
        self.gaps().fold(0.0_f32, f32::max)
    }

    /// Find sparse regions where gap exceeds threshold.
    ///
    /// Returns ascending, non-overlapping [`TimeRange`]s that need fill; the
    /// ordering invariants downstream region merging relies on come with the
    /// type, not a comment.
    pub fn sparse_regions(&self, threshold: f32) -> Vec<TimeRange> {
        let mut regions = Vec::new();
        let mut push = |start: f32, end: f32| {
            // Timestamps are verified non-decreasing and finite; a gap that
            // beat `threshold` (> 0) is therefore a valid range.
            regions.push(TimeRange::new(start, end).expect("monotonic timestamps form a range"));
        };

        // Gap before first keyframe
        if let Some(&first) = self.timestamps.first()
            && first > threshold
        {
            push(0.0, first);
        }

        // Gaps between keyframes
        for window in self.timestamps.windows(2) {
            let gap = window[1] - window[0];
            if gap > threshold {
                push(window[0], window[1]);
            }
        }

        // Gap after last keyframe
        if let Some(&last) = self.timestamps.last()
            && self.duration - last > threshold
        {
            push(last, self.duration);
        }

        regions
    }
}

/// Probe video duration only (container header; no packet scanning).
pub fn probe_duration(input: &Source) -> Result<f32> {
    let ictx = media::open_input(input)?;
    media::duration_seconds(&ictx, input)
}

/// Probe a video for I-frame timestamps and duration.
///
/// A single demux pass: every packet of the best video stream is inspected
/// for the keyframe flag; nothing is decoded.
pub fn probe_keyframes(input: &Source) -> Result<KeyframeInfo> {
    use ffmpeg_next as ffmpeg;

    let mut ictx = media::open_input(input)?;
    let duration = media::duration_seconds(&ictx, input)?;
    let video = media::best_video_stream(&ictx, input)?;

    let mut timestamps = Vec::new();
    let mut packet = ffmpeg::Packet::empty();
    let mut read_budget = media::ErrorBudget::new("scanning packets for keyframes");
    loop {
        match packet.read(&mut ictx) {
            Ok(()) => read_budget.ok(),
            Err(ffmpeg::Error::Eof) => break,
            Err(e) => {
                // Demux hiccups are skipped here exactly as they are during
                // decoding, behind the same circuit breaker: an input that
                // fails forever without reaching EOF (e.g. a dropped network
                // stream) must error out, not spin.
                read_budget
                    .fail(e)
                    .map_err(|err| err.with_input(input.as_path()))?;
                continue;
            }
        }
        if packet.stream() != video.index || !packet.is_key() {
            continue;
        }
        // pts falls back to dts: some containers omit pts on packets, and for
        // I-frames dts equals presentation order anyway.
        let Some(pts) = packet.pts().or(packet.dts()) else {
            continue;
        };
        let ts = video.ts_seconds(pts);
        if ts.is_finite() {
            timestamps.push(ts);
        }
    }

    let timestamps = MonotonicTimestamps::new(timestamps).map_err(|nm| {
        Error::new(
            ErrorCode::InputCorrupt,
            format!(
                "keyframe timestamps are not monotonic (index {}: {} after {}); \
                 the timeline cannot be trusted — use --strategy fixed-fps",
                nm.index, nm.next, nm.prev
            ),
        )
        .with_context(ErrorContext::default().with_input(input.as_path()))
    })?;

    Ok(KeyframeInfo {
        timestamps,
        duration,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mono(ts: Vec<f32>) -> MonotonicTimestamps {
        MonotonicTimestamps::new(ts).expect("ascending test data")
    }

    fn tr(start: f32, end: f32) -> TimeRange {
        TimeRange::new(start, end).expect("valid test range")
    }

    #[test]
    fn accepts_equal_adjacent_timestamps() {
        let ts = mono(vec![0.0, 1.0, 1.0, 2.0]);
        assert_eq!(&*ts, &[0.0, 1.0, 1.0, 2.0]);
    }

    #[test]
    fn rejects_non_monotonic_timestamps() {
        // A demux order that disagrees with pts order must fail construction,
        // not be sorted: sorting would hide that the timeline is broken.
        let err = MonotonicTimestamps::new(vec![0.0, 2.0, 1.0]).expect_err("descending pair");
        assert_eq!(
            err,
            NonMonotonic {
                index: 2,
                prev: 2.0,
                next: 1.0
            }
        );
    }

    #[test]
    fn rejects_nan_timestamps_as_non_monotonic() {
        // NaN compares false against everything, so it can never satisfy the
        // non-decreasing contract.
        MonotonicTimestamps::new(vec![0.0, f32::NAN]).expect_err("NaN never compares");
    }

    #[test]
    fn sparse_regions_finds_gaps() {
        let info = KeyframeInfo {
            timestamps: mono(vec![0.0, 2.0, 4.0, 10.0, 12.0]),
            duration: 15.0,
        };
        let regions = info.sparse_regions(5.0);
        assert_eq!(regions, vec![tr(4.0, 10.0)]);
    }

    #[test]
    fn sparse_regions_includes_start_gap() {
        let info = KeyframeInfo {
            timestamps: mono(vec![8.0, 10.0]),
            duration: 12.0,
        };
        let regions = info.sparse_regions(5.0);
        assert_eq!(regions, vec![tr(0.0, 8.0)]);
    }

    #[test]
    fn sparse_regions_includes_end_gap() {
        let info = KeyframeInfo {
            timestamps: mono(vec![0.0, 2.0]),
            duration: 10.0,
        };
        let regions = info.sparse_regions(5.0);
        assert_eq!(regions, vec![tr(2.0, 10.0)]);
    }

    #[test]
    fn mean_gap_handles_single_keyframe() {
        let info = KeyframeInfo {
            timestamps: mono(vec![5.0]),
            duration: 10.0,
        };
        assert_eq!(info.mean_gap(), 0.0);
    }
}
