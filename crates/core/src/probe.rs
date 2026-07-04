//! Video metadata probing via ffprobe.
//!
//! Extracts I-frame (keyframe) timestamps via a keyframe-only decode pass
//! (`-skip_frame nokey`) and video duration from the container header. Used to
//! enable keyframe-based sampling strategies.

use std::path::Path;
use std::process::Command;

use crate::error::{Error, ErrorCode, ErrorContext, Result};

/// Upper bound on buffered ffprobe stdout. Even an all-intra 24h stream lists
/// its keyframes in well under this; a stream reaching it is runaway output,
/// not a probe result.
const PROBE_STDOUT_CAP: u64 = 64 * 1024 * 1024;

/// Keyframe pts in decoder-emission order, verified non-decreasing.
///
/// Keyframe extraction pairs the i-th decoded frame with the i-th probed
/// timestamp; that pairing is trustworthy only because ffprobe and ffmpeg
/// decode the same stream in the same order. Sorting would repair nothing and
/// silently break the pairing whenever emission order differs from pts order
/// (spliced or timestamp-reset streams), so such a sequence fails
/// construction instead.
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

    /// Consume the verified list for index pairing.
    pub fn into_vec(self) -> Vec<f32> {
        self.0
    }
}

impl std::ops::Deref for MonotonicTimestamps {
    type Target = [f32];
    fn deref(&self) -> &[f32] {
        &self.0
    }
}

/// Sampling strategy for frame extraction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SampleStrategy {
    /// Fixed-rate sampling at base_fps (current behavior).
    #[default]
    FixedFps,
    /// Sample only I-frames (keyframes).
    Keyframes,
    /// I-frames with fixed-fps fill in sparse regions.
    Adaptive,
}

/// Information extracted via ffprobe about video keyframes.
#[derive(Debug, Clone)]
pub struct KeyframeInfo {
    /// Timestamps of I-frames in seconds, in decoder-emission order (verified
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
    /// Returns `(start, end)` pairs representing time ranges that need fill.
    pub fn sparse_regions(&self, threshold: f32) -> Vec<(f32, f32)> {
        let mut regions = Vec::new();

        // Gap before first keyframe
        if let Some(&first) = self.timestamps.first()
            && first > threshold
        {
            regions.push((0.0, first));
        }

        // Gaps between keyframes
        for window in self.timestamps.windows(2) {
            let gap = window[1] - window[0];
            if gap > threshold {
                regions.push((window[0], window[1]));
            }
        }

        // Gap after last keyframe
        if let Some(&last) = self.timestamps.last()
            && self.duration - last > threshold
        {
            regions.push((last, self.duration));
        }

        regions
    }
}

/// Probe video duration only (fast, no frame scanning).
pub fn probe_duration(input: &Path) -> Result<f32> {
    let mut cmd = Command::new("ffprobe");
    cmd.args([
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
    ])
    .arg(input);
    let stdout = crate::child::run_capped(
        &mut cmd,
        "ffprobe duration probe",
        ErrorCode::FfprobeNotFound,
        ErrorCode::FfprobeFailed,
        PROBE_STDOUT_CAP,
    )
    .map_err(|e| e.with_input(input))?;

    // "N/A" (fragmented MP4, raw streams, some webm) is a property of the
    // input, not a probe failure — code it in the input decade so the harness
    // can distinguish it from a broken environment. Non-finite or negative
    // values parse but are equally unusable as a duration.
    match stdout.trim().parse::<f32>() {
        Ok(d) if d.is_finite() && d >= 0.0 => Ok(d),
        _ => Err(Error::new(
            ErrorCode::InputCorrupt,
            format!(
                "cannot parse video duration from ffprobe output {:?}; \
                 the container may not report one",
                stdout.trim()
            ),
        )
        .with_context(ErrorContext::default().with_input(input))),
    }
}

/// Probe a video for I-frame timestamps and duration.
pub fn probe_keyframes(input: &Path) -> Result<KeyframeInfo> {
    let duration = probe_duration(input)?;
    let timestamps = probe_iframe_timestamps(input)?;
    Ok(KeyframeInfo {
        timestamps,
        duration,
    })
}

fn probe_iframe_timestamps(input: &Path) -> Result<MonotonicTimestamps> {
    let mut cmd = Command::new("ffprobe");
    cmd.args([
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-skip_frame",
        "nokey",
        "-show_entries",
        "frame=pts_time",
        "-of",
        "csv=p=0",
    ])
    .arg(input);
    let stdout = crate::child::run_capped(
        &mut cmd,
        "ffprobe I-frame scan",
        ErrorCode::FfprobeNotFound,
        ErrorCode::FfprobeFailed,
        PROBE_STDOUT_CAP,
    )
    .map_err(|e| e.with_input(input))?;

    let (timestamps, parse_errors) = parse_pts_csv(&stdout);

    // A skipped line is not recoverable: frames are paired with these
    // timestamps by index, so one missing entry shifts every later keyframe
    // onto its successor's timestamp. Fail loudly instead of shipping grids
    // with silently misattributed time ranges.
    if let Some((line_num, content)) = parse_errors.first() {
        return Err(Error::new(
            ErrorCode::FfprobeFailed,
            format!(
                "{} unparseable pts_time line(s) from ffprobe (first: line {line_num}: \
                 {content:?}); keyframe timestamps would misalign — use --strategy fixed-fps",
                parse_errors.len()
            ),
        )
        .with_context(ErrorContext::default().with_input(input)));
    }

    MonotonicTimestamps::new(timestamps).map_err(|nm| {
        Error::new(
            ErrorCode::InputCorrupt,
            format!(
                "keyframe pts_time from ffprobe is not monotonic (index {}: {} after {}); \
                 decoded frames cannot be paired with timestamps — use --strategy fixed-fps",
                nm.index, nm.next, nm.prev
            ),
        )
        .with_context(ErrorContext::default().with_input(input))
    })
}

/// Parse ffprobe's per-frame pts_time csv into timestamps (kept in emission
/// order — see [`MonotonicTimestamps`]) plus the 1-indexed lines that failed
/// to parse.
///
/// Only the first csv field is the timestamp: ffmpeg appends empty trailing
/// fields for frames carrying side data (e.g. `"0.000000,"`), and treating the
/// whole line as the number would drop that frame and shift every timestamp by
/// one keyframe position. Non-finite values ("nan", "inf") parse as f32 but
/// are not usable timestamps, so they count as parse failures too.
fn parse_pts_csv(stdout: &str) -> (Vec<f32>, Vec<(usize, String)>) {
    let mut timestamps: Vec<f32> = Vec::new();
    let mut parse_errors: Vec<(usize, String)> = Vec::new();

    for (line_num, line) in stdout.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let field = trimmed
            .split(',')
            .next()
            .expect("split yields at least one field");
        match field.parse::<f32>() {
            Ok(ts) if ts.is_finite() => timestamps.push(ts),
            _ => parse_errors.push((line_num + 1, trimmed.to_string())),
        }
    }

    (timestamps, parse_errors)
}

/// Check if ffprobe is available on PATH.
pub fn ffprobe_available() -> bool {
    crate::child::run_capped(
        Command::new("ffprobe").arg("-version"),
        "ffprobe version check",
        ErrorCode::FfprobeNotFound,
        ErrorCode::FfprobeFailed,
        64 * 1024,
    )
    .is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mono(ts: Vec<f32>) -> MonotonicTimestamps {
        MonotonicTimestamps::new(ts).expect("ascending test data")
    }

    #[test]
    fn accepts_equal_adjacent_timestamps() {
        let ts = mono(vec![0.0, 1.0, 1.0, 2.0]);
        assert_eq!(&*ts, &[0.0, 1.0, 1.0, 2.0]);
    }

    #[test]
    fn rejects_non_monotonic_timestamps() {
        // A decode order that disagrees with pts order must fail construction,
        // not be sorted: sorting silently breaks index pairing with the frame
        // stream.
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
        assert_eq!(regions, vec![(4.0, 10.0)]);
    }

    #[test]
    fn sparse_regions_includes_start_gap() {
        let info = KeyframeInfo {
            timestamps: mono(vec![8.0, 10.0]),
            duration: 12.0,
        };
        let regions = info.sparse_regions(5.0);
        assert_eq!(regions, vec![(0.0, 8.0)]);
    }

    #[test]
    fn sparse_regions_includes_end_gap() {
        let info = KeyframeInfo {
            timestamps: mono(vec![0.0, 2.0]),
            duration: 10.0,
        };
        let regions = info.sparse_regions(5.0);
        assert_eq!(regions, vec![(2.0, 10.0)]);
    }

    #[test]
    fn parses_pts_lines_with_trailing_csv_fields() {
        // ffmpeg 8.x emits "0.000000," for frames with side data; the frame
        // must be kept, or every later timestamp shifts one keyframe position.
        let (ts, errors) = parse_pts_csv("0.000000,\n2.000000\nbogus\n\n4.000000,extra\n");
        assert_eq!(ts, vec![0.0, 2.0, 4.0]);
        assert_eq!(errors, vec![(3, "bogus".to_string())]);
    }

    #[test]
    fn rejects_non_finite_pts_values() {
        // "nan" and "inf" parse as f32 but would poison chunk keying and
        // monotonicity downstream; they must count as parse failures.
        let (ts, errors) = parse_pts_csv("0.0\nnan\ninf\n2.0\n");
        assert_eq!(ts, vec![0.0, 2.0]);
        assert_eq!(errors.len(), 2, "nan and inf are rejected: {errors:?}");
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
