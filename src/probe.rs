//! Video metadata probing via ffprobe.
//!
//! Extracts I-frame (keyframe) timestamps and video duration without decoding.
//! Used to enable keyframe-based sampling strategies.

use std::path::Path;
use std::process::Command;

use crate::error::{Error, ErrorCode, ErrorContext, Result};

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
    /// Timestamps of I-frames in seconds, sorted ascending.
    pub timestamps: Vec<f32>,
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
        if self.timestamps.first().is_some_and(|&t| t > threshold) {
            regions.push((0.0, self.timestamps[0]));
        }

        // Gaps between keyframes
        for window in self.timestamps.windows(2) {
            let gap = window[1] - window[0];
            if gap > threshold {
                regions.push((window[0], window[1]));
            }
        }

        // Gap after last keyframe
        if self
            .timestamps
            .last()
            .is_some_and(|&t| self.duration - t > threshold)
        {
            regions.push((*self.timestamps.last().unwrap(), self.duration));
        }

        regions
    }
}

/// Probe video duration only (fast, no frame scanning).
pub fn probe_duration(input: &Path) -> Result<f32> {
    let output = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
        ])
        .arg(input)
        .output()
        .map_err(|e| {
            Error::new(ErrorCode::FfprobeNotFound, "ffprobe not found on PATH").with_source(e)
        })?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(Error::new(
            ErrorCode::FfprobeFailed,
            format!("ffprobe failed: {}", stderr.trim()),
        )
        .with_context(
            ErrorContext::default()
                .with_input(input)
                .with_exit_status(output.status)
                .with_stderr(stderr.to_string()),
        ));
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    stdout.trim().parse::<f32>().map_err(|e| {
        Error::new(
            ErrorCode::FfprobeFailed,
            format!("failed to parse duration from ffprobe: {}", stdout.trim()),
        )
        .with_source(e)
    })
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

fn probe_iframe_timestamps(input: &Path) -> Result<Vec<f32>> {
    let output = Command::new("ffprobe")
        .args([
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
        .arg(input)
        .output()
        .map_err(|e| {
            Error::new(ErrorCode::FfprobeNotFound, "ffprobe not found on PATH").with_source(e)
        })?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(Error::new(
            ErrorCode::FfprobeFailed,
            format!("ffprobe I-frame scan failed: {}", stderr.trim()),
        )
        .with_context(
            ErrorContext::default()
                .with_input(input)
                .with_exit_status(output.status)
                .with_stderr(stderr.to_string()),
        ));
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let (timestamps, parse_errors) = parse_pts_csv(&stdout);

    // Warn about parse failures but don't fail entirely if we got some timestamps
    if !parse_errors.is_empty() {
        let count = parse_errors.len();
        eprintln!("warning: skipped {count} unparseable pts_time lines from ffprobe");
        for (line_num, content) in parse_errors.iter().take(3) {
            eprintln!("  line {line_num}: {content:?}");
        }
        if count > 3 {
            eprintln!("  ... and {} more", count - 3);
        }
    }

    Ok(timestamps)
}

/// Parse ffprobe's per-frame pts_time csv into sorted timestamps plus the
/// 1-indexed lines that failed to parse.
///
/// Only the first csv field is the timestamp: ffmpeg appends empty trailing
/// fields for frames carrying side data (e.g. `"0.000000,"`), and treating the
/// whole line as the number would drop that frame and shift every timestamp by
/// one keyframe position.
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
            Ok(ts) => timestamps.push(ts),
            Err(_) => parse_errors.push((line_num + 1, trimmed.to_string())),
        }
    }

    timestamps.sort_by(|a, b| a.total_cmp(b));
    (timestamps, parse_errors)
}

/// Check if ffprobe is available on PATH.
pub fn ffprobe_available() -> bool {
    Command::new("ffprobe")
        .arg("-version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sparse_regions_finds_gaps() {
        let info = KeyframeInfo {
            timestamps: vec![0.0, 2.0, 4.0, 10.0, 12.0],
            duration: 15.0,
        };
        let regions = info.sparse_regions(5.0);
        assert_eq!(regions, vec![(4.0, 10.0)]);
    }

    #[test]
    fn sparse_regions_includes_start_gap() {
        let info = KeyframeInfo {
            timestamps: vec![8.0, 10.0],
            duration: 12.0,
        };
        let regions = info.sparse_regions(5.0);
        assert_eq!(regions, vec![(0.0, 8.0)]);
    }

    #[test]
    fn sparse_regions_includes_end_gap() {
        let info = KeyframeInfo {
            timestamps: vec![0.0, 2.0],
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
    fn mean_gap_handles_single_keyframe() {
        let info = KeyframeInfo {
            timestamps: vec![5.0],
            duration: 10.0,
        };
        assert_eq!(info.mean_gap(), 0.0);
    }
}
