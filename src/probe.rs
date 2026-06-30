//! Video metadata probing via ffprobe.
//!
//! Extracts I-frame (keyframe) timestamps and video duration without decoding.
//! Used to enable keyframe-based sampling strategies.

use std::path::Path;
use std::process::Command;

use anyhow::{Context, Result, bail};

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
        .context("ffprobe not found on PATH")?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        bail!("ffprobe failed: {}", stderr.trim());
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    stdout
        .trim()
        .parse::<f32>()
        .context("failed to parse duration from ffprobe")
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
        .context("ffprobe not found on PATH")?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        bail!("ffprobe I-frame scan failed: {}", stderr.trim());
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut timestamps: Vec<f32> = stdout
        .lines()
        .filter_map(|line| line.trim().parse::<f32>().ok())
        .collect();
    timestamps.sort_by(|a, b| a.total_cmp(b));

    Ok(timestamps)
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
    fn mean_gap_handles_single_keyframe() {
        let info = KeyframeInfo {
            timestamps: vec![5.0],
            duration: 10.0,
        };
        assert_eq!(info.mean_gap(), 0.0);
    }
}
