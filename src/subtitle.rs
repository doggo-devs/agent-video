//! Subtitle extraction: embedded streams via ffmpeg, or external sidecar files.
//!
//! Parses SRT and provides time-range slicing for aligning subtitles with grids.

use std::fs;
use std::path::Path;
use std::process::Command;

use anyhow::{Context, bail};

/// A single parsed subtitle cue.
#[derive(Debug, Clone)]
pub struct Cue {
    pub start: f32,
    pub end: f32,
    pub text: String,
}

/// Embedded subtitle track info.
#[derive(Debug)]
struct EmbeddedTrack {
    index: usize,
    language: Option<String>,
}

/// Extract subtitles from an explicit file path.
pub fn extract_from_file(path: &Path) -> anyhow::Result<String> {
    fs::read_to_string(path)
        .with_context(|| format!("reading subtitle file: {}", path.display()))
}

/// Extract a specific embedded subtitle track by index.
pub fn extract_track(input: &Path, track: usize) -> anyhow::Result<String> {
    let map_arg = format!("0:s:{track}");
    let out = Command::new("ffmpeg")
        .arg("-i")
        .arg(input)
        .args(["-map", &map_arg, "-f", "srt", "pipe:1"])
        .output()?;

    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        bail!("failed to extract subtitle track {track}: {}", stderr.trim());
    }
    let text = String::from_utf8_lossy(&out.stdout).trim().to_string();
    if text.is_empty() {
        bail!("subtitle track {track} is empty");
    }
    Ok(text)
}

/// Extract subtitles with automatic source detection.
///
/// Fails if multiple sources exist (embedded + sidecar, or multiple sidecars/tracks).
/// Returns `None` when no subtitles are found.
pub fn extract(input: &Path) -> anyhow::Result<Option<String>> {
    let sidecars = find_all_sidecars(input);
    let embedded = list_embedded_tracks(input);

    let total = sidecars.len() + embedded.len();
    if total > 1 {
        let mut sources = Vec::new();
        for s in &sidecars {
            sources.push(format!("  sidecar: {}", s.display()));
        }
        for t in &embedded {
            let lang = t.language.as_deref().unwrap_or("unknown");
            sources.push(format!("  track {}: {lang}", t.index));
        }
        bail!(
            "multiple subtitle sources detected, use --subtitle or --subtitle-track:\n{}",
            sources.join("\n")
        );
    }

    // Single sidecar
    if let Some(path) = sidecars.into_iter().next() {
        return read_nonempty(&path)
            .map(Some)
            .ok_or_else(|| anyhow::anyhow!("sidecar file empty: {}", path.display()));
    }

    // Single embedded stream
    if embedded.len() == 1 {
        return Ok(Some(extract_embedded(input, 0)?));
    }

    Ok(None)
}

/// Find all sidecar subtitle files alongside the video.
fn find_all_sidecars(input: &Path) -> Vec<std::path::PathBuf> {
    let Some(stem) = input.file_stem().and_then(|s| s.to_str()) else {
        return Vec::new();
    };
    let Some(parent) = input.parent() else {
        return Vec::new();
    };

    let mut results = Vec::new();

    // Check exact match: video.srt
    let exact = parent.join(format!("{stem}.srt"));
    if exact.is_file() {
        results.push(exact);
    }

    // Check pattern: video.*.srt (e.g. video.zh.srt)
    if let Ok(entries) = fs::read_dir(parent) {
        for entry in entries.flatten() {
            let name = entry.file_name();
            let name = name.to_string_lossy();
            let is_pattern_match =
                name.starts_with(stem) && name.ends_with(".srt") && name.len() > stem.len() + 4;
            if is_pattern_match {
                results.push(entry.path());
            }
        }
    }

    results
}

/// List embedded subtitle tracks with language info.
fn list_embedded_tracks(input: &Path) -> Vec<EmbeddedTrack> {
    let output = Command::new("ffprobe")
        .args([
            "-v", "error",
            "-select_streams", "s",
            "-show_entries", "stream=index:stream_tags=language",
            "-of", "csv=p=0",
        ])
        .arg(input)
        .output()
        .ok();

    let Some(output) = output else {
        return Vec::new();
    };

    String::from_utf8_lossy(&output.stdout)
        .lines()
        .enumerate()
        .map(|(i, line)| {
            let language = line.split(',').nth(1).map(|s| s.trim().to_string());
            EmbeddedTrack {
                index: i,
                language: language.filter(|s| !s.is_empty()),
            }
        })
        .collect()
}

fn read_nonempty(path: &Path) -> Option<String> {
    fs::read_to_string(path)
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

/// Extract an embedded subtitle track by index as SRT text via ffmpeg.
fn extract_embedded(input: &Path, track: usize) -> anyhow::Result<String> {
    let map_arg = format!("0:s:{track}");
    let out = Command::new("ffmpeg")
        .arg("-i")
        .arg(input)
        .args(["-map", &map_arg, "-f", "srt", "pipe:1"])
        .output()?;

    if !out.status.success() {
        bail!("no subtitle stream found");
    }
    let text = String::from_utf8_lossy(&out.stdout).trim().to_string();
    if text.is_empty() {
        bail!("subtitle stream is empty");
    }
    Ok(text)
}

/// Parse SRT text into a list of cues.
pub fn parse_srt(srt: &str) -> Vec<Cue> {
    let mut cues = Vec::new();
    let mut lines = srt.lines().peekable();

    while lines.peek().is_some() {
        // Skip cue index (numeric line)
        if let Some(line) = lines.next() {
            if line.trim().is_empty() {
                continue;
            }
            // Cue index line — next should be timestamp
            if !line.trim().chars().all(|c| c.is_ascii_digit()) {
                continue;
            }
        }

        // Parse timestamp line: "00:00:01,234 --> 00:00:04,567"
        let Some(ts_line) = lines.next() else {
            break;
        };
        let Some((start, end)) = parse_timestamp_line(ts_line) else {
            continue;
        };

        // Collect text lines until blank line
        let mut text_lines = Vec::new();
        while let Some(line) = lines.peek() {
            if line.trim().is_empty() {
                lines.next();
                break;
            }
            text_lines.push(lines.next().unwrap());
        }

        if !text_lines.is_empty() {
            cues.push(Cue {
                start,
                end,
                text: text_lines.join(" "),
            });
        }
    }

    cues
}

/// Parse "00:00:01,234 --> 00:00:04,567" into (start_secs, end_secs).
fn parse_timestamp_line(line: &str) -> Option<(f32, f32)> {
    let parts: Vec<&str> = line.split("-->").collect();
    if parts.len() != 2 {
        return None;
    }
    let start = parse_srt_time(parts[0].trim())?;
    let end = parse_srt_time(parts[1].trim())?;
    Some((start, end))
}

/// Parse "00:01:23,456" or "00:01:23.456" into seconds.
fn parse_srt_time(s: &str) -> Option<f32> {
    let s = s.replace(',', ".");
    let parts: Vec<&str> = s.split(':').collect();
    if parts.len() != 3 {
        return None;
    }
    let hours: f32 = parts[0].parse().ok()?;
    let mins: f32 = parts[1].parse().ok()?;
    let secs: f32 = parts[2].parse().ok()?;
    Some(hours * 3600.0 + mins * 60.0 + secs)
}

/// Extract subtitles that overlap with a time range [start, end].
pub fn slice_for_range(cues: &[Cue], start: f32, end: f32) -> Option<String> {
    let text: String = cues
        .iter()
        .filter(|c| c.end > start && c.start < end)
        .map(|c| c.text.as_str())
        .collect::<Vec<_>>()
        .join(" ");
    if text.is_empty() { None } else { Some(text) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_srt() {
        let srt = r#"1
00:00:01,000 --> 00:00:03,500
Hello world

2
00:00:04,000 --> 00:00:06,000
Second line
with continuation
"#;
        let cues = parse_srt(srt);
        assert_eq!(cues.len(), 2);
        assert!((cues[0].start - 1.0).abs() < 0.01);
        assert!((cues[0].end - 3.5).abs() < 0.01);
        assert_eq!(cues[0].text, "Hello world");
        assert_eq!(cues[1].text, "Second line with continuation");
    }

    #[test]
    fn slices_by_range() {
        let cues = vec![
            Cue {
                start: 1.0,
                end: 3.0,
                text: "one".into(),
            },
            Cue {
                start: 5.0,
                end: 8.0,
                text: "two".into(),
            },
            Cue {
                start: 10.0,
                end: 12.0,
                text: "three".into(),
            },
        ];
        assert_eq!(slice_for_range(&cues, 0.0, 4.0), Some("one".into()));
        assert_eq!(slice_for_range(&cues, 4.0, 9.0), Some("two".into()));
        assert_eq!(slice_for_range(&cues, 2.0, 6.0), Some("one two".into()));
        assert_eq!(slice_for_range(&cues, 20.0, 30.0), None);
    }
}
