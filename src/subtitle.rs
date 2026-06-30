//! Subtitle extraction: embedded streams via ffmpeg, or external sidecar files.
//!
//! Parses SRT and provides time-range slicing for aligning subtitles with grids.

use std::fs;
use std::path::Path;
use std::process::Command;

use anyhow::{Context, bail};

use crate::child::TrackedChild;
use crate::error::{Error, ErrorCode, ErrorContext};

/// A single parsed subtitle cue.
#[derive(Debug, Clone)]
pub struct Cue {
    pub start: f32,
    pub end: f32,
    pub text: String,
}

/// Resolve subtitle text with priority: explicit path > explicit track > auto-detect.
pub fn resolve(
    input: &Path,
    path: Option<&Path>,
    track: Option<usize>,
) -> anyhow::Result<Option<String>> {
    match (path, track) {
        (Some(p), _) => Ok(Some(extract_from_file(p)?)),
        (None, Some(t)) => Ok(Some(extract_track(input, t)?)),
        (None, None) => extract(input),
    }
}

/// Embedded subtitle track info.
#[derive(Debug)]
struct EmbeddedTrack {
    index: usize,
    language: Option<String>,
}

/// Extract subtitles from an explicit file path.
fn extract_from_file(path: &Path) -> anyhow::Result<String> {
    fs::read_to_string(path)
        .with_context(|| format!("reading subtitle file: {}", path.display()))
}

/// Extract a specific embedded subtitle track by index.
fn extract_track(input: &Path, track: usize) -> anyhow::Result<String> {
    let map_arg = format!("0:s:{track}");
    let mut cmd = Command::new("ffmpeg");
    cmd.arg("-i")
        .arg(input)
        .args(["-map", &map_arg, "-f", "srt", "pipe:1"]);

    let (child, mut stdout) = TrackedChild::spawn(
        &mut cmd,
        "ffmpeg for subtitle extraction",
        ErrorCode::FfmpegNotFound,
    )?;

    let mut text = String::new();
    std::io::Read::read_to_string(&mut stdout, &mut text)
        .context("reading subtitle output")?;

    child.wait("ffmpeg subtitle extraction").map_err(|e| {
        anyhow::anyhow!("failed to extract subtitle track {track}: {e}")
    })?;

    let text = text.trim().to_string();
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
    let sidecars = find_all_sidecars(input)?;
    let embedded = list_embedded_tracks(input)?;

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
        return Err(Error::new(
            ErrorCode::AmbiguousSubtitle,
            format!(
                "multiple subtitle sources detected, use --subtitle or --subtitle-track:\n{}",
                sources.join("\n")
            ),
        )
        .into());
    }

    // Single sidecar
    if let Some(path) = sidecars.into_iter().next() {
        return match read_sidecar(&path)? {
            Some(content) => Ok(Some(content)),
            None => Err(Error::new(
                ErrorCode::InputCorrupt,
                format!("sidecar file is empty: {}", path.display()),
            )
            .into()),
        };
    }

    // Single embedded stream
    if embedded.len() == 1 {
        return Ok(Some(extract_embedded(input, 0)?));
    }

    Ok(None)
}

/// Find all sidecar subtitle files alongside the video.
///
/// Returns an error if directory listing fails (e.g., permission denied).
fn find_all_sidecars(input: &Path) -> crate::error::Result<Vec<std::path::PathBuf>> {
    let Some(stem) = input.file_stem().and_then(|s| s.to_str()) else {
        return Ok(Vec::new());
    };
    let Some(parent) = sidecar_dir(input) else {
        return Ok(Vec::new());
    };

    let mut results = Vec::new();

    // Check exact match: video.srt
    let exact = parent.join(format!("{stem}.srt"));
    if exact.is_file() {
        results.push(exact);
    }

    // Check pattern: video.*.srt (e.g. video.zh.srt)
    let entries = fs::read_dir(parent).map_err(|e| {
        Error::new(
            ErrorCode::InputUnreadable,
            format!("cannot list directory {}: {e}", parent.display()),
        )
        .with_context(ErrorContext::default().with_input(parent))
        .with_source(e)
    })?;

    for entry in entries {
        let entry = entry.map_err(|e| {
            Error::new(
                ErrorCode::InputUnreadable,
                format!("cannot read directory entry: {e}"),
            )
            .with_source(e)
        })?;
        let name = entry.file_name();
        let name = name.to_string_lossy();
        let is_pattern_match =
            name.starts_with(stem) && name.ends_with(".srt") && name.len() > stem.len() + 4;
        if is_pattern_match {
            results.push(entry.path());
        }
    }

    Ok(results)
}

/// Directory to scan for sidecar files: the input's parent, or `.` for a bare
/// filename — a bare name's parent is the empty path, which `fs::read_dir`
/// rejects.
fn sidecar_dir(input: &Path) -> Option<&Path> {
    let parent = input.parent()?;
    Some(if parent.as_os_str().is_empty() {
        Path::new(".")
    } else {
        parent
    })
}

/// List embedded subtitle tracks with language info.
///
/// Returns an error if ffprobe fails; returns empty Vec if no subtitle streams.
fn list_embedded_tracks(input: &Path) -> crate::error::Result<Vec<EmbeddedTrack>> {
    let output = Command::new("ffprobe")
        .args([
            "-v", "error",
            "-select_streams", "s",
            "-show_entries", "stream=index:stream_tags=language",
            "-of", "csv=p=0",
        ])
        .arg(input)
        .output()
        .map_err(|e| {
            Error::new(ErrorCode::FfprobeNotFound, "ffprobe not found on PATH")
                .with_source(e)
        })?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(Error::new(
            ErrorCode::FfprobeFailed,
            format!("ffprobe subtitle scan failed: {}", stderr.trim()),
        )
        .with_context(
            ErrorContext::default()
                .with_input(input)
                .with_exit_status(output.status)
                .with_stderr(stderr.to_string()),
        ));
    }

    Ok(String::from_utf8_lossy(&output.stdout)
        .lines()
        .enumerate()
        .map(|(i, line)| {
            let language = line.split(',').nth(1).map(|s| s.trim().to_string());
            EmbeddedTrack {
                index: i,
                language: language.filter(|s| !s.is_empty()),
            }
        })
        .collect())
}

/// Read a sidecar file, distinguishing "not found" from "read error" from "empty".
///
/// Returns:
/// - `Ok(Some(content))` if the file exists and has content
/// - `Ok(None)` if the file doesn't exist
/// - `Err` if the file exists but cannot be read (permissions, I/O error)
fn read_sidecar(path: &Path) -> anyhow::Result<Option<String>> {
    match fs::read_to_string(path) {
        Ok(content) => {
            let trimmed = content.trim().to_string();
            if trimmed.is_empty() {
                Ok(None)
            } else {
                Ok(Some(trimmed))
            }
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(anyhow::anyhow!("reading sidecar {}: {e}", path.display())),
    }
}

/// Extract an embedded subtitle track by index as SRT text via ffmpeg.
fn extract_embedded(input: &Path, track: usize) -> anyhow::Result<String> {
    let map_arg = format!("0:s:{track}");
    let mut cmd = Command::new("ffmpeg");
    cmd.arg("-i")
        .arg(input)
        .args(["-map", &map_arg, "-f", "srt", "pipe:1"]);

    let (child, mut stdout) = TrackedChild::spawn(
        &mut cmd,
        "ffmpeg for embedded subtitle",
        ErrorCode::FfmpegNotFound,
    )?;

    let mut text = String::new();
    std::io::Read::read_to_string(&mut stdout, &mut text)
        .context("reading subtitle output")?;

    child
        .wait("ffmpeg embedded subtitle")
        .map_err(|e| anyhow::anyhow!("{e}"))?;

    let text = text.trim().to_string();
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

    #[test]
    fn scans_current_dir_for_bare_filename() {
        // A bare filename's parent is the empty path; scanning must fall back
        // to `.` instead of handing read_dir an unreadable "".
        assert_eq!(sidecar_dir(Path::new("video.mp4")), Some(Path::new(".")));
    }

    #[test]
    fn scans_parent_dir_for_nested_path() {
        assert_eq!(
            sidecar_dir(Path::new("/data/clips/video.mp4")),
            Some(Path::new("/data/clips"))
        );
    }
}
