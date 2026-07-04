//! Subtitle resolution: embedded streams via ffmpeg, or external sidecar files.
//!
//! [`Subtitles`] is the single boundary of the subtitle domain: source
//! resolution (explicit path > explicit track > auto-detection), SRT parsing,
//! and time-range slicing all live behind it, and every failure leaves with a
//! structured [`Error`] code regardless of which source was used.

use std::fs;
use std::io::Read;
use std::path::Path;
use std::process::Command;

use crate::child::TrackedChild;
use crate::error::{Error, ErrorCode, ErrorContext, Result};
use crate::types::TimeRange;

/// Upper bound on subtitle stdout we will buffer. Real subtitle tracks are far
/// smaller; a stream that reaches this is treated as runaway output.
const MAX_SUBTITLE_BYTES: u64 = 16 * 1024 * 1024;

/// Read an ffmpeg subtitle stream to a string, bounded by [`MAX_SUBTITLE_BYTES`].
/// Fails if the cap is exceeded rather than returning silently truncated text.
fn read_capped(r: impl Read) -> Result<String> {
    let mut text = String::new();
    // Take one byte past the cap: `take(cap)` alone cannot distinguish
    // "exactly cap" (acceptable) from "more than cap" (runaway).
    let read = r
        .take(MAX_SUBTITLE_BYTES + 1)
        .read_to_string(&mut text)
        .map_err(|e| {
            Error::new(
                ErrorCode::FfmpegFailed,
                "reading subtitle output from ffmpeg",
            )
            .with_source(e)
        })?;
    if read as u64 > MAX_SUBTITLE_BYTES {
        return Err(Error::new(
            ErrorCode::InputCorrupt,
            format!(
                "subtitle output exceeded {MAX_SUBTITLE_BYTES} bytes; refusing to buffer runaway stream"
            ),
        ));
    }
    Ok(text)
}

/// A single parsed subtitle cue.
#[derive(Debug, Clone)]
pub struct Cue {
    pub start: f32,
    pub end: f32,
    pub text: String,
}

/// Where subtitles come from, decided at CLI parsing time.
#[derive(Debug, Clone, Copy)]
pub enum SubtitleSource<'a> {
    /// `--subtitle`: explicit file; every failure is fatal.
    File(&'a Path),
    /// `--subtitle-track`: explicit embedded track; every failure is fatal.
    Track(usize),
    /// Neither flag: best-effort discovery. Environmental failures (an
    /// unlistable directory, a missing ffprobe) degrade to warnings, because
    /// this path runs on every invocation including ones that need no
    /// subtitles; a source that is *found* but unreadable or corrupt stays
    /// fatal.
    Auto,
}

/// Resolved subtitles for one run: the raw SRT text plus its parsed cues.
///
/// Constructed only by [`Subtitles::resolve`], so holding a value means every
/// discovery, read, extraction, and parse failure has already been mapped to a
/// structured [`Error`] — the machine-readable exit code is the same whether
/// the source was an explicit `--subtitle` path, a `--subtitle-track` index,
/// or an auto-discovered sidecar/embedded stream. A resolved value always
/// holds at least one cue: a source that parses to zero cues is rejected, not
/// carried as silently empty metadata.
#[derive(Debug)]
pub struct Subtitles {
    raw: String,
    cues: Vec<Cue>,
}

impl Subtitles {
    /// Resolve subtitles from the given source. Returns `Ok(None)` only on
    /// the [`SubtitleSource::Auto`] path, when no source exists. A source
    /// that yields text but no parseable SRT cues fails with
    /// [`ErrorCode::InputCorrupt`]: every grid's `subtitles` field would be
    /// empty while the run reports success — a corrupt input, not a resolved
    /// one.
    pub fn resolve(input: &Path, source: SubtitleSource<'_>) -> Result<Option<Self>> {
        let raw = match source {
            SubtitleSource::File(p) => Some(extract_from_file(p)?),
            SubtitleSource::Track(t) => Some(extract_embedded(input, t, "subtitle track")?),
            SubtitleSource::Auto => auto_detect(input)?,
        };
        let Some(raw) = raw else { return Ok(None) };
        let cues = parse_srt(&raw);
        if cues.is_empty() {
            return Err(Error::new(
                ErrorCode::InputCorrupt,
                "subtitle source contains no parseable SRT cues; is it a valid SRT file?",
            )
            .with_context(ErrorContext::default().with_input(input)));
        }
        Ok(Some(Subtitles { raw, cues }))
    }

    /// Text of every cue overlapping `range`, for a grid's metadata entry.
    pub fn slice(&self, range: TimeRange) -> Option<String> {
        slice_for_range(&self.cues, range.start(), range.end())
    }

    /// The full resolved SRT text — the `subtitles.txt` payload.
    pub fn raw(&self) -> &str {
        &self.raw
    }
}

/// Embedded subtitle track info.
#[derive(Debug)]
struct EmbeddedTrack {
    index: usize,
    language: Option<String>,
}

/// Extract subtitles from an explicit file path. A missing or unreadable file
/// maps to the same coded errors as the video input.
fn extract_from_file(path: &Path) -> Result<String> {
    fs::read_to_string(path).map_err(|e| {
        let code = match e.kind() {
            std::io::ErrorKind::NotFound => ErrorCode::InputNotFound,
            _ => ErrorCode::InputUnreadable,
        };
        Error::new(code, format!("reading subtitle file {}", path.display()))
            .with_context(ErrorContext::default().with_input(path))
            .with_source(e)
    })
}

/// Extract subtitles with automatic source detection.
///
/// Fails if multiple sources exist (embedded + sidecar, or multiple sidecars/tracks).
/// Returns `None` when no subtitles are found.
fn auto_detect(input: &Path) -> Result<Option<String>> {
    let sidecars = find_all_sidecars(input);
    // ffprobe is optional on this path: `--strategy fixed-fps --force` and
    // `extract` are contracted to run without it, and auto-detection is the
    // only ffprobe use they reach. A missing binary downgrades embedded
    // detection to a warning; an ffprobe that exists but fails still errors.
    let embedded = match list_embedded_tracks(input) {
        Ok(tracks) => tracks,
        Err(e) if e.code == ErrorCode::FfprobeNotFound => {
            eprintln!("warning: ffprobe not found; skipping embedded subtitle detection");
            Vec::new()
        }
        Err(e) => return Err(e),
    };

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
        ));
    }

    // Single sidecar
    if let Some(path) = sidecars.into_iter().next() {
        return match read_sidecar(&path)? {
            Some(content) => Ok(Some(content)),
            None => Err(Error::new(
                ErrorCode::InputCorrupt,
                format!("sidecar file is empty: {}", path.display()),
            )),
        };
    }

    // Single embedded stream
    if embedded.len() == 1 {
        return Ok(Some(extract_embedded(
            input,
            0,
            "embedded subtitle stream",
        )?));
    }

    Ok(None)
}

/// Find all sidecar subtitle files alongside the video.
///
/// Best-effort by design: this runs on every auto-detect invocation,
/// including ones that need no subtitles, so a directory that cannot be
/// listed (permissions, sandbox) degrades to a warning and whatever the
/// exact-name check already found — mirroring the missing-ffprobe downgrade
/// in [`auto_detect`]. A sidecar that is *found* but unreadable still fails,
/// in [`read_sidecar`].
pub(crate) fn find_all_sidecars(input: &Path) -> Vec<std::path::PathBuf> {
    let Some(stem) = input.file_stem().and_then(|s| s.to_str()) else {
        return Vec::new();
    };
    let Some(parent) = sidecar_dir(input) else {
        return Vec::new();
    };

    let mut results = Vec::new();

    // Check exact match: video.srt — a direct stat, so it works even when the
    // directory below cannot be listed.
    let exact = parent.join(format!("{stem}.srt"));
    if exact.is_file() {
        results.push(exact);
    }

    // Check pattern: video.*.srt (e.g. video.zh.srt)
    let entries = match fs::read_dir(parent) {
        Ok(entries) => entries,
        Err(e) => {
            eprintln!(
                "warning: cannot list directory {} for subtitle sidecars: {e}; \
                 pass --subtitle to use one explicitly",
                parent.display()
            );
            return results;
        }
    };

    for entry in entries {
        let entry = match entry {
            Ok(entry) => entry,
            Err(e) => {
                eprintln!("warning: skipping unreadable directory entry during sidecar scan: {e}");
                continue;
            }
        };
        let name = entry.file_name();
        let name = name.to_string_lossy();
        // The stem must be followed by a literal `.`, or `video2.srt` would be
        // claimed as a sidecar of `video.mp4`. `rest.len() > 4` excludes the
        // exact `video.srt` (already collected above) where rest is just ".srt".
        let is_pattern_match = name
            .strip_prefix(stem)
            .is_some_and(|rest| rest.starts_with('.') && rest.ends_with(".srt") && rest.len() > 4);
        if is_pattern_match {
            results.push(entry.path());
        }
    }

    results
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
    let mut cmd = Command::new("ffprobe");
    cmd.args([
        "-v",
        "error",
        "-select_streams",
        "s",
        // `stream=index` is requested only to anchor one CSV row per
        // stream: a stream lacking a language tag would otherwise emit an
        // empty line and could desync the enumerate() below. The value
        // itself is the *global* stream index and is deliberately unused
        // — `-map 0:s:{i}` takes the subtitle-relative index, which is
        // exactly the enumerate position.
        "-show_entries",
        "stream=index:stream_tags=language",
        "-of",
        "csv=p=0",
    ])
    .arg(input);
    // One short CSV row per subtitle stream; 1 MiB covers thousands of them.
    let stdout = crate::child::run_capped(
        &mut cmd,
        "ffprobe subtitle scan",
        ErrorCode::FfprobeNotFound,
        ErrorCode::FfprobeFailed,
        1024 * 1024,
    )
    .map_err(|e| e.with_input(input))?;

    Ok(stdout
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
/// - `Err` if the file exists but cannot be read (permissions, I/O error),
///   carrying the same coded error an explicit `--subtitle` path would
fn read_sidecar(path: &Path) -> Result<Option<String>> {
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
        Err(e) => Err(Error::new(
            ErrorCode::InputUnreadable,
            format!("reading sidecar {}", path.display()),
        )
        .with_context(ErrorContext::default().with_input(path))
        .with_source(e)),
    }
}

/// Extract embedded subtitle stream `track` (subtitle-relative index) as SRT
/// text via ffmpeg. `what` names the source in errors — "subtitle track" for
/// an explicit `--subtitle-track`, "embedded subtitle stream" when
/// auto-detection picked it.
fn extract_embedded(input: &Path, track: usize, what: &str) -> Result<String> {
    let map_arg = format!("0:s:{track}");
    let mut cmd = Command::new("ffmpeg");
    cmd.arg("-i")
        .arg(input)
        .args(["-map", &map_arg, "-f", "srt", "pipe:1"]);

    let (child, stdout) = TrackedChild::spawn(
        &mut cmd,
        "ffmpeg for subtitle extraction",
        ErrorCode::FfmpegNotFound,
    )?;

    let text = read_capped(stdout)?;

    child.wait("ffmpeg subtitle extraction").map_err(|e| {
        let mut err = e.into_error(ErrorCode::FfmpegFailed);
        err.reason = format!("extracting {what} {track}: {}", err.reason);
        err
    })?;

    let text = text.trim().to_string();
    if text.is_empty() {
        return Err(
            Error::new(ErrorCode::InputCorrupt, format!("{what} {track} is empty"))
                .with_context(ErrorContext::default().with_input(input)),
        );
    }
    Ok(text)
}

/// Parse SRT text into a list of cues.
fn parse_srt(srt: &str) -> Vec<Cue> {
    // Windows subtitle editors commonly write a UTF-8 BOM. It is not
    // whitespace, so left in place it makes the first index line fail the
    // digit check and silently drops the first cue.
    let srt = srt.trim_start_matches('\u{feff}');
    let mut cues = Vec::new();
    let mut lines = srt.lines();

    while let Some(line) = lines.next() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        // The numeric cue index is optional: some tools emit cues that start
        // directly with the timestamp line, and requiring an index would
        // silently parse such files to zero cues.
        let ts_line = if trimmed.contains("-->") {
            trimmed
        } else if trimmed.chars().all(|c| c.is_ascii_digit()) {
            match lines.next() {
                Some(l) => l,
                None => break,
            }
        } else {
            continue;
        };

        // Parse timestamp line: "00:00:01,234 --> 00:00:04,567"
        let Some((start, end)) = parse_timestamp_line(ts_line) else {
            continue;
        };

        // Collect text lines until blank line
        let mut text_lines = Vec::new();
        for line in lines.by_ref() {
            if line.trim().is_empty() {
                break;
            }
            text_lines.push(line);
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

/// Parse a cue timing line `start --> end [settings…]` into (start_secs,
/// end_secs). Settings after the end time — SubRip `X1:…` coordinates
/// (danmaku converters), WebVTT `position:…`/`align:…` (CC downloaders) —
/// are ignored, not treated as part of the timestamp.
fn parse_timestamp_line(line: &str) -> Option<(f32, f32)> {
    let (lhs, rhs) = line.split_once("-->")?;
    Some((parse_srt_time(lhs)?, parse_srt_time(rhs)?))
}

/// Parse the leading `[HH:]MM:SS[,.]mmm` token of `s` into seconds; anything
/// after the first whitespace is the caller's settings to discard. The hours
/// group is optional (WebVTT permits `MM:SS.mmm`).
fn parse_srt_time(s: &str) -> Option<f32> {
    let token = s.split_whitespace().next()?.replace(',', ".");
    let mut parts = token.split(':').rev();
    let secs: f32 = parts.next()?.parse().ok()?;
    let mins: f32 = parts.next()?.parse().ok()?;
    let hours: f32 = match parts.next() {
        Some(h) => h.parse().ok()?,
        None => 0.0,
    };
    if parts.next().is_some() {
        return None;
    }
    Some(hours * 3600.0 + mins * 60.0 + secs)
}

/// Extract subtitles that overlap with a time range [start, end].
fn slice_for_range(cues: &[Cue], start: f32, end: f32) -> Option<String> {
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
    fn parses_srt_without_cue_indexes() {
        // Some tools emit SRT whose cues start directly with the timestamp
        // line; the numeric index is optional.
        let srt = "00:00:01,000 --> 00:00:03,500\nHello world\n\n\
                   00:00:04,000 --> 00:00:06,000\nSecond\n";
        let cues = parse_srt(srt);
        assert_eq!(cues.len(), 2, "index-less cues must not be dropped");
        assert_eq!(cues[0].text, "Hello world");
        assert!((cues[1].start - 4.0).abs() < 0.01);
    }

    #[test]
    fn parses_timing_line_with_subrip_coordinates() {
        // Danmaku-to-SRT converters position each cue with SubRip coordinate
        // extensions on the timing line; they are settings, not timestamp.
        let srt = "1\n00:00:01,120 --> 00:00:05,120 X1:120 Y1:50 X2:320 Y2:50\nbullet\n";
        let cues = parse_srt(srt);
        assert_eq!(cues.len(), 1, "coordinates must not break the cue");
        assert!((cues[0].start - 1.12).abs() < 0.01);
        assert!((cues[0].end - 5.12).abs() < 0.01);
    }

    #[test]
    fn parses_timing_line_with_vtt_settings() {
        // CC downloaders keep WebVTT cue settings when writing SRT.
        let (start, end) =
            parse_timestamp_line("00:00:01.000 --> 00:00:04.000 position:10% align:middle")
                .expect("settings after the end time are ignored");
        assert!((start - 1.0).abs() < 0.01);
        assert!((end - 4.0).abs() < 0.01);
    }

    #[test]
    fn parses_time_without_hours_group() {
        // WebVTT permits MM:SS.mmm.
        let secs = parse_srt_time("01:23.500").expect("two-group time is valid");
        assert!((secs - 83.5).abs() < 0.01);
    }

    #[test]
    fn rejects_malformed_time_token() {
        assert_eq!(parse_srt_time("00:00:01junk"), None);
        assert_eq!(parse_srt_time("1:2:3:4,000"), None, "too many groups");
        assert_eq!(parse_srt_time(""), None);
    }

    #[test]
    fn parses_srt_with_utf8_bom() {
        let srt = "\u{feff}1\n00:00:01,000 --> 00:00:02,000\nFirst cue\n\n\
                   2\n00:00:03,000 --> 00:00:04,000\nSecond\n";
        let cues = parse_srt(srt);
        assert_eq!(cues.len(), 2, "BOM must not swallow the first cue");
        assert_eq!(cues[0].text, "First cue");
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
    fn rejects_subtitle_source_with_no_parseable_cues() {
        // A readable file whose content parses to zero cues must fail the
        // run, not resolve into silently empty per-grid subtitles.
        let dir = std::env::temp_dir().join(format!("agent-video-nocue-{}", std::process::id()));
        fs::create_dir_all(&dir).expect("create test dir");
        let sub = dir.join("broken.srt");
        fs::write(&sub, "this is not SRT\njust prose\n").expect("write test subtitle");

        let err = Subtitles::resolve(Path::new("video.mp4"), SubtitleSource::File(&sub))
            .expect_err("unparseable subtitle must fail");
        assert_eq!(err.code, ErrorCode::InputCorrupt);

        fs::remove_dir_all(&dir).expect("cleanup test dir");
    }

    #[test]
    fn accepts_subtitle_at_exact_byte_cap() {
        // Exactly MAX_SUBTITLE_BYTES is within the cap, not a runaway stream.
        let data = vec![b'a'; MAX_SUBTITLE_BYTES as usize];
        let text = read_capped(&data[..]).expect("exact-cap input is acceptable");
        assert_eq!(text.len() as u64, MAX_SUBTITLE_BYTES);
    }

    #[test]
    fn rejects_subtitle_past_byte_cap() {
        let data = vec![b'a'; MAX_SUBTITLE_BYTES as usize + 1];
        let err = read_capped(&data[..]).expect_err("over-cap input is runaway");
        assert_eq!(err.code, ErrorCode::InputCorrupt);
    }

    #[test]
    fn missing_explicit_subtitle_maps_to_input_not_found() {
        let err = extract_from_file(Path::new("definitely_missing_subtitle.srt"))
            .expect_err("missing subtitle file must fail");
        assert_eq!(err.code, ErrorCode::InputNotFound);
    }

    #[test]
    fn maps_unreadable_sidecar_to_coded_error() {
        // A directory cannot be read as a file and is not NotFound: the
        // auto-detect path must fail with the same structured code an explicit
        // --subtitle path gets, not an uncoded exit-1 error.
        let err = read_sidecar(Path::new(".")).expect_err("directory read must fail");
        assert_eq!(err.code, ErrorCode::InputUnreadable);
    }

    /// Run `f` with a fresh temp dir containing `files`, cleaning up after.
    fn with_sidecar_dir(tag: &str, files: &[&str], f: impl FnOnce(&Path)) {
        let dir =
            std::env::temp_dir().join(format!("agent-video-sidecar-{tag}-{}", std::process::id()));
        fs::create_dir_all(&dir).expect("create test dir");
        for name in files {
            fs::write(dir.join(name), b"1\n00:00:01,000 --> 00:00:02,000\nx\n")
                .expect("write test sidecar");
        }
        f(&dir);
        fs::remove_dir_all(&dir).expect("cleanup test dir");
    }

    #[test]
    fn rejects_prefix_named_sibling_as_sidecar() {
        // `video2.srt` shares the prefix but is not `video.srt` or
        // `video.<tag>.srt`; claiming it would attach another video's subtitles.
        with_sidecar_dir("prefix", &["video2.srt"], |dir| {
            let found = find_all_sidecars(&dir.join("video.mp4"));
            assert!(found.is_empty(), "video2.srt is not a sidecar: {found:?}");
        });
    }

    #[test]
    fn finds_exact_and_language_tagged_sidecars() {
        with_sidecar_dir("tagged", &["video.srt", "video.zh.srt"], |dir| {
            let found = find_all_sidecars(&dir.join("video.mp4"));
            let mut names: Vec<_> = found
                .iter()
                .map(|p| {
                    p.file_name()
                        .expect("file name")
                        .to_string_lossy()
                        .into_owned()
                })
                .collect();
            names.sort();
            assert_eq!(names, ["video.srt", "video.zh.srt"]);
        });
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
