// SPDX-License-Identifier: Apache-2.0
//! Subtitle resolution: embedded streams via the in-process decoder, or
//! external sidecar files.
//!
//! [`Subtitles`] is the single boundary of the subtitle domain: source
//! resolution (explicit path > explicit track > auto-detection), SRT parsing,
//! and time-range slicing all live behind it, and every failure leaves with a
//! structured [`Error`] code regardless of which source was used.
//!
//! This module also owns the SRT grammar in both directions: [`parse_srt`]
//! reads it and [`serialize_srt`] writes it. Every producer of SRT text in
//! the workspace — embedded-track rendering here, the harness's whisper
//! transcript writer — goes through [`serialize_srt`], so writer and reader
//! cannot drift apart.

use std::borrow::Cow;
use std::fs;
use std::path::Path;

use crate::error::{Error, ErrorCode, ErrorContext, Result};
use crate::event::{Discard, Event, EventSink};
use crate::media;
use crate::source::Source;
use crate::types::{GridSubtitles, TimeRange, format_ts};

/// A single parsed subtitle cue.
#[derive(Debug, Clone)]
pub struct Cue {
    pub start: f32,
    pub end: f32,
    pub text: String,
}

impl Cue {
    /// Build a cue clipped to the visible timeline `[0, ∞)`.
    ///
    /// The sole cue gate on every parse path — SRT text ([`parse_srt`]) and
    /// embedded tracks — so a parsed [`Cue`] has finite bounds with
    /// `0 <= start < end` and non-empty text by construction. Non-finite
    /// bounds yield `None`: NaN or infinity here is corrupt input, not a
    /// clampable time.
    ///
    /// Embedded streams legitimately place cues before the first frame
    /// (edit-list pre-roll, shifted tracks, priming cues): a cue ending at
    /// or before 0 yields `None`, one straddling 0 is clamped to start at 0.
    /// Empty text also yields `None`.
    pub fn visible(start: f64, end: f64, text: String) -> Option<Cue> {
        // Narrow first, so an f64 that overflows f32 becomes inf here and
        // fails the finiteness check with the other non-finite inputs; the
        // check precedes the clamp because NaN.max(0.0) would silently
        // launder NaN into 0.
        let (start, end) = (start as f32, end as f32);
        if !start.is_finite() || !end.is_finite() {
            return None;
        }
        let start = start.max(0.0);
        (end > start && !text.is_empty()).then_some(Cue { start, end, text })
    }
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

/// Resolved subtitles for one run: parsed cues plus the provenance of
/// their SRT text ([`Subtitles::srt_text`]).
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
    payload: SrtPayload,
    cues: Vec<Cue>,
}

/// Provenance of the `subtitles.txt` payload.
///
/// A file or sidecar source keeps its original text byte-faithfully; an
/// embedded track keeps only its cues and serializes SRT on demand — the
/// cues reside in memory exactly once for the life of a pipeline run,
/// instead of doubling into an equal-size string at construction.
#[derive(Debug)]
enum SrtPayload {
    /// The byte-faithful original SRT text of a file/sidecar source.
    Verbatim(String),
    /// Cues decoded from an embedded track; text derives on demand.
    Derived,
}

impl Subtitles {
    /// Resolve subtitles from the given source. Returns `Ok(None)` only on
    /// the [`SubtitleSource::Auto`] path: when no source exists, or when the
    /// single embedded track turns out to hold no text cues — a legal
    /// container state that degrades to running without subtitles. An
    /// *explicitly requested* source that yields no cues fails with
    /// [`ErrorCode::InputCorrupt`] instead: every grid's `subtitles` field
    /// would be empty while the run reports success — not what the user
    /// asked for.
    ///
    /// `events` receives the auto path's environmental warnings (an empty
    /// embedded track, an unlistable sidecar directory) as typed events; the
    /// explicit-source paths emit none. Pass [`crate::event::Discard`] for
    /// silence. `stop` bounds the embedded-track reads on a stalled source;
    /// see [`media::open_input`].
    pub fn resolve(
        input: &Source,
        source: SubtitleSource<'_>,
        stop: &media::Stop,
        events: &dyn EventSink,
    ) -> Result<Option<Self>> {
        match source {
            SubtitleSource::File(p) => Ok(Some(Self::from_srt_text(input, extract_from_file(p)?)?)),
            SubtitleSource::Track(t) => {
                let cues =
                    media::embedded_subtitle::extract_track(input, t, "subtitle track", stop)?;
                if cues.is_empty() {
                    return Err(Error::new(
                        ErrorCode::InputCorrupt,
                        format!("subtitle track {t} contains no text cues"),
                    )
                    .with_context(ErrorContext::default().with_input(input.as_path())));
                }
                Ok(Some(Self::from_cues(cues)))
            }
            SubtitleSource::Auto => match auto_detect(input, stop, events)? {
                Some(AutoSource::Sidecar(text)) => Ok(Some(Self::from_srt_text(input, text)?)),
                Some(AutoSource::Embedded(cues)) => Ok(Some(Self::from_cues(cues))),
                None => Ok(None),
            },
        }
    }

    /// Build from SRT text (explicit file or sidecar), keeping the raw text
    /// as the `subtitles.txt` payload.
    fn from_srt_text(input: &Source, raw: String) -> Result<Self> {
        let cues = parse_srt(&raw);
        if cues.is_empty() {
            return Err(Error::new(
                ErrorCode::InputCorrupt,
                "subtitle source contains no parseable SRT cues; is it a valid SRT file?",
            )
            .with_context(ErrorContext::default().with_input(input.as_path())));
        }
        Ok(Subtitles {
            payload: SrtPayload::Verbatim(raw),
            cues,
        })
    }

    /// Build from decoded embedded cues; the `subtitles.txt` payload
    /// derives from them on demand ([`Self::srt_text`]). Callers guarantee
    /// at least one cue — both resolution paths reject or degrade on an
    /// empty extraction first.
    fn from_cues(cues: Vec<Cue>) -> Self {
        Subtitles {
            payload: SrtPayload::Derived,
            cues,
        }
    }

    /// Every cue overlapping `processed`, assigned to exactly one grid: the
    /// one whose frame span lies nearest the cue's midpoint (ties go to the
    /// earlier grid). Returns one entry per span, in order; a grid that won
    /// no cue is [`GridSubtitles::NoCueInRange`], distinct by type from "the
    /// run had no source".
    ///
    /// Totality is the contract: grid spans cover picked frames, not the
    /// timeline, so overlap slicing would drop every cue in the gaps between
    /// grids. Nearest-span assignment has no gaps, and overlapping spans
    /// (static stretches straddle their neighbors) never duplicate a cue.
    ///
    /// Each entry renders as `MM:SS text` lines (cue start times), so the
    /// text stays addressable for follow-up `--from/--until` calls.
    pub fn assign(&self, processed: TimeRange, spans: &[TimeRange]) -> Vec<GridSubtitles> {
        let mut per_grid: Vec<Vec<&Cue>> = spans.iter().map(|_| Vec::new()).collect();
        for cue in &self.cues {
            if cue.end <= processed.start() || cue.start >= processed.end() {
                continue;
            }
            let mid = 0.5 * (cue.start + cue.end);
            if let Some(nearest) = nearest_span(mid, spans) {
                per_grid[nearest].push(cue);
            }
        }
        per_grid
            .into_iter()
            .map(|mut cues| {
                if cues.is_empty() {
                    return GridSubtitles::NoCueInRange;
                }
                cues.sort_by(|a, b| a.start.total_cmp(&b.start));
                GridSubtitles::Text(render_cue_lines(&cues))
            })
            .collect()
    }

    /// The full resolved SRT text — the `subtitles.txt` payload: the
    /// byte-faithful original for file and sidecar sources, or SRT
    /// serialized on demand from embedded cues. A derived payload lives
    /// only as long as the returned value, so embedded cues are never
    /// stored twice.
    pub fn srt_text(&self) -> Cow<'_, str> {
        match &self.payload {
            SrtPayload::Verbatim(raw) => Cow::Borrowed(raw.as_str()),
            SrtPayload::Derived => Cow::Owned(serialize_srt(self.cues.iter().map(SrtCue::from))),
        }
    }
}

/// What auto-detection resolved to: sidecar SRT text, or decoded embedded
/// cues.
enum AutoSource {
    Sidecar(String),
    Embedded(Vec<Cue>),
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
/// Returns `None` when no subtitles are found. Sidecar discovery is
/// local-only: a remote source has no directory to scan, so only its embedded
/// tracks are considered.
fn auto_detect(
    input: &Source,
    stop: &media::Stop,
    events: &dyn EventSink,
) -> Result<Option<AutoSource>> {
    let sidecars = match input {
        Source::Local(path) => find_sidecars_inner(path, events),
        Source::Remote(_) => Vec::new(),
    };
    let embedded = media::embedded_subtitle::list_tracks(input, stop)?;

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

    if let Some(path) = sidecars.into_iter().next() {
        return match read_sidecar(&path)? {
            Some(content) => Ok(Some(AutoSource::Sidecar(content))),
            None => Err(Error::new(
                ErrorCode::InputCorrupt,
                format!("sidecar file is empty: {}", path.display()),
            )),
        };
    }

    if embedded.len() == 1 {
        let cues =
            media::embedded_subtitle::extract_track(input, 0, "embedded subtitle stream", stop)?;
        if cues.is_empty() {
            // A present-but-empty text track is a legal container state, not
            // corruption; unrequested, the auto path degrades to no subtitles.
            events.event(Event::EmbeddedSubtitlesEmpty);
            return Ok(None);
        }
        return Ok(Some(AutoSource::Embedded(cues)));
    }

    Ok(None)
}

/// Sidecar subtitle extensions [`find_sidecars`] discovers.
///
/// This is the auto-detection contract: anything advertising "subtitles
/// available" for a file (the harness probe) must derive from this list,
/// not restate it — a format core cannot read must never be promised.
pub const SIDECAR_SUBTITLE_EXTS: &[&str] = &["srt"];

/// Find all sidecar subtitle files alongside the video: `video.<ext>` and
/// `video.*.<ext>` for every extension in [`SIDECAR_SUBTITLE_EXTS`].
///
/// Best-effort: it runs on every auto-detect, so an unlistable directory
/// degrades to whatever the exact-name check found; a sidecar that is found
/// but unreadable still fails in `read_sidecar`. Scan diagnostics are
/// discarded here; the pipeline uses `find_sidecars_inner` to keep them.
pub fn find_sidecars(input: &Path) -> Vec<std::path::PathBuf> {
    find_sidecars_inner(input, &Discard)
}

/// Sidecar discovery that reports scan diagnostics into `events`: an
/// unlistable directory or an unreadable entry becomes a typed event rather
/// than a hardcoded stderr line. [`find_sidecars`] is the silent wrapper.
fn find_sidecars_inner(input: &Path, events: &dyn EventSink) -> Vec<std::path::PathBuf> {
    let Some(stem) = input.file_stem().and_then(|s| s.to_str()) else {
        return Vec::new();
    };
    let Some(parent) = sidecar_dir(input) else {
        return Vec::new();
    };

    let mut results = Vec::new();

    // Direct stats, so exact names resolve even when the directory can't be listed.
    for ext in SIDECAR_SUBTITLE_EXTS {
        let exact = parent.join(format!("{stem}.{ext}"));
        if exact.is_file() {
            results.push(exact);
        }
    }

    let entries = match fs::read_dir(parent) {
        Ok(entries) => entries,
        Err(e) => {
            events.event(Event::SidecarScanFailed {
                dir: parent,
                error: e.to_string(),
            });
            return results;
        }
    };

    for entry in entries {
        let entry = match entry {
            Ok(entry) => entry,
            Err(e) => {
                events.event(Event::SidecarEntryUnreadable {
                    error: e.to_string(),
                });
                continue;
            }
        };
        let name = entry.file_name();
        let name = name.to_string_lossy();
        // Require a literal `.` after the stem (else `video2.srt` matches
        // `video.mp4`); `rest.len() > suffix` excludes the exact-name case above.
        let is_pattern_match = name.strip_prefix(stem).is_some_and(|rest| {
            rest.starts_with('.')
                && SIDECAR_SUBTITLE_EXTS.iter().any(|ext| {
                    let suffix = format!(".{ext}");
                    rest.ends_with(&suffix) && rest.len() > suffix.len()
                })
        });
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

/// One cue at SRT's native resolution — integral milliseconds.
///
/// `HH:MM:SS,mmm` carries exactly integral milliseconds, so this is the
/// canonical currency of [`serialize_srt`]: whisper's centisecond cues embed
/// exactly (×10), and the pipeline's f32-second [`Cue`]s round to the nearest
/// millisecond — lossy only below what the format itself can express.
/// Callers convert at this boundary; the grammar has one writer.
#[derive(Debug, Clone, Copy)]
pub struct SrtCue<'a> {
    pub start_ms: u64,
    pub end_ms: u64,
    pub text: &'a str,
}

impl<'a> From<&'a Cue> for SrtCue<'a> {
    /// Round f32 seconds to the nearest millisecond. Parsed cues are finite
    /// and non-negative by [`Cue::visible`]'s construction; a hand-built cue
    /// bypassing that gate saturates via the float-to-int cast (negatives to
    /// 0, infinity to `u64::MAX` ms) rather than wrapping.
    fn from(cue: &'a Cue) -> Self {
        SrtCue {
            start_ms: (f64::from(cue.start) * 1000.0).round() as u64,
            end_ms: (f64::from(cue.end) * 1000.0).round() as u64,
            text: &cue.text,
        }
    }
}

/// Serialize cues as SRT text: numbered blocks, `HH:MM:SS,mmm --> …` timing
/// lines with comma milliseconds, blank-line separated.
///
/// The inverse of [`parse_srt`] up to the parser's normalizations: text
/// containing newlines serializes as multiple cue lines and parses back
/// joined with a single space.
pub fn serialize_srt<'a>(cues: impl IntoIterator<Item = SrtCue<'a>>) -> String {
    use std::fmt::Write;

    let mut srt = String::new();
    for (i, cue) in cues.into_iter().enumerate() {
        let _ = write!(
            srt,
            "{}\n{} --> {}\n{}\n\n",
            i + 1,
            srt_timestamp(cue.start_ms),
            srt_timestamp(cue.end_ms),
            cue.text
        );
    }
    srt
}

/// `HH:MM:SS,mmm` from integral milliseconds.
fn srt_timestamp(ms: u64) -> String {
    let (h, rem) = (ms / 3_600_000, ms % 3_600_000);
    let (m, rem) = (rem / 60_000, rem % 60_000);
    let (s, ms) = (rem / 1000, rem % 1000);
    format!("{h:02}:{m:02}:{s:02},{ms:03}")
}

/// Parse SRT text into a list of cues.
///
/// The reader half of the SRT contract whose writer is [`serialize_srt`].
/// Lenient by design: the numeric cue index is optional, `.` and `,`
/// millisecond separators are both accepted, bare-seconds timing (`5 --> 10`)
/// parses, timing-line settings after the end time are ignored, and
/// multi-line cue text joins with a space. Every cue passes through
/// [`Cue::visible`], so non-finite, inverted, or zero-length timing drops
/// the cue rather than representing it.
pub fn parse_srt(srt: &str) -> Vec<Cue> {
    // Windows editors write a UTF-8 BOM; left in, it fails the first index
    // line's digit check and silently drops the first cue.
    let srt = srt.trim_start_matches('\u{feff}');
    let mut cues = Vec::new();
    let mut lines = srt.lines();

    while let Some(line) = lines.next() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        // The numeric cue index is optional: some tools emit cues starting
        // directly with the timestamp line.
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

        let Some((start, end)) = parse_timestamp_line(ts_line) else {
            continue;
        };

        let mut text_lines = Vec::new();
        for line in lines.by_ref() {
            if line.trim().is_empty() {
                break;
            }
            text_lines.push(line);
        }

        // Cue::visible is the single gate: empty text, inverted, and
        // zero-length cues all drop here instead of being representable.
        if let Some(cue) = Cue::visible(f64::from(start), f64::from(end), text_lines.join(" ")) {
            cues.push(cue);
        }
    }

    cues
}

/// Parse a cue timing line `start --> end [settings…]` into (start_secs,
/// end_secs). Each side's leading whitespace-delimited token is a timestamp
/// in the crate's shared clock grammar (`crate::types::parse_clock`); the
/// rest is settings to discard — SubRip `X1:…` coordinates (danmaku
/// converters), WebVTT `position:…`/`align:…` (CC downloaders) — not part
/// of the timestamp.
fn parse_timestamp_line(line: &str) -> Option<(f32, f32)> {
    let (lhs, rhs) = line.split_once("-->")?;
    let time = |s: &str| crate::types::parse_clock(s.split_whitespace().next()?);
    Some((time(lhs)?, time(rhs)?))
}

/// Index of the span nearest to instant `t` (distance 0 inside a span).
/// The first minimum wins, so assignment stays deterministic when spans
/// overlap; `None` only for an empty span list.
fn nearest_span(t: f32, spans: &[TimeRange]) -> Option<usize> {
    let distance = |s: &TimeRange| (s.start() - t).max(t - s.end()).max(0.0);
    let mut best: Option<(usize, f32)> = None;
    for (i, span) in spans.iter().enumerate() {
        let d = distance(span);
        if best.is_none_or(|(_, held)| d < held) {
            best = Some((i, d));
        }
    }
    best.map(|(i, _)| i)
}

/// `MM:SS text` per cue, newline-joined — the grid subtitle payload.
fn render_cue_lines(cues: &[&Cue]) -> String {
    cues.iter()
        .map(|c| format!("{} {}", format_ts(c.start), c.text))
        .collect::<Vec<_>>()
        .join("\n")
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
        let (start, _) =
            parse_timestamp_line("01:23.500 --> 01:25.000").expect("two-group time is valid");
        assert!((start - 83.5).abs() < 0.01);
    }

    #[test]
    fn rejects_malformed_time_token() {
        assert_eq!(parse_timestamp_line("00:00:01junk --> 00:00:02,000"), None);
        assert_eq!(
            parse_timestamp_line("1:2:3:4,000 --> 00:00:05,000"),
            None,
            "too many groups"
        );
        assert_eq!(parse_timestamp_line(" --> 00:00:02,000"), None);
    }

    #[test]
    fn rejects_nonfinite_srt_time() {
        // f32::parse happily reads "inf" and "nan"; the shared clock grammar
        // must not let them become cue bounds.
        let cues = parse_srt("1\ninf:00 --> nan:30\nghost\n");
        assert!(cues.is_empty(), "non-finite timing must drop the cue");
    }

    #[test]
    fn rejects_negative_srt_time_component() {
        let cues = parse_srt("1\n0:-30,000 --> 0:40,000\nearly\n");
        assert!(cues.is_empty(), "negative component must drop the cue");
    }

    #[test]
    fn drops_inverted_srt_cue() {
        // Routing parse_srt through Cue::visible makes inverted (and
        // zero-length) cues unrepresentable, not merely unlikely.
        let cues = parse_srt("1\n00:00:05,000 --> 00:00:01,000\nbackwards\n");
        assert!(cues.is_empty(), "inverted timing must drop the cue");
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
    fn drops_cue_ending_before_zero() {
        // A broadcast priming cue lives entirely before the first video
        // frame; it is initialization, not content.
        assert!(Cue::visible(-0.5, -0.1, "prime".into()).is_none());
        assert!(Cue::visible(-0.5, 0.0, "prime".into()).is_none());
    }

    #[test]
    fn clamps_cue_straddling_zero() {
        // Edit-list pre-roll: the cue must be on screen at the first frame,
        // so its visible interval starts at 0.
        let cue = Cue::visible(-0.2, 1.0, "early".into()).expect("straddling cue is visible");
        assert_eq!(cue.start, 0.0);
        assert!((cue.end - 1.0).abs() < 1e-6);
    }

    #[test]
    fn rejects_empty_or_inverted_cue() {
        assert!(Cue::visible(1.0, 2.0, String::new()).is_none());
        assert!(Cue::visible(2.0, 1.0, "inverted".into()).is_none());
    }

    #[test]
    fn rejects_nonfinite_cue_bounds() {
        assert!(Cue::visible(f64::NAN, 1.0, "x".into()).is_none());
        assert!(Cue::visible(0.0, f64::NAN, "x".into()).is_none());
        assert!(Cue::visible(0.0, f64::INFINITY, "x".into()).is_none());
        assert!(Cue::visible(f64::NEG_INFINITY, 1.0, "x".into()).is_none());
        // An f64 finite value past f32::MAX becomes inf on narrowing and
        // must be caught by the same check.
        assert!(Cue::visible(0.0, f64::MAX, "x".into()).is_none());
    }

    fn cue(start: f32, end: f32, text: &str) -> Cue {
        Cue {
            start,
            end,
            text: text.into(),
        }
    }

    fn subs(cues: Vec<Cue>) -> Subtitles {
        Subtitles {
            payload: SrtPayload::Derived,
            cues,
        }
    }

    #[test]
    fn derives_srt_text_from_cues_on_demand() {
        let s = subs(vec![cue(1.0, 2.5, "hi")]);
        let text = s.srt_text();
        assert!(
            matches!(text, Cow::Owned(_)),
            "embedded cues serialize on demand"
        );
        assert_eq!(text, serialize_srt(s.cues.iter().map(SrtCue::from)));
    }

    #[test]
    fn borrows_verbatim_srt_text_for_file_sources() {
        let raw = "1\n00:00:01,000 --> 00:00:02,000\nHello\n\n";
        let s = Subtitles::from_srt_text(&Source::classify("video.mp4"), raw.to_string())
            .expect("a valid SRT parses");
        let text = s.srt_text();
        assert!(
            matches!(text, Cow::Borrowed(_)),
            "a file source's payload borrows the original"
        );
        assert_eq!(text, raw, "byte-faithful to the source text");
    }

    fn range(start: f32, end: f32) -> TimeRange {
        TimeRange::new(start, end).expect("valid test range")
    }

    #[test]
    fn assigns_every_cue_despite_sampling_gaps() {
        // Grid spans cover frames at 6-58s and 180-186s of a 188s video:
        // cues in the head (0-6), the inter-grid gap (58-180 midpointed),
        // and the tail (186-188) all still land somewhere.
        let s = subs(vec![
            cue(0.0, 4.0, "head"),
            cue(30.0, 33.0, "inside first"),
            cue(100.0, 104.0, "gap, nearer first"),
            cue(170.0, 175.0, "gap, nearer last"),
            cue(186.5, 188.0, "tail"),
        ]);
        let assigned = s.assign(range(0.0, 188.0), &[range(6.0, 58.0), range(180.0, 186.0)]);
        assert_eq!(
            assigned,
            vec![
                GridSubtitles::Text(
                    "00:00 head\n00:30 inside first\n01:40 gap, nearer first".into()
                ),
                GridSubtitles::Text("02:50 gap, nearer last\n03:07 tail".into()),
            ]
        );
    }

    #[test]
    fn assigns_boundary_cue_to_exactly_one_grid() {
        // A cue straddling two adjacent spans must not appear in both.
        let s = subs(vec![cue(9.0, 11.0, "straddler")]);
        let assigned = s.assign(range(0.0, 20.0), &[range(0.0, 10.0), range(10.0, 20.0)]);
        assert_eq!(
            assigned,
            vec![
                GridSubtitles::Text("00:09 straddler".into()),
                GridSubtitles::NoCueInRange,
            ]
        );
    }

    #[test]
    fn excludes_cues_outside_the_processed_range() {
        // A --from/--until run must not adopt cues it never attended to.
        let s = subs(vec![
            cue(1.0, 2.0, "before"),
            cue(65.0, 68.0, "inside"),
            cue(200.0, 204.0, "after"),
        ]);
        let assigned = s.assign(range(60.0, 120.0), &[range(62.0, 118.0)]);
        assert_eq!(assigned, vec![GridSubtitles::Text("01:05 inside".into())]);
    }

    #[test]
    fn ties_between_overlapping_spans_go_to_the_earlier_grid() {
        // A static stretch's span overlaps its neighbor; a cue inside both
        // has distance 0 to each and must resolve deterministically.
        let s = subs(vec![cue(5.0, 6.0, "shared")]);
        let assigned = s.assign(range(0.0, 20.0), &[range(0.0, 10.0), range(4.0, 20.0)]);
        assert_eq!(
            assigned,
            vec![
                GridSubtitles::Text("00:05 shared".into()),
                GridSubtitles::NoCueInRange,
            ]
        );
    }

    #[test]
    fn renders_cues_as_timestamped_lines_in_time_order() {
        let s = subs(vec![cue(83.0, 85.0, "later"), cue(1.0, 2.0, "earlier")]);
        let assigned = s.assign(range(0.0, 90.0), &[range(0.0, 90.0)]);
        assert_eq!(
            assigned,
            vec![GridSubtitles::Text("00:01 earlier\n01:23 later".into())]
        );
    }

    #[test]
    fn formats_srt_timestamps() {
        assert_eq!(srt_timestamp(0), "00:00:00,000");
        assert_eq!(srt_timestamp(83_500), "00:01:23,500");
        assert_eq!(srt_timestamp(3_661_007), "01:01:01,007");
    }

    #[test]
    fn serializes_numbered_srt_blocks() {
        let cues = [
            SrtCue {
                start_ms: 1_000,
                end_ms: 2_500,
                text: "hi",
            },
            SrtCue {
                start_ms: 3_000,
                end_ms: 4_000,
                text: "there",
            },
        ];
        assert_eq!(
            serialize_srt(cues),
            "1\n00:00:01,000 --> 00:00:02,500\nhi\n\n\
             2\n00:00:03,000 --> 00:00:04,000\nthere\n\n"
        );
    }

    /// Serialize, parse back, and assert cue-for-cue equality within the
    /// format's millisecond resolution.
    fn assert_round_trips(cues: &[Cue]) {
        let srt = serialize_srt(cues.iter().map(SrtCue::from));
        let parsed = parse_srt(&srt);
        assert_eq!(parsed.len(), cues.len(), "cue count survives:\n{srt}");
        for (a, b) in cues.iter().zip(&parsed) {
            assert!(
                (a.start - b.start).abs() < 0.001,
                "{} vs {}",
                a.start,
                b.start
            );
            assert!((a.end - b.end).abs() < 0.001, "{} vs {}", a.end, b.end);
            assert_eq!(a.text, b.text);
        }
    }

    #[test]
    fn round_trips_comma_millisecond_times() {
        assert_round_trips(&[Cue {
            start: 1.007,
            end: 2.503,
            text: "precise".into(),
        }]);
    }

    #[test]
    fn round_trips_across_hour_rollover() {
        assert_round_trips(&[Cue {
            start: 3599.5,
            end: 3600.5,
            text: "straddles the hour".into(),
        }]);
    }

    #[test]
    fn round_trips_blank_line_separated_cues() {
        assert_round_trips(&[
            Cue {
                start: 1.0,
                end: 2.0,
                text: "first".into(),
            },
            Cue {
                start: 3.0,
                end: 4.0,
                text: "second".into(),
            },
        ]);
    }

    #[test]
    fn round_trips_multi_line_text_as_space_joined() {
        // Text with a newline serializes as two cue lines; the parser's
        // normalization joins them with a space — the documented lossy edge.
        let srt = serialize_srt([SrtCue {
            start_ms: 1_000,
            end_ms: 2_000,
            text: "line one\nline two",
        }]);
        let parsed = parse_srt(&srt);
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0].text, "line one line two");
    }

    #[test]
    fn rejects_subtitle_source_with_no_parseable_cues() {
        let dir = std::env::temp_dir().join(format!("agent-video-nocue-{}", std::process::id()));
        fs::create_dir_all(&dir).expect("create test dir");
        let sub = dir.join("broken.srt");
        fs::write(&sub, "this is not SRT\njust prose\n").expect("write test subtitle");

        let err = Subtitles::resolve(
            &Source::classify("video.mp4"),
            SubtitleSource::File(&sub),
            &media::Stop::new(),
            &Discard,
        )
        .expect_err("unparseable subtitle must fail");
        assert_eq!(err.code, ErrorCode::InputCorrupt);

        fs::remove_dir_all(&dir).expect("cleanup test dir");
    }

    #[test]
    fn rejects_sidecar_with_only_nonfinite_cues() {
        // The finding's trigger surface end-to-end: a hostile subtitle file
        // whose only cue carries non-finite timing must fail as corrupt, not
        // ride NaN bounds into assign().
        let dir = std::env::temp_dir().join(format!("agent-video-nonfin-{}", std::process::id()));
        fs::create_dir_all(&dir).expect("create test dir");
        let sub = dir.join("hostile.srt");
        fs::write(&sub, "1\ninf:00 --> nan:30\nghost\n").expect("write test subtitle");

        let err = Subtitles::resolve(
            &Source::classify("video.mp4"),
            SubtitleSource::File(&sub),
            &media::Stop::new(),
            &Discard,
        )
        .expect_err("non-finite-only subtitle must fail");
        assert_eq!(err.code, ErrorCode::InputCorrupt);

        fs::remove_dir_all(&dir).expect("cleanup test dir");
    }

    #[test]
    fn missing_explicit_subtitle_maps_to_input_not_found() {
        let err = extract_from_file(Path::new("definitely_missing_subtitle.srt"))
            .expect_err("missing subtitle file must fail");
        assert_eq!(err.code, ErrorCode::InputNotFound);
    }

    #[test]
    fn maps_unreadable_sidecar_to_coded_error() {
        // A directory is neither a readable file nor NotFound; it must still
        // map to a structured code, not an uncoded exit-1.
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
            let found = find_sidecars(&dir.join("video.mp4"));
            assert!(found.is_empty(), "video2.srt is not a sidecar: {found:?}");
        });
    }

    #[test]
    fn finds_exact_and_language_tagged_sidecars() {
        with_sidecar_dir("tagged", &["video.srt", "video.zh.srt"], |dir| {
            let found = find_sidecars(&dir.join("video.mp4"));
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
        // A bare filename's parent is the empty path; scan `.` instead.
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
