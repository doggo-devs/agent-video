//! Embedded subtitle extraction via the in-process subtitle decoder.
//!
//! Replaces two subprocess round-trips (an ffprobe stream scan and an ffmpeg
//! SRT transcode): tracks are listed from the opened container's stream
//! table, and cues are decoded directly from the subtitle stream's packets.
//! Text-based codecs (SubRip, ASS/SSA, mov_text, WebVTT) decode to ASS event
//! lines, which are stripped down to plain cue text; bitmap subtitle tracks
//! (DVD, PGS) carry no text and are rejected with an input-decade error, the
//! same way the old `-f srt` transcode failed on them.

use std::path::Path;

use ffmpeg::codec::Context as CodecContext;
use ffmpeg::codec::subtitle::Rect;
use ffmpeg::{Packet, Subtitle as AvSubtitle};
use ffmpeg_next as ffmpeg;

use crate::error::{Error, ErrorCode, ErrorContext, Result};
use crate::subtitle::Cue;

use super::{MAX_READ_ERRORS, decode_error, open_input};

/// Embedded subtitle track info, in subtitle-relative index order.
#[derive(Debug)]
pub struct EmbeddedTrack {
    pub index: usize,
    pub language: Option<String>,
}

/// List the container's subtitle streams with their language tags.
pub fn list_tracks(input: &Path) -> Result<Vec<EmbeddedTrack>> {
    let ictx = open_input(input)?;
    Ok(ictx
        .streams()
        .filter(|s| {
            CodecContext::from_parameters(s.parameters())
                .map(|c| c.medium() == ffmpeg::media::Type::Subtitle)
                .unwrap_or(false)
        })
        .enumerate()
        .map(|(i, s)| EmbeddedTrack {
            index: i,
            language: s
                .metadata()
                .get("language")
                .map(str::trim)
                .filter(|l| !l.is_empty())
                .map(String::from),
        })
        .collect())
}

/// Decode subtitle track `track` (subtitle-relative index) into cues.
///
/// `what` names the source in errors — "subtitle track" for an explicit
/// `--subtitle-track`, "embedded subtitle stream" when auto-detection picked
/// it.
pub fn extract_track(input: &Path, track: usize, what: &str) -> Result<Vec<Cue>> {
    let mut ictx = open_input(input)?;

    // Resolve the subtitle-relative index to a global stream index.
    let (stream_index, time_base, parameters) = {
        let mut subtitle_streams = ictx.streams().filter(|s| {
            CodecContext::from_parameters(s.parameters())
                .map(|c| c.medium() == ffmpeg::media::Type::Subtitle)
                .unwrap_or(false)
        });
        let stream = subtitle_streams.nth(track).ok_or_else(|| {
            Error::new(
                ErrorCode::InputCorrupt,
                format!("{what} {track} does not exist in {}", input.display()),
            )
            .with_context(ErrorContext::default().with_input(input))
        })?;
        let tb = stream.time_base();
        (
            stream.index(),
            tb.numerator() as f64 / tb.denominator() as f64,
            stream.parameters(),
        )
    };

    let codec_ctx = CodecContext::from_parameters(parameters)
        .map_err(|e| decode_error("reading subtitle codec parameters", e).with_input(input))?;
    let mut decoder = codec_ctx
        .decoder()
        .subtitle()
        .map_err(|e| decode_error(&format!("opening {what} decoder"), e).with_input(input))?;

    let mut cues = Vec::new();
    let mut packet = Packet::empty();
    let mut read_errors = 0u32;
    loop {
        match packet.read(&mut ictx) {
            Ok(()) => read_errors = 0,
            Err(ffmpeg::Error::Eof) => break,
            Err(e) => {
                // Transient demux errors are skipped, but a permanently
                // failing input must not spin forever — same cap as the
                // frame-decode loop.
                read_errors += 1;
                if read_errors > MAX_READ_ERRORS {
                    return Err(decode_error("reading subtitle packets", e).with_input(input));
                }
                continue;
            }
        }
        if packet.stream() != stream_index {
            continue;
        }

        let mut sub = AvSubtitle::new();
        // A packet that fails to decode or decodes to nothing is skipped;
        // wholesale failure surfaces as zero cues below.
        let Ok(true) = decoder.decode(&packet, &mut sub) else {
            continue;
        };

        let Some(start) = packet.pts().or(packet.dts()) else {
            continue;
        };
        let start = (start as f64 * time_base) as f32;
        // AVSubtitle display times are ms offsets from the packet pts; fall
        // back to the packet duration when the codec does not set them.
        let end = if sub.end() > sub.start() {
            start + (sub.end() - sub.start()) as f32 / 1000.0
        } else {
            start + (packet.duration() as f64 * time_base) as f32
        };
        let start = start + sub.start() as f32 / 1000.0;

        let mut lines: Vec<String> = Vec::new();
        for rect in sub.rects() {
            match rect {
                Rect::Text(t) => lines.push(t.get().trim().to_string()),
                Rect::Ass(a) => lines.push(ass_event_text(a.get())),
                Rect::Bitmap(_) => {
                    return Err(Error::new(
                        ErrorCode::InputCorrupt,
                        format!(
                            "{what} {track} is a bitmap subtitle track (no text to extract); \
                             provide --subtitle with a text file instead"
                        ),
                    )
                    .with_context(ErrorContext::default().with_input(input)));
                }
                Rect::None(_) => {}
            }
        }
        let text = lines.join(" ").trim().to_string();
        if !text.is_empty() && end > start {
            cues.push(Cue { start, end, text });
        }
    }

    if cues.is_empty() {
        return Err(Error::new(
            ErrorCode::InputCorrupt,
            format!("{what} {track} contains no text cues"),
        )
        .with_context(ErrorContext::default().with_input(input)));
    }
    Ok(cues)
}

/// Extract the plain text from a decoded ASS event line.
///
/// Modern libavcodec subtitle decoders emit events as
/// `ReadOrder,Layer,Style,Name,MarginL,MarginR,MarginV,Effect,Text` (9
/// fields); older versions prefix `Dialogue:` and carry Start/End timing
/// instead of ReadOrder, for 10 fields total. Only the Text field matters;
/// ASS override blocks `{\...}` are dropped and `\N`/`\n` soft breaks become
/// spaces.
fn ass_event_text(event: &str) -> String {
    let event = event.trim();
    let (event, fields) = match event.strip_prefix("Dialogue:") {
        Some(rest) => (rest.trim(), 10),
        None => (event, 9),
    };
    let text = event.splitn(fields, ',').nth(fields - 1).unwrap_or(event);

    let mut out = String::with_capacity(text.len());
    let mut in_override = false;
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '{' => in_override = true,
            '}' if in_override => in_override = false,
            _ if in_override => {}
            '\\' => match chars.peek() {
                Some('N') | Some('n') | Some('h') => {
                    chars.next();
                    if !out.ends_with(' ') && !out.is_empty() {
                        out.push(' ');
                    }
                }
                _ => out.push(c),
            },
            _ => out.push(c),
        }
    }
    out.trim().to_string()
}

/// Render cues back to SRT text — the `subtitles.txt` payload for embedded
/// sources, which no longer pass through an intermediate SRT transcode.
pub fn cues_to_srt(cues: &[Cue]) -> String {
    use std::fmt::Write;

    let mut srt = String::new();
    for (i, cue) in cues.iter().enumerate() {
        let _ = write!(
            srt,
            "{}\n{} --> {}\n{}\n\n",
            i + 1,
            srt_timestamp(cue.start),
            srt_timestamp(cue.end),
            cue.text
        );
    }
    srt
}

fn srt_timestamp(secs: f32) -> String {
    let ms_total = (secs.max(0.0) as f64 * 1000.0).round() as u64;
    let (h, rem) = (ms_total / 3_600_000, ms_total % 3_600_000);
    let (m, rem) = (rem / 60_000, rem % 60_000);
    let (s, ms) = (rem / 1000, rem % 1000);
    format!("{h:02}:{m:02}:{s:02},{ms:03}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strips_ass_event_down_to_text() {
        assert_eq!(
            ass_event_text("1,0,Default,,0,0,0,,{\\an8}Hello\\Nworld"),
            "Hello world"
        );
        assert_eq!(
            ass_event_text("Dialogue: 0,0:00:01.00,0:00:02.00,Default,,0,0,0,,Plain line"),
            "Plain line"
        );
    }

    #[test]
    fn keeps_commas_inside_the_text_field() {
        assert_eq!(
            ass_event_text("1,0,Default,,0,0,0,,one, two, three"),
            "one, two, three"
        );
    }

    #[test]
    fn formats_srt_timestamps() {
        assert_eq!(srt_timestamp(0.0), "00:00:00,000");
        assert_eq!(srt_timestamp(83.5), "00:01:23,500");
        assert_eq!(srt_timestamp(3661.007), "01:01:01,007");
    }

    #[test]
    fn renders_cues_as_srt() {
        let cues = vec![Cue {
            start: 1.0,
            end: 2.5,
            text: "hi".into(),
        }];
        assert_eq!(
            cues_to_srt(&cues),
            "1\n00:00:01,000 --> 00:00:02,500\nhi\n\n"
        );
    }
}
