// SPDX-License-Identifier: Apache-2.0
//! Embedded subtitle extraction via the in-process subtitle decoder.
//!
//! Tracks are listed from the opened container's stream table, and cues are
//! decoded directly from the subtitle stream's packets. Text-based codecs
//! (SubRip, ASS/SSA, mov_text, WebVTT) decode to ASS event lines, which are
//! stripped down to plain cue text; bitmap subtitle tracks (DVD, PGS) carry
//! no text and are rejected with an input-decade error.

use ffmpeg::codec::Context as CodecContext;
use ffmpeg::codec::subtitle::Rect;
use ffmpeg::{Packet, Subtitle as AvSubtitle};
use ffmpeg_next as ffmpeg;

use crate::error::{Error, ErrorCode, ErrorContext, Result};
use crate::source::Source;
use crate::subtitle::Cue;

use super::{ErrorBudget, Timeline, decode_error, open_input};

/// Embedded subtitle track info, in subtitle-relative index order.
#[derive(Debug)]
pub struct EmbeddedTrack {
    pub index: usize,
    pub language: Option<String>,
}

/// List the container's subtitle streams with their language tags.
pub fn list_tracks(input: &Source) -> Result<Vec<EmbeddedTrack>> {
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
///
/// The returned list may be empty: a present-but-empty text track is a legal
/// container state, and whether emptiness is fatal is the caller's policy
/// (an explicitly requested track fails the run; auto-detection degrades to
/// running without subtitles).
pub fn extract_track(input: &Source, track: usize, what: &str) -> Result<Vec<Cue>> {
    let mut ictx = open_input(input)?;

    // Cue times must land on the same 0-based clock the video frames use,
    // or grid time-range slicing would systematically misattribute text on
    // containers whose streams do not start at 0 (broadcast TS captures).
    let timeline = Timeline::from_input(&ictx);

    let (stream_index, time_base, parameters) = {
        let mut subtitle_streams = ictx.streams().filter(|s| {
            CodecContext::from_parameters(s.parameters())
                .map(|c| c.medium() == ffmpeg::media::Type::Subtitle)
                .unwrap_or(false)
        });
        let stream = subtitle_streams.nth(track).ok_or_else(|| {
            Error::new(
                ErrorCode::InputCorrupt,
                format!("{what} {track} does not exist in {input}"),
            )
            .with_context(ErrorContext::default().with_input(input.as_path()))
        })?;
        let tb = stream.time_base();
        (
            stream.index(),
            tb.numerator() as f64 / tb.denominator() as f64,
            stream.parameters(),
        )
    };

    // Learn the field count before the decoder consumes the parameters;
    // non-ASS tracks fall back to the canonical libavcodec layout.
    let schema = AssSchema::from_extradata(codec_extradata(&parameters));

    let codec_ctx = CodecContext::from_parameters(parameters).map_err(|e| {
        decode_error("reading subtitle codec parameters", e).with_input(input.as_path())
    })?;
    let mut decoder = codec_ctx.decoder().subtitle().map_err(|e| {
        decode_error(&format!("opening {what} decoder"), e).with_input(input.as_path())
    })?;

    let mut cues = Vec::new();
    let mut packet = Packet::empty();
    let mut read_budget = ErrorBudget::new("reading subtitle packets");
    loop {
        match packet.read(&mut ictx) {
            Ok(()) => read_budget.ok(),
            Err(ffmpeg::Error::Eof) => break,
            Err(e) => {
                read_budget
                    .fail(e)
                    .map_err(|err| err.with_input(input.as_path()))?;
                continue;
            }
        }
        if packet.stream() != stream_index {
            continue;
        }

        let mut sub = AvSubtitle::new();
        let Ok(true) = decoder.decode(&packet, &mut sub) else {
            continue;
        };

        let Some(pts) = packet.pts().or(packet.dts()) else {
            continue;
        };
        let packet_start = timeline.seconds(pts, time_base);
        let declared = declared_duration(packet.duration(), time_base);
        let (start, end) = cue_window(packet_start, sub.start(), sub.end(), declared);

        let mut lines: Vec<String> = Vec::new();
        for rect in sub.rects() {
            match rect {
                Rect::Text(t) => lines.push(t.get().trim().to_string()),
                Rect::Ass(a) => match schema.classify(a.get()) {
                    AssLine::Event(text) => lines.push(strip_ass_markup(text)),
                    AssLine::PlainText(text) => lines.push(text.to_string()),
                    AssLine::Malformed => {}
                },
                Rect::Bitmap(_) => {
                    return Err(Error::new(
                        ErrorCode::InputCorrupt,
                        format!(
                            "{what} {track} is a bitmap subtitle track (no text to extract); \
                             provide --subtitle with a text file instead"
                        ),
                    )
                    .with_context(ErrorContext::default().with_input(input.as_path())));
                }
                Rect::None(_) => {}
            }
        }
        let text = lines.join(" ").trim().to_string();
        // Cues before the video's first frame (edit-list pre-roll, priming
        // cues) are clipped or dropped by the cue's own visibility rule.
        if let Some(cue) = Cue::visible(start, end, text) {
            cues.push(cue);
        }
    }

    Ok(cues)
}

/// Display duration used when neither the codec nor the container declares
/// one. Broadcast TS routinely omits both: subrip-in-TS packets carry no
/// AVSubtitle display times and a zero (or `AV_NOPTS_VALUE`) packet
/// duration. Dropping such cues would silently lose valid text; a fixed
/// fallback keeps them visible at a typical subtitle display length.
const DEFAULT_CUE_SECONDS: f64 = 3.0;

/// The container-declared packet duration in seconds, or `None` when it is
/// undeclared. libav uses 0 and negative values (`AV_NOPTS_VALUE` included)
/// as "unknown" sentinels — neither ever means an instantaneous cue, so
/// both map to absence here instead of leaking into window arithmetic.
fn declared_duration(raw: i64, time_base: f64) -> Option<f64> {
    (raw > 0).then_some(raw as f64 * time_base)
}

/// Resolve one cue's display window on the logical clock.
///
/// AVSubtitle display times (`start_ms..end_ms`, offsets from the packet
/// pts) win when the codec sets them; otherwise the container's declared
/// duration applies, and failing that, [`DEFAULT_CUE_SECONDS`].
///
/// Postcondition: `end > start`. Every arm of the fallback chain adds a
/// strictly positive duration, so a degenerate window cannot be built —
/// which leaves [`Cue::visible`]'s `end > start` rule its one real job:
/// clipping cues that end before the video's first frame.
fn cue_window(packet_start: f64, start_ms: u32, end_ms: u32, declared: Option<f64>) -> (f64, f64) {
    let start = packet_start + start_ms as f64 / 1000.0;
    let end = if end_ms > start_ms {
        packet_start + end_ms as f64 / 1000.0
    } else {
        start + declared.unwrap_or(DEFAULT_CUE_SECONDS)
    };
    (start, end)
}

/// Event-line schema of one subtitle track's ASS payload.
///
/// libavcodec's text subtitle decoders (subrip, mov_text, webvtt) synthesize
/// canonical events — `ReadOrder,Layer,Style,Name,MarginL,MarginR,MarginV,
/// Effect,Text`, 9 fields — but the `ass` decoder passes Matroska packet
/// payloads through verbatim, and those follow whatever `Format:` line the
/// muxer wrote into the track header (codec extradata): the file's field
/// list minus Start/End, with ReadOrder prepended. Learning the count from
/// extradata turns "9" from a guess into the track's own declaration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct AssSchema {
    /// Field count of a packet-form event line; Text is always the last
    /// field, so commas inside the text never split.
    fields: usize,
}

/// One decoded ASS payload line, classified against a track schema.
///
/// Parsing an event line is fallible, and the three outcomes demand three
/// different downstream policies — so the distinction is a type, not a
/// silent fallback.
#[derive(Debug, PartialEq, Eq)]
enum AssLine<'a> {
    /// Matched the schema; this is the Text field, markup still intact.
    Event(&'a str),
    /// No event shape at all (first field is not a number): a container
    /// that muxed plain text as `ass`. The whole line is the text.
    PlainText(&'a str),
    /// Event-shaped but too few fields for the schema — a truncated or
    /// mis-declared line whose "text" would be metadata. Dropped.
    Malformed,
}

impl AssSchema {
    /// The layout every modern libavcodec text-subtitle decoder emits.
    const CANONICAL: AssSchema = AssSchema { fields: 9 };

    /// Learn the packet-form field count from the track's script header,
    /// falling back to [`Self::CANONICAL`] when the extradata is absent, is
    /// not an ASS header, or declares no usable `[Events]` format.
    fn from_extradata(extradata: Option<&[u8]>) -> AssSchema {
        extradata
            .and_then(events_format_fields)
            .filter(|&fields| fields >= 2)
            .map(|fields| AssSchema { fields })
            .unwrap_or(Self::CANONICAL)
    }

    /// Classify one decoded payload line against this schema.
    fn classify<'a>(&self, line: &'a str) -> AssLine<'a> {
        let line = line.trim();
        // Old libav decoders emit full `Dialogue:` lines, which keep
        // Start/End and lack ReadOrder: one extra field vs packet form.
        let (line, fields, prefixed) = match line.strip_prefix("Dialogue:") {
            Some(rest) => (rest.trim(), self.fields + 1, true),
            None => (line, self.fields, false),
        };
        // Every event form leads with a numeric field (ReadOrder, Layer,
        // or SSA's Marked); a line that doesn't is not an event at all.
        let event_shaped = prefixed
            || line
                .split(',')
                .next()
                .is_some_and(|f| f.trim().parse::<u64>().is_ok());
        if !event_shaped {
            return AssLine::PlainText(line);
        }
        match line.splitn(fields, ',').nth(fields - 1) {
            Some(text) => AssLine::Event(text),
            None => AssLine::Malformed,
        }
    }
}

/// Field count of a packet-form event line, as declared by the `Format:`
/// line in the `[Events]` section of an ASS script header.
///
/// Models the Matroska muxing transform directly: Start/End move into block
/// timestamps and ReadOrder is prepended. Returns `None` when there is no
/// `[Events]` format or it does not end with a Text field.
fn events_format_fields(extradata: &[u8]) -> Option<usize> {
    let header = String::from_utf8_lossy(extradata);
    let mut in_events = false;
    for line in header.lines() {
        let line = line.trim();
        if line.starts_with('[') {
            in_events = line.eq_ignore_ascii_case("[events]");
            continue;
        }
        if !in_events {
            continue;
        }
        let Some(rest) = line
            .get(.."Format:".len())
            .filter(|p| p.eq_ignore_ascii_case("Format:"))
            .map(|_| &line["Format:".len()..])
        else {
            continue;
        };
        let names: Vec<&str> = rest.split(',').map(str::trim).collect();
        if !names.last()?.eq_ignore_ascii_case("text") {
            return None;
        }
        let timing = names
            .iter()
            .filter(|n| n.eq_ignore_ascii_case("start") || n.eq_ignore_ascii_case("end"))
            .count();
        return Some(names.len() - timing + 1);
    }
    None
}

/// The track's codec extradata — for ASS tracks, the full script header.
fn codec_extradata(params: &ffmpeg::codec::Parameters) -> Option<&[u8]> {
    // SAFETY: `params` guards a valid AVCodecParameters for its lifetime
    // (it owns the allocation or holds an Rc on the owning context), and
    // extradata/extradata_size are libav's canonical pair describing one
    // buffer. The returned slice borrows `params`, so it cannot outlive it.
    unsafe {
        let p = params.as_ptr();
        let (data, size) = ((*p).extradata, (*p).extradata_size);
        (!data.is_null() && size > 0).then(|| std::slice::from_raw_parts(data, size as usize))
    }
}

/// Strip ASS inline markup from an event's Text field: override blocks
/// `{\...}` are dropped and `\N`/`\n`/`\h` soft breaks become spaces.
fn strip_ass_markup(text: &str) -> String {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uses_codec_display_window_when_declared() {
        assert_eq!(cue_window(10.0, 500, 2500, Some(1.0)), (10.5, 12.5));
    }

    #[test]
    fn falls_back_to_container_duration_without_display_times() {
        assert_eq!(cue_window(10.0, 0, 0, Some(1.5)), (10.0, 11.5));
    }

    #[test]
    fn falls_back_to_default_duration_when_nothing_is_declared() {
        let (start, end) = cue_window(10.0, 0, 0, None);
        assert_eq!((start, end), (10.0, 10.0 + DEFAULT_CUE_SECONDS));
    }

    #[test]
    fn treats_sentinel_packet_durations_as_undeclared() {
        assert_eq!(declared_duration(0, 0.001), None);
        assert_eq!(declared_duration(ffmpeg::ffi::AV_NOPTS_VALUE, 0.001), None);
        assert_eq!(declared_duration(2000, 0.001), Some(2.0));
    }

    #[test]
    fn classifies_canonical_event() {
        assert_eq!(
            AssSchema::CANONICAL.classify("1,0,Default,,0,0,0,,one, two, three"),
            AssLine::Event("one, two, three")
        );
    }

    #[test]
    fn classifies_dialogue_prefixed_event() {
        assert_eq!(
            AssSchema::CANONICAL
                .classify("Dialogue: 0,0:00:01.00,0:00:02.00,Default,,0,0,0,,Plain line"),
            AssLine::Event("Plain line")
        );
    }

    #[test]
    fn classifies_truncated_event_as_malformed() {
        assert_eq!(
            AssSchema::CANONICAL.classify("0,0,Default,0,0,0,leaked text"),
            AssLine::Malformed
        );
    }

    #[test]
    fn passes_plain_text_through() {
        assert_eq!(
            AssSchema::CANONICAL.classify("hi, friend, where, are we going?"),
            AssLine::PlainText("hi, friend, where, are we going?")
        );
    }

    #[test]
    fn reads_field_count_from_events_format_line() {
        // A stripped muxer that dropped Name and Effect: 8 file fields
        // become 7 packet fields (minus Start/End, plus ReadOrder). The
        // [V4+ Styles] Format line must not be mistaken for it.
        let extradata = b"[Script Info]\nScriptType: v4.00+\n\
            [V4+ Styles]\nFormat: Name, Fontname, Fontsize\n\
            [Events]\nFormat: Layer, Start, End, Style, MarginL, MarginR, MarginV, Text\n";
        let schema = AssSchema::from_extradata(Some(extradata));
        assert_eq!(schema, AssSchema { fields: 7 });
        assert_eq!(
            schema.classify("0,0,Default,0,0,0,the real subtitle"),
            AssLine::Event("the real subtitle")
        );
    }

    #[test]
    fn falls_back_to_canonical_without_events_format() {
        assert_eq!(AssSchema::from_extradata(None), AssSchema::CANONICAL);
        assert_eq!(
            AssSchema::from_extradata(Some(b"[Script Info]\nTitle: x\n")),
            AssSchema::CANONICAL
        );
    }

    #[test]
    fn strips_markup_from_event_text() {
        assert_eq!(strip_ass_markup("{\\an8}Hello\\Nworld"), "Hello world");
    }
}
