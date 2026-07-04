//! Container inspection for the `probe` subcommand: format and stream facts
//! read from the demuxed headers in one container open.
//!
//! No decoder is opened — like ffprobe, the report describes what the file
//! declares (codec parameters), so a video probes fine even when its codec
//! has no decoder built into this binary.

use std::path::Path;

use ffmpeg::Rational;
use ffmpeg::media::Type;
use ffmpeg_next as ffmpeg;

use crate::error::{Error, ErrorCode, ErrorContext, Result};

use super::{duration_seconds, open_input};

/// Stream/format facts backing the `probe` report.
#[derive(Debug)]
pub struct MediaInfo {
    /// Container duration in seconds.
    pub duration: f32,
    pub width: u32,
    pub height: u32,
    /// Average frame rate; `0.0` when the container does not report one.
    pub fps: f32,
    /// Codec name as libavcodec spells it (`h264`, `av1`, ...), independent
    /// of which decoder implementation would handle it.
    pub video_codec: String,
    /// Demuxer `format_name`, verbatim (e.g. `mov,mp4,m4a,3gp,3g2,mj2`) —
    /// the same string ffprobe reports.
    pub container: String,
    pub has_audio: bool,
    /// Subtitle streams in subtitle-relative index order; `index` matches
    /// `--subtitle-track`.
    pub subtitle_tracks: Vec<SubtitleTrackInfo>,
}

/// One embedded subtitle stream.
#[derive(Debug)]
pub struct SubtitleTrackInfo {
    pub index: usize,
    pub lang: Option<String>,
    pub codec: String,
}

/// Inspect `input`'s container and stream headers.
pub fn inspect(input: &Path) -> Result<MediaInfo> {
    let ictx = open_input(input)?;
    let duration = duration_seconds(&ictx, input)?;
    let container = ictx.format().name().to_string();

    // The first video stream with real dimensions, mirroring ffprobe's
    // classification: decode picks the "best" stream to play, but the report
    // describes what the file declares.
    let video = ictx
        .streams()
        .find_map(|s| {
            let params = s.parameters();
            if params.medium() != Type::Video {
                return None;
            }
            // SAFETY: `parameters()` wraps the stream's AVCodecParameters,
            // valid while `ictx` is open; width/height are plain ints with no
            // aliasing or threading concerns. ffmpeg-next exposes them only
            // through an opened decoder, which we deliberately avoid.
            let (w, h) = unsafe {
                let p = params.as_ptr();
                ((*p).width, (*p).height)
            };
            (w > 0 && h > 0).then(|| VideoFacts {
                width: w as u32,
                height: h as u32,
                fps: frame_rate(s.avg_frame_rate()).or_else(|| frame_rate(s.rate())),
                codec: params.id().name().to_string(),
            })
        })
        .ok_or_else(|| {
            Error::new(
                ErrorCode::InputCorrupt,
                "no video stream found — is this actually a video file?",
            )
            .with_context(ErrorContext::default().with_input(input))
        })?;

    let has_audio = ictx
        .streams()
        .any(|s| s.parameters().medium() == Type::Audio);

    let subtitle_tracks = ictx
        .streams()
        .filter(|s| s.parameters().medium() == Type::Subtitle)
        .enumerate()
        .map(|(index, s)| SubtitleTrackInfo {
            index,
            lang: s
                .metadata()
                .get("language")
                .map(str::trim)
                .filter(|l| !l.is_empty())
                .map(String::from),
            codec: s.parameters().id().name().to_string(),
        })
        .collect();

    Ok(MediaInfo {
        duration,
        width: video.width,
        height: video.height,
        fps: video.fps.unwrap_or(0.0),
        video_codec: video.codec,
        container,
        has_audio,
        subtitle_tracks,
    })
}

struct VideoFacts {
    width: u32,
    height: u32,
    fps: Option<f32>,
    codec: String,
}

/// A stream rate as fps, `None` when degenerate (`0/0`) or malformed —
/// `avg_frame_rate` reports `0/0` on some streams, so the caller falls back
/// to `r_frame_rate`.
fn frame_rate(rate: Rational) -> Option<f32> {
    if rate.denominator() == 0 {
        return None;
    }
    let fps = rate.numerator() as f32 / rate.denominator() as f32;
    (fps.is_finite() && fps > 0.0).then_some(fps)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_degenerate_frame_rates() {
        assert_eq!(frame_rate(Rational::new(0, 0)), None);
        assert_eq!(frame_rate(Rational::new(0, 1)), None);
        assert_eq!(frame_rate(Rational::new(-30, 1)), None);
    }

    #[test]
    fn converts_rational_rates_to_fps() {
        assert_eq!(frame_rate(Rational::new(30, 1)), Some(30.0));
        let ntsc = frame_rate(Rational::new(30000, 1001)).expect("valid rate");
        assert!((ntsc - 29.97).abs() < 0.01);
    }
}
