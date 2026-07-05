//! The single boundary to the in-process FFmpeg libraries (libavformat,
//! libavcodec, libswscale) via `ffmpeg-next`.
//!
//! Everything media-shaped funnels through here: opening containers, probing
//! duration and keyframes, decoding sampled frames, and extracting embedded
//! subtitles. No other module names an `ffmpeg` type.
//!
//! Because decoding is in-process, a frame's timestamp comes from the frame
//! itself (`pts × time_base`), produced by the same decoder that produced the
//! pixels. The old subprocess design had to pair an ffprobe timestamp list
//! with an ffmpeg frame stream by index — a whole class of misattribution
//! failures that this module makes unrepresentable.

pub mod decode;
pub mod inspect;
mod scale;
pub mod subtitle;

use std::path::Path;
use std::sync::Once;

use ffmpeg_next as ffmpeg;

use crate::error::{Error, ErrorCode, ErrorContext, Result};

/// Initialize libav once per process and quiet its logging to errors.
///
/// libav writes its log to stderr directly; at `Error` level a fatal decode
/// complaint still reaches the user in real time (the moral successor of the
/// captured child stderr), without per-frame warning noise.
pub fn init() {
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        // init() registers everything; failure here means a broken FFmpeg
        // installation, which every later call would also report.
        let _ = ffmpeg::init();
        ffmpeg::log::set_level(ffmpeg::log::Level::Error);
    });
}

/// Open a media container for reading.
///
/// The file's existence/readability is validated by the pipeline before any
/// media work, so an open failure here means the content is not a readable
/// media container — an input problem, not an environment one.
pub fn open_input(path: &Path) -> Result<ffmpeg::format::context::Input> {
    init();
    ffmpeg::format::input(&path).map_err(|e| {
        Error::new(
            ErrorCode::InputCorrupt,
            format!("cannot open {} as a media container: {e}", path.display()),
        )
        .with_context(ErrorContext::default().with_input(path))
    })
}

/// The best video stream's identity and timing, resolved once at open.
#[derive(Debug, Clone, Copy)]
pub struct VideoStream {
    /// Global stream index used to filter demuxed packets.
    pub index: usize,
    /// Seconds per pts unit.
    pub time_base: f64,
    /// Stream start offset in pts units (0 when unreported), subtracted so
    /// reported timestamps start at 0 like the rest of the pipeline expects.
    pub start_pts: i64,
}

impl VideoStream {
    /// Convert a raw pts to seconds on the pipeline's 0-based clock.
    pub fn ts_seconds(&self, pts: i64) -> f32 {
        ((pts - self.start_pts) as f64 * self.time_base) as f32
    }
}

/// Locate the best video stream, or fail with an input-decade error.
pub fn best_video_stream(
    ictx: &ffmpeg::format::context::Input,
    path: &Path,
) -> Result<VideoStream> {
    let stream = ictx
        .streams()
        .best(ffmpeg::media::Type::Video)
        .ok_or_else(|| {
            Error::new(
                ErrorCode::InputCorrupt,
                format!("{} contains no video stream", path.display()),
            )
            .with_context(ErrorContext::default().with_input(path))
        })?;
    let tb = stream.time_base();
    let start = stream.start_time();
    Ok(VideoStream {
        index: stream.index(),
        time_base: tb.numerator() as f64 / tb.denominator() as f64,
        start_pts: if start == ffmpeg::ffi::AV_NOPTS_VALUE {
            0
        } else {
            start
        },
    })
}

/// Container duration in seconds.
///
/// Prefers the container header (`AVFormatContext.duration`), falling back to
/// the best video stream's own duration. A container that reports neither
/// (fragmented MP4, raw streams) yields an input-decade error, mirroring the
/// old ffprobe "N/A" handling.
pub fn duration_seconds(ictx: &ffmpeg::format::context::Input, path: &Path) -> Result<f32> {
    let container = ictx.duration();
    if container > 0 {
        return Ok(container as f64 as f32 / ffmpeg::ffi::AV_TIME_BASE as f32);
    }
    if let Some(stream) = ictx.streams().best(ffmpeg::media::Type::Video) {
        let d = stream.duration();
        if d > 0 {
            let tb = stream.time_base();
            return Ok((d as f64 * tb.numerator() as f64 / tb.denominator() as f64) as f32);
        }
    }
    Err(Error::new(
        ErrorCode::InputCorrupt,
        "the container reports no duration (fragmented or raw stream?)",
    )
    .with_context(ErrorContext::default().with_input(path)))
}

/// Consecutive demux read failures tolerated before the stream is declared
/// broken. Shared by frame decoding and subtitle extraction: both loops skip
/// transient errors, and both must turn a permanently failing input into an
/// error instead of a spin loop.
pub(crate) const MAX_READ_ERRORS: u32 = 64;

/// Map a mid-stream libav failure into the pipeline's coded error space.
pub fn decode_error(context: &str, e: ffmpeg::Error) -> Error {
    let code = match e {
        ffmpeg::Error::DecoderNotFound => ErrorCode::DecoderMissing,
        _ => ErrorCode::DecodeFailed,
    };
    Error::new(code, format!("{context}: {e}"))
}
