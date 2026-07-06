// SPDX-License-Identifier: Apache-2.0
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
mod scale;
pub mod subtitle;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Once};

use ffmpeg_next as ffmpeg;

use crate::error::{Error, ErrorCode, ErrorContext, Result};
use crate::source::Source;

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
/// A local file's existence/readability is validated by the pipeline before
/// any media work, so an open failure here means the content is not a
/// readable media container. For a remote source this is the first contact:
/// a protocol the linked FFmpeg lacks maps to
/// [`ErrorCode::ProtocolUnsupported`]; every other failure (unreachable host,
/// HTTP error, non-media payload) stays an input-decade error.
pub fn open_input(source: &Source) -> Result<ffmpeg::format::context::Input> {
    init();
    ffmpeg::format::input(source.as_path()).map_err(|e| open_error(source, e))
}

/// Like [`open_input`], but wires `stop` into libav's interrupt callback:
/// raising the flag makes any blocking I/O inside the returned context —
/// the open itself, seeks, and every subsequent packet read — return
/// `AVERROR_EXIT` instead of blocking until the operation completes. This is
/// what bounds a consumer's shutdown on a stalled source (dead network peer,
/// hung pipe), where a plain read would never return and a joining thread
/// would hang with it.
pub fn open_input_interruptible(
    source: &Source,
    stop: Arc<AtomicBool>,
) -> Result<ffmpeg::format::context::Input> {
    init();
    // ffmpeg-next boxes the closure and never reclaims it (the callback must
    // stay valid for the context's whole lifetime and the wrapper has no
    // teardown hook): a few leaked bytes per open, bounded by the handful of
    // opens a run performs.
    ffmpeg::format::input_with_interrupt(source.as_path(), move || stop.load(Ordering::Relaxed))
        .map_err(|e| open_error(source, e))
}

/// Map a container-open failure into the pipeline's coded error space.
fn open_error(source: &Source, e: ffmpeg::Error) -> Error {
    let err = match e {
        ffmpeg::Error::ProtocolNotFound => Error::new(
            ErrorCode::ProtocolUnsupported,
            format!("cannot open {source}: {e} (not compiled into this build's FFmpeg)"),
        ),
        _ => Error::new(
            ErrorCode::InputCorrupt,
            format!("cannot open {source} as a media container: {e}"),
        ),
    };
    err.with_context(ErrorContext::default().with_input(source.as_path()))
}

/// The presentation clock of one opened container: origin at the best video
/// stream's start, seconds thereafter.
///
/// Every timestamp the pipeline reports — frame ts, keyframe ts, subtitle
/// cue bounds — is produced by this one value, so the video and subtitle
/// timelines cannot drift apart by construction. Streams carry their own
/// `time_base`, so the raw-pts conversion takes it as a parameter; the
/// origin is the shared part.
#[derive(Debug, Clone, Copy)]
pub struct Timeline {
    /// Container-absolute origin in seconds: the best video stream's start
    /// time, or 0 when unreported or when the container has no video.
    origin: f64,
}

/// A demuxer seek position on the container's absolute clock, in
/// `AV_TIME_BASE` units — the clock `avformat_seek_file` speaks when no
/// stream index is bound.
///
/// Only [`Timeline::seek_target`] can mint one, so a logical 0-based
/// timestamp cannot reach the seek call unprojected: on containers whose
/// streams do not start at 0 (broadcast TS, live archives) an unprojected
/// seek lands before the stream start and silently degrades to a full
/// decode-and-discard from the file head.
#[derive(Debug, Clone, Copy)]
pub struct SeekTarget(i64);

impl SeekTarget {
    /// The raw `AV_TIME_BASE` position for the libav seek call.
    pub fn as_av_time(self) -> i64 {
        self.0
    }
}

impl Timeline {
    /// Resolve the clock from the container's best video stream.
    ///
    /// A container without a video stream anchors at 0 — the pipeline
    /// rejects such inputs on the video path anyway, and for subtitle
    /// extraction the container-absolute times are the best available.
    pub fn from_input(ictx: &ffmpeg::format::context::Input) -> Timeline {
        ictx.streams()
            .best(ffmpeg::media::Type::Video)
            .map(|s| Timeline::from_stream(&s))
            .unwrap_or(Timeline { origin: 0.0 })
    }

    fn from_stream(stream: &ffmpeg::format::stream::Stream) -> Timeline {
        let tb = stream.time_base();
        let time_base = tb.numerator() as f64 / tb.denominator() as f64;
        let start = stream.start_time();
        Timeline {
            origin: if start == ffmpeg::ffi::AV_NOPTS_VALUE {
                0.0
            } else {
                start as f64 * time_base
            },
        }
    }

    /// Convert a raw pts in `time_base` units (seconds per pts unit) to
    /// seconds on the pipeline's 0-based clock. Negative results are real:
    /// they place the pts before the video's first frame (edit-list
    /// pre-roll, priming cues) and it is the caller's domain rule how to
    /// treat them.
    pub fn seconds(&self, pts: i64, time_base: f64) -> f64 {
        pts as f64 * time_base - self.origin
    }

    /// Reverse projection: logical 0-based seconds back to the container's
    /// absolute clock, as a demuxer seek position.
    pub fn seek_target(&self, logical_secs: f64) -> SeekTarget {
        SeekTarget(((logical_secs + self.origin) * ffmpeg::ffi::AV_TIME_BASE as f64) as i64)
    }
}

/// The best video stream's identity and timing, resolved once at open.
#[derive(Debug, Clone, Copy)]
pub struct VideoStream {
    /// Global stream index used to filter demuxed packets.
    pub index: usize,
    /// Seconds per pts unit.
    pub time_base: f64,
    /// The container's presentation clock, anchored at this stream's start.
    pub timeline: Timeline,
}

impl VideoStream {
    /// Convert a raw pts to seconds on the pipeline's 0-based clock.
    pub fn ts_seconds(&self, pts: i64) -> f32 {
        self.timeline.seconds(pts, self.time_base) as f32
    }
}

/// Locate the best video stream, or fail with an input-decade error.
pub fn best_video_stream(
    ictx: &ffmpeg::format::context::Input,
    source: &Source,
) -> Result<VideoStream> {
    let stream = ictx
        .streams()
        .best(ffmpeg::media::Type::Video)
        .ok_or_else(|| {
            Error::new(
                ErrorCode::InputCorrupt,
                format!("{source} contains no video stream"),
            )
            .with_context(ErrorContext::default().with_input(source.as_path()))
        })?;
    let tb = stream.time_base();
    Ok(VideoStream {
        index: stream.index(),
        time_base: tb.numerator() as f64 / tb.denominator() as f64,
        timeline: Timeline::from_stream(&stream),
    })
}

/// Container duration in seconds.
///
/// Prefers the container header (`AVFormatContext.duration`), falling back to
/// the best video stream's own duration. A container that reports neither
/// (fragmented MP4, raw streams) yields an input-decade error, mirroring the
/// old ffprobe "N/A" handling.
pub fn duration_seconds(ictx: &ffmpeg::format::context::Input, source: &Source) -> Result<f32> {
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
    .with_context(ErrorContext::default().with_input(source.as_path())))
}

/// Consecutive demux read failures tolerated before the stream is declared
/// broken. Shared by every packet-read loop (frame decoding, subtitle
/// extraction, keyframe probing): all of them skip transient errors, and all
/// must turn a permanently failing input into an error instead of a spin
/// loop.
pub(crate) const MAX_READ_ERRORS: u32 = 64;

/// Circuit breaker for demux/decode loops that skip transient failures.
///
/// Every packet-read loop follows the same discipline: a successful read
/// resets the budget ([`ErrorBudget::ok`]), a failed one spends it
/// ([`ErrorBudget::fail`]), and an exhausted budget converts the last
/// failure into a coded error. Owning the counter here makes the breaker a
/// property of the loop's type shape — a new loop cannot forget it the way a
/// hand-rolled counter can be forgotten.
pub(crate) struct ErrorBudget {
    left: u32,
    /// Names the loop in the error message, e.g. "reading packets".
    context: &'static str,
}

impl ErrorBudget {
    pub fn new(context: &'static str) -> Self {
        Self {
            left: MAX_READ_ERRORS,
            context,
        }
    }

    /// A successful iteration: consecutive-failure tracking starts over.
    pub fn ok(&mut self) {
        self.left = MAX_READ_ERRORS;
    }

    /// A failed iteration: spend one unit, or convert `e` into the loop's
    /// coded error once the budget is exhausted.
    pub fn fail(&mut self, e: ffmpeg::Error) -> Result<()> {
        match self.left.checked_sub(1) {
            Some(left) => {
                self.left = left;
                Ok(())
            }
            None => Err(decode_error(self.context, e)),
        }
    }
}

/// Map a mid-stream libav failure into the pipeline's coded error space.
pub fn decode_error(context: &str, e: ffmpeg::Error) -> Error {
    let code = match e {
        ffmpeg::Error::DecoderNotFound => ErrorCode::DecoderMissing,
        _ => ErrorCode::DecodeFailed,
    };
    Error::new(code, format!("{context}: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rebases_pts_to_the_video_clock() {
        // A stream starting at 10.0s container time: a pts at 10.0s maps to
        // 0, one at 9.8s maps to -0.2 — before the first video frame.
        let timeline = Timeline { origin: 10.0 };
        let tb = 1.0 / 90_000.0;
        assert!((timeline.seconds(900_000, tb) - 0.0).abs() < 1e-9);
        assert!((timeline.seconds(882_000, tb) - (-0.2)).abs() < 1e-9);
    }

    #[test]
    fn projects_seek_target_past_nonzero_origin() {
        // Logical 5.0s on a stream starting at 10.0s container time must
        // seek to absolute 15.0s, not 5.0s — the unprojected value lands
        // before the stream start and kills fast seeking.
        let timeline = Timeline { origin: 10.0 };
        let target = timeline.seek_target(5.0);
        assert_eq!(
            target.as_av_time(),
            (15.0 * ffmpeg::ffi::AV_TIME_BASE as f64) as i64
        );
    }

    #[test]
    fn seek_target_is_identity_on_zero_origin() {
        let timeline = Timeline { origin: 0.0 };
        assert_eq!(
            timeline.seek_target(5.0).as_av_time(),
            5 * ffmpeg::ffi::AV_TIME_BASE as i64
        );
    }

    fn io_error() -> ffmpeg::Error {
        ffmpeg::Error::Other {
            errno: ffmpeg::error::EAGAIN,
        }
    }

    #[test]
    fn trips_after_consecutive_failures_beyond_budget() {
        let mut budget = ErrorBudget::new("test loop");
        for _ in 0..MAX_READ_ERRORS {
            budget.fail(io_error()).expect("failures within the budget");
        }
        let err = budget
            .fail(io_error())
            .expect_err("exhausted budget must trip");
        assert_eq!(err.code, ErrorCode::DecodeFailed);
    }

    #[test]
    fn resets_budget_on_success() {
        let mut budget = ErrorBudget::new("test loop");
        for _ in 0..MAX_READ_ERRORS {
            budget.fail(io_error()).expect("failures within the budget");
        }
        budget.ok();
        budget
            .fail(io_error())
            .expect("a success resets consecutive-failure tracking");
    }
}
