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
//! pixels, so a frame and its timestamp cannot be misattributed by index
//! pairing.

pub mod audio;
pub mod decode;
pub mod embedded_subtitle;
mod scale;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Once};

use ffmpeg_next as ffmpeg;

use crate::error::{Error, ErrorCode, ErrorContext, Result};
use crate::source::Source;

/// Initialize libav once per process and quiet its logging to errors.
///
/// libav writes its log to stderr directly; at `Error` level a fatal decode
/// complaint still reaches the user in real time, without per-frame warning
/// noise.
pub fn init() {
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        // Failure here means a broken FFmpeg install, which every later call
        // would also report.
        let _ = ffmpeg::init();
        ffmpeg::log::set_level(ffmpeg::log::Level::Error);
    });
}

/// The stall guard every container open requires: a cloneable stop token
/// wired into libav's interrupt callback.
///
/// Tripping the token makes any blocking I/O inside the opened context —
/// the open itself, seeks, and every subsequent packet read — return
/// `AVERROR_EXIT` instead of blocking until the operation completes. This
/// is what bounds a consumer's shutdown on a stalled source (dead network
/// peer, hung pipe), where a plain read would never return and a joining
/// thread would hang with it. `ErrorBudget` only bounds reads that
/// *return* errors; this token bounds the ones that never return.
///
/// The guard's *existence* is the type-level guarantee ([`open_input`]
/// takes one unconditionally); *when* it trips is the caller's policy — a
/// manual token ([`Stop::new`]) for local files, a deadline
/// ([`Stop::after`]) for network probes, a Drop-tripped token for a
/// consumer-owned decode thread.
#[derive(Clone)]
pub struct Stop {
    flag: Arc<AtomicBool>,
    /// Held only by deadline-armed tokens: dropping the last clone
    /// disconnects the channel, waking the deadline thread immediately —
    /// a finished open never leaks a sleeping thread past a scheduler
    /// tick, and disarming cannot be forgotten on any return path.
    _disarm: Option<Arc<std::sync::mpsc::Sender<()>>>,
}

impl Stop {
    /// A manual token: trips only via [`Stop::trip`], never on its own.
    pub fn new() -> Stop {
        Stop {
            flag: Arc::new(AtomicBool::new(false)),
            _disarm: None,
        }
    }

    /// A deadline token: trips itself after `timeout` unless every clone
    /// is dropped first.
    pub fn after(timeout: std::time::Duration) -> Stop {
        let flag = Arc::new(AtomicBool::new(false));
        let (tx, rx) = std::sync::mpsc::channel::<()>();
        let armed = Arc::clone(&flag);
        std::thread::spawn(move || {
            if rx.recv_timeout(timeout) == Err(std::sync::mpsc::RecvTimeoutError::Timeout) {
                armed.store(true, Ordering::Relaxed);
            }
        });
        Stop {
            flag,
            _disarm: Some(Arc::new(tx)),
        }
    }

    /// Trip the token: every blocking libav call it guards aborts with
    /// `AVERROR_EXIT` at its next interrupt poll. Idempotent; visible to
    /// all clones.
    pub fn trip(&self) {
        self.flag.store(true, Ordering::Relaxed);
    }

    /// Whether the token has tripped (manually or by deadline).
    pub fn is_tripped(&self) -> bool {
        self.flag.load(Ordering::Relaxed)
    }

    /// The flag the interrupt callback polls through a raw pointer.
    fn flag(&self) -> &Arc<AtomicBool> {
        &self.flag
    }
}

impl Default for Stop {
    fn default() -> Stop {
        Stop::new()
    }
}

/// Open a media container for reading, with `stop` wired into libav's
/// interrupt callback and `options` forwarded as protocol/demuxer knobs.
///
/// A local file's existence/readability is validated by the pipeline before
/// any media work, so an open failure here means the content is not a
/// readable media container. For a remote source this is the first contact:
/// a protocol the linked FFmpeg lacks maps to
/// [`ErrorCode::ProtocolUnsupported`]; every other failure (unreachable host,
/// HTTP error, non-media payload) stays an input-decade error.
///
/// `stop` is mandatory by design: "opened without a stall guard" is not a
/// representable state (see [`Stop`]). A caller that never intends to trip
/// passes `&Stop::new()`.
///
/// `options` become an `AVDictionary` handed to `avformat_open_input`, so
/// each pair is a libav protocol/demuxer option — e.g.
/// `("http_proxy", "http://127.0.0.1:port")` to route every connection the
/// demuxer makes through a forward proxy. This module forwards them
/// verbatim and never interprets their meaning; pass `&[]` for a plain open.
///
/// The returned [`InterruptibleInput`] derefs to the plain input context
/// and owns its clone of the token — nothing leaks per open.
pub fn open_input(
    source: &Source,
    stop: &Stop,
    options: &[(&str, &str)],
) -> Result<InterruptibleInput> {
    init();
    let path = path_cstring(source)?;
    open_with_interrupt_and_options(&path, stop.clone(), options).map_err(|e| open_error(source, e))
}

/// The source path as the C string libav's open call takes.
///
/// Rejecting a non-UTF-8 or interior-NUL path here — rather than asserting
/// invariants maintained layers away (URL vetting upstream, file opens that
/// reject NUL) — keeps a new call path from turning a weird filename into a
/// process abort: this library's servers build with `panic = "abort"`.
fn path_cstring(source: &Source) -> Result<std::ffi::CString> {
    let path = source.ffmpeg_path();
    path.as_os_str()
        .to_str()
        .ok_or("is not valid UTF-8")
        .and_then(|text| std::ffi::CString::new(text).map_err(|_| "contains an interior NUL byte"))
        .map_err(|why| {
            Error::new(
                ErrorCode::InputUnreadable,
                format!("cannot open {source}: the path {why}"),
            )
            .with_context(ErrorContext::default().with_input(source.as_path()))
        })
}

/// An opened container whose blocking I/O observes a [`Stop`] token.
///
/// The interrupt callback installed by [`open_input`] polls the
/// `AtomicBool` inside the token through a raw pointer, so the flag must
/// outlive every callback invocation. This wrapper is that proof: field
/// order drops `input` first — `avformat_close_input` runs, after which
/// libav never calls back — and only then releases the token clone.
/// Unlike `ffmpeg::util::interrupt`'s boxed-closure API (which leaks the
/// closure per open, unreclaimable because the context has no teardown
/// hook for it), nothing here outlives the open — which matters in a
/// long-lived server probing one URL per call.
///
/// The proof also pins the inner value's identity: it derefs to `&Input`
/// but deliberately exposes no `&mut Input` (no `DerefMut`, no accessor),
/// because a `&mut Input` admits `mem::replace` — safe code could extract
/// the callback-armed context, outlive the wrapper with it, and leave the
/// callback polling a freed flag. The mutating operations the pipeline
/// needs are delegated below ([`Self::seek`], [`Self::read_packet`]), so
/// the context this wrapper opened is the context it closes.
pub struct InterruptibleInput {
    /// Declared (and therefore dropped) first: the context must close
    /// before the flag it polls is released.
    input: ffmpeg::format::context::Input,
    /// Keeps the callback's pointee alive (and any deadline armed) for the
    /// context's lifetime.
    _stop: Stop,
}

impl InterruptibleInput {
    pub fn seek<R: ffmpeg::util::range::Range<i64>>(
        &mut self,
        ts: i64,
        range: R,
    ) -> std::result::Result<(), ffmpeg::Error> {
        self.input.seek(ts, range)
    }

    pub fn read_packet(
        &mut self,
        packet: &mut ffmpeg::Packet,
    ) -> std::result::Result<(), ffmpeg::Error> {
        packet.read(&mut self.input)
    }
}

impl std::ops::Deref for InterruptibleInput {
    type Target = ffmpeg::format::context::Input;

    fn deref(&self) -> &Self::Target {
        &self.input
    }
}

/// libav's interrupt poll: nonzero aborts the blocking call.
///
/// # Safety
///
/// `opaque` must point at an `AtomicBool` that outlives every possible
/// invocation — [`InterruptibleInput`] upholds this by holding the owning
/// `Arc` until after its context closes.
unsafe extern "C" fn interrupt_poll(opaque: *mut std::ffi::c_void) -> std::ffi::c_int {
    // SAFETY: `opaque` is `Arc::as_ptr` of the flag held by the
    // `InterruptibleInput` that owns this context; the wrapper's drop
    // order guarantees the pointee is alive whenever libav calls here.
    let stop = unsafe { &*opaque.cast::<AtomicBool>() };
    std::ffi::c_int::from(stop.load(Ordering::Relaxed))
}

/// Open an input with both an interrupt callback and an options dictionary.
///
/// `ffmpeg-next` exposes `input_with_interrupt` and `input_with_dictionary`
/// but not their union, so this drops to FFI to set the interrupt callback
/// and pass the options dictionary on one `AVFormatContext`. Returning
/// libav's raw error keeps the coded mapping in one place ([`open_error`]).
fn open_with_interrupt_and_options(
    path: &std::ffi::CStr,
    stop: Stop,
    options: &[(&str, &str)],
) -> std::result::Result<InterruptibleInput, ffmpeg::Error> {
    use ffmpeg::format::context::Input;

    let mut dict = ffmpeg::Dictionary::new();
    for (key, value) in options {
        dict.set(key, value);
    }

    // SAFETY: `ps` is allocated by libav and handed straight to
    // `avformat_open_input`, which on success adopts it (freed later by the
    // wrapping `Input`'s Drop) and on error frees it via
    // `avformat_close_input`. The interrupt callback's opaque pointer
    // aliases the `AtomicBool` inside `stop`, whose owning `Stop` clone
    // travels in the returned `InterruptibleInput` and is released only
    // after the context closes — see that type's drop-order contract; on
    // every error path the context is already gone before `stop` drops
    // here. `opts` is
    // disowned into the call and re-owned after so the reclaimed dictionary
    // frees any options libav did not consume.
    unsafe {
        let mut ps = ffmpeg::ffi::avformat_alloc_context();
        if ps.is_null() {
            // Allocation failed (OOM); return before any deref. `dict` is
            // never disowned on this path, so it drops and frees normally.
            return Err(ffmpeg::Error::Other {
                errno: ffmpeg::error::ENOMEM,
            });
        }
        (*ps).interrupt_callback = ffmpeg::ffi::AVIOInterruptCB {
            callback: Some(interrupt_poll),
            opaque: Arc::as_ptr(stop.flag()).cast_mut().cast(),
        };
        let mut opts = dict.disown();
        let res = ffmpeg::ffi::avformat_open_input(
            &mut ps,
            path.as_ptr(),
            std::ptr::null_mut(),
            &mut opts,
        );
        ffmpeg::Dictionary::own(opts);
        if res < 0 {
            return Err(ffmpeg::Error::from(res));
        }
        match ffmpeg::ffi::avformat_find_stream_info(ps, std::ptr::null_mut()) {
            r if r >= 0 => Ok(InterruptibleInput {
                input: Input::wrap(ps),
                _stop: stop,
            }),
            e => {
                ffmpeg::ffi::avformat_close_input(&mut ps);
                Err(ffmpeg::Error::from(e))
            }
        }
    }
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

    /// A timeline anchored at an explicitly elected origin (seconds on the
    /// container's absolute clock) instead of the stream-declared start —
    /// the corrected clock of an origin-rebased decode pass (see
    /// [`crate::media::decode`]'s origin-rebase contract).
    pub(crate) fn rebased(origin: f64) -> Timeline {
        Timeline { origin }
    }

    /// The container-absolute origin in seconds — what [`Timeline::seconds`]
    /// subtracts and [`Timeline::seek_target`] adds back.
    pub(crate) fn origin(&self) -> f64 {
        self.origin
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
        // The `as i64` cannot lose a valid seek: callers pass a validated
        // window position (≤ the probed duration), and i64::MAX at
        // AV_TIME_BASE (1 µs) ticks is ~292k years; float→int `as` also
        // saturates rather than wrapping.
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
/// (fragmented MP4, raw streams) yields an input-decade error.
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

/// Per-track timeline coverage, for post-download integrity checks.
///
/// A truncated download often keeps a valid container header — the muxer
/// wrote the *claimed* duration — while a track's own duration records
/// where its samples actually stop. Comparing the two exposes a served-
/// truncated stream without decoding a single frame.
#[derive(Debug, Clone, Copy)]
pub struct StreamCoverage {
    pub container_seconds: f64,
    /// Best audio stream's own duration; `None` = no stream or no duration.
    pub audio_seconds: Option<f64>,
    /// Best video stream's own duration; `None` = no stream or no duration.
    pub video_seconds: Option<f64>,
}

/// Read [`StreamCoverage`] from a container header. `stop` bounds the
/// header reads on a stalled source; see [`open_input`].
pub fn stream_coverage(source: &Source, stop: &Stop) -> Result<StreamCoverage> {
    let ictx = open_input(source, stop, &[])?;
    let container_seconds = f64::from(duration_seconds(&ictx, source)?);
    let track = |ty: ffmpeg::media::Type| -> Option<f64> {
        let stream = ictx.streams().best(ty)?;
        let d = stream.duration();
        if d <= 0 {
            return None;
        }
        let tb = stream.time_base();
        Some(d as f64 * tb.numerator() as f64 / tb.denominator() as f64)
    };
    Ok(StreamCoverage {
        container_seconds,
        audio_seconds: track(ffmpeg::media::Type::Audio),
        video_seconds: track(ffmpeg::media::Type::Video),
    })
}

/// Container-level facts for the probe surface: dimensions, duration,
/// container name, and subtitle tracks. Reading this decodes no pixels —
/// it is all header data, cheap even over http(s).
#[derive(Debug, Clone)]
pub struct MediaSummary {
    /// Seconds, from the container header.
    pub duration: f32,
    pub width: u32,
    pub height: u32,
    /// Demuxer name, a comma-list like `mov,mp4,m4a,3gp,3g2,mj2`.
    pub container: String,
    pub subtitle_tracks: Vec<SubtitleTrackSummary>,
}

/// One embedded subtitle stream, as the probe surface reports it.
#[derive(Debug, Clone)]
pub struct SubtitleTrackSummary {
    /// Language tag from stream metadata, when present.
    pub language: Option<String>,
    /// Codec name, e.g. `subrip`, `ass`, `hdmv_pgs_subtitle`.
    pub codec: String,
}

/// Summarize an already-open container; see [`MediaSummary`].
///
/// Fails when the container has no video stream with usable dimensions or
/// reports no duration — the callers' contract is "this is a watchable
/// video", so both are input-decade errors.
pub fn summarize_input(
    ictx: &ffmpeg::format::context::Input,
    source: &Source,
) -> Result<MediaSummary> {
    use ffmpeg::codec::context::Context as CodecContext;

    let duration = duration_seconds(ictx, source)?;

    let stream = ictx
        .streams()
        .best(ffmpeg::media::Type::Video)
        .ok_or_else(|| {
            Error::new(
                ErrorCode::InputCorrupt,
                "no video stream found — is this actually a video file?",
            )
            .with_input(source.as_path())
        })?;
    // Distinct failures stay distinct: unreadable codec parameters are an
    // input problem, a missing decoder is an environment problem — neither
    // may be misreported as "not a video file".
    let video = CodecContext::from_parameters(stream.parameters())
        .map_err(|e| {
            decode_error("reading video stream parameters", e).with_input(source.as_path())
        })?
        .decoder()
        .video()
        .map_err(|e| decode_error("opening the video decoder", e).with_input(source.as_path()))?;
    if video.width() == 0 || video.height() == 0 {
        return Err(Error::new(
            ErrorCode::InputCorrupt,
            "the video stream reports no dimensions",
        )
        .with_input(source.as_path()));
    }

    let subtitle_tracks = ictx
        .streams()
        .filter_map(|s| {
            let ctx = CodecContext::from_parameters(s.parameters()).ok()?;
            (ctx.medium() == ffmpeg::media::Type::Subtitle).then(|| SubtitleTrackSummary {
                language: s
                    .metadata()
                    .get("language")
                    .map(str::trim)
                    .filter(|l| !l.is_empty())
                    .map(String::from),
                codec: ctx.id().name().to_string(),
            })
        })
        .collect();

    Ok(MediaSummary {
        duration,
        width: video.width(),
        height: video.height(),
        container: ictx.format().name().to_string(),
        subtitle_tracks,
    })
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
    /// Cumulative failures, kept across [`ErrorBudget::ok`] resets —
    /// diagnostics only (the "0 frames decoded, N bad packets" verdict
    /// and its message), never the trip condition.
    total: u64,
    /// Names the loop in the error message, e.g. "reading packets".
    context: &'static str,
}

impl ErrorBudget {
    pub fn new(context: &'static str) -> Self {
        Self {
            left: MAX_READ_ERRORS,
            total: 0,
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
        self.total += 1;
        match self.left.checked_sub(1) {
            Some(left) => {
                self.left = left;
                Ok(())
            }
            None => Err(decode_error(self.context, e)),
        }
    }

    /// Cumulative failures over the budget's lifetime, across resets.
    pub fn failures(&self) -> u64 {
        self.total
    }
}

/// One `receive_frame` outcome, classified. Three states demand three
/// downstream policies — a frame to handle, a drained decoder to leave
/// for the next feed, a hard error to charge or surface — so the
/// distinction is a type, and `while receive_frame().is_ok()` (which
/// silently conflates [`Recv::Broken`] with [`Recv::Drained`] and
/// truncates the tail) has no shape left to be written in.
pub(crate) enum Recv {
    /// A frame arrived; the receive buffer holds it.
    Frame,
    /// The decoder owes nothing right now: `EAGAIN` (feed more input) or
    /// `EOF` (fully flushed). Both end this drain pass.
    Drained,
    /// A hard decoder error — never silently absorbed.
    Broken(ffmpeg::Error),
}

/// Classify one `receive_frame` result; shared by every drain loop.
pub(crate) fn classify_receive(result: std::result::Result<(), ffmpeg::Error>) -> Recv {
    match result {
        Ok(()) => Recv::Frame,
        Err(ffmpeg::Error::Other { errno }) if errno == ffmpeg::error::EAGAIN => Recv::Drained,
        Err(ffmpeg::Error::Eof) => Recv::Drained,
        Err(e) => Recv::Broken(e),
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
    fn rejects_path_with_interior_nul() {
        let source = Source::Local(std::path::PathBuf::from("/tmp/vid\0eo.mp4"));
        let err = path_cstring(&source).expect_err("an interior NUL cannot reach libav");
        assert_eq!(err.code, ErrorCode::InputUnreadable);
    }

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

    #[test]
    fn keeps_total_failures_across_resets() {
        let mut budget = ErrorBudget::new("test loop");
        budget.fail(io_error()).expect("within budget");
        budget.ok();
        budget.fail(io_error()).expect("reset restores the budget");
        assert_eq!(
            budget.failures(),
            2,
            "cumulative count survives the consecutive reset"
        );
    }

    #[test]
    fn classifies_drained_receive_results() {
        assert!(matches!(classify_receive(Ok(())), Recv::Frame));
        assert!(matches!(
            classify_receive(Err(ffmpeg::Error::Other {
                errno: ffmpeg::error::EAGAIN,
            })),
            Recv::Drained
        ));
        assert!(matches!(
            classify_receive(Err(ffmpeg::Error::Eof)),
            Recv::Drained
        ));
    }

    #[test]
    fn classifies_hard_receive_error_as_broken() {
        assert!(matches!(
            classify_receive(Err(ffmpeg::Error::InvalidData)),
            Recv::Broken(ffmpeg::Error::InvalidData)
        ));
    }

    #[test]
    fn trips_manual_token_across_clones() {
        let stop = Stop::new();
        let clone = stop.clone();
        assert!(!clone.is_tripped(), "a fresh token is untripped");
        stop.trip();
        assert!(clone.is_tripped(), "a trip is visible to every clone");
    }

    #[test]
    fn trips_deadline_token_after_timeout() {
        let stop = Stop::after(std::time::Duration::from_millis(5));
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while !stop.is_tripped() {
            assert!(
                std::time::Instant::now() < deadline,
                "the deadline thread must trip the token"
            );
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
    }
}
