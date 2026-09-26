// SPDX-License-Identifier: Apache-2.0
//! In-process frame decoding: a dedicated decode thread feeding a bounded
//! channel of small RGB frames. The channel's capacity is the backpressure,
//! and a dropped receiver stops the producer.
//!
//! Timestamps come from the decoded frames (`best-effort pts × time_base`,
//! rebased to the stream start). Container clocks may legally jump (MPEG-TS
//! discontinuities, pts wrap-arounds, splices) and nothing local tells a
//! legal jump from a corrupt one, so every frame passes the `Segmenter`,
//! which cuts the clock into monotone segments and projects them onto one
//! continuous logical clock. Everything downstream consumes only that
//! clock; a single-segment stream projects identically.
//!
//! Every decodable frame is either placed on the logical timeline or
//! dropped with a signal ([`Event::UnplaceableTimestamp`] for the first,
//! [`Event::TimestampsDropped`] for the total); every seam is announced as
//! [`Event::TimelineDiscontinuity`]. Timestamps alone never fail a run.
//!
//! Seeking speaks the container clock, so a windowed decode is exact only
//! within the first segment. A seek that lands past every decodable frame
//! (a poisoned index) is detected at EOF with zero frames, announced as
//! [`Event::SeekFallback`], and retried once from the stream head.

use std::num::NonZeroUsize;
use std::sync::mpsc::{Receiver, SyncSender, sync_channel};
use std::thread::JoinHandle;

use ffmpeg::codec::Context as CodecContext;
use ffmpeg::{Discard, Packet, threading};
use ffmpeg_next as ffmpeg;
use image::RgbImage;

use crate::error::{Error, ErrorCode, Result};
use crate::event::{Event, EventSink};
use crate::source::Source;
use crate::types::{TimeRange, Ts};

use super::clock::{ContainerTs, Discontinuity, FALLBACK_NOMINAL_GAP, SegStep, Segmenter};
use super::orientation;
use super::scale::Rescaler;
use super::{
    ErrorBudget, Recv, Stop, VideoStream, best_video_stream, classify_receive, decode_error,
    duration_seconds, open_input,
};

/// Recover a suspect header origin from the same confirmed video head used
/// for frame labels. Only the head is decoded; no RGB conversion is needed.
pub(super) fn head_timeline(
    source: &Source,
    stop: &Stop,
    audio_start: f64,
) -> Result<super::Timeline> {
    let mut input = open_input(source, stop, &[])?;
    let video = best_video_stream(&input, source)?;
    let stream = input.stream(video.index).expect("video stream");
    let mut decoder = CodecContext::from_parameters(stream.parameters())
        .and_then(|ctx| ctx.decoder().video())
        .map_err(|e| decode_error("opening video clock decoder", e))?;
    let mut clock = Segmenter::new(duration_seconds(&input, source).ok(), FALLBACK_NOMINAL_GAP);
    let mut packet = Packet::empty();
    let mut budget = ErrorBudget::new("recovering video clock origin");
    let mut flushing = false;
    loop {
        let mut frame = ffmpeg::frame::Video::empty();
        let mut finished = false;
        let step = match classify_receive(decoder.receive_frame(&mut frame)) {
            Recv::Frame => {
                budget.ok();
                let Some(pts) = frame.timestamp().or(frame.pts()) else {
                    continue;
                };
                clock.admit(
                    ContainerTs(video.timeline.seconds(pts, video.time_base)),
                    frame,
                )
            }
            Recv::Broken(e) => {
                budget.fail(e)?;
                continue;
            }
            Recv::Drained if flushing => {
                finished = true;
                clock.finish()
            }
            Recv::Drained => {
                match input.read_packet(&mut packet) {
                    Ok(()) if packet.stream() == video.index => {
                        decoder
                            .send_packet(&packet)
                            .map_err(|e| decode_error("decoding video clock head", e))?;
                    }
                    Ok(()) => {}
                    Err(ffmpeg::Error::Eof) => {
                        decoder
                            .send_eof()
                            .map_err(|e| decode_error("flushing video clock head", e))?;
                        flushing = true;
                    }
                    Err(e) => budget.fail(e)?,
                }
                continue;
            }
        };
        for (logical, frame) in step.released {
            let pts = frame
                .timestamp()
                .or(frame.pts())
                .expect("admitted timestamp");
            let timeline = super::Timeline {
                origin: pts as f64 * video.time_base - logical.0,
            };
            // A long poisoned head can itself confirm. Wait for the run
            // on the audio clock instead of returning that same bad origin.
            if audio_start - timeline.origin >= super::clock::PREROLL_ALLOWANCE {
                return Ok(timeline);
            }
        }
        if finished {
            return Ok(video.timeline);
        }
    }
}

/// One sampled RGB frame, already downscaled for composition.
#[derive(Debug)]
pub struct RgbFrame {
    /// Presentation timestamp, 0-based at the stream start.
    pub ts: Ts,
    /// Downscaled RGB image used later by composition.
    pub rgb: RgbImage,
}

/// Which frames of the video to sample.
#[derive(Debug, Clone)]
pub enum SampleSpec {
    /// One frame per `1/fps` tick, decoding clamped to `[start, end)` —
    /// seek to `start`, stop at `end` (`None` reads to EOF). Ticks are
    /// anchored at `start`, so timestamps stay absolute to the file.
    ///
    /// Callers guarantee `start <= end`; both CLI doors ([`TimeRange`] for
    /// `extract --range`, `Window::from_args` for `process --from/--until`)
    /// validate the ordering at parse time.
    Fps {
        fps: f32,
        start: f32,
        end: Option<f32>,
    },
    /// Only keyframes (I-frames); timestamps are their own pts.
    Keyframes,
    /// Keyframes everywhere, plus `1/fps` ticks inside each `fill` range —
    /// the adaptive plan, executed by one decoder in one pass.
    ///
    /// `fill` ranges must be ascending and non-overlapping (the invariant
    /// [`crate::probe::KeyframeInfo::sparse_regions`] provides). Inside a
    /// fill range every frame competes at the ticks, keyframes included; a
    /// sparse region opens at a keyframe, so its first tick lands on it.
    Mixed { fps: f32, fill: Vec<TimeRange> },
}

/// Frames buffered in the channel between the decode thread and the consumer.
///
/// The backpressure knob: large enough to keep both sides busy, small
/// enough that a stalled consumer bounds memory at a few small frames.
const CHANNEL_FRAMES: usize = 8;

/// One message from the decode thread: a sampled frame or a timeline
/// signal (a dropped frame, a committed clock seam). Signals ride the same
/// channel because the worker cannot borrow the caller's event sink across
/// the thread boundary; [`FrameSource::next_frame`] turns them into events
/// on the consumer side.
enum Msg {
    Frame(RgbFrame),
    Dropped {
        ts: f32,
        last: Option<f32>,
    },
    Discontinuity {
        container: f64,
        logical: f64,
    },
    /// A window seek reported success but landed beyond every decodable
    /// frame; the worker rewound to the stream head and reads sequentially.
    SeekRecovered {
        start: f32,
    },
}

/// A stream of sampled frames decoded on a background thread.
///
/// Pull frames with [`FrameSource::next_frame`]; `None` means clean end of
/// stream. Dropping the source stops the decode thread at its next send.
pub struct FrameSource<'e> {
    rx: Option<Receiver<std::result::Result<Msg, Error>>>,
    thread: Option<JoinHandle<()>>,
    /// Tripped on drop so the worker exits at its next packet-loop
    /// iteration instead of decoding on until the next sampled frame's
    /// failed send. Also wired into libav's interrupt callback at open, so
    /// a packet read blocked inside a libav C call (stalled network source,
    /// hung pipe) aborts with `AVERROR_EXIT` instead of blocking the join
    /// forever.
    stop: Stop,
    aspect: Option<f32>,
    done: bool,
    /// Receives the run's timeline signals: the first-drop warning, the
    /// end-of-decode drop summary, and every committed clock seam.
    events: &'e dyn EventSink,
    /// Frames the decode pass dropped off the timeline so far.
    drops: u64,
}

impl<'e> FrameSource<'e> {
    /// Open `input` and start decoding per `spec`.
    ///
    /// Frames are pre-scaled so their shorter edge is `min_edge` pixels, and
    /// turned upright per the stream's display matrix. Only right-angle
    /// rotations and reflections with uniform scale are supported; other
    /// display transforms fail with [`ErrorCode::DecodeFailed`]. `threads` caps the
    /// decoder's thread count (`None` = auto). Container, stream, decoder,
    /// and seek failures surface here with precise error codes; mid-stream
    /// failures surface from `next_frame`. `events` receives
    /// [`Event::SeekFallback`] when a window seek fails or lands past every
    /// decodable frame, and the timeline signals of
    /// [`FrameSource::next_frame`].
    ///
    /// The [`Stop`] guarding this open is self-minted: [`Drop`] needs the
    /// exclusive right to trip it. A pipeline-level token such as
    /// `ProcessOpts.stop` therefore does not reach this decode thread.
    pub fn open(
        input: &Source,
        spec: SampleSpec,
        min_edge: u32,
        threads: Option<NonZeroUsize>,
        events: &'e dyn EventSink,
    ) -> Result<Self> {
        // Built before the container so it can ride into libav's interrupt
        // callback, arming every blocking I/O call in this context.
        let stop = Stop::new();
        let mut ictx = open_input(input, &stop, &[])?;
        let video = best_video_stream(&ictx, input)?;
        // The declared duration seeds the segmenter's extent fence — a
        // label prior, never a verdict (see [`Segmenter`]); a container
        // that declares none leaves only the flat floor.
        let duration = duration_seconds(&ictx, input).ok();

        // Seek before building the decoder so no flush is needed. The seek
        // speaks the container's absolute clock, so the logical 0-based
        // `start` is projected through the timeline; this is exact only
        // within the stream's first timeline segment (see the module docs).
        // An unseekable stream degrades to a decode-and-discard up to the
        // sampler's first tick. A seek that succeeds is still unverified
        // until a frame decodes — the worker rewinds on an empty landing.
        let mut pending_seek = None;
        if let SampleSpec::Fps { start, .. } = &spec
            && *start > 0.0
        {
            let target = video.timeline.seek_target(*start as f64).as_av_time();
            match ictx.seek(target, ..target) {
                Ok(()) => pending_seek = Some(*start),
                Err(e) => events.event(Event::SeekFallback {
                    start: *start,
                    error: e.to_string(),
                }),
            }
        }

        let stream = ictx
            .stream(video.index)
            .expect("best stream index is valid");
        let orientation =
            orientation::of_stream(&stream).map_err(|e| e.with_input(input.as_path()))?;
        let mut codec_ctx = CodecContext::from_parameters(stream.parameters())
            .map_err(|e| decode_error("reading codec parameters", e).with_input(input.as_path()))?;

        // Count 0 = auto-detect cores. Without explicit threading the decoder
        // defaults to a single thread, unlike the ffmpeg CLI.
        let mut tc = threading::Config::count(threads.map_or(0, NonZeroUsize::get));
        tc.kind = threading::Type::Frame;
        codec_ctx.set_threading(tc);

        let mut decoder = codec_ctx.decoder();
        if matches!(spec, SampleSpec::Keyframes) {
            // Redundant with the packet gate in the pump: also skip non-key.
            decoder.skip_frame(Discard::NonKey);
        }
        let decoder = decoder
            .video()
            .map_err(|e| decode_error("opening video decoder", e).with_input(input.as_path()))?;

        let (tx, rx) = sync_channel(CHANNEL_FRAMES);
        let stop_flag = stop.clone();
        let thread = std::thread::Builder::new()
            .name("agent-video-decode".into())
            .spawn(move || {
                let mut worker = Worker {
                    ictx,
                    video,
                    decoder,
                    gate: PacketGate::new(&spec),
                    path: FramePath::new(spec, duration),
                    rescaler: Rescaler::new(min_edge, orientation),
                    tx,
                    stop: stop_flag,
                    frames_sent: 0,
                    pending_seek,
                };
                worker.run();
            })
            .map_err(|e| {
                Error::new(
                    ErrorCode::DecodeFailed,
                    format!("cannot spawn decode thread: {e}"),
                )
            })?;

        Ok(Self {
            rx: Some(rx),
            thread: Some(thread),
            stop,
            aspect: None,
            done: false,
            events,
            drops: 0,
        })
    }

    /// Source frame aspect ratio, known once at least one frame was read.
    pub fn aspect(&self) -> Option<f32> {
        self.aspect
    }

    /// Read the next sampled frame, or `None` after a clean end of stream.
    ///
    /// Frames arrive in non-decreasing *logical* timestamp order (see
    /// `Segmenter`). Timeline trouble is signal, never verdict: the first
    /// dropped frame emits [`Event::UnplaceableTimestamp`], a clean end of
    /// stream with any drops emits [`Event::TimestampsDropped`], and every
    /// committed clock seam emits [`Event::TimelineDiscontinuity`] — no
    /// combination of timestamps alone produces an error here.
    ///
    /// A decode-thread panic surfaces here as an [`Error`] with
    /// [`ErrorCode::DecodeFailed`], never as a silently truncated stream —
    /// in unwind builds (tests, downstream consumers on the default
    /// profile). This workspace's release binaries build with
    /// `panic = "abort"`, so a panic terminates the process before ever
    /// reaching this path. Mid-stream decode failures carry the worker's
    /// typed code unchanged.
    pub fn next_frame(&mut self) -> Result<Option<RgbFrame>> {
        if self.done {
            return Ok(None);
        }
        loop {
            let rx = self.rx.as_ref().expect("receiver lives until drop");
            match rx.recv() {
                Ok(Ok(Msg::Frame(frame))) => {
                    let (w, h) = frame.rgb.dimensions();
                    self.aspect = Some(w as f32 / h as f32);
                    return Ok(Some(frame));
                }
                Ok(Ok(Msg::Dropped { ts, last })) => {
                    if self.drops == 0 {
                        self.events.event(Event::UnplaceableTimestamp { ts, last });
                    }
                    self.drops += 1;
                }
                Ok(Ok(Msg::Discontinuity { container, logical })) => {
                    self.events.event(Event::TimelineDiscontinuity {
                        container_ts: container,
                        logical_ts: logical,
                    });
                }
                Ok(Ok(Msg::SeekRecovered { start })) => {
                    self.events.event(Event::SeekFallback {
                        start,
                        error: "seek landed beyond every decodable frame; \
                                rewound to the stream head"
                            .into(),
                    });
                }
                Ok(Err(e)) => {
                    self.done = true;
                    return Err(e);
                }
                // Channel closed: the worker is gone, but a panic also drops
                // `tx` on unwind. Joining tells a clean end of stream from a
                // crashed worker, which must not masquerade as EOF.
                Err(_) => {
                    self.done = true;
                    return match self.join() {
                        Ok(()) => {
                            if self.drops > 0 {
                                self.events
                                    .event(Event::TimestampsDropped { count: self.drops });
                            }
                            Ok(None)
                        }
                        Err(payload) => Err(Error::new(
                            ErrorCode::DecodeFailed,
                            format!(
                                "decode thread panicked: {}",
                                panic_message(payload.as_ref())
                            ),
                        )),
                    };
                }
            }
        }
    }

    fn join(&mut self) -> std::thread::Result<()> {
        self.thread.take().map_or(Ok(()), JoinHandle::join)
    }
}

/// Best-effort text from a panic payload: `panic!` carries `&str` or
/// `String`; anything else gets a placeholder.
fn panic_message(payload: &(dyn std::any::Any + Send)) -> &str {
    payload
        .downcast_ref::<&str>()
        .copied()
        .or_else(|| payload.downcast_ref::<String>().map(String::as_str))
        .unwrap_or("non-string panic payload")
}

impl Drop for FrameSource<'_> {
    fn drop(&mut self) {
        // The token stops the worker at its next loop iteration and aborts a
        // packet read blocked inside libav (interrupt callback); dropping the
        // receiver unblocks a send in progress. Together they bound the join
        // even on a source that will never deliver another byte.
        self.stop.trip();
        self.rx = None;
        // A panic here was already reported by the panic hook; propagating
        // out of Drop would escalate it to an abort.
        let _ = self.join();
    }
}

/// Tick-based frame selection: emits the first frame whose ts reaches the
/// next tick, then advances the tick past that frame — at most one frame per
/// tick, one tick per frame, robust to pts gaps. Frames decoded before a seek
/// target (keyframe preroll) fall before the first tick and are discarded
/// here.
///
/// Tick instants are indexed as `start + idx * step`, never accumulated, so
/// a far-future pts cannot spin the advance and hours of ticks accrue no
/// rounding error. The index never rewinds: timestamps must be
/// non-decreasing, which [`Segmenter`] guarantees upstream.
struct Tick {
    start: f64,
    step: f64,
    /// Index of the next tick: its instant is `start + idx * step`.
    idx: u64,
}

impl Tick {
    fn new(start: f32, step: f64) -> Self {
        Self {
            start: f64::from(start),
            step,
            idx: 0,
        }
    }

    /// The next tick instant, exact up to one f64 rounding.
    fn next_instant(&self) -> f64 {
        self.start + self.idx as f64 * self.step
    }

    fn wants(&mut self, ts: f32) -> bool {
        // A sliver of tolerance absorbs float noise in pts arithmetic;
        // anything larger would visibly shift the sampling instants.
        let reach = f64::from(ts) + self.step * 1e-3;
        if reach < self.next_instant() {
            return false;
        }
        // The last tick index at or before `reach`, computed directly. The
        // `as` cast saturates on out-of-range values; `max` keeps the index
        // monotonic against float rounding at the tick boundary.
        let reached = ((reach - self.start) / self.step) as u64;
        self.idx = reached.max(self.idx).saturating_add(1);
        true
    }
}

/// Frame-level selection per sampling mode.
enum Sampler {
    Fps(Tick),
    All,
    /// Keyframes outside the fill ranges, ticks inside them. `tick` is the
    /// current range's tick state, created on entry with its first tick at
    /// the range start (for a sparse region, the keyframe that opens it).
    Mixed {
        fill: Vec<TimeRange>,
        idx: usize,
        tick: Option<Tick>,
        step: f64,
    },
}

impl Sampler {
    fn new(spec: SampleSpec) -> Self {
        match spec {
            SampleSpec::Fps { fps, start, .. } => {
                Sampler::Fps(Tick::new(start, 1.0 / f64::from(fps)))
            }
            SampleSpec::Keyframes => Sampler::All,
            SampleSpec::Mixed { fps, fill } => Sampler::Mixed {
                fill,
                idx: 0,
                tick: None,
                step: 1.0 / f64::from(fps),
            },
        }
    }

    fn wants(&mut self, ts: f32, is_key: bool) -> bool {
        match self {
            Sampler::All => true,
            Sampler::Fps(tick) => tick.wants(ts),
            Sampler::Mixed {
                fill,
                idx,
                tick,
                step,
            } => {
                while fill.get(*idx).is_some_and(|r| ts >= r.end()) {
                    *idx += 1;
                    *tick = None;
                }
                match fill.get(*idx) {
                    Some(r) if ts >= r.start() => tick
                        .get_or_insert_with(|| Tick::new(r.start(), *step))
                        .wants(ts),
                    _ => is_key,
                }
            }
        }
    }
}

/// Demux-level packet admission per sampling mode: a packet the gate rejects
/// costs no decode at all.
enum PacketGate {
    /// Every video packet decodes ([`SampleSpec::Fps`]).
    All,
    /// Only key packets decode ([`SampleSpec::Keyframes`]).
    KeyOnly,
    /// Key packets always decode; non-key packets only inside a fill range
    /// ([`SampleSpec::Mixed`]). A fill range opens at a keyframe, so every
    /// frame inside it decodes from references that also passed the gate.
    KeyOrFill { fill: Vec<TimeRange>, idx: usize },
}

impl PacketGate {
    fn new(spec: &SampleSpec) -> Self {
        match spec {
            SampleSpec::Fps { .. } => PacketGate::All,
            SampleSpec::Keyframes => PacketGate::KeyOnly,
            SampleSpec::Mixed { fill, .. } => PacketGate::KeyOrFill {
                fill: fill.clone(),
                idx: 0,
            },
        }
    }

    fn admits(&mut self, packet: &Packet, video: &VideoStream) -> bool {
        match self {
            PacketGate::All => true,
            PacketGate::KeyOnly => packet.is_key(),
            PacketGate::KeyOrFill { fill, idx } => {
                if packet.is_key() {
                    return true;
                }
                // B-frame reordering jitters pts only within a GOP, never
                // across the keyframe opening the next fill range, so the
                // forward-only cursor is safe. A packet with no pts/dts
                // cannot be placed and is dropped.
                let Some(pts) = packet.pts().or(packet.dts()) else {
                    return false;
                };
                let ts = video.ts_seconds(pts);
                while fill.get(*idx).is_some_and(|r| ts >= r.end()) {
                    *idx += 1;
                }
                fill.get(*idx).is_some_and(|r| ts >= r.start())
            }
        }
    }
}

/// One dropped frame's identity, reported downstream as a diagnostic.
#[derive(Debug, Clone, Copy)]
struct Unplaced {
    /// Container timestamp of the dropped frame.
    ts: f32,
    /// Last released logical timestamp, when any.
    last: Option<f32>,
}

/// The verdict path between one decoded frame and the channel — segmenter,
/// range-end stop, sampler — composed exactly as the worker runs them.
/// Pure over `(ts, frame)` pairs; the worker adds only scaling and
/// sending, so the whole admission path (including the EOF flush) is
/// testable without a decoder.
struct FramePath {
    seg: Segmenter<ffmpeg::frame::Video>,
    sampler: Sampler,
    /// Stop decoding once the released logical clock reaches this bound
    /// (range sampling). Released instants are non-decreasing by
    /// construction, so the first release at or past the end is final.
    end: Option<f32>,
    /// Last released logical instant — context for drop reports.
    last: Option<f32>,
}

/// What the worker must do after feeding one frame (or EOF) to the path.
#[derive(Default)]
struct Step {
    /// Sampler-approved frames to scale and send, in logical order.
    send: Vec<(Ts, ffmpeg::frame::Video)>,
    /// Frames dropped off the timeline, to report downstream.
    dropped: Vec<Unplaced>,
    /// A committed clock seam, to report downstream.
    cut: Option<Discontinuity>,
    /// The released logical clock reached the range end.
    stop: bool,
}

impl FramePath {
    fn new(spec: SampleSpec, duration: Option<f32>) -> Self {
        let end = match &spec {
            SampleSpec::Fps { end, .. } => *end,
            _ => None,
        };
        // The segmenter's initial nominal gap: the tick step when fps is
        // known, the fallback for keyframe-paced decodes — both only until
        // the first released delta replaces them.
        let gap = match &spec {
            SampleSpec::Fps { fps, .. } | SampleSpec::Mixed { fps, .. } => 1.0 / f64::from(*fps),
            SampleSpec::Keyframes => FALLBACK_NOMINAL_GAP,
        };
        Self {
            seg: Segmenter::new(duration, gap),
            sampler: Sampler::new(spec),
            end,
            last: None,
        }
    }

    fn admit(&mut self, ts: ContainerTs, frame: ffmpeg::frame::Video) -> Step {
        let seg = self.seg.admit(ts, frame);
        self.place(seg)
    }

    /// End of stream: flush what the segmenter still buffers through the
    /// same range-end/sampler path as any other frame.
    fn finish(&mut self) -> Step {
        let seg = self.seg.finish();
        self.place(seg)
    }

    fn place(&mut self, seg: SegStep<ffmpeg::frame::Video>) -> Step {
        let mut step = Step {
            cut: seg.cut,
            ..Step::default()
        };
        for ts in seg.dropped {
            step.dropped.push(Unplaced {
                ts: ts.0 as f32,
                last: self.last,
            });
        }
        for (logical, frame) in seg.released {
            // `f32` overflow of a fenced logical instant is unreachable;
            // the finite-by-type boundary skips rather than panics.
            let Some(ts) = logical.to_ts() else { continue };
            self.last = Some(ts.get());
            if let Some(end) = self.end
                && ts.get() >= end
            {
                // Monotone releases: everything from here on is past the
                // end. Stop without sending.
                step.stop = true;
                continue;
            }
            if self.sampler.wants(ts.get(), frame.is_key()) {
                step.send.push((ts, frame));
            }
        }
        step
    }
}

/// The decode thread's state; consumed by [`Worker::run`].
struct Worker {
    ictx: super::InterruptibleInput,
    video: VideoStream,
    decoder: ffmpeg::decoder::Video,
    /// Demux-level packet gate: what it rejects never reaches the decoder.
    gate: PacketGate,
    /// Frame-level admission: segmenter → range end → sampler. What it
    /// sends is monotone on the logical clock by construction; what it
    /// drops or relabels is reported through the channel.
    path: FramePath,
    rescaler: Rescaler,
    tx: SyncSender<std::result::Result<Msg, Error>>,
    /// Checked once per packet: the consumer's drop requests an early exit.
    stop: Stop,
    frames_sent: u64,
    /// `Some(start)` while a demuxer-successful window seek has produced
    /// no decoded frame yet. A demuxer EOF in that state means the seek
    /// landed beyond every decodable frame (a poisoned head drags the
    /// container's index arbitrarily): the worker rewinds to the stream
    /// head for one announced sequential retry — the sampler discards
    /// everything before its first tick, so only cost changes, not
    /// correctness. Cleared by the first decoded frame.
    pending_seek: Option<f32>,
}

/// Whether to keep pumping after handling one decoded frame.
enum Flow {
    Continue,
    Stop,
}

impl Worker {
    fn run(&mut self) {
        if let Err(e) = self.pump() {
            let _ = self.tx.send(Err(e));
        }
        // Dropping tx (with self) closes the channel = clean EOF marker.
    }

    fn pump(&mut self) -> Result<()> {
        let mut packet = Packet::empty();
        let mut read_budget = ErrorBudget::new("reading packets");
        // Corrupt-packet tolerance is consecutive, not cumulative: every
        // accepted packet resets it, so scattered corruption over a long
        // video never accumulates into a mid-run abort.
        let mut decode_budget = ErrorBudget::new("decoding video packet");
        loop {
            if self.stop.is_tripped() {
                return Ok(());
            }
            match self.ictx.read_packet(&mut packet) {
                Ok(()) => read_budget.ok(),
                Err(ffmpeg::Error::Eof) => {
                    // A window seek that never yielded a frame landed past
                    // all content. Rewind to the stream head (the absolute
                    // minimum, not logical 0 — the declared origin may be
                    // the poison) and read sequentially, announced. Taken
                    // at most once: the flag never re-arms.
                    if let Some(start) = self.pending_seek.take()
                        && self.ictx.seek(i64::MIN, ..).is_ok()
                    {
                        self.decoder.flush();
                        if self.tx.send(Ok(Msg::SeekRecovered { start })).is_err() {
                            return Ok(());
                        }
                        continue;
                    }
                    break;
                }
                // Interrupt callback tripped: a requested stop, not a failure.
                Err(ffmpeg::Error::Exit) => return Ok(()),
                Err(e) => {
                    read_budget.fail(e)?;
                    continue;
                }
            }
            if packet.stream() != self.video.index {
                continue;
            }
            if !self.gate.admits(&packet, &self.video) {
                continue;
            }

            match self.decoder.send_packet(&packet) {
                Ok(()) => decode_budget.ok(),
                // A corrupt packet does not abandon the rest of the stream.
                Err(e) => {
                    decode_budget.fail(e)?;
                    continue;
                }
            }
            if matches!(self.drain()?, Flow::Stop) {
                return Ok(());
            }
        }

        // The EOF flush squeezes the decoder's buffered tail (B-frame
        // reordering delay) out; an unchecked failure here would silently
        // truncate the stream by exactly those frames. `Eof` (already
        // flushed) is not a failure — the drain below still empties what is
        // buffered.
        match self.decoder.send_eof() {
            Ok(()) | Err(ffmpeg::Error::Eof) => {}
            Err(e) => return Err(decode_error("flushing decoder at end of stream", e)),
        }
        self.drain()?;
        // Flush the segmenter: a forward tail challenger within the extent
        // fence is released in place (nothing left to contradict it); any
        // other unconfirmed challenger is dropped with signals; an
        // unconfirmed head — a stream shorter than one confirmation — is
        // released whole.
        let step = self.path.finish();
        if matches!(self.apply(step)?, Flow::Stop) {
            return Ok(());
        }

        if self.frames_sent == 0 && decode_budget.failures() > 0 {
            return Err(Error::new(
                ErrorCode::DecodeFailed,
                format!(
                    "no frames decoded ({} packet errors); is the input corrupt?",
                    decode_budget.failures()
                ),
            ));
        }
        Ok(())
    }

    /// Receive every frame currently available from the decoder, sample,
    /// scale, and send. `Flow::Stop` when past the range end or the consumer
    /// hung up.
    fn drain(&mut self) -> Result<Flow> {
        loop {
            // A fresh buffer per frame: the segmenter takes ownership of
            // each decoded frame, buffering it while its run awaits
            // confirmation.
            let mut frame = ffmpeg::frame::Video::empty();
            match classify_receive(self.decoder.receive_frame(&mut frame)) {
                Recv::Frame => self.pending_seek = None,
                Recv::Drained => return Ok(Flow::Continue),
                Recv::Broken(e) => return Err(decode_error("receiving decoded frame", e)),
            }

            // best-effort timestamp, pts fallback. No pts at all, or a pts
            // whose seconds conversion is non-finite, is unplaceable: skip.
            // Past here the container clock is finite by construction.
            let Some(pts) = frame.timestamp().or(frame.pts()) else {
                continue;
            };
            let secs = self.video.timeline.seconds(pts, self.video.time_base);
            if !secs.is_finite() {
                continue;
            }
            let step = self.path.admit(ContainerTs(secs), frame);
            if matches!(self.apply(step)?, Flow::Stop) {
                return Ok(Flow::Stop);
            }
        }
    }

    /// Perform one [`Step`]'s effects: report drops and seams, scale and
    /// send released frames. `Flow::Stop` when past the range end or the
    /// consumer hung up.
    fn apply(&mut self, step: Step) -> Result<Flow> {
        for unplaced in step.dropped {
            let msg = Msg::Dropped {
                ts: unplaced.ts,
                last: unplaced.last,
            };
            if self.tx.send(Ok(msg)).is_err() {
                return Ok(Flow::Stop);
            }
        }
        if let Some(cut) = step.cut {
            let msg = Msg::Discontinuity {
                container: cut.container.0,
                logical: cut.logical.0,
            };
            if self.tx.send(Ok(msg)).is_err() {
                return Ok(Flow::Stop);
            }
        }
        for (ts, frame) in step.send {
            let rgb = self.rescaler.rgb(&frame)?;
            if self.tx.send(Ok(Msg::Frame(RgbFrame { ts, rgb }))).is_err() {
                return Ok(Flow::Stop);
            }
            self.frames_sent += 1;
        }
        Ok(if step.stop {
            Flow::Stop
        } else {
            Flow::Continue
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tr(start: f32, end: f32) -> TimeRange {
        TimeRange::new(start, end).expect("valid test range")
    }

    fn mixed(fill: Vec<TimeRange>, fps: f32) -> Sampler {
        Sampler::new(SampleSpec::Mixed { fps, fill })
    }

    #[test]
    fn tick_takes_one_frame_per_step() {
        let mut tick = Tick::new(0.0, 1.0);
        assert!(tick.wants(0.0));
        assert!(!tick.wants(0.5), "still inside the first step");
        assert!(tick.wants(1.2));
        assert!(!tick.wants(1.8), "the 2.0 tick has not arrived");
        assert!(tick.wants(2.0));
    }

    #[test]
    fn tick_absorbs_float_noise_just_below_the_instant() {
        let mut tick = Tick::new(1.0, 1.0);
        assert!(tick.wants(0.999999), "owns the 1.0 tick despite pts noise");
        assert!(!tick.wants(1.5));
    }

    #[test]
    fn keeps_tick_instants_on_grid_after_hours_of_ticks() {
        // Ten hours of single-step advances at a step inexact in binary:
        // an accumulator would drift by whole seconds.
        let mut tick = Tick::new(0.0, 0.1);
        for _ in 0..360_000 {
            assert!(tick.wants((tick.next_instant() + 0.01) as f32));
        }
        let expected = 360_000.0 * 0.1;
        assert!(
            (tick.next_instant() - expected).abs() < 1e-3,
            "tick grid drifted: {} vs {expected}",
            tick.next_instant()
        );
    }

    /// Drive one container timestamp through the path and expose sent/dropped times.
    fn path_admit(path: &mut FramePath, v: f64) -> (Vec<f32>, Vec<f32>, bool) {
        let step = path.admit(ContainerTs(v), ffmpeg::frame::Video::empty());
        step_view(step)
    }

    fn step_view(step: Step) -> (Vec<f32>, Vec<f32>, bool) {
        (
            step.send.iter().map(|(t, _)| t.get()).collect(),
            step.dropped.iter().map(|u| u.ts).collect(),
            step.stop,
        )
    }

    fn range_path(end: f32) -> FramePath {
        // A dense tick grid so every released ts is sampled.
        FramePath::new(
            SampleSpec::Fps {
                fps: 1000.0,
                start: 0.0,
                end: Some(end),
            },
            None,
        )
    }

    #[test]
    fn stops_the_range_at_the_first_release_past_the_end() {
        let mut path = range_path(2.0);
        assert_eq!(path_admit(&mut path, 0.0).0, Vec::<f32>::new());
        assert_eq!(path_admit(&mut path, 0.5).0, Vec::<f32>::new());
        assert_eq!(path_admit(&mut path, 1.0).0, vec![0.0, 0.5, 1.0]);
        let (sent, _, stop) = path_admit(&mut path, 2.0);
        assert!(sent.is_empty(), "past the end sends nothing");
        assert!(stop, "monotone releases make the first crossing final");
    }

    #[test]
    fn does_not_stop_the_range_on_an_out_of_tolerance_outlier() {
        // A lone corrupt pts past the range end must not truncate the
        // range: it opens a challenger, the next legal frame kills it, and
        // sampling continues.
        let mut path = range_path(2.0);
        path_admit(&mut path, 0.0);
        path_admit(&mut path, 0.5);
        path_admit(&mut path, 1.0);
        assert_eq!(
            path_admit(&mut path, 15.0),
            (vec![], vec![], false),
            "the outlier is buffered, not trusted to stop anything"
        );
        assert_eq!(
            path_admit(&mut path, 1.1),
            (vec![1.1], vec![15.0], false),
            "the next legal frame contradicts the outlier"
        );
        assert_eq!(path_admit(&mut path, 1.2).0, vec![1.2], "keeps sampling");
    }

    #[test]
    fn finish_flushes_a_short_head_through_the_sampler() {
        let mut path = FramePath::new(SampleSpec::Keyframes, None);
        path_admit(&mut path, 0.0);
        path_admit(&mut path, 1.0);
        let (sent, dropped, stop) = step_view(path.finish());
        assert_eq!(sent, vec![0.0, 1.0], "a short clean stream is content");
        assert!(dropped.is_empty() && !stop);
    }

    #[test]
    fn finish_applies_the_range_end_to_the_flush() {
        let mut path = range_path(2.0);
        path_admit(&mut path, 0.0);
        path_admit(&mut path, 2.5);
        let (sent, _, stop) = step_view(path.finish());
        assert_eq!(sent, vec![0.0], "the in-range head frame is sent");
        assert!(stop, "the tail past the range end is released but not sent");
    }

    #[test]
    fn mixed_takes_only_keyframes_outside_fill_ranges() {
        let mut s = mixed(vec![tr(10.0, 20.0)], 1.0);
        assert!(s.wants(0.0, true));
        assert!(!s.wants(1.0, false), "dense area: non-key frames drop");
        assert!(s.wants(3.0, true));
    }

    #[test]
    fn mixed_ticks_inside_a_fill_range_keyframes_included() {
        let mut s = mixed(vec![tr(10.0, 13.0)], 1.0);
        assert!(s.wants(10.0, true), "region opens at its keyframe = tick 0");
        assert!(!s.wants(10.5, false), "between ticks");
        assert!(!s.wants(10.9, true), "a keyframe between ticks drops too");
        assert!(s.wants(11.0, false));
        assert!(s.wants(12.0, false));
    }

    #[test]
    fn mixed_resumes_keyframe_selection_past_the_range_end() {
        let mut s = mixed(vec![tr(10.0, 12.0)], 1.0);
        assert!(s.wants(10.0, true));
        assert!(s.wants(11.0, false));
        assert!(s.wants(12.0, true), "the keyframe closing the region");
        assert!(!s.wants(12.5, false), "dense again: non-key drops");
    }

    #[test]
    fn mixed_restarts_ticks_at_each_fill_range_start() {
        let mut s = mixed(vec![tr(0.0, 2.0), tr(10.3, 12.0)], 1.0);
        assert!(s.wants(0.0, false));
        assert!(s.wants(1.0, false));
        // Next region starts between the first region's tick instants: its
        // ticks are anchored at 10.3, not carried over from the 1.0 grid.
        assert!(s.wants(10.3, true));
        assert!(!s.wants(11.0, false), "next tick is 11.3");
        assert!(s.wants(11.3, false));
    }
}
