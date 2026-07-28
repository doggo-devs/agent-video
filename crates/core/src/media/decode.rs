// SPDX-License-Identifier: Apache-2.0
//! In-process frame decoding: a dedicated decode thread feeding a bounded
//! channel of small RGB frames.
//!
//! The decode thread and `sync_channel` pipeline decoding against scoring:
//! the channel's capacity bounds buffered frames and provides backpressure,
//! and a dropped receiver stops the producer.
//!
//! Timestamps come from the decoded frames themselves (`best-effort pts ×
//! time_base`, rebased to the stream start), produced by the same decoder
//! that produced the pixels. Before sampling, every decoded frame passes
//! [`TimelineGate`], which enforces the invariant everything downstream
//! (sampler ticks, range end, chunk placement) relies on: released
//! timestamps are non-decreasing *and* lie within the stream's own declared
//! extent ([`PlausibleWindow`]). Order is confirmed locally by each frame's
//! successor; containment is confirmed globally against the container
//! header probed before any frame is decoded — two corrupt frames can vouch
//! for each other's order, but not for an extent the container never
//! declared. Every dropped frame is reported downstream as an event, and a
//! timeline the gate cannot trust fails the run with a coded error: no
//! combination of inputs yields silent content loss on a successful exit.
//!
//! One failure shape earns a second chance: when the failing pass's own
//! audit shows the *rejected* frames forming a longer self-consistent
//! timeline than everything the declared origin released, the error
//! carries an [`AnchorFault`] indicting the clock anchor rather than the
//! frames, and the retry layer above [`FrameSource`] re-runs the whole
//! pass once with the elected origin ([`Anchor::Rebased`]) — container
//! reopened, seek reprojected, everything re-decoded. A rebased pass that
//! fails again is final: the anchor token has no third state, so no retry
//! loop exists to fabricate a timeline for a majority-corrupt input.

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

use super::scale::Rescaler;
use super::{
    ErrorBudget, Recv, Stop, Timeline, VideoStream, best_video_stream, classify_receive,
    decode_error, duration_seconds, open_input,
};

/// The clock anchor one decode pass trusts — the pass token of the origin
/// rebase, with two states and one legal transition.
///
/// Every first pass runs [`Anchor::Declared`] and behaves exactly as it
/// always has. Only when that pass fails with an [`AnchorFault`] does the
/// retry layer mint the [`Anchor::Rebased`] anchor for the single retry;
/// a rebased pass has no successor state, so "retry loop" is not a
/// representable program.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Anchor {
    /// Trust the container header: origin at the best video stream's
    /// declared start, extent from the declared duration. The fast path;
    /// normal files never leave it.
    Declared,
    /// Trust the origin elected by a failed declared pass's dropped-frame
    /// evidence, in seconds on the container's absolute clock.
    Rebased {
        /// The elected origin ([`AnchorFault::elected`]).
        origin: f64,
    },
}

/// The typed verdict a timeline failure carries when its own evidence
/// indicts the clock anchor rather than the frames: the dropped frames'
/// original pts form a monotone chain *longer* than everything the
/// declared origin managed to release, so the rejected timeline is more
/// self-consistent than the accepted one.
///
/// Travels as the [`Error`]'s source — the same channel, the same error
/// code ([`ErrorCode::InputCorrupt`]), zero new plumbing through the
/// decode thread — and is read back by the retry layer above
/// [`FrameSource`] via [`AnchorFault::in_error`].
#[derive(Debug, Clone, Copy)]
pub struct AnchorFault {
    /// The origin the failed pass trusted, seconds on the container's
    /// absolute clock.
    pub declared: f64,
    /// The elected origin: the dropped chain's head pts, seconds on the
    /// container's absolute clock.
    pub elected: f64,
}

impl std::fmt::Display for AnchorFault {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "the timeline anchor is suspect: declared origin {:.3}s, \
             the dropped frames' own chain elects {:.3}s",
            self.declared, self.elected
        )
    }
}

impl std::error::Error for AnchorFault {}

impl AnchorFault {
    /// The fault carried by `e`, when `e` is a timeline failure whose
    /// dropped-frame evidence elected a rebased origin; `None` for every
    /// other error — including timeline failures whose evidence sided
    /// with the declared anchor.
    pub fn in_error(e: &Error) -> Option<&AnchorFault> {
        std::error::Error::source(e)?.downcast_ref()
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

/// One message from the decode thread: a sampled frame, or the record of a
/// frame the timeline gate dropped. Drops ride the same channel because the
/// worker cannot borrow the caller's event sink across the thread boundary;
/// [`FrameSource::next_frame`] turns them into events on the consumer side.
enum Msg {
    Frame(RgbFrame),
    Dropped { ts: f32, committed: Option<f32> },
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
    /// Receives the first-drop warning ([`Event::UnplaceableTimestamp`]) and
    /// the end-of-decode drop summary ([`Event::TimestampsDropped`]).
    events: &'e dyn EventSink,
    /// Frames the decode pass dropped off the timeline so far.
    drops: u64,
}

impl<'e> FrameSource<'e> {
    /// Open `input` and start decoding per `spec`.
    ///
    /// Frames are pre-scaled so their shorter edge is `min_edge` pixels.
    /// `threads` caps the decoder's thread count (`None` = auto). Container,
    /// stream, decoder, and seek failures surface here, synchronously, with
    /// precise error codes; mid-stream failures surface from `next_frame`.
    /// `events` receives a [`Event::SeekFallback`] when a seek to a window
    /// start fails and decoding degrades to a sequential read, then rides
    /// along for the stream's lifetime to report timeline-gate drops (see
    /// [`FrameSource::next_frame`]).
    ///
    /// The [`Stop`] guarding this open is self-minted, not taken from the
    /// caller: [`Drop`] needs the exclusive right to trip it (a shared
    /// token tripped elsewhere would kill an unrelated decode mid-stream).
    /// A pipeline-level stop token (e.g. `ProcessOpts.stop`) therefore
    /// covers the probe/subtitle paths but not this decode thread; future
    /// ctrl-C wiring must bridge the two tokens explicitly.
    pub fn open(
        input: &Source,
        spec: SampleSpec,
        min_edge: u32,
        threads: Option<NonZeroUsize>,
        events: &'e dyn EventSink,
    ) -> Result<Self> {
        Self::open_anchored(input, spec, min_edge, threads, events, Anchor::Declared)
    }

    /// [`FrameSource::open`] with an explicit clock anchor — the door the
    /// origin-rebase retry walks through.
    ///
    /// [`Anchor::Declared`] is byte-identical to [`FrameSource::open`].
    /// [`Anchor::Rebased`] re-anchors the pass on the elected origin: every
    /// logical timestamp and the window seek below are projected through
    /// the corrected clock, and the plausible extent degrades to unbounded
    /// — a container whose declared origin was voted down cannot be
    /// trusted about its declared extent either (on MPEG-TS the duration
    /// derives from the same poisoned pts extremes; a poisoned head was
    /// observed to yield a 0.03s "duration" that would out-window every
    /// legal frame). The monotone gate, the consecutive cap, and the EOF
    /// audit still bound the damage, and a rebased pass that cannot form a
    /// majority timeline stays a final [`ErrorCode::InputCorrupt`].
    pub fn open_anchored(
        input: &Source,
        spec: SampleSpec,
        min_edge: u32,
        threads: Option<NonZeroUsize>,
        events: &'e dyn EventSink,
        anchor: Anchor,
    ) -> Result<Self> {
        // Built before the container so it can ride into libav's interrupt
        // callback, arming every blocking I/O call in this context.
        let stop = Stop::new();
        let mut ictx = open_input(input, &stop, &[])?;
        let mut video = best_video_stream(&ictx, input)?;
        // The timeline gate's global anchor. Declared: read from the
        // container header before any frame is decoded (a container that
        // declares no duration leaves the extent unbounded and the gate
        // falls back to purely local confirmation). Rebased: the elected
        // origin replaces the clock and the extent is unbounded — see
        // this constructor's contract.
        let window = match anchor {
            Anchor::Declared => {
                PlausibleWindow::from_declared(duration_seconds(&ictx, input).ok())
            }
            Anchor::Rebased { origin } => {
                video.timeline = Timeline::rebased(origin);
                PlausibleWindow::from_declared(None)
            }
        };

        // Seek before building the decoder so no flush is needed. The seek
        // speaks the container's absolute clock, so the logical 0-based
        // `start` is projected through the timeline. An unseekable stream
        // degrades to a decode-and-discard up to the sampler's first tick.
        if let SampleSpec::Fps { start, .. } = &spec
            && *start > 0.0
        {
            let target = video.timeline.seek_target(*start as f64).as_av_time();
            if let Err(e) = ictx.seek(target, ..target) {
                events.event(Event::SeekFallback {
                    start: *start,
                    error: e.to_string(),
                });
            }
        }

        let stream = ictx
            .stream(video.index)
            .expect("best stream index is valid");
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
        let clock = AnchorClock {
            declared: video.timeline.origin(),
            time_base: video.time_base,
        };
        let stop_flag = stop.clone();
        let thread = std::thread::Builder::new()
            .name("agent-video-decode".into())
            .spawn(move || {
                let mut worker = Worker {
                    ictx,
                    video,
                    decoder,
                    gate: PacketGate::new(&spec),
                    path: FramePath::new(spec, window, clock),
                    rescaler: Rescaler::new(min_edge),
                    tx,
                    stop: stop_flag,
                    frames_sent: 0,
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
    /// Frames arrive in non-decreasing timestamp order, within the stream's
    /// declared extent ([`TimelineGate`]). Dropped frames are never silent:
    /// the first drop emits [`Event::UnplaceableTimestamp`], a clean end of
    /// stream with any drops emits [`Event::TimestampsDropped`], and a
    /// timeline the gate cannot trust (see [`MAX_TS_OUTLIERS`]) surfaces
    /// here as [`ErrorCode::InputCorrupt`] — carrying an [`AnchorFault`]
    /// when the dropped frames' evidence indicts the clock anchor (the
    /// origin-rebase election; see the module docs).
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
                Ok(Ok(Msg::Dropped { ts, committed })) => {
                    if self.drops == 0 {
                        self.events.event(Event::UnplaceableTimestamp {
                            ts,
                            last: committed,
                        });
                    }
                    self.drops += 1;
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
                                self.events.event(Event::TimestampsDropped { count: self.drops });
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
/// Tick instants are indexed, never accumulated: the next instant is
/// `start + idx * step`, and advancing past a frame computes the new index
/// directly. Two failure shapes of the accumulating form (`next += step`)
/// are unrepresentable here: a corrupt far-future pts (finite by [`Ts`],
/// but arbitrary) has no loop to spin — the old advance never terminated
/// once `step` fell below the accumulator's ULP — and hours of ticks have
/// no error to accumulate, each instant being one multiplication away from
/// `start`.
///
/// The grid state is a ratchet: once advanced it never rewinds, so the
/// caller must feed non-decreasing timestamps. [`TimelineGate`] enforces
/// that upstream; the saturation in [`Tick::wants`] is a float-rounding
/// guard, not a corruption policy.
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

/// Ceiling shared by the gate's two failure counters. Below it, drops
/// degrade the output (with signal); above it, the timeline is declared
/// broken and the run fails — the same discipline as the demux-error cap
/// in [`crate::media`].
const MAX_TS_OUTLIERS: u32 = 64;

/// Earliest plausible presentation timestamp, in seconds. Legitimate
/// negative timestamps exist (edit-list pre-roll, priming frames) but stay
/// within single seconds of zero; this bound is generous by an order of
/// magnitude, so it only excludes wrap-around garbage.
const EARLIEST_PLAUSIBLE_TS: f32 = -60.0;

/// Slack over the declared duration: `max = duration * FACTOR + SECONDS`.
/// The slack covers header imprecision (rounded durations, files that grew
/// after the header was written) — not corruption magnitude. Corruption
/// that lands *inside* the slack is indistinguishable from a legal splice,
/// and its damage is bounded by the extent the container itself declared.
const EXTENT_SLACK_FACTOR: f32 = 2.0;
const EXTENT_SLACK_SECONDS: f32 = 300.0;

/// Unconfirmed forward reach granted to the final frame at EOF when the
/// confirmed timeline never jumped itself, in seconds. The final frame has
/// no successor to vouch for it, so *some* bound must exist — an unbounded
/// release lets a single corrupt tail pts append a garbage chunk (and, on
/// containers whose declared duration derives from pts extremes, the
/// extent window cannot catch it). Ten seconds admits every sane
/// inter-frame or inter-keyframe gap while capping what an unvouched tail
/// can fabricate at below chunk scale.
const EOF_JUMP_FLOOR: f32 = 10.0;

/// The stream's plausible presentation extent: the timeline gate's global
/// anchor, resolved from the container's own declared duration before any
/// frame is decoded.
///
/// Monotonicity is confirmable only locally — a successor vouches for
/// order, and two corrupt frames can vouch for each other — so order alone
/// cannot tell a legal timeline from a mutually-consistent corrupt one.
/// Containment can: the container states its extent up front, and a
/// corrupt pts (bit garbage, a wrapped 33-bit counter) lands orders of
/// magnitude outside it. Frames outside the extent are dropped before they
/// can be held, committed, or confirm anything; frames inside it are
/// treated as content, because at that scale corruption is
/// indistinguishable from a legal splice and bounded by the declared
/// extent itself.
///
/// Two honest degradations: a container that declares no duration
/// (fragmented MP4, raw streams) leaves the extent unbounded above, and a
/// container whose declared duration is itself computed from pts extremes
/// (MPEG-TS) can be widened *upward* by the very corruption — never
/// inward — so legal frames always stay plausible and the residual risk is
/// admitting in-extent corruption, which the monotone gate, the failure
/// counters, and the EOF jump rule still bound.
#[derive(Debug, Clone, Copy)]
struct PlausibleWindow {
    max: f32,
}

impl PlausibleWindow {
    fn from_declared(duration: Option<f32>) -> Self {
        Self {
            max: duration.map_or(f32::INFINITY, |d| {
                d.max(0.0) * EXTENT_SLACK_FACTOR + EXTENT_SLACK_SECONDS
            }),
        }
    }

    fn contains(self, ts: Ts) -> bool {
        (EARLIEST_PLAUSIBLE_TS..=self.max).contains(&ts.get())
    }
}

/// One dropped frame's identity, reported downstream as a diagnostic.
#[derive(Debug, Clone, Copy)]
struct Unplaced {
    ts: Ts,
    committed: Option<Ts>,
}

/// A decoded frame's identity at the gate: the logical timestamp the
/// pipeline speaks and the original stream pts it was minted from. The pts
/// rides along so a dropped frame's evidence survives verbatim in the
/// anchor election ([`DropRing`]) — logical seconds are `f32` and already
/// relative to the possibly-poisoned origin; the raw pts is neither.
#[derive(Debug, Clone, Copy)]
struct Stamp {
    ts: Ts,
    pts: i64,
}

/// The conversion facts the gate needs to express an elected origin on the
/// container's absolute clock: the origin this pass trusted (seconds) and
/// the stream's seconds per pts unit.
#[derive(Debug, Clone, Copy)]
struct AnchorClock {
    declared: f64,
    time_base: f64,
}

/// The original pts of the frames the gate dropped, ring-bounded at
/// [`MAX_TS_OUTLIERS`] entries — the same ceiling that bounds the gate's
/// failure counters bounds its evidence, so the buffer is saturated by the
/// time any counter can trip.
struct DropRing {
    pts: Vec<i64>,
    /// Overwrite cursor once the ring is full: index of the oldest entry.
    next: usize,
}

impl DropRing {
    fn new() -> Self {
        Self {
            pts: Vec::new(),
            next: 0,
        }
    }

    fn record(&mut self, pts: i64) {
        if self.pts.len() < MAX_TS_OUTLIERS as usize {
            self.pts.push(pts);
        } else {
            self.pts[self.next] = pts;
            self.next = (self.next + 1) % self.pts.len();
        }
    }

    /// The retained drops, oldest first.
    fn in_order(&self) -> Vec<i64> {
        let (tail, head) = self.pts.split_at(self.next);
        head.iter().chain(tail).copied().collect()
    }
}

/// A monotone chain elected from dropped pts: its first element (the
/// candidate origin) and its length (its vote weight).
#[derive(Debug, Clone, Copy, PartialEq)]
struct Chain {
    head: i64,
    len: u64,
}

/// The longest non-decreasing chain (subsequence) in `pts`, which must be
/// in arrival order. Ties prefer the earliest head — the more conservative
/// origin, keeping every chain member at a non-negative logical time.
/// O(n²) dynamic programming; `pts` is ring-bounded at
/// [`MAX_TS_OUTLIERS`] entries.
fn monotone_chain(pts: &[i64]) -> Option<Chain> {
    // best[i]: the strongest chain ending at pts[i].
    let mut best: Vec<Chain> = Vec::with_capacity(pts.len());
    for (i, &x) in pts.iter().enumerate() {
        let mut b = Chain { head: x, len: 1 };
        for j in 0..i {
            if pts[j] <= x {
                let c = Chain {
                    head: best[j].head,
                    len: best[j].len + 1,
                };
                if c.len > b.len || (c.len == b.len && c.head < b.head) {
                    b = c;
                }
            }
        }
        best.push(b);
    }
    best.into_iter()
        .reduce(|a, b| match (b.len > a.len, b.len == a.len && b.head < a.head) {
            (true, _) | (false, true) => b,
            _ => a,
        })
}

/// What one gate step yielded: at most one released frame and at most one
/// drop record; both `None` when the frame was held.
struct GateStep {
    release: Option<(Ts, ffmpeg::frame::Video)>,
    dropped: Option<Unplaced>,
}

impl GateStep {
    fn hold() -> Self {
        Self {
            release: None,
            dropped: None,
        }
    }
}

/// Admits decoded frames onto a monotone timeline inside the stream's
/// declared extent ([`PlausibleWindow`]).
///
/// A frame outside the extent is dropped immediately: it can never be
/// held, never commit the timeline, and never confirm another frame — so
/// mutually-vouching corrupt pairs die at the anchor instead of hijacking
/// `committed`. A frame inside the extent is released one frame late, once
/// its successor confirms it (`successor ts >= its ts`). An in-extent
/// outlier is identifiable only in retrospect, by three points — the
/// committed timeline `c` (last released ts), the held frame `h`, and the
/// newcomer `n`:
///
/// - `n >= h`: the stream vouches for `h`; release it. A legitimate
///   forward jump (splice, timelapse gap) is exactly this case.
/// - `c <= n < h`: `n` continues the committed timeline, so `h` was the
///   outlier; drop `h`.
/// - `n < c`: `n` fell behind frames already released; drop `n`.
///
/// At the stream head (`committed == None`) two contradicting points
/// genuinely cannot be attributed; the gate sides with the successor and
/// *accounts* the drop — the honest contract is a conservative pass with a
/// signal, not a guess dressed up as knowledge. The extent anchor makes
/// this ambiguity rare (real head corruption is usually out of extent),
/// and the damage is one frame either way.
///
/// Failure accounting distinguishes what a drop means:
///
/// - `implausible` (out of extent): certainly corrupt; dropping them loses
///   no placeable content, so they only feed the consecutive counter and
///   the end-of-stream sanity check.
/// - `conflicted` (in extent): frames that *could* have been content. When
///   they exceed [`MAX_TS_OUTLIERS`] *and* reach a third of all placeable
///   frames (`2 * conflicted >= released`), the output's completeness is
///   no longer vouched for and the run fails — alternating corruption can
///   no longer buy unlimited silent drops with interleaved releases.
/// - `consecutive` (either kind, reset by every release): a timeline that
///   never recovers — e.g. a poisoned `committed` shadowing every later
///   legal frame — fails fast instead of draining the stream.
///
/// All failures are [`ErrorCode::InputCorrupt`]; every drop below the
/// thresholds is reported to the consumer (see [`Msg::Dropped`]), so
/// degraded output is never silent. Every dropped frame's original pts is
/// additionally retained in a [`DropRing`]; a failure verdict holds the
/// anchor election over that evidence ([`TimelineGate::elect`]) and, when
/// it passes, carries the resulting [`AnchorFault`] to the retry layer.
struct TimelineGate {
    window: PlausibleWindow,
    /// Converts elected raw pts back onto the container's absolute clock.
    clock: AnchorClock,
    /// The frame awaiting its successor's confirmation.
    held: Option<(Stamp, ffmpeg::frame::Video)>,
    /// Timestamp of the last released frame: the committed timeline.
    committed: Option<Ts>,
    /// Consecutive drops since the last release.
    consecutive: u32,
    /// Cumulative in-extent drops: frames that could have been content.
    conflicted: u64,
    /// Cumulative out-of-extent drops: certainly-corrupt timestamps.
    implausible: u64,
    /// Cumulative released frames.
    released: u64,
    /// Largest forward gap between consecutively released frames: the
    /// stream's own jump precedent, consulted by the EOF release rule.
    max_released_gap: f32,
    /// Original pts of every dropped frame (most recent
    /// [`MAX_TS_OUTLIERS`]): the anchor election's evidence.
    drops: DropRing,
}

impl TimelineGate {
    fn new(window: PlausibleWindow, clock: AnchorClock) -> Self {
        Self {
            window,
            clock,
            held: None,
            committed: None,
            consecutive: 0,
            conflicted: 0,
            implausible: 0,
            released: 0,
            max_released_gap: 0.0,
            drops: DropRing::new(),
        }
    }

    /// Feed the next decoded frame; get back its confirmed predecessor when
    /// that predecessor earned release, the drop record when a frame fell
    /// off the timeline, or the failure verdict.
    fn admit(&mut self, stamp: Stamp, frame: ffmpeg::frame::Video) -> Result<GateStep> {
        let ts = stamp.ts;
        if !self.window.contains(ts) {
            self.implausible += 1;
            return self.charge(stamp);
        }
        let Some((held, held_frame)) = self.held.take() else {
            self.held = Some((stamp, frame));
            return Ok(GateStep::hold());
        };
        if ts >= held.ts {
            self.held = Some((stamp, frame));
            Ok(GateStep {
                release: Some(self.release(held.ts, held_frame)),
                dropped: None,
            })
        } else if self.committed.is_some_and(|c| ts < c) {
            // The newcomer fell behind released frames: it is the outlier.
            self.held = Some((held, held_frame));
            self.conflicted += 1;
            self.charge(stamp)
        } else {
            // The newcomer continues the committed timeline (or no timeline
            // is committed yet — the unattributable stream head): the held
            // frame it contradicts was the outlier.
            self.held = Some((stamp, frame));
            self.conflicted += 1;
            self.charge(held)
        }
    }

    /// End of stream: the final frame has no successor to vouch for it.
    /// Release it only when it stays within the stream's own jump
    /// precedent; an unvouched jump beyond precedent is dropped (with
    /// signal) rather than fabricating timeline the stream never showed.
    /// Then verify the whole pass: more drops than releases means the
    /// timestamps were garbage wholesale, not scattered noise.
    fn finish(&mut self) -> Result<GateStep> {
        let step = match self.held.take() {
            None => GateStep::hold(),
            Some((stamp, frame)) => match self.committed {
                Some(c) if stamp.ts.get() - c.get() > self.eof_allowance() => {
                    self.conflicted += 1;
                    self.charge(stamp)?
                }
                _ => GateStep {
                    release: Some(self.release(stamp.ts, frame)),
                    dropped: None,
                },
            },
        };
        let dropped = self.conflicted + self.implausible;
        if dropped > self.released {
            return Err(self.corrupt(format!(
                "{dropped} of {} decoded frames fell off the plausible timeline; \
                 the input's timestamps are broken",
                dropped + self.released
            )));
        }
        Ok(step)
    }

    /// Unconfirmed forward reach granted at EOF: twice the largest jump the
    /// confirmed timeline itself exhibited, floored at [`EOF_JUMP_FLOOR`].
    /// A stream that spliced before may splice at its end; one that never
    /// jumped cannot suddenly do so on its very last, unvouched frame.
    fn eof_allowance(&self) -> f32 {
        (2.0 * self.max_released_gap).max(EOF_JUMP_FLOOR)
    }

    fn release(&mut self, ts: Ts, frame: ffmpeg::frame::Video) -> (Ts, ffmpeg::frame::Video) {
        if let Some(c) = self.committed {
            self.max_released_gap = self.max_released_gap.max(ts.get() - c.get());
        }
        self.committed = Some(ts);
        self.consecutive = 0;
        self.released += 1;
        (ts, frame)
    }

    /// Account one dropped frame and pass verdict on the counters.
    fn charge(&mut self, offending: Stamp) -> Result<GateStep> {
        self.drops.record(offending.pts);
        self.consecutive += 1;
        if self.consecutive > MAX_TS_OUTLIERS {
            return Err(self.corrupt(format!(
                "{} consecutive frames off the timeline (last released {:?}, \
                 offending {}); the input's timeline never recovers",
                self.consecutive,
                self.committed.map(Ts::get),
                offending.ts.get()
            )));
        }
        if self.conflicted > u64::from(MAX_TS_OUTLIERS) && 2 * self.conflicted >= self.released {
            return Err(self.corrupt(format!(
                "{} of {} plausible frames conflicted with the timeline \
                 (last released {:?}); the input's timeline is untrustworthy",
                self.conflicted,
                self.conflicted + self.released,
                self.committed.map(Ts::get)
            )));
        }
        Ok(GateStep {
            release: None,
            dropped: Some(Unplaced {
                ts: offending.ts,
                committed: self.committed,
            }),
        })
    }

    /// A timeline failure, carrying the anchor election when it passes.
    fn corrupt(&self, reason: String) -> Error {
        let err = Error::new(ErrorCode::InputCorrupt, reason);
        match self.elect() {
            Some(fault) => err.with_source(fault),
            None => err,
        }
    }

    /// The anchor election over this pass's evidence: `Some` exactly when
    /// the dropped frames' original pts form a monotone chain longer than
    /// the released count — the rejected timeline is more self-consistent
    /// than the accepted one, so the failure indicts the anchor, not the
    /// frames. Threshold-free by design: a majority vote between the two
    /// timelines the pass actually observed.
    fn elect(&self) -> Option<AnchorFault> {
        let chain = monotone_chain(&self.drops.in_order())?;
        (chain.len > self.released).then_some(AnchorFault {
            declared: self.clock.declared,
            elected: chain.head as f64 * self.clock.time_base,
        })
    }
}

/// The verdict path between one decoded frame and the channel — timeline
/// gate, range-end stop, sampler — composed exactly as the worker runs
/// them. Pure over `(ts, frame)` pairs; the worker adds only scaling and
/// sending, so the whole admission path (including the EOF release) is
/// testable without a decoder.
struct FramePath {
    gate: TimelineGate,
    sampler: Sampler,
    /// Stop decoding once the released timeline confirmably reaches this
    /// bound (range sampling).
    end: Option<f32>,
    /// One released frame has reached `end`; the next released frame
    /// confirms the stop. Stopping is irreversible and unsignaled, so it
    /// demands the same standard as a release plus one: released
    /// timestamps are non-decreasing, so a second released frame past the
    /// end cannot be the mutually-vouched half of an isolated corrupt
    /// pair — that pair's survivor stays held and is contradicted by the
    /// next legal frame instead of truncating the range.
    past_end: bool,
}

/// What the worker must do after feeding one frame (or EOF) to the path.
struct Step {
    /// Sampler-approved frame to scale and send.
    send: Option<(Ts, ffmpeg::frame::Video)>,
    /// A frame dropped off the timeline, to report downstream.
    dropped: Option<Unplaced>,
    /// The released timeline has confirmably passed the range end.
    stop: bool,
}

impl FramePath {
    fn new(spec: SampleSpec, window: PlausibleWindow, clock: AnchorClock) -> Self {
        let end = match &spec {
            SampleSpec::Fps { end, .. } => *end,
            _ => None,
        };
        Self {
            gate: TimelineGate::new(window, clock),
            sampler: Sampler::new(spec),
            end,
            past_end: false,
        }
    }

    fn admit(&mut self, stamp: Stamp, frame: ffmpeg::frame::Video) -> Result<Step> {
        let gate = self.gate.admit(stamp, frame)?;
        Ok(self.place(gate))
    }

    /// End of stream: release what the gate still holds (subject to its
    /// EOF rule) through the same range-end/sampler path as any other
    /// frame.
    fn finish(&mut self) -> Result<Step> {
        let gate = self.gate.finish()?;
        Ok(self.place(gate))
    }

    fn place(&mut self, gate: GateStep) -> Step {
        let mut step = Step {
            send: None,
            dropped: gate.dropped,
            stop: false,
        };
        let Some((ts, frame)) = gate.release else {
            return step;
        };
        if let Some(end) = self.end
            && ts.get() >= end
        {
            step.stop = self.past_end;
            self.past_end = true;
            return step;
        }
        if self.sampler.wants(ts.get(), frame.is_key()) {
            step.send = Some((ts, frame));
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
    /// Frame-level admission: timeline gate → range end → sampler. What it
    /// sends is non-decreasing and in-extent by construction; what it drops
    /// is reported through the channel.
    path: FramePath,
    rescaler: Rescaler,
    tx: SyncSender<std::result::Result<Msg, Error>>,
    /// Checked once per packet: the consumer's drop requests an early exit.
    stop: Stop,
    frames_sent: u64,
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
                Err(ffmpeg::Error::Eof) => break,
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
        // The last decoded frame has no successor to vouch for it; the
        // gate's EOF rule decides whether it is released or dropped, and
        // the whole pass's drop accounting is verified.
        let step = self.path.finish()?;
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
            // A fresh buffer per frame: the gate takes ownership of each
            // decoded frame, holding it until its successor confirms it.
            let mut frame = ffmpeg::frame::Video::empty();
            match classify_receive(self.decoder.receive_frame(&mut frame)) {
                Recv::Frame => {}
                Recv::Drained => return Ok(Flow::Continue),
                Recv::Broken(e) => return Err(decode_error("receiving decoded frame", e)),
            }

            // best-effort timestamp, pts fallback. No pts at all, or a pts
            // whose seconds conversion is non-finite (absurd pts × time_base
            // overflowing f32) = unplaceable, skip. This is the single mint
            // point of [`Ts`]: past here timestamps are finite by type.
            let Some(pts) = frame.timestamp().or(frame.pts()) else {
                continue;
            };
            let Some(ts) = Ts::new(self.video.ts_seconds(pts)) else {
                continue;
            };
            let step = self.path.admit(Stamp { ts, pts }, frame)?;
            if matches!(self.apply(step)?, Flow::Stop) {
                return Ok(Flow::Stop);
            }
        }
    }

    /// Perform one [`Step`]'s effects: report a drop, scale and send a
    /// released frame. `Flow::Stop` when past the range end or the consumer
    /// hung up.
    fn apply(&mut self, step: Step) -> Result<Flow> {
        if let Some(unplaced) = step.dropped
            && self
                .tx
                .send(Ok(Msg::Dropped {
                    ts: unplaced.ts.get(),
                    committed: unplaced.committed.map(Ts::get),
                }))
                .is_err()
        {
            return Ok(Flow::Stop);
        }
        if let Some((ts, frame)) = step.send {
            let rgb = self.rescaler.rgb(&frame)?;
            if self.tx.send(Ok(Msg::Frame(RgbFrame { ts, rgb }))).is_err() {
                return Ok(Flow::Stop);
            }
            self.frames_sent += 1;
        }
        Ok(if step.stop { Flow::Stop } else { Flow::Continue })
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
        // Ten hours of single-step advances at a step that is inexact in
        // binary: the old f32 accumulator drifted by whole seconds here;
        // indexed instants stay on the grid.
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

    /// Pts unit used by the test clock: pts are minted from seconds so the
    /// gate's raw-pts evidence stays exact for the values tests use.
    const TB: f64 = 1.0 / 90_000.0;

    fn clock() -> AnchorClock {
        AnchorClock {
            declared: 0.0,
            time_base: TB,
        }
    }

    fn stamp(v: f32) -> Stamp {
        Stamp {
            ts: Ts::new(v).expect("finite test ts"),
            pts: (f64::from(v) / TB).round() as i64,
        }
    }

    /// A gate with no extent anchor: purely local (successor) confirmation.
    fn unanchored() -> TimelineGate {
        TimelineGate::new(PlausibleWindow::from_declared(None), clock())
    }

    /// A gate anchored to a container declaring `duration` seconds.
    fn anchored(duration: f32) -> TimelineGate {
        TimelineGate::new(PlausibleWindow::from_declared(Some(duration)), clock())
    }

    /// Feed one timestamp through the gate; map a released frame to its ts.
    fn admit(gate: &mut TimelineGate, v: f32) -> Result<Option<f32>> {
        gate.admit(stamp(v), ffmpeg::frame::Video::empty())
            .map(|step| step.release.map(|(t, _)| t.get()))
    }

    fn release(gate: &mut TimelineGate, v: f32) -> Option<f32> {
        admit(gate, v).expect("within the gate's failure thresholds")
    }

    fn finish(gate: &mut TimelineGate) -> Option<f32> {
        gate.finish()
            .expect("finish verdict passes")
            .release
            .map(|(t, _)| t.get())
    }

    #[test]
    fn releases_frames_once_their_successor_confirms_them() {
        let mut gate = unanchored();
        assert_eq!(release(&mut gate, 0.0), None, "first frame is held");
        assert_eq!(release(&mut gate, 0.5), Some(0.0));
        assert_eq!(release(&mut gate, 1.0), Some(0.5));
        assert_eq!(
            finish(&mut gate),
            Some(1.0),
            "EOF releases the final in-precedent frame"
        );
        assert_eq!(finish(&mut gate), None);
    }

    #[test]
    fn drops_a_far_future_outlier_and_keeps_the_frames_after_it() {
        // The adversarial shape: one corrupt pts must die alone instead of
        // making every later legal frame unsampleable (silent truncation).
        // Unanchored, so the three-point rule alone must handle it.
        let mut gate = unanchored();
        assert_eq!(release(&mut gate, 0.0), None);
        assert_eq!(release(&mut gate, 1e30), Some(0.0), "not yet contradicted");
        assert_eq!(
            release(&mut gate, 0.5),
            None,
            "0.5 continues the committed timeline, so 1e30 is dropped"
        );
        assert_eq!(release(&mut gate, 1.0), Some(0.5), "legal frames flow on");
        assert_eq!(finish(&mut gate), Some(1.0));
    }

    #[test]
    fn drops_implausible_timestamps_before_they_can_vouch_for_anything() {
        // The extent anchor: an out-of-extent frame is never held, never
        // committed, and never confirms the held frame — so a corrupt
        // successor cannot release a frame, and a corrupt pair cannot
        // hijack the committed timeline.
        let mut gate = anchored(10.0);
        assert_eq!(release(&mut gate, 1.0), None, "held");
        let step = gate
            .admit(stamp(86_400.0), ffmpeg::frame::Video::empty())
            .expect("one implausible drop is within thresholds");
        assert!(step.release.is_none(), "corrupt frame confirms nothing");
        assert_eq!(
            step.dropped.map(|u| u.ts.get()),
            Some(86_400.0),
            "the drop is reported, not silent"
        );
        assert_eq!(
            release(&mut gate, 1.5),
            Some(1.0),
            "the held frame survives the corrupt interloper"
        );
    }

    #[test]
    fn survives_a_mutually_vouching_corrupt_pair_via_the_extent_anchor() {
        // Two increasing corrupt frames vouch for each other's order; only
        // the extent anchor can tell them from a legal splice. With it,
        // committed is never poisoned and every legal frame flows.
        let mut gate = anchored(10.0);
        release(&mut gate, 0.0);
        assert_eq!(release(&mut gate, 1e29), None, "out of extent: dropped");
        assert_eq!(release(&mut gate, 1e30), None, "out of extent: dropped");
        assert_eq!(release(&mut gate, 0.5), Some(0.0));
        assert_eq!(release(&mut gate, 1.0), Some(0.5));
        assert_eq!(finish(&mut gate), Some(1.0));
    }

    #[test]
    fn admits_a_forward_jump_its_successors_follow() {
        // A splice / timelapse gap: the jump target is vouched for by the
        // frame after it and lies within the declared extent.
        let mut gate = anchored(4000.0);
        assert_eq!(release(&mut gate, 0.0), None);
        assert_eq!(release(&mut gate, 1.0), Some(0.0));
        assert_eq!(release(&mut gate, 3600.0), Some(1.0));
        assert_eq!(release(&mut gate, 3600.5), Some(3600.0), "the jump holds");
    }

    #[test]
    fn drops_a_backward_outlier_below_the_committed_timeline() {
        let mut gate = unanchored();
        assert_eq!(release(&mut gate, 5.0), None);
        assert_eq!(release(&mut gate, 6.0), Some(5.0));
        assert_eq!(release(&mut gate, 2.0), None, "behind the timeline: drop");
        assert_eq!(release(&mut gate, 7.0), Some(6.0), "the held frame stays");
    }

    #[test]
    fn trusts_the_successor_when_no_timeline_is_committed() {
        // The stream head is genuinely unattributable: two points, no
        // anchor between them. The contract is a conservative pass — side
        // with the successor — plus an accounted, reported drop; either
        // choice can be wrong and either way loses exactly one frame.
        let mut gate = unanchored();
        assert_eq!(release(&mut gate, 10.0), None);
        let step = gate
            .admit(stamp(5.0), ffmpeg::frame::Video::empty())
            .expect("one head drop is within thresholds");
        assert!(step.release.is_none(), "10.0 is dropped");
        assert_eq!(
            step.dropped.map(|u| u.ts.get()),
            Some(10.0),
            "the ambiguous head drop is reported"
        );
        assert_eq!(release(&mut gate, 5.5), Some(5.0));
    }

    #[test]
    fn fails_when_an_in_extent_pair_hijacks_the_timeline() {
        // In-extent corruption that poisons `committed` shadows every later
        // legal frame; the consecutive counter must end that loudly instead
        // of draining the stream into silence.
        let mut gate = anchored(3600.0);
        release(&mut gate, 0.0);
        release(&mut gate, 3000.0);
        release(&mut gate, 3001.0);
        let mut verdict = Ok(None);
        for i in 0..=MAX_TS_OUTLIERS {
            verdict = admit(&mut gate, 1.0 + i as f32 * 0.1);
            if verdict.is_err() {
                break;
            }
        }
        let err = verdict.expect_err("a never-recovering timeline must fail");
        assert_eq!(err.code, ErrorCode::InputCorrupt);
    }

    #[test]
    fn heals_scattered_conflicts_below_the_density_threshold() {
        // Isolated in-extent outliers over a long stream cost nothing: each
        // is contradicted by the next legal frame, drops stay far below the
        // release count, and no failure accrues.
        let mut gate = unanchored();
        release(&mut gate, 0.0);
        for i in 1..=60u32 {
            let t = i as f32;
            assert_eq!(release(&mut gate, t), Some(t - 1.0));
            assert_eq!(release(&mut gate, t - 2.0), None, "outlier dropped");
        }
        assert_eq!(release(&mut gate, 61.0), Some(60.0), "stream flows on");
    }

    #[test]
    fn fails_when_conflicted_drops_stop_being_isolated() {
        // Alternating good/bad frames interleave releases with drops, so a
        // consecutive counter never fires — the gen-3 blind spot. The
        // cumulative density rule fails the run once in-extent drops exceed
        // the budget AND a third of all placeable frames.
        let mut gate = unanchored();
        release(&mut gate, 0.0);
        let mut verdict = Ok(None);
        'outer: for i in 1..=(2 * MAX_TS_OUTLIERS) {
            let t = i as f32;
            for v in [t, t - 0.5] {
                verdict = admit(&mut gate, v);
                if verdict.is_err() {
                    break 'outer;
                }
            }
        }
        let err = verdict.expect_err("pervasive alternating conflicts must fail");
        assert_eq!(err.code, ErrorCode::InputCorrupt);
    }

    #[test]
    fn fails_at_eof_when_drops_outnumber_releases() {
        // A short stream whose frames are mostly off-timeline must not
        // report success on the residue: finish() audits the whole pass.
        let mut gate = anchored(10.0);
        release(&mut gate, 0.0);
        release(&mut gate, 0.5);
        for i in 0..4 {
            assert_eq!(release(&mut gate, 90_000.0 + i as f32), None);
        }
        let err = gate
            .finish()
            .map(|_| ())
            .expect_err("garbage-majority stream fails");
        assert_eq!(err.code, ErrorCode::InputCorrupt);
    }

    #[test]
    fn releases_the_final_frame_within_the_streams_gap_precedent() {
        let mut gate = unanchored();
        release(&mut gate, 0.0);
        release(&mut gate, 1.0);
        assert_eq!(
            finish(&mut gate),
            Some(1.0),
            "a clean tail is released, not truncated"
        );
    }

    #[test]
    fn releases_an_eof_jump_the_stream_has_precedent_for() {
        // Keyframe-sparse streams gap by minutes; the EOF rule must scale
        // with the stream's own observed jumps, not a fixed constant.
        let mut gate = anchored(400.0);
        release(&mut gate, 0.0);
        release(&mut gate, 100.0);
        release(&mut gate, 200.0);
        assert_eq!(finish(&mut gate), Some(200.0), "100s precedent covers it");
    }

    #[test]
    fn drops_an_unconfirmed_eof_jump_without_precedent() {
        // S2's shape: a corrupt pts on the very last frame has no successor
        // to contradict it. A stream that never jumped cannot suddenly jump
        // at EOF; the frame is dropped with a report, not released into a
        // garbage chunk.
        let mut gate = unanchored();
        release(&mut gate, 0.0);
        release(&mut gate, 0.5);
        release(&mut gate, 500.0);
        let step = gate.finish().expect("one tail drop is within thresholds");
        assert!(step.release.is_none(), "the unvouched jump is not released");
        assert_eq!(
            step.dropped.map(|u| u.ts.get()),
            Some(500.0),
            "the tail drop is reported"
        );
    }

    #[test]
    fn releases_a_lone_frame_at_eof_without_an_anchor() {
        // A single-frame stream has no committed timeline to measure a jump
        // against; the extent window is the only filter.
        let mut gate = anchored(10.0);
        release(&mut gate, 5.0);
        assert_eq!(finish(&mut gate), Some(5.0));
    }

    #[test]
    fn rebases_when_dropped_chain_outweighs_released() {
        // The anchor-fault signature: a released minority against a longer
        // monotone chain of drops. The EOF audit fails the pass, and the
        // error carries the election — the chain's head pts, back on the
        // container's absolute clock, is the corrected origin.
        let mut gate = anchored(10.0);
        release(&mut gate, 0.0);
        release(&mut gate, 0.5);
        for i in 0..4 {
            assert_eq!(release(&mut gate, 90_000.0 + i as f32), None, "dropped");
        }
        let err = gate
            .finish()
            .map(|_| ())
            .expect_err("drops outnumber releases");
        assert_eq!(err.code, ErrorCode::InputCorrupt);
        let fault = AnchorFault::in_error(&err)
            .expect("a 4-drop chain against 2 releases elects a rebase");
        assert_eq!(fault.declared, 0.0);
        assert!(
            (fault.elected - 90_000.0).abs() < 1e-3,
            "the chain head is the elected origin: {}",
            fault.elected
        );
    }

    #[test]
    fn withholds_the_fault_when_releases_outnumber_the_chain() {
        // The election is a majority vote: when the accepted timeline is
        // the longer one, a timeline failure stays a plain InputCorrupt —
        // the guarantee that keeps release-majority inputs (edit-list
        // pre-roll among them) from ever being re-anchored.
        let mut gate = unanchored();
        release(&mut gate, 0.0);
        let mut verdict = Ok(None);
        'outer: for i in 1..=(2 * MAX_TS_OUTLIERS) {
            let t = i as f32;
            for v in [t, t - 0.5] {
                verdict = admit(&mut gate, v);
                if verdict.is_err() {
                    break 'outer;
                }
            }
        }
        let err = verdict.expect_err("pervasive alternating conflicts must fail");
        assert!(
            AnchorFault::in_error(&err).is_none(),
            "released majority: the declared anchor stands"
        );
    }

    #[test]
    fn carries_the_fault_out_of_a_consecutive_failure() {
        // The budget branch: a poisoned committed timeline shadows every
        // later frame until the consecutive counter trips; the shadowed
        // frames' own chain wins the election. The first drop was evicted
        // from the full ring, so the retained chain heads at the second.
        let mut gate = anchored(3600.0);
        release(&mut gate, 3000.0);
        release(&mut gate, 3001.0);
        let mut verdict = Ok(None);
        for i in 0..=MAX_TS_OUTLIERS {
            verdict = admit(&mut gate, 1.0 + i as f32 * 0.1);
            if verdict.is_err() {
                break;
            }
        }
        let err = verdict.expect_err("a never-recovering timeline must fail");
        let fault = AnchorFault::in_error(&err).expect("a 64-chain against 1 release elects");
        assert!(
            (fault.elected - 1.1).abs() < 1e-3,
            "elected origin {} heads the retained chain",
            fault.elected
        );
    }

    #[test]
    fn finds_the_longest_chain_across_noise() {
        let chain = monotone_chain(&[10, 1, 2, 100, 3]).expect("non-empty evidence");
        assert_eq!(chain, Chain { head: 1, len: 3 });
    }

    #[test]
    fn prefers_the_earliest_head_between_equal_chains() {
        let chain = monotone_chain(&[5, 6, 1, 2]).expect("non-empty evidence");
        assert_eq!(chain, Chain { head: 1, len: 2 });
    }

    #[test]
    fn elects_nothing_from_no_evidence() {
        assert!(monotone_chain(&[]).is_none());
    }

    #[test]
    fn evicts_the_oldest_drops_beyond_ring_capacity() {
        let mut ring = DropRing::new();
        let cap = i64::from(MAX_TS_OUTLIERS);
        for pts in 0..cap + 2 {
            ring.record(pts);
        }
        let ordered = ring.in_order();
        assert_eq!(ordered.len(), MAX_TS_OUTLIERS as usize);
        assert_eq!(ordered.first().copied(), Some(2), "oldest two evicted");
        assert_eq!(ordered.last().copied(), Some(cap + 1), "newest retained");
    }

    /// Drive one timestamp through a [`FramePath`]; map the step to
    /// `(sent ts, dropped ts, stop)`.
    fn path_admit(path: &mut FramePath, v: f32) -> Result<(Option<f32>, Option<f32>, bool)> {
        path.admit(stamp(v), ffmpeg::frame::Video::empty())
            .map(step_view)
    }

    fn step_view(step: Step) -> (Option<f32>, Option<f32>, bool) {
        (
            step.send.map(|(t, _)| t.get()),
            step.dropped.map(|u| u.ts.get()),
            step.stop,
        )
    }

    fn range_path(end: f32) -> FramePath {
        // A dense tick grid so every distinct released ts is sampled.
        FramePath::new(
            SampleSpec::Fps {
                fps: 1000.0,
                start: 0.0,
                end: Some(end),
            },
            PlausibleWindow::from_declared(None),
            clock(),
        )
    }

    #[test]
    fn stops_only_after_two_released_frames_confirm_the_range_end() {
        let mut path = range_path(2.0);
        assert_eq!(path_admit(&mut path, 0.0).expect("ok"), (None, None, false));
        assert_eq!(
            path_admit(&mut path, 1.0).expect("ok"),
            (Some(0.0), None, false)
        );
        assert_eq!(
            path_admit(&mut path, 2.5).expect("ok"),
            (Some(1.0), None, false)
        );
        assert_eq!(
            path_admit(&mut path, 3.0).expect("ok"),
            (None, None, false),
            "first release past the end arms the stop but sends nothing"
        );
        assert_eq!(
            path_admit(&mut path, 3.5).expect("ok"),
            (None, None, true),
            "the second released frame past the end confirms the stop"
        );
    }

    #[test]
    fn does_not_stop_the_range_on_an_isolated_future_outlier() {
        // A lone in-extent corrupt pts past the range end must not truncate
        // the range: it is contradicted by the next legal frame and dropped,
        // and sampling continues.
        let mut path = range_path(2.0);
        path_admit(&mut path, 0.0).expect("ok");
        assert_eq!(
            path_admit(&mut path, 1.0).expect("ok"),
            (Some(0.0), None, false)
        );
        assert_eq!(
            path_admit(&mut path, 5.0).expect("ok"),
            (Some(1.0), None, false),
            "the outlier is held, not yet trusted to stop anything"
        );
        assert_eq!(
            path_admit(&mut path, 1.2).expect("ok"),
            (None, Some(5.0), false),
            "the next legal frame contradicts the outlier"
        );
        assert_eq!(
            path_admit(&mut path, 1.4).expect("ok"),
            (Some(1.2), None, false),
            "the range keeps sampling"
        );
    }

    #[test]
    fn finish_applies_the_range_end_to_the_eof_release() {
        // The EOF release goes through the same range gate as any frame: a
        // tail at or past the end is not sent (S2's deliver-path shape).
        let mut path = range_path(2.0);
        path_admit(&mut path, 0.0).expect("ok");
        assert_eq!(
            path_admit(&mut path, 2.5).expect("ok"),
            (Some(0.0), None, false)
        );
        let step = path.finish().expect("finish verdict passes");
        assert_eq!(
            step_view(step),
            (None, None, false),
            "the tail past the range end is released by the gate but not sent"
        );
    }

    #[test]
    fn finish_sends_the_eof_release_through_the_sampler() {
        let mut path = FramePath::new(
            SampleSpec::Keyframes,
            PlausibleWindow::from_declared(None),
            clock(),
        );
        path_admit(&mut path, 0.0).expect("ok");
        assert_eq!(
            path_admit(&mut path, 1.0).expect("ok"),
            (Some(0.0), None, false)
        );
        let step = path.finish().expect("finish verdict passes");
        assert_eq!(
            step_view(step),
            (Some(1.0), None, false),
            "a clean tail reaches the consumer"
        );
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
