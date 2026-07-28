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
//! that produced the pixels. Container clocks are allowed to jump — MPEG-TS
//! PCR discontinuities, 33-bit pts wrap-arounds, and spliced clock domains
//! are all legal — and no local evidence can distinguish a legal jump from
//! a corrupt one, so this module never tries. Instead, every decoded frame
//! passes the [`Segmenter`], which cuts the container clock into monotone
//! *segments* and projects them onto one continuous logical clock;
//! everything downstream (sampler ticks, range end, chunk placement, wire
//! time ranges) consumes only the logical clock. A single-segment stream —
//! every normal file — projects identically: logical time equals container
//! time to the last bit.
//!
//! The contract this buys: every decodable frame is either sampled onto the
//! logical timeline or dropped with a signal (the first drop emits
//! [`Event::UnplaceableTimestamp`], the pass total
//! [`Event::TimestampsDropped`]); every committed clock seam is announced
//! as [`Event::TimelineDiscontinuity`]; and timestamps alone never fail a
//! run — a wrong segmentation threshold mislabels a seam, it cannot lose
//! content wholesale. Decode and I/O failures keep their own coded errors.
//!
//! Segment state is owned by the decode thread alone; only released frames
//! and signals cross the channel. Fast seeking speaks the container clock
//! and is therefore exact only within the stream's first segment: a later
//! segment's container offset is unknowable before its frames are decoded,
//! so a windowed decode of a multi-segment stream finds its window where
//! the container clock says it is, while labels follow the logical clock.

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
    ErrorBudget, Recv, Stop, VideoStream, best_video_stream, classify_receive,
    decode_error, duration_seconds, open_input,
};

/// Seconds on the container's presentation clock, 0-based at the declared
/// stream start — the clock decoded frames arrive on. A distinct type from
/// the logical clock so a container timestamp cannot be handed downstream
/// unprojected.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
struct ContainerTs(f64);

/// Seconds on the run's logical clock — the monotone timeline everything
/// downstream (sampler ticks, range end, chunker, wire ranges) consumes.
///
/// Only [`Segment::project`] mints values, and the segmenter advances
/// segment anchors monotonically, so the instants a decode pass releases
/// are non-decreasing by construction, not by runtime check.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
struct LogicalTs(f64);

impl LogicalTs {
    /// Narrow to the pipeline's frame timestamp; `None` when the value
    /// overflows `f32` — unreachable for fenced segments, checked anyway
    /// because [`Ts`] is the finite-by-construction boundary.
    fn to_ts(self) -> Option<Ts> {
        Ts::new(self.0 as f32)
    }
}

/// One committed clock seam: the container clock read `container` where
/// the logical clock continues at `logical`. Reported downstream as
/// [`Event::TimelineDiscontinuity`].
#[derive(Debug, Clone, Copy)]
struct Discontinuity {
    container: ContainerTs,
    logical: LogicalTs,
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
    Dropped { ts: f32, last: Option<f32> },
    Discontinuity { container: f64, logical: f64 },
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
    /// Frames are pre-scaled so their shorter edge is `min_edge` pixels.
    /// `threads` caps the decoder's thread count (`None` = auto). Container,
    /// stream, decoder, and seek failures surface here, synchronously, with
    /// precise error codes; mid-stream failures surface from `next_frame`.
    /// `events` receives a [`Event::SeekFallback`] when a seek to a window
    /// start fails and decoding degrades to a sequential read, then rides
    /// along for the stream's lifetime to report timeline signals (see
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
        // sampler's first tick.
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
    /// Frames arrive in non-decreasing *logical* timestamp order (see
    /// [`Segmenter`]). Timeline trouble is signal, never verdict: the first
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
/// caller must feed non-decreasing timestamps. [`Segmenter`] enforces
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

/// Frames a candidate run must accumulate before it becomes (or replaces)
/// the committed segment. Three defeats the observed one- and two-frame
/// poisoned stream heads outright, bounds every false candidate's cost to
/// two signaled frames (under 0.1s at 30 fps), and commits a real
/// discontinuity within three frames of its seam.
const SEGMENT_CONFIRM_FRAMES: usize = 3;

/// Floor of the in-segment forward fence, seconds — the previous
/// paradigm's EOF jump floor, value and rationale unchanged: admits every
/// sane inter-frame gap while keeping an unvouched forward jump below
/// chunk scale.
const FORWARD_FENCE_FLOOR: f64 = 10.0;

/// Extent-fence slack over the declared duration: `fence = duration ×
/// FACTOR + SLACK` — the previous paradigm's plausible-extent slack,
/// repurposed from a corruption verdict into a label prior. The slack
/// covers header imprecision (rounded durations, files that grew after
/// the header was written); on containers whose declared duration derives
/// from pts extremes (MPEG-TS) corruption can widen the fence upward,
/// never inward, so legal cadences always fit.
const EXTENT_FENCE_FACTOR: f64 = 2.0;
const EXTENT_FENCE_SLACK: f64 = 300.0;

/// Pre-roll allowance for the stream's first segment, seconds. Legitimate
/// negative timestamps (edit-list pre-roll, priming frames) stay within
/// single seconds of zero; a first segment opening below this bound means
/// the declared origin was dragged (or the pts counter wrapped), and the
/// segment is re-anchored at the stream head instead of trusted verbatim.
const PREROLL_ALLOWANCE: f64 = -60.0;

/// Nominal inter-frame gap before any is observed, seconds (25 fps).
/// Seeds the seam width and the jitter bound for keyframe-paced decodes
/// until the first released delta replaces it.
const FALLBACK_NOMINAL_GAP: f64 = 0.04;

/// One committed span of the logical timeline: a run of frames whose
/// container timestamps advanced plausibly. `logical = container + shift`;
/// the first segment of a well-anchored stream has `shift == 0.0` exactly,
/// so a single-segment file — every normal file — reproduces the container
/// clock bit for bit.
#[derive(Debug, Clone, Copy)]
struct Segment {
    shift: f64,
    /// Container ts of the last frame this segment released.
    last: ContainerTs,
}

impl Segment {
    /// The segment for a confirmed run starting at `first`, projected so
    /// its first frame lands on `anchor`. `anchor == first` yields the
    /// exact identity: IEEE subtraction of equal finite values is `0.0`,
    /// and adding `0.0` never changes a finite timestamp.
    fn anchored(anchor: f64, first: ContainerTs) -> Segment {
        Segment {
            shift: anchor - first.0,
            last: first,
        }
    }

    /// Project a container timestamp onto the logical clock.
    fn project(&self, ts: ContainerTs) -> LogicalTs {
        LogicalTs(ts.0 + self.shift)
    }
}

/// A run of frames awaiting confirmation: the head of the stream before
/// any segment is confirmed, or a challenger opened by a discontinuity
/// candidate. Non-empty by construction, bounded by
/// [`SEGMENT_CONFIRM_FRAMES`]; the frames are owned here until they are
/// released into a segment or dropped wholesale — no third fate exists.
struct Buffered {
    frames: Vec<(ContainerTs, ffmpeg::frame::Video)>,
}

impl Buffered {
    fn new(ts: ContainerTs, frame: ffmpeg::frame::Video) -> Buffered {
        Buffered {
            frames: vec![(ts, frame)],
        }
    }

    fn push(&mut self, ts: ContainerTs, frame: ffmpeg::frame::Video) {
        self.frames.push((ts, frame));
    }

    fn first(&self) -> ContainerTs {
        self.frames.first().expect("buffered runs are non-empty").0
    }

    fn last(&self) -> ContainerTs {
        self.frames.last().expect("buffered runs are non-empty").0
    }

    fn len(&self) -> usize {
        self.frames.len()
    }
}

/// The committed side of the segmenter: the stream head still gathering
/// its first confirmation, or a confirmed segment streaming frames.
enum Current {
    Head(Buffered),
    Confirmed(Segment),
}

/// Segmenter state. A challenger can only exist alongside a current run:
/// "challenger with nothing to challenge" is not representable.
enum State {
    /// No frame seen yet.
    Unopened,
    Open {
        current: Current,
        /// A candidate run contradicting `current`, gathering confirmation.
        challenger: Option<Buffered>,
    },
}

/// What one segmenter step yielded. Drops precede releases: a killed
/// challenger's frames fell off the timeline before the step's releases
/// continued it.
#[derive(Default)]
struct SegStep {
    /// Frames released onto the logical clock, in order.
    released: Vec<(LogicalTs, ffmpeg::frame::Video)>,
    /// Container timestamps of frames dropped off the timeline.
    dropped: Vec<ContainerTs>,
    /// A clock seam committed by this step.
    cut: Option<Discontinuity>,
}

/// Drop a whole buffered run off the timeline, one signal per frame.
fn drop_run(buf: Buffered, step: &mut SegStep) {
    step.dropped.extend(buf.frames.into_iter().map(|(ts, _)| ts));
}

/// Cuts the container clock into monotone segments and projects them onto
/// one continuous logical clock.
///
/// The paradigm: there is no ground truth separating a broken clock from a
/// legal discontinuity (MPEG-TS PCR jumps, 26.5-hour pts wrap-arounds, and
/// spliced clock domains are all legal), so nothing here passes verdicts
/// on the input. Frames are content; timestamps are labels; a wrong
/// threshold mislabels a seam instead of losing content.
///
/// Three local heuristics decide where segments end, all bounded, none
/// fatal:
///
/// - A backward jump beyond the jitter bound (twice the nominal frame gap
///   — the depth B-frame reordering can jitter presentation order) opens a
///   challenger; a backward step within it is decoder noise and drops that
///   frame with a signal, as the previous paradigm did.
/// - A forward jump beyond the fence (`max(`[`FORWARD_FENCE_FLOOR`]`, 2 ×`
///   the largest released gap`)` — the previous EOF rule generalized to
///   every frame) opens a challenger; smaller forward gaps are legal
///   pacing (timelapse splices, keyframe cadence) and pass through.
/// - Runs without released precedent — the stream head and every
///   challenger — fence instead against the declared extent
///   ([`EXTENT_FENCE_FACTOR`]`/`[`EXTENT_FENCE_SLACK`]), so a keyframe-only
///   decode teaches itself minute-scale cadence at the head while a
///   far-future poisoned timestamp still cannot join any run.
///
/// A challenger must gather [`SEGMENT_CONFIRM_FRAMES`] frames to commit.
/// Committing seals the current segment at its last released instant and
/// continues the logical clock one nominal gap later (a
/// [`Discontinuity`]); a challenger contradicted first is dropped whole,
/// with signals, at most `SEGMENT_CONFIRM_FRAMES - 1` frames each. That
/// bound is what keeps alternating poisoned timestamps from shredding the
/// stream: the poisoned half dies challenger by challenger while the legal
/// half streams on uninterrupted.
///
/// The first confirmed segment projects identically (`shift == 0`) when it
/// opens within [`PREROLL_ALLOWANCE`]; below that — a poisoned head packet
/// dragged the container's declared start, so legal content rebased far
/// negative — it re-anchors at the first container timestamp the stream
/// produced, giving content sane labels without any global verdict on
/// which clock was "right".
///
/// Owned by the decode thread alone; only released frames and signals
/// cross the channel. Infallible by design: timestamps relabel or drop
/// frames with signals, they never construct an error.
struct Segmenter {
    state: State,
    /// The first container ts the stream produced: the re-anchor point for
    /// a first segment that opens outside the pre-roll allowance.
    stream_head: Option<ContainerTs>,
    /// Forward fence for runs without released precedent, from the
    /// declared extent.
    extent_fence: f64,
    /// Last released in-segment gap: the seam width and the jitter scale.
    /// In a keyframe-only decode this is the keyframe cadence, so the
    /// jitter bound scales with it — a small backward resync there costs a
    /// frame or two (signaled) instead of a seam.
    nominal_gap: f64,
    /// Largest released in-segment gap: the forward fence's precedent.
    /// Learning is capped at the extent fence so a confirmed challenger
    /// carrying a corrupt gap cannot widen the fence without bound.
    max_released_gap: f64,
    /// Highest logical instant released — the monotonicity invariant,
    /// stated where it is produced.
    #[cfg(debug_assertions)]
    watermark: f64,
}

impl Segmenter {
    fn new(duration: Option<f32>, nominal_gap: f64) -> Segmenter {
        let declared = duration
            .map(f64::from)
            .filter(|d| d.is_finite() && *d > 0.0)
            .unwrap_or(0.0);
        Segmenter {
            state: State::Unopened,
            stream_head: None,
            extent_fence: declared * EXTENT_FENCE_FACTOR + EXTENT_FENCE_SLACK,
            nominal_gap,
            max_released_gap: 0.0,
            #[cfg(debug_assertions)]
            watermark: f64::NEG_INFINITY,
        }
    }

    /// Feed the next decoded frame; get back the frames it released, the
    /// frames it dropped, and any clock seam it committed.
    fn admit(&mut self, ts: ContainerTs, frame: ffmpeg::frame::Video) -> SegStep {
        let mut step = SegStep::default();
        if self.stream_head.is_none() {
            self.stream_head = Some(ts);
        }
        // Work on the state by value: every arm builds the successor state
        // whole, so a half-transitioned segmenter cannot be observed.
        match std::mem::replace(&mut self.state, State::Unopened) {
            State::Unopened => {
                self.state = State::Open {
                    current: Current::Head(Buffered::new(ts, frame)),
                    challenger: None,
                };
            }
            State::Open { current, challenger } => {
                self.state = self.step(current, challenger, ts, frame, &mut step);
            }
        }
        step
    }

    /// End of stream. An unconfirmed challenger is dropped — the same rule
    /// as mid-stream: an unvouched tail cannot commit a seam. An
    /// unconfirmed head is the whole stream (nothing ever contradicted
    /// it), so it confirms: a two-frame video is content, not a fragment.
    fn finish(&mut self) -> SegStep {
        let mut step = SegStep::default();
        if let State::Open { current, challenger } =
            std::mem::replace(&mut self.state, State::Unopened)
        {
            if let Some(ch) = challenger {
                drop_run(ch, &mut step);
            }
            if let Current::Head(buf) = current {
                self.confirm_first(buf, &mut step);
            }
        }
        step
    }

    /// One step of an opened stream: route the frame to the run it
    /// continues, or make it the new challenger. Returns the successor
    /// state.
    fn step(
        &mut self,
        current: Current,
        challenger: Option<Buffered>,
        ts: ContainerTs,
        frame: ffmpeg::frame::Video,
        step: &mut SegStep,
    ) -> State {
        if self.fits_current(&current, ts) {
            // The established run continues: a pending challenger was a
            // blip (an isolated poisoned run) and dies with a signal.
            if let Some(ch) = challenger {
                drop_run(ch, step);
            }
            State::Open {
                current: self.advance(current, ts, frame, step),
                challenger: None,
            }
        } else if challenger.as_ref().is_some_and(|ch| self.fits_run(ch, ts)) {
            let mut ch = challenger.expect("checked by is_some_and");
            ch.push(ts, frame);
            if ch.len() < SEGMENT_CONFIRM_FRAMES {
                State::Open {
                    current,
                    challenger: Some(ch),
                }
            } else {
                State::Open {
                    current: Current::Confirmed(self.commit(current, ch, step)),
                    challenger: None,
                }
            }
        } else {
            // Fits nothing: this frame opens the (replacement) challenger.
            if let Some(old) = challenger {
                drop_run(old, step);
            }
            State::Open {
                current,
                challenger: Some(Buffered::new(ts, frame)),
            }
        }
    }

    /// Whether `ts` continues the committed run.
    fn fits_current(&self, current: &Current, ts: ContainerTs) -> bool {
        match current {
            Current::Head(buf) => self.fits_run(buf, ts),
            Current::Confirmed(seg) => {
                let gap = ts.0 - seg.last.0;
                (-self.jitter()..=self.fence()).contains(&gap)
            }
        }
    }

    /// Whether `ts` continues an unconfirmed run: monotone, within the
    /// extent fence (no released precedent exists to fence with).
    fn fits_run(&self, buf: &Buffered, ts: ContainerTs) -> bool {
        let gap = ts.0 - buf.last().0;
        (0.0..=self.extent_fence).contains(&gap)
    }

    /// Backward tolerance inside a confirmed segment: ~2 frame durations,
    /// the depth B-frame reordering can jitter presentation order.
    fn jitter(&self) -> f64 {
        2.0 * self.nominal_gap
    }

    /// Forward fence inside a confirmed segment: twice the largest jump
    /// the stream itself released, floored at [`FORWARD_FENCE_FLOOR`].
    fn fence(&self) -> f64 {
        FORWARD_FENCE_FLOOR.max(2.0 * self.max_released_gap)
    }

    /// Advance the committed run with a frame that fits it.
    fn advance(
        &mut self,
        current: Current,
        ts: ContainerTs,
        frame: ffmpeg::frame::Video,
        step: &mut SegStep,
    ) -> Current {
        match current {
            Current::Head(mut buf) => {
                buf.push(ts, frame);
                if buf.len() < SEGMENT_CONFIRM_FRAMES {
                    Current::Head(buf)
                } else {
                    Current::Confirmed(self.confirm_first(buf, step))
                }
            }
            Current::Confirmed(mut seg) => {
                if ts < seg.last {
                    // Backward within the jitter bound: decoder noise, not
                    // a discontinuity — drop the frame, keep the segment.
                    step.dropped.push(ts);
                } else {
                    self.release(&mut seg, ts, frame, step);
                }
                Current::Confirmed(seg)
            }
        }
    }

    /// Confirm the stream's first segment: identity within the pre-roll
    /// allowance, re-anchored at the stream head otherwise (a dragged
    /// declared origin — see the type docs).
    fn confirm_first(&mut self, buf: Buffered, step: &mut SegStep) -> Segment {
        let first = buf.first();
        let anchor = if first.0 >= PREROLL_ALLOWANCE {
            first.0
        } else {
            self.stream_head
                .expect("set on the first admitted frame")
                .0
        };
        self.open_segment(Segment::anchored(anchor, first), buf, step)
    }

    /// Commit a confirmed challenger: seal the current run and continue
    /// the logical clock one nominal gap past its last released instant.
    fn commit(&mut self, current: Current, ch: Buffered, step: &mut SegStep) -> Segment {
        match current {
            // The head never confirmed: its short run is dropped whole
            // (with signals) and the challenger becomes the stream's first
            // segment. No seam is announced — there is no committed clock
            // to seam from.
            Current::Head(buf) => {
                drop_run(buf, step);
                self.confirm_first(ch, step)
            }
            Current::Confirmed(seg) => {
                let seam = seg.project(seg.last).0 + self.nominal_gap;
                let first = ch.first();
                step.cut = Some(Discontinuity {
                    container: first,
                    logical: LogicalTs(seam),
                });
                self.open_segment(Segment::anchored(seam, first), ch, step)
            }
        }
    }

    /// Release a confirmed run into `seg`, oldest first.
    fn open_segment(&mut self, mut seg: Segment, buf: Buffered, step: &mut SegStep) -> Segment {
        let mut frames = buf.frames.into_iter();
        let (first, frame) = frames.next().expect("buffered runs are non-empty");
        debug_assert!(first == seg.last, "a segment opens at its first frame");
        self.emit(&seg, first, frame, step);
        for (ts, frame) in frames {
            self.release(&mut seg, ts, frame, step);
        }
        seg
    }

    /// Release one frame into `seg`, learning its gap statistics.
    fn release(
        &mut self,
        seg: &mut Segment,
        ts: ContainerTs,
        frame: ffmpeg::frame::Video,
        step: &mut SegStep,
    ) {
        let gap = ts.0 - seg.last.0;
        debug_assert!(gap >= 0.0, "in-segment releases are monotone");
        // Precedent learning is capped at the extent fence: a confirmed
        // challenger may legally carry a large gap, but nothing beyond the
        // declared extent may widen future fences.
        if gap > 0.0 && gap <= self.extent_fence {
            self.nominal_gap = gap;
            self.max_released_gap = self.max_released_gap.max(gap);
        }
        seg.last = ts;
        self.emit(seg, ts, frame, step);
    }

    fn emit(
        &mut self,
        seg: &Segment,
        ts: ContainerTs,
        frame: ffmpeg::frame::Video,
        step: &mut SegStep,
    ) {
        let logical = seg.project(ts);
        #[cfg(debug_assertions)]
        {
            debug_assert!(
                logical.0 >= self.watermark,
                "the logical clock never rewinds"
            );
            self.watermark = logical.0;
        }
        step.released.push((logical, frame));
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
    seg: Segmenter,
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

    fn place(&mut self, seg: SegStep) -> Step {
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
        // Flush the segmenter: an unconfirmed tail challenger is dropped
        // (with signals); an unconfirmed head — a stream shorter than one
        // confirmation — is released whole.
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
                Recv::Frame => {}
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

    /// A segmenter as a fixed-fps decode at 30 fps would build it, over a
    /// container declaring `duration` seconds (`None` = undeclared).
    fn segmenter(duration: Option<f32>) -> Segmenter {
        Segmenter::new(duration, 1.0 / 30.0)
    }

    fn admit(seg: &mut Segmenter, v: f64) -> SegStep {
        seg.admit(ContainerTs(v), ffmpeg::frame::Video::empty())
    }

    fn released(step: &SegStep) -> Vec<f64> {
        step.released.iter().map(|(l, _)| l.0).collect()
    }

    fn dropped(step: &SegStep) -> Vec<f64> {
        step.dropped.iter().map(|ts| ts.0).collect()
    }

    /// Feed `values` then finish; collect every released logical instant.
    fn drain_releases(seg: &mut Segmenter, values: &[f64]) -> Vec<f64> {
        let mut out = Vec::new();
        for &v in values {
            out.extend(released(&admit(seg, v)));
        }
        out.extend(released(&seg.finish()));
        out
    }

    fn assert_close(actual: &[f64], expected: &[f64]) {
        assert_eq!(actual.len(), expected.len(), "{actual:?} vs {expected:?}");
        for (a, e) in actual.iter().zip(expected) {
            assert!((a - e).abs() < 1e-9, "{actual:?} vs {expected:?}");
        }
    }

    #[test]
    fn projects_a_clean_stream_identically() {
        let mut seg = segmenter(Some(2.0));
        let out = drain_releases(&mut seg, &[0.0, 0.1, 0.2, 0.3, 0.4]);
        // Bit-exact, not approximate: the identity segment adds 0.0.
        assert_eq!(out, vec![0.0, 0.1, 0.2, 0.3, 0.4]);
    }

    #[test]
    fn buffers_the_head_until_confirmation() {
        let mut seg = segmenter(Some(2.0));
        assert!(released(&admit(&mut seg, 0.0)).is_empty());
        assert!(released(&admit(&mut seg, 0.1)).is_empty());
        assert_eq!(released(&admit(&mut seg, 0.2)), vec![0.0, 0.1, 0.2]);
        assert_eq!(released(&admit(&mut seg, 0.3)), vec![0.3], "streams on");
    }

    #[test]
    fn emits_no_signals_on_a_clean_pass() {
        let mut seg = segmenter(Some(2.0));
        for i in 0..60 {
            let step = admit(&mut seg, f64::from(i) * 0.1);
            assert!(step.dropped.is_empty(), "clean streams drop nothing");
            assert!(step.cut.is_none(), "clean streams commit no seams");
        }
        let step = seg.finish();
        assert!(step.dropped.is_empty() && step.cut.is_none());
    }

    #[test]
    fn drops_backward_jitter_inside_a_segment() {
        let mut seg = segmenter(Some(2.0));
        for v in [0.0, 0.1, 0.2] {
            admit(&mut seg, v);
        }
        // nominal gap is 0.1, jitter bound 0.2: a small backward step is
        // decoder noise, dropped with a signal, no challenger.
        let step = admit(&mut seg, 0.15);
        assert_eq!(dropped(&step), vec![0.15]);
        assert!(step.cut.is_none());
        assert_eq!(released(&admit(&mut seg, 0.3)), vec![0.3]);
    }

    #[test]
    fn commits_a_seam_on_a_confirmed_backward_jump() {
        let mut seg = segmenter(Some(100.0));
        for v in [5.0, 5.1, 5.2] {
            admit(&mut seg, v);
        }
        assert!(released(&admit(&mut seg, 1.0)).is_empty(), "buffered");
        assert!(released(&admit(&mut seg, 1.1)).is_empty(), "buffered");
        let step = admit(&mut seg, 1.2);
        let cut = step.cut.expect("a confirmed backward jump commits a seam");
        // The new segment continues one nominal gap (0.1) past the seal.
        assert!((cut.logical.0 - 5.3).abs() < 1e-9);
        assert!((cut.container.0 - 1.0).abs() < 1e-9);
        assert_close(&released(&step), &[5.3, 5.4, 5.5]);
        assert_close(&released(&admit(&mut seg, 1.3)), &[5.6]);
    }

    #[test]
    fn kills_an_unconfirmed_challenger_when_the_segment_resumes() {
        let mut seg = segmenter(Some(2.0));
        for v in [0.0, 0.1, 0.2] {
            admit(&mut seg, v);
        }
        assert!(released(&admit(&mut seg, 500.0)).is_empty(), "buffered");
        let step = admit(&mut seg, 0.3);
        assert_eq!(dropped(&step), vec![500.0], "the blip dies with a signal");
        assert_eq!(released(&step), vec![0.3], "the segment streams on");
        assert!(step.cut.is_none(), "no seam was committed");
    }

    #[test]
    fn bounds_alternating_poison_to_the_poisoned_half() {
        // The fragment-storm shape: every other frame jumps +24h. Each
        // poisoned frame opens a challenger that the next legal frame
        // kills, so loss is exactly the poisoned half, every drop is
        // signaled, no seam is committed, and the logical clock stays on
        // the legal half.
        let mut seg = segmenter(Some(2.0));
        let mut kept = Vec::new();
        let mut lost = 0;
        for i in 0..60 {
            let t = f64::from(i) * 0.033;
            let step = admit(&mut seg, t);
            kept.extend(released(&step));
            lost += step.dropped.len();
            assert!(step.cut.is_none());
            let step = admit(&mut seg, t + 86_400.0);
            kept.extend(released(&step));
            lost += step.dropped.len();
            assert!(step.cut.is_none());
        }
        let tail = seg.finish();
        lost += tail.dropped.len();
        kept.extend(released(&tail));
        assert_eq!(kept.len(), 60, "the legal half survives whole");
        assert_eq!(lost, 60, "loss is bounded to the poisoned half");
        assert!(kept.iter().all(|&t| t < 2.1), "duration does not explode");
        assert!(kept.windows(2).all(|w| w[0] <= w[1]));
    }

    #[test]
    fn releases_an_in_fence_forward_gap_in_segment() {
        // Legal pacing (a timelapse splice) below the fence floor passes
        // through on the identity projection — it must not be shredded
        // into segments.
        let mut seg = segmenter(Some(60.0));
        for v in [0.0, 0.1, 0.2] {
            admit(&mut seg, v);
        }
        assert_eq!(released(&admit(&mut seg, 5.0)), vec![5.0]);
        assert_eq!(released(&admit(&mut seg, 5.1)), vec![5.1]);
    }

    #[test]
    fn learns_the_forward_fence_from_released_gaps() {
        let mut seg = segmenter(Some(100.0));
        // Head learns an 8s cadence: fence becomes max(10, 2×8) = 16.
        for v in [0.0, 8.0, 16.0] {
            admit(&mut seg, v);
        }
        assert_eq!(
            released(&admit(&mut seg, 31.0)),
            vec![31.0],
            "a 15s gap fits the learned fence"
        );
        assert!(
            released(&admit(&mut seg, 100.0)).is_empty(),
            "a 69s gap beyond the fence opens a challenger"
        );
    }

    #[test]
    fn teaches_keyframe_cadence_through_the_head() {
        // A keyframe-only decode of a sparse stream: minute-scale gaps are
        // its legal cadence. The head fences against the declared extent,
        // not the flat floor, so the cadence confirms and then becomes its
        // own precedent.
        let mut seg = Segmenter::new(Some(500.0), FALLBACK_NOMINAL_GAP);
        let out = drain_releases(&mut seg, &[0.0, 100.0, 200.0, 300.0, 400.0]);
        assert_eq!(out, vec![0.0, 100.0, 200.0, 300.0, 400.0]);
    }

    #[test]
    fn fences_the_head_against_a_far_future_jump() {
        let mut seg = segmenter(Some(2.0));
        assert!(released(&admit(&mut seg, 0.0)).is_empty());
        assert!(
            released(&admit(&mut seg, 86_400.0)).is_empty(),
            "beyond the extent fence: a challenger, not a head frame"
        );
        let step = admit(&mut seg, 0.033);
        assert_eq!(dropped(&step), vec![86_400.0], "the blip dies signaled");
        assert_eq!(released(&admit(&mut seg, 0.066)), vec![0.0, 0.033, 0.066]);
    }

    #[test]
    fn replaces_an_unconfirmed_head_with_its_confirmed_challenger() {
        // The poisoned-head shape: the first packets dragged the declared
        // origin a day forward, so the head reads ~0 and legal content
        // rebases to -86398. The challenger confirms, the dead head is
        // dropped with signals, and the first segment re-anchors at the
        // stream head — content keeps sane labels, no verdict, no retry.
        let mut seg = segmenter(Some(0.1));
        admit(&mut seg, 0.0);
        admit(&mut seg, 0.033);
        assert!(released(&admit(&mut seg, -86_398.0)).is_empty());
        assert!(released(&admit(&mut seg, -86_397.966)).is_empty());
        let step = admit(&mut seg, -86_397.933);
        assert_eq!(
            dropped(&step),
            vec![0.0, 0.033],
            "the dead head is dropped with signals"
        );
        assert!(step.cut.is_none(), "no committed clock existed to seam");
        assert_close(&released(&step), &[0.0, 0.034, 0.067]);
    }

    #[test]
    fn anchors_an_opening_segment_within_preroll_identically() {
        // Edit-list pre-roll: legitimate negative timestamps close to zero
        // project identically — byte-for-byte what the container said.
        let mut seg = segmenter(Some(4.0));
        let out = drain_releases(&mut seg, &[-0.5, -0.4, -0.3, 0.0, 0.1]);
        assert_eq!(out, vec![-0.5, -0.4, -0.3, 0.0, 0.1]);
    }

    #[test]
    fn drops_an_unconfirmed_challenger_at_eof() {
        // A corrupt pts on the very last frame has no successor to vouch
        // for it: dropped with a signal, never released into a garbage
        // label.
        let mut seg = segmenter(Some(2.0));
        for v in [0.0, 0.1, 0.2] {
            admit(&mut seg, v);
        }
        admit(&mut seg, 86_400.0);
        let step = seg.finish();
        assert_eq!(dropped(&step), vec![86_400.0]);
        assert!(step.released.is_empty());
    }

    #[test]
    fn confirms_a_short_head_at_eof() {
        // A stream shorter than one confirmation is content, not a
        // fragment: nothing ever contradicted it.
        let mut seg = segmenter(Some(1.0));
        admit(&mut seg, 0.0);
        admit(&mut seg, 0.1);
        assert_eq!(released(&seg.finish()), vec![0.0, 0.1]);
    }

    #[test]
    fn releases_a_lone_frame_at_eof() {
        let mut seg = segmenter(Some(10.0));
        admit(&mut seg, 5.0);
        assert_eq!(released(&seg.finish()), vec![5.0]);
    }

    #[test]
    fn keeps_the_logical_clock_monotone_across_seams() {
        // Three clock domains, each internally sane: two seams, all
        // content preserved, and the concatenation is non-decreasing.
        let mut seg = segmenter(Some(3.0));
        let mut values = Vec::new();
        values.extend((0..10).map(|i| f64::from(i) * 0.033));
        values.extend((0..10).map(|i| 43_200.0 + f64::from(i) * 0.033));
        values.extend((0..10).map(|i| 20.0 + f64::from(i) * 0.033));
        let out = drain_releases(&mut seg, &values);
        assert_eq!(out.len(), 30, "every frame survives");
        assert!(out.windows(2).all(|w| w[0] <= w[1]), "monotone: {out:?}");
        assert!(out.iter().all(|&t| t < 2.0), "labels stay near content");
    }

    /// Drive one container timestamp through a [`FramePath`]; map the step
    /// to `(sent, dropped, stop)` timestamps.
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
        let (sent, _, stop) = path_admit(&mut path, 2.5);
        assert!(sent.is_empty(), "past the end sends nothing");
        assert!(stop, "monotone releases make the first crossing final");
    }

    #[test]
    fn does_not_stop_the_range_on_an_out_of_fence_outlier() {
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
