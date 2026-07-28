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
//! that produced the pixels.

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
    ErrorBudget, Recv, Stop, VideoStream, best_video_stream, classify_receive, decode_error,
    open_input,
};

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

/// A stream of sampled frames decoded on a background thread.
///
/// Pull frames with [`FrameSource::next_frame`]; `None` means clean end of
/// stream. Dropping the source stops the decode thread at its next send.
pub struct FrameSource {
    rx: Option<Receiver<std::result::Result<RgbFrame, Error>>>,
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
}

impl FrameSource {
    /// Open `input` and start decoding per `spec`.
    ///
    /// Frames are pre-scaled so their shorter edge is `min_edge` pixels.
    /// `threads` caps the decoder's thread count (`None` = auto). Container,
    /// stream, decoder, and seek failures surface here, synchronously, with
    /// precise error codes; mid-stream failures surface from `next_frame`.
    /// `events` receives a [`Event::SeekFallback`] when a seek to a window
    /// start fails and decoding degrades to a sequential read.
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
        events: &dyn EventSink,
    ) -> Result<Self> {
        // Built before the container so it can ride into libav's interrupt
        // callback, arming every blocking I/O call in this context.
        let stop = Stop::new();
        let mut ictx = open_input(input, &stop, &[])?;
        let video = best_video_stream(&ictx, input)?;

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
        let stop_flag = stop.clone();
        let thread = std::thread::Builder::new()
            .name("agent-video-decode".into())
            .spawn(move || {
                let end = match &spec {
                    SampleSpec::Fps { end, .. } => *end,
                    _ => None,
                };
                let mut worker = Worker {
                    ictx,
                    video,
                    decoder,
                    gate: PacketGate::new(&spec),
                    sampler: Sampler::new(spec),
                    end,
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
        })
    }

    /// Source frame aspect ratio, known once at least one frame was read.
    pub fn aspect(&self) -> Option<f32> {
        self.aspect
    }

    /// Read the next sampled frame, or `None` after a clean end of stream.
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
        let rx = self.rx.as_ref().expect("receiver lives until drop");
        match rx.recv() {
            Ok(Ok(frame)) => {
                let (w, h) = frame.rgb.dimensions();
                self.aspect = Some(w as f32 / h as f32);
                Ok(Some(frame))
            }
            Ok(Err(e)) => {
                self.done = true;
                Err(e)
            }
            // Channel closed: the worker is gone, but a panic also drops
            // `tx` on unwind. Joining tells a clean end of stream from a
            // crashed worker, which must not masquerade as EOF.
            Err(_) => {
                self.done = true;
                match self.join() {
                    Ok(()) => Ok(None),
                    Err(payload) => Err(Error::new(
                        ErrorCode::DecodeFailed,
                        format!(
                            "decode thread panicked: {}",
                            panic_message(payload.as_ref())
                        ),
                    )),
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

impl Drop for FrameSource {
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
        // `as` cast saturates, bounding a corrupt far-future pts at
        // `u64::MAX`; `max` keeps the index monotonic against float
        // rounding at the tick boundary.
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

/// The decode thread's state; consumed by [`Worker::run`].
struct Worker {
    ictx: super::InterruptibleInput,
    video: VideoStream,
    decoder: ffmpeg::decoder::Video,
    sampler: Sampler,
    /// Demux-level packet gate: what it rejects never reaches the decoder.
    gate: PacketGate,
    /// Stop decoding once a frame's ts reaches this bound (range sampling).
    end: Option<f32>,
    rescaler: Rescaler,
    tx: SyncSender<std::result::Result<RgbFrame, Error>>,
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
        let mut frame = ffmpeg::frame::Video::empty();
        loop {
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
            if let Some(end) = self.end
                && ts.get() >= end
            {
                return Ok(Flow::Stop);
            }
            if !self.sampler.wants(ts.get(), frame.is_key()) {
                continue;
            }

            let rgb = self.rescaler.rgb(&frame)?;
            if self.tx.send(Ok(RgbFrame { ts, rgb })).is_err() {
                return Ok(Flow::Stop);
            }
            self.frames_sent += 1;
        }
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
    fn stops_when_a_corrupt_pts_outruns_the_step() {
        // A finite but absurd pts (what `Ts` still admits): on the old
        // accumulating advance, `next += step` became a floating no-op once
        // `step < ULP(next) / 2` and the decode thread spun forever. The
        // indexed advance must answer immediately.
        let mut tick = Tick::new(0.0, 1.0 / 30.0);
        assert!(tick.wants(1e30), "corrupt frame still owns the pending tick");
        assert!(
            !tick.wants(0.0),
            "the tick grid stays monotonic after the jump"
        );
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
