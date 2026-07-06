// SPDX-License-Identifier: Apache-2.0
//! In-process frame decoding: a dedicated decode thread feeding a bounded
//! channel of small RGB frames.
//!
//! The old design got decode/score pipelining for free from the process
//! boundary (ffmpeg decoded while Rust scored, the pipe provided
//! backpressure). The decode thread and `sync_channel` reproduce exactly that
//! shape in-process: the channel is the pipe, its capacity the buffer, and a
//! dropped receiver the closed pipe that stops the producer.
//!
//! Timestamps come from the decoded frames themselves (`best-effort pts ×
//! time_base`, rebased to the stream start), produced by the same decoder
//! that produced the pixels — there is no cross-process index pairing left to
//! go wrong.

use std::num::NonZeroUsize;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, SyncSender, sync_channel};
use std::thread::JoinHandle;

use ffmpeg::codec::Context as CodecContext;
use ffmpeg::{Discard, Packet, threading};
use ffmpeg_next as ffmpeg;
use image::RgbImage;

use crate::error::{Error, ErrorCode, Result};
use crate::source::Source;
use crate::types::TimeRange;

use super::scale::Rescaler;
use super::{
    ErrorBudget, MAX_READ_ERRORS, VideoStream, best_video_stream, decode_error, open_input,
};

/// One sampled RGB frame, already downscaled for composition.
#[derive(Debug)]
pub struct RgbFrame {
    /// Presentation timestamp in seconds, 0-based at the stream start.
    pub ts: f32,
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
/// This is the backpressure knob the OS pipe used to provide: large enough to
/// keep both sides busy, small enough that a stalled consumer bounds memory
/// at a few small frames.
const CHANNEL_FRAMES: usize = 8;

/// A stream of sampled frames decoded on a background thread.
///
/// Pull frames with [`FrameSource::next_frame`]; `None` means clean end of
/// stream. Dropping the source stops the decode thread at its next send.
pub struct FrameSource {
    rx: Option<Receiver<std::result::Result<RgbFrame, Error>>>,
    thread: Option<JoinHandle<()>>,
    /// Set on drop so the worker exits at its next packet-loop iteration
    /// instead of decoding on until the next sampled frame's failed send.
    stop: Arc<AtomicBool>,
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
    pub fn open(
        input: &Source,
        spec: SampleSpec,
        min_edge: u32,
        threads: Option<NonZeroUsize>,
    ) -> Result<Self> {
        let mut ictx = open_input(input)?;
        let video = best_video_stream(&ictx, input)?;

        // Seek before building the decoder so no flush is needed. Seeking to
        // the latest keyframe at or before the target mirrors `-ss` accurate
        // input seeking: decode from the keyframe, discard until the target
        // (the tick sampler below does the discarding naturally). `start` is
        // a logical 0-based timestamp but the unbound seek call speaks the
        // container's absolute clock, so the timeline projects it back. An
        // unseekable stream (e.g. HTTP without Range support) degrades to a
        // sequential read: the sampler's first tick sits at `start`, so
        // everything before it is decoded and discarded — slower, same output.
        if let SampleSpec::Fps { start, .. } = &spec
            && *start > 0.0
        {
            let target = video.timeline.seek_target(*start as f64).as_av_time();
            if let Err(e) = ictx.seek(target, ..target) {
                eprintln!(
                    "warning: cannot seek to {start:.2}s ({e}); \
                     reading sequentially from the start"
                );
            }
        }

        let stream = ictx
            .stream(video.index)
            .expect("best stream index is valid");
        let mut codec_ctx = CodecContext::from_parameters(stream.parameters())
            .map_err(|e| decode_error("reading codec parameters", e).with_input(input.as_path()))?;

        // Frame threading with count 0 = auto-detect cores; an explicit
        // `--threads` caps it. Without this the decoder defaults to a single
        // thread — the ffmpeg CLI always configures threading, so must we.
        let mut tc = threading::Config::count(threads.map_or(0, NonZeroUsize::get));
        tc.kind = threading::Type::Frame;
        codec_ctx.set_threading(tc);

        let mut decoder = codec_ctx.decoder();
        if matches!(spec, SampleSpec::Keyframes) {
            // Belt and suspenders with the packet-level filter in the pump:
            // the decoder also skips anything that is not a keyframe.
            decoder.skip_frame(Discard::NonKey);
        }
        let decoder = decoder
            .video()
            .map_err(|e| decode_error("opening video decoder", e).with_input(input.as_path()))?;

        let (tx, rx) = sync_channel(CHANNEL_FRAMES);
        let stop = Arc::new(AtomicBool::new(false));
        let stop_flag = Arc::clone(&stop);
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
                    decode_errors: 0,
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
    pub fn next_frame(&mut self) -> anyhow::Result<Option<RgbFrame>> {
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
                Err(e.into())
            }
            // Channel closed: the decode thread finished cleanly.
            Err(_) => {
                self.done = true;
                self.join();
                Ok(None)
            }
        }
    }

    fn join(&mut self) {
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

impl Drop for FrameSource {
    fn drop(&mut self) {
        // The flag stops the worker at its next packet-loop iteration;
        // dropping the receiver additionally unblocks a send in progress.
        // Together they bound the join wait to one in-flight libav call
        // instead of however long decoding to the next sampled frame takes.
        self.stop.store(true, Ordering::Relaxed);
        self.rx = None;
        self.join();
    }
}

/// Tick-based frame selection: emits the first frame whose ts reaches the
/// next tick, then advances the tick past that frame — at most one frame per
/// tick, one tick per frame, robust to pts gaps. Frames decoded before a seek
/// target (keyframe preroll) fall before the first tick and are discarded
/// here.
struct Tick {
    next: f32,
    step: f32,
}

impl Tick {
    fn wants(&mut self, ts: f32) -> bool {
        // A sliver of tolerance absorbs float noise in pts arithmetic
        // (a frame at 0.999999 owns the 1.0 tick); anything larger
        // would visibly shift the sampling instants.
        let eps = self.step * 1e-3;
        if ts + eps < self.next {
            return false;
        }
        while self.next <= ts + eps {
            self.next += self.step;
        }
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
        step: f32,
    },
}

impl Sampler {
    fn new(spec: SampleSpec) -> Self {
        match spec {
            SampleSpec::Fps { fps, start, .. } => Sampler::Fps(Tick {
                next: start,
                step: 1.0 / fps,
            }),
            SampleSpec::Keyframes => Sampler::All,
            SampleSpec::Mixed { fps, fill } => Sampler::Mixed {
                fill,
                idx: 0,
                tick: None,
                step: 1.0 / fps,
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
                        .get_or_insert(Tick {
                            next: r.start(),
                            step: *step,
                        })
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
                // pts (dts fallback) places the packet on the timeline. A
                // non-key packet without either cannot be placed and is
                // dropped — the sampler could not emit its frame anyway.
                // B-frame reordering jitters pts only within a GOP, never
                // across the keyframe that opens the next fill range, so a
                // forward-only cursor is safe.
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
    ictx: ffmpeg::format::context::Input,
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
    stop: Arc<AtomicBool>,
    frames_sent: u64,
    decode_errors: u32,
}

/// Whether to keep pumping after handling one decoded frame.
enum Flow {
    Continue,
    Stop,
}

impl Worker {
    fn run(&mut self) {
        if let Err(e) = self.pump() {
            // The consumer may already be gone; nothing to do then.
            let _ = self.tx.send(Err(e));
        }
        // Dropping tx (with self) closes the channel = clean EOF marker.
    }

    fn pump(&mut self) -> Result<()> {
        let mut packet = Packet::empty();
        let mut read_budget = ErrorBudget::new("reading packets");
        loop {
            if self.stop.load(Ordering::Relaxed) {
                return Ok(());
            }
            match packet.read(&mut self.ictx) {
                Ok(()) => read_budget.ok(),
                Err(ffmpeg::Error::Eof) => break,
                Err(e) => {
                    // Transient demux errors are skipped like the CLI does,
                    // but a permanently failing input must not spin forever.
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

            if let Err(e) = self.decoder.send_packet(&packet) {
                // A corrupt packet is the input's fault, not a reason to
                // abandon the rest of the stream — the CLI keeps going too.
                self.decode_errors += 1;
                if self.decode_errors > MAX_READ_ERRORS {
                    return Err(decode_error("decoding video packet", e));
                }
                continue;
            }
            if matches!(self.drain()?, Flow::Stop) {
                return Ok(());
            }
        }

        // Flush delayed frames out of the decoder.
        let _ = self.decoder.send_eof();
        self.drain()?;

        // Decoding produced nothing and errors piled up: report the cause
        // instead of a silent empty stream.
        if self.frames_sent == 0 && self.decode_errors > 0 {
            return Err(Error::new(
                ErrorCode::DecodeFailed,
                format!(
                    "no frames decoded ({} packet errors); is the input corrupt?",
                    self.decode_errors
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
            match self.decoder.receive_frame(&mut frame) {
                Ok(()) => {}
                Err(ffmpeg::Error::Other { errno }) if errno == ffmpeg::error::EAGAIN => {
                    return Ok(Flow::Continue);
                }
                Err(ffmpeg::Error::Eof) => return Ok(Flow::Continue),
                Err(e) => return Err(decode_error("receiving decoded frame", e)),
            }

            // best-effort timestamp falls back to pts; a frame with neither
            // cannot be placed on the timeline and is skipped.
            let Some(pts) = frame.timestamp().or(frame.pts()) else {
                continue;
            };
            let ts = self.video.ts_seconds(pts);
            if let Some(end) = self.end
                && ts >= end
            {
                return Ok(Flow::Stop);
            }
            if !self.sampler.wants(ts, frame.is_key()) {
                continue;
            }

            let rgb = self.rescaler.rgb(&frame)?;
            if self.tx.send(Ok(RgbFrame { ts, rgb })).is_err() {
                // Receiver dropped: the consumer is done with us.
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
        let mut tick = Tick {
            next: 0.0,
            step: 1.0,
        };
        assert!(tick.wants(0.0));
        assert!(!tick.wants(0.5), "still inside the first step");
        assert!(tick.wants(1.2));
        assert!(!tick.wants(1.8), "the 2.0 tick has not arrived");
        assert!(tick.wants(2.0));
    }

    #[test]
    fn tick_absorbs_float_noise_just_below_the_instant() {
        let mut tick = Tick {
            next: 1.0,
            step: 1.0,
        };
        assert!(tick.wants(0.999999), "owns the 1.0 tick despite pts noise");
        assert!(!tick.wants(1.5));
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
