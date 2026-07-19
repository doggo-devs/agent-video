// SPDX-License-Identifier: Apache-2.0
//! Audio decoding for transcription: any input's best audio stream to
//! 16 kHz mono f32 PCM (what whisper.cpp consumes), in-process via
//! libswresample. [`AudioSamples`] is a pull stream — the caller decides
//! how much decoded audio is resident at once, so a consumer that
//! processes as it pulls holds hours of audio in bounded memory.

use ffmpeg_next as ffmpeg;

use ffmpeg::software::resampling;
use ffmpeg::util::channel_layout::ChannelLayout;
use ffmpeg::util::format::sample::{Sample, Type};
use ffmpeg::util::frame;

use super::{ErrorBudget, open_input};
use crate::error::Result;
use crate::source::Source;

/// The sample rate whisper.cpp expects.
pub const WHISPER_SAMPLE_RATE: u32 = 16_000;

/// Decode `source`'s best audio stream to 16 kHz mono f32 samples, whole.
///
/// The convenience form of [`AudioSamples`] for callers that need the
/// entire track in memory (VAD wants random access). `Ok(None)` when the
/// container carries no audio stream at all: absence is a fact about the
/// input, not a decode failure, and the caller decides what "nothing to
/// transcribe" means. Errors are reserved for a container that cannot be
/// opened or a decode that breaks mid-stream.
pub fn decode_audio_16k_mono(source: &Source) -> Result<Option<Vec<f32>>> {
    let Some(mut stream) = AudioSamples::open(source)? else {
        return Ok(None);
    };
    let mut samples = Vec::new();
    stream.fill(&mut samples, usize::MAX)?;
    Ok(Some(samples))
}

/// A pull stream of 16 kHz mono f32 samples from one audio track.
///
/// Owns the demux → decode → resample pipeline; [`Self::fill`] advances it
/// on demand. Per-packet failures are absorbed by an [`ErrorBudget`], so
/// only a permanently broken stream surfaces as an error.
pub struct AudioSamples {
    ictx: ffmpeg::format::context::Input,
    stream_index: usize,
    decoder: ffmpeg::decoder::Audio,
    sink: MonoSink,
    /// Reused demux buffer.
    packet: ffmpeg::Packet,
    /// Reused decode buffer.
    frame: frame::Audio,
    budget: ErrorBudget,
    stage: Stage,
}

/// Where the pipeline stands; stages only move forward.
enum Stage {
    /// Packets still flow from the demuxer.
    Demux,
    /// Demux EOF seen: the decoder's lookahead and the resampler's
    /// carry-over still owe samples.
    Flush,
    /// Everything delivered; [`AudioSamples::fill`] keeps answering
    /// "no more".
    Done,
}

impl AudioSamples {
    /// Open `source`'s best audio stream as a sample stream. `Ok(None)`
    /// when the container has no audio stream at all — absence is the
    /// input's shape, not a decode failure.
    pub fn open(source: &Source) -> Result<Option<AudioSamples>> {
        let ictx = open_input(source)?;
        let Some(stream) = ictx.streams().best(ffmpeg::media::Type::Audio) else {
            return Ok(None);
        };
        let stream_index = stream.index();

        let ctx = ffmpeg::codec::context::Context::from_parameters(stream.parameters())
            .map_err(|e| super::decode_error("opening audio decoder", e))?;
        let mut decoder = ctx
            .decoder()
            .audio()
            .map_err(|e| super::decode_error("opening audio decoder", e))?;

        // Decoders may leave the layout unset; the resampler needs one.
        if decoder.channel_layout().is_empty() {
            decoder.set_channel_layout(ChannelLayout::default(decoder.channels() as i32));
        }

        let sink = MonoSink::new(decoder.format(), decoder.channel_layout(), decoder.rate())
            .map_err(|e| super::decode_error("configuring audio resampler", e))?;

        Ok(Some(AudioSamples {
            ictx,
            stream_index,
            decoder,
            sink,
            packet: ffmpeg::Packet::empty(),
            frame: frame::Audio::empty(),
            budget: ErrorBudget::new("decoding audio packets"),
            stage: Stage::Demux,
        }))
    }

    /// Append decoded samples to `out` until `out.len() >= min_samples` or
    /// the track ends. Returns whether more samples may still arrive;
    /// `false` is final — further calls keep answering `false` without
    /// touching `out`. `out` may overshoot `min_samples` by the delivery
    /// granularity (one packet's worth of resampled audio), never by more.
    pub fn fill(&mut self, out: &mut Vec<f32>, min_samples: usize) -> Result<bool> {
        while out.len() < min_samples {
            if !self.step(out)? {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// Advance the pipeline by one increment: one demuxed packet, or the
    /// end-of-stream flush. `false` means exhausted.
    fn step(&mut self, out: &mut Vec<f32>) -> Result<bool> {
        match self.stage {
            Stage::Demux => {
                match self.packet.read(&mut self.ictx) {
                    Ok(()) if self.packet.stream() != self.stream_index => {}
                    Ok(()) => match self.decoder.send_packet(&self.packet) {
                        Ok(()) => self.receive_frames(out)?,
                        Err(e) => self.budget.fail(e)?,
                    },
                    Err(ffmpeg::Error::Eof) => {
                        // The EOF flush is the decoder's last squeeze:
                        // lookahead frames surface only now, so a refused
                        // flush spends error budget instead of silently
                        // dropping the tail.
                        if let Err(e) = self.decoder.send_eof() {
                            self.budget.fail(e)?;
                        }
                        self.stage = Stage::Flush;
                    }
                    // Unlike `PacketIter` (which retries read errors
                    // unbudgeted, forever), a failed read spends budget, so
                    // a permanently failing input trips the breaker instead
                    // of spinning.
                    Err(e) => self.budget.fail(e)?,
                }
                Ok(true)
            }
            Stage::Flush => {
                self.receive_frames(out)?;
                self.sink
                    .drain(out)
                    .map_err(|e| super::decode_error("flushing audio resampler", e))?;
                self.stage = Stage::Done;
                Ok(true)
            }
            Stage::Done => Ok(false),
        }
    }

    /// Move every frame the decoder currently holds through the sink.
    fn receive_frames(&mut self, out: &mut Vec<f32>) -> Result<()> {
        while self.decoder.receive_frame(&mut self.frame).is_ok() {
            match self.sink.push(&mut self.frame, out) {
                Ok(()) => self.budget.ok(),
                Err(e) => self.budget.fail(e)?,
            }
        }
        Ok(())
    }
}

/// Streaming converter to 16 kHz mono f32 that follows mid-stream input
/// parameter changes.
///
/// A decoder's header parameters are a prediction, not a contract: HE-AAC
/// SBR doubles the rate after the first frame, DASH splices can change the
/// layout, and a decoder recovering from bitstream corruption may emit a
/// frame with nonsense parameters. A fixed swresample context answers all
/// of those with `AVERROR_INPUT_CHANGED` and would kill the stream; this
/// sink instead reconfigures for coherent changes and reports garbage
/// frames as per-frame errors the caller's [`ErrorBudget`] can absorb.
struct MonoSink {
    ctx: resampling::Context,
    format: Sample,
    layout: ChannelLayout,
    rate: u32,
}

impl MonoSink {
    fn new(
        format: Sample,
        layout: ChannelLayout,
        rate: u32,
    ) -> std::result::Result<Self, ffmpeg::Error> {
        let ctx = resampling::Context::get(
            format,
            layout,
            rate,
            Sample::F32(Type::Packed),
            ChannelLayout::MONO,
            WHISPER_SAMPLE_RATE,
        )?;
        Ok(MonoSink {
            ctx,
            format,
            layout,
            rate,
        })
    }

    /// Feed one decoded frame. `Err` marks this frame as unusable (decoder
    /// garbage after corruption) — the stream itself may continue.
    fn push(
        &mut self,
        frame: &mut frame::Audio,
        samples: &mut Vec<f32>,
    ) -> std::result::Result<(), ffmpeg::Error> {
        if frame.channel_layout().is_empty() {
            frame.set_channel_layout(ChannelLayout::default(frame.channels() as i32));
        }
        let params = (frame.format(), frame.channel_layout().bits(), frame.rate());
        if params != (self.format, self.layout.bits(), self.rate) {
            // Coherent parameters reconfigure; incoherent ones (a 17-channel
            // frame out of a stereo stream) fail context creation and are
            // charged to the caller's error budget.
            let next = MonoSink::new(frame.format(), frame.channel_layout(), frame.rate())?;
            self.drain(samples)?;
            *self = next;
        }
        let capacity = scaled_capacity(frame.samples(), self.rate);
        let mut out = frame::Audio::new(Sample::F32(Type::Packed), capacity, ChannelLayout::MONO);
        self.ctx.run(frame, &mut out)?;
        append_samples(&out, samples);
        Ok(())
    }

    /// Drain the resampler's internal carry-over into `samples`.
    fn drain(&mut self, samples: &mut Vec<f32>) -> std::result::Result<(), ffmpeg::Error> {
        loop {
            let mut out = frame::Audio::new(Sample::F32(Type::Packed), 4096, ChannelLayout::MONO);
            let delay = self.ctx.flush(&mut out)?;
            append_samples(&out, samples);
            if delay.is_none() || out.samples() == 0 {
                return Ok(());
            }
        }
    }
}

/// Output-frame capacity for `n` input samples at `in_rate`, with headroom
/// for resampler carry-over.
fn scaled_capacity(n: usize, in_rate: u32) -> usize {
    let scaled = (n as u64 * u64::from(WHISPER_SAMPLE_RATE)).div_ceil(u64::from(in_rate.max(1)));
    scaled as usize + 256
}

/// Append the valid samples of a packed mono f32 frame.
fn append_samples(out: &frame::Audio, samples: &mut Vec<f32>) {
    let n = out.samples();
    if n == 0 {
        return;
    }
    // Packed mono: one plane, one sample per frame slot.
    samples.extend_from_slice(&out.plane::<f32>(0)[..n]);
}
