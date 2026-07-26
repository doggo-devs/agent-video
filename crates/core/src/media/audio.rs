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

use super::{ErrorBudget, InterruptibleInput, Recv, Stop, classify_receive, open_input};
use crate::error::{Error, ErrorCode, Result};
use crate::source::Source;

/// The sample rate whisper.cpp expects.
pub const WHISPER_SAMPLE_RATE: u32 = 16_000;

/// Hard upper bound on any whole-track ceiling: 24 hours. Past it no
/// caller policy is sane (24 h × 16 kHz × 4 B ≈ 5.5 GB resident), and the
/// bound keeps `samples() + 1` arithmetic overflow-free on usize.
const MAX_CEILING_SECONDS: u32 = 24 * 3600;

/// A validated ceiling on whole-track PCM materialization, in seconds of
/// 16 kHz audio.
///
/// [`AudioSamples::collect`] — the only whole-track entry point — demands
/// one, so "decode the entire track into memory" is inexpressible without
/// a bound: a hostile file claiming hundreds of hours of DTX/silence audio
/// fails with a typed error instead of decoding into a multi-GB `Vec`.
#[derive(Debug, Clone, Copy)]
pub struct AudioCeiling {
    seconds: u32,
}

impl AudioCeiling {
    /// Build a ceiling of `secs` seconds. Rejects 0 and anything past the
    /// 24-hour hard cap with [`ErrorCode::InvalidConfig`]: both describe a
    /// misconfigured caller, never a property of the input.
    pub fn from_seconds(secs: u32) -> Result<AudioCeiling> {
        if secs == 0 || secs > MAX_CEILING_SECONDS {
            return Err(Error::new(
                ErrorCode::InvalidConfig,
                format!(
                    "invalid audio ceiling: {secs}s (must be 1..={MAX_CEILING_SECONDS} seconds)"
                ),
            ));
        }
        Ok(AudioCeiling { seconds: secs })
    }

    /// The ceiling in 16 kHz samples.
    pub fn samples(&self) -> usize {
        self.seconds as usize * WHISPER_SAMPLE_RATE as usize
    }
}

/// Decode `source`'s best audio stream to 16 kHz mono f32 samples, whole,
/// bounded by `cap`.
///
/// The convenience form of [`AudioSamples`] for callers that need the
/// entire track in memory (VAD wants random access). `Ok(None)` when the
/// container carries no audio stream at all: absence is a fact about the
/// input, not a decode failure, and the caller decides what "nothing to
/// transcribe" means. Errors are reserved for a container that cannot be
/// opened, a decode that breaks mid-stream, or a track that decodes past
/// `cap` ([`AudioSamples::collect`]). `stop` bounds every blocking read on
/// a stalled source; see [`super::open_input`].
pub fn decode_audio_16k_mono(
    source: &Source,
    stop: &Stop,
    cap: AudioCeiling,
) -> Result<Option<Vec<f32>>> {
    let Some(mut stream) = AudioSamples::open(source, stop)? else {
        return Ok(None);
    };
    Ok(Some(stream.collect(cap)?))
}

/// A pull stream of 16 kHz mono f32 samples from one audio track.
///
/// Owns the demux → decode → resample pipeline; [`Self::fill`] advances it
/// on demand. Per-packet failures are absorbed by an `ErrorBudget`, so
/// only a permanently broken stream surfaces as an error.
pub struct AudioSamples {
    ictx: InterruptibleInput,
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
    /// input's shape, not a decode failure. `stop` bounds every blocking
    /// read on a stalled source; see [`super::open_input`].
    pub fn open(source: &Source, stop: &Stop) -> Result<Option<AudioSamples>> {
        let ictx = open_input(source, stop, &[])?;
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

    /// Decode the remainder of the track into one `Vec`, failing once the
    /// decoded audio passes `cap` with [`ErrorCode::InputTooLong`].
    ///
    /// The only whole-track entry point. Overflow is an error, never a
    /// silent truncation: a partial transcript masquerading as complete
    /// would misreport the input. Peak memory on overflow is `cap` plus
    /// [`Self::fill`]'s one-packet overshoot — asking for one sample past
    /// the ceiling is enough to prove the track exceeds it.
    pub fn collect(&mut self, cap: AudioCeiling) -> Result<Vec<f32>> {
        let mut samples = Vec::new();
        self.fill(&mut samples, cap.samples() + 1)?;
        if samples.len() > cap.samples() {
            return Err(Error::new(
                ErrorCode::InputTooLong,
                format!(
                    "audio track decodes past the {}-second whole-track ceiling \
                     ({} samples at 16 kHz)",
                    cap.seconds,
                    cap.samples()
                ),
            ));
        }
        Ok(samples)
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
                match self.ictx.read_packet(&mut self.packet) {
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
        loop {
            match classify_receive(self.decoder.receive_frame(&mut self.frame)) {
                Recv::Frame => match self.sink.push(&mut self.frame, out) {
                    Ok(()) => self.budget.ok(),
                    Err(e) => self.budget.fail(e)?,
                },
                Recv::Drained => return Ok(()),
                // A hard receive error is charged, never mistaken for
                // "drained": a decoder that fails every receive trips the
                // budget instead of silently truncating the audio tail.
                Recv::Broken(e) => {
                    self.budget.fail(e)?;
                    return Ok(());
                }
            }
        }
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
/// frames as per-frame errors the caller's `ErrorBudget` can absorb.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn audio_ceiling_rejects_zero_and_beyond_the_hard_cap() {
        assert_eq!(
            AudioCeiling::from_seconds(0)
                .expect_err("zero seconds is not a ceiling")
                .code,
            ErrorCode::InvalidConfig
        );
        assert_eq!(
            AudioCeiling::from_seconds(MAX_CEILING_SECONDS + 1)
                .expect_err("past the 24h hard cap")
                .code,
            ErrorCode::InvalidConfig
        );
        let cap = AudioCeiling::from_seconds(MAX_CEILING_SECONDS).expect("the cap itself is legal");
        assert_eq!(cap.samples(), 24 * 3600 * 16_000);
    }
}
