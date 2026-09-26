// SPDX-License-Identifier: Apache-2.0
//! Pull-based audio decoding to timed, continuous blocks of 16 kHz mono PCM.
//! Gaps carry no samples; every block retains its position on the video clock.

use std::collections::VecDeque;
use std::num::NonZeroUsize;

use ffmpeg::software::resampling;
use ffmpeg::util::channel_layout::ChannelLayout;
use ffmpeg::util::format::sample::{Sample, Type};
use ffmpeg::util::frame;
use ffmpeg_next as ffmpeg;

use super::clock::{ContainerTs, LogicalTs, PREROLL_ALLOWANCE, SegStep, Segmenter};
use super::{ErrorBudget, InterruptibleInput, Recv, Stop, Timeline, classify_receive, open_input};
use crate::error::Result;
use crate::source::Source;

/// Sample rate of decoded audio and both transcription routes.
pub const WHISPER_SAMPLE_RATE: u32 = 16_000;

/// Position on the media clock, measured in 16 kHz samples.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct SamplePos(i64);

impl SamplePos {
    /// Construct a position in 16 kHz samples, including negative pre-roll.
    pub const fn from_samples(samples: i64) -> Self {
        Self(samples)
    }

    /// Position in seconds relative to the video's start.
    pub fn seconds(self) -> f64 {
        self.0 as f64 / f64::from(WHISPER_SAMPLE_RATE)
    }

    /// Advance by a sample count without wrapping the media clock.
    pub fn advance(self, samples: usize) -> Self {
        Self(
            self.0
                .saturating_add(i64::try_from(samples).unwrap_or(i64::MAX)),
        )
    }

    /// Nonnegative sample distance from an earlier position, saturated to `usize`.
    pub fn samples_since(self, earlier: Self) -> usize {
        usize::try_from(self.0.saturating_sub(earlier.0).max(0)).unwrap_or(usize::MAX)
    }

    fn from_seconds(seconds: f64) -> std::result::Result<Self, ffmpeg::Error> {
        let samples = (seconds * f64::from(WHISPER_SAMPLE_RATE)).round();
        if !samples.is_finite() || samples < i64::MIN as f64 || samples >= i64::MAX as f64 {
            return Err(ffmpeg::Error::InvalidData);
        }
        Ok(Self(samples as i64))
    }
}

/// A continuous run of PCM at its original media position. Emitted blocks
/// are nonempty, nonnegative and nonoverlapping; gaps are not materialized.
#[derive(Debug)]
pub struct AudioChunk {
    /// Position of the first sample on the source clock.
    pub start: SamplePos,
    /// Consecutive 16 kHz mono samples, with no timeline gaps inside.
    pub samples: Vec<f32>,
}

impl AudioChunk {
    /// Exclusive end on the media clock.
    pub fn end(&self) -> SamplePos {
        self.start.advance(self.samples.len())
    }
}

/// A pull stream owning demux, decode and resampling. Memory is bounded by
/// the requested block size plus bounded clock confirmation and resampler carry.
pub struct AudioSamples {
    ictx: InterruptibleInput,
    stream_index: usize,
    decoder: ffmpeg::decoder::Audio,
    sink: MonoSink,
    clock: Segmenter<frame::Audio>,
    timeline: Timeline,
    time_base: f64,
    fallback_start: f64,
    packet: ffmpeg::Packet,
    budget: ErrorBudget,
    stage: Stage,
    ready: VecDeque<AudioChunk>,
    decoded_end: Option<SamplePos>,
    dropped_frames: usize,
}

enum Stage {
    Demux,
    Flush,
    Done,
}

impl AudioSamples {
    /// Open the best audio stream; `None` means the source has no audio.
    /// Blocking reads observe `stop`. Positions share the video's origin.
    pub fn open(source: &Source, stop: &Stop) -> Result<Option<Self>> {
        let ictx = open_input(source, stop, &[])?;
        let Some(stream) = ictx.streams().best(ffmpeg::media::Type::Audio) else {
            return Ok(None);
        };
        let stream_index = stream.index();
        let tb = stream.time_base();
        let time_base = f64::from(tb);
        let mut timeline = Timeline::from_input(&ictx);
        let start = stream.start_time();
        if start != ffmpeg::ffi::AV_NOPTS_VALUE
            && timeline.seconds(start, time_base) < PREROLL_ALLOWANCE
        {
            timeline = super::decode::head_timeline(source, stop, start as f64 * time_base)?;
        }
        let fallback_start = if start == ffmpeg::ffi::AV_NOPTS_VALUE {
            0.0
        } else {
            timeline.seconds(start, time_base)
        };
        let ctx = ffmpeg::codec::context::Context::from_parameters(stream.parameters())
            .map_err(|e| super::decode_error("opening audio decoder", e))?;
        let mut decoder = ctx.decoder();
        // Decoder-side priming removal adjusts frame PTS in this time base.
        decoder.set_packet_time_base(tb);
        let decoder = decoder
            .audio()
            .map_err(|e| super::decode_error("opening audio decoder", e))?;
        let duration = super::duration_seconds(&ictx, source).ok();
        Ok(Some(Self {
            ictx,
            stream_index,
            decoder,
            timeline,
            time_base,
            fallback_start,
            clock: Segmenter::new(duration, 0.02),
            sink: MonoSink {
                time_base,
                run: None,
                published_end: SamplePos(0),
            },
            packet: ffmpeg::Packet::empty(),
            budget: ErrorBudget::new("decoding audio packets"),
            stage: Stage::Demux,
            ready: VecDeque::new(),
            decoded_end: None,
            dropped_frames: 0,
        }))
    }

    /// Pull at most `max_samples` contiguous samples. Stops at a time gap
    /// even when the block is smaller than requested. `None` is final.
    pub fn next_chunk(&mut self, max_samples: NonZeroUsize) -> Result<Option<AudioChunk>> {
        let mut result: Option<AudioChunk> = None;
        loop {
            while self.ready.is_empty() && !matches!(self.stage, Stage::Done) {
                self.step()?;
            }
            let Some(front) = self.ready.front() else {
                break;
            };
            if result.as_ref().is_some_and(|out| out.end() != front.start) {
                break;
            }
            let mut chunk = self.ready.pop_front().expect("front exists");
            let out = result.get_or_insert_with(|| AudioChunk {
                start: chunk.start,
                samples: Vec::new(),
            });
            let take = chunk
                .samples
                .len()
                .min(max_samples.get() - out.samples.len());
            if take < chunk.samples.len() {
                let rest = chunk.samples.split_off(take);
                self.ready.push_front(AudioChunk {
                    start: chunk.start.advance(take),
                    samples: rest,
                });
            }
            out.samples.extend(chunk.samples);
            if out.samples.len() == max_samples.get() {
                break;
            }
        }
        if let Some(chunk) = &result {
            self.decoded_end = Some(chunk.end());
        }
        Ok(result)
    }

    /// End of the last delivered audio block, before any VAD filtering.
    /// `None` means no presentable samples were decoded.
    pub fn decoded_end(&self) -> Option<SamplePos> {
        self.decoded_end
    }

    /// Decoded frames discarded because their timestamps could not be confirmed.
    pub fn dropped_frames(&self) -> usize {
        self.dropped_frames
    }

    fn step(&mut self) -> Result<()> {
        let mut frame = frame::Audio::empty();
        match classify_receive(self.decoder.receive_frame(&mut frame)) {
            Recv::Frame => {
                if frame.rate() == 0 || frame.samples() == 0 {
                    self.budget.fail(ffmpeg::Error::InvalidData)?;
                    return Ok(());
                }
                let at = frame.pts().or(frame.timestamp()).map_or_else(
                    || self.clock.next_timestamp().unwrap_or(self.fallback_start),
                    |pts| self.timeline.seconds(pts, self.time_base),
                );
                let step = self.clock.admit(ContainerTs(at), frame);
                self.place(step)?;
                return Ok(());
            }
            Recv::Broken(e) => self.budget.fail(e)?,
            Recv::Drained => {}
        }
        match self.stage {
            Stage::Demux => match self.ictx.read_packet(&mut self.packet) {
                Ok(()) if self.packet.stream() != self.stream_index => {}
                Ok(()) => {
                    if let Err(e) = self.decoder.send_packet(&self.packet) {
                        self.budget.fail(e)?;
                    }
                }
                Err(ffmpeg::Error::Eof) => {
                    if let Err(e) = self.decoder.send_eof() {
                        self.budget.fail(e)?;
                    }
                    self.stage = Stage::Flush;
                }
                Err(e) => self.budget.fail(e)?,
            },
            Stage::Flush => {
                let step = self.clock.finish();
                self.place(step)?;
                self.sink
                    .drain(&mut self.ready)
                    .map_err(|e| super::decode_error("flushing audio resampler", e))?;
                self.stage = Stage::Done;
            }
            Stage::Done => {}
        }
        Ok(())
    }

    fn place(&mut self, step: SegStep<frame::Audio>) -> Result<()> {
        self.dropped_frames += step.dropped.len();
        for (at, mut frame) in step.released {
            match self.sink.push(&mut frame, at, &mut self.ready) {
                Ok(()) => self.budget.ok(),
                Err(e) => self.budget.fail(e)?,
            }
        }
        Ok(())
    }
}

struct Run {
    ctx: resampling::Context,
    input_end: f64,
    output_pos: SamplePos,
}

struct MonoSink {
    time_base: f64,
    run: Option<Run>,
    published_end: SamplePos,
}

impl MonoSink {
    fn push(
        &mut self,
        frame: &mut frame::Audio,
        at: LogicalTs,
        ready: &mut VecDeque<AudioChunk>,
    ) -> std::result::Result<(), ffmpeg::Error> {
        if frame.rate() == 0 || frame.samples() == 0 {
            return Err(ffmpeg::Error::InvalidData);
        }
        if frame.channel_layout().is_empty() {
            frame.set_channel_layout(ChannelLayout::default(frame.channels() as i32));
        }
        let at = at.0;
        let expected = self.run.as_ref().map_or(at, |run| run.input_end);
        let position = SamplePos::from_seconds(at)?;
        // Container PTS rounding must not split a continuous sample clock.
        let tolerance = self.time_base.abs() + 1.0 / f64::from(frame.rate());
        let continuous = self.run.is_some() && (at - expected).abs() <= tolerance;
        let params = resampling::context::Definition {
            format: frame.format(),
            channel_layout: frame.channel_layout(),
            rate: frame.rate(),
        };
        if !continuous
            || self
                .run
                .as_ref()
                .is_none_or(|run| *run.ctx.input() != params)
        {
            let ctx = resampling::Context::get(
                params.format,
                params.channel_layout,
                params.rate,
                Sample::F32(Type::Packed),
                ChannelLayout::MONO,
                WHISPER_SAMPLE_RATE,
            )?;
            self.drain(ready)?;
            let output_pos = if continuous {
                self.run.as_ref().expect("continuous run").output_pos
            } else {
                position
            };
            self.run = Some(Run {
                ctx,
                input_end: expected,
                output_pos,
            });
        }
        let run = self.run.as_mut().expect("resampler configured");
        let capacity = (frame.samples() as u64 * u64::from(WHISPER_SAMPLE_RATE))
            .div_ceil(u64::from(frame.rate())) as usize
            + 256;
        let mut output =
            frame::Audio::new(Sample::F32(Type::Packed), capacity, ChannelLayout::MONO);
        run.ctx.run(frame, &mut output)?;
        run.input_end = if continuous { expected } else { at }
            + frame.samples() as f64 / f64::from(frame.rate());
        Self::publish(run, &mut self.published_end, &output, ready);
        Ok(())
    }

    fn drain(
        &mut self,
        ready: &mut VecDeque<AudioChunk>,
    ) -> std::result::Result<(), ffmpeg::Error> {
        let Some(run) = self.run.as_mut() else {
            return Ok(());
        };
        loop {
            let mut output =
                frame::Audio::new(Sample::F32(Type::Packed), 4096, ChannelLayout::MONO);
            let delay = run.ctx.flush(&mut output)?;
            Self::publish(run, &mut self.published_end, &output, ready);
            if delay.is_none() || output.samples() == 0 {
                return Ok(());
            }
        }
    }

    fn publish(
        run: &mut Run,
        end: &mut SamplePos,
        output: &frame::Audio,
        ready: &mut VecDeque<AudioChunk>,
    ) {
        let start = run.output_pos;
        run.output_pos = start.advance(output.samples());
        let skip = end.0.saturating_sub(start.0).max(0) as u64;
        if skip >= output.samples() as u64 {
            return;
        }
        let chunk = AudioChunk {
            start: start.advance(skip as usize),
            samples: output.plane::<f32>(0)[skip as usize..output.samples()].to_vec(),
        };
        *end = chunk.end();
        ready.push_back(chunk);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trims_preroll_and_overlap() {
        let mut sink = MonoSink {
            time_base: 1.0 / 16_000.0,
            run: None,
            published_end: SamplePos(0),
        };
        let mut chunks = VecDeque::new();
        for (at, n, value) in [(-0.5, 16_000, 1.0), (0.25, 16_000, 2.0), (1.25, 8_000, 3.0)] {
            let mut audio = frame::Audio::new(Sample::F32(Type::Packed), n, ChannelLayout::MONO);
            audio.set_rate(16_000);
            audio.plane_mut::<f32>(0).fill(value);
            sink.push(&mut audio, LogicalTs(at), &mut chunks)
                .expect("coherent audio frame");
        }
        sink.drain(&mut chunks).expect("resampler tail");
        let samples: Vec<_> = chunks
            .into_iter()
            .flat_map(|chunk| {
                chunk
                    .samples
                    .into_iter()
                    .enumerate()
                    .map(move |(i, sample)| (chunk.start.advance(i), sample))
            })
            .collect();
        assert_eq!(samples.len(), 28_000);
        for (i, (position, sample)) in samples.into_iter().enumerate() {
            assert_eq!(position, SamplePos::from_samples(i as i64));
            assert_eq!(
                sample,
                if i < 8_000 {
                    1.0
                } else if i < 20_000 {
                    2.0
                } else {
                    3.0
                }
            );
        }
    }
}
