// SPDX-License-Identifier: Apache-2.0
//! Audio decoding for transcription: any input's best audio stream to
//! 16 kHz mono f32 PCM (what whisper.cpp consumes), in-process via
//! libswresample — the moral successor of the old `ffmpeg -ac 1 -ar 16000`
//! subprocess, minus the subprocess.

use ffmpeg_next as ffmpeg;

use ffmpeg::software::resampling;
use ffmpeg::util::channel_layout::ChannelLayout;
use ffmpeg::util::format::sample::{Sample, Type};
use ffmpeg::util::frame;

use super::{ErrorBudget, open_input};
use crate::error::{Error, ErrorCode, ErrorContext, Result};
use crate::source::Source;

/// The sample rate whisper.cpp expects.
pub const WHISPER_SAMPLE_RATE: u32 = 16_000;

/// Decode `source`'s best audio stream to 16 kHz mono f32 samples.
///
/// Fails with an input-decade error when the container has no audio stream
/// (the caller turns that into "nothing to transcribe") or when decoding
/// breaks mid-stream.
pub fn decode_audio_16k_mono(source: &Source) -> Result<Vec<f32>> {
    let mut ictx = open_input(source)?;

    let stream = ictx
        .streams()
        .best(ffmpeg::media::Type::Audio)
        .ok_or_else(|| {
            Error::new(
                ErrorCode::InputCorrupt,
                format!("{source} contains no audio stream"),
            )
            .with_context(ErrorContext::default().with_input(source.as_path()))
        })?;
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

    let mut resampler = resampling::Context::get(
        decoder.format(),
        decoder.channel_layout(),
        decoder.rate(),
        Sample::F32(Type::Packed),
        ChannelLayout::MONO,
        WHISPER_SAMPLE_RATE,
    )
    .map_err(|e| super::decode_error("configuring audio resampler", e))?;

    let mut samples: Vec<f32> = Vec::new();
    let mut budget = ErrorBudget::new("reading audio packets");
    let in_rate = decoder.rate();

    let sink = |resampler: &mut resampling::Context,
                decoded: &frame::Audio,
                samples: &mut Vec<f32>|
     -> Result<()> {
        let capacity = scaled_capacity(decoded.samples(), in_rate);
        let mut out = frame::Audio::new(Sample::F32(Type::Packed), capacity, ChannelLayout::MONO);
        resampler
            .run(decoded, &mut out)
            .map_err(|e| super::decode_error("resampling audio", e))?;
        append_samples(&out, samples);
        Ok(())
    };

    let mut decoded = frame::Audio::empty();
    for (stream, packet) in ictx.packets() {
        if stream.index() != stream_index {
            continue;
        }
        match decoder.send_packet(&packet) {
            Ok(()) => budget.ok(),
            Err(e) => {
                budget.fail(e)?;
                continue;
            }
        }
        while decoder.receive_frame(&mut decoded).is_ok() {
            sink(&mut resampler, &decoded, &mut samples)?;
        }
    }

    let _ = decoder.send_eof();
    while decoder.receive_frame(&mut decoded).is_ok() {
        sink(&mut resampler, &decoded, &mut samples)?;
    }

    // Drain the resampler's internal buffer.
    loop {
        let mut out = frame::Audio::new(Sample::F32(Type::Packed), 4096, ChannelLayout::MONO);
        let delay = resampler
            .flush(&mut out)
            .map_err(|e| super::decode_error("flushing audio resampler", e))?;
        append_samples(&out, &mut samples);
        if delay.is_none() || out.samples() == 0 {
            break;
        }
    }

    Ok(samples)
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
