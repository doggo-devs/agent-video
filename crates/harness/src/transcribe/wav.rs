// SPDX-License-Identifier: Apache-2.0
//! Minimal 16-bit PCM mono WAV encoder. A general-purpose codec: it knows
//! sample rates and RIFF framing, not transcription.

/// Bytes of RIFF + fmt + data-chunk framing before the first sample.
pub(super) const HEADER_BYTES: usize = 44;

/// Bytes each f32 sample occupies once encoded as 16-bit PCM.
pub(super) const BYTES_PER_SAMPLE: usize = 2;

/// Encode f32 samples (clamped to [-1, 1]) as a 16-bit PCM mono WAV at
/// `sample_rate` Hz.
pub(super) fn encode(samples: &[f32], sample_rate: u32) -> Vec<u8> {
    let num_samples = samples.len();
    let data_bytes = num_samples * BYTES_PER_SAMPLE;
    let file_size = 36 + data_bytes as u32;
    let mut buf = Vec::with_capacity(HEADER_BYTES + data_bytes);

    // RIFF header
    buf.extend_from_slice(b"RIFF");
    buf.extend_from_slice(&file_size.to_le_bytes());
    buf.extend_from_slice(b"WAVE");

    // fmt chunk
    buf.extend_from_slice(b"fmt ");
    buf.extend_from_slice(&16u32.to_le_bytes()); // chunk size
    buf.extend_from_slice(&1u16.to_le_bytes()); // PCM
    buf.extend_from_slice(&1u16.to_le_bytes()); // mono
    buf.extend_from_slice(&sample_rate.to_le_bytes()); // sample rate
    buf.extend_from_slice(&(sample_rate * BYTES_PER_SAMPLE as u32).to_le_bytes()); // byte rate
    buf.extend_from_slice(&(BYTES_PER_SAMPLE as u16).to_le_bytes()); // block align
    buf.extend_from_slice(&16u16.to_le_bytes()); // bits per sample

    // data chunk
    buf.extend_from_slice(b"data");
    buf.extend_from_slice(&(data_bytes as u32).to_le_bytes());
    for &s in samples {
        let clamped = s.clamp(-1.0, 1.0);
        let i = (clamped * 32767.0) as i16;
        buf.extend_from_slice(&i.to_le_bytes());
    }

    buf
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encodes_valid_wav_header() {
        let samples = vec![0.0f32; 16000]; // 1 second at 16 kHz
        let wav = encode(&samples, 16_000);
        assert_eq!(&wav[0..4], b"RIFF");
        assert_eq!(&wav[8..12], b"WAVE");
        assert_eq!(&wav[12..16], b"fmt ");
        assert_eq!(
            u32::from_le_bytes(wav[24..28].try_into().expect("4 bytes")),
            16_000,
            "the sample rate is the caller's, not a baked-in constant"
        );
        assert_eq!(&wav[36..40], b"data");
        // data size = 16000 * 2 = 32000
        assert_eq!(
            u32::from_le_bytes(wav[40..44].try_into().expect("4 bytes")),
            32000
        );
        assert_eq!(wav.len(), HEADER_BYTES + samples.len() * BYTES_PER_SAMPLE);
    }
}
