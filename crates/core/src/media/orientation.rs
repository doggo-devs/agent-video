// SPDX-License-Identifier: Apache-2.0
//! Convert FFmpeg display matrices to the eight rectangular image orientations.

use ffmpeg_next as ffmpeg;
use image::metadata::Orientation;

use crate::error::{Error, ErrorCode, Result};

/// Read a stream's orientation, defaulting only when metadata is absent.
/// Malformed matrices are corrupt input; transforms beyond quarter turns and
/// reflections with uniform scale are unsupported decode operations.
pub(super) fn of_stream(stream: &ffmpeg::format::stream::Stream) -> Result<Orientation> {
    stream
        .side_data()
        .find(|sd| sd.kind() == ffmpeg::codec::packet::side_data::Type::DisplayMatrix)
        .map_or(Ok(Orientation::NoTransforms), |sd| {
            from_display_matrix(sd.data())
        })
}

fn from_display_matrix(bytes: &[u8]) -> Result<Orientation> {
    let (words, trailing) = bytes.as_chunks::<4>();
    let words = <&[[u8; 4]; 9]>::try_from(words)
        .ok()
        .filter(|_| trailing.is_empty())
        .ok_or_else(|| {
            Error::new(
                ErrorCode::InputCorrupt,
                "display matrix must contain nine i32 values",
            )
        })?;
    let [a, b, u, c, d, v, _, _, w] = words.map(i32::from_ne_bytes);
    let unsupported = || {
        Error::new(
            ErrorCode::DecodeFailed,
            "unsupported video display matrix: expected a right-angle rotation or reflection with uniform scale",
        )
    };
    if u != 0 || v != 0 {
        return Err(unsupported());
    }
    if w == 0 || i128::from(a) * i128::from(d) == i128::from(b) * i128::from(c) {
        return Err(Error::new(
            ErrorCode::InputCorrupt,
            "singular video display matrix",
        ));
    }
    // Translation disappears when tightly framing the image, and uniform scale
    // cancels when pinning its shorter edge. Perspective and unequal scale do not.
    let axes = [a, b, c, d];
    let scale = axes
        .map(i32::unsigned_abs)
        .into_iter()
        .max()
        .expect("four axes");
    if axes
        .iter()
        .any(|&axis| axis != 0 && axis.unsigned_abs() != scale)
    {
        return Err(unsupported());
    }

    // FFmpeg maps (p, q) to (a*p + c*q, b*p + d*q) / w; pixel y grows downward.
    match axes.map(|axis| axis.signum() * w.signum()) {
        [1, 0, 0, 1] => Ok(Orientation::NoTransforms),
        [0, 1, -1, 0] => Ok(Orientation::Rotate90),
        [-1, 0, 0, -1] => Ok(Orientation::Rotate180),
        [0, -1, 1, 0] => Ok(Orientation::Rotate270),
        [-1, 0, 0, 1] => Ok(Orientation::FlipHorizontal),
        [1, 0, 0, -1] => Ok(Orientation::FlipVertical),
        [0, 1, 1, 0] => Ok(Orientation::Rotate90FlipH),
        [0, -1, -1, 0] => Ok(Orientation::Rotate270FlipH),
        _ => Err(unsupported()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ONE: i32 = 1 << 16;
    const IDENTITY: [i32; 9] = [ONE, 0, 0, 0, ONE, 0, 0, 0, 1 << 30];

    fn bytes(matrix: [i32; 9]) -> Vec<u8> {
        matrix.into_iter().flat_map(i32::to_ne_bytes).collect()
    }

    #[test]
    fn rejects_malformed_or_singular_matrices() {
        let identity = bytes(IDENTITY);
        let mut inputs = vec![
            vec![],
            identity[..32].to_vec(),
            identity[..35].to_vec(),
            [identity.clone(), vec![0]].concat(),
            [identity, vec![0; 4]].concat(),
            bytes([0; 9]),
            bytes([ONE, ONE, 0, ONE, ONE, 0, 0, 0, 1 << 30]),
        ];
        let mut zero_w = IDENTITY;
        zero_w[8] = 0;
        inputs.push(bytes(zero_w));
        for input in inputs {
            assert_eq!(
                from_display_matrix(&input)
                    .expect_err("invalid matrix")
                    .code,
                ErrorCode::InputCorrupt,
                "{input:?}"
            );
        }
    }

    #[test]
    fn rejects_transforms_that_cannot_preserve_rectangular_geometry() {
        let matrices = [
            [ONE, ONE, 0, -ONE, ONE, 0, 0, 0, 1 << 30], // 45 degrees
            [ONE, 0, 0, ONE, ONE, 0, 0, 0, 1 << 30],    // shear
            [ONE, 0, 1, 0, ONE, 0, 0, 0, 1 << 30],      // perspective
            [ONE, 0, 0, 0, 2 * ONE, 0, 0, 0, 1 << 30],  // unequal scale
        ];
        for matrix in matrices {
            assert_eq!(
                from_display_matrix(&bytes(matrix))
                    .expect_err("unsupported transform")
                    .code,
                ErrorCode::DecodeFailed,
                "{matrix:?}"
            );
        }
    }

    #[test]
    fn preserves_orientation_under_translation_and_uniform_scale() {
        let matrix = [-2 * ONE, 0, 0, 0, 2 * ONE, 0, 160 * ONE, 96 * ONE, 1 << 30];

        assert_eq!(
            from_display_matrix(&bytes(matrix)).expect("scaled and translated reflection"),
            Orientation::FlipHorizontal
        );
    }
}
