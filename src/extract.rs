//! Frame extraction: the only place that talks to ffmpeg for decoding.
//!
//! ffmpeg samples frames at a fixed fps and pre-scales each one (via swscale,
//! its optimized C path) down to the size we actually need before writing a raw
//! PPM stream to stdout — so the pipe carries small frames, not full-resolution
//! ones, and Rust does almost no resizing. We parse the stream incrementally;
//! stderr is sent to `/dev/null`, and we deliberately do *not* use the
//! `showinfo` filter, which would force a second pipe and risk a deadlock when
//! its buffer fills. Timestamps are derived as `index / fps`.
//!
//! Grayscale derivation (one per frame, for scoring) is independent across
//! frames, so it runs in parallel after the frames are read off the pipe.

use std::io::{BufRead, BufReader};
use std::path::Path;
use std::process::{Command, Stdio};

use anyhow::{Context, Result, bail};
use image::imageops::FilterType;
use image::{GrayImage, RgbImage};
use rayon::prelude::*;

use crate::types::{RawScores, ScoredFrame};

/// Fixed minimum edge for the grayscale scoring image. Scoring on a uniform
/// scale keeps Laplacian variance / Sobel energy comparable across frames
/// regardless of source resolution.
const SCORE_EDGE: u32 = 256;

/// Result of sampling a video: the frames plus the source aspect ratio the
/// layout engine needs.
pub struct ExtractResult {
    /// Source frame aspect ratio `width / height`. Constant for the whole clip.
    pub aspect: f32,
    pub frames: Vec<ScoredFrame>,
}

/// Sample `input` at `fps`, returning downscaled frames.
///
/// `cell_min_edge` is the layout's minimum cell edge; the composition RGB is
/// kept at that edge. Hero cells (up to 2×2) upscale modestly from it — an
/// acceptable trade for bounded memory, since every extracted frame is held in
/// RAM before dedup/selection runs. Raw scores are left at their defaults —
/// [`crate::score`] fills them per chunk.
pub fn extract_frames(input: &Path, fps: f32, cell_min_edge: u32) -> Result<ExtractResult> {
    extract_inner(input, fps, cell_min_edge, None)
}

/// Like [`extract_frames`] but restricted to the time window `[start, end)`
/// seconds. Timestamps in the result are absolute (offset by `start`).
pub fn extract_range(
    input: &Path,
    fps: f32,
    cell_min_edge: u32,
    start: f32,
    end: f32,
) -> Result<ExtractResult> {
    extract_inner(input, fps, cell_min_edge, Some((start, end)))
}

fn extract_inner(
    input: &Path,
    fps: f32,
    cell_min_edge: u32,
    range: Option<(f32, f32)>,
) -> Result<ExtractResult> {
    let rgb_edge = cell_min_edge.max(1);
    // ffmpeg pre-scales the shorter edge down to this in its optimized C
    // (swscale) before the PPM hits the pipe — far cheaper than shipping
    // full-resolution frames and resizing them in Rust, and it shrinks pipe
    // bandwidth. `rgb_edge >= SCORE_EDGE` always (the smallest cell is 256),
    // so the piped frame already serves as the composition RGB as-is; only the
    // grayscale needs a further (cheap) downscale.
    let pipe_edge = rgb_edge.max(SCORE_EDGE);

    let mut cmd = Command::new("ffmpeg");
    // Seek before -i for speed; -t bounds the duration read.
    if let Some((start, end)) = range {
        cmd.args(["-ss", &start.to_string()]);
        cmd.args(["-t", &(end - start).max(0.0).to_string()]);
    }
    // `force_original_aspect_ratio=increase` scales so the shorter edge lands
    // exactly on `pipe_edge`, preserving aspect.
    let vf = format!(
        "fps={fps},scale={pipe_edge}:{pipe_edge}:force_original_aspect_ratio=increase"
    );
    let mut child = cmd
        .arg("-i")
        .arg(input)
        .arg("-vf")
        .arg(vf)
        .args(["-f", "image2pipe", "-vcodec", "ppm", "pipe:1"])
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .context("failed to spawn ffmpeg (is it on PATH?)")?;

    let stdout = child.stdout.take().expect("stdout was piped");
    let mut reader = BufReader::new(stdout);

    // Read the (already small) RGB frames sequentially off the pipe; deriving
    // each grayscale is independent per frame, so that runs in parallel after.
    let mut rgbs: Vec<RgbImage> = Vec::new();
    let mut aspect = 1.0_f32;
    while let Some(frame) = read_ppm(&mut reader)? {
        if rgbs.is_empty() {
            let (w, h) = frame.dimensions();
            aspect = w as f32 / h as f32;
        }
        rgbs.push(frame);
    }

    let status = child.wait().context("waiting for ffmpeg")?;
    if !status.success() && rgbs.is_empty() {
        bail!("ffmpeg exited with {status} and produced no frames");
    }

    let start_offset = range.map(|(s, _)| s).unwrap_or(0.0);
    let frames: Vec<ScoredFrame> = rgbs
        .into_par_iter()
        .enumerate()
        .map(|(idx, rgb)| ScoredFrame {
            ts: start_offset + idx as f32 / fps,
            gray: to_gray(&rgb, SCORE_EDGE),
            rgb,
            raw: RawScores::default(),
            score: 0.0,
        })
        .collect();

    Ok(ExtractResult { aspect, frames })
}

/// Derive a grayscale image (shorter edge `min_edge`) from an already-small RGB
/// frame. Converts in place (no full-image clone) and only resizes when the
/// source is larger than the target.
fn to_gray(img: &RgbImage, min_edge: u32) -> GrayImage {
    let gray = image::imageops::grayscale(img);
    let (w, h) = gray.dimensions();
    let (nw, nh) = target_dims(w, h, min_edge);
    if (nw, nh) == (w, h) {
        gray
    } else {
        image::imageops::resize(&gray, nw, nh, FilterType::Triangle)
    }
}

/// Aspect-preserving target dimensions for a shorter-edge of `min_edge`.
/// Returns the original dimensions unchanged if it would mean upscaling.
fn target_dims(w: u32, h: u32, min_edge: u32) -> (u32, u32) {
    let cur_min = w.min(h);
    if cur_min <= min_edge {
        return (w, h);
    }
    let scale = min_edge as f32 / cur_min as f32;
    let nw = ((w as f32 * scale).round() as u32).max(1);
    let nh = ((h as f32 * scale).round() as u32).max(1);
    (nw, nh)
}

/// Read one PPM (P6) image from a buffered reader, or `None` at clean EOF.
fn read_ppm<R: BufRead>(r: &mut R) -> Result<Option<RgbImage>> {
    let magic = match read_token(r)? {
        None => return Ok(None),
        Some(m) => m,
    };
    if magic != "P6" {
        bail!("unexpected PPM magic: {magic:?}");
    }
    let w: u32 = read_token(r)?.context("eof in PPM header")?.parse()?;
    let h: u32 = read_token(r)?.context("eof in PPM header")?.parse()?;
    let maxval: u32 = read_token(r)?.context("eof in PPM header")?.parse()?;
    if maxval != 255 {
        bail!("unsupported PPM maxval {maxval} (expected 255)");
    }

    let mut data = vec![0u8; (w as usize) * (h as usize) * 3];
    r.read_exact(&mut data).context("eof in PPM pixel data")?;

    RgbImage::from_raw(w, h, data)
        .map(Some)
        .context("PPM dimensions do not match pixel data")
}

/// Read one whitespace-delimited token from a PPM header, skipping leading
/// whitespace and `#` comment lines. Consumes exactly one trailing whitespace
/// byte as the terminator, so binary pixel data following `maxval` is untouched.
/// Returns `None` at EOF before any token byte.
fn read_token<R: BufRead>(r: &mut R) -> Result<Option<String>> {
    let mut buf = Vec::new();
    // Skip leading whitespace and comment lines.
    loop {
        let mut b = [0u8; 1];
        if r.read(&mut b)? == 0 {
            return Ok(None);
        }
        let c = b[0];
        if c == b'#' {
            loop {
                let mut d = [0u8; 1];
                if r.read(&mut d)? == 0 {
                    return Ok(None);
                }
                if d[0] == b'\n' {
                    break;
                }
            }
            continue;
        }
        if c.is_ascii_whitespace() {
            continue;
        }
        buf.push(c);
        break;
    }
    // Accumulate until the terminating whitespace byte.
    loop {
        let mut b = [0u8; 1];
        if r.read(&mut b)? == 0 {
            break;
        }
        if b[0].is_ascii_whitespace() {
            break;
        }
        buf.push(b[0]);
    }
    Ok(Some(String::from_utf8(buf)?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn parses_single_ppm_frame() {
        // 2x1 image: red, green pixels.
        let bytes = b"P6\n2 1\n255\n\xff\x00\x00\x00\xff\x00";
        let mut r = Cursor::new(&bytes[..]);
        let img = read_ppm(&mut r).expect("parse ok").expect("one frame");
        assert_eq!(img.dimensions(), (2, 1));
        assert_eq!(img.get_pixel(0, 0).0, [255, 0, 0]);
        assert_eq!(img.get_pixel(1, 0).0, [0, 255, 0]);
    }

    #[test]
    fn returns_none_at_clean_eof() {
        let mut r = Cursor::new(&b""[..]);
        assert!(read_ppm(&mut r).expect("ok").is_none());
    }

    #[test]
    fn reads_two_concatenated_frames() {
        let one = b"P6\n1 1\n255\n\x11\x22\x33";
        let mut bytes = Vec::new();
        bytes.extend_from_slice(one);
        bytes.extend_from_slice(one);
        let mut r = Cursor::new(bytes);
        assert!(read_ppm(&mut r).expect("ok").is_some());
        assert!(read_ppm(&mut r).expect("ok").is_some());
        assert!(read_ppm(&mut r).expect("ok").is_none());
    }

    #[test]
    fn skips_comment_in_header() {
        let bytes = b"P6\n# generated by test\n1 1\n255\n\xaa\xbb\xcc";
        let mut r = Cursor::new(&bytes[..]);
        let img = read_ppm(&mut r).expect("ok").expect("frame");
        assert_eq!(img.get_pixel(0, 0).0, [0xaa, 0xbb, 0xcc]);
    }

    #[test]
    fn does_not_upscale_small_source() {
        assert_eq!(target_dims(100, 50, 256), (100, 50));
    }

    #[test]
    fn downscales_preserving_aspect() {
        assert_eq!(target_dims(1920, 1080, 270), (480, 270));
    }
}
