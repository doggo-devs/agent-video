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
//! Grayscale derivation (one per frame, for scoring) is independent across a
//! caller-sized batch, so it can run in parallel without retaining the whole
//! video in memory.

use std::io::{BufRead, BufReader, Read};
use std::path::Path;
use std::process::{Child, ChildStdout, Command, Stdio};

use anyhow::{Context, Result, bail};
use image::imageops::FilterType;
use image::{GrayImage, RgbImage};
use rayon::prelude::*;

use crate::types::{RawScores, ScoredFrame};

/// Default edge length for the grayscale scoring image.
const DEFAULT_SCORE_EDGE: u32 = 256;

/// One sampled RGB frame from ffmpeg, already downscaled for composition.
///
/// The frame owns its pixel buffer. Call [`scored_frames_from_rgbs`] on a small
/// batch to derive scoring grayscale images while keeping memory bounded by the
/// batch size rather than by the full video.
#[derive(Debug)]
pub struct RgbFrame {
    /// Presentation timestamp in seconds.
    pub ts: f32,
    /// Downscaled RGB image used later by composition.
    pub rgb: RgbImage,
}

/// Streaming reader over ffmpeg's PPM image pipe.
///
/// Frames are pulled one by one with [`FrameStream::next_frame`]. The stream
/// waits for ffmpeg at EOF and reports a decoding failure if ffmpeg produced no
/// frames. Dropping the stream early leaves child cleanup to the OS; callers in
/// this crate consume it to EOF.
pub struct FrameStream {
    child: Child,
    reader: BufReader<ChildStdout>,
    tok: Vec<u8>,
    fps: f32,
    start_offset: f32,
    idx: usize,
    aspect: Option<f32>,
    eof: bool,
}

impl FrameStream {
    /// Return the source frame aspect ratio once at least one frame was read.
    pub fn aspect(&self) -> Option<f32> {
        self.aspect
    }

    /// Read the next sampled frame, or `None` after a clean EOF.
    ///
    /// The timestamp is derived as `start + index / fps`. On EOF this method
    /// waits for ffmpeg and surfaces a non-zero exit status if no frame was
    /// decoded.
    pub fn next_frame(&mut self) -> Result<Option<RgbFrame>> {
        if self.eof {
            return Ok(None);
        }

        let Some(rgb) = read_ppm(&mut self.reader, &mut self.tok)? else {
            self.eof = true;
            let status = self.child.wait().context("waiting for ffmpeg")?;
            if !status.success() && self.idx == 0 {
                bail!("ffmpeg exited with {status} and produced no frames");
            }
            return Ok(None);
        };

        if self.aspect.is_none() {
            let (w, h) = rgb.dimensions();
            self.aspect = Some(w as f32 / h as f32);
        }

        let ts = self.start_offset + self.idx as f32 / self.fps;
        self.idx += 1;
        Ok(Some(RgbFrame { ts, rgb }))
    }
}

/// Open a streaming extractor for the whole video.
pub fn stream_frames(
    input: &Path,
    fps: f32,
    cell_min_edge: u32,
    score_edge: u32,
) -> Result<FrameStream> {
    stream_inner(input, fps, cell_min_edge, score_edge, None)
}

/// Open a streaming extractor for `[start, end)` seconds.
///
/// Timestamps produced by [`FrameStream::next_frame`] are absolute, offset by
/// `start`.
pub fn stream_range(
    input: &Path,
    fps: f32,
    cell_min_edge: u32,
    score_edge: u32,
    start: f32,
    end: f32,
) -> Result<FrameStream> {
    stream_inner(input, fps, cell_min_edge, score_edge, Some((start, end)))
}

/// Convert a bounded RGB batch to [`ScoredFrame`]s, deriving grayscale in parallel.
pub fn scored_frames_from_rgbs(frames: Vec<RgbFrame>, score_edge: u32) -> Vec<ScoredFrame> {
    let score_edge = if score_edge == 0 {
        DEFAULT_SCORE_EDGE
    } else {
        score_edge
    };
    frames
        .into_par_iter()
        .map(|frame| scored_frame_from_rgb(frame, score_edge))
        .collect()
}

/// Convert a single RGB frame to a [`ScoredFrame`] with grayscale for scoring.
pub fn scored_frame_from_rgb(frame: RgbFrame, score_edge: u32) -> ScoredFrame {
    let score_edge = if score_edge == 0 {
        DEFAULT_SCORE_EDGE
    } else {
        score_edge
    };
    ScoredFrame {
        ts: frame.ts,
        gray: to_gray(&frame.rgb, score_edge),
        rgb: frame.rgb,
        raw: RawScores::default(),
        score: 0.0,
    }
}

fn stream_inner(
    input: &Path,
    fps: f32,
    cell_min_edge: u32,
    score_edge: u32,
    range: Option<(f32, f32)>,
) -> Result<FrameStream> {
    let rgb_edge = cell_min_edge.max(1);
    let score_edge = if score_edge == 0 {
        DEFAULT_SCORE_EDGE
    } else {
        score_edge
    };
    // The piped frame doubles as the composition RGB, so pre-scale to the larger
    // edge; `rgb_edge >= score_edge` typically, so only the grayscale downscales
    // further.
    let pipe_edge = rgb_edge.max(score_edge);

    let mut cmd = Command::new("ffmpeg");
    // Seek before -i for speed; -t bounds the duration read.
    if let Some((start, end)) = range {
        cmd.args(["-ss", &start.to_string()]);
        cmd.args(["-t", &(end - start).max(0.0).to_string()]);
    }
    // `force_original_aspect_ratio=increase` scales so the shorter edge lands
    // exactly on `pipe_edge`, preserving aspect.
    let vf =
        format!("fps={fps},scale={pipe_edge}:{pipe_edge}:force_original_aspect_ratio=increase");
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
    Ok(FrameStream {
        child,
        reader: BufReader::new(stdout),
        tok: Vec::new(),
        fps,
        start_offset: range.map(|(s, _)| s).unwrap_or(0.0),
        idx: 0,
        aspect: None,
        eof: false,
    })
}

/// Downscale RGB to grayscale with shorter edge `min_edge`; no-op if already smaller.
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
/// `tok` is a scratch buffer reused across calls so header parsing allocates
/// nothing per frame.
fn read_ppm<R: BufRead>(r: &mut R, tok: &mut Vec<u8>) -> Result<Option<RgbImage>> {
    if !read_token(r, tok)? {
        return Ok(None);
    }
    if tok.as_slice() != b"P6" {
        bail!("unexpected PPM magic: {:?}", String::from_utf8_lossy(tok));
    }
    let w = read_header_u32(r, tok)?;
    let h = read_header_u32(r, tok)?;
    let maxval = read_header_u32(r, tok)?;
    if maxval != 255 {
        bail!("unsupported PPM maxval {maxval} (expected 255)");
    }

    // Avoid pre-zeroing: `vec![0u8; n]` would memset a frame that the read then
    // fully overwrites. `take` + `read_to_end` fills uninitialised spare capacity
    // directly (std's BorrowedBuf path) — no memset, no unsafe.
    let n = (w as usize) * (h as usize) * 3;
    let mut data = Vec::with_capacity(n);
    (&mut *r)
        .take(n as u64)
        .read_to_end(&mut data)
        .context("reading PPM pixel data")?;
    if data.len() != n {
        bail!("eof in PPM pixel data");
    }

    RgbImage::from_raw(w, h, data)
        .map(Some)
        .context("PPM dimensions do not match pixel data")
}

/// Read a numeric PPM-header token into `tok` and parse it, erroring on EOF.
fn read_header_u32<R: BufRead>(r: &mut R, tok: &mut Vec<u8>) -> Result<u32> {
    if !read_token(r, tok)? {
        bail!("eof in PPM header");
    }
    let s = std::str::from_utf8(tok).context("non-utf8 in PPM header")?;
    s.parse()
        .with_context(|| format!("invalid PPM header number: {s:?}"))
}

/// Read one whitespace-delimited PPM-header token into `buf` (cleared first),
/// skipping leading whitespace and `#` comment lines and consuming exactly one
/// trailing whitespace byte, so the binary pixel data after `maxval` stays
/// intact. Reuses `buf` across calls (no per-token allocation). Returns `false`
/// at EOF before any token byte.
fn read_token<R: BufRead>(r: &mut R, buf: &mut Vec<u8>) -> Result<bool> {
    buf.clear();
    let (mut in_comment, mut in_token) = (false, false);
    loop {
        let chunk = r.fill_buf()?;
        if chunk.is_empty() {
            // EOF finishes a token already being accumulated; otherwise none.
            return Ok(in_token);
        }
        let mut consumed = chunk.len();
        let mut done = false;
        for (i, &c) in chunk.iter().enumerate() {
            if in_comment {
                if c == b'\n' {
                    in_comment = false;
                }
            } else if !in_token {
                if c == b'#' {
                    in_comment = true;
                } else if !c.is_ascii_whitespace() {
                    in_token = true;
                    buf.push(c);
                }
            } else if c.is_ascii_whitespace() {
                consumed = i + 1; // consume through the terminator
                done = true;
                break;
            } else {
                buf.push(c);
            }
        }
        r.consume(consumed); // chunk borrow ends here, so consume is allowed
        if done {
            return Ok(true);
        }
    }
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
        let img = read_ppm(&mut r, &mut Vec::new())
            .expect("parse ok")
            .expect("one frame");
        assert_eq!(img.dimensions(), (2, 1));
        assert_eq!(img.get_pixel(0, 0).0, [255, 0, 0]);
        assert_eq!(img.get_pixel(1, 0).0, [0, 255, 0]);
    }

    #[test]
    fn returns_none_at_clean_eof() {
        let mut r = Cursor::new(&b""[..]);
        assert!(read_ppm(&mut r, &mut Vec::new()).expect("ok").is_none());
    }

    #[test]
    fn reads_two_concatenated_frames() {
        let one = b"P6\n1 1\n255\n\x11\x22\x33";
        let mut bytes = Vec::new();
        bytes.extend_from_slice(one);
        bytes.extend_from_slice(one);
        let mut r = Cursor::new(bytes);
        let mut tok = Vec::new();
        assert!(read_ppm(&mut r, &mut tok).expect("ok").is_some());
        assert!(read_ppm(&mut r, &mut tok).expect("ok").is_some());
        assert!(read_ppm(&mut r, &mut tok).expect("ok").is_none());
    }

    #[test]
    fn skips_comment_in_header() {
        let bytes = b"P6\n# generated by test\n1 1\n255\n\xaa\xbb\xcc";
        let mut r = Cursor::new(&bytes[..]);
        let img = read_ppm(&mut r, &mut Vec::new())
            .expect("ok")
            .expect("frame");
        assert_eq!(img.get_pixel(0, 0).0, [0xaa, 0xbb, 0xcc]);
    }

    #[test]
    fn parses_header_split_across_buffer_refills() {
        // Capacity-1 BufReader makes fill_buf yield one byte at a time, so the
        // token state machine must survive a refill mid-token and mid-comment.
        let bytes = b"P6\n# split header\n2 1\n255\n\xff\x00\x00\x00\xff\x00";
        let mut r = BufReader::with_capacity(1, Cursor::new(&bytes[..]));
        let img = read_ppm(&mut r, &mut Vec::new())
            .expect("ok")
            .expect("frame");
        assert_eq!(img.dimensions(), (2, 1));
        assert_eq!(img.get_pixel(1, 0).0, [0, 255, 0]);
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
