//! Frame extraction: the only place that talks to ffmpeg for decoding.
//!
//! ffmpeg samples frames at a fixed fps and pre-scales each one (via swscale,
//! its optimized C path) down to the size we actually need before writing a raw
//! PPM stream to stdout — so the pipe carries small frames, not full-resolution
//! ones, and Rust does almost no resizing. We parse the stream incrementally;
//! stderr is captured and reported on failure. Timestamps are derived as
//! `index / fps`.
//!
//! Grayscale derivation (one per frame, for scoring) is independent across a
//! caller-sized batch, so it can run in parallel without retaining the whole
//! video in memory.

use std::io::{BufRead, BufReader, Read};
use std::num::NonZeroUsize;
use std::path::Path;
use std::process::{ChildStdout, Command};
use std::sync::Arc;

use anyhow::{Context, Result, bail};
use image::imageops::FilterType;
use image::{GrayImage, RgbImage};
use rayon::prelude::*;

use crate::child::TrackedChild;
use crate::error::{Error, ErrorCode};
use crate::probe::MonotonicTimestamps;
use crate::types::CandidateFrame;

/// One sampled RGB frame from ffmpeg, already downscaled for composition.
///
/// The frame owns its pixel buffer. Call [`candidate_frames_from_rgbs`] on a
/// small batch to derive scoring grayscale images while keeping memory bounded
/// by the batch size rather than by the full video.
#[derive(Debug)]
pub struct RgbFrame {
    /// Presentation timestamp in seconds.
    pub ts: f32,
    /// Downscaled RGB image used later by composition.
    pub rgb: RgbImage,
}

/// How a decoded frame's timestamp is derived.
enum TsPolicy {
    /// ts = start + idx / fps (fixed-rate sampling).
    Fps { fps: f32, start: f32 },
    /// ts from a pre-probed keyframe list.
    List(Vec<f32>),
}

impl TsPolicy {
    /// Timestamp for the frame at `idx`, or `None` once a probed keyframe list
    /// is exhausted — the decoder produced more frames than ffprobe listed, so
    /// we cannot attribute a timestamp. The `Fps` arm is always `Some`.
    fn ts_at(&self, idx: usize) -> Option<f32> {
        match self {
            TsPolicy::Fps { fps, start } => Some(start + idx as f32 / fps),
            TsPolicy::List(list) => list.get(idx).copied(),
        }
    }
}

/// Streaming reader over ffmpeg's PPM image pipe.
///
/// Frames are pulled one by one with [`FrameStream::next_frame`]. The stream
/// waits for ffmpeg at EOF and reports a decoding failure (with stderr) if
/// ffmpeg produced no frames. A frame's timestamp is derived by the stream's
/// [`TsPolicy`]: computed from a fixed fps, or read from a pre-probed keyframe
/// list.
pub struct FrameStream {
    child: Option<TrackedChild>,
    reader: BufReader<ChildStdout>,
    tok: Vec<u8>,
    ts: TsPolicy,
    idx: usize,
    aspect: Option<f32>,
    eof: bool,
    /// Context string for ffmpeg wait errors, so messages stay mode-specific.
    wait_context: &'static str,
}

impl Drop for FrameStream {
    fn drop(&mut self) {
        if let Some(child) = self.child.take() {
            child.kill_and_wait();
        }
    }
}

impl FrameStream {
    /// Return the source frame aspect ratio once at least one frame was read.
    pub fn aspect(&self) -> Option<f32> {
        self.aspect
    }

    /// Read the next sampled frame, or `None` after a clean EOF.
    ///
    /// The timestamp is derived by the stream's [`TsPolicy`]. On EOF this method
    /// waits for ffmpeg and surfaces a non-zero exit status with stderr if no
    /// frame was decoded.
    pub fn next_frame(&mut self) -> Result<Option<RgbFrame>> {
        if self.eof {
            return Ok(None);
        }

        let rgb = match read_ppm(&mut self.reader, &mut self.tok) {
            Ok(rgb) => rgb,
            Err(e) => {
                // A broken PPM stream is usually a symptom: ffmpeg's stderr
                // carries the real cause (codec errors, corrupt input). Kill
                // rather than wait — the child may be blocked writing to the
                // pipe we just stopped draining.
                self.eof = true;
                let stderr = self
                    .child
                    .take()
                    .map(TrackedChild::kill_collect_stderr)
                    .unwrap_or_default();
                if stderr.is_empty() {
                    return Err(e);
                }
                return Err(e.context(format!("ffmpeg stderr: {stderr}")));
            }
        };
        let Some(rgb) = rgb else {
            self.eof = true;
            if let Some(child) = self.child.take()
                && let Err(e) = child.wait(self.wait_context)
            {
                if self.idx == 0 {
                    return Err(e.into_error(ErrorCode::NoFramesExtracted).into());
                }
                return Err(anyhow::Error::from(e.into_error(ErrorCode::FfmpegFailed))
                    .context(format!("after {} frames (possible truncation)", self.idx)));
            }
            // ffmpeg exited cleanly but decoded fewer frames than the probed
            // keyframe list: a dropped frame anywhere before the tail shifts
            // every later frame onto the wrong timestamp, silently. Fail with
            // a code, mirroring the more-frames-than-probed case below.
            if let TsPolicy::List(list) = &self.ts
                && self.idx < list.len()
            {
                return Err(Error::new(
                    ErrorCode::InputCorrupt,
                    format!(
                        "decoder produced {} frames but ffprobe listed {} keyframes; \
                         timestamps cannot be trusted — use --strategy fixed-fps",
                        self.idx,
                        list.len()
                    ),
                )
                .into());
            }
            return Ok(None);
        };

        let Some(ts) = self.ts.ts_at(self.idx) else {
            // More decoded frames than probed keyframes: index pairing has
            // broken down, so the frames already streamed may carry shifted
            // timestamps and the tail cannot be attributed at all. Fail with a
            // code rather than exit 0 with silently partial, misaligned output.
            self.eof = true;
            if let Some(child) = self.child.take() {
                child.kill_and_wait();
            }
            return Err(Error::new(
                ErrorCode::InputCorrupt,
                format!(
                    "decoder produced more frames than the {} probed keyframes; \
                     timestamps cannot be attributed — use --strategy fixed-fps",
                    self.idx
                ),
            )
            .into());
        };

        if self.aspect.is_none() {
            let (w, h) = rgb.dimensions();
            self.aspect = Some(w as f32 / h as f32);
        }

        self.idx += 1;
        Ok(Some(RgbFrame { ts, rgb }))
    }
}

/// Open a streaming extractor for the whole video.
///
/// `threads` caps the ffmpeg decoder's thread count (`None` = ffmpeg default);
/// see [`apply_thread_cap`].
pub fn stream_frames(
    input: &Path,
    fps: f32,
    cell_min_edge: u32,
    score_edge: u32,
    threads: Option<NonZeroUsize>,
) -> Result<FrameStream> {
    stream_inner(input, fps, cell_min_edge, score_edge, None, threads)
}

/// Open a streaming extractor for `[start, end)` seconds.
///
/// Timestamps produced by [`FrameStream::next_frame`] are absolute, offset by
/// `start`. `threads` caps the ffmpeg decoder's thread count (`None` = ffmpeg
/// default).
pub fn stream_range(
    input: &Path,
    fps: f32,
    cell_min_edge: u32,
    score_edge: u32,
    start: f32,
    end: f32,
    threads: Option<NonZeroUsize>,
) -> Result<FrameStream> {
    stream_inner(
        input,
        fps,
        cell_min_edge,
        score_edge,
        Some((start, end)),
        threads,
    )
}

/// Open a keyframe-only streaming extractor.
///
/// Uses `-skip_frame nokey` so ffmpeg decodes only I-frames. Each decoded
/// frame is paired with the corresponding timestamp by index; taking
/// [`MonotonicTimestamps`] (from [`crate::probe::probe_keyframes`]) makes a
/// list whose order disagrees with the decoder's emission order
/// unconstructible. `threads` caps the ffmpeg decoder's thread count (`None`
/// = ffmpeg default).
pub fn stream_keyframes(
    input: &Path,
    timestamps: MonotonicTimestamps,
    cell_min_edge: u32,
    score_edge: u32,
    threads: Option<NonZeroUsize>,
) -> Result<FrameStream> {
    let rgb_edge = cell_min_edge.max(1);
    let pipe_edge = rgb_edge.max(score_edge);

    let vf = format!("scale={pipe_edge}:{pipe_edge}:force_original_aspect_ratio=increase");
    let mut cmd = Command::new("ffmpeg");
    apply_thread_cap(&mut cmd, threads);
    // `-map 0:v:0` pins the same stream ffprobe scanned (`-select_streams
    // v:0`); ffmpeg's default "best stream" pick can differ (e.g. cover art),
    // which would desync the index pairing between frames and timestamps.
    cmd.arg("-skip_frame")
        .arg("nokey")
        .arg("-i")
        .arg(input)
        .args(["-map", "0:v:0"])
        .arg("-vf")
        .arg(vf)
        .arg("-vsync")
        .arg("0")
        .args(["-f", "image2pipe", "-vcodec", "ppm", "pipe:1"]);

    let (child, stdout) = TrackedChild::spawn(
        &mut cmd,
        "ffmpeg for keyframe extraction",
        ErrorCode::FfmpegNotFound,
    )?;

    Ok(FrameStream {
        child: Some(child),
        reader: BufReader::new(stdout),
        tok: Vec::new(),
        ts: TsPolicy::List(timestamps.into_vec()),
        idx: 0,
        aspect: None,
        eof: false,
        wait_context: "ffmpeg keyframe extraction",
    })
}

/// Convert a bounded RGB batch to [`CandidateFrame`]s, deriving grayscale in parallel.
pub fn candidate_frames_from_rgbs(frames: Vec<RgbFrame>, score_edge: u32) -> Vec<CandidateFrame> {
    frames
        .into_par_iter()
        .map(|frame| candidate_frame_from_rgb(frame, score_edge))
        .collect()
}

/// Convert a single RGB frame to a [`CandidateFrame`] with grayscale for scoring.
fn candidate_frame_from_rgb(frame: RgbFrame, score_edge: u32) -> CandidateFrame {
    CandidateFrame {
        ts: frame.ts,
        gray: to_gray(&frame.rgb, score_edge),
        rgb: Arc::new(frame.rgb),
    }
}

fn stream_inner(
    input: &Path,
    fps: f32,
    cell_min_edge: u32,
    score_edge: u32,
    range: Option<(f32, f32)>,
    threads: Option<NonZeroUsize>,
) -> Result<FrameStream> {
    let rgb_edge = cell_min_edge.max(1);
    // The piped frame doubles as the composition RGB, so pre-scale to the larger
    // edge; `rgb_edge >= score_edge` typically, so only the grayscale downscales
    // further.
    let pipe_edge = rgb_edge.max(score_edge);

    let mut cmd = Command::new("ffmpeg");
    apply_thread_cap(&mut cmd, threads);
    // Seek before -i for speed; -t bounds the duration read.
    if let Some((start, end)) = range {
        cmd.args(["-ss", &start.to_string()]);
        cmd.args(["-t", &(end - start).max(0.0).to_string()]);
    }
    // `force_original_aspect_ratio=increase` scales so the shorter edge lands
    // exactly on `pipe_edge`, preserving aspect.
    let vf =
        format!("fps={fps},scale={pipe_edge}:{pipe_edge}:force_original_aspect_ratio=increase");
    cmd.arg("-i").arg(input).arg("-vf").arg(vf).args([
        "-f",
        "image2pipe",
        "-vcodec",
        "ppm",
        "pipe:1",
    ]);

    let (child, stdout) = TrackedChild::spawn(
        &mut cmd,
        "ffmpeg (is it on PATH?)",
        ErrorCode::FfmpegNotFound,
    )?;

    Ok(FrameStream {
        child: Some(child),
        reader: BufReader::new(stdout),
        tok: Vec::new(),
        ts: TsPolicy::Fps {
            fps,
            start: range.map(|(s, _)| s).unwrap_or(0.0),
        },
        idx: 0,
        aspect: None,
        eof: false,
        wait_context: "ffmpeg frame extraction",
    })
}

/// Cap the ffmpeg decoder's thread count.
///
/// `--threads` bounds the rayon pool, but ffmpeg subprocesses schedule their
/// own decoder threads outside Rust; without this cap a core-limited
/// environment (container, CI) still sees each ffmpeg spawn threads per its
/// own defaults. Must be applied before `-i` so it acts as an input (decoder)
/// option; PPM "encoding" on the output side is trivial and needs no cap.
fn apply_thread_cap(cmd: &mut Command, threads: Option<NonZeroUsize>) {
    if let Some(n) = threads {
        cmd.args(["-threads", &n.to_string()]);
    }
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

/// Upper bound on frames we will allocate from a PPM header: 67 MP (~200 MB
/// RGB). Our own piped frames are ≤ ~3 MP even at extreme aspect ratios, so a
/// larger dimension means a desynced or corrupt stream, not a real frame.
const MAX_FRAME_PIXELS: u64 = 1 << 26;

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
    // Never size an allocation from an untrusted header alone: a desynced stream
    // can present absurd dimensions, turning a huge allocation into an abort
    // instead of a clean error.
    if w == 0 || h == 0 || (w as u64) * (h as u64) > MAX_FRAME_PIXELS {
        bail!("implausible PPM dimensions {w}x{h}");
    }
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
    fn rejects_implausible_ppm_dimensions() {
        // A P6 header claiming a gigantic frame must error, not attempt a huge
        // allocation; a zero-size frame is likewise rejected.
        let huge = b"P6\n1000000 1000000\n255\n";
        assert!(
            read_ppm(&mut Cursor::new(&huge[..]), &mut Vec::new()).is_err(),
            "gigantic frame rejected"
        );
        let zero = b"P6\n0 0\n255\n";
        assert!(
            read_ppm(&mut Cursor::new(&zero[..]), &mut Vec::new()).is_err(),
            "zero-size frame rejected"
        );
    }

    #[test]
    fn stops_at_probed_timestamp_list_exhaustion() {
        // A probed keyframe list attributes only as many frames as it lists;
        // extras get `None` instead of a fabricated `idx`-as-seconds timestamp.
        let policy = TsPolicy::List(vec![0.0, 2.5]);
        assert_eq!(policy.ts_at(0), Some(0.0));
        assert_eq!(policy.ts_at(1), Some(2.5));
        assert_eq!(
            policy.ts_at(2),
            None,
            "no fabricated ts past the probed list"
        );
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
