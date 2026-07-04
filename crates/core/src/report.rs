//! `probe` subcommand: single-pass media inspection to JSON.
//!
//! One `ffprobe -show_format -show_streams` call reduced to the machine
//! interface the harness mirrors in `contract.ts`; `--keyframes` appends the
//! slower packet-scan statistics used to pick a strategy for long videos.

use std::path::Path;
use std::process::Command;

use serde::{Deserialize, Serialize};

use crate::error::{Error, ErrorCode, ErrorContext, Result};
use crate::{pipeline, probe, subtitle};

/// ffprobe emits stream/format metadata here, never packet data; a probe
/// document approaching this cap is runaway output, not a real result.
const REPORT_STDOUT_CAP: u64 = 16 * 1024 * 1024;

/// Machine-readable output of `agent-video probe`.
#[derive(Debug, Serialize)]
pub struct ProbeReport {
    /// Container duration in seconds.
    pub duration: f32,
    pub width: u32,
    pub height: u32,
    /// Average frame rate; `0` when the container does not report one.
    pub fps: f32,
    pub video_codec: String,
    /// ffprobe `format_name`, verbatim (e.g. `mov,mp4,m4a,3gp,3g2,mj2`).
    pub container: String,
    pub size_bytes: u64,
    pub has_audio: bool,
    pub embedded_subtitles: Vec<SubtitleTrack>,
    /// Sidecar file names the subtitle auto-detection would consider.
    pub sidecar_subtitles: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub keyframes: Option<KeyframeStats>,
}

/// One embedded subtitle stream; `index` matches `--subtitle-track`.
#[derive(Debug, Serialize)]
pub struct SubtitleTrack {
    pub index: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lang: Option<String>,
    pub codec: String,
}

/// Keyframe distribution from the `--keyframes` packet scan.
#[derive(Debug, Serialize)]
pub struct KeyframeStats {
    pub count: usize,
    pub mean_gap_s: f32,
    pub max_gap_s: f32,
}

// ── ffprobe document (the subset we read) ──

#[derive(Debug, Default, Deserialize)]
struct FfprobeDoc {
    format: Option<FfprobeFormat>,
    #[serde(default)]
    streams: Vec<FfprobeStream>,
}

#[derive(Debug, Default, Deserialize)]
struct FfprobeFormat {
    duration: Option<String>,
    size: Option<String>,
    format_name: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
struct FfprobeStream {
    codec_type: Option<String>,
    codec_name: Option<String>,
    width: Option<u32>,
    height: Option<u32>,
    duration: Option<String>,
    avg_frame_rate: Option<String>,
    r_frame_rate: Option<String>,
    tags: Option<FfprobeTags>,
}

#[derive(Debug, Default, Deserialize)]
struct FfprobeTags {
    language: Option<String>,
}

/// Run the `probe` subcommand: inspect `input` and print JSON to stdout.
pub fn run(input: &Path, keyframes: bool) -> anyhow::Result<()> {
    pipeline::validate_input(input)?;
    let doc = ffprobe_doc(input)?;
    let size_fallback = std::fs::metadata(input).map(|m| m.len()).unwrap_or(0);
    let sidecars = subtitle::find_all_sidecars(input)
        .into_iter()
        .filter_map(|p| p.file_name().map(|n| n.to_string_lossy().into_owned()))
        .collect();

    let mut report = reduce(doc, size_fallback, sidecars).map_err(|e| e.with_input(input))?;
    if keyframes {
        let info = probe::probe_keyframes(input)?;
        report.keyframes = Some(KeyframeStats {
            count: info.count(),
            mean_gap_s: info.mean_gap(),
            max_gap_s: info.max_gap(),
        });
    }

    println!("{}", serde_json::to_string(&report)?);
    Ok(())
}

fn ffprobe_doc(input: &Path) -> Result<FfprobeDoc> {
    let mut cmd = Command::new("ffprobe");
    cmd.args([
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
    ])
    .arg(input);
    let stdout = crate::child::run_capped(
        &mut cmd,
        "ffprobe media inspection",
        ErrorCode::FfprobeNotFound,
        ErrorCode::FfprobeFailed,
        REPORT_STDOUT_CAP,
    )
    .map_err(|e| e.with_input(input))?;

    serde_json::from_str(&stdout).map_err(|e| {
        Error::new(
            ErrorCode::FfprobeFailed,
            format!("cannot parse ffprobe JSON output: {e}"),
        )
        .with_context(ErrorContext::default().with_input(input))
    })
}

/// Reduce an ffprobe document to the report; pure so the parsing rules
/// (duration fallback, rate strings, stream classification) stay testable.
fn reduce(doc: FfprobeDoc, size_fallback: u64, sidecars: Vec<String>) -> Result<ProbeReport> {
    let video = doc
        .streams
        .iter()
        .find(|s| {
            s.codec_type.as_deref() == Some("video")
                && s.width.unwrap_or(0) > 0
                && s.height.unwrap_or(0) > 0
        })
        .ok_or_else(|| {
            Error::new(
                ErrorCode::InputCorrupt,
                "no video stream found — is this actually a video file?",
            )
        })?;

    // Container duration is authoritative; some containers only carry it on
    // the stream. "N/A" or negative values are unusable either way.
    let duration = [
        doc.format.as_ref().and_then(|f| f.duration.as_deref()),
        video.duration.as_deref(),
    ]
    .into_iter()
    .flatten()
    .find_map(|s| {
        s.trim()
            .parse::<f32>()
            .ok()
            .filter(|d| d.is_finite() && *d >= 0.0)
    })
    .ok_or_else(|| {
        Error::new(
            ErrorCode::InputCorrupt,
            "cannot determine video duration; the container may not report one",
        )
    })?;

    // avg_frame_rate degenerates to "0/0" on some streams; r_frame_rate is
    // the fallback estimate.
    let fps = [
        video.avg_frame_rate.as_deref(),
        video.r_frame_rate.as_deref(),
    ]
    .into_iter()
    .flatten()
    .find_map(parse_rate)
    .unwrap_or(0.0);

    let size_bytes = doc
        .format
        .as_ref()
        .and_then(|f| f.size.as_deref())
        .and_then(|s| s.trim().parse::<u64>().ok())
        .unwrap_or(size_fallback);

    let embedded_subtitles = doc
        .streams
        .iter()
        .filter(|s| s.codec_type.as_deref() == Some("subtitle"))
        .enumerate()
        .map(|(index, s)| SubtitleTrack {
            index,
            lang: s.tags.as_ref().and_then(|t| t.language.clone()),
            codec: s.codec_name.clone().unwrap_or_else(|| "unknown".into()),
        })
        .collect();

    Ok(ProbeReport {
        duration,
        width: video.width.unwrap_or(0),
        height: video.height.unwrap_or(0),
        fps,
        video_codec: video.codec_name.clone().unwrap_or_else(|| "unknown".into()),
        container: doc
            .format
            .and_then(|f| f.format_name)
            .unwrap_or_else(|| "unknown".into()),
        size_bytes,
        has_audio: doc
            .streams
            .iter()
            .any(|s| s.codec_type.as_deref() == Some("audio")),
        embedded_subtitles,
        sidecar_subtitles: sidecars,
        keyframes: None,
    })
}

/// Parse an ffprobe rational rate ("30000/1001") into fps, `None` when the
/// rate is degenerate ("0/0") or malformed.
fn parse_rate(rate: &str) -> Option<f32> {
    let (num, den) = rate.split_once('/')?;
    let (num, den) = (num.parse::<f32>().ok()?, den.parse::<f32>().ok()?);
    let fps = num / den;
    (fps.is_finite() && fps > 0.0).then_some(fps)
}
