//! `probe` subcommand: single-pass media inspection to JSON.
//!
//! One in-process container open ([`crate::media::inspect`]) reduced to the
//! machine interface the harness mirrors in `contract.ts`; `--keyframes`
//! appends the slower packet-scan statistics used to pick a strategy for long
//! videos.

use std::path::Path;

use serde::Serialize;

use crate::{media, pipeline, probe, subtitle};

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

/// Run the `probe` subcommand: inspect `input` and print JSON to stdout.
pub fn run(input: &Path, keyframes: bool) -> anyhow::Result<()> {
    pipeline::validate_input(input)?;
    let info = media::inspect::inspect(input)?;
    let sidecar_subtitles = subtitle::find_all_sidecars(input)
        .into_iter()
        .filter_map(|p| p.file_name().map(|n| n.to_string_lossy().into_owned()))
        .collect();

    let mut report = ProbeReport {
        duration: info.duration,
        width: info.width,
        height: info.height,
        fps: info.fps,
        video_codec: info.video_codec,
        container: info.container,
        size_bytes: std::fs::metadata(input).map(|m| m.len()).unwrap_or(0),
        has_audio: info.has_audio,
        embedded_subtitles: info
            .subtitle_tracks
            .into_iter()
            .map(|t| SubtitleTrack {
                index: t.index,
                lang: t.lang,
                codec: t.codec,
            })
            .collect(),
        sidecar_subtitles,
        keyframes: None,
    };
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
