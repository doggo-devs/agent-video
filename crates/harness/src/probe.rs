// SPDX-License-Identifier: Apache-2.0
//! Probe adapter: local files and direct media URLs into the model-facing
//! [`ProbeSummary`].
//!
//! Probing runs in-process over the linked FFmpeg (`media::summarize_input`)
//! — no `ffprobe` on PATH. URL probing reads the container header without
//! downloading the payload; a deadline thread arms libav's interrupt
//! callback so a hung server cannot stall the call forever.

use std::path::Path;
use std::time::Duration;

use super::schema::ProbeSummary;
use crate::fault::Fault;
use crate::relay::PolicyRelay;
use agent_video_core::media;
use agent_video_core::source::Source;
use agent_video_core::subtitle::find_sidecars;
use agent_video_core::types::format_ts;

/// How long an over-http probe may hang before we call it a non-file.
pub const URL_PROBE_TIMEOUT: Duration = Duration::from_secs(20);

/// Short container label from libav's demuxer name, which is a comma-list
/// of aliases (`mov,mp4,m4a,3gp,3g2,mj2`). This is what the model sees in
/// [`ProbeSummary::container`] *and* the extension a downloaded rip gets,
/// so every arm yields an ext-shaped token; unrecognizable input falls
/// back to `mp4`, the safest default for direct media links.
pub(crate) fn container_label(demuxer: &str) -> String {
    let first = demuxer.split(',').next().unwrap_or("").trim();
    match first {
        "matroska" => "mkv",
        "webm" => "webm",
        "mov" | "mp4" => "mp4",
        "mpegts" => "ts",
        "avi" => "avi",
        "flv" => "flv",
        "asf" => "wmv",
        "ogg" => "ogg",
        other if is_ext_shaped(other) => other,
        _ => "mp4",
    }
    .to_string()
}

/// `[a-z0-9]{2,5}` — plausible as a file extension.
pub(crate) fn is_ext_shaped(s: &str) -> bool {
    (2..=5).contains(&s.len())
        && s.bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit())
}

/// Model-facing seconds: the f32 probe value widens into f64 with a tail
/// of noise digits (406.037353515625); centiseconds are already finer than
/// any frame the tool samples, and the noise only wastes model tokens.
fn round_centis(seconds: f32) -> f64 {
    (f64::from(seconds) * 100.0).round() / 100.0
}

/// Human-readable embedded-track descriptions, mirroring the TS wire text
/// (`embedded #0 (eng, subrip)`). Sidecar lines (`sidecar x.srt`) are a
/// property of the file's directory, not its content, and are appended
/// live by callers via [`scan_sidecars`].
fn describe_subtitles(summary: &media::MediaSummary) -> Vec<String> {
    summary
        .subtitle_tracks
        .iter()
        .enumerate()
        .map(|(i, t)| match &t.language {
            Some(lang) => format!("embedded #{i} ({lang}, {})", t.codec),
            None => format!("embedded #{i} ({})", t.codec),
        })
        .collect()
}

/// Basenames of the sidecar subtitle files core's auto-detection would
/// actually pick up — the advertisement and the behavior are the same
/// discovery function, so "Subtitles available" can never promise a format
/// or a name pattern core does not read.
pub fn scan_sidecars(video: &Path) -> Vec<String> {
    find_sidecars(video)
        .into_iter()
        .filter_map(|p| p.file_name().map(|n| n.to_string_lossy().into_owned()))
        .collect()
}

/// Probe a local video file. The summary is a deterministic function of
/// the file's *content* only — no sidecar lines — so it may be cached by
/// content fingerprint. Callers who want the sidecar advertisement append
/// it live via [`scan_sidecars`].
pub fn probe_local_file(file: &Path) -> Result<ProbeSummary, Fault> {
    let source = Source::classify(&file.to_string_lossy());
    // A failed stat is an unknown size, not a zero-byte claim.
    let size = std::fs::metadata(file).ok().map(|m| m.len());
    // A local file cannot stall like a dead peer; a manual token that
    // never trips satisfies the open's stall-guard contract.
    let ictx = media::open_input(&source, &media::Stop::new(), &[])?;
    let summary = media::summarize_input(&ictx, &source)?;
    Ok(ProbeSummary {
        duration: round_centis(summary.duration),
        width: summary.width,
        height: summary.height,
        size_bytes: size,
        container: Some(container_label(&summary.container)),
        subtitles: describe_subtitles(&summary),
    })
}

/// Whether a track covering `track_s` of a `container_s`-long timeline is
/// materially truncated: more than 10 s missing *and* under 90% coverage.
/// The absolute bound ignores trailing-silence jitter on long videos, the
/// relative one keeps short videos from tripping on it.
pub(crate) fn materially_truncated(track_s: f64, container_s: f64) -> bool {
    container_s - track_s > 10.0 && track_s < container_s * 0.9
}

/// The media track whose coverage is shorter than the container timeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrackKind {
    Audio,
    Video,
}

impl std::fmt::Display for TrackKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Audio => "audio",
            Self::Video => "video",
        })
    }
}

/// Evidence of a track ending materially before its container timeline.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CoverageShortfall {
    pub track: TrackKind,
    pub actual_seconds: f64,
    pub expected_seconds: f64,
}

impl std::fmt::Display for CoverageShortfall {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "the {} track covers only {} of the {} timeline",
            self.track,
            format_ts(self.actual_seconds as f32),
            format_ts(self.expected_seconds as f32),
        )
    }
}

/// Post-download integrity: identify the track that covers materially less of
/// the timeline than the container claims, if any. `None` also when the
/// file cannot be probed — integrity has its own verdicts elsewhere.
pub fn coverage_shortfall(file: &Path) -> Option<CoverageShortfall> {
    let source = Source::Local(file.to_path_buf());
    let coverage = media::stream_coverage(&source, &media::Stop::new()).ok()?;
    let check = |track: TrackKind, seconds: Option<f64>| -> Option<CoverageShortfall> {
        let s = seconds?;
        materially_truncated(s, coverage.container_seconds).then_some(CoverageShortfall {
            track,
            actual_seconds: s,
            expected_seconds: coverage.container_seconds,
        })
    };
    check(TrackKind::Audio, coverage.audio_seconds)
        .or_else(|| check(TrackKind::Video, coverage.video_seconds))
}

/// Probe a direct media URL over http(s): the demuxer reads the container
/// header (moov etc.) without downloading the payload; `size_bytes` is
/// unknown (the caller may fill it via HEAD). Fails when the URL is not
/// direct media (HTML pages land here) or the probe exceeds
/// [`URL_PROBE_TIMEOUT`].
///
/// `relay` routes libav's connection through the policy relay via the
/// `http_proxy` demuxer option, so the demuxer's own hostname resolution
/// and redirect-following are re-screened per connection rather than
/// trusted from the up-front [`crate::netguard::vet`].
pub fn probe_media_url(url: &str, relay: &PolicyRelay) -> Result<ProbeSummary, Fault> {
    let source = Source::classify(url);
    // A deadline token arms libav's interrupt callback, turning a stalled
    // open/read into AVERROR_EXIT after the timeout. The context's own
    // clone keeps the deadline live for every later header read.
    let stop = media::Stop::after(URL_PROBE_TIMEOUT);
    let proxy = relay.url();
    let ictx = media::open_input(&source, &stop, &[("http_proxy", &proxy)])?;
    let summary = media::summarize_input(&ictx, &source)?;
    Ok(ProbeSummary {
        duration: round_centis(summary.duration),
        width: summary.width,
        height: summary.height,
        size_bytes: None,
        container: Some(container_label(&summary.container)),
        subtitles: describe_subtitles(&summary),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn labels_demuxer_alias_lists_with_one_name() {
        assert_eq!(container_label("mov,mp4,m4a,3gp,3g2,mj2"), "mp4");
        assert_eq!(container_label("matroska,webm"), "mkv");
        assert_eq!(container_label("asf"), "wmv");
        assert_eq!(container_label("avi"), "avi");
        assert_eq!(container_label("weird_thing"), "mp4");
        assert_eq!(container_label(""), "mp4");
    }

    #[test]
    fn container_label_is_idempotent() {
        // Cached probe docs may re-enter through this mapping; a label must
        // map to itself.
        for raw in ["mov,mp4,m4a,3gp,3g2,mj2", "matroska,webm", "mpegts"] {
            let once = container_label(raw);
            assert_eq!(container_label(&once), once, "{raw}");
        }
    }

    #[test]
    fn rounds_duration_to_centiseconds() {
        assert_eq!(round_centis(406.03735_f32), 406.04);
        assert_eq!(round_centis(90.0_f32), 90.0);
    }

    #[test]
    fn advertises_only_what_core_discovers() {
        let dir = crate::tempdir::TempDir::new("av-probe");
        let video = dir.join("talk.mp4");
        std::fs::write(&video, b"").expect("write");
        std::fs::write(dir.join("talk.srt"), b"").expect("write");
        std::fs::write(dir.join("talk.zh.srt"), b"").expect("write");
        // Formats core's auto-detection does not read must not be promised.
        std::fs::write(dir.join("talk.vtt"), b"").expect("write");
        std::fs::write(dir.join("other.srt"), b"").expect("write");

        let mut found = scan_sidecars(&video);
        found.sort();
        assert_eq!(found, ["talk.srt".to_string(), "talk.zh.srt".to_string()]);
    }

    #[test]
    fn flags_materially_truncated_tracks_only() {
        assert!(
            materially_truncated(50.0, 437.0),
            "50s of 437s is truncated"
        );
        assert!(
            !materially_truncated(430.0, 437.0),
            "a 7s shortfall is trailing-silence jitter"
        );
        assert!(
            !materially_truncated(55.0, 60.0),
            "short videos need the absolute bound"
        );
        assert!(
            !materially_truncated(500.0, 437.0),
            "longer than the container is not truncation"
        );
    }

    #[test]
    fn describes_embedded_tracks_only() {
        let summary = media::MediaSummary {
            duration: 10.0,
            width: 100,
            height: 100,
            container: "mp4".into(),
            subtitle_tracks: vec![
                media::SubtitleTrackSummary {
                    language: Some("eng".into()),
                    codec: "subrip".into(),
                },
                media::SubtitleTrackSummary {
                    language: None,
                    codec: "ass".into(),
                },
            ],
        };
        assert_eq!(
            describe_subtitles(&summary),
            ["embedded #0 (eng, subrip)", "embedded #1 (ass)"]
        );
    }
}
