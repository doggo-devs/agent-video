// SPDX-License-Identifier: Apache-2.0
//! Probe adapter: local files and direct media URLs into the model-facing
//! [`ProbeSummary`].
//!
//! Probing runs in-process over the linked FFmpeg (`media::summarize_input`)
//! — no `ffprobe` on PATH. URL probing reads the container header without
//! downloading the payload; a deadline thread arms libav's interrupt
//! callback so a hung server cannot stall the call forever.

use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, mpsc};
use std::time::Duration;

use super::schema::ProbeSummary;
use crate::fault::Fault;
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

/// Arms `stop` after `timeout` unless dropped first. Dropping the guard
/// disconnects the channel, waking the deadline thread immediately — so a
/// finished probe never leaks a sleeping thread past a scheduler tick, and
/// disarming cannot be forgotten on any return path.
struct Deadline {
    _disarm: mpsc::Sender<()>,
}

impl Deadline {
    fn arm(stop: Arc<AtomicBool>, timeout: Duration) -> Self {
        let (tx, rx) = mpsc::channel::<()>();
        std::thread::spawn(move || {
            if rx.recv_timeout(timeout) == Err(mpsc::RecvTimeoutError::Timeout) {
                stop.store(true, Ordering::Relaxed);
            }
        });
        Deadline { _disarm: tx }
    }
}

/// Human-readable subtitle descriptions: embedded tracks then sidecars,
/// mirroring the TS wire text (`embedded #0 (eng, subrip)`, `sidecar x.srt`).
fn describe_subtitles(summary: &media::MediaSummary, sidecars: &[String]) -> Vec<String> {
    let mut out: Vec<String> = summary
        .subtitle_tracks
        .iter()
        .enumerate()
        .map(|(i, t)| match &t.language {
            Some(lang) => format!("embedded #{i} ({lang}, {})", t.codec),
            None => format!("embedded #{i} ({})", t.codec),
        })
        .collect();
    out.extend(sidecars.iter().map(|name| format!("sidecar {name}")));
    out
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

/// Probe a local video file; sidecar subtitles are discovered exactly as a
/// core run would discover them.
pub fn probe_local_file(file: &Path) -> Result<ProbeSummary, Fault> {
    let source = Source::classify(&file.to_string_lossy());
    // A failed stat is an unknown size, not a zero-byte claim.
    let size = std::fs::metadata(file).ok().map(|m| m.len());
    let ictx = media::open_input(&source)?;
    let summary = media::summarize_input(&ictx, &source)?;
    let sidecars = scan_sidecars(file);
    Ok(ProbeSummary {
        duration: round_centis(summary.duration),
        width: summary.width,
        height: summary.height,
        size_bytes: size,
        container: Some(container_label(&summary.container)),
        subtitles: describe_subtitles(&summary, &sidecars),
    })
}

/// Whether a track covering `track_s` of a `container_s`-long timeline is
/// materially truncated: more than 10 s missing *and* under 90% coverage.
/// The absolute bound ignores trailing-silence jitter on long videos, the
/// relative one keeps short videos from tripping on it.
pub(crate) fn materially_truncated(track_s: f64, container_s: f64) -> bool {
    container_s - track_s > 10.0 && track_s < container_s * 0.9
}

/// Post-download integrity: name the track that covers materially less of
/// the timeline than the container claims, if any. `None` also when the
/// file cannot be probed — integrity has its own verdicts elsewhere.
pub fn coverage_shortfall(file: &Path) -> Option<String> {
    let source = Source::classify(&file.to_string_lossy());
    let coverage = media::stream_coverage(&source).ok()?;
    let check = |name: &str, seconds: Option<f64>| -> Option<String> {
        let s = seconds?;
        materially_truncated(s, coverage.container_seconds).then(|| {
            format!(
                "the {name} track covers only {} of the {} timeline",
                format_ts(s as f32),
                format_ts(coverage.container_seconds as f32)
            )
        })
    };
    check("audio", coverage.audio_seconds).or_else(|| check("video", coverage.video_seconds))
}

/// Probe a direct media URL over http(s): the demuxer reads the container
/// header (moov etc.) without downloading the payload; `size_bytes` is
/// unknown (the caller may fill it via HEAD). Fails when the URL is not
/// direct media (HTML pages land here) or the probe exceeds
/// [`URL_PROBE_TIMEOUT`].
pub fn probe_media_url(url: &str) -> Result<ProbeSummary, Fault> {
    let source = Source::classify(url);
    // Deadline: arms libav's interrupt callback, turning a stalled
    // open/read into AVERROR_EXIT. The atomic is the FFI boundary — the C
    // callback polls it.
    let stop = Arc::new(AtomicBool::new(false));
    let _deadline = Deadline::arm(Arc::clone(&stop), URL_PROBE_TIMEOUT);
    let ictx = media::open_input_interruptible(&source, stop)?;
    let summary = media::summarize_input(&ictx, &source)?;
    Ok(ProbeSummary {
        duration: round_centis(summary.duration),
        width: summary.width,
        height: summary.height,
        size_bytes: None,
        container: Some(container_label(&summary.container)),
        subtitles: describe_subtitles(&summary, &[]),
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
        let dir = std::env::temp_dir().join(format!(
            "av-probe-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("clock")
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).expect("mkdir");
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
    fn describes_embedded_then_sidecar() {
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
            describe_subtitles(&summary, &["talk.srt".to_string()]),
            [
                "embedded #0 (eng, subrip)",
                "embedded #1 (ass)",
                "sidecar talk.srt"
            ]
        );
    }
}
