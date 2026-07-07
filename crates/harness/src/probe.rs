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

/// How long an over-http probe may hang before we call it a non-file.
pub const URL_PROBE_TIMEOUT: Duration = Duration::from_secs(20);

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
        duration: f64::from(summary.duration),
        width: summary.width,
        height: summary.height,
        size_bytes: size,
        container: Some(summary.container.clone()),
        subtitles: describe_subtitles(&summary, &sidecars),
    })
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
        duration: f64::from(summary.duration),
        width: summary.width,
        height: summary.height,
        size_bytes: None,
        container: Some(summary.container.clone()),
        subtitles: describe_subtitles(&summary, &[]),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

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
