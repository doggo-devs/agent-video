// SPDX-License-Identifier: Apache-2.0
//! Synthesized media fixtures: ffmpeg's lavfi generators, cached per spec in
//! the temp dir so one encode serves every test in the run.
//!
//! Fixtures are code, not vendored assets: every byte of test media is
//! reproducible from the recipe in this file. Each recipe is hashed (args +
//! duration) into a cache directory under [`std::env::temp_dir`], synthesized
//! into a scratch directory, and atomically renamed into place — concurrent
//! test processes never observe a half-written file, and repeated runs reuse
//! the encode.

use std::collections::hash_map::DefaultHasher;
use std::fs;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Mutex, OnceLock};

/// Bump to invalidate every cached fixture when a recipe's meaning changes
/// without its argument list changing (e.g. the byte-concat step).
const RECIPE_VERSION: u32 = 1;

/// A synthesized video on disk, plus the duration it was synthesized with.
pub struct Fixture {
    /// The media file (mp4 or mpeg-ts, per recipe).
    pub media: PathBuf,
    /// A generated SRT sidecar, present only for [`with_srt_sidecar`].
    pub sidecar: Option<PathBuf>,
    /// The duration the recipe requested, in seconds.
    pub duration: f32,
}

/// testsrc2 video + sine audio, mp4 (libx264 + aac), keyframe every second.
/// The pattern rotates continuously (`rotate=t`): plain testsrc2 is
/// perceptual-hash-static at 1 fps and the whole file would dedup into a
/// single frame plus a static stretch.
///
/// `None` = no ffmpeg CLI on PATH and the caller skips — unless
/// `AV_REQUIRE_MEDIA_TESTS` is set, where absence panics: CI must never
/// silently degrade to skipping the only real-decode coverage this
/// workspace has.
pub fn h264_talk(duration: f32) -> Option<Fixture> {
    require_ffmpeg()?;
    let args = talk_args(duration);
    let dir = cached("h264-talk", &args, &["talk.mp4"], |scratch| {
        run_ffmpeg(scratch, &args);
    });
    Some(Fixture {
        media: dir.join("talk.mp4"),
        sidecar: None,
        duration,
    })
}

/// `-g 9999 -keyint_min 9999 -sc_threshold 0`: one keyframe in the whole
/// file, so keyframe-driven strategies have nothing to work with.
pub fn keyframe_poor(duration: f32) -> Option<Fixture> {
    require_ffmpeg()?;
    let args: Vec<String> = [
        "-f",
        "lavfi",
        "-i",
        &format!("testsrc2=duration={duration}:size=640x360:rate=30,rotate=t"),
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-pix_fmt",
        "yuv420p",
        "-g",
        "9999",
        "-keyint_min",
        "9999",
        "-sc_threshold",
        "0",
        "poor.mp4",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    let dir = cached("keyframe-poor", &args, &["poor.mp4"], |scratch| {
        run_ffmpeg(scratch, &args);
    });
    Some(Fixture {
        media: dir.join("poor.mp4"),
        sidecar: None,
        duration,
    })
}

/// [`h264_talk`] plus a generated `.srt` sidecar with one cue per two
/// seconds. The sidecar's stem deliberately differs from the media's, so a
/// run with `SubtitleSource::Auto` on the shared media never discovers it.
pub fn with_srt_sidecar(duration: f32) -> Option<Fixture> {
    let base = h264_talk(duration)?;
    let srt = srt_text(duration);
    let dir = cached(
        "srt-sidecar",
        std::slice::from_ref(&srt),
        &["sidecar.srt"],
        |scratch| {
            fs::write(scratch.join("sidecar.srt"), &srt).expect("writing sidecar srt");
        },
    );
    Some(Fixture {
        media: base.media,
        sidecar: Some(dir.join("sidecar.srt")),
        duration,
    })
}

/// [`h264_talk`] re-cut with `-ss` before `-i` plus stream copy: the mp4
/// edit list makes presentation start at 0 (header `start_time = 0`) while
/// the copied packets from the preceding keyframe keep legitimate negative
/// pts (down to `cut`'s distance from that keyframe) — the legal shape the
/// origin rebase must never fire on: releases hold the majority, so the
/// declared anchor always wins the election.
pub fn edit_list_cut(duration: f32, cut: f32) -> Option<Fixture> {
    let base = h264_talk(duration)?;
    let args: Vec<String> = [
        "-ss",
        &format!("{cut}"),
        "-i",
        base.media.to_str().expect("fixture path is utf-8"),
        "-c",
        "copy",
        "cut.mp4",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    let dir = cached("edit-list-cut", &args, &["cut.mp4"], |scratch| {
        run_ffmpeg(scratch, &args);
    });
    Some(Fixture {
        media: dir.join("cut.mp4"),
        sidecar: None,
        duration: duration - cut,
    })
}

/// Regression shape for 7f7191f: two sine segments concat'd with different
/// sample rates — audio decode must follow the mid-stream change.
///
/// Two MPEG-TS segments (aac at 44100 Hz, then 22050 Hz; the second offset
/// to continue the first's timeline) byte-concatenated, so the decoder sees
/// the audio parameters change mid-stream while the container demuxes as
/// one continuous file.
pub fn audio_param_change(duration: f32) -> Option<Fixture> {
    require_ffmpeg()?;
    let half = duration / 2.0;
    // Each lavfi segment restarts its clock at 0, so with identical filters
    // the second segment would replay the first's rotations frame for frame
    // and dedup would discard it wholesale; a distinct look (negated, other
    // rotation phase and speed) keeps both segments' frames alive.
    let seg = |rate: u32, offset: f32, look: &str, out: &str| -> Vec<String> {
        [
            "-f",
            "lavfi",
            "-i",
            &format!("testsrc2=duration={half}:size=640x360:rate=30,{look}"),
            "-f",
            "lavfi",
            "-i",
            &format!("sine=frequency=440:sample_rate={rate}:duration={half}"),
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-ar",
            &rate.to_string(),
            "-muxdelay",
            "0",
            "-muxpreload",
            "0",
            "-output_ts_offset",
            &format!("{offset}"),
            "-f",
            "mpegts",
            out,
        ]
        .iter()
        .map(|s| s.to_string())
        .collect()
    };
    let first = seg(44_100, 0.0, "rotate=t", "seg1.ts");
    let second = seg(22_050, half, "negate,rotate=0.7+1.3*t", "seg2.ts");
    let key: Vec<String> = first.iter().chain(second.iter()).cloned().collect();
    let dir = cached("audio-param-change", &key, &["concat.ts"], |scratch| {
        run_ffmpeg(scratch, &first);
        run_ffmpeg(scratch, &second);
        let mut bytes = fs::read(scratch.join("seg1.ts")).expect("reading seg1.ts");
        bytes.extend(fs::read(scratch.join("seg2.ts")).expect("reading seg2.ts"));
        fs::write(scratch.join("concat.ts"), bytes).expect("writing concatenated ts");
    });
    Some(Fixture {
        media: dir.join("concat.ts"),
        sidecar: None,
        duration,
    })
}

/// A clean testsrc2 MPEG-TS remuxed with a `setts` bitstream filter
/// rewriting selected video packet timestamps: the raw material for the
/// decode pass's timeline-gate integration tests.
///
/// `expr` is an ffmpeg `setts` pts expression over `N` (packet index) and
/// `PTS` (90 kHz ticks); commas inside function calls must be escaped as
/// `\,` for the bsf option parser. B-frames are disabled so decode order
/// equals presentation order and packet index maps 1:1 onto the frames the
/// gate sees; MPEG-TS is used because its 33-bit pts field round-trips
/// far-future values that MP4's composition offsets cannot represent.
pub fn corrupt_pts(tag: &str, duration: f32, expr: &str) -> Option<Fixture> {
    require_ffmpeg()?;
    let clean: Vec<String> = [
        "-f",
        "lavfi",
        "-i",
        &format!("testsrc2=duration={duration}:size=320x180:rate=30,rotate=t"),
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-pix_fmt",
        "yuv420p",
        "-bf",
        "0",
        "-g",
        "30",
        "-muxdelay",
        "0",
        "-muxpreload",
        "0",
        "-f",
        "mpegts",
        "clean.ts",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    let remux: Vec<String> = [
        "-i",
        "clean.ts",
        "-c",
        "copy",
        "-bsf:v",
        &format!("setts=pts={expr}"),
        "-copyts",
        "-muxdelay",
        "0",
        "-muxpreload",
        "0",
        "-f",
        "mpegts",
        "corrupt.ts",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    let key: Vec<String> = clean.iter().chain(remux.iter()).cloned().collect();
    let dir = cached(tag, &key, &["corrupt.ts"], |scratch| {
        run_ffmpeg(scratch, &clean);
        run_ffmpeg(scratch, &remux);
    });
    Some(Fixture {
        media: dir.join("corrupt.ts"),
        sidecar: None,
        duration,
    })
}

/// `Some(())` when the ffmpeg CLI is on PATH (detected once per process).
/// Absent + `AV_REQUIRE_MEDIA_TESTS` set panics; absent otherwise returns
/// `None` so the caller can skip.
fn require_ffmpeg() -> Option<()> {
    static PRESENT: OnceLock<bool> = OnceLock::new();
    let present = *PRESENT.get_or_init(|| {
        Command::new("ffmpeg")
            .arg("-version")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    });
    if present {
        return Some(());
    }
    if std::env::var_os("AV_REQUIRE_MEDIA_TESTS").is_some() {
        panic!(
            "AV_REQUIRE_MEDIA_TESTS is set but no ffmpeg CLI is on PATH; \
             install ffmpeg — CI must not silently skip the real-decode tests"
        );
    }
    None
}

/// The [`h264_talk`] encode recipe, shared with [`with_srt_sidecar`] so both
/// resolve to the same cached media.
fn talk_args(duration: f32) -> Vec<String> {
    [
        "-f",
        "lavfi",
        "-i",
        &format!("testsrc2=duration={duration}:size=640x360:rate=30,rotate=t"),
        "-f",
        "lavfi",
        "-i",
        &format!("sine=frequency=440:sample_rate=44100:duration={duration}"),
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-pix_fmt",
        "yuv420p",
        "-g",
        "30",
        "-c:a",
        "aac",
        "-shortest",
        "talk.mp4",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect()
}

/// One SRT cue per two seconds, covering `[0, duration)`.
fn srt_text(duration: f32) -> String {
    let cues = ((duration / 2.0).ceil() as u32).max(1);
    let mut out = String::new();
    for i in 0..cues {
        let start = i * 2;
        out.push_str(&format!(
            "{}\n00:00:{:02},000 --> 00:00:{:02},500\nSynthesized cue number {}\n\n",
            i + 1,
            start,
            start + 1,
            i + 1
        ));
    }
    out
}

/// Resolve the recipe's cache directory, synthesizing it on a miss.
///
/// `build` runs in a scratch directory next to the final one (same
/// filesystem); afterwards each named artifact is renamed into place, so a
/// concurrent test process either sees the complete artifact or a cache
/// miss — never a half-written file. A process-local mutex additionally
/// keeps concurrent tests in this process from encoding the same recipe
/// twice.
fn cached(tag: &str, key: &[String], artifacts: &[&str], build: impl FnOnce(&Path)) -> PathBuf {
    let dir = fixture_dir(tag, key);
    let ready = |d: &Path| artifacts.iter().all(|a| d.join(a).exists());
    if ready(&dir) {
        return dir;
    }

    static SYNTH: Mutex<()> = Mutex::new(());
    let _guard = SYNTH.lock().expect("fixture synthesis lock");
    if ready(&dir) {
        return dir;
    }

    let scratch = dir.with_extension(format!("build-{}", std::process::id()));
    let _ = fs::remove_dir_all(&scratch);
    fs::create_dir_all(&scratch).expect("creating fixture scratch dir");
    build(&scratch);
    fs::create_dir_all(&dir).expect("creating fixture cache dir");
    for artifact in artifacts {
        // Same-filesystem rename: atomic promotion into the cache dir.
        fs::rename(scratch.join(artifact), dir.join(artifact))
            .unwrap_or_else(|e| panic!("promoting fixture artifact {artifact}: {e}"));
    }
    let _ = fs::remove_dir_all(&scratch);
    dir
}

/// `temp_dir()/agent-video-fixtures/<tag>-<hash of recipe version + key>`.
fn fixture_dir(tag: &str, key: &[String]) -> PathBuf {
    let mut hasher = DefaultHasher::new();
    RECIPE_VERSION.hash(&mut hasher);
    tag.hash(&mut hasher);
    key.hash(&mut hasher);
    std::env::temp_dir()
        .join("agent-video-fixtures")
        .join(format!("{tag}-{:016x}", hasher.finish()))
}

/// Run one ffmpeg invocation in `dir`, panicking with its stderr on failure.
fn run_ffmpeg(dir: &Path, args: &[String]) {
    let output = Command::new("ffmpeg")
        .current_dir(dir)
        .args(["-hide_banner", "-nostdin", "-y"])
        .args(args)
        .output()
        .expect("spawning ffmpeg (presence was checked at fixture request)");
    if !output.status.success() {
        panic!(
            "ffmpeg {args:?} failed ({}):\n{}",
            output.status,
            String::from_utf8_lossy(&output.stderr)
        );
    }
}
