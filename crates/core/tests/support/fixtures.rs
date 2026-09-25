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

/// A legal long-GOP recording with a sparse keyframe tail: keyframes
/// forced at 0,2,..,10 then 25 and `duration - 0.1`, everything else
/// suppressed (`-g 9999 -sc_threshold 0`) — the screen-capture/`-g 600`
/// cadence where minute-scale keyframe gaps are the file's legal pacing.
/// A keyframe-only decode must surface the tail keyframes: dropping them
/// loses the last `duration - 10` seconds of a healthy file.
pub fn sparse_tail_keyframes(duration: f32) -> Option<Fixture> {
    require_ffmpeg()?;
    let last = duration - 0.1;
    let args: Vec<String> = [
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
        "-force_key_frames",
        &format!("0,2,4,6,8,10,25,{last}"),
        "-g",
        "9999",
        "-keyint_min",
        "9999",
        "-sc_threshold",
        "0",
        "sparse.mp4",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    let dir = cached("sparse-tail-keyframes", &args, &["sparse.mp4"], |scratch| {
        run_ffmpeg(scratch, &args);
    });
    Some(Fixture {
        media: dir.join("sparse.mp4"),
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
/// segmenter must treat as one identity segment: pre-roll stays within the
/// allowance, so no seam, no drop, no re-anchor.
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
/// decode pass's timeline-segmenter integration tests.
///
/// `expr` is an ffmpeg `setts` pts expression over `N` (packet index) and
/// `PTS` (90 kHz ticks); commas inside function calls must be escaped as
/// `\,` for the bsf option parser. B-frames are disabled so decode order
/// equals presentation order and packet index maps 1:1 onto the frames the
/// segmenter sees; MPEG-TS is used because its 33-bit pts field round-trips
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

/// Red/green/blue/white quadrants, with counterclockwise display rotation
/// followed by optional flips. Identity has no display matrix.
pub fn display_oriented(rotation: i32, hflip: bool, vflip: bool) -> Option<Fixture> {
    require_ffmpeg()?;
    let encode = strings(&[
        "-f",
        "lavfi",
        "-i",
        "color=white:size=160x96:rate=1:duration=1,\
         drawbox=x=0:y=0:w=80:h=48:color=red:t=fill,\
         drawbox=x=80:y=0:w=80:h=48:color=lime:t=fill,\
         drawbox=x=0:y=48:w=80:h=48:color=blue:t=fill",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-crf",
        "0",
        "corners.mp4",
    ]);
    let base = cached(
        "orientation-corners",
        &encode,
        &["corners.mp4"],
        |scratch| {
            run_ffmpeg(scratch, &encode);
        },
    );
    let mut fixture = Fixture {
        media: base.join("corners.mp4"),
        sidecar: None,
        duration: 1.0,
    };
    if rotation == 0 && !hflip && !vflip {
        return Some(fixture);
    }
    let mut args = strings(&["-display_rotation", &rotation.to_string()]);
    if hflip {
        args.push("-display_hflip".into());
    }
    if vflip {
        args.push("-display_vflip".into());
    }
    args.extend(strings(&[
        "-i",
        fixture.media.to_str().expect("fixture path is utf-8"),
        "-c",
        "copy",
        "oriented.mp4",
    ]));
    let dir = cached("display-oriented", &args, &["oriented.mp4"], |scratch| {
        run_ffmpeg(scratch, &args);
    });
    fixture.media = dir.join("oriented.mp4");
    Some(fixture)
}

/// [`h264_talk`] with an embedded SubRip track of `cues` one-kilobyte
/// cues, one every 10ms: enough decoded subtitle data per pass that a
/// per-cue leak shows in process memory.
pub fn embedded_subtitle_track(duration: f32, cues: u32) -> Option<Fixture> {
    let base = h264_talk(duration)?;
    let text = "x".repeat(1000);
    let mut srt = String::new();
    for i in 0..cues {
        let (start, end) = (i * 10, i * 10 + 5);
        srt.push_str(&format!(
            "{}\n00:00:{:02},{:03} --> 00:00:{:02},{:03}\n{text}\n\n",
            i + 1,
            start / 1000,
            start % 1000,
            end / 1000,
            end % 1000,
        ));
    }
    let args = strings(&[
        "-i",
        base.media.to_str().expect("fixture path is utf-8"),
        "-i",
        "track.srt",
        "-map",
        "0",
        "-map",
        "1",
        "-c",
        "copy",
        "-c:s",
        "srt",
        "subs.mkv",
    ]);
    let key: Vec<String> = args.iter().cloned().chain([srt.clone()]).collect();
    let dir = cached("embedded-subtitle", &key, &["subs.mkv"], |scratch| {
        fs::write(scratch.join("track.srt"), &srt).expect("writing subtitle track");
        run_ffmpeg(scratch, &args);
    });
    Some(Fixture {
        media: dir.join("subs.mkv"),
        sidecar: None,
        duration,
    })
}

fn strings(args: &[&str]) -> Vec<String> {
    args.iter().map(|s| s.to_string()).collect()
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
