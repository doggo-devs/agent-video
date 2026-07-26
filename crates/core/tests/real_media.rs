// SPDX-License-Identifier: Apache-2.0
//! Real-decode integration tests over synthesized media (see
//! [`support::fixtures`]). Every assertion is invariant-level — durations,
//! coverage, ordering, container magic — never pixel content, so the suite
//! survives encoder and ffmpeg-version drift.

mod support;

use agent_video_core::config::Overrides;
use agent_video_core::event::Discard;
use agent_video_core::media::Stop;
use agent_video_core::media::audio::{
    AudioCeiling, AudioSamples, WHISPER_SAMPLE_RATE, decode_audio_16k_mono,
};
use agent_video_core::pipeline::{self, OutputSink, SubtitleSource, Window};
use agent_video_core::probe;
use agent_video_core::source::Source;
use agent_video_core::types::{GridImage, TimeRange};
use agent_video_core::wire;

use std::path::Path;

/// Wire timestamps round to the nearest second, so every parsed value
/// carries up to ±0.5s of quantization.
const ROUNDING: f32 = 0.5;

/// Fixed-fps sampling at 1 fps leaves up to 1s between a frame and the
/// content it covers; combined with rounding on both endpoints, two grids
/// (or a grid and the file edge) may legitimately sit this far apart.
const COVERAGE_SLACK: f32 = 2.0;

/// A ceiling far above any fixture's duration: overflow tests pick their
/// own tight cap; everything else must never trip this one.
fn generous_cap() -> AudioCeiling {
    AudioCeiling::from_seconds(60).expect("60s is a valid ceiling")
}

macro_rules! require_fixture {
    ($e:expr) => {
        match $e {
            Some(fx) => fx,
            None => {
                eprintln!("skipping: no ffmpeg CLI on PATH");
                return;
            }
        }
    };
}

fn source_of(media: &Path) -> Source {
    Source::classify(media.to_str().expect("fixture path is utf-8"))
}

/// Run `process` into a fresh file-mode output dir and parse its metadata.
fn run_process(
    media: &Path,
    tag: &str,
    overrides: Overrides,
    subs: SubtitleSource<'_>,
    window: Option<Window>,
) -> wire::Metadata {
    let dir = support::fresh_output_dir(tag);
    let cfg = support::config(2048, overrides);
    pipeline::process(
        &source_of(media),
        OutputSink::Files(&dir),
        &cfg,
        pipeline::ProcessOpts {
            threads: 0,
            max_in_flight: pipeline::DEFAULT_IN_FLIGHT,
            subs,
            window,
            stop: Stop::new(),
            events: &Discard,
        },
    )
    .expect("process succeeds on the synthesized fixture");
    support::read_metadata(&dir)
}

#[test]
fn covers_full_duration_without_overlap() {
    let fx = require_fixture!(support::fixtures::h264_talk(6.0));
    // A 512px canvas holds ~2 cells at the quality floor, and 2s chunks at
    // 1 fps feed it 2 frames each: several grids, so ordering and coverage
    // are asserted across real boundaries instead of a single trivial range.
    let dir = support::fresh_output_dir("covers-full");
    let cfg = support::config(
        512,
        Overrides {
            base_fps: Some(1.0),
            chunk_seconds: Some(2.0),
            ..Default::default()
        },
    );
    pipeline::process(
        &source_of(&fx.media),
        OutputSink::Files(&dir),
        &cfg,
        pipeline::ProcessOpts {
            threads: 0,
            max_in_flight: pipeline::DEFAULT_IN_FLIGHT,
            subs: SubtitleSource::Auto,
            window: None,
            stop: Stop::new(),
            events: &Discard,
        },
    )
    .expect("process succeeds on the synthesized fixture");
    let meta = support::read_metadata(&dir);

    assert!(!meta.grids.is_empty(), "at least one grid");
    let total = meta.total_duration().expect("process document");
    assert!(
        (total - fx.duration).abs() <= ROUNDING,
        "reported duration {} tracks the synthesized {}",
        total,
        fx.duration
    );

    let ranges: Vec<(f32, f32)> = meta
        .grids
        .iter()
        .map(|g| support::parse_wire_range(&g.time_range))
        .collect();
    for (start, end) in &ranges {
        assert!(start <= end, "range {start}-{end} is ordered");
    }
    for pair in ranges.windows(2) {
        assert!(
            pair[1].0 >= pair[0].1,
            "grids are monotonic and non-overlapping: {:?} then {:?}",
            pair[0],
            pair[1]
        );
        assert!(
            pair[1].0 - pair[0].1 <= COVERAGE_SLACK,
            "no coverage hole between {:?} and {:?}",
            pair[0],
            pair[1]
        );
    }
    let first = ranges.first().expect("non-empty");
    let last = ranges.last().expect("non-empty");
    assert!(
        first.0 <= COVERAGE_SLACK,
        "coverage starts at the file head"
    );
    assert!(
        last.1 >= total - COVERAGE_SLACK,
        "coverage reaches the file tail: last grid ends at {} of {}",
        last.1,
        total
    );
}

#[test]
fn bounds_every_key_frame_timestamp_by_the_duration() {
    let fx = require_fixture!(support::fixtures::h264_talk(6.0));
    let meta = run_process(
        &fx.media,
        "keyframe-bounds",
        Overrides::default(),
        SubtitleSource::Auto,
        None,
    );
    for grid in &meta.grids {
        for kf in &grid.key_frames {
            let ts = support::parse_wire_ts(&kf.timestamp);
            assert!(
                (0.0..=fx.duration + ROUNDING).contains(&ts),
                "key frame at {ts}s lies within [0, {}]",
                fx.duration
            );
        }
    }
}

#[test]
fn window_bounds_every_timestamp() {
    let fx = require_fixture!(support::fixtures::h264_talk(6.0));
    let window = Window::from_args(Some(1.0), Some(4.0))
        .expect("valid window")
        .expect("both bounds given");
    let meta = run_process(
        &fx.media,
        "window-bounds",
        Overrides::default(),
        SubtitleSource::Auto,
        Some(window),
    );
    assert!(!meta.grids.is_empty(), "windowed run still produces grids");
    let inside = |ts: f32| (1.0 - ROUNDING..=4.0 + ROUNDING).contains(&ts);
    for grid in &meta.grids {
        let (start, end) = support::parse_wire_range(&grid.time_range);
        assert!(
            inside(start) && inside(end),
            "grid range {}s-{}s falls inside the 1s-4s window",
            start,
            end
        );
        for kf in &grid.key_frames {
            let ts = support::parse_wire_ts(&kf.timestamp);
            assert!(inside(ts), "key frame at {ts}s falls inside the window");
        }
    }
}

#[test]
fn fills_grid_from_keyframe_poor_source() {
    let fx = require_fixture!(support::fixtures::keyframe_poor(4.0));
    // Default (adaptive) strategy on a one-keyframe file: the sparse-
    // keyframe path must degrade to fixed-fps and still produce output.
    let meta = run_process(
        &fx.media,
        "keyframe-poor",
        Overrides::default(),
        SubtitleSource::Auto,
        None,
    );
    assert!(
        !meta.grids.is_empty(),
        "a single-keyframe source still yields at least one grid"
    );
}

#[test]
fn carries_sidecar_subtitles_into_grids() {
    let fx = require_fixture!(support::fixtures::with_srt_sidecar(4.0));
    let sidecar = fx.sidecar.as_deref().expect("fixture carries a sidecar");
    let meta = run_process(
        &fx.media,
        "srt-sidecar",
        Overrides::default(),
        SubtitleSource::File(sidecar),
        None,
    );
    assert!(
        meta.grids.iter().any(|g| g
            .subtitles
            .as_deref()
            .is_some_and(|s| s.contains("Synthesized cue"))),
        "at least one grid carries the sidecar's cue text"
    );
}

#[test]
fn extracts_single_valid_jpeg_grid() {
    let fx = require_fixture!(support::fixtures::h264_talk(6.0));
    let dir = support::fresh_output_dir("extract-jpeg");
    let cfg = support::config(2048, Overrides::default());
    let range = TimeRange::new(1.0, 3.0).expect("valid range");
    pipeline::extract_range(
        &source_of(&fx.media),
        OutputSink::Files(&dir),
        &cfg,
        range,
        SubtitleSource::Auto,
        &Stop::new(),
        &Discard,
    )
    .expect("extract succeeds on the synthesized fixture");

    let meta = support::read_metadata(&dir);
    assert_eq!(meta.grids.len(), 1, "extract emits exactly one grid");
    assert_eq!(
        meta.segment().map(|s| (s.start(), s.end())),
        Some((1.0, 3.0)),
        "the document records the harvested range"
    );
    let GridImage::File(name) = &meta.grids[0].image else {
        panic!("file-mode extract references a written file");
    };
    assert_eq!(name, "grid_extract.jpg");
    let bytes = std::fs::read(dir.join(name)).expect("grid image exists on disk");
    assert!(
        bytes.starts_with(&[0xff, 0xd8]),
        "grid image starts with the JPEG SOI marker"
    );
    assert!(
        bytes.ends_with(&[0xff, 0xd9]),
        "grid image ends with the JPEG EOI marker"
    );
}

#[test]
fn probes_duration_within_half_second() {
    let fx = require_fixture!(support::fixtures::h264_talk(6.0));
    let duration = probe::probe_duration(&source_of(&fx.media), &Stop::new())
        .expect("probe succeeds on the fixture");
    assert!(
        (duration - fx.duration).abs() <= 0.5,
        "probed {duration}s is within 0.5s of the synthesized {}s",
        fx.duration
    );
}

#[test]
fn survives_mid_stream_audio_param_change_in_process() {
    let fx = require_fixture!(support::fixtures::audio_param_change(6.0));
    let meta = run_process(
        &fx.media,
        "audio-param-change",
        Overrides::default(),
        SubtitleSource::Auto,
        None,
    );
    let total = meta.total_duration().expect("process document");
    assert!(
        total >= fx.duration - ROUNDING,
        "the concatenated stream demuxes to its full {}s, got {}",
        fx.duration,
        total
    );
    let last_end = meta
        .grids
        .iter()
        .map(|g| support::parse_wire_range(&g.time_range).1)
        .fold(0.0_f32, f32::max);
    assert!(
        last_end >= fx.duration - COVERAGE_SLACK,
        "grids cover the splice and reach the tail: last end {last_end}s of {}s",
        fx.duration
    );
}

#[test]
fn reports_a_missing_audio_track_as_absence_not_failure() {
    // keyframe_poor synthesizes video only — no audio input at all. A
    // legitimate silent video (surveillance, screen capture) must decode
    // to "no audio here", never to a corrupt-input error.
    let fx = require_fixture!(support::fixtures::keyframe_poor(2.0));
    let decoded = decode_audio_16k_mono(&source_of(&fx.media), &Stop::new(), generous_cap())
        .expect("a video without an audio track is not a decode failure");
    assert!(decoded.is_none(), "absence is Ok(None), not an error");
}

#[test]
fn rejects_audio_decoding_past_the_ceiling() {
    // The finding's shape: a track longer than the caller's ceiling must
    // fail with a typed error before materializing, never decode whole and
    // never truncate silently. 6s of audio against a 1s cap pins the
    // mechanism without synthesizing a hostile multi-hour file.
    let fx = require_fixture!(support::fixtures::audio_param_change(6.0));
    let cap = AudioCeiling::from_seconds(1).expect("1s is a valid ceiling");
    let err = decode_audio_16k_mono(&source_of(&fx.media), &Stop::new(), cap)
        .expect_err("6s of audio must overflow a 1s ceiling");
    assert_eq!(err.code, agent_video_core::error::ErrorCode::InputTooLong);
}

#[test]
fn follows_mid_stream_sample_rate_change_in_audio_decode() {
    let fx = require_fixture!(support::fixtures::audio_param_change(6.0));
    // The 7f7191f regression path: a fixed swresample context answered the
    // 44100 -> 22050 Hz mid-stream change with AVERROR_INPUT_CHANGED and
    // discarded everything already decoded. Following the change means the
    // decode succeeds AND its output spans both segments, not just the first.
    let samples = decode_audio_16k_mono(&source_of(&fx.media), &Stop::new(), generous_cap())
        .expect("audio decode follows the mid-stream sample-rate change")
        .expect("the fixture has an audio track");
    let expected = fx.duration * WHISPER_SAMPLE_RATE as f32;
    assert!(
        samples.len() as f32 >= 0.9 * expected,
        "decoded {} samples, expected ~{expected} — decode stopped early?",
        samples.len()
    );
    assert!(
        samples.len() as f32 <= 1.15 * expected,
        "decoded {} samples, expected ~{expected} — timeline duplicated?",
        samples.len()
    );
}

#[test]
fn collects_whole_track_under_a_generous_ceiling() {
    // The streaming contract: pulling the track through small bounded
    // fills must yield bit-identical samples to the capped whole-track
    // decode — same pipeline, same carry-over handling, chunk seams
    // invisible, and a generous ceiling changes nothing on legit input.
    let fx = require_fixture!(support::fixtures::audio_param_change(6.0));
    let whole = decode_audio_16k_mono(&source_of(&fx.media), &Stop::new(), generous_cap())
        .expect("whole decode succeeds under a generous ceiling")
        .expect("the fixture has an audio track");

    let mut stream = AudioSamples::open(&source_of(&fx.media), &Stop::new())
        .expect("stream opens")
        .expect("the fixture has an audio track");
    let mut streamed = Vec::new();
    let mut buf = Vec::new();
    loop {
        let more = stream.fill(&mut buf, 10_000).expect("fill succeeds");
        streamed.append(&mut buf);
        if !more {
            break;
        }
    }
    assert!(
        !stream
            .fill(&mut buf, 1)
            .expect("post-exhaustion fill is fine"),
        "exhaustion is final: later fills keep answering false"
    );
    assert!(buf.is_empty(), "a drained stream appends nothing");
    assert_eq!(
        streamed, whole,
        "streamed fills and the whole decode are the same samples"
    );
}
