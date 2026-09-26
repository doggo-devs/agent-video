// SPDX-License-Identifier: Apache-2.0
//! Real-decode integration tests over synthesized media (see
//! [`support::fixtures`]). Assertions cover timing, coverage, and decoded
//! geometry; color checks allow codec rounding instead of requiring exact pixels.

// The support module is shared with timeline_segments.rs; each test
// binary uses its own subset, so unused items are expected here.
#[allow(dead_code)]
mod support;

use agent_video_core::config::Overrides;
use agent_video_core::event::Discard;
use agent_video_core::media::Stop;
use agent_video_core::media::audio::{AudioChunk, AudioSamples, WHISPER_SAMPLE_RATE};
use agent_video_core::media::decode::{FrameSource, SampleSpec};
use agent_video_core::pipeline::{self, OutputSink, SubtitleSource, Window};
use agent_video_core::probe;
use agent_video_core::source::Source;
use agent_video_core::types::{GridImage, TimeRange};
use agent_video_core::wire;
use std::num::NonZeroUsize;

use std::path::Path;

/// Wire timestamps round to the nearest second, so every parsed value
/// carries up to ±0.5s of quantization.
const ROUNDING: f32 = 0.5;

/// Fixed-fps sampling at 1 fps leaves up to 1s between a frame and the
/// content it covers; combined with rounding on both endpoints, two grids
/// (or a grid and the file edge) may legitimately sit this far apart.
const COVERAGE_SLACK: f32 = 2.0;

fn audio_blocks(path: &Path, bound: usize) -> Vec<AudioChunk> {
    let mut stream = AudioSamples::open(&source_of(path), &Stop::new())
        .expect("audio opens")
        .expect("audio track");
    let mut blocks = Vec::new();
    while let Some(chunk) = stream
        .next_chunk(NonZeroUsize::new(bound).expect("positive bound"))
        .expect("audio decodes")
    {
        assert!(chunk.samples.len() <= bound);
        blocks.push(chunk);
    }
    assert!(
        stream
            .next_chunk(NonZeroUsize::new(bound).expect("positive bound"))
            .expect("exhaustion is final")
            .is_none()
    );
    blocks
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
        .map(|g| (g.time_range.start(), g.time_range.end()))
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
        let (start, end) = (grid.time_range.start(), grid.time_range.end());
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
        .map(|g| g.time_range.end())
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
    let decoded = AudioSamples::open(&source_of(&fx.media), &Stop::new())
        .expect("a video without an audio track is not a decode failure");
    assert!(decoded.is_none(), "absence is Ok(None), not an error");
}

#[test]
fn follows_mid_stream_sample_rate_change_in_audio_decode() {
    let fx = require_fixture!(support::fixtures::audio_param_change(6.0));
    // The 7f7191f regression path: a fixed swresample context answered the
    // 44100 -> 22050 Hz mid-stream change with AVERROR_INPUT_CHANGED and
    // discarded everything already decoded. Following the change means the
    // decode succeeds AND its output spans both segments, not just the first.
    let blocks = audio_blocks(&fx.media, 10_000);
    let samples: Vec<_> = blocks.iter().flat_map(|b| b.samples.iter()).collect();
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
fn bounds_audio_blocks_without_changing_samples_or_timing() {
    let fx = require_fixture!(support::fixtures::audio_param_change(6.0));
    let large = audio_blocks(&fx.media, 1_000_000);
    let small = audio_blocks(&fx.media, 511);
    let flatten = |blocks: Vec<AudioChunk>| -> Vec<_> {
        blocks
            .into_iter()
            .flat_map(|b| {
                b.samples
                    .into_iter()
                    .enumerate()
                    .map(move |(i, sample)| (b.start.advance(i), sample))
            })
            .collect()
    };
    assert_eq!(flatten(large), flatten(small));
}

#[test]
fn preserves_delayed_audio_and_gaps_on_the_video_clock() {
    for rate in [16_000, 44_100, 48_000] {
        let fx = require_fixture!(support::fixtures::sparse_audio(rate));
        let blocks = audio_blocks(&fx.media, 100_000);
        assert_eq!(
            blocks.len(),
            2,
            "only the actual audio runs are materialized"
        );
        assert_eq!(
            (blocks[0].start.seconds(), blocks[0].end().seconds()),
            (1.0, 2.0)
        );
        assert_eq!(
            (blocks[1].start.seconds(), blocks[1].end().seconds()),
            (3.0, 5.0)
        );
        assert!(blocks[0].samples.iter().all(|s| (*s - 0.25).abs() < 0.001));
        assert!(blocks[1].samples.iter().all(|s| (*s - 0.75).abs() < 0.001));
    }
}

#[test]
fn keeps_audio_and_video_on_continuous_clocks_across_a_ts_reset() {
    let fx = require_fixture!(support::fixtures::reset_av_clock(6.0));
    let blocks = audio_blocks(&fx.media, 100_000);
    let samples: usize = blocks.iter().map(|b| b.samples.len()).sum();
    assert!(
        (5.9..6.2).contains(&(samples as f64 / 16_000.0)),
        "all speech survives: {samples}"
    );
    let audio_end = blocks.last().expect("audio").end().seconds();
    let mut video = FrameSource::open(
        &source_of(&fx.media),
        SampleSpec::Fps {
            fps: 120.0,
            start: 0.0,
            end: None,
        },
        48,
        None,
        &Discard,
    )
    .expect("video");
    let mut video_end = 0.0;
    while let Some(frame) = video.next_frame().expect("video decodes") {
        video_end = f64::from(frame.ts.get()) + 1.0 / 30.0;
    }
    assert!(
        (audio_end - video_end).abs() < 0.08,
        "audio {audio_end}, video {video_end}"
    );
}

#[test]
fn confines_a_future_audio_timestamp_to_its_corrupt_frame() {
    for packet in [0, 1, 2, 30, 165] {
        let fx = require_fixture!(support::fixtures::corrupt_av_pts(
            "audio-clock-blip",
            4.0,
            "PTS",
            &format!("if(eq(N\\,{packet})\\,PTS+7776000000\\,PTS)")
        ));
        let blocks = audio_blocks(&fx.media, 100_000);
        let samples: usize = blocks.iter().map(|b| b.samples.len()).sum();
        assert!(
            samples > 3 * 16_000,
            "the legal audio tail survives: {samples}"
        );
        let end = blocks.last().expect("audio").end().seconds();
        assert!(
            (3.9..4.1).contains(&end),
            "the corrupt timestamp never poisons the end: {end}"
        );
    }
}

#[test]
fn recovers_the_video_origin_without_losing_delayed_audio() {
    for poisoned in [1, 2, 65] {
        let fx = require_fixture!(support::fixtures::corrupt_av_pts(
            "audio-poisoned-video-head",
            4.0,
            &format!("if(lt(N\\,{poisoned})\\,PTS+7776000000\\,PTS)"),
            "PTS"
        ));
        let blocks = audio_blocks(&fx.media, 100_000);
        let start = blocks
            .first()
            .expect("audio survives the poisoned origin")
            .start
            .seconds();
        let end = blocks.last().expect("audio").end().seconds();
        // Short poisoned heads are dropped; long ones occupy their original duration.
        assert!(
            (0.40..0.50).contains(&start),
            "the half-second delay survives: {start}"
        );
        assert!(
            (3.9..4.1).contains(&end),
            "the whole audio track survives: {end}"
        );
    }
}

#[test]
fn preserves_aac_and_opus_samples_and_marker_positions() {
    for codec in ["aac", "libopus"] {
        let fx = require_fixture!(support::fixtures::primed_audio(codec));
        let blocks = audio_blocks(&fx.media, 100_000);
        assert_eq!(
            blocks.len(),
            1,
            "{codec}: PTS quantization must not fragment the track"
        );
        let block = &blocks[0];
        // Let the codec/container declare its padding rather than assuming
        // that every encoder produces exactly three seconds of decoded PCM.
        // The CLI is an independent path through decode and resampling: our
        // clock/sink must neither trim those samples again nor append any.
        let reference = std::process::Command::new("ffmpeg")
            .args(["-v", "error", "-i"])
            .arg(&fx.media)
            .args([
                "-map", "0:a:0", "-ar", "16000", "-ac", "1", "-f", "f32le", "pipe:1",
            ])
            .output()
            .expect("decode reference PCM");
        assert!(
            reference.status.success(),
            "{codec}: reference decode failed: {}",
            String::from_utf8_lossy(&reference.stderr)
        );
        assert_eq!(reference.stdout.len() % size_of::<f32>(), 0);
        assert_eq!(
            block.samples.len(),
            reference.stdout.len() / size_of::<f32>(),
            "{codec}: preserve the reference decoder's sample count"
        );

        // Energy centroids locate actual sound, not just the chunk's label.
        // Smooth symmetric markers avoid a codec-dependent silence threshold.
        // 2ms allows container millisecond rounding and lossy encoding while
        // remaining smaller than an extra Opus pre-skip (~6.5ms) or AAC frame.
        for expected in [0.25, 2.75] {
            let mut energy = 0.0;
            let mut moment = 0.0;
            for (i, sample) in block.samples.iter().enumerate() {
                let at = block.start.advance(i).seconds();
                if (at - expected).abs() < 0.1 {
                    let weight = f64::from(*sample).powi(2);
                    energy += weight;
                    moment += at * weight;
                }
            }
            assert!(energy > 1.0, "{codec}: missing tone marker at {expected}s");
            let actual = moment / energy;
            assert!(
                (actual - expected).abs() <= 0.002,
                "{codec}: tone marker at {actual:.6}s, expected {expected}s"
            );
        }
    }
}

#[test]
fn displays_rotated_and_mirrored_corners_in_their_declared_positions() {
    let red = [255u8, 0, 0];
    let green = [0, 255, 0];
    let blue = [0, 0, 255];
    let white = [255, 255, 255];
    // Each row names the displayed top-left, top-right, bottom-left, bottom-right.
    let cases = [
        (0, false, false, (80, 48), [red, green, blue, white]),
        (90, false, false, (48, 80), [green, white, red, blue]),
        (-90, false, false, (48, 80), [blue, red, white, green]),
        (180, false, false, (80, 48), [white, blue, green, red]),
        (0, true, false, (80, 48), [green, red, white, blue]),
        (0, false, true, (80, 48), [blue, white, red, green]),
        (90, true, false, (48, 80), [white, green, blue, red]),
        (90, false, true, (48, 80), [red, blue, green, white]),
        (0, true, true, (80, 48), [white, blue, green, red]),
        (180, true, true, (80, 48), [red, green, blue, white]),
    ];
    for (rotation, hflip, vflip, dimensions, corners) in cases {
        let fx = require_fixture!(support::fixtures::display_oriented(rotation, hflip, vflip));
        let case = format!("rotation={rotation}, hflip={hflip}, vflip={vflip}");
        let input = agent_video_core::media::open_input(&source_of(&fx.media), &Stop::new(), &[])
            .expect("probe input");
        let summary = agent_video_core::media::summarize_input(&input, &source_of(&fx.media))
            .expect("probe orientation");
        assert_eq!(
            (summary.width, summary.height),
            (dimensions.0 * 2, dimensions.1 * 2),
            "{case}"
        );
        let mut frames = FrameSource::open(
            &source_of(&fx.media),
            SampleSpec::Keyframes,
            48,
            None,
            &Discard,
        )
        .expect("opens");
        let frame = frames.next_frame().expect("decodes").expect("has a frame");
        let (w, h) = frame.rgb.dimensions();

        assert_eq!((w, h), dimensions, "{case}");
        assert_eq!(frames.aspect(), Some(w as f32 / h as f32), "{case}");
        let positions = [
            (w / 4, h / 4),
            (3 * w / 4, h / 4),
            (w / 4, 3 * h / 4),
            (3 * w / 4, 3 * h / 4),
        ];
        for ((x, y), expected) in positions.into_iter().zip(corners) {
            let actual = frame.rgb.get_pixel(x, y).0;
            assert!(
                actual
                    .into_iter()
                    .zip(expected)
                    .all(|(a, e)| a.abs_diff(e) < 12),
                "{case}, pixel ({x}, {y}): expected {expected:?}, got {actual:?}"
            );
        }
    }
}

#[test]
fn rejects_unsupported_display_rotation_instead_of_returning_sideways_pixels() {
    let fx = require_fixture!(support::fixtures::display_oriented(45, false, false));
    let input = agent_video_core::media::open_input(&source_of(&fx.media), &Stop::new(), &[])
        .expect("probe input");
    let error = agent_video_core::media::summarize_input(&input, &source_of(&fx.media))
        .expect_err("probe rejects unsupported rotation");
    assert_eq!(error.code, agent_video_core::error::ErrorCode::DecodeFailed);
    let error = FrameSource::open(
        &source_of(&fx.media),
        SampleSpec::Keyframes,
        48,
        None,
        &Discard,
    )
    .err()
    .expect("a non-right-angle matrix cannot be represented");

    assert_eq!(error.code, agent_video_core::error::ErrorCode::DecodeFailed);
    assert_eq!(
        error.context().and_then(|c| c.input_path.as_deref()),
        Some(fx.media.as_path())
    );
}

#[test]
fn renders_the_held_scene_after_revisits_and_brightness_flashes() {
    use agent_video_core::types::LayoutKind;
    for (uniform_flashes, expected_start) in [(false, 3.0), (true, 19.0)] {
        let fx = require_fixture!(support::fixtures::revisited_static_scene(uniform_flashes));
        let dir = support::fresh_output_dir(if uniform_flashes {
            "flash-hold"
        } else {
            "revisit-hold"
        });
        let cfg = support::config(
            512,
            Overrides {
                base_fps: Some(1.0),
                ..Default::default()
            },
        );
        pipeline::process(
            &source_of(&fx.media),
            OutputSink::Files(&dir),
            &cfg,
            pipeline::ProcessOpts {
                threads: 1,
                max_in_flight: pipeline::DEFAULT_IN_FLIGHT,
                subs: SubtitleSource::Auto,
                window: None,
                stop: Stop::new(),
                events: &Discard,
            },
        )
        .expect("processes the 80-second fixture");
        let meta = support::read_metadata(&dir);
        let statics: Vec<_> = meta
            .grids
            .iter()
            .filter(|g| g.layout == LayoutKind::Static)
            .collect();
        assert_eq!(
            statics.len(),
            1,
            "short transitions must not emit static entries"
        );
        assert_eq!(
            (statics[0].time_range.start(), statics[0].time_range.end()),
            (expected_start, 79.0)
        );
        assert_eq!(meta.total_duration(), Some(80.0));
        let GridImage::File(name) = &statics[0].image else {
            panic!("file output expected")
        };
        let image = image::open(dir.join(name)).expect("static JPEG").to_rgb8();
        let (width, height) = image.dimensions();
        let left = image.get_pixel(width / 4, 3 * height / 4).0[0];
        let right = image.get_pixel(3 * width / 4, 3 * height / 4).0[0];
        assert!(left > 200, "B is bright on the left, got {left}");
        if uniform_flashes {
            assert!(right > 200, "held white frame, got {right}");
        } else {
            assert!(right < 60, "B is dark on the right, got {right}");
        }
    }
}
