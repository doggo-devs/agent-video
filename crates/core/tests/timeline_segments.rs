// SPDX-License-Identifier: Apache-2.0
//! Integration tests for the decode pass's timeline segmenter: real media
//! whose packet timestamps were rewritten (see
//! [`support::fixtures::corrupt_pts`]) decoded through [`FrameSource`], the
//! same path production uses — packet pump, segmenter, range end, sampler,
//! EOF flush, and the signal plumbing back to the event sink.
//!
//! Each test pins one adversarial shape from the five-generation attack
//! history under the segment paradigm's contract: every decodable frame is
//! either sampled onto a monotone logical timeline or dropped with a
//! signal; committed clock seams are announced; timestamps alone never
//! fail a run; and a normal file passes with zero timeline events. Errors
//! are label errors — a wrong seam mislabels, it cannot lose content
//! wholesale.

#[allow(dead_code)]
mod support;

use std::path::Path;
use std::sync::Mutex;

use agent_video_core::config::{Config, Overrides, StrategyOpts};
use agent_video_core::error::Error;
use agent_video_core::event::{Event, EventSink};
use agent_video_core::media::Stop;
use agent_video_core::media::decode::{FrameSource, SampleSpec};
use agent_video_core::pipeline::{self, OutputSink, SubtitleSource};
use agent_video_core::probe::SampleStrategy;
use agent_video_core::source::Source;
use agent_video_core::types::TimeRange;

/// +24 hours in 90 kHz ticks: far outside any fixture's declared extent,
/// yet still within MPEG-TS's 33-bit pts field.
const FUTURE: &str = "7776000000";

/// +12 hours in 90 kHz ticks: a second, mutually inconsistent corrupt
/// clock regime for the three-clock-domain fixture.
const HALF_FUTURE: &str = "3888000000";

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

/// Records every event's Debug rendering, so assertions can pin which
/// diagnostics a run emitted without depending on CLI wording.
#[derive(Default)]
struct Recorder(Mutex<Vec<String>>);

impl EventSink for Recorder {
    fn event(&self, event: Event<'_>) {
        self.0
            .lock()
            .expect("recorder lock")
            .push(format!("{event:?}"));
    }
}

impl Recorder {
    fn saw(&self, needle: &str) -> bool {
        self.0
            .lock()
            .expect("recorder lock")
            .iter()
            .any(|e| e.contains(needle))
    }
}

/// Decode `media` per `spec` through the production frame source and
/// collect every sampled timestamp.
fn drain(media: &Path, spec: SampleSpec, events: &dyn EventSink) -> Result<Vec<f32>, Error> {
    let source = Source::classify(media.to_str().expect("fixture path is utf-8"));
    let mut stream = FrameSource::open(&source, spec, 64, None, events)?;
    let mut ts = Vec::new();
    while let Some(frame) = stream.next_frame()? {
        ts.push(frame.ts.get());
    }
    Ok(ts)
}

/// A tick grid denser than the fixture's frame rate: every released frame
/// is sampled, so the collected timestamps mirror the segmenter's releases.
fn full_speed() -> SampleSpec {
    SampleSpec::Fps {
        fps: 120.0,
        start: 0.0,
        end: None,
    }
}

fn assert_monotone(ts: &[f32]) {
    assert!(
        ts.windows(2).all(|w| w[0] <= w[1]),
        "released timeline must be non-decreasing: {ts:?}"
    );
}

/// An explicitly fixed-fps, probe-skipping config: the corrupt fixtures
/// poison the probed duration and keyframe order, and this suite tests the
/// decode pass, not the probe layer.
fn fixed_fps_config() -> Config {
    Config::new(
        512,
        None,
        &Overrides {
            base_fps: Some(10.0),
            chunk_seconds: Some(1.0),
            ..Default::default()
        },
        StrategyOpts {
            strategy: Some(SampleStrategy::FixedFps),
            force: true,
            ..Default::default()
        },
    )
    .expect("test config builds")
}

fn process(media: &Path, dir: &Path, events: &Recorder) -> Result<(), Error> {
    pipeline::process(
        &Source::classify(media.to_str().expect("fixture path is utf-8")),
        OutputSink::Files(dir),
        &fixed_fps_config(),
        pipeline::ProcessOpts {
            threads: 0,
            max_in_flight: pipeline::DEFAULT_IN_FLIGHT,
            subs: SubtitleSource::Auto,
            window: None,
            stop: Stop::new(),
            events,
        },
    )
}

/// The poisoned-head fixture: only packet 0's pts is shifted +24h, which
/// drags the container's declared `start_time` (observed: 86400.0) — so
/// every legal frame rebases to about -86398s. Under the old paradigm this
/// killed the whole intact video as INPUT_CORRUPT and needed a re-decode
/// retry; the segmenter re-anchors the first confirmed segment at the
/// stream head in one pass.
fn poisoned_head() -> Option<support::fixtures::Fixture> {
    support::fixtures::corrupt_pts(
        "ts-origin-poisoned-head",
        2.0,
        &format!("if(eq(N\\,0)\\,PTS+{FUTURE}\\,PTS)"),
    )
}

#[test]
fn keeps_a_clean_stream_event_free() {
    let fx = require_fixture!(support::fixtures::h264_talk(6.0));
    let events = Recorder::default();

    let ts = drain(&fx.media, full_speed(), &events).expect("a clean stream decodes");

    assert!(ts.len() >= 150, "content decodes: {} frames", ts.len());
    assert_monotone(&ts);
    assert!(
        !events.saw("UnplaceableTimestamp")
            && !events.saw("TimestampsDropped")
            && !events.saw("TimelineDiscontinuity"),
        "a normal file emits zero timeline events"
    );
}

/// A single poisoned head packet drags the declared origin a day forward.
/// The head candidate dies against the confirmed legal challenger; the
/// first segment re-anchors at the stream head, so content keeps ~0-based
/// labels — no verdict, no retry, one signaled drop.
#[test]
fn survives_a_single_frame_poisoned_head() {
    let fx = require_fixture!(poisoned_head());
    let events = Recorder::default();

    let ts = drain(&fx.media, full_speed(), &events).expect("timestamps never fail a run");

    assert!(ts.len() >= 50, "content survives: {} frames", ts.len());
    assert!(
        ts.iter().all(|&t| (-1.0..100.0).contains(&t)),
        "labels are re-anchored near zero: {:?}..{:?}",
        ts.first(),
        ts.last()
    );
    assert!(
        ts.last().is_some_and(|&t| t > 1.5),
        "the tail is not truncated: last {:?}",
        ts.last()
    );
    assert_monotone(&ts);
    assert!(events.saw("UnplaceableTimestamp"), "the drop is signaled");
    assert!(events.saw("TimestampsDropped"), "drop total is signaled");
}

/// Two mutually-vouching poisoned head packets — the shape that defeated
/// the gen-5 election. A two-frame head candidate is still below the
/// confirmation quorum, so the legal challenger replaces it wholesale.
#[test]
fn survives_a_double_frame_poisoned_head() {
    let fx = require_fixture!(support::fixtures::corrupt_pts(
        "ts-seg-double-head",
        2.0,
        &format!("if(lte(N\\,1)\\,PTS+{FUTURE}\\,PTS)"),
    ));
    let events = Recorder::default();

    let ts = drain(&fx.media, full_speed(), &events).expect("timestamps never fail a run");

    assert!(ts.len() >= 50, "content survives: {} frames", ts.len());
    assert!(
        ts.iter().all(|&t| (-1.0..100.0).contains(&t)),
        "labels are re-anchored near zero"
    );
    assert!(ts.last().is_some_and(|&t| t > 1.5), "tail not truncated");
    assert_monotone(&ts);
    assert!(events.saw("UnplaceableTimestamp"), "drops are signaled");
}

/// 65 poisoned frames over a 300-frame base: enough to confirm as the
/// stream's first segment (the gen-5 election lost here). The paradigm's
/// answer is a label error with zero content loss: the poisoned run keeps
/// its 65 frames at the head of the logical clock, a seam is committed,
/// and the 235 legal frames follow — nothing is dropped, nothing fails.
#[test]
fn survives_a_sixty_five_frame_poisoned_head() {
    let fx = require_fixture!(support::fixtures::corrupt_pts(
        "ts-seg-65-head",
        10.0,
        &format!("if(lte(N\\,64)\\,PTS+{FUTURE}\\,PTS)"),
    ));
    let events = Recorder::default();

    let ts = drain(&fx.media, full_speed(), &events).expect("timestamps never fail a run");

    assert!(
        ts.len() >= 290,
        "all content survives, poisoned labels included: {} frames",
        ts.len()
    );
    assert!(
        ts.iter().all(|&t| (-1.0..20.0).contains(&t)),
        "the logical duration does not explode: last {:?}",
        ts.last()
    );
    assert_monotone(&ts);
    assert!(
        events.saw("TimelineDiscontinuity"),
        "the committed seam is announced"
    );
}

/// Three mutually inconsistent clock domains (the majority-corrupt shape
/// that stayed INPUT_CORRUPT after its one rebase under the old paradigm).
/// Now: each domain is a segment, content is preserved across two seams,
/// and the run succeeds — the paradigm flip made explicit.
#[test]
fn survives_three_clock_domains() {
    let fx = require_fixture!(support::fixtures::corrupt_pts(
        "ts-origin-corrupt-majority",
        3.0,
        &format!("if(lte(N\\,1)\\,PTS+{FUTURE}\\,if(lte(N\\,46)\\,PTS+{HALF_FUTURE}\\,PTS))"),
    ));
    let events = Recorder::default();

    let ts = drain(&fx.media, full_speed(), &events).expect("timestamps never fail a run");

    assert!(
        ts.len() >= 80,
        "the two confirmed domains survive whole: {} frames",
        ts.len()
    );
    assert!(
        ts.iter().all(|&t| (-1.0..20.0).contains(&t)),
        "the logical duration does not explode"
    );
    assert_monotone(&ts);
    assert!(
        events.saw("TimelineDiscontinuity") || events.saw("UnplaceableTimestamp"),
        "seams or drops are signaled, never silent"
    );
}

/// Alternating poisoned pts — the fragment-storm shape. Loss is bounded to
/// the poisoned frames themselves (each dies as an unconfirmed challenger,
/// signaled), the legal half streams on, and the logical duration stays at
/// content scale. (The decoder's best-effort timestamps heal most of the
/// alternation from clean dts; whatever poison survives to the segmenter
/// is what this pins.)
#[test]
fn bounds_alternating_poison_with_signals() {
    let fx = require_fixture!(support::fixtures::corrupt_pts(
        "ts-s3-alternating",
        2.0,
        &format!("if(eq(mod(N\\,2)\\,1)\\,PTS+{FUTURE}\\,PTS)"),
    ));
    let events = Recorder::default();

    let ts = drain(&fx.media, full_speed(), &events).expect("timestamps never fail a run");

    assert!(ts.len() >= 30, "the legal half survives: {} frames", ts.len());
    assert!(
        ts.iter().all(|&t| (-1.0..100.0).contains(&t)),
        "the logical duration does not explode"
    );
    assert_monotone(&ts);
    assert!(
        events.saw("UnplaceableTimestamp"),
        "dropped frames are never silent"
    );
    assert!(events.saw("TimestampsDropped"), "the total is signaled");
}

/// The mutually-vouching corrupt pair *after* a legal head (S1): the pair
/// buffers as a challenger, the next legal frame kills it, and every legal
/// frame — the head included — survives.
#[test]
fn survives_a_mutually_vouching_corrupt_pair() {
    let fx = require_fixture!(support::fixtures::corrupt_pts(
        "ts-s1-head-pair",
        2.0,
        &format!("if(between(N\\,1\\,2)\\,PTS+{FUTURE}\\,PTS)"),
    ));
    let events = Recorder::default();

    let ts = drain(&fx.media, full_speed(), &events).expect("timestamps never fail a run");

    assert!(ts.len() >= 50, "content survives: {} frames", ts.len());
    assert!(
        ts.iter().all(|&t| t < 100.0),
        "no corrupt timestamp reaches the output: max {:?}",
        ts.last()
    );
    assert!(ts.last().is_some_and(|&t| t > 1.5), "tail not truncated");
    assert_monotone(&ts);
    assert!(events.saw("UnplaceableTimestamp"), "drops are signaled");
    assert!(events.saw("TimestampsDropped"), "the total is signaled");
}

/// A corrupt pts on the very last frame (S2): no successor ever vouches
/// for it, so the challenger it opened is dropped at EOF with a signal —
/// never released into a garbage label.
#[test]
fn drops_an_unvouched_corrupt_tail_at_eof() {
    let fx = require_fixture!(support::fixtures::corrupt_pts(
        "ts-s2-tail",
        2.0,
        &format!("if(gte(PTS\\,176000)\\,PTS+{FUTURE}\\,PTS)"),
    ));
    let events = Recorder::default();

    let ts = drain(&fx.media, full_speed(), &events).expect("timestamps never fail a run");

    assert!(ts.len() >= 50, "content survives: {} frames", ts.len());
    assert!(
        ts.iter().all(|&t| t < 100.0),
        "the unvouched tail is not released: max {:?}",
        ts.last()
    );
    assert!(ts.last().is_some_and(|&t| t > 1.5), "legal tail arrives");
    assert_monotone(&ts);
    assert!(events.saw("UnplaceableTimestamp"), "the drop is signaled");
    assert!(events.saw("TimestampsDropped"), "the total is signaled");
}

/// A corrupt (but near-zero) pts on the very first frame (S4): the head is
/// genuinely unattributable, the confirmed challenger replaces it, exactly
/// one frame is lost — with a signal, not a guess dressed up as knowledge.
#[test]
fn resolves_an_ambiguous_stream_head_with_a_signal() {
    let fx = require_fixture!(support::fixtures::corrupt_pts(
        "ts-s4-head",
        2.0,
        "if(eq(N\\,0)\\,135000\\,PTS)",
    ));
    let events = Recorder::default();

    let ts = drain(&fx.media, full_speed(), &events).expect("an ambiguous head must not fail");

    assert!(
        ts.len() >= 10,
        "the stream flows past the ambiguous head: {} frames",
        ts.len()
    );
    assert!(
        ts.iter().all(|&t| (-0.1..1.0).contains(&t)),
        "only the legal timeline is sampled: {ts:?}"
    );
    assert_monotone(&ts);
    assert!(
        events.saw("UnplaceableTimestamp"),
        "the unattributable head drop is signaled"
    );
}

/// A corrupt far-future pts inside a bounded range must not trigger the
/// range end: it buffers as a challenger, never releases, and the range
/// keeps sampling to its true end.
#[test]
fn does_not_let_a_corrupt_pts_trigger_the_range_end_early() {
    let fx = require_fixture!(support::fixtures::corrupt_pts(
        "ts-s1-head-pair",
        2.0,
        &format!("if(between(N\\,1\\,2)\\,PTS+{FUTURE}\\,PTS)"),
    ));
    let events = Recorder::default();

    let ts = drain(
        &fx.media,
        SampleSpec::Fps {
            fps: 120.0,
            start: 0.0,
            end: Some(1.0),
        },
        &events,
    )
    .expect("a bounded range must survive corrupt timestamps");

    assert!(
        ts.len() >= 20,
        "the range is fully sampled, not truncated: {} frames",
        ts.len()
    );
    assert!(
        ts.iter().all(|&t| t < 1.0),
        "the range end still bounds the output: {ts:?}"
    );
    assert_monotone(&ts);
}

/// The poisoned-head stream end to end: `process` succeeds in a single
/// pass (no retry layer exists anymore), grids cover the real two-second
/// video, and every reported timestamp is measured from the re-anchored
/// origin.
#[test]
fn process_reports_the_real_video_despite_a_poisoned_head() {
    let fx = require_fixture!(poisoned_head());
    let events = Recorder::default();
    let dir = support::fresh_output_dir("segments-poisoned-head");

    process(&fx.media, &dir, &events).expect("one pass, no verdict, no retry");

    let meta = support::read_metadata(&dir);
    assert!(!meta.grids.is_empty(), "grids cover the surviving content");
    let total = meta.total_duration().expect("process document");
    assert!(
        (1.5..5.0).contains(&total),
        "reported duration {total} tracks the real 2s video, not the corrupt pts"
    );
    for grid in &meta.grids {
        let (start, end) = support::parse_wire_range(&grid.time_range);
        assert!(
            start >= 0.0 && end < 100.0,
            "grid timestamps are measured from the re-anchored origin: {}",
            grid.time_range
        );
    }
    assert!(
        events.saw("TimestampsDropped"),
        "the pipeline surfaces the decode-pass drops"
    );
}

/// The extract door shares the decode pass: a range extraction over the
/// poisoned-head stream composes its grid from the re-anchored timeline in
/// one pass.
#[test]
fn extracts_a_range_across_a_poisoned_head() {
    let fx = require_fixture!(poisoned_head());
    let events = Recorder::default();
    let dir = support::fresh_output_dir("segments-extract");

    pipeline::extract_range(
        &Source::classify(fx.media.to_str().expect("fixture path is utf-8")),
        OutputSink::Files(&dir),
        &fixed_fps_config(),
        TimeRange::new(0.0, 2.0).expect("valid range"),
        SubtitleSource::Auto,
        &Stop::new(),
        &events,
    )
    .expect("extract succeeds in one pass");

    assert!(
        dir.join("grid_extract.jpg").exists(),
        "the extract grid is written from the re-anchored timeline"
    );
}

/// S1: a single forward blip below the old 10-second fence floor (+9.5s
/// on a 30 fps stream). The two-door design admitted it unconfirmed — it
/// advanced the committed cursor, inflated the learned gap statistics,
/// and pushed the next ~49 legal frames into the backward-jitter drop
/// path: one poisoned frame cost 49 legal ones, with no seam signal.
/// Single-door: it buffers as a challenger, the next legal frame kills
/// it, and the stream survives whole.
#[test]
fn quarantines_a_sub_ten_second_forward_blip() {
    let fx = require_fixture!(support::fixtures::corrupt_pts(
        "ts-s1-fence-blip",
        2.0,
        "if(eq(N\\,10)\\,PTS+855000\\,PTS)",
    ));
    let events = Recorder::default();

    let ts = drain(&fx.media, full_speed(), &events).expect("timestamps never fail a run");

    assert!(ts.len() >= 55, "the legal frames survive whole: {} frames", ts.len());
    assert!(
        ts.iter().all(|&t| (-0.1..3.0).contains(&t)),
        "the blip never enters the released history: {:?}..{:?}",
        ts.first(),
        ts.last()
    );
    assert_monotone(&ts);
}

/// S2: a legal long-GOP recording (screen capture, `-g 600` encodes)
/// whose keyframes run 0,2,..,10 then jump to 25 and 39.9. The tail
/// keyframes are content at a plausible position inside the declared
/// 40s extent with nothing left to contradict them; dropping them at EOF
/// as an unconfirmed challenger loses the last 30 seconds of a healthy
/// file — the exact content loss this paradigm's contract forbids.
#[test]
fn keyframe_decode_surfaces_a_legal_sparse_tail() {
    let fx = require_fixture!(support::fixtures::sparse_tail_keyframes(40.0));
    let events = Recorder::default();

    let ts = drain(&fx.media, SampleSpec::Keyframes, &events).expect("a legal file never fails");

    assert!(
        ts.iter().any(|&t| (24.0..26.0).contains(&t)),
        "the 25s keyframe is visible: {ts:?}"
    );
    assert!(
        ts.last().is_some_and(|&t| t > 35.0),
        "the tail keyframe is visible: {ts:?}"
    );
    assert_monotone(&ts);
    assert!(
        !events.saw("UnplaceableTimestamp")
            && !events.saw("TimestampsDropped")
            && !events.saw("TimelineDiscontinuity"),
        "a legal file emits zero timeline events"
    );
}

/// S3: a window seek on the poisoned-head stream speaks the dragged
/// container clock (logical 1.0 → ~86401s absolute), reports success,
/// and decodes nothing — under the old assembly the run died as
/// NoFramesExtracted with zero events, timestamps alone failing a run.
/// The decode pass now detects the empty landing at demuxer EOF, rewinds
/// to the stream head, announces the fallback, and serves the window
/// sequentially from content.
#[test]
fn recovers_a_window_seek_dragged_by_a_poisoned_head() {
    let fx = require_fixture!(poisoned_head());
    let events = Recorder::default();

    let ts = drain(
        &fx.media,
        SampleSpec::Fps {
            fps: 120.0,
            start: 1.0,
            end: Some(2.0),
        },
        &events,
    )
    .expect("a poisoned index must not fail the window");

    assert!(!ts.is_empty(), "the window is served from content");
    assert!(
        ts.iter().all(|&t| (0.9..2.0).contains(&t)),
        "window labels stay inside the request: {ts:?}"
    );
    assert_monotone(&ts);
    assert!(events.saw("SeekFallback"), "the degraded seek is announced");
}

/// The extract door over the same S3 shape: `extract_range(1.0, 2.0)` on
/// the poisoned-head stream must produce a grid, not NoFramesExtracted.
#[test]
fn extracts_a_mid_range_across_a_poisoned_head() {
    let fx = require_fixture!(poisoned_head());
    let events = Recorder::default();
    let dir = support::fresh_output_dir("segments-extract-mid");

    pipeline::extract_range(
        &Source::classify(fx.media.to_str().expect("fixture path is utf-8")),
        OutputSink::Files(&dir),
        &fixed_fps_config(),
        TimeRange::new(1.0, 2.0).expect("valid range"),
        SubtitleSource::Auto,
        &Stop::new(),
        &events,
    )
    .expect("a timestamp alone never fails a run");

    assert!(
        dir.join("grid_extract.jpg").exists(),
        "the mid-range grid is written from recovered content"
    );
    assert!(events.saw("SeekFallback"), "the degraded seek is announced");
}

/// The regression pinned since the first design: an edit-list cut mp4 has
/// legitimate negative packet pts (pre-roll) against a zero header origin.
/// It is a normal file: one identity segment, zero timeline events, and
/// the duration tracks the cut.
#[test]
fn processes_an_edit_list_cut_as_a_single_identity_segment() {
    let fx = require_fixture!(support::fixtures::edit_list_cut(4.0, 1.5));
    let events = Recorder::default();
    let dir = support::fresh_output_dir("segments-edit-list");

    process(&fx.media, &dir, &events).expect("an edit-list cut is a normal file");

    assert!(
        !events.saw("TimelineDiscontinuity")
            && !events.saw("UnplaceableTimestamp")
            && !events.saw("TimestampsDropped"),
        "legal pre-roll emits zero timeline events"
    );
    let meta = support::read_metadata(&dir);
    assert!(!meta.grids.is_empty(), "grids cover the cut video");
    let total = meta.total_duration().expect("process document");
    assert!(
        (2.0..4.0).contains(&total),
        "reported duration {total} tracks the 2.5s cut"
    );
}
