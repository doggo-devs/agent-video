// SPDX-License-Identifier: Apache-2.0
//! Worker-level integration tests for the decode pass's timeline gate: real
//! media whose packet timestamps were rewritten (see
//! [`support::fixtures::corrupt_pts`]) decoded through [`FrameSource`], the
//! same path production uses — packet pump, gate, range end, sampler, EOF
//! release, and the drop-report plumbing back to the event sink.
//!
//! Each test pins one adversarial shape from the review history: a
//! mutually-vouching corrupt head pair (S1), a corrupt pts on the very last
//! frame released only by EOF (S2), silent drops without any signal (S3),
//! the unattributable stream head (S4), and a corrupt pts triggering the
//! range end early. The shared contract: no input may combine significant
//! content loss with zero signal and a successful exit.

#[allow(dead_code)]
mod support;

use std::path::Path;
use std::sync::Mutex;

use agent_video_core::config::Overrides;
use agent_video_core::error::Error;
use agent_video_core::event::{Event, EventSink};
use agent_video_core::media::Stop;
use agent_video_core::media::decode::{FrameSource, SampleSpec};
use agent_video_core::pipeline::{self, OutputSink, SubtitleSource};
use agent_video_core::source::Source;

/// +24 hours in 90 kHz ticks: far outside any fixture's declared extent,
/// yet still within MPEG-TS's 33-bit pts field.
const FUTURE: &str = "7776000000";

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
/// is sampled, so the collected timestamps mirror the gate's releases.
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

/// S1: the first two frames carry corrupt, mutually increasing far-future
/// pts. Gen-3's gate released the pair (each vouched for the other),
/// poisoned the committed timeline, silently dropped every legal frame
/// (fewer than the 64-frame budget on a short stream), and reported
/// success with the whole video replaced by two garbage timestamps. The
/// extent anchor must drop the pair before it can vouch for anything.
#[test]
fn survives_a_mutually_vouching_corrupt_head_pair() {
    let fx = require_fixture!(support::fixtures::corrupt_pts(
        "ts-s1-head-pair",
        2.0,
        &format!("if(between(N\\,1\\,2)\\,PTS+{FUTURE}\\,PTS)"),
    ));
    let events = Recorder::default();

    let ts = drain(&fx.media, full_speed(), &events).expect("the legal stream must survive");

    assert!(ts.len() >= 50, "content survives: {} frames", ts.len());
    assert!(
        ts.iter().all(|&t| t < 100.0),
        "no corrupt timestamp reaches the output: max {:?}",
        ts.last()
    );
    assert!(
        ts.last().is_some_and(|&t| t > 1.5),
        "the tail of the video is not truncated: last {:?}",
        ts.last()
    );
    assert_monotone(&ts);
    assert!(events.saw("UnplaceableTimestamp"), "first drop is signaled");
    assert!(events.saw("TimestampsDropped"), "drop total is signaled");
}

/// S2: the corrupt pts sits on the very last frame, so no successor ever
/// contradicts it and only `finish()` decides. Gen-3 released it
/// unconfirmed into the sampled output (garbage chunk, garbage
/// time_range, successful exit) — a path with zero tests. Now it must be
/// dropped with a signal: by the extent anchor when the container's
/// declared duration is clean, or by the EOF jump rule when the corruption
/// poisoned the declared duration itself.
#[test]
fn refuses_an_unconfirmed_corrupt_tail_at_eof() {
    let fx = require_fixture!(support::fixtures::corrupt_pts(
        "ts-s2-tail",
        2.0,
        &format!("if(gte(PTS\\,176000)\\,PTS+{FUTURE}\\,PTS)"),
    ));
    let events = Recorder::default();

    let ts = drain(&fx.media, full_speed(), &events).expect("one corrupt tail must not fail");

    assert!(ts.len() >= 50, "content survives: {} frames", ts.len());
    assert!(
        ts.iter().all(|&t| t < 100.0),
        "the unvouched tail is not released into the output: max {:?}",
        ts.last()
    );
    assert!(
        ts.last().is_some_and(|&t| t > 1.5),
        "the legal tail still arrives: last {:?}",
        ts.last()
    );
    assert_monotone(&ts);
    assert!(events.saw("UnplaceableTimestamp"), "the drop is signaled");
    assert!(events.saw("TimestampsDropped"), "drop total is signaled");
}

/// S3: alternating good/corrupt timestamps. Gen-3 dropped the corrupt half
/// with zero events, zero logs, and a successful exit — observability
/// regressed below the pre-gate baseline. Every drop must now surface:
/// a first-drop event and an end-of-decode total. (The decoder's own
/// best-effort heuristic heals most alternating corruption from dts, so
/// what reaches the gate is the first corrupt timestamp; the contract
/// under test is that *whatever* is dropped is signaled.)
#[test]
fn signals_timeline_drops_instead_of_silence() {
    let fx = require_fixture!(support::fixtures::corrupt_pts(
        "ts-s3-alternating",
        2.0,
        &format!("if(eq(mod(N\\,2)\\,1)\\,PTS+{FUTURE}\\,PTS)"),
    ));
    let events = Recorder::default();

    let ts = drain(&fx.media, full_speed(), &events).expect("healable corruption must not fail");

    assert!(ts.len() >= 50, "the legal half survives: {} frames", ts.len());
    assert!(
        ts.iter().all(|&t| t < 100.0),
        "no corrupt timestamp reaches the output"
    );
    assert_monotone(&ts);
    assert!(
        events.saw("UnplaceableTimestamp"),
        "dropped frames must not be silent: first drop is signaled"
    );
    assert!(
        events.saw("TimestampsDropped"),
        "dropped frames must not be silent: the total is signaled"
    );
}

/// S4: the very first frame carries a corrupt (but in-extent) pts, so the
/// gate's first decision has two points and no anchor — genuinely
/// unattributable. The pinned contract is honesty, not clairvoyance: side
/// with the successor, lose exactly one frame, and *account* for it with a
/// signal instead of pretending the choice was knowledge. (The corrupt
/// head also drags the container's reported origin, so legal timestamps
/// rebase below zero; the run must still succeed with the content intact.)
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
        "the unattributable head drop is signaled, not guessed away"
    );
}

/// A corrupt far-future pts inside a `--range` decode must not trigger the
/// range end: gen-3 moved the end check behind the gate, but a released
/// corrupt frame still stopped the decode and silently truncated the
/// range. The extent anchor drops it before the end check ever sees it.
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
        "the range is fully sampled, not truncated at the corrupt pts: {} frames",
        ts.len()
    );
    assert!(
        ts.iter().all(|&t| t < 1.0),
        "the range end still bounds the output: {ts:?}"
    );
    assert_monotone(&ts);
}

/// The S1 shape end to end: `process` must report the real two-second
/// video — full coverage, sane duration, grids written — and surface the
/// drops as events through the pipeline's own sink. Gen-3 reported a
/// 24-hour duration around three frames and exited successfully.
#[test]
fn process_reports_the_real_video_despite_a_corrupt_head_pair() {
    let fx = require_fixture!(support::fixtures::corrupt_pts(
        "ts-s1-head-pair",
        2.0,
        &format!("if(between(N\\,1\\,2)\\,PTS+{FUTURE}\\,PTS)"),
    ));
    let events = Recorder::default();
    let dir = support::fresh_output_dir("timeline-corrupt-head");
    let cfg = support::config(
        512,
        Overrides {
            base_fps: Some(10.0),
            chunk_seconds: Some(1.0),
            ..Default::default()
        },
    );

    pipeline::process(
        &Source::classify(fx.media.to_str().expect("fixture path is utf-8")),
        OutputSink::Files(&dir),
        &cfg,
        pipeline::ProcessOpts {
            threads: 0,
            max_in_flight: pipeline::DEFAULT_IN_FLIGHT,
            subs: SubtitleSource::Auto,
            window: None,
            stop: Stop::new(),
            events: &events,
        },
    )
    .expect("process succeeds on the healable fixture");

    let meta = support::read_metadata(&dir);
    assert!(!meta.grids.is_empty(), "grids cover the surviving content");
    let total = meta.total_duration().expect("process document");
    assert!(
        (1.5..5.0).contains(&total),
        "reported duration {total} tracks the real 2s video, not the corrupt pts"
    );
    assert!(
        events.saw("TimestampsDropped"),
        "the pipeline surfaces the decode-pass drops"
    );
}
