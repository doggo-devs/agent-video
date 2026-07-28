// SPDX-License-Identifier: Apache-2.0
//! End-to-end tests of the origin rebase: a corrupt first packet drags an
//! MPEG-TS container's declared `start_time` (and with it the pipeline's
//! clock origin) a day into the future, so every legal frame rebases far
//! out of the plausible window and the whole, essentially intact video
//! used to die as `INPUT_CORRUPT`. The failure's own audit — a minority
//! released, a longer monotone chain rejected — now indicts the anchor
//! instead: the run re-decodes once anchored at the dropped chain's head
//! ([`Event::OriginRebased`]) and succeeds with correct timestamps.
//!
//! The boundaries are pinned too: a rebased pass that still cannot form a
//! trustworthy timeline is a final `INPUT_CORRUPT` (no second retry), and
//! an edit-list cut mp4 — legal negative packet pts against a zero header
//! origin, releases in the majority — never triggers a rebase at all.

#[allow(dead_code)]
mod support;

use std::path::Path;
use std::sync::Mutex;

use agent_video_core::config::{Config, Overrides, StrategyOpts};
use agent_video_core::error::{Error, ErrorCode};
use agent_video_core::event::{Event, EventSink};
use agent_video_core::media::Stop;
use agent_video_core::media::decode::{AnchorFault, FrameSource, SampleSpec};
use agent_video_core::pipeline::{self, OutputSink, SubtitleSource};
use agent_video_core::probe::SampleStrategy;
use agent_video_core::source::Source;
use agent_video_core::types::TimeRange;

/// +24 hours in 90 kHz ticks: far outside any fixture's declared extent,
/// yet still within MPEG-TS's 33-bit pts field.
const FUTURE: &str = "7776000000";

/// +12 hours in 90 kHz ticks: a second, mutually inconsistent corrupt
/// clock regime for the majority-corrupt fixture.
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

/// The poisoned-head fixture: only packet 0's pts is shifted +24h, which
/// is enough to drag the container's `start_time` (observed: 86400.0) and
/// its derived duration (observed: 0.033s) — the anchor, not the frames,
/// is what breaks.
fn poisoned_head() -> Option<support::fixtures::Fixture> {
    support::fixtures::corrupt_pts(
        "ts-origin-poisoned-head",
        2.0,
        &format!("if(eq(N\\,0)\\,PTS+{FUTURE}\\,PTS)"),
    )
}

/// An explicitly fixed-fps, probe-skipping config: the corrupt fixtures
/// poison the probed duration and keyframe order, and this suite tests the
/// decode-pass rebase, not the probe layer.
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

/// The first pass alone (no retry layer): the timeline failure must carry
/// the election so the layer above can rebase — this pins the evidence
/// plumbing from the gate through the decode thread into the coded error.
#[test]
fn carries_the_anchor_fault_out_of_the_failing_first_pass() {
    let fx = require_fixture!(poisoned_head());
    let events = Recorder::default();
    let source = Source::classify(fx.media.to_str().expect("fixture path is utf-8"));

    let mut stream = FrameSource::open(
        &source,
        SampleSpec::Fps {
            fps: 120.0,
            start: 0.0,
            end: None,
        },
        64,
        None,
        &events,
    )
    .expect("open succeeds; the failure is a decode-pass verdict");
    let err = loop {
        match stream.next_frame() {
            Ok(Some(_)) => {}
            Ok(None) => panic!("the poisoned declared pass must fail"),
            Err(e) => break e,
        }
    };

    assert_eq!(err.code, ErrorCode::InputCorrupt);
    let fault = AnchorFault::in_error(&err).expect("the dropped majority elects a rebase");
    assert!(
        (fault.declared - 86_400.0).abs() < 1.0,
        "declared origin is the poisoned start_time: {}",
        fault.declared
    );
    assert!(
        (0.0..0.1).contains(&fault.elected),
        "elected origin is the first legal frame's pts: {}",
        fault.elected
    );
}

/// The headline behavior: the poisoned-head stream, processed end to end,
/// succeeds after one rebase — grids cover the real two-second video and
/// every reported timestamp is measured from the true origin, not the
/// corrupt one.
#[test]
fn rebases_the_origin_when_a_poisoned_head_drags_the_clock() {
    let fx = require_fixture!(poisoned_head());
    let events = Recorder::default();
    let dir = support::fresh_output_dir("origin-rebase-head");

    process(&fx.media, &dir, &events).expect("the rebased second pass must succeed");

    assert!(
        events.saw("OriginRebased"),
        "the rebase is announced as a diagnostic event"
    );
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
            "grid timestamps are measured from the true origin: {}",
            grid.time_range
        );
    }
}

/// The final-death boundary: a majority-corrupt stream (a poisoned head
/// pair plus a second corrupt clock regime over most frames) earns its one
/// rebase, and when the rebased pass cannot form a trustworthy timeline
/// either, the run is a final `INPUT_CORRUPT` — no second election, no
/// retry loop.
#[test]
fn stays_input_corrupt_when_the_rebased_pass_is_still_broken() {
    let fx = require_fixture!(support::fixtures::corrupt_pts(
        "ts-origin-corrupt-majority",
        3.0,
        &format!("if(lte(N\\,1)\\,PTS+{FUTURE}\\,if(lte(N\\,46)\\,PTS+{HALF_FUTURE}\\,PTS))"),
    ));
    let events = Recorder::default();
    let dir = support::fresh_output_dir("origin-rebase-majority");

    let err = process(&fx.media, &dir, &events)
        .expect_err("a majority-corrupt stream must die after its one retry");

    assert_eq!(err.code, ErrorCode::InputCorrupt);
    assert!(
        events.saw("OriginRebased"),
        "the one rebase was attempted before the final verdict"
    );
}

/// The regression the design pinned up front: an edit-list cut mp4 has
/// legitimate negative packet pts (edit-list pre-roll) against a header
/// start_time of 0. Releases hold the majority on such files, so they
/// must process cleanly without ever triggering a rebase.
#[test]
fn processes_an_edit_list_cut_without_rebasing() {
    let fx = require_fixture!(support::fixtures::edit_list_cut(4.0, 1.5));
    let events = Recorder::default();
    let dir = support::fresh_output_dir("origin-rebase-edit-list");

    process(&fx.media, &dir, &events).expect("an edit-list cut is a normal file");

    assert!(
        !events.saw("OriginRebased"),
        "legal edit-list pre-roll must not look like a broken anchor"
    );
    let meta = support::read_metadata(&dir);
    assert!(!meta.grids.is_empty(), "grids cover the cut video");
    let total = meta.total_duration().expect("process document");
    assert!(
        (2.0..4.0).contains(&total),
        "reported duration {total} tracks the 2.5s cut"
    );
}

/// The extract door shares the retry layer: a range extraction over the
/// poisoned-head stream rebases once and composes its grid from the real
/// timeline.
#[test]
fn rebases_through_the_extract_door() {
    let fx = require_fixture!(poisoned_head());
    let events = Recorder::default();
    let dir = support::fresh_output_dir("origin-rebase-extract");

    pipeline::extract_range(
        &Source::classify(fx.media.to_str().expect("fixture path is utf-8")),
        OutputSink::Files(&dir),
        &fixed_fps_config(),
        TimeRange::new(0.0, 2.0).expect("valid range"),
        SubtitleSource::Auto,
        &Stop::new(),
        &events,
    )
    .expect("the rebased second pass must succeed");

    assert!(events.saw("OriginRebased"), "the rebase is announced");
    assert!(
        dir.join("grid_extract.jpg").exists(),
        "the extract grid is written from the rebased timeline"
    );
}
