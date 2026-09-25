// SPDX-License-Identifier: Apache-2.0
//! Native-allocation leak check, alone in its own test binary so no other
//! test's allocations share the process RSS it measures.

#[allow(dead_code)]
mod support;

use agent_video_core::media::Stop;
use agent_video_core::media::embedded_subtitle::extract_track;
use agent_video_core::source::Source;

/// Resident set size in KiB, from `/proc/self/status`.
#[cfg(target_os = "linux")]
fn rss_kib() -> u64 {
    let status = std::fs::read_to_string("/proc/self/status").expect("procfs is readable");
    status
        .lines()
        .find_map(|l| l.strip_prefix("VmRSS:"))
        .and_then(|v| v.trim().trim_end_matches("kB").trim().parse().ok())
        .expect("VmRSS is reported")
}

#[cfg(target_os = "linux")]
#[test]
fn repeated_subtitle_extraction_holds_memory_flat() {
    // Every pass reads each packet of the file and decodes 400 KB of
    // subtitle text; libav owns both allocations, so a pass that fails to
    // release them grows RSS by roughly that much.
    let Some(fx) = support::fixtures::embedded_subtitle_track(6.0, 400) else {
        eprintln!("skipping: no ffmpeg CLI on PATH");
        return;
    };
    let source = Source::classify(fx.media.to_str().expect("fixture path is utf-8"));
    let pass = || {
        let cues = extract_track(&source, 0, "subtitle track", &Stop::new()).expect("extracts");
        assert_eq!(cues.len(), 400);
    };
    pass(); // warm the allocator
    let before = rss_kib();

    for _ in 0..30 {
        pass();
    }

    let grown = rss_kib().saturating_sub(before);
    assert!(grown < 8 * 1024, "30 passes grew RSS by {grown} KiB");
}
