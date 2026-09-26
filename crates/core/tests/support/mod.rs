// SPDX-License-Identifier: Apache-2.0
//! Shared plumbing for the real-media integration tests: fixture synthesis
//! ([`fixtures`]), output-directory management, and parsing of the wire
//! contract's `"MM:SS"` timestamp strings back into seconds.

pub mod fixtures;

use std::fs;
use std::path::{Path, PathBuf};

use agent_video_core::config::{Config, Overrides, StrategyOpts};
use agent_video_core::wire;

/// Build a test [`Config`] with no profile and default strategy options.
pub fn config(max_edge: u32, overrides: Overrides) -> Config {
    Config::new(max_edge, None, &overrides, StrategyOpts::default()).expect("test config builds")
}

/// RAII output directory for one pipeline run: unique per test tag and
/// process, removed on drop. Assertion failures unwind through Drop, so
/// failed runs stop littering /tmp; only an abort leaks. The directory
/// itself is created by the pipeline's file sink, not here.
pub struct OutputDir(PathBuf);

impl std::ops::Deref for OutputDir {
    type Target = Path;

    fn deref(&self) -> &Path {
        &self.0
    }
}

impl Drop for OutputDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

/// A fresh output path under the temp dir; any stale run's leftovers are
/// cleared so `metadata.json` can only come from this run.
pub fn fresh_output_dir(tag: &str) -> OutputDir {
    let dir = std::env::temp_dir().join(format!(
        "agent-video-real-media-{tag}-{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    OutputDir(dir)
}

/// Read and parse the `metadata.json` a committed file-mode run wrote.
pub fn read_metadata(dir: &Path) -> wire::Metadata {
    let text = fs::read_to_string(dir.join("metadata.json"))
        .expect("metadata.json exists after a committed run");
    serde_json::from_str(&text).expect("metadata.json parses as wire::Metadata")
}

/// Parse a wire timestamp (`"MM:SS"` or `"H:MM:SS"`) into seconds.
pub fn parse_wire_ts(s: &str) -> f32 {
    s.split(':').fold(0.0, |acc, part| {
        acc * 60.0
            + part
                .parse::<f32>()
                .unwrap_or_else(|_| panic!("bad wire timestamp {s:?}"))
    })
}
