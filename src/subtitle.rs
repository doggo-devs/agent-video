//! Embedded-subtitle extraction via ffmpeg.
//!
//! Pulls the first subtitle stream as SRT to stdout. Videos without a subtitle
//! track yield `None` rather than an error — absence is normal.

use std::path::Path;
use std::process::Command;

/// Extract the first embedded subtitle track as SRT text.
///
/// Returns `None` when there is no subtitle stream or ffmpeg produces no
/// output. Errors (ffmpeg missing, spawn failure) propagate.
pub fn extract(input: &Path) -> anyhow::Result<Option<String>> {
    let out = Command::new("ffmpeg")
        .arg("-i")
        .arg(input)
        .args(["-map", "0:s:0", "-f", "srt", "pipe:1"])
        .output()?;

    if !out.status.success() {
        return Ok(None);
    }
    let text = String::from_utf8_lossy(&out.stdout).trim().to_string();
    if text.is_empty() {
        Ok(None)
    } else {
        Ok(Some(text))
    }
}
