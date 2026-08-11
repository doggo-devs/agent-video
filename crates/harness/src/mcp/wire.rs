// SPDX-License-Identifier: Apache-2.0
//! Wire shapes of one `video` call: argument coercion on the way in, result
//! rendering on the way out.
//!
//! The core JSON is read through [`agent_video_core::wire`] — the same types
//! the child process serialized, so producer and consumer cannot drift.

use std::time::Duration;

use rmcp::model::{CallToolResult, ContentBlock};

use super::download::Pace;
use crate::fault::Fault;
use crate::plan::format_bytes;
use crate::schema::{Look, Strategy, TimeValue, VideoArgs, VideoBody, wire_names};
use agent_video_core::types::{GridImage, format_ts};
use agent_video_core::wire::Coverage;

/// The document core's `--json` mode emits; one definition for both sides.
pub(super) use agent_video_core::wire::Metadata as CoreMetadata;

// ── argument coercion ──
// Clients do not reliably enforce additionalProperties; coerce defensively.

/// Argument names accepted by `coerce_args`, derived from the published
/// input schema — the whitelist cannot drift from what `tools/list`
/// advertises, because it *is* what `tools/list` advertises.
fn valid_arg_names() -> &'static [String] {
    use std::sync::OnceLock;
    static NAMES: OnceLock<Vec<String>> = OnceLock::new();
    NAMES.get_or_init(|| {
        crate::schema::video_input_schema()["properties"]
            .as_object()
            .expect("video_input_schema properties is an object by construction")
            .keys()
            .cloned()
            .collect()
    })
}

pub(super) fn coerce_args(
    raw: &serde_json::Map<String, serde_json::Value>,
) -> Result<VideoArgs, Fault> {
    let valid = valid_arg_names();
    for key in raw.keys() {
        if !valid.iter().any(|name| name == key) {
            return Err(Fault::bad_input(format!(
                "unknown argument \"{key}\"; valid: {}",
                valid.join(", ")
            )));
        }
    }
    let source = match raw.get("source") {
        Some(serde_json::Value::String(s)) if !s.is_empty() => s.clone(),
        _ => {
            return Err(Fault::bad_input(
                r#"source must be a non-empty string, e.g. {"source": "/videos/talk.mp4"}"#,
            ));
        }
    };
    let time = |key: &str, example: &str| -> Result<Option<TimeValue>, Fault> {
        match raw.get(key) {
            None => Ok(None),
            Some(serde_json::Value::String(s)) => Ok(Some(TimeValue::Text(s.clone()))),
            Some(v) if v.is_number() => Ok(Some(TimeValue::Number(
                v.as_f64()
                    .expect("serde_json numbers are f64-representable"),
            ))),
            Some(_) => Err(Fault::bad_input(format!(
                "{key} must be seconds or a timestamp string, e.g. {example}"
            ))),
        }
    };
    let from = time("from", r#"{"from": "1:23"}"#)?;
    let until = time("until", r#"{"until": 300}"#)?;
    let look = match raw.get("look") {
        None => None,
        Some(v) => Some(serde_json::from_value::<Look>(v.clone()).map_err(|_| {
            Fault::bad_input(format!(
                r#"look must be one of {}, e.g. {{"look": "overview"}}"#,
                wire_names(&Look::ALL).join(", ")
            ))
        })?),
    };
    let strategy = match raw.get("strategy") {
        None => None,
        Some(v) => Some(serde_json::from_value::<Strategy>(v.clone()).map_err(|_| {
            Fault::bad_input(format!(
                r#"strategy must be one of {}, e.g. {{"strategy": "uniform"}}"#,
                wire_names(&Strategy::ALL).join(", ")
            ))
        })?),
    };
    let string = |key: &str, message: &str| -> Result<Option<String>, Fault> {
        match raw.get(key) {
            None => Ok(None),
            Some(serde_json::Value::String(s)) => Ok(Some(s.clone())),
            Some(_) => Err(Fault::bad_input(message)),
        }
    };
    let raw_arg = string(
        "raw",
        r#"raw must be a string of core CLI arguments, e.g. {"raw": "process --strategy keyframes"}"#,
    )?;
    let confirm = string(
        "confirm",
        "confirm must be a string token echoed from a previous response",
    )?;
    let transcribe = match raw.get("transcribe") {
        None => None,
        Some(serde_json::Value::Bool(b)) => Some(*b),
        Some(_) => {
            return Err(Fault::bad_input(
                r#"transcribe must be a boolean, e.g. {"transcribe": true}"#,
            ));
        }
    };
    Ok(VideoArgs {
        source,
        from,
        until,
        look,
        strategy,
        raw: raw_arg,
        confirm,
        transcribe,
    })
}

pub(super) fn is_url(source: &str) -> bool {
    let lower = source.to_ascii_lowercase();
    lower.starts_with("http://") || lower.starts_with("https://")
}

/// The scheme of a source that names a protocol this layer does not
/// accept — it parses as a protocol URL by core's classifier (one
/// definition, not a third), but is not an http(s) URL this server can
/// fetch. `None` for http(s) URLs and local paths. Without this check an
/// `rtsp://cam/1` source would fall into the local flow and be diagnosed
/// as a missing file, which is a lie about what the caller sent.
pub(super) fn foreign_scheme(source: &str) -> Option<&str> {
    use agent_video_core::source::Source;
    if is_url(source) {
        return None;
    }
    match Source::classify(source) {
        // The comma matters only for `subfile,…`: the split names the
        // scheme, not the option blob after it.
        Source::Remote(_) => source.split([':', ',']).next(),
        Source::Local(_) => None,
    }
}

// ── result rendering ──

/// The evicted-underfoot fault: a rip that resolved from the cache and
/// then vanished before it could be used — a concurrent process's
/// eviction, never the caller's arguments. Transient by the repo's own
/// taxonomy (retrying the same call, after a re-download, succeeds), and
/// the healing action rides in the prose. One constructor, so no call
/// site can misfile the race under another kind.
pub(super) fn evicted_rip() -> Fault {
    Fault::transient(
        "the cached rip for this URL is gone (evicted underfoot by a concurrent process). \
         Call again with just the URL to re-download; a finished transcript stays in the \
         shared cache and is adopted without re-running whisper.",
    )
}

pub(super) fn failed(fault: &Fault) -> CallToolResult {
    let body = VideoBody::Failed {
        error: fault.message.clone(),
        kind: fault.kind,
    };
    let text = serde_json::to_string(&body).expect("VideoBody serializes");
    CallToolResult::error(vec![ContentBlock::text(text)])
}

pub(super) fn json_result(body: &VideoBody) -> CallToolResult {
    let text = serde_json::to_string_pretty(body).expect("VideoBody serializes");
    CallToolResult::success(vec![ContentBlock::text(text)])
}

/// One text block for all grids: ranges, key frames, subtitles, notices.
fn describe_grids(meta: &CoreMetadata, notices: &[String]) -> String {
    let mut lines: Vec<String> = notices.iter().map(|n| format!("note: {n}")).collect();
    // The coverage view is the wire contract's discrimination rule made
    // type-level: an extract's `duration` is the segment's span, not the
    // video's total — present it as such.
    lines.push(match meta.coverage() {
        Coverage::Segment(seg) => format!(
            "{}: segment {}-{}, {} grid(s) below, read top-left to bottom-right.",
            meta.source,
            format_ts(seg.start()),
            format_ts(seg.end()),
            meta.grids.len()
        ),
        Coverage::Source(duration) => format!(
            "{}: {}, {} grid(s) below, read top-left to bottom-right.",
            meta.source,
            format_ts(duration),
            meta.grids.len()
        ),
    });
    for (i, grid) in meta.grids.iter().enumerate() {
        lines.push(String::new());
        lines.push(format!(
            "[grid {}/{}] {} · layout {} · {} frames",
            i + 1,
            meta.grids.len(),
            grid.time_range,
            grid.layout.as_str(),
            grid.total_frames
        ));
        if !grid.key_frames.is_empty() {
            let keys: Vec<String> = grid
                .key_frames
                .iter()
                .map(|k| format!("{} @{}", k.timestamp, k.grid_position.as_str()))
                .collect();
            lines.push(format!("  key frames (bordered): {}", keys.join(", ")));
        }
        if let Some(subs) = &grid.subtitles
            && !subs.is_empty()
        {
            lines.push("  subtitles:".to_string());
            for line in subs.split('\n') {
                lines.push(format!("    {line}"));
            }
        }
    }
    lines.join("\n")
}

/// Render a parsed core document: one text block, then the grid images.
/// Takes the document by value so base64 payloads move instead of clone.
pub(super) fn ready_result(meta: CoreMetadata, notices: &[String]) -> CallToolResult {
    let mut content = vec![ContentBlock::text(describe_grids(&meta, notices))];
    for grid in meta.grids {
        if let GridImage::Base64(b64) = grid.image {
            content.push(ContentBlock::image(b64, "image/jpeg"));
        }
    }
    CallToolResult::success(content)
}

pub(super) fn pending_result(pace: &Pace) -> CallToolResult {
    let mut head = format!("Download in progress: {}", format_bytes(pace.downloaded));
    if let Some(total) = pace.total.filter(|t| *t > 0) {
        let percent = pace.downloaded.min(total) * 100 / total;
        head += &format!(" / {} ({percent}%)", format_bytes(total));
    }
    if let Some(rate) = pace.rate {
        head += &format!(", ~{}/s", format_bytes(rate as u64));
        if let Some(left) = pace.eta() {
            head += &format!(", ~{} left", format_span(left));
        }
    }
    let waited = format_span(pace.waited);
    json_result(&VideoBody::Pending {
        message: format!(
            "{head}. This call already waited {waited} for it, and the transfer continues in \
             the background — nothing failed and nothing was lost. Call again with just the \
             same source: that call waits the same {waited} and returns the finished result \
             the moment it lands. No argument you can pass makes it faster, so a poll buys \
             only a fresh byte count — if the numbers above say the wait is long, report the \
             progress to the user or do other work first, and poll after."
        ),
    })
}

fn format_span(span: Duration) -> String {
    let secs = span.as_secs();
    match secs {
        0..60 => format!("{secs}s"),
        _ => format!("{}m{:02}s", secs / 60, secs % 60),
    }
}
