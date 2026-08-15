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
use crate::schema::{Look, Strategy, TimeValue, VideoArgs, VideoBody, Want, wire_names};
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
    NAMES.get_or_init(|| property_names(&crate::schema::video_input_schema()))
}

fn property_names(schema: &serde_json::Value) -> Vec<String> {
    schema["properties"]
        .as_object()
        .expect("input schemas carry an object of properties by construction")
        .keys()
        .cloned()
        .collect()
}

/// Reject any key the published schema does not advertise.
fn reject_unknown(
    raw: &serde_json::Map<String, serde_json::Value>,
    valid: &[String],
) -> Result<(), Fault> {
    for key in raw.keys() {
        if !valid.iter().any(|name| name == key) {
            return Err(Fault::bad_input(format!(
                "unknown argument \"{key}\"; valid: {}",
                valid.join(", ")
            )));
        }
    }
    Ok(())
}

/// A non-empty string argument, or a `bad_input` naming the example form.
fn required_string(
    raw: &serde_json::Map<String, serde_json::Value>,
    key: &str,
    example: &str,
) -> Result<String, Fault> {
    match raw.get(key) {
        Some(serde_json::Value::String(s)) if !s.is_empty() => Ok(s.clone()),
        _ => Err(Fault::bad_input(format!(
            "{key} must be a non-empty string, e.g. {example}"
        ))),
    }
}

/// Coerce `video_confirm` arguments into the source and token it names.
pub(super) fn coerce_confirm_args(
    raw: &serde_json::Map<String, serde_json::Value>,
) -> Result<(String, String), Fault> {
    let valid = property_names(&crate::schema::confirm_input_schema());
    reject_unknown(raw, &valid)?;
    let source = required_string(raw, "source", r#"{"source": "/videos/talk.mp4"}"#)?;
    let token = required_string(raw, "token", r#"{"token": "t1"}"#)?;
    Ok((source, token))
}

pub(super) fn coerce_args(
    raw: &serde_json::Map<String, serde_json::Value>,
) -> Result<VideoArgs, Fault> {
    let valid = valid_arg_names();
    reject_unknown(raw, valid)?;
    let source = required_string(raw, "source", r#"{"source": "/videos/talk.mp4"}"#)?;
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
    let want = match raw.get("want") {
        None => None,
        Some(v) => Some(serde_json::from_value::<Want>(v.clone()).map_err(|_| {
            Fault::bad_input(format!(
                r#"want must be one of {}, e.g. {{"want": "transcript"}}"#,
                wire_names(&Want::ALL).join(", ")
            ))
        })?),
    };
    let raw_arg = string(
        "raw",
        r#"raw must be a string of core CLI arguments, e.g. {"raw": "process --strategy keyframes"}"#,
    )?;
    Ok(VideoArgs {
        source,
        from,
        until,
        look,
        strategy,
        want,
        raw: raw_arg,
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

/// Budgeted renderer for grid results.
///
/// [`Wire::ready`] is the only path that builds a grid answer: it drops
/// tail grids that would exceed the configured response ceiling and names
/// the `from` value that resumes the window.
pub(super) struct Wire {
    max_bytes: usize,
}

/// Charged once per response for the JSON-RPC envelope and the result
/// wrapper around the content blocks. A fixed allowance, not a
/// measurement: the exact framing belongs to the transport, and the
/// ceiling is a backstop with headroom rather than a tight fit.
const RESPONSE_OVERHEAD: usize = 4 * 1024;

/// Charged per image block for its JSON keys and quoting around the
/// base64 payload.
const IMAGE_BLOCK_OVERHEAD: usize = 128;

impl Wire {
    pub(super) fn new(cfg: &crate::config::Config) -> Self {
        Wire {
            max_bytes: usize::try_from(cfg.budget.max_response_bytes.get()).unwrap_or(usize::MAX),
        }
    }

    /// Render a document into a result that fits the ceiling.
    pub(super) fn ready(&self, mut meta: CoreMetadata, notices: &[String]) -> CallToolResult {
        let mut notices = notices.to_vec();
        // The text block is charged first and never dropped: it carries
        // the ranges and subtitles that make any returned image legible,
        // and a transcript can make it large enough to matter.
        let spent = describe_grids(&meta, &notices).len() + RESPONSE_OVERHEAD;
        let kept = self.fits(&meta.grids, spent);
        if kept < meta.grids.len() {
            let resumes_at = meta.grids[kept]
                .time_range
                .split('-')
                .next()
                .unwrap_or_default()
                .to_string();
            let dropped = meta.grids.len() - kept;
            let total = meta.grids.len();
            let ceiling = crate::plan::format_bytes(self.max_bytes as u64);
            notices.push(format!(
                "response budget: {dropped} of {total} grids did not fit the {ceiling} \
                 ceiling and were dropped. Call again with from: {resumes_at} for the rest, \
                 or use the CLI (`agent-video watch -o dir/`), which writes grids as files \
                 and has no such ceiling."
            ));
            meta.grids.truncate(kept);
        }
        ready_result(meta, &notices)
    }

    /// How many leading grids fit the ceiling, given what the response
    /// already spends. At least one always does: a lone grid past the
    /// budget is still better answered than withheld, and truncating to
    /// nothing would leave the caller no way forward.
    fn fits(&self, grids: &[agent_video_core::wire::Grid], already_spent: usize) -> usize {
        let mut spent = already_spent;
        for (i, grid) in grids.iter().enumerate() {
            let cost = match &grid.image {
                GridImage::Base64(b64) => b64.len() + IMAGE_BLOCK_OVERHEAD,
                GridImage::File(_) => 0,
            };
            spent = spent.saturating_add(cost);
            if spent > self.max_bytes {
                return i.max(1);
            }
        }
        grids.len()
    }
}

/// Render a parsed core document: one text block, then the grid images.
/// Takes the document by value so base64 payloads move instead of clone.
fn ready_result(meta: CoreMetadata, notices: &[String]) -> CallToolResult {
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

#[cfg(test)]
mod tests {
    use super::*;
    use agent_video_core::types::LayoutKind;
    use agent_video_core::wire::Grid;

    fn grid(range: &str, bytes: usize) -> Grid {
        Grid {
            image: GridImage::Base64("x".repeat(bytes)),
            time_range: range.to_string(),
            total_frames: 1,
            key_frames: Vec::new(),
            layout: LayoutKind::Uniform,
            subtitles: None,
        }
    }

    fn doc(grids: Vec<Grid>) -> CoreMetadata {
        let json = serde_json::json!({
            "source": "v.mp4",
            "duration": 600.0,
            "settings": { "max_edge": 2048, "quality": "low", "detail": "standard" },
            "grids": [],
        });
        let mut meta: CoreMetadata = serde_json::from_value(json).expect("document");
        meta.grids = grids;
        meta
    }

    fn images(result: &CallToolResult) -> usize {
        result
            .content
            .iter()
            .filter(|c| matches!(c, ContentBlock::Image(_)))
            .count()
    }

    fn text(result: &CallToolResult) -> String {
        result
            .content
            .iter()
            .filter_map(|c| match c {
                ContentBlock::Text(t) => Some(t.text.clone()),
                _ => None,
            })
            .collect()
    }

    fn wire_bytes(result: &CallToolResult) -> usize {
        serde_json::to_string(result)
            .expect("a result serializes")
            .len()
    }

    fn ceiling_for(n: usize) -> usize {
        RESPONSE_OVERHEAD + 4 * 1024 + n * (64 * 1024 + IMAGE_BLOCK_OVERHEAD)
    }

    #[test]
    fn passes_a_document_that_fits() {
        let wire = Wire {
            max_bytes: ceiling_for(2),
        };
        let r = wire.ready(
            doc(vec![
                grid("00:00-01:00", 64 * 1024),
                grid("01:00-02:00", 64 * 1024),
            ]),
            &[],
        );
        assert_eq!(images(&r), 2);
        assert!(!text(&r).contains("response budget"), "{}", text(&r));
    }

    #[test]
    fn drops_grids_past_the_ceiling_from_the_tail() {
        let wire = Wire {
            max_bytes: ceiling_for(2),
        };
        let r = wire.ready(
            doc(vec![
                grid("00:00-01:00", 64 * 1024),
                grid("01:00-02:00", 64 * 1024),
                grid("02:00-03:00", 64 * 1024),
            ]),
            &[],
        );
        assert_eq!(images(&r), 2, "the tail goes, the window's start stays");
    }

    #[test]
    fn keeps_the_whole_response_under_the_ceiling() {
        let wire = Wire {
            max_bytes: ceiling_for(2),
        };
        let r = wire.ready(
            doc(vec![
                grid("00:00-01:00", 64 * 1024),
                grid("01:00-02:00", 64 * 1024),
                grid("02:00-03:00", 64 * 1024),
            ]),
            &[],
        );
        assert!(
            wire_bytes(&r) <= ceiling_for(2),
            "the ceiling bounds the response, not just its images: {} > {}",
            wire_bytes(&r),
            ceiling_for(2)
        );
    }

    #[test]
    fn a_fat_text_block_is_charged_against_the_same_ceiling() {
        let wire = Wire {
            max_bytes: ceiling_for(2),
        };
        let mut grids = vec![
            grid("00:00-01:00", 64 * 1024),
            grid("01:00-02:00", 64 * 1024),
        ];
        grids[0].subtitles = Some("x".repeat(64 * 1024));
        let r = wire.ready(doc(grids), &[]);
        assert_eq!(
            images(&r),
            1,
            "subtitles spend the budget the second image wanted"
        );
    }

    #[test]
    fn names_the_call_that_resumes_from_the_first_dropped_grid() {
        let wire = Wire {
            max_bytes: ceiling_for(1),
        };
        let r = wire.ready(
            doc(vec![
                grid("00:00-01:00", 64 * 1024),
                grid("01:00-02:00", 64 * 1024),
            ]),
            &[],
        );
        let text = text(&r);
        assert!(text.contains("from: 01:00"), "{text}");
    }

    #[test]
    fn answers_one_oversized_grid_rather_than_withholding_everything() {
        let wire = Wire { max_bytes: 10 };
        let r = wire.ready(doc(vec![grid("00:00-01:00", 64 * 1024)]), &[]);
        assert_eq!(
            images(&r),
            1,
            "a lone grid past the budget beats an answer with no way forward"
        );
    }
}
