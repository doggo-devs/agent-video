// SPDX-License-Identifier: Apache-2.0
//! Single source of truth for the `video` tool.
//!
//! Defines the argument types, the input schema handed to MCP `tools/list`
//! (hand-written JSON, byte-compatible with the TS harness it replaces),
//! and the result family serialized back to the model.

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use super::time::ArgError;

/// Intent-level presets; mapped to core parameters in `look.rs`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Look {
    Overview,
    Detailed,
    ReadText,
    TrackMotion,
}

pub const LOOKS: [&str; 4] = ["overview", "detailed", "read_text", "track_motion"];

/// Temporal sampling choice: which moments become frames. Orthogonal to
/// [`Look`], which controls frame size/density within the chosen moments.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Strategy {
    /// Keyframes plus fill in sparse stretches; the default.
    Adaptive,
    /// Only I-frames: cheapest scan, blind inside static shots.
    Keyframes,
    /// Evenly spaced in time; for content where keyframes mislead.
    Uniform,
}

pub const STRATEGIES: [&str; 3] = ["adaptive", "keyframes", "uniform"];

impl Strategy {
    /// The core `--strategy` value, or `None` for the lenient default.
    ///
    /// `Adaptive` deliberately maps to no flag: core honors an explicit
    /// `--strategy` strictly (too few keyframes becomes an error), while an
    /// omitted one degrades gracefully. A model saying "adaptive" means "the
    /// default is fine", not "fail rather than degrade" — strictness belongs
    /// to `raw`. `Uniform` is the model-facing name for core's `fixed-fps`:
    /// the fps knob itself is not exposed here.
    pub fn core_flag(self) -> Option<&'static str> {
        match self {
            Strategy::Adaptive => None,
            Strategy::Keyframes => Some("keyframes"),
            Strategy::Uniform => Some("fixed-fps"),
        }
    }
}

/// A timestamp argument as the model sends it: seconds or `"MM:SS"` text.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum TimeValue {
    Number(f64),
    Text(String),
}

/// Arguments for the `video` tool.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VideoArgs {
    /// Local file path or URL.
    pub source: String,
    /// Window start; omit = start of video.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub from: Option<TimeValue>,
    /// Window end; omit = end of video.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub until: Option<TimeValue>,
    /// How to look at the window; omit = balanced default.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub look: Option<Look>,
    /// Which moments to sample; omit = lenient adaptive default.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub strategy: Option<Strategy>,
    /// Expert escape hatch: verbatim core CLI arguments.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub raw: Option<String>,
    /// Echo of a consent token issued by a previous response.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub confirm: Option<String>,
    /// `true` = generate (or reuse) a whisper transcript for the source.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub transcribe: Option<bool>,
}

impl VideoArgs {
    /// A bare invocation carrying only the source.
    pub fn bare(source: &str) -> Self {
        VideoArgs {
            source: source.to_string(),
            from: None,
            until: None,
            look: None,
            strategy: None,
            raw: None,
            confirm: None,
            transcribe: None,
        }
    }

    /// Refine the loose wire struct into its one valid reading. This is the
    /// single place the exclusivity rules of `docs/protocol.md` are
    /// enforced (`confirm` alone, `transcribe` alone, `raw` without grid
    /// intent); every consumer matches on the result instead of
    /// re-checking. `transcribe: false` reads as absent. The error message
    /// goes to the model verbatim.
    pub fn intent(&self) -> Result<Intent<'_>, ArgError> {
        let grid = GridSpec {
            from: self.from.as_ref(),
            until: self.until.as_ref(),
            look: self.look,
            strategy: self.strategy,
        };
        let transcribe = self.transcribe == Some(true);
        if let Some(token) = &self.confirm {
            if self.raw.is_some() || transcribe || !grid.is_bare() {
                return Err(ArgError(CONFIRM_ALONE.to_string()));
            }
            return Ok(Intent::Confirm(token));
        }
        if transcribe {
            if self.raw.is_some() || !grid.is_bare() {
                return Err(ArgError(TRANSCRIBE_ALONE.to_string()));
            }
            return Ok(Intent::Transcribe);
        }
        if let Some(raw) = &self.raw {
            if !grid.is_bare() {
                return Err(ArgError(RAW_EXCLUSIVE.to_string()));
            }
            return Ok(Intent::Raw(raw));
        }
        Ok(Intent::Grid(grid))
    }
}

/// Rejection for `confirm` combined with anything else; also the story
/// [`crate::plan`] tells when a confirm reaches it out of band.
pub const CONFIRM_ALONE: &str = "confirm must be passed alone (with source): it authorizes \
                                 exactly the pending plan, nothing else";

/// Rejection for `transcribe` combined with `raw` or grid intent.
pub const TRANSCRIBE_ALONE: &str = "transcribe must be passed alone (with source); once the transcript exists it rides along \
     with every grid call automatically";

/// Rejection for `raw` combined with grid intent.
pub const RAW_EXCLUSIVE: &str = "raw is mutually exclusive with from/until/look/strategy: raw \
                                 callers own the strategy entirely";

/// The four mutually exclusive modes of a `video` call, refined from
/// [`VideoArgs`] by [`VideoArgs::intent`] — a combination that mixes modes
/// is not constructible here.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Intent<'a> {
    /// `confirm` alone: redeem a pending consent token.
    Confirm(&'a str),
    /// `transcribe: true` alone: generate (or reuse) a whisper transcript.
    Transcribe,
    /// `raw` alone: verbatim core CLI passthrough.
    Raw(&'a str),
    /// The grid path; every field optional, a bare source when all are.
    Grid(GridSpec<'a>),
}

/// The grid-path arguments of one call; all optional.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct GridSpec<'a> {
    pub from: Option<&'a TimeValue>,
    pub until: Option<&'a TimeValue>,
    pub look: Option<Look>,
    pub strategy: Option<Strategy>,
}

impl GridSpec<'_> {
    /// Whether no field carries model intent — a bare source.
    pub fn is_bare(&self) -> bool {
        self.from.is_none()
            && self.until.is_none()
            && self.look.is_none()
            && self.strategy.is_none()
    }
}

/// Single-line tool description shown in the tool list.
pub const VIDEO_TOOL_DESCRIPTION: &str = "Watch a video (local path or URL) as VLM-readable frame grids with timestamps and subtitles, or transcribe its audio with `transcribe: true`. Call with just `source` first: short videos run immediately, everything else answers with metadata and copyable next calls.";

/// JSON Schema for MCP `tools/list`, kept in sync with [`VideoArgs`].
pub fn video_input_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "source": {
                "type": "string",
                "description": "Local video file path or URL. Call with only `source` first; the response tells you what to do next."
            },
            "from": {
                "anyOf": [{ "type": "string" }, { "type": "number" }],
                "description": "Window start: seconds (83) or timestamp (\"1:23\", \"1:02:03\"). Omit to start from the beginning."
            },
            "until": {
                "anyOf": [{ "type": "string" }, { "type": "number" }],
                "description": "Window end, same formats. Omit to go to the end of the video."
            },
            "look": {
                "type": "string",
                "enum": LOOKS,
                "description": "How to look: overview (fast triage), detailed (miss nothing), read_text (few large frames to read on-screen text/UI), track_motion (dense small frames for fast action). Subtitle text always rides along in metadata regardless; this only affects frame size/density. Omit for a balanced default."
            },
            "strategy": {
                "type": "string",
                "enum": STRATEGIES,
                "description": "Which moments become frames: adaptive (default: keyframes plus fill in sparse stretches, right for most footage), keyframes (cheapest scan, frames land on cuts — triage for edited footage, blind to changes inside static shots), uniform (evenly spaced in time — use when keyframes mislead: screen recordings, slides, long static takes). Orthogonal to look, which sets frame size/density."
            },
            "raw": {
                "type": "string",
                "description": "Expert escape hatch: verbatim agent-video core CLI arguments starting with the subcommand, e.g. \"process --strategy keyframes --base-fps 0.2\". Mutually exclusive with from/until/look/strategy. Input/output plumbing is injected by the server."
            },
            "confirm": {
                "type": "string",
                "description": "Echo a consent token issued by a previous response to authorize the pending action. Tokens stay valid until a newer plan for the same source replaces them; pass confirm alone with source."
            },
            "transcribe": {
                "type": "boolean",
                "description": "Set true (alone with source) to generate a whisper transcript of the audio; it is cached and rides along with every grid afterwards. Needs local media: for a URL, download it first (bare call, then confirm), then repeat with transcribe."
            }
        },
        "required": ["source"],
        "additionalProperties": false
    })
}

// ── result family ──

/// Probe summary shown to the model in a `plan` response. Field names are the
/// wire contract (`sizeBytes` stays camelCase for TS-harness compatibility).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProbeSummary {
    /// Seconds, centisecond precision — probes round before emitting so
    /// f32-widening noise never reaches the model.
    pub duration: f64,
    pub width: u32,
    pub height: u32,
    /// `None` = unknown (URL probes without Content-Length, stat failures);
    /// absent on the wire, never a fake `0`.
    #[serde(rename = "sizeBytes", default, skip_serializing_if = "Option::is_none")]
    pub size_bytes: Option<u64>,
    /// Short container label (`mp4`, `mkv`), never libav's demuxer alias
    /// list; see `crate::probe::container_label`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub container: Option<String>,
    /// Human-readable descriptions, e.g. `embedded #0 (eng, subrip)`.
    pub subtitles: Vec<String>,
}

/// Argument object the model can copy verbatim, with a rationale.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RecommendedCall {
    pub call: VideoArgs,
    pub note: String,
}

/// Consent token sent to the model, spelling out its effect.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OfferedToken {
    pub token: String,
    pub effect: String,
}

/// JSON-body responses of the `video` tool (grids are rendered separately).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum VideoBody {
    /// Probe-only response: summary plus ready-to-paste follow-up calls.
    Plan {
        probe: ProbeSummary,
        guidance: String,
        recommended_calls: Vec<RecommendedCall>,
    },
    /// Request requires explicit consent; retry with `confirm`.
    ConfirmRequired {
        consequence: String,
        tokens: Vec<OfferedToken>,
    },
    /// A download continues in the background; poll with the same source.
    Pending { message: String },
    /// The call failed; `error` is the prose, `kind` the machine-readable
    /// class (`bad_input`: fix the arguments; `environment`: this
    /// installation lacks something — tell the user; `transient`: retrying
    /// may succeed; `internal`: an agent-video bug).
    Failed {
        error: String,
        kind: crate::fault::FaultKind,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deserializes_minimal_args() {
        let args: VideoArgs = serde_json::from_str(r#"{"source":"/v.mp4"}"#).expect("valid");
        assert_eq!(args.source, "/v.mp4");
        assert!(
            matches!(args.intent(), Ok(Intent::Grid(g)) if g.is_bare()),
            "a bare source refines to a bare grid"
        );
    }

    #[test]
    fn accepts_number_and_text_timestamps() {
        let args: VideoArgs =
            serde_json::from_str(r#"{"source":"v","from":83,"until":"1:23"}"#).expect("valid");
        assert_eq!(args.from, Some(TimeValue::Number(83.0)));
        assert_eq!(args.until, Some(TimeValue::Text("1:23".into())));
    }

    #[test]
    fn rejects_unknown_look() {
        assert!(serde_json::from_str::<VideoArgs>(r#"{"source":"v","look":"zoom"}"#).is_err());
    }

    #[test]
    fn rejects_unknown_strategy() {
        assert!(
            serde_json::from_str::<VideoArgs>(r#"{"source":"v","strategy":"fixed-fps"}"#).is_err(),
            "core spelling is not the wire spelling"
        );
    }

    #[test]
    fn strategy_counts_as_intent() {
        let args: VideoArgs =
            serde_json::from_str(r#"{"source":"v","strategy":"uniform"}"#).expect("valid");
        assert!(matches!(args.intent(), Ok(Intent::Grid(g)) if !g.is_bare()));
    }

    fn expect_rejection(args: &VideoArgs, message: &str) {
        assert_eq!(args.intent(), Err(ArgError(message.to_string())));
    }

    #[test]
    fn rejects_confirm_combined_with_grid_intent() {
        let args = VideoArgs {
            confirm: Some("t".into()),
            look: Some(Look::Overview),
            ..VideoArgs::bare("v")
        };
        expect_rejection(
            &args,
            "confirm must be passed alone (with source): it authorizes exactly the pending \
             plan, nothing else",
        );
    }

    #[test]
    fn rejects_confirm_combined_with_raw() {
        let args = VideoArgs {
            confirm: Some("t".into()),
            raw: Some("process".into()),
            ..VideoArgs::bare("v")
        };
        expect_rejection(&args, CONFIRM_ALONE);
    }

    #[test]
    fn rejects_confirm_combined_with_transcribe() {
        let args = VideoArgs {
            confirm: Some("t".into()),
            transcribe: Some(true),
            ..VideoArgs::bare("v")
        };
        expect_rejection(&args, CONFIRM_ALONE);
    }

    #[test]
    fn rejects_transcribe_combined_with_grid_intent() {
        let args = VideoArgs {
            transcribe: Some(true),
            from: Some(TimeValue::Number(3.0)),
            ..VideoArgs::bare("v")
        };
        expect_rejection(
            &args,
            "transcribe must be passed alone (with source); once the transcript exists it \
             rides along with every grid call automatically",
        );
    }

    #[test]
    fn rejects_transcribe_combined_with_raw() {
        let args = VideoArgs {
            transcribe: Some(true),
            raw: Some("process".into()),
            ..VideoArgs::bare("v")
        };
        expect_rejection(&args, TRANSCRIBE_ALONE);
    }

    #[test]
    fn rejects_raw_combined_with_grid_intent() {
        let args = VideoArgs {
            raw: Some("process".into()),
            strategy: Some(Strategy::Uniform),
            ..VideoArgs::bare("v")
        };
        expect_rejection(
            &args,
            "raw is mutually exclusive with from/until/look/strategy: raw callers own the \
             strategy entirely",
        );
    }

    #[test]
    fn refines_each_lone_mode_to_its_intent() {
        let confirm = VideoArgs {
            confirm: Some("tok".into()),
            ..VideoArgs::bare("v")
        };
        assert_eq!(confirm.intent(), Ok(Intent::Confirm("tok")));
        let transcribe = VideoArgs {
            transcribe: Some(true),
            ..VideoArgs::bare("v")
        };
        assert_eq!(transcribe.intent(), Ok(Intent::Transcribe));
        let raw = VideoArgs {
            raw: Some("extract --range 1-2".into()),
            ..VideoArgs::bare("v")
        };
        assert_eq!(raw.intent(), Ok(Intent::Raw("extract --range 1-2")));
    }

    #[test]
    fn reads_transcribe_false_as_absent() {
        let args = VideoArgs {
            transcribe: Some(false),
            look: Some(Look::Overview),
            ..VideoArgs::bare("v")
        };
        assert!(
            matches!(args.intent(), Ok(Intent::Grid(g)) if !g.is_bare()),
            "transcribe: false must not claim the call"
        );
    }

    #[test]
    fn adaptive_maps_to_no_flag() {
        assert_eq!(Strategy::Adaptive.core_flag(), None, "lenient default");
        assert_eq!(Strategy::Keyframes.core_flag(), Some("keyframes"));
        assert_eq!(Strategy::Uniform.core_flag(), Some("fixed-fps"));
    }

    #[test]
    fn serializes_status_tags_snake_case() {
        let body = VideoBody::ConfirmRequired {
            consequence: "big".into(),
            tokens: vec![],
        };
        let v = serde_json::to_value(&body).expect("serializable");
        assert_eq!(v["status"], "confirm_required");
    }

    #[test]
    fn probe_summary_keeps_ts_wire_names() {
        let p = ProbeSummary {
            duration: 1.0,
            width: 2,
            height: 3,
            size_bytes: Some(4),
            container: None,
            subtitles: vec![],
        };
        let v = serde_json::to_value(&p).expect("serializable");
        assert!(v.get("sizeBytes").is_some(), "camelCase on the wire");

        let unknown = ProbeSummary {
            size_bytes: None,
            ..p
        };
        let v = serde_json::to_value(&unknown).expect("serializable");
        assert!(
            v.get("sizeBytes").is_none(),
            "unknown size is absent, never a fake 0"
        );
    }

    /// Golden file: `surfaces/video.schema.json` feeds the surface generator
    /// (`scripts/gen-surfaces.ts`), which runs without a Rust toolchain. This
    /// test keeps the pinned copy in lockstep with the schema here;
    /// regenerate with `UPDATE_SURFACES=1 cargo test`, then
    /// `bun run gen:surfaces`.
    #[test]
    fn pinned_surface_schema_is_current() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../surfaces/video.schema.json");
        let expected = json!({
            "description": VIDEO_TOOL_DESCRIPTION,
            "inputSchema": video_input_schema(),
        });
        if std::env::var_os("UPDATE_SURFACES").is_some() {
            let doc = serde_json::to_string_pretty(&expected).expect("serializable");
            std::fs::write(&path, doc + "\n").expect("write surfaces/video.schema.json");
        }
        let pinned: Value = serde_json::from_str(
            &std::fs::read_to_string(&path)
                .expect("surfaces/video.schema.json exists — UPDATE_SURFACES=1 cargo test"),
        )
        .expect("pinned schema is valid JSON");
        assert_eq!(
            pinned, expected,
            "surfaces/video.schema.json is stale — regenerate with UPDATE_SURFACES=1 cargo \
             test, then bun run gen:surfaces"
        );
    }

    #[test]
    fn schema_lists_all_looks() {
        let schema = video_input_schema();
        assert_eq!(
            schema["properties"]["look"]["enum"],
            json!(["overview", "detailed", "read_text", "track_motion"])
        );
        assert_eq!(schema["required"], json!(["source"]));
    }

    #[test]
    fn schema_lists_all_strategies() {
        let schema = video_input_schema();
        assert_eq!(
            schema["properties"]["strategy"]["enum"],
            json!(["adaptive", "keyframes", "uniform"])
        );
    }
}
