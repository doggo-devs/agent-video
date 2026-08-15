// SPDX-License-Identifier: Apache-2.0
//! Single source of truth for the `video` tool.
//!
//! Defines the argument types, the input schema handed to MCP `tools/list`
//! (hand-written JSON, byte-compatible with the TS harness it replaces),
//! and the result family serialized back to the model.

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use super::time::ArgError;

/// Which view of the source a call asks for. The discriminant of the
/// argument sum type: `from`/`until` window every view, `look`/`strategy`
/// belong to [`Want::Grids`] alone.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Want {
    /// Frame grids over the window.
    #[default]
    Grids,
    /// The spoken text over the window, as `MM:SS text` cue lines.
    Transcript,
    /// Probe metadata and follow-up calls; decodes nothing.
    Plan,
}

impl Want {
    /// Variant count, witnessed by an exhaustive match; same drift guard as
    /// [`Look::COUNT`].
    const COUNT: usize = match Want::Grids {
        Want::Grids | Want::Transcript | Want::Plan => 3,
    };

    /// Every view, in the order the schema advertises them.
    pub const ALL: [Want; Self::COUNT] = [Want::Grids, Want::Transcript, Want::Plan];
}

/// Intent-level presets; mapped to core parameters in `look.rs`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Look {
    Overview,
    Detailed,
    ReadText,
    TrackMotion,
}

impl Look {
    /// Variant count, witnessed by an exhaustive match: a new look fails to
    /// compile here until it is counted, and [`Look::ALL`]'s array type then
    /// refuses to build until the variant joins it — the advertised schema
    /// cannot silently fork from the accepted values.
    const COUNT: usize = match Look::Overview {
        Look::Overview | Look::Detailed | Look::ReadText | Look::TrackMotion => 4,
    };

    /// Every look, in the order the schema advertises them.
    pub const ALL: [Look; Self::COUNT] = [
        Look::Overview,
        Look::Detailed,
        Look::ReadText,
        Look::TrackMotion,
    ];
}

/// The serde wire names of `all`, read back through serialization: the
/// `rename_all` attribute stays the single truth for what is advertised
/// (`tools/list`), what is accepted (deserialization), and what error
/// messages list.
pub fn wire_names<T: Serialize>(all: &[T]) -> Vec<String> {
    all.iter()
        .map(|v| {
            serde_json::to_value(v)
                .expect("wire enums serialize")
                .as_str()
                .expect("wire enums serialize to strings")
                .to_string()
        })
        .collect()
}

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

impl Strategy {
    /// Variant count, witnessed by an exhaustive match — see [`Look::COUNT`].
    const COUNT: usize = match Strategy::Adaptive {
        Strategy::Adaptive | Strategy::Keyframes | Strategy::Uniform => 3,
    };

    /// Every strategy, in the order the schema advertises them.
    pub const ALL: [Strategy; Self::COUNT] =
        [Strategy::Adaptive, Strategy::Keyframes, Strategy::Uniform];
}

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
    /// Which view to return; omit = grids.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub want: Option<Want>,
    /// Expert escape hatch: verbatim core CLI arguments.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub raw: Option<String>,
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
            want: None,
            raw: None,
        }
    }

    /// Refine the loose wire struct into its one valid reading.
    ///
    /// `want` names the view; the window rides every view, so only `raw` —
    /// which owns core's whole argument tail — still excludes anything. The
    /// error message goes to the model verbatim.
    pub fn intent(&self) -> Result<Intent<'_>, ArgError> {
        let window = WindowSpec {
            from: self.from.as_ref(),
            until: self.until.as_ref(),
        };
        if let Some(raw) = &self.raw {
            if self.want.is_some()
                || self.look.is_some()
                || self.strategy.is_some()
                || !window.is_bare()
            {
                return Err(ArgError(RAW_EXCLUSIVE.to_string()));
            }
            return Ok(Intent::Raw(raw));
        }
        Ok(match self.want.unwrap_or_default() {
            Want::Transcript => Intent::Transcript(window),
            Want::Plan => Intent::Plan,
            Want::Grids => Intent::Grid(GridSpec {
                window,
                look: self.look,
                strategy: self.strategy,
            }),
        })
    }
}

/// Rejection for `raw` combined with any other argument.
pub const RAW_EXCLUSIVE: &str = "raw is mutually exclusive with want/from/until/look/strategy: \
                                 raw callers own core's arguments entirely";

/// The one valid reading of a `video` call, refined from [`VideoArgs`] by
/// [`VideoArgs::intent`] — a combination that mixes views is not
/// constructible here.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Intent<'a> {
    /// Frame grids over the window.
    Grid(GridSpec<'a>),
    /// The spoken text over the window.
    Transcript(WindowSpec<'a>),
    /// Probe metadata only; decodes nothing.
    Plan,
    /// Verbatim core CLI passthrough.
    Raw(&'a str),
}

/// The window every view shares; absent bounds mean the whole source.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct WindowSpec<'a> {
    pub from: Option<&'a TimeValue>,
    pub until: Option<&'a TimeValue>,
}

impl WindowSpec<'_> {
    /// Whether the window is the whole source.
    pub fn is_bare(&self) -> bool {
        self.from.is_none() && self.until.is_none()
    }
}

/// The grid view's arguments; all optional.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct GridSpec<'a> {
    pub window: WindowSpec<'a>,
    pub look: Option<Look>,
    pub strategy: Option<Strategy>,
}

impl GridSpec<'_> {
    /// Whether no field carries model intent.
    pub fn is_bare(&self) -> bool {
        self.window.is_bare() && self.look.is_none() && self.strategy.is_none()
    }
}

/// Single-line tool description shown in the tool list.
pub const VIDEO_TOOL_DESCRIPTION: &str = "Watch a video (local path or URL): `want` picks the view \u{2014} grids of timestamped frames (default), the spoken transcript, or probe metadata alone. `from`/`until` window any of them. Local files render on the spot; a URL answers with download options to confirm first.";

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
                "enum": Look::ALL,
                "description": "How to look: overview (fast triage), detailed (miss nothing), read_text (few large frames to read on-screen text/UI), track_motion (dense small frames for fast action). Subtitle text always rides along in metadata regardless; this only affects frame size/density. Omit for a balanced default."
            },
            "strategy": {
                "type": "string",
                "enum": Strategy::ALL,
                "description": "Which moments become frames: adaptive (default: keyframes plus fill in sparse stretches, right for most footage), keyframes (cheapest scan, frames land on cuts — triage for edited footage, blind to changes inside static shots), uniform (evenly spaced in time — use when keyframes mislead: screen recordings, slides, long static takes). Orthogonal to look, which sets frame size/density."
            },
            "want": {
                "type": "string",
                "enum": Want::ALL,
                "description": "Which view to return over the window: grids (default \u{2014} frame grids you read directly), transcript (the spoken text as \"MM:SS text\" lines; whisper runs once and the result then rides along with every grid), plan (probe metadata and follow-up calls, decoding nothing). For a long video, transcript first is the cheapest way to find what is worth looking at."
            },
            "raw": {
                "type": "string",
                "description": "Expert escape hatch: verbatim agent-video core CLI arguments starting with the subcommand, e.g. \"process --strategy keyframes --base-fps 0.2\". Mutually exclusive with every other argument. Input/output plumbing is injected by the server."
            }
        },
        "required": ["source"],
        "additionalProperties": false
    })
}

/// Single-line description of the consent tool.
pub const CONFIRM_TOOL_DESCRIPTION: &str = "Authorize one pending action a `video` response asked for. Pass the source it concerned and one token copied from that response's `tokens`.";

/// JSON Schema for the `video_confirm` tool.
///
/// Consent is its own tool because redeeming a token is a different act
/// from asking for a view: the tool name carries the discriminant, so the
/// host's tool dispatch enforces what prose in a shared argument bag only
/// ever asked for.
pub fn confirm_input_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "source": {
                "type": "string",
                "description": "The exact source the pending action concerned."
            },
            "token": {
                "type": "string",
                "description": "One token copied verbatim from a `confirm_required` response. Tokens stay valid until a newer plan for the same source replaces them."
            }
        },
        "required": ["source", "token"],
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
    /// The product of `transcribe: true`: the transcript itself, whole, as
    /// `MM:SS text` cue lines. It is also cached and rides along with every
    /// grid afterwards, sliced per grid.
    Transcript {
        transcript: String,
        guidance: String,
        recommended_calls: Vec<RecommendedCall>,
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
    fn rejects_raw_combined_with_grid_intent() {
        let args = VideoArgs {
            raw: Some("process".into()),
            strategy: Some(Strategy::Uniform),
            ..VideoArgs::bare("v")
        };
        expect_rejection(
            &args,
            "raw is mutually exclusive with want/from/until/look/strategy: raw callers own \
             core's arguments entirely",
        );
    }

    #[test]
    fn want_picks_the_view() {
        let args = |want| VideoArgs {
            want: Some(want),
            ..VideoArgs::bare("v")
        };
        assert_eq!(args(Want::Plan).intent(), Ok(Intent::Plan));
        assert!(matches!(
            args(Want::Transcript).intent(),
            Ok(Intent::Transcript(_))
        ));
        assert!(matches!(args(Want::Grids).intent(), Ok(Intent::Grid(_))));
    }

    #[test]
    fn omitted_want_reads_as_grids() {
        assert!(matches!(
            VideoArgs::bare("v").intent(),
            Ok(Intent::Grid(g)) if g.is_bare()
        ));
    }

    #[test]
    fn transcript_carries_the_window() {
        let args = VideoArgs {
            want: Some(Want::Transcript),
            from: Some(TimeValue::Text("1:00".into())),
            until: Some(TimeValue::Number(120.0)),
            ..VideoArgs::bare("v")
        };
        let Ok(Intent::Transcript(window)) = args.intent() else {
            panic!("expected a transcript view, got {:?}", args.intent());
        };
        assert_eq!(window.from, Some(&TimeValue::Text("1:00".into())));
        assert_eq!(window.until, Some(&TimeValue::Number(120.0)));
    }

    #[test]
    fn raw_refines_to_a_passthrough() {
        let raw = VideoArgs {
            raw: Some("extract --range 1-2".into()),
            ..VideoArgs::bare("v")
        };
        assert_eq!(raw.intent(), Ok(Intent::Raw("extract --range 1-2")));
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
        // The golden copy is the whole tool set: a second tool that no
        // surface knows about is the drift this test exists to catch.
        let expected = json!({
            "description": VIDEO_TOOL_DESCRIPTION,
            "inputSchema": video_input_schema(),
            "tools": [
                {
                    "name": "video",
                    "description": VIDEO_TOOL_DESCRIPTION,
                    "inputSchema": video_input_schema(),
                },
                {
                    "name": "video_confirm",
                    "description": CONFIRM_TOOL_DESCRIPTION,
                    "inputSchema": confirm_input_schema(),
                },
            ],
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
