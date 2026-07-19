// SPDX-License-Identifier: Apache-2.0
//! Decision engine.
//!
//! Pure `(args, probe, config) → action`; consent state lives in the passed
//! [`Tokens`]. Grid count picks extract vs process; the budget ladder
//! gates on stretch and consent. Never panics on model input — invalid input
//! becomes a `Reject` with a model-presentable message.

use super::config::Config;
use super::look::preset_for;
use super::raw::{RawSubcommand, parse_raw};
use super::schema::{
    CONFIRM_ALONE, GridSpec, Intent, Look, ProbeSummary, RecommendedCall, Strategy, VideoArgs,
};
use super::time::{ArgError, Window, resolve_window};
use super::tokens::{DownloadPlan, OfferDraft, PendingAction, Redemption, Token, Tokens};
use agent_video_core::types::format_ts;

/// A fully-decided core run; the server injects source and I/O flags.
#[derive(Debug, Clone, PartialEq)]
pub struct CoreInvocation {
    pub subcommand: RawSubcommand,
    /// Strategy flags only; never contains the input path, `-o`, `--json`,
    /// or `--subtitle` (a typed field below).
    pub flags: Vec<String>,
    /// Explicit subtitle choice; `None` lets the server attach a cached
    /// whisper transcript.
    pub subtitle: Option<std::path::PathBuf>,
    /// Attention window for `process` (core clamp); `extract` encodes
    /// `--range` in flags. A [`Window`] is a valid sub-range of the video
    /// by construction, so an inverted or NaN window cannot ride here.
    pub window: Option<Window>,
    pub predicted_grids: u32,
}

/// Probe-only decision; also reused verbatim after analysis downloads.
#[derive(Debug, Clone, PartialEq)]
pub struct PlanDecision {
    pub probe: ProbeSummary,
    pub guidance: String,
    pub recommended_calls: Vec<RecommendedCall>,
}

/// What a `video` call does.
#[derive(Debug, Clone, PartialEq)]
pub enum Decision {
    Run {
        invocation: CoreInvocation,
        notices: Vec<String>,
    },
    Download {
        plan: DownloadPlan,
        /// The redeemed token, so a completed keep download can be recorded
        /// back into the registry ([`Tokens::fulfill`]) as a receipt.
        token: Token,
        notices: Vec<String>,
    },
    /// A keep token echoed after its download already completed: the file
    /// at `path` is the answer, no work left to authorize.
    Saved {
        path: std::path::PathBuf,
    },
    Plan(PlanDecision),
    /// Consent required: the drafts carry the model-facing effect lines
    /// and the actions to register — the caller commits them on render,
    /// and the registry mints the tokens ([`Tokens::commit`]).
    Confirm {
        consequence: String,
        offers: Vec<OfferDraft>,
    },
    Reject {
        message: String,
    },
}

/// Decide what a video call does; invalid input becomes `Reject`. Pure:
/// consent offers ride the returned decision, the caller commits them.
pub fn decide(args: &VideoArgs, probe: &ProbeSummary, cfg: &Config) -> Decision {
    match decide_inner(args, probe, cfg) {
        Ok(d) => d,
        Err(ArgError(message)) => Decision::Reject { message },
    }
}

/// Redeem a consent token: a lookup against the pending store, no probe
/// needed (a URL source may not exist locally until its download runs).
/// Only a pure confirm redeems; anything else is told to pass it alone
/// (the handler routes here exactly when `confirm` is present).
pub fn redeem(args: &VideoArgs, store: &Tokens) -> Decision {
    let token = match args.intent() {
        Ok(Intent::Confirm(token)) => Token::new(token),
        _ => {
            return Decision::Reject {
                message: CONFIRM_ALONE.to_string(),
            };
        }
    };
    let Some(redemption) = store.take(&args.source, &token) else {
        return Decision::Reject {
            message: format!(
                "token \"{token}\" is unknown or stale for this source. Tokens stay valid \
                 until a newer plan replaces them; call again without confirm to see the \
                 current plan and its tokens."
            ),
        };
    };
    let notices = vec![format!("confirmed: {token}")];
    match redemption {
        Redemption::Fresh(PendingAction::Core(invocation)) => Decision::Run {
            invocation,
            notices,
        },
        Redemption::Fresh(PendingAction::Download(plan)) => Decision::Download {
            plan,
            token,
            notices,
        },
        Redemption::Fulfilled(path) => Decision::Saved { path },
    }
}

fn decide_inner(
    args: &VideoArgs,
    probe: &ProbeSummary,
    cfg: &Config,
) -> Result<Decision, ArgError> {
    let grid = match args.intent()? {
        // The caller redeems consent before deciding; a confirm reaching
        // the decision engine — even alone — is out of band and told the
        // same story as redeem's, so the model sees one message either way.
        Intent::Confirm(_) => return Err(ArgError(CONFIRM_ALONE.to_string())),
        Intent::Raw(raw) => return decide_raw(raw, probe, cfg),
        // The caller routes transcription before deciding; a stray
        // transcribe carries no grid fields (`intent` guarantees it), so it
        // decides as a bare source.
        Intent::Transcribe => GridSpec::default(),
        Intent::Grid(grid) => grid,
    };

    let (window, mut notices) = resolve_window(grid.from, grid.until, probe.duration)?;
    let preset = preset_for(grid.look);
    let span = window.span();
    let natural = ((span / preset.chunk_seconds).ceil() as u32).max(1);

    // Bare source on anything non-trivial: plan first, decode later.
    if grid.is_bare() && natural > cfg.budget.auto_run.get() {
        return Ok(Decision::Plan(plan_response(&args.source, probe, cfg)));
    }

    let mut preset_flags: Vec<String> = preset.flags.iter().map(|s| s.to_string()).collect();
    if let Some(flag) = grid.strategy.and_then(Strategy::core_flag) {
        preset_flags.extend(["--strategy".to_string(), flag.to_string()]);
    }

    // An explicit strategy is a temporal-selection request extract's dense
    // range sampling would ignore, so it stays on the process path.
    if natural == 1 && grid.strategy.is_none() {
        let range = format!("{}-{}", window.start().floor(), window.end().ceil());
        let mut flags = preset_flags;
        flags.extend(["--range".to_string(), range]);
        return Ok(Decision::Run {
            invocation: CoreInvocation {
                subcommand: RawSubcommand::Extract,
                flags,
                subtitle: None,
                window: None,
                predicted_grids: 1,
            },
            notices,
        });
    }

    let max = cfg.budget.max.get();
    if natural <= max {
        return Ok(Decision::Run {
            invocation: process_invocation(preset_flags, window, natural),
            notices,
        });
    }

    // Over budget: stretch chunks so the window fits in budget.max grids.
    let stretched = (span / f64::from(max)).ceil();
    let mut flags = preset_flags;
    flags.extend(["--chunk-seconds".to_string(), format!("{stretched}")]);
    let invocation = process_invocation(flags, window, max);
    // The degradation floor (budget.max_stretch_seconds): past it, grids
    // are too thin to run without consent.
    if stretched <= f64::from(cfg.budget.max_stretch_seconds.get()) {
        notices.push(format!(
            "budget: {natural} grids at requested density exceeds the {max}-grid budget; each \
             grid now covers ~{stretched}s (degraded temporal resolution). Zoom into ranges \
             with from/until for full density."
        ));
        return Ok(Decision::Run {
            invocation,
            notices,
        });
    }

    // Even stretched, grids would be absurdly thin: require consent.
    Ok(Decision::Confirm {
        consequence: format!(
            "covering {} in {max} grids means each grid spans ~{} minutes — likely too thin \
             to be useful. Better: survey sub-windows with from/until (look=overview), or rely \
             on subtitles for long talks.",
            format_ts(span as f32),
            (stretched / 60.0).round()
        ),
        offers: vec![OfferDraft {
            effect: format!("run anyway: {max} grids, ~{stretched}s of video per grid"),
            action: PendingAction::Core(invocation),
        }],
    })
}

fn decide_raw(raw: &str, probe: &ProbeSummary, cfg: &Config) -> Result<Decision, ArgError> {
    let raw = parse_raw(raw)?;
    if raw.subcommand == RawSubcommand::Extract {
        return Ok(Decision::Run {
            invocation: CoreInvocation {
                subcommand: RawSubcommand::Extract,
                flags: raw.flags,
                subtitle: raw.subtitle,
                window: None,
                predicted_grids: 1,
            },
            notices: Vec::new(),
        });
    }
    let predicted = ((probe.duration / raw.chunk_seconds).ceil() as u32).max(1);
    let invocation = CoreInvocation {
        subcommand: RawSubcommand::Process,
        flags: raw.flags,
        subtitle: raw.subtitle,
        window: None,
        predicted_grids: predicted,
    };
    let max = cfg.budget.max.get();
    if predicted <= max {
        return Ok(Decision::Run {
            invocation,
            notices: Vec::new(),
        });
    }
    Ok(Decision::Confirm {
        consequence: format!(
            "this raw plan emits ~{predicted} grids over {} — far past the {max}-grid budget \
             and likely to flood your context.",
            format_ts(probe.duration as f32)
        ),
        offers: vec![OfferDraft {
            effect: format!("run anyway, emitting ~{predicted} grids"),
            action: PendingAction::Core(invocation),
        }],
    })
}

fn process_invocation(flags: Vec<String>, window: Window, predicted_grids: u32) -> CoreInvocation {
    CoreInvocation {
        subcommand: RawSubcommand::Process,
        flags,
        subtitle: None,
        window: (!window.is_full()).then_some(window),
        predicted_grids,
    }
}

/// Plan response for a bare source: probe summary plus verbatim next calls.
pub fn plan_response(source: &str, probe: &ProbeSummary, cfg: &Config) -> PlanDecision {
    let max = cfg.budget.max.get();
    let overview_grids = (probe.duration / preset_for(Some(Look::Overview)).chunk_seconds).ceil();
    let capped = (overview_grids as u32).min(max);
    let mid = probe.duration / 2.0;
    let zoom_from = format_ts((mid - 30.0).max(0.0) as f32);
    let zoom_until = format_ts((mid + 30.0).min(probe.duration) as f32);

    let subs = if probe.subtitles.is_empty() {
        String::new()
    } else {
        format!(
            " Subtitles available ({}) — they ride along with every grid.",
            probe.subtitles.join("; ")
        )
    };

    let overview_note = if overview_grids as u32 > max {
        format!(
            "full pass, degraded to {capped} grids (~{}s each)",
            (probe.duration / f64::from(max)).ceil()
        )
    } else {
        format!("full pass, ~{overview_grids} grids")
    };

    let recommended_calls = vec![
        RecommendedCall {
            call: VideoArgs {
                look: Some(Look::Overview),
                ..VideoArgs::bare(source)
            },
            note: overview_note,
        },
        RecommendedCall {
            call: VideoArgs {
                from: Some(super::schema::TimeValue::Text(zoom_from)),
                until: Some(super::schema::TimeValue::Text(zoom_until)),
                look: Some(Look::Detailed),
                ..VideoArgs::bare(source)
            },
            note: "zoom pattern: any ~1-minute window at full density — adjust from/until"
                .to_string(),
        },
    ];

    PlanDecision {
        probe: probe.clone(),
        guidance: format!(
            "{} video, {}x{}, {}.{subs} Pick a call below (copy the args verbatim), or answer \
             from metadata alone if that already satisfies the user.",
            format_ts(probe.duration as f32),
            probe.width,
            probe.height,
            size_label(probe.size_bytes)
        ),
        recommended_calls,
    }
}

/// [`format_bytes`] for possibly-unknown sizes.
pub fn size_label(n: Option<u64>) -> String {
    match n {
        Some(n) => format_bytes(n),
        None => "size unknown".to_string(),
    }
}

/// Human-readable byte sizes for consequence text: `1.2 GiB`, `34 MiB`,
/// `12 KiB`.
pub fn format_bytes(n: u64) -> String {
    const KIB: f64 = 1024.0;
    const MIB: f64 = 1024.0 * 1024.0;
    const GIB: f64 = 1024.0 * 1024.0 * 1024.0;
    let n = n as f64;
    if n >= GIB {
        format!("{:.1} GiB", n / GIB)
    } else if n >= MIB {
        format!("{:.0} MiB", n / MIB)
    } else {
        format!("{} KiB", (n / KIB).ceil())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{default_config, test_dirs};
    use crate::schema::TimeValue;
    use std::path::Path;

    fn cfg() -> Config {
        default_config(&test_dirs(Path::new("/home/u")))
    }

    fn probe(duration: f64) -> ProbeSummary {
        ProbeSummary {
            duration,
            width: 1920,
            height: 1080,
            size_bytes: Some(100 * 1024 * 1024),
            container: Some("mp4".to_string()),
            subtitles: vec![],
        }
    }

    fn decide_bare(source_duration: f64, args: VideoArgs) -> Decision {
        decide(&args, &probe(source_duration), &cfg())
    }

    #[test]
    fn bare_short_video_runs_immediately() {
        // 90s / 60s-chunks = 2 grids = auto_run threshold.
        let d = decide_bare(90.0, VideoArgs::bare("/v.mp4"));
        let Decision::Run { invocation, .. } = d else {
            panic!("expected Run, got {d:?}");
        };
        assert_eq!(invocation.subcommand, RawSubcommand::Process);
        assert_eq!(invocation.predicted_grids, 2);
        assert_eq!(invocation.window, None, "full window is implicit");
    }

    #[test]
    fn bare_long_video_answers_with_plan() {
        let d = decide_bare(3600.0, VideoArgs::bare("/v.mp4"));
        let Decision::Plan(plan) = d else {
            panic!("expected Plan, got {d:?}");
        };
        assert_eq!(plan.recommended_calls.len(), 2);
        assert!(plan.guidance.contains("1:00:00"));
    }

    #[test]
    fn sub_minute_window_extracts_single_grid() {
        let args = VideoArgs {
            from: Some(TimeValue::Text("1:00".into())),
            until: Some(TimeValue::Text("1:30".into())),
            ..VideoArgs::bare("/v.mp4")
        };
        let d = decide_bare(3600.0, args);
        let Decision::Run { invocation, .. } = d else {
            panic!("expected Run, got {d:?}");
        };
        assert_eq!(invocation.subcommand, RawSubcommand::Extract);
        assert!(invocation.flags.contains(&"--range".to_string()));
        assert!(invocation.flags.contains(&"60-90".to_string()));
    }

    #[test]
    fn intent_within_budget_runs_with_window() {
        let args = VideoArgs {
            from: Some(TimeValue::Number(0.0)),
            until: Some(TimeValue::Number(300.0)),
            ..VideoArgs::bare("/v.mp4")
        };
        let d = decide_bare(3600.0, args);
        let Decision::Run { invocation, .. } = d else {
            panic!("expected Run, got {d:?}");
        };
        assert_eq!(invocation.predicted_grids, 5);
        let window = invocation
            .window
            .expect("explicit sub-range rides as a window");
        assert_eq!((window.start(), window.end()), (0.0, 300.0));
    }

    #[test]
    fn over_budget_stretches_chunks_with_notice() {
        let args = VideoArgs {
            look: Some(Look::Overview),
            ..VideoArgs::bare("/v.mp4")
        };
        // 1200s at 60s chunks = 20 grids > 8; stretched = ceil(1200/8) = 150s.
        let d = decide_bare(1200.0, args);
        let Decision::Run {
            invocation,
            notices,
        } = d
        else {
            panic!("expected Run, got {d:?}");
        };
        assert_eq!(invocation.predicted_grids, 8);
        assert!(invocation.flags.contains(&"--chunk-seconds".to_string()));
        assert!(invocation.flags.contains(&"150".to_string()));
        assert!(notices.iter().any(|n| n.contains("budget")), "{notices:?}");
    }

    #[test]
    fn absurd_stretch_requires_consent() {
        // 6000s per grid > 600s floor.
        let args = VideoArgs {
            look: Some(Look::Overview),
            ..VideoArgs::bare("/v.mp4")
        };
        let d = decide(&args, &probe(48000.0), &cfg());
        let Decision::Confirm { offers, .. } = d else {
            panic!("expected Confirm, got {d:?}");
        };
        assert!(
            offers[0].effect.contains("8 grids"),
            "the effect line spells out the consequence: {}",
            offers[0].effect
        );

        // The echoed token authorizes exactly the committed invocation.
        let store = Tokens::default();
        let committed = store.commit("/v.mp4", offers);
        let redeemed = redeem(
            &VideoArgs {
                confirm: Some(committed[0].token.as_str().to_string()),
                ..VideoArgs::bare("/v.mp4")
            },
            &store,
        );
        assert!(matches!(redeemed, Decision::Run { .. }), "{redeemed:?}");
    }

    #[test]
    fn stretch_floor_is_config_not_code() {
        // The same video that needs consent under the default floor runs
        // once the operator raises budget.max_stretch_seconds: the floor
        // is the third rung of the config ladder, not a code constant.
        let args = VideoArgs {
            look: Some(Look::Overview),
            ..VideoArgs::bare("/v.mp4")
        };
        let mut cfg = cfg();
        cfg.budget.max_stretch_seconds = std::num::NonZeroU32::new(6000).expect("non-zero");
        // 48000s / 8 grids = 6000s per grid, exactly at the raised floor.
        let d = decide(&args, &probe(48000.0), &cfg);
        assert!(matches!(d, Decision::Run { .. }), "{d:?}");
    }

    #[test]
    fn raw_extract_runs_unbudgeted() {
        let args = VideoArgs {
            raw: Some("extract --range 10-20".into()),
            ..VideoArgs::bare("/v.mp4")
        };
        let d = decide_bare(48000.0, args);
        assert!(matches!(d, Decision::Run { .. }), "{d:?}");
    }

    #[test]
    fn raw_over_budget_requires_consent() {
        let args = VideoArgs {
            raw: Some("process".into()),
            ..VideoArgs::bare("/v.mp4")
        };
        let d = decide_bare(3600.0, args); // 60 grids
        let Decision::Confirm { offers, .. } = d else {
            panic!("expected Confirm, got {d:?}");
        };
        assert!(
            offers[0].effect.contains("~60 grids"),
            "the effect line spells out the consequence: {}",
            offers[0].effect
        );
    }

    #[test]
    fn raw_with_intent_rejects() {
        let args = VideoArgs {
            raw: Some("process".into()),
            look: Some(Look::Overview),
            ..VideoArgs::bare("/v.mp4")
        };
        let d = decide_bare(60.0, args);
        assert!(matches!(d, Decision::Reject { .. }), "{d:?}");
    }

    #[test]
    fn raw_with_strategy_rejects() {
        let args = VideoArgs {
            raw: Some("process".into()),
            strategy: Some(Strategy::Uniform),
            ..VideoArgs::bare("/v.mp4")
        };
        let d = decide_bare(60.0, args);
        let Decision::Reject { message } = d else {
            panic!("expected Reject, got {d:?}");
        };
        assert!(message.contains("strategy"), "{message}");
    }

    #[test]
    fn uniform_strategy_rides_as_fixed_fps_flag() {
        let args = VideoArgs {
            strategy: Some(Strategy::Uniform),
            until: Some(TimeValue::Number(300.0)),
            ..VideoArgs::bare("/v.mp4")
        };
        let d = decide_bare(3600.0, args);
        let Decision::Run { invocation, .. } = d else {
            panic!("expected Run, got {d:?}");
        };
        assert_eq!(invocation.subcommand, RawSubcommand::Process);
        assert!(invocation.flags.contains(&"--strategy".to_string()));
        assert!(invocation.flags.contains(&"fixed-fps".to_string()));
    }

    #[test]
    fn adaptive_strategy_adds_no_flag_but_counts_as_intent() {
        // Explicit adaptive on a long video: skips the plan gate (intent),
        // but the invocation carries no --strategy so core stays lenient.
        let args = VideoArgs {
            strategy: Some(Strategy::Adaptive),
            until: Some(TimeValue::Number(300.0)),
            ..VideoArgs::bare("/v.mp4")
        };
        let d = decide_bare(3600.0, args);
        let Decision::Run { invocation, .. } = d else {
            panic!("expected Run, got {d:?}");
        };
        assert!(!invocation.flags.contains(&"--strategy".to_string()));
    }

    #[test]
    fn explicit_strategy_forces_process_on_single_grid_window() {
        let args = VideoArgs {
            from: Some(TimeValue::Text("1:00".into())),
            until: Some(TimeValue::Text("1:30".into())),
            strategy: Some(Strategy::Keyframes),
            ..VideoArgs::bare("/v.mp4")
        };
        let d = decide_bare(3600.0, args);
        let Decision::Run { invocation, .. } = d else {
            panic!("expected Run, got {d:?}");
        };
        assert_eq!(
            invocation.subcommand,
            RawSubcommand::Process,
            "extract would silently ignore the strategy"
        );
        assert!(invocation.flags.contains(&"keyframes".to_string()));
        let window = invocation
            .window
            .expect("explicit sub-range rides as a window");
        assert_eq!((window.start(), window.end()), (60.0, 90.0));
    }

    #[test]
    fn confirm_with_intent_rejects() {
        let args = VideoArgs {
            confirm: Some("t".into()),
            look: Some(Look::Overview),
            ..VideoArgs::bare("/v.mp4")
        };
        let d = decide_bare(60.0, args);
        let Decision::Reject { message } = d else {
            panic!("expected Reject, got {d:?}");
        };
        assert!(message.contains("alone"), "{message}");
    }

    #[test]
    fn unknown_token_rejects_with_refresh_hint() {
        let store = Tokens::default();
        let d = redeem(
            &VideoArgs {
                confirm: Some("nope".into()),
                ..VideoArgs::bare("/v.mp4")
            },
            &store,
        );
        let Decision::Reject { message } = d else {
            panic!("expected Reject, got {d:?}");
        };
        assert!(message.contains("stale"), "{message}");
    }

    #[test]
    fn plan_mentions_subtitles_when_present() {
        let mut p = probe(3600.0);
        p.subtitles = vec!["embedded #0 (eng, subrip)".to_string()];
        let plan = plan_response("/v.mp4", &p, &cfg());
        assert!(plan.guidance.contains("Subtitles available"));
    }

    #[test]
    fn formats_bytes_in_ts_tiers() {
        assert_eq!(format_bytes(3 * 1024 * 1024 * 1024), "3.0 GiB");
        assert_eq!(format_bytes(1288490189), "1.2 GiB");
        assert_eq!(format_bytes(34 * 1024 * 1024), "34 MiB");
        assert_eq!(format_bytes(1), "1 KiB");
    }
}
