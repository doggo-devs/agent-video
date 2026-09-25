// SPDX-License-Identifier: Apache-2.0
//! The `video` call flow, from coerced arguments to a wire result.
//!
//! Couriers one call through cached probe, policy ([`crate::plan`],
//! [`crate::offer`]), and execution: core runs (self-spawned, JSON
//! captured), downloads (grace window then detach into the
//! [`crate::mcp::tracker::Tracker`]), and whisper transcription. Everything
//! here is synchronous; the rmcp server hosts it via `spawn_blocking`.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use rmcp::model::CallToolResult;

use super::deps::{Notifier, ServerDeps};
use super::download::{Waited, execute_download, keep_response, slot_key, wait_out};
use super::invoke::{run_invocation, transcribe_flow};
use super::probe_cache::{MediaMissing, ProbeError, probe_media};
use super::wire::{
    coerce_args, coerce_confirm_args, evicted_rip, failed, foreign_scheme, is_url, json_result,
    pending_result,
};
use crate::fault::Fault;
use crate::fetch::rip_key;
use crate::offer::decide_url;
use crate::plan::{Decision, PlanDecision, decide, redeem};
use crate::schema::OfferedToken;
use crate::schema::{Intent, ProbeSummary, RecommendedCall, VideoArgs, VideoBody, Want};
use crate::time::ArgError;
use crate::tokens::{Dest, OfferDraft};

// ── tool handlers ──

/// Handle one `video_confirm` call: redeem a token against the plan it was
/// issued for.
///
/// Consent runs ahead of every slot poll — a keep confirm during a running
/// analysis download starts the keep job rather than observing the analysis
/// one. A confirm whose own download already runs converges inside
/// `execute_download` (same slot, one writer).
pub fn handle_confirm_call(
    raw: &serde_json::Map<String, serde_json::Value>,
    deps: &Arc<ServerDeps>,
    notify: &Notifier,
) -> CallToolResult {
    let (source, token) = match coerce_confirm_args(raw) {
        Ok(pair) => pair,
        Err(e) => return failed(&e),
    };
    let decision = redeem(&source, &token, &deps.tokens);
    let args = VideoArgs::bare(&source);
    execute_decision(&args, decision, None, Vec::new(), deps, notify)
}

/// Handle one `video` call end to end; never panics on model input, errors
/// become `Failed` bodies.
pub fn handle_video_call(
    raw: &serde_json::Map<String, serde_json::Value>,
    deps: &Arc<ServerDeps>,
    notify: &Notifier,
) -> CallToolResult {
    let args = match coerce_args(raw) {
        Ok(a) => a,
        Err(e) => return failed(&e),
    };

    // A protocol URL this layer does not fetch (rtsp, rtmp, …) is rejected
    // by name, before any flow can misread it as a local path and diagnose
    // a missing file.
    if let Some(scheme) = foreign_scheme(&args.source) {
        return failed(&Fault::bad_input(format!(
            "\"{}\" is a {scheme}:// URL, and this tool cannot fetch {scheme} sources — only \
             http(s) URLs and local file paths are accepted",
            args.source
        )));
    }

    // Refine once: every route below matches a view that is valid by
    // construction.
    match args.intent() {
        Err(ArgError(message)) => return failed(&Fault::bad_input(message)),
        // Transcription is an explicit request, not a consent dance.
        Ok(Intent::Transcript(window)) => {
            let (from, until) = (window.from.cloned(), window.until.cloned());
            return transcribe_flow(&args, from, until, deps, notify);
        }
        Ok(Intent::Raw(_) | Intent::Grid(_) | Intent::Plan) => {}
    }

    // Background downloads: wait each slot so a URL never sits on Pending
    // for the whole session. Every observer waits the same window, so a
    // poll costs wall clock instead of being free. Progress samples ride
    // the *polling* request's own token.
    if is_url(&args.source) {
        for dest in [Dest::Cache, Dest::Keep] {
            match wait_out(&slot_key(&args.source, dest), deps, notify) {
                Waited::Vacant => {}
                Waited::Settled(result) => return result,
                Waited::Running(pace) => return pending_result(&pace),
                Waited::Abandoned => {
                    return failed(&Fault::transient(
                        "the background download stopped reporting progress and was abandoned — \
                         call again with the URL to start over",
                    ));
                }
            }
        }
        // A cache hit that evaporates before use is indistinguishable
        // from a miss: another process's eviction can unlink the rip
        // between this resolve and the probe's open. Re-resolve once — a
        // rival's re-download or a surviving rival container may answer —
        // then fall through to first contact (a fresh download offer),
        // exactly as if the entry had never been found. Never a
        // `bad_input`: the model's arguments were right.
        let key = rip_key(&args.source, &deps.config);
        for _ in 0..2 {
            let Some(cached) = deps.cache.find_download(&key) else {
                break;
            };
            match local_flow(
                &args,
                &cached,
                vec!["serving from the cached analysis rip".to_string()],
                deps,
                notify,
            ) {
                Ok(result) => return result,
                Err(MediaMissing) => continue,
            }
        }
        return url_first_contact(&args, deps, notify);
    }
    let path = PathBuf::from(&args.source);
    // A user-supplied path that is missing is the caller's error — the
    // miss semantics above are reserved for cache-derived paths.
    local_flow(&args, &path, Vec::new(), deps, notify).unwrap_or_else(|MediaMissing| {
        failed(&Fault::bad_input(format!(
            "file not found: {}",
            path.display()
        )))
    })
}

/// Probe a resolved media path and run the local decision engine. An
/// absent file is returned as [`MediaMissing`], never rendered here: only
/// the caller knows what absence means for this path's provenance — a
/// user-supplied argument is the caller's error, a cache-derived rip was
/// evicted underfoot and must answer like a cache miss.
fn local_flow(
    args: &VideoArgs,
    media_path: &Path,
    extra_notices: Vec<String>,
    deps: &Arc<ServerDeps>,
    notify: &Notifier,
) -> Result<CallToolResult, MediaMissing> {
    let (probe, _) = match probe_media(media_path, deps) {
        Ok(v) => v,
        Err(ProbeError::Missing(m)) => return Err(m),
        Err(ProbeError::Fault(e)) => return Ok(failed(&e)),
    };
    let mut decision = decide(args, &probe, &deps.config);
    offer_transcription(&args.source, &probe, &mut decision, deps);
    Ok(execute_decision(
        args,
        decision,
        Some(media_path.to_path_buf()),
        extra_notices,
        deps,
        notify,
    ))
}

/// A subtitle-less source grows a transcription hint: a recommended call on
/// a plan, a notice on a render — never a gate, so learning that whisper
/// exists costs no round trip.
pub(super) fn offer_transcription(
    source: &str,
    probe: &ProbeSummary,
    decision: &mut Decision,
    deps: &ServerDeps,
) {
    if !probe.subtitles.is_empty() {
        return;
    }
    match decision {
        Decision::Plan(plan) => offer_transcription_plan(source, plan, deps),
        Decision::Run { notices, .. } if deps.whisper.available(&deps.config) => {
            notices.push(format!(
                "no subtitles found for this source — call again with {} to generate a \
                 whisper transcript; it then rides along with every grid",
                serde_json::to_string(&transcript_call(source)).expect("VideoArgs serializes")
            ))
        }
        _ => {}
    }
}

/// The plan-shaped delivery, for callers already holding a [`PlanDecision`].
pub(super) fn offer_transcription_plan(source: &str, plan: &mut PlanDecision, deps: &ServerDeps) {
    if !plan.probe.subtitles.is_empty() {
        return;
    }
    if deps.whisper.available(&deps.config) {
        plan.recommended_calls.push(RecommendedCall {
            call: transcript_call(source),
            note: "no subtitles found — generate a whisper transcript; it then rides \
                   along with every grid"
                .to_string(),
        });
    } else {
        plan.guidance.push_str(&install_hint());
    }
}

/// The call that transcribes `source`; both hint deliveries quote it.
fn transcript_call(source: &str) -> VideoArgs {
    VideoArgs {
        want: Some(Want::Transcript),
        ..VideoArgs::bare(source)
    }
}

/// Told when a source has no subtitles and this installation cannot make
/// any. The remedy comes from `transcribe`, which knows which tiers this
/// build carries.
fn install_hint() -> String {
    format!(
        " No subtitles found and no transcription available — {}.",
        crate::transcribe::unavailable_hint()
    )
}

/// First contact with a URL that has no cached rip: offer downloads.
fn url_first_contact(
    args: &VideoArgs,
    deps: &Arc<ServerDeps>,
    notify: &Notifier,
) -> CallToolResult {
    let relay = match deps.relay() {
        Ok(r) => r,
        Err(e) => return failed(&e),
    };
    let info = match deps.fetcher.probe_url(&args.source, &deps.config, relay) {
        Ok(i) => i,
        Err(e) => return failed(&e),
    };
    let decision = decide_url(args, &info, &deps.config);
    execute_decision(args, decision, None, Vec::new(), deps, notify)
}

/// Carry out a policy decision. Pure decisions (plan, reject) render
/// directly; effectful ones execute through their named executors —
/// committing consent offers, starting downloads, running core — and
/// render the outcome. One dispatch, so a new [`Decision`] variant does
/// not compile until it names where its execution lives.
fn execute_decision(
    args: &VideoArgs,
    decision: Decision,
    media_path: Option<PathBuf>,
    extra_notices: Vec<String>,
    deps: &Arc<ServerDeps>,
    notify: &Notifier,
) -> CallToolResult {
    match decision {
        // Policy rejections answer the model's arguments (bad range, stale
        // token): the caller fixes them.
        Decision::Reject { message } => failed(&Fault::bad_input(message)),
        Decision::Plan(p) => json_result(&VideoBody::Plan {
            probe: p.probe,
            guidance: p.guidance,
            recommended_calls: p.recommended_calls,
        }),
        Decision::Confirm {
            consequence,
            offers,
        } => commit_offers(&args.source, consequence, offers, deps),
        Decision::Run {
            invocation,
            notices,
        } => {
            let mut all = extra_notices;
            all.extend(notices);
            run_core(&args.source, &invocation, all, media_path, deps)
        }
        Decision::Download { plan, token, .. } => {
            execute_download(&args.source, plan, token, deps, notify)
        }
        Decision::Saved { path } => replay_saved(&path, deps),
    }
}

/// Commit a decision's consent offers and render the confirm body.
/// Committing here makes "the model saw it" and "it is redeemable" the
/// same event: the registry mints the tokens, and the wire body renders
/// exactly the offers it returned.
fn commit_offers(
    source: &str,
    consequence: String,
    offers: Vec<OfferDraft>,
    deps: &ServerDeps,
) -> CallToolResult {
    let tokens: Vec<OfferedToken> = deps
        .tokens
        .commit(source, offers)
        .into_iter()
        .map(|o| OfferedToken {
            token: o.token.as_str().to_string(),
            effect: o.effect,
        })
        .collect();
    json_result(&VideoBody::ConfirmRequired {
        consequence,
        tokens,
    })
}

/// Run a decided core invocation against the media the source resolves to
/// (the caller's probed path when it has one, else the URL's cached rip).
fn run_core(
    source: &str,
    invocation: &crate::plan::CoreInvocation,
    notices: Vec<String>,
    media_path: Option<PathBuf>,
    deps: &Arc<ServerDeps>,
) -> CallToolResult {
    let Some(resolved) = media_path.or_else(|| resolve_media(source, deps)) else {
        return failed(&evicted_rip());
    };
    run_invocation(&resolved, invocation, &notices, deps)
}

/// Replay a completed keep download's receipt: a keep token echoed after
/// its download finished answers with the saved file instead of ripping a
/// duplicate into the keep dir.
fn replay_saved(path: &Path, deps: &Arc<ServerDeps>) -> CallToolResult {
    if path.exists() {
        keep_response(path, None, deps)
    } else {
        failed(&Fault::bad_input(format!(
            "this download already completed and was saved to {}, but the file has \
             since been moved or removed. Call again without confirm for a fresh plan.",
            path.display()
        )))
    }
}

/// Local paths resolve to themselves; URLs to their cached rip — under the
/// *current* recipe — if any.
pub(super) fn resolve_media(source: &str, deps: &ServerDeps) -> Option<PathBuf> {
    if !is_url(source) {
        return Some(PathBuf::from(source));
    }
    deps.cache.find_download(&rip_key(source, &deps.config))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fetch::SourceInfo;
    use crate::mcp::download::DOWNLOAD_ABANDON_LIMIT;
    use crate::mcp::testutil::*;
    use rmcp::model::ContentBlock;
    use std::sync::Mutex;
    use std::sync::mpsc::{Receiver, Sender};
    use std::time::Duration;

    #[test]
    fn rejects_unknown_argument() {
        let env = env(probe(90.0, vec![]), false);
        let r = call(&env, serde_json::json!({"source": "/v.mp4", "zoom": 1}));
        assert_eq!(r.is_error, Some(true));
        let body = body_of(&r);
        assert!(
            body["error"]
                .as_str()
                .expect("error text")
                .contains("unknown argument \"zoom\""),
            "{body}"
        );
    }

    #[test]
    fn rejects_missing_source() {
        let env = env(probe(90.0, vec![]), false);
        let r = call(&env, serde_json::json!({}));
        assert!(text_of(&r).contains("source must be a non-empty string"));
    }

    #[test]
    fn missing_file_fails_cleanly() {
        let env = env(probe(90.0, vec![]), false);
        let r = call(&env, serde_json::json!({"source": "/no/such/file.mp4"}));
        let body = body_of(&r);
        assert!(
            body["error"]
                .as_str()
                .expect("error text")
                .contains("file not found"),
            "{body}"
        );
        // The failure class rides the wire: the model can tell "fix the
        // argument" from "this build lacks a capability" without parsing prose.
        assert_eq!(body["kind"], "bad_input", "{body}");
    }

    #[test]
    fn absent_media_is_a_typed_miss_for_the_caller_to_classify() {
        // `local_flow` must not bake in a diagnosis: the URL branch maps
        // this miss to a re-resolve/fresh offer, the local branch to
        // `bad_input` — the provenance decides, not the probe.
        let env = env(probe(90.0, vec![]), false);
        let args = VideoArgs::bare("/no/such/file.mp4");
        let r = local_flow(
            &args,
            Path::new("/no/such/file.mp4"),
            Vec::new(),
            &env.deps,
            &crate::mcp::deps::null_notifier(),
        );
        assert!(r.is_err(), "an absent path surfaces as MediaMissing");
    }

    #[test]
    fn evicted_rip_answers_as_a_miss_not_a_caller_error() {
        // The rip landed, then vanished (eviction in another process).
        // The next call must re-offer the download — never tell the model
        // its arguments are wrong.
        let env = env(probe(90.0, vec![]), false);
        let url = "https://example.com/v.mp4";
        let offer = body_of(&call(&env, serde_json::json!({"source": url})));
        assert_eq!(offer["status"], "confirm_required", "{offer}");
        let token = offered_tokens(&offer).remove(0);
        let done = body_of(&confirm(&env, url, &token));
        assert_eq!(done["status"], "plan", "{done}");

        let downloads = env.deps.config.paths.cache.join("downloads");
        for entry in std::fs::read_dir(&downloads).expect("downloads dir") {
            std::fs::remove_file(entry.expect("entry").path()).expect("evict the rip");
        }

        let again = body_of(&call(&env, serde_json::json!({"source": url})));
        assert_eq!(
            again["status"], "confirm_required",
            "an evaporated rip re-offers the download: {again}"
        );
    }

    #[test]
    fn rtsp_source_is_diagnosed_as_unsupported_scheme() {
        let env = env(probe(90.0, vec![]), false);
        let r = call(&env, serde_json::json!({"source": "rtsp://cam/1"}));
        let body = body_of(&r);
        assert_eq!(body["kind"], "bad_input", "{body}");
        let error = body["error"].as_str().expect("error text");
        assert!(
            error.contains("rtsp") && error.contains("http"),
            "the diagnosis names the scheme and what is accepted: {error}"
        );
        assert!(
            !error.contains("file not found"),
            "a protocol URL must not be misdiagnosed as a missing file: {error}"
        );
    }

    #[test]
    fn short_video_runs_and_renders_grids() {
        let env = env(probe(90.0, vec![]), false);
        let file = media_file(&env, "v.mp4");
        let r = call(&env, serde_json::json!({"source": file}));
        assert_ne!(r.is_error, Some(true), "{}", text_of(&r));
        let text = text_of(&r);
        assert!(text.contains("[grid 1/1]"), "{text}");
        assert!(
            text.contains("key frames (bordered): 00:10 @hero"),
            "{text}"
        );
        let images = r
            .content
            .iter()
            .filter(|c| matches!(c, ContentBlock::Image(_)))
            .count();
        assert_eq!(images, 1, "one base64 grid becomes one image block");
    }

    #[test]
    fn bare_local_source_renders_instead_of_answering_with_a_menu() {
        let env = env(probe(3600.0, vec![]), true);
        let file = media_file(&env, "long.mp4");
        let r = call(&env, serde_json::json!({"source": file}));
        assert!(
            r.content
                .iter()
                .any(|c| matches!(c, ContentBlock::Image(_))),
            "a file already on disk is decoded, not planned around: {:?}",
            r.content
        );
    }

    #[test]
    fn subtitleless_render_hints_a_call_the_server_accepts() {
        let env = env(probe(3600.0, vec![]), true);
        let file = media_file(&env, "long.mp4");
        let text = text_of(&call(&env, serde_json::json!({"source": file})));
        let (_, quoted) = text
            .split_once("call again with ")
            .unwrap_or_else(|| panic!("the hint rides with the frames: {text}"));
        let hinted: serde_json::Value = serde_json::Deserializer::from_str(quoted)
            .into_iter()
            .next()
            .expect("the hint quotes a call")
            .expect("the quoted call is JSON");

        let followed = body_of(&call(&env, hinted));

        assert_eq!(followed["status"], "transcript", "{followed}");
    }

    #[test]
    fn subtitleless_plan_without_whisper_hints_install() {
        let env = env(probe(3600.0, vec![]), false);
        let summary = probe(3600.0, vec![]);
        let mut plan = crate::plan::plan_response("/v.mp4", &summary, &env.deps.config);
        offer_transcription_plan("/v.mp4", &mut plan, &env.deps);
        // The remedy is whatever this build can honour, quoted from the
        // one place that knows.
        assert!(
            plan.guidance
                .contains(crate::transcribe::unavailable_hint()),
            "{plan:?}"
        );
    }

    #[test]
    fn transcribe_arg_runs_whisper_and_rides_along() {
        let env = env(probe(3600.0, vec![]), true);
        let file = media_file(&env, "talk.mp4");
        let source = file.to_string_lossy().into_owned();

        let after = body_of(&call(
            &env,
            serde_json::json!({"source": source, "want": "transcript"}),
        ));
        assert_eq!(after["status"], "transcript", "{after}");
        assert_eq!(
            after["transcript"], "00:00 hi",
            "the reply carries the transcript text itself: {after}"
        );
        assert!(
            after["guidance"]
                .as_str()
                .expect("guidance")
                .contains("Transcript cached"),
            "{after}"
        );

        // The transcript now rides along with core runs.
        let r = call(
            &env,
            serde_json::json!({"source": source, "from": 0, "until": 90}),
        );
        assert_ne!(r.is_error, Some(true), "{}", text_of(&r));
        let calls = env.core.calls.lock().expect("calls");
        let last = calls.last().expect("core ran");
        assert!(
            last.iter().any(|a| a == "--subtitle"),
            "transcript must ride along: {last:?}"
        );
    }

    #[test]
    fn repeated_transcribe_reuses_cached_transcript() {
        let env = env(probe(3600.0, vec![]), true);
        let file = media_file(&env, "talk.mp4");
        let source = file.to_string_lossy().into_owned();

        let first = body_of(&call(
            &env,
            serde_json::json!({"source": source, "want": "transcript"}),
        ));
        assert_eq!(first["status"], "transcript", "{first}");
        let again = body_of(&call(
            &env,
            serde_json::json!({"source": source, "want": "transcript"}),
        ));
        assert_eq!(again["status"], "transcript", "{again}");
        assert_eq!(
            again["transcript"], first["transcript"],
            "the cache hit answers with the same text"
        );
        assert_eq!(
            *env.whisper.calls.lock().expect("calls"),
            1,
            "second transcribe must hit the transcript cache"
        );
    }

    #[test]
    fn truncated_audio_warns_in_transcribe_guidance() {
        let env = env(probe(437.0, vec![]), true);
        *env.whisper.audio_seconds.lock().expect("audio_seconds") = 49.0;
        let file = media_file(&env, "trunc.mp4");
        let body = body_of(&call(
            &env,
            serde_json::json!({"source": file, "want": "transcript"}),
        ));
        assert_eq!(body["status"], "transcript", "{body}");
        let guidance = body["guidance"].as_str().expect("guidance");
        assert!(
            guidance.starts_with("Warning: only 00:49"),
            "truncation leads the guidance: {guidance}"
        );
        assert!(guidance.contains("Transcript cached"), "{guidance}");
    }

    #[test]
    fn transcribe_without_whisper_reports_environment() {
        let env = env(probe(90.0, vec![]), false);
        let file = media_file(&env, "v.mp4");
        let r = call(
            &env,
            serde_json::json!({"source": file, "want": "transcript"}),
        );
        let body = body_of(&r);
        assert_eq!(body["kind"], "environment", "{body}");
        assert!(
            body["error"]
                .as_str()
                .expect("error")
                .contains(crate::transcribe::unavailable_hint()),
            "{body}"
        );
    }

    #[test]
    fn transcribe_on_undownloaded_url_rejects() {
        let env = env(probe(90.0, vec![]), true);
        let r = call(
            &env,
            serde_json::json!({"source": "https://example.com/v.mp4", "want": "transcript"}),
        );
        let body = body_of(&r);
        assert_eq!(body["kind"], "bad_input", "{body}");
        assert!(
            body["error"]
                .as_str()
                .expect("error")
                .contains("confirm a download"),
            "{body}"
        );
    }

    #[test]
    fn reconfirmed_download_serves_cached_rip_without_refetch() {
        let env = env(probe(90.0, vec![]), false);
        let url = "https://example.com/v.mp4";

        let offer = body_of(&call(&env, serde_json::json!({"source": url})));
        assert_eq!(offer["status"], "confirm_required", "{offer}");
        let token = offered_tokens(&offer).remove(0);
        let done = body_of(&confirm(&env, url, &token));
        assert_eq!(done["status"], "plan", "{done}");

        // The token survives redemption; echoing it again must not refetch.
        let again = body_of(&confirm(&env, url, &token));
        assert_eq!(again["status"], "plan", "{again}");
        assert_eq!(
            *env.fetcher.downloads.lock().expect("downloads"),
            1,
            "re-confirmation must serve the cached rip"
        );
    }

    #[test]
    fn a_short_rip_costs_the_caller_no_consent_round_trip() {
        // The defect this pins: a CDN backend serving a truncated part made
        // the download fail, and every retry had to come from the model as
        // a fresh confirm — a full consent round-trip apiece. One confirm
        // must outlast the burst.
        let env = env_with_fetcher(
            probe(90.0, vec![]),
            false,
            FakeFetcher {
                info: SourceInfo {
                    headline: "A video.".to_string(),
                    title: None,
                    options: vec![analysis_option()],
                },
                gate: None,
                warning: None,
                downloads: Mutex::new(0),
                finished: Mutex::new(0),
                last_cancel: Mutex::new(None),
                short_attempts: Mutex::new(2),
            },
        );
        let url = "https://example.com/v.mp4";

        let offer = body_of(&call(&env, serde_json::json!({"source": url})));
        let token = offered_tokens(&offer).remove(0);
        let done = body_of(&confirm(&env, url, &token));

        assert_eq!(
            done["status"], "plan",
            "one confirm must land the rip: {done}"
        );
        assert_eq!(
            *env.fetcher.downloads.lock().expect("downloads"),
            3,
            "the re-extractions happen here, behind the one consent the model gave"
        );
    }

    #[test]
    fn surfaces_download_warning_before_guidance() {
        // A degraded download (media landed, subtitles did not) must inform
        // the model instead of failing or staying silent.
        let env = env_with_fetcher(
            probe(90.0, vec![]),
            false,
            FakeFetcher {
                info: SourceInfo {
                    headline: "A video.".to_string(),
                    title: None,
                    options: vec![analysis_option()],
                },
                gate: None,
                warning: Some("subtitles could not be fetched (login required)".to_string()),
                downloads: Mutex::new(0),
                finished: Mutex::new(0),
                last_cancel: Mutex::new(None),
                short_attempts: Mutex::new(0),
            },
        );
        let url = "https://example.com/v.mp4";

        let offer = body_of(&call(&env, serde_json::json!({"source": url})));
        assert_eq!(offer["status"], "confirm_required", "{offer}");

        let token = offered_tokens(&offer).remove(0);
        let done = body_of(&confirm(&env, url, &token));
        assert_eq!(done["status"], "plan", "{done}");
        let guidance = done["guidance"].as_str().expect("guidance");
        assert!(
            guidance.starts_with("Warning: subtitles could not be fetched"),
            "warning leads the guidance: {guidance}"
        );
        assert!(
            guidance.contains("Analysis rip downloaded"),
            "success text still follows: {guidance}"
        );
    }

    #[test]
    fn url_flow_offers_confirm_then_downloads_inline() {
        let env = env(probe(90.0, vec![]), false);
        let url = "https://example.com/v.mp4";

        let offer = body_of(&call(&env, serde_json::json!({"source": url})));
        assert_eq!(offer["status"], "confirm_required", "{offer}");
        let token = offered_tokens(&offer).remove(0);
        assert!(
            offer["tokens"][0]["effect"]
                .as_str()
                .expect("effect")
                .contains("analysis"),
            "the effect line names the tier: {offer}"
        );

        let done = body_of(&confirm(&env, url, &token));
        assert_eq!(done["status"], "plan", "{done}");
        assert!(
            done["guidance"]
                .as_str()
                .expect("guidance")
                .contains("Analysis rip downloaded"),
            "{done}"
        );

        // The URL now resolves to the cached rip.
        let again = call(&env, serde_json::json!({"source": url}));
        let text = text_of(&again);
        assert!(
            text.contains("cached analysis rip") || text.contains("[grid"),
            "{text}"
        );
    }

    #[test]
    fn slow_download_detaches_and_settles_on_poll() {
        let (release, gate): (Sender<()>, Receiver<()>) = std::sync::mpsc::channel();
        let mut env = env_with_fetcher(
            probe(90.0, vec![]),
            false,
            FakeFetcher {
                info: SourceInfo {
                    headline: "A video.".to_string(),
                    title: None,
                    options: vec![analysis_option()],
                },
                gate: Some(Mutex::new(gate)),
                warning: None,
                downloads: Mutex::new(0),
                finished: Mutex::new(0),
                last_cancel: Mutex::new(None),
                short_attempts: Mutex::new(0),
            },
        );
        // Shortest representable grace window keeps the test fast.
        Arc::get_mut(&mut env.deps)
            .expect("sole owner before threads spawn")
            .config
            .download
            .sync_wait_seconds = std::num::NonZeroU32::new(1).expect("non-zero");
        let url = "https://example.com/slow.mp4";

        let offer = body_of(&call(&env, serde_json::json!({"source": url})));
        assert_eq!(offer["status"], "confirm_required", "{offer}");

        let token = offered_tokens(&offer).remove(0);
        let pending = body_of(&confirm(&env, url, &token));
        assert_eq!(pending["status"], "pending", "{pending}");

        release.send(()).expect("release download");
        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        loop {
            let body = body_of(&call(&env, serde_json::json!({"source": url})));
            if body["status"] == "plan" {
                assert!(
                    body["guidance"]
                        .as_str()
                        .expect("guidance")
                        .contains("Analysis rip downloaded"),
                    "{body}"
                );
                break;
            }
            assert_eq!(body["status"], "pending", "{body}");
            assert!(
                std::time::Instant::now() < deadline,
                "download never settled"
            );
            std::thread::sleep(Duration::from_millis(20));
        }
    }

    #[test]
    fn abandons_downloads_that_stop_reporting() {
        let (release, gate): (Sender<()>, Receiver<()>) = std::sync::mpsc::channel();
        let mut env = env_with_fetcher(
            probe(90.0, vec![]),
            false,
            FakeFetcher {
                info: SourceInfo {
                    headline: "A video.".to_string(),
                    title: None,
                    options: vec![analysis_option()],
                },
                gate: Some(Mutex::new(gate)),
                warning: None,
                downloads: Mutex::new(0),
                finished: Mutex::new(0),
                last_cancel: Mutex::new(None),
                short_attempts: Mutex::new(0),
            },
        );
        Arc::get_mut(&mut env.deps)
            .expect("sole owner before threads spawn")
            .config
            .download
            .sync_wait_seconds = std::num::NonZeroU32::new(1).expect("non-zero");
        let url = "https://example.com/wedged.mp4";

        let offer = body_of(&call(&env, serde_json::json!({"source": url})));
        assert_eq!(offer["status"], "confirm_required", "{offer}");
        let token = offered_tokens(&offer).remove(0);
        let pending = body_of(&confirm(&env, url, &token));
        assert_eq!(pending["status"], "pending", "{pending}");

        // Rewind the job's progress clock past the abandonment limit — the
        // downloader-side watchdogs have failed and the thread is wedged.
        env.deps.downloads.age(
            &slot_key(url, Dest::Cache),
            DOWNLOAD_ABANDON_LIMIT + Duration::from_secs(60),
        );

        let r = call(&env, serde_json::json!({"source": url}));
        assert!(text_of(&r).contains("abandoned"), "{}", text_of(&r));

        // The URL is unpinned: the next call starts over with a fresh offer.
        let again = body_of(&call(&env, serde_json::json!({"source": url})));
        assert_eq!(again["status"], "confirm_required", "{again}");
        drop(release);
    }

    #[test]
    fn abandonment_cancels_the_in_flight_download() {
        let (env, release) = gated_env(vec![analysis_option()]);
        let url = "https://example.com/v.mp4";

        let offer = body_of(&call(&env, serde_json::json!({"source": url})));
        assert_eq!(offer["status"], "confirm_required", "{offer}");
        let token = offered_tokens(&offer).remove(0);
        let pending = body_of(&confirm(&env, url, &token));
        assert_eq!(pending["status"], "pending", "{pending}");
        let cancel = env
            .fetcher
            .last_cancel
            .lock()
            .expect("last_cancel")
            .clone()
            .expect("download started");
        assert!(!cancel.is_cancelled(), "a live job is not cancelled");

        env.deps.downloads.age(
            &slot_key(url, Dest::Cache),
            DOWNLOAD_ABANDON_LIMIT + Duration::from_secs(60),
        );
        let r = call(&env, serde_json::json!({"source": url}));
        assert!(text_of(&r).contains("abandoned"), "{}", text_of(&r));

        assert!(
            cancel.is_cancelled(),
            "abandoning the slot must reach the fetch's cancel signal"
        );
        drop(release);
    }

    #[test]
    fn zombie_keep_settle_cannot_fulfill_after_abandonment() {
        use crate::tokens::{Redemption, Token};
        let (env, release) = gated_env(vec![keep_option()]);
        let url = "https://example.com/v.mp4";

        let offer = body_of(&call(&env, serde_json::json!({"source": url})));
        assert_eq!(offer["status"], "confirm_required", "{offer}");
        let keep_token = offered_tokens(&offer).remove(0);
        let pending = body_of(&confirm(&env, url, &keep_token));
        assert_eq!(pending["status"], "pending", "{pending}");

        env.deps.downloads.age(
            &slot_key(url, Dest::Keep),
            DOWNLOAD_ABANDON_LIMIT + Duration::from_secs(60),
        );
        let r = call(&env, serde_json::json!({"source": url}));
        assert!(text_of(&r).contains("abandoned"), "{}", text_of(&r));

        // Let the zombie finish: it writes the keep file and settles into a
        // channel whose receiving slot is gone. The fetch counter is the
        // barrier — the file itself is transient now, because the disowned
        // settlement's compensation removes what the zombie published.
        release.send(()).expect("release the zombie");
        let keep_file = env.deps.config.paths.keep.join("v.mp4");
        let deadline = std::time::Instant::now() + Duration::from_secs(10);
        while *env.fetcher.finished.lock().expect("finished") == 0 {
            assert!(
                std::time::Instant::now() < deadline,
                "the zombie never finished its download"
            );
            std::thread::sleep(Duration::from_millis(10));
        }
        // The compensation runs inside the failed settle, so once the file
        // is gone the settle has provably completed and the receipt check
        // below cannot race it.
        while keep_file.exists() {
            assert!(
                std::time::Instant::now() < deadline,
                "a disowned keep settlement must remove the file it published"
            );
            std::thread::sleep(Duration::from_millis(10));
        }

        assert!(
            matches!(
                env.deps.tokens.take(url, &Token::new(keep_token)),
                Some(Redemption::Fresh(_))
            ),
            "a job that lost its slot must not record a receipt — the token \
             stays fresh for a retry"
        );
    }

    #[test]
    fn dropping_the_deps_graph_disowns_a_queued_keep_settlement() {
        let (env, release) = gated_env(vec![keep_option()]);
        let url = "https://example.com/v.mp4";

        let offer = body_of(&call(&env, serde_json::json!({"source": url})));
        assert_eq!(offer["status"], "confirm_required", "{offer}");
        let keep_token = offered_tokens(&offer).remove(0);
        let pending = body_of(&confirm(&env, url, &keep_token));
        assert_eq!(pending["status"], "pending", "{pending}");

        // Let the download finish and settle into a channel nobody polls:
        // the keep receipt (owned effect) and its compensation now sit
        // queued inside the tracker's slot.
        release.send(()).expect("release download");
        let keep_file = env.deps.config.paths.keep.join("v.mp4");
        let deadline = std::time::Instant::now() + Duration::from_secs(10);
        while *env.fetcher.finished.lock().expect("finished") == 0 {
            assert!(
                std::time::Instant::now() < deadline,
                "the download never finished"
            );
            std::thread::sleep(Duration::from_millis(10));
        }
        assert!(keep_file.exists(), "the keep file landed");

        // Drop every external handle. The queued receipt captures a Tokens
        // handle — not Arc<ServerDeps> — so the deps graph is actually
        // freeable: the tracker tears down, the settlement drops unadopted,
        // and its compensation removes the file no receipt points to. (A
        // deps-capturing receipt would cycle deps → tracker → settlement →
        // deps and leak the graph with the drop bomb defused forever.)
        let TestEnv {
            deps, dir: _dir, ..
        } = env;
        drop(deps);
        while keep_file.exists() {
            assert!(
                std::time::Instant::now() < deadline,
                "dropping the deps graph must fire the queued settlement's compensation"
            );
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    #[test]
    fn stale_confirm_token_rejects() {
        let env = env(probe(90.0, vec![]), false);
        let file = media_file(&env, "v.mp4");
        let r = confirm(&env, file, "nope");
        assert!(text_of(&r).contains("stale"), "{}", text_of(&r));
    }

    /// A counting notifier plus its call count, for pinning who may send
    /// progress notifications.
    fn counting_notifier() -> (Notifier, Arc<std::sync::atomic::AtomicU32>) {
        let count = Arc::new(std::sync::atomic::AtomicU32::new(0));
        let seen = Arc::clone(&count);
        let notifier: Notifier = Arc::new(move |_, _, _| {
            seen.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        });
        (notifier, count)
    }

    fn gated_env(options: Vec<crate::fetch::DownloadOption>) -> (TestEnv, Sender<()>) {
        let (release, gate): (Sender<()>, Receiver<()>) = std::sync::mpsc::channel();
        let mut env = env_with_fetcher(
            probe(90.0, vec![]),
            false,
            FakeFetcher {
                info: SourceInfo {
                    headline: "A video.".to_string(),
                    title: None,
                    options,
                },
                gate: Some(Mutex::new(gate)),
                warning: None,
                downloads: Mutex::new(0),
                finished: Mutex::new(0),
                last_cancel: Mutex::new(None),
                short_attempts: Mutex::new(0),
            },
        );
        Arc::get_mut(&mut env.deps)
            .expect("sole owner before threads spawn")
            .config
            .download
            .sync_wait_seconds = std::num::NonZeroU32::new(1).expect("non-zero");
        (env, release)
    }

    #[test]
    fn keep_confirm_during_a_running_analysis_download_is_not_swallowed() {
        let (env, release) = gated_env(vec![analysis_option(), keep_option()]);
        let url = "https://example.com/v.mp4";

        let offer = body_of(&call(&env, serde_json::json!({"source": url})));
        assert_eq!(offer["status"], "confirm_required", "{offer}");
        let tokens = offered_tokens(&offer);
        let (analysis_token, keep_token) = (&tokens[0], &tokens[1]);

        let analysis = body_of(&confirm(&env, url, analysis_token));
        assert_eq!(analysis["status"], "pending", "{analysis}");

        // The keep confirm must start its own job — not observe the running
        // analysis download and hand its caller someone else's response.
        let keep = body_of(&confirm(&env, url, keep_token));
        assert_eq!(keep["status"], "pending", "{keep}");
        assert_eq!(
            *env.fetcher.downloads.lock().expect("downloads"),
            2,
            "both jobs must be running"
        );

        release.send(()).expect("release analysis");
        release.send(()).expect("release keep");
        // Both outcomes surface across successive polls, each exactly once.
        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        let (mut saw_analysis, mut saw_keep) = (false, false);
        while !(saw_analysis && saw_keep) {
            let text = text_of(&call(&env, serde_json::json!({"source": url})));
            saw_analysis |= text.contains("Analysis rip downloaded");
            saw_keep |= text.contains("Saved to");
            assert!(
                std::time::Instant::now() < deadline,
                "downloads never settled (analysis: {saw_analysis}, keep: {saw_keep})"
            );
            std::thread::sleep(Duration::from_millis(20));
        }
    }

    #[test]
    fn detached_download_sends_no_notifications_after_pending() {
        let (env, release) = gated_env(vec![analysis_option()]);
        let url = "https://example.com/v.mp4";
        let (notifier, count) = counting_notifier();

        let offer = body_of(&call(&env, serde_json::json!({"source": url})));
        assert_eq!(offer["status"], "confirm_required", "{offer}");
        let token = offered_tokens(&offer).remove(0);
        let pending = body_of(&confirm_with_notify(&env, url, &token, &notifier));
        assert_eq!(pending["status"], "pending", "{pending}");
        let at_pending = count.load(std::sync::atomic::Ordering::SeqCst);

        release.send(()).expect("release download");
        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        loop {
            // Null-notifier polls: any notification from here on would have
            // to come from the detached thread — whose request is over.
            let body = body_of(&call(&env, serde_json::json!({"source": url})));
            if body["status"] == "plan" {
                break;
            }
            assert!(std::time::Instant::now() < deadline, "never settled");
            std::thread::sleep(Duration::from_millis(20));
        }
        assert_eq!(
            count.load(std::sync::atomic::Ordering::SeqCst),
            at_pending,
            "a completed request's progress token must never be used again"
        );
    }

    #[test]
    fn polling_call_reports_progress_through_its_own_notifier() {
        let (env, release) = gated_env(vec![analysis_option()]);
        let url = "https://example.com/v.mp4";

        let offer = body_of(&call(&env, serde_json::json!({"source": url})));
        assert_eq!(offer["status"], "confirm_required", "{offer}");
        let token = offered_tokens(&offer).remove(0);
        let pending = body_of(&confirm(&env, url, &token));
        assert_eq!(pending["status"], "pending", "{pending}");

        // The fake reported 1024/4096 before blocking on the gate; the
        // polling call renders that sample under its own token.
        let (notifier, count) = counting_notifier();
        let poll = body_of(&call_with_notify(
            &env,
            serde_json::json!({"source": url}),
            &notifier,
        ));
        assert_eq!(poll["status"], "pending", "{poll}");
        assert_eq!(
            count.load(std::sync::atomic::Ordering::SeqCst),
            1,
            "one running snapshot, one notification"
        );
        drop(release);
    }
}
