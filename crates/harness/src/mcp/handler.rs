// SPDX-License-Identifier: Apache-2.0
//! The `video` call flow, from coerced arguments to a wire result.
//!
//! Couriers one call through cached probe, policy ([`crate::plan`],
//! [`crate::url_policy`]), and execution: core runs (self-spawned, JSON
//! captured), downloads (grace window then detach into the
//! [`crate::mcp::tracker::Tracker`]), and whisper transcription. Everything
//! here is synchronous; the rmcp server hosts it via `spawn_blocking`.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use rmcp::model::CallToolResult;

use super::deps::{Notifier, ServerDeps};
use super::download::{DOWNLOAD_ABANDON_LIMIT, execute_download, keep_response};
use super::invoke::{run_invocation, transcribe_flow};
use super::probe_cache::cached_probe;
use super::tracker::Poll;
use super::wire::{coerce_args, failed, is_url, json_result, pending_result};
use crate::cache::url_key;
use crate::fault::Fault;
use crate::fetch::rip_key;
use crate::plan::{Decision, PlanDecision, decide, redeem};
use crate::schema::OfferedToken;
use crate::schema::{RecommendedCall, VideoArgs, VideoBody};
use crate::url_policy::decide_url;

// ── tool handler ──

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

    // Background downloads first: one poll resolves the slot's whole state
    // machine, so a URL can never sit on Pending for the whole session.
    if is_url(&args.source) {
        match deps
            .downloads
            .poll(&url_key(&args.source), DOWNLOAD_ABANDON_LIMIT)
        {
            Poll::Vacant => {}
            Poll::Settled(result) => return result,
            Poll::Running { downloaded, total } => return pending_result(downloaded, total),
            Poll::Abandoned => {
                return failed(&Fault::transient(
                    "the background download stopped reporting progress and was abandoned — \
                     call again with the URL to start over",
                ));
            }
        }
    }

    // Consent next: a URL source may not exist locally until its download runs.
    if args.confirm.is_some() {
        let decision = redeem(&args, &deps.tokens);
        return render_decision(&args, decision, None, Vec::new(), deps, notify);
    }

    // Transcription is an explicit request, not a consent dance: the
    // argument itself is the intent.
    if args.transcribe == Some(true) {
        return transcribe_flow(&args, deps, notify);
    }

    if is_url(&args.source) {
        return match deps
            .cache
            .find_download(&rip_key(&args.source, &deps.config))
        {
            None => url_first_contact(&args, deps, notify),
            Some(cached) => local_flow(
                &args,
                &cached,
                vec!["serving from the cached analysis rip".to_string()],
                deps,
                notify,
            ),
        };
    }
    let path = PathBuf::from(&args.source);
    local_flow(&args, &path, Vec::new(), deps, notify)
}

/// Probe a local media file and run the local decision engine.
fn local_flow(
    args: &VideoArgs,
    media_path: &Path,
    extra_notices: Vec<String>,
    deps: &Arc<ServerDeps>,
    notify: &Notifier,
) -> CallToolResult {
    let (probe, _) = match cached_probe(media_path, deps) {
        Ok(v) => v,
        Err(e) => return failed(&e),
    };
    let mut decision = decide(args, &probe, &deps.config);
    if let Decision::Plan(plan) = &mut decision {
        offer_transcription(&args.source, plan, deps);
    }
    render_decision(
        args,
        decision,
        Some(media_path.to_path_buf()),
        extra_notices,
        deps,
        notify,
    )
}

/// Subtitle-less plans grow a transcribe recommendation (or an install hint).
pub(super) fn offer_transcription(source: &str, plan: &mut PlanDecision, deps: &ServerDeps) {
    if !plan.probe.subtitles.is_empty() {
        return;
    }
    if deps.whisper.available(&deps.config) {
        plan.recommended_calls.push(RecommendedCall {
            call: VideoArgs {
                transcribe: Some(true),
                ..VideoArgs::bare(source)
            },
            note: "no subtitles found — generate a whisper transcript; it then rides \
                   along with every grid"
                .to_string(),
        });
    } else {
        plan.guidance.push_str(
            " No subtitles found and no whisper transcription available — set \
             WHISPER_API_KEY for API transcription, or compile with a GPU feature and run \
             `agent-video install-whisper` for local inference.",
        );
    }
}

/// First contact with a URL that has no cached rip: offer downloads.
fn url_first_contact(
    args: &VideoArgs,
    deps: &Arc<ServerDeps>,
    notify: &Notifier,
) -> CallToolResult {
    let info = match deps.fetcher.probe_url(&args.source, &deps.config) {
        Ok(i) => i,
        Err(e) => return failed(&e),
    };
    let decision = decide_url(args, &info, &deps.config);
    render_decision(args, decision, None, Vec::new(), deps, notify)
}

/// Turn a policy decision into the wire result, executing side effects.
fn render_decision(
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
        } => {
            // Commit here so "the model saw it" and "it is redeemable" are
            // the same event, from the same offers.
            let tokens: Vec<OfferedToken> = offers
                .iter()
                .map(|o| OfferedToken {
                    token: o.token.as_str().to_string(),
                    effect: o.effect.clone(),
                })
                .collect();
            deps.tokens.commit(&args.source, offers);
            json_result(&VideoBody::ConfirmRequired {
                consequence,
                tokens,
            })
        }
        Decision::Run {
            invocation,
            notices,
        } => {
            let resolved = media_path.or_else(|| resolve_media(&args.source, deps));
            let Some(resolved) = resolved else {
                return failed(&Fault::transient(
                    "the cached rip for this URL is gone (evicted). Call again with just the \
                     URL to re-download.",
                ));
            };
            let mut all = extra_notices;
            all.extend(notices);
            run_invocation(&resolved, &invocation, &all, deps)
        }
        Decision::Download { plan, token, .. } => {
            execute_download(&args.source, plan, token, deps, notify)
        }
        // A keep token echoed after its download completed: replay the
        // receipt instead of ripping a duplicate into the keep dir.
        Decision::Saved { path } => {
            if path.exists() {
                keep_response(&path, None, deps)
            } else {
                failed(&Fault::bad_input(format!(
                    "this download already completed and was saved to {}, but the file has \
                     since been moved or removed. Call again without confirm for a fresh plan.",
                    path.display()
                )))
            }
        }
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
    fn long_video_plans_and_offers_transcription() {
        let env = env(probe(3600.0, vec![]), true);
        let file = media_file(&env, "long.mp4");
        let r = call(&env, serde_json::json!({"source": file}));
        let body = body_of(&r);
        assert_eq!(body["status"], "plan");
        let calls = body["recommended_calls"]
            .as_array()
            .expect("recommended calls");
        assert!(
            calls.iter().any(|c| c["call"]["transcribe"] == true),
            "{body}"
        );
    }

    #[test]
    fn plan_without_whisper_hints_install() {
        let env = env(probe(3600.0, vec![]), false);
        let file = media_file(&env, "long.mp4");
        let body = body_of(&call(&env, serde_json::json!({"source": file})));
        assert!(
            body["guidance"]
                .as_str()
                .expect("guidance")
                .contains("install-whisper"),
            "{body}"
        );
    }

    #[test]
    fn transcribe_arg_runs_whisper_and_rides_along() {
        let env = env(probe(3600.0, vec![]), true);
        let file = media_file(&env, "talk.mp4");
        let source = file.to_string_lossy().into_owned();

        let after = body_of(&call(
            &env,
            serde_json::json!({"source": source, "transcribe": true}),
        ));
        assert_eq!(after["status"], "plan", "{after}");
        assert!(
            after["guidance"]
                .as_str()
                .expect("guidance")
                .contains("Transcript ready"),
            "{after}"
        );
        assert!(
            after["probe"]["subtitles"]
                .as_array()
                .expect("subtitles")
                .iter()
                .any(|s| s == "transcript (whisper)"),
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
            serde_json::json!({"source": source, "transcribe": true}),
        ));
        assert_eq!(first["status"], "plan", "{first}");
        let again = body_of(&call(
            &env,
            serde_json::json!({"source": source, "transcribe": true}),
        ));
        assert_eq!(again["status"], "plan", "{again}");
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
            serde_json::json!({"source": file, "transcribe": true}),
        ));
        assert_eq!(body["status"], "plan", "{body}");
        let guidance = body["guidance"].as_str().expect("guidance");
        assert!(
            guidance.starts_with("Warning: only 00:49"),
            "truncation leads the guidance: {guidance}"
        );
        assert!(guidance.contains("Transcript ready"), "{guidance}");
    }

    #[test]
    fn transcribe_with_intent_rejects() {
        let env = env(probe(90.0, vec![]), true);
        let file = media_file(&env, "v.mp4");
        let r = call(
            &env,
            serde_json::json!({"source": file, "transcribe": true, "look": "overview"}),
        );
        let body = body_of(&r);
        assert_eq!(body["kind"], "bad_input", "{body}");
        assert!(
            body["error"].as_str().expect("error").contains("alone"),
            "{body}"
        );
    }

    #[test]
    fn transcribe_without_whisper_reports_environment() {
        let env = env(probe(90.0, vec![]), false);
        let file = media_file(&env, "v.mp4");
        let r = call(
            &env,
            serde_json::json!({"source": file, "transcribe": true}),
        );
        let body = body_of(&r);
        assert_eq!(body["kind"], "environment", "{body}");
        assert!(
            body["error"]
                .as_str()
                .expect("error")
                .contains("install-whisper"),
            "{body}"
        );
    }

    #[test]
    fn transcribe_on_undownloaded_url_rejects() {
        let env = env(probe(90.0, vec![]), true);
        let r = call(
            &env,
            serde_json::json!({"source": "https://example.com/v.mp4", "transcribe": true}),
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
        let done = body_of(&call(
            &env,
            serde_json::json!({"source": url, "confirm": "dl-analysis"}),
        ));
        assert_eq!(done["status"], "plan", "{done}");

        // The token survives redemption; echoing it again must not refetch.
        let again = body_of(&call(
            &env,
            serde_json::json!({"source": url, "confirm": "dl-analysis"}),
        ));
        assert_eq!(again["status"], "plan", "{again}");
        assert_eq!(
            *env.fetcher.downloads.lock().expect("downloads"),
            1,
            "re-confirmation must serve the cached rip"
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
                    options: vec![analysis_option()],
                },
                gate: None,
                warning: Some("subtitles could not be fetched (login required)".to_string()),
                downloads: Mutex::new(0),
            },
        );
        let url = "https://example.com/v.mp4";

        let offer = body_of(&call(&env, serde_json::json!({"source": url})));
        assert_eq!(offer["status"], "confirm_required", "{offer}");

        let done = body_of(&call(
            &env,
            serde_json::json!({"source": url, "confirm": "dl-analysis"}),
        ));
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
        assert_eq!(offer["tokens"][0]["token"], "dl-analysis");

        let done = body_of(&call(
            &env,
            serde_json::json!({"source": url, "confirm": "dl-analysis"}),
        ));
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
                    options: vec![analysis_option()],
                },
                gate: Some(Mutex::new(gate)),
                warning: None,
                downloads: Mutex::new(0),
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

        let pending = body_of(&call(
            &env,
            serde_json::json!({"source": url, "confirm": "dl-analysis"}),
        ));
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
                    options: vec![analysis_option()],
                },
                gate: Some(Mutex::new(gate)),
                warning: None,
                downloads: Mutex::new(0),
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
        let pending = body_of(&call(
            &env,
            serde_json::json!({"source": url, "confirm": "dl-analysis"}),
        ));
        assert_eq!(pending["status"], "pending", "{pending}");

        // Rewind the job's progress clock past the abandonment limit — the
        // downloader-side watchdogs have failed and the thread is wedged.
        env.deps.downloads.age(
            &url_key(url),
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
    fn stale_confirm_token_rejects() {
        let env = env(probe(90.0, vec![]), false);
        let file = media_file(&env, "v.mp4");
        let r = call(&env, serde_json::json!({"source": file, "confirm": "nope"}));
        assert!(text_of(&r).contains("stale"), "{}", text_of(&r));
    }
}
