// SPDX-License-Identifier: Apache-2.0
//! Core invocation and transcription execution: build the core child's
//! argument vector, run it (JSON captured, exit class recovered from the
//! published exit-code contract), and the `transcribe: true` flow that
//! resolves local media, reuses or runs whisper (single-flight per
//! transcript key), and answers with a plan.

use std::path::Path;
use std::sync::Arc;

use rmcp::model::CallToolResult;

use super::deps::{Notifier, ServerDeps};
use super::handler::resolve_media;
use super::probe_cache::{cached_probe, effective_prompt_for, transcript_id_for};
use super::wire::{CoreMetadata, failed, json_result, ready_result};
use crate::fault::{Fault, FaultKind};
use crate::identity::fingerprint_file;
use crate::plan::{CoreInvocation, plan_response};
use crate::schema::{VideoArgs, VideoBody};
use crate::transcribe;
use agent_video_core::types::format_ts;

// ── core invocation ──

pub(super) fn run_invocation(
    media_path: &Path,
    invocation: &CoreInvocation,
    notices: &[String],
    deps: &ServerDeps,
) -> CallToolResult {
    // `--json` (pushed below) selects the stdout sink; no `-o` exists in
    // that mode.
    let mut args: Vec<String> = vec![
        invocation.subcommand.as_str().to_string(),
        media_path.to_string_lossy().into_owned(),
    ];
    if let Some(window) = invocation.window {
        args.extend([
            "--from".to_string(),
            window.start().to_string(),
            "--until".to_string(),
            window.end().to_string(),
        ]);
    }
    // Explicit pick, else a cached whisper transcript.
    let subtitle = invocation.subtitle.clone().or_else(|| {
        let fp = fingerprint_file(media_path).ok()?;
        deps.cache
            .find_transcript(&fp.id, &transcript_id_for(&fp.id, deps))
    });
    if let Some(subtitle) = subtitle {
        args.extend([
            "--subtitle".to_string(),
            subtitle.to_string_lossy().into_owned(),
        ]);
    }
    args.extend(invocation.flags.iter().cloned());
    args.push("--json".to_string());

    let run = match deps.core.run(&args) {
        Ok(run) => run,
        Err(e) => return failed(&Fault::internal(format!("core failed to start: {e}"))),
    };
    if !run.exit.is_success() {
        let detail: Vec<&str> = run.stderr.trim().lines().rev().take(5).collect();
        let detail: Vec<&str> = detail.into_iter().rev().collect();
        return failed(&Fault::new(
            core_exit_kind(run.exit),
            format!(
                "core {} failed ({}):\n{}",
                invocation.subcommand.as_str(),
                run.exit,
                detail.join("\n")
            ),
        ));
    }
    // BadInput, not transient: retrying the identical call would hit the
    // same capture cap; narrowing the request is an action the model
    // itself can take.
    if run.stdout_truncated {
        return failed(&Fault::bad_input(
            "core produced more output than one call can carry — narrow the \
             window or lower the frame density and call again",
        ));
    }
    // Same wire types on both sides and truncation handled above, so a
    // parse failure is a bug (stray bytes), not contract drift.
    match serde_json::from_str::<CoreMetadata>(&run.stdout) {
        Ok(meta) => ready_result(meta, notices),
        Err(e) => failed(&Fault::internal(format!(
            "core output failed to parse ({e}) — this is an agent-video bug"
        ))),
    }
}

/// Recover the failure class from a core child's exit status: the exit
/// code decades are core's published contract ([`ErrorCode::exit_code`]),
/// so a subprocess seam loses no more than the in-process one. Uncoded
/// exits (bugs, panics, clap usage — the harness built those arguments)
/// and signals are ours.
fn core_exit_kind(exit: super::core_run::CoreExit) -> FaultKind {
    use agent_video_core::error::ErrorCode;
    match exit {
        super::core_run::CoreExit::Failure(code) => u8::try_from(code.get())
            .ok()
            .and_then(ErrorCode::from_exit_code)
            .map(FaultKind::of)
            .unwrap_or(FaultKind::Internal),
        _ => FaultKind::Internal,
    }
}

// ── transcription execution ──

/// The `transcribe: true` flow: resolve local media, reuse a cached
/// transcript, else run whisper; answer with the refreshed plan. The
/// handler routes here on [`crate::schema::Intent::Transcribe`], so
/// exclusivity with `raw`/`confirm`/grid intent is already proven.
pub(super) fn transcribe_flow(
    args: &VideoArgs,
    deps: &Arc<ServerDeps>,
    notify: &Notifier,
) -> CallToolResult {
    let Some(media) = resolve_media(&args.source, deps) else {
        return failed(&Fault::bad_input(
            "transcription needs local media, and this URL has no cached rip. Call with just \
             the URL first, confirm a download, then repeat with transcribe.",
        ));
    };
    if !deps.whisper.available(&deps.config) {
        return failed(&Fault::environment(
            "no whisper transcription available — set WHISPER_API_KEY for API transcription, \
             or compile with a GPU feature and run `agent-video install-whisper` for local \
             inference",
        ));
    }
    let fingerprint = match fingerprint_file(&media) {
        Ok(fp) => fp.id,
        Err(e) => {
            return failed(&Fault::bad_input(format!(
                "cannot read {}: {e}",
                media.display()
            )));
        }
    };
    // The conditioning is content-determined (operator knobs × recorded
    // title), so the transcript identity — and with it the single-flight
    // key and the cache path — is the same for every caller of this
    // content.
    let prompt = effective_prompt_for(&fingerprint, deps);
    let transcript_id = transcribe::transcript_id(&deps.config, &prompt);
    let mut decoded_seconds: Option<f64> = None;
    {
        // Single-flight per transcript (content × recipe): the cache check
        // and the whisper run share one keyed critical section, so a
        // concurrent transcribe for the same content blocks here and then
        // takes the cache hit instead of running the model — and paying
        // the GPU load or the API bill — a second time. Blocking is this
        // flow's own shape: the winner's caller waits out the whole
        // inference anyway. The wait is per-waiter bounded, though: the
        // winner's legitimate wall clock compounds per chunk (hours for a
        // long track), and every waiter pins a `spawn_blocking` thread, so
        // a rival that outwaits its own bound answers "in flight" instead
        // of inheriting the winner's whole account.
        let Some(_admitted) = deps.transcriptions.admit_within(
            &format!("{fingerprint}.{transcript_id}"),
            deps.transcribe_wait,
        ) else {
            return failed(&Fault::transient(
                "a transcription of this content is already in flight; it lands in the shared \
                 cache when it finishes — call again with transcribe to adopt it (or to wait \
                 again)",
            ));
        };
        if deps
            .cache
            .find_transcript(&fingerprint, &transcript_id)
            .is_none()
        {
            let out = deps.cache.transcript_path(&fingerprint, &transcript_id);
            let mut step = 0.0f64;
            let notify = Arc::clone(notify);
            let result =
                deps.whisper
                    .transcribe(&deps.config, &media, &out, &prompt, &mut |message| {
                        step += 1.0;
                        notify(step, None, message.to_string());
                    });
            match result {
                Ok(outcome) => decoded_seconds = Some(outcome.audio_seconds),
                Err(e) => return failed(&e),
            }
        }
    }
    let (probe, _) = match cached_probe(&media, deps) {
        Ok(v) => v,
        Err(e) => return failed(&e),
    };
    // Honesty over silence: a transcript that stops mid-video says so.
    let truncation_note = decoded_seconds
        .filter(|&a| crate::probe::materially_truncated(a, probe.duration))
        .map(|a| {
            format!(
                "Warning: only {} of audio decoded for a {} video — the transcript stops \
                 there; the source's audio track is likely truncated or corrupt. ",
                format_ts(a as f32),
                format_ts(probe.duration as f32)
            )
        })
        .unwrap_or_default();
    let next = plan_response(&args.source, &probe, &deps.config);
    json_result(&VideoBody::Plan {
        probe: next.probe,
        guidance: format!(
            "{truncation_note}Transcript ready and cached — it rides along with every grid \
             from now on. {}",
            next.guidance
        ),
        recommended_calls: next.recommended_calls,
    })
}

#[cfg(test)]
mod tests {
    use crate::mcp::testutil::{body_of, call, env, media_file, offered_tokens, probe};

    /// Run the URL flow up to a cached analysis rip, then transcribe it.
    fn download_then_transcribe(env: &crate::mcp::testutil::TestEnv, url: &str) {
        let offer = body_of(&call(env, serde_json::json!({"source": url})));
        assert_eq!(offer["status"], "confirm_required", "{offer}");
        let token = offered_tokens(&offer).remove(0);
        let done = body_of(&call(
            env,
            serde_json::json!({"source": url, "confirm": token}),
        ));
        assert_eq!(done["status"], "plan", "{done}");
        let after = body_of(&call(
            env,
            serde_json::json!({"source": url, "transcribe": true}),
        ));
        assert_eq!(after["status"], "plan", "{after}");
    }

    fn titled_env() -> crate::mcp::testutil::TestEnv {
        use crate::fetch::SourceInfo;
        use crate::mcp::testutil::{FakeFetcher, analysis_option, env_with_fetcher};
        use std::sync::Mutex;
        env_with_fetcher(
            probe(3600.0, vec![]),
            true,
            FakeFetcher {
                info: SourceInfo {
                    headline: "\"Rust Ownership Explained\" — a video.".to_string(),
                    title: Some("Rust Ownership Explained".to_string()),
                    options: vec![analysis_option()],
                },
                gate: None,
                warning: None,
                downloads: Mutex::new(0),
                finished: Mutex::new(0),
                last_cancel: Mutex::new(None),
            },
        )
    }

    #[test]
    fn derived_title_prompt_conditions_url_transcription() {
        let env = titled_env();
        download_then_transcribe(&env, "https://example.com/v.mp4");
        assert_eq!(
            *env.whisper.prompts.lock().expect("prompts"),
            ["Rust Ownership Explained."],
            "the recorded title, shaped as continuation text, primes whisper"
        );
    }

    #[test]
    fn operator_prompt_wins_over_the_derived_title() {
        let mut env = titled_env();
        std::sync::Arc::get_mut(&mut env.deps)
            .expect("sole owner before threads spawn")
            .config
            .whisper
            .prompt = Some("Kubernetes, etcd, kubelet.".to_string());
        download_then_transcribe(&env, "https://example.com/v.mp4");
        assert_eq!(
            *env.whisper.prompts.lock().expect("prompts"),
            ["Kubernetes, etcd, kubelet."],
            "whisper.prompt overrides the derived default"
        );
    }

    #[test]
    fn local_media_transcribes_without_a_derived_prompt() {
        let env = env(probe(3600.0, vec![]), true);
        let file = media_file(&env, "talk.mp4");
        let after = body_of(&call(
            &env,
            serde_json::json!({"source": file.to_string_lossy(), "transcribe": true}),
        ));
        assert_eq!(after["status"], "plan", "{after}");
        assert_eq!(
            *env.whisper.prompts.lock().expect("prompts"),
            [""],
            "a filename is noise, not vocabulary — no conditioning for local files"
        );
    }

    #[test]
    fn concurrent_transcribes_run_whisper_once() {
        use std::sync::mpsc;
        let env = env(probe(3600.0, vec![]), true);
        let file = media_file(&env, "talk.mp4");
        let request = serde_json::json!({
            "source": file.to_string_lossy(),
            "transcribe": true,
        });
        let (entered_tx, entered_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        *env.whisper.gate.lock().expect("whisper gate") = Some((entered_tx, release_rx));

        let (winner, rival) = std::thread::scope(|s| {
            let winner = s.spawn(|| call(&env, request.clone()));
            entered_rx
                .recv()
                .expect("the first request reached whisper");
            let rival = s.spawn(|| call(&env, request.clone()));
            // Two releases on purpose: if single-flight were broken, the
            // second whisper run unblocks too and the test fails on the
            // call count instead of deadlocking.
            release_tx.send(()).expect("release the winner");
            let _ = release_tx.send(());
            (
                winner.join().expect("winner completes"),
                rival.join().expect("rival completes"),
            )
        });

        assert_eq!(
            *env.whisper.calls.lock().expect("calls"),
            1,
            "one content key, one whisper run — the rival adopts the cached transcript"
        );
        for (name, result) in [("winner", &winner), ("rival", &rival)] {
            let body = body_of(result);
            assert_eq!(body["status"], "plan", "{name}: {body}");
            assert!(
                body["guidance"]
                    .as_str()
                    .expect("guidance")
                    .contains("Transcript ready"),
                "{name}: {body}"
            );
        }
    }

    #[test]
    fn rival_outwaiting_the_bound_reports_in_flight() {
        use std::sync::mpsc;
        let mut env = env(probe(3600.0, vec![]), true);
        std::sync::Arc::get_mut(&mut env.deps)
            .expect("sole owner before threads spawn")
            .transcribe_wait = std::time::Duration::from_millis(50);
        let file = media_file(&env, "talk.mp4");
        let request = serde_json::json!({
            "source": file.to_string_lossy(),
            "transcribe": true,
        });
        let (entered_tx, entered_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        *env.whisper.gate.lock().expect("whisper gate") = Some((entered_tx, release_rx));

        std::thread::scope(|s| {
            let winner = s.spawn(|| call(&env, request.clone()));
            entered_rx
                .recv()
                .expect("the first request reached whisper");
            // The winner is wedged inside whisper; the rival must give up
            // at its own bound instead of riding out the winner's wall
            // clock on a pinned thread.
            let rival = body_of(&call(&env, request.clone()));
            assert_eq!(rival["kind"], "transient", "{rival}");
            assert!(
                rival["error"]
                    .as_str()
                    .expect("error text")
                    .contains("in flight"),
                "{rival}"
            );
            release_tx.send(()).expect("release the winner");
            let winner = body_of(&winner.join().expect("winner completes"));
            assert_eq!(winner["status"], "plan", "the winner is unaffected");
        });
        assert_eq!(
            *env.whisper.calls.lock().expect("calls"),
            1,
            "the refused rival never started a second whisper run"
        );
    }

    #[test]
    fn truncated_core_stdout_rejects_as_bad_input() {
        let env = env(probe(90.0, vec![]), false);
        let file = media_file(&env, "v.mp4");
        env.core
            .truncate
            .store(true, std::sync::atomic::Ordering::SeqCst);
        let r = call(&env, serde_json::json!({"source": file}));
        let body = body_of(&r);
        // Clipped output is the caller's request being too big, not a wire
        // bug: the model must be told to narrow, not to report an agent bug.
        assert_eq!(body["kind"], "bad_input", "{body}");
        assert!(
            body["error"]
                .as_str()
                .expect("error text")
                .contains("narrow"),
            "{body}"
        );
    }
}
