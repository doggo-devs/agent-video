// SPDX-License-Identifier: Apache-2.0
//! Core invocation and transcription execution: build the core child's
//! argument vector, run it (JSON captured, exit class recovered from the
//! published exit-code contract), and the `transcribe: true` flow that
//! resolves local media, reuses or runs whisper, and answers with a plan.

use std::path::Path;
use std::sync::Arc;

use rmcp::model::CallToolResult;

use super::deps::{Notifier, ServerDeps};
use super::handler::resolve_media;
use super::probe_cache::cached_probe;
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
    if let Some((start, end)) = invocation.window {
        args.extend([
            "--from".to_string(),
            start.to_string(),
            "--until".to_string(),
            end.to_string(),
        ]);
    }
    // Explicit pick, else a cached whisper transcript.
    let subtitle = invocation.subtitle.clone().or_else(|| {
        let fp = fingerprint_file(media_path).ok()?;
        deps.cache
            .find_transcript(&fp.id, &transcribe::transcript_id(&deps.config))
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
/// transcript, else run whisper; answer with the refreshed plan.
pub(super) fn transcribe_flow(
    args: &VideoArgs,
    deps: &Arc<ServerDeps>,
    notify: &Notifier,
) -> CallToolResult {
    if args.raw.is_some() || args.has_intent() {
        return failed(&Fault::bad_input(
            "transcribe must be passed alone (with source); once the transcript exists it \
             rides along with every grid call automatically",
        ));
    }
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
    let transcript_id = transcribe::transcript_id(&deps.config);
    let mut decoded_seconds: Option<f64> = None;
    if deps
        .cache
        .find_transcript(&fingerprint, &transcript_id)
        .is_none()
    {
        let out = deps.cache.transcript_path(&fingerprint, &transcript_id);
        let mut step = 0.0f64;
        let notify = Arc::clone(notify);
        let result = deps
            .whisper
            .transcribe(&deps.config, &media, &out, &mut |message| {
                step += 1.0;
                notify(step, None, message.to_string());
            });
        match result {
            Ok(outcome) => decoded_seconds = Some(outcome.audio_seconds),
            Err(e) => return failed(&e),
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
    use crate::mcp::testutil::{body_of, call, env, media_file, probe};

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
