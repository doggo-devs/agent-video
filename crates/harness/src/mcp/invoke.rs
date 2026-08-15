// SPDX-License-Identifier: Apache-2.0
//! Core invocation and transcription execution: build the core child's
//! argument vector, run it (JSON captured, exit class recovered from the
//! published exit-code contract), and the `transcribe: true` flow that
//! resolves local media, reuses or runs whisper (single-flight per
//! transcript key), and answers with the transcript itself.

use std::path::Path;
use std::sync::Arc;

use rmcp::model::CallToolResult;

use super::deps::{Notifier, ServerDeps};
use super::handler::resolve_media;
use super::probe_cache::{
    MediaMissing, ProbeError, effective_prompt_for, probe_media, transcript_id_for,
};
use super::wire::{Wire, evicted_rip, failed, is_url, json_result};
use crate::core_run::run_core;
use crate::fault::Fault;
use crate::identity::{ContentId, fingerprint_file};
use crate::plan::{CoreInvocation, Sink, plan_response};
use crate::schema::{TimeValue, VideoArgs, VideoBody};
use crate::time::{ArgError, Window, resolve_window};
use crate::transcribe;
use agent_video_core::types::format_ts;

// ── core invocation ──

pub(super) fn run_invocation(
    media_path: &Path,
    invocation: &CoreInvocation,
    notices: &[String],
    deps: &ServerDeps,
) -> CallToolResult {
    // Explicit pick, else the cached whisper transcript — looked up through
    // the transcription gate, so a grid landing while whisper still runs
    // waits for the transcript instead of silently rendering without it.
    // The wait is this waiter's own bound; a render that outlasts it
    // proceeds and says so in a notice rather than staying silent. The
    // hold is dropped before core runs: the gate guards the rendezvous,
    // not the render.
    //
    // This is the concurrency the server owns, which is why it lives here
    // and not in `core_run`: a one-shot caller has no rival to wait for.
    let mut notices = notices.to_vec();
    let mut transcript_in_flight = false;
    let subtitle = invocation.subtitle.clone().or_else(|| {
        let fp = fingerprint_file(media_path).ok()?;
        let transcript_id = transcript_id_for(&fp.id, deps);
        let key = transcript_gate_key(&fp.id, &transcript_id);
        match deps.transcriptions.admit_within(&key, deps.transcribe_wait) {
            Some(_hold) => deps.cache.find_transcript(&fp.id, &transcript_id),
            None => {
                transcript_in_flight = true;
                None
            }
        }
    });
    if transcript_in_flight {
        notices.push(
            "a transcription of this content is still in flight; this render carries no \
             subtitles — repeat the call once it lands to re-render with them"
                .to_string(),
        );
    }
    match run_core(
        deps.core.as_ref(),
        media_path,
        invocation,
        subtitle.as_deref(),
        &Sink::Json,
    ) {
        Ok(meta) => Wire::new(&deps.config).ready(meta, &notices),
        Err(fault) => failed(&fault),
    }
}

/// The single-flight key for one transcript: content × recipe. One
/// definition for the transcribe flow (the holder) and the grid flow (the
/// rendezvous), so the two sides cannot key apart.
fn transcript_gate_key(fingerprint: &ContentId, transcript_id: &str) -> String {
    format!("{fingerprint}.{transcript_id}")
}

// ── transcription execution ──

/// The `transcribe: true` flow: resolve local media, reuse a cached
/// transcript, else run whisper; answer with the transcript text. The
/// handler routes here on [`crate::schema::Intent::Transcribe`], so
/// exclusivity with `raw`/`confirm`/grid intent is already proven.
pub(super) fn transcribe_flow(
    args: &VideoArgs,
    from: Option<TimeValue>,
    until: Option<TimeValue>,
    deps: &Arc<ServerDeps>,
    notify: &Notifier,
) -> CallToolResult {
    let Some(media) = resolve_media(&args.source, deps) else {
        return failed(&no_cached_rip());
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
        // A cache-derived rip that vanished between resolve and open was
        // evicted underfoot by a concurrent process: a transient race,
        // never the caller's arguments — unlike the never-downloaded
        // resolve miss above.
        Err(e) if e.kind() == std::io::ErrorKind::NotFound && is_url(&args.source) => {
            return failed(&evicted_rip());
        }
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
            &transcript_gate_key(&fingerprint, &transcript_id),
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
    let (probe, _) = match probe_media(&media, deps) {
        Ok(v) => v,
        // Same eviction race as above, later in the flow: the transcript
        // already landed in the shared cache, so re-download-then-
        // transcribe adopts it without another whisper run.
        Err(ProbeError::Missing(MediaMissing)) if is_url(&args.source) => {
            return failed(&evicted_rip());
        }
        Err(ProbeError::Missing(MediaMissing)) => {
            return failed(&Fault::bad_input(format!(
                "file not found: {}",
                media.display()
            )));
        }
        Err(ProbeError::Fault(e)) => return failed(&e),
    };
    // The transcript is this call's product, not a side effect: read it
    // back and answer with the text itself. With age-sweep removals vetoed
    // by a concurrent touch ([`crate::cache::Cache::evict`]), the only
    // remaining way for it to be absent here is the narrow
    // recheck-to-unlink window of a rival process's sweep — a benign race
    // at the age boundary, not a bug: a retry regenerates it.
    let Some(srt_path) = deps.cache.find_transcript(&fingerprint, &transcript_id) else {
        return failed(&transcript_swept());
    };
    let srt = match std::fs::read_to_string(&srt_path) {
        Ok(s) => s,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            return failed(&transcript_swept());
        }
        Err(e) => {
            return failed(&Fault::internal(format!(
                "cannot read the cached transcript {}: {e}",
                srt_path.display()
            )));
        }
    };
    // The window is resolved against the real duration, so an out-of-range
    // slice is rejected here rather than answering with silence.
    let window = match resolve_window(from.as_ref(), until.as_ref(), probe.duration) {
        Ok((window, _)) => window,
        Err(ArgError(message)) => return failed(&Fault::bad_input(message)),
    };
    let transcript = render_transcript(&srt, window);
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
    json_result(&VideoBody::Transcript {
        transcript,
        guidance: format!(
            "{truncation_note}Transcript cached — it rides along with every grid \
             from now on. {}",
            next.guidance
        ),
        recommended_calls: next.recommended_calls,
    })
}

/// The transcription miss for a URL that was never downloaded: `resolve`
/// found no rip, so the model skipped the consent dance — fixing the call
/// sequence is the model's own move, hence `bad_input`. A rip that
/// resolved and *then* vanished is a different state — the transient
/// [`evicted_rip`], constructed in one place for every flow.
fn no_cached_rip() -> Fault {
    Fault::bad_input(
        "transcription needs local media, and this URL has no cached rip. Call with just \
         the URL first, confirm a download, then repeat with transcribe.",
    )
}

/// A transcript removed by a concurrent cache sweep between its creation
/// (or adoption) and its readback: transient, not an agent-video bug.
fn transcript_swept() -> Fault {
    Fault::transient(
        "the transcript was evicted by a concurrent cache sweep just as it was being read \
         back — call again with transcribe to regenerate it",
    )
}

/// Render the cached SRT as `MM:SS text` lines, keeping the cues that
/// overlap `window`. Timestamps stay absolute to the source, so a slice
/// lines up with the grids of the same range.
fn render_transcript(srt: &str, window: Window) -> String {
    let mut out = String::new();
    for cue in agent_video_core::subtitle::parse_srt(srt) {
        if f64::from(cue.end) < window.start() || f64::from(cue.start) > window.end() {
            continue;
        }
        out.push_str(&format!("{} {}\n", format_ts(cue.start), cue.text));
    }
    out.trim_end().to_string()
}

#[cfg(test)]
mod tests {
    use crate::mcp::testutil::{
        body_of, call, confirm, env, media_file, offered_tokens, probe, text_of,
    };

    /// Run the URL flow up to a cached analysis rip, then transcribe it.
    fn download_then_transcribe(env: &crate::mcp::testutil::TestEnv, url: &str) {
        let offer = body_of(&call(env, serde_json::json!({"source": url})));
        assert_eq!(offer["status"], "confirm_required", "{offer}");
        let token = offered_tokens(&offer).remove(0);
        let done = body_of(&confirm(env, url, &token));
        assert_eq!(done["status"], "plan", "{done}");
        let after = body_of(&call(
            env,
            serde_json::json!({"source": url, "want": "transcript"}),
        ));
        assert_eq!(after["status"], "transcript", "{after}");
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
                short_attempts: Mutex::new(0),
            },
        )
    }

    #[test]
    fn transcribe_after_full_eviction_is_indistinguishable_from_a_miss() {
        // The rip was evicted *before* this call resolved it: the flow
        // observes exactly what a never-downloaded URL looks like, so it
        // must answer the resolve miss — never diagnose a broken file
        // argument. (An eviction observed *mid-flow* is distinguishable
        // and transient; see the underfoot test below.)
        let env = titled_env();
        let url = "https://example.com/v.mp4";
        download_then_transcribe(&env, url);
        let downloads = env.deps.config.paths.cache.join("downloads");
        for entry in std::fs::read_dir(&downloads).expect("downloads dir") {
            std::fs::remove_file(entry.expect("entry").path()).expect("evict the rip");
        }

        let body = crate::mcp::testutil::body_of(&call(
            &env,
            serde_json::json!({"source": url, "want": "transcript"}),
        ));
        assert_eq!(body["kind"], "bad_input", "{body}");
        let error = body["error"].as_str().expect("error text");
        assert!(
            error.contains("no cached rip"),
            "the miss names its healing action: {error}"
        );
        assert!(
            !error.contains("cannot read") && !error.contains("file not found"),
            "a cache-internal path must never be diagnosed to the model: {error}"
        );
    }

    #[test]
    fn transcribe_rip_evicted_underfoot_answers_transient_with_the_healing_action() {
        // The rip resolved, then vanished while whisper held the flow open
        // (a concurrent process's eviction): a transient race, never the
        // caller's arguments — a kind-branching agent must keep retrying,
        // and never see a cache-internal path diagnosed.
        use std::sync::mpsc;
        let env = titled_env();
        let url = "https://example.com/v.mp4";
        let offer = body_of(&call(&env, serde_json::json!({"source": url})));
        assert_eq!(offer["status"], "confirm_required", "{offer}");
        let token = offered_tokens(&offer).remove(0);
        let done = body_of(&confirm(&env, url, &token));
        assert_eq!(done["status"], "plan", "{done}");

        let (entered_tx, entered_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        *env.whisper.gate.lock().expect("whisper gate") = Some((entered_tx, release_rx));

        let body = std::thread::scope(|s| {
            let transcriber = s.spawn(|| {
                call(
                    &env,
                    serde_json::json!({"source": url, "want": "transcript"}),
                )
            });
            entered_rx
                .recv()
                .expect("the transcription reached whisper");
            let downloads = env.deps.config.paths.cache.join("downloads");
            for entry in std::fs::read_dir(&downloads).expect("downloads dir") {
                std::fs::remove_file(entry.expect("entry").path()).expect("evict the rip");
            }
            release_tx.send(()).expect("release whisper");
            body_of(&transcriber.join().expect("transcribe completes"))
        });

        assert_eq!(body["kind"], "transient", "{body}");
        let error = body["error"].as_str().expect("error text");
        assert!(
            error.contains("URL"),
            "the race names its healing action: {error}"
        );
        assert!(
            !error.contains("cannot read") && !error.contains("file not found"),
            "a cache-internal path must never be diagnosed to the model: {error}"
        );
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
            serde_json::json!({"source": file.to_string_lossy(), "want": "transcript"}),
        ));
        assert_eq!(after["status"], "transcript", "{after}");
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
            "want": "transcript",
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
            assert_eq!(body["status"], "transcript", "{name}: {body}");
            assert_eq!(
                body["transcript"], "00:00 hi",
                "{name}: both callers answer with the cached transcript text"
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
            "want": "transcript",
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
            assert_eq!(winner["status"], "transcript", "the winner is unaffected");
        });
        assert_eq!(
            *env.whisper.calls.lock().expect("calls"),
            1,
            "the refused rival never started a second whisper run"
        );
    }

    #[test]
    fn grid_render_waits_for_an_in_flight_transcription() {
        use std::sync::mpsc;
        let env = env(probe(90.0, vec![]), true);
        let file = media_file(&env, "v.mp4");
        let source = file.to_string_lossy().into_owned();
        let (entered_tx, entered_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        *env.whisper.gate.lock().expect("whisper gate") = Some((entered_tx, release_rx));

        std::thread::scope(|s| {
            let transcriber = s.spawn(|| {
                call(
                    &env,
                    serde_json::json!({"source": source.clone(), "want": "transcript"}),
                )
            });
            entered_rx
                .recv()
                .expect("the transcription reached whisper");
            // The grid call lands mid-transcription: it must rendezvous on
            // the transcript gate and attach the fresh transcript, never
            // render silently without it.
            let grid = s.spawn(|| call(&env, serde_json::json!({"source": source.clone()})));
            release_tx.send(()).expect("release whisper");
            let grid = grid.join().expect("grid completes");
            assert_ne!(grid.is_error, Some(true), "{}", text_of(&grid));
            transcriber.join().expect("transcribe completes");
        });

        let calls = env.core.calls.lock().expect("calls");
        let last = calls.last().expect("core ran");
        assert!(
            last.iter().any(|a| a == "--subtitle"),
            "the mid-flight transcript must ride along: {last:?}"
        );
    }

    #[test]
    fn grid_render_notes_a_transcription_it_outwaited() {
        use std::sync::mpsc;
        let mut env = env(probe(90.0, vec![]), true);
        std::sync::Arc::get_mut(&mut env.deps)
            .expect("sole owner before threads spawn")
            .transcribe_wait = std::time::Duration::from_millis(50);
        let file = media_file(&env, "v.mp4");
        let source = file.to_string_lossy().into_owned();
        let (entered_tx, entered_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        *env.whisper.gate.lock().expect("whisper gate") = Some((entered_tx, release_rx));

        std::thread::scope(|s| {
            let transcriber = s.spawn(|| {
                call(
                    &env,
                    serde_json::json!({"source": source.clone(), "want": "transcript"}),
                )
            });
            entered_rx
                .recv()
                .expect("the transcription reached whisper");
            // The waiter outlasts its own bound: the render proceeds, but
            // the silent-degradation state is gone — the response says the
            // transcript is still in flight.
            let grid = call(&env, serde_json::json!({"source": source.clone()}));
            assert_ne!(grid.is_error, Some(true), "{}", text_of(&grid));
            assert!(
                text_of(&grid).contains("still in flight"),
                "the render must disclose the missing transcript: {}",
                text_of(&grid)
            );
            release_tx.send(()).expect("release whisper");
            transcriber.join().expect("transcribe completes");
        });

        let calls = env.core.calls.lock().expect("calls");
        let grid_run = calls
            .iter()
            .find(|args| args.iter().any(|a| a == "process"))
            .expect("core ran the grid");
        assert!(
            !grid_run.iter().any(|a| a == "--subtitle"),
            "an outwaited transcript cannot half-attach: {grid_run:?}"
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
