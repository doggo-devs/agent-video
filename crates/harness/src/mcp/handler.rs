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
use std::time::{Duration, SystemTime};

use rmcp::model::CallToolResult;

use super::deps::{Notifier, ServerDeps};
use super::tracker::Poll;
use super::wire::{
    CoreMetadata, coerce_args, failed, is_url, json_result, pending_result, ready_result,
};
use crate::cache::url_key;
use crate::fault::{Fault, FaultKind};
use crate::fetch::{DownloadIo, ProgressFn, rip_key};
use crate::identity::{ContentId, fingerprint_file};
use crate::plan::{
    CoreInvocation, Decision, PlanDecision, decide, format_bytes, plan_response, redeem, size_label,
};
use crate::schema::OfferedToken;
use crate::schema::{ProbeSummary, RecommendedCall, VideoArgs, VideoBody};
use crate::tokens::{Dest, DownloadPlan, PendingAction, Token, TranscribePlan};
use crate::url_policy::decide_url;

// ── probe with cache ──

/// On-disk probe document: a version-tagged envelope, so a summary written
/// under an older schema — or older field semantics — misses instead of
/// being read as current. Bump the version when [`ProbeSummary`] changes
/// shape or meaning; pre-envelope documents lack the tag and miss too.
#[derive(serde::Serialize, serde::Deserialize)]
struct ProbeDoc {
    v: u32,
    probe: ProbeSummary,
}

const PROBE_DOC_VERSION: u32 = 1;

fn parse_probe_doc(text: &str) -> Option<ProbeSummary> {
    serde_json::from_str::<ProbeDoc>(text)
        .ok()
        .filter(|doc| doc.v == PROBE_DOC_VERSION)
        .map(|doc| doc.probe)
}

fn render_probe_doc(probe: &ProbeSummary) -> Option<String> {
    serde_json::to_string(&ProbeDoc {
        v: PROBE_DOC_VERSION,
        probe: probe.clone(),
    })
    .ok()
}

/// Probe via content-fingerprint cache; unstable files are probed fresh,
/// never cached. A cached whisper transcript decorates the summary (never
/// the cache entry).
fn cached_probe(file: &Path, deps: &ServerDeps) -> Result<(ProbeSummary, ContentId), Fault> {
    let fp = fingerprint_file(file).map_err(|e| match e.kind() {
        std::io::ErrorKind::NotFound => {
            Fault::bad_input(format!("file not found: {}", file.display()))
        }
        _ => Fault::bad_input(format!("cannot read {}: {e}", file.display())),
    })?;
    let cached: Option<ProbeSummary> = fp
        .stable
        .then(|| deps.cache.get_probe(&fp.id).ok().flatten())
        .flatten()
        .and_then(|doc| parse_probe_doc(&doc));
    let mut probe = match cached {
        Some(p) => p,
        None => {
            let p = deps.prober.probe(file)?;
            if fp.stable
                && let Some(doc) = render_probe_doc(&p)
            {
                let _ = deps.cache.put_probe(&fp.id, &doc);
            }
            p
        }
    };
    if deps
        .cache
        .find_transcript(&fp.id, &deps.config.whisper.model.cache_id())
        .is_some()
    {
        probe.subtitles.push("transcript (whisper)".to_string());
    }
    Ok((probe, fp.id))
}

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
    let (probe, fingerprint) = match cached_probe(media_path, deps) {
        Ok(v) => v,
        Err(e) => return failed(&e),
    };
    let mut decision = decide(args, &probe, &deps.config);
    if let Decision::Plan(plan) = &mut decision {
        offer_transcription(&args.source, media_path, &fingerprint, plan, deps);
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

/// Subtitle-less plans grow a transcribe offer (or an install hint).
fn offer_transcription(
    source: &str,
    media_path: &Path,
    fingerprint: &ContentId,
    plan: &mut PlanDecision,
    deps: &ServerDeps,
) {
    if !plan.probe.subtitles.is_empty() {
        return;
    }
    if deps.whisper.available(&deps.config) {
        deps.tokens.offer_one(
            source,
            Token::new("transcribe"),
            PendingAction::Transcribe(TranscribePlan {
                file: media_path.to_path_buf(),
                fingerprint: fingerprint.clone(),
            }),
        );
        plan.recommended_calls.push(RecommendedCall {
            call: VideoArgs {
                confirm: Some("transcribe".to_string()),
                ..VideoArgs::bare(source)
            },
            note: "no subtitles found — generate a whisper transcript locally (CPU-bound, can \
                   take minutes); it then rides along with every grid"
                .to_string(),
        });
    } else {
        plan.guidance.push_str(
            " No subtitles found and no whisper model installed — run `agent-video \
             install-whisper` to enable transcription.",
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
        Decision::Download { plan, .. } => execute_download(&args.source, plan, deps, notify),
        Decision::Transcribe { plan, .. } => execute_transcribe(&args.source, &plan, deps, notify),
    }
}

/// Local paths resolve to themselves; URLs to their cached rip — under the
/// *current* recipe — if any.
fn resolve_media(source: &str, deps: &ServerDeps) -> Option<PathBuf> {
    if !is_url(source) {
        return Some(PathBuf::from(source));
    }
    deps.cache.find_download(&rip_key(source, &deps.config))
}

// ── core invocation ──

/// `-o` is mandatory in the core CLI but never touched in `--json` mode.
fn unused_out_dir() -> PathBuf {
    std::env::temp_dir().join("agent-video-json-unused")
}

fn run_invocation(
    media_path: &Path,
    invocation: &CoreInvocation,
    notices: &[String],
    deps: &ServerDeps,
) -> CallToolResult {
    let mut args: Vec<String> = vec![
        invocation.subcommand.as_str().to_string(),
        media_path.to_string_lossy().into_owned(),
        "--output".to_string(),
        unused_out_dir().to_string_lossy().into_owned(),
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
            .find_transcript(&fp.id, &deps.config.whisper.model.cache_id())
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
    // Same wire types on both sides, so a parse failure is a bug (truncated
    // stdout, stray bytes), not contract drift.
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

/// Run whisper against local media; answer with the refreshed plan.
fn execute_transcribe(
    source: &str,
    plan: &TranscribePlan,
    deps: &Arc<ServerDeps>,
    notify: &Notifier,
) -> CallToolResult {
    let out = deps
        .cache
        .transcript_path(&plan.fingerprint, &deps.config.whisper.model.cache_id());
    let mut step = 0.0f64;
    let notify = Arc::clone(notify);
    let result = deps
        .whisper
        .transcribe(&deps.config, &plan.file, &out, &mut |message| {
            step += 1.0;
            notify(step, None, message.to_string());
        });
    if let Err(e) = result {
        return failed(&e);
    }
    let (probe, _) = match cached_probe(&plan.file, deps) {
        Ok(v) => v,
        Err(e) => return failed(&e),
    };
    let next = plan_response(source, &probe, &deps.config);
    json_result(&VideoBody::Plan {
        probe: next.probe,
        guidance: format!(
            "Transcript generated and cached — it rides along with every grid from now on. {}",
            next.guidance
        ),
        recommended_calls: next.recommended_calls,
    })
}

// ── download execution ──

/// A job with no progress for this long is presumed wedged and abandoned on
/// the next poll. Longer than every downloader-side stall bound (the yt-dlp
/// watchdog, ureq's phase timeouts), so it only fires when those failed to.
const DOWNLOAD_ABANDON_LIMIT: Duration = Duration::from_secs(10 * 60);

/// Start the download on its own thread, then wait out a sync grace window:
/// fast downloads answer inline, slow ones detach into the tracker and are
/// polled by later calls.
fn execute_download(
    source: &str,
    plan: DownloadPlan,
    deps: &Arc<ServerDeps>,
    notify: &Notifier,
) -> CallToolResult {
    let key = url_key(source);
    let reporter = deps.downloads.start(&key);
    let (tx, rx) = std::sync::mpsc::channel::<CallToolResult>();
    {
        let deps = Arc::clone(deps);
        let source = source.to_string();
        let notify = Arc::clone(notify);
        std::thread::spawn(move || {
            let mut progress = |bytes: u64, total: Option<u64>| {
                reporter.progress(bytes, total);
                let total_text = total
                    .map(|t| format!(" / {}", format_bytes(t)))
                    .unwrap_or_default();
                notify(
                    bytes as f64,
                    total.map(|t| t as f64),
                    format!("downloading: {}{total_text}", format_bytes(bytes)),
                );
            };
            let result = perform_download(&source, &plan, &deps, &mut progress);
            reporter.settle(result.clone());
            // Receiver gone after the grace window: the settled job waits in
            // the tracker for the next poll.
            let _ = tx.send(result);
        });
    }
    let window = Duration::from_secs(u64::from(deps.config.download.sync_wait_seconds.get()));
    match rx.recv_timeout(window) {
        Ok(result) => {
            deps.downloads.clear(&key);
            result
        }
        // Timed out (slow detach) or the thread died; one poll resolves either.
        Err(_) => match deps.downloads.poll(&key, DOWNLOAD_ABANDON_LIMIT) {
            Poll::Settled(result) => result,
            Poll::Running { downloaded, total } => pending_result(downloaded, total),
            Poll::Vacant | Poll::Abandoned => failed(&Fault::transient(
                "the download thread died before reporting a result — call again to retry",
            )),
        },
    }
}

/// The blocking download flow: fetch, respond, sweep; known errors become
/// `Failed`.
fn perform_download(
    source: &str,
    plan: &DownloadPlan,
    deps: &Arc<ServerDeps>,
    progress: ProgressFn<'_>,
) -> CallToolResult {
    let downloaded = match plan.dest {
        Dest::Cache => deps
            .cache
            .ensure_downloads_dir()
            .map_err(|e| Fault::environment(format!("cannot create the downloads directory: {e}")))
            .and_then(|dir| {
                let key = rip_key(&plan.url, &deps.config);
                deps.fetcher.download(
                    plan,
                    DownloadIo::Cache { dir, key: &key },
                    &deps.config,
                    progress,
                )
            }),
        Dest::Keep => std::fs::create_dir_all(&deps.config.paths.keep)
            .map_err(|e| {
                Fault::environment(format!(
                    "cannot create {}: {e}",
                    deps.config.paths.keep.display()
                ))
            })
            .and_then(|()| {
                deps.fetcher.download(
                    plan,
                    DownloadIo::Keep {
                        dir: &deps.config.paths.keep,
                    },
                    &deps.config,
                    progress,
                )
            }),
    };
    let filepath = match downloaded {
        Ok(p) => p,
        Err(e) => return failed(&e),
    };
    // Sweep before building the response, which describes files on disk. The
    // rip that just landed survives via the eviction grace window.
    let _ = deps.cache.evict(SystemTime::now());
    match plan.dest {
        Dest::Cache => analysis_response(source, &filepath, deps),
        Dest::Keep => keep_response(&filepath, deps),
    }
}

/// Analysis rip landed in the evictable cache; answer with the next plan.
fn analysis_response(source: &str, filepath: &Path, deps: &Arc<ServerDeps>) -> CallToolResult {
    let (probe, fingerprint) = match cached_probe(filepath, deps) {
        Ok(v) => v,
        Err(e) => return failed(&e),
    };
    let mut next = plan_response(source, &probe, &deps.config);
    next.guidance = format!(
        "Analysis rip downloaded ({}, cached, evictable). The URL now resolves to it \
         automatically. {}",
        size_label(probe.size_bytes),
        next.guidance
    );
    offer_transcription(source, filepath, &fingerprint, &mut next, deps);
    json_result(&VideoBody::Plan {
        probe: next.probe,
        guidance: next.guidance,
        recommended_calls: next.recommended_calls,
    })
}

/// Full-quality download landed in the keep dir; the file belongs to the user.
fn keep_response(filepath: &Path, deps: &Arc<ServerDeps>) -> CallToolResult {
    let (probe, _) = match cached_probe(filepath, deps) {
        Ok(v) => v,
        Err(e) => return failed(&e),
    };
    let size = size_label(probe.size_bytes);
    json_result(&VideoBody::Plan {
        guidance: format!(
            "Saved to {} ({size}). This file belongs to the user: hand the path over if they \
             want to watch it, and they may move or rename it freely. To analyze it, call \
             again with the (possibly moved) path as source — probe results are keyed by \
             content, so a move stays a cache hit.",
            filepath.display()
        ),
        probe,
        recommended_calls: vec![RecommendedCall {
            call: VideoArgs::bare(&filepath.to_string_lossy()),
            note: "analyze the saved file".to_string(),
        }],
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{Config, default_config, test_dirs};
    use crate::fetch::{DownloadOption, Recipe, SourceInfo, direct::DirectRecipe};
    use crate::mcp::core_run::{CoreRun, CoreRunner};
    use crate::mcp::deps::{FileProber, UrlFetcher, Whisper, null_notifier};
    use crate::mcp::tracker::Tracker;
    use crate::tokens::Tokens;
    use rmcp::model::ContentBlock;
    use std::io::Write as _;
    use std::sync::Mutex;
    use std::sync::mpsc::{Receiver, Sender};

    // ── fakes ──

    struct FakeProber(ProbeSummary);

    impl FileProber for FakeProber {
        fn probe(&self, _file: &Path) -> Result<ProbeSummary, Fault> {
            Ok(self.0.clone())
        }
    }

    struct FakeCore {
        stdout: String,
        calls: Mutex<Vec<Vec<String>>>,
    }

    impl CoreRunner for FakeCore {
        fn run(&self, args: &[String]) -> std::io::Result<CoreRun> {
            self.calls.lock().expect("calls").push(args.to_vec());
            Ok(CoreRun {
                exit: super::super::core_run::CoreExit::Success,
                stdout: self.stdout.clone(),
                stderr: String::new(),
            })
        }
    }

    /// Shared handles let tests inspect fakes after `ServerDeps` boxes them.
    impl<T: CoreRunner> CoreRunner for Arc<T> {
        fn run(&self, args: &[String]) -> std::io::Result<CoreRun> {
            (**self).run(args)
        }
    }

    struct FakeWhisper {
        available: bool,
        calls: Mutex<u32>,
    }

    impl Whisper for FakeWhisper {
        fn available(&self, _cfg: &Config) -> bool {
            self.available
        }

        fn transcribe(
            &self,
            _cfg: &Config,
            _file: &Path,
            out_srt: &Path,
            on_step: &mut dyn FnMut(&str),
        ) -> Result<(), Fault> {
            *self.calls.lock().expect("calls") += 1;
            on_step("transcribing");
            std::fs::create_dir_all(out_srt.parent().expect("parent")).expect("transcripts dir");
            std::fs::write(out_srt, "1\n00:00:00,000 --> 00:00:01,000\nhi\n\n").expect("write srt");
            Ok(())
        }
    }

    /// Downloads block on `gate` when present, making slow downloads
    /// deterministic in tests.
    struct FakeFetcher {
        info: SourceInfo,
        gate: Option<Mutex<Receiver<()>>>,
    }

    impl UrlFetcher for FakeFetcher {
        fn probe_url(&self, _url: &str, _cfg: &Config) -> Result<SourceInfo, Fault> {
            Ok(self.info.clone())
        }

        fn download(
            &self,
            _plan: &DownloadPlan,
            io: DownloadIo<'_>,
            _cfg: &Config,
            progress: ProgressFn<'_>,
        ) -> Result<PathBuf, Fault> {
            if let Some(gate) = &self.gate {
                progress(1024, Some(4096));
                gate.lock()
                    .expect("gate")
                    .recv()
                    .map_err(|_| Fault::transient("gate closed"))?;
            }
            let dest = match io {
                DownloadIo::Cache { dir, key } => dir.join(format!("{key}.mp4")),
                DownloadIo::Keep { dir } => dir.join("v.mp4"),
            };
            let mut f = std::fs::File::create(&dest).expect("create download");
            f.write_all(b"fake video bytes").expect("write");
            progress(16, Some(16));
            Ok(dest)
        }
    }

    // ── scaffolding ──

    fn probe(duration: f64, subtitles: Vec<String>) -> ProbeSummary {
        ProbeSummary {
            duration,
            width: 1920,
            height: 1080,
            size_bytes: Some(10 * 1024 * 1024),
            container: Some("mp4".to_string()),
            subtitles,
        }
    }

    /// Built by serializing the real wire types: the fake core's stdout is
    /// in-contract by construction, not by hand-maintained JSON.
    fn core_json() -> String {
        use agent_video_core::types::{
            Detail, GridImage, GridPosition, KeyFrameMeta, LayoutKind, Quality, Settings,
        };
        use agent_video_core::wire::{Grid, Metadata};
        serde_json::to_string(&Metadata {
            source: "v.mp4".to_string(),
            duration: 90.0,
            range: None,
            settings: Settings {
                max_edge: 2048,
                quality: Quality::Low,
                detail: Detail::Standard,
            },
            grids: vec![Grid {
                image: GridImage::Base64("QUJD".to_string()),
                time_range: "00:00-01:30".to_string(),
                total_frames: 12,
                key_frames: vec![KeyFrameMeta {
                    timestamp: "00:10".to_string(),
                    score: 5,
                    grid_position: GridPosition::Hero,
                }],
                layout: LayoutKind::Uniform,
                subtitles: Some("00:01 hi".to_string()),
            }],
        })
        .expect("wire metadata serializes")
    }

    struct TestEnv {
        deps: Arc<ServerDeps>,
        dir: PathBuf,
        core: Arc<FakeCore>,
    }

    fn env(probe_summary: ProbeSummary, whisper_available: bool) -> TestEnv {
        env_with_fetcher(
            probe_summary,
            whisper_available,
            FakeFetcher {
                info: SourceInfo {
                    headline: "A video.".to_string(),
                    options: vec![analysis_option()],
                },
                gate: None,
            },
        )
    }

    fn env_with_fetcher(
        probe_summary: ProbeSummary,
        whisper_available: bool,
        fetcher: FakeFetcher,
    ) -> TestEnv {
        let dir = std::env::temp_dir().join(format!(
            "av-server-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("clock")
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).expect("mkdir");
        let mut config = default_config(&test_dirs(&dir));
        config.paths.cache = dir.join("cache");
        config.paths.keep = dir.join("keep");
        let cache = crate::cache::Cache::new(&config.paths.cache, config.cache.max_bytes);
        let core = Arc::new(FakeCore {
            stdout: core_json(),
            calls: Mutex::new(Vec::new()),
        });
        let deps = Arc::new(ServerDeps {
            cache,
            tokens: Tokens::default(),
            downloads: Tracker::default(),
            prober: Box::new(FakeProber(probe_summary)),
            fetcher: Box::new(fetcher),
            whisper: Box::new(FakeWhisper {
                available: whisper_available,
                calls: Mutex::new(0),
            }),
            core: Box::new(Arc::clone(&core)),
            config,
        });
        TestEnv { deps, dir, core }
    }

    fn analysis_option() -> DownloadOption {
        DownloadOption {
            token: Token::new("dl-analysis"),
            dest: Dest::Cache,
            label: "480p".to_string(),
            effect: "download the analysis rip".to_string(),
            estimated_bytes: Some(1024),
            recipe: Recipe::Direct(DirectRecipe {
                ext: "mp4".to_string(),
                filename: "v.mp4".to_string(),
            }),
        }
    }

    fn media_file(env: &TestEnv, name: &str) -> PathBuf {
        let p = env.dir.join(name);
        std::fs::write(&p, b"pretend this is a video").expect("write media");
        p
    }

    fn call(env: &TestEnv, args: serde_json::Value) -> CallToolResult {
        let serde_json::Value::Object(map) = args else {
            panic!("args must be an object");
        };
        handle_video_call(&map, &env.deps, &null_notifier())
    }

    fn text_of(result: &CallToolResult) -> String {
        result
            .content
            .iter()
            .filter_map(|c| match c {
                ContentBlock::Text(t) => Some(t.text.clone()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn body_of(result: &CallToolResult) -> serde_json::Value {
        serde_json::from_str(&text_of(result)).expect("body is JSON")
    }

    // ── tests ──

    #[test]
    fn probe_doc_round_trips_current_version() {
        let summary = probe(90.0, vec!["sidecar talk.srt".to_string()]);
        let doc = render_probe_doc(&summary).expect("renders");
        let back = parse_probe_doc(&doc).expect("current version parses");
        assert_eq!(back.duration, 90.0);
        assert_eq!(back.subtitles, ["sidecar talk.srt"]);
    }

    #[test]
    fn rejects_probe_docs_from_other_schemas() {
        // A pre-envelope document (bare summary) and a future version must
        // both miss — never be consumed under current semantics.
        let bare = serde_json::to_string(&probe(90.0, vec![])).expect("serialize");
        assert!(parse_probe_doc(&bare).is_none(), "unversioned doc misses");

        let doc = render_probe_doc(&probe(90.0, vec![])).expect("renders");
        let future = doc.replacen("\"v\":1", "\"v\":2", 1);
        assert_ne!(doc, future, "version tag must be present to rewrite");
        assert!(parse_probe_doc(&future).is_none(), "future version misses");
    }

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
            calls.iter().any(|c| c["call"]["confirm"] == "transcribe"),
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
    fn confirmed_transcribe_runs_whisper_and_rides_along() {
        let env = env(probe(3600.0, vec![]), true);
        let file = media_file(&env, "talk.mp4");
        let source = file.to_string_lossy().into_owned();

        let plan = body_of(&call(&env, serde_json::json!({"source": source})));
        assert_eq!(plan["status"], "plan");

        let after = body_of(&call(
            &env,
            serde_json::json!({"source": source, "confirm": "transcribe"}),
        ));
        assert_eq!(after["status"], "plan", "{after}");
        assert!(
            after["guidance"]
                .as_str()
                .expect("guidance")
                .contains("Transcript generated"),
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
