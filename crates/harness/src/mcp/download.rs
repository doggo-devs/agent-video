// SPDX-License-Identifier: Apache-2.0
//! Download execution: start the fetch on its own thread, wait out a sync
//! grace window (fast downloads answer inline, slow ones detach into the
//! tracker), verify coverage before caching, sweep, and shape the analysis
//! or keep response. The tracker is the one channel between the download
//! thread and the protocol: progress and the final result both land there,
//! and the inline waiter is just the slot's first poller. Authority flows
//! the same way: the slot owns the job's cancel signal, and the keep
//! receipt rides the settlement, so a job that lost its slot can neither
//! keep working nor record anything.

use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};

use rmcp::model::CallToolResult;

use super::deps::{Notifier, ServerDeps};
use super::handler::offer_transcription;
use super::probe_cache::cached_probe;
use super::tracker::{Admission, Poll, Settlement};
use super::wire::{failed, json_result, pending_result};
use crate::cache::url_key;
use crate::fault::Fault;
use crate::fetch::{Attempt, CancelSignal, DownloadIo, Downloaded, ProgressFn, Shortfall, rip_key};
use crate::plan::{format_bytes, plan_response, size_label};
use crate::schema::{RecommendedCall, VideoArgs, VideoBody};
use crate::tokens::{Dest, DownloadPlan, Token};

/// A job with no progress for this long is presumed wedged and abandoned on
/// the next poll — and abandonment has teeth: dropping the slot trips the
/// job's cancel signal, which the direct client observes between reads and
/// on every stall-guard tick. Longer than every downloader-side stall bound
/// (the yt-dlp watchdog, the direct client's idle limit — both pinned by
/// the assertions below), so it only fires when those failed to.
pub(super) const DOWNLOAD_ABANDON_LIMIT: Duration = Duration::from_secs(10 * 60);

/// Sampling tick of the sync grace window's poll loop.
const SYNC_POLL_TICK: Duration = Duration::from_millis(200);

/// How a job spends its retries. The ladder's length *is* the retry count:
/// there is no way to schedule an attempt without naming the wait before
/// it, so a tight loop against a bursty backend cannot be written here.
#[derive(Debug, Clone, Copy)]
struct Retries {
    /// Wait before each retry, in order.
    ladder: &'static [Duration],
    /// Wall clock since the job began past which no further attempt
    /// starts, whatever the ladder still offers.
    budget: Duration,
}

/// Production policy. The bursts this exists for outlast a tight loop —
/// observed as minutes of a CDN backend serving a truncated part — so the
/// waits are tens of seconds, not milliseconds. The budget keeps a job
/// that already burned its time on a stalled attempt (the yt-dlp watchdog
/// gives up after `NO_PROGRESS_LIMIT`) from starting a second one it
/// cannot finish.
const RETRIES: Retries = Retries {
    // In-crate tests drive the ladder's shape, never its clock: a test
    // that waited out a real burst would put every download test in this
    // crate on a two-minute leash.
    ladder: if cfg!(test) {
        &[Duration::ZERO, Duration::ZERO]
    } else {
        &[Duration::from_secs(30), Duration::from_secs(90)]
    },
    budget: Duration::from_secs(5 * 60),
};

/// Longest a backoff may leave the tracker without a word. Abandonment is
/// judged on silence, so a wait longer than this is split into ticks that
/// re-report the last byte count.
const HEARTBEAT_TICK: Duration = Duration::from_secs(60);

const _: () = assert!(HEARTBEAT_TICK.as_secs() < DOWNLOAD_ABANDON_LIMIT.as_secs());

// Each downloader's stall bound must sit inside the tracker's abandon
// limit, so a network-stalled download always dies (yt-dlp: the watchdog
// kills the child; direct: the stall guard unblocks the read) before its
// slot is given up. What the bounds cannot cover — a thread wedged outside
// its network loop — is defused instead: abandonment trips the slot's
// cancel signal, and the settlement (keep receipt included) of a job that
// lost its slot is dropped by the tracker unexecuted.
const _: () =
    assert!(crate::fetch::ytdlp::NO_PROGRESS_LIMIT.as_secs() <= DOWNLOAD_ABANDON_LIMIT.as_secs());
const _: () =
    assert!(crate::fetch::direct::IDLE_LIMIT.as_secs() <= DOWNLOAD_ABANDON_LIMIT.as_secs());

/// Tracker slot key for one download job. An analysis rip and a keep
/// download of the same URL are different jobs — different destinations,
/// different consent tokens — so they must not share a state machine: a
/// keep confirm during a running analysis download starts (or observes)
/// the keep job, never the analysis one.
pub(super) fn slot_key(source: &str, dest: Dest) -> String {
    match dest {
        Dest::Cache => format!("cache.{}", url_key(source)),
        Dest::Keep => format!("keep.{}", url_key(source)),
    }
}

/// Progress line for MCP notifications and pending bodies.
pub(super) fn progress_message(downloaded: u64, total: Option<u64>) -> String {
    let total_text = total
        .map(|t| format!(" / {}", format_bytes(t)))
        .unwrap_or_default();
    format!("downloading: {}{total_text}", format_bytes(downloaded))
}

/// Start the download on its own thread, then wait out a sync grace window:
/// fast downloads answer inline, slow ones detach into the tracker and are
/// polled by later calls.
pub(super) fn execute_download(
    source: &str,
    plan: DownloadPlan,
    token: Token,
    deps: &Arc<ServerDeps>,
    notify: &Notifier,
) -> CallToolResult {
    // Idempotent redemption: a re-confirmed analysis download serves the
    // rip that already landed instead of fetching it again.
    if plan.dest == Dest::Cache
        && let Some(cached) = deps.cache.find_download(&rip_key(&plan.url, &deps.config))
    {
        return analysis_response(source, &cached, None, deps);
    }
    let key = slot_key(source, plan.dest);
    // Claim the slot atomically: a concurrent confirm for the same URL
    // must observe this job instead of spawning a second downloader into
    // the same cache entry.
    let reporter = match deps.downloads.admit(&key, DOWNLOAD_ABANDON_LIMIT) {
        Admission::Started(reporter) => reporter,
        Admission::Running { downloaded, total } => return pending_result(downloaded, total),
        Admission::Settled(result) => return result,
    };
    {
        let deps = Arc::clone(deps);
        let source = source.to_string();
        // The reporter is the thread's only voice. MCP progress
        // notifications are rendered by whoever holds a live request —
        // never by this thread, whose spawning request (and its progress
        // token) may complete long before the download does.
        std::thread::spawn(move || {
            let cancel = reporter.cancel_signal();
            let mut progress = |bytes: u64, total: Option<u64>| reporter.progress(bytes, total);
            let settlement =
                perform_download(&source, &plan, &token, &deps, &cancel, &mut progress);
            reporter.settle(settlement);
        });
    }
    let window = Duration::from_secs(u64::from(deps.config.download.sync_wait_seconds.get()));
    let deadline = Instant::now() + window;
    let mut last_sent: u64 = 0;
    loop {
        match deps.downloads.poll(&key, DOWNLOAD_ABANDON_LIMIT) {
            Poll::Settled(result) => return result,
            Poll::Running { downloaded, total } => {
                // MCP progress must not go backwards; a coverage retry
                // rewinds the byte count, so only growth is reported.
                if downloaded > last_sent {
                    last_sent = downloaded;
                    notify(
                        downloaded as f64,
                        total.map(|t| t as f64),
                        progress_message(downloaded, total),
                    );
                }
                if Instant::now() >= deadline {
                    return pending_result(downloaded, total);
                }
            }
            // Our own thread died without settling: reported once, the slot
            // is free for a retry.
            Poll::Abandoned => {
                return failed(&Fault::transient(
                    "the download thread died before reporting a result — call again to retry",
                ));
            }
            // The slot vanished mid-wait: a concurrent call for the same
            // source consumed the outcome. The executed actions are
            // idempotent — a retry serves the landed rip or replays the
            // keep receipt — so redirecting the caller is safe.
            Poll::Vacant => {
                return failed(&Fault::transient(
                    "this download was resolved by a concurrent call for the same source — \
                     call again",
                ));
            }
        }
        std::thread::sleep(SYNC_POLL_TICK);
    }
}

/// The blocking download flow: fetch, verify coverage, respond, sweep;
/// known errors become `Failed`. Returns a [`Settlement`] because the one
/// durable side effect — the keep receipt — must ride with the result: the
/// tracker executes it only if this job still owns its slot.
fn perform_download(
    source: &str,
    plan: &DownloadPlan,
    token: &Token,
    deps: &Arc<ServerDeps>,
    cancel: &CancelSignal,
    progress: ProgressFn<'_>,
) -> Settlement {
    let relay = match deps.relay() {
        Ok(r) => r,
        Err(e) => return failed(&e).into(),
    };
    let mut fetch = |progress: ProgressFn<'_>| -> Result<Attempt, Fault> {
        match plan.dest {
            Dest::Cache => deps
                .cache
                .ensure_downloads_dir()
                .map_err(|e| {
                    Fault::environment(format!("cannot create the downloads directory: {e}"))
                })
                .and_then(|dir| {
                    let key = rip_key(&plan.url, &deps.config);
                    deps.fetcher.download(
                        plan,
                        DownloadIo::Cache { dir, key: &key },
                        &deps.config,
                        relay,
                        cancel,
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
                        relay,
                        cancel,
                        progress,
                    )
                }),
        }
    };
    let mut beat = Beat::new(progress);
    let Downloaded { path, warning } =
        match fetch_complete(plan.dest, cancel, &mut beat, &mut fetch, RETRIES) {
            Ok(d) => d,
            Err(e) => return failed(&e).into(),
        };
    // Sweep before building the response, which describes files on disk. The
    // rip that just landed survives via the eviction grace window.
    let _ = deps.cache.evict(SystemTime::now());
    // Record the probe's title against the landed bytes: transcription
    // derives its whisper prompt (and thus the transcript's cache key)
    // from content identity alone, so the title must be recoverable from
    // the file with no URL in hand — a kept file re-analyzed by path
    // included. First writer wins (`put_title` is write-once): a
    // re-download under a retitled upload must not shift the derived
    // prompt. Best-effort: a rip without a recorded title only loses
    // the derived prompt, never the download — but the loss must be
    // observable, because it is not local: an untitled rip derives a
    // different transcript id than titled rips of the same content on
    // other machines, splitting the shared transcript cache and
    // re-running whisper for identical bytes.
    if let Some(title) = &plan.title
        && let Ok(fp) = crate::identity::fingerprint_file(&path)
        && fp.stable
        && let Err(e) = deps.cache.put_title(&fp.id, title)
    {
        tracing::warn!(
            error = %e,
            "title sidecar publication failed; transcripts of this content derive an \
             untitled prompt on this machine, splitting the shared transcript cache"
        );
    }
    match plan.dest {
        Dest::Cache => analysis_response(source, &path, warning.as_deref(), deps).into(),
        Dest::Keep => {
            // The receipt makes future echoes of this token replays instead
            // of duplicate rips — keep is the one non-idempotent action, so
            // its idempotency is manufactured here. Truncation warnings
            // still fulfill: the file landed and belongs to the user. The
            // receipt rides the settlement as its owned effect, so a job
            // whose slot was abandoned can never write it — the tracker
            // drops a disowned settlement whole, and a zombie cannot
            // overwrite the receipt a retry just recorded. The disowned
            // compensation is the receipt's mirror: a file published by a
            // job that then lost its slot (cancellation can land between
            // the fetcher's last cancel check and its rename) would orphan
            // under a name no receipt points to and no retry can take, so
            // an unadopted settlement removes it — the file exists iff its
            // settlement was adopted, up to a process crash in between.
            let response = keep_response(&path, warning.as_deref(), deps);
            let unadopted = path.clone();
            let receipt = {
                // A `Tokens` handle, never `Arc<ServerDeps>`: this closure
                // is queued into a channel owned by `deps.downloads`, so
                // capturing the deps graph would close a strong cycle
                // (deps → tracker → queued settlement → deps) that leaks
                // the whole graph and defuses the drop bomb below forever.
                // `Tokens` owns no tracker, so no cycle is constructible.
                let tokens = deps.tokens.clone();
                let source = source.to_string();
                let token = token.clone();
                move || tokens.fulfill(&source, &token, path)
            };
            Settlement::with_owned_effect(response, receipt).or_disowned(move || {
                let _ = std::fs::remove_file(&unadopted);
            })
        }
    }
}

/// The job's voice to the tracker, and its memory of what it last said.
/// Every byte count the downloaders report passes through here, so a
/// backoff can repeat the last one instead of inventing a number or
/// falling silent.
struct Beat<'a> {
    inner: ProgressFn<'a>,
    seen: (u64, Option<u64>),
}

impl<'a> Beat<'a> {
    fn new(inner: ProgressFn<'a>) -> Self {
        Beat {
            inner,
            seen: (0, None),
        }
    }

    fn report(&mut self, downloaded: u64, total: Option<u64>) {
        self.seen = (downloaded, total);
        (self.inner)(downloaded, total);
    }

    /// Wait `delay`, telling the tracker the job is alive. Abandonment is
    /// judged on silence ([`super::tracker::JobSlot`]'s `last_progress`),
    /// and a sleeping thread is indistinguishable from a wedged one, so
    /// the wait is split into [`HEARTBEAT_TICK`] pieces that re-report the
    /// last byte count — never backwards from the caller's view, since the
    /// sync waiter forwards growth only.
    ///
    /// Returns whether the job still owns itself: a disowned job stops
    /// waiting and stops retrying.
    fn wait(&mut self, delay: Duration, cancel: &CancelSignal) -> bool {
        let deadline = Instant::now() + delay;
        loop {
            if cancel.is_cancelled() {
                return false;
            }
            let left = deadline.saturating_duration_since(Instant::now());
            if left.is_zero() {
                return true;
            }
            std::thread::sleep(left.min(HEARTBEAT_TICK));
            let (downloaded, total) = self.seen;
            (self.inner)(downloaded, total);
        }
    }
}

/// Why an attempt did not settle the job.
enum Setback {
    /// The source served short; a fresh extraction may land elsewhere.
    Short(Shortfall),
    /// The attempt failed outright, transiently.
    Failed(Fault),
}

/// Fetch until the rip is complete, the budget runs out, or the failure is
/// one no fresh extraction can heal.
///
/// The loop lives at this layer — above both downloaders, below the
/// protocol — because only a whole new invocation re-runs extraction, and
/// re-extraction is the only thing that changes the signed URL a flaky CDN
/// backend pinned the rip to. yt-dlp's own retries resume that same URL
/// and cannot help. And it lives here rather than in the model's hands
/// because a retry from up there costs a consent round-trip apiece: the
/// job is already detached behind the tracker, so spending the time here
/// costs the caller nothing but a pending poll.
///
/// A short rip that nonetheless landed is kept as the fallback: retries
/// adopt over the same cache names, so a failed retry still leaves the
/// first copy, and a caller who asked to keep a file is told what is
/// wrong with it rather than having it second-guessed.
fn fetch_complete(
    dest: Dest,
    cancel: &CancelSignal,
    beat: &mut Beat<'_>,
    fetch: &mut dyn for<'p> FnMut(ProgressFn<'p>) -> Result<Attempt, Fault>,
    retries: Retries,
) -> Result<Downloaded, Fault> {
    let started = Instant::now();
    let mut fallback: Option<Downloaded> = None;
    let mut attempts: u32 = 0;
    let mut waits = retries.ladder.iter();
    loop {
        attempts += 1;
        let setback = {
            let mut report = |downloaded, total| beat.report(downloaded, total);
            match fetch(&mut report) {
                Ok(Attempt::Complete(done)) => return Ok(done),
                Ok(Attempt::Short { evidence, landed }) => {
                    // Keep destinations belong to the user: a landed file is
                    // reported, never re-fetched behind their back.
                    if landed.is_some() && dest == Dest::Keep {
                        return Ok(short_landing(landed, evidence, attempts)
                            .expect("a landed rip yields a landing"));
                    }
                    fallback = fallback.or(landed);
                    Setback::Short(evidence)
                }
                // Only a transient fault is worth a second attempt; a
                // missing binary or a rejected URL fails the same way
                // forever, and repeating it just burns the caller's clock.
                Err(fault) if !fault.kind.retriable() => return Err(fault),
                Err(fault) => Setback::Failed(fault),
            }
        };
        let Some(&delay) = waits.next() else {
            return verdict(setback, fallback, attempts, started.elapsed());
        };
        if started.elapsed() + delay >= retries.budget || !beat.wait(delay, cancel) {
            return verdict(setback, fallback, attempts, started.elapsed());
        }
    }
}

/// Settle a job whose attempts are spent: serve the incomplete rip if one
/// landed, otherwise report the last setback with the shape of the
/// failure — how many attempts, over how long — because a caller who
/// retries immediately, three times, loses to a burst that outlasts all
/// three.
fn verdict(
    last: Setback,
    fallback: Option<Downloaded>,
    attempts: u32,
    span: Duration,
) -> Result<Downloaded, Fault> {
    match last {
        Setback::Short(evidence) => {
            if let Some(landed) = short_landing(fallback, evidence.clone(), attempts) {
                return Ok(landed);
            }
            Err(Fault::transient(format!(
                "the source served an incomplete rip on every attempt ({}, {}): {}. \
                 {RETRY_HINT}, or pick a different format so the download resolves to a \
                 different stored object",
                attempt_count(attempts),
                span_label(span),
                evidence.describe()
            )))
        }
        Setback::Failed(fault) => match fallback {
            Some(landed) => Ok(Downloaded {
                warning: join_warning(
                    landed.warning,
                    format!(
                        "the rip is incomplete and a fresh attempt failed ({}: {}) — keeping \
                         this copy; grids and transcription stop where the tracks end",
                        fault.kind, fault.message
                    ),
                ),
                ..landed
            }),
            None if attempts > 1 => Err(Fault::new(
                fault.kind,
                format!(
                    "{} ({}, {}). {RETRY_HINT}",
                    fault.message,
                    attempt_count(attempts),
                    span_label(span)
                ),
            )),
            None => Err(fault),
        },
    }
}

/// What a caller can actually do about an exhausted ladder. Named once:
/// the wait is the whole advice, and it is the part every immediate retry
/// in the wild got wrong.
const RETRY_HINT: &str = "the attempts above were already spread over that window, so this \
     fault outlasts an immediate retry — wait a few minutes before confirming again";

/// Serve an incomplete rip that landed anyway, with the evidence attached:
/// the file is real and usable up to where its tracks end, and saying so
/// beats failing a download the caller can still work with.
fn short_landing(
    landed: Option<Downloaded>,
    evidence: Shortfall,
    attempts: u32,
) -> Option<Downloaded> {
    let landed = landed?;
    let note = format!(
        "{} after {} — the source itself serves a truncated stream; grids and transcription \
         stop where the tracks end",
        evidence.describe(),
        attempt_count(attempts)
    );
    Some(Downloaded {
        warning: join_warning(landed.warning, note),
        ..landed
    })
}

fn attempt_count(attempts: u32) -> String {
    match attempts {
        1 => "1 attempt".to_string(),
        n => format!("{n} attempts"),
    }
}

fn span_label(span: Duration) -> String {
    let secs = span.as_secs();
    match secs {
        0..60 => format!("over {secs}s"),
        _ => format!("over {}m{:02}s", secs / 60, secs % 60),
    }
}

/// Append a degradation note to an optional existing warning.
fn join_warning(existing: Option<String>, note: String) -> Option<String> {
    Some(match existing {
        Some(w) => format!("{w}; {note}"),
        None => note,
    })
}

/// Prefix for a degraded-but-successful download: the note leads the
/// guidance so the model sees what is missing (and why) before the next
/// steps, instead of retrying a download that already succeeded.
fn warning_prefix(warning: Option<&str>) -> String {
    warning
        .map(|w| format!("Warning: {w}\n"))
        .unwrap_or_default()
}

/// Analysis rip landed in the evictable cache; answer with the next plan.
fn analysis_response(
    source: &str,
    filepath: &Path,
    warning: Option<&str>,
    deps: &Arc<ServerDeps>,
) -> CallToolResult {
    let (probe, _) = match cached_probe(filepath, deps) {
        Ok(v) => v,
        Err(e) => return failed(&e),
    };
    let mut next = plan_response(source, &probe, &deps.config);
    next.guidance = format!(
        "{}Analysis rip downloaded ({}, cached, evictable). The URL now resolves to it \
         automatically. {}",
        warning_prefix(warning),
        size_label(probe.size_bytes),
        next.guidance
    );
    offer_transcription(source, &mut next, deps);
    json_result(&VideoBody::Plan {
        probe: next.probe,
        guidance: next.guidance,
        recommended_calls: next.recommended_calls,
    })
}

/// Full-quality download landed in the keep dir; the file belongs to the
/// user. Also the replay body when a fulfilled keep token is echoed again.
pub(super) fn keep_response(
    filepath: &Path,
    warning: Option<&str>,
    deps: &Arc<ServerDeps>,
) -> CallToolResult {
    let (probe, _) = match cached_probe(filepath, deps) {
        Ok(v) => v,
        Err(e) => return failed(&e),
    };
    let size = size_label(probe.size_bytes);
    json_result(&VideoBody::Plan {
        guidance: format!(
            "{}Saved to {} ({size}). This file belongs to the user: hand the path over if they \
             want to watch it, and they may move or rename it freely. To analyze it, call \
             again with the (possibly moved) path as source — probe results are keyed by \
             content, so a move stays a cache hit.",
            warning_prefix(warning),
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
    use std::path::PathBuf;

    fn landed(name: &str) -> Downloaded {
        Downloaded {
            path: PathBuf::from(name),
            warning: None,
        }
    }

    fn short(landed: Option<Downloaded>) -> Result<Attempt, Fault> {
        Ok(Attempt::Short {
            evidence: Shortfall::Transfer {
                got: 75_497_472,
                expected: 103_894_225,
            },
            landed,
        })
    }

    /// Drive `fetch_complete` over a scripted sequence of attempts;
    /// returns its verdict and how many attempts it spent.
    fn run(dest: Dest, script: Vec<Result<Attempt, Fault>>) -> (Result<Downloaded, Fault>, usize) {
        let mut script = script.into_iter();
        let mut spent = 0;
        let mut sink = |_, _| {};
        let mut beat = Beat::new(&mut sink);
        let mut fetch = |_: ProgressFn<'_>| {
            spent += 1;
            script.next().expect("the script covers every attempt")
        };
        let result = fetch_complete(dest, &CancelSignal::new(), &mut beat, &mut fetch, RETRIES);
        (result, spent)
    }

    #[test]
    fn retries_a_short_rip_without_the_caller() {
        let (result, attempts) = run(
            Dest::Cache,
            vec![
                short(None),
                short(None),
                Ok(Attempt::Complete(landed("v.mp4"))),
            ],
        );

        assert_eq!(
            attempts, 3,
            "the burst must be outlasted here, not by the model"
        );
        assert_eq!(
            result.expect("the third attempt landed").path,
            PathBuf::from("v.mp4")
        );
    }

    #[test]
    fn reports_an_exhausted_burst_with_its_shape() {
        let (result, attempts) = run(Dest::Cache, vec![short(None), short(None), short(None)]);

        assert_eq!(attempts, 3, "the ladder bounds the attempts");
        let fault = result.expect_err("nothing landed");
        assert_eq!(fault.kind, crate::fault::FaultKind::Transient);
        assert!(fault.message.contains("3 attempts"), "{fault}");
        assert!(
            fault.message.contains("72 MiB of the 99 MiB"),
            "the evidence rides along: {fault}"
        );
        assert!(
            fault.message.contains("wait a few minutes"),
            "a burst outlasts an immediate retry, and the caller must hear it: {fault}"
        );
    }

    #[test]
    fn serves_a_short_rip_that_landed_when_retries_run_out() {
        let (result, _) = run(
            Dest::Cache,
            vec![short(Some(landed("first.mp4"))), short(None), short(None)],
        );

        let served = result.expect("an incomplete file still beats no file");
        assert_eq!(
            served.path,
            PathBuf::from("first.mp4"),
            "retries adopt over the same names, so the first copy survives"
        );
        let warning = served.warning.expect("the shortfall must be disclosed");
        assert!(warning.contains("truncated"), "{warning}");
    }

    #[test]
    fn keeps_a_landed_file_without_second_guessing_the_user() {
        let (result, attempts) = run(Dest::Keep, vec![short(Some(landed("user.mp4")))]);

        assert_eq!(
            attempts, 1,
            "a keep file belongs to the user, not to the retry loop"
        );
        let served = result.expect("the file landed");
        assert!(served.warning.expect("disclosed").contains("truncated"));
    }

    #[test]
    fn retries_a_keep_download_that_landed_nothing() {
        let (result, attempts) = run(
            Dest::Keep,
            vec![short(None), Ok(Attempt::Complete(landed("user.mp4")))],
        );

        assert_eq!(
            attempts, 2,
            "nothing landed, so there is nothing to second-guess"
        );
        assert!(result.is_ok());
    }

    #[test]
    fn refuses_to_retry_a_fault_no_re_extraction_can_heal() {
        let (result, attempts) = run(
            Dest::Cache,
            vec![Err(Fault::environment("yt-dlp is not installed"))],
        );

        assert_eq!(attempts, 1, "a missing binary fails the same way forever");
        assert_eq!(
            result.expect_err("fatal").kind,
            crate::fault::FaultKind::Environment
        );
    }

    #[test]
    fn stops_retrying_once_the_job_is_disowned() {
        let cancel = CancelSignal::new();
        let mut sink = |_, _| {};
        let mut beat = Beat::new(&mut sink);
        let mut spent = 0;
        let mut fetch = |_: ProgressFn<'_>| {
            spent += 1;
            cancel.cancel();
            short(None)
        };

        let result = fetch_complete(Dest::Cache, &cancel, &mut beat, &mut fetch, RETRIES);

        assert_eq!(spent, 1, "a job that lost its slot must not keep working");
        assert!(result.is_err());
    }

    #[test]
    fn backoff_heartbeats_keep_the_slot_alive() {
        let mut reported: Vec<(u64, Option<u64>)> = Vec::new();
        {
            let mut sink = |downloaded, total| reported.push((downloaded, total));
            let mut beat = Beat::new(&mut sink);
            beat.report(1024, Some(4096));
            let alive = beat.wait(Duration::from_millis(1), &CancelSignal::new());
            assert!(alive);
        }

        assert_eq!(
            reported,
            vec![(1024, Some(4096)), (1024, Some(4096))],
            "a silent backoff is indistinguishable from a wedged thread, so the wait \
             re-reports the last count instead of inventing one"
        );
    }

    #[test]
    fn a_spent_ladder_reports_the_cause_that_ended_it() {
        let (result, _) = run(
            Dest::Cache,
            vec![
                Err(Fault::transient("connection reset by peer")),
                Err(Fault::transient("connection reset by peer")),
                Err(Fault::transient("connection reset by peer")),
            ],
        );
        let fault = result.expect_err("nothing landed");
        assert!(
            fault.message.contains("connection reset by peer"),
            "{fault}"
        );
        assert!(fault.message.contains("3 attempts"), "{fault}");
    }
}
