// SPDX-License-Identifier: Apache-2.0
//! Download execution: start the fetch on its own thread, wait out a sync
//! grace window (fast downloads answer inline, slow ones detach into the
//! tracker), verify coverage before caching, sweep, and shape the analysis
//! or keep response. The tracker is the one channel between the download
//! thread and the protocol: progress and the final result both land there,
//! and the inline waiter is just the slot's first poller.

use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};

use rmcp::model::CallToolResult;

use super::deps::{Notifier, ServerDeps};
use super::handler::offer_transcription;
use super::probe_cache::cached_probe;
use super::tracker::{Admission, Poll};
use super::wire::{failed, json_result, pending_result};
use crate::cache::url_key;
use crate::fault::Fault;
use crate::fetch::{DownloadIo, Downloaded, ProgressFn, rip_key};
use crate::plan::{format_bytes, plan_response, size_label};
use crate::schema::{RecommendedCall, VideoArgs, VideoBody};
use crate::tokens::{Dest, DownloadPlan, Token};

/// A job with no progress for this long is presumed wedged and abandoned on
/// the next poll. Longer than every downloader-side stall bound (the yt-dlp
/// watchdog, ureq's phase timeouts), so it only fires when those failed to.
pub(super) const DOWNLOAD_ABANDON_LIMIT: Duration = Duration::from_secs(10 * 60);

/// Sampling tick of the sync grace window's poll loop.
const SYNC_POLL_TICK: Duration = Duration::from_millis(200);

// The watchdog's no-progress bound must sit inside the tracker's abandon
// limit: the child dies before its slot is given up, so an abandoned
// download can never leave a live yt-dlp (and its staging dir) behind.
const _: () =
    assert!(crate::fetch::ytdlp::NO_PROGRESS_LIMIT.as_secs() <= DOWNLOAD_ABANDON_LIMIT.as_secs());

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
            let mut progress = |bytes: u64, total: Option<u64>| reporter.progress(bytes, total);
            let result = perform_download(&source, &plan, &token, &deps, &mut progress);
            reporter.settle(result);
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
/// known errors become `Failed`.
fn perform_download(
    source: &str,
    plan: &DownloadPlan,
    token: &Token,
    deps: &Arc<ServerDeps>,
    progress: ProgressFn<'_>,
) -> CallToolResult {
    let relay = match deps.relay() {
        Ok(r) => r,
        Err(e) => return failed(&e),
    };
    let fetch = |progress: ProgressFn<'_>| -> Result<Downloaded, Fault> {
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
                        progress,
                    )
                }),
        }
    };
    let Downloaded {
        mut path,
        mut warning,
    } = match fetch(&mut *progress) {
        Ok(d) => d,
        Err(e) => return failed(&e),
    };
    // A container whose tracks stop early was served truncated (an expired
    // CDN session, a flaky segment fetch). Caching it would pin the URL to
    // a silently broken rip, so verify before answering.
    if let Some(issue) = crate::probe::coverage_shortfall(&path) {
        match plan.dest {
            // One fresh attempt: a new fetch gets a new CDN session, which
            // heals most truncations. The retry adopts over the same cache
            // names, so a failed retry keeps the first copy.
            Dest::Cache => match fetch(&mut *progress) {
                Ok(retried) => match crate::probe::coverage_shortfall(&retried.path) {
                    None => {
                        path = retried.path;
                        warning = retried.warning;
                    }
                    Some(issue) => {
                        path = retried.path;
                        warning = join_warning(
                            retried.warning,
                            format!(
                                "{issue} even after a fresh retry — the source itself serves \
                                 a truncated stream; grids and transcription stop where the \
                                 tracks end"
                            ),
                        );
                    }
                },
                Err(_) => {
                    warning = join_warning(
                        warning,
                        format!(
                            "{issue}, and a fresh retry failed — keeping this copy; grids \
                             and transcription stop where the tracks end"
                        ),
                    );
                }
            },
            // Keep files belong to the user: report, never second-guess.
            Dest::Keep => {
                warning = join_warning(
                    warning,
                    format!("{issue} — the source may have served a truncated stream"),
                );
            }
        }
    }
    // Sweep before building the response, which describes files on disk. The
    // rip that just landed survives via the eviction grace window.
    let _ = deps.cache.evict(SystemTime::now());
    match plan.dest {
        Dest::Cache => analysis_response(source, &path, warning.as_deref(), deps),
        Dest::Keep => {
            // The receipt makes future echoes of this token replays instead
            // of duplicate rips — keep is the one non-idempotent action, so
            // its idempotency is manufactured here. Truncation warnings
            // still fulfill: the file landed and belongs to the user.
            deps.tokens.fulfill(source, token, path.clone());
            keep_response(&path, warning.as_deref(), deps)
        }
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
