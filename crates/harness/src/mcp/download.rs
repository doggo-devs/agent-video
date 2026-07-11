// SPDX-License-Identifier: Apache-2.0
//! Download execution: start the fetch on its own thread, wait out a sync
//! grace window (fast downloads answer inline, slow ones detach into the
//! tracker), verify coverage before caching, sweep, and shape the analysis
//! or keep response.

use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, SystemTime};

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
    let key = url_key(source);
    // Claim the slot atomically: a concurrent confirm for the same URL
    // must observe this job instead of spawning a second downloader into
    // the same cache entry.
    let reporter = match deps.downloads.admit(&key, DOWNLOAD_ABANDON_LIMIT) {
        Admission::Started(reporter) => reporter,
        Admission::Running { downloaded, total } => return pending_result(downloaded, total),
        Admission::Settled(result) => return result,
    };
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
            let result = perform_download(&source, &plan, &token, &deps, &mut progress);
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

/// The blocking download flow: fetch, verify coverage, respond, sweep;
/// known errors become `Failed`.
fn perform_download(
    source: &str,
    plan: &DownloadPlan,
    token: &Token,
    deps: &Arc<ServerDeps>,
    progress: ProgressFn<'_>,
) -> CallToolResult {
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
pub(super) fn keep_response(filepath: &Path, warning: Option<&str>, deps: &Arc<ServerDeps>) -> CallToolResult {
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
