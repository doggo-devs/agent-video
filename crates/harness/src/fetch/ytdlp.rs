// SPDX-License-Identifier: Apache-2.0
//! yt-dlp source adapter.
//!
//! Probes and downloads 1800+ sites via an optional yt-dlp on PATH; its
//! absence becomes an install hint while direct media links keep working
//! without it. Machine interfaces only: `-J` for probing,
//! `--progress-template` for progress, `--print after_move:filepath` for the
//! final path; the human progress bar is never parsed.

use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::mpsc;
use std::time::{Duration, Instant};

use super::{Attempt, DownloadIo, Downloaded, ProgressFn, Shortfall, SourceInfo};
use crate::config::Config;
use crate::cookies::CookieHandle;
use crate::fault::Fault;
use crate::netguard::VettedUrl;
use crate::relay::PolicyRelay;

mod document;

pub use document::{
    KEEP_TIER_LIMIT, KeepTier, SiteUrlInfo, YtdlpFormat, YtdlpJson, site_info, site_source_info,
};

const INSTALL_HINT: &str = "yt-dlp not found on PATH — install it (https://github.com/yt-dlp/yt-dlp) to handle \
     site URLs, or download the video manually and pass the local file path as source";

/// Recipe carried by yt-dlp download plans.
#[derive(Debug, Clone, PartialEq)]
pub struct YtdlpRecipe {
    /// yt-dlp `-f` format selector, decided at offer time.
    pub format: String,
    /// Also fetch subtitle sidecars (analysis copy only).
    pub subs: bool,
}

// ── sidecar promotion ──

/// Name segment separating downloaded subtitles from the media's sidecar
/// namespace: they land as `<key>_subs.<lang>.<ext>`, which core's sidecar
/// discovery (stem followed by a literal `.`) never claims. Without this,
/// every multi-language download would make core's auto-detection fail as
/// ambiguous — promotion elects the single sidecar instead.
const SUBS_INFIX: &str = "_subs";

/// Copy the preferred-language downloaded subtitle to `<key>.<ext>` — the
/// rip's one sidecar, picked up by core's auto-detection. `preferred_langs`
/// is the election policy (`subtitles.preferred_langs` in config.toml):
/// prefix-matched, first hit wins, no hit falls back to the first candidate.
///
/// Only formats core actually reads are eligible
/// ([`agent_video_core::subtitle::SIDECAR_SUBTITLE_EXTS`]): an unconverted leftover (`--convert-subs`
/// can fail) must not become a sidecar core cannot parse.
///
/// On failure, no partial file remains under the sidecar's name: the copy
/// goes to a temp name and is renamed into place only once complete, so
/// adoption can never publish a truncated sidecar. The error is the
/// caller's to surface — the media itself is unaffected.
pub fn promote_subtitle(dir: &Path, key: &str, preferred_langs: &[String]) -> std::io::Result<()> {
    use agent_video_core::subtitle::SIDECAR_SUBTITLE_EXTS;

    let mut names: Vec<String> = std::fs::read_dir(dir)?
        .flatten()
        .filter_map(|e| e.file_name().into_string().ok())
        .collect();
    names.sort();

    // `<key>_subs.<lang>.<ext>` — lang is a single dot-free segment.
    let prefix = format!("{key}{SUBS_INFIX}.");
    let candidates: Vec<(&str, &str, &str)> = names
        .iter()
        .filter_map(|n| {
            let rest = n.strip_prefix(&prefix)?;
            let (lang, ext) = rest.rsplit_once('.')?;
            let valid =
                !lang.is_empty() && !lang.contains('.') && SIDECAR_SUBTITLE_EXTS.contains(&ext);
            valid.then_some((n.as_str(), lang, ext))
        })
        .collect();
    let preferred = preferred_langs
        .iter()
        .find_map(|want| {
            candidates
                .iter()
                .find(|(_, lang, _)| lang.starts_with(want.as_str()))
        })
        .or_else(|| candidates.first());
    let Some((name, _, ext)) = preferred else {
        return Ok(());
    };
    let target = dir.join(format!("{key}.{ext}"));
    if target.exists() {
        return Ok(());
    }
    // Staging publishes every file it holds (`adopt_into`), so a torn copy
    // must never rest under a publishable sidecar name: copy to a temp
    // name, rename into place only when complete, delete the temp on error.
    let tmp = dir.join(format!("{key}.{ext}.promote"));
    let copied = std::fs::copy(dir.join(name), &tmp)
        .map(drop)
        .and_then(|()| std::fs::rename(&tmp, &target));
    if copied.is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
    copied
}

/// Fold a failed sidecar promotion into the caller-facing warning channel
/// — the same channel [`settle_exit`] uses for degraded extras: the media
/// is intact, only the subtitle sidecar is affected.
fn promotion_warning(result: std::io::Result<()>, warning: Option<String>) -> Option<String> {
    let Err(e) = result else { return warning };
    stack_warnings(
        warning,
        Some(format!(
            "the downloaded subtitles could not be promoted to a sidecar ({e}); subtitles may be missing"
        )),
    )
}

/// Stack a later degradation note under an earlier one; either may be
/// absent. The warning channel is append-only: no note ever replaces
/// another.
fn stack_warnings(first: Option<String>, second: Option<String>) -> Option<String> {
    match (first, second) {
        (Some(a), Some(b)) => Some(format!("{a}\n{b}")),
        (a, b) => a.or(b),
    }
}

// ── spawning ──

/// Diagnostics only ever read the tail (`stderr_tail`, `is_auth_failure`,
/// `short_transfer`), so keep at most this many bytes per pipe and let
/// whole lines fall off the front.
const LOG_TAIL_LIMIT: usize = 64 * 1024;

/// Line-oriented tail buffer bounded to `limit` bytes: once appending a
/// line pushes the total over, whole lines fall off the front. The newest
/// line always survives intact, even alone over the limit — a truncated
/// diagnostic is worse than a briefly oversized buffer.
struct TailBuf {
    buf: String,
    limit: usize,
}

impl TailBuf {
    fn new(limit: usize) -> Self {
        TailBuf {
            buf: String::new(),
            limit,
        }
    }

    /// Append `line` (newline added here) and evict whole lines from the
    /// front until the buffer fits `limit` or one line remains.
    fn push_line(&mut self, line: &str) {
        self.buf.push_str(line);
        self.buf.push('\n');
        while self.buf.len() > self.limit {
            // Every line ends in '\n'; a boundary before the trailing one
            // means more than one line is stored.
            let Some(i) = self.buf[..self.buf.len() - 1].find('\n') else {
                break;
            };
            self.buf.drain(..=i);
        }
    }

    fn into_string(self) -> String {
        self.buf
    }
}

fn stderr_tail(stderr: &str) -> String {
    let tail: Vec<&str> = stderr.trim().lines().rev().take(5).collect();
    if tail.is_empty() {
        "no error output".to_string()
    } else {
        tail.into_iter().rev().collect::<Vec<_>>().join("\n")
    }
}

fn spawn_error(e: std::io::Error) -> Fault {
    if e.kind() == std::io::ErrorKind::NotFound {
        Fault::environment(INSTALL_HINT)
    } else {
        Fault::environment(format!("could not spawn yt-dlp: {e}"))
    }
}

/// Acquire the configured cookie source; a jar that cannot be staged is
/// the user's environment, not the URL.
fn acquire_cookies(cfg: &Config) -> Result<CookieHandle, Fault> {
    CookieHandle::acquire(&cfg.cookies)
        .map_err(|e| Fault::environment(format!("cannot stage the cookie jar: {e}")))
}

/// Adopt rotated cookies back into the canonical jar. A failed write-back
/// is a half-committed persistent effect — the site may already have
/// rotated the session (invalidating the old credentials) while the
/// canonical jar still holds the old page — so it comes back as a
/// caller-facing warning instead of vanishing into a log line. The media
/// verdict itself must not depend on jar bookkeeping; the warning channel
/// is exactly that shape.
#[must_use]
fn commit_cookies(cookies: CookieHandle) -> Option<String> {
    let e = cookies.commit().err()?;
    Some(format!(
        "rotated cookies could not be written back to the jar ({e}); the stored login \
         session may now be stale — if the next download fails asking for a login, \
         re-import cookies with `agent-video cookie import`"
    ))
}

/// yt-dlp's own error lines. The auth heuristic below runs on these only:
/// a WARNING quoting "premium" or progress noise mentioning a member
/// stream is not a login wall. No `ERROR:` lines at all (a bare traceback,
/// a signal kill) falls back to the full text — a mislabeled hint beats a
/// lost one.
fn error_lines(stderr: &str) -> String {
    let errors: Vec<&str> = stderr
        .lines()
        .filter(|l| l.trim_start().starts_with("ERROR"))
        .collect();
    if errors.is_empty() {
        stderr.to_string()
    } else {
        errors.join("\n")
    }
}

/// yt-dlp's own short-transfer verdict, parsed for its two numbers:
/// `Downloaded 75497472 bytes, expected 103894225 bytes`. An exact parse,
/// not a keyword sniff — the numbers *are* the evidence, and they ride
/// into the caller-facing message when every attempt comes up short.
///
/// yt-dlp reports this on its screen channel (stdout) while retries are
/// still pending and on stderr once they are exhausted, so callers search
/// both tails. The last occurrence wins: it is the attempt that gave up.
fn short_transfer(log: &str) -> Option<Shortfall> {
    log.lines().rev().find_map(|line| {
        let (_, rest) = line.rsplit_once("Downloaded ")?;
        let (got, rest) = rest.split_once(" bytes, expected ")?;
        let (expected, _) = rest.split_once(" bytes")?;
        let got: u64 = got.trim().parse().ok()?;
        let expected: u64 = expected.trim().parse().ok()?;
        (got < expected).then_some(Shortfall::Transfer { got, expected })
    })
}

/// yt-dlp stderr markers meaning "this source wants a login", not "retry":
/// yt-dlp's own auth errors suggest `--cookies`, and extractors phrase
/// gated content as sign-in / members-only / private. Judged on yt-dlp's
/// error lines only ([`error_lines`]).
fn is_auth_failure(stderr: &str) -> bool {
    let lower = error_lines(stderr).to_lowercase();
    [
        "--cookies",
        "sign in",
        "log in",
        "login required",
        "private video",
        "members-only",
        "members only",
        "premium",
    ]
    .iter()
    .any(|m| lower.contains(m))
}

/// The actionable way out of an auth failure, appended to its fault.
const AUTH_HINT: &str = "this source requires a login — export cookies from a logged-in browser \
     session and run `agent-video cookie import <file>`, or set cookies.from_browser in \
     config.toml";

/// How long a `yt-dlp -J` probe may run before it is killed: it is pure
/// network metadata work, so a minute of silence means a hung site, not a
/// slow one.
const PROBE_DEADLINE: Duration = Duration::from_secs(60);

/// Assemble the `yt-dlp -J` probe arguments, routing every connection
/// through the relay with `--proxy`. Extracted so a test can assert the
/// proxy is wired without spawning yt-dlp.
fn probe_args(url: &VettedUrl, relay: &PolicyRelay, cookies: &CookieHandle) -> Vec<String> {
    let mut args: Vec<String> = ["-J", "--no-playlist", "--proxy"]
        .into_iter()
        .map(String::from)
        .collect();
    args.push(relay.url());
    args.extend(cookies.args());
    args.push(url.as_str().to_string());
    args
}

/// Set proxy environment variables so the ffmpeg yt-dlp spawns internally
/// inherits the same egress route as yt-dlp itself — `--proxy` binds
/// yt-dlp's own requests, the env vars bind its children's.
///
/// `no_proxy` is blanked, not inherited: ffmpeg's http protocol consults it
/// at connect time even when a proxy is explicitly configured, so an
/// inherited value would silently route matching hosts around the policy
/// relay.
fn set_proxy_env(cmd: &mut Command, relay: &PolicyRelay) {
    let url = relay.url();
    for key in ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"] {
        cmd.env(key, &url);
    }
    for key in ["no_proxy", "NO_PROXY"] {
        cmd.env(key, "");
    }
}

/// Probe a site page: `yt-dlp -J` reduced through [`site_info`].
pub fn probe(url: &VettedUrl, cfg: &Config, relay: &PolicyRelay) -> Result<SourceInfo, Fault> {
    let cookies = acquire_cookies(cfg)?;
    let mut cmd = Command::new("yt-dlp");
    cmd.args(probe_args(url, relay, &cookies));
    set_proxy_env(&mut cmd, relay);
    let out = crate::proc::output_with_deadline(&mut cmd, PROBE_DEADLINE).map_err(spawn_error)?;
    // Even a probe rotates cookies on some sites; adopt them whatever the
    // verdict (a deadline-killed child left an empty delta). A probe's
    // SourceInfo has no degradation channel, so a failed write-back is
    // logged here; the download path surfaces the same warning on-wire.
    if let Some(warning) = commit_cookies(cookies) {
        tracing::warn!("{warning}");
    }
    let Some(status) = out.status else {
        return Err(Fault::transient(format!(
            "yt-dlp probing {url} hung for {}s and was aborted — the site may be \
             unreachable; retry, or download manually and pass the local path",
            PROBE_DEADLINE.as_secs()
        )));
    };
    if !status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr.bytes);
        if is_auth_failure(&stderr) {
            return Err(Fault::environment(format!(
                "yt-dlp could not read {url}: {} — {AUTH_HINT}",
                stderr_tail(&stderr)
            )));
        }
        return Err(Fault::transient(format!(
            "yt-dlp could not read {url} (exit {}): {}",
            status.code().unwrap_or(1),
            stderr_tail(&stderr)
        )));
    }
    if out.stdout.truncated {
        return Err(Fault::bad_input(format!(
            "yt-dlp -J output for {url} exceeded the capture limit — the site's \
             metadata is implausibly large; this source cannot be probed"
        )));
    }
    let doc = parse_probe_doc(&out.stdout.bytes)?;
    Ok(site_source_info(
        &site_info(&doc, cfg.download.analysis_max_height.get()),
        cfg,
    ))
}

/// Parse the `-J` document. A parse failure keeps the serde error — the
/// actual evidence — in both the message and the source chain; the
/// version hint is a guess and must not replace it.
fn parse_probe_doc(bytes: &[u8]) -> Result<YtdlpJson, Fault> {
    serde_json::from_slice(bytes).map_err(|e| {
        Fault::environment(format!(
            "yt-dlp -J emitted invalid JSON ({e}) — yt-dlp may be too old"
        ))
        .with_source(e)
    })
}

// ── downloading ──

/// No output on either pipe for this long means the download hung; the
/// watchdog kills it. Progress lines arrive at least once a second while
/// bytes flow, so this is generous even across silent phase transitions.
const STALL_LIMIT: Duration = Duration::from_secs(300);

/// No *byte progress* for this long means the download is wedged even if
/// it keeps logging (a fragment-retry loop chatters forever without moving
/// bytes). Long enough to cover the chatter-only pre-download phase
/// (extractor negotiation) and post-download remuxing, and strictly inside
/// the tracker's abandon limit — see the compile-time assertion in
/// `mcp::download` — so an abandoned slot can never leave a live child
/// behind.
pub(crate) const NO_PROGRESS_LIMIT: Duration = Duration::from_secs(480);

/// One liveness signal from the pipe readers. The two variants feed two
/// clocks: `Chatter` proves the child is talking, `Bytes` proves it is
/// working. Only the stdout progress parse — the same parse that feeds
/// the tracker — may claim `Bytes`, so the watchdog and the tracker's
/// abandon clock share one definition of progress and cannot disagree.
#[derive(Debug, Clone, Copy)]
enum Pulse {
    /// A parsed progress line: bytes moved. Resets both clocks.
    Bytes,
    /// Any other output line. Resets only the silence clock.
    Chatter,
    /// The downloader is unwinding: kill the child now. Sent only by
    /// [`WatchdogGuard`]'s `Drop`.
    Abort,
}

/// Absolute wall-clock ceiling on a yt-dlp child's lifetime, progress or
/// not: a server drip-feeding one progress tick every few minutes resets
/// both stall clocks forever, so the stall machinery alone never ends it.
/// Matches the direct path's `BODY_CEILING`.
const LIFETIME_CEILING: Duration = Duration::from_secs(4 * 60 * 60);

/// The watchdog's deadlines, injectable so tests run in milliseconds.
#[derive(Debug, Clone, Copy)]
struct StallLimits {
    /// Kill after this long with no output at all.
    silence: Duration,
    /// Kill after this long with no byte progress, chatter or not.
    no_progress: Duration,
    /// Kill after this long since spawn, whatever the progress.
    lifetime: Duration,
}

const STALL_LIMITS: StallLimits = StallLimits {
    silence: STALL_LIMIT,
    no_progress: NO_PROGRESS_LIMIT,
    lifetime: LIFETIME_CEILING,
};

/// How long a child may linger after closing its output before being killed.
const REAP_GRACE: Duration = Duration::from_secs(30);

/// Grace between SIGINT and SIGKILL when aborting a stalled child: enough
/// for yt-dlp's KeyboardInterrupt path to save rotated cookies into the
/// staged jar, far below every caller-facing wait bound.
const INT_GRACE: Duration = Duration::from_secs(10);

/// Terminal state of the yt-dlp child, decided by the watchdog thread that
/// owns it.
#[derive(Debug)]
enum ChildOutcome {
    /// Exited on its own after closing both pipes.
    Exited(ExitStatus),
    /// Silent past [`StallLimits::silence`]; killed mid-download.
    Stalled,
    /// Kept chattering but moved no bytes for [`StallLimits::no_progress`];
    /// killed mid-download.
    NoProgress,
    /// Alive past [`StallLimits::lifetime`], progress or not; killed.
    Expired,
    /// Closed both pipes but did not exit within [`REAP_GRACE`]; killed.
    Unreaped,
    /// Killed on [`Pulse::Abort`]: the downloader unwound mid-flight.
    Aborted,
}

/// Own the child until it terminates. Liveness is typed: a [`Pulse::Bytes`]
/// resets both clocks, a [`Pulse::Chatter`] only the silence clock, so a
/// child that logs forever without moving bytes still dies. All senders
/// dropping (both pipes at EOF) means the work is done and the child only
/// needs reaping. Sole owner of the child, so the kill paths cannot race a
/// wait.
fn watch_child(
    mut child: Child,
    pulses: mpsc::Receiver<Pulse>,
    limits: StallLimits,
) -> ChildOutcome {
    let start = Instant::now();
    let mut last_any = start;
    let mut last_bytes = start;
    loop {
        let deadline = (last_any + limits.silence)
            .min(last_bytes + limits.no_progress)
            .min(start + limits.lifetime);
        let wait = deadline.saturating_duration_since(Instant::now());
        match pulses.recv_timeout(wait) {
            Ok(Pulse::Bytes) => {
                let now = Instant::now();
                last_any = now;
                last_bytes = now;
            }
            Ok(Pulse::Chatter) => last_any = Instant::now(),
            Ok(Pulse::Abort) => {
                // Gently, as on a stall: the INT path saves rotated cookies.
                crate::proc::kill_tree_graceful(&mut child, INT_GRACE);
                return ChildOutcome::Aborted;
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                return match crate::proc::wait_with_deadline(&mut child, REAP_GRACE) {
                    Some(status) => ChildOutcome::Exited(status),
                    None => ChildOutcome::Unreaped,
                };
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {
                // Gently first: a SIGKILLed yt-dlp never saves rotated
                // cookies to its jar, so a stall would burn the login
                // session on top of the download. The whole group either
                // way: a stalled yt-dlp may have live ffmpeg children
                // holding the pipes, and the pipe readers only unblock
                // once every write end is gone.
                crate::proc::kill_tree_graceful(&mut child, INT_GRACE);
                return if start.elapsed() >= limits.lifetime {
                    ChildOutcome::Expired
                } else if last_any.elapsed() >= limits.silence {
                    ChildOutcome::Stalled
                } else {
                    ChildOutcome::NoProgress
                };
            }
        }
    }
}

/// Pins the child's lifetime to the download call. The tracker
/// (`crate::mcp::tracker`) re-admits a URL's slot the moment this thread's
/// reporter drops, so a child that outlives its call would race the retry
/// on the same output paths — the keep directory has no per-job staging.
/// A normal path takes the watchdog's verdict through [`Self::join`]; an
/// unwind (a panicking progress callback) reaches `Drop`, which orders the
/// kill and waits for it before the unwinding continues.
struct WatchdogGuard {
    abort: mpsc::Sender<Pulse>,
    handle: Option<std::thread::JoinHandle<ChildOutcome>>,
}

impl WatchdogGuard {
    /// Spawn the watchdog thread owning `child`; `abort` must be a sender
    /// into the same channel `pulses` reads.
    fn spawn(
        child: Child,
        pulses: mpsc::Receiver<Pulse>,
        abort: mpsc::Sender<Pulse>,
        limits: StallLimits,
    ) -> Self {
        WatchdogGuard {
            abort,
            handle: Some(std::thread::spawn(move || {
                watch_child(child, pulses, limits)
            })),
        }
    }

    /// Wait for the watchdog's verdict. The abort sender is released before
    /// the join: the watchdog's normal exit condition is the pulse
    /// channel's disconnect, which a sender parked here would veto.
    fn join(mut self) -> ChildOutcome {
        let handle = self.handle.take().expect("join consumes the only handle");
        drop(self); // no handle left: Drop only releases the abort sender
        handle.join().expect("watchdog thread does not panic")
    }
}

impl Drop for WatchdogGuard {
    fn drop(&mut self) {
        let Some(handle) = self.handle.take() else {
            return;
        };
        // Unwinding: order the kill and wait for the reap, so the child is
        // provably dead before the tracker can hand its key to a retry. A
        // failed send means the watchdog already returned; join either way.
        let _ = self.abort.send(Pulse::Abort);
        let _ = handle.join();
    }
}

const PROGRESS_PREFIX: &str = "AVDL ";
const PROGRESS_TEMPLATE: &str = "download:AVDL %(progress.downloaded_bytes)s %(progress.total_bytes)s %(progress.total_bytes_estimate)s";

/// Sentinel marking the one stdout line that names the media's final path.
/// stdout carries three kinds of line — progress, this, and yt-dlp's own
/// chatter — and each is recognised by what it *is*: the open-world
/// default this replaces ("whatever did not parse as progress is the
/// filepath") let a `[download] Destination: …` line pass for the media
/// path, which is why the chatter had to be muted with `--quiet` — taking
/// yt-dlp's short-transfer diagnostics down with it.
const FILEPATH_PREFIX: &str = "AVFP ";

/// One classified stdout line. Closed world by construction: adding a
/// channel means adding a variant, not widening a default.
#[derive(Debug, PartialEq)]
enum Line<'a> {
    /// A `--progress-template` line: bytes moved, total when known.
    Progress(u64, Option<u64>),
    /// The `--print after_move:` line: the media's final path.
    Filepath(&'a str),
    /// Anything else yt-dlp says. Never a path, always a diagnostic.
    Chatter(&'a str),
}

/// Classify one stdout line by its sentinel.
fn classify(line: &str) -> Line<'_> {
    if let Some((downloaded, total)) = parse_progress(line) {
        return Line::Progress(downloaded, total);
    }
    match line.strip_prefix(FILEPATH_PREFIX) {
        Some(path) => Line::Filepath(path.trim()),
        None => Line::Chatter(line),
    }
}

/// Parse one `--progress-template` line; `None` for foreign lines.
pub fn parse_progress(line: &str) -> Option<(u64, Option<u64>)> {
    let rest = line.strip_prefix(PROGRESS_PREFIX)?;
    let mut fields = rest.split_whitespace();
    let num = |s: Option<&str>| -> Option<u64> {
        let s = s?;
        if s == "NA" {
            return None;
        }
        // yt-dlp may render byte counts as floats.
        let v: f64 = s.parse().ok()?;
        (v.is_finite() && v > 0.0).then_some(v as u64)
    };
    let downloaded = num(fields.next())?;
    let total = num(fields.next());
    let estimate = num(fields.next());
    Some((downloaded, total.or(estimate)))
}

/// A per-writer staging directory inside the shared cache: two jobs
/// confirming the same URL — whether in one server or two — would
/// otherwise hand yt-dlp the same output template, and yt-dlp's own
/// `.part` handling interleaves their writes. The name comes from the
/// cache's one staging grammar
/// ([`crate::cache::naming::staging_dir_name`]): writer-unique, not
/// per-process — a pid alone would let two threads of one server share a
/// directory. Each job downloads into its own directory and adopts the
/// results into the cache with per-file renames (same filesystem, atomic
/// per writer; last-writer-wins is safe because equal keys mean equal
/// recipes). Removed on drop; a crash leaves it for
/// [`crate::cache::Cache::evict`]'s stale-staging sweep, which recognizes
/// it through the same grammar's predicate.
struct StagingDir {
    path: PathBuf,
}

impl StagingDir {
    fn create(parent: &Path, key: &str) -> std::io::Result<Self> {
        let path = parent.join(crate::cache::naming::staging_dir_name(key));
        std::fs::create_dir_all(&path)?;
        Ok(StagingDir { path })
    }

    /// Move every staged file up into `dest_dir`; the media named by
    /// `filepath` (yt-dlp's report, inside staging) maps to its final path.
    ///
    /// Each file is fsynced before its rename — yt-dlp does not sync, so
    /// without this a crash after adoption could publish a cache name whose
    /// bytes are still in the page cache. Best-effort: a platform refusing
    /// fsync on a read-only handle degrades to plain rename durability,
    /// which is not worth failing a finished download over.
    fn adopt_into(&self, dest_dir: &Path, filepath: &Path) -> std::io::Result<PathBuf> {
        for entry in std::fs::read_dir(&self.path)? {
            let entry = entry?;
            if let Ok(file) = std::fs::File::open(entry.path()) {
                let _ = file.sync_all();
            }
            std::fs::rename(entry.path(), dest_dir.join(entry.file_name()))?;
        }
        let name = filepath
            .file_name()
            .ok_or_else(|| std::io::Error::other("yt-dlp reported a nameless filepath"))?;
        Ok(dest_dir.join(name))
    }
}

impl Drop for StagingDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

/// Judge a finished yt-dlp run. A non-zero exit does not decide failure by
/// itself: yt-dlp reflects any recorded error in its exit code, including
/// best-effort side work such as subtitle fetching or conversion (e.g. a
/// site serving login-gated subtitles). The `after_move:filepath` line
/// prints only once the media file reached its final path, so `media_landed`
/// is the ground truth: media on disk downgrades the exit code to a warning
/// the caller surfaces; no media means the download really failed.
fn settle_exit(
    status: ExitStatus,
    media_landed: bool,
    stderr: &str,
) -> Result<Option<String>, Fault> {
    if status.success() {
        return Ok(None);
    }
    if media_landed {
        return Ok(Some(format!(
            "yt-dlp reported errors after the media file completed (exit {}); best-effort \
             extras such as subtitles may be missing: {}",
            status.code().unwrap_or(1),
            stderr_tail(stderr)
        )));
    }
    if is_auth_failure(stderr) {
        return Err(Fault::environment(format!(
            "yt-dlp download failed: {} — {AUTH_HINT}",
            stderr_tail(stderr)
        )));
    }
    Err(Fault::transient(format!(
        "yt-dlp download failed (exit {}): {}",
        status.code().unwrap_or(1),
        stderr_tail(stderr)
    )))
}

/// Download via yt-dlp, blocking until the file is at its final path; yt-dlp
/// owns `.part` handling and atomic renames within the staging directory.
///
/// One straight pipeline, destination-blind between its ends: prepare the
/// `Landing`, assemble `download_args`, pump the child (`run_child`),
/// judge its verdict (`settle_outcome`), publish (`Landing::publish`).
pub fn download(
    url: &VettedUrl,
    recipe: &YtdlpRecipe,
    io: DownloadIo<'_>,
    cfg: &Config,
    relay: &PolicyRelay,
    progress: ProgressFn<'_>,
) -> Result<Attempt, Fault> {
    let landing = Landing::prepare(io, recipe.subs)?;
    let cookies = acquire_cookies(cfg)?;
    let args = download_args(url, recipe, &landing, cfg, relay, &cookies);
    let run = run_child(&args, relay, progress)?;
    // Whatever the verdict: a normal exit and a graceful abort both saved
    // the staged jar, a hard-killed child left an empty delta. A failed
    // write-back rides the warning channel — the remote session may have
    // rotated past the jar, and the model must hear it from this call, not
    // from the next download's auth failure.
    let jar_warning = commit_cookies(cookies);
    // yt-dlp's own completeness verdict outranks its exit code: a child
    // that never finished the transfer served a short rip, whatever it
    // exited with, and a short rip is the retry loop's business — the
    // exit code alone cannot tell it from a dead host.
    let short = short_transfer(&run.chatter).or_else(|| short_transfer(&run.stderr));
    let warning = match settle_outcome(&run) {
        Ok(warning) => stack_warnings(warning, jar_warning),
        Err(fault) => return short_or(short, jar_warning, fault),
    };
    if let Some(e) = run.read_error {
        return Err(Fault::transient(format!(
            "reading yt-dlp output failed mid-download: {e}"
        )));
    }
    let Some(filepath) = run.filepath else {
        return short_or(
            short,
            warning,
            Fault::bad_input(format!(
                "yt-dlp finished without producing a file — the size cap (download.max_bytes) \
                 may have aborted it: {}",
                stderr_tail(&run.stderr)
            )),
        );
    };
    Ok(Attempt::Complete(landing.publish(filepath, warning, cfg)?))
}

/// Report a failed run as short when yt-dlp said so, otherwise as the
/// fault. A warning collected on the way (a failed cookie write-back) has
/// no caller to reach on either path — the loop above only ever surfaces
/// its own final verdict — so it is logged rather than dropped: the remote
/// session may have rotated past the jar.
fn short_or(
    short: Option<Shortfall>,
    warning: Option<String>,
    fault: Fault,
) -> Result<Attempt, Fault> {
    if let Some(warning) = warning {
        tracing::warn!("{warning}");
    }
    match short {
        Some(evidence) => Ok(Attempt::Short {
            evidence,
            landed: None,
        }),
        None => Err(fault),
    }
}

/// Where a yt-dlp run's files land and how they are published once the
/// child settles. Constructed before the spawn and consumed by
/// `Landing::publish`, so the whole Cache/Keep divergence — staging vs
/// direct write, subtitle side work, sidecar promotion, adoption — lives
/// at this type's two ends; the pipeline between (argument assembly,
/// spawn, watchdog, verdict) is destination-blind.
enum Landing {
    /// Per-writer staging inside the shared cache, adopted into `dest_dir`
    /// once complete, with the preferred subtitle promoted to the rip's
    /// one sidecar. `subs` mirrors the recipe: it turns on both the
    /// subtitle templates and the promotion step.
    Cache {
        staging: StagingDir,
        dest_dir: PathBuf,
        key: String,
        subs: bool,
    },
    /// Straight into the keep directory; yt-dlp names the file and no side
    /// work follows — the file belongs to the user.
    Keep { dir: PathBuf },
}

impl Landing {
    fn prepare(io: DownloadIo<'_>, subs: bool) -> Result<Self, Fault> {
        match io {
            DownloadIo::Cache { dir, key } => {
                let staging = StagingDir::create(dir, key).map_err(|e| {
                    Fault::environment(format!("cannot create a staging directory in cache: {e}"))
                })?;
                Ok(Landing::Cache {
                    staging,
                    dest_dir: dir.to_path_buf(),
                    key: key.to_string(),
                    subs,
                })
            }
            DownloadIo::Keep { dir } => Ok(Landing::Keep {
                dir: dir.to_path_buf(),
            }),
        }
    }

    /// yt-dlp `-o` output template for the media file.
    fn media_template(&self) -> PathBuf {
        match self {
            Landing::Cache { staging, key, .. } => staging.path.join(format!("{key}.%(ext)s")),
            Landing::Keep { dir } => dir.join("%(title)s [%(id)s].%(ext)s"),
        }
    }

    /// Subtitle output template, outside the media's sidecar namespace
    /// (see [`SUBS_INFIX`]); cache rips only, and only when asked.
    fn subtitle_template(&self) -> Option<PathBuf> {
        match self {
            Landing::Cache {
                staging,
                key,
                subs: true,
                ..
            } => Some(
                staging
                    .path
                    .join(format!("{key}{SUBS_INFIX}.%(lang)s.%(ext)s")),
            ),
            _ => None,
        }
    }

    /// Publish a finished download. Keep files are already final; cache
    /// rips elect their one sidecar first — adoption must publish a
    /// complete entry, and a failed election degrades to a warning since
    /// the media is already on disk — then adopt out of staging.
    fn publish(
        self,
        filepath: PathBuf,
        warning: Option<String>,
        cfg: &Config,
    ) -> Result<Downloaded, Fault> {
        match self {
            Landing::Keep { .. } => Ok(Downloaded {
                path: filepath,
                warning,
            }),
            Landing::Cache {
                staging,
                dest_dir,
                key,
                subs,
            } => {
                let warning = if subs {
                    promotion_warning(
                        promote_subtitle(&staging.path, &key, &cfg.subtitles.preferred_langs),
                        warning,
                    )
                } else {
                    warning
                };
                let path = staging.adopt_into(&dest_dir, &filepath).map_err(|e| {
                    Fault::environment(format!("cannot move the download into the cache: {e}"))
                })?;
                Ok(Downloaded { path, warning })
            }
        }
    }
}

/// Assemble the yt-dlp download arguments: machine progress on stdout, the
/// final path via `--print after_move:filepath`, every connection through
/// the policy relay. Extracted so tests can pin the wiring — in particular
/// that `-o` is followed by its template — without spawning yt-dlp.
fn download_args(
    url: &VettedUrl,
    recipe: &YtdlpRecipe,
    landing: &Landing,
    cfg: &Config,
    relay: &PolicyRelay,
    cookies: &CookieHandle,
) -> Vec<String> {
    // `--no-quiet` is load-bearing, and not a default: `--print` turns
    // quiet on by itself. Quiet suppresses yt-dlp's screen channel, and
    // that channel carries the diagnostic that tells a short transfer from
    // a dead host — `Got error: Downloaded N bytes, expected M bytes`,
    // reported per pending retry and never repeated on stderr, which only
    // ever shows the terminal `Did not get any data blocks`. The chatter
    // this lets through is safe because stdout lines are typed by sentinel
    // ([`classify`]) rather than by falling through a default.
    //
    // No `--retries` either: yt-dlp already defaults to 10, and every one
    // of them resumes the *same* signed URL — the one pinned to the backend
    // that served short. Only a fresh invocation re-runs extraction, which
    // is why the retry loop lives above this process, in `mcp::download`.
    let mut args: Vec<String> = [
        "--no-playlist",
        "--no-simulate",
        "--no-quiet",
        "--progress",
        "--newline",
        "--progress-delta",
        "1",
        "--progress-template",
        PROGRESS_TEMPLATE,
        "-f",
        &recipe.format,
        "--max-filesize",
        &cfg.download.max_bytes.get().to_string(),
        // Every yt-dlp connection (and its ffmpeg children, via the env
        // set at spawn) routes through the policy relay.
        "--proxy",
        &relay.url(),
    ]
    .into_iter()
    .map(String::from)
    .collect();
    args.push("--print".to_string());
    args.push(format!("after_move:{FILEPATH_PREFIX}%(filepath)s"));
    args.push("-o".to_string());
    args.push(landing.media_template().to_string_lossy().into_owned());
    if let Some(sub) = landing.subtitle_template() {
        args.extend(
            [
                "--write-subs",
                "--sub-langs",
                // live_chat and danmaku are sidecar formats core can never
                // read; danmaku xml additionally makes `--convert-subs`
                // fail hard on an ffmpeg error.
                "all,-live_chat,-danmaku",
                "--convert-subs",
                "srt",
                "-o",
            ]
            .map(String::from),
        );
        args.push(format!("subtitle:{}", sub.display()));
    }
    args.extend(cookies.args());
    args.push(url.as_str().to_string());
    args
}

/// What a yt-dlp child left behind once the watchdog settled it.
struct ChildRun {
    /// The watchdog's terminal verdict.
    outcome: ChildOutcome,
    /// Bounded stderr tail for diagnostics.
    stderr: String,
    /// Bounded tail of yt-dlp's stdout chatter — its screen channel, where
    /// pending-retry diagnostics are reported ([`short_transfer`]).
    chatter: String,
    /// yt-dlp's `after_move:` line: the media's final path, ground truth
    /// that the file landed.
    filepath: Option<PathBuf>,
    /// A failed stdout read — not EOF: a lost filepath line must not be
    /// misdiagnosed as "no file produced".
    read_error: Option<std::io::Error>,
}

/// Spawn yt-dlp and pump it until it terminates. stdout progress lines
/// feed `progress` and the watchdog's byte clock — one parse, one
/// definition of "moving" — everything else is chatter; stderr drains into
/// a bounded tail on a helper thread. The watchdog owns the child and
/// kills it on stall, no-progress, lifetime, or an unwind of this thread
/// ([`WatchdogGuard`]).
fn run_child(
    args: &[String],
    relay: &PolicyRelay,
    progress: ProgressFn<'_>,
) -> Result<ChildRun, Fault> {
    let mut cmd = Command::new("yt-dlp");
    set_proxy_env(&mut cmd, relay);
    let mut child = crate::proc::isolate_group(&mut cmd)
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(spawn_error)?;
    let stdout_pipe = child.stdout.take().expect("stdout was piped");
    let stderr_pipe = child.stderr.take().expect("stderr was piped");

    // Pipe readers feed the watchdog typed pulses: only the stdout
    // progress parse claims Bytes, everything else is Chatter, so a child
    // that logs (fragment retries) without moving bytes still expires.
    let (activity, watchdog_rx) = mpsc::channel::<Pulse>();

    // stderr drains on a helper thread (yt-dlp writes log noise there);
    // stdout — progress lines plus the final filepath — parses here.
    let stderr_thread = {
        let activity = activity.clone();
        std::thread::spawn(move || {
            let mut tail = TailBuf::new(LOG_TAIL_LIMIT);
            for line in BufReader::new(stderr_pipe).lines() {
                let Ok(line) = line else { break };
                let _ = activity.send(Pulse::Chatter);
                tail.push_line(&line);
            }
            tail.into_string()
        })
    };

    let watchdog = WatchdogGuard::spawn(child, watchdog_rx, activity.clone(), STALL_LIMITS);

    let mut filepath: Option<PathBuf> = None;
    let mut chatter = TailBuf::new(LOG_TAIL_LIMIT);
    let mut read_error: Option<std::io::Error> = None;
    for line in BufReader::new(stdout_pipe).lines() {
        let line = match line {
            Ok(line) => line,
            Err(e) => {
                read_error = Some(e);
                break;
            }
        };
        match classify(&line) {
            // The one site that claims byte progress — the same parse feeds
            // the tracker, so both stall clocks agree on what "moving" means.
            Line::Progress(downloaded, total) => {
                let _ = activity.send(Pulse::Bytes);
                progress(downloaded, total);
            }
            Line::Filepath(path) => {
                let _ = activity.send(Pulse::Chatter);
                if !path.is_empty() {
                    filepath = Some(PathBuf::from(path));
                }
            }
            Line::Chatter(text) => {
                let _ = activity.send(Pulse::Chatter);
                chatter.push_line(text);
            }
        }
    }

    // Last sender: with the stderr thread's clone gone too, the watchdog
    // sees the disconnect and reaps.
    drop(activity);
    let stderr = stderr_thread.join().unwrap_or_default();
    let outcome = watchdog.join();
    Ok(ChildRun {
        outcome,
        stderr,
        chatter: chatter.into_string(),
        filepath,
        read_error,
    })
}

/// Map the watchdog's verdict to the caller-facing result: every kill
/// verdict is a transient fault naming what tripped; a normal exit is
/// judged by [`settle_exit`] against whether the media actually landed.
fn settle_outcome(run: &ChildRun) -> Result<Option<String>, Fault> {
    match &run.outcome {
        ChildOutcome::Stalled => Err(Fault::transient(format!(
            "the download stalled — no output from yt-dlp for {}s — and was aborted; \
             confirm again to retry",
            STALL_LIMIT.as_secs()
        ))),
        ChildOutcome::NoProgress => Err(Fault::transient(format!(
            "yt-dlp kept producing output but moved no bytes for {}s — likely wedged \
             in a retry loop — and was aborted; confirm again to retry",
            NO_PROGRESS_LIMIT.as_secs()
        ))),
        ChildOutcome::Expired => Err(Fault::transient(format!(
            "the download exceeded the {}h lifetime ceiling and was aborted — the \
             source may be serving impossibly slowly; confirm again to retry",
            LIFETIME_CEILING.as_secs() / 3600
        ))),
        ChildOutcome::Unreaped => Err(Fault::transient(
            "yt-dlp did not exit after its output closed and was killed",
        )),
        // Only the guard's Drop sends Abort, and that path discards the
        // outcome while unwinding — but the type cannot promise it.
        ChildOutcome::Aborted => Err(Fault::transient("the download was aborted mid-flight")),
        ChildOutcome::Exited(status) => {
            let media_landed = run.filepath.as_deref().is_some_and(Path::is_file);
            settle_exit(*status, media_landed, &run.stderr)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(unix)]
    fn exit_status(code: i32) -> ExitStatus {
        use std::os::unix::process::ExitStatusExt;
        ExitStatus::from_raw(code << 8)
    }

    #[cfg(unix)]
    #[test]
    fn clean_exit_settles_without_warning() {
        let settled = settle_exit(exit_status(0), true, "").expect("success");
        assert_eq!(settled, None);
    }

    #[cfg(unix)]
    #[test]
    fn degrades_exit_error_to_warning_when_media_landed() {
        // BiliBili without login: the rip completes, then subtitle
        // conversion fails and yt-dlp exits 1. The finished media must
        // survive as a warning, not be discarded as a failed download.
        let warning = settle_exit(exit_status(1), true, "ERROR: Preprocessing: Invalid data")
            .expect("media landed ⇒ success")
            .expect("non-zero exit ⇒ warning");
        assert!(warning.contains("subtitles may be missing"), "{warning}");
        assert!(warning.contains("Invalid data"), "stderr rides along");
    }

    #[cfg(unix)]
    #[test]
    fn auth_failure_becomes_an_environment_fault_with_the_import_hint() {
        // YouTube's login wall: the fix is user action (import cookies),
        // not a retry — the fault class and message must say so.
        let stderr = "ERROR: [youtube] abc: Sign in to confirm you’re not a bot. \
                      Use --cookies-from-browser or --cookies for the authentication";
        let fault = settle_exit(exit_status(1), false, stderr).expect_err("no media ⇒ failure");
        assert_eq!(fault.kind, crate::fault::FaultKind::Environment);
        assert!(
            fault.to_string().contains("agent-video cookie import"),
            "names the way out: {fault}"
        );
    }

    #[cfg(unix)]
    #[test]
    fn fails_exit_error_when_media_never_landed() {
        let fault = settle_exit(exit_status(1), false, "ERROR: no formats")
            .expect_err("no media ⇒ failure");
        assert!(fault.to_string().contains("no formats"), "{fault}");
    }

    #[test]
    fn parses_the_short_transfer_verdict_with_its_numbers() {
        // yt-dlp's own line, as it reaches the screen channel while
        // retries are still pending.
        let log = "[download] Destination: /tmp/rip.mp4\n\
                   [download] Got error: Downloaded 75497472 bytes, expected 103894225 bytes. \
                   Retrying (1/10)...";

        assert_eq!(
            short_transfer(log),
            Some(Shortfall::Transfer {
                got: 75_497_472,
                expected: 103_894_225
            })
        );
    }

    #[test]
    fn reads_the_last_short_transfer_report() {
        let log = "Got error: Downloaded 10 bytes, expected 100 bytes. Retrying (1/10)...\n\
                   ERROR: [download] Got error: Downloaded 40 bytes, expected 100 bytes. \
                   Giving up after 10 retries";

        assert_eq!(
            short_transfer(log),
            Some(Shortfall::Transfer {
                got: 40,
                expected: 100
            }),
            "the attempt that gave up is the evidence"
        );
    }

    #[test]
    fn ignores_logs_without_a_shortfall() {
        assert_eq!(short_transfer("ERROR: Did not get any data blocks"), None);
        assert_eq!(
            short_transfer("Downloaded 100 bytes, expected 100 bytes"),
            None,
            "a complete transfer is not a shortfall"
        );
    }

    #[test]
    fn chatter_never_becomes_the_media_path() {
        // Regression: with stdout typed by "anything that is not progress
        // is the filepath", this line published a log message as the media
        // path — which is why the chatter had to be muted, taking the
        // short-transfer diagnostics with it.
        assert_eq!(
            classify("[download] Destination: /tmp/rip.f30032.mp4"),
            Line::Chatter("[download] Destination: /tmp/rip.f30032.mp4")
        );
        assert_eq!(
            classify(&format!("{FILEPATH_PREFIX}/tmp/rip.mp4")),
            Line::Filepath("/tmp/rip.mp4")
        );
        assert_eq!(
            classify("AVDL 1024 4096 NA"),
            Line::Progress(1024, Some(4096))
        );
    }

    #[test]
    fn download_args_name_the_media_path_channel_and_keep_yt_dlp_talking() {
        let relay = crate::relay::PolicyRelay::start(false).expect("relay starts");
        let mut cfg =
            crate::config::default_config(&crate::config::test_dirs(Path::new("/home/u")));
        cfg.network.allow_private_hosts = true;
        let cookies = acquire_cookies(&cfg).expect("anonymous cookies stage cleanly");
        let url = crate::netguard::vet("https://example.com/watch", &cfg).expect("well-formed");
        let landing = Landing::prepare(
            DownloadIo::Keep {
                dir: Path::new("/keep"),
            },
            false,
        )
        .expect("keep landing");
        let recipe = YtdlpRecipe {
            format: "best".to_string(),
            subs: false,
        };

        let args = download_args(&url, &recipe, &landing, &cfg, &relay, &cookies);

        let print = args.iter().position(|a| a == "--print").expect("--print");
        assert!(
            args[print + 1].starts_with(&format!("after_move:{FILEPATH_PREFIX}")),
            "the media path must arrive on its own sentinel: {}",
            args[print + 1]
        );
        assert!(
            args.iter().any(|a| a == "--no-quiet"),
            "`--print` turns quiet on by itself, and quiet mutes the diagnostics that \
             tell a short transfer from a dead host: {args:?}"
        );
        assert!(!args.iter().any(|a| a == "--quiet"), "{args:?}");
    }

    #[test]
    fn auth_markers_outside_error_lines_do_not_fire() {
        // A WARNING quoting "premium" next to a plain network error must
        // settle as transient, not as a login wall.
        let stderr = "WARNING: [site] skipping premium-quality subtitles\n\
                      ERROR: unable to download video data: timed out";
        assert!(!is_auth_failure(stderr), "warning noise must not fire");
        assert!(
            is_auth_failure("ERROR: [youtube] private video, sign in"),
            "real auth errors still fire"
        );
        assert!(
            is_auth_failure("Traceback: use --cookies for authentication"),
            "no ERROR lines falls back to the full text"
        );
    }

    #[test]
    fn probe_args_route_through_the_relay() {
        let relay = crate::relay::PolicyRelay::start(false).expect("relay starts");
        let mut cfg =
            crate::config::default_config(&crate::config::test_dirs(Path::new("/home/u")));
        // Skip DNS in `vet` so the test never depends on how a public
        // hostname resolves on the runner (offline CI, or a resolver that
        // maps example.com into a non-global range); the arg wiring under
        // test is independent of the address policy.
        cfg.network.allow_private_hosts = true;
        let cookies = acquire_cookies(&cfg).expect("anonymous cookies stage cleanly");
        let url = crate::netguard::vet("https://example.com/watch", &cfg).expect("well-formed url");
        let args = probe_args(&url, &relay, &cookies);
        let proxy_at = args
            .iter()
            .position(|a| a == "--proxy")
            .expect("probe passes --proxy");
        assert_eq!(
            args[proxy_at + 1],
            relay.url(),
            "--proxy points at the policy relay"
        );
        assert_eq!(args.last().expect("url is last"), url.as_str());
    }

    #[test]
    fn proxy_env_blanks_no_proxy_for_children() {
        let relay = crate::relay::PolicyRelay::start(false).expect("relay starts");
        let mut cmd = Command::new("yt-dlp");
        set_proxy_env(&mut cmd, &relay);
        let envs: std::collections::HashMap<_, _> = cmd
            .get_envs()
            .map(|(k, v)| (k.to_os_string(), v.map(|v| v.to_os_string())))
            .collect();
        for key in ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"] {
            assert_eq!(
                envs.get(std::ffi::OsStr::new(key)),
                Some(&Some(relay.url().into())),
                "{key} points at the relay"
            );
        }
        for key in ["no_proxy", "NO_PROXY"] {
            assert_eq!(
                envs.get(std::ffi::OsStr::new(key)),
                Some(&Some(std::ffi::OsString::new())),
                "{key} is blanked so no inherited value bypasses the relay"
            );
        }
    }

    #[test]
    fn parses_progress_template_lines() {
        assert_eq!(
            parse_progress("AVDL 1024 2048 NA"),
            Some((1024, Some(2048)))
        );
        assert_eq!(
            parse_progress("AVDL 1024 NA 4096"),
            Some((1024, Some(4096)))
        );
        assert_eq!(parse_progress("AVDL 1024 NA NA"), Some((1024, None)));
        assert_eq!(parse_progress("AVDL 1536.0 NA NA"), Some((1536, None)));
        assert_eq!(parse_progress("/some/file/path.mp4"), None);
        assert_eq!(parse_progress("AVDL NA NA NA"), None);
    }

    fn temp_dir(tag: &str) -> crate::tempdir::TempDir {
        crate::tempdir::TempDir::new(&format!("av-ytdlp-{tag}"))
    }

    fn langs(prefs: &[&str]) -> Vec<String> {
        prefs.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn promotes_the_configured_language_first() {
        let dir = temp_dir("promote");
        std::fs::write(dir.join("k_subs.zh-Hans.srt"), b"zh").expect("write");
        std::fs::write(dir.join("k_subs.en-US.srt"), b"en").expect("write");
        promote_subtitle(&dir, "k", &langs(&["en", "zh"])).expect("promote");
        assert_eq!(std::fs::read(dir.join("k.srt")).expect("promoted"), b"en");
    }

    #[test]
    fn promotion_preference_is_policy_not_hardcode() {
        let dir = temp_dir("promote-zh");
        std::fs::write(dir.join("k_subs.zh-Hans.srt"), b"zh").expect("write");
        std::fs::write(dir.join("k_subs.en-US.srt"), b"en").expect("write");
        promote_subtitle(&dir, "k", &langs(&["zh", "en"])).expect("promote");
        assert_eq!(std::fs::read(dir.join("k.srt")).expect("promoted"), b"zh");
    }

    #[test]
    fn promote_ignores_unrelated_files() {
        let dir = temp_dir("unrelated");
        std::fs::write(dir.join("k.mp4"), b"v").expect("write");
        promote_subtitle(&dir, "k", &langs(&["en"])).expect("promote");
        assert!(!dir.join("k.srt").exists());
    }

    #[test]
    fn promote_skips_formats_core_cannot_read() {
        // `--convert-subs srt` can fail; an unconverted leftover must not
        // become a sidecar core would refuse to parse.
        let dir = temp_dir("vtt");
        std::fs::write(dir.join("k_subs.en.vtt"), b"vtt").expect("write");
        promote_subtitle(&dir, "k", &langs(&["en"])).expect("promote");
        assert!(!dir.join("k.vtt").exists());
        assert!(!dir.join("k.srt").exists());
    }

    #[test]
    fn failed_promotion_leaves_no_partial_file_in_staging() {
        // A candidate that cannot be copied (a directory stands in for a
        // mid-copy I/O error): the sidecar name and the temp name must both
        // stay absent, or adoption would publish a torn subtitle for the
        // URL's whole cache lifetime.
        let dir = temp_dir("torn");
        std::fs::create_dir(dir.join("k_subs.en.srt")).expect("uncopyable candidate");
        promote_subtitle(&dir, "k", &langs(&["en"])).expect_err("copy must fail");
        assert!(!dir.join("k.srt").exists(), "no publishable sidecar");
        assert!(!dir.join("k.srt.promote").exists(), "no temp leftover");
    }

    #[test]
    fn failed_promotion_surfaces_a_warning() {
        let warning = promotion_warning(Err(std::io::Error::other("disk full")), None)
            .expect("failure reaches the warning channel");
        assert!(warning.contains("subtitles may be missing"), "{warning}");
        assert!(warning.contains("disk full"), "cause rides along");
    }

    #[test]
    fn promotion_warning_preserves_the_exit_warning() {
        let warning = promotion_warning(
            Err(std::io::Error::other("boom")),
            Some("exit 1: extras failed".to_string()),
        )
        .expect("warning present");
        assert!(warning.contains("exit 1: extras failed"));
        assert!(warning.contains("boom"));
        assert_eq!(
            promotion_warning(Ok(()), Some("prior".to_string())).as_deref(),
            Some("prior"),
            "success passes the channel through untouched"
        );
    }

    #[test]
    fn probe_parse_fault_retains_the_serde_source() {
        let fault = parse_probe_doc(b"not json").expect_err("parse must fail");
        assert!(fault.to_string().contains("invalid JSON"), "{fault}");
        let source = std::error::Error::source(&fault).expect("serde error rides in the chain");
        assert!(
            source.downcast_ref::<serde_json::Error>().is_some(),
            "source is the typed serde error"
        );
    }

    #[test]
    fn clean_cookie_commit_yields_no_warning() {
        assert_eq!(commit_cookies(CookieHandle::Anonymous), None);
    }

    #[cfg(unix)]
    #[test]
    fn failed_cookie_commit_surfaces_a_warning() {
        // The child rotated a cookie into the staged jar, but the write
        // back to the canonical jar fails: the remote session may already
        // have moved on, so the half-committed state must reach the
        // caller's warning channel, not just a log line.
        use std::os::unix::fs::PermissionsExt;
        let dir = temp_dir("jar");
        let cookie_cfg = crate::config::CookiesCfg {
            from_browser: None,
            jar: dir.join("cookies.txt"),
        };
        let jar_line = |value: &str| format!(".a.com\tTRUE\t/\tFALSE\t0\tn\t{value}\n");
        std::fs::write(&cookie_cfg.jar, jar_line("old")).expect("write jar");
        let handle = CookieHandle::acquire(&cookie_cfg).expect("stage jar");
        let CookieHandle::Jar(lease) = &handle else {
            panic!("a present jar acquires in jar mode");
        };
        std::fs::write(lease.staged(), jar_line("rotated")).expect("rotate staged");
        std::fs::set_permissions(&cookie_cfg.jar, std::fs::Permissions::from_mode(0o000))
            .expect("chmod");
        if std::fs::read_to_string(&cookie_cfg.jar).is_ok() {
            return; // privileged (e.g. root): unreadable cannot be simulated
        }

        let warning = commit_cookies(handle).expect("a failed write-back must surface");

        assert!(
            warning.contains("could not be written back"),
            "names the failure: {warning}"
        );
        assert!(
            warning.contains("cookie import"),
            "names the way out: {warning}"
        );
    }

    #[test]
    fn download_args_bind_the_output_template_to_dash_o() {
        // Regression: the --proxy pair was once inserted between `-o` and
        // its value, so yt-dlp's parser took "--proxy" as the output
        // template and the relay URL became a positional download target.
        let relay = crate::relay::PolicyRelay::start(false).expect("relay starts");
        let mut cfg =
            crate::config::default_config(&crate::config::test_dirs(Path::new("/home/u")));
        cfg.network.allow_private_hosts = true; // skip DNS in `vet`; see probe_args test
        let cookies = acquire_cookies(&cfg).expect("anonymous cookies stage cleanly");
        let url = crate::netguard::vet("https://example.com/watch", &cfg).expect("well-formed");
        let dir = temp_dir("args");
        let landing = Landing::prepare(
            DownloadIo::Cache {
                dir: &dir,
                key: "k",
            },
            true,
        )
        .expect("staging dir");
        let recipe = YtdlpRecipe {
            format: "best".to_string(),
            subs: true,
        };

        let args = download_args(&url, &recipe, &landing, &cfg, &relay, &cookies);

        let o = args.iter().position(|a| a == "-o").expect("-o present");
        assert_eq!(
            args[o + 1],
            landing.media_template().to_string_lossy(),
            "-o must be immediately followed by the media template"
        );
        let proxy = args.iter().position(|a| a == "--proxy").expect("--proxy");
        assert_eq!(args[proxy + 1], relay.url(), "--proxy names the relay");
        let sub_o = args.iter().rposition(|a| a == "-o").expect("subtitle -o");
        assert_ne!(sub_o, o, "subtitle output gets its own -o");
        assert!(
            args[sub_o + 1].starts_with("subtitle:"),
            "{}",
            args[sub_o + 1]
        );
        assert_eq!(args.last().expect("url is last"), url.as_str());
    }

    #[test]
    fn keep_download_args_skip_subtitle_work() {
        let relay = crate::relay::PolicyRelay::start(false).expect("relay starts");
        let mut cfg =
            crate::config::default_config(&crate::config::test_dirs(Path::new("/home/u")));
        cfg.network.allow_private_hosts = true;
        let cookies = acquire_cookies(&cfg).expect("anonymous cookies stage cleanly");
        let url = crate::netguard::vet("https://example.com/watch", &cfg).expect("well-formed");
        let landing = Landing::prepare(
            DownloadIo::Keep {
                dir: Path::new("/keep"),
            },
            // The recipe never asks for subs on keep downloads; even if it
            // did, the keep landing has no subtitle template.
            true,
        )
        .expect("keep landing");
        let recipe = YtdlpRecipe {
            format: "best".to_string(),
            subs: true,
        };

        let args = download_args(&url, &recipe, &landing, &cfg, &relay, &cookies);

        assert!(!args.iter().any(|a| a == "--write-subs"), "{args:?}");
        let o = args.iter().position(|a| a == "-o").expect("-o present");
        assert_eq!(
            args[o + 1],
            landing.media_template().to_string_lossy(),
            "keep template rides -o"
        );
    }

    #[test]
    fn cache_landing_publishes_media_with_promoted_sidecar() {
        let cfg = crate::config::default_config(&crate::config::test_dirs(Path::new("/home/u")));
        let dir = temp_dir("landing");
        let landing = Landing::prepare(
            DownloadIo::Cache {
                dir: &dir,
                key: "k",
            },
            true,
        )
        .expect("staging dir");
        let staging_path = match &landing {
            Landing::Cache { staging, .. } => staging.path.clone(),
            Landing::Keep { .. } => unreachable!("cache io prepares a cache landing"),
        };
        std::fs::write(staging_path.join("k.mp4"), b"v").expect("write media");
        std::fs::write(staging_path.join("k_subs.en.srt"), b"en").expect("write subs");

        let downloaded = landing
            .publish(staging_path.join("k.mp4"), None, &cfg)
            .expect("publish");

        assert_eq!(downloaded.path, dir.join("k.mp4"));
        assert!(dir.join("k.srt").exists(), "sidecar elected and adopted");
        assert!(!staging_path.exists(), "staging cleaned up");
    }

    #[test]
    fn keep_landing_publishes_the_reported_path_verbatim() {
        let cfg = crate::config::default_config(&crate::config::test_dirs(Path::new("/home/u")));
        let landing = Landing::prepare(
            DownloadIo::Keep {
                dir: Path::new("/keep"),
            },
            false,
        )
        .expect("keep landing");

        let downloaded = landing
            .publish(
                PathBuf::from("/keep/My Video [id].mp4"),
                Some("degraded".to_string()),
                &cfg,
            )
            .expect("publish");

        assert_eq!(downloaded.path, Path::new("/keep/My Video [id].mp4"));
        assert_eq!(downloaded.warning.as_deref(), Some("degraded"));
    }

    #[test]
    fn staging_adoption_publishes_every_file_and_cleans_up() {
        let dir = temp_dir("adopt");
        let staging = StagingDir::create(&dir, "k").expect("staging");
        let staging_path = staging.path.clone();
        std::fs::write(staging_path.join("k.mp4"), b"v").expect("write");
        std::fs::write(staging_path.join("k.srt"), b"s").expect("write");

        let reported = staging_path.join("k.mp4");
        let final_path = staging.adopt_into(&dir, &reported).expect("adopt");
        assert_eq!(final_path, dir.join("k.mp4"), "reported path maps to dest");
        assert!(dir.join("k.srt").exists(), "sidecar adopted alongside");

        drop(staging);
        assert!(!staging_path.exists(), "staging removed once adopted");
    }

    #[test]
    fn promoted_layout_yields_exactly_one_core_sidecar() {
        // The point of the `_subs` namespace: after a multi-language
        // download plus promotion, core's auto-detection sees exactly one
        // sidecar instead of failing as ambiguous.
        let dir = temp_dir("sidecar");
        std::fs::write(dir.join("k.mp4"), b"v").expect("write");
        std::fs::write(dir.join("k_subs.en.srt"), b"en").expect("write");
        std::fs::write(dir.join("k_subs.zh.srt"), b"zh").expect("write");
        promote_subtitle(&dir, "k", &langs(&["en", "zh"])).expect("promote");

        let sidecars = agent_video_core::subtitle::find_sidecars(&dir.join("k.mp4"));
        assert_eq!(sidecars, vec![dir.join("k.srt")]);
    }

    /// A long-sleeping child in its own process group, as production spawns
    /// them, so the watchdog's group kill cannot reach the test process.
    #[cfg(unix)]
    fn sleeping_child() -> Child {
        crate::proc::isolate_group(&mut Command::new("sleep"))
            .arg("60")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("spawn sleep")
    }

    #[cfg(unix)]
    const TEST_LIMITS: StallLimits = StallLimits {
        silence: Duration::from_millis(100),
        no_progress: Duration::from_millis(250),
        lifetime: Duration::from_secs(60),
    };

    #[cfg(unix)]
    #[test]
    fn guard_drop_kills_the_child_before_unwinding_continues() {
        // The Fold::Dead race: a panicking download thread drops its
        // reporter, the tracker re-admits the URL at once, and a surviving
        // yt-dlp would share the keep path with the retry. The guard's
        // Drop must therefore not return until the child is dead.
        let child = sleeping_child();
        let pid = child.id() as i32;
        let (tx, rx) = mpsc::channel::<Pulse>();
        let guard = WatchdogGuard::spawn(child, rx, tx, TEST_LIMITS);
        drop(guard);
        // SAFETY: signal 0 probes existence only; the pid was just reaped
        // by the guard's kill path, so ESRCH is the expected verdict.
        let alive = unsafe { libc::kill(pid, 0) } == 0;
        assert!(!alive, "the child must be dead when Drop returns");
    }

    #[cfg(unix)]
    #[test]
    fn silent_child_is_killed_as_stalled() {
        let child = sleeping_child();
        let (_tx, rx) = mpsc::channel::<Pulse>();
        let outcome = watch_child(child, rx, TEST_LIMITS);
        assert!(matches!(outcome, ChildOutcome::Stalled), "{outcome:?}");
    }

    #[cfg(unix)]
    #[test]
    fn chatter_without_bytes_is_killed_as_no_progress() {
        // Chatter faster than the silence limit, never a byte: the old
        // single-clock watchdog lived forever here while the tracker
        // abandoned the slot — the orphan-process bug.
        //
        // Load-tolerant limits: the verdict still requires chatter to have
        // reset the silence clock (silence < no_progress), but the nominal
        // 50ms chatter gap sits 40× under the silence limit, so a scheduling
        // stall on a contended machine cannot misread chatter as Stalled.
        let child = sleeping_child();
        let limits = StallLimits {
            silence: Duration::from_secs(2),
            no_progress: Duration::from_secs(5),
            lifetime: Duration::from_secs(60),
        };
        let (tx, rx) = mpsc::channel::<Pulse>();
        let chatterer = std::thread::spawn(move || {
            // Chatter until the watchdog hangs up (its verdict drops the
            // receiver) — no fixed count that could fall short under load.
            while tx.send(Pulse::Chatter).is_ok() {
                std::thread::sleep(Duration::from_millis(50));
            }
        });
        let outcome = watch_child(child, rx, limits);
        chatterer.join().expect("chatterer");
        assert!(matches!(outcome, ChildOutcome::NoProgress), "{outcome:?}");
    }

    #[cfg(unix)]
    #[test]
    fn byte_pulses_reset_both_clocks() {
        // Bytes arriving slower than the silence limit but faster than the
        // no-progress limit must keep the child alive: a slow but moving
        // download is progress, not a stall.
        let child = crate::proc::isolate_group(&mut Command::new("true"))
            .stdin(Stdio::null())
            .spawn()
            .expect("spawn true");
        let limits = StallLimits {
            silence: Duration::from_millis(150),
            no_progress: Duration::from_millis(300),
            lifetime: Duration::from_secs(60),
        };
        let (tx, rx) = mpsc::channel::<Pulse>();
        let pulser = std::thread::spawn(move || {
            // 8 × 50ms = 400ms of liveness, past both raw limits.
            for _ in 0..8 {
                if tx.send(Pulse::Bytes).is_err() {
                    break;
                }
                std::thread::sleep(Duration::from_millis(50));
            }
            // Sender drops: EOF, the exited child is reaped as Exited.
        });
        let outcome = watch_child(child, rx, limits);
        pulser.join().expect("pulser");
        assert!(matches!(outcome, ChildOutcome::Exited(_)), "{outcome:?}");
    }

    #[cfg(unix)]
    #[test]
    fn aborts_a_child_past_the_lifetime_ceiling() {
        // Byte pulses faster than both stall clocks — a drip-feeding
        // server resets them forever; only the lifetime ceiling may end
        // the child.
        let child = sleeping_child();
        let limits = StallLimits {
            silence: Duration::from_millis(200),
            no_progress: Duration::from_millis(400),
            lifetime: Duration::from_millis(300),
        };
        let (tx, rx) = mpsc::channel::<Pulse>();
        let pulser = std::thread::spawn(move || {
            for _ in 0..20 {
                if tx.send(Pulse::Bytes).is_err() {
                    break;
                }
                std::thread::sleep(Duration::from_millis(50));
            }
        });
        let outcome = watch_child(child, rx, limits);
        pulser.join().expect("pulser");
        assert!(matches!(outcome, ChildOutcome::Expired), "{outcome:?}");
    }

    #[test]
    fn caps_stderr_to_a_bounded_tail() {
        let mut tail = TailBuf::new(64);
        for i in 0..100 {
            tail.push_line(&format!("line-{i:03}"));
        }
        let s = tail.into_string();
        assert!(s.len() <= 64, "stays within the byte limit: {}", s.len());
        assert!(!s.contains("line-000"), "oldest lines fall off the front");
        assert!(s.ends_with("line-099\n"), "newest line survives intact");
        assert!(
            s.lines().all(|l| l.len() == "line-000".len()),
            "eviction drops whole lines, never splits one: {s:?}"
        );
    }

    #[test]
    fn keeps_a_single_line_over_the_limit_intact() {
        // One oversized line must not be evicted down to nothing: a whole
        // diagnostic beats an empty buffer.
        let mut tail = TailBuf::new(8);
        tail.push_line("a line far past eight bytes");
        assert_eq!(tail.into_string(), "a line far past eight bytes\n");
    }
}
