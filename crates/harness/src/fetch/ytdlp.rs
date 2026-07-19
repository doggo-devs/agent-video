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

use super::{DownloadIo, Downloaded, ProgressFn, SourceInfo};
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
pub fn promote_subtitle(dir: &Path, key: &str, preferred_langs: &[String]) -> std::io::Result<()> {
    use agent_video_core::subtitle::SIDECAR_SUBTITLE_EXTS;

    let Ok(entries) = std::fs::read_dir(dir) else {
        return Ok(());
    };
    let mut names: Vec<String> = entries
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
    if !target.exists() {
        std::fs::copy(dir.join(name), target)?;
    }
    Ok(())
}

// ── spawning ──

/// Diagnostics only ever read the tail (`stderr_tail`, `is_auth_failure`),
/// so keep at most this many bytes and let whole lines fall off the front.
const STDERR_TAIL_LIMIT: usize = 64 * 1024;

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

/// Adopt rotated cookies back into the canonical jar; best-effort — the
/// media verdict must not depend on jar bookkeeping.
fn commit_cookies(cookies: CookieHandle) {
    if let Err(e) = cookies.commit() {
        tracing::warn!("cookie write-back failed: {e}");
    }
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
    // verdict (a deadline-killed child left an empty delta).
    commit_cookies(cookies);
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
    let doc: YtdlpJson = serde_json::from_slice(&out.stdout.bytes)
        .map_err(|_| Fault::environment("yt-dlp -J emitted invalid JSON — version too old?"))?;
    Ok(site_source_info(
        &site_info(&doc, cfg.download.analysis_max_height.get()),
        cfg,
    ))
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
/// `.part` handling interleaves their writes. Ownership is per writer,
/// not per process: a pid alone would let two threads of one server
/// share a directory. Each job downloads into its own directory and
/// adopts the results into the cache with per-file renames (same
/// filesystem, atomic per writer; last-writer-wins is safe because equal
/// keys mean equal recipes). Removed on drop; a crash leaves it for
/// [`crate::cache::Cache::evict`]'s stale-staging sweep.
struct StagingDir {
    path: PathBuf,
}

/// Prefix marking staging directories; the eviction sweep recognizes and
/// ages them out after a crash.
pub const STAGING_PREFIX: &str = ".staging-";

impl StagingDir {
    fn create(parent: &Path, key: &str) -> std::io::Result<Self> {
        use std::sync::atomic::{AtomicU64, Ordering};
        static WRITER_SEQ: AtomicU64 = AtomicU64::new(0);
        let seq = WRITER_SEQ.fetch_add(1, Ordering::Relaxed);
        let path = parent.join(format!(
            "{STAGING_PREFIX}{key}-{}-{seq}",
            std::process::id()
        ));
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
pub fn download(
    url: &VettedUrl,
    recipe: &YtdlpRecipe,
    io: DownloadIo<'_>,
    cfg: &Config,
    relay: &PolicyRelay,
    progress: ProgressFn<'_>,
) -> Result<Downloaded, Fault> {
    let (staging, output, subtitle_output, promote_key) = match io {
        DownloadIo::Cache { dir, key } => {
            let staging = StagingDir::create(dir, key).map_err(|e| {
                Fault::environment(format!("cannot create a staging directory in cache: {e}"))
            })?;
            let output = staging.path.join(format!("{key}.%(ext)s"));
            // Outside the media's sidecar namespace; see [`SUBS_INFIX`].
            let subs = recipe.subs.then(|| {
                staging
                    .path
                    .join(format!("{key}{SUBS_INFIX}.%(lang)s.%(ext)s"))
            });
            let promote = recipe.subs.then(|| key.to_string());
            (Some((staging, dir.to_path_buf())), output, subs, promote)
        }
        DownloadIo::Keep { dir } => (None, dir.join("%(title)s [%(id)s].%(ext)s"), None, None),
    };

    let mut args: Vec<String> = [
        "--no-playlist",
        "--no-simulate",
        "--quiet",
        "--progress",
        "--newline",
        "--progress-delta",
        "1",
        "--progress-template",
        PROGRESS_TEMPLATE,
        "--print",
        "after_move:filepath",
        "-f",
        &recipe.format,
        "--max-filesize",
        &cfg.download.max_bytes.get().to_string(),
        "-o",
    ]
    .into_iter()
    .map(String::from)
    .collect();
    // Every yt-dlp connection (and its ffmpeg children, via the env below)
    // routes through the policy relay.
    args.push("--proxy".to_string());
    args.push(relay.url());
    args.push(output.to_string_lossy().into_owned());
    if let Some(sub) = &subtitle_output {
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
    let cookies = acquire_cookies(cfg)?;
    args.extend(cookies.args());
    args.push(url.as_str().to_string());

    let mut cmd = Command::new("yt-dlp");
    set_proxy_env(&mut cmd, relay);
    let mut child = crate::proc::isolate_group(&mut cmd)
        .args(&args)
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
            let mut tail = TailBuf::new(STDERR_TAIL_LIMIT);
            for line in BufReader::new(stderr_pipe).lines() {
                let Ok(line) = line else { break };
                let _ = activity.send(Pulse::Chatter);
                tail.push_line(&line);
            }
            tail.into_string()
        })
    };

    // Watchdog: a child that stops producing output has hung (network stall,
    // dead peer). Kill it so the job settles as a retryable failure instead
    // of pinning this thread and the URL's pending slot until server exit.
    // The guard additionally kills the child if this thread unwinds — see
    // [`WatchdogGuard`].
    let watchdog = WatchdogGuard::spawn(child, watchdog_rx, activity.clone(), STALL_LIMITS);

    let mut filepath: Option<String> = None;
    let mut read_error: Option<std::io::Error> = None;
    for line in BufReader::new(stdout_pipe).lines() {
        let line = match line {
            Ok(line) => line,
            // A failed pipe read is not EOF: record it, or a lost filepath
            // line gets misdiagnosed as "no file produced".
            Err(e) => {
                read_error = Some(e);
                break;
            }
        };
        if let Some((downloaded, total)) = parse_progress(&line) {
            // The one site that claims byte progress — the same parse feeds
            // the tracker, so both stall clocks agree on what "moving" means.
            let _ = activity.send(Pulse::Bytes);
            progress(downloaded, total);
            continue;
        }
        let _ = activity.send(Pulse::Chatter);
        let trimmed = line.trim();
        if !trimmed.is_empty() {
            filepath = Some(trimmed.to_string());
        }
    }

    // Last sender: with the stderr thread's clone gone too, the watchdog
    // sees the disconnect and reaps.
    drop(activity);
    let stderr = stderr_thread.join().unwrap_or_default();
    let outcome = watchdog.join();

    // Whatever the verdict: a normal exit and a graceful abort both saved
    // the staged jar, a hard-killed child left an empty delta.
    commit_cookies(cookies);

    let warning = match outcome {
        ChildOutcome::Stalled => {
            return Err(Fault::transient(format!(
                "the download stalled — no output from yt-dlp for {}s — and was aborted; \
                 confirm again to retry",
                STALL_LIMIT.as_secs()
            )));
        }
        ChildOutcome::NoProgress => {
            return Err(Fault::transient(format!(
                "yt-dlp kept producing output but moved no bytes for {}s — likely wedged \
                 in a retry loop — and was aborted; confirm again to retry",
                NO_PROGRESS_LIMIT.as_secs()
            )));
        }
        ChildOutcome::Expired => {
            return Err(Fault::transient(format!(
                "the download exceeded the {}h lifetime ceiling and was aborted — the \
                 source may be serving impossibly slowly; confirm again to retry",
                LIFETIME_CEILING.as_secs() / 3600
            )));
        }
        ChildOutcome::Unreaped => {
            return Err(Fault::transient(
                "yt-dlp did not exit after its output closed and was killed",
            ));
        }
        // Only the guard's Drop sends Abort, and that path discards the
        // outcome while unwinding — but the type cannot promise it.
        ChildOutcome::Aborted => {
            return Err(Fault::transient("the download was aborted mid-flight"));
        }
        ChildOutcome::Exited(status) => {
            let media_landed = filepath.as_deref().is_some_and(|p| Path::new(p).is_file());
            settle_exit(status, media_landed, &stderr)?
        }
    };
    if let Some(e) = read_error {
        return Err(Fault::transient(format!(
            "reading yt-dlp output failed mid-download: {e}"
        )));
    }
    let Some(filepath) = filepath else {
        return Err(Fault::bad_input(format!(
            "yt-dlp finished without producing a file — the size cap (download.max_bytes) \
             may have aborted it: {}",
            stderr_tail(&stderr)
        )));
    };
    let filepath = PathBuf::from(filepath);
    let Some((staging, dest_dir)) = staging else {
        return Ok(Downloaded {
            path: filepath,
            warning,
        });
    };
    // Elect the one sidecar before the results become visible to other
    // processes: adoption must publish a complete entry.
    if let Some(key) = promote_key {
        let _ = promote_subtitle(&staging.path, &key, &cfg.subtitles.preferred_langs);
    }
    let path = staging
        .adopt_into(&dest_dir, &filepath)
        .map_err(|e| Fault::environment(format!("cannot move the download into the cache: {e}")))?;
    Ok(Downloaded { path, warning })
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

    fn temp_dir(tag: &str) -> crate::testutil::TempDir {
        crate::testutil::TempDir::new(&format!("av-ytdlp-{tag}"))
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
        let child = sleeping_child();
        let (tx, rx) = mpsc::channel::<Pulse>();
        let chatterer = std::thread::spawn(move || {
            // Outlives no_progress (250ms) with gaps well under silence (100ms).
            for _ in 0..40 {
                if tx.send(Pulse::Chatter).is_err() {
                    break;
                }
                std::thread::sleep(Duration::from_millis(10));
            }
        });
        let outcome = watch_child(child, rx, TEST_LIMITS);
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
