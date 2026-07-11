// SPDX-License-Identifier: Apache-2.0
//! yt-dlp source adapter.
//!
//! Probes and downloads 1800+ sites via an optional yt-dlp on PATH; its
//! absence becomes an install hint while direct media links keep working
//! without it. Machine interfaces only: `-J` for probing,
//! `--progress-template` for progress, `--print after_move:filepath` for the
//! final path; the human progress bar is never parsed.

use std::collections::BTreeMap;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::mpsc;
use std::time::Duration;

use serde::Deserialize;

use super::{
    DownloadIo, DownloadOption, Downloaded, ProgressFn, Recipe, SourceInfo, estimate_label,
};
use crate::config::Config;
use crate::cookies::CookieHandle;
use crate::fault::Fault;
use crate::netguard::VettedUrl;
use crate::plan::format_bytes;
use crate::tokens::{Dest, Token};
use agent_video_core::types::format_ts;

const INSTALL_HINT: &str = "yt-dlp not found on PATH — install it (https://github.com/yt-dlp/yt-dlp) to handle \
     site URLs, or download the video manually and pass the local file path as source";

/// How many keep tiers (distinct heights) to offer at most.
pub const KEEP_TIER_LIMIT: usize = 3;

/// Recipe carried by yt-dlp download plans.
#[derive(Debug, Clone, PartialEq)]
pub struct YtdlpRecipe {
    /// yt-dlp `-f` format selector, decided at offer time.
    pub format: String,
    /// Also fetch subtitle sidecars (analysis copy only).
    pub subs: bool,
}

// ── -J document ──

#[derive(Debug, Default, Deserialize)]
pub struct YtdlpFormat {
    #[serde(default)]
    pub height: Option<f64>,
    /// `Some` = the format carries video. yt-dlp encodes absence as the
    /// literal string `"none"`; that sentinel dies at this boundary.
    #[serde(default, deserialize_with = "none_is_absent")]
    pub vcodec: Option<String>,
    /// `Some` = the format carries audio; same sentinel rule as `vcodec`.
    #[serde(default, deserialize_with = "none_is_absent")]
    pub acodec: Option<String>,
    #[serde(default)]
    pub filesize: Option<f64>,
    #[serde(default)]
    pub filesize_approx: Option<f64>,
}

/// Map yt-dlp's `"none"` codec sentinel to a plain absent value, so no
/// in-memory consumer ever compares against the magic string.
fn none_is_absent<'de, D>(deserializer: D) -> Result<Option<String>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let value = Option::<String>::deserialize(deserializer)?;
    Ok(value.filter(|v| v != "none"))
}

#[derive(Debug, Default, Deserialize)]
pub struct YtdlpJson {
    #[serde(default)]
    pub title: Option<String>,
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub duration: Option<f64>,
    #[serde(default)]
    pub subtitles: Option<BTreeMap<String, serde_json::Value>>,
    #[serde(default)]
    pub formats: Option<Vec<YtdlpFormat>>,
}

/// One keep tier derived from a site's format list.
#[derive(Debug, Clone, PartialEq)]
pub struct KeepTier {
    pub height: u32,
    pub estimated_bytes: Option<u64>,
}

/// A site page probed via `yt-dlp -J`.
#[derive(Debug, Clone, PartialEq)]
pub struct SiteUrlInfo {
    pub title: String,
    /// Seconds; live streams may not report one.
    pub duration: Option<f64>,
    pub subtitle_langs: Vec<String>,
    /// Distinct available heights, descending, already capped in count.
    pub keep_tiers: Vec<KeepTier>,
    /// Estimate for the analysis rip (best video ≤ cap plus audio).
    pub analysis_estimated_bytes: Option<u64>,
}

fn size_of(f: &YtdlpFormat) -> Option<u64> {
    f.filesize
        .or(f.filesize_approx)
        .filter(|&n| n > 0.0)
        .map(|n| n as u64)
}

fn height_of(f: &YtdlpFormat) -> u32 {
    f.height.filter(|&h| h > 0.0).map(|h| h as u32).unwrap_or(0)
}

fn is_video(f: &YtdlpFormat) -> bool {
    f.vcodec.is_some() && height_of(f) > 0
}

fn is_audio_only(f: &YtdlpFormat) -> bool {
    f.acodec.is_some() && !is_video(f)
}

/// Largest known size among `fmts`, or `None` when none reports one.
fn max_size<'a>(fmts: impl Iterator<Item = &'a YtdlpFormat>) -> Option<u64> {
    fmts.filter_map(size_of).max()
}

/// Reduce a `-J` document to [`SiteUrlInfo`]; pure. Size estimates mirror
/// download selectors: keep = best video at tier + best audio, analysis =
/// best video-only rendition at or under `analysis_max_height`, plus audio.
pub fn site_info(doc: &YtdlpJson, analysis_max_height: u32) -> SiteUrlInfo {
    let empty = Vec::new();
    let formats = doc.formats.as_ref().unwrap_or(&empty);
    let video: Vec<&YtdlpFormat> = formats.iter().filter(|f| is_video(f)).collect();
    let audio_size = max_size(formats.iter().filter(|f| is_audio_only(f)));

    let mut heights: Vec<u32> = video.iter().map(|f| height_of(f)).collect();
    heights.sort_unstable_by(|a, b| b.cmp(a));
    heights.dedup();
    heights.truncate(KEEP_TIER_LIMIT);

    let keep_tiers: Vec<KeepTier> = heights
        .into_iter()
        .map(|height| {
            let at_height: Vec<&YtdlpFormat> = video
                .iter()
                .copied()
                .filter(|f| height_of(f) == height)
                .collect();
            let video_size = max_size(at_height.iter().copied());
            let muxed = at_height.iter().any(|f| f.acodec.is_some());
            KeepTier {
                height,
                estimated_bytes: video_size
                    .map(|v| v + if muxed { 0 } else { audio_size.unwrap_or(0) }),
            }
        })
        .collect();

    let video_only: Vec<&YtdlpFormat> = video
        .iter()
        .copied()
        .filter(|f| height_of(f) <= analysis_max_height && f.acodec.is_none())
        .collect();
    let pool: Vec<&YtdlpFormat> = if video_only.is_empty() {
        video
            .iter()
            .copied()
            .filter(|f| height_of(f) <= analysis_max_height)
            .collect()
    } else {
        video_only
    };
    let best_height = pool.iter().map(|f| height_of(f)).max();
    let video_estimate =
        best_height.and_then(|h| max_size(pool.iter().copied().filter(|f| height_of(f) == h)));
    // Add audio unless the chosen rendition already carries it.
    let chosen_muxed = best_height
        .map(|h| pool.iter().any(|f| height_of(f) == h && f.acodec.is_some()))
        .unwrap_or(false);
    let analysis_estimated_bytes = video_estimate.map(|v| {
        v + if chosen_muxed {
            0
        } else {
            audio_size.unwrap_or(0)
        }
    });

    SiteUrlInfo {
        title: doc
            .title
            .clone()
            .or_else(|| doc.id.clone())
            .unwrap_or_else(|| "untitled".to_string()),
        duration: doc.duration.filter(|&d| d > 0.0),
        subtitle_langs: doc
            .subtitles
            .as_ref()
            .map(|s| s.keys().cloned().collect())
            .unwrap_or_default(),
        keep_tiers,
        analysis_estimated_bytes,
    }
}

/// Turn site facts into the source info: analysis rip plus keep tiers.
pub fn site_source_info(info: &SiteUrlInfo, cfg: &Config) -> SourceInfo {
    let h = cfg.download.analysis_max_height.get();
    let duration = info
        .duration
        .map(|d| format_ts(d as f32))
        .unwrap_or_else(|| "unknown duration".to_string());
    let subs = if info.subtitle_langs.is_empty() {
        "no subtitles listed".to_string()
    } else {
        format!("subtitles: {}", info.subtitle_langs.join(", "))
    };

    let mut options = vec![DownloadOption {
        token: Token::new("dl-analysis"),
        dest: Dest::Cache,
        label: match info.analysis_estimated_bytes {
            Some(n) => format!("analysis rip ({})", format_bytes(n)),
            None => "analysis rip".to_string(),
        },
        effect: format!(
            "download a rip (≤{h}p video + audio, {}) plus subtitles into the analysis cache \
             (evictable)",
            estimate_label(info.analysis_estimated_bytes)
        ),
        estimated_bytes: info.analysis_estimated_bytes,
        recipe: Recipe::Ytdlp(YtdlpRecipe {
            format: format!("bv[height<={h}]+ba/bv*[height<={h}]+ba/b[height<={h}]/b"),
            subs: true,
        }),
    }];
    for tier in &info.keep_tiers {
        options.push(DownloadOption {
            token: Token::new(format!("dl-keep-{}", tier.height)),
            dest: Dest::Keep,
            label: match tier.estimated_bytes {
                Some(n) => format!("{}p ({})", tier.height, format_bytes(n)),
                None => format!("{}p", tier.height),
            },
            effect: format!(
                "save best audio+video at ≤{}p ({}) to {} for the user",
                tier.height,
                estimate_label(tier.estimated_bytes),
                cfg.paths.keep.display()
            ),
            estimated_bytes: tier.estimated_bytes,
            recipe: Recipe::Ytdlp(YtdlpRecipe {
                format: format!("bv*[height<={h}]+ba/b[height<={h}]/b", h = tier.height),
                subs: false,
            }),
        });
    }

    SourceInfo {
        headline: format!(
            "\"{}\" — {duration}, {subs}. Nothing is downloaded yet.",
            info.title
        ),
        options,
    }
}

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

/// Probe a site page: `yt-dlp -J` reduced through [`site_info`].
pub fn probe(url: &VettedUrl, cfg: &Config) -> Result<SourceInfo, Fault> {
    let cookies = acquire_cookies(cfg)?;
    let mut cmd = Command::new("yt-dlp");
    cmd.args(["-J", "--no-playlist"])
        .args(cookies.args())
        .arg(url.as_str());
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
    /// Silent past [`STALL_LIMIT`]; killed mid-download.
    Stalled,
    /// Closed both pipes but did not exit within [`REAP_GRACE`]; killed.
    Unreaped,
}

/// Own the child until it terminates. Liveness is message arrival: each
/// ping on `activity` resets the stall clock, and all senders dropping
/// (both pipes at EOF) means the work is done and the child only needs
/// reaping. Sole owner of the child, so the kill paths cannot race a wait.
fn watch_child(mut child: Child, activity: mpsc::Receiver<()>) -> ChildOutcome {
    loop {
        match activity.recv_timeout(STALL_LIMIT) {
            Ok(()) => continue,
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
                return ChildOutcome::Stalled;
            }
        }
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
    let mut child = crate::proc::isolate_group(&mut cmd)
        .args(&args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(spawn_error)?;
    let stdout_pipe = child.stdout.take().expect("stdout was piped");
    let stderr_pipe = child.stderr.take().expect("stderr was piped");

    // Any output on either pipe pings the watchdog, so a silent
    // post-processing phase that still logs is not mistaken for a stall.
    let (activity, watchdog_rx) = mpsc::channel::<()>();

    // stderr drains on a helper thread (yt-dlp writes progress noise there
    // too); stdout — progress lines plus the final filepath — parses here.
    let stderr_thread = {
        let activity = activity.clone();
        std::thread::spawn(move || {
            let mut buf = String::new();
            for line in BufReader::new(stderr_pipe).lines() {
                let Ok(line) = line else { break };
                let _ = activity.send(());
                buf.push_str(&line);
                buf.push('\n');
            }
            buf
        })
    };

    // Watchdog: a child that stops producing output has hung (network stall,
    // dead peer). Kill it so the job settles as a retryable failure instead
    // of pinning this thread and the URL's pending slot until server exit.
    let watchdog = std::thread::spawn(move || watch_child(child, watchdog_rx));

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
        let _ = activity.send(());
        if let Some((downloaded, total)) = parse_progress(&line) {
            progress(downloaded, total);
            continue;
        }
        let trimmed = line.trim();
        if !trimmed.is_empty() {
            filepath = Some(trimmed.to_string());
        }
    }

    // Last sender: with the stderr thread's clone gone too, the watchdog
    // sees the disconnect and reaps.
    drop(activity);
    let stderr = stderr_thread.join().unwrap_or_default();
    let outcome = watchdog.join().expect("watchdog thread does not panic");

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
        ChildOutcome::Unreaped => {
            return Err(Fault::transient(
                "yt-dlp did not exit after its output closed and was killed",
            ));
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

    /// Mirrors the wire boundary: `"none"` arrives as absent.
    fn fmt(height: f64, vcodec: &str, acodec: &str, size: Option<f64>) -> YtdlpFormat {
        YtdlpFormat {
            height: Some(height),
            vcodec: (vcodec != "none").then(|| vcodec.to_string()),
            acodec: (acodec != "none").then(|| acodec.to_string()),
            filesize: size,
            filesize_approx: None,
        }
    }

    #[test]
    fn codec_none_sentinel_deserializes_to_absent() {
        let f: YtdlpFormat =
            serde_json::from_str(r#"{"vcodec":"none","acodec":"opus"}"#).expect("valid json");
        assert_eq!(f.vcodec, None, "sentinel dies at the wire");
        assert_eq!(f.acodec.as_deref(), Some("opus"));
    }

    #[test]
    fn reduces_formats_to_keep_tiers_descending() {
        let doc = YtdlpJson {
            title: Some("T".into()),
            formats: Some(vec![
                fmt(360.0, "avc1", "none", Some(10.0)),
                fmt(720.0, "avc1", "none", Some(50.0)),
                fmt(1080.0, "avc1", "none", Some(100.0)),
                fmt(1440.0, "avc1", "none", Some(200.0)),
                fmt(0.0, "none", "opus", Some(5.0)),
            ]),
            ..Default::default()
        };
        let info = site_info(&doc, 480);
        let heights: Vec<u32> = info.keep_tiers.iter().map(|t| t.height).collect();
        assert_eq!(heights, [1440, 1080, 720], "top 3 distinct, descending");
        // Video-only tiers add best audio.
        assert_eq!(info.keep_tiers[0].estimated_bytes, Some(205));
    }

    #[test]
    fn muxed_tier_does_not_add_audio() {
        let doc = YtdlpJson {
            formats: Some(vec![
                fmt(720.0, "avc1", "mp4a", Some(80.0)),
                fmt(0.0, "none", "opus", Some(5.0)),
            ]),
            ..Default::default()
        };
        let info = site_info(&doc, 480);
        assert_eq!(info.keep_tiers[0].estimated_bytes, Some(80));
    }

    #[test]
    fn analysis_estimate_prefers_video_only_under_cap_plus_audio() {
        let doc = YtdlpJson {
            formats: Some(vec![
                fmt(480.0, "avc1", "none", Some(30.0)),
                fmt(360.0, "avc1", "none", Some(20.0)),
                fmt(480.0, "avc1", "mp4a", Some(60.0)),
                fmt(0.0, "none", "opus", Some(5.0)),
            ]),
            ..Default::default()
        };
        let info = site_info(&doc, 480);
        assert_eq!(info.analysis_estimated_bytes, Some(35));
    }

    #[test]
    fn falls_back_to_muxed_when_no_video_only() {
        let doc = YtdlpJson {
            formats: Some(vec![fmt(480.0, "avc1", "mp4a", Some(60.0))]),
            ..Default::default()
        };
        let info = site_info(&doc, 480);
        assert_eq!(info.analysis_estimated_bytes, Some(60));
    }

    #[test]
    fn untitled_falls_back_to_id_then_placeholder() {
        let doc = YtdlpJson {
            id: Some("abc123".into()),
            ..Default::default()
        };
        assert_eq!(site_info(&doc, 480).title, "abc123");
        assert_eq!(site_info(&YtdlpJson::default(), 480).title, "untitled");
    }

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

    #[test]
    fn source_info_offers_analysis_plus_tiers() {
        let cfg = crate::config::default_config(&crate::config::test_dirs(Path::new("/home/u")));
        let info = SiteUrlInfo {
            title: "Talk".into(),
            duration: Some(90.0),
            subtitle_langs: vec!["en".into()],
            keep_tiers: vec![KeepTier {
                height: 1080,
                estimated_bytes: Some(1024 * 1024),
            }],
            analysis_estimated_bytes: Some(512 * 1024),
        };
        let si = site_source_info(&info, &cfg);
        assert_eq!(si.options.len(), 2);
        assert_eq!(si.options[0].token, Token::new("dl-analysis"));
        assert_eq!(si.options[1].token, Token::new("dl-keep-1080"));
        assert!(si.headline.contains("\"Talk\""));
        assert!(si.headline.contains("subtitles: en"));
        let Recipe::Ytdlp(r) = &si.options[0].recipe else {
            panic!("analysis recipe is ytdlp");
        };
        assert_eq!(
            r.format,
            "bv[height<=480]+ba/bv*[height<=480]+ba/b[height<=480]/b"
        );
        assert!(r.subs);
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
}
