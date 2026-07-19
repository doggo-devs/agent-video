// SPDX-License-Identifier: Apache-2.0
//! Direct http(s) source.
//!
//! Probes direct media links in-process (the demuxer reads the container
//! header over http without downloading the payload) and streams downloads
//! to `<dest>.part` against a hard cap, so a file at the final path is
//! always whole.
//!
//! Stall discipline: ureq's per-phase timeouts cannot express "abort when
//! the peer goes silent" (see [`agent`]), so every connection is wrapped in
//! a [`StallGuard`] transport that bounds each receive wait to a short
//! tick. Sends are not guarded — request heads fit the socket buffer and
//! the whole-call ceiling still bounds them. The guard aborts a transfer
//! whose peer sent nothing for [`IDLE_LIMIT`]
//! and observes the job's [`CancelSignal`] every tick, so a download whose
//! tracker slot was abandoned stops promptly instead of pinning its thread,
//! socket, `.part` file, and reserved keep filename.

use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use ureq::unversioned::transport::{
    Buffers, ConnectionDetails, Connector, DefaultConnector, NextTimeout, Transport,
    time::Duration as UreqDuration,
};

use super::{
    CancelSignal, DownloadIo, DownloadOption, Downloaded, ProgressFn, Recipe, SourceInfo,
    estimate_label,
};
use crate::config::Config;
use crate::fault::Fault;
use crate::netguard::{GuardedResolver, VettedUrl};
use crate::plan::format_bytes;
use crate::probe::{container_label, is_ext_shaped, probe_media_url};
use crate::relay::PolicyRelay;
use crate::schema::ProbeSummary;
use crate::tokens::{Dest, Token};
use agent_video_core::types::format_ts;

/// Minimum interval between progress callbacks.
const PROGRESS_INTERVAL: Duration = Duration::from_secs(1);

/// Connection establishment bound for HEAD/GET.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(15);

/// A peer that sends nothing for this long is presumed dead and the
/// transfer is aborted — the direct client's stall bound, enforced by
/// [`StallGuard`] on every receive wait (headers and body alike). Kept
/// inside the tracker's abandon limit by a compile-time assertion in
/// `crate::mcp::download`, so a stalled read always unblocks before the
/// slot it reports to is given up.
pub(crate) const IDLE_LIMIT: Duration = Duration::from_secs(60);

/// Granularity of [`StallGuard`]'s waits: the longest a blocked socket
/// read can go without re-checking the job's [`CancelSignal`], and thus
/// the bound on how long a cancelled transfer keeps waiting on the peer.
const CANCEL_TICK: Duration = Duration::from_secs(2);

/// Whole-call ceiling for a download (connect through last body byte). A
/// termination guarantee against a peer that dribbles forever, not stall
/// detection — [`IDLE_LIMIT`] handles silence. Sized for
/// `download.max_bytes` on slow links.
const BODY_CEILING: Duration = Duration::from_secs(4 * 60 * 60);

/// Error note for a transfer aborted because its job was disowned.
const CANCELLED_NOTE: &str = "download cancelled: its tracker slot was abandoned";

/// Agent wired for one download job: explicit deadlines, the
/// egress-screening resolver (every connection this client makes —
/// redirect targets included — passes the host policy), and a
/// [`StallGuard`] wrapping every transport.
///
/// Deadlines are deliberately only `connect` + `global`. ureq 3 has no
/// idle timeout, and its phase timeouts cannot fill in: `recv_body` is
/// re-anchored at every socket wait (a bound per read, never per body),
/// and a configured `recv_response` leaks a stale `headers_end +
/// recv_response` deadline into the body phase, killing any transfer that
/// pauses ≥ 1s past that mark (verified against ureq 3.3.0). So the agent
/// keeps ureq's phase timeouts off and the [`StallGuard`] owns stall
/// detection uniformly: no receive wait outlives [`IDLE_LIMIT`] of peer
/// silence.
fn agent(cfg: &Config, cancel: &CancelSignal) -> ureq::Agent {
    let config = ureq::Agent::config_builder()
        .timeout_connect(Some(CONNECT_TIMEOUT))
        .timeout_global(Some(BODY_CEILING))
        .build();
    let connector = DefaultConnector::default().chain(StallGuardConnector(cancel.clone()));
    ureq::Agent::with_parts(config, connector, GuardedResolver::new(cfg))
}

/// Terminal link of the connector chain: wraps whatever transport the
/// default chain produced (TCP or TLS) in a [`StallGuard`] sharing the
/// job's cancel signal.
#[derive(Debug)]
struct StallGuardConnector(CancelSignal);

impl Connector<Box<dyn Transport>> for StallGuardConnector {
    type Out = StallGuard<Box<dyn Transport>>;

    fn connect(
        &self,
        _details: &ConnectionDetails,
        chained: Option<Box<dyn Transport>>,
    ) -> Result<Option<Self::Out>, ureq::Error> {
        Ok(chained.map(|inner| StallGuard::new(inner, self.0.clone(), CANCEL_TICK, IDLE_LIMIT)))
    }
}

/// Transport wrapper that chops every receive wait into `tick`-sized
/// slices: each slice re-checks the cancel signal, and slices accumulate
/// into at most `idle_limit` of peer silence before the wait fails. Sends
/// pass through unguarded — see the module doc. Data flowing resets
/// nothing here — a wait that yields bytes returns immediately, and
/// the next wait starts a fresh idle budget — which is exactly an idle
/// timeout. The caller's own deadline (the global ceiling) still applies
/// as an upper bound on every slice.
struct StallGuard<T> {
    inner: T,
    cancel: CancelSignal,
    tick: Duration,
    idle_limit: Duration,
}

impl<T: Transport> StallGuard<T> {
    fn new(inner: T, cancel: CancelSignal, tick: Duration, idle_limit: Duration) -> Self {
        StallGuard {
            inner,
            cancel,
            tick,
            idle_limit,
        }
    }
}

impl<T> std::fmt::Debug for StallGuard<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StallGuard").finish_non_exhaustive()
    }
}

impl<T: Transport> Transport for StallGuard<T> {
    fn buffers(&mut self) -> &mut dyn Buffers {
        self.inner.buffers()
    }

    fn transmit_output(&mut self, amount: usize, timeout: NextTimeout) -> Result<(), ureq::Error> {
        self.inner.transmit_output(amount, timeout)
    }

    fn await_input(&mut self, timeout: NextTimeout) -> Result<bool, ureq::Error> {
        let started = Instant::now();
        // `NotHappening` derefs to an effectively-infinite duration.
        let ceiling = *timeout.after;
        loop {
            if self.cancel.is_cancelled() {
                return Err(ureq::Error::Io(std::io::Error::other(CANCELLED_NOTE)));
            }
            let idle_left = self.idle_limit.saturating_sub(started.elapsed());
            if idle_left.is_zero() {
                return Err(ureq::Error::Io(std::io::Error::other(format!(
                    "no data from the peer for {}s — presuming the transfer dead",
                    self.idle_limit.as_secs()
                ))));
            }
            let ceiling_left = ceiling.saturating_sub(started.elapsed());
            if ceiling_left.is_zero() {
                return Err(ureq::Error::Timeout(timeout.reason));
            }
            let slice = self.tick.min(idle_left).min(ceiling_left);
            let slice_timeout = NextTimeout {
                after: UreqDuration::Exact(slice),
                reason: timeout.reason,
            };
            match self.inner.await_input(slice_timeout) {
                Ok(made_progress) => return Ok(made_progress),
                // A slice expiring is our tick, not a failure: loop back to
                // the cancel and idle checks.
                Err(ureq::Error::Timeout(_)) => {}
                Err(ureq::Error::Io(e)) if e.kind() == std::io::ErrorKind::TimedOut => {}
                Err(e) => return Err(e),
            }
        }
    }

    fn is_open(&mut self) -> bool {
        self.inner.is_open()
    }

    fn is_tls(&self) -> bool {
        self.inner.is_tls()
    }
}

/// Recipe carried by direct-link download plans.
#[derive(Debug, Clone, PartialEq)]
pub struct DirectRecipe {
    /// File extension for the cache path.
    pub ext: String,
    /// Human-readable filename used when dest is keep.
    pub filename: String,
}

/// Probe a URL as a direct media link.
///
/// Any failure — an HTML page the demuxer rejects, a DNS error, a timeout —
/// is returned with its real reason; libav cannot reliably tell "not media"
/// from "network trouble", so this source never claims that distinction.
/// The chain in [`super::probe_url`] falls through to yt-dlp either way and
/// only surfaces this error when yt-dlp fails too.
pub fn probe(url: &VettedUrl, cfg: &Config, relay: &PolicyRelay) -> Result<SourceInfo, Fault> {
    let mut probe = probe_media_url(url.as_str(), relay)?;
    if probe.size_bytes.is_none() {
        probe.size_bytes = head_content_length(url, cfg);
    }
    Ok(direct_info(url.as_str(), &probe, cfg))
}

/// Build the source info of a direct media link: one file, one keep tier.
pub fn direct_info(url: &str, probe: &ProbeSummary, cfg: &Config) -> SourceInfo {
    let recipe = direct_identity(url, probe);
    let size = probe.size_bytes;
    let size_text = match size {
        Some(n) => format_bytes(n),
        None => "size unknown".to_string(),
    };
    let label = format!("the file ({size_text})");
    SourceInfo {
        headline: format!(
            "Direct media link: {}x{}, {}, {size_text}. Nothing is downloaded yet.",
            probe.width,
            probe.height,
            format_ts(probe.duration as f32)
        ),
        // A direct link has no title — only a filename, which is noise,
        // not vocabulary; transcription derives no prompt from it.
        title: None,
        options: vec![
            DownloadOption {
                token: Token::new("dl-analysis"),
                dest: Dest::Cache,
                label: label.clone(),
                effect: format!(
                    "download the file ({}) into the analysis cache (evictable) and analyze \
                     from there",
                    estimate_label(size)
                ),
                estimated_bytes: size,
                recipe: Recipe::Direct(recipe.clone()),
            },
            DownloadOption {
                token: Token::new(format!("dl-keep-{}", probe.height)),
                dest: Dest::Keep,
                label,
                effect: format!(
                    "save the file ({}) to {} for the user",
                    estimate_label(size),
                    cfg.paths.keep.join(&recipe.filename).display()
                ),
                estimated_bytes: size,
                recipe: Recipe::Direct(recipe),
            },
        ],
    }
}

/// Execute a direct download plan into cache or keep space.
///
/// Cancellation (the tracker disowned the job) surfaces as an error from
/// the fetch, so it shares the failure path: the `.part` file is removed
/// and a reserved keep filename is released — a retry can take the
/// original name instead of inheriting a squatting placeholder.
pub fn download(
    url: &VettedUrl,
    recipe: &DirectRecipe,
    io: DownloadIo<'_>,
    cfg: &Config,
    cancel: &CancelSignal,
    progress: ProgressFn<'_>,
) -> Result<Downloaded, Fault> {
    let (dest, reserved) = match io {
        DownloadIo::Cache { dir, key } => (dir.join(format!("{key}.{}", recipe.ext)), false),
        DownloadIo::Keep { dir } => (reserve_available_path(&dir.join(&recipe.filename))?, true),
    };
    if let Err(e) = download_direct(url, &dest, cfg, cancel, progress) {
        // Release the name: a failed keep download must not leave an empty
        // placeholder squatting on the user's filename.
        if reserved {
            let _ = std::fs::remove_file(&dest);
        }
        return Err(e);
    }
    Ok(Downloaded {
        path: dest,
        warning: None,
    })
}

// ── link identity ──

/// Extension plus a keep-dir filename for a direct media link.
pub fn direct_identity(url: &str, probe: &ProbeSummary) -> DirectRecipe {
    let base = url_basename(url)
        .map(|b| sanitize_filename(&b))
        .unwrap_or_default();
    let url_ext = Path::new(&base)
        .extension()
        .and_then(|e| e.to_str())
        .map(str::to_lowercase)
        .unwrap_or_default();
    let ext = if is_ext_shaped(&url_ext) {
        url_ext.clone()
    } else {
        // The probe canonicalizes container to an ext-shaped label already;
        // the mapping also swallows raw demuxer names from older callers.
        container_label(probe.container.as_deref().unwrap_or(""))
    };
    let filename = if base.is_empty() || base == format!(".{url_ext}") {
        format!("download.{ext}")
    } else if url_ext.is_empty() {
        format!("{base}.{ext}")
    } else {
        base
    };
    DirectRecipe { ext, filename }
}

/// Last path segment of a URL, percent-decoded; `None` when the URL does
/// not parse as a `Uri` (ureq would reject it at download time anyway).
/// Malformed escapes pass through verbatim and invalid UTF-8 decodes
/// lossily — the result is a best-effort filename, and `sanitize_filename`
/// guards it downstream.
fn url_basename(url: &str) -> Option<String> {
    let uri: ureq::http::Uri = url.parse().ok()?;
    let segment = uri.path().rsplit('/').next().unwrap_or("");
    Some(
        percent_encoding::percent_decode_str(segment)
            .decode_utf8_lossy()
            .into_owned(),
    )
}

/// Strip path separators, reserved punctuation, and control characters.
fn sanitize_filename(name: &str) -> String {
    name.chars()
        .filter(|c| !matches!(c, '/' | '\\' | '<' | '>' | ':' | '"' | '|' | '?' | '*'))
        .filter(|c| !c.is_control())
        .collect::<String>()
        .trim()
        .to_string()
}

/// First non-colliding variant of `p` (`name.ext`, `name (1).ext`, …),
/// *reserved* at return: the winning name is claimed with `create_new`
/// (O_EXCL), so the probe and the claim are one atomic step and two
/// concurrent downloads of the same filename can never both adopt it — a
/// bare `exists()` probe would let the loser silently overwrite the
/// winner at publish time. The zero-byte placeholder is replaced by the
/// final rename on success and removed by the caller on failure; a crash
/// leaves an empty file behind, harmless — the next reservation skips it.
pub fn reserve_available_path(p: &Path) -> Result<PathBuf, Fault> {
    let dir = p.parent().unwrap_or(Path::new("."));
    std::fs::create_dir_all(dir)
        .map_err(|e| Fault::environment(format!("cannot create {}: {e}", dir.display())))?;
    let ext = p.extension().and_then(|e| e.to_str());
    let stem = p.file_stem().and_then(|s| s.to_str()).unwrap_or("download");
    for n in 0..100 {
        let candidate = if n == 0 {
            p.to_path_buf()
        } else {
            match ext {
                Some(ext) => dir.join(format!("{stem} ({n}).{ext}")),
                None => dir.join(format!("{stem} ({n})")),
            }
        };
        match std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&candidate)
        {
            Ok(_) => return Ok(candidate),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {}
            Err(e) => {
                return Err(Fault::environment(format!(
                    "cannot reserve {}: {e}",
                    candidate.display()
                )));
            }
        }
    }
    Err(Fault::environment(format!(
        "could not find a free filename for {} in {}",
        p.display(),
        dir.display()
    )))
}

// ── plumbing ──

/// Content-Length via HEAD, or `None` when the server hides it. Probes
/// have no tracker job, so the agent gets a signal nobody can trip; the
/// stall guard's idle bound still applies.
pub fn head_content_length(url: &VettedUrl, cfg: &Config) -> Option<u64> {
    let res = agent(cfg, &CancelSignal::new())
        .head(url.as_str())
        .call()
        .ok()?;
    res.body().content_length().filter(|&len| len > 0)
}

/// Download `url` to `dest`, refusing to exceed `max_bytes`: known oversize
/// is refused before the first byte, unknown-length streams are cut off
/// mid-flight. The `.part` file never survives a failure.
pub fn download_direct(
    url: &VettedUrl,
    dest: &Path,
    cfg: &Config,
    cancel: &CancelSignal,
    progress: ProgressFn<'_>,
) -> Result<u64, Fault> {
    let max_bytes = cfg.download.max_bytes.get();
    let mut res = agent(cfg, cancel)
        .get(url.as_str())
        .call()
        .map_err(|e| match e {
            // A definitive 4xx is the URL's fault; everything else — 5xx,
            // transport, DNS — may pass.
            ureq::Error::StatusCode(code) if (400..500).contains(&code) => {
                Fault::bad_input(format!("download failed: GET {url} → HTTP {code}"))
            }
            ureq::Error::StatusCode(code) => {
                Fault::transient(format!("download failed: GET {url} → HTTP {code}"))
            }
            other => Fault::transient(format!("download failed: GET {url} → {other}")),
        })?;
    let total = res.body().content_length().filter(|&len| len > 0);
    if let Some(total) = total
        && total > max_bytes
    {
        return Err(Fault::bad_input(format!(
            "the file is {}, past the {} ceiling (download.max_bytes) — download it manually \
             and pass the local path as source",
            format_bytes(total),
            format_bytes(max_bytes)
        )));
    }

    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| Fault::environment(format!("cannot create {}: {e}", parent.display())))?;
    }
    let part = crate::cache::naming::part_path(dest);
    let result = stream_to_part(
        &mut res.body_mut().as_reader(),
        &part,
        max_bytes,
        total,
        cancel,
        progress,
    )
    .and_then(|(file, bytes)| publish(file, &part, dest, cancel).map(|()| bytes));
    result.inspect_err(|_| {
        let _ = std::fs::remove_file(&part);
    })
}

/// Publish a fully-written staging file at its final name. The staging
/// name comes from [`crate::cache::naming::part_path`] — writer-unique,
/// because the cache directory is shared across server processes and
/// cache keys are deterministic, so a fixed `.part` name would let two
/// processes interleave writes into one file and rename the corruption
/// into the cache; last-writer-wins on the rename is safe because equal
/// keys mean equal recipes. `sync_all` runs before the rename — the
/// writer's own handle, no reopen — so a crash can never leave the
/// published name pointing at bytes still in the page cache; the rename
/// is atomic per writer. A crash between sync and rename leaves only a
/// `.part` for the staleness sweep. The directory entry itself is not
/// fsynced, accepted: losing the rename to a power cut is a cache miss,
/// never a torn published file.
///
/// A disowned job must not publish: its settlement is void, and for a
/// keep destination the published file would orphan under a name the
/// retry can no longer take. The cancel check sits between the sync
/// (seconds on a large file) and the rename, the narrowest window this
/// side of the syscall; a cancellation landing inside it publishes anyway
/// — a rename cannot be checked-and-done atomically against the signal —
/// and is compensated upstream: the disowned settlement removes the
/// published file (`crate::mcp::download`), so only a process crash in
/// that window leaves an orphan.
fn publish(
    file: std::fs::File,
    part: &Path,
    dest: &Path,
    cancel: &CancelSignal,
) -> Result<(), Fault> {
    file.sync_all()
        .map_err(|e| Fault::environment(format!("cannot sync {}: {e}", part.display())))?;
    if cancel.is_cancelled() {
        return Err(Fault::transient(CANCELLED_NOTE));
    }
    std::fs::rename(part, dest)
        .map_err(|e| Fault::environment(format!("cannot finalize {}: {e}", dest.display())))
}

/// Stream the body into `part`; returns the still-open write handle so the
/// caller can decide durability before publishing — ownership of the bytes
/// and of the fsync travel together. The cancel signal is checked before
/// every read: while data flows, that is at least once per received chunk,
/// and a read blocked on a silent peer is unblocked by the stall guard
/// within [`CANCEL_TICK`].
fn stream_to_part(
    reader: &mut dyn Read,
    part: &Path,
    max_bytes: u64,
    total: Option<u64>,
    cancel: &CancelSignal,
    progress: ProgressFn<'_>,
) -> Result<(std::fs::File, u64), Fault> {
    let mut file = std::fs::File::create(part)
        .map_err(|e| Fault::environment(format!("cannot write {}: {e}", part.display())))?;
    let mut buf = [0u8; 64 * 1024];
    let mut bytes: u64 = 0;
    let mut last_report = Instant::now();
    loop {
        if cancel.is_cancelled() {
            return Err(Fault::transient(CANCELLED_NOTE));
        }
        let n = reader
            .read(&mut buf)
            .map_err(|e| Fault::transient(format!("download failed mid-stream: {e}")))?;
        if n == 0 {
            break;
        }
        bytes += n as u64;
        if bytes > max_bytes {
            return Err(Fault::bad_input(format!(
                "download aborted at {}: the stream exceeded the {} ceiling \
                 (download.max_bytes) — download it manually",
                format_bytes(bytes),
                format_bytes(max_bytes)
            )));
        }
        file.write_all(&buf[..n])
            .map_err(|e| Fault::environment(format!("cannot write {}: {e}", part.display())))?;
        if last_report.elapsed() >= PROGRESS_INTERVAL {
            last_report = Instant::now();
            progress(bytes, total);
        }
    }
    progress(bytes, total.or(Some(bytes)));
    Ok((file, bytes))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn probe_with(container: Option<&str>, height: u32, size: u64) -> ProbeSummary {
        ProbeSummary {
            duration: 90.0,
            width: 1280,
            height,
            size_bytes: (size > 0).then_some(size),
            container: container.map(String::from),
            subtitles: vec![],
        }
    }

    #[test]
    fn takes_extension_from_url_path() {
        let r = direct_identity(
            "https://cdn.example.com/videos/talk.MKV?sig=abc",
            &probe_with(Some("matroska,webm"), 720, 0),
        );
        assert_eq!(r.ext, "mkv");
        assert_eq!(r.filename, "talk.MKV");
    }

    #[test]
    fn decodes_percent_escapes_in_filename() {
        let r = direct_identity(
            "https://ex.com/My%20Talk.mp4",
            &probe_with(Some("mov,mp4"), 720, 0),
        );
        assert_eq!(r.filename, "My Talk.mp4");
    }

    #[test]
    fn falls_back_to_container_extension() {
        let r = direct_identity(
            "https://ex.com/stream",
            &probe_with(Some("matroska,webm"), 720, 0),
        );
        assert_eq!(r.ext, "mkv");
        assert_eq!(r.filename, "stream.mkv");
    }

    #[test]
    fn names_pathless_urls_download() {
        let r = direct_identity("https://ex.com/", &probe_with(Some("mov,mp4"), 720, 0));
        assert_eq!(r.filename, "download.mp4");
    }

    #[test]
    fn sanitizes_reserved_characters() {
        let r = direct_identity(
            "https://ex.com/a%3Cb%3Ec.mp4",
            &probe_with(Some("mov,mp4"), 720, 0),
        );
        assert_eq!(r.filename, "abc.mp4");
    }

    #[test]
    fn offers_analysis_and_keep_tiers() {
        let cfg = crate::config::default_config(&crate::config::test_dirs(Path::new("/home/u")));
        let info = direct_info(
            "https://ex.com/v.mp4",
            &probe_with(Some("mov,mp4"), 720, 100 * 1024 * 1024),
            &cfg,
        );
        assert_eq!(info.options.len(), 2);
        assert_eq!(info.options[0].token, Token::new("dl-analysis"));
        assert_eq!(info.options[0].dest, Dest::Cache);
        assert_eq!(info.options[1].token, Token::new("dl-keep-720"));
        assert_eq!(info.options[1].dest, Dest::Keep);
        assert!(info.headline.contains("1280x720"), "{}", info.headline);
    }

    #[test]
    fn refuses_loopback_at_the_resolver() {
        // The resolver rejects before any connect; no packet leaves.
        let cfg = crate::config::default_config(&crate::config::test_dirs(Path::new("/home/u")));
        let err = agent(&cfg, &CancelSignal::new())
            .get("http://127.0.0.1:9/x")
            .call()
            .expect_err("loopback must be refused");

        assert!(err.to_string().contains("loopback"), "{err}");
    }

    #[test]
    fn reservation_skips_taken_names() {
        let dir = crate::testutil::TempDir::new("av-direct");
        let p = dir.join("v.mp4");
        std::fs::write(&p, b"x").expect("write");
        let next = reserve_available_path(&p).expect("free name");
        assert_eq!(next, dir.join("v (1).mp4"));
    }

    #[test]
    fn reservation_claims_the_name_it_returns() {
        // The atomicity contract, observed: once a caller holds a reserved
        // path, a second probe of the same name cannot win it — the old
        // exists()-based probe handed both callers "v.mp4".
        let dir = crate::testutil::TempDir::new("av-direct-claim");
        let p = dir.join("v.mp4");
        let first = reserve_available_path(&p).expect("free name");
        assert_eq!(first, p);
        assert!(first.exists(), "the returned name is claimed on disk");
        let second = reserve_available_path(&p).expect("free name");
        assert_eq!(second, dir.join("v (1).mp4"));
    }

    /// Endless data source that trips the shared cancel signal after a set
    /// number of reads — the tracker abandoning the slot mid-stream.
    struct TrippingReader {
        cancel: CancelSignal,
        reads: u32,
        trip_after: u32,
    }

    impl Read for TrippingReader {
        fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
            self.reads += 1;
            if self.reads > self.trip_after {
                self.cancel.cancel();
            }
            buf[0] = b'x';
            Ok(1)
        }
    }

    #[test]
    fn cancellation_stops_the_stream_between_reads() {
        let dir = crate::testutil::TempDir::new("av-direct-stop");
        let cancel = CancelSignal::new();
        let mut reader = TrippingReader {
            cancel: cancel.clone(),
            reads: 0,
            trip_after: 3,
        };
        let part = dir.join("v.mp4.part");

        let err = stream_to_part(&mut reader, &part, u64::MAX, None, &cancel, &mut |_, _| {})
            .expect_err("a cancelled stream must not complete");

        assert!(err.message.contains("cancelled"), "{err}");
        assert_eq!(
            reader.reads, 4,
            "the check after the tripping read must be the last"
        );
    }

    /// Peer that never sends a byte: every wait sleeps out its slice and
    /// reports a timeout, like a socket read hitting `SO_RCVTIMEO`.
    #[derive(Debug)]
    struct SilentPeer {
        buffers: ureq::unversioned::transport::LazyBuffers,
    }

    impl SilentPeer {
        fn new() -> Self {
            SilentPeer {
                buffers: ureq::unversioned::transport::LazyBuffers::new(1024, 1024),
            }
        }
    }

    impl Transport for SilentPeer {
        fn buffers(&mut self) -> &mut dyn Buffers {
            &mut self.buffers
        }

        fn transmit_output(&mut self, _: usize, _: NextTimeout) -> Result<(), ureq::Error> {
            Ok(())
        }

        fn await_input(&mut self, timeout: NextTimeout) -> Result<bool, ureq::Error> {
            std::thread::sleep(*timeout.after);
            Err(ureq::Error::Timeout(timeout.reason))
        }

        fn is_open(&mut self) -> bool {
            true
        }
    }

    /// A caller deadline that never fires, isolating the guard's own bounds.
    fn no_deadline() -> NextTimeout {
        NextTimeout {
            after: UreqDuration::NotHappening,
            reason: ureq::Timeout::RecvBody,
        }
    }

    #[test]
    fn guard_unblocks_a_silent_peer_at_the_idle_limit() {
        let mut guard = StallGuard::new(
            SilentPeer::new(),
            CancelSignal::new(),
            Duration::from_millis(5),
            Duration::from_millis(40),
        );
        let started = Instant::now();
        let err = guard
            .await_input(no_deadline())
            .expect_err("a silent peer must not pin the wait");
        assert!(
            started.elapsed() >= Duration::from_millis(40),
            "the idle budget must be spent first"
        );
        assert!(
            started.elapsed() < Duration::from_secs(5),
            "the wait must end at the idle limit, not the caller deadline"
        );
        assert!(
            err.to_string().contains("presuming the transfer dead"),
            "{err}"
        );
    }

    #[test]
    fn guard_refuses_to_wait_for_a_cancelled_job() {
        let cancel = CancelSignal::new();
        cancel.cancel();
        let mut guard = StallGuard::new(
            SilentPeer::new(),
            cancel,
            Duration::from_millis(5),
            Duration::from_secs(3600),
        );
        let err = guard
            .await_input(no_deadline())
            .expect_err("a disowned job must not wait at all");
        assert!(err.to_string().contains("cancelled"), "{err}");
    }

    #[test]
    fn guard_observes_cancellation_within_a_tick() {
        let cancel = CancelSignal::new();
        let tripper = cancel.clone();
        let mut guard = StallGuard::new(
            SilentPeer::new(),
            cancel,
            Duration::from_millis(5),
            Duration::from_secs(3600),
        );
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(20));
            tripper.cancel();
        });
        let started = Instant::now();
        let err = guard
            .await_input(no_deadline())
            .expect_err("cancellation must end the wait");
        assert!(
            started.elapsed() < Duration::from_secs(5),
            "observed within ticks, not at the idle limit"
        );
        assert!(err.to_string().contains("cancelled"), "{err}");
    }

    #[test]
    fn cancelled_keep_download_releases_the_reserved_name() {
        let dir = crate::testutil::TempDir::new("av-direct-cancel");
        let mut cfg = crate::config::default_config(&crate::config::test_dirs(Path::new("/h/u")));
        cfg.network.allow_private_hosts = true;
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind");
        let addr = listener.local_addr().expect("addr");
        std::thread::spawn(move || {
            if let Ok((mut s, _)) = listener.accept() {
                // Swallow the request and never answer; the client must
                // abort on its own.
                let _ = std::io::Read::read(&mut s, &mut [0u8; 1024]);
            }
        });
        let url = crate::netguard::vet(&format!("http://{addr}/v.mp4"), &cfg).expect("vetted");
        let cancel = CancelSignal::new();
        cancel.cancel(); // the tracker disowned the job
        let keep = dir.join("keep");
        let recipe = DirectRecipe {
            ext: "mp4".to_string(),
            filename: "v.mp4".to_string(),
        };

        let err = download(
            &url,
            &recipe,
            DownloadIo::Keep { dir: &keep },
            &cfg,
            &cancel,
            &mut |_, _| {},
        )
        .expect_err("a cancelled download must fail");

        assert!(err.message.contains("cancelled"), "{err}");
        let leftovers: Vec<_> = std::fs::read_dir(&keep)
            .expect("keep dir")
            .map(|e| e.expect("entry").file_name())
            .collect();
        assert!(
            leftovers.is_empty(),
            "placeholder and .part must be gone: {leftovers:?}"
        );
        let again = reserve_available_path(&keep.join("v.mp4")).expect("free name");
        assert_eq!(again, keep.join("v.mp4"), "a retry takes the original name");
    }
}
