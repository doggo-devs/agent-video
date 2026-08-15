// SPDX-License-Identifier: Apache-2.0
//! Injectable capabilities and the shared per-server state.
//!
//! [`ServerDeps`] bundles everything one `video` call needs — config, cache,
//! token registry, download tracker — behind capability traits so tests swap
//! in fakes without touching the call flow.

use std::path::Path;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

use crate::cache::Cache;
use crate::config::Config;
use crate::core_run::{CoreRunner, SelfSpawn};
use crate::fault::Fault;
use crate::fetch::{self, Attempt, CancelSignal, DownloadIo, ProgressFn, SourceInfo};
use crate::gate::KeyedGate;
use crate::mcp::tracker::Tracker;
use crate::probe::probe_local_file;
use crate::relay::PolicyRelay;
use crate::schema::ProbeSummary;
use crate::tokens::{DownloadPlan, Tokens};
use crate::transcribe;

/// Probes local media files; injectable for tests.
pub trait FileProber: Send + Sync {
    fn probe(&self, file: &Path) -> Result<ProbeSummary, Fault>;
}

/// Probes and downloads URLs; injectable for tests.
///
/// Contract: implementations that reach a real network own the egress
/// policy — the production impl vets every URL through [`crate::netguard`]
/// before any request and again before executing a download.
pub trait UrlFetcher: Send + Sync {
    fn probe_url(&self, url: &str, cfg: &Config, relay: &PolicyRelay) -> Result<SourceInfo, Fault>;
    fn download(
        &self,
        plan: &DownloadPlan,
        io: DownloadIo<'_>,
        cfg: &Config,
        relay: &PolicyRelay,
        cancel: &CancelSignal,
        progress: ProgressFn<'_>,
    ) -> Result<Attempt, Fault>;
}

/// Transcription capability; absence degrades to an install hint.
pub trait Whisper: Send + Sync {
    fn available(&self, cfg: &Config) -> bool;
    /// `prompt` is the effective conditioning text
    /// ([`transcribe::effective_prompt`]); the caller keys `out_srt` by
    /// [`transcribe::transcript_id`] over the same text.
    fn transcribe(
        &self,
        cfg: &Config,
        file: &Path,
        out_srt: &Path,
        prompt: &str,
        on_step: &mut dyn FnMut(&str),
    ) -> Result<transcribe::Transcribed, Fault>;
}

struct RealProber;

impl FileProber for RealProber {
    fn probe(&self, file: &Path) -> Result<ProbeSummary, Fault> {
        probe_local_file(file)
    }
}

struct RealFetcher;

impl UrlFetcher for RealFetcher {
    fn probe_url(&self, url: &str, cfg: &Config, relay: &PolicyRelay) -> Result<SourceInfo, Fault> {
        // The rejection carries its own fault class: a bad or forbidden URL
        // is the caller's, a resolution failure may be the network's.
        let vetted = crate::netguard::vet(url, cfg)?;
        fetch::probe_url(&vetted, cfg, relay)
    }

    fn download(
        &self,
        plan: &DownloadPlan,
        io: DownloadIo<'_>,
        cfg: &Config,
        relay: &PolicyRelay,
        cancel: &CancelSignal,
        progress: ProgressFn<'_>,
    ) -> Result<Attempt, Fault> {
        // Re-vet at execution time: consent may be redeemed long after the
        // probe, and DNS may have moved underneath the plan.
        let vetted = crate::netguard::vet(&plan.url, cfg)?;
        fetch::download(&vetted, plan, io, cfg, relay, cancel, progress)
    }
}

struct RealWhisper;

impl Whisper for RealWhisper {
    fn available(&self, cfg: &Config) -> bool {
        transcribe::available(cfg)
    }

    fn transcribe(
        &self,
        cfg: &Config,
        file: &Path,
        out_srt: &Path,
        prompt: &str,
        on_step: &mut dyn FnMut(&str),
    ) -> Result<transcribe::Transcribed, Fault> {
        transcribe::transcribe(cfg, file, out_srt, prompt, on_step)
    }
}

/// Everything one `video` call needs; shared across calls and with detached
/// download threads.
pub struct ServerDeps {
    pub config: Config,
    pub cache: Cache,
    pub tokens: Tokens,
    pub downloads: Tracker,
    /// Single-flight admission for transcription, keyed by content ×
    /// transcript recipe: the cache check and the whisper run share one
    /// critical section, so concurrent requests for the same transcript
    /// run the model once and the rest adopt the cached result.
    pub transcriptions: KeyedGate,
    /// Per-waiter bound on `transcriptions` admission: how long a rival
    /// call may wait for an in-flight transcription of the same content
    /// before answering "in flight — call again" instead of pinning its
    /// thread for the holder's whole (per-chunk compounding) wall clock.
    /// [`TRANSCRIBE_WAIT`] in production; a field so tests can shrink it.
    pub transcribe_wait: Duration,
    pub prober: Box<dyn FileProber>,
    pub fetcher: Box<dyn UrlFetcher>,
    pub whisper: Box<dyn Whisper>,
    pub core: Box<dyn CoreRunner>,
    /// The egress policy relay, bound lazily on first URL use so a
    /// local-file-only session never opens a listener. See [`ServerDeps::relay`].
    pub(crate) relay: RelaySlot,
}

/// Lazily started [`PolicyRelay`] slot: the first successful start is kept
/// for the life of the deps, but a start failure is returned to the caller
/// and never cached, so transient port/fd pressure at first use cannot
/// poison every later URL request.
pub(crate) struct RelaySlot {
    started: OnceLock<PolicyRelay>,
    /// Serializes start attempts so a lost race cannot bind a second
    /// listener that would be dropped unused.
    starting: Mutex<()>,
}

impl RelaySlot {
    pub(crate) fn new() -> Self {
        RelaySlot {
            started: OnceLock::new(),
            starting: Mutex::new(()),
        }
    }

    /// The started relay, or the outcome of one fresh `start` attempt.
    fn get_or_start(
        &self,
        start: impl FnOnce() -> std::io::Result<PolicyRelay>,
    ) -> std::io::Result<&PolicyRelay> {
        if let Some(relay) = self.started.get() {
            return Ok(relay);
        }
        let _starting = self.starting.lock().expect("relay start lock poisoned");
        if let Some(relay) = self.started.get() {
            return Ok(relay);
        }
        let relay = start()?;
        Ok(self.started.get_or_init(|| relay))
    }
}

/// The production [`ServerDeps::transcribe_wait`]: long enough that a
/// rival to a short transcription still answers inline (the one-blocked-
/// call-over-poll-chains stance of docs/config.md), short enough that a
/// long-running winner cannot pin every rival's thread for hours. Mirrors
/// the download sync grace window's default.
pub const TRANSCRIBE_WAIT: Duration = Duration::from_secs(90);

impl ServerDeps {
    /// Production wiring: disk cache, in-process probe, real sources,
    /// whisper-rs, self-spawned core.
    pub fn production(config: Config) -> Self {
        let cache = Cache::new(&config.paths.cache, config.cache.max_bytes);
        ServerDeps {
            cache,
            tokens: Tokens::default(),
            downloads: Tracker::default(),
            transcriptions: KeyedGate::default(),
            transcribe_wait: TRANSCRIBE_WAIT,
            prober: Box::new(RealProber),
            fetcher: Box::new(RealFetcher),
            whisper: Box::new(RealWhisper),
            core: Box::new(SelfSpawn),
            relay: RelaySlot::new(),
            config,
        }
    }

    /// The policy relay, started on first call and shared thereafter. Every
    /// URL probe and download routes its subprocess/demuxer traffic through
    /// it; local-file sessions never call this, so they never bind a
    /// listener. A bind failure is the server's environment, not the URL's,
    /// and is retried on the next call rather than cached.
    pub fn relay(&self) -> Result<&PolicyRelay, Fault> {
        self.relay
            .get_or_start(|| PolicyRelay::start(self.config.network.allow_private_hosts))
            .map_err(|e| Fault::environment(format!("cannot start the egress policy relay: {e}")))
    }
}

/// Progress sink: `(progress, total, message)`, forwarded to MCP progress
/// notifications when the client asked for them.
pub type Notifier = Arc<dyn Fn(f64, Option<f64>, String) + Send + Sync>;

/// A notifier that drops everything (no progress token, tests).
pub fn null_notifier() -> Notifier {
    Arc::new(|_, _, _| {})
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retries_relay_start_after_a_failed_attempt() {
        let slot = RelaySlot::new();
        let err = match slot.get_or_start(|| Err(std::io::Error::other("port pressure"))) {
            Err(err) => err,
            Ok(_) => panic!("first start must fail"),
        };
        assert_eq!(err.to_string(), "port pressure");
        let url = slot
            .get_or_start(|| PolicyRelay::start(false))
            .expect("second attempt starts fresh")
            .url();
        assert!(url.starts_with("http://127.0.0.1:"), "{url}");
    }

    #[test]
    fn keeps_the_first_started_relay() {
        let slot = RelaySlot::new();
        let first = slot
            .get_or_start(|| PolicyRelay::start(false))
            .expect("first start succeeds")
            .url();
        let second = slot
            .get_or_start(|| panic!("must not start a second relay"))
            .expect("cached relay returned")
            .url();
        assert_eq!(first, second);
    }
}
