// SPDX-License-Identifier: Apache-2.0
//! Injectable capabilities and the shared per-server state.
//!
//! [`ServerDeps`] bundles everything one `video` call needs — config, cache,
//! token registry, download tracker — behind capability traits so tests swap
//! in fakes without touching the call flow.

use std::path::Path;
use std::sync::Arc;

use crate::cache::Cache;
use crate::config::Config;
use crate::fault::Fault;
use crate::fetch::{self, DownloadIo, Downloaded, ProgressFn, SourceInfo};
use crate::mcp::core_run::{CoreRunner, SelfSpawn};
use crate::mcp::tracker::Tracker;
use crate::probe::probe_local_file;
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
    fn probe_url(&self, url: &str, cfg: &Config) -> Result<SourceInfo, Fault>;
    fn download(
        &self,
        plan: &DownloadPlan,
        io: DownloadIo<'_>,
        cfg: &Config,
        progress: ProgressFn<'_>,
    ) -> Result<Downloaded, Fault>;
}

/// Transcription capability; absence degrades to an install hint.
pub trait Whisper: Send + Sync {
    fn available(&self, cfg: &Config) -> bool;
    fn transcribe(
        &self,
        cfg: &Config,
        file: &Path,
        out_srt: &Path,
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
    fn probe_url(&self, url: &str, cfg: &Config) -> Result<SourceInfo, Fault> {
        // A netguard rejection is a verdict on the URL, not on the network.
        let vetted = crate::netguard::vet(url, cfg).map_err(|e| Fault::bad_input(e.to_string()))?;
        fetch::probe_url(&vetted, cfg)
    }

    fn download(
        &self,
        plan: &DownloadPlan,
        io: DownloadIo<'_>,
        cfg: &Config,
        progress: ProgressFn<'_>,
    ) -> Result<Downloaded, Fault> {
        // Re-vet at execution time: consent may be redeemed long after the
        // probe, and DNS may have moved underneath the plan.
        let vetted =
            crate::netguard::vet(&plan.url, cfg).map_err(|e| Fault::bad_input(e.to_string()))?;
        fetch::download(&vetted, plan, io, cfg, progress)
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
        on_step: &mut dyn FnMut(&str),
    ) -> Result<transcribe::Transcribed, Fault> {
        transcribe::transcribe(cfg, file, out_srt, on_step)
    }
}

/// Everything one `video` call needs; shared across calls and with detached
/// download threads.
pub struct ServerDeps {
    pub config: Config,
    pub cache: Cache,
    pub tokens: Tokens,
    pub downloads: Tracker,
    pub prober: Box<dyn FileProber>,
    pub fetcher: Box<dyn UrlFetcher>,
    pub whisper: Box<dyn Whisper>,
    pub core: Box<dyn CoreRunner>,
}

impl ServerDeps {
    /// Production wiring: disk cache, in-process probe, real sources,
    /// whisper-rs, self-spawned core.
    pub fn production(config: Config) -> Self {
        let cache = Cache::new(&config.paths.cache, config.cache.max_bytes);
        ServerDeps {
            cache,
            tokens: Tokens::default(),
            downloads: Tracker::default(),
            prober: Box::new(RealProber),
            fetcher: Box::new(RealFetcher),
            whisper: Box::new(RealWhisper),
            core: Box::new(SelfSpawn),
            config,
        }
    }
}

/// Progress sink: `(progress, total, message)`, forwarded to MCP progress
/// notifications when the client asked for them.
pub type Notifier = Arc<dyn Fn(f64, Option<f64>, String) + Send + Sync>;

/// A notifier that drops everything (no progress token, tests).
pub fn null_notifier() -> Notifier {
    Arc::new(|_, _, _| {})
}
