// SPDX-License-Identifier: Apache-2.0
//! URL sources: turn URLs into download offers and execute the chosen recipe.
//!
//! Two sources exist — direct http(s) media links and yt-dlp site pages —
//! walked in that order; the first probe claim wins. Policy owns tokens and
//! caps, the server owns paths and cache; a source only knows how to probe
//! and how to fetch exactly what its recipe describes.
//!
//! Every entry point that touches the network takes a
//! [`VettedUrl`]: the egress policy has run
//! before any byte leaves, by construction.

pub mod direct;
pub mod ytdlp;

use std::path::{Path, PathBuf};

use super::cache::url_key;
use super::config::Config;
use super::fault::{Fault, FaultKind};
use super::netguard::VettedUrl;
use super::plan::format_bytes;
use super::relay::PolicyRelay;
use super::tokens::{Dest, DownloadPlan, Token};

/// Version of the download recipe logic. Bump when the bytes a download
/// produces change for reasons outside the config — e.g. the yt-dlp format
/// selector template in [`ytdlp`] — so old cache entries miss instead of
/// impersonating the new recipe's output.
const RECIPE_VERSION: u32 = 1;

/// Cache identity of one analysis rip: the URL plus every knob that shapes
/// the downloaded bytes (`download.analysis_max_height`, the recipe logic
/// version, whether cookies are in play — a member-only stream and the
/// anonymous fallback are different bytes). Same URL, different recipe ⇒
/// different cache entry, so a config change can never serve stale bytes
/// as a hit.
///
/// Not the in-flight job key: the tracker dedupes by [`url_key`] alone —
/// "a download for this URL is running" is URL identity, not byte identity.
pub fn rip_key(url: &str, cfg: &Config) -> String {
    let auth = match crate::cookies::mode(&cfg.cookies) {
        crate::cookies::CookieMode::Anonymous => "",
        _ => "-auth",
    };
    format!(
        "{}-h{}r{RECIPE_VERSION}{auth}",
        url_key(url),
        cfg.download.analysis_max_height
    )
}

/// A source-typed download recipe; the variant names the source that issued
/// (and will execute) the plan, so a plan can never be handed to the wrong
/// downloader.
#[derive(Debug, Clone, PartialEq)]
pub enum Recipe {
    Direct(direct::DirectRecipe),
    Ytdlp(ytdlp::YtdlpRecipe),
}

/// One download candidate offered to the model behind a consent token.
#[derive(Debug, Clone)]
pub struct DownloadOption {
    pub token: Token,
    pub dest: Dest,
    /// Short human label, used when the option is withheld for size
    /// (`"720p (1.2 GiB)"`).
    pub label: String,
    /// Sentence for the offered-token list.
    pub effect: String,
    /// Best-effort size; `None` = unknown, never withheld.
    pub estimated_bytes: Option<u64>,
    pub recipe: Recipe,
}

/// Probe outcome of one source: summary line plus download candidates.
#[derive(Debug, Clone)]
pub struct SourceInfo {
    pub headline: String,
    pub options: Vec<DownloadOption>,
}

/// Byte-level progress callback; total is set when the server reveals it.
pub type ProgressFn<'a> = &'a mut (dyn FnMut(u64, Option<u64>) + Send);

/// Cooperative cancellation for one in-flight download. Clones share a
/// flag whose one writer is the owner of the job's tracker slot: tripping
/// it tells the download thread its work has been disowned. The direct
/// client checks it between reads and inside its stall guard; yt-dlp does
/// not need it — its watchdog kills the child on its own stall bound.
#[derive(Debug, Clone, Default)]
pub struct CancelSignal(std::sync::Arc<std::sync::atomic::AtomicBool>);

impl CancelSignal {
    /// A fresh, untripped signal.
    pub fn new() -> Self {
        Self::default()
    }

    /// Trip the signal; every clone observes it. Irreversible.
    pub fn cancel(&self) {
        self.0.store(true, std::sync::atomic::Ordering::Release);
    }

    /// Whether the job this signal belongs to has been disowned.
    pub fn is_cancelled(&self) -> bool {
        self.0.load(std::sync::atomic::Ordering::Acquire)
    }
}

/// A completed download: the final media path plus an optional degradation
/// note. The note reports best-effort side work that failed without
/// invalidating the media — e.g. subtitles that could not be fetched or
/// converted — so the caller can inform the model instead of failing.
#[derive(Debug, Clone)]
pub struct Downloaded {
    pub path: PathBuf,
    pub warning: Option<String>,
}

/// Walk the source chain: a direct media link claims the URL first,
/// anything it cannot open goes to yt-dlp.
///
/// A direct-probe failure is never fully classified (libav cannot tell an
/// HTML page from a flaky network) and never discarded: it rides along, and
/// if yt-dlp fails too the caller sees both reasons — a transient network
/// error on a real media link must not be misdiagnosed as "install yt-dlp".
/// The combined kind is the join of both faults (`combine_kinds`).
pub fn probe_url(url: &VettedUrl, cfg: &Config, relay: &PolicyRelay) -> Result<SourceInfo, Fault> {
    let direct_failure = match direct::probe(url, cfg, relay) {
        Ok(info) => return Ok(info),
        Err(e) => e,
    };
    ytdlp::probe(url, cfg, relay).map_err(|yt| {
        Fault::new(
            combine_kinds(yt.kind, direct_failure.kind),
            format!("{yt}\nProbing it as a direct media link also failed: {direct_failure}"),
        )
    })
}

/// When both sources fail, pick the class of the cheapest fruitful next
/// step: if either failure is plausibly transient, a retry may open either
/// path, so the pair is transient; otherwise if either side names a missing
/// capability, installing it is the actionable move; otherwise the URL
/// itself is the problem.
fn combine_kinds(ytdlp: FaultKind, direct: FaultKind) -> FaultKind {
    use FaultKind::*;
    match (ytdlp, direct) {
        (Transient, _) | (_, Transient) => Transient,
        (Environment, _) | (_, Environment) => Environment,
        (Internal, _) | (_, Internal) => Internal,
        (BadInput, BadInput) => BadInput,
    }
}

/// Filesystem surface handed to a download: cache entries are keyed, keep
/// files are free-form.
#[derive(Debug, Clone, Copy)]
pub enum DownloadIo<'a> {
    Cache { dir: &'a Path, key: &'a str },
    Keep { dir: &'a Path },
}

/// Execute a redeemed download plan; dispatches on the recipe's source.
/// Returns the final media path plus any degradation note.
///
/// `url` is the caller's fresh vetting of `plan.url` — consent may be
/// redeemed long after the probe that built the plan, so the egress check
/// is repeated at execution time rather than trusted from then.
pub fn download(
    url: &VettedUrl,
    plan: &DownloadPlan,
    io: DownloadIo<'_>,
    cfg: &Config,
    relay: &PolicyRelay,
    cancel: &CancelSignal,
    progress: ProgressFn<'_>,
) -> Result<Downloaded, Fault> {
    match &plan.recipe {
        // The direct client owns its own per-connection screen via
        // `GuardedResolver`, so it needs no relay; only the subprocess path
        // does.
        Recipe::Direct(recipe) => direct::download(url, recipe, io, cfg, cancel, progress),
        // yt-dlp takes no cancel signal: its watchdog kills a stalled child
        // within `ytdlp::NO_PROGRESS_LIMIT`, which a compile-time assertion
        // in `mcp::download` keeps inside the tracker's abandon limit.
        Recipe::Ytdlp(recipe) => ytdlp::download(url, recipe, io, cfg, relay, progress),
    }
}

/// `"~1.2 GiB"` or `"size unknown"`, shared by offer effects.
pub fn estimate_label(bytes: Option<u64>) -> String {
    match bytes {
        Some(n) => format!("~{}", format_bytes(n)),
        None => "size unknown".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{default_config, test_dirs};
    use std::path::Path;

    #[test]
    fn rip_key_changes_with_analysis_height() {
        let dirs = test_dirs(Path::new("/home/u"));
        let mut cfg = default_config(&dirs);
        let url = "https://example.com/v";
        let before = rip_key(url, &cfg);
        assert_eq!(before, rip_key(url, &cfg), "stable for the same config");

        cfg.download.analysis_max_height =
            std::num::NonZeroU32::new(1080).expect("non-zero literal");
        assert_ne!(
            before,
            rip_key(url, &cfg),
            "a different recipe must be a different cache entry"
        );
    }

    #[test]
    fn rip_key_changes_when_cookies_are_in_play() {
        let dirs = test_dirs(Path::new("/home/u"));
        let mut cfg = default_config(&dirs);
        let url = "https://example.com/v";
        let anonymous = rip_key(url, &cfg);

        cfg.cookies.from_browser = Some("firefox".to_string());
        assert_ne!(
            anonymous,
            rip_key(url, &cfg),
            "an authenticated rip must never answer for an anonymous one"
        );
    }
}
