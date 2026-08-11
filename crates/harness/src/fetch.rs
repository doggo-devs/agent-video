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

use std::num::NonZeroU32;
use std::path::{Path, PathBuf};

use super::cache::source::SourceId;
use super::config::{Config, CookiesCfg, DownloadCfg, NetworkCfg};
use super::fault::{Fault, FaultKind};
use super::netguard::VettedUrl;
use super::plan::format_bytes;
use super::relay::PolicyRelay;
use super::tokens::{Dest, DownloadPlan};

/// Version of the download recipe logic. Bump when the bytes a download
/// produces change for reasons outside the config — e.g. the yt-dlp format
/// selector template in [`ytdlp`] — so old cache entries miss instead of
/// impersonating the new recipe's output. Key-*input* changes need no
/// bump: when [`SourceId`] normalization converges spellings, a canonical
/// key still means exactly the bytes it always meant, and keys that
/// existed only under non-canonical spellings orphan into natural misses
/// the LRU ages out.
const RECIPE_VERSION: u32 = 1;

/// Cache identity of one analysis rip: the canonical source plus every
/// config knob that shapes the downloaded bytes. "Every" is enforced by
/// construction, not by review: [`RipId::derive`] is the only constructor,
/// and it destructures the whole [`Config`] without `..`.
#[derive(Debug, PartialEq, Eq)]
struct RipId {
    source: SourceId,
    /// `download.analysis_max_height`: the frame-height cap both fetchers
    /// select formats under.
    max_height: NonZeroU32,
    /// Whether any cookie source is configured — a member-only stream and
    /// the anonymous fallback are different bytes, so an authenticated rip
    /// must never answer for an anonymous one (or vice versa).
    authenticated: bool,
    /// `network.allow_private_hosts`: under split-horizon DNS the same URL
    /// serves different bytes from the private vantage than from the
    /// public one, so a rip fetched under the permissive policy must never
    /// answer for the strict one (or vice versa) — isomorphic to
    /// `authenticated` (intranet view vs public view ⇔ member view vs
    /// anonymous view).
    private_hosts_allowed: bool,
}

impl RipId {
    /// Classify the whole config into byte-shaping identity inputs and
    /// explicit discards. No `..` in any pattern: a field added to the
    /// config — in any section — fails to compile here until it is either
    /// folded into the identity or bound to a discard with the reason it
    /// cannot shape the downloaded bytes.
    fn derive(source: SourceId, cfg: &Config) -> RipId {
        let Config {
            // Where artifacts land, never which bytes land there.
            paths: _,
            // When cache entries die, not what they contain.
            cache: _,
            // Shapes transcripts, which carry their own recipe identity
            // ([`crate::transcribe::transcript_id`]), not downloads.
            whisper: _,
            download:
                DownloadCfg {
                    // Refusal ceiling: decides whether a download runs,
                    // never the bytes of one that does.
                    max_bytes: _,
                    analysis_max_height,
                    // Response latency, invisible to the fetched bytes.
                    sync_wait_seconds: _,
                },
            // Consent thresholds for grid rendering, downstream of the rip.
            budget: _,
            // Not just "whether a host may be reached": under
            // split-horizon DNS the vantage shapes the bytes served
            // ([`RipId::private_hosts_allowed`]).
            network: NetworkCfg {
                allow_private_hosts,
            },
            cookies:
                cookies @ CookiesCfg {
                    from_browser: _,
                    jar: _,
                },
            // Sidecar election among already-downloaded subtitles; the
            // media bytes — the rip's identity — are untouched, and
            // sidecars are best-effort decoration that may be absent
            // entirely ([`Downloaded::warning`]).
            subtitles: _,
        } = cfg;
        // Both cookie fields feed `mode`: any credential source flips the
        // marker, which is why the whole struct — not one field — is the
        // classified input.
        let authenticated = !matches!(
            crate::cookies::mode(cookies),
            crate::cookies::CookieMode::Anonymous
        );
        RipId {
            source,
            max_height: *analysis_max_height,
            authenticated,
            private_hosts_allowed: *allow_private_hosts,
        }
    }

    /// Render the on-disk key. Destructures `self` exhaustively, so an
    /// identity field cannot exist without entering the key; the spelling
    /// keeps the pre-`RipId` format, so existing cache entries stay valid.
    fn key(&self) -> String {
        let RipId {
            source,
            max_height,
            authenticated,
            private_hosts_allowed,
        } = self;
        let auth = if *authenticated { "-auth" } else { "" };
        let net = if *private_hosts_allowed { "-priv" } else { "" };
        format!(
            "{}-h{max_height}r{RECIPE_VERSION}{auth}{net}",
            source.cache_key()
        )
    }
}

/// Cache key of the analysis rip for `url` under `cfg`: the rendered
/// `RipId`. Same source, different recipe ⇒ different cache entry, so a
/// config change can never serve stale bytes as a hit; different spelling,
/// same source ⇒ the same entry, by [`SourceId`] construction.
///
/// Not the in-flight job key: the tracker dedupes by
/// [`crate::cache::url_key`] alone — "a download for this URL is running"
/// is source identity, not byte identity.
pub fn rip_key(url: &str, cfg: &Config) -> String {
    RipId::derive(SourceId::from_spelling(url), cfg).key()
}

/// A source-typed download recipe; the variant names the source that issued
/// (and will execute) the plan, so a plan can never be handed to the wrong
/// downloader.
#[derive(Debug, Clone, PartialEq)]
pub enum Recipe {
    Direct(direct::DirectRecipe),
    Ytdlp(ytdlp::YtdlpRecipe),
}

/// One download candidate offered to the model behind a consent token
/// (minted by the registry at commit time; a source only describes the
/// candidate, it never spells token text).
#[derive(Debug, Clone)]
pub struct DownloadOption {
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
    /// The media title as the site reports it (yt-dlp sources); `None`
    /// for direct links, whose only "title" would be a filename — noise,
    /// not vocabulary. Rides into every [`DownloadPlan`] so the download
    /// can record it by content ([`crate::cache::Cache::put_title`]).
    pub title: Option<String>,
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

/// Evidence that a fetch delivered less than the source promised.
///
/// The two arms are one fact seen by two observers — the downloader
/// counting bytes against the advertised length, and the demuxer reading
/// the landed container — and they share one cure: run extraction again,
/// so a fresh signed URL (and, on a multi-tier CDN, a different backend)
/// serves the bytes. Resuming cannot help; a resumed range is pinned to
/// the backend that already served short.
///
/// Deliberately not a [`Fault`]: a short rip is the retry loop's input,
/// never the caller's verdict.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Shortfall {
    /// The transfer ended before the advertised length.
    Transfer { got: u64, expected: u64 },
    /// The container's tracks stop before its declared extent, named as
    /// [`crate::probe::coverage_shortfall`] phrases it.
    Coverage(String),
}

impl Shortfall {
    /// One clause naming the evidence, for warnings and faults.
    pub fn describe(&self) -> String {
        match self {
            Shortfall::Transfer { got, expected } => format!(
                "the transfer stopped at {} of the {} the source advertised",
                format_bytes(*got),
                format_bytes(*expected)
            ),
            Shortfall::Coverage(issue) => issue.clone(),
        }
    }
}

/// What one fetch attempt produced. Completeness is the axis, not
/// success: a short rip is neither `Ok` for the caller nor a `Fault`, and
/// splitting it across `Result` is what once made a downloader-observed
/// shortfall cost a consent round-trip while a disk-observed one was
/// retried for free.
#[derive(Debug)]
pub enum Attempt {
    /// Complete media, ready to publish.
    Complete(Downloaded),
    /// The source served short. `landed` carries the incomplete media when
    /// it reached its final path anyway — a truncated container is still a
    /// file — and is `None` when nothing publishable exists.
    Short {
        evidence: Shortfall,
        landed: Option<Downloaded>,
    },
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
) -> Result<Attempt, Fault> {
    let attempt = match &plan.recipe {
        // The direct client owns its own per-connection screen via
        // `GuardedResolver`, so it needs no relay; only the subprocess path
        // does.
        Recipe::Direct(recipe) => direct::download(url, recipe, io, cfg, cancel, progress)?,
        // yt-dlp takes no cancel signal: its watchdog kills a stalled child
        // within `ytdlp::NO_PROGRESS_LIMIT`, which a compile-time assertion
        // in `mcp::download` keeps inside the tracker's abandon limit.
        Recipe::Ytdlp(recipe) => ytdlp::download(url, recipe, io, cfg, relay, progress)?,
    };
    Ok(verify_coverage(attempt))
}

/// The second observer: a rip whose tracks stop before the container's
/// declared extent is short even when every advertised byte arrived — a
/// fragment fetch that gave up mid-stream still muxes a well-formed file.
/// Applied once, above both sources, so every downloader and every retry
/// is judged by the same evidence.
fn verify_coverage(attempt: Attempt) -> Attempt {
    judge_coverage(attempt, crate::probe::coverage_shortfall)
}

/// Which observer to ask is production's business; what its answer *means*
/// is a state transition, and it is spelled here so it can be exercised
/// without a truncated container on disk. An attempt already known to be
/// short is never re-judged: the first evidence is the specific one, and a
/// second opinion could only blur it.
fn judge_coverage(attempt: Attempt, observe: impl Fn(&Path) -> Option<String>) -> Attempt {
    let Attempt::Complete(landed) = attempt else {
        return attempt;
    };
    match observe(&landed.path) {
        Some(issue) => Attempt::Short {
            evidence: Shortfall::Coverage(issue),
            landed: Some(landed),
        },
        None => Attempt::Complete(landed),
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

    fn complete(name: &str) -> Attempt {
        Attempt::Complete(Downloaded {
            path: PathBuf::from(name),
            warning: None,
        })
    }

    #[test]
    fn a_rip_whose_tracks_stop_early_is_short_but_still_served() {
        let judged = judge_coverage(complete("v.mp4"), |_| {
            Some("the audio track covers only 00:31 of the 08:51 timeline".to_string())
        });

        match judged {
            Attempt::Short {
                evidence: Shortfall::Coverage(issue),
                landed,
            } => {
                assert!(issue.contains("00:31"), "{issue}");
                assert_eq!(
                    landed.expect("a truncated container is still a file").path,
                    PathBuf::from("v.mp4"),
                    "the file must ride along, or a spent ladder has nothing to serve"
                );
            }
            other => panic!("expected a coverage shortfall, got {other:?}"),
        }
    }

    #[test]
    fn a_clean_rip_passes_the_coverage_observer_untouched() {
        let judged = judge_coverage(complete("v.mp4"), |_| None);

        assert!(matches!(judged, Attempt::Complete(_)), "{judged:?}");
    }

    #[test]
    fn a_transfer_shortfall_is_never_relabelled_as_a_coverage_one() {
        // The downloader's evidence carries the two byte counts; the
        // container observer's carries a timeline. Re-judging an attempt
        // that is already short would trade the specific for the vague.
        let short = Attempt::Short {
            evidence: Shortfall::Transfer {
                got: 10,
                expected: 100,
            },
            landed: None,
        };

        let judged = judge_coverage(short, |_| panic!("a short attempt must not be re-observed"));

        assert!(
            matches!(
                judged,
                Attempt::Short {
                    evidence: Shortfall::Transfer { .. },
                    ..
                }
            ),
            "{judged:?}"
        );
    }

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
    fn rip_key_converges_spellings_of_one_source() {
        let dirs = test_dirs(Path::new("/home/u"));
        let cfg = default_config(&dirs);
        let canonical = rip_key("https://example.com/v.mp4", &cfg);
        assert_eq!(
            canonical,
            rip_key("HTTPS://Example.COM/v.mp4", &cfg),
            "scheme and host case never split the cache"
        );
        assert_ne!(
            canonical,
            rip_key("https://example.com/V.mp4", &cfg),
            "path case names a different resource"
        );
    }

    #[test]
    fn rip_key_changes_when_private_hosts_are_allowed() {
        let dirs = test_dirs(Path::new("/home/u"));
        let mut cfg = default_config(&dirs);
        let url = "https://example.com/v";
        let strict = rip_key(url, &cfg);

        cfg.network.allow_private_hosts = true;
        assert_ne!(
            strict,
            rip_key(url, &cfg),
            "a permissive-network rip must never answer for a strict one"
        );
    }

    #[test]
    fn default_policy_rip_keys_keep_their_spelling() {
        let dirs = test_dirs(Path::new("/home/u"));
        let cfg = default_config(&dirs);
        let key = rip_key("https://example.com/v", &cfg);
        assert!(
            !key.contains("-priv") && !key.contains("-auth"),
            "default-policy keys keep the pre-field spelling, so existing \
             cache entries stay valid: {key}"
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
