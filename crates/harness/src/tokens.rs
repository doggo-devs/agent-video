// SPDX-License-Identifier: Apache-2.0
//! Consent token registry, in-memory for one MCP stdio session.
//!
//! The registry is the only token mint: policy emits [`OfferDraft`]s —
//! which carry no token text — and [`Tokens::commit`] assigns each one a
//! session-unique token from a monotonic counter, so two distinct plans
//! can never alias — uniqueness is a registry fact, not a property of any
//! text policy could construct. [`Token::new`] wraps the text the model
//! echoes back, and wrapping is not minting: a constructed token is a
//! lookup claim that redeems nothing unless a commit minted its text.
//!
//! A token only authorizes the plan and source it was issued for. Redemption
//! is idempotent: an offer stays redeemable until a newer plan for the same
//! source replaces it, so a failed or repeated attempt never strands the
//! model with a stale token. Idempotency has two shapes: naturally idempotent
//! actions (core runs, cache downloads that serve the existing rip) stay
//! [`Redemption::Fresh`] forever, while a completed keep download — the one
//! action whose repetition duplicates a user-owned file — is recorded via
//! [`Tokens::fulfill`] so later echoes replay the receipt instead.
//!
//! Offer sets are keyed by canonical source identity
//! ([`crate::cache::source::SourceId`] — the same identity the disk cache
//! keys on), not by the model's spelling, and the registry is bounded: past
//! `MAX_OFFER_SETS` live sources the stalest set is evicted, its tokens
//! answering as stale — a state the model is already told how to leave.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use super::cache::source::SourceId;
use super::fetch::Recipe;
use super::plan::CoreInvocation;

/// A consent token, echoed verbatim by the model to authorize a plan.
/// Opaque: [`Tokens::commit`] mints it, the registry keys on it, nothing
/// parses it.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Token(String);

impl Token {
    /// Wrap echoed text as a lookup claim. This constructs the *key*, not
    /// the authority: a `Token` grants nothing until [`Tokens::take`]
    /// matches it against a minted entry, and policy code cannot smuggle
    /// one into an offer — [`OfferDraft`] carries no token text.
    pub fn new(text: impl Into<String>) -> Self {
        Token(text.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for Token {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// One consent offer before its token exists: the model-facing effect line
/// and the action it authorizes. Policy emits drafts; only
/// [`Tokens::commit`] can turn one into a redeemable [`Offer`], so a
/// decision layer cannot construct (let alone alias) token text.
#[derive(Debug, Clone, PartialEq)]
pub struct OfferDraft {
    pub effect: String,
    pub action: PendingAction,
}

/// One committed consent offer: the minted token and its model-facing
/// effect line, returned by [`Tokens::commit`] for rendering. The wire
/// rendering and the registry entry both derive from the same commit —
/// they cannot drift apart.
#[derive(Debug, Clone, PartialEq)]
pub struct Offer {
    pub token: Token,
    pub effect: String,
}

/// Where a download lands: `cache` = disposable analysis copy under the LRU
/// cache; `keep` = user-owned file in the keep directory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dest {
    Cache,
    Keep,
}

/// A self-contained download decision, no re-probing needed at exec time.
#[derive(Debug, Clone, PartialEq)]
pub struct DownloadPlan {
    pub url: String,
    pub dest: Dest,
    /// Source-typed recipe describing exactly what to fetch; the variant
    /// names the source that issued (and will execute) this plan.
    pub recipe: Recipe,
    /// Best-effort size estimate; `None` = unknown before download.
    pub estimated_bytes: Option<u64>,
    /// The media title from the probe that issued this plan, when the
    /// source reports one; recorded by content once the file lands so
    /// transcription can derive its whisper prompt from it.
    pub title: Option<String>,
}

/// The action authorized once a token is redeemed.
#[derive(Debug, Clone, PartialEq)]
pub enum PendingAction {
    Core(CoreInvocation),
    Download(DownloadPlan),
}

/// What redeeming a token yields.
#[derive(Debug, Clone, PartialEq)]
pub enum Redemption {
    /// First redemption, or a retry after a failed execution: run the action.
    Fresh(PendingAction),
    /// The action already completed with a durable, non-idempotent effect
    /// (a keep download); serve the recorded path instead of repeating it.
    Fulfilled(PathBuf),
}

/// One token's registry state. A redeemed-and-completed non-idempotent
/// action turns into a receipt rather than a hole, so the module's
/// idempotency promise holds for keep downloads too.
#[derive(Debug, Clone, PartialEq)]
enum Slot {
    Offered(PendingAction),
    Fulfilled(PathBuf),
}

/// Ceiling on concurrently live offer sets. A session's live consent
/// conversation is a handful of sources; the cap only bites in long
/// sessions that touch many, where the stalest set is evicted instead of
/// the registry growing for the session's whole lifetime.
const MAX_OFFER_SETS: usize = 64;

/// One source's live offers plus the sequence number of the commit that
/// inserted them — the eviction order at capacity.
#[derive(Debug)]
struct OfferSet {
    stamp: u64,
    slots: HashMap<Token, Slot>,
}

/// Pending offers keyed by canonical source identity ([`SourceId`]),
/// behind the one lock that guards them; guards never escape this type.
/// One live offer set per source, at most `MAX_OFFER_SETS` sources.
///
/// The contract — one live offer set per *source* — is about the source,
/// not the model's spelling of it: `/v.mp4`, `/./v.mp4`, a trailing-space
/// echo, or a re-cased URL host must be one conversation, or a superseded
/// plan would stay redeemable under an alias. The registry derives the
/// [`SourceId`] itself — the same type whose hash keys the disk cache —
/// so no caller can mis-key it and the two sides cannot diverge.
///
/// A cheaply-cloneable shared handle: clones share one registry (and one
/// mint counter). Owning a `Tokens` pins nothing but the registry itself —
/// deliberately, so an effect queued into the download tracker (the keep
/// receipt) can capture a clone without owning the deps graph that owns
/// the tracker, which would close a strong reference cycle and leak both.
#[derive(Debug, Clone, Default)]
pub struct Tokens(Arc<Mutex<Registry>>);

/// Everything the registry knows, under its one lock. The counters live
/// inside the critical section with the map they describe, so a token's
/// uniqueness and an offer set's eviction stamp are both facts of the same
/// serialized insertion event — two racing commits (`spawn_blocking`
/// threads are genuinely parallel) cannot mint equal or inverted stamps.
#[derive(Debug, Default)]
struct Registry {
    pending: HashMap<SourceId, OfferSet>,
    /// Session-local mint counter; every commit draws fresh values, so no
    /// two offers of this registry's lifetime ever share a token.
    minted: u64,
    /// Commit sequence: each insertion stamps itself with the next value.
    /// Strictly monotonic under the lock — even for a commit that mints no
    /// tokens — so "the stalest offer set" is a total order and
    /// `min_by_key` never has a tie to break.
    commits: u64,
}

impl Tokens {
    /// Commit the offers of one decision for `source`, minting each draft a
    /// session-unique token and wholesale replacing any prior offer set
    /// (including fulfilled receipts: a newer plan supersedes the whole
    /// conversation about this source). At capacity the stalest *other*
    /// source's set is evicted first — the fresh commit always survives.
    /// Returns the completed offers for rendering — the model sees exactly
    /// what became redeemable.
    pub fn commit(&self, source: &str, drafts: Vec<OfferDraft>) -> Vec<Offer> {
        let key = SourceId::from_spelling(source);
        let mut reg = self.lock();
        let mut slots = HashMap::with_capacity(drafts.len());
        let offers: Vec<Offer> = drafts
            .into_iter()
            .map(|draft| {
                reg.minted += 1;
                let token = Token(format!("consent-{}", reg.minted));
                slots.insert(token.clone(), Slot::Offered(draft.action));
                Offer {
                    token,
                    effect: draft.effect,
                }
            })
            .collect();
        if !reg.pending.contains_key(&key) && reg.pending.len() >= MAX_OFFER_SETS {
            let stalest = reg
                .pending
                .iter()
                .min_by_key(|(_, set)| set.stamp)
                .map(|(k, _)| k.clone())
                .expect("registry is at capacity, so it is non-empty");
            reg.pending.remove(&stalest);
        }
        reg.commits += 1;
        let stamp = reg.commits;
        reg.pending.insert(key, OfferSet { stamp, slots });
        offers
    }

    /// Redeem: return the authorized action or the receipt of its completed
    /// run, else `None`. The offer set survives — only a newer `commit` for
    /// the same source invalidates it — so retries after a failed execution
    /// redeem the same token again.
    pub fn take(&self, source: &str, token: &Token) -> Option<Redemption> {
        Some(
            match self
                .lock()
                .pending
                .get(&SourceId::from_spelling(source))?
                .slots
                .get(token)?
            {
                Slot::Offered(action) => Redemption::Fresh(action.clone()),
                Slot::Fulfilled(path) => Redemption::Fulfilled(path.clone()),
            },
        )
    }

    /// Record that a redeemed token's action completed with a durable effect
    /// at `path`. Only non-idempotent actions (keep downloads) call this;
    /// idempotent ones stay `Offered` so retries re-run them. A no-op when a
    /// newer `commit` already replaced the offer set — the receipt would be
    /// stale the moment it landed.
    pub fn fulfill(&self, source: &str, token: &Token, path: PathBuf) {
        if let Some(slot) = self
            .lock()
            .pending
            .get_mut(&SourceId::from_spelling(source))
            .and_then(|s| s.slots.get_mut(token))
        {
            *slot = Slot::Fulfilled(path);
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Registry> {
        self.0.lock().expect("token registry poisoned")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::look::preset_for;
    use crate::plan::CoreArgs;

    fn core_action(tag: u32) -> PendingAction {
        // Distinct pending actions told apart by their stretched cadence —
        // a semantically valid process run, not a placeholder flag.
        PendingAction::Core(CoreInvocation {
            args: CoreArgs::Process {
                look: preset_for(None),
                strategy: None,
                chunk_seconds: Some(tag),
                window: None,
            },
            subtitle: None,
        })
    }

    fn draft(tag: u32) -> OfferDraft {
        OfferDraft {
            effect: format!("effect {tag}"),
            action: core_action(tag),
        }
    }

    #[test]
    fn redeemed_token_stays_valid_for_retries() {
        let reg = Tokens::default();
        let offers = reg.commit("src", vec![draft(1)]);
        assert!(reg.take("src", &offers[0].token).is_some());
        assert!(
            reg.take("src", &offers[0].token).is_some(),
            "a failed execution must be retryable with the same token"
        );
    }

    #[test]
    fn sibling_offers_survive_redemption() {
        let reg = Tokens::default();
        let offers = reg.commit("src", vec![draft(1), draft(2)]);
        assert!(reg.take("src", &offers[0].token).is_some());
        assert!(
            reg.take("src", &offers[1].token).is_some(),
            "picking one tier must not strand the others"
        );
    }

    #[test]
    fn new_offer_replaces_prior_offer() {
        let reg = Tokens::default();
        let old = reg.commit("src", vec![draft(1)]);
        reg.commit("src", vec![draft(2)]);
        assert!(
            reg.take("src", &old[0].token).is_none(),
            "stale token rejected"
        );
    }

    #[test]
    fn rejects_foreign_source() {
        let reg = Tokens::default();
        let offers = reg.commit("src-a", vec![draft(1)]);
        assert!(reg.take("src-b", &offers[0].token).is_none());
    }

    #[test]
    fn keys_offers_by_source_identity_not_spelling() {
        let reg = Tokens::default();
        let offers = reg.commit("/v.mp4", vec![draft(1)]);
        for spelling in ["/v.mp4 ", " /v.mp4", "/./v.mp4"] {
            assert!(
                reg.take(spelling, &offers[0].token).is_some(),
                "{spelling:?} names the same file as /v.mp4"
            );
        }
    }

    #[test]
    fn keys_url_offers_by_identity_across_case_spellings() {
        let reg = Tokens::default();
        let offers = reg.commit("https://example.com/v.mp4", vec![draft(1)]);
        for spelling in [
            "HTTPS://example.com/v.mp4",
            "https://EXAMPLE.com/v.mp4",
            "https://example.com/v.mp4 ",
        ] {
            assert!(
                reg.take(spelling, &offers[0].token).is_some(),
                "{spelling:?} names the same source"
            );
        }
    }

    #[test]
    fn rejects_a_url_differing_past_the_authority() {
        let reg = Tokens::default();
        let offers = reg.commit("https://example.com/v.mp4", vec![draft(1)]);
        assert!(
            reg.take("https://example.com/V.mp4", &offers[0].token)
                .is_none(),
            "path case names a different resource"
        );
    }

    #[test]
    fn newer_plan_supersedes_offers_across_spellings() {
        let reg = Tokens::default();
        let old = reg.commit("/v.mp4", vec![draft(1)]);
        reg.commit("/./v.mp4", vec![draft(2)]);
        assert!(
            reg.take("/v.mp4", &old[0].token).is_none(),
            "an alias spelling must not keep a superseded plan redeemable"
        );
    }

    #[test]
    fn evicts_the_stalest_offer_set_at_capacity() {
        let reg = Tokens::default();
        let first = reg.commit("/v0.mp4", vec![draft(0)]);
        let mut last = Vec::new();
        for i in 1..=MAX_OFFER_SETS {
            last = reg.commit(&format!("/v{i}.mp4"), vec![draft(i as u32)]);
        }
        assert!(
            reg.take("/v0.mp4", &first[0].token).is_none(),
            "the stalest set leaves at capacity"
        );
        assert!(
            reg.take(&format!("/v{MAX_OFFER_SETS}.mp4"), &last[0].token)
                .is_some(),
            "the fresh commit always survives its own eviction"
        );
    }

    #[test]
    fn evicts_in_commit_order_when_a_commit_mints_no_tokens() {
        // The eviction stamp is the commit's own insertion event, not a
        // token count: a commit with no drafts (superseding a source with
        // nothing) still ranks as fresher than every commit before it.
        // Under a mint-counter stamp it would tie with its predecessor and
        // min_by_key would break the tie by hash order.
        let reg = Tokens::default();
        let first = reg.commit("/first.mp4", vec![draft(1)]);
        reg.commit("/second.mp4", Vec::new());
        for i in 2..=MAX_OFFER_SETS {
            reg.commit(&format!("/v{i}.mp4"), vec![draft(i as u32)]);
        }
        assert!(
            reg.take("/first.mp4", &first[0].token).is_none(),
            "the first commit is strictly the stalest and leaves first"
        );
    }

    #[test]
    fn fulfilled_token_replays_receipt_instead_of_action() {
        let reg = Tokens::default();
        let offers = reg.commit("src", vec![draft(1)]);
        let token = &offers[0].token;
        assert!(matches!(reg.take("src", token), Some(Redemption::Fresh(_))));
        reg.fulfill("src", token, PathBuf::from("/keep/video.mp4"));
        assert_eq!(
            reg.take("src", token),
            Some(Redemption::Fulfilled(PathBuf::from("/keep/video.mp4"))),
            "an echo after completion must serve the receipt, not re-run"
        );
    }

    #[test]
    fn new_offer_replaces_fulfilled_receipt() {
        let reg = Tokens::default();
        let offers = reg.commit("src", vec![draft(1)]);
        reg.fulfill("src", &offers[0].token, PathBuf::from("/keep/video.mp4"));
        reg.commit("src", vec![draft(2)]);
        assert!(
            reg.take("src", &offers[0].token).is_none(),
            "a newer plan supersedes receipts along with offers"
        );
    }

    #[test]
    fn fulfill_after_replacement_is_noop() {
        let reg = Tokens::default();
        let old = reg.commit("src", vec![draft(1)]);
        reg.commit("src", vec![draft(2)]);
        reg.fulfill("src", &old[0].token, PathBuf::from("/keep/video.mp4"));
        assert!(
            reg.take("src", &old[0].token).is_none(),
            "a stale token's completion must not resurrect it"
        );
    }

    #[test]
    fn never_aliases_tokens_across_plans() {
        // Two decisions authorizing the same action for the same source
        // still mint distinct tokens — even through a clone, which shares
        // the one counter — so a superseded token dies instead of silently
        // rebinding to the newer plan.
        let reg = Tokens::default();
        let handle = reg.clone();
        let a = reg.commit("src", vec![draft(1)]);
        let b = handle.commit("src", vec![draft(1)]);
        assert_ne!(a[0].token, b[0].token, "distinct plans never share a token");
        assert!(
            reg.take("src", &a[0].token).is_none(),
            "the superseded token is dead, not rebound"
        );
        assert!(reg.take("src", &b[0].token).is_some());
    }
}
