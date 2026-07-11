// SPDX-License-Identifier: Apache-2.0
//! Consent token registry, in-memory for one MCP stdio session.
//!
//! A token only authorizes the plan and source it was issued for. Redemption
//! is idempotent: an offer stays redeemable until a newer plan for the same
//! source replaces it, so a failed or repeated attempt never strands the
//! model with a stale token. Idempotency has two shapes: naturally idempotent
//! actions (core runs, cache downloads that serve the existing rip) stay
//! [`Redemption::Fresh`] forever, while a completed keep download — the one
//! action whose repetition duplicates a user-owned file — is recorded via
//! [`Tokens::fulfill`] so later echoes replay the receipt instead.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Mutex;

use super::fetch::Recipe;
use super::plan::CoreInvocation;

/// A consent token, echoed verbatim by the model to authorize a plan.
/// Opaque: policy mints it, the registry keys on it, nothing parses it.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Token(String);

impl Token {
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

/// One consent offer: the token, its model-facing effect line, and the
/// action it authorizes. The wire rendering and the registry entry both
/// derive from this — they cannot drift apart.
#[derive(Debug, Clone, PartialEq)]
pub struct Offer {
    pub token: Token,
    pub effect: String,
    pub action: PendingAction,
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

/// Pending offers keyed by source, behind the one lock that guards them;
/// guards never escape this type. One live offer set per source.
#[derive(Debug, Default)]
pub struct Tokens {
    pending: Mutex<HashMap<String, HashMap<Token, Slot>>>,
}

impl Tokens {
    /// Register the offers of one decision under `key`, wholesale replacing
    /// any prior offer set (including fulfilled receipts: a newer plan
    /// supersedes the whole conversation about this source).
    pub fn commit(&self, key: &str, offers: Vec<Offer>) {
        let entry = offers
            .into_iter()
            .map(|o| (o.token, Slot::Offered(o.action)))
            .collect();
        self.lock().insert(key.to_string(), entry);
    }

    /// Redeem: return the authorized action or the receipt of its completed
    /// run, else `None`. The offer set survives — only a newer `commit` for
    /// the same key invalidates it — so retries after a failed execution
    /// redeem the same token again.
    pub fn take(&self, key: &str, token: &Token) -> Option<Redemption> {
        Some(match self.lock().get(key)?.get(token)? {
            Slot::Offered(action) => Redemption::Fresh(action.clone()),
            Slot::Fulfilled(path) => Redemption::Fulfilled(path.clone()),
        })
    }

    /// Record that a redeemed token's action completed with a durable effect
    /// at `path`. Only non-idempotent actions (keep downloads) call this;
    /// idempotent ones stay `Offered` so retries re-run them. A no-op when a
    /// newer `commit` already replaced the offer set — the receipt would be
    /// stale the moment it landed.
    pub fn fulfill(&self, key: &str, token: &Token, path: PathBuf) {
        if let Some(slot) = self.lock().get_mut(key).and_then(|s| s.get_mut(token)) {
            *slot = Slot::Fulfilled(path);
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, HashMap<String, HashMap<Token, Slot>>> {
        self.pending.lock().expect("token registry poisoned")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::raw::RawSubcommand;

    fn core_action(tag: u32) -> PendingAction {
        PendingAction::Core(CoreInvocation {
            subcommand: RawSubcommand::Process,
            flags: vec![format!("--tag{tag}")],
            subtitle: None,
            window: None,
            predicted_grids: tag,
        })
    }

    fn offer(token: &str, tag: u32) -> Offer {
        Offer {
            token: Token::new(token),
            effect: format!("{token} effect"),
            action: core_action(tag),
        }
    }

    #[test]
    fn redeemed_token_stays_valid_for_retries() {
        let reg = Tokens::default();
        reg.commit("src", vec![offer("t1", 1)]);
        assert!(reg.take("src", &Token::new("t1")).is_some());
        assert!(
            reg.take("src", &Token::new("t1")).is_some(),
            "a failed execution must be retryable with the same token"
        );
    }

    #[test]
    fn sibling_offers_survive_redemption() {
        let reg = Tokens::default();
        reg.commit("src", vec![offer("a", 1), offer("b", 2)]);
        assert!(reg.take("src", &Token::new("a")).is_some());
        assert!(
            reg.take("src", &Token::new("b")).is_some(),
            "picking one tier must not strand the others"
        );
    }

    #[test]
    fn new_offer_replaces_prior_offer() {
        let reg = Tokens::default();
        reg.commit("src", vec![offer("old", 1)]);
        reg.commit("src", vec![offer("new", 2)]);
        assert!(
            reg.take("src", &Token::new("old")).is_none(),
            "stale token rejected"
        );
    }

    #[test]
    fn rejects_foreign_source() {
        let reg = Tokens::default();
        reg.commit("src-a", vec![offer("t", 1)]);
        assert!(reg.take("src-b", &Token::new("t")).is_none());
    }

    #[test]
    fn fulfilled_token_replays_receipt_instead_of_action() {
        let reg = Tokens::default();
        reg.commit("src", vec![offer("t", 1)]);
        let token = Token::new("t");
        assert!(matches!(
            reg.take("src", &token),
            Some(Redemption::Fresh(_))
        ));
        reg.fulfill("src", &token, PathBuf::from("/keep/video.mp4"));
        assert_eq!(
            reg.take("src", &token),
            Some(Redemption::Fulfilled(PathBuf::from("/keep/video.mp4"))),
            "an echo after completion must serve the receipt, not re-run"
        );
    }

    #[test]
    fn new_offer_replaces_fulfilled_receipt() {
        let reg = Tokens::default();
        reg.commit("src", vec![offer("t", 1)]);
        reg.fulfill("src", &Token::new("t"), PathBuf::from("/keep/video.mp4"));
        reg.commit("src", vec![offer("t2", 2)]);
        assert!(
            reg.take("src", &Token::new("t")).is_none(),
            "a newer plan supersedes receipts along with offers"
        );
    }

    #[test]
    fn fulfill_after_replacement_is_noop() {
        let reg = Tokens::default();
        reg.commit("src", vec![offer("old", 1)]);
        reg.commit("src", vec![offer("new", 2)]);
        reg.fulfill("src", &Token::new("old"), PathBuf::from("/keep/video.mp4"));
        assert!(
            reg.take("src", &Token::new("old")).is_none(),
            "a stale token's completion must not resurrect it"
        );
    }
}
