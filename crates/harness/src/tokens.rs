// SPDX-License-Identifier: Apache-2.0
//! Consent token registry, in-memory for one MCP stdio session.
//!
//! A token only authorizes the plan and source it was issued for; redeeming
//! any token consumes the whole offer, so every token is single-use and
//! bound to the latest plan.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Mutex;

use super::fetch::Recipe;
use super::identity::ContentId;
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

/// A transcription decision: the media is local by probe time.
#[derive(Debug, Clone, PartialEq)]
pub struct TranscribePlan {
    pub file: PathBuf,
    /// Content identity keying the cached transcript.
    pub fingerprint: ContentId,
}

/// The action authorized once a token is redeemed.
#[derive(Debug, Clone, PartialEq)]
pub enum PendingAction {
    Core(CoreInvocation),
    Download(DownloadPlan),
    Transcribe(TranscribePlan),
}

/// Pending offers keyed by source, behind the one lock that guards them;
/// guards never escape this type. One live offer set per source.
#[derive(Debug, Default)]
pub struct Tokens {
    pending: Mutex<HashMap<String, HashMap<Token, PendingAction>>>,
}

impl Tokens {
    /// Register the offers of one decision under `key`, wholesale replacing
    /// any prior offer set.
    pub fn commit(&self, key: &str, offers: Vec<Offer>) {
        let entry = offers.into_iter().map(|o| (o.token, o.action)).collect();
        self.lock().insert(key.to_string(), entry);
    }

    /// Register a single-token offer for `key`.
    pub fn offer_one(&self, key: &str, token: Token, action: PendingAction) {
        self.lock()
            .insert(key.to_string(), HashMap::from([(token, action)]));
    }

    /// Redeem: return the action and consume the whole offer, else `None`.
    pub fn take(&self, key: &str, token: &Token) -> Option<PendingAction> {
        let mut pending = self.lock();
        let action = pending.get(key)?.get(token)?.clone();
        pending.remove(key);
        Some(action)
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, HashMap<String, HashMap<Token, PendingAction>>> {
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
    fn redeems_offered_token_once() {
        let reg = Tokens::default();
        reg.offer_one("src", Token::new("t1"), core_action(1));
        assert!(reg.take("src", &Token::new("t1")).is_some());
        assert!(
            reg.take("src", &Token::new("t1")).is_none(),
            "tokens are single-use"
        );
    }

    #[test]
    fn redeeming_consumes_sibling_offers() {
        let reg = Tokens::default();
        reg.commit("src", vec![offer("a", 1), offer("b", 2)]);
        assert!(reg.take("src", &Token::new("a")).is_some());
        assert!(
            reg.take("src", &Token::new("b")).is_none(),
            "whole offer is consumed"
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
}
