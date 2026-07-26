// SPDX-License-Identifier: Apache-2.0
//! Canonical source identity.
//!
//! The one definition of "same source" that every keyed surface derives
//! from: the consent-token registry keys offer sets on [`SourceId`] values
//! and the disk cache hashes [`SourceId::cache_key`], so the two sides
//! cannot disagree about which spellings name one source — there is no
//! second normalization to drift from this one.
//!
//! Normalization is strictly semantics-preserving, each step licensed by
//! RFC 3986, never by convenience:
//!
//! - Surrounding whitespace is trimmed: no URI production contains
//!   whitespace (§2), so it can only be transport debris around the
//!   spelling, never part of the source.
//! - Scheme and host fold to lowercase (§6.2.2.1, via §3.1 and §3.2.2:
//!   both are defined case-insensitive). Userinfo, path, query, and
//!   fragment stay byte-exact — §6.2.2.1 assumes every other component
//!   case-sensitive unless the scheme says otherwise, and http(s) does not.
//! - Local paths are made absolute and lexically normalized (`.` segments
//!   collapsed); symlinks are deliberately not resolved — that would make
//!   identity depend on filesystem state at key-derivation time.

use std::path::PathBuf;

use sha2::{Digest, Sha256};

use crate::identity::hex;

/// Canonical identity of one media source. [`SourceId::from_spelling`] is
/// the only constructor and the representation is private, so a `SourceId`
/// that skipped normalization cannot exist: everything keyed on one — or
/// on its [`cache_key`](SourceId::cache_key) — inherits the same identity
/// by construction.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SourceId(Repr);

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum Repr {
    /// An http(s) URL, scheme and host folded to lowercase.
    Url(String),
    /// A local path, absolute and lexically normalized.
    Path(PathBuf),
}

impl SourceId {
    /// Derive the canonical identity from a source as the model spelled it.
    pub fn from_spelling(source: &str) -> SourceId {
        let source = source.trim();
        match fold_url(source) {
            Some(url) => SourceId(Repr::Url(url)),
            // `absolute` is lexical (`.` collapsed, no symlink or `..`
            // resolution) and fails only on an empty path, which then keys
            // as its own spelling — a miss, never a collision.
            None => SourceId(Repr::Path(
                std::path::absolute(source).unwrap_or_else(|_| PathBuf::from(source)),
            )),
        }
    }

    /// Stable disk-cache key: sha256 of the canonical spelling, truncated
    /// to 16 bytes hex. Hashing the *canonical* form is what makes every
    /// spelling of one source hit one cache entry; for spellings that were
    /// already canonical the digested bytes are unchanged, so existing
    /// cache entries keep hitting.
    pub fn cache_key(&self) -> String {
        let canonical: &[u8] = match &self.0 {
            Repr::Url(url) => url.as_bytes(),
            Repr::Path(path) => path.as_os_str().as_encoded_bytes(),
        };
        hex(&Sha256::digest(canonical)[..16])
    }
}

/// The RFC 3986 §6.2.2.1 case fold for an http(s) URL; `None` when the
/// source is not one. The fold is scoped precisely:
///
/// - scheme: the text before `"://"` (§3.1: case-insensitive);
/// - host: the authority after the last `@` (§3.2.2: case-insensitive;
///   the userinfo before it is not, §3.2.1), including any `:port`, whose
///   digits the fold cannot alter;
/// - the authority ends at the first `/`, `?`, or `#` (§3.2), where the
///   case-sensitive path, query, and fragment begin — those pass through
///   byte-exact.
fn fold_url(source: &str) -> Option<String> {
    let (scheme, rest) = source.split_once("://")?;
    if !scheme.eq_ignore_ascii_case("http") && !scheme.eq_ignore_ascii_case("https") {
        return None;
    }
    let authority_len = rest.find(['/', '?', '#']).unwrap_or(rest.len());
    let (authority, tail) = rest.split_at(authority_len);
    let host_start = authority.rfind('@').map_or(0, |at| at + 1);
    let (userinfo, host_port) = authority.split_at(host_start);
    Some(format!(
        "{}://{userinfo}{}{tail}",
        scheme.to_ascii_lowercase(),
        host_port.to_ascii_lowercase()
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn folds_scheme_and_host_case_to_one_identity() {
        let canonical = SourceId::from_spelling("https://example.com/v.mp4");
        for spelling in [
            "HTTPS://example.com/v.mp4",
            "https://EXAMPLE.com/v.mp4",
            "https://Example.COM/v.mp4",
        ] {
            assert_eq!(
                canonical,
                SourceId::from_spelling(spelling),
                "{spelling:?} names the same source"
            );
        }
    }

    #[test]
    fn keeps_path_query_and_fragment_byte_exact() {
        let lower = SourceId::from_spelling("https://example.com/v.mp4?t=1#a");
        for spelling in [
            "https://example.com/V.mp4?t=1#a",
            "https://example.com/v.mp4?T=1#a",
            "https://example.com/v.mp4?t=1#A",
        ] {
            assert_ne!(
                lower,
                SourceId::from_spelling(spelling),
                "case is significant past the authority: {spelling:?}"
            );
        }
    }

    #[test]
    fn keeps_userinfo_byte_exact() {
        assert_ne!(
            SourceId::from_spelling("https://User@example.com/v"),
            SourceId::from_spelling("https://user@example.com/v"),
            "userinfo is case-sensitive (RFC 3986 §3.2.1)"
        );
        assert_eq!(
            SourceId::from_spelling("https://User@Example.com/v"),
            SourceId::from_spelling("https://User@example.com/v"),
            "the host after the last @ still folds"
        );
    }

    #[test]
    fn folds_host_case_when_a_port_is_present() {
        assert_eq!(
            SourceId::from_spelling("https://Example.com:8443/v"),
            SourceId::from_spelling("https://example.com:8443/v"),
        );
    }

    #[test]
    fn trims_surrounding_whitespace() {
        assert_eq!(
            SourceId::from_spelling(" https://example.com/v.mp4 "),
            SourceId::from_spelling("https://example.com/v.mp4"),
        );
    }

    #[test]
    fn converges_lexical_path_spellings() {
        let canonical = SourceId::from_spelling("/v.mp4");
        for spelling in ["/v.mp4 ", " /v.mp4", "/./v.mp4"] {
            assert_eq!(
                canonical,
                SourceId::from_spelling(spelling),
                "{spelling:?} names the same file"
            );
        }
    }

    #[test]
    fn keeps_local_path_case_significant() {
        assert_ne!(
            SourceId::from_spelling("/v.mp4"),
            SourceId::from_spelling("/V.mp4"),
            "path identity is byte-exact"
        );
    }

    #[test]
    fn cache_key_of_a_canonical_url_matches_the_verbatim_hash() {
        // Compatibility pin: the cache used to hash the verbatim spelling,
        // so a spelling that is already canonical must key exactly as
        // before — existing shared-cache entries stay valid.
        let url = "https://example.com/v.mp4";
        let verbatim = hex(&Sha256::digest(url.as_bytes())[..16]);
        assert_eq!(SourceId::from_spelling(url).cache_key(), verbatim);
    }

    #[test]
    fn urls_and_paths_never_share_an_identity() {
        assert_ne!(
            SourceId::from_spelling("https://example.com/v.mp4"),
            SourceId::from_spelling("/example.com/v.mp4"),
        );
    }
}
