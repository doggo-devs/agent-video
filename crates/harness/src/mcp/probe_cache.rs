// SPDX-License-Identifier: Apache-2.0
//! Probe-with-cache: a version-tagged on-disk envelope so a summary written
//! under an older schema — or older field semantics — misses instead of
//! being read as current, plus the content-fingerprint cache lookup that
//! decorates every summary with live-rescanned sidecar files and any
//! cached whisper transcript.

use std::path::Path;

use super::deps::ServerDeps;
use crate::fault::Fault;
use crate::identity::{ContentId, fingerprint_file};
use crate::schema::ProbeSummary;
use crate::transcribe;

/// The effective whisper conditioning text for this content: the
/// operator's `whisper.prompt`, else a continuation shaped from the
/// first-sighted title recorded at download time
/// ([`crate::cache::Cache::put_title`], write-once — stable per content,
/// not a pure function of the bytes), else empty. Content-keyed, so
/// every caller — transcribe, ride-along lookup, probe decoration —
/// derives the same text and therefore the same transcript identity.
pub(super) fn effective_prompt_for(fingerprint: &ContentId, deps: &ServerDeps) -> String {
    let title = deps.cache.get_title(fingerprint);
    transcribe::effective_prompt(&deps.config, title.as_deref())
}

/// The transcript cache identity this content would get today: whisper
/// route × language × effective prompt ([`transcribe::transcript_id`]).
pub(super) fn transcript_id_for(fingerprint: &ContentId, deps: &ServerDeps) -> String {
    transcribe::transcript_id(&deps.config, &effective_prompt_for(fingerprint, deps))
}

/// On-disk probe document: a version-tagged envelope, so a summary written
/// under an older schema — or older field semantics — misses instead of
/// being read as current. Bump the version when [`ProbeSummary`] changes
/// shape or meaning; pre-envelope documents lack the tag and miss too.
#[derive(serde::Serialize, serde::Deserialize)]
struct ProbeDoc {
    v: u32,
    probe: ProbeSummary,
}

// Bumped to 2: v1 documents baked path-dependent sidecar lines into the
// content-keyed summary, so a moved file could advertise a stale sidecar
// for the cache lifetime. v2 documents carry embedded tracks only;
// sidecars are rescanned live on every lookup.
const PROBE_DOC_VERSION: u32 = 2;

fn parse_probe_doc(text: &str) -> Option<ProbeSummary> {
    serde_json::from_str::<ProbeDoc>(text)
        .ok()
        .filter(|doc| doc.v == PROBE_DOC_VERSION)
        .map(|doc| doc.probe)
}

fn render_probe_doc(probe: &ProbeSummary) -> Option<String> {
    serde_json::to_string(&ProbeDoc {
        v: PROBE_DOC_VERSION,
        probe: probe.clone(),
    })
    .ok()
}

/// Probe via content-fingerprint cache; unstable files are probed fresh,
/// never cached. The cached document is a function of content only:
/// sidecar subtitle files — a property of the file's directory — are
/// rescanned live and appended after the cache read, and a cached whisper
/// transcript decorates the summary last (never the cache entry).
pub(super) fn cached_probe(
    file: &Path,
    deps: &ServerDeps,
) -> Result<(ProbeSummary, ContentId), Fault> {
    let fp = fingerprint_file(file).map_err(|e| match e.kind() {
        std::io::ErrorKind::NotFound => {
            Fault::bad_input(format!("file not found: {}", file.display()))
        }
        _ => Fault::bad_input(format!("cannot read {}: {e}", file.display())),
    })?;
    let cached: Option<ProbeSummary> = fp
        .stable
        .then(|| match deps.cache.get_probe(&fp.id) {
            Ok(doc) => doc,
            Err(e) => {
                // Same degradation — and the same alarm — as the write
                // side below: a fresh probe still answers the call, but a
                // broken cache dir must not silently turn every call into
                // a full re-probe.
                tracing::warn!(
                    file = %file.display(),
                    error = %e,
                    "probe cache read failed; probing fresh until it heals"
                );
                None
            }
        })
        .flatten()
        .and_then(|doc| parse_probe_doc(&doc));
    let mut probe = match cached {
        Some(p) => p,
        None => {
            let p = deps.prober.probe(file)?;
            if fp.stable
                && let Some(doc) = render_probe_doc(&p)
                && let Err(e) = deps.cache.put_probe(&fp.id, &doc)
            {
                // Miss semantics stay (a fresh probe answered the call), but
                // a permanently broken cache dir would otherwise degrade to
                // a full re-probe on every call with no trace.
                tracing::warn!(
                    file = %file.display(),
                    error = %e,
                    "probe cache write failed; probing fresh until it heals"
                );
            }
            p
        }
    };
    // Sidecars live next to the file, not inside it: rescan on every call
    // so the content-keyed doc stays deterministic and moves are honest.
    probe.subtitles.extend(
        crate::probe::scan_sidecars(file)
            .into_iter()
            .map(|name| format!("sidecar {name}")),
    );
    if deps
        .cache
        .find_transcript(&fp.id, &transcript_id_for(&fp.id, deps))
        .is_some()
    {
        probe.subtitles.push("transcript (whisper)".to_string());
    }
    Ok((probe, fp.id))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mcp::testutil::{env, media_file, probe};

    #[test]
    fn probe_doc_round_trips_current_version() {
        let summary = probe(90.0, vec!["embedded #0 (eng, subrip)".to_string()]);
        let doc = render_probe_doc(&summary).expect("renders");
        let back = parse_probe_doc(&doc).expect("current version parses");
        assert_eq!(back.duration, 90.0);
        assert_eq!(back.subtitles, ["embedded #0 (eng, subrip)"]);
    }

    #[test]
    fn rejects_probe_docs_from_other_schemas() {
        // A pre-envelope document (bare summary) and a future version must
        // both miss — never be consumed under current semantics.
        let bare = serde_json::to_string(&probe(90.0, vec![])).expect("serialize");
        assert!(parse_probe_doc(&bare).is_none(), "unversioned doc misses");

        let doc = render_probe_doc(&probe(90.0, vec![])).expect("renders");
        let future = doc.replacen(
            &format!("\"v\":{PROBE_DOC_VERSION}"),
            &format!("\"v\":{}", PROBE_DOC_VERSION + 1),
            1,
        );
        assert_ne!(doc, future, "version tag must be present to rewrite");
        assert!(parse_probe_doc(&future).is_none(), "future version misses");
    }

    #[test]
    fn expired_title_yields_a_clean_miss_not_a_stale_hit() {
        let env = env(probe(90.0, vec![]), false);
        let fp = ContentId::new("fp");
        env.deps
            .cache
            .put_title(&fp, "Rust Ownership Explained")
            .expect("record title");
        let conditioned_id = transcript_id_for(&fp, &env.deps);
        let srt = env.deps.cache.transcript_path(&fp, &conditioned_id);
        std::fs::create_dir_all(srt.parent().expect("parent")).expect("mkdir");
        std::fs::write(&srt, b"conditioned cues").expect("write transcript");

        // The sidecar ages out (simulated by removal) while the
        // conditioned transcript is still on disk — the state the
        // hit-side sidecar bump makes unreachable in normal operation.
        let title_path = env
            .deps
            .config
            .paths
            .cache
            .join("probe")
            .join(format!("{fp}.title"));
        std::fs::remove_file(&title_path).expect("expire the sidecar");

        let bare_id = transcript_id_for(&fp, &env.deps);
        assert_ne!(bare_id, conditioned_id, "the derived identity changed");
        assert!(
            env.deps.cache.find_transcript(&fp, &bare_id).is_none(),
            "the lookup misses cleanly (re-transcribe) — the conditioned \
             transcript must never answer under the bare id"
        );
    }

    #[test]
    fn a_later_title_cannot_change_the_derived_prompt() {
        let env = env(probe(90.0, vec![]), false);
        let fp = ContentId::new("fp");
        env.deps
            .cache
            .put_title(&fp, "First Sighted Title")
            .expect("record title");
        let frozen = effective_prompt_for(&fp, &env.deps);
        assert_eq!(frozen, "First Sighted Title.");

        env.deps
            .cache
            .put_title(&fp, "Retitled Upload")
            .expect("a repeated put is a successful no-op");

        assert_eq!(
            effective_prompt_for(&fp, &env.deps),
            frozen,
            "the conditioning prompt — and with it the transcript id — is frozen at first sighting"
        );
    }

    #[test]
    fn a_broken_cache_read_degrades_to_a_fresh_probe_not_a_fault() {
        let env = env(probe(90.0, vec![]), false);
        let file = media_file(&env, "talk.mp4");
        let (_, id) = cached_probe(&file, &env.deps).expect("first probe stores the doc");
        // Corrupt the cache slot into something read_to_string cannot
        // consume: a directory where the document should be.
        let doc_path = env
            .deps
            .config
            .paths
            .cache
            .join("probe")
            .join(format!("{id}.json"));
        std::fs::remove_file(&doc_path).expect("clear the slot");
        std::fs::create_dir(&doc_path).expect("plant an unreadable entry");

        let (summary, _) =
            cached_probe(&file, &env.deps).expect("a broken read is a miss, never a fault");
        assert_eq!(summary.duration, 90.0, "the fresh probe answered");
    }

    #[test]
    fn sidecars_never_enter_the_cached_doc() {
        let env = env(
            probe(90.0, vec!["embedded #0 (eng, subrip)".to_string()]),
            false,
        );
        let file = media_file(&env, "talk.mp4");
        std::fs::write(env.dir.join("talk.srt"), b"").expect("write sidecar");

        let (summary, id) = cached_probe(&file, &env.deps).expect("probe");

        assert_eq!(
            summary.subtitles,
            ["embedded #0 (eng, subrip)", "sidecar talk.srt"],
            "the live summary advertises the sidecar"
        );
        let doc = env
            .deps
            .cache
            .get_probe(&id)
            .expect("cache read")
            .expect("stable file stores a doc");
        let stored = parse_probe_doc(&doc).expect("current version parses");
        assert_eq!(stored.subtitles, ["embedded #0 (eng, subrip)"]);
    }

    #[test]
    fn cached_probe_rescans_sidecars_on_hit() {
        let env = env(probe(90.0, vec![]), false);
        let dir_b = env.dir.join("b");
        std::fs::create_dir_all(&dir_b).expect("create dir b");
        let file_a = media_file(&env, "talk.mp4");
        let file_b = dir_b.join("talk.mp4");
        std::fs::copy(&file_a, &file_b).expect("same content, new directory");
        std::fs::write(dir_b.join("talk.srt"), b"").expect("write sidecar in b");
        let (_, id) = cached_probe(&file_a, &env.deps).expect("first probe stores the doc");
        // A marker only the cache can produce: seeing it back proves the
        // second call was a hit, not a fresh probe.
        let marker = render_probe_doc(&probe(90.0, vec!["embedded #0 (marker)".to_string()]))
            .expect("renders");
        env.deps
            .cache
            .put_probe(&id, &marker)
            .expect("seed marker doc");

        let (hit, _) = cached_probe(&file_b, &env.deps).expect("second probe");

        assert_eq!(
            hit.subtitles,
            ["embedded #0 (marker)", "sidecar talk.srt"],
            "a cache hit reflects the new directory's sidecars"
        );
    }
}
