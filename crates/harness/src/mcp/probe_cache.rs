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
        .then(|| deps.cache.get_probe(&fp.id).ok().flatten())
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
        .find_transcript(&fp.id, &transcribe::transcript_id(&deps.config))
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
