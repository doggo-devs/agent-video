// SPDX-License-Identifier: Apache-2.0
//! Probe-with-cache: a version-tagged on-disk envelope so a summary written
//! under an older schema — or older field semantics — misses instead of
//! being read as current, plus the content-fingerprint cache lookup that
//! decorates hits with any cached whisper transcript.

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

const PROBE_DOC_VERSION: u32 = 1;

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
/// never cached. A cached whisper transcript decorates the summary (never
/// the cache entry).
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
            {
                let _ = deps.cache.put_probe(&fp.id, &doc);
            }
            p
        }
    };
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
    use crate::mcp::testutil::probe;

    #[test]
    fn probe_doc_round_trips_current_version() {
        let summary = probe(90.0, vec!["sidecar talk.srt".to_string()]);
        let doc = render_probe_doc(&summary).expect("renders");
        let back = parse_probe_doc(&doc).expect("current version parses");
        assert_eq!(back.duration, 90.0);
        assert_eq!(back.subtitles, ["sidecar talk.srt"]);
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
}
