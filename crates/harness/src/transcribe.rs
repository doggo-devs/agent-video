// SPDX-License-Identifier: Apache-2.0
//! Transcription behind a resolved [`Route`]: an OpenAI-compatible API
//! when a key is configured, local CPU recognition otherwise.
//!
//! **Local** (`sense`): SenseVoice over Silero VAD spans, in-process on
//! the CPU. Weights are provisioned by `install-asr` and never fetched
//! here. No GPU, no conditioning — the tier exists to be cheap and
//! offline, not to match the API's accuracy or its language coverage.
//!
//! **API** (`api`): audio is encoded as 16-bit WAV and POSTed one bounded
//! chunk (~13 min) at a time, with no VAD; only the chunk in flight is
//! resident. Short gaps are padded; long gaps and the upload bound split chunks.
//! A cue spanning a capacity cut may lose or repeat a word. Each
//! chunk after the first is conditioned on the previous chunk's transcript
//! tail (`prompt`) so vocabulary and script carry across the cut.
//!
//! This module owns the policy seams — route resolution, cache identity,
//! the entry point, SRT publication; `wav` and `multipart` are
//! generic codecs with no transcription knowledge.

mod api;
mod multipart;
mod prompt;
#[cfg(feature = "local-asr")]
mod sense;
mod wav;

pub use prompt::{PROMPT_MAX_BYTES, effective_prompt};

use std::path::Path;
#[cfg(feature = "local-asr")]
use std::path::PathBuf;

use super::config::Config;
use crate::cache::Cache;
use crate::fault::Fault;
use crate::identity::ContentId;
use crate::time::Window;
use agent_video_core::media::Stop;
use agent_video_core::media::audio::AudioSamples;
use agent_video_core::source::Source;
use agent_video_core::subtitle as core_subtitle;
use agent_video_core::types::format_ts;

// ── local weights ──

/// Pinned filenames under `paths.models`, provisioned by `install-asr`.
///
/// Everything in this section is the local tier's, and exists only where
/// that tier does.
#[cfg(feature = "local-asr")]
mod names {
    use super::{Config, PathBuf};

    /// Pinned filenames under `paths.models`, provisioned by `install-asr`.
    /// The recognizer's date suffix is part of the name: a re-export of the
    /// same architecture is a different model, and must not answer for the
    /// old one's cached transcripts.
    pub const SENSE_MODEL_FILENAME: &str = "sense-voice-2024-07-17.int8.onnx";
    pub const SENSE_TOKENS_FILENAME: &str = "sense-voice-2024-07-17.tokens.txt";
    pub const VAD_FILENAME: &str = "silero_vad.onnx";

    /// Where the local recognizer's weights, its token table, and the VAD
    /// model must live for the local route to resolve.
    pub fn sense_model_path(cfg: &Config) -> PathBuf {
        cfg.paths.models.join(SENSE_MODEL_FILENAME)
    }

    pub fn sense_tokens_path(cfg: &Config) -> PathBuf {
        cfg.paths.models.join(SENSE_TOKENS_FILENAME)
    }

    pub fn vad_model_path(cfg: &Config) -> PathBuf {
        cfg.paths.models.join(VAD_FILENAME)
    }

    /// Cache identity of the local recognizer. A different export of the
    /// same architecture is a different engine, so the pinned date rides in
    /// the key.
    pub const SENSE_CACHE_ID: &str = "sense-voice-2024-07-17";

    /// The install hint shown when local weights are missing.
    pub fn install_hint(cfg: &Config) -> String {
        format!(
            "no local recognizer under {} — run `agent-video install-asr` to provision it",
            cfg.paths.models.display()
        )
    }
}

#[cfg(feature = "local-asr")]
pub use names::*;

// ── route ──

/// A resolved transcription route: which engine will run, carrying
/// everything its path needs. Resolution happens once, up front —
/// downstream code never re-derives availability. In builds without
/// `local-asr` the local route is unrepresentable.
#[derive(Debug)]
pub enum Route {
    /// OpenAI-compatible transcription endpoint.
    Api { key: String, url: String },
    /// In-process SenseVoice on the CPU.
    #[cfg(feature = "local-asr")]
    Sense(sense::Weights),
}

impl Route {
    /// Resolve the route from config.
    ///
    /// A configured key wins: the endpoint is the accurate, wide-language
    /// tier, and an operator who set a key asked for it. Local
    /// recognition is what answers without one — provisioned weights and
    /// no device to probe, so "installed" is the whole condition.
    pub fn resolve(cfg: &Config) -> Result<Route, Fault> {
        if let Some(key) = cfg.whisper.api_key.clone() {
            return Ok(Route::Api {
                key,
                url: cfg.whisper.api_url.clone(),
            });
        }
        #[cfg(feature = "local-asr")]
        {
            let weights = sense::Weights {
                model: sense_model_path(cfg),
                tokens: sense_tokens_path(cfg),
                vad: vad_model_path(cfg),
            };
            if [&weights.model, &weights.tokens, &weights.vad]
                .iter()
                .all(|p| p.is_file())
            {
                return Ok(Route::Sense(weights));
            }
        }
        Err(Fault::environment(format!(
            "no transcription path available — {}",
            unavailable_hint()
        )))
    }

    /// Transcript cache identity: cues from one engine × model must never
    /// answer for another's (the [`crate::cache::Cache::transcript_path`]
    /// contract), so the API route keys by the endpoint model, not by
    /// anything configured locally.
    pub fn cache_id(&self) -> String {
        match self {
            Route::Api { .. } => api_cache_id(),
            #[cfg(feature = "local-asr")]
            Route::Sense(_) => SENSE_CACHE_ID.to_string(),
        }
    }

    /// Whether this engine's output is shaped by the effective prompt.
    /// SenseVoice takes no conditioning text, so keying its transcripts
    /// by one would split the cache into entries that hold identical cues.
    fn takes_prompt(&self) -> bool {
        match self {
            Route::Api { .. } => true,
            #[cfg(feature = "local-asr")]
            Route::Sense(_) => false,
        }
    }
}

/// The API route's cache identity, derivable without a resolved route so
/// a transcript written before the key was removed stays findable.
fn api_cache_id() -> String {
    format!("api-{}", api::API_MODEL)
}

/// What to do about a host that cannot transcribe, worded for the tiers
/// this build actually carries. Every surface that has to say it reads it
/// from here: a build with no local recognizer must not send anyone to
/// `install-asr`.
pub fn unavailable_hint() -> &'static str {
    #[cfg(feature = "local-asr")]
    {
        "set WHISPER_API_KEY (or whisper.api_key in config.toml) for API transcription, or run \
         `agent-video install-asr` for local recognition"
    }
    #[cfg(not(feature = "local-asr"))]
    {
        "set WHISPER_API_KEY (or whisper.api_key in config.toml) for API transcription; this \
         build carries no local recognizer"
    }
}

/// Whether any transcription route is available.
pub fn available(cfg: &Config) -> bool {
    Route::resolve(cfg).is_ok()
}

// ── cache identity ──

/// Transcript identity: engine, model and conditioning.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TranscriptId(String);

impl std::fmt::Display for TranscriptId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

#[cfg(test)]
impl TranscriptId {
    /// Construct a recipe identity for cache fixtures.
    pub(crate) fn fixture(id: &str) -> Self {
        Self(id.to_owned())
    }
}

/// Render the cached SRT as `MM:SS text` lines, keeping the cues that
/// overlap `window`. Timestamps stay absolute to the source, so a slice
/// lines up with the grids of the same range.
pub fn render_transcript(srt: &str, window: Window) -> String {
    let mut out = String::new();
    for cue in agent_video_core::subtitle::parse_srt(srt) {
        if f64::from(cue.end) < window.start() || f64::from(cue.start) > window.end() {
            continue;
        }
        out.push_str(&format!("{} {}\n", format_ts(cue.start), cue.text));
    }
    out.trim_end().to_string()
}

/// The prompt this content gets today: the operator's, else one derived
/// from a cached title.
///
/// Lives here rather than beside either surface because both must derive
/// it identically — a CLI run and a tool call that disagreed would miss
/// each other's cached transcripts and pay for one video twice.
pub fn effective_prompt_for(cache: &Cache, cfg: &Config, fingerprint: &ContentId) -> String {
    effective_prompt(cfg, cache.get_title(fingerprint).as_deref())
}

/// The current transcript recipe for this content ([`transcript_id`]).
pub fn transcript_id_for(cache: &Cache, cfg: &Config, fingerprint: &ContentId) -> TranscriptId {
    transcript_id(cfg, &effective_prompt_for(cache, cfg, fingerprint))
}

/// Cache identity of the transcript the active route would produce for
/// this conditioning: engine, model, language and
/// effective prompt (only for engines that consume it). Falls back to the
/// API identity when no route resolves, preserving lookup after key removal.
pub fn transcript_id(cfg: &Config, prompt: &str) -> TranscriptId {
    let route = Route::resolve(cfg);
    let base = route
        .as_ref()
        .map(Route::cache_id)
        .unwrap_or_else(|_| api_cache_id());
    let prompt = match route.as_ref().map(Route::takes_prompt) {
        Ok(false) => "",
        _ => prompt,
    };
    if cfg.whisper.language.is_none() && prompt.is_empty() {
        return TranscriptId(base);
    }
    use sha2::{Digest, Sha256};
    let digest = Sha256::digest(prompt.as_bytes());
    TranscriptId(format!(
        "{base}-{}-{}",
        cfg.whisper.language.as_deref().unwrap_or("auto"),
        crate::identity::hex(&digest[..4])
    ))
}

// ── cue type ──

/// One recognized cue with absolute media timestamps in centiseconds.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Cue {
    pub start_cs: i64,
    pub end_cs: i64,
    pub text: String,
}

// ── entry point ──

/// Facts about a finished transcription the caller can judge honesty by.
#[derive(Debug, Clone, Copy)]
pub struct Transcribed {
    /// Source-time end of decoded audio, measured before speech filtering.
    pub decoded_end_seconds: f64,
}

/// Transcribe `file` into an SRT at `out_srt` (written atomically via
/// `.part`-then-rename). `prompt` is the effective conditioning text
/// ([`effective_prompt`]); the caller must key `out_srt` by
/// [`transcript_id`] over the same text. `on_step` receives coarse phase
/// messages suitable for MCP progress notifications.
pub fn transcribe(
    cfg: &Config,
    file: &Path,
    out_srt: &Path,
    prompt: &str,
    on_step: &mut dyn FnMut(&str),
) -> Result<Transcribed, Fault> {
    let route = Route::resolve(cfg)?;

    on_step("extracting audio");
    // `file` is a local path by this function's contract; re-classifying its
    // string form would misread a colon-named file (`clip:1.mp4`) as an
    // FFmpeg protocol URL. The typed knowledge is threaded, not re-guessed.
    let source = Source::Local(file.to_path_buf());
    // Both flavors of silence are the input's shape, not a decode failure:
    // a container with no audio track (screen capture, surveillance) and a
    // track that decodes to zero samples each get named for what they are.
    // Local-file contract (see above): no dead peer to stall on, so a
    // manual token that never trips satisfies the stall-guard requirement.
    let Some(mut stream) = AudioSamples::open(&source, &Stop::new())? else {
        return Err(Fault::bad_input(format!(
            "{} has no audio track — nothing to transcribe",
            file.display()
        )));
    };

    let mut next = |max| stream.next_chunk(max).map_err(Fault::from);
    let cues = match route {
        #[cfg(feature = "local-asr")]
        Route::Sense(weights) => sense::recognize(
            &weights,
            &mut next,
            cfg.whisper.language.as_deref(),
            on_step,
        )?,
        Route::Api { key, url } => api::recognize(
            &key,
            &url,
            cfg.whisper.language.as_deref(),
            prompt,
            &mut next,
            on_step,
        )?,
    };
    let decoded_end = stream
        .decoded_end()
        .ok_or_else(|| zero_samples_fault(file))?;
    if stream.dropped_frames() > 0 {
        on_step(&format!(
            "discarded {} audio frames with unconfirmed timestamps",
            stream.dropped_frames()
        ));
    }
    write_srt(out_srt, &cues)?;
    Ok(Transcribed {
        decoded_end_seconds: decoded_end.seconds(),
    })
}

/// A track that decoded to zero samples, named for what it is.
fn zero_samples_fault(file: &Path) -> Fault {
    Fault::bad_input(format!(
        "{} decoded to zero audio samples — nothing to transcribe",
        file.display()
    ))
}

// ── SRT formatting ──

/// Serialize cues as SRT via core's one SRT writer
/// ([`agent_video_core::subtitle::serialize_srt`]), so the transcript is
/// guaranteed to parse back with core's `parse_srt`. Centiseconds embed
/// exactly in the format's millisecond resolution; negative times (a
/// hallucinated pre-roll segment) clamp to zero.
pub fn format_srt(cues: &[Cue]) -> String {
    core_subtitle::serialize_srt(cues.iter().map(|cue| core_subtitle::SrtCue {
        start_ms: cue.start_cs.max(0) as u64 * 10,
        end_ms: cue.end_cs.max(0) as u64 * 10,
        text: &cue.text,
    }))
}

/// Write the SRT atomically: parent dirs, per-writer `.part`, rename. The
/// transcript cache is shared across server processes and the destination
/// name is deterministic (fingerprint × route identity), so the staging
/// name must be per-writer, not per-process — two writers, even two
/// threads in one process, must not interleave into one file. The name
/// comes from the cache's one staging grammar
/// ([`crate::cache::naming::part_path`], docs/shared-cache.md Pattern 2).
/// A failed write removes its `.part` rather than leaving it for the
/// staleness sweep.
fn write_srt(out_srt: &Path, cues: &[Cue]) -> Result<(), Fault> {
    if let Some(parent) = out_srt.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| Fault::environment(format!("cannot create {}: {e}", parent.display())))?;
    }
    let part = crate::cache::naming::part_path(out_srt);
    std::fs::write(&part, format_srt(cues))
        .map_err(|e| Fault::environment(format!("cannot write {}: {e}", part.display())))
        .and_then(|()| {
            std::fs::rename(&part, out_srt).map_err(|e| {
                Fault::environment(format!("cannot finalize {}: {e}", out_srt.display()))
            })
        })
        .inspect_err(|_| {
            let _ = std::fs::remove_file(&part);
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{default_config, test_dirs};

    fn cfg() -> Config {
        default_config(&test_dirs(Path::new("/home/u")))
    }

    /// Three cues at 0-2s, 4-6s and 8-10s.
    const SRT: &str = "1\n00:00:00,000 --> 00:00:02,000\nfirst\n\n\
         2\n00:00:04,000 --> 00:00:06,000\nsecond\n\n\
         3\n00:00:08,000 --> 00:00:10,000\nthird\n";

    fn window(from: f64, until: f64) -> Window {
        use crate::schema::TimeValue;
        crate::time::resolve_window(
            Some(&TimeValue::Number(from)),
            Some(&TimeValue::Number(until)),
            30.0,
        )
        .expect("a valid sub-range")
        .0
    }

    #[test]
    fn renders_every_cue_of_a_full_window() {
        let text = render_transcript(SRT, window(0.0, 30.0));
        assert_eq!(text, "00:00 first\n00:04 second\n00:08 third");
    }

    #[test]
    fn keeps_only_cues_overlapping_the_window() {
        let text = render_transcript(SRT, window(3.0, 7.0));
        assert_eq!(text, "00:04 second");
    }

    #[test]
    fn keeps_a_cue_straddling_the_window_edge() {
        // The cue runs 4-6s and the window opens at 5s: a caller asking
        // about 5s wants the sentence being spoken at 5s, not silence.
        let text = render_transcript(SRT, window(5.0, 7.0));
        assert_eq!(text, "00:04 second");
    }

    #[test]
    fn timestamps_stay_absolute_to_the_source() {
        let text = render_transcript(SRT, window(7.0, 11.0));
        assert_eq!(text, "00:08 third", "not re-based to the window start");
    }

    #[test]
    fn resolves_api_route_when_key_configured() {
        let mut cfg = cfg();
        cfg.whisper.api_key = Some("sk-test".to_string());
        match Route::resolve(&cfg).expect("api route resolves") {
            Route::Api { key, url } => {
                assert_eq!(key, "sk-test");
                assert_eq!(url, cfg.whisper.api_url);
            }
            #[cfg(feature = "local-asr")]
            other => panic!("expected the API route, got {other:?}"),
        }
    }

    #[cfg(feature = "local-asr")]
    #[test]
    fn local_weights_answer_without_a_key() {
        let dir = crate::tempdir::TempDir::new("route-sense");
        let mut cfg = cfg();
        cfg.paths.models = dir.to_path_buf();
        for name in [SENSE_MODEL_FILENAME, SENSE_TOKENS_FILENAME, VAD_FILENAME] {
            std::fs::write(dir.join(name), b"weights").expect("stage a weight file");
        }
        match Route::resolve(&cfg).expect("installed weights resolve") {
            Route::Sense(w) => assert_eq!(w.model, sense_model_path(&cfg)),
            other => panic!("expected the local route, got {other:?}"),
        }
        assert_eq!(transcript_id(&cfg, "").to_string(), SENSE_CACHE_ID);
    }

    #[cfg(feature = "local-asr")]
    #[test]
    fn a_configured_key_outranks_installed_weights() {
        let dir = crate::tempdir::TempDir::new("route-both");
        let mut cfg = cfg();
        cfg.paths.models = dir.to_path_buf();
        cfg.whisper.api_key = Some("sk-test".to_string());
        for name in [SENSE_MODEL_FILENAME, SENSE_TOKENS_FILENAME, VAD_FILENAME] {
            std::fs::write(dir.join(name), b"weights").expect("stage a weight file");
        }
        assert!(matches!(
            Route::resolve(&cfg).expect("a key resolves"),
            Route::Api { .. }
        ));
    }

    #[cfg(feature = "local-asr")]
    #[test]
    fn a_partial_install_is_not_a_route() {
        let dir = crate::tempdir::TempDir::new("route-partial");
        let mut cfg = cfg();
        cfg.paths.models = dir.to_path_buf();
        std::fs::write(dir.join(SENSE_MODEL_FILENAME), b"weights").expect("stage");
        let err = Route::resolve(&cfg).expect_err("tokens and VAD are missing");
        assert!(err.message.contains("install-asr"), "{err}");
    }

    #[cfg(feature = "local-asr")]
    #[test]
    fn the_local_route_is_not_keyed_by_the_prompt() {
        // SenseVoice takes no conditioning text, so two prompts must not
        // split one recognition into two cache entries.
        let dir = crate::tempdir::TempDir::new("route-prompt");
        let mut cfg = cfg();
        cfg.paths.models = dir.to_path_buf();
        for name in [SENSE_MODEL_FILENAME, SENSE_TOKENS_FILENAME, VAD_FILENAME] {
            std::fs::write(dir.join(name), b"weights").expect("stage a weight file");
        }
        assert_eq!(
            transcript_id(&cfg, "Rust Ownership."),
            transcript_id(&cfg, "Kubernetes, etcd.")
        );
    }

    #[test]
    fn reports_no_route_without_a_key() {
        let err = Route::resolve(&cfg()).expect_err("nothing is provisioned");
        assert_eq!(err.kind, crate::fault::FaultKind::Environment);
        assert!(err.message.contains("WHISPER_API_KEY"), "{err}");
    }

    #[cfg(not(feature = "local-asr"))]
    #[test]
    fn a_build_without_the_local_tier_says_so() {
        // Staged weights must not change the answer: this build cannot
        // load them, and pointing at `install-asr` would be a lie.
        let dir = crate::tempdir::TempDir::new("route-none");
        let mut cfg = cfg();
        cfg.paths.models = dir.to_path_buf();
        for name in [
            "sense-voice-2024-07-17.int8.onnx",
            "sense-voice-2024-07-17.tokens.txt",
            "silero_vad.onnx",
        ] {
            std::fs::write(dir.join(name), b"weights").expect("stage a weight file");
        }
        let err = Route::resolve(&cfg).expect_err("no key, no local tier");
        assert!(err.message.contains("no local recognizer"), "{err}");
        assert!(!err.message.contains("install-asr"), "{err}");
    }

    #[test]
    fn api_transcripts_key_by_endpoint_model() {
        let mut cfg = cfg();
        cfg.whisper.api_key = Some("sk-test".to_string());
        assert_eq!(
            transcript_id(&cfg, "").to_string(),
            format!("api-{}", api::API_MODEL)
        );
    }

    #[test]
    fn transcript_lookup_survives_a_removed_key() {
        // No route resolves, but a transcript generated before the key was
        // removed should still be findable under the endpoint identity.
        assert_eq!(
            transcript_id(&cfg(), "").to_string(),
            format!("api-{}", api::API_MODEL)
        );
    }

    #[test]
    fn transcript_id_is_stable_for_the_same_conditioning() {
        let mut cfg = cfg();
        cfg.whisper.api_key = Some("sk-test".to_string());
        cfg.whisper.language = Some("zh".to_string());
        assert_eq!(
            transcript_id(&cfg, "Rust Ownership."),
            transcript_id(&cfg, "Rust Ownership."),
            "same model × language × prompt must be the same cache entry"
        );
    }

    #[test]
    fn transcript_id_changes_with_the_language_hint() {
        let mut cfg = cfg();
        cfg.whisper.api_key = Some("sk-test".to_string());
        let auto = transcript_id(&cfg, "");
        cfg.whisper.language = Some("zh".to_string());
        assert_ne!(
            auto,
            transcript_id(&cfg, ""),
            "a language-pinned transcript must not answer for an auto-detected one"
        );
    }

    #[test]
    fn transcript_id_changes_with_the_prompt() {
        let mut cfg = cfg();
        cfg.whisper.api_key = Some("sk-test".to_string());
        let bare = transcript_id(&cfg, "");
        let a = transcript_id(&cfg, "Rust Ownership.");
        let b = transcript_id(&cfg, "Kubernetes, etcd.");
        assert_ne!(bare, a, "conditioned output is a different cache entry");
        assert_ne!(a, b, "different prompts, different entries");
        for id in [&a, &b] {
            assert!(
                id.to_string()
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || c == '-'),
                "transcript id {id:?} must stay filename-safe"
            );
        }
    }

    #[test]
    fn unresolved_route_fails_transcription_with_hint() {
        let err = transcribe(
            &cfg(),
            Path::new("/nonexistent.mp4"),
            Path::new("/tmp/out.srt"),
            "",
            &mut |_| {},
        )
        .expect_err("no route is provisioned");
        assert_eq!(err.kind, crate::fault::FaultKind::Environment);
    }

    #[test]
    fn clamps_negative_cue_times_to_zero() {
        let cues = vec![Cue {
            start_cs: -5,
            end_cs: 150,
            text: "pre-roll".into(),
        }];
        assert!(
            format_srt(&cues).contains("00:00:00,000 --> 00:00:01,500"),
            "negatives clamp: {}",
            format_srt(&cues)
        );
    }

    #[test]
    fn formats_numbered_srt_blocks() {
        let cues = vec![
            Cue {
                start_cs: 0,
                end_cs: 150,
                text: "hello".into(),
            },
            Cue {
                start_cs: 200,
                end_cs: 350,
                text: "world".into(),
            },
        ];
        let srt = format_srt(&cues);
        assert_eq!(
            srt,
            "1\n00:00:00,000 --> 00:00:01,500\nhello\n\n\
             2\n00:00:02,000 --> 00:00:03,500\nworld\n\n"
        );
    }

    #[test]
    fn whisper_srt_parses_with_core_parser() {
        // The cross-crate contract: whatever the whisper writer emits must
        // round-trip through core's SRT reader with centisecond fidelity.
        let cues = vec![
            Cue {
                start_cs: 7,
                end_cs: 253,
                text: "first cue".into(),
            },
            Cue {
                start_cs: 360_099,
                end_cs: 366_150,
                text: "past the hour".into(),
            },
        ];
        let parsed = core_subtitle::parse_srt(&format_srt(&cues));
        assert_eq!(parsed.len(), cues.len());
        for (ours, theirs) in cues.iter().zip(&parsed) {
            assert!((ours.start_cs as f32 / 100.0 - theirs.start).abs() < 0.001);
            assert!((ours.end_cs as f32 / 100.0 - theirs.end).abs() < 0.001);
            assert_eq!(ours.text, theirs.text);
        }
    }

    #[test]
    fn decodes_colon_named_local_file_as_a_file_not_a_protocol() {
        // A relative path like `clip:1.mp4` looks like an FFmpeg protocol
        // URL to string re-classification; transcribe receives a &Path that
        // is local by contract and must reach the decoder as a file. The
        // file does not exist, so the failure must be a file-open error
        // naming the path — never "Protocol not found".
        let mut cfg = cfg();
        cfg.whisper.api_key = Some("sk-test".to_string()); // route resolves; decode runs
        let err = transcribe(
            &cfg,
            Path::new("clip:1.mp4"),
            Path::new("/tmp/out.srt"),
            "",
            &mut |_| {},
        )
        .expect_err("the file does not exist");
        assert!(
            !err.message.to_lowercase().contains("protocol"),
            "misclassified as a protocol URL: {err}"
        );
        assert!(err.message.contains("clip:1.mp4"), "names the file: {err}");
    }
}
