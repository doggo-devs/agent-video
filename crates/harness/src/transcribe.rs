// SPDX-License-Identifier: Apache-2.0
//! Whisper transcription behind a resolved [`Route`]: local GPU inference
//! when compiled and provisioned, an OpenAI-compatible API otherwise.
//!
//! **Local** (`local`, `whisper-local` feature + GPU backend):
//! in-process whisper.cpp; Silero VAD trims silence and music before
//! recognition. Weights are provisioned by `install-whisper` and never
//! fetched here.
//!
//! **API** (`api`): audio is decoded, encoded as 16-bit WAV, and POSTed
//! one bounded chunk at a time (~13 min each) — no VAD pre-filtering; the
//! endpoint owns hallucination suppression. Only the chunk in flight is
//! resident, so peak memory is one chunk, never the track. Chunks split
//! at hard byte boundaries, so a cue spanning a cut may lose or repeat a
//! word: accepted for the simplicity of stateless uploads. Each chunk
//! after the first is conditioned on the tail of the previous chunk's
//! transcript (`prompt`), so vocabulary, casing, and script choice
//! carry across the cut instead of lurching.
//!
//! This module owns the policy seams — route resolution, cache identity,
//! the entry point, SRT publication; `wav` and `multipart` are
//! generic codecs with no transcription knowledge.

mod api;
#[cfg(feature = "whisper-local")]
mod local;
mod multipart;
mod prompt;
mod wav;

pub use prompt::{PROMPT_MAX_BYTES, effective_prompt};

use std::path::{Path, PathBuf};
#[cfg(feature = "whisper-local")]
use std::time::Instant;

#[cfg(feature = "whisper-local")]
use super::config::Backend;
use super::config::{Config, WhisperModel};
use crate::fault::Fault;
use agent_video_core::media::audio::{AudioSamples, WHISPER_SAMPLE_RATE};
use agent_video_core::source::Source;
use agent_video_core::subtitle as core_subtitle;

// ── model / path helpers ──

/// Pinned ggml filename for a named model under `paths.models`.
pub fn model_filename(model: &WhisperModel) -> Option<&'static str> {
    match model {
        WhisperModel::Medium => Some("ggml-medium.bin"),
        WhisperModel::LargeV3 => Some("ggml-large-v3.bin"),
        WhisperModel::LargeV3Turbo => Some("ggml-large-v3-turbo.bin"),
        WhisperModel::File(_) => None,
    }
}

/// Pinned Silero VAD model filename under `paths.models`.
pub const VAD_FILENAME: &str = "ggml-silero-v5.1.2.bin";

/// Where the configured whisper model must live: the models path for named
/// models, the configured path itself for the self-supplied escape hatch.
pub fn model_path(cfg: &Config) -> PathBuf {
    match &cfg.whisper.model {
        WhisperModel::File(p) => p.clone(),
        named => cfg
            .paths
            .models
            .join(model_filename(named).expect("named model has a pinned filename")),
    }
}

/// The VAD model's expected location.
pub fn vad_model_path(cfg: &Config) -> PathBuf {
    cfg.paths.models.join(VAD_FILENAME)
}

/// The install hint shown when local weights are missing.
pub fn install_hint(cfg: &Config) -> String {
    format!(
        "no whisper model at {} — run `agent-video install-whisper` to provision it, or point \
         whisper.model at a ggml file",
        model_path(cfg).display()
    )
}

// ── route ──

/// A resolved transcription route: which engine will run, carrying
/// everything its path needs. Resolution happens once, up front —
/// downstream code never re-derives availability. In builds without
/// `whisper-local` the local route is unrepresentable.
#[derive(Debug)]
pub enum Route {
    /// In-process whisper.cpp on a compiled-in GPU backend.
    #[cfg(feature = "whisper-local")]
    Local {
        backend: Backend,
        model: PathBuf,
        /// Cache identity of the model ([`WhisperModel::cache_id`]).
        model_id: String,
    },
    /// OpenAI-compatible transcription endpoint.
    Api { key: String, url: String },
}

impl Route {
    /// Resolve the route from config.
    ///
    /// Local inference wins when fully provisioned: compiled backend, a
    /// device that passes the runtime probe ([`crate::gpu`]), and the
    /// model on disk. An *explicit* `whisper.backend` pins the local
    /// route: a missing model or a failed probe is then an error, never a
    /// silent fallback to the API — that would swap the engine and the
    /// model behind the operator's back. Otherwise the API route needs a
    /// key.
    pub fn resolve(cfg: &Config) -> Result<Route, Fault> {
        #[cfg(feature = "whisper-local")]
        {
            if let Some(backend) = local_backend(cfg, crate::gpu::device_available)? {
                let model = model_path(cfg);
                if model.is_file() {
                    return Ok(Route::Local {
                        backend,
                        model,
                        model_id: cfg.whisper.model.cache_id(),
                    });
                }
                if cfg.whisper.backend.is_some() {
                    return Err(Fault::environment(install_hint(cfg)));
                }
            }
        }
        match cfg.whisper.api_key.clone() {
            Some(key) => Ok(Route::Api {
                key,
                url: cfg.whisper.api_url.clone(),
            }),
            None => Err(Fault::environment(
                "no transcription path available — set WHISPER_API_KEY (or whisper.api_key in \
                 config.toml) for API transcription, or build with a GPU feature and run \
                 `agent-video install-whisper` for local inference",
            )),
        }
    }

    /// Transcript cache identity: cues from one engine × model must never
    /// answer for another's (the [`crate::cache::Cache::transcript_path`]
    /// contract), so the API route keys by the endpoint model, not the
    /// locally configured one.
    pub fn cache_id(&self) -> String {
        match self {
            #[cfg(feature = "whisper-local")]
            Route::Local { model_id, .. } => model_id.clone(),
            Route::Api { .. } => format!("api-{}", api::API_MODEL),
        }
    }
}

/// The local backend this host can actually run, or `None` to yield to
/// the API route. Probe-before-decide: device presence is settled here,
/// up front, never discovered mid-transcription inside ggml (whose
/// failure paths abort). The pin contract holds throughout: an explicit
/// `whisper.backend` that fails its probe is a hard error, never a silent
/// engine swap.
#[cfg(feature = "whisper-local")]
fn local_backend(
    cfg: &Config,
    device_available: impl Fn(Backend) -> bool,
) -> Result<Option<Backend>, Fault> {
    if let Some(pinned) = cfg.whisper.backend {
        if !device_available(pinned) {
            return Err(Fault::environment(format!(
                "whisper.backend is pinned to '{name}' but no usable {name} device was found \
                 on this host — fix the driver/loader, or remove the pin to allow API \
                 transcription",
                name = pinned.name()
            )));
        }
        return Ok(Some(pinned));
    }
    Ok(Backend::compiled()
        .iter()
        .copied()
        .find(|&b| device_available(b)))
}

/// Whether any transcription route is available.
pub fn available(cfg: &Config) -> bool {
    Route::resolve(cfg).is_ok()
}

// ── cache identity ──

/// Cache identity of the transcript the active route would produce for
/// this conditioning. Everything that shapes the cues is in the key:
/// engine × model ([`Route::cache_id`]), the language hint, and a short
/// hash of the effective `prompt`. Unconditioned runs keep the bare model
/// identity, so transcripts from before these knobs existed stay hits.
/// Falls back to the local model identity when no route resolves, so a
/// transcript generated before a key was removed still rides along.
pub fn transcript_id(cfg: &Config, prompt: &str) -> String {
    let base = Route::resolve(cfg)
        .map(|r| r.cache_id())
        .unwrap_or_else(|_| cfg.whisper.model.cache_id());
    if cfg.whisper.language.is_none() && prompt.is_empty() {
        return base;
    }
    use sha2::{Digest, Sha256};
    let digest = Sha256::digest(prompt.as_bytes());
    format!(
        "{base}-{}-{}",
        cfg.whisper.language.as_deref().unwrap_or("auto"),
        crate::identity::hex(&digest[..4])
    )
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
    /// Seconds of audio actually decoded and fed to recognition — less
    /// than the container duration when the audio track was truncated.
    pub audio_seconds: f64,
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
    let Some(mut stream) = AudioSamples::open(&source)? else {
        return Err(Fault::bad_input(format!(
            "{} has no audio track — nothing to transcribe",
            file.display()
        )));
    };

    let (cues, sample_count) = match route {
        #[cfg(feature = "whisper-local")]
        Route::Local { backend, model, .. } => {
            // The local route decodes whole: VAD wants random access.
            let t = Instant::now();
            let mut samples = Vec::new();
            stream.fill(&mut samples, usize::MAX)?;
            tracing::info!(
                elapsed_ms = t.elapsed().as_millis() as u64,
                samples = samples.len(),
                duration_s = samples.len() as f64 / f64::from(WHISPER_SAMPLE_RATE),
                "audio decode complete",
            );
            if samples.is_empty() {
                return Err(zero_samples_fault(file));
            }
            let count = samples.len();
            (
                local::recognize(cfg, backend, &model, &samples, prompt, on_step)?,
                count,
            )
        }
        Route::Api { key, url } => {
            // The API route streams: decode, encode, and upload one bounded
            // chunk at a time — the track is never materialized.
            let mut chunks =
                api::Chunker::new(|out, min| stream.fill(out, min).map_err(Fault::from));
            let cues = api::recognize(
                &key,
                &url,
                cfg.whisper.language.as_deref(),
                prompt,
                &mut chunks,
                on_step,
            )?;
            let count = chunks.samples_pulled();
            if count == 0 {
                return Err(zero_samples_fault(file));
            }
            tracing::info!(
                samples = count,
                duration_s = count as f64 / f64::from(WHISPER_SAMPLE_RATE),
                "audio decode complete",
            );
            (cues, count)
        }
    };

    write_srt(out_srt, &cues)?;
    Ok(Transcribed {
        audio_seconds: sample_count as f64 / f64::from(WHISPER_SAMPLE_RATE),
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

    #[test]
    fn named_models_resolve_under_models_path() {
        assert_eq!(
            model_path(&cfg()),
            Path::new("/home/u/data-base/agent-video/models/ggml-large-v3-turbo.bin")
        );
    }

    #[test]
    fn file_model_is_the_escape_hatch() {
        let mut cfg = cfg();
        cfg.whisper.model = WhisperModel::File(PathBuf::from("/opt/ggml-tiny.bin"));
        assert_eq!(model_path(&cfg), Path::new("/opt/ggml-tiny.bin"));
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
            #[cfg(feature = "whisper-local")]
            other => panic!("expected the API route, got {other:?}"),
        }
    }

    #[test]
    fn reports_no_route_without_key_or_local_model() {
        let err = Route::resolve(&cfg()).expect_err("nothing is provisioned");
        assert_eq!(err.kind, crate::fault::FaultKind::Environment);
        assert!(err.message.contains("WHISPER_API_KEY"), "{err}");
        assert!(err.message.contains("install-whisper"), "{err}");
    }

    #[test]
    fn api_transcripts_key_by_endpoint_model() {
        let mut cfg = cfg();
        cfg.whisper.api_key = Some("sk-test".to_string());
        let id = transcript_id(&cfg, "");
        assert_eq!(id, format!("api-{}", api::API_MODEL));
        assert_ne!(
            id,
            cfg.whisper.model.cache_id(),
            "an API transcript must never answer for a local model's"
        );
    }

    #[test]
    fn transcript_lookup_falls_back_to_local_model_id() {
        // No route resolves, but a transcript generated before a key was
        // removed should still be findable under the local identity.
        assert_eq!(transcript_id(&cfg(), ""), cfg().whisper.model.cache_id());
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
                id.chars().all(|c| c.is_ascii_alphanumeric() || c == '-'),
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

    #[cfg(feature = "whisper-local")]
    #[test]
    fn pinned_backend_failing_probe_is_a_hard_error() {
        let mut cfg = cfg();
        cfg.whisper.backend = Some(Backend::Vulkan);
        cfg.whisper.api_key = Some("sk-test".to_string());
        let err = local_backend(&cfg, |_| false).expect_err("pin commits to local");
        assert_eq!(err.kind, crate::fault::FaultKind::Environment);
        assert!(err.message.contains("vulkan"), "names the backend: {err}");
    }

    #[cfg(feature = "whisper-local")]
    #[test]
    fn pinned_backend_passing_probe_is_selected() {
        let mut cfg = cfg();
        cfg.whisper.backend = Some(Backend::Vulkan);
        let backend = local_backend(&cfg, |_| true).expect("probe passes");
        assert_eq!(backend, Some(Backend::Vulkan));
    }

    #[cfg(feature = "whisper-local")]
    #[test]
    fn auto_detection_yields_to_api_when_no_device_probes() {
        let backend = local_backend(&cfg(), |_| false).expect("auto never errors");
        assert_eq!(backend, None, "no usable device ⇒ the API route decides");
    }

    #[cfg(feature = "whisper-local")]
    #[test]
    fn auto_detection_picks_the_first_usable_compiled_backend() {
        let backend = local_backend(&cfg(), |_| true).expect("auto never errors");
        assert_eq!(backend, Backend::compiled().first().copied());
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
