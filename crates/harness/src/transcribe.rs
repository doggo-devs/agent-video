// SPDX-License-Identifier: Apache-2.0
//! In-process whisper transcription via whisper.cpp (`whisper-rs`).
//!
//! Replaces the TS harness's spawned `whisper-cli`/`whisper` processes:
//! audio decodes in-process (`media::audio`), Silero VAD trims silence and
//! music before recognition (the dominant hallucination source on real
//! videos), and the transcript lands as an SRT file whose cue times feed the
//! grid subtitle ride-along. Weights are provisioned by `install-whisper`
//! and never fetched here.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use whisper_rs::{
    FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters, WhisperVadContext,
    WhisperVadContextParams, WhisperVadParams,
};

use super::config::{Config, WhisperModel};
use crate::fault::Fault;
use agent_video_core::media::audio::{WHISPER_SAMPLE_RATE, decode_audio_16k_mono};
use agent_video_core::source::Source;

/// Pinned ggml filename for a named model under `paths.models`.
pub fn model_filename(model: &WhisperModel) -> Option<&'static str> {
    match model {
        WhisperModel::Medium => Some("ggml-medium.bin"),
        WhisperModel::LargeV3 => Some("ggml-large-v3.bin"),
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

/// The install hint shown when no weights are on disk.
pub fn install_hint(cfg: &Config) -> String {
    format!(
        "no whisper model at {} — run `agent-video install-whisper` to provision it, or point \
         whisper.model at a ggml file",
        model_path(cfg).display()
    )
}

/// One recognized cue with absolute media timestamps in centiseconds.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Cue {
    pub start_cs: i64,
    pub end_cs: i64,
    pub text: String,
}

/// Transcribe `file` into an SRT at `out_srt` (written atomically via
/// `.part`-then-rename). `on_step` receives coarse phase messages suitable
/// for MCP progress notifications.
///
/// Requires the configured model on disk ([`install_hint`] otherwise). The
/// VAD model is optional: when present, only speech segments are recognized;
/// when absent, the whole track is.
pub fn transcribe(
    cfg: &Config,
    file: &Path,
    out_srt: &Path,
    on_step: &mut dyn FnMut(&str),
) -> Result<(), Fault> {
    let model = model_path(cfg);
    if !model.is_file() {
        return Err(Fault::environment(install_hint(cfg)));
    }

    on_step("extracting audio");
    let source = Source::classify(&file.to_string_lossy());
    let samples = decode_audio_16k_mono(&source)?;
    if samples.is_empty() {
        return Err(Fault::bad_input(format!(
            "{} decoded to zero audio samples — nothing to transcribe",
            file.display()
        )));
    }

    let spans = speech_spans(cfg, &samples)?;
    if spans.is_empty() {
        return Err(Fault::bad_input(
            "voice activity detection found no speech — nothing to transcribe",
        ));
    }

    on_step("transcribing (CPU-bound, may take minutes)");
    let cues = recognize(cfg, &model, &samples, &spans)?;
    write_srt(out_srt, &cues)
}

/// Sample-index spans to recognize: VAD speech segments when the VAD model
/// is installed, the whole track otherwise.
fn speech_spans(cfg: &Config, samples: &[f32]) -> Result<Vec<(usize, usize)>, Fault> {
    let vad_model = vad_model_path(cfg);
    let Some(vad_model) = vad_model.to_str().filter(|_| vad_model.is_file()) else {
        return Ok(vec![(0, samples.len())]);
    };
    let mut vad_ctx_params = WhisperVadContextParams::new();
    vad_ctx_params.set_n_threads(threads());
    let mut vad = WhisperVadContext::new(vad_model, vad_ctx_params)
        .map_err(|e| Fault::environment(format!("loading VAD model {vad_model}: {e}")))?;
    let segments = vad
        .segments_from_samples(WhisperVadParams::new(), samples)
        .map_err(|e| Fault::internal(format!("voice activity detection failed: {e}")))?;
    Ok(segments
        .map(|seg| {
            // VAD timestamps are centiseconds; 16 kHz ⇒ 160 samples per cs.
            let per_cs = WHISPER_SAMPLE_RATE as usize / 100;
            let start = (seg.start as usize)
                .saturating_mul(per_cs)
                .min(samples.len());
            let end = (seg.end as usize).saturating_mul(per_cs).min(samples.len());
            (start, end)
        })
        .filter(|(start, end)| end > start)
        .collect())
}

/// Run whisper over each span, rebasing cue times to absolute media time.
fn recognize(
    cfg: &Config,
    model: &Path,
    samples: &[f32],
    spans: &[(usize, usize)],
) -> Result<Vec<Cue>, Fault> {
    let model = model.to_str().ok_or_else(|| {
        Fault::environment(format!("model path {} is not UTF-8", model.display()))
    })?;
    let mut ctx_params = WhisperContextParameters::default();
    ctx_params.use_gpu(cfg.whisper.backend.use_gpu());
    let ctx = WhisperContext::new_with_params(model, ctx_params)
        .map_err(|e| Fault::environment(format!("loading whisper model {model}: {e}")))?;
    let mut state = ctx
        .create_state()
        .map_err(|e| Fault::environment(format!("initializing whisper: {e}")))?;

    let mut cues = Vec::new();
    for &(start, end) in spans {
        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        params.set_language(Some("auto"));
        // Spans are independent; conditioning on prior text re-invites the
        // repeat loops VAD exists to prevent.
        params.set_no_context(true);
        params.set_print_special(false);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);
        params.set_n_threads(threads());

        state
            .full(params, &samples[start..end])
            .map_err(|e| Fault::internal(format!("whisper inference failed: {e}")))?;

        let offset_cs = (start / (WHISPER_SAMPLE_RATE as usize / 100)) as i64;
        for segment in state.as_iter() {
            let text = match segment.to_str_lossy() {
                Ok(t) => t.trim().to_string(),
                Err(_) => continue,
            };
            if text.is_empty() {
                continue;
            }
            cues.push(Cue {
                start_cs: segment.start_timestamp() + offset_cs,
                end_cs: segment.end_timestamp() + offset_cs,
                text,
            });
        }
    }
    Ok(cues)
}

fn threads() -> i32 {
    std::thread::available_parallelism()
        .map(|n| n.get() as i32)
        .unwrap_or(4)
}

/// Serialize cues as SRT.
pub fn format_srt(cues: &[Cue]) -> String {
    let mut out = String::new();
    for (i, cue) in cues.iter().enumerate() {
        let _ = writeln!(out, "{}", i + 1);
        let _ = writeln!(
            out,
            "{} --> {}",
            srt_timestamp(cue.start_cs),
            srt_timestamp(cue.end_cs)
        );
        let _ = writeln!(out, "{}\n", cue.text);
    }
    out
}

/// `HH:MM:SS,mmm` from centiseconds, clamping negatives to zero.
fn srt_timestamp(cs: i64) -> String {
    let ms = cs.max(0) * 10;
    let (h, m, s, ms) = (
        ms / 3_600_000,
        ms % 3_600_000 / 60_000,
        ms % 60_000 / 1000,
        ms % 1000,
    );
    format!("{h:02}:{m:02}:{s:02},{ms:03}")
}

/// Write the SRT atomically: parent dirs, per-writer `.part`, rename. The
/// transcript cache is shared across server processes and the destination
/// name is deterministic (fingerprint × model), so the staging name carries
/// the pid — two processes transcribing the same content must not
/// interleave into one file.
fn write_srt(out_srt: &Path, cues: &[Cue]) -> Result<(), Fault> {
    if let Some(parent) = out_srt.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| Fault::environment(format!("cannot create {}: {e}", parent.display())))?;
    }
    let part = PathBuf::from(format!("{}.{}.part", out_srt.display(), std::process::id()));
    std::fs::write(&part, format_srt(cues))
        .map_err(|e| Fault::environment(format!("cannot write {}: {e}", part.display())))?;
    std::fs::rename(&part, out_srt).map_err(|e| {
        let _ = std::fs::remove_file(&part);
        Fault::environment(format!("cannot finalize {}: {e}", out_srt.display()))
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
            Path::new("/home/u/data-base/agent-video/models/ggml-medium.bin")
        );
    }

    #[test]
    fn file_model_is_the_escape_hatch() {
        let mut cfg = cfg();
        cfg.whisper.model = WhisperModel::File(PathBuf::from("/opt/ggml-tiny.bin"));
        assert_eq!(model_path(&cfg), Path::new("/opt/ggml-tiny.bin"));
    }

    #[test]
    fn formats_srt_timestamps_past_the_hour() {
        assert_eq!(srt_timestamp(0), "00:00:00,000");
        assert_eq!(srt_timestamp(123), "00:00:01,230");
        assert_eq!(srt_timestamp(366_150), "01:01:01,500");
        assert_eq!(srt_timestamp(-5), "00:00:00,000", "negatives clamp");
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
    fn missing_model_reports_install_hint() {
        let err = transcribe(
            &cfg(),
            Path::new("/nonexistent.mp4"),
            Path::new("/tmp/out.srt"),
            &mut |_| {},
        )
        .expect_err("no model on disk");
        assert_eq!(err.kind, crate::fault::FaultKind::Environment);
        assert!(err.message.contains("install-whisper"), "{err}");
    }
}
