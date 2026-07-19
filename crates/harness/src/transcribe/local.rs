// SPDX-License-Identifier: Apache-2.0
//! The local GPU route: in-process whisper.cpp via `whisper-rs`, with
//! Silero VAD trimming silence and music before recognition (the dominant
//! hallucination source on real videos). Compiled only with the
//! `whisper-local` feature; CPU-only local inference is not supported.

use std::path::Path;
use std::time::Instant;

use super::{Cue, vad_model_path};
use crate::config::{Backend, Config};
use crate::fault::Fault;
use agent_video_core::media::audio::WHISPER_SAMPLE_RATE;

/// Run VAD then whisper over each speech span, rebasing cue times to
/// absolute media time.
pub(super) fn recognize(
    cfg: &Config,
    backend: Backend,
    model: &Path,
    samples: &[f32],
    prompt: &str,
    on_step: &mut dyn FnMut(&str),
) -> Result<Vec<Cue>, Fault> {
    use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

    let t = Instant::now();
    let spans = speech_spans(cfg, samples)?;
    tracing::info!(
        elapsed_ms = t.elapsed().as_millis() as u64,
        segments = spans.len(),
        "VAD complete",
    );
    if spans.is_empty() {
        return Err(Fault::bad_input(
            "voice activity detection found no speech — nothing to transcribe",
        ));
    }

    let model_str = model.to_str().ok_or_else(|| {
        Fault::environment(format!("model path {} is not UTF-8", model.display()))
    })?;
    let n_threads = threads();
    tracing::info!(
        model = model_str,
        ?backend,
        n_threads,
        "loading whisper model"
    );

    on_step("transcribing (GPU-accelerated)");
    let t = Instant::now();
    let mut ctx_params = WhisperContextParameters::default();
    ctx_params.use_gpu(true);
    let ctx = WhisperContext::new_with_params(model_str, ctx_params)
        .map_err(|e| Fault::environment(format!("loading whisper model {model_str}: {e}")))?;
    tracing::info!(
        elapsed_ms = t.elapsed().as_millis() as u64,
        "whisper model loaded"
    );

    let mut state = ctx
        .create_state()
        .map_err(|e| Fault::environment(format!("initializing whisper: {e}")))?;

    let mut cues = Vec::new();
    let total = spans.len();
    for (i, &(start, end)) in spans.iter().enumerate() {
        let span_dur = (end - start) as f64 / WHISPER_SAMPLE_RATE as f64;
        tracing::info!(
            span = i + 1,
            total,
            duration_s = format_args!("{span_dur:.1}"),
            "recognizing span"
        );

        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        params.set_language(Some(cfg.whisper.language.as_deref().unwrap_or("auto")));
        // Spans are independent; conditioning on prior text re-invites the
        // repeat loops VAD exists to prevent. The initial prompt is not
        // that: `full` re-injects it at the start of every call, so each
        // span opens with the same vocabulary bias and inherits no
        // neighbor's decoded text.
        params.set_no_context(true);
        if !prompt.is_empty() {
            params.set_initial_prompt(prompt);
        }
        params.set_print_special(false);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);
        params.set_n_threads(n_threads);

        let t = Instant::now();
        state
            .full(params, &samples[start..end])
            .map_err(|e| Fault::internal(format!("whisper inference failed: {e}")))?;
        tracing::info!(
            span = i + 1,
            elapsed_ms = t.elapsed().as_millis() as u64,
            "span done"
        );

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

/// Sample-index spans to recognize: VAD speech segments when the VAD model
/// is installed, the whole track otherwise.
fn speech_spans(cfg: &Config, samples: &[f32]) -> Result<Vec<(usize, usize)>, Fault> {
    use whisper_rs::{WhisperVadContext, WhisperVadContextParams, WhisperVadParams};

    let vad_model = vad_model_path(cfg);
    let Some(vad_model) = vad_model.to_str().filter(|_| vad_model.is_file()) else {
        tracing::info!("no VAD model on disk, skipping VAD");
        return Ok(vec![(0, samples.len())]);
    };
    // Silero VAD is a tiny LSTM (~885 KB). whisper.cpp's own default is 4
    // threads; using more creates 55k barrier-sync rounds that stall on
    // large core counts.
    let n_threads = threads().min(4);
    tracing::info!(model = vad_model, n_threads, "loading VAD model");
    let t = Instant::now();
    let mut vad_ctx_params = WhisperVadContextParams::new();
    vad_ctx_params.set_n_threads(n_threads);
    let mut vad = WhisperVadContext::new(vad_model, vad_ctx_params)
        .map_err(|e| Fault::environment(format!("loading VAD model {vad_model}: {e}")))?;
    tracing::info!(
        elapsed_ms = t.elapsed().as_millis() as u64,
        "VAD model loaded"
    );

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

fn threads() -> i32 {
    std::thread::available_parallelism()
        .map(|n| n.get() as i32)
        .unwrap_or(4)
}
