// SPDX-License-Identifier: Apache-2.0
//! Local recognition: Silero VAD carves the track into speech spans,
//! SenseVoice recognizes each one, and the recognizer's own token
//! timestamps cut a span into cues.
//!
//! Everything here runs on the CPU through statically linked onnxruntime;
//! weights come from `install-asr` and are never fetched at run time.
//!
//! The module owns two policy numbers ([`MAX_CUE_SECONDS`],
//! [`CUE_GAP_SECONDS`]) and nothing else: span detection belongs to the
//! VAD, text to the recognizer, and SRT serialization to the parent.

use std::path::Path;

use sherpa_onnx::{
    OfflineRecognizer, OfflineRecognizerConfig, OfflineSenseVoiceModelConfig, SileroVadModelConfig,
    VadModelConfig, VoiceActivityDetector,
};

use super::Cue;
use crate::fault::Fault;
use agent_video_core::media::audio::WHISPER_SAMPLE_RATE;

/// Samples handed to the VAD at a time. Silero v5 consumes exactly 512
/// samples (32 ms at 16 kHz) per inference; feeding any other window
/// makes it pad internally and drift.
const VAD_WINDOW: usize = 512;

/// Seconds of audio the VAD may buffer before a caller must drain it.
/// Bounds resident memory independently of track length: spans are popped
/// as they close, so this only has to outlast the longest single span.
const VAD_BUFFER_SECONDS: f32 = 60.0;

/// Longest cue this module will emit. A VAD span averages several
/// seconds; past this the line stops being a subtitle and becomes a
/// paragraph a reader cannot align with the picture.
const MAX_CUE_SECONDS: f32 = 6.0;

/// Silence between two recognized tokens that ends a cue. Below the
/// threshold the pause is phrasing; above it, it is a sentence boundary
/// the speaker themselves marked.
const CUE_GAP_SECONDS: f32 = 0.6;

/// Weight files the local route needs, checked to exist before the route
/// resolves so a missing file is an install hint, not a decode failure.
#[derive(Debug)]
pub struct Weights {
    pub model: std::path::PathBuf,
    pub tokens: std::path::PathBuf,
    pub vad: std::path::PathBuf,
}

/// Recognize `samples` (16 kHz mono f32) into cues with absolute
/// centisecond timestamps.
///
/// `language` is a two-letter hint or `None` for SenseVoice's own
/// detection; the recognizer accepts only the languages it was trained on
/// (zh, en, ja, ko, yue) and falls back to detection for anything else.
pub fn recognize(
    weights: &Weights,
    samples: &[f32],
    language: Option<&str>,
    on_step: &mut dyn FnMut(&str),
) -> Result<Vec<Cue>, Fault> {
    on_step("loading recognizer");
    let recognizer = build_recognizer(weights, language)?;
    let vad = build_vad(&weights.vad)?;

    on_step("detecting speech");
    let mut cues = Vec::new();
    let mut chunks = samples.chunks(VAD_WINDOW).peekable();
    while let Some(chunk) = chunks.next() {
        // Silero rejects a short tail window; the flush below reclaims
        // whatever speech it holds.
        if chunk.len() == VAD_WINDOW {
            vad.accept_waveform(chunk);
        }
        if chunks.peek().is_none() {
            vad.flush();
        }
        drain(&vad, &recognizer, &mut cues);
    }
    // A track shorter than one window never entered the loop body's flush.
    vad.flush();
    drain(&vad, &recognizer, &mut cues);

    tracing::info!(
        cues = cues.len(),
        duration_s = samples.len() as f64 / f64::from(WHISPER_SAMPLE_RATE),
        "local recognition complete",
    );
    Ok(cues)
}

/// Pop every closed span and append its cues.
fn drain(vad: &VoiceActivityDetector, recognizer: &OfflineRecognizer, cues: &mut Vec<Cue>) {
    while !vad.is_empty() {
        let Some(segment) = vad.front() else {
            break;
        };
        vad.pop();
        let offset = segment.start() as f32 / WHISPER_SAMPLE_RATE as f32;
        let stream = recognizer.create_stream();
        stream.accept_waveform(WHISPER_SAMPLE_RATE as i32, segment.samples());
        recognizer.decode(&stream);
        let Some(result) = stream.get_result() else {
            continue;
        };
        let span_end = offset + segment.n() as f32 / WHISPER_SAMPLE_RATE as f32;
        cues.extend(split(
            &result.tokens,
            result.timestamps.as_deref(),
            offset,
            span_end,
        ));
    }
}

/// Cut one span's tokens into cues at pauses and at [`MAX_CUE_SECONDS`].
///
/// Without timestamps the span can only answer as a single cue — that is
/// the honest shape of the information, not a degraded one.
fn split(tokens: &[String], timestamps: Option<&[f32]>, offset: f32, span_end: f32) -> Vec<Cue> {
    let Some(times) = timestamps.filter(|t| t.len() == tokens.len()) else {
        let text = join(tokens.iter().filter(|t| !is_control(t)).map(String::as_str));
        return one_cue(text, offset, span_end).into_iter().collect();
    };

    let mut cues = Vec::new();
    let mut pending: Vec<&str> = Vec::new();
    let mut start = 0.0f32;
    let mut previous = 0.0f32;

    for (token, &at) in tokens.iter().zip(times) {
        if is_control(token) {
            continue;
        }
        // A cue may only open on speech: punctuation belongs to the
        // phrase it closes, and a line holding nothing else is a line
        // the reader has to skip.
        let breaks = !pending.is_empty()
            && carries_speech(token)
            && (at - previous >= CUE_GAP_SECONDS || at - start >= MAX_CUE_SECONDS);
        if breaks {
            cues.extend(one_cue(
                join(pending.drain(..)),
                offset + start,
                offset + previous,
            ));
            start = at;
        }
        if pending.is_empty() {
            start = at;
        }
        pending.push(token);
        previous = at;
    }
    cues.extend(one_cue(join(pending.drain(..)), offset + start, span_end));
    cues
}

/// A cue, unless the text is blank — an empty subtitle line is noise the
/// reader has to skip, and a cue with no text has nothing to say.
fn one_cue(text: String, start: f32, end: f32) -> Option<Cue> {
    let text = text.trim().to_string();
    if text.is_empty() {
        return None;
    }
    Some(Cue {
        start_cs: (start * 100.0).round() as i64,
        end_cs: (end.max(start) * 100.0).round() as i64,
        text,
    })
}

/// SenseVoice emits SentencePiece pieces: `▁` marks a word boundary in
/// space-delimited scripts, and CJK characters arrive bare and must not
/// be spaced apart.
fn join<'a>(tokens: impl IntoIterator<Item = &'a str>) -> String {
    let mut out = String::new();
    for token in tokens {
        match token.strip_prefix('\u{2581}') {
            Some(word) => {
                if !out.is_empty() {
                    out.push(' ');
                }
                out.push_str(word);
            }
            None => out.push_str(token),
        }
    }
    out
}

/// Whether a token is one of SenseVoice's control tags (`<|zh|>`,
/// `<|NEUTRAL|>`, `<|Speech|>`, `<|woitn|>` …) rather than transcript.
fn is_control(token: &str) -> bool {
    token.starts_with("<|") && token.ends_with("|>")
}

/// Whether a token says anything on its own. Pure punctuation does not:
/// it modifies the token before it.
fn carries_speech(token: &str) -> bool {
    token.chars().any(char::is_alphanumeric)
}

fn build_recognizer(weights: &Weights, language: Option<&str>) -> Result<OfflineRecognizer, Fault> {
    let mut config = OfflineRecognizerConfig::default();
    config.model_config.sense_voice = OfflineSenseVoiceModelConfig {
        model: Some(path_arg(&weights.model)?),
        language: Some(language.unwrap_or("auto").to_string()),
        // Inverse text normalization: "二零二五" becomes "2025". The
        // transcript is read by a model that reasons over numbers.
        use_itn: true,
    };
    config.model_config.tokens = Some(path_arg(&weights.tokens)?);
    config.model_config.num_threads = threads();
    OfflineRecognizer::create(&config).ok_or_else(|| {
        Fault::environment(format!(
            "cannot load the recognizer at {} — re-run `agent-video install-asr` to repair it",
            weights.model.display()
        ))
    })
}

fn build_vad(model: &Path) -> Result<VoiceActivityDetector, Fault> {
    let config = VadModelConfig {
        silero_vad: SileroVadModelConfig {
            model: Some(path_arg(model)?),
            window_size: VAD_WINDOW as i32,
            ..SileroVadModelConfig::default()
        },
        sample_rate: WHISPER_SAMPLE_RATE as i32,
        num_threads: threads(),
        ..VadModelConfig::default()
    };
    VoiceActivityDetector::create(&config, VAD_BUFFER_SECONDS).ok_or_else(|| {
        Fault::environment(format!(
            "cannot load the VAD model at {} — re-run `agent-video install-asr` to repair it",
            model.display()
        ))
    })
}

/// The C API takes paths as UTF-8 C strings, so a path that is not UTF-8
/// is refused here by name rather than truncated on the way in.
fn path_arg(path: &Path) -> Result<String, Fault> {
    path.to_str().map(str::to_string).ok_or_else(|| {
        Fault::environment(format!(
            "{} is not valid UTF-8; move the models directory somewhere it is",
            path.display()
        ))
    })
}

/// Recognition threads, capped: past a handful of cores the ONNX graphs
/// gain nothing and start contending with the decoder feeding them.
fn threads() -> i32 {
    std::thread::available_parallelism()
        .map(|n| n.get().min(8) as i32)
        .unwrap_or(2)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tokens(words: &[&str]) -> Vec<String> {
        words.iter().map(|w| w.to_string()).collect()
    }

    #[test]
    fn joins_cjk_without_spaces() {
        assert_eq!(join(["从", "前", "有"]), "从前有");
    }

    #[test]
    fn joins_word_pieces_on_the_boundary_marker() {
        assert_eq!(join(["▁the", "▁quick", "est"]), "the quickest");
    }

    #[test]
    fn drops_control_tags_from_cue_text() {
        let cues = split(
            &tokens(&["<|en|>", "<|NEUTRAL|>", "▁hello"]),
            Some(&[0.0, 0.0, 0.1]),
            0.0,
            1.0,
        );
        assert_eq!(cues.len(), 1);
        assert_eq!(cues[0].text, "hello");
    }

    #[test]
    fn splits_a_span_at_a_pause() {
        let cues = split(
            &tokens(&["▁one", "▁two", "▁three"]),
            Some(&[0.0, 0.1, 2.0]),
            0.0,
            3.0,
        );
        assert_eq!(cues.len(), 2, "the 1.9s gap ends a cue: {cues:?}");
        assert_eq!(cues[0].text, "one two");
        assert_eq!(cues[1].text, "three");
    }

    #[test]
    fn splits_a_span_that_runs_past_the_cue_ceiling() {
        let times: Vec<f32> = (0..20).map(|i| i as f32 * 0.5).collect();
        let words = tokens(&["▁w"; 20]);
        let cues = split(&words, Some(&times), 0.0, 10.0);
        assert!(cues.len() > 1, "a 10s span cannot be one cue: {cues:?}");
        for cue in &cues {
            let seconds = (cue.end_cs - cue.start_cs) as f32 / 100.0;
            assert!(
                seconds <= MAX_CUE_SECONDS + 0.5,
                "{cue:?} exceeds the ceiling"
            );
        }
    }

    #[test]
    fn offsets_cues_to_absolute_media_time() {
        let cues = split(&tokens(&["▁hi"]), Some(&[0.25]), 10.0, 11.0);
        assert_eq!(cues[0].start_cs, 1025, "span offset plus token time");
        assert_eq!(cues[0].end_cs, 1100, "the span's own end closes the cue");
    }

    #[test]
    fn a_span_without_timestamps_stays_one_cue() {
        let cues = split(&tokens(&["▁hello", "▁world"]), None, 4.0, 6.0);
        assert_eq!(cues.len(), 1);
        assert_eq!(cues[0].text, "hello world");
        assert_eq!((cues[0].start_cs, cues[0].end_cs), (400, 600));
    }

    #[test]
    fn drops_a_span_that_recognized_nothing() {
        assert!(split(&tokens(&["<|zh|>"]), Some(&[0.0]), 0.0, 1.0).is_empty());
    }

    #[test]
    fn keeps_trailing_punctuation_with_the_phrase_it_closes() {
        // The full stop lands well after the pause threshold, but a line
        // holding only "。" is not a subtitle.
        let cues = split(
            &tokens(&["好", "吧", "。"]),
            Some(&[0.0, 0.1, 3.0]),
            0.0,
            4.0,
        );
        assert_eq!(cues.len(), 1, "punctuation cannot open a cue: {cues:?}");
        assert_eq!(cues[0].text, "好吧。");
    }
}
