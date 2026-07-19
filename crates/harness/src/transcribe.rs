// SPDX-License-Identifier: Apache-2.0
//! Whisper transcription behind a resolved [`Route`]: local GPU inference
//! when compiled and provisioned, an OpenAI-compatible API otherwise.
//!
//! **Local** (`whisper-local` feature + GPU backend): in-process
//! whisper.cpp via `whisper-rs`; Silero VAD trims silence and music before
//! recognition (the dominant hallucination source on real videos).
//! CPU-only local inference is not supported. Weights are provisioned by
//! `install-whisper` and never fetched here.
//!
//! **API**: audio is encoded as 16-bit WAV chunks and POSTed whole — no
//! VAD pre-filtering; the endpoint owns hallucination suppression. Chunks
//! split at hard byte boundaries (~13 min each), so a cue spanning a cut
//! may lose or repeat a word: accepted for the simplicity of stateless
//! uploads. Each chunk after the first is conditioned on the tail of the
//! previous chunk's transcript ([`continuation_prompt`]), so vocabulary,
//! casing, and script choice carry across the cut instead of lurching.
//!
//! Either way the transcript lands as an SRT whose cue times feed the grid
//! subtitle ride-along.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

#[cfg(feature = "whisper-local")]
use super::config::Backend;
use super::config::{Config, WhisperModel};
use crate::fault::Fault;
use agent_video_core::media::audio::{WHISPER_SAMPLE_RATE, decode_audio_16k_mono};
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
            Route::Api { .. } => format!("api-{API_MODEL}"),
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

// ── conditioning ──

/// Hard byte cap on the conditioning text handed to whisper. The model
/// conditions on at most ~224 tokens of prompt; 800 bytes of UTF-8 is
/// roughly 200 BPE tokens for ASCII text and ~260 codepoints for CJK —
/// under the budget either way. Anything longer would be silently cut by
/// the tokenizer while still bloating the upload and the cache key.
pub const PROMPT_MAX_BYTES: usize = 800;

/// The effective conditioning text both engines receive: the operator's
/// `whisper.prompt` wins; otherwise a known media `title` is shaped into
/// continuation-style text (a fragment the transcript could plausibly
/// follow — never an instruction, which whisper does not obey); otherwise
/// empty. NUL bytes are stripped (the local path crosses a `CString`
/// boundary) and the result is truncated to [`PROMPT_MAX_BYTES`] on a
/// char boundary — what returns from here is exactly what ships, so its
/// hash is the prompt's cache identity ([`transcript_id`]).
pub fn effective_prompt(cfg: &Config, title: Option<&str>) -> String {
    let text = match (&cfg.whisper.prompt, title) {
        (Some(prompt), _) => prompt.clone(),
        (None, Some(title)) => title_as_continuation(title),
        (None, None) => String::new(),
    };
    let mut text: String = text.chars().filter(|&c| c != '\0').collect();
    while text.len() > PROMPT_MAX_BYTES {
        text.pop();
    }
    text
}

/// Shape a title as text that reads like preceding transcript: a finished
/// sentence of its own, so the decoder treats it as context, not as the
/// opening words of the audio.
fn title_as_continuation(title: &str) -> String {
    let title = title.trim();
    if title.is_empty() {
        return String::new();
    }
    if title.ends_with(['.', '!', '?', '。', '！', '？']) {
        title.to_string()
    } else {
        format!("{title}.")
    }
}

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
    let t = Instant::now();
    let decoded = decode_audio_16k_mono(&source)?;
    let sample_count = decoded.as_deref().map_or(0, <[f32]>::len);
    tracing::info!(
        elapsed_ms = t.elapsed().as_millis() as u64,
        samples = sample_count,
        duration_s = sample_count as f64 / WHISPER_SAMPLE_RATE as f64,
        "audio decode complete",
    );
    // Both flavors of silence are the input's shape, not a decode failure:
    // a container with no audio track (screen capture, surveillance) and a
    // track that decoded to zero samples each get named for what they are.
    let samples = match decoded {
        Some(samples) if !samples.is_empty() => samples,
        Some(_) => {
            return Err(Fault::bad_input(format!(
                "{} decoded to zero audio samples — nothing to transcribe",
                file.display()
            )));
        }
        None => {
            return Err(Fault::bad_input(format!(
                "{} has no audio track — nothing to transcribe",
                file.display()
            )));
        }
    };

    let audio_seconds = samples.len() as f64 / f64::from(WHISPER_SAMPLE_RATE);
    let cues = match route {
        #[cfg(feature = "whisper-local")]
        Route::Local { backend, model, .. } => {
            recognize_local(cfg, backend, &model, &samples, prompt, on_step)?
        }
        Route::Api { key, url } => recognize_api(
            &key,
            &url,
            cfg.whisper.language.as_deref(),
            prompt,
            &samples,
            on_step,
        )?,
    };
    write_srt(out_srt, &cues)?;
    Ok(Transcribed { audio_seconds })
}

// ── API path ──

/// Maximum upload size per chunk (bytes). Groq / OpenAI limit is 25 MB.
const MAX_UPLOAD_BYTES: usize = 24 * 1024 * 1024; // 24 MiB with margin

/// API model name sent to the endpoint; also part of the transcript cache
/// identity ([`Route::cache_id`]).
const API_MODEL: &str = "whisper-large-v3-turbo";

/// Connection bound for the whisper API. TCP + TLS to a live endpoint
/// completes in seconds; only a dead host needs longer, and a dead host
/// must not pin the MCP call's `spawn_blocking` thread — transcription
/// runs under a single-flight gate, so one wedged upload would make every
/// later transcribe of the same content inherit the wait. Matches the
/// download client's bound (`crate::fetch::direct`).
const API_CONNECT_TIMEOUT: Duration = Duration::from_secs(15);

/// Whole-call ceiling per chunk: upload, server-side recognition, and the
/// JSON body read all live under this one deadline. Sized to separate
/// dead from slow, not to police slowness: a full [`MAX_UPLOAD_BYTES`]
/// chunk on a ~50 KiB/s uplink is ~8 minutes, and recognition of the
/// ~13 minutes of audio it carries can take minutes more on a loaded
/// endpoint — 20 minutes covers both with margin, while a hung endpoint
/// unwedges the gate within one ceiling instead of never. A `StallGuard`
/// idle bound (as in `crate::fetch::direct`) would be sharper, but it is
/// private to that module and this path has no cancel signal to feed it;
/// the constant pair keeps the fix local.
const API_CALL_CEILING: Duration = Duration::from_secs(20 * 60);

/// Agent for whisper API calls: explicit deadlines, nothing else.
/// Deliberately only `connect` + `global` — ureq's per-phase timeouts
/// misbehave across the body phase (see `crate::fetch::direct::agent`),
/// and `global` already spans the entire exchange, so the response body
/// read is covered by the same deadline as the upload. Status-as-error
/// is off because error statuses must surface the response body (the
/// endpoint's own diagnosis); classification is ours.
fn api_agent(connect_timeout: Duration, call_ceiling: Duration) -> ureq::Agent {
    ureq::Agent::config_builder()
        .timeout_connect(Some(connect_timeout))
        .timeout_global(Some(call_ceiling))
        .http_status_as_error(false)
        .build()
        .new_agent()
}

fn recognize_api(
    api_key: &str,
    api_url: &str,
    language: Option<&str>,
    prompt: &str,
    samples: &[f32],
    on_step: &mut dyn FnMut(&str),
) -> Result<Vec<Cue>, Fault> {
    let chunks = wav_chunks(samples, MAX_UPLOAD_BYTES);
    tracing::info!(total_chunks = chunks.len(), "uploading to API");

    let agent = api_agent(API_CONNECT_TIMEOUT, API_CALL_CEILING);
    recognize_chunked(chunks.len(), prompt, |i, chunk_prompt| {
        let (wav, offset_samples) = &chunks[i];
        on_step(&format!(
            "transcribing via API (chunk {}/{})",
            i + 1,
            chunks.len()
        ));
        let t = Instant::now();
        let cues = upload_and_parse(
            &agent,
            api_key,
            api_url,
            language,
            chunk_prompt,
            wav,
            *offset_samples,
        )?;
        tracing::info!(
            chunk = i + 1,
            elapsed_ms = t.elapsed().as_millis() as u64,
            cues = cues.len(),
            "chunk transcribed",
        );
        Ok(cues)
    })
}

/// Drive a multi-chunk recognition with continuation prompting.
/// `recognize` maps `(chunk_index, prompt)` to that chunk's cues; the
/// prompt is [`continuation_prompt`] over the transcript tail of the last
/// chunk that produced any cues. A silent chunk keeps the previous tail
/// alive — one stretch of silence must not wipe the chain and reset the
/// next chunk to the bare initial prompt.
///
/// Cache note: the tail derives from the previous chunk's *output*,
/// which an endpoint need not reproduce bit-for-bit — and that is fine
/// for the key. [`transcript_id`] over the initial prompt names the
/// recipe, not one run of it: the whole SRT is produced under that one
/// initial conditioning and published atomically, so whichever run lands
/// is a valid value for its key. No cache-key change is needed here.
fn recognize_chunked(
    chunk_count: usize,
    initial_prompt: &str,
    mut recognize: impl FnMut(usize, &str) -> Result<Vec<Cue>, Fault>,
) -> Result<Vec<Cue>, Fault> {
    let mut all_cues = Vec::new();
    let mut tail = String::new();
    for i in 0..chunk_count {
        let prompt = continuation_prompt(initial_prompt, &tail);
        let cues = recognize(i, &prompt)?;
        if !cues.is_empty() {
            tail = cues
                .iter()
                .map(|cue| cue.text.as_str())
                .collect::<Vec<_>>()
                .join(" ");
        }
        all_cues.extend(cues);
    }
    Ok(all_cues)
}

/// Conditioning text for one chunk: the initial prompt first (domain
/// vocabulary), the previous chunk's transcript tail last — nearest the
/// audio, where the decoder reads it as the immediately preceding text.
/// The pair is truncated from the *front* to [`PROMPT_MAX_BYTES`] on a
/// char boundary, keeping the most recent text: when the tail alone fills
/// the budget the initial prompt gives way, because for a mid-stream
/// chunk local continuity (casing, script choice, in-flight vocabulary)
/// beats global vocabulary — which the tail already carries implicitly,
/// being output that was itself conditioned on the initial prompt.
/// With an empty tail (first chunk, or every chunk of single-chunk audio)
/// this is the initial prompt verbatim.
fn continuation_prompt(initial: &str, tail: &str) -> String {
    let combined = match (initial.is_empty(), tail.is_empty()) {
        (_, true) => return initial.to_string(),
        (true, false) => tail.to_string(),
        (false, false) => format!("{initial} {tail}"),
    };
    let mut start = combined.len().saturating_sub(PROMPT_MAX_BYTES);
    while !combined.is_char_boundary(start) {
        start += 1;
    }
    combined[start..].to_string()
}

/// Split samples into WAV byte buffers, each under `max_bytes`.
/// Returns `(wav_bytes, offset_in_samples)` pairs. Cuts land at hard byte
/// boundaries, not silence — see the module doc for the trade-off.
fn wav_chunks(samples: &[f32], max_bytes: usize) -> Vec<(Vec<u8>, usize)> {
    // WAV header = 44 bytes; each sample = 2 bytes (i16).
    let max_samples = (max_bytes - 44) / 2;
    let mut chunks = Vec::new();
    let mut offset = 0;
    while offset < samples.len() {
        let end = (offset + max_samples).min(samples.len());
        let wav = encode_wav(&samples[offset..end]);
        chunks.push((wav, offset));
        offset = end;
    }
    chunks
}

/// Encode f32 samples as 16-bit PCM WAV.
fn encode_wav(samples: &[f32]) -> Vec<u8> {
    let num_samples = samples.len();
    let data_bytes = num_samples * 2;
    let file_size = 36 + data_bytes as u32;
    let mut buf = Vec::with_capacity(44 + data_bytes);

    // RIFF header
    buf.extend_from_slice(b"RIFF");
    buf.extend_from_slice(&file_size.to_le_bytes());
    buf.extend_from_slice(b"WAVE");

    // fmt chunk
    buf.extend_from_slice(b"fmt ");
    buf.extend_from_slice(&16u32.to_le_bytes()); // chunk size
    buf.extend_from_slice(&1u16.to_le_bytes()); // PCM
    buf.extend_from_slice(&1u16.to_le_bytes()); // mono
    buf.extend_from_slice(&WHISPER_SAMPLE_RATE.to_le_bytes()); // sample rate
    buf.extend_from_slice(&(WHISPER_SAMPLE_RATE * 2).to_le_bytes()); // byte rate
    buf.extend_from_slice(&2u16.to_le_bytes()); // block align
    buf.extend_from_slice(&16u16.to_le_bytes()); // bits per sample

    // data chunk
    buf.extend_from_slice(b"data");
    buf.extend_from_slice(&(data_bytes as u32).to_le_bytes());
    for &s in samples {
        let clamped = s.clamp(-1.0, 1.0);
        let i = (clamped * 32767.0) as i16;
        buf.extend_from_slice(&i.to_le_bytes());
    }

    buf
}

/// The multipart request body for one WAV chunk, plus its boundary.
/// Built manually — ureq 3.x has no built-in multipart. A boundary derived
/// from the payload's own hash cannot occur inside it short of a SHA-256
/// preimage, so no RNG is needed. `language` and `prompt` ride only when
/// configured: the OpenAI contract auto-detects when `language` is absent
/// ("auto" is not a valid ISO-639-1 value), and an empty prompt is no
/// prompt, not a field.
fn multipart_body(wav: &[u8], language: Option<&str>, prompt: &str) -> (Vec<u8>, String) {
    let boundary = {
        use sha2::{Digest, Sha256};
        let digest = Sha256::digest(wav);
        format!("----avwhisper{}", crate::identity::hex(&digest[..16]))
    };
    let mut body = Vec::new();
    write_multipart_field(
        &mut body,
        &boundary,
        "file",
        Some("audio.wav"),
        "application/octet-stream",
        wav,
    );
    write_multipart_text(&mut body, &boundary, "model", API_MODEL);
    write_multipart_text(&mut body, &boundary, "response_format", "verbose_json");
    if let Some(language) = language {
        write_multipart_text(&mut body, &boundary, "language", language);
    }
    if !prompt.is_empty() {
        write_multipart_text(&mut body, &boundary, "prompt", prompt);
    }
    body.extend_from_slice(format!("--{boundary}--\r\n").as_bytes());
    (body, boundary)
}

/// POST a WAV chunk to the API, parse the verbose-JSON response into cues.
/// The agent's deadlines ([`api_agent`]) bound the whole exchange —
/// including the [`read_to_string`](ureq::Body::read_to_string) below —
/// so a hung endpoint errors out instead of pinning the thread.
fn upload_and_parse(
    agent: &ureq::Agent,
    api_key: &str,
    api_url: &str,
    language: Option<&str>,
    prompt: &str,
    wav: &[u8],
    offset_samples: usize,
) -> Result<Vec<Cue>, Fault> {
    let (body, boundary) = multipart_body(wav, language, prompt);
    let content_type = format!("multipart/form-data; boundary={boundary}");

    let response = agent
        .post(api_url)
        .header("Authorization", &format!("Bearer {api_key}"))
        .header("Content-Type", &content_type)
        .send(&body)
        .map_err(|e| Fault::transient(format!("whisper API request failed: {e}")))?;

    let status = response.status().as_u16();
    let resp_body = response
        .into_body()
        .read_to_string()
        .map_err(|e| Fault::transient(format!("reading API response: {e}")))?;

    match status {
        200..=299 => parse_verbose_json(&resp_body, offset_samples),
        // Auth failures are the deployment's problem, not the request's:
        // retrying without a key change cannot succeed.
        401 | 403 => Err(Fault::environment(format!(
            "whisper API rejected the credentials ({status}): {resp_body} — check \
             WHISPER_API_KEY / whisper.api_key"
        ))),
        429 | 500..=599 => Err(Fault::transient(format!(
            "whisper API returned {status}: {resp_body}"
        ))),
        // Any other 4xx indicts the request we built — an agent-video bug.
        _ => Err(Fault::internal(format!(
            "whisper API returned {status}: {resp_body}"
        ))),
    }
}

fn write_multipart_field(
    body: &mut Vec<u8>,
    boundary: &str,
    name: &str,
    filename: Option<&str>,
    content_type: &str,
    data: &[u8],
) {
    body.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
    let disposition = match filename {
        Some(fname) => {
            format!("Content-Disposition: form-data; name=\"{name}\"; filename=\"{fname}\"\r\n")
        }
        None => format!("Content-Disposition: form-data; name=\"{name}\"\r\n"),
    };
    body.extend_from_slice(disposition.as_bytes());
    body.extend_from_slice(format!("Content-Type: {content_type}\r\n\r\n").as_bytes());
    body.extend_from_slice(data);
    body.extend_from_slice(b"\r\n");
}

fn write_multipart_text(body: &mut Vec<u8>, boundary: &str, name: &str, value: &str) {
    write_multipart_field(body, boundary, name, None, "text/plain", value.as_bytes());
}

/// Parse the OpenAI-compatible verbose_json response.
fn parse_verbose_json(body: &str, offset_samples: usize) -> Result<Vec<Cue>, Fault> {
    #[derive(serde::Deserialize)]
    struct Response {
        segments: Option<Vec<Segment>>,
    }
    #[derive(serde::Deserialize)]
    struct Segment {
        start: f64,
        end: f64,
        text: String,
    }

    let resp: Response = serde_json::from_str(body)
        .map_err(|e| Fault::internal(format!("parsing API response: {e}")))?;

    let offset_s = offset_samples as f64 / WHISPER_SAMPLE_RATE as f64;
    let cues = resp
        .segments
        .unwrap_or_default()
        .into_iter()
        .filter_map(|seg| {
            let text = seg.text.trim().to_string();
            if text.is_empty() {
                return None;
            }
            Some(Cue {
                start_cs: ((seg.start + offset_s) * 100.0).round() as i64,
                end_cs: ((seg.end + offset_s) * 100.0).round() as i64,
                text,
            })
        })
        .collect();
    Ok(cues)
}

// ── local GPU path (feature-gated) ──

/// Run VAD then whisper over each speech span, rebasing cue times to
/// absolute media time.
#[cfg(feature = "whisper-local")]
fn recognize_local(
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
#[cfg(feature = "whisper-local")]
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

#[cfg(feature = "whisper-local")]
fn threads() -> i32 {
    std::thread::available_parallelism()
        .map(|n| n.get() as i32)
        .unwrap_or(4)
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
        assert_eq!(id, format!("api-{API_MODEL}"));
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
    fn operator_prompt_overrides_the_derived_title() {
        let mut cfg = cfg();
        cfg.whisper.prompt = Some("Kubernetes, etcd, kubelet.".to_string());
        assert_eq!(
            effective_prompt(&cfg, Some("Unrelated Title")),
            "Kubernetes, etcd, kubelet.",
            "the operator's knob wins over the derived default"
        );
    }

    #[test]
    fn derives_a_continuation_prompt_from_the_title() {
        assert_eq!(
            effective_prompt(&cfg(), Some("  Rust Ownership Explained ")),
            "Rust Ownership Explained.",
            "the title becomes a finished sentence, not an instruction"
        );
        assert_eq!(
            effective_prompt(&cfg(), Some("什么是所有权？")),
            "什么是所有权？",
            "a title that already ends a sentence stays untouched"
        );
    }

    #[test]
    fn no_prompt_and_no_title_yields_empty_conditioning() {
        assert_eq!(effective_prompt(&cfg(), None), "");
        assert_eq!(effective_prompt(&cfg(), Some("   ")), "");
    }

    #[test]
    fn truncates_the_prompt_on_a_char_boundary() {
        let mut cfg = cfg();
        cfg.whisper.prompt = Some("语".repeat(PROMPT_MAX_BYTES)); // 3 bytes per char
        let prompt = effective_prompt(&cfg, None);
        assert!(prompt.len() <= PROMPT_MAX_BYTES, "{}", prompt.len());
        assert!(!prompt.is_empty());
        assert!(prompt.chars().all(|c| c == '语'), "prefix survives intact");
    }

    #[test]
    fn strips_nul_bytes_before_the_cstring_boundary() {
        let mut cfg = cfg();
        cfg.whisper.prompt = Some("a\0b".to_string());
        assert_eq!(effective_prompt(&cfg, None), "ab");
    }

    #[test]
    fn multipart_body_carries_language_and_prompt_when_set() {
        let (body, _) = multipart_body(b"RIFF", Some("zh"), "Rust Ownership.");
        let text = String::from_utf8_lossy(&body);
        assert!(text.contains("name=\"language\""), "{text}");
        assert!(text.contains("zh"), "{text}");
        assert!(text.contains("name=\"prompt\""), "{text}");
        assert!(text.contains("Rust Ownership."), "{text}");
    }

    #[test]
    fn multipart_body_omits_unset_conditioning() {
        let (body, _) = multipart_body(b"RIFF", None, "");
        let text = String::from_utf8_lossy(&body);
        assert!(
            !text.contains("name=\"language\""),
            "absent language means auto-detect, never a fake value: {text}"
        );
        assert!(!text.contains("name=\"prompt\""), "{text}");
        assert!(text.contains("name=\"model\""), "{text}");
    }

    #[test]
    fn hostile_title_cannot_break_multipart_framing() {
        // A title (and thus a derived prompt) is attacker-influenced text.
        // The worst case ships CRLF framing, a boundary lookalike, and a
        // forged field header — all of which must land inert inside the
        // prompt field's value. This pins the by-construction safety (the
        // boundary is the WAV payload's SHA-256, unguessable from the
        // prompt) against a refactor to predictable boundaries.
        let wav = b"RIFFfake wav payload";
        let hostile = "benign words\r\n--boundary\r\n\
                       Content-Disposition: form-data; name=\"model\"\r\n\r\n\
                       injected-model\r\n";
        let (body, boundary) = multipart_body(wav, None, hostile);
        assert!(
            !hostile.contains(&boundary),
            "the crafted text cannot contain the payload-hash boundary"
        );

        // Parse the body the way a server does: split on the real
        // delimiter, then split each part into header block and value.
        let text = String::from_utf8_lossy(&body).into_owned();
        let delim = format!("--{boundary}");
        let parts: Vec<&str> = text.split(delim.as_str()).collect();
        assert_eq!(
            parts.first(),
            Some(&""),
            "no preamble before the first boundary"
        );
        assert_eq!(parts.last(), Some(&"--\r\n"), "closing delimiter intact");
        let fields: Vec<(&str, &str)> = parts[1..parts.len() - 1]
            .iter()
            .map(|part| {
                let part = part.strip_prefix("\r\n").expect("CRLF after delimiter");
                let (headers, value) = part.split_once("\r\n\r\n").expect("header/value split");
                let name = headers
                    .split("name=\"")
                    .nth(1)
                    .and_then(|rest| rest.split('"').next())
                    .expect("a field name in the part header");
                (
                    name,
                    value
                        .strip_suffix("\r\n")
                        .expect("CRLF before next delimiter"),
                )
            })
            .collect();

        let names: Vec<&str> = fields.iter().map(|(n, _)| *n).collect();
        assert_eq!(
            names,
            ["file", "model", "response_format", "prompt"],
            "exactly the expected fields, none forged into existence"
        );
        let model = fields
            .iter()
            .find(|(n, _)| *n == "model")
            .expect("model field");
        assert_eq!(model.1, API_MODEL, "the forged model line did not take");
        let prompt = fields
            .iter()
            .find(|(n, _)| *n == "prompt")
            .expect("prompt field");
        assert_eq!(
            prompt.1, hostile,
            "the hostile text lands whole inside the prompt value"
        );
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

    #[test]
    fn encodes_valid_wav_header() {
        let samples = vec![0.0f32; 16000]; // 1 second
        let wav = encode_wav(&samples);
        assert_eq!(&wav[0..4], b"RIFF");
        assert_eq!(&wav[8..12], b"WAVE");
        assert_eq!(&wav[12..16], b"fmt ");
        assert_eq!(&wav[36..40], b"data");
        // data size = 16000 * 2 = 32000
        assert_eq!(
            u32::from_le_bytes(wav[40..44].try_into().expect("4 bytes")),
            32000
        );
    }

    #[test]
    fn chunks_large_audio_into_bounded_wavs() {
        // 2 minutes at 16kHz = 1,920,000 samples = 3.84 MB as WAV
        let samples = vec![0.0f32; 1_920_000];
        let chunks = wav_chunks(&samples, 2 * 1024 * 1024); // 2 MB limit
        assert!(chunks.len() >= 2, "should split into multiple chunks");
        for (wav, _) in &chunks {
            assert!(wav.len() <= 2 * 1024 * 1024, "each chunk respects limit");
        }
    }

    #[test]
    fn second_chunk_carries_the_first_chunks_tail() {
        let mut prompts = Vec::new();
        let cues = recognize_chunked(2, "Rust Ownership.", |i, prompt| {
            prompts.push(prompt.to_string());
            Ok(vec![
                Cue {
                    start_cs: 0,
                    end_cs: 100,
                    text: format!("chunk {i} opens"),
                },
                Cue {
                    start_cs: 100,
                    end_cs: 200,
                    text: format!("chunk {i} closes"),
                },
            ])
        })
        .expect("both chunks succeed");
        assert_eq!(prompts[0], "Rust Ownership.", "first chunk: initial only");
        assert_eq!(
            prompts[1], "Rust Ownership. chunk 0 opens chunk 0 closes",
            "initial first, previous transcript last (nearest the audio)"
        );
        assert_eq!(cues.len(), 4, "all cues survive in order");
    }

    #[test]
    fn silent_chunk_does_not_erase_the_prompt_chain() {
        let mut prompts = Vec::new();
        recognize_chunked(3, "", |i, prompt| {
            prompts.push(prompt.to_string());
            Ok(if i == 1 {
                Vec::new() // silence: the endpoint found nothing to say
            } else {
                vec![Cue {
                    start_cs: 0,
                    end_cs: 100,
                    text: format!("speech {i}"),
                }]
            })
        })
        .expect("all chunks succeed");
        assert_eq!(prompts[1], "speech 0");
        assert_eq!(
            prompts[2], "speech 0",
            "a silent chunk keeps the last non-empty tail alive"
        );
    }

    #[test]
    fn single_chunk_gets_the_initial_prompt_verbatim() {
        let mut prompts = Vec::new();
        recognize_chunked(1, "Rust Ownership.", |_, prompt| {
            prompts.push(prompt.to_string());
            Ok(Vec::new())
        })
        .expect("the chunk succeeds");
        assert_eq!(
            prompts,
            ["Rust Ownership."],
            "no tail machinery is visible on single-chunk audio"
        );
    }

    #[test]
    fn truncates_the_tail_from_the_front_on_char_boundaries() {
        // 3 bytes per char: the byte cut cannot land mid-codepoint.
        let old = "旧".repeat(400);
        let recent = "新".repeat(100);
        let prompt = continuation_prompt("", &format!("{old} {recent}"));
        assert!(prompt.len() <= PROMPT_MAX_BYTES, "{}", prompt.len());
        assert!(
            prompt.ends_with(&recent),
            "the most recent text survives; the oldest falls off the front"
        );
        assert!(
            prompt.starts_with('旧'),
            "the cut lands on a char boundary: {:?}",
            &prompt[..4]
        );
    }

    #[test]
    fn overflowing_tail_evicts_the_initial_prompt() {
        let tail = "语".repeat(PROMPT_MAX_BYTES); // 3× the budget in bytes
        let prompt = continuation_prompt("Rust Ownership.", &tail);
        assert!(prompt.len() <= PROMPT_MAX_BYTES, "{}", prompt.len());
        assert!(
            !prompt.contains("Rust"),
            "mid-stream, local continuity outranks global vocabulary"
        );
        assert!(prompt.chars().all(|c| c == '语'));
    }

    #[test]
    fn combined_prompt_respects_the_byte_cap() {
        let initial = "a".repeat(PROMPT_MAX_BYTES); // already at the cap
        let prompt = continuation_prompt(&initial, "tail text");
        assert!(prompt.len() <= PROMPT_MAX_BYTES, "{}", prompt.len());
        assert!(prompt.ends_with("tail text"), "the tail stays whole");
    }

    #[test]
    fn hung_api_endpoint_errors_within_the_call_ceiling() {
        // A server that accepts, swallows the request, and never answers —
        // the wedged endpoint that used to pin the spawn_blocking thread
        // (and, through the single-flight gate, every later transcribe of
        // the same content). The injected ceiling stands in for
        // API_CALL_CEILING so the test runs in milliseconds.
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind");
        let addr = listener.local_addr().expect("addr");
        std::thread::spawn(move || {
            if let Ok((mut s, _)) = listener.accept() {
                let _ = std::io::Read::read(&mut s, &mut [0u8; 64 * 1024]);
                std::thread::sleep(std::time::Duration::from_secs(20));
            }
        });
        let agent = api_agent(
            std::time::Duration::from_secs(5),
            std::time::Duration::from_millis(300),
        );
        let started = Instant::now();

        let err = upload_and_parse(
            &agent,
            "sk-test",
            &format!("http://{addr}/v1"),
            None,
            "",
            b"RIFF",
            0,
        )
        .expect_err("a silent endpoint must not pin the call");

        assert!(
            started.elapsed() < std::time::Duration::from_secs(5),
            "the call must end at the ceiling, not hang: {:?}",
            started.elapsed()
        );
        assert_eq!(err.kind, crate::fault::FaultKind::Transient);
        assert!(err.message.contains("whisper API"), "{err}");
    }

    #[test]
    fn parses_verbose_json_segments() {
        let body = r#"{"segments":[
            {"id":0,"start":0.0,"end":2.5,"text":" Hello world"},
            {"id":1,"start":3.0,"end":5.0,"text":" Goodbye"}
        ]}"#;
        let cues = parse_verbose_json(body, 0).expect("valid json");
        assert_eq!(cues.len(), 2);
        assert_eq!(cues[0].text, "Hello world");
        assert_eq!(cues[0].start_cs, 0);
        assert_eq!(cues[0].end_cs, 250);
        assert_eq!(cues[1].start_cs, 300);
    }

    #[test]
    fn parses_verbose_json_with_sample_offset() {
        let body = r#"{"segments":[{"id":0,"start":1.0,"end":2.0,"text":"hi"}]}"#;
        // offset = 160000 samples = 10 seconds
        let cues = parse_verbose_json(body, 160_000).expect("valid json");
        assert_eq!(
            cues[0].start_cs, 1100,
            "1.0s + 10.0s offset = 11.0s = 1100cs"
        );
    }
}
