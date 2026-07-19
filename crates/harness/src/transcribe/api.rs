// SPDX-License-Identifier: Apache-2.0
//! The OpenAI-compatible API route: pull bounded sample chunks, encode
//! each as WAV, POST it multipart, and parse verbose-JSON segments into
//! cues. Policy lives here — budgets, deadlines, fault classes, the
//! whisper form layout; the WAV and multipart codecs are [`super::wav`]
//! and [`super::multipart`].

use std::time::{Duration, Instant};

use super::prompt::PromptChain;
use super::{Cue, multipart, wav};
use crate::fault::Fault;
use agent_video_core::media::audio::WHISPER_SAMPLE_RATE;

/// Maximum upload size per chunk (bytes). Groq / OpenAI limit is 25 MB.
const MAX_UPLOAD_BYTES: usize = 24 * 1024 * 1024; // 24 MiB with margin

/// API model name sent to the endpoint; also part of the transcript cache
/// identity ([`super::Route::cache_id`]).
pub(super) const API_MODEL: &str = "whisper-large-v3-turbo";

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

/// Recognize a whole track by streaming it chunk-by-chunk to the API,
/// carrying the conditioning chain ([`PromptChain`]) across cuts.
pub(super) fn recognize<F>(
    api_key: &str,
    api_url: &str,
    language: Option<&str>,
    initial_prompt: &str,
    chunks: &mut Chunker<F>,
    on_step: &mut dyn FnMut(&str),
) -> Result<Vec<Cue>, Fault>
where
    F: FnMut(&mut Vec<f32>, usize) -> Result<bool, Fault>,
{
    let agent = api_agent(API_CONNECT_TIMEOUT, API_CALL_CEILING);
    let mut chain = PromptChain::new(initial_prompt);
    let mut all_cues = Vec::new();
    let mut n = 0usize;
    while let Some(chunk) = chunks.next_chunk()? {
        n += 1;
        on_step(&format!("transcribing via API (chunk {n})"));
        let wav = wav::encode(&chunk.samples, WHISPER_SAMPLE_RATE);
        let offset = chunk.offset;
        // The chunk's f32 samples are spent once encoded; only the WAV
        // (half their size) rides through the upload.
        drop(chunk);
        let t = Instant::now();
        let cues = upload_and_parse(
            &agent,
            api_key,
            api_url,
            language,
            &chain.prompt(),
            &wav,
            offset,
        )?;
        tracing::info!(
            chunk = n,
            elapsed_ms = t.elapsed().as_millis() as u64,
            cues = cues.len(),
            "chunk transcribed",
        );
        chain.absorb(&cues);
        all_cues.extend(cues);
    }
    tracing::info!(chunks = n, "API transcription complete");
    Ok(all_cues)
}

/// Maximum samples per upload chunk: the byte budget minus the WAV
/// header, at [`wav::BYTES_PER_SAMPLE`] each.
const MAX_CHUNK_SAMPLES: usize = (MAX_UPLOAD_BYTES - wav::HEADER_BYTES) / wav::BYTES_PER_SAMPLE;

// The bound and the budget must agree: a full chunk's WAV fits one upload.
const _: () =
    assert!(wav::HEADER_BYTES + MAX_CHUNK_SAMPLES * wav::BYTES_PER_SAMPLE <= MAX_UPLOAD_BYTES);

/// One bounded run of samples with its absolute position in the track.
pub(super) struct Chunk {
    pub(super) samples: Vec<f32>,
    /// Offset of `samples[0]` from the track start, in samples.
    pub(super) offset: usize,
}

/// Pull bounded chunks off a sample stream: only the chunk in flight (plus
/// the stream's delivery overshoot) is ever resident, which is what keeps
/// a 3-hour track from materializing. `fill` follows
/// [`AudioSamples::fill`](agent_video_core::media::audio::AudioSamples::fill)'s
/// contract: append until the threshold or track end; `false` ("nothing
/// more will arrive") is final. Cuts land at hard sample boundaries, not
/// silence — see the module doc of [`super`] for the trade-off.
pub(super) struct Chunker<F> {
    fill: F,
    max_samples: usize,
    buf: Vec<f32>,
    consumed: usize,
}

impl<F: FnMut(&mut Vec<f32>, usize) -> Result<bool, Fault>> Chunker<F> {
    /// A chunker bounded to the upload budget — the only bound production
    /// code can construct.
    pub(super) fn new(fill: F) -> Self {
        Chunker::with_bound(fill, MAX_CHUNK_SAMPLES)
    }

    fn with_bound(fill: F, max_samples: usize) -> Self {
        Chunker {
            fill,
            max_samples,
            buf: Vec::new(),
            consumed: 0,
        }
    }

    /// The next chunk, or `None` once the stream is exhausted.
    pub(super) fn next_chunk(&mut self) -> Result<Option<Chunk>, Fault> {
        if self.buf.len() < self.max_samples {
            (self.fill)(&mut self.buf, self.max_samples)?;
        }
        if self.buf.is_empty() {
            return Ok(None);
        }
        let take = self.buf.len().min(self.max_samples);
        let rest = self.buf.split_off(take);
        let samples = std::mem::replace(&mut self.buf, rest);
        let offset = self.consumed;
        self.consumed += samples.len();
        Ok(Some(Chunk { samples, offset }))
    }

    /// Total samples handed out so far; after exhaustion, the track's
    /// decoded length.
    pub(super) fn samples_pulled(&self) -> usize {
        self.consumed
    }
}

/// The multipart form for one WAV chunk. The boundary is keyed to the WAV
/// payload ([`multipart::Form::keyed_to`]), so attacker-influenced text
/// (a derived title riding in `prompt`) cannot break the framing.
/// `language` and `prompt` ride only when configured: the OpenAI contract
/// auto-detects when `language` is absent ("auto" is not a valid
/// ISO-639-1 value), and an empty prompt is no prompt, not a field.
fn whisper_form(wav: &[u8], language: Option<&str>, prompt: &str) -> multipart::EncodedForm {
    let mut form = multipart::Form::keyed_to(wav);
    form.file("file", "audio.wav", "application/octet-stream", wav);
    form.text("model", API_MODEL);
    form.text("response_format", "verbose_json");
    if let Some(language) = language {
        form.text("language", language);
    }
    if !prompt.is_empty() {
        form.text("prompt", prompt);
    }
    form.finish()
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
    let form = whisper_form(wav, language, prompt);

    let response = agent
        .post(api_url)
        .header("Authorization", &format!("Bearer {api_key}"))
        .header("Content-Type", &form.content_type())
        .send(&form.body)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn whisper_form_carries_language_and_prompt_when_set() {
        let form = whisper_form(b"RIFF", Some("zh"), "Rust Ownership.");
        let text = String::from_utf8_lossy(&form.body);
        assert!(text.contains("name=\"language\""), "{text}");
        assert!(text.contains("zh"), "{text}");
        assert!(text.contains("name=\"prompt\""), "{text}");
        assert!(text.contains("Rust Ownership."), "{text}");
    }

    #[test]
    fn whisper_form_omits_unset_conditioning() {
        let form = whisper_form(b"RIFF", None, "");
        let text = String::from_utf8_lossy(&form.body);
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
        let form = whisper_form(wav, None, hostile);
        assert!(
            !hostile.contains(&form.boundary),
            "the crafted text cannot contain the payload-hash boundary"
        );

        // Parse the body the way a server does: split on the real
        // delimiter, then split each part into header block and value.
        let text = String::from_utf8_lossy(&form.body).into_owned();
        let delim = format!("--{}", form.boundary);
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

    /// A fake sample stream honoring the fill contract: appends in
    /// `granule`-sized slabs (the delivery overshoot) until the threshold
    /// or the source ends; exhaustion is final.
    fn slab_fill(
        source: Vec<f32>,
        granule: usize,
    ) -> impl FnMut(&mut Vec<f32>, usize) -> Result<bool, Fault> {
        let mut pos = 0;
        move |out, min| {
            while out.len() < min && pos < source.len() {
                let end = (pos + granule).min(source.len());
                out.extend_from_slice(&source[pos..end]);
                pos = end;
            }
            Ok(pos < source.len())
        }
    }

    #[test]
    fn chunker_bounds_chunks_and_keeps_offsets_contiguous() {
        let source: Vec<f32> = (0..1000).map(|i| i as f32).collect();
        // A granule that divides nothing evenly: overshoot must carry over.
        let mut chunks = Chunker::with_bound(slab_fill(source.clone(), 7), 100);
        let mut rebuilt = Vec::new();
        while let Some(chunk) = chunks.next_chunk().expect("fake stream cannot fail") {
            assert!(chunk.samples.len() <= 100, "chunk respects the bound");
            assert_eq!(chunk.offset, rebuilt.len(), "offsets are contiguous");
            rebuilt.extend(chunk.samples);
        }
        assert_eq!(rebuilt, source, "no sample lost or repeated across cuts");
        assert_eq!(chunks.samples_pulled(), 1000);
        assert!(
            chunks.next_chunk().expect("still no failure").is_none(),
            "exhaustion is final"
        );
    }

    #[test]
    fn chunker_yields_nothing_for_an_empty_stream() {
        let mut chunks = Chunker::with_bound(slab_fill(Vec::new(), 7), 100);
        assert!(
            chunks
                .next_chunk()
                .expect("empty is not an error")
                .is_none()
        );
        assert_eq!(chunks.samples_pulled(), 0);
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
