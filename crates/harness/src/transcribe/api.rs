// SPDX-License-Identifier: Apache-2.0
//! The OpenAI-compatible API route: pull bounded sample chunks, encode
//! each as WAV, POST it multipart, and parse verbose-JSON segments into
//! cues. Policy lives here — budgets, deadlines, fault classes, the
//! whisper form layout; the WAV and multipart codecs are [`super::wav`]
//! and [`super::multipart`].

use std::num::NonZeroUsize;
use std::ops::Range;
use std::time::{Duration, Instant};

use super::prompt::PromptChain;
use super::{Cue, multipart, wav};
use crate::fault::Fault;
use agent_video_core::media::audio::{AudioChunk, SamplePos, WHISPER_SAMPLE_RATE};

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
    next: &mut F,
    on_step: &mut dyn FnMut(&str),
) -> Result<Vec<Cue>, Fault>
where
    F: FnMut(NonZeroUsize) -> Result<Option<AudioChunk>, Fault>,
{
    let agent = api_agent(API_CONNECT_TIMEOUT, API_CALL_CEILING);
    let mut chain = PromptChain::new(initial_prompt);
    let mut all_cues = Vec::new();
    let mut n = 0usize;
    let mut pending = None;
    while let Some(mut chunk) = next_upload(next, &mut pending, MAX_CHUNK_SAMPLES)? {
        n += 1;
        on_step(&format!("transcribing via API (chunk {n})"));
        let span = chunk.start..chunk.end();
        // Padding makes tiny tails acceptable to ASR endpoints; cues are
        // still clipped to the source span, excluding this synthetic tail.
        chunk
            .samples
            .resize(chunk.samples.len().max(WHISPER_SAMPLE_RATE as usize), 0.0);
        let wav = wav::encode(&chunk.samples, WHISPER_SAMPLE_RATE);
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
            span,
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

/// Short pauses stay in one request; long absent spans cost no upload bytes.
const MAX_GAP_SAMPLES: usize = 2 * WHISPER_SAMPLE_RATE as usize;

fn next_upload<F>(
    next: &mut F,
    pending: &mut Option<AudioChunk>,
    capacity: usize,
) -> Result<Option<AudioChunk>, Fault>
where
    F: FnMut(NonZeroUsize) -> Result<Option<AudioChunk>, Fault>,
{
    let mut output: Option<AudioChunk> = None;
    loop {
        let remaining = capacity - output.as_ref().map_or(0, |chunk| chunk.samples.len());
        let Some(bound) = NonZeroUsize::new(remaining) else {
            break;
        };
        let chunk = match pending.take() {
            Some(chunk) => Some(chunk),
            None => next(bound)?,
        };
        let Some(mut chunk) = chunk else { break };
        let out = output.get_or_insert_with(|| AudioChunk {
            start: chunk.start,
            samples: Vec::new(),
        });
        let gap = chunk.start.samples_since(out.end());
        if gap > MAX_GAP_SAMPLES || gap >= remaining {
            *pending = Some(chunk);
            break;
        }
        out.samples.resize(out.samples.len() + gap, 0.0);
        let take = chunk.samples.len().min(remaining - gap);
        if take < chunk.samples.len() {
            *pending = Some(AudioChunk {
                start: chunk.start.advance(take),
                samples: chunk.samples.split_off(take),
            });
        }
        out.samples.extend(chunk.samples);
    }
    Ok(output)
}

// The bound and the budget must agree: a full chunk's WAV fits one upload.
const _: () =
    assert!(wav::HEADER_BYTES + MAX_CHUNK_SAMPLES * wav::BYTES_PER_SAMPLE <= MAX_UPLOAD_BYTES);

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
    span: Range<SamplePos>,
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
        200..=299 => parse_verbose_json(&resp_body, span),
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
fn parse_verbose_json(body: &str, span: Range<SamplePos>) -> Result<Vec<Cue>, Fault> {
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

    let offset_s = span.start.seconds();
    let duration = span.end.seconds() - offset_s;
    let cues = resp
        .segments
        .unwrap_or_default()
        .into_iter()
        .filter_map(|seg| {
            let text = seg.text.trim().to_string();
            if text.is_empty() || !seg.start.is_finite() || !seg.end.is_finite() {
                return None;
            }
            let start = seg.start.clamp(0.0, duration);
            let end = seg.end.clamp(0.0, duration);
            if end <= start {
                return None;
            }
            Some(Cue {
                start_cs: ((start + offset_s) * 100.0).round() as i64,
                end_cs: ((end + offset_s) * 100.0).round() as i64,
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
    fn bounds_uploads_including_gap_padding_and_leaves_long_gaps_absent() {
        let mut chunks = [
            AudioChunk {
                start: SamplePos::from_samples(0),
                samples: vec![1.0; 10],
            },
            AudioChunk {
                start: SamplePos::from_samples(15),
                samples: vec![2.0; 10],
            },
            AudioChunk {
                start: SamplePos::from_samples(1_000_000),
                samples: vec![3.0; 5],
            },
        ]
        .into_iter();
        let mut next = |_| Ok(chunks.next());
        let mut pending = None;
        let first = next_upload(&mut next, &mut pending, 20)
            .expect("upload")
            .expect("first");
        assert_eq!(
            first.samples,
            [vec![1.0; 10], vec![0.0; 5], vec![2.0; 5]].concat()
        );
        let second = next_upload(&mut next, &mut pending, 20)
            .expect("upload")
            .expect("second");
        assert_eq!(second.start, SamplePos::from_samples(20));
        assert_eq!(second.samples, vec![2.0; 5]);
        let third = next_upload(&mut next, &mut pending, 20)
            .expect("upload")
            .expect("third");
        assert_eq!(third.start, SamplePos::from_samples(1_000_000));
        assert_eq!(third.samples, vec![3.0; 5]);
        assert!(
            next_upload(&mut next, &mut pending, 20)
                .expect("end")
                .is_none()
        );
    }

    #[test]
    fn uploads_many_tiny_runs_once_and_clips_cues_before_tail_padding() {
        use std::io::{BufRead, Read, Write};
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("API peer");
        let addr = listener.local_addr().expect("address");
        let peer = std::thread::spawn(move || {
            let (mut socket, _) = listener.accept().expect("one request");
            socket
                .set_read_timeout(Some(Duration::from_secs(5)))
                .expect("timeout");
            let mut reader = std::io::BufReader::new(&mut socket);
            let mut length = 0;
            loop {
                let mut line = String::new();
                assert!(reader.read_line(&mut line).expect("header") > 0);
                if line == "\r\n" {
                    break;
                }
                if let Some((name, value)) = line.split_once(':')
                    && name.eq_ignore_ascii_case("content-length")
                {
                    length = value.trim().parse().expect("length");
                }
            }
            let mut body = vec![0; length];
            reader.read_exact(&mut body).expect("body");
            let riff = body.windows(4).position(|b| b == b"RIFF").expect("WAV");
            let bytes = u32::from_le_bytes(body[riff + 40..riff + 44].try_into().expect("size"));
            assert_eq!(bytes, 32_000, "subsecond audio is padded to one second");
            let json = r#"{"segments":[{"start":0,"end":1,"text":"speech"},{"start":0.8,"end":1,"text":"padding"}]}"#;
            write!(
                socket,
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{json}",
                json.len()
            )
            .expect("response");
        });
        let mut runs = (0..20).map(|i| AudioChunk {
            start: SamplePos::from_samples(16_000 + i * 176),
            samples: vec![0.5; 160],
        });
        let cues = recognize(
            "key",
            &format!("http://{addr}"),
            None,
            "",
            &mut |_| Ok(runs.next()),
            &mut |_| {},
        )
        .expect("one upload");
        peer.join().expect("WAV expectations");
        assert_eq!(
            cues,
            vec![Cue {
                start_cs: 100,
                end_cs: 122,
                text: "speech".into()
            }]
        );
    }

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
            SamplePos::from_samples(0)..SamplePos::from_samples(160_000),
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
        let cues = parse_verbose_json(
            body,
            SamplePos::from_samples(0)..SamplePos::from_samples(160_000),
        )
        .expect("valid json");
        assert_eq!(cues.len(), 2);
        assert_eq!(cues[0].text, "Hello world");
        assert_eq!(cues[0].start_cs, 0);
        assert_eq!(cues[0].end_cs, 250);
        assert_eq!(cues[1].start_cs, 300);
    }

    #[test]
    fn parses_verbose_json_with_sample_offset() {
        let body = r#"{"segments":[{"id":0,"start":1.0,"end":22.0,"text":"hi"}]}"#;
        // offset = 160000 samples = 10 seconds
        let cues = parse_verbose_json(
            body,
            SamplePos::from_samples(160_000)..SamplePos::from_samples(192_000),
        )
        .expect("valid json");
        assert_eq!(
            cues[0].start_cs, 1100,
            "1.0s + 10.0s offset = 11.0s = 1100cs"
        );
        assert_eq!(
            cues[0].end_cs, 1200,
            "a cue cannot extend into the following gap"
        );
    }
}
