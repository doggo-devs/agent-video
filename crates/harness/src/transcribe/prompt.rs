// SPDX-License-Identifier: Apache-2.0
//! Whisper conditioning text: the effective prompt an operator or a media
//! title yields, and the continuation chain that carries transcript
//! context across API chunks. Everything here is text shaping — no
//! network, no audio.

use super::Cue;
use crate::config::Config;

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
/// hash is the prompt's cache identity ([`super::transcript_id`]).
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

/// The conditioning chain across chunks: each chunk's prompt is
/// [`continuation_prompt`] over the transcript tail of the last chunk that
/// produced any cues. A silent chunk keeps the previous tail alive — one
/// stretch of silence must not wipe the chain and reset the next chunk to
/// the bare initial prompt.
///
/// The retained tail is bounded near [`PROMPT_MAX_BYTES`]: only the cues
/// that can survive [`continuation_prompt`]'s front-truncation are kept,
/// so the chain's memory never grows with the transcript.
///
/// Cache note: the tail derives from the previous chunk's *output*, which
/// an endpoint need not reproduce bit-for-bit — and that is fine for the
/// key. [`super::transcript_id`] over the initial prompt names the recipe,
/// not one run of it: the whole SRT is produced under that one initial
/// conditioning and published atomically, so whichever run lands is a
/// valid value for its key. No cache-key change is needed here.
pub(super) struct PromptChain {
    initial: String,
    tail: String,
}

impl PromptChain {
    pub(super) fn new(initial: &str) -> Self {
        PromptChain {
            initial: initial.to_string(),
            tail: String::new(),
        }
    }

    /// The conditioning text for the next chunk.
    pub(super) fn prompt(&self) -> String {
        continuation_prompt(&self.initial, &self.tail)
    }

    /// Feed one chunk's cues into the chain. Empty cues (a silent chunk)
    /// leave the tail untouched.
    pub(super) fn absorb(&mut self, cues: &[Cue]) {
        if cues.is_empty() {
            return;
        }
        // Walk from the back until the prompt budget is covered: anything
        // older would fall off the front-truncation anyway.
        let mut from = cues.len();
        let mut bytes = 0;
        while from > 0 && bytes < PROMPT_MAX_BYTES {
            from -= 1;
            bytes += cues[from].text.len() + 1;
        }
        self.tail = cues[from..]
            .iter()
            .map(|cue| cue.text.as_str())
            .collect::<Vec<_>>()
            .join(" ");
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{default_config, test_dirs};
    use std::path::Path;

    fn cfg() -> Config {
        default_config(&test_dirs(Path::new("/home/u")))
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
    fn second_chunk_carries_the_first_chunks_tail() {
        let mut chain = PromptChain::new("Rust Ownership.");
        assert_eq!(
            chain.prompt(),
            "Rust Ownership.",
            "first chunk: initial only — no tail machinery visible"
        );
        chain.absorb(&[
            Cue {
                start_cs: 0,
                end_cs: 100,
                text: "chunk 0 opens".into(),
            },
            Cue {
                start_cs: 100,
                end_cs: 200,
                text: "chunk 0 closes".into(),
            },
        ]);
        assert_eq!(
            chain.prompt(),
            "Rust Ownership. chunk 0 opens chunk 0 closes",
            "initial first, previous transcript last (nearest the audio)"
        );
    }

    #[test]
    fn silent_chunk_does_not_erase_the_prompt_chain() {
        let mut chain = PromptChain::new("");
        chain.absorb(&[Cue {
            start_cs: 0,
            end_cs: 100,
            text: "speech 0".into(),
        }]);
        chain.absorb(&[]); // silence: the endpoint found nothing to say
        assert_eq!(
            chain.prompt(),
            "speech 0",
            "a silent chunk keeps the last non-empty tail alive"
        );
    }

    #[test]
    fn chain_tail_memory_stays_bounded() {
        // An hour-long chunk can carry thousands of cues; the chain must
        // retain only what the prompt budget can ever use, not the chapter.
        let cues: Vec<Cue> = (0..10_000)
            .map(|i| Cue {
                start_cs: i,
                end_cs: i + 1,
                text: format!("cue {i:05}"),
            })
            .collect();
        let mut chain = PromptChain::new("");
        chain.absorb(&cues);
        let longest = cues.iter().map(|c| c.text.len() + 1).max().expect("cues");
        assert!(
            chain.tail.len() <= PROMPT_MAX_BYTES + longest,
            "tail holds one budget's worth, not the transcript: {}",
            chain.tail.len()
        );
    }

    #[test]
    fn bounded_tail_matches_the_unbounded_join() {
        // The bound is invisible in the produced prompt: same text as
        // joining every cue and front-truncating.
        let cues: Vec<Cue> = (0..500)
            .map(|i| Cue {
                start_cs: i,
                end_cs: i + 1,
                text: format!("词{i:04}"),
            })
            .collect();
        let full_join = cues
            .iter()
            .map(|c| c.text.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        let mut chain = PromptChain::new("Rust Ownership.");
        chain.absorb(&cues);
        assert_eq!(
            chain.prompt(),
            continuation_prompt("Rust Ownership.", &full_join)
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
}
