# Local ASR engine survey

Measured 2026-08-03. This exists so the engine choice is decided by numbers
once, instead of re-argued from model announcements every quarter. It records
what was measured, on what, with which normalization — and, equally important,
which conclusions turned out to be artifacts of a bad measurement.

**The decision it supports:** SenseVoice (via sherpa-onnx, CPU) is the only
local engine. whisper stays, but only where it was already reached over the
network — the default API route is Groq's `whisper-large-v3-turbo`. Local
whisper (whisper-rs, GPU) is gone; see [What was cut](#what-was-cut).

## What was measured

**Chinese** — `test2.mp4`, 6m46s of Mandarin tech commentary, dense with
internet slang and product names. Reference is `test2.zh.srt`, the human
sidecar the repo already uses as ground truth in
`crates/harness/examples/bench_transcribe.rs`. Metric is that example's CER:
whole transcript concatenated in time order, Unicode alphanumerics only, ASCII
lowercased, Levenshtein over reference length.

**English** — LibriSpeech test-clean, 150 utterances (17.5 min), seeded random
sample. Metric is standard WER: lowercased, punctuation stripped, whitespace
tokenized.

**Conditions** — every sherpa engine ran int8 on CPU with 8 threads, over
identical Silero VAD spans (45 spans, 373.1s of speech from 406s of audio).
The whisper row marked *GPU path* is the existing in-tree route: whisper-rs,
ggml, `large-v3-turbo`, Vulkan on an RTX 4090, with the project's own VAD and
language hint.

## Results

| engine | zh CER | en WER | timestamps | xRT (CPU) | size |
| --- | --- | --- | --- | --- | --- |
| **SenseVoice-small** | **18.02%** | **3.44%** | character-level, 45/45 | **67–73x** | 226 MB |
| whisper `large-v3-turbo` (GPU path) | **16.50%** | not measured | segment-level, 183 cues | 26.5x *on GPU* | 1.6 GB |
| Qwen3-ASR-0.6B | 28.38% | 2.79% | none, 0/45 | 7–8.5x | 1.0 GB |
| Cohere Transcribe | 50.58% | **1.72%** | none | 9–11x | 2.9 GB |
| Paraformer-zh | 31.94% | — (zh only) | none | 82x | 242 MB |
| FireRedASR2-CTC | 33.01% | 11.05% | character-level | 10x | 778 MB |
| FireRedASR2-AED | 32.74% | — | none | **1.2x** | 1.2 GB |
| Dolphin-small | 49.96% | — | character-level | 63x | 250 MB |
| whisper via sherpa-onnx | 40.39% | 2.83% | none | 5–6x | 1.0 GB |

Two rows for whisper because the engine is fine and one of its runtimes is not;
see [The sherpa whisper CJK defect](#the-sherpa-whisper-cjk-defect). The two
rows are not interchangeable halves of one engine: the GPU path is ggml fp16
with the project's conditioning, the sherpa row is int8 on CPU. English was
measured only on the sherpa row, Chinese is trustworthy only on the GPU row.

Every English figure excludes the 5 of 150 utterances where some engine emitted
digits — LibriSpeech references spell numbers out, so inverse text
normalization ("13th" against "THIRTEENTH") would otherwise be scored as error.
The exclusion is applied to all engines alike and moves SenseVoice most
(3.68% → 3.44%), every other engine by under 0.05pp.

## What the numbers say

**No engine wins both languages.** Whisper leads Chinese, Cohere leads English
by a wide margin and trails by a wide margin on Chinese, Qwen3 is second tier
in both. SenseVoice is second on Chinese and fourth on English — but the whole
English field spans 1.7pp, so fourth there costs far less than second place on
Chinese earns, and it is the only engine carrying timestamps *and* running an
order of magnitude faster without a GPU.

**Whisper's Chinese lead is real but small, and expensive.** 1.5pp over
SenseVoice — about 34 characters on a 2248-character reference — bought with
1.6 GB of weights, a GPU, and the whole ggml build stack. On CPU whisper drops
to 6x realtime; SenseVoice does 67x. For hosts without a GPU the comparison is
not 1.5pp, it is "works" versus "unsupported" (`whisper-local` requires a GPU
backend by construction).

**"Leaderboard rank" does not transfer across languages.** Cohere Transcribe
tops the (English) Open ASR Leaderboard and reproduces here at 1.72% WER — and
lands dead last on Chinese at 50.58%. Any future model shortlisted from an
English leaderboard must be measured on Chinese before it is believed.

**Autoregressive decoders do not emit timestamps.** Qwen3-ASR, Cohere,
FireRedASR2-AED and whisper-via-sherpa all returned 0/45 spans with token
times. Only the non-autoregressive heads (SenseVoice, FireRedASR2-CTC,
Dolphin) produced them. This is structural, not a configuration miss: without
timestamps a cue can only span a whole VAD segment — 8.3s average here, which
is not a usable subtitle.

**Qwen3-ASR's `hotwords` did not help.** Seeded with the exact phrases every
engine got wrong, CER moved 28.38% → 29.18%. It is not a rescue for the
accuracy gap.

## What was cut

The two rows for whisper describe **the same model** the default API route
already serves: `API_MODEL` is `whisper-large-v3-turbo` and `DEFAULT_API_URL`
is Groq. So local whisper bought exactly one state — *offline* — for 1.6 GB of
weights, four GPU cargo features, a crash-isolated device probe, and 24
`#[cfg]` sites. SenseVoice buys the same state for 226 MB of CPU weights.

Read that way the table is not a ranking, it is a price list:

- **16.50% CER = the price of being online** (free, Groq, 99 languages)
- **18.02% CER = the price of being offline** (226 MB, no GPU, 5 languages)

The 1.5pp is the offline premium, and part of it is quantization (fp16-on-GPU
vs int8-on-CPU) rather than the engine.

What the cut removes, honestly: **offline recognition of anything outside
zh/en/ja/ko/yue**. Those five are also still unmeasured here (see
[Not measured](#not-measured)) — accepted deliberately, because the local tier
no longer promises to match whisper. It promises to be cheap and to run
without a key. Anyone who needs accuracy or coverage has a free API for it.

Coexistence was verified before the cut and no longer matters, but the finding
stands for anyone re-treading this path: sherpa-onnx (onnxruntime) and
whisper-rs (ggml) do **not** collide the way llama.cpp and whisper.cpp do —
both FFI boundaries were crossed at runtime in one 75.6 MB binary.

Measured after the cut, on the code that shipped: 18.06% CER bare, 17.57% with
`--language zh`, 406 s of audio in 8.1 s wall (~50x realtime, 8 CPU threads).
The small drift from 18.02% is cue splitting — the shipped engine cuts VAD
spans at token-timestamp gaps, which the survey did not.

## Rejected paths

**llama.cpp + Qwen3-ASR (GGUF).** Three independent blockers: duplicate ggml
symbols against whisper-rs (upstream llama.cpp #9267, #11303, ggml #1148); no
timestamps, which the GGUF path cannot fix because timestamps live in a
separate model (`Qwen3-ForcedAligner-0.6B`, not supported by llama.cpp); and
empty output on audio beyond ~2 minutes (llama.cpp #21847). Qwen3-ASR is also
available through sherpa-onnx, where the first and third blockers disappear —
and it still loses to SenseVoice on Chinese by 10pp.

**Cohere Transcribe, FireRedASR2, Dolphin, Paraformer.** Cohere and Dolphin on
Chinese accuracy; FireRedASR2 on both (33% CER, 11% WER, despite the `zh_en`
label); FireRedASR2-AED additionally on speed (1.2x realtime is slower than
the video plays); Paraformer on language coverage (Chinese only) despite being
the fastest and smallest thing measured.

## The sherpa whisper CJK defect

whisper through sherpa-onnx drops Chinese characters. Observed directly:

```text
[   4.55] 从怎么到家     (expected: 从飞龙骑脸怎么输,到状态奇佳大杀四方)
raw tokens: ["从","","","","","","","","","怎么","","","到","","","","","","","家"]
```

Whisper uses byte-level BPE, so one Chinese character spans several tokens that
are each invalid UTF-8 on their own; the result crosses the FFI boundary as
JSON (`SherpaOnnxGetOfflineStreamResultAsJson` → `to_string_lossy` →
`serde_json`) and those pieces come back empty. English through the identical
code path is unaffected — the tokens are ASCII fragments and reassemble
correctly, which is why the English WER row for that runtime is sound.

Independently reported upstream as k2-fsa/sherpa-onnx#2900 (same audio: 0.25
CER with faster-whisper, 0.81 with sherpa-onnx), open and unanswered. Until it
is fixed, whisper's Chinese must come from whisper-rs, not from sherpa.

## Corrections to earlier readings

Recorded because both errors were made confidently, and the same mistake is
easy to repeat.

**The reference was misidentified.** `test2.zh.srt` was initially read as
whisper output, which made whisper look flawless on hard phrases and made its
uncovered 22s look like a recall failure. It is a human sidecar — the file
carries a UTF-8 BOM that `core::subtitle::serialize_srt` never writes, is
untracked by git, and matches `SIDECAR_SUBTITLE_EXTS`, the *input* convention.
The 22s is speech the subtitler chose to skip.

**Eight cherry-picked phrases are not a metric.** Judging by hard-phrase
spot-checks put FireRedASR2-AED first ("the accuracy champion") and suggested
SenseVoice was weak. Measured by CER over all 45 spans, AED sits at 32.74% —
tied with its own CTC head while 9x slower — and SenseVoice leads the field at
18.02%. Small probe sets rank engines by their rare-word behaviour, which is
not what a subtitle track is made of.

## Not measured

- Real-world English (accent, noise, spontaneous speech). LibriSpeech is clean
  read speech; the 0.6pp spread among the top four is roughly a dozen words and
  should not be treated as a durable ordering.
- Chinese beyond one speaker and one domain. One 6m46s video decides the
  Chinese column.
- Precision parity. Whisper's Chinese number is fp16-on-GPU; every sherpa
  number is int8-on-CPU. Part of the 1.5pp gap is quantization, in unknown
  proportion.
- Languages other than Chinese and English, including the ja/ko/yue that
  SenseVoice claims and the 99 that whisper claims.
- Qwen3-ASR-1.7B. sherpa-onnx ships only the 0.6B export; the 1.7B exists
  upstream as HF weights and community GGUF, both on the rejected llama.cpp
  path.

## The musl wall

The local tier does not ship in the fully-static builds. sherpa links a
prebuilt onnxruntime compiled against glibc, and under musl it segfaults
inside musl's `__libc_free` while creating a recognizer — mallocng handed a
pointer it never allocated.

What was ruled out, each by measurement rather than reasoning:

- **Not our linking, and not Rust.** A plain C program against the same static
  libs crashes identically.
- **Not TLS or PIE.** A non-PIE fully static binary crashes the same way.
- **Not the thread stack** (musl gives 128 KiB where glibc gives 8 MiB, the
  usual suspect). `PT_GNU_STACK` was confirmed raised to 8 MiB; no change.
- **Not sherpa's own code.** Rebuilding sherpa from source against musl works
  and takes 53 s — onnxruntime stays prebuilt because it references no
  glibc-only symbol. The crash survives that rebuild.
- **Not glibc-static.** The same C program, statically linked against glibc,
  returns a working recognizer.

So the wall is onnxruntime's, and getting through it means building
onnxruntime itself against musl. Until someone does, `local-asr` is a default
feature that `build-static.sh` turns off.

## Supply chain note

`sherpa-onnx-sys` downloads prebuilt static libraries from GitHub releases at
build time rather than compiling from source (which is why a cold build is 24s,
not minutes). Before this ships, that fetch must be verified the way
`crate::install` verifies model weights: pinned URL, pinned sha256.
