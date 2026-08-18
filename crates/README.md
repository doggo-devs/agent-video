# agent-video

Turn a video into VLM-readable grid images. Instead of feeding a model one frame
at a time, agent-video samples frames, drops blurry and near-duplicate ones, scores
the rest, and packs them into grid images sized to a single pixel budget. Key
frames get larger cells so important moments keep their detail.

## Requirements

- Rust (edition 2024)
- FFmpeg shared libraries + headers to build against (e.g. `pacman -S ffmpeg`,
  `apt install libavcodec-dev libavformat-dev libswscale-dev`). Decoding,
  probing, and subtitle extraction all run in-process — no `ffmpeg`/`ffprobe`
  executables are spawned or required on `PATH`.

## Build

```bash
cargo build --release
# binary at target/release/agent-video
```

The default build links the system FFmpeg libraries dynamically, so the host
needs a matching FFmpeg installed at runtime.

For a self-contained release binary with **zero runtime dependencies**, build
FFmpeg from source and link it statically (adds static dav1d for software AV1):

```bash
./scripts/build-static.sh
# requires: git, nasm, meson, ninja, pkg-config, a C toolchain
```

The static binary still links glibc. For a single-file artifact with no dynamic
deps, build against musl (needs `musl-tools` + `rustup target add
x86_64-unknown-linux-musl`):

```bash
TARGET=x86_64-unknown-linux-musl ./scripts/build-static.sh
```

The static build has **no TLS backend**: `https://` inputs to `process`/
`extract` fail with `PROTOCOL_UNSUPPORTED`. This is deliberate — URL sources
go through the harness downloader (rustls, CA roots compiled in), so the MCP
path is unaffected; for direct CLI use, download the file first. Dynamic
builds keep whatever protocols the system FFmpeg provides.

The static binary has **no hardware acceleration** — FFmpeg is configured with
`--disable-autodetect`, so all decoding is pure CPU (dav1d SIMD, dispatched at
runtime). This is the right trade-off for a portable binary; use the dynamic
build if you need the host's hardware decoders.

The static binary also has **no local recognizer**: `build-static.sh` passes
`--no-default-features`, dropping `local-asr`, so there is no `install-asr`
subcommand and transcription needs `WHISPER_API_KEY`. The recognizer links a
prebuilt onnxruntime built against glibc, and under musl it segfaults inside
musl's allocator while loading a model — reproducible in a plain C program
with no Rust involved, so the constraint is upstream's. Fixing it means
building onnxruntime itself against musl. Dynamic builds carry the local
tier; `cargo build --no-default-features` reproduces the API-only shape
anywhere.

`.cargo/config.toml` targets `x86-64-v3` (AVX2, ~all x86-64 CPUs since 2015).
When distributing to older hardware, override with
`RUSTFLAGS="-C target-cpu=x86-64"` and set `FFMPEG_TARGET_CPU=x86-64` so the
statically compiled FFmpeg matches the same baseline.

## Usage

```bash
# Smart frame selection + grid composition
agent-video process video.mp4 -o out/

# Process straight from a URL (any protocol the build's FFmpeg can open;
# static builds have no TLS — download first or use the MCP path)
agent-video process https://example.com/talk.mp4 -o out/

# Override the canvas edge length
agent-video process video.mp4 --max-edge 4096 -o out/

# High-density sampling of one time range into a single grid
agent-video extract video.mp4 --range 01:23-01:35 -o out/

# Output JSON with base64 images to stdout (no files written)
agent-video process video.mp4 -o /dev/null --json

# Use a named profile
agent-video process video.mp4 --profile fast -o out/

# Control parallelism
agent-video process video.mp4 -j 4 --in-flight 2 -o out/
```

Output is written to the directory given by `-o`: one or more `grid_*.jpg` images
plus a `metadata.json` describing each grid, its time range, and its key frames.

The input may be a local path or any URL the linked FFmpeg can open. `--from`/
`--until`/`--range` seek by timestamp; over `http(s)://` this needs server Range
support, else it falls back to a sequential read.

File output streams grids to disk as they finish, so memory stays flat over any
video length. `--json` instead holds every grid in memory (base64-inline) until
the run completes — fine for inline consumption of short videos, but prefer
file output for long ones.

## Options

| Flag | Values | Default | Effect |
|------|--------|---------|--------|
| `--max-edge` | integer | `2048` | Canvas edge length in pixels |
| `--quality` | `low`, `medium`, `high` | `low` | Cell size floor; lower packs more frames per grid |
| `--detail` | `coarse`, `standard`, `fine` | `standard` | Sampling fps |
| `--profile` | `balanced`, `fast`, `quality`, `action` | `balanced` | Named parameter preset |
| `--strategy` | `fixed-fps`, `keyframes`, `adaptive` | `fixed-fps` | Frame sampling strategy |
| `--json` | flag | off | Output JSON with base64 to stdout |
| `-j` / `--threads` | integer | CPU cores | Rayon worker threads |
| `--in-flight` | integer | 2 | Max chunks buffered in pipeline |

Run `agent-video process --help` for the full list of override knobs (sampling,
scoring weights, timing, subtitles).

## How it works

1. **Extract** — an in-process decoder samples frames at an fps set by
   `--detail`, streaming through a dedup filter that drops perceptual-hash
   duplicates.
2. **Score** — each frame gets a composite score from scene change, clarity,
   and edge density; top scorers become key frames.
3. **Lay out & compose** — frames are packed into a grid within the pixel budget,
   with key frames given larger cells, then rendered to JPEG.

## Benchmarking transcription

A manual benchmark (never run in CI) measures whether conditioning —
the `--prompt` continuation text and the `--language` hint — improves
transcription quality on real media. It transcribes the same file under a
2×2 conditioning matrix and reports CER (character error rate, Levenshtein
over cue text with whitespace/punctuation stripped) against a human
reference SRT:

```bash
cargo run -p agent-video-harness --example bench_transcribe --release -- \
    <media> <reference.srt> [--prompt "..."] [--language zh]
```

With no arguments it uses `test2.mp4` / `test2.zh.srt` at the repo root
(Chinese speech, hence the `zh` default). It needs a provisioned route:
local weights (`agent-video install-asr`) or `WHISPER_API_KEY`. The
prompt rows only move the API route — SenseVoice takes no conditioning
text, so locally they measure the same recognition twice. Each row writes to a throwaway temp directory — the
real transcript cache is never touched.

**Leakage caveat:** never feed text from the reference SRT into `--prompt`.
The prompt must be knowledge an operator has *without* the ground truth —
a video title, domain vocabulary — or the numbers flatter conditioning it
cannot earn on unseen media. When `--prompt` is absent the prompt rows are
skipped rather than derived from the reference, for exactly this reason.

## License

Apache-2.0. The embedded timestamp font (Liberation Sans Bold) is licensed under
the SIL Open Font License 1.1 — see `assets/OFL.txt`. FFmpeg libraries are
linked (dynamically by default, statically with `--features static-ffmpeg`) and
carry their own LGPL/GPL licensing depending on the build configuration.
