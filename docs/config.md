# config.toml reference

The harness reads one optional TOML file. Every key overrides a built-in
default; an absent file (or an empty document) yields the defaults verbatim.
Validation runs at load time and is strict: an unknown key, a zero where a
positive integer is required, an empty path, an uncompiled whisper backend, or
an unknown browser all fail the load, and a misconfigured server refuses to
start rather than run on a half-valid config.

## Location

The file lives at `<config_dir>/agent-video/config.toml`, where `<config_dir>`
comes from the `dirs` crate (`$XDG_CONFIG_HOME` or `~/.config` on Linux,
`~/Library/Application Support` on macOS, `%APPDATA%` on Windows). Set
`$AGENT_VIDEO_CONFIG` to point at an explicit path instead.

`agent-video config init` writes a fully-commented template there,
sshd_config-style: every key at its resolved default for the machine,
disabled with `#` — the file changes nothing until a line is uncommented.
It refuses to overwrite an existing file unless given `--force`.

Defaults that name a directory hang off the platform's base dirs (cache, data,
download, config), so the concrete default paths below are the Linux/XDG
spelling; on other platforms the same roles resolve to that platform's
conventions.

## Path handling

Every path value is tilde-expanded: a leading `~/` (or a bare `~`) resolves
against `$HOME`; anything else is taken as-is. An empty path string is a load
error.

## Sections

### `[paths]`

The three path roles, kept distinct. Each is an optional string; the default is
platform-derived.

| Key | Type | Default | Controls |
|-----|------|---------|----------|
| `cache` | path | `<cache_dir>/agent-video` | Evictable analysis artifacts: download rips (`downloads/`), probe documents (`probe/`), whisper transcripts (`transcripts/`). Subject to the `[cache]` LRU budget and age sweeps. |
| `models` | path | `<data_dir>/agent-video/models` | Whisper weights (model + Silero VAD) provisioned by `install-whisper`. Persistent and LRU-exempt — never evicted. |
| `keep` | path | `<download_dir>` | Landing directory for keep downloads. Files here belong to the user; the harness never evicts or rewrites them. |

### `[cache]`

| Key | Type | Default | Validation | Controls |
|-----|------|---------|-----------|----------|
| `max_bytes` | integer | `10737418240` (10 GiB) | must be > 0 | LRU eviction threshold over `downloads/` only (probe docs and transcripts are evicted by age, not size). A soft target: entries younger than the 5-minute grace window and in-flight `.part` files are never evicted, so the cache may briefly exceed this. |

### `[whisper]`

| Key | Type | Default | Validation | Controls |
|-----|------|---------|-----------|----------|
| `model` | string | `large-v3-turbo` | known name or path-shaped | Model identity for transcription and for keying transcripts in the cache. One of the pinned names `medium`, `large-v3`, `large-v3-turbo` (provisioned by `install-whisper`), or a path to a self-supplied ggml file (tilde-expanded) — recognized by its shape: it contains a path separator or starts with `~` or `.`. Any other value (a typo like `large-v3-trubo`) fails the load. A transcript is keyed by content × model, so changing this never serves another model's output. |
| `backend` | string | unset (auto-detect) | one of `metal`, `coreml`, `cuda`, `vulkan`, and must be compiled into this build | Pins the GPU inference backend. Unset auto-detects from compiled features. Pinning commits to local inference: a missing weight then errors instead of silently falling back to the API. CPU-only inference is not supported; a build with no `whisper-*` feature offers no local transcription at all. |
| `api_key` | string | unset | — | API key for remote transcription (Groq / OpenAI-compatible endpoint). The `WHISPER_API_KEY` environment variable overrides this at load time; past load, nothing reads the environment again. |
| `api_url` | string | `https://api.groq.com/openai/v1/audio/transcriptions` | — | Remote transcription endpoint. |
| `prompt` | string | unset (derived) | non-empty | Conditioning text for whisper (its *initial prompt*). A continuation bias, not an instruction channel: write text that reads like preceding transcript — domain vocabulary, proper nouns, the desired script (简体 vs 繁体) — never directives, which the model does not obey. Budget is ~224 tokens; the harness truncates at 800 bytes. When unset, URL downloads derive a prompt from the media title reported by the site; local files get none. Part of the transcript cache key (hashed), so changing it produces a fresh transcript rather than serving a differently-conditioned one. |
| `language` | string | unset (auto-detect) | two lowercase ASCII letters (ISO 639-1), e.g. `zh`; `auto` is invalid — auto-detection is the key's absence | Language hint for transcription. Sent to the API as the `language` field and pinned in local inference. Part of the transcript cache key. |

### `[download]`

| Key | Type | Default | Validation | Controls |
|-----|------|---------|-----------|----------|
| `max_bytes` | integer | `2147483648` (2 GiB) | must be > 0 | Hard refusal ceiling for any single download. Known-oversize downloads are refused before the first byte; unknown-length streams are cut off mid-flight. |
| `analysis_max_height` | integer | `480` | must be > 0 | Frame-height ceiling for analysis rips (the grid-quality ceiling). Part of the rip's cache key, so changing it produces a distinct cache entry rather than serving stale bytes. |
| `sync_wait_seconds` | integer | `90` | must be > 0 | How long a confirmed download may block the `video` call before detaching into the background tracker. Generous by design: for an agent one blocked call is cheaper than a chain of pending polls, each of which costs a full model round trip. |

### `[budget]`

The grid-count ladder that gates the decision engine (`crate::plan`).

| Key | Type | Default | Validation | Controls |
|-----|------|---------|-----------|----------|
| `max` | integer | `8` | must be > 0 | Chunk-count ceiling. Above it the run's chunks are widened to fit or, if even the widened chunks are too thin, it requires explicit consent. A bound on grids, never a prediction: dedup decides the real count after decode. |
| `max_stretch_seconds` | integer | `600` | must be > 0 | Degradation floor: the most seconds one widened chunk may cover. Past it the plan is likely too thin to be useful, so running it requires explicit consent. |

### `[network]`

| Key | Type | Default | Controls |
|-----|------|---------|----------|
| `allow_private_hosts` | boolean | `false` | Permit URL sources whose hosts resolve to private, loopback, or otherwise non-global addresses. Off by default so a model-supplied URL cannot reach internal services (SSRF). Enforcement lives in `crate::netguard`. |

### `[cookies]`

Cookie forwarding for login-gated sites, consumed only by the yt-dlp source.
Acquisition is always explicit — this key or `agent-video cookie import` —
never an implicit browser read during a tool call.

| Key | Type | Default | Validation | Controls |
|-----|------|---------|-----------|----------|
| `from_browser` | string | unset | base name (before any `:profile` / `+keyring` suffix) must be a lowercase alphanumeric word; the browser set itself is yt-dlp's, checked at spawn time | yt-dlp's `--cookies-from-browser` value (e.g. `"firefox"`, `"chrome:Profile 1"`): read cookies live from the browser store at every spawn. When set, it wins over the jar. |

The canonical Netscape jar lives at `<config_dir>/agent-video/cookies.txt` and
is **not** file-configurable — it is managed by `agent-video cookie import`
and used automatically when the file exists and `from_browser` is unset.

### `[subtitles]`

| Key | Type | Default | Validation | Controls |
|-----|------|---------|-----------|----------|
| `preferred_langs` | array of strings | `["en", "zh"]` | non-empty list of non-empty prefixes | Language preference for electing the one promoted subtitle sidecar from a multi-language yt-dlp download. Prefix-matched (`"en"` claims `"en-US"`), first hit wins; no hit falls back to the first candidate. |

## Complete example

Every key set to its default value. Keys that default to *unset* (whisper
`backend` and `api_key`, cookies `from_browser`) are shown commented out —
their default is absence. Path values use the Linux/XDG spelling; on other
platforms the same roles resolve to that platform's base directories.

```toml
[paths]
cache = "~/.cache/agent-video"
models = "~/.local/share/agent-video/models"
keep = "~/Downloads"

[cache]
max_bytes = 10737418240  # 10 GiB

[whisper]
model = "large-v3-turbo"
# backend unset by default: auto-detect from compiled features.
# backend = "vulkan"
# api_key unset by default; WHISPER_API_KEY overrides it at load time.
# api_key = "gsk_..."
api_url = "https://api.groq.com/openai/v1/audio/transcriptions"
# prompt unset by default: URL downloads derive one from the media title.
# Continuation-style vocabulary bias, not instructions.
# prompt = "本期课程讲解 Rust 的所有权、借用检查器与生命周期。"
# language unset by default: whisper auto-detects.
# language = "zh"

[download]
max_bytes = 2147483648  # 2 GiB
analysis_max_height = 480
sync_wait_seconds = 90

[budget]
max = 8
max_stretch_seconds = 600

[network]
allow_private_hosts = false

[cookies]
# from_browser unset by default (anonymous); the canonical jar under the
# config dir is used automatically when it exists.
# from_browser = "firefox"

[subtitles]
preferred_langs = ["en", "zh"]
```
