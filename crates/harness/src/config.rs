// SPDX-License-Identifier: Apache-2.0
//! Harness configuration: `config.toml`.
//!
//! Defaults come from the `dirs` crate; a file at
//! `<config_dir>/agent-video/config.toml` (or `$AGENT_VIDEO_CONFIG`) overrides
//! any key; `agent-video config init` writes a fully-commented template of
//! every key at its default ([`render_template`]). Invalid values fail loudly
//! at load time — a misconfigured server
//! must not start. Positive integers are encoded as `NonZero` types so a zero
//! budget or byte ceiling is unrepresentable past this boundary.

use std::fmt;
use std::num::{NonZeroU32, NonZeroU64};
use std::path::{Path, PathBuf};

use serde::Deserialize;

const GIB: u64 = 1024 * 1024 * 1024;

/// A config value failed validation; the message names the key and value.
#[derive(Debug)]
pub struct ConfigError(pub String);

impl ConfigError {
    /// Exit code for CLI paths: a bad `config.toml` is the same category as
    /// core's invalid configuration, not an uncoded bug (exit 1).
    pub fn exit_code(&self) -> u8 {
        agent_video_core::error::ErrorCode::InvalidConfig.exit_code()
    }
}

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "config: {}", self.0)
    }
}

impl std::error::Error for ConfigError {}

/// Resolved harness configuration; all byte values are absolute counts.
#[derive(Debug, Clone, PartialEq)]
pub struct Config {
    pub paths: Paths,
    pub cache: CacheCfg,
    pub whisper: WhisperCfg,
    pub download: DownloadCfg,
    pub budget: BudgetCfg,
    pub network: NetworkCfg,
    pub cookies: CookiesCfg,
    pub subtitles: SubtitlesCfg,
}

/// Subtitle sidecar policy; consumed by the yt-dlp source when electing
/// the one promoted sidecar from a multi-language download.
#[derive(Debug, Clone, PartialEq)]
pub struct SubtitlesCfg {
    /// Language preference for the election: prefix-matched (`"en"` claims
    /// `"en-US"`), first hit wins, no hit falls back to the first
    /// candidate. Never empty — validation refuses an empty list.
    pub preferred_langs: Vec<String>,
}

/// The three path roles, kept distinct.
#[derive(Debug, Clone, PartialEq)]
pub struct Paths {
    /// Evictable analysis artifacts: download rips, probe docs, transcripts.
    pub cache: PathBuf,
    /// Whisper weights: persistent, LRU-exempt.
    pub models: PathBuf,
    /// Keep-download landing directory; files there belong to the user.
    pub keep: PathBuf,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CacheCfg {
    /// LRU eviction threshold over `downloads/`.
    pub max_bytes: NonZeroU64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct WhisperCfg {
    pub model: WhisperModel,
    /// Explicit backend pin; `None` auto-detects from compiled features.
    /// Pinning a backend commits to local inference: a missing model is
    /// then an error, never a silent fallback to the API (see
    /// [`crate::transcribe::Route::resolve`]).
    pub backend: Option<Backend>,
    /// API key for remote transcription (Groq / OpenAI-compatible).
    /// [`load_config`] lets `WHISPER_API_KEY` override the file value, so
    /// past load time this field is the resolved answer — nothing reads
    /// the environment afterwards.
    pub api_key: Option<String>,
    /// API endpoint URL. Defaults to Groq.
    pub api_url: String,
    /// Operator conditioning text for whisper. An initial prompt is a
    /// continuation bias, not an instruction channel: text that reads
    /// like preceding transcript steers vocabulary, proper nouns, and
    /// script (简体 vs 繁体). When unset, transcription derives one from
    /// the media title recorded at download time. Part of the transcript
    /// cache key ([`crate::transcribe::transcript_id`]).
    pub prompt: Option<String>,
    /// ISO 639-1 language hint (`"zh"`, `"en"`); `None` auto-detects.
    /// Sent to the API verbatim and pinned in local inference; part of
    /// the transcript cache key.
    pub language: Option<String>,
}

impl WhisperCfg {
    /// Default Groq Whisper endpoint.
    pub const DEFAULT_API_URL: &str = "https://api.groq.com/openai/v1/audio/transcriptions";
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DownloadCfg {
    /// Hard refusal ceiling for any single download.
    pub max_bytes: NonZeroU64,
    /// Analysis rips are capped at this frame height (grid quality ceiling).
    pub analysis_max_height: NonZeroU32,
    /// How long a `video` call may block on a running download — the one
    /// that started it and every later call that observes it — before
    /// answering Pending.
    ///
    /// Sized against the job, not the impatience: a download that outlives
    /// this window is answered with a poll, and every poll costs a full
    /// model round trip that buys nothing but a byte count. Five minutes
    /// covers the retry ladder's own budget (the download module's fetch
    /// retry window), so the common shapes — a rip that lands, a burst
    /// that is outlasted, a ladder that gives up — all resolve inside
    /// one call. Pending remains the fallback for genuinely long
    /// transfers, where it states a real fact about the source rather
    /// than a fact about this timer.
    pub sync_wait_seconds: NonZeroU32,
}

/// Egress policy knobs; enforcement lives in [`crate::netguard`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NetworkCfg {
    /// Permit URL sources whose hosts resolve to private, loopback, or
    /// otherwise non-global addresses. Off by default: a model-supplied
    /// URL must not be able to reach internal services (SSRF).
    pub allow_private_hosts: bool,
}

/// Cookie forwarding for login-gated sites; consumed by the yt-dlp source
/// ([`crate::fetch::ytdlp`]). Acquisition is always explicit — a config key
/// or `agent-video cookie import` — never an implicit browser read during a
/// tool call.
#[derive(Debug, Clone, PartialEq)]
pub struct CookiesCfg {
    /// `--cookies-from-browser` value (e.g. `"firefox"`,
    /// `"chrome:Profile 1"`): read cookies live from this browser's store
    /// at every yt-dlp spawn. Set = wins over the jar.
    pub from_browser: Option<String>,
    /// Canonical Netscape jar maintained by `agent-video cookie import`;
    /// used when the file exists and `from_browser` is unset.
    pub jar: PathBuf,
}

/// Validate a `cookies.from_browser` value by shape, not by list: the set
/// of browsers yt-dlp can extract cookies from is yt-dlp's to own and
/// grows upstream, so a frozen snapshot here would reject tomorrow's
/// valid config at load time. Only the base name (before any `:profile` /
/// `+keyring` suffix) is checked — a lowercase alphanumeric word, the
/// shape every yt-dlp browser name has. A typo'd browser fails at spawn
/// time with yt-dlp's own error naming the currently supported set.
fn parse_from_browser(value: &str) -> Result<String, ConfigError> {
    let base = value
        .split([':', '+'])
        .next()
        .expect("split yields at least one item");
    let word = !base.is_empty()
        && base
            .bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit());
    if !word {
        return Err(ConfigError(format!(
            "cookies.from_browser must start with a browser name yt-dlp knows, like \
             'firefox' or 'chrome:Profile 1' — got '{value}'"
        )));
    }
    Ok(value.to_string())
}

/// Validate a `whisper.language` value: exactly two lowercase ASCII
/// letters, the ISO 639-1 shape both engines accept. Notably this rejects
/// `"auto"` — auto-detection is the *absence* of the key, and the OpenAI
/// contract has no `auto` value.
fn parse_language(value: &str) -> Result<String, ConfigError> {
    let iso639_1 = value.len() == 2 && value.bytes().all(|b| b.is_ascii_lowercase());
    if !iso639_1 {
        return Err(ConfigError(format!(
            "whisper.language must be a two-letter lowercase ISO 639-1 code (e.g. 'zh') — \
             got '{value}'; unset it for auto-detection"
        )));
    }
    Ok(value.to_string())
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BudgetCfg {
    /// Chunk-count ceiling: above it the run's chunks are widened so the
    /// render stays bounded. An upper bound on grids, never a prediction —
    /// dedup decides the real count after decode.
    pub max: NonZeroU32,
    /// Degradation floor: max seconds one widened chunk may cover before
    /// running the plan requires explicit consent.
    pub max_stretch_seconds: NonZeroU32,
}

/// Whisper model selection: a pinned name provisioned by `install-whisper`,
/// or a self-supplied ggml file (the escape hatch — spelled as a path:
/// `"./models/ggml-tiny.bin"`, `"~/models/ggml-tiny.bin"`, or any value
/// containing a path separator).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WhisperModel {
    Medium,
    LargeV3,
    LargeV3Turbo,
    File(PathBuf),
}

impl WhisperModel {
    /// Parse a `whisper.model` value. Known names parse to their variant;
    /// a path-shaped value (contains a separator, or starts with `~` or
    /// `.`) is the self-supplied-file escape hatch. Anything else — a typo
    /// like `"large-v3-trubo"` — fails the load naming the valid options,
    /// instead of masquerading as a model file that will never exist.
    fn parse(s: &str) -> Result<Self, ConfigError> {
        match s {
            "medium" => Ok(WhisperModel::Medium),
            "large-v3" => Ok(WhisperModel::LargeV3),
            "large-v3-turbo" => Ok(WhisperModel::LargeV3Turbo),
            path if path.contains(['/', std::path::MAIN_SEPARATOR])
                || path.starts_with('~')
                || path.starts_with('.') =>
            {
                Ok(WhisperModel::File(PathBuf::from(path)))
            }
            other => Err(ConfigError(format!(
                "whisper.model must be one of medium, large-v3, large-v3-turbo, or a path \
                 to a self-supplied ggml file — got '{other}'"
            ))),
        }
    }

    /// Stable, filename-safe identity for cache keying: transcripts from
    /// different models must never answer for each other, so the model is
    /// part of the transcript's cache key.
    ///
    /// A self-supplied ggml is identified by its (hashed) path; swapping the
    /// bytes behind the same path is the operator's own cache-poisoning.
    pub fn cache_id(&self) -> String {
        use sha2::{Digest, Sha256};
        match self {
            WhisperModel::Medium => "medium".to_string(),
            WhisperModel::LargeV3 => "large-v3".to_string(),
            WhisperModel::LargeV3Turbo => "large-v3-turbo".to_string(),
            WhisperModel::File(p) => {
                let digest = Sha256::digest(p.as_os_str().as_encoded_bytes());
                format!("file-{}", crate::identity::hex(&digest[..8]))
            }
        }
    }
}

/// Whisper hardware-accelerated backend. CPU-only inference is not
/// supported — transcription requires a compiled-in GPU backend.
/// Builds without any `whisper-*` feature simply do not offer
/// transcription.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    Metal,
    Coreml,
    Cuda,
    Vulkan,
}

impl Backend {
    fn parse(s: &str) -> Result<Self, ConfigError> {
        let Some(backend) = Backend::from_name(s) else {
            return Err(ConfigError(format!(
                "whisper.backend must be one of metal, coreml, cuda, vulkan — got '{s}'"
            )));
        };
        if !Self::compiled().contains(&backend) {
            let available: Vec<&str> = Self::compiled().iter().map(|b| b.name()).collect();
            return Err(ConfigError(format!(
                "whisper.backend '{s}' is not compiled into this build; available: {}",
                available.join(", ")
            )));
        }
        Ok(backend)
    }

    /// Canonical lowercase name, as accepted by `whisper.backend` and the
    /// hidden `probe-gpu` subcommand.
    pub fn name(self) -> &'static str {
        match self {
            Backend::Metal => "metal",
            Backend::Coreml => "coreml",
            Backend::Cuda => "cuda",
            Backend::Vulkan => "vulkan",
        }
    }

    /// Parse a canonical name, without the compiled-in check.
    pub fn from_name(s: &str) -> Option<Self> {
        [
            Backend::Metal,
            Backend::Coreml,
            Backend::Cuda,
            Backend::Vulkan,
        ]
        .into_iter()
        .find(|b| b.name() == s)
    }

    /// Whether device presence must be probed at runtime
    /// ([`crate::gpu::device_available`]). Metal/CoreML are ambient on
    /// Apple hardware; CUDA/Vulkan depend on loaders, drivers, and devices
    /// that may be absent on the machine running this build.
    pub fn needs_probe(self) -> bool {
        matches!(self, Backend::Cuda | Backend::Vulkan)
    }

    /// All backends compiled into this build, in preference order.
    pub fn compiled() -> &'static [Backend] {
        &[
            #[cfg(feature = "whisper-metal")]
            Backend::Metal,
            #[cfg(feature = "whisper-coreml")]
            Backend::Coreml,
            #[cfg(feature = "whisper-cuda")]
            Backend::Cuda,
            #[cfg(feature = "whisper-vulkan")]
            Backend::Vulkan,
        ]
    }
}

// ── file shape ──

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct FileConfig {
    #[serde(default)]
    paths: FilePaths,
    #[serde(default)]
    cache: FileCache,
    #[serde(default)]
    whisper: FileWhisper,
    #[serde(default)]
    download: FileDownload,
    #[serde(default)]
    budget: FileBudget,
    #[serde(default)]
    network: FileNetwork,
    #[serde(default)]
    cookies: FileCookies,
    #[serde(default)]
    subtitles: FileSubtitles,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct FilePaths {
    cache: Option<String>,
    models: Option<String>,
    keep: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct FileCache {
    max_bytes: Option<u64>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct FileWhisper {
    model: Option<String>,
    backend: Option<String>,
    api_key: Option<String>,
    api_url: Option<String>,
    prompt: Option<String>,
    language: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct FileDownload {
    max_bytes: Option<u64>,
    analysis_max_height: Option<u32>,
    sync_wait_seconds: Option<u32>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct FileBudget {
    max: Option<u32>,
    max_stretch_seconds: Option<u32>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct FileNetwork {
    allow_private_hosts: Option<bool>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct FileCookies {
    from_browser: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct FileSubtitles {
    preferred_langs: Option<Vec<String>>,
}

// ── loading ──

/// Filesystem anchors the defaults hang off; injectable for tests.
#[derive(Debug, Clone)]
pub struct BaseDirs {
    pub home: PathBuf,
    pub cache: PathBuf,
    pub data: PathBuf,
    pub download: PathBuf,
    pub config: PathBuf,
}

impl BaseDirs {
    /// Platform directories via the `dirs` crate, falling back to `~`-relative
    /// conventions when the platform reports none.
    pub fn detect() -> Self {
        let home = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));
        BaseDirs {
            cache: dirs::cache_dir().unwrap_or_else(|| home.join(".cache")),
            data: dirs::data_dir().unwrap_or_else(|| home.join(".local/share")),
            download: dirs::download_dir().unwrap_or_else(|| home.join("Downloads")),
            config: dirs::config_dir().unwrap_or_else(|| home.join(".config")),
            home,
        }
    }
}

/// Default config file location; `$AGENT_VIDEO_CONFIG` overrides the path.
pub fn config_file_path(dirs: &BaseDirs) -> PathBuf {
    match std::env::var_os("AGENT_VIDEO_CONFIG") {
        Some(p) => PathBuf::from(p),
        None => dirs.config.join("agent-video/config.toml"),
    }
}

/// Built-in defaults for the given base directories.
pub fn default_config(dirs: &BaseDirs) -> Config {
    let nz64 = |v: u64| NonZeroU64::new(v).expect("non-zero literal");
    let nz32 = |v: u32| NonZeroU32::new(v).expect("non-zero literal");
    Config {
        paths: Paths {
            cache: dirs.cache.join("agent-video"),
            models: dirs.data.join("agent-video/models"),
            keep: dirs.download.clone(),
        },
        cache: CacheCfg {
            max_bytes: nz64(10 * GIB),
        },
        whisper: WhisperCfg {
            model: WhisperModel::LargeV3Turbo,
            backend: None,
            api_key: None,
            api_url: WhisperCfg::DEFAULT_API_URL.to_string(),
            prompt: None,
            language: None,
        },
        download: DownloadCfg {
            max_bytes: nz64(2 * GIB),
            analysis_max_height: nz32(480),
            sync_wait_seconds: nz32(300),
        },
        budget: BudgetCfg {
            max: nz32(8),
            max_stretch_seconds: nz32(600),
        },
        network: NetworkCfg {
            allow_private_hosts: false,
        },
        cookies: CookiesCfg {
            from_browser: None,
            jar: dirs.config.join("agent-video/cookies.txt"),
        },
        subtitles: SubtitlesCfg {
            preferred_langs: vec!["en".to_string(), "zh".to_string()],
        },
    }
}

/// Load configuration: defaults ← `config.toml` (when present) ← env.
pub fn load_config(dirs: &BaseDirs) -> Result<Config, ConfigError> {
    let file = config_file_path(dirs);
    let mut cfg = match std::fs::read_to_string(&file) {
        Ok(text) => parse_config(&text, dirs)
            .map_err(|e| ConfigError(format!("{}: {}", file.display(), e.0)))?,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => default_config(dirs),
        Err(e) => return Err(ConfigError(format!("cannot read {}: {e}", file.display()))),
    };
    // Env overrides land here, once: past load time the config is plain
    // data, and no answer derived from it drifts with the environment.
    if let Ok(key) = std::env::var("WHISPER_API_KEY")
        && !key.is_empty()
    {
        cfg.whisper.api_key = Some(key);
    }
    Ok(cfg)
}

/// Parse a `config.toml` document over the defaults for `dirs`.
pub fn parse_config(text: &str, dirs: &BaseDirs) -> Result<Config, ConfigError> {
    let file: FileConfig = toml::from_str(text).map_err(|e| ConfigError(e.to_string()))?;
    let mut cfg = default_config(dirs);

    let dir = |v: Option<String>, key: &str, home: &Path| -> Result<Option<PathBuf>, ConfigError> {
        match v {
            None => Ok(None),
            Some(s) if s.is_empty() => Err(ConfigError(format!("{key} must be a non-empty path"))),
            Some(s) => Ok(Some(expand_tilde(&s, home))),
        }
    };
    if let Some(p) = dir(file.paths.cache, "paths.cache", &dirs.home)? {
        cfg.paths.cache = p;
    }
    if let Some(p) = dir(file.paths.models, "paths.models", &dirs.home)? {
        cfg.paths.models = p;
    }
    if let Some(p) = dir(file.paths.keep, "paths.keep", &dirs.home)? {
        cfg.paths.keep = p;
    }

    let nz64 = |v: Option<u64>, key: &str| -> Result<Option<NonZeroU64>, ConfigError> {
        v.map(|n| NonZeroU64::new(n).ok_or_else(|| ConfigError(format!("{key} must be positive"))))
            .transpose()
    };
    let nz32 = |v: Option<u32>, key: &str| -> Result<Option<NonZeroU32>, ConfigError> {
        v.map(|n| NonZeroU32::new(n).ok_or_else(|| ConfigError(format!("{key} must be positive"))))
            .transpose()
    };

    if let Some(v) = nz64(file.cache.max_bytes, "cache.max_bytes")? {
        cfg.cache.max_bytes = v;
    }
    if let Some(v) = nz64(file.download.max_bytes, "download.max_bytes")? {
        cfg.download.max_bytes = v;
    }
    if let Some(v) = nz32(
        file.download.analysis_max_height,
        "download.analysis_max_height",
    )? {
        cfg.download.analysis_max_height = v;
    }
    if let Some(v) = nz32(
        file.download.sync_wait_seconds,
        "download.sync_wait_seconds",
    )? {
        cfg.download.sync_wait_seconds = v;
    }
    if let Some(v) = nz32(file.budget.max, "budget.max")? {
        cfg.budget.max = v;
    }
    if let Some(v) = nz32(
        file.budget.max_stretch_seconds,
        "budget.max_stretch_seconds",
    )? {
        cfg.budget.max_stretch_seconds = v;
    }
    if let Some(v) = file.network.allow_private_hosts {
        cfg.network.allow_private_hosts = v;
    }
    if let Some(v) = file.cookies.from_browser {
        cfg.cookies.from_browser = Some(parse_from_browser(&v)?);
    }
    if let Some(langs) = file.subtitles.preferred_langs {
        if langs.is_empty() || langs.iter().any(|l| l.is_empty()) {
            return Err(ConfigError(
                "subtitles.preferred_langs must be a non-empty list of non-empty language \
                 prefixes"
                    .into(),
            ));
        }
        cfg.subtitles.preferred_langs = langs;
    }

    if let Some(m) = file.whisper.model {
        if m.is_empty() {
            return Err(ConfigError("whisper.model must be non-empty".into()));
        }
        cfg.whisper.model = match WhisperModel::parse(&m)? {
            WhisperModel::File(p) => WhisperModel::File(expand_tilde(
                p.to_str().expect("came from a &str"),
                &dirs.home,
            )),
            named => named,
        };
    }
    if let Some(b) = file.whisper.backend {
        cfg.whisper.backend = Some(Backend::parse(&b)?);
    }
    if let Some(k) = file.whisper.api_key {
        cfg.whisper.api_key = Some(k);
    }
    if let Some(u) = file.whisper.api_url {
        cfg.whisper.api_url = u;
    }
    if let Some(p) = file.whisper.prompt {
        if p.is_empty() {
            return Err(ConfigError(
                "whisper.prompt must be non-empty — unset it for the derived default".into(),
            ));
        }
        cfg.whisper.prompt = Some(p);
    }
    if let Some(l) = file.whisper.language {
        cfg.whisper.language = Some(parse_language(&l)?);
    }

    Ok(cfg)
}

/// Expand a leading `~/` against `home`; anything else passes through.
fn expand_tilde(path: &str, home: &Path) -> PathBuf {
    match path.strip_prefix("~/") {
        Some(rest) => home.join(rest),
        None if path == "~" => home.to_path_buf(),
        None => PathBuf::from(path),
    }
}

// ── template ──

/// Render the sshd_config-style `config.toml` template for `cfg`: every key
/// present at its resolved value and commented out, so the file as written
/// changes nothing.
///
/// Two comment shapes, machine-distinguishable: `#key = value` (no space
/// after `#`) is a disabled key whose value is `cfg`'s — uncommenting it
/// verbatim is a no-op, guaranteed by the round-trip test; `# text` (space
/// after `#`) is prose, including examples for keys whose default is the
/// absence of the key.
///
/// The exhaustive destructuring below is the drift guard: a new config field
/// fails compilation here until the template learns to render it.
pub fn render_template(cfg: &Config) -> String {
    use std::fmt::Write;

    fn t(s: &str) -> String {
        toml::Value::String(s.to_string()).to_string()
    }
    fn tp(p: &Path) -> String {
        t(&p.to_string_lossy())
    }

    let Config {
        paths,
        cache,
        whisper,
        download,
        budget,
        network,
        cookies,
        subtitles,
    } = cfg;
    let Paths {
        cache: cache_path,
        models,
        keep,
    } = paths;
    let CacheCfg {
        max_bytes: cache_max_bytes,
    } = cache;
    let WhisperCfg {
        model,
        backend,
        api_key,
        api_url,
        prompt,
        language,
    } = whisper;
    let DownloadCfg {
        max_bytes: download_max_bytes,
        analysis_max_height,
        sync_wait_seconds,
    } = download;
    let BudgetCfg {
        max,
        max_stretch_seconds,
    } = budget;
    let NetworkCfg {
        allow_private_hosts,
    } = network;
    let CookiesCfg { from_browser, jar } = cookies;
    let SubtitlesCfg { preferred_langs } = subtitles;

    let model_value = match model {
        WhisperModel::Medium => t("medium"),
        WhisperModel::LargeV3 => t("large-v3"),
        WhisperModel::LargeV3Turbo => t("large-v3-turbo"),
        WhisperModel::File(p) => tp(p),
    };
    let langs_value = toml::Value::Array(
        preferred_langs
            .iter()
            .map(|l| toml::Value::String(l.clone()))
            .collect(),
    )
    .to_string();

    // An optional key: `Some` renders as a disabled `#key = value` line,
    // `None` as a prose example — uncommenting must never invent a value
    // whose default is absence.
    let opt = |out: &mut String, key: &str, value: Option<String>, example: &str| {
        match value {
            Some(v) => writeln!(out, "#{key} = {v}"),
            None => writeln!(out, "# {key} = {example}"),
        }
        .expect("writing to a String cannot fail");
    };

    fn line(out: &mut String, s: &str) {
        out.push_str(s);
        out.push('\n');
    }

    let mut out = String::new();

    line(
        &mut out,
        "# agent-video config.toml — generated by `agent-video config init`.",
    );
    line(&mut out, "#");
    line(
        &mut out,
        "# Every key is shown at its resolved default for this machine and",
    );
    line(
        &mut out,
        "# commented out: as written, this file changes nothing. Uncomment a",
    );
    line(
        &mut out,
        "# `#key = value` line and edit it to override. Keys whose default is",
    );
    line(
        &mut out,
        "# the key's absence carry a `# key = ...` example instead. Unknown",
    );
    line(
        &mut out,
        "# keys and invalid values refuse the load. Full reference:",
    );
    line(&mut out, "# docs/config.md in the source tree.");
    line(&mut out, "");
    line(&mut out, "[paths]");
    line(
        &mut out,
        "# Evictable analysis artifacts: download rips, probe docs, transcripts.",
    );
    line(&mut out, &format!("#cache = {}", tp(cache_path)));
    line(
        &mut out,
        "# Whisper weights (model + Silero VAD): persistent, never evicted.",
    );
    line(&mut out, &format!("#models = {}", tp(models)));
    line(
        &mut out,
        "# Landing directory for keep downloads; files here belong to you.",
    );
    line(&mut out, &format!("#keep = {}", tp(keep)));
    line(&mut out, "");
    line(&mut out, "[cache]");
    line(
        &mut out,
        "# LRU eviction threshold over downloads/, in bytes (default 10 GiB).",
    );
    line(&mut out, &format!("#max_bytes = {cache_max_bytes}"));
    line(&mut out, "");
    line(&mut out, "[whisper]");
    line(
        &mut out,
        "# Transcription model: medium, large-v3, large-v3-turbo (provisioned",
    );
    line(
        &mut out,
        "# by `agent-video install-whisper`), or a path to a self-supplied",
    );
    line(&mut out, "# ggml file.");
    line(&mut out, &format!("#model = {model_value}"));
    line(
        &mut out,
        "# GPU backend pin: metal, coreml, cuda, vulkan; must be compiled into",
    );
    line(
        &mut out,
        "# this build. Unset: auto-detect from compiled features.",
    );
    opt(
        &mut out,
        "backend",
        backend.map(|b| t(b.name())),
        "\"vulkan\"",
    );
    line(
        &mut out,
        "# API key for remote transcription (Groq / OpenAI-compatible);",
    );
    line(
        &mut out,
        "# the WHISPER_API_KEY environment variable overrides it at load time.",
    );
    opt(
        &mut out,
        "api_key",
        api_key.as_deref().map(t),
        "\"gsk_...\"",
    );
    line(&mut out, "# Remote transcription endpoint.");
    line(&mut out, &format!("#api_url = {}", t(api_url)));
    line(
        &mut out,
        "# Whisper conditioning text: continuation-style vocabulary bias, not",
    );
    line(
        &mut out,
        "# instructions. Unset: derived from the media title.",
    );
    opt(
        &mut out,
        "prompt",
        prompt.as_deref().map(t),
        "\"Preceding-transcript-style text with domain vocabulary.\"",
    );
    line(
        &mut out,
        "# ISO 639-1 language hint, e.g. \"zh\". Unset: auto-detect.",
    );
    opt(&mut out, "language", language.as_deref().map(t), "\"zh\"");
    line(&mut out, "");
    line(&mut out, "[download]");
    line(
        &mut out,
        "# Hard refusal ceiling for any single download, in bytes (default 2 GiB).",
    );
    line(&mut out, &format!("#max_bytes = {download_max_bytes}"));
    line(
        &mut out,
        "# Frame-height ceiling for analysis rips (grid quality ceiling).",
    );
    line(
        &mut out,
        &format!("#analysis_max_height = {analysis_max_height}"),
    );
    line(
        &mut out,
        "# Seconds a confirmed download may block the call before detaching.",
    );
    line(
        &mut out,
        &format!("#sync_wait_seconds = {sync_wait_seconds}"),
    );
    line(&mut out, "");
    line(&mut out, "[budget]");
    line(
        &mut out,
        "# Predicted grid count at or below which a bare call runs immediately.",
    );
    line(
        &mut out,
        "# Predicted grid count above which the plan degrades / needs consent.",
    );
    line(&mut out, &format!("#max = {max}"));
    line(
        &mut out,
        "# Max seconds one stretched grid may cover before requiring consent.",
    );
    line(
        &mut out,
        &format!("#max_stretch_seconds = {max_stretch_seconds}"),
    );
    line(&mut out, "");
    line(&mut out, "[network]");
    line(
        &mut out,
        "# Permit URL sources resolving to private/loopback addresses. Off by",
    );
    line(
        &mut out,
        "# default so a model-supplied URL cannot reach internal services (SSRF).",
    );
    line(
        &mut out,
        &format!("#allow_private_hosts = {allow_private_hosts}"),
    );
    line(&mut out, "");
    line(&mut out, "[cookies]");
    line(
        &mut out,
        "# yt-dlp --cookies-from-browser value, e.g. \"firefox\" or",
    );
    line(
        &mut out,
        "# \"chrome:Profile 1\": read cookies live from the browser store at",
    );
    line(
        &mut out,
        "# every spawn. Unset: anonymous, or the jar below when it exists.",
    );
    opt(
        &mut out,
        "from_browser",
        from_browser.as_deref().map(t),
        "\"firefox\"",
    );
    line(
        &mut out,
        "# The canonical jar is managed by `agent-video cookie import`, not",
    );
    line(&mut out, "# by this file; it lives at:");
    line(&mut out, &format!("#   {}", jar.display()));
    line(&mut out, "");
    line(&mut out, "[subtitles]");
    line(
        &mut out,
        "# Language preference for electing the promoted subtitle sidecar;",
    );
    line(
        &mut out,
        "# prefix-matched (\"en\" claims \"en-US\"), first hit wins.",
    );
    line(&mut out, &format!("#preferred_langs = {langs_value}"));

    out
}

#[cfg(test)]
pub(crate) fn test_dirs(root: &Path) -> BaseDirs {
    BaseDirs {
        home: root.to_path_buf(),
        cache: root.join("cache-base"),
        data: root.join("data-base"),
        download: root.join("downloads-base"),
        config: root.join("config-base"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    fn dirs() -> BaseDirs {
        test_dirs(Path::new("/home/u"))
    }

    /// The last ```` ```toml ```` block in a doc, verbatim between fences.
    /// The example is the first fence under the doc's own `## Complete
    /// example` heading — anchored to structure, not to global block order,
    /// so a per-key snippet added elsewhere can never silently steal the
    /// guard from the load-bearing block.
    fn extract_toml_block(doc: &str) -> Option<&str> {
        let section = &doc[doc.find("## Complete example")?..];
        let fence = section.find("```toml")?;
        let after = &section[fence + "```toml".len()..];
        let body = &after[after.find('\n')? + 1..];
        Some(&body[..body.find("```")?])
    }

    #[test]
    fn config_doc_example_parses_cleanly() {
        let doc = include_str!("../../../docs/config.md");
        let example = extract_toml_block(doc).expect("doc carries the example");
        parse_config(example, &dirs()).expect("the documented config is a valid config");
    }

    #[test]
    fn defaults_hang_off_base_dirs() {
        let cfg = default_config(&dirs());
        assert_eq!(cfg.paths.cache, Path::new("/home/u/cache-base/agent-video"));
        assert_eq!(
            cfg.paths.models,
            Path::new("/home/u/data-base/agent-video/models")
        );
        assert_eq!(cfg.paths.keep, Path::new("/home/u/downloads-base"));
        assert_eq!(cfg.budget.max.get(), 8);
        assert_eq!(cfg.budget.max_stretch_seconds.get(), 600);
        assert_eq!(cfg.download.max_bytes.get(), 2 * GIB);
        assert_eq!(cfg.download.analysis_max_height.get(), 480);
        assert_eq!(cfg.whisper.model, WhisperModel::LargeV3Turbo);
        assert_eq!(cfg.whisper.backend, None);
        assert!(
            !cfg.network.allow_private_hosts,
            "private destinations are blocked by default"
        );
    }

    #[test]
    fn empty_document_yields_defaults() {
        let cfg = parse_config("", &dirs()).expect("empty toml is valid");
        assert_eq!(cfg.cache.max_bytes.get(), 10 * GIB);
    }

    #[test]
    fn overrides_every_section() {
        let cfg = parse_config(
            r#"
            [paths]
            cache = "/tmp/c"
            models = "/tmp/m"
            keep = "/tmp/k"
            [cache]
            max_bytes = 1
            [whisper]
            model = "large-v3"
            [download]
            max_bytes = 2
            analysis_max_height = 3
            sync_wait_seconds = 4
            [budget]
            max = 6
            max_stretch_seconds = 7
            [network]
            allow_private_hosts = true
            "#,
            &dirs(),
        )
        .expect("valid config");
        assert_eq!(cfg.paths.cache, Path::new("/tmp/c"));
        assert_eq!(cfg.paths.models, Path::new("/tmp/m"));
        assert_eq!(cfg.paths.keep, Path::new("/tmp/k"));
        assert_eq!(cfg.cache.max_bytes.get(), 1);
        assert_eq!(cfg.whisper.model, WhisperModel::LargeV3);
        assert_eq!(cfg.download.max_bytes.get(), 2);
        assert_eq!(cfg.download.analysis_max_height.get(), 3);
        assert_eq!(cfg.download.sync_wait_seconds.get(), 4);
        assert_eq!(cfg.budget.max.get(), 6);
        assert_eq!(cfg.budget.max_stretch_seconds.get(), 7);
        assert!(cfg.network.allow_private_hosts);
    }

    #[test]
    fn rejects_unknown_keys() {
        let err = parse_config("[budget]\nautorun = 3\n", &dirs()).expect_err("unknown key");
        assert!(err.0.contains("autorun"), "message names the key: {err}");
    }

    #[test]
    fn rejects_zero_integers() {
        let err = parse_config("[budget]\nmax = 0\n", &dirs()).expect_err("zero is invalid");
        assert!(err.0.contains("budget.max"), "names the key: {err}");
    }

    #[test]
    fn expands_tilde_in_paths() {
        let cfg = parse_config("[paths]\nkeep = \"~/Videos\"\n", &dirs()).expect("valid");
        assert_eq!(cfg.paths.keep, Path::new("/home/u/Videos"));
    }

    #[test]
    fn model_path_escape_hatch() {
        let cfg = parse_config("[whisper]\nmodel = \"~/models/ggml-tiny.bin\"\n", &dirs())
            .expect("valid");
        assert_eq!(
            cfg.whisper.model,
            WhisperModel::File(PathBuf::from("/home/u/models/ggml-tiny.bin"))
        );
    }

    #[test]
    fn rejects_typoed_model_name_at_load() {
        let err = parse_config("[whisper]\nmodel = \"large-v3-trubo\"\n", &dirs())
            .expect_err("a typo must not become a self-supplied model file");
        assert!(err.0.contains("large-v3-trubo"), "names the value: {err}");
        assert!(err.0.contains("large-v3-turbo"), "lists the options: {err}");
    }

    #[test]
    fn model_cache_ids_are_distinct_and_filename_safe() {
        let ids = [
            WhisperModel::Medium.cache_id(),
            WhisperModel::LargeV3.cache_id(),
            WhisperModel::LargeV3Turbo.cache_id(),
            WhisperModel::File(PathBuf::from("/a/ggml.bin")).cache_id(),
            WhisperModel::File(PathBuf::from("/b/ggml.bin")).cache_id(),
        ];
        for (i, id) in ids.iter().enumerate() {
            assert!(
                id.chars().all(|c| c.is_ascii_alphanumeric() || c == '-'),
                "cache id {id:?} must be filename-safe"
            );
            for other in &ids[i + 1..] {
                assert_ne!(id, other, "distinct models must key distinct entries");
            }
        }
    }

    #[test]
    fn subtitle_preference_defaults_and_overrides() {
        let cfg = default_config(&dirs());
        assert_eq!(cfg.subtitles.preferred_langs, ["en", "zh"]);

        let cfg = parse_config("[subtitles]\npreferred_langs = [\"ja\", \"en\"]\n", &dirs())
            .expect("valid");
        assert_eq!(cfg.subtitles.preferred_langs, ["ja", "en"]);
    }

    #[test]
    fn rejects_empty_subtitle_preference() {
        let err = parse_config("[subtitles]\npreferred_langs = []\n", &dirs())
            .expect_err("empty list is invalid");
        assert!(err.0.contains("preferred_langs"), "names the key: {err}");
    }

    #[test]
    fn cookies_default_to_anonymous_with_a_config_dir_jar() {
        let cfg = default_config(&dirs());
        assert_eq!(cfg.cookies.from_browser, None);
        assert_eq!(
            cfg.cookies.jar,
            Path::new("/home/u/config-base/agent-video/cookies.txt")
        );
    }

    #[test]
    fn accepts_a_known_browser_with_profile_suffix() {
        let cfg = parse_config("[cookies]\nfrom_browser = \"chrome:Profile 1\"\n", &dirs())
            .expect("valid");
        assert_eq!(
            cfg.cookies.from_browser.as_deref(),
            Some("chrome:Profile 1")
        );
    }

    #[test]
    fn accepts_a_browser_this_build_never_heard_of() {
        // The browser list is yt-dlp's, and yt-dlp updates independently
        // of this binary: an unlisted-but-shaped name must load, not be
        // rejected by a stale snapshot.
        let cfg = parse_config("[cookies]\nfrom_browser = \"ladybird\"\n", &dirs()).expect("valid");
        assert_eq!(cfg.cookies.from_browser.as_deref(), Some("ladybird"));
    }

    #[test]
    fn rejects_a_malformed_from_browser() {
        for junk in ["", ":Profile 1", "Fire Fox", "fire/fox"] {
            let err = parse_config(&format!("[cookies]\nfrom_browser = {junk:?}\n"), &dirs())
                .expect_err("a value with no browser-name shape must fail the load");
            assert!(err.0.contains("from_browser"), "names the key: {err}");
        }
    }

    #[test]
    fn accepts_iso639_language_hint() {
        let cfg = parse_config("[whisper]\nlanguage = \"zh\"\n", &dirs()).expect("valid");
        assert_eq!(cfg.whisper.language.as_deref(), Some("zh"));
    }

    #[test]
    fn rejects_malformed_language() {
        for junk in ["", "ZH", "auto", "zh-CN", "中文"] {
            let err = parse_config(&format!("[whisper]\nlanguage = {junk:?}\n"), &dirs())
                .expect_err("junk language must fail the load");
            assert!(err.0.contains("whisper.language"), "names the key: {err}");
        }
    }

    #[test]
    fn rejects_empty_prompt() {
        let err = parse_config("[whisper]\nprompt = \"\"\n", &dirs())
            .expect_err("empty prompt is invalid");
        assert!(err.0.contains("whisper.prompt"), "names the key: {err}");
    }

    #[test]
    fn rejects_uncompiled_backend() {
        let err = parse_config("[whisper]\nbackend = \"cuda\"\n", &dirs())
            .expect_err("cuda is not compiled into test builds");
        assert!(err.0.contains("not compiled"), "explains why: {err}");
    }

    #[test]
    fn rejects_unknown_backend() {
        let err = parse_config("[whisper]\nbackend = \"tpu\"\n", &dirs()).expect_err("unknown");
        assert!(err.0.contains("tpu"), "names the value: {err}");
    }

    /// Strip the leading `#` from every disabled-key line (`#key = value`,
    /// no space after `#`), leaving prose comments (`# text`) alone — the
    /// mechanical inverse of the template's comment convention.
    fn uncomment(template: &str) -> String {
        template
            .lines()
            .map(|l| match l.strip_prefix('#') {
                Some(rest) if rest.starts_with(|c: char| c.is_ascii_lowercase()) => rest,
                _ => l,
            })
            .fold(String::new(), |mut acc, l| {
                acc.push_str(l);
                acc.push('\n');
                acc
            })
    }

    #[test]
    fn template_is_inert_as_written() {
        let d = dirs();
        let cfg = parse_config(&render_template(&default_config(&d)), &d)
            .expect("the template must parse as-is");
        assert_eq!(
            cfg,
            default_config(&d),
            "a fully-commented template is a no-op"
        );
    }

    #[test]
    fn uncommented_template_round_trips_to_defaults() {
        let d = dirs();
        let text = uncomment(&render_template(&default_config(&d)));
        assert!(
            text.contains("\nmax_stretch_seconds = "),
            "uncommenting must actually enable keys:\n{text}"
        );
        let cfg = parse_config(&text, &d).expect("every uncommented default must be valid");
        assert_eq!(
            cfg,
            default_config(&d),
            "uncommenting a default line must be a no-op"
        );
    }

    #[test]
    fn template_renders_set_optionals_as_disabled_keys() {
        let d = dirs();
        let mut cfg = default_config(&d);
        cfg.whisper.backend = Some(Backend::Cuda);
        cfg.whisper.api_key = Some("gsk_live".to_string());
        cfg.whisper.prompt = Some("Rust 所有权与生命周期。".to_string());
        cfg.whisper.language = Some("zh".to_string());
        cfg.cookies.from_browser = Some("chrome:Profile 1".to_string());
        let text = render_template(&cfg);
        for needle in [
            "#backend = \"cuda\"",
            "#api_key = \"gsk_live\"",
            "#prompt = \"Rust 所有权与生命周期。\"",
            "#language = \"zh\"",
            "#from_browser = \"chrome:Profile 1\"",
        ] {
            assert!(text.contains(needle), "missing {needle:?} in:\n{text}");
        }
    }
}
