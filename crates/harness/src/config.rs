// SPDX-License-Identifier: Apache-2.0
//! Harness configuration: `config.toml`.
//!
//! Defaults come from the `dirs` crate; a file at
//! `<config_dir>/agent-video/config.toml` (or `$AGENT_VIDEO_CONFIG`) overrides
//! any key. Invalid values fail loudly at load time — a misconfigured server
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
#[derive(Debug, Clone)]
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
#[derive(Debug, Clone)]
pub struct SubtitlesCfg {
    /// Language preference for the election: prefix-matched (`"en"` claims
    /// `"en-US"`), first hit wins, no hit falls back to the first
    /// candidate. Never empty — validation refuses an empty list.
    pub preferred_langs: Vec<String>,
}

/// The three path roles, kept distinct.
#[derive(Debug, Clone)]
pub struct Paths {
    /// Evictable analysis artifacts: download rips, probe docs, transcripts.
    pub cache: PathBuf,
    /// Whisper weights: persistent, LRU-exempt.
    pub models: PathBuf,
    /// `dl-keep-*` landing directory; files there belong to the user.
    pub keep: PathBuf,
}

#[derive(Debug, Clone, Copy)]
pub struct CacheCfg {
    /// LRU eviction threshold over `downloads/`.
    pub max_bytes: NonZeroU64,
}

#[derive(Debug, Clone)]
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
}

impl WhisperCfg {
    /// Default Groq Whisper endpoint.
    pub const DEFAULT_API_URL: &str = "https://api.groq.com/openai/v1/audio/transcriptions";
}

#[derive(Debug, Clone, Copy)]
pub struct DownloadCfg {
    /// Hard refusal ceiling for any single download.
    pub max_bytes: NonZeroU64,
    /// Analysis rips are capped at this frame height (grid quality ceiling).
    pub analysis_max_height: NonZeroU32,
    /// How long a confirmed download may block the call before detaching.
    /// Generous by default: for an agent, one blocked call is cheaper than
    /// a chain of pending polls — each poll costs a full model round trip.
    /// Pending remains the fallback for genuinely long downloads.
    pub sync_wait_seconds: NonZeroU32,
}

/// Egress policy knobs; enforcement lives in [`crate::netguard`].
#[derive(Debug, Clone, Copy)]
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
#[derive(Debug, Clone)]
pub struct CookiesCfg {
    /// `--cookies-from-browser` value (e.g. `"firefox"`,
    /// `"chrome:Profile 1"`): read cookies live from this browser's store
    /// at every yt-dlp spawn. Set = wins over the jar.
    pub from_browser: Option<String>,
    /// Canonical Netscape jar maintained by `agent-video cookie import`;
    /// used when the file exists and `from_browser` is unset.
    pub jar: PathBuf,
}

/// Browsers yt-dlp can extract cookies from; the config value's base name
/// (before any `:profile` / `+keyring` suffix) must be one of these.
const KNOWN_BROWSERS: &[&str] = &[
    "brave", "chrome", "chromium", "edge", "firefox", "opera", "safari", "vivaldi", "whale",
];

fn parse_from_browser(value: &str) -> Result<String, ConfigError> {
    let base = value
        .split([':', '+'])
        .next()
        .expect("split yields at least one item");
    if !KNOWN_BROWSERS.contains(&base) {
        return Err(ConfigError(format!(
            "cookies.from_browser must start with one of {} — got '{value}'",
            KNOWN_BROWSERS.join(", ")
        )));
    }
    Ok(value.to_string())
}

#[derive(Debug, Clone, Copy)]
pub struct BudgetCfg {
    /// Predicted grid count at or below which a call runs immediately.
    pub auto_run: NonZeroU32,
    /// Predicted grid count above which the plan degrades / needs consent.
    pub max: NonZeroU32,
}

/// Whisper model selection: a pinned name provisioned by `install-whisper`,
/// or a self-supplied ggml file (the escape hatch).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WhisperModel {
    Medium,
    LargeV3,
    LargeV3Turbo,
    File(PathBuf),
}

impl WhisperModel {
    fn parse(s: &str) -> Self {
        match s {
            "medium" => WhisperModel::Medium,
            "large-v3" => WhisperModel::LargeV3,
            "large-v3-turbo" => WhisperModel::LargeV3Turbo,
            other => WhisperModel::File(PathBuf::from(other)),
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
    auto_run: Option<u32>,
    max: Option<u32>,
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
        },
        download: DownloadCfg {
            max_bytes: nz64(2 * GIB),
            analysis_max_height: nz32(480),
            sync_wait_seconds: nz32(90),
        },
        budget: BudgetCfg {
            auto_run: nz32(2),
            max: nz32(8),
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
    if let Some(v) = nz32(file.budget.auto_run, "budget.auto_run")? {
        cfg.budget.auto_run = v;
    }
    if let Some(v) = nz32(file.budget.max, "budget.max")? {
        cfg.budget.max = v;
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
        cfg.whisper.model = match WhisperModel::parse(&m) {
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

    #[test]
    fn defaults_hang_off_base_dirs() {
        let cfg = default_config(&dirs());
        assert_eq!(cfg.paths.cache, Path::new("/home/u/cache-base/agent-video"));
        assert_eq!(
            cfg.paths.models,
            Path::new("/home/u/data-base/agent-video/models")
        );
        assert_eq!(cfg.paths.keep, Path::new("/home/u/downloads-base"));
        assert_eq!(cfg.budget.auto_run.get(), 2);
        assert_eq!(cfg.budget.max.get(), 8);
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
            auto_run = 5
            max = 6
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
        assert_eq!(cfg.budget.auto_run.get(), 5);
        assert_eq!(cfg.budget.max.get(), 6);
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
    fn rejects_an_unknown_browser() {
        let err = parse_config("[cookies]\nfrom_browser = \"netscape\"\n", &dirs())
            .expect_err("unknown browser");
        assert!(err.0.contains("netscape"), "names the value: {err}");
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
}
