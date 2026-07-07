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
    pub backend: Backend,
}

#[derive(Debug, Clone, Copy)]
pub struct DownloadCfg {
    /// Hard refusal ceiling for any single download.
    pub max_bytes: NonZeroU64,
    /// Analysis rips are capped at this frame height (grid quality ceiling).
    pub analysis_max_height: NonZeroU32,
    /// How long a confirmed download may block the call before detaching.
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
    File(PathBuf),
}

impl WhisperModel {
    fn parse(s: &str) -> Self {
        match s {
            "medium" => WhisperModel::Medium,
            "large-v3" => WhisperModel::LargeV3,
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
            WhisperModel::File(p) => {
                let digest = Sha256::digest(p.as_os_str().as_encoded_bytes());
                format!("file-{}", crate::identity::hex(&digest[..8]))
            }
        }
    }
}

/// Whisper compute backend, selectable among compiled-in backends only:
/// an uncompiled backend cannot be lit up at runtime, so naming one is a
/// config error, not a fallback.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    Cpu,
    Metal,
    Coreml,
    Cuda,
    Vulkan,
}

impl Backend {
    fn parse(s: &str) -> Result<Self, ConfigError> {
        let all = [
            ("cpu", Backend::Cpu, true),
            ("metal", Backend::Metal, cfg!(feature = "whisper-metal")),
            ("coreml", Backend::Coreml, cfg!(feature = "whisper-coreml")),
            ("cuda", Backend::Cuda, cfg!(feature = "whisper-cuda")),
            ("vulkan", Backend::Vulkan, cfg!(feature = "whisper-vulkan")),
        ];
        let Some(&(_, backend, compiled)) = all.iter().find(|(name, ..)| *name == s) else {
            return Err(ConfigError(format!(
                "whisper.backend must be one of cpu, metal, coreml, cuda, vulkan — got '{s}'"
            )));
        };
        if !compiled {
            let available: Vec<&str> = all
                .iter()
                .filter(|&&(_, _, c)| c)
                .map(|&(name, ..)| name)
                .collect();
            return Err(ConfigError(format!(
                "whisper.backend '{s}' is not compiled into this build; available: {}",
                available.join(", ")
            )));
        }
        Ok(backend)
    }

    /// Whether whisper should attempt GPU offload.
    pub fn use_gpu(self) -> bool {
        self != Backend::Cpu
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
            model: WhisperModel::Medium,
            backend: Backend::Cpu,
        },
        download: DownloadCfg {
            max_bytes: nz64(2 * GIB),
            analysis_max_height: nz32(480),
            sync_wait_seconds: nz32(20),
        },
        budget: BudgetCfg {
            auto_run: nz32(2),
            max: nz32(8),
        },
        network: NetworkCfg {
            allow_private_hosts: false,
        },
    }
}

/// Load configuration: defaults ← `config.toml` (when present).
pub fn load_config(dirs: &BaseDirs) -> Result<Config, ConfigError> {
    let file = config_file_path(dirs);
    match std::fs::read_to_string(&file) {
        Ok(text) => parse_config(&text, dirs)
            .map_err(|e| ConfigError(format!("{}: {}", file.display(), e.0))),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(default_config(dirs)),
        Err(e) => Err(ConfigError(format!("cannot read {}: {e}", file.display()))),
    }
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
        cfg.whisper.backend = Backend::parse(&b)?;
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
        assert_eq!(cfg.whisper.model, WhisperModel::Medium);
        assert_eq!(cfg.whisper.backend, Backend::Cpu);
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
