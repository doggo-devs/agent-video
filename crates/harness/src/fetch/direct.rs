// SPDX-License-Identifier: Apache-2.0
//! Direct http(s) source.
//!
//! Probes direct media links in-process (the demuxer reads the container
//! header over http without downloading the payload) and streams downloads
//! to `<dest>.part` against a hard cap, so a file at the final path is
//! always whole.

use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use super::{DownloadIo, DownloadOption, ProgressFn, Recipe, SourceInfo, estimate_label};
use crate::config::Config;
use crate::fault::Fault;
use crate::netguard::{GuardedResolver, VettedUrl};
use crate::plan::format_bytes;
use crate::probe::probe_media_url;
use crate::schema::ProbeSummary;
use crate::tokens::{Dest, Token};
use agent_video_core::types::format_ts;

/// Minimum interval between progress callbacks.
const PROGRESS_INTERVAL: Duration = Duration::from_secs(1);

/// Connection establishment bound for HEAD/GET.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(15);

/// Response-header bound: a server that takes longer to answer is hung.
const RESPONSE_TIMEOUT: Duration = Duration::from_secs(30);

/// Whole-body ceiling for a download. A termination guarantee, not stall
/// detection: ureq has no idle timeout, so a dead peer pins the thread
/// until this expires — the tracker's staleness check surfaces the stall
/// to the model long before. Sized for `download.max_bytes` on slow links.
const BODY_CEILING: Duration = Duration::from_secs(4 * 60 * 60);

/// Agent with explicit deadlines (the default agent has none, and a
/// stalled peer would otherwise pin the download thread forever) and the
/// egress-screening resolver, so every connection this client makes —
/// redirect targets included — passes the host policy.
fn agent(cfg: &Config) -> ureq::Agent {
    let config = ureq::Agent::config_builder()
        .timeout_connect(Some(CONNECT_TIMEOUT))
        .timeout_recv_response(Some(RESPONSE_TIMEOUT))
        .timeout_recv_body(Some(BODY_CEILING))
        .build();
    ureq::Agent::with_parts(
        config,
        ureq::unversioned::transport::DefaultConnector::default(),
        GuardedResolver::new(cfg),
    )
}

/// Recipe carried by direct-link download plans.
#[derive(Debug, Clone, PartialEq)]
pub struct DirectRecipe {
    /// File extension for the cache path.
    pub ext: String,
    /// Human-readable filename used when dest is keep.
    pub filename: String,
}

/// Probe a URL as a direct media link.
///
/// Any failure — an HTML page the demuxer rejects, a DNS error, a timeout —
/// is returned with its real reason; libav cannot reliably tell "not media"
/// from "network trouble", so this source never claims that distinction.
/// The chain in [`super::probe_url`] falls through to yt-dlp either way and
/// only surfaces this error when yt-dlp fails too.
pub fn probe(url: &VettedUrl, cfg: &Config) -> Result<SourceInfo, Fault> {
    let mut probe = probe_media_url(url.as_str())?;
    if probe.size_bytes.is_none() {
        probe.size_bytes = head_content_length(url, cfg);
    }
    Ok(direct_info(url.as_str(), &probe, cfg))
}

/// Build the source info of a direct media link: one file, one keep tier.
pub fn direct_info(url: &str, probe: &ProbeSummary, cfg: &Config) -> SourceInfo {
    let recipe = direct_identity(url, probe);
    let size = probe.size_bytes;
    let size_text = match size {
        Some(n) => format_bytes(n),
        None => "size unknown".to_string(),
    };
    let label = format!("the file ({size_text})");
    SourceInfo {
        headline: format!(
            "Direct media link: {}x{}, {}, {size_text}. Nothing is downloaded yet.",
            probe.width,
            probe.height,
            format_ts(probe.duration as f32)
        ),
        options: vec![
            DownloadOption {
                token: Token::new("dl-analysis"),
                dest: Dest::Cache,
                label: label.clone(),
                effect: format!(
                    "download the file ({}) into the analysis cache (evictable) and analyze \
                     from there",
                    estimate_label(size)
                ),
                estimated_bytes: size,
                recipe: Recipe::Direct(recipe.clone()),
            },
            DownloadOption {
                token: Token::new(format!("dl-keep-{}", probe.height)),
                dest: Dest::Keep,
                label,
                effect: format!(
                    "save the file ({}) to {} for the user",
                    estimate_label(size),
                    cfg.paths.keep.join(&recipe.filename).display()
                ),
                estimated_bytes: size,
                recipe: Recipe::Direct(recipe),
            },
        ],
    }
}

/// Execute a direct download plan into cache or keep space.
pub fn download(
    url: &VettedUrl,
    recipe: &DirectRecipe,
    io: DownloadIo<'_>,
    cfg: &Config,
    progress: ProgressFn<'_>,
) -> Result<PathBuf, Fault> {
    let dest = match io {
        DownloadIo::Cache { dir, key } => dir.join(format!("{key}.{}", recipe.ext)),
        DownloadIo::Keep { dir } => available_path(&dir.join(&recipe.filename))?,
    };
    download_direct(url, &dest, cfg, progress)?;
    Ok(dest)
}

// ── link identity ──

/// Best-effort extension from a demuxer format name (comma-list).
pub fn ext_from_container(container: Option<&str>) -> String {
    let first = container
        .and_then(|c| c.split(',').next())
        .unwrap_or("")
        .trim();
    match first {
        "matroska" => "mkv",
        "webm" => "webm",
        "mov" | "mp4" => "mp4",
        "mpegts" => "ts",
        "avi" => "avi",
        "flv" => "flv",
        "asf" => "wmv",
        "ogg" => "ogg",
        other if is_ext_shaped(other) => other,
        _ => "mp4",
    }
    .to_string()
}

/// `[a-z0-9]{2,5}` — plausible as a file extension.
fn is_ext_shaped(s: &str) -> bool {
    (2..=5).contains(&s.len())
        && s.bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit())
}

/// Extension plus a keep-dir filename for a direct media link.
pub fn direct_identity(url: &str, probe: &ProbeSummary) -> DirectRecipe {
    let base = url_basename(url)
        .map(|b| sanitize_filename(&b))
        .unwrap_or_default();
    let url_ext = Path::new(&base)
        .extension()
        .and_then(|e| e.to_str())
        .map(str::to_lowercase)
        .unwrap_or_default();
    let ext = if is_ext_shaped(&url_ext) {
        url_ext.clone()
    } else {
        ext_from_container(probe.container.as_deref())
    };
    let filename = if base.is_empty() || base == format!(".{url_ext}") {
        format!("download.{ext}")
    } else if url_ext.is_empty() {
        format!("{base}.{ext}")
    } else {
        base
    };
    DirectRecipe { ext, filename }
}

/// Last path segment of a URL, percent-decoded; `None` when the URL does
/// not parse as a `Uri` (ureq would reject it at download time anyway).
/// Malformed escapes pass through verbatim and invalid UTF-8 decodes
/// lossily — the result is a best-effort filename, and `sanitize_filename`
/// guards it downstream.
fn url_basename(url: &str) -> Option<String> {
    let uri: ureq::http::Uri = url.parse().ok()?;
    let segment = uri.path().rsplit('/').next().unwrap_or("");
    Some(
        percent_encoding::percent_decode_str(segment)
            .decode_utf8_lossy()
            .into_owned(),
    )
}

/// Strip path separators, reserved punctuation, and control characters.
fn sanitize_filename(name: &str) -> String {
    name.chars()
        .filter(|c| !matches!(c, '/' | '\\' | '<' | '>' | ':' | '"' | '|' | '?' | '*'))
        .filter(|c| !c.is_control())
        .collect::<String>()
        .trim()
        .to_string()
}

/// First non-colliding variant of `p`: `name.ext`, `name (1).ext`, …
pub fn available_path(p: &Path) -> Result<PathBuf, Fault> {
    let dir = p.parent().unwrap_or(Path::new("."));
    let ext = p.extension().and_then(|e| e.to_str());
    let stem = p.file_stem().and_then(|s| s.to_str()).unwrap_or("download");
    for n in 0..100 {
        let candidate = if n == 0 {
            p.to_path_buf()
        } else {
            match ext {
                Some(ext) => dir.join(format!("{stem} ({n}).{ext}")),
                None => dir.join(format!("{stem} ({n})")),
            }
        };
        if !candidate.exists() {
            return Ok(candidate);
        }
    }
    Err(Fault::environment(format!(
        "could not find a free filename for {} in {}",
        p.display(),
        dir.display()
    )))
}

// ── plumbing ──

/// Content-Length via HEAD, or `None` when the server hides it.
pub fn head_content_length(url: &VettedUrl, cfg: &Config) -> Option<u64> {
    let res = agent(cfg).head(url.as_str()).call().ok()?;
    res.body().content_length().filter(|&len| len > 0)
}

/// Download `url` to `dest`, refusing to exceed `max_bytes`: known oversize
/// is refused before the first byte, unknown-length streams are cut off
/// mid-flight. The `.part` file never survives a failure.
pub fn download_direct(
    url: &VettedUrl,
    dest: &Path,
    cfg: &Config,
    progress: ProgressFn<'_>,
) -> Result<u64, Fault> {
    let max_bytes = cfg.download.max_bytes.get();
    let mut res = agent(cfg).get(url.as_str()).call().map_err(|e| match e {
        // A definitive 4xx is the URL's fault; everything else — 5xx,
        // transport, DNS — may pass.
        ureq::Error::StatusCode(code) if (400..500).contains(&code) => {
            Fault::bad_input(format!("download failed: GET {url} → HTTP {code}"))
        }
        ureq::Error::StatusCode(code) => {
            Fault::transient(format!("download failed: GET {url} → HTTP {code}"))
        }
        other => Fault::transient(format!("download failed: GET {url} → {other}")),
    })?;
    let total = res.body().content_length().filter(|&len| len > 0);
    if let Some(total) = total
        && total > max_bytes
    {
        return Err(Fault::bad_input(format!(
            "the file is {}, past the {} ceiling (download.max_bytes) — download it manually \
             and pass the local path as source",
            format_bytes(total),
            format_bytes(max_bytes)
        )));
    }

    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| Fault::environment(format!("cannot create {}: {e}", parent.display())))?;
    }
    let part = part_path(dest);
    let result = stream_to_part(&mut res, &part, max_bytes, total, progress);
    match result {
        Ok(bytes) => {
            std::fs::rename(&part, dest).map_err(|e| {
                Fault::environment(format!("cannot finalize {}: {e}", dest.display()))
            })?;
            Ok(bytes)
        }
        Err(e) => {
            let _ = std::fs::remove_file(&part);
            Err(e)
        }
    }
}

/// Per-writer staging path for `dest`. The cache directory is shared across
/// server processes and cache keys are deterministic, so a fixed `.part`
/// name would let two processes interleave writes into one file and rename
/// the corruption into the cache. A unique name gives each writer sole
/// ownership of its bytes; the final rename is atomic per writer and
/// last-writer-wins is safe because equal keys mean equal recipes.
fn part_path(dest: &Path) -> PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static PART_SEQ: AtomicU64 = AtomicU64::new(0);
    let seq = PART_SEQ.fetch_add(1, Ordering::Relaxed);
    PathBuf::from(format!(
        "{}.{}-{seq}.part",
        dest.display(),
        std::process::id()
    ))
}

fn stream_to_part(
    res: &mut ureq::http::Response<ureq::Body>,
    part: &Path,
    max_bytes: u64,
    total: Option<u64>,
    progress: ProgressFn<'_>,
) -> Result<u64, Fault> {
    let mut file = std::fs::File::create(part)
        .map_err(|e| Fault::environment(format!("cannot write {}: {e}", part.display())))?;
    let mut reader = res.body_mut().as_reader();
    let mut buf = [0u8; 64 * 1024];
    let mut bytes: u64 = 0;
    let mut last_report = Instant::now();
    loop {
        let n = reader
            .read(&mut buf)
            .map_err(|e| Fault::transient(format!("download failed mid-stream: {e}")))?;
        if n == 0 {
            break;
        }
        bytes += n as u64;
        if bytes > max_bytes {
            return Err(Fault::bad_input(format!(
                "download aborted at {}: the stream exceeded the {} ceiling \
                 (download.max_bytes) — download it manually",
                format_bytes(bytes),
                format_bytes(max_bytes)
            )));
        }
        file.write_all(&buf[..n])
            .map_err(|e| Fault::environment(format!("cannot write {}: {e}", part.display())))?;
        if last_report.elapsed() >= PROGRESS_INTERVAL {
            last_report = Instant::now();
            progress(bytes, total);
        }
    }
    progress(bytes, total.or(Some(bytes)));
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn probe_with(container: Option<&str>, height: u32, size: u64) -> ProbeSummary {
        ProbeSummary {
            duration: 90.0,
            width: 1280,
            height,
            size_bytes: (size > 0).then_some(size),
            container: container.map(String::from),
            subtitles: vec![],
        }
    }

    #[test]
    fn maps_container_names_to_extensions() {
        assert_eq!(ext_from_container(Some("matroska,webm")), "mkv");
        assert_eq!(ext_from_container(Some("mov,mp4,m4a,3gp,3g2,mj2")), "mp4");
        assert_eq!(ext_from_container(Some("asf")), "wmv");
        assert_eq!(ext_from_container(Some("weird_thing")), "mp4");
        assert_eq!(ext_from_container(None), "mp4");
    }

    #[test]
    fn takes_extension_from_url_path() {
        let r = direct_identity(
            "https://cdn.example.com/videos/talk.MKV?sig=abc",
            &probe_with(Some("matroska,webm"), 720, 0),
        );
        assert_eq!(r.ext, "mkv");
        assert_eq!(r.filename, "talk.MKV");
    }

    #[test]
    fn decodes_percent_escapes_in_filename() {
        let r = direct_identity(
            "https://ex.com/My%20Talk.mp4",
            &probe_with(Some("mov,mp4"), 720, 0),
        );
        assert_eq!(r.filename, "My Talk.mp4");
    }

    #[test]
    fn falls_back_to_container_extension() {
        let r = direct_identity(
            "https://ex.com/stream",
            &probe_with(Some("matroska,webm"), 720, 0),
        );
        assert_eq!(r.ext, "mkv");
        assert_eq!(r.filename, "stream.mkv");
    }

    #[test]
    fn names_pathless_urls_download() {
        let r = direct_identity("https://ex.com/", &probe_with(Some("mov,mp4"), 720, 0));
        assert_eq!(r.filename, "download.mp4");
    }

    #[test]
    fn sanitizes_reserved_characters() {
        let r = direct_identity(
            "https://ex.com/a%3Cb%3Ec.mp4",
            &probe_with(Some("mov,mp4"), 720, 0),
        );
        assert_eq!(r.filename, "abc.mp4");
    }

    #[test]
    fn offers_analysis_and_keep_tiers() {
        let cfg = crate::config::default_config(&crate::config::test_dirs(Path::new("/home/u")));
        let info = direct_info(
            "https://ex.com/v.mp4",
            &probe_with(Some("mov,mp4"), 720, 100 * 1024 * 1024),
            &cfg,
        );
        assert_eq!(info.options.len(), 2);
        assert_eq!(info.options[0].token, Token::new("dl-analysis"));
        assert_eq!(info.options[0].dest, Dest::Cache);
        assert_eq!(info.options[1].token, Token::new("dl-keep-720"));
        assert_eq!(info.options[1].dest, Dest::Keep);
        assert!(info.headline.contains("1280x720"), "{}", info.headline);
    }

    #[test]
    fn part_paths_are_writer_unique() {
        let dest = Path::new("/cache/downloads/k.mp4");
        let a = part_path(dest);
        assert!(a.to_string_lossy().ends_with(".part"));
        assert!(
            a.to_string_lossy().starts_with("/cache/downloads/k.mp4."),
            "stays next to its destination: {a:?}"
        );
        // Two writers (even in one process) must not share a staging file.
        let b = part_path(dest);
        assert_ne!(a, b);
    }

    #[test]
    fn refuses_loopback_at_the_resolver() {
        // The resolver rejects before any connect; no packet leaves.
        let cfg = crate::config::default_config(&crate::config::test_dirs(Path::new("/home/u")));
        let err = agent(&cfg)
            .get("http://127.0.0.1:9/x")
            .call()
            .expect_err("loopback must be refused");
        assert!(err.to_string().contains("loopback"), "{err}");
    }

    #[test]
    fn available_path_appends_counter() {
        let dir = std::env::temp_dir().join(format!("av-direct-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("mkdir");
        let p = dir.join("v.mp4");
        std::fs::write(&p, b"x").expect("write");
        let next = available_path(&p).expect("free name");
        assert_eq!(next, dir.join("v (1).mp4"));
        let _ = std::fs::remove_file(&p);
    }
}
