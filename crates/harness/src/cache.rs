// SPDX-License-Identifier: Apache-2.0
//! Disk cache under `paths.cache`.
//!
//! `downloads/` holds analysis rips (LRU by size), `probe/` holds probe
//! documents and `transcripts/` whisper output (both evicted by age);
//! mtime is the LRU clock, disk is the index — no manifest to corrupt.

use std::fs;
use std::fs::File;
use std::io::ErrorKind;
use std::num::NonZeroU64;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

use sha2::{Digest, Sha256};

use super::identity::{ContentId, hex};

/// Probe documents and transcripts older than this are dropped during sweeps.
pub const PROBE_MAX_AGE: Duration = Duration::from_secs(30 * 24 * 60 * 60);

/// Files younger than this are never evicted: a rip that just landed — or
/// a concurrent process's freshly renamed result — must survive the sweep
/// that its own download triggers, even when the budget is exceeded. The
/// budget is a soft target; correctness of just-published entries is not.
pub const EVICT_GRACE: Duration = Duration::from_secs(5 * 60);

/// Staging directories (crashed downloads) older than this are removed.
const STALE_STAGING_AGE: Duration = Duration::from_secs(24 * 60 * 60);

/// Subtitle sidecar extensions living next to cached downloads.
pub const SUBTITLE_EXTS: [&str; 3] = ["srt", "ass", "vtt"];

/// Stable cache key for a URL: sha256 truncated to 16 bytes hex.
pub fn url_key(url: &str) -> String {
    hex(&Sha256::digest(url.as_bytes())[..16])
}

/// Disk cache rooted at `paths.cache`; all methods create dirs lazily.
#[derive(Debug, Clone)]
pub struct Cache {
    downloads_dir: PathBuf,
    probe_dir: PathBuf,
    transcripts_dir: PathBuf,
    max_bytes: NonZeroU64,
}

impl Cache {
    pub fn new(dir: &Path, max_bytes: NonZeroU64) -> Self {
        Cache {
            downloads_dir: dir.join("downloads"),
            probe_dir: dir.join("probe"),
            transcripts_dir: dir.join("transcripts"),
            max_bytes,
        }
    }

    // ── probe documents ──

    /// Read a cached probe document, refreshing its age clock.
    pub fn get_probe(&self, fingerprint: &ContentId) -> std::io::Result<Option<String>> {
        let p = self.probe_path(fingerprint);
        match fs::read_to_string(&p) {
            Ok(text) => {
                touch(&p);
                Ok(Some(text))
            }
            Err(e) if e.kind() == ErrorKind::NotFound => Ok(None),
            Err(e) => Err(e),
        }
    }

    /// Persist a probe document; the caller must have verified file stability.
    pub fn put_probe(&self, fingerprint: &ContentId, doc: &str) -> std::io::Result<()> {
        fs::create_dir_all(&self.probe_dir)?;
        fs::write(self.probe_path(fingerprint), doc)
    }

    fn probe_path(&self, fingerprint: &ContentId) -> PathBuf {
        self.probe_dir.join(format!("{fingerprint}.json"))
    }

    // ── transcripts ──

    /// Target path for a whisper transcript, keyed by content fingerprint
    /// × model identity ([`crate::config::WhisperModel::cache_id`]): a
    /// transcript produced by one model must never answer for another, or a
    /// model upgrade keeps serving the old model's output indefinitely —
    /// every hit refreshes the age clock.
    pub fn transcript_path(&self, fingerprint: &ContentId, model_id: &str) -> PathBuf {
        self.transcripts_dir
            .join(format!("{fingerprint}.{model_id}.srt"))
    }

    /// Locate a finished transcript for this content × model; hits refresh
    /// the age clock.
    pub fn find_transcript(&self, fingerprint: &ContentId, model_id: &str) -> Option<PathBuf> {
        let p = self.transcript_path(fingerprint, model_id);
        if p.is_file() {
            touch(&p);
            Some(p)
        } else {
            None
        }
    }

    // ── downloads ──

    /// Ensure the downloads directory exists; returns it.
    pub fn ensure_downloads_dir(&self) -> std::io::Result<&Path> {
        fs::create_dir_all(&self.downloads_dir)?;
        Ok(&self.downloads_dir)
    }

    /// Locate the finished media download for a URL key (sidecars and `.part`
    /// never match); hits refresh the LRU clock.
    pub fn find_download(&self, key: &str) -> Option<PathBuf> {
        let entries = fs::read_dir(&self.downloads_dir).ok()?;
        let prefix = format!("{key}.");
        for entry in entries.flatten() {
            let name = entry.file_name();
            let Some(name) = name.to_str() else { continue };
            let is_media = name.starts_with(&prefix)
                && !name.ends_with(".part")
                && !SUBTITLE_EXTS
                    .iter()
                    .any(|ext| name.ends_with(&format!(".{ext}")));
            if is_media {
                let p = entry.path();
                touch(&p);
                return Some(p);
            }
        }
        None
    }

    // ── eviction ──

    /// Evict downloads down to `max_bytes` (oldest first, sparing in-flight
    /// `.part` files and anything younger than [`EVICT_GRACE`]), drop probe
    /// documents / transcripts past [`PROBE_MAX_AGE`], and clear staging
    /// directories abandoned by crashed downloads; returns removed paths.
    ///
    /// Accounting is honest: a file that cannot be deleted keeps counting
    /// toward the total and never appears in the removed list, so a
    /// permission problem shows up as a persistently over-budget cache
    /// instead of silently unbounded growth reported as trimmed.
    pub fn evict(&self, now: SystemTime) -> Vec<PathBuf> {
        let mut removed = Vec::new();

        // In-flight staging: another process may be mid-download. Its
        // `.part` files and staging dirs are transient, not LRU entries.
        let is_in_flight = |f: &Stat| {
            f.path.file_name().is_some_and(|n| {
                let n = n.to_string_lossy();
                n.ends_with(".part") || n.starts_with(crate::fetch::STAGING_PREFIX)
            })
        };
        let mut downloads = list_with_stats(&self.downloads_dir);
        downloads.retain(|f| !is_in_flight(f));
        let mut total: u64 = downloads.iter().map(|f| f.size).sum();
        downloads.sort_by_key(|f| f.mtime);
        for f in &downloads {
            if total <= self.max_bytes.get() {
                break;
            }
            let fresh = now
                .duration_since(f.mtime)
                .is_ok_and(|age| age < EVICT_GRACE);
            if fresh || !remove_if_unbumped(f) {
                continue;
            }
            total -= f.size;
            removed.push(f.path.clone());
        }

        for dir in [&self.probe_dir, &self.transcripts_dir] {
            for f in list_with_stats(dir) {
                let expired = now
                    .duration_since(f.mtime)
                    .is_ok_and(|age| age > PROBE_MAX_AGE);
                if expired && fs::remove_file(&f.path).is_ok() {
                    removed.push(f.path);
                }
            }
        }

        self.sweep_stale_staging(now);
        removed
    }

    /// Remove staging directories whose downloads crashed: normal completion
    /// (and failure) removes them on drop, so anything past
    /// [`STALE_STAGING_AGE`] has no owner left.
    fn sweep_stale_staging(&self, now: SystemTime) {
        let Ok(entries) = fs::read_dir(&self.downloads_dir) else {
            return;
        };
        for entry in entries.flatten() {
            let name = entry.file_name();
            let is_staging = name
                .to_string_lossy()
                .starts_with(crate::fetch::STAGING_PREFIX);
            let abandoned = entry.metadata().is_ok_and(|m| {
                m.is_dir()
                    && m.modified().is_ok_and(|mtime| {
                        now.duration_since(mtime)
                            .is_ok_and(|age| age > STALE_STAGING_AGE)
                    })
            });
            if is_staging && abandoned {
                let _ = fs::remove_dir_all(entry.path());
            }
        }
    }
}

/// Refresh the LRU/age clock; best-effort, a failed touch only ages the entry.
fn touch(p: &Path) {
    let _ = File::options().append(true).open(p).and_then(|f| {
        f.set_modified(SystemTime::now())?;
        Ok(())
    });
}

/// Remove `f` only if its mtime is still the one the sweep ranked it by: a
/// concurrent [`touch`] (another process's cache hit) moves the mtime
/// forward and vetoes the deletion. The re-check-to-unlink window that
/// remains is accepted: on Unix an open reader keeps the unlinked file
/// readable, so the worst case is a re-download, never corruption. A veto
/// keeps the file counting toward the budget, consistent with the honest
/// accounting above.
fn remove_if_unbumped(f: &Stat) -> bool {
    fs::metadata(&f.path)
        .and_then(|m| m.modified())
        .is_ok_and(|mtime| mtime == f.mtime)
        && fs::remove_file(&f.path).is_ok()
}

struct Stat {
    path: PathBuf,
    size: u64,
    mtime: SystemTime,
}

fn list_with_stats(dir: &Path) -> Vec<Stat> {
    let Ok(entries) = fs::read_dir(dir) else {
        return Vec::new();
    };
    entries
        .flatten()
        .filter_map(|entry| {
            let meta = entry.metadata().ok()?;
            meta.is_file().then(|| Stat {
                path: entry.path(),
                size: meta.len(),
                mtime: meta.modified().unwrap_or(SystemTime::UNIX_EPOCH),
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cache(max_bytes: u64) -> (Cache, PathBuf) {
        let dir = std::env::temp_dir().join(format!(
            "av-cache-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("clock")
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).expect("mkdir");
        (
            Cache::new(&dir, NonZeroU64::new(max_bytes).expect("non-zero")),
            dir,
        )
    }

    #[test]
    fn url_key_is_stable_hex() {
        let k = url_key("https://example.com/v.mp4");
        assert_eq!(k.len(), 32);
        assert_eq!(k, url_key("https://example.com/v.mp4"));
        assert_ne!(k, url_key("https://example.com/w.mp4"));
    }

    #[test]
    fn probe_roundtrips_and_misses_cleanly() {
        let (cache, _dir) = cache(1024);
        assert!(
            cache
                .get_probe(&ContentId::new("abc"))
                .expect("io")
                .is_none()
        );
        cache
            .put_probe(&ContentId::new("abc"), r#"{"duration":1}"#)
            .expect("put");
        assert_eq!(
            cache
                .get_probe(&ContentId::new("abc"))
                .expect("io")
                .as_deref(),
            Some(r#"{"duration":1}"#)
        );
    }

    #[test]
    fn find_download_skips_partials_and_sidecars() {
        let (cache, _dir) = cache(1024);
        let dir = cache.ensure_downloads_dir().expect("mkdir").to_path_buf();
        std::fs::write(dir.join("k1.mp4.part"), b"x").expect("write");
        std::fs::write(dir.join("k1.srt"), b"x").expect("write");
        assert!(cache.find_download("k1").is_none());
        std::fs::write(dir.join("k1.mp4"), b"x").expect("write");
        assert_eq!(cache.find_download("k1").expect("hit"), dir.join("k1.mp4"));
    }

    #[test]
    fn evicts_oldest_downloads_over_budget() {
        let (cache, _dir) = cache(5);
        let dir = cache.ensure_downloads_dir().expect("mkdir").to_path_buf();
        let old = dir.join("old.mp4");
        let new = dir.join("new.mp4");
        std::fs::write(&old, b"aaaa").expect("write");
        std::fs::write(&new, b"bbbb").expect("write");
        let past = SystemTime::now() - Duration::from_secs(3600);
        File::options()
            .append(true)
            .open(&old)
            .and_then(|f| f.set_modified(past))
            .expect("age the old file");

        let removed = cache.evict(SystemTime::now());
        assert_eq!(removed, vec![old.clone()]);
        assert!(!old.exists());
        assert!(new.exists());
    }

    #[test]
    fn transcripts_are_keyed_by_model() {
        let (cache, _dir) = cache(1024);
        let p = cache.transcript_path(&ContentId::new("fp"), "medium");
        std::fs::create_dir_all(p.parent().expect("parent")).expect("mkdir");
        std::fs::write(&p, b"cues").expect("write");

        assert_eq!(
            cache.find_transcript(&ContentId::new("fp"), "medium"),
            Some(p)
        );
        assert_eq!(
            cache.find_transcript(&ContentId::new("fp"), "large-v3"),
            None,
            "one model's transcript must not answer for another"
        );
    }

    #[test]
    fn evict_spares_fresh_and_in_flight_files() {
        let (cache, _dir) = cache(1);
        let dir = cache.ensure_downloads_dir().expect("mkdir").to_path_buf();
        // All three exceed the 1-byte budget; only the aged plain file may go.
        let fresh = dir.join("fresh.mp4");
        let aged = dir.join("aged.mp4");
        let part = dir.join("aged.mp4.123-0.part");
        for p in [&fresh, &aged, &part] {
            std::fs::write(p, b"xxxx").expect("write");
        }
        let past = SystemTime::now() - EVICT_GRACE - Duration::from_secs(60);
        for p in [&aged, &part] {
            File::options()
                .append(true)
                .open(p)
                .and_then(|f| f.set_modified(past))
                .expect("age the file");
        }

        let removed = cache.evict(SystemTime::now());
        assert_eq!(removed, vec![aged.clone()]);
        assert!(fresh.exists(), "files inside the grace window survive");
        assert!(part.exists(), "in-flight .part files are never LRU entries");
    }

    #[test]
    fn eviction_vetoes_a_file_touched_after_ranking() {
        // Models the cross-process race: another server's cache hit bumps
        // the mtime between the sweep's listing and the unlink. The stale
        // stat must not authorize the deletion.
        let (cache, _dir) = cache(1024);
        let dir = cache.ensure_downloads_dir().expect("mkdir").to_path_buf();
        let p = dir.join("hit.mp4");
        std::fs::write(&p, b"xxxx").expect("write");
        let past = SystemTime::now() - Duration::from_secs(3600);
        File::options()
            .append(true)
            .open(&p)
            .and_then(|f| f.set_modified(past))
            .expect("age the file");
        let stale = Stat {
            path: p.clone(),
            size: 4,
            mtime: past,
        };
        touch(&p); // the concurrent cache hit

        assert!(!remove_if_unbumped(&stale), "bumped mtime vetoes");
        assert!(p.exists(), "the just-used file survives");

        let current = fs::metadata(&p)
            .and_then(|m| m.modified())
            .expect("mtime");
        assert!(
            remove_if_unbumped(&Stat {
                path: p.clone(),
                size: 4,
                mtime: current,
            }),
            "an unbumped ranking still deletes"
        );
        assert!(!p.exists());
    }

    #[test]
    fn evict_clears_abandoned_staging_dirs() {
        let (cache, _dir) = cache(1024);
        let dir = cache.ensure_downloads_dir().expect("mkdir").to_path_buf();
        let staging = dir.join(format!("{}k-999", crate::fetch::STAGING_PREFIX));
        std::fs::create_dir_all(&staging).expect("mkdir staging");
        std::fs::write(staging.join("k.mp4.part"), b"x").expect("write");

        cache.evict(SystemTime::now());
        assert!(staging.exists(), "a recent staging dir may be in flight");

        let long_ago = SystemTime::now() + Duration::from_secs(2 * 24 * 60 * 60);
        cache.evict(long_ago);
        assert!(!staging.exists(), "a crashed download's staging is swept");
    }

    #[test]
    fn evicts_expired_probe_documents() {
        let (cache, _dir) = cache(1024);
        cache
            .put_probe(&ContentId::new("stale"), "{}")
            .expect("put");
        let p = cache.get_probe(&ContentId::new("stale")).expect("io");
        assert!(p.is_some());
        let long_ago = SystemTime::now() + PROBE_MAX_AGE + Duration::from_secs(60);
        let removed = cache.evict(long_ago);
        assert_eq!(removed.len(), 1);
        assert!(
            cache
                .get_probe(&ContentId::new("stale"))
                .expect("io")
                .is_none()
        );
    }
}
