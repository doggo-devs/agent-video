// SPDX-License-Identifier: Apache-2.0
//! Disk cache under `paths.cache`.
//!
//! `downloads/` holds analysis rips (LRU by size), `probe/` holds probe
//! documents and `transcripts/` whisper output (both evicted by age);
//! mtime is the LRU clock, disk is the index — no manifest to corrupt.
//! Staging names — what the sweep must spare or collect — are classified
//! by [`naming`]'s predicates, the same module the writers construct them
//! from.

pub mod naming;

use std::fs;
use std::fs::File;
use std::io::{ErrorKind, Write};
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

/// Staging entries (`.part` files and `.staging-…` directories) older than
/// this are removed: their writer crashed and no owner is left.
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

    /// Persist a probe document; the caller must have verified file
    /// stability. Staged as `.part` and renamed like every other cache
    /// write, so a reader never sees a torn document and a crashed
    /// writer's debris is age-swept.
    pub fn put_probe(&self, fingerprint: &ContentId, doc: &str) -> std::io::Result<()> {
        fs::create_dir_all(&self.probe_dir)?;
        let dest = self.probe_path(fingerprint);
        let part = naming::part_path(&dest);
        fs::write(&part, doc)
            .and_then(|()| fs::rename(&part, &dest))
            .inspect_err(|_| {
                let _ = fs::remove_file(&part);
            })
    }

    fn probe_path(&self, fingerprint: &ContentId) -> PathBuf {
        self.probe_dir.join(format!("{fingerprint}.json"))
    }

    // ── media titles ──

    /// Record the media title learned from the source's probe document at
    /// download time, keyed by content: any later caller holding the same
    /// bytes recovers the *first-sighted* title — a stable-per-content
    /// value, not a pure function of the bytes — for whisper prompt
    /// derivation ([`crate::transcribe::effective_prompt`]) with no URL
    /// in hand. Write-once, first writer wins: the title exists only to
    /// derive a stable conditioning prompt, and the first sighted title
    /// serves that end as well as any later one; freezing it removes the
    /// re-download-under-a-new-title → new transcript id →
    /// re-transcription fragmentation window. A repeated put is a
    /// successful no-op. Lives beside the probe documents and ages out
    /// with them.
    pub fn put_title(&self, fingerprint: &ContentId, title: &str) -> std::io::Result<()> {
        fs::create_dir_all(&self.probe_dir)?;
        match File::create_new(self.title_path(fingerprint)) {
            Ok(mut f) => f.write_all(title.as_bytes()),
            Err(e) if e.kind() == ErrorKind::AlreadyExists => Ok(()),
            Err(e) => Err(e),
        }
    }

    /// The recorded title for this content, refreshing its age clock;
    /// `None` when nothing was recorded (local files, pre-title rips).
    pub fn get_title(&self, fingerprint: &ContentId) -> Option<String> {
        let p = self.title_path(fingerprint);
        let title = fs::read_to_string(&p).ok()?;
        touch(&p);
        Some(title)
    }

    fn title_path(&self, fingerprint: &ContentId) -> PathBuf {
        self.probe_dir.join(format!("{fingerprint}{TITLE_SUFFIX}"))
    }

    // ── transcripts ──

    /// Target path for a whisper transcript, keyed by content fingerprint
    /// × transcript identity ([`crate::transcribe::transcript_id`]: engine
    /// × model × conditioning): cues produced under one recipe must never
    /// answer for another's, or a model upgrade — or a prompt/language
    /// change — keeps serving the old recipe's output indefinitely —
    /// every hit refreshes the age clock.
    pub fn transcript_path(&self, fingerprint: &ContentId, transcript_id: &str) -> PathBuf {
        self.transcripts_dir
            .join(format!("{fingerprint}.{transcript_id}.srt"))
    }

    /// Locate a finished transcript for this content × recipe; hits refresh
    /// the transcript's age clock. Its title sidecar — the input its id
    /// derives from — needs no bump here: the sweep itself never expires
    /// a sidecar while a transcript for its content lives ([`Cache::evict`]),
    /// so no read path can reopen the sidecar-gone-transcript-alive window.
    pub fn find_transcript(&self, fingerprint: &ContentId, transcript_id: &str) -> Option<PathBuf> {
        let p = self.transcript_path(fingerprint, transcript_id);
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

    /// Locate the finished media download for a URL key; hits refresh the
    /// LRU clock.
    ///
    /// A key's media file has one spelling — `{key}.{ext}` with a single
    /// dot-free extension outside [`SUBTITLE_EXTS`] — exactly what the
    /// fetchers publish. Everything else is excluded by shape, not by
    /// blacklist: staging files (`….<writer>.part`) and promotion temps
    /// (`….promote`) carry extra dots, sidecars carry subtitle
    /// extensions. When re-downloads under different containers leave
    /// rivals (`k.mp4` beside `k.webm`), the newest mtime wins
    /// deterministically — and the winning touch keeps it winning while
    /// the loser ages toward eviction.
    pub fn find_download(&self, key: &str) -> Option<PathBuf> {
        let entries = fs::read_dir(&self.downloads_dir).ok()?;
        let prefix = format!("{key}.");
        let (_, p) = entries
            .flatten()
            .filter_map(|entry| {
                let name = entry.file_name();
                let ext = name.to_str()?.strip_prefix(&prefix)?;
                let is_media =
                    !ext.is_empty() && !ext.contains('.') && !SUBTITLE_EXTS.contains(&ext);
                let meta = entry.metadata().ok()?;
                (is_media && meta.is_file()).then(|| {
                    let mtime = meta.modified().unwrap_or(SystemTime::UNIX_EPOCH);
                    (mtime, entry.path())
                })
            })
            .max()?;
        touch(&p);
        Some(p)
    }

    // ── eviction ──

    /// Evict downloads down to `max_bytes` (oldest first, sparing in-flight
    /// `.part` files and anything younger than [`EVICT_GRACE`]), drop probe
    /// documents / transcripts past [`PROBE_MAX_AGE`] — except a title
    /// sidecar whose content still has a living transcript, since the
    /// transcript's cache id derives from it — and clear staging files and
    /// directories abandoned by crashed writers; returns removed paths.
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
                naming::is_part_file(&n) || naming::is_staging_dir(&n)
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

        // Age sweep, transcripts first: a transcript that dies here frees
        // its title sidecar for the probe-dir pass below. The veto makes
        // the pairing invariant mechanical — a living transcript never
        // outlives the sidecar its id derives from; the pair expires in
        // one sweep or not at all — instead of relying on every transcript
        // read path to remember a compensating sidecar bump.
        let expired = |f: &Stat| {
            now.duration_since(f.mtime)
                .is_ok_and(|age| age > PROBE_MAX_AGE)
        };
        let file_name = |f: &Stat| f.path.file_name().map(|n| n.to_string_lossy().into_owned());
        let mut living_transcripts = Vec::new();
        for f in list_with_stats(&self.transcripts_dir) {
            let Some(name) = file_name(&f) else { continue };
            if naming::is_part_file(&name) {
                continue; // in flight or crashed: sweep_stale_staging owns it
            }
            if expired(&f) && fs::remove_file(&f.path).is_ok() {
                removed.push(f.path);
            } else {
                living_transcripts.push(name);
            }
        }
        for f in list_with_stats(&self.probe_dir) {
            let Some(name) = file_name(&f) else { continue };
            if naming::is_part_file(&name) || !expired(&f) {
                continue;
            }
            let vetoed = name.strip_suffix(TITLE_SUFFIX).is_some_and(|fp| {
                let prefix = format!("{fp}.");
                living_transcripts.iter().any(|t| t.starts_with(&prefix))
            });
            if !vetoed && fs::remove_file(&f.path).is_ok() {
                removed.push(f.path);
            }
        }

        self.sweep_stale_staging(now);
        removed
    }

    /// Remove staging entries whose writers crashed, in every directory
    /// writers stage in: `.part` files (direct downloads in `downloads/`,
    /// whisper output in `transcripts/`, probe documents in `probe/`) and
    /// `.staging-…` directories (yt-dlp jobs in `downloads/`). Normal
    /// completion and failure both remove staging on the spot, so anything
    /// past [`STALE_STAGING_AGE`] has no owner left; a live writer keeps
    /// its entry's mtime fresh, so age is the ownership test.
    fn sweep_stale_staging(&self, now: SystemTime) {
        for dir in [&self.downloads_dir, &self.transcripts_dir, &self.probe_dir] {
            let Ok(entries) = fs::read_dir(dir) else {
                continue;
            };
            for entry in entries.flatten() {
                let name = entry.file_name();
                let name = name.to_string_lossy();
                if !naming::is_part_file(&name) && !naming::is_staging_dir(&name) {
                    continue;
                }
                let Ok(meta) = entry.metadata() else { continue };
                let abandoned = meta.modified().is_ok_and(|mtime| {
                    now.duration_since(mtime)
                        .is_ok_and(|age| age > STALE_STAGING_AGE)
                });
                if !abandoned {
                    continue;
                }
                let _ = if meta.is_dir() {
                    fs::remove_dir_all(entry.path())
                } else {
                    fs::remove_file(entry.path())
                };
            }
        }
    }
}

/// Suffix of the write-once title sidecar living beside probe documents;
/// the eviction sweep parses the content id back out of it to veto
/// expiring a sidecar whose transcripts still live.
const TITLE_SUFFIX: &str = ".title";

/// Refresh the LRU/age clock. Best-effort — a hit is a hit even when the
/// clock write fails — but a failed bump is a lost state transition (the
/// hot entry keeps aging toward eviction), so it is logged. `NotFound`
/// stays silent: a concurrent sweep already unlinked the entry, and there
/// is no clock left to refresh.
fn touch(p: &Path) {
    let refreshed = File::options()
        .append(true)
        .open(p)
        .and_then(|f| f.set_modified(SystemTime::now()));
    if let Err(e) = refreshed
        && e.kind() != ErrorKind::NotFound
    {
        tracing::warn!(
            path = %p.display(),
            error = %e,
            "cache clock refresh failed; the entry keeps aging toward eviction"
        );
    }
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

    fn cache(max_bytes: u64) -> (Cache, crate::tempdir::TempDir) {
        let dir = crate::tempdir::TempDir::new("av-cache");
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
    fn put_probe_publishes_atomically_leaving_no_staging() {
        let (cache, _dir) = cache(1024);
        cache.put_probe(&ContentId::new("abc"), "{}").expect("put");
        let names: Vec<String> = std::fs::read_dir(&cache.probe_dir)
            .expect("probe dir exists")
            .map(|e| e.expect("entry").file_name().to_string_lossy().into_owned())
            .collect();
        assert_eq!(names, ["abc.json"], "only the published name remains");
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
    fn find_download_rejects_multidot_names_as_media() {
        // A promotion temp (`k.srt.promote`) is neither `.part` nor a
        // subtitle name; only the one-dot-free-extension shape keeps it
        // from answering as media.
        let (cache, _dir) = cache(1024);
        let dir = cache.ensure_downloads_dir().expect("mkdir").to_path_buf();
        std::fs::write(dir.join("k1.srt.promote"), b"x").expect("write");
        assert!(cache.find_download("k1").is_none());
    }

    #[test]
    fn find_download_prefers_the_newest_rival_container() {
        // A re-download under a different container leaves two media
        // candidates; the answer must be the latest publication, not
        // whatever readdir yields first.
        let (cache, _dir) = cache(1024);
        let dir = cache.ensure_downloads_dir().expect("mkdir").to_path_buf();
        let stale = dir.join("k1.mp4");
        let current = dir.join("k1.webm");
        std::fs::write(&stale, b"x").expect("write");
        std::fs::write(&current, b"x").expect("write");
        File::options()
            .append(true)
            .open(&stale)
            .and_then(|f| f.set_modified(SystemTime::now() - Duration::from_secs(3600)))
            .expect("age the superseded rip");

        assert_eq!(cache.find_download("k1").expect("hit"), current);
        assert_eq!(
            cache.find_download("k1").expect("hit"),
            current,
            "the winning touch keeps the winner winning"
        );
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

        let current = fs::metadata(&p).and_then(|m| m.modified()).expect("mtime");
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
        // The name a real writer would stage under, straight from the
        // grammar's constructor: the sweep must recognize its own inverse.
        let staging = dir.join(naming::staging_dir_name("k"));
        std::fs::create_dir_all(&staging).expect("mkdir staging");
        std::fs::write(staging.join("k.mp4.part"), b"x").expect("write");

        cache.evict(SystemTime::now());
        assert!(staging.exists(), "a recent staging dir may be in flight");

        let long_ago = SystemTime::now() + Duration::from_secs(2 * 24 * 60 * 60);
        cache.evict(long_ago);
        assert!(!staging.exists(), "a crashed download's staging is swept");
    }

    #[test]
    fn sweeps_aged_crashed_part_files_in_every_staging_dir() {
        let (cache, _dir) = cache(1024);
        let dl = cache.ensure_downloads_dir().expect("mkdir").to_path_buf();
        // The names real writers stage under, straight from the grammar's
        // constructor: a direct download, a whisper transcript, a probe doc.
        let dl_part = naming::part_path(&dl.join("k.mp4"));
        let srt = cache.transcript_path(&ContentId::new("fp"), "m");
        std::fs::create_dir_all(srt.parent().expect("parent")).expect("mkdir");
        let srt_part = naming::part_path(&srt);
        std::fs::create_dir_all(&cache.probe_dir).expect("mkdir");
        let probe_part = naming::part_path(&cache.probe_path(&ContentId::new("fp")));
        for p in [&dl_part, &srt_part, &probe_part] {
            std::fs::write(p, b"x").expect("write");
        }

        let long_ago = SystemTime::now() + Duration::from_secs(2 * 24 * 60 * 60);
        cache.evict(long_ago);
        assert!(!dl_part.exists(), "a crashed download's .part is swept");
        assert!(!srt_part.exists(), "a crashed transcript's .part is swept");
        assert!(
            !probe_part.exists(),
            "a crashed probe write's .part is swept"
        );
    }

    #[test]
    fn spares_fresh_part_files_from_the_staleness_sweep() {
        let (cache, _dir) = cache(1024);
        let srt = cache.transcript_path(&ContentId::new("fp"), "m");
        std::fs::create_dir_all(srt.parent().expect("parent")).expect("mkdir");
        let part = naming::part_path(&srt);
        std::fs::write(&part, b"x").expect("write");

        cache.evict(SystemTime::now());
        assert!(part.exists(), "a fresh .part may be in flight");
    }

    #[test]
    fn staleness_sweep_leaves_published_entries_untouched() {
        let (cache, _dir) = cache(1024);
        let dl = cache.ensure_downloads_dir().expect("mkdir").to_path_buf();
        let media = dl.join("k.mp4");
        std::fs::write(&media, b"x").expect("write");
        let srt = cache.transcript_path(&ContentId::new("fp"), "m");
        std::fs::create_dir_all(srt.parent().expect("parent")).expect("mkdir");
        std::fs::write(&srt, b"cues").expect("write");

        // Past STALE_STAGING_AGE but inside PROBE_MAX_AGE: only staging
        // names may go, and there are none.
        let later = SystemTime::now() + STALE_STAGING_AGE + Duration::from_secs(60);
        let removed = cache.evict(later);
        assert!(removed.is_empty());
        assert!(media.exists() && srt.exists());
    }

    #[test]
    fn title_roundtrips_by_content() {
        let (cache, _dir) = cache(1024);
        assert_eq!(cache.get_title(&ContentId::new("fp")), None);
        cache
            .put_title(&ContentId::new("fp"), "Rust Ownership Explained")
            .expect("put");
        assert_eq!(
            cache.get_title(&ContentId::new("fp")).as_deref(),
            Some("Rust Ownership Explained")
        );
        assert_eq!(
            cache.get_title(&ContentId::new("other")),
            None,
            "titles answer only for their own content"
        );
    }

    #[test]
    fn later_titles_cannot_repoint_recorded_content() {
        let (cache, _dir) = cache(1024);
        cache
            .put_title(&ContentId::new("fp"), "First Sighted Title")
            .expect("first put");
        cache
            .put_title(&ContentId::new("fp"), "Retitled Upload")
            .expect("a repeated put is a successful no-op");
        assert_eq!(
            cache.get_title(&ContentId::new("fp")).as_deref(),
            Some("First Sighted Title"),
            "first writer wins: the recorded title is frozen"
        );
    }

    #[test]
    fn sweep_spares_the_title_sidecar_of_a_living_transcript() {
        let (cache, _dir) = cache(1024);
        let fp = ContentId::new("fp");
        cache
            .put_title(&fp, "Rust Ownership Explained")
            .expect("put");
        let srt = cache.transcript_path(&fp, "m-auto-abcd1234");
        std::fs::create_dir_all(srt.parent().expect("parent")).expect("mkdir");
        std::fs::write(&srt, b"cues").expect("write");
        // Age the sidecar past expiry while the transcript stays fresh: the
        // sweep itself — no read-path bump required — must keep the input
        // the transcript's id derives from.
        let title = cache.probe_dir.join(format!("{fp}{TITLE_SUFFIX}"));
        File::options()
            .append(true)
            .open(&title)
            .and_then(|f| {
                f.set_modified(SystemTime::now() - PROBE_MAX_AGE - Duration::from_secs(60))
            })
            .expect("age the sidecar");

        cache.evict(SystemTime::now());

        assert_eq!(
            cache.get_title(&fp).as_deref(),
            Some("Rust Ownership Explained"),
            "a living transcript's key premise survives the sweep"
        );
    }

    #[test]
    fn expired_transcript_takes_its_title_along_in_one_sweep() {
        let (cache, _dir) = cache(1024);
        let fp = ContentId::new("fp");
        cache
            .put_title(&fp, "Rust Ownership Explained")
            .expect("put");
        let srt = cache.transcript_path(&fp, "m-auto-abcd1234");
        std::fs::create_dir_all(srt.parent().expect("parent")).expect("mkdir");
        std::fs::write(&srt, b"cues").expect("write");

        let long_ago = SystemTime::now() + PROBE_MAX_AGE + Duration::from_secs(60);
        cache.evict(long_ago);

        assert!(!srt.exists(), "the transcript expired");
        assert_eq!(
            cache.get_title(&fp),
            None,
            "the pair expires in the same sweep: no orphaned sidecar lingers"
        );
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
