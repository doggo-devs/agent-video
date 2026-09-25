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
pub mod source;

use std::fs;
use std::fs::File;
use std::io::ErrorKind;
use std::num::NonZeroU64;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

use self::source::SourceId;
use super::identity::ContentId;

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

/// Stable cache key for a source spelling: the canonical [`SourceId`]'s
/// hash — the same identity the token registry keys offer sets on, so the
/// two sides cannot disagree about which spellings share an entry.
pub fn url_key(url: &str) -> String {
    SourceId::from_spelling(url).cache_key()
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
    /// download time, keyed by content, so a later caller holding the same
    /// bytes but no URL can derive the same whisper prompt
    /// ([`crate::transcribe::effective_prompt`]). First writer wins and a
    /// repeated put is a no-op: a frozen title keeps the transcript id
    /// stable across re-downloads. Ages out with the probe documents.
    ///
    /// Publication is atomic (docs/shared-cache.md, Pattern 2a): the title
    /// is staged to a writer-unique `.part` and hard-linked into place, so
    /// readers never see a torn sidecar and a crash leaves only staging. A
    /// blank title records nothing; a blank on-disk winner (a torn write
    /// from releases before this scheme) is replaced. Without hard links
    /// (FAT, some network mounts) the publish falls back to a rename:
    /// still atomic, but last writer wins inside the race window.
    pub fn put_title(&self, fingerprint: &ContentId, title: &str) -> std::io::Result<()> {
        if title.trim().is_empty() {
            return Ok(());
        }
        fs::create_dir_all(&self.probe_dir)?;
        let dest = self.title_path(fingerprint);
        let part = naming::part_path(&dest);
        let published = fs::write(&part, title)
            .and_then(|()| publish_write_once(&part, &dest, |p, d| fs::hard_link(p, d)));
        // A successful rename fallback already consumed the staging;
        // `NotFound` here is the harmless residue of that path.
        let _ = fs::remove_file(&part);
        published
    }

    /// The recorded title for this content, refreshing its age clock;
    /// `None` when nothing was recorded (local files, pre-title rips).
    /// A blank sidecar — a legacy torn write — is not a title: it answers
    /// `None` and its clock is left alone, so it ages out instead of
    /// living forever as an unreadable winner.
    pub fn get_title(&self, fingerprint: &ContentId) -> Option<String> {
        let p = self.title_path(fingerprint);
        let title = fs::read_to_string(&p).ok()?;
        if title.trim().is_empty() {
            return None;
        }
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
    /// A key's media file is `{key}.{ext}` with one dot-free extension
    /// outside [`SUBTITLE_EXTS`], exactly what the fetchers publish; staging
    /// and promotion temps carry extra dots and are excluded by shape. Among
    /// rivals from different containers (`k.mp4` beside `k.webm`) the newest
    /// mtime wins, and the touch keeps it winning.
    ///
    /// A hit also refreshes the key's subtitle sidecars, which grids read
    /// alongside the media; otherwise budget pressure would evict them from
    /// under a hot rip.
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
        for ext in SUBTITLE_EXTS {
            touch(&self.downloads_dir.join(format!("{prefix}{ext}")));
        }
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
    /// Every LRU/age removal goes through the mtime recheck
    /// (`remove_if_unbumped`): a concurrent process's cache hit between
    /// this sweep's listing and its unlink bumps the clock and vetoes the
    /// deletion, for age-swept families exactly as for downloads.
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
            if expired(&f) && remove_if_unbumped(&f) {
                removed.push(f.path);
            } else {
                // A veto (a rival's hit landed after ranking) keeps the
                // transcript living — and with it the sidecar it derives
                // from, via the pairing below.
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
            if !vetoed && remove_if_unbumped(&f) {
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

/// Install fully-staged `part` at `dest`, write-once. `link` is the
/// hard-link primitive (injected so tests can model filesystems without
/// one): it installs the complete staged bytes or fails because a winner
/// already exists. A blank winner is a legacy torn write, not a title —
/// it is dethroned and the publish contested once more; losing that
/// second link means a rival published a real title meanwhile, and theirs
/// serves the freeze as well as ours. Any other link error means the
/// filesystem cannot hard-link at all and falls back to
/// [`publish_by_rename`].
fn publish_write_once(
    part: &Path,
    dest: &Path,
    link: impl Fn(&Path, &Path) -> std::io::Result<()>,
) -> std::io::Result<()> {
    match link(part, dest) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == ErrorKind::AlreadyExists => {
            if fs::read_to_string(dest).is_ok_and(|t| t.trim().is_empty()) {
                let _ = fs::remove_file(dest);
                match link(part, dest) {
                    Ok(()) => Ok(()),
                    Err(e) if e.kind() == ErrorKind::AlreadyExists => Ok(()),
                    Err(_) => publish_by_rename(part, dest),
                }
            } else {
                Ok(())
            }
        }
        Err(_) => publish_by_rename(part, dest),
    }
}

/// Publication for filesystems without hard links: a same-directory
/// rename is still an atomic install of the complete staged bytes, so a
/// reader never sees a torn sidecar — but rename replaces, so the
/// write-once freeze weakens to last-writer-wins inside the racing
/// window. An existing non-blank winner is deferred to first, exactly
/// like a lost link, keeping the weakening confined to concurrent racers
/// (each holding a complete title that serves the freeze equally well).
fn publish_by_rename(part: &Path, dest: &Path) -> std::io::Result<()> {
    if fs::read_to_string(dest).is_ok_and(|t| !t.trim().is_empty()) {
        return Ok(());
    }
    fs::rename(part, dest)
}

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
/// readable, and a caller holding only a *path* re-derives the entry (a
/// re-download, a re-transcribe, a fresh probe) — never corruption. A veto
/// keeps a download counting toward the budget, consistent with the honest
/// accounting above, and keeps a transcript in the living set that shields
/// its title sidecar.
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
    fn url_key_converges_spellings_of_one_source() {
        let k = url_key("https://example.com/v.mp4");
        assert_eq!(k, url_key("HTTPS://Example.COM/v.mp4"));
        assert_ne!(
            k,
            url_key("https://example.com/V.mp4"),
            "path case names a different resource"
        );
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
    fn find_download_moves_sidecar_clocks_with_the_media() {
        // A grid served from this hit reads the sidecars too: their LRU
        // clock must move with the media's, or budget pressure evicts the
        // subtitles out from under a hot rip.
        let (cache, _dir) = cache(1024);
        let dir = cache.ensure_downloads_dir().expect("mkdir").to_path_buf();
        let media = dir.join("k1.mp4");
        let srt = dir.join("k1.srt");
        let vtt = dir.join("k1.vtt");
        let foreign = dir.join("k2.srt");
        let past = SystemTime::now() - Duration::from_secs(3600);
        for p in [&media, &srt, &vtt, &foreign] {
            std::fs::write(p, b"x").expect("write");
            File::options()
                .append(true)
                .open(p)
                .and_then(|f| f.set_modified(past))
                .expect("age the file");
        }

        cache.find_download("k1").expect("hit");

        let mtime = |p: &Path| fs::metadata(p).and_then(|m| m.modified()).expect("mtime");
        for p in [&media, &srt, &vtt] {
            assert!(mtime(p) > past, "{} clock moved with the hit", p.display());
        }
        assert_eq!(mtime(&foreign), past, "another key's sidecar is left alone");
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
    fn age_sweep_removal_is_vetoed_by_a_touch_after_ranking() {
        // Models the cross-process race for the age-swept families: a
        // rival's `find_transcript` hit bumps the mtime between this
        // sweep's listing and its unlink. The stale stat must not
        // authorize the deletion — the same veto downloads enjoy.
        let (cache, _dir) = cache(1024);
        let p = cache.transcript_path(&ContentId::new("fp"), "m");
        std::fs::create_dir_all(p.parent().expect("parent")).expect("mkdir");
        std::fs::write(&p, b"cues").expect("write");
        let past = SystemTime::now() - PROBE_MAX_AGE - Duration::from_secs(60);
        File::options()
            .append(true)
            .open(&p)
            .and_then(|f| f.set_modified(past))
            .expect("age the transcript");
        let stale = Stat {
            path: p.clone(),
            size: 4,
            mtime: past,
        };
        touch(&p); // the concurrent cache hit

        assert!(!remove_if_unbumped(&stale), "bumped mtime vetoes");
        assert!(p.exists(), "the just-served transcript survives");
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
    fn publishes_titles_atomically_leaving_no_staging() {
        let (cache, _dir) = cache(1024);
        cache
            .put_title(&ContentId::new("fp"), "Rust Ownership Explained")
            .expect("put");
        let names: Vec<String> = std::fs::read_dir(&cache.probe_dir)
            .expect("probe dir exists")
            .map(|e| e.expect("entry").file_name().to_string_lossy().into_owned())
            .collect();
        assert_eq!(names, ["fp.title"], "only the published name remains");
    }

    #[test]
    fn freezes_the_winning_title_under_racing_writers() {
        // Two processes sight the same content under different titles and
        // publish concurrently: exactly one link wins, the loser reports
        // success, and the winner is frozen — never torn, never blank.
        let (cache, _dir) = cache(1024);
        let fp = ContentId::new("fp");
        let titles = ["First Sighting", "Rival Sighting"];
        std::thread::scope(|s| {
            for title in titles {
                let cache = &cache;
                let fp = &fp;
                s.spawn(move || cache.put_title(fp, title).expect("a loser is a no-op"));
            }
        });
        let won = cache.get_title(&fp).expect("one racer published");
        assert!(
            titles.contains(&won.as_str()),
            "a complete title won: {won}"
        );
        cache.put_title(&fp, "Late Arrival").expect("no-op");
        assert_eq!(cache.get_title(&fp), Some(won), "the winner stays frozen");
        let names: Vec<String> = std::fs::read_dir(&cache.probe_dir)
            .expect("probe dir exists")
            .map(|e| e.expect("entry").file_name().to_string_lossy().into_owned())
            .collect();
        assert_eq!(names, ["fp.title"], "losers remove their staging");
    }

    #[test]
    fn falls_back_to_rename_when_hard_links_are_unsupported() {
        // exFAT/FAT and some SMB/NFS mounts fail hard_link permanently
        // with a non-AlreadyExists error; publication must still land —
        // silently losing every title splits the shared transcript cache.
        let (cache, _dir) = cache(1024);
        fs::create_dir_all(&cache.probe_dir).expect("mkdir");
        let dest = cache.title_path(&ContentId::new("fp"));
        let part = naming::part_path(&dest);
        fs::write(&part, "Rust Ownership Explained").expect("stage");

        publish_write_once(&part, &dest, |_, _| {
            Err(std::io::Error::from(ErrorKind::Unsupported))
        })
        .expect("rename fallback publishes");

        assert_eq!(
            cache.get_title(&ContentId::new("fp")).as_deref(),
            Some("Rust Ownership Explained")
        );
        assert!(!part.exists(), "the rename consumed the staging");
    }

    #[test]
    fn rename_fallback_defers_to_an_existing_winner() {
        // The fallback must not weaken write-once beyond the racing
        // window: a winner that already exists stays frozen, exactly as
        // a lost hard link would leave it.
        let (cache, _dir) = cache(1024);
        fs::create_dir_all(&cache.probe_dir).expect("mkdir");
        let dest = cache.title_path(&ContentId::new("fp"));
        fs::write(&dest, "First Sighted Title").expect("winner");
        let part = naming::part_path(&dest);
        fs::write(&part, "Retitled Upload").expect("stage");

        publish_write_once(&part, &dest, |_, _| {
            Err(std::io::Error::from(ErrorKind::Unsupported))
        })
        .expect("deferring is a successful no-op");

        assert_eq!(
            cache.get_title(&ContentId::new("fp")).as_deref(),
            Some("First Sighted Title"),
            "the recorded title stays frozen"
        );
    }

    #[test]
    fn rename_fallback_dethrones_a_blank_legacy_winner() {
        // A blank winner is no winner on the fallback path either: the
        // healing put must land instead of deferring to poison forever.
        let (cache, _dir) = cache(1024);
        fs::create_dir_all(&cache.probe_dir).expect("mkdir");
        let dest = cache.title_path(&ContentId::new("fp"));
        fs::write(&dest, "").expect("legacy poison");
        let part = naming::part_path(&dest);
        fs::write(&part, "Rust Ownership Explained").expect("stage");

        publish_write_once(&part, &dest, |_, _| {
            Err(std::io::Error::from(ErrorKind::Unsupported))
        })
        .expect("rename fallback publishes over poison");

        assert_eq!(
            cache.get_title(&ContentId::new("fp")).as_deref(),
            Some("Rust Ownership Explained")
        );
    }

    #[test]
    fn records_nothing_for_a_blank_title() {
        let (cache, _dir) = cache(1024);
        cache
            .put_title(&ContentId::new("fp"), "  \n")
            .expect("a blank put is a successful no-op");
        assert_eq!(cache.get_title(&ContentId::new("fp")), None);
        assert!(
            !cache.probe_dir.join("fp.title").exists(),
            "a blank title never reaches disk"
        );
    }

    #[test]
    fn blank_legacy_title_answers_none_and_ages_out() {
        // A pre-atomic-path cache may hold an empty sidecar. It must not
        // answer as a title, and reading it must not refresh its clock:
        // left alone, the age sweep retires it.
        let (cache, _dir) = cache(1024);
        std::fs::create_dir_all(&cache.probe_dir).expect("mkdir");
        let poisoned = cache.probe_dir.join("fp.title");
        std::fs::write(&poisoned, b"").expect("write");
        let past = SystemTime::now() - PROBE_MAX_AGE - Duration::from_secs(60);
        File::options()
            .append(true)
            .open(&poisoned)
            .and_then(|f| f.set_modified(past))
            .expect("age the sidecar");

        assert_eq!(cache.get_title(&ContentId::new("fp")), None);

        let removed = cache.evict(SystemTime::now());
        assert_eq!(removed, vec![poisoned.clone()], "the read did not bump it");
        assert!(!poisoned.exists(), "the poison self-retires");
    }

    #[test]
    fn dethrones_a_blank_legacy_winner_on_put() {
        // An empty winner is no winner: the next sighting replaces it
        // instead of deferring to it forever.
        let (cache, _dir) = cache(1024);
        std::fs::create_dir_all(&cache.probe_dir).expect("mkdir");
        std::fs::write(cache.probe_dir.join("fp.title"), b"").expect("write");

        cache
            .put_title(&ContentId::new("fp"), "Rust Ownership Explained")
            .expect("put");

        assert_eq!(
            cache.get_title(&ContentId::new("fp")).as_deref(),
            Some("Rust Ownership Explained"),
            "the healing put wins over legacy poison"
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
