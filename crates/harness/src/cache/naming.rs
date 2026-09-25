// SPDX-License-Identifier: Apache-2.0
//! The shared cache's staging-name grammar: the one owner of every
//! transient spelling that rests directly under the cache directories, and
//! of the predicates the eviction sweep recognizes them by. Names inside a
//! per-writer staging directory are that writer's business; the sweep
//! classifies only the directory.
//!
//! Writers ([`crate::fetch::direct`], [`crate::fetch::ytdlp`],
//! [`crate::transcribe`], and [`crate::install`] for the models directory)
//! construct names here; [`crate::cache::Cache::evict`] classifies them
//! here, so live work is never evicted and crashed work is never missed.
//!
//! The on-disk spellings are frozen so existing caches stay sweepable:
//! `<dest>.<pid>-<seq>.part` for files, `.staging-<key>-<pid>-<seq>` for
//! directories. Tests pin them against verbatim strings.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

/// Suffix of every staging file. Never a published extension: finished
/// downloads carry a media extension and transcripts `.srt`, so the
/// [`is_part_file`] predicate cannot claim a published entry.
const PART_SUFFIX: &str = ".part";

/// Prefix of every staging directory. Published cache names are
/// `<key>.<ext>` with a hex key, so no published name starts with it.
const STAGING_PREFIX: &str = ".staging-";

/// Writer-unique `pid-seq` token. Uniqueness is per writer, not per
/// process: a pid alone would let two threads of one server share a
/// staging name and interleave their bytes.
fn writer_id() -> String {
    static WRITER_SEQ: AtomicU64 = AtomicU64::new(0);
    let seq = WRITER_SEQ.fetch_add(1, Ordering::Relaxed);
    format!("{}-{seq}", std::process::id())
}

/// Writer-unique staging path for `dest`: `<dest>.<pid>-<seq>.part`,
/// always a sibling of `dest` so the final rename stays on one filesystem
/// (docs/shared-cache.md, Pattern 2). Recognized by [`is_part_file`].
pub fn part_path(dest: &Path) -> PathBuf {
    PathBuf::from(format!("{}.{}{PART_SUFFIX}", dest.display(), writer_id()))
}

/// Writer-unique staging directory name for cache key `key`:
/// `.staging-<key>-<pid>-<seq>` (docs/shared-cache.md, Pattern 2).
/// Recognized by [`is_staging_dir`].
pub fn staging_dir_name(key: &str) -> String {
    format!("{STAGING_PREFIX}{key}-{}", writer_id())
}

/// Sweeper predicate: `name` is a staging file — in flight or crashed —
/// and never an LRU entry. Matches every [`part_path`] product and no
/// published cache name.
pub fn is_part_file(name: &str) -> bool {
    name.ends_with(PART_SUFFIX)
}

/// Sweeper predicate: `name` is a per-writer staging directory. Matches
/// every [`staging_dir_name`] product and no published cache name.
pub fn is_staging_dir(name: &str) -> bool {
    name.starts_with(STAGING_PREFIX)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn published_names_match_no_staging_predicate() {
        // The names writers actually publish: media rips, sidecars,
        // transcripts, probe documents.
        for name in ["k.mp4", "k.mkv", "k.srt", "fp.large-v3.srt", "fp.json"] {
            assert!(!is_part_file(name), "{name} must stay an LRU entry");
            assert!(!is_staging_dir(name), "{name} must stay an LRU entry");
        }
    }

    #[test]
    fn pins_the_on_disk_part_spelling() {
        // The frozen grammar: `<dest>.<pid>-<seq>.part`. A change here
        // strands `.part` files written by existing installs.
        let part = part_path(Path::new("/c/k.mp4")).display().to_string();
        let middle = part
            .strip_prefix("/c/k.mp4.")
            .and_then(|rest| rest.strip_suffix(".part"))
            .expect("spelling is <dest>.<writer>.part: {part}");
        let (pid, seq) = middle.rsplit_once('-').expect("writer id is pid-seq");
        assert_eq!(pid, std::process::id().to_string());
        assert!(seq.chars().all(|c| c.is_ascii_digit()), "{seq}");
    }

    #[test]
    fn pins_the_on_disk_staging_spelling() {
        // The frozen grammar: `.staging-<key>-<pid>-<seq>`.
        let dir = staging_dir_name("abc123");
        let middle = dir
            .strip_prefix(".staging-abc123-")
            .expect("spelling is .staging-<key>-<writer>: {dir}");
        let (pid, seq) = middle.rsplit_once('-').expect("writer id is pid-seq");
        assert_eq!(pid, std::process::id().to_string());
        assert!(seq.chars().all(|c| c.is_ascii_digit()), "{seq}");
    }

    #[test]
    fn recognizes_legacy_on_disk_spellings() {
        // Verbatim names earlier releases wrote; an existing cache must
        // stay sweepable across upgrades.
        assert!(is_part_file("aged.mp4.123-0.part"));
        assert!(is_part_file("fp.large-v3.srt.4242-7.part"));
        assert!(is_staging_dir(".staging-k-999-0"));
    }

    #[test]
    fn part_paths_are_writer_unique() {
        let dest = Path::new("/cache/downloads/k.mp4");
        let a = part_path(dest);
        assert!(
            a.to_string_lossy().starts_with("/cache/downloads/k.mp4."),
            "stays next to its destination: {a:?}"
        );
        // Two writers (even two threads of one process) must not share a
        // staging file.
        assert_ne!(a, part_path(dest));
    }

    #[test]
    fn staging_dir_names_are_writer_unique() {
        assert_ne!(staging_dir_name("k"), staging_dir_name("k"));
    }
}
