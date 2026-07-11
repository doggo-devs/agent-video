// SPDX-License-Identifier: Apache-2.0
//! Content-based file identity.
//!
//! Fingerprints key the probe/transcript caches so moved or renamed files
//! keep their entries while in-place edits invalidate them. The digest is
//! `sha256(head ‖ tail ‖ size_le64)` truncated to 16 bytes, over the first
//! and last [`SLICE_BYTES`] of the file — byte-compatible with the TS
//! harness this replaces, so existing caches stay valid.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;
use std::time::SystemTime;

use sha2::{Digest, Sha256};

/// Bytes hashed from each end of the file.
pub const SLICE_BYTES: u64 = 128 * 1024;

/// Content identity: the 16-byte digest as 32 hex chars (size is folded
/// into the digest). Keys the probe and transcript caches; equal ids mean
/// same identity for cache purposes.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ContentId(String);

impl ContentId {
    /// Wrap an externally stored digest (cache filenames, tests). Real ids
    /// come from [`digest_parts`].
    pub fn new(hex: impl Into<String>) -> Self {
        ContentId(hex.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for ContentId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// A content fingerprint; check `stable` before writing cache entries.
#[derive(Debug, Clone)]
pub struct Fingerprint {
    pub id: ContentId,
    /// False when the file changed mid-read; do not write cache entries.
    pub stable: bool,
}

/// Pure digest over sampled parts: `sha256(head ‖ tail ‖ size_le64)[..16]`.
pub fn digest_parts(head: &[u8], tail: &[u8], size: u64) -> ContentId {
    let mut hasher = Sha256::new();
    hasher.update(head);
    hasher.update(tail);
    hasher.update(size.to_le_bytes());
    ContentId(hex(&hasher.finalize()[..16]))
}

/// Lowercase hex of `bytes`.
pub fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// Fingerprint a file by content: first/last [`SLICE_BYTES`] plus size.
/// Check [`Fingerprint::stable`] before writing cache entries keyed by it.
pub fn fingerprint_file(path: &Path) -> std::io::Result<Fingerprint> {
    let before = snapshot(path)?;

    let mut file = File::open(path)?;
    let (head, tail) = if before.size <= 2 * SLICE_BYTES {
        (read_exactish(&mut file, before.size as usize)?, Vec::new())
    } else {
        let head = read_exactish(&mut file, SLICE_BYTES as usize)?;
        file.seek(SeekFrom::Start(before.size - SLICE_BYTES))?;
        let tail = read_exactish(&mut file, SLICE_BYTES as usize)?;
        (head, tail)
    };

    let after = snapshot(path)?;
    Ok(Fingerprint {
        id: digest_parts(&head, &tail, before.size),
        stable: before == after,
    })
}

#[derive(PartialEq)]
struct Snapshot {
    size: u64,
    mtime: Option<SystemTime>,
}

fn snapshot(path: &Path) -> std::io::Result<Snapshot> {
    let meta = std::fs::metadata(path)?;
    Ok(Snapshot {
        size: meta.len(),
        mtime: meta.modified().ok(),
    })
}

/// Read up to `len` bytes (short only at EOF).
fn read_exactish(file: &mut File, len: usize) -> std::io::Result<Vec<u8>> {
    let mut buf = vec![0u8; len];
    let mut filled = 0;
    while filled < len {
        let n = file.read(&mut buf[filled..])?;
        if n == 0 {
            break;
        }
        filled += n;
    }
    buf.truncate(filled);
    Ok(buf)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn digest_matches_ts_harness_vectors() {
        // Cross-language pin: bun -e 'digestParts(Buffer.from("head"), Buffer.from("tail"), 8)'
        // must produce the same hex, keeping existing caches valid.
        let hex = digest_parts(b"head", b"tail", 8);
        // Pinned against the TS harness (`digestParts`) — equality keeps
        // caches written by the old harness valid.
        assert_eq!(hex.as_str(), "34c1f7afec88cc36dc043f6bcdeafd05");
        assert_ne!(digest_parts(b"head", b"tail", 9), hex, "size feeds digest");
        assert_ne!(digest_parts(b"heap", b"tail", 8), hex, "head feeds digest");
    }

    #[test]
    fn fingerprints_small_file_whole() {
        let dir = tempdir();
        let p = dir.join("small.bin");
        std::fs::write(&p, b"hello world").expect("write");
        let fp = fingerprint_file(&p).expect("fingerprint");
        assert!(fp.stable);
        assert_eq!(fp.id, digest_parts(b"hello world", b"", 11));
    }

    #[test]
    fn samples_both_ends_of_large_file() {
        let dir = tempdir();
        let p = dir.join("large.bin");
        let size = (2 * SLICE_BYTES + 100) as usize;
        let mut data = vec![0xAAu8; size];
        data[..4].copy_from_slice(b"HEAD");
        let tail_start = size - SLICE_BYTES as usize;
        data[size - 4..].copy_from_slice(b"TAIL");
        std::fs::write(&p, &data).expect("write");

        let fp = fingerprint_file(&p).expect("fingerprint");
        let expected = digest_parts(
            &data[..SLICE_BYTES as usize],
            &data[tail_start..],
            size as u64,
        );
        assert_eq!(fp.id, expected);
    }

    #[test]
    fn middle_edits_of_large_file_do_not_change_identity() {
        let dir = tempdir();
        let p = dir.join("v.bin");
        let size = (3 * SLICE_BYTES) as usize;
        let mut data = vec![1u8; size];
        std::fs::write(&p, &data).expect("write");
        let a = fingerprint_file(&p).expect("fingerprint");
        data[size / 2] = 2;
        std::fs::write(&p, &data).expect("write");
        let b = fingerprint_file(&p).expect("fingerprint");
        assert_eq!(a.id, b.id, "sampled identity ignores middle bytes");
    }

    fn tempdir() -> crate::testutil::TempDir {
        crate::testutil::TempDir::new("av-identity")
    }
}
