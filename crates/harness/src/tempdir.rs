// SPDX-License-Identifier: Apache-2.0
//! RAII temp directories for tests.

use std::ops::Deref;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

/// RAII temp dir: unique per construction, removed on drop. Assertion
/// failures unwind through Drop, so failed runs stop littering /tmp;
/// only an abort leaks.
pub struct TempDir(PathBuf);

impl TempDir {
    /// Creates a fresh, empty directory under [`std::env::temp_dir`], its name
    /// woven from `tag`, the process id, a nanosecond clock reading and a
    /// monotonic counter so concurrent constructions never collide. `tag`
    /// survives into the path so a leaked directory names its origin test.
    pub fn new(tag: &str) -> Self {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        let seq = COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("{tag}-{}-{nanos}-{seq}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("mkdir");
        Self(dir)
    }
}

impl Deref for TempDir {
    type Target = Path;

    fn deref(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}
