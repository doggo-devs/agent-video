//! Output writing: the per-run output session, atomic file operations, and
//! metadata assembly.

use std::cell::Cell;
use std::fs;
use std::path::Path;

use anyhow::{Context, Result};

use crate::config::Config;
use crate::error::{Error, ErrorCode, ErrorContext};
use crate::subtitle::Subtitles;
use crate::types::{GridMeta, Metadata};

use super::OutputMode;

/// One run's output-directory lifecycle.
///
/// Run-level contract: the directory always reflects the most recent run that
/// produced output. Stale vidgrid artifacts are swept lazily, just before the
/// first write — so a run that fails before producing anything leaves the
/// previous run's outputs intact. Grids and statics are written incrementally
/// as they finish, and [`OutputSession::commit`] consumes the session and
/// writes `metadata.json` last — so a `metadata.json` on disk marks a run that
/// completed, and its absence marks one that did not. Files this tool never
/// writes are left untouched. In JSON mode the session touches no files at all.
pub struct OutputSession<'a> {
    dir: &'a Path,
    mode: OutputMode,
    swept: Cell<bool>,
}

impl<'a> OutputSession<'a> {
    /// Open the session: create the output directory so an unwritable
    /// destination fails before any frame work. The destructive sweep of
    /// previous artifacts is deferred to the first write (see [`Self::sweep_stale`]).
    pub fn open(dir: &'a Path, mode: OutputMode) -> Result<Self> {
        let session = Self {
            dir,
            mode,
            swept: Cell::new(false),
        };
        if mode == OutputMode::Files {
            fs::create_dir_all(dir).map_err(|e| session.dir_error("creating", e))?;
        }
        Ok(session)
    }

    pub fn mode(&self) -> OutputMode {
        self.mode
    }

    fn dir_error(&self, what: &str, e: std::io::Error) -> Error {
        Error::new(
            ErrorCode::OutputDirCreate,
            format!("{what} output dir {}: {e}", self.dir.display()),
        )
        .with_context(ErrorContext::default().with_output(self.dir))
        .with_source(e)
    }

    /// Remove artifacts left by a previous run, once, before the first write.
    /// Without the sweep, re-running with settings that produce fewer grids
    /// leaves stale `grid_*.jpg` files that the new `metadata.json` no longer
    /// lists. Deferring it to the first write (rather than `open`) means a run
    /// that fails during extraction destroys nothing.
    fn sweep_stale(&self) -> Result<()> {
        if self.swept.get() {
            return Ok(());
        }
        for entry in fs::read_dir(self.dir).map_err(|e| self.dir_error("listing", e))? {
            let entry = entry.map_err(|e| self.dir_error("listing", e))?;
            if entry.file_name().to_str().is_some_and(is_vidgrid_artifact) {
                fs::remove_file(entry.path()).map_err(|e| self.dir_error("cleaning", e))?;
            }
        }
        self.swept.set(true);
        Ok(())
    }

    /// Write one incremental image artifact (grid or static entry). A no-op in
    /// JSON mode, where images travel base64-inline in the metadata instead.
    pub fn write_image(&self, name: &str, jpeg: &[u8]) -> Result<()> {
        match self.mode {
            OutputMode::Files => {
                self.sweep_stale()?;
                atomic_write(self.dir, name, jpeg)
            }
            OutputMode::Json => Ok(()),
        }
    }

    /// Commit the run: write `subtitles.txt` (when given), then
    /// `metadata.json` last. Consuming `self` makes the commit point a
    /// type-level fact — one commit per session, nothing written after it.
    /// JSON mode prints the metadata to stdout instead.
    pub fn commit(self, meta: &Metadata, subtitles_txt: Option<&str>) -> Result<()> {
        let json = serde_json::to_string_pretty(meta).context("serializing metadata")?;
        match self.mode {
            OutputMode::Files => {
                self.sweep_stale()?;
                if let Some(text) = subtitles_txt {
                    atomic_write(self.dir, "subtitles.txt", text.as_bytes())?;
                }
                atomic_write(self.dir, "metadata.json", json.as_bytes())?;
                eprintln!(
                    "wrote {} grid(s) to {}",
                    meta.grids.len(),
                    self.dir.display()
                );
            }
            OutputMode::Json => println!("{json}"),
        }
        Ok(())
    }
}

/// Whether `name` is a file this tool writes into the output directory:
/// grids, static entries, metadata, subtitles, or their atomic-write temps.
fn is_vidgrid_artifact(name: &str) -> bool {
    if let Some(tmp) = name.strip_prefix('.').and_then(|n| n.strip_suffix(".tmp")) {
        return is_vidgrid_artifact(tmp);
    }
    let numbered_jpg = |prefix: &str| {
        name.strip_prefix(prefix)
            .and_then(|rest| rest.strip_suffix(".jpg"))
            .is_some_and(|digits| digits.len() >= 3 && digits.bytes().all(|b| b.is_ascii_digit()))
    };
    matches!(name, "metadata.json" | "subtitles.txt" | "grid_extract.jpg")
        || numbered_jpg("grid_")
        || numbered_jpg("static_")
}

/// Slice per-grid subtitles, assemble [`Metadata`], and commit the session.
///
/// `subs` is already resolved by the caller — resolution can fail (ambiguous
/// sources, unreadable files) and must happen before any grid hits the disk,
/// not here at the tail of the run.
pub fn write_outputs(
    session: OutputSession<'_>,
    input: &Path,
    cfg: &Config,
    duration: f32,
    mut grid_metas: Vec<GridMeta>,
    subs: Option<&Subtitles>,
) -> Result<()> {
    if let Some(subs) = subs {
        for grid in &mut grid_metas {
            grid.subtitles = subs.slice(grid.time_range);
        }
    }

    let meta = Metadata {
        source: input.display().to_string(),
        duration,
        settings: cfg.settings(),
        grids: grid_metas,
    };

    session.commit(&meta, subs.map(Subtitles::raw))
}

/// Write a file atomically: write to .tmp then rename.
fn atomic_write(dir: &Path, name: &str, data: &[u8]) -> Result<()> {
    let final_path = dir.join(name);
    let tmp_path = dir.join(format!(".{name}.tmp"));

    fs::write(&tmp_path, data).map_err(|e| {
        Error::new(ErrorCode::OutputWriteFailed, format!("writing {name}: {e}"))
            .with_context(ErrorContext::default().with_output(&tmp_path))
            .with_source(e)
    })?;

    if let Err(e) = fs::rename(&tmp_path, &final_path) {
        if let Err(rm_err) = fs::remove_file(&tmp_path) {
            eprintln!(
                "warning: failed to clean up {}: {rm_err}",
                tmp_path.display()
            );
        }
        return Err(Error::new(
            ErrorCode::OutputWriteFailed,
            format!("finalizing {name}: {e}"),
        )
        .with_context(ErrorContext::default().with_output(&final_path))
        .with_source(e)
        .into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recognizes_own_artifacts_and_spares_foreign_files() {
        for own in [
            "grid_001.jpg",
            "grid_1000.jpg",
            "static_007.jpg",
            "grid_extract.jpg",
            "metadata.json",
            "subtitles.txt",
            ".grid_001.jpg.tmp",
            ".metadata.json.tmp",
        ] {
            assert!(is_vidgrid_artifact(own), "{own} is ours");
        }
        for foreign in [
            "grid_01.jpg",
            "grid_abc.jpg",
            "grid_001.png",
            "notes.txt",
            "video.mp4",
            ".hidden",
        ] {
            assert!(!is_vidgrid_artifact(foreign), "{foreign} is not ours");
        }
    }

    #[test]
    fn sweeps_stale_artifacts_on_first_write_but_keeps_foreign_files() {
        let dir = std::env::temp_dir().join(format!("vidgrid-sweep-test-{}", std::process::id()));
        fs::create_dir_all(&dir).expect("create test dir");
        let stale = dir.join("grid_007.jpg");
        let foreign = dir.join("keep.me");
        fs::write(&stale, b"old").expect("write stale grid");
        fs::write(&foreign, b"user data").expect("write foreign file");

        let session = OutputSession::open(&dir, OutputMode::Files).expect("open session");
        assert!(
            stale.exists(),
            "sweep deferred: a run failing before output must destroy nothing"
        );

        session.write_image("grid_001.jpg", b"new").expect("write");

        assert!(!stale.exists(), "stale grid removed at first write");
        assert!(foreign.exists(), "foreign file untouched");
        assert!(dir.join("grid_001.jpg").exists(), "new grid written");
        fs::remove_dir_all(&dir).expect("cleanup test dir");
    }

    #[test]
    fn json_session_touches_no_files() {
        let dir = std::env::temp_dir().join(format!("vidgrid-json-test-{}", std::process::id()));
        // The directory is never created: open, write_image, and commit must
        // all stay off the filesystem in JSON mode.
        let session = OutputSession::open(&dir, OutputMode::Json).expect("open session");
        session
            .write_image("grid_001.jpg", b"jpeg")
            .expect("write is a no-op");
        assert!(!dir.exists(), "json session created the output dir");
    }
}
