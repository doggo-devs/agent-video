//! Output writing: the per-run output session, staged-commit file lifecycle,
//! and metadata assembly.

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

use crate::config::Config;
use crate::error::{Error, ErrorCode, ErrorContext};
use crate::source::Source;
use crate::subtitle::Subtitles;
use crate::types::{GridImage, GridMeta, Metadata};

use super::OutputMode;

/// Name of the staging directory inside the output directory. Starts with a
/// dot so [`is_own_artifact`]'s tmp-prefix rule never mistakes it for a
/// final artifact, and lives on the same filesystem so promotion is a rename.
const STAGING_DIR: &str = ".staging.tmp";

/// One run's output-directory lifecycle, with a staged commit.
///
/// Run-level contract: destruction of the previous run's artifacts happens
/// only inside [`OutputSession::commit`], after every byte of the new run is
/// already durable in the staging directory. Grids and statics are written
/// incrementally — into staging — as they finish; `commit` then writes
/// `subtitles.txt` and `metadata.json` into staging, sweeps stale artifacts,
/// and promotes the staged files into place with `metadata.json` renamed
/// last. So at every instant the directory is one of exactly two states: the
/// previous completed run (any abort before the sweep), or the new run
/// mid-promotion — already past the point where all its files exist locally
/// — with `metadata.json` still marking completion. A leftover staging
/// directory (crash mid-run) is removed on drop and, belt and suspenders, at
/// the next `open`. Files this tool never writes are left untouched. In JSON
/// mode the session touches no files at all.
pub struct OutputSession<'a> {
    dir: &'a Path,
    sink: Sink,
}

/// Where emitted images go: a staging directory awaiting promotion, or
/// base64 blobs in the JSON payload. The one-of is structural: a JSON
/// session has no staging path to misuse.
enum Sink {
    Files { staging: PathBuf },
    Json,
}

impl<'a> OutputSession<'a> {
    /// Open the session: create the output and staging directories so an
    /// unwritable destination fails before any frame work. A staging
    /// directory left behind by a crashed run is discarded here.
    pub fn open(dir: &'a Path, mode: OutputMode) -> Result<Self> {
        let sink = match mode {
            OutputMode::Json => Sink::Json,
            OutputMode::Files => {
                let staging = dir.join(STAGING_DIR);
                fs::create_dir_all(dir).map_err(|e| dir_error(dir, "creating", e))?;
                if staging.exists() {
                    fs::remove_dir_all(&staging)
                        .map_err(|e| dir_error(dir, "clearing stale staging in", e))?;
                }
                fs::create_dir(&staging).map_err(|e| dir_error(dir, "creating staging in", e))?;
                Sink::Files { staging }
            }
        };
        Ok(Self { dir, sink })
    }

    /// Emit one image artifact (grid or static entry) and return its metadata
    /// payload. In file mode the JPEG is written into the staging directory
    /// and referenced by name; in JSON mode nothing touches the filesystem
    /// and the bytes travel base64-inline. This is the only constructor of
    /// [`GridImage`] in the pipeline: the file-vs-base64 one-of invariant
    /// lives here, so no worker needs to know the output mode exists.
    pub fn emit_image(&self, name: &str, jpeg: &[u8]) -> Result<GridImage> {
        match &self.sink {
            Sink::Files { staging } => {
                write_file(staging, name, jpeg)?;
                Ok(GridImage::File(name.to_string()))
            }
            Sink::Json => {
                use base64::{Engine, engine::general_purpose::STANDARD};
                Ok(GridImage::Base64(STANDARD.encode(jpeg)))
            }
        }
    }

    /// Commit the run. Consuming `self` makes the commit point a type-level
    /// fact — one commit per session, nothing written after it. JSON mode
    /// prints the metadata to stdout instead.
    ///
    /// File mode orders the steps so every abort window is safe:
    /// 1. `subtitles.txt` (when given) and `metadata.json` land in staging —
    ///    the new run is now complete on disk;
    /// 2. stale artifacts of the previous run are swept — the only
    ///    destructive step, gated on step 1;
    /// 3. staged files are renamed into place, `metadata.json` last — its
    ///    presence keeps marking a completed run;
    /// 4. the staging directory is removed.
    pub fn commit(self, meta: &Metadata, subtitles_txt: Option<&str>) -> Result<()> {
        let json = serde_json::to_string_pretty(meta).context("serializing metadata")?;
        match &self.sink {
            Sink::Files { staging } => {
                if let Some(text) = subtitles_txt {
                    write_file(staging, "subtitles.txt", text.as_bytes())?;
                }
                write_file(staging, "metadata.json", json.as_bytes())?;
                self.sweep_stale()?;
                self.promote(staging)?;
                eprintln!(
                    "wrote {} grid(s) to {}",
                    meta.grids.len(),
                    self.dir.display()
                );
            }
            Sink::Json => println!("{json}"),
        }
        Ok(())
    }

    /// Remove artifacts left by a previous run. Without the sweep, re-running
    /// with settings that produce fewer grids leaves stale `grid_*.jpg` files
    /// that the new `metadata.json` no longer lists.
    fn sweep_stale(&self) -> Result<()> {
        for entry in fs::read_dir(self.dir).map_err(|e| dir_error(self.dir, "listing", e))? {
            let entry = entry.map_err(|e| dir_error(self.dir, "listing", e))?;
            let is_file = entry.file_type().is_ok_and(|t| t.is_file());
            if is_file && entry.file_name().to_str().is_some_and(is_own_artifact) {
                fs::remove_file(entry.path()).map_err(|e| dir_error(self.dir, "cleaning", e))?;
            }
        }
        Ok(())
    }

    /// Rename every staged file into the output directory, `metadata.json`
    /// strictly last, then remove the (now empty) staging directory.
    fn promote(&self, staging: &Path) -> Result<()> {
        let mut names: Vec<String> = Vec::new();
        for entry in
            fs::read_dir(staging).map_err(|e| dir_error(self.dir, "listing staging in", e))?
        {
            let entry = entry.map_err(|e| dir_error(self.dir, "listing staging in", e))?;
            let name = entry.file_name().to_string_lossy().into_owned();
            names.push(name);
        }
        // Completion-marker ordering: metadata.json must be the last rename.
        names.sort_unstable_by_key(|n| n == "metadata.json");
        for name in &names {
            let from = staging.join(name);
            let to = self.dir.join(name);
            fs::rename(&from, &to).map_err(|e| {
                Error::new(
                    ErrorCode::OutputWriteFailed,
                    format!("finalizing {name}: {e}"),
                )
                .with_context(ErrorContext::default().with_output(&to))
                .with_source(e)
            })?;
        }
        fs::remove_dir(staging).map_err(|e| dir_error(self.dir, "removing staging in", e))?;
        Ok(())
    }
}

fn dir_error(dir: &Path, what: &str, e: std::io::Error) -> Error {
    Error::new(
        ErrorCode::OutputDirCreate,
        format!("{what} output dir {}: {e}", dir.display()),
    )
    .with_context(ErrorContext::default().with_output(dir))
    .with_source(e)
}

impl Drop for OutputSession<'_> {
    /// Best-effort cleanup of the staging directory when the run aborts
    /// before `commit` (which removes it itself; this remove then finds
    /// nothing). The previous run's artifacts are untouched either way.
    fn drop(&mut self) {
        if let Sink::Files { staging } = &self.sink {
            let _ = fs::remove_dir_all(staging);
        }
    }
}

/// Whether `name` is a file this tool writes into the output directory:
/// grids, static entries, metadata, subtitles, or their legacy atomic-write
/// temps (pre-staging versions wrote `.<name>.tmp` files in place).
fn is_own_artifact(name: &str) -> bool {
    if let Some(tmp) = name.strip_prefix('.').and_then(|n| n.strip_suffix(".tmp")) {
        return is_own_artifact(tmp);
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
    input: &Source,
    cfg: &Config,
    duration: f32,
    mut grid_metas: Vec<GridMeta>,
    subs: Option<&Subtitles>,
) -> Result<()> {
    for grid in &mut grid_metas {
        grid.subtitles = match subs {
            Some(subs) => subs.slice(grid.time_range),
            None => crate::types::GridSubtitles::NoSource,
        };
    }

    let meta = Metadata {
        source: input.to_string(),
        duration,
        settings: cfg.settings(),
        grids: grid_metas,
    };

    session.commit(&meta, subs.map(Subtitles::raw))
}

/// Write a staged file, mapping failures into the output error decade.
fn write_file(dir: &Path, name: &str, data: &[u8]) -> Result<()> {
    let path = dir.join(name);
    fs::write(&path, data).map_err(|e| {
        Error::new(ErrorCode::OutputWriteFailed, format!("writing {name}: {e}"))
            .with_context(ErrorContext::default().with_output(&path))
            .with_source(e)
            .into()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A fresh temp output dir per test, cleaned up by the caller.
    fn test_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("agent-video-{tag}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        dir
    }

    fn committed_meta() -> Metadata {
        Metadata {
            source: "test.mp4".into(),
            duration: 1.0,
            settings: crate::types::Settings {
                max_edge: 2048,
                quality: crate::types::Quality::High,
                detail: crate::types::Detail::Standard,
            },
            grids: Vec::new(),
        }
    }

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
            assert!(is_own_artifact(own), "{own} is ours");
        }
        for foreign in [
            "grid_01.jpg",
            "grid_abc.jpg",
            "grid_001.png",
            "notes.txt",
            "video.mp4",
            ".hidden",
            STAGING_DIR,
        ] {
            assert!(!is_own_artifact(foreign), "{foreign} is not ours");
        }
    }

    #[test]
    fn destroys_previous_run_only_at_commit() {
        let dir = test_dir("commit");
        fs::create_dir_all(&dir).expect("create test dir");
        let stale = dir.join("grid_007.jpg");
        let foreign = dir.join("keep.me");
        fs::write(&stale, b"old").expect("write stale grid");
        fs::write(&foreign, b"user data").expect("write foreign file");

        let session = OutputSession::open(&dir, OutputMode::Files).expect("open session");
        let image = session.emit_image("grid_001.jpg", b"new").expect("write");
        assert!(
            matches!(image, GridImage::File(name) if name == "grid_001.jpg"),
            "file mode emits a file reference"
        );
        assert!(
            stale.exists(),
            "incremental writes are staged: nothing is destroyed before commit"
        );
        assert!(
            !dir.join("grid_001.jpg").exists(),
            "the new grid stays in staging until commit"
        );

        session
            .commit(&committed_meta(), Some("cues"))
            .expect("commit");

        assert!(!stale.exists(), "stale grid swept at commit");
        assert!(foreign.exists(), "foreign file untouched");
        assert!(dir.join("grid_001.jpg").exists(), "new grid promoted");
        assert!(dir.join("metadata.json").exists(), "metadata promoted");
        assert!(dir.join("subtitles.txt").exists(), "subtitles promoted");
        assert!(!dir.join(STAGING_DIR).exists(), "staging dir removed");
        fs::remove_dir_all(&dir).expect("cleanup test dir");
    }

    #[test]
    fn abort_after_partial_output_leaves_previous_run_intact() {
        let dir = test_dir("abort");
        fs::create_dir_all(&dir).expect("create test dir");
        let old_grid = dir.join("grid_007.jpg");
        let old_meta = dir.join("metadata.json");
        fs::write(&old_grid, b"old").expect("write old grid");
        fs::write(&old_meta, b"{}").expect("write old metadata");

        {
            let session = OutputSession::open(&dir, OutputMode::Files).expect("open session");
            session.emit_image("grid_001.jpg", b"new").expect("write");
            // Dropped without commit: the run aborted mid-flight.
        }

        assert!(old_grid.exists(), "old grid survives the abort");
        assert!(old_meta.exists(), "old metadata survives the abort");
        assert!(
            !dir.join("grid_001.jpg").exists(),
            "partial new output never reaches the directory"
        );
        assert!(!dir.join(STAGING_DIR).exists(), "staging cleaned on drop");
        fs::remove_dir_all(&dir).expect("cleanup test dir");
    }

    #[test]
    fn open_discards_staging_left_by_a_crashed_run() {
        let dir = test_dir("crash");
        let leftover = dir.join(STAGING_DIR).join("grid_001.jpg");
        fs::create_dir_all(leftover.parent().expect("staging path has parent"))
            .expect("create leftover staging");
        fs::write(&leftover, b"half-written").expect("write leftover");

        let session = OutputSession::open(&dir, OutputMode::Files).expect("open session");

        assert!(
            !leftover.exists(),
            "a crashed run's staging is discarded at open"
        );
        drop(session);
        fs::remove_dir_all(&dir).expect("cleanup test dir");
    }

    #[test]
    fn json_session_touches_no_files() {
        let dir = test_dir("json");
        // The directory is never created: open, emit_image, and commit must
        // all stay off the filesystem in JSON mode.
        let session = OutputSession::open(&dir, OutputMode::Json).expect("open session");
        let image = session
            .emit_image("grid_001.jpg", b"jpeg")
            .expect("emit touches no files");
        assert!(
            matches!(image, GridImage::Base64(_)),
            "json mode embeds the payload"
        );
        drop(session);
        assert!(!dir.exists(), "json session created the output dir");
    }
}
