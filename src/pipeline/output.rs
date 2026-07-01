//! Output writing: atomic file operations, metadata assembly.

use std::fs;
use std::path::Path;

use anyhow::{Context, Result};

use crate::config::Config;
use crate::error::{Error, ErrorCode, ErrorContext};
use crate::subtitle;
use crate::types::{GridMeta, Metadata};

use super::{OutputMode, SubtitleOpts};

/// Write grid images, metadata.json, and subtitles.
pub fn write_outputs(
    input: &Path,
    out_dir: &Path,
    cfg: &Config,
    mode: OutputMode,
    duration: f32,
    mut grid_metas: Vec<GridMeta>,
    subs: SubtitleOpts<'_>,
) -> Result<()> {
    let write_files = mode == OutputMode::Files;

    let srt_text = subtitle::resolve(input, subs.path, subs.track)?;
    if let Some(ref text) = srt_text {
        let cues = subtitle::parse_srt(text);
        for grid in &mut grid_metas {
            grid.subtitles =
                subtitle::slice_for_range(&cues, grid.time_range.start(), grid.time_range.end());
        }
        if write_files {
            atomic_write(out_dir, "subtitles.txt", text.as_bytes())?;
        }
    }

    let meta = Metadata {
        source: input.display().to_string(),
        duration,
        settings: cfg.settings(),
        grids: grid_metas,
    };

    if write_files {
        let json = serde_json::to_string_pretty(&meta).context("serializing metadata")?;
        atomic_write(out_dir, "metadata.json", json.as_bytes())?;
        eprintln!(
            "wrote {} grid(s) to {}",
            meta.grids.len(),
            out_dir.display()
        );
    } else {
        let json = serde_json::to_string_pretty(&meta).context("serializing metadata")?;
        println!("{json}");
    }

    Ok(())
}

/// Write a file atomically: write to .tmp then rename.
pub fn atomic_write(dir: &Path, name: &str, data: &[u8]) -> Result<()> {
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
