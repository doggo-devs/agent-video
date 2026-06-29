//! Pipeline orchestration: extract → dedup → chunk → (per-chunk) score → select
//! → layout → compose, then write grids, `metadata.json`, and subtitles.
//!
//! Rank normalization and selection are deliberately per-chunk: scores are only
//! meaningful relative to the frames they compete with, and a "minute" is the
//! v1 unit of competition.

use std::fs;
use std::path::Path;

use anyhow::{Context, Result};

use crate::types::{
    GridMeta, KeyFrameMeta, Metadata, Preset, ScoredFrame, Settings,
};
use crate::{compose, dedup, extract, layout, score, subtitle};

/// Seconds per grid chunk (the v1 "minute" unit).
const CHUNK_SECONDS: f32 = 60.0;

/// Floor on the temporal NMS gap so selection never collapses to one moment.
const MIN_GAP_FLOOR: f32 = 0.5;

/// Run the full `process` pipeline and write all outputs into `out_dir`.
pub fn process(
    input: &Path,
    out_dir: &Path,
    preset: &Preset,
    quality_label: &str,
    detail_label: &str,
) -> Result<()> {
    fs::create_dir_all(out_dir)
        .with_context(|| format!("creating output dir {}", out_dir.display()))?;

    let res = extract::extract_frames(input, preset.base_fps, preset.min_frame_size)?;
    let aspect = res.aspect;
    let deduped = dedup::deduplicate(res.frames, preset.hash_threshold);

    let duration = deduped.last().map(|f| f.ts + 1.0 / preset.base_fps).unwrap_or(0.0);
    let chunks = chunk_by_seconds(deduped, CHUNK_SECONDS);
    let capacity = layout::grid_capacity(preset, aspect);

    let mut grid_metas = Vec::new();
    for (ci, mut chunk) in chunks.into_iter().enumerate() {
        if chunk.is_empty() {
            continue;
        }
        // Score within the chunk.
        score::compute_raw(&mut chunk);
        score::rank_normalize_and_score(&mut chunk);

        // Select top-k with temporal NMS.
        let span = chunk_span(&chunk);
        let min_gap = (0.5 * span / capacity.max(1) as f32).max(MIN_GAP_FLOOR);
        let selected = score::select_top_k(&chunk, capacity, min_gap);
        let picked: Vec<ScoredFrame> = selected.iter().map(|&i| chunk[i].clone()).collect();

        // Heroes drive the layout template.
        let heroes = score::pick_heroes(&picked);
        let lay = layout::compute(picked.len(), &heroes, aspect, preset);

        // Compose and write.
        let jpeg = compose::render(&picked, &lay, &heroes)?;
        let file = format!("grid_{:03}.jpg", ci + 1);
        fs::write(out_dir.join(&file), &jpeg)
            .with_context(|| format!("writing {file}"))?;

        grid_metas.push(build_grid_meta(&file, &picked, &heroes, lay.template));
    }

    // Subtitles (optional).
    if let Some(text) = subtitle::extract(input).ok().flatten() {
        fs::write(out_dir.join("subtitles.txt"), text).context("writing subtitles.txt")?;
    }

    // Metadata.
    let meta = Metadata {
        source: input.display().to_string(),
        duration,
        settings: Settings {
            max_pixels: preset.max_pixels,
            quality: quality_label.to_string(),
            detail: detail_label.to_string(),
        },
        grids: grid_metas,
    };
    let json = serde_json::to_string_pretty(&meta).context("serializing metadata")?;
    fs::write(out_dir.join("metadata.json"), json).context("writing metadata.json")?;

    println!("wrote {} grid(s) to {}", meta.grids.len(), out_dir.display());
    Ok(())
}

/// Run the `extract` pipeline: high-density sampling of `[start, end)` into a
/// single equal-cell grid, frames in time order (no hero emphasis), then write
/// `grid_extract.jpg` and `metadata.json`.
pub fn extract_range(
    input: &Path,
    out_dir: &Path,
    preset: &Preset,
    quality_label: &str,
    detail_label: &str,
    start: f32,
    end: f32,
) -> Result<()> {
    fs::create_dir_all(out_dir)
        .with_context(|| format!("creating output dir {}", out_dir.display()))?;

    // Triple the base fps for dense sampling within the window.
    let fps = preset.base_fps * 3.0;
    let res = extract::extract_range(input, fps, preset.min_frame_size, start, end)?;
    let aspect = res.aspect;
    let mut frames = dedup::deduplicate(res.frames, preset.hash_threshold);

    score::compute_raw(&mut frames);
    score::rank_normalize_and_score(&mut frames);

    let capacity = layout::grid_capacity(preset, aspect);
    // Equal layout, time order, no heroes: take the top-k then sort by time.
    let selected = score::select_top_k(&frames, capacity, MIN_GAP_FLOOR);
    let picked: Vec<ScoredFrame> = selected.iter().map(|&i| frames[i].clone()).collect();

    let lay = layout::compute(picked.len(), &[], aspect, preset);
    let jpeg = compose::render(&picked, &lay, &[])?;
    let file = "grid_extract.jpg";
    fs::write(out_dir.join(file), &jpeg).context("writing grid_extract.jpg")?;

    let meta = Metadata {
        source: input.display().to_string(),
        duration: end - start,
        settings: Settings {
            max_pixels: preset.max_pixels,
            quality: quality_label.to_string(),
            detail: detail_label.to_string(),
        },
        grids: vec![build_grid_meta(file, &picked, &[], lay.template)],
    };
    let json = serde_json::to_string_pretty(&meta).context("serializing metadata")?;
    fs::write(out_dir.join("metadata.json"), json).context("writing metadata.json")?;

    println!("wrote {} to {}", file, out_dir.display());
    Ok(())
}

/// Parse a `MM:SS-MM:SS` (or `SS-SS`) range into `(start, end)` seconds.
pub fn parse_range(s: &str) -> Result<(f32, f32)> {
    let (a, b) = s
        .split_once('-')
        .with_context(|| format!("range '{s}' must be START-END"))?;
    let start = parse_clock(a)?;
    let end = parse_clock(b)?;
    if end <= start {
        anyhow::bail!("range end ({end}s) must be after start ({start}s)");
    }
    Ok((start, end))
}

/// Parse `MM:SS`, `HH:MM:SS`, or plain seconds into seconds.
fn parse_clock(s: &str) -> Result<f32> {
    let parts: Vec<&str> = s.split(':').collect();
    let secs = match parts.as_slice() {
        [s] => s.parse::<f32>()?,
        [m, s] => m.parse::<f32>()? * 60.0 + s.parse::<f32>()?,
        [h, m, s] => h.parse::<f32>()? * 3600.0 + m.parse::<f32>()? * 60.0 + s.parse::<f32>()?,
        _ => anyhow::bail!("bad time '{s}'"),
    };
    Ok(secs)
}

/// Build the metadata record for one rendered grid.
fn build_grid_meta(
    file: &str,
    picked: &[ScoredFrame],
    heroes: &[usize],
    template: &'static str,
) -> GridMeta {
    let (first, last) = (
        picked.first().map(|f| f.ts).unwrap_or(0.0),
        picked.last().map(|f| f.ts).unwrap_or(0.0),
    );
    let key_frames = heroes
        .iter()
        .map(|&i| KeyFrameMeta {
            timestamp: compose::format_ts(picked[i].ts),
            score: picked[i].score.round() as u32,
            grid_position: "hero".to_string(),
        })
        .collect();
    GridMeta {
        file: file.to_string(),
        time_range: format!("{}-{}", compose::format_ts(first), compose::format_ts(last)),
        total_frames: picked.len(),
        key_frames,
        layout: template.to_string(),
    }
}

/// Largest timestamp span within a chunk.
fn chunk_span(chunk: &[ScoredFrame]) -> f32 {
    match (chunk.first(), chunk.last()) {
        (Some(a), Some(b)) => (b.ts - a.ts).max(0.0),
        _ => 0.0,
    }
}

/// Partition frames (assumed time-ordered) into chunks of `seconds` each, keyed
/// by `floor(ts / seconds)`. Empty chunks are not emitted.
fn chunk_by_seconds(frames: Vec<ScoredFrame>, seconds: f32) -> Vec<Vec<ScoredFrame>> {
    let mut chunks: Vec<Vec<ScoredFrame>> = Vec::new();
    let mut cur_key = usize::MAX;
    for f in frames {
        let key = (f.ts / seconds).floor() as usize;
        if key != cur_key {
            chunks.push(Vec::new());
            cur_key = key;
        }
        chunks.last_mut().expect("just pushed").push(f);
    }
    chunks
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::RawScores;
    use image::{GrayImage, RgbImage};

    fn frame(ts: f32) -> ScoredFrame {
        ScoredFrame {
            ts,
            gray: GrayImage::new(1, 1),
            rgb: RgbImage::new(1, 1),
            raw: RawScores::default(),
            score: 0.0,
        }
    }

    #[test]
    fn splits_frames_at_minute_boundaries() {
        let frames = vec![frame(0.0), frame(30.0), frame(59.9), frame(60.1), frame(125.0)];
        let chunks = chunk_by_seconds(frames, 60.0);
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].len(), 3);
        assert_eq!(chunks[1].len(), 1);
        assert_eq!(chunks[2].len(), 1);
    }

    #[test]
    fn keeps_single_chunk_for_short_clip() {
        let frames = vec![frame(0.0), frame(5.0), frame(10.0)];
        let chunks = chunk_by_seconds(frames, 60.0);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].len(), 3);
    }

    #[test]
    fn parses_mmss_range() {
        let (s, e) = parse_range("01:23-01:35").expect("valid range");
        assert_eq!(s, 83.0);
        assert_eq!(e, 95.0);
    }

    #[test]
    fn parses_plain_seconds_range() {
        let (s, e) = parse_range("5-12").expect("valid range");
        assert_eq!((s, e), (5.0, 12.0));
    }

    #[test]
    fn rejects_inverted_range() {
        assert!(parse_range("01:35-01:23").is_err());
    }
}
