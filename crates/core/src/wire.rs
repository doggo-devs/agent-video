// SPDX-License-Identifier: Apache-2.0
//! The JSON output contract, shared by producer and consumer.
//!
//! `process`/`extract` serialize these types (stdout in `--json` mode,
//! `metadata.json` in file mode), and the harness deserializes the very same
//! types from the self-spawned child's stdout — one definition, so a field
//! rename is a compile error in the consumer instead of a runtime parse
//! failure. Rich pipeline types convert *into* this module at the output
//! boundary ([`Grid::from`]) and are never reconstructed from it: wire
//! shapes are plain data with no invariants to bypass.

use serde::{Deserialize, Serialize};

use crate::types::{GridImage, GridMeta, GridSubtitles, KeyFrameMeta, LayoutKind, Settings};

/// Top-level output document of one run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Metadata {
    pub source: String,
    /// `process`: total source duration in seconds (probed when available,
    /// floored by the frame-derived estimate). `extract`: the span of
    /// [`Self::range`].
    pub duration: f32,
    /// The harvested sub-range `(start, end)` in seconds; present exactly
    /// when the document came from `extract`, so consumers can tell a
    /// segment's span from a video's total duration.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub range: Option<(f32, f32)>,
    pub settings: Settings,
    pub grids: Vec<Grid>,
}

/// One rendered grid image and its description.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Grid {
    /// Exactly one of `{"file": name}` / `{"base64": blob}`, flattened into
    /// this object under the variant's key.
    #[serde(flatten)]
    pub image: GridImage,
    /// `"MM:SS-MM:SS"` (`"H:MM:SS"` past an hour).
    pub time_range: String,
    pub total_frames: usize,
    #[serde(default)]
    pub key_frames: Vec<KeyFrameMeta>,
    pub layout: LayoutKind,
    /// Concatenated cue text overlapping the grid's range; absent when the
    /// run resolved no subtitle source or no cue overlaps.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub subtitles: Option<String>,
}

impl From<GridMeta> for Grid {
    fn from(meta: GridMeta) -> Self {
        Grid {
            image: meta.image,
            time_range: meta.time_range.to_string(),
            total_frames: meta.total_frames,
            key_frames: meta.key_frames,
            layout: meta.layout,
            subtitles: match meta.subtitles {
                GridSubtitles::Text(text) => Some(text),
                _ => None,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Detail, GridPosition, Quality, TimeRange};

    fn grid_meta(image: GridImage) -> GridMeta {
        GridMeta {
            image,
            time_range: TimeRange::new(0.0, 90.0).expect("valid range"),
            total_frames: 12,
            key_frames: vec![KeyFrameMeta {
                timestamp: "00:10".to_string(),
                score: 5,
                grid_position: GridPosition::Hero,
            }],
            layout: LayoutKind::Uniform,
            subtitles: GridSubtitles::Text("00:01 hi".to_string()),
        }
    }

    fn doc(grids: Vec<Grid>) -> Metadata {
        Metadata {
            source: "v.mp4".to_string(),
            duration: 90.0,
            range: None,
            settings: Settings {
                max_edge: 2048,
                quality: Quality::Low,
                detail: Detail::Standard,
            },
            grids,
        }
    }

    #[test]
    fn round_trips_through_json() {
        let meta = doc(vec![Grid::from(grid_meta(GridImage::Base64(
            "QUJD".to_string(),
        )))]);
        let json = serde_json::to_string(&meta).expect("serialize");
        let back: Metadata = serde_json::from_str(&json).expect("deserialize own output");
        assert_eq!(back.source, "v.mp4");
        assert_eq!(back.duration, 90.0);
        assert_eq!(back.range, None);
        assert_eq!(back.grids.len(), 1);
        let grid = &back.grids[0];
        assert!(matches!(&grid.image, GridImage::Base64(b) if b == "QUJD"));
        assert_eq!(grid.time_range, "00:00-01:30");
        assert_eq!(grid.key_frames[0].timestamp, "00:10");
        assert_eq!(grid.layout, LayoutKind::Uniform);
        assert_eq!(grid.subtitles.as_deref(), Some("00:01 hi"));
    }

    #[test]
    fn serializes_grid_image_under_original_key() {
        // The flattened enum must keep the exact `"file"` / `"base64"` JSON
        // keys, with only one present at a time.
        let json = serde_json::to_value(Grid::from(grid_meta(GridImage::File(
            "grid_001.jpg".to_string(),
        ))))
        .expect("serialize file grid");
        assert_eq!(json["file"], "grid_001.jpg");
        assert!(json.get("base64").is_none(), "file variant omits base64");

        let json =
            serde_json::to_value(Grid::from(grid_meta(GridImage::Base64("QUJD".to_string()))))
                .expect("serialize base64 grid");
        assert_eq!(json["base64"], "QUJD");
        assert!(json.get("file").is_none(), "base64 variant omits file");
    }

    #[test]
    fn omits_absent_range_and_subtitles() {
        let mut rich = grid_meta(GridImage::File("grid_001.jpg".to_string()));
        rich.subtitles = GridSubtitles::NoSource;
        let json = serde_json::to_value(doc(vec![rich.into()])).expect("serialize");
        assert!(
            json.get("range").is_none(),
            "process documents carry no range key"
        );
        assert!(
            json["grids"][0].get("subtitles").is_none(),
            "non-text subtitle states stay absent, matching the old contract"
        );
    }

    #[test]
    fn rejects_grid_with_no_image_payload() {
        // The one-of survives deserialization: a grid missing both `file`
        // and `base64` is not representable.
        let json = r#"{"time_range": "00:00-01:30", "total_frames": 1, "layout": "UNIFORM"}"#;
        assert!(serde_json::from_str::<Grid>(json).is_err());
    }
}
