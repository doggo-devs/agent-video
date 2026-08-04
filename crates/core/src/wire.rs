// SPDX-License-Identifier: Apache-2.0
//! The JSON output contract, shared by producer and consumer.
//!
//! `process`/`extract` serialize these types (stdout in `--json` mode,
//! `metadata.json` in file mode), and the harness deserializes the very same
//! types from the self-spawned child's stdout — one definition, so a field
//! rename is a compile error in the consumer instead of a runtime parse
//! failure. Rich pipeline types convert *into* this module at the output
//! boundary ([`Grid::new`]) and are never reconstructed from it: wire
//! shapes are plain data with no invariants to bypass.

use serde::{Deserialize, Serialize};

use crate::types::{
    GridImage, GridMeta, GridSubtitles, KeyFrameMeta, LayoutKind, Settings, TimeRange,
};

/// Top-level output document of one run.
///
/// `duration` and `range` are a mode pair — the discrimination rule is the
/// wire contract: `range` present marks an `extract` document and `duration`
/// is that segment's span; `range` absent marks a `process` document and
/// `duration` is the total source duration in seconds (probed when
/// available, floored by the frame-derived estimate). The pair is private so
/// the two readings cannot be confused: construct through
/// [`Metadata::process`] / [`Metadata::extract`], read through
/// [`Metadata::coverage`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Metadata {
    /// The input path or URL, echoed verbatim.
    pub source: String,
    duration: f32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    range: Option<Segment>,
    /// The resolved settings echoed back to the consumer.
    pub settings: Settings,
    /// One entry per rendered grid, in time order.
    pub grids: Vec<Grid>,
}

impl Metadata {
    /// A `process` document: `duration` is the total source duration.
    pub fn process(source: String, duration: f32, settings: Settings, grids: Vec<Grid>) -> Self {
        Metadata {
            source,
            duration,
            range: None,
            settings,
            grids,
        }
    }

    /// An `extract` document: the wire `duration` is derived here from the
    /// segment — always the harvested span, never a source total.
    pub fn extract(source: String, segment: Segment, settings: Settings, grids: Vec<Grid>) -> Self {
        Metadata {
            source,
            duration: segment.span(),
            range: Some(segment),
            settings,
            grids,
        }
    }

    /// The mode-discriminated reading of the `duration`/`range` pair.
    pub fn coverage(&self) -> Coverage {
        match self.range {
            Some(segment) => Coverage::Segment(segment),
            None => Coverage::Source(self.duration),
        }
    }

    /// Total source duration; `None` for `extract` documents, whose
    /// `duration` is a segment span, not a total.
    pub fn total_duration(&self) -> Option<f32> {
        self.range.is_none().then_some(self.duration)
    }

    /// The harvested segment; `None` for `process` documents.
    pub fn segment(&self) -> Option<Segment> {
        self.range
    }
}

/// What a document's `duration` measures, discriminated by `range`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Coverage {
    /// A `process` document: the total source duration in seconds.
    Source(f32),
    /// An `extract` document: one harvested segment of the source.
    Segment(Segment),
}

/// A harvested sub-range of the source, in seconds. On the wire it is the
/// `[start, end]` two-element array `metadata.json` has always carried;
/// deserialization enforces what construction guarantees: both bounds
/// finite, `start <= end`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "(f32, f32)", into = "(f32, f32)")]
pub struct Segment {
    start: f32,
    end: f32,
}

impl Segment {
    /// Start of the segment in seconds.
    pub fn start(self) -> f32 {
        self.start
    }

    /// End of the segment in seconds; at least `start`.
    pub fn end(self) -> f32 {
        self.end
    }

    /// Seconds covered; non-negative by construction.
    pub fn span(self) -> f32 {
        self.end - self.start
    }
}

impl From<TimeRange> for Segment {
    fn from(r: TimeRange) -> Self {
        Segment {
            start: r.start(),
            end: r.end(),
        }
    }
}

impl From<Segment> for (f32, f32) {
    fn from(s: Segment) -> (f32, f32) {
        (s.start, s.end)
    }
}

impl TryFrom<(f32, f32)> for Segment {
    type Error = String;

    fn try_from((start, end): (f32, f32)) -> Result<Self, Self::Error> {
        TimeRange::new(start, end)
            .map(Segment::from)
            .ok_or_else(|| format!("range [{start}, {end}] must be finite and ordered"))
    }
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
    /// Number of frames composed onto the grid.
    pub total_frames: usize,
    /// Hero frames; uniform grids carry none. Always serialized — an empty
    /// list is `[]` on the wire — while `default` lets the consumer read
    /// documents that omit the key.
    #[serde(default)]
    pub key_frames: Vec<KeyFrameMeta>,
    /// Layout template label.
    pub layout: LayoutKind,
    /// `MM:SS text` cue lines assigned to this grid (every cue in the
    /// processed range lands on exactly one grid); absent when the run
    /// resolved no subtitle source or this grid won no cue.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub subtitles: Option<String>,
}

impl Grid {
    /// Pair one rendered grid with its resolved subtitle outcome — the only
    /// door onto the wire shape, so a grid cannot reach `metadata.json`
    /// before subtitle resolution has run for it.
    pub fn new(meta: GridMeta, subtitles: GridSubtitles) -> Self {
        Grid {
            image: meta.image,
            time_range: meta.time_range.to_string(),
            total_frames: meta.total_frames,
            key_frames: meta.key_frames,
            layout: meta.layout,
            subtitles: match subtitles {
                GridSubtitles::Text(text) => Some(text),
                GridSubtitles::NoSource | GridSubtitles::NoCueInRange => None,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Detail, GridPosition, Quality};

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
        }
    }

    fn grid(image: GridImage) -> Grid {
        Grid::new(
            grid_meta(image),
            GridSubtitles::Text("00:01 hi".to_string()),
        )
    }

    fn settings() -> Settings {
        Settings {
            max_edge: 2048,
            quality: Quality::Low,
            detail: Detail::Standard,
        }
    }

    fn doc(grids: Vec<Grid>) -> Metadata {
        Metadata::process("v.mp4".to_string(), 90.0, settings(), grids)
    }

    fn segment(start: f32, end: f32) -> Segment {
        Segment::try_from((start, end)).expect("valid segment")
    }

    #[test]
    fn round_trips_through_json() {
        let meta = doc(vec![grid(GridImage::Base64("QUJD".to_string()))]);
        let json = serde_json::to_string(&meta).expect("serialize");
        let back: Metadata = serde_json::from_str(&json).expect("deserialize own output");
        assert_eq!(back.source, "v.mp4");
        assert_eq!(back.coverage(), Coverage::Source(90.0));
        assert_eq!(back.total_duration(), Some(90.0));
        assert_eq!(back.segment(), None);
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
        let json = serde_json::to_value(grid(GridImage::File("grid_001.jpg".to_string())))
            .expect("serialize file grid");
        assert_eq!(json["file"], "grid_001.jpg");
        assert!(json.get("base64").is_none(), "file variant omits base64");

        let json = serde_json::to_value(grid(GridImage::Base64("QUJD".to_string())))
            .expect("serialize base64 grid");
        assert_eq!(json["base64"], "QUJD");
        assert!(json.get("file").is_none(), "base64 variant omits file");
    }

    #[test]
    fn omits_absent_range_and_subtitles() {
        let rich = Grid::new(
            grid_meta(GridImage::File("grid_001.jpg".to_string())),
            GridSubtitles::NoSource,
        );
        let json = serde_json::to_value(doc(vec![rich])).expect("serialize");
        assert!(
            json.get("range").is_none(),
            "process documents carry no range key"
        );
        assert!(
            json["grids"][0].get("subtitles").is_none(),
            "non-text subtitle states serialize as absent"
        );
    }

    /// Empty `key_frames` still serializes as `[]`: the frozen wire shape
    /// carries the key unconditionally.
    #[test]
    fn serializes_empty_key_frames_as_empty_array() {
        let mut meta = grid_meta(GridImage::File("grid_001.jpg".to_string()));
        meta.key_frames = Vec::new();
        let json =
            serde_json::to_string(&Grid::new(meta, GridSubtitles::NoSource)).expect("serialize");
        assert!(
            json.contains(r#""key_frames":[]"#),
            "empty key_frames must stay on the wire as []: {json}"
        );
    }

    #[test]
    fn defaults_absent_key_frames_to_empty_on_read() {
        let json =
            r#"{"file":"g.jpg","time_range":"00:00-01:30","total_frames":1,"layout":"UNIFORM"}"#;
        let grid: Grid = serde_json::from_str(json).expect("deserialize without key_frames");
        assert!(grid.key_frames.is_empty());
    }

    #[test]
    fn rejects_grid_with_no_image_payload() {
        // The one-of survives deserialization: a grid missing both `file`
        // and `base64` is not representable.
        let json = r#"{"time_range": "00:00-01:30", "total_frames": 1, "layout": "UNIFORM"}"#;
        assert!(serde_json::from_str::<Grid>(json).is_err());
    }

    /// Byte-for-byte pin of a `process` document — the compatibility proof
    /// for the public `metadata.json` / `--json` contract.
    #[test]
    fn keeps_process_wire_json_byte_identical() {
        let meta = doc(vec![grid(GridImage::File("grid_001.jpg".to_string()))]);
        assert_eq!(
            serde_json::to_string(&meta).expect("serialize"),
            r#"{"source":"v.mp4","duration":90.0,"settings":{"max_edge":2048,"quality":"low","detail":"standard"},"grids":[{"file":"grid_001.jpg","time_range":"00:00-01:30","total_frames":12,"key_frames":[{"timestamp":"00:10","score":5,"grid_position":"hero"}],"layout":"UNIFORM","subtitles":"00:01 hi"}]}"#
        );
    }

    /// Byte-for-byte pin of an `extract` document: `range` stays a bare
    /// `[start, end]` array and `duration` its span.
    #[test]
    fn keeps_extract_wire_json_byte_identical() {
        let meta = Metadata::extract(
            "v.mp4".to_string(),
            segment(1.0, 3.0),
            settings(),
            vec![grid(GridImage::File("grid_001.jpg".to_string()))],
        );
        assert_eq!(
            serde_json::to_string(&meta).expect("serialize"),
            r#"{"source":"v.mp4","duration":2.0,"range":[1.0,3.0],"settings":{"max_edge":2048,"quality":"low","detail":"standard"},"grids":[{"file":"grid_001.jpg","time_range":"00:00-01:30","total_frames":12,"key_frames":[{"timestamp":"00:10","score":5,"grid_position":"hero"}],"layout":"UNIFORM","subtitles":"00:01 hi"}]}"#
        );
    }

    #[test]
    fn round_trips_segment_as_tuple() {
        let json = serde_json::to_string(&segment(1.0, 3.0)).expect("serialize");
        assert_eq!(json, "[1.0,3.0]");
        let back: Segment = serde_json::from_str(&json).expect("deserialize");
        assert_eq!((back.start(), back.end(), back.span()), (1.0, 3.0, 2.0));
    }

    #[test]
    fn rejects_inverted_segment_on_deserialize() {
        assert!(serde_json::from_str::<Segment>("[3.0,1.0]").is_err());
    }

    #[test]
    fn reads_extract_coverage_as_segment() {
        let meta = Metadata::extract("v.mp4".to_string(), segment(1.0, 3.0), settings(), vec![]);
        assert_eq!(meta.total_duration(), None, "span is not a total");
        let Coverage::Segment(seg) = meta.coverage() else {
            panic!("extract documents read as segments");
        };
        assert_eq!((seg.start(), seg.end()), (1.0, 3.0));
    }
}
