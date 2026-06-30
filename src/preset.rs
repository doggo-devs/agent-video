//! Preset resolution: merge a quality tier and a detail tier into a single
//! [`Preset`]. The layout engine reads from `max_pixels` (canvas edge length).

use crate::types::{Detail, Preset, Quality};

/// Minimum readable frame edge and canvas scale for a quality tier.
fn quality_params(q: Quality) -> (u32, f32) {
    match q {
        Quality::Low => (256, 0.5),
        Quality::Medium => (384, 0.75),
        Quality::High => (512, 1.0),
    }
}

/// Base fps and pHash Hamming threshold for a detail tier.
fn detail_params(d: Detail) -> (f32, u32) {
    match d {
        Detail::Coarse => (0.5, 4),
        Detail::Standard => (1.0, 8),
        Detail::Fine => (3.0, 12),
    }
}

/// Merge quality and detail tiers into a resolved [`Preset`].
///
/// `max_edge` is taken as-is (canvas edge length). `min_frame_size` and
/// `canvas_scale` come from `quality`; `base_fps` and `hash_threshold` from
/// `detail`.
pub fn resolve_preset(max_edge: u32, quality: Quality, detail: Detail) -> Preset {
    let (min_frame_size, canvas_scale) = quality_params(quality);
    let (base_fps, hash_threshold) = detail_params(detail);
    Preset {
        max_edge,
        min_frame_size,
        canvas_scale,
        base_fps,
        hash_threshold,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merges_high_standard_tiers() {
        let p = resolve_preset(8000, Quality::High, Detail::Standard);
        assert_eq!(p.min_frame_size, 512);
        assert_eq!(p.canvas_scale, 1.0);
        assert_eq!(p.base_fps, 1.0);
        assert_eq!(p.hash_threshold, 8);
    }
}
