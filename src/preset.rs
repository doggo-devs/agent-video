//! Preset resolution: merge a model preset, a quality tier, and a detail tier
//! into a single [`Preset`]. The whole layout engine reads from `max_pixels`,
//! so changing target model is a one-number change.

use crate::types::{Detail, Preset, Quality};

/// Model presets only seed the `max_pixels` default; `--max-pixels` overrides.
///
/// `max_pixels` is a canvas **area** budget. Returns `None` for an unknown name
/// so the caller can surface a clear error.
pub fn model_max_pixels(model: &str) -> Option<u32> {
    let v = match model {
        "qwen2.5vl" => 8000,
        "gpt4o" => 2048,
        "claude" => 2048,
        "internvl" => 4096,
        _ => return None,
    };
    Some(v)
}

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

/// Merge the three tiers into a resolved [`Preset`].
///
/// `max_pixels` is taken as-is (already resolved from `--max-pixels` or the
/// model preset by the caller). `min_frame_size` and `canvas_scale` come from
/// `quality`; `base_fps` and `hash_threshold` from `detail`.
pub fn resolve_preset(max_pixels: u32, quality: Quality, detail: Detail) -> Preset {
    let (min_frame_size, canvas_scale) = quality_params(quality);
    let (base_fps, hash_threshold) = detail_params(detail);
    Preset {
        max_pixels,
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
    fn resolves_qwen_default_pixels() {
        assert_eq!(model_max_pixels("qwen2.5vl"), Some(8000));
    }

    #[test]
    fn rejects_unknown_model() {
        assert_eq!(model_max_pixels("nope"), None);
    }

    #[test]
    fn merges_high_standard_tiers() {
        let p = resolve_preset(8000, Quality::High, Detail::Standard);
        assert_eq!(p.min_frame_size, 512);
        assert_eq!(p.canvas_scale, 1.0);
        assert_eq!(p.base_fps, 1.0);
        assert_eq!(p.hash_threshold, 8);
    }
}
