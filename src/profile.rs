//! Profile system: named parameter presets for scoring, timing, and output.
//!
//! Profiles are resolved with priority: CLI overrides → named profile (built-in
//! or user file) → defaults. User profiles live at `~/.config/vidgrid/profiles.toml`.

use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

use anyhow::{Context, Result};
use serde::Deserialize;

/// Tunable parameters exposed through profiles.
#[derive(Debug, Clone, Copy)]
pub struct Profile {
    pub w_scene: f32,
    pub w_clarity: f32,
    pub w_edge: f32,

    /// Base sampling fps.
    pub base_fps: f32,
    /// pHash Hamming distance threshold for dedup.
    pub hash_threshold: u32,
    /// Dedup sliding window size in frames.
    pub dedup_window: usize,
    /// Edge length of grayscale image for scoring convolutions.
    pub score_edge: u32,

    /// Seconds per grid chunk.
    pub chunk_seconds: f32,
    /// Floor on temporal NMS gap in seconds.
    pub min_gap: f32,

    /// JPEG output quality 0-100.
    pub jpeg_quality: u8,
    /// Hero frame border thickness in pixels.
    pub hero_border_px: u32,
}

impl Default for Profile {
    fn default() -> Self {
        Self {
            w_scene: 0.40,
            w_clarity: 0.35,
            w_edge: 0.25,
            base_fps: 1.0,
            hash_threshold: 8,
            dedup_window: 8,
            score_edge: 256,
            chunk_seconds: 60.0,
            min_gap: 0.5,
            jpeg_quality: 92,
            hero_border_px: 4,
        }
    }
}

/// Partial profile for TOML deserialization and CLI overrides — all fields optional.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct ProfileOverrides {
    pub w_scene: Option<f32>,
    pub w_clarity: Option<f32>,
    pub w_edge: Option<f32>,
    pub base_fps: Option<f32>,
    pub hash_threshold: Option<u32>,
    pub dedup_window: Option<usize>,
    pub score_edge: Option<u32>,
    pub chunk_seconds: Option<f32>,
    pub min_gap: Option<f32>,
    pub jpeg_quality: Option<u8>,
    pub hero_border_px: Option<u32>,
}

impl Profile {
    fn apply(&mut self, patch: &ProfileOverrides) {
        macro_rules! merge {
            ($($field:ident),+ $(,)?) => {
                $(if let Some(v) = patch.$field { self.$field = v; })+
            };
        }
        merge!(
            w_scene,
            w_clarity,
            w_edge,
            base_fps,
            hash_threshold,
            dedup_window,
            score_edge,
            chunk_seconds,
            min_gap,
            jpeg_quality,
            hero_border_px,
        );
    }
}

/// Built-in named profiles.
fn builtin(name: &str) -> Option<Profile> {
    let mut p = Profile::default();
    match name {
        "balanced" => {}
        "fast" => {
            p.chunk_seconds = 90.0;
            p.min_gap = 1.0;
            p.jpeg_quality = 85;
        }
        "quality" => {
            p.chunk_seconds = 45.0;
            p.min_gap = 0.3;
            p.jpeg_quality = 95;
        }
        "action" => {
            p.w_scene = 0.55;
            p.w_clarity = 0.25;
            p.w_edge = 0.20;
            p.min_gap = 0.3;
        }
        _ => return None,
    }
    Some(p)
}

/// Load user profiles from `~/.config/vidgrid/profiles.toml`.
fn load_user_profiles() -> HashMap<String, ProfileOverrides> {
    let path = match dirs::config_dir() {
        Some(d) => d.join("vidgrid").join("profiles.toml"),
        None => return HashMap::new(),
    };
    if !path.exists() {
        return HashMap::new();
    }
    match load_profiles_from(&path) {
        Ok(profiles) => profiles,
        Err(e) => {
            eprintln!("warning: {e:#}");
            HashMap::new()
        }
    }
}

fn load_profiles_from(path: &PathBuf) -> Result<HashMap<String, ProfileOverrides>> {
    let content = fs::read_to_string(path).context("reading profiles.toml")?;
    toml::from_str(&content).context("parsing profiles.toml")
}

/// Resolve a profile by name with optional CLI overrides.
///
/// Priority: CLI overrides → named profile (user file → built-in) → defaults.
pub fn resolve(name: Option<&str>, overrides: &ProfileOverrides) -> Result<Profile> {
    let mut profile = Profile::default();

    if let Some(n) = name {
        let user_profiles = load_user_profiles();
        if let Some(patch) = user_profiles.get(n) {
            profile.apply(patch);
        } else if let Some(p) = builtin(n) {
            profile = p;
        } else {
            let available = list_available().join(", ");
            anyhow::bail!("unknown profile '{n}'; available: {available}");
        }
    }

    profile.apply(overrides);
    Ok(profile)
}

/// List available profile names (built-in + user-defined).
pub fn list_available() -> Vec<String> {
    let mut names: Vec<String> = vec![
        "balanced".into(),
        "fast".into(),
        "quality".into(),
        "action".into(),
    ];
    for key in load_user_profiles().keys() {
        if !names.contains(key) {
            names.push(key.clone());
        }
    }
    names.sort();
    names
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_weights_sum_to_one() {
        let p = Profile::default();
        let sum = p.w_scene + p.w_clarity + p.w_edge;
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn builtin_fast_differs_from_default() {
        let fast = builtin("fast").unwrap();
        let def = Profile::default();
        assert!(fast.chunk_seconds > def.chunk_seconds);
        assert!(fast.jpeg_quality < def.jpeg_quality);
    }

    #[test]
    fn overrides_take_precedence() {
        let overrides = ProfileOverrides {
            jpeg_quality: Some(50),
            ..Default::default()
        };
        let p = resolve(Some("quality"), &overrides).unwrap();
        assert_eq!(p.jpeg_quality, 50);
    }

    #[test]
    fn unknown_profile_errors() {
        let r = resolve(Some("nonexistent"), &ProfileOverrides::default());
        assert!(r.is_err());
    }
}
