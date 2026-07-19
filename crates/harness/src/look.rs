// SPDX-License-Identifier: Apache-2.0
//! `look` presets: intent-level names mapped to core knobs.

use super::schema::Look;
use agent_video_core::config::{Profile, default_chunk_seconds};

/// The action profile's core flags. `Profile::Action` spells the flag
/// value, so the flag and the chunk-length lookup below cannot name
/// different profiles.
const ACTION_FLAGS: &[&str] = &["--profile", Profile::Action.cli_name(), "--quality", "low"];

/// The core knobs behind a single look value.
#[derive(Debug, Clone)]
pub struct LookPreset {
    /// Core flags ready to append; no I/O flags involved.
    pub flags: &'static [&'static str],
    /// Effective chunk seconds, i.e. the denominator of the grid count.
    pub chunk_seconds: f64,
}

/// The preset behind `look` (`None` = balanced default).
pub fn preset_for(look: Option<Look>) -> LookPreset {
    // (flags, the profile a preset rides on — chunk length resolves through
    // core's own table, never a copied number).
    let (flags, profile): (&'static [&'static str], Option<Profile>) = match look {
        None => (&["--detail", "standard", "--quality", "low"], None),
        Some(Look::Overview) => (&["--detail", "coarse", "--quality", "low"], None),
        Some(Look::Detailed) => (&["--detail", "fine", "--quality", "medium"], None),
        Some(Look::ReadText) => (&["--detail", "standard", "--quality", "high"], None),
        // Action profile favors scene changes and tightens min-gap; quality
        // low makes cells small and dense.
        Some(Look::TrackMotion) => (ACTION_FLAGS, Some(Profile::Action)),
    };
    LookPreset {
        flags,
        chunk_seconds: f64::from(default_chunk_seconds(profile)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn balanced_default_when_look_omitted() {
        let p = preset_for(None);
        assert_eq!(p.flags, ["--detail", "standard", "--quality", "low"]);
        assert_eq!(
            p.chunk_seconds,
            f64::from(agent_video_core::config::DEFAULT_CHUNK_SECONDS)
        );
    }

    #[test]
    fn track_motion_uses_action_profile() {
        let p = preset_for(Some(Look::TrackMotion));
        assert_eq!(p.flags, ["--profile", "action", "--quality", "low"]);
    }
}
