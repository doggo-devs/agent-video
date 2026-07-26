// SPDX-License-Identifier: Apache-2.0
//! `look` presets: intent-level names mapped to core knobs.

use super::schema::Look;
use agent_video_core::config::{Profile, default_chunk_seconds};
use agent_video_core::types::{Detail, Quality};

/// Sampling-density choice of a look: an explicit detail tier at the
/// default cadence, or a named core profile that brings its own tier and
/// chunk cadence. One choice by construction — a look naming both (which
/// core would let fight) or neither cannot exist.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sampling {
    Detail(Detail),
    Profile(Profile),
}

/// The core knobs behind a single look value. Rendered to core flags in
/// exactly one place (`LookKnobs::render_into`); the chunk cadence is
/// derived from the same value, so flags and prediction cannot disagree.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LookKnobs {
    pub sampling: Sampling,
    pub quality: Quality,
}

impl LookKnobs {
    /// Effective chunk seconds — the grid-count denominator — resolved
    /// through core's own profile table, so the prediction cannot drift
    /// from what the run will do.
    pub fn chunk_seconds(&self) -> f64 {
        let profile = match self.sampling {
            Sampling::Profile(p) => Some(p),
            Sampling::Detail(_) => None,
        };
        f64::from(default_chunk_seconds(profile))
    }

    /// Append this look's core flags; no I/O flags involved.
    pub(crate) fn render_into(&self, argv: &mut Vec<String>) {
        match self.sampling {
            Sampling::Detail(d) => {
                argv.extend(["--detail".to_string(), detail_name(d).to_string()]);
            }
            Sampling::Profile(p) => {
                argv.extend(["--profile".to_string(), p.cli_name().to_string()]);
            }
        }
        argv.extend([
            "--quality".to_string(),
            quality_name(self.quality).to_string(),
        ]);
    }
}

/// Core's CLI spelling of a detail tier — the clap `ValueEnum` names,
/// mirrored exhaustively so a new core tier fails to compile here until
/// it is spelled.
const fn detail_name(detail: Detail) -> &'static str {
    match detail {
        Detail::Coarse => "coarse",
        Detail::Standard => "standard",
        Detail::Fine => "fine",
    }
}

/// Core's CLI spelling of a quality tier; same mirror discipline as
/// [`detail_name`].
const fn quality_name(quality: Quality) -> &'static str {
    match quality {
        Quality::Low => "low",
        Quality::Medium => "medium",
        Quality::High => "high",
    }
}

/// The knobs behind `look` (`None` = balanced default).
pub fn preset_for(look: Option<Look>) -> LookKnobs {
    let (sampling, quality) = match look {
        None => (Sampling::Detail(Detail::Standard), Quality::Low),
        Some(Look::Overview) => (Sampling::Detail(Detail::Coarse), Quality::Low),
        Some(Look::Detailed) => (Sampling::Detail(Detail::Fine), Quality::Medium),
        Some(Look::ReadText) => (Sampling::Detail(Detail::Standard), Quality::High),
        // Action profile favors scene changes and tightens min-gap; quality
        // low makes cells small and dense.
        Some(Look::TrackMotion) => (Sampling::Profile(Profile::Action), Quality::Low),
    };
    LookKnobs { sampling, quality }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rendered(knobs: LookKnobs) -> Vec<String> {
        let mut argv = Vec::new();
        knobs.render_into(&mut argv);
        argv
    }

    #[test]
    fn balanced_default_when_look_omitted() {
        let knobs = preset_for(None);
        assert_eq!(knobs.sampling, Sampling::Detail(Detail::Standard));
        assert_eq!(knobs.quality, Quality::Low);
        assert_eq!(
            knobs.chunk_seconds(),
            f64::from(agent_video_core::config::DEFAULT_CHUNK_SECONDS)
        );
    }

    #[test]
    fn track_motion_uses_action_profile() {
        let knobs = preset_for(Some(Look::TrackMotion));
        assert_eq!(knobs.sampling, Sampling::Profile(Profile::Action));
        assert_eq!(
            knobs.chunk_seconds(),
            f64::from(default_chunk_seconds(Some(Profile::Action))),
            "cadence resolves through the same profile the flags name"
        );
    }

    #[test]
    fn renders_detail_look_in_core_spelling() {
        assert_eq!(
            rendered(preset_for(None)),
            ["--detail", "standard", "--quality", "low"]
        );
    }

    #[test]
    fn renders_profile_look_in_core_spelling() {
        assert_eq!(
            rendered(preset_for(Some(Look::TrackMotion))),
            ["--profile", "action", "--quality", "low"]
        );
    }
}
