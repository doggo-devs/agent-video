// SPDX-License-Identifier: Apache-2.0
//! URL-source decisions.
//!
//! Turns source facts into download tiers behind consent tokens, or refusals.
//! Nothing is downloaded without an echoed token; tiers over
//! `download.max_bytes` are dropped.

use super::config::Config;
use super::fetch::SourceInfo;
use super::plan::{Decision, format_bytes};
use super::schema::{Intent, VideoArgs};
use super::time::ArgError;
use super::tokens::{DownloadPlan, OfferDraft, PendingAction};

/// Model-facing guidance attached to every download offer.
const CHOICE_GUIDANCE: &str = "The analysis option fetches a low-res rip plus subtitles into an evictable \
     cache — enough for you to watch and report on the video. If the user wants \
     the video for themselves, pick an option whose effect saves the file for \
     the user instead: never serve them the analysis rip. Echo exactly one \
     token via confirm.";

/// First contact with a not-yet-downloaded URL: offer tiers behind tokens,
/// or refuse. Pure: the offers ride the returned decision.
pub fn decide_url(args: &VideoArgs, info: &SourceInfo, cfg: &Config) -> Decision {
    let intent = match args.intent() {
        Ok(intent) => intent,
        Err(ArgError(message)) => return Decision::Reject { message },
    };
    if matches!(intent, Intent::Raw(_)) {
        return Decision::Reject {
            message: "raw needs a local file. Call with just the URL first, confirm a download, \
                      then repeat the raw call."
                .to_string(),
        };
    }

    let max_bytes = cfg.download.max_bytes.get();
    let mut offers: Vec<OfferDraft> = Vec::new();
    let mut dropped: Vec<&str> = Vec::new();
    for opt in &info.options {
        if opt.estimated_bytes.is_some_and(|est| est > max_bytes) {
            dropped.push(&opt.label);
            continue;
        }
        offers.push(OfferDraft {
            effect: opt.effect.clone(),
            action: PendingAction::Download(DownloadPlan {
                url: args.source.clone(),
                dest: opt.dest,
                recipe: opt.recipe.clone(),
                estimated_bytes: opt.estimated_bytes,
                title: info.title.clone(),
            }),
        });
    }

    if offers.is_empty() {
        return Decision::Reject {
            message: format!(
                "{} Every available tier exceeds the {} download ceiling \
                 (download.max_bytes). Download the file manually and pass the local path as \
                 source.",
                info.headline,
                format_bytes(max_bytes)
            ),
        };
    }

    let mut notes: Vec<String> = vec![info.headline.clone()];
    if !dropped.is_empty() {
        // Display-side dedup: distinct tiers may share a human label.
        let mut labels: Vec<&str> = Vec::new();
        for label in dropped {
            if !labels.contains(&label) {
                labels.push(label);
            }
        }
        notes.push(format!(
            "Tiers over the {} ceiling were not offered: {}.",
            format_bytes(max_bytes),
            labels.join(", ")
        ));
    }
    if matches!(intent, Intent::Grid(grid) if !grid.is_bare()) {
        notes.push(
            "Your from/until/look cannot apply yet — the download comes first. Repeat them in \
             the call after it completes."
                .to_string(),
        );
    }
    notes.push(CHOICE_GUIDANCE.to_string());

    Decision::Confirm {
        consequence: notes.join(" "),
        offers,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{default_config, test_dirs};
    use crate::fetch::{DownloadOption, Recipe, direct::DirectRecipe};
    use crate::schema::Look;
    use crate::tokens::Dest;
    use std::path::Path;

    fn cfg() -> Config {
        default_config(&test_dirs(Path::new("/home/u")))
    }

    fn option(label: &str, dest: Dest, bytes: Option<u64>) -> DownloadOption {
        DownloadOption {
            dest,
            label: label.to_string(),
            effect: format!("{label} effect"),
            estimated_bytes: bytes,
            recipe: Recipe::Direct(DirectRecipe {
                ext: "mp4".to_string(),
                filename: "v.mp4".to_string(),
            }),
        }
    }

    fn info(options: Vec<DownloadOption>) -> SourceInfo {
        SourceInfo {
            headline: "A video.".to_string(),
            title: None,
            options,
        }
    }

    #[test]
    fn offers_tokens_behind_confirm() {
        let args = VideoArgs::bare("https://ex.com/v");
        let d = decide_url(
            &args,
            &info(vec![
                option("analysis", Dest::Cache, Some(1024)),
                option("keep-720", Dest::Keep, None),
            ]),
            &cfg(),
        );
        let Decision::Confirm {
            offers,
            consequence,
        } = d
        else {
            panic!("expected Confirm, got {d:?}");
        };
        assert_eq!(offers.len(), 2);
        assert!(consequence.contains("A video."));
        assert_eq!(offers[0].effect, "analysis effect");
        assert!(matches!(offers[0].action, PendingAction::Download(_)));
    }

    #[test]
    fn drops_oversized_tiers_with_note() {
        let args = VideoArgs::bare("https://ex.com/v");
        let over = cfg().download.max_bytes.get() + 1;
        let d = decide_url(
            &args,
            &info(vec![
                option("analysis", Dest::Cache, Some(1024)),
                option("keep-2160", Dest::Keep, Some(over)),
            ]),
            &cfg(),
        );
        let Decision::Confirm {
            offers,
            consequence,
        } = d
        else {
            panic!("expected Confirm, got {d:?}");
        };
        assert_eq!(offers.len(), 1);
        assert!(consequence.contains("were not offered"), "{consequence}");
    }

    #[test]
    fn refuses_when_every_tier_is_oversized() {
        let args = VideoArgs::bare("https://ex.com/v");
        let over = cfg().download.max_bytes.get() + 1;
        let d = decide_url(
            &args,
            &info(vec![option("analysis", Dest::Cache, Some(over))]),
            &cfg(),
        );
        let Decision::Reject { message } = d else {
            panic!("expected Reject, got {d:?}");
        };
        assert!(message.contains("download.max_bytes"), "{message}");
    }

    #[test]
    fn rejects_raw_on_undownloaded_url() {
        let args = VideoArgs {
            raw: Some("process".into()),
            ..VideoArgs::bare("https://ex.com/v")
        };
        let d = decide_url(&args, &info(vec![]), &cfg());
        assert!(matches!(d, Decision::Reject { .. }), "{d:?}");
    }

    #[test]
    fn notes_deferred_intent() {
        let args = VideoArgs {
            look: Some(Look::Overview),
            ..VideoArgs::bare("https://ex.com/v")
        };
        let d = decide_url(
            &args,
            &info(vec![option("analysis", Dest::Cache, None)]),
            &cfg(),
        );
        let Decision::Confirm { consequence, .. } = d else {
            panic!("expected Confirm, got {d:?}");
        };
        assert!(consequence.contains("cannot apply yet"), "{consequence}");
    }
}
