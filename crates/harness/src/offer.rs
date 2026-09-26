// SPDX-License-Identifier: Apache-2.0
//! Download offers: expose the source's available tiers and size estimates.
//! Nothing is downloaded without an echoed consent token.

use super::fetch::SourceInfo;
use super::plan::Decision;
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
pub fn decide_url(args: &VideoArgs, info: &SourceInfo) -> Decision {
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

    if info.options.is_empty() {
        return Decision::Reject {
            message: format!("{} No downloadable formats are available.", info.headline),
        };
    }
    let offers = info
        .options
        .iter()
        .map(|opt| OfferDraft {
            effect: opt.effect.clone(),
            action: PendingAction::Download(DownloadPlan {
                url: args.source.clone(),
                dest: opt.dest,
                recipe: opt.recipe.clone(),
                estimated_bytes: opt.estimated_bytes,
                title: info.title.clone(),
            }),
        })
        .collect();
    let mut notes = vec![info.headline.clone()];
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
    use crate::fetch::{DownloadOption, Recipe, direct::DirectRecipe};
    use crate::schema::Look;
    use crate::tokens::Dest;

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
    fn offers_large_and_unknown_sizes_without_filtering_tiers() {
        let args = VideoArgs::bare("https://ex.com/v");
        let large = 20 * 1024 * 1024 * 1024;
        let d = decide_url(
            &args,
            &info(vec![
                option("keep-2160", Dest::Keep, Some(large)),
                option("analysis", Dest::Cache, None),
            ]),
        );
        let Decision::Confirm { offers, .. } = d else {
            panic!("expected download offers")
        };
        let estimates: Vec<_> = offers
            .iter()
            .map(|offer| {
                let PendingAction::Download(plan) = &offer.action else {
                    panic!("download plan")
                };
                plan.estimated_bytes
            })
            .collect();
        assert_eq!(estimates, vec![Some(large), None]);
    }

    #[test]
    fn rejects_raw_on_undownloaded_url() {
        let args = VideoArgs {
            raw: Some("process".into()),
            ..VideoArgs::bare("https://ex.com/v")
        };
        let d = decide_url(&args, &info(vec![]));
        assert!(matches!(d, Decision::Reject { .. }), "{d:?}");
    }

    #[test]
    fn notes_deferred_intent() {
        let args = VideoArgs {
            look: Some(Look::Overview),
            ..VideoArgs::bare("https://ex.com/v")
        };
        let d = decide_url(&args, &info(vec![option("analysis", Dest::Cache, None)]));
        let Decision::Confirm { consequence, .. } = d else {
            panic!("expected Confirm, got {d:?}");
        };
        assert!(consequence.contains("cannot apply yet"), "{consequence}");
    }
}
