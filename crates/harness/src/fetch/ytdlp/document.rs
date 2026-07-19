// SPDX-License-Identifier: Apache-2.0
//! The `yt-dlp -J` probe document: its wire model, the pure reduction to
//! site facts ([`site_info`]), and the offer built from those facts
//! ([`site_source_info`]). Nothing here spawns a process — the parent
//! module owns yt-dlp itself.

use std::collections::BTreeMap;

use serde::Deserialize;

use super::super::{DownloadOption, Recipe, SourceInfo, estimate_label};
use super::YtdlpRecipe;
use crate::config::Config;
use crate::plan::format_bytes;
use crate::tokens::{Dest, Token};
use agent_video_core::types::format_ts;

/// How many keep tiers (distinct heights) to offer at most.
pub const KEEP_TIER_LIMIT: usize = 3;

#[derive(Debug, Default, Deserialize)]
pub struct YtdlpFormat {
    /// `Some` = the rendition's pixel height. Live streams and some HLS
    /// variants report none (`null`); a non-positive value dies at this
    /// boundary too, so no consumer ever compares against 0.
    #[serde(default, deserialize_with = "positive_height")]
    pub height: Option<u32>,
    /// `Some` = the format carries video. yt-dlp encodes absence as the
    /// literal string `"none"`; that sentinel dies at this boundary.
    #[serde(default, deserialize_with = "none_is_absent")]
    pub vcodec: Option<String>,
    /// `Some` = the format carries audio; same sentinel rule as `vcodec`.
    #[serde(default, deserialize_with = "none_is_absent")]
    pub acodec: Option<String>,
    #[serde(default)]
    pub filesize: Option<f64>,
    #[serde(default)]
    pub filesize_approx: Option<f64>,
}

/// Map yt-dlp's `"none"` codec sentinel to a plain absent value, so no
/// in-memory consumer ever compares against the magic string.
fn none_is_absent<'de, D>(deserializer: D) -> Result<Option<String>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let value = Option::<String>::deserialize(deserializer)?;
    Ok(value.filter(|v| v != "none"))
}

/// Map an absent, `null`, or non-positive height to a plain absent value;
/// like the codec sentinel above, it dies at the wire boundary.
fn positive_height<'de, D>(deserializer: D) -> Result<Option<u32>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let value = Option::<f64>::deserialize(deserializer)?;
    Ok(value.filter(|&h| h >= 1.0).map(|h| h as u32))
}

#[derive(Debug, Default, Deserialize)]
pub struct YtdlpJson {
    #[serde(default)]
    pub title: Option<String>,
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub duration: Option<f64>,
    #[serde(default)]
    pub subtitles: Option<BTreeMap<String, serde_json::Value>>,
    #[serde(default)]
    pub formats: Option<Vec<YtdlpFormat>>,
}

/// One keep tier derived from a site's format list.
#[derive(Debug, Clone, PartialEq)]
pub struct KeepTier {
    /// Pixel height, when the site reports one; renditions without it
    /// (live streams, some HLS variants) form their own tier rather than
    /// vanish from the offer.
    pub height: Option<u32>,
    pub estimated_bytes: Option<u64>,
}

/// A site page probed via `yt-dlp -J`.
#[derive(Debug, Clone, PartialEq)]
pub struct SiteUrlInfo {
    pub title: String,
    /// Seconds; live streams may not report one.
    pub duration: Option<f64>,
    pub subtitle_langs: Vec<String>,
    /// Distinct available renditions, best first, already capped in count;
    /// a tier may lack a reported height.
    pub keep_tiers: Vec<KeepTier>,
    /// Estimate for the analysis rip (best video ≤ cap plus audio).
    pub analysis_estimated_bytes: Option<u64>,
}

fn size_of(f: &YtdlpFormat) -> Option<u64> {
    f.filesize
        .or(f.filesize_approx)
        .filter(|&n| n > 0.0)
        .map(|n| n as u64)
}

fn is_video(f: &YtdlpFormat) -> bool {
    f.vcodec.is_some()
}

fn is_audio_only(f: &YtdlpFormat) -> bool {
    f.acodec.is_some() && f.vcodec.is_none()
}

/// Largest known size among `fmts`, or `None` when none reports one.
fn max_size<'a>(fmts: impl Iterator<Item = &'a YtdlpFormat>) -> Option<u64> {
    fmts.filter_map(size_of).max()
}

/// Estimate a rendition pool the way the download selector picks from it:
/// elect one representative row — the largest known size, the closest
/// observable proxy for yt-dlp's "best" — and read size and audio-ness off
/// that single row, adding best audio only when the electee lacks it.
/// `None` when no row in the pool reports a size.
fn estimate<'a>(
    pool: impl Iterator<Item = &'a YtdlpFormat>,
    audio_size: Option<u64>,
) -> Option<u64> {
    let (size, muxed) = pool
        .filter_map(|f| size_of(f).map(|s| (s, f.acodec.is_some())))
        .max_by_key(|&(s, _)| s)?;
    Some(size + if muxed { 0 } else { audio_size.unwrap_or(0) })
}

/// Reduce a `-J` document to [`SiteUrlInfo`]; pure. Size estimates mirror
/// download selectors: keep = best video at tier + best audio, analysis =
/// best video-only rendition at or under `analysis_max_height`, plus audio.
pub fn site_info(doc: &YtdlpJson, analysis_max_height: u32) -> SiteUrlInfo {
    let empty = Vec::new();
    let formats = doc.formats.as_ref().unwrap_or(&empty);
    let video: Vec<&YtdlpFormat> = formats.iter().filter(|f| is_video(f)).collect();
    let audio_size = max_size(formats.iter().filter(|f| is_audio_only(f)));

    // Distinct heights, best first; renditions without a reported height
    // form one trailing tier instead of vanishing from the offer.
    let mut heights: Vec<Option<u32>> = video.iter().map(|f| f.height).collect();
    heights.sort_unstable_by_key(|&h| (h.is_none(), std::cmp::Reverse(h)));
    heights.dedup();
    heights.truncate(KEEP_TIER_LIMIT);

    let keep_tiers: Vec<KeepTier> = heights
        .into_iter()
        .map(|height| KeepTier {
            height,
            estimated_bytes: estimate(
                video.iter().copied().filter(|f| f.height == height),
                audio_size,
            ),
        })
        .collect();

    // A rendition that reports no height cannot be excluded by the height
    // cap; the analysis selector's final `/b` fallback would take it anyway.
    let under_cap = |f: &&YtdlpFormat| f.height.is_none_or(|h| h <= analysis_max_height);
    let video_only: Vec<&YtdlpFormat> = video
        .iter()
        .copied()
        .filter(|f| under_cap(f) && f.acodec.is_none())
        .collect();
    let pool: Vec<&YtdlpFormat> = if video_only.is_empty() {
        video.iter().copied().filter(under_cap).collect()
    } else {
        video_only
    };
    // The selector takes the highest rendition under the cap first, so the
    // estimate reads the same slice; all-unknown heights leave the whole
    // pool in play.
    let best_height = pool.iter().filter_map(|f| f.height).max();
    let analysis_estimated_bytes = estimate(
        pool.iter()
            .copied()
            .filter(|f| best_height.is_none() || f.height == best_height),
        audio_size,
    );

    SiteUrlInfo {
        title: doc
            .title
            .clone()
            .or_else(|| doc.id.clone())
            .unwrap_or_else(|| "untitled".to_string()),
        duration: doc.duration.filter(|&d| d > 0.0),
        subtitle_langs: doc
            .subtitles
            .as_ref()
            .map(|s| s.keys().cloned().collect())
            .unwrap_or_default(),
        keep_tiers,
        analysis_estimated_bytes,
    }
}

/// Turn site facts into the source info: analysis rip plus keep tiers.
pub fn site_source_info(info: &SiteUrlInfo, cfg: &Config) -> SourceInfo {
    let h = cfg.download.analysis_max_height.get();
    let duration = info
        .duration
        .map(|d| format_ts(d as f32))
        .unwrap_or_else(|| "unknown duration".to_string());
    let subs = if info.subtitle_langs.is_empty() {
        "no subtitles listed".to_string()
    } else {
        format!("subtitles: {}", info.subtitle_langs.join(", "))
    };

    let mut options = vec![DownloadOption {
        token: Token::new("dl-analysis"),
        dest: Dest::Cache,
        label: match info.analysis_estimated_bytes {
            Some(n) => format!("analysis rip ({})", format_bytes(n)),
            None => "analysis rip".to_string(),
        },
        effect: format!(
            "download a rip (≤{h}p video + audio, {}) plus subtitles into the analysis cache \
             (evictable)",
            estimate_label(info.analysis_estimated_bytes)
        ),
        estimated_bytes: info.analysis_estimated_bytes,
        recipe: Recipe::Ytdlp(YtdlpRecipe {
            format: format!("bv[height<={h}]+ba/bv*[height<={h}]+ba/b[height<={h}]/b"),
            subs: true,
        }),
    }];
    for tier in &info.keep_tiers {
        // A tier without a reported height is still downloadable: the plain
        // best selector fetches it, and the offer states what is known.
        let (slug, name, bound, selector) = match tier.height {
            Some(h) => (
                h.to_string(),
                format!("{h}p"),
                format!("≤{h}p"),
                format!("bv*[height<={h}]+ba/b[height<={h}]/b"),
            ),
            None => (
                "best".to_string(),
                "best available".to_string(),
                "the source's best quality (height unreported)".to_string(),
                "bv*+ba/b".to_string(),
            ),
        };
        options.push(DownloadOption {
            token: Token::new(format!("dl-keep-{slug}")),
            dest: Dest::Keep,
            label: match tier.estimated_bytes {
                Some(n) => format!("{name} ({})", format_bytes(n)),
                None => name,
            },
            effect: format!(
                "save best audio+video at {bound} ({}) to {} for the user",
                estimate_label(tier.estimated_bytes),
                cfg.paths.keep.display()
            ),
            estimated_bytes: tier.estimated_bytes,
            recipe: Recipe::Ytdlp(YtdlpRecipe {
                format: selector,
                subs: false,
            }),
        });
    }

    SourceInfo {
        headline: format!(
            "\"{}\" — {duration}, {subs}. Nothing is downloaded yet.",
            info.title
        ),
        options,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    /// Mirrors the wire boundary: `"none"` codecs arrive as absent, and so
    /// does a missing height.
    fn fmt(height: Option<u32>, vcodec: &str, acodec: &str, size: Option<f64>) -> YtdlpFormat {
        YtdlpFormat {
            height,
            vcodec: (vcodec != "none").then(|| vcodec.to_string()),
            acodec: (acodec != "none").then(|| acodec.to_string()),
            filesize: size,
            filesize_approx: None,
        }
    }

    #[test]
    fn codec_none_sentinel_deserializes_to_absent() {
        let f: YtdlpFormat =
            serde_json::from_str(r#"{"vcodec":"none","acodec":"opus"}"#).expect("valid json");
        assert_eq!(f.vcodec, None, "sentinel dies at the wire");
        assert_eq!(f.acodec.as_deref(), Some("opus"));
    }

    #[test]
    fn non_positive_height_deserializes_to_absent() {
        let f: YtdlpFormat =
            serde_json::from_str(r#"{"height":0,"vcodec":"avc1"}"#).expect("valid json");
        assert_eq!(f.height, None, "0-as-missing dies at the wire");
        let f: YtdlpFormat =
            serde_json::from_str(r#"{"height":720,"vcodec":"avc1"}"#).expect("valid json");
        assert_eq!(f.height, Some(720));
    }

    #[test]
    fn reduces_formats_to_keep_tiers_descending() {
        let doc = YtdlpJson {
            title: Some("T".into()),
            formats: Some(vec![
                fmt(Some(360), "avc1", "none", Some(10.0)),
                fmt(Some(720), "avc1", "none", Some(50.0)),
                fmt(Some(1080), "avc1", "none", Some(100.0)),
                fmt(Some(1440), "avc1", "none", Some(200.0)),
                fmt(None, "none", "opus", Some(5.0)),
            ]),
            ..Default::default()
        };
        let info = site_info(&doc, 480);
        let heights: Vec<Option<u32>> = info.keep_tiers.iter().map(|t| t.height).collect();
        assert_eq!(
            heights,
            [Some(1440), Some(1080), Some(720)],
            "top 3 distinct, descending"
        );
        // Video-only tiers add best audio.
        assert_eq!(info.keep_tiers[0].estimated_bytes, Some(205));
    }

    #[test]
    fn heightless_video_format_still_forms_a_keep_tier() {
        // A live/HLS rendition with a codec but no reported height must not
        // vanish from the offer: it becomes its own tier.
        let doc = YtdlpJson {
            formats: Some(vec![fmt(None, "avc1", "mp4a", Some(50.0))]),
            ..Default::default()
        };
        let info = site_info(&doc, 480);
        assert_eq!(
            info.keep_tiers,
            [KeepTier {
                height: None,
                estimated_bytes: Some(50),
            }]
        );
    }

    #[test]
    fn heightless_tier_sorts_after_known_heights() {
        let doc = YtdlpJson {
            formats: Some(vec![
                fmt(None, "avc1", "none", Some(40.0)),
                fmt(Some(720), "avc1", "none", Some(50.0)),
            ]),
            ..Default::default()
        };
        let info = site_info(&doc, 480);
        let heights: Vec<Option<u32>> = info.keep_tiers.iter().map(|t| t.height).collect();
        assert_eq!(heights, [Some(720), None], "known quality outranks unknown");
    }

    #[test]
    fn analysis_pool_admits_heightless_renditions() {
        // A height cap cannot exclude a rendition that reports no height;
        // the selector's `/b` fallback would fetch it, so the estimate
        // must count it too.
        let doc = YtdlpJson {
            formats: Some(vec![
                fmt(None, "avc1", "none", Some(40.0)),
                fmt(None, "none", "opus", Some(5.0)),
            ]),
            ..Default::default()
        };
        let info = site_info(&doc, 480);
        assert_eq!(info.analysis_estimated_bytes, Some(45));
    }

    #[test]
    fn tier_estimate_takes_size_and_audio_from_one_row() {
        // Muxed and video-only coexist at 720p and the video-only row is
        // larger: stitching "largest size" with "any row muxed" would
        // claim 100 and silently omit audio. The representative row is the
        // video-only 100, so best audio (5) must ride along.
        let doc = YtdlpJson {
            formats: Some(vec![
                fmt(Some(720), "avc1", "mp4a", Some(80.0)),
                fmt(Some(720), "avc1", "none", Some(100.0)),
                fmt(None, "none", "opus", Some(5.0)),
            ]),
            ..Default::default()
        };
        let info = site_info(&doc, 480);
        assert_eq!(info.keep_tiers[0].estimated_bytes, Some(105));
    }

    #[test]
    fn muxed_tier_does_not_add_audio() {
        let doc = YtdlpJson {
            formats: Some(vec![
                fmt(Some(720), "avc1", "mp4a", Some(80.0)),
                fmt(None, "none", "opus", Some(5.0)),
            ]),
            ..Default::default()
        };
        let info = site_info(&doc, 480);
        assert_eq!(info.keep_tiers[0].estimated_bytes, Some(80));
    }

    #[test]
    fn analysis_estimate_prefers_video_only_under_cap_plus_audio() {
        let doc = YtdlpJson {
            formats: Some(vec![
                fmt(Some(480), "avc1", "none", Some(30.0)),
                fmt(Some(360), "avc1", "none", Some(20.0)),
                fmt(Some(480), "avc1", "mp4a", Some(60.0)),
                fmt(None, "none", "opus", Some(5.0)),
            ]),
            ..Default::default()
        };
        let info = site_info(&doc, 480);
        assert_eq!(info.analysis_estimated_bytes, Some(35));
    }

    #[test]
    fn falls_back_to_muxed_when_no_video_only() {
        let doc = YtdlpJson {
            formats: Some(vec![fmt(Some(480), "avc1", "mp4a", Some(60.0))]),
            ..Default::default()
        };
        let info = site_info(&doc, 480);
        assert_eq!(info.analysis_estimated_bytes, Some(60));
    }

    #[test]
    fn untitled_falls_back_to_id_then_placeholder() {
        let doc = YtdlpJson {
            id: Some("abc123".into()),
            ..Default::default()
        };
        assert_eq!(site_info(&doc, 480).title, "abc123");
        assert_eq!(site_info(&YtdlpJson::default(), 480).title, "untitled");
    }

    #[test]
    fn source_info_offers_analysis_plus_tiers() {
        let cfg = crate::config::default_config(&crate::config::test_dirs(Path::new("/home/u")));
        let info = SiteUrlInfo {
            title: "Talk".into(),
            duration: Some(90.0),
            subtitle_langs: vec!["en".into()],
            keep_tiers: vec![KeepTier {
                height: Some(1080),
                estimated_bytes: Some(1024 * 1024),
            }],
            analysis_estimated_bytes: Some(512 * 1024),
        };
        let si = site_source_info(&info, &cfg);
        assert_eq!(si.options.len(), 2);
        assert_eq!(si.options[0].token, Token::new("dl-analysis"));
        assert_eq!(si.options[1].token, Token::new("dl-keep-1080"));
        assert!(si.headline.contains("\"Talk\""));
        assert!(si.headline.contains("subtitles: en"));
        let Recipe::Ytdlp(r) = &si.options[0].recipe else {
            panic!("analysis recipe is ytdlp");
        };
        assert_eq!(
            r.format,
            "bv[height<=480]+ba/bv*[height<=480]+ba/b[height<=480]/b"
        );
        assert!(r.subs);
    }

    #[test]
    fn offers_a_heightless_tier_with_the_plain_best_selector() {
        let cfg = crate::config::default_config(&crate::config::test_dirs(Path::new("/home/u")));
        let info = SiteUrlInfo {
            title: "Live".into(),
            duration: None,
            subtitle_langs: vec![],
            keep_tiers: vec![KeepTier {
                height: None,
                estimated_bytes: None,
            }],
            analysis_estimated_bytes: None,
        };
        let si = site_source_info(&info, &cfg);
        assert_eq!(si.options[1].token, Token::new("dl-keep-best"));
        assert_eq!(si.options[1].label, "best available");
        let Recipe::Ytdlp(r) = &si.options[1].recipe else {
            panic!("keep recipe is ytdlp");
        };
        assert_eq!(r.format, "bv*+ba/b", "no height cap without a height");
    }
}
