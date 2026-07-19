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
    #[serde(default)]
    pub height: Option<f64>,
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
    pub height: u32,
    pub estimated_bytes: Option<u64>,
}

/// A site page probed via `yt-dlp -J`.
#[derive(Debug, Clone, PartialEq)]
pub struct SiteUrlInfo {
    pub title: String,
    /// Seconds; live streams may not report one.
    pub duration: Option<f64>,
    pub subtitle_langs: Vec<String>,
    /// Distinct available heights, descending, already capped in count.
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

fn height_of(f: &YtdlpFormat) -> u32 {
    f.height.filter(|&h| h > 0.0).map(|h| h as u32).unwrap_or(0)
}

fn is_video(f: &YtdlpFormat) -> bool {
    f.vcodec.is_some() && height_of(f) > 0
}

fn is_audio_only(f: &YtdlpFormat) -> bool {
    f.acodec.is_some() && !is_video(f)
}

/// Largest known size among `fmts`, or `None` when none reports one.
fn max_size<'a>(fmts: impl Iterator<Item = &'a YtdlpFormat>) -> Option<u64> {
    fmts.filter_map(size_of).max()
}

/// Reduce a `-J` document to [`SiteUrlInfo`]; pure. Size estimates mirror
/// download selectors: keep = best video at tier + best audio, analysis =
/// best video-only rendition at or under `analysis_max_height`, plus audio.
pub fn site_info(doc: &YtdlpJson, analysis_max_height: u32) -> SiteUrlInfo {
    let empty = Vec::new();
    let formats = doc.formats.as_ref().unwrap_or(&empty);
    let video: Vec<&YtdlpFormat> = formats.iter().filter(|f| is_video(f)).collect();
    let audio_size = max_size(formats.iter().filter(|f| is_audio_only(f)));

    let mut heights: Vec<u32> = video.iter().map(|f| height_of(f)).collect();
    heights.sort_unstable_by(|a, b| b.cmp(a));
    heights.dedup();
    heights.truncate(KEEP_TIER_LIMIT);

    let keep_tiers: Vec<KeepTier> = heights
        .into_iter()
        .map(|height| {
            let at_height: Vec<&YtdlpFormat> = video
                .iter()
                .copied()
                .filter(|f| height_of(f) == height)
                .collect();
            let video_size = max_size(at_height.iter().copied());
            let muxed = at_height.iter().any(|f| f.acodec.is_some());
            KeepTier {
                height,
                estimated_bytes: video_size
                    .map(|v| v + if muxed { 0 } else { audio_size.unwrap_or(0) }),
            }
        })
        .collect();

    let video_only: Vec<&YtdlpFormat> = video
        .iter()
        .copied()
        .filter(|f| height_of(f) <= analysis_max_height && f.acodec.is_none())
        .collect();
    let pool: Vec<&YtdlpFormat> = if video_only.is_empty() {
        video
            .iter()
            .copied()
            .filter(|f| height_of(f) <= analysis_max_height)
            .collect()
    } else {
        video_only
    };
    let best_height = pool.iter().map(|f| height_of(f)).max();
    let video_estimate =
        best_height.and_then(|h| max_size(pool.iter().copied().filter(|f| height_of(f) == h)));
    // Add audio unless the chosen rendition already carries it.
    let chosen_muxed = best_height
        .map(|h| pool.iter().any(|f| height_of(f) == h && f.acodec.is_some()))
        .unwrap_or(false);
    let analysis_estimated_bytes = video_estimate.map(|v| {
        v + if chosen_muxed {
            0
        } else {
            audio_size.unwrap_or(0)
        }
    });

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
        options.push(DownloadOption {
            token: Token::new(format!("dl-keep-{}", tier.height)),
            dest: Dest::Keep,
            label: match tier.estimated_bytes {
                Some(n) => format!("{}p ({})", tier.height, format_bytes(n)),
                None => format!("{}p", tier.height),
            },
            effect: format!(
                "save best audio+video at ≤{}p ({}) to {} for the user",
                tier.height,
                estimate_label(tier.estimated_bytes),
                cfg.paths.keep.display()
            ),
            estimated_bytes: tier.estimated_bytes,
            recipe: Recipe::Ytdlp(YtdlpRecipe {
                format: format!("bv*[height<={h}]+ba/b[height<={h}]/b", h = tier.height),
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

    /// Mirrors the wire boundary: `"none"` arrives as absent.
    fn fmt(height: f64, vcodec: &str, acodec: &str, size: Option<f64>) -> YtdlpFormat {
        YtdlpFormat {
            height: Some(height),
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
    fn reduces_formats_to_keep_tiers_descending() {
        let doc = YtdlpJson {
            title: Some("T".into()),
            formats: Some(vec![
                fmt(360.0, "avc1", "none", Some(10.0)),
                fmt(720.0, "avc1", "none", Some(50.0)),
                fmt(1080.0, "avc1", "none", Some(100.0)),
                fmt(1440.0, "avc1", "none", Some(200.0)),
                fmt(0.0, "none", "opus", Some(5.0)),
            ]),
            ..Default::default()
        };
        let info = site_info(&doc, 480);
        let heights: Vec<u32> = info.keep_tiers.iter().map(|t| t.height).collect();
        assert_eq!(heights, [1440, 1080, 720], "top 3 distinct, descending");
        // Video-only tiers add best audio.
        assert_eq!(info.keep_tiers[0].estimated_bytes, Some(205));
    }

    #[test]
    fn muxed_tier_does_not_add_audio() {
        let doc = YtdlpJson {
            formats: Some(vec![
                fmt(720.0, "avc1", "mp4a", Some(80.0)),
                fmt(0.0, "none", "opus", Some(5.0)),
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
                fmt(480.0, "avc1", "none", Some(30.0)),
                fmt(360.0, "avc1", "none", Some(20.0)),
                fmt(480.0, "avc1", "mp4a", Some(60.0)),
                fmt(0.0, "none", "opus", Some(5.0)),
            ]),
            ..Default::default()
        };
        let info = site_info(&doc, 480);
        assert_eq!(info.analysis_estimated_bytes, Some(35));
    }

    #[test]
    fn falls_back_to_muxed_when_no_video_only() {
        let doc = YtdlpJson {
            formats: Some(vec![fmt(480.0, "avc1", "mp4a", Some(60.0))]),
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
                height: 1080,
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
}
