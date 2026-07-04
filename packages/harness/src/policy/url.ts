//! URL-source decisions: which download tiers to offer, with what
//! consequence text, and which requests to refuse outright. Pure —
//! probing and downloading live in probe/ and source/; this module
//! only turns probe facts into a consent offer.
//!
//! Contract (docs/R4-IMPL.md "URL stack"): nothing is downloaded
//! without an echoed token. `dl-analysis` feeds the grid pipeline from
//! the evictable cache; `dl-keep-<res>` saves a user-owned file. Tiers
//! whose estimated size breaches `download.max_bytes` are not offered.

import * as path from "node:path";

import type { Config } from "../store/config.ts";
import type { OfferedToken, ProbeSummary, VideoArgs } from "../tool/schema.ts";
import { formatBytes, type Decision } from "./plan.ts";
import { formatTimestamp } from "./time.ts";
import type { DownloadPlan, PendingAction, PendingStore } from "./tokens.ts";

/** A direct media link: ffprobe-over-http succeeded, one file, one tier. */
export interface DirectUrlInfo {
  kind: "direct";
  /** `sizeBytes` 0 means the server did not reveal a length. */
  probe: ProbeSummary;
  /** Extension for cache paths, from the URL or the probed container. */
  ext: string;
  /** Human filename used for keep downloads. */
  filename: string;
}

/** One keep tier derived from a site's format list. */
export interface KeepTier {
  height: number;
  estimatedBytes?: number;
}

/** A site page probed via yt-dlp -J. */
export interface SiteUrlInfo {
  kind: "site";
  title: string;
  /** Seconds; live streams may not report one. */
  duration?: number;
  subtitleLangs: string[];
  /** Distinct available heights, descending, already capped in count. */
  keepTiers: KeepTier[];
  /** Estimate for the analysis rip (video-only, height-capped). */
  analysisEstimatedBytes?: number;
}

export type UrlInfo = DirectUrlInfo | SiteUrlInfo;

/** Model-facing guidance attached to every download offer. */
const CHOICE_GUIDANCE =
  "dl-analysis fetches a low-res rip plus subtitles into an evictable cache — " +
  "enough for you to watch and report on the video. If the user wants the video " +
  "for themselves, pick a dl-keep tier instead: never serve them the analysis rip. " +
  "Echo exactly one token via confirm.";

/**
 * Decide the first contact with a not-yet-downloaded URL: offer download
 * tiers behind consent tokens, or refuse when nothing fits the cap.
 * `confirm` is redeemed by the server before this is ever called.
 */
export function decideUrl(
  args: VideoArgs,
  info: UrlInfo,
  cfg: Config,
  store: PendingStore,
): Decision {
  if (args.raw !== undefined) {
    return {
      kind: "reject",
      message:
        "raw needs a local file. Call with just the URL first, confirm a download, then repeat the raw call.",
    };
  }

  const { offers, tokens, dropped } =
    info.kind === "direct" ? directOffers(args.source, info, cfg) : siteOffers(args.source, info, cfg);

  if (tokens.length === 0) {
    return {
      kind: "reject",
      message:
        `${headline(info)} Every available tier exceeds the ${formatBytes(cfg.download.maxBytes)} ` +
        "download ceiling (download.max_bytes). Download the file manually and pass the local path as source.",
    };
  }

  store.offer(args.source, offers);

  const notes: string[] = [headline(info)];
  if (dropped.length > 0) {
    notes.push(
      `Tiers over the ${formatBytes(cfg.download.maxBytes)} ceiling were not offered: ${dropped.join(", ")}.`,
    );
  }
  if (args.from !== undefined || args.until !== undefined || args.look !== undefined) {
    notes.push(
      "Your from/until/look cannot apply yet — the download comes first. Repeat them in the call after it completes.",
    );
  }
  notes.push(CHOICE_GUIDANCE);

  return { kind: "confirm", consequence: notes.join(" "), tokens };
}

function headline(info: UrlInfo): string {
  if (info.kind === "direct") {
    const size = info.probe.sizeBytes > 0 ? formatBytes(info.probe.sizeBytes) : "size unknown";
    return (
      `Direct media link: ${info.probe.width}x${info.probe.height}, ` +
      `${formatTimestamp(info.probe.duration)}, ${size}. Nothing is downloaded yet.`
    );
  }
  const duration = info.duration !== undefined ? formatTimestamp(info.duration) : "unknown duration";
  const subs =
    info.subtitleLangs.length > 0 ? `subtitles: ${info.subtitleLangs.join(", ")}` : "no subtitles listed";
  return `"${info.title}" — ${duration}, ${subs}. Nothing is downloaded yet.`;
}

interface OfferSet {
  offers: Record<string, PendingAction>;
  tokens: OfferedToken[];
  /** Human labels of tiers withheld for breaching the size cap. */
  dropped: string[];
}

function estimateLabel(bytes: number | undefined): string {
  return bytes !== undefined ? `~${formatBytes(bytes)}` : "size unknown";
}

function directOffers(url: string, info: DirectUrlInfo, cfg: Config): OfferSet {
  const size = info.probe.sizeBytes > 0 ? info.probe.sizeBytes : undefined;
  const set: OfferSet = { offers: {}, tokens: [], dropped: [] };
  if (size !== undefined && size > cfg.download.maxBytes) {
    set.dropped.push(`the file (${formatBytes(size)})`);
    return set;
  }

  const method = { type: "direct", ext: info.ext, filename: info.filename } as const;
  add(set, "dl-analysis", {
    url,
    dest: "cache",
    method,
    estimatedBytes: size,
  }, `download the file (${estimateLabel(size)}) into the analysis cache (evictable) and analyze from there`);
  add(set, `dl-keep-${info.probe.height}`, {
    url,
    dest: "keep",
    method,
    estimatedBytes: size,
  }, `save the file (${estimateLabel(size)}) to ${path.join(cfg.download.keepDir, info.filename)} for the user`);
  return set;
}

function siteOffers(url: string, info: SiteUrlInfo, cfg: Config): OfferSet {
  const set: OfferSet = { offers: {}, tokens: [], dropped: [] };
  const cap = cfg.download.maxBytes;
  const h = cfg.download.analysisMaxHeight;

  if (info.analysisEstimatedBytes !== undefined && info.analysisEstimatedBytes > cap) {
    set.dropped.push(`analysis rip (${formatBytes(info.analysisEstimatedBytes)})`);
  } else {
    add(set, "dl-analysis", {
      url,
      dest: "cache",
      method: { type: "ytdlp", format: `bv[height<=${h}]/bv*[height<=${h}]/b`, subs: true },
      estimatedBytes: info.analysisEstimatedBytes,
    }, `download a video-only rip (≤${h}p, ${estimateLabel(info.analysisEstimatedBytes)}) plus subtitles into the analysis cache (evictable)`);
  }

  for (const tier of info.keepTiers) {
    if (tier.estimatedBytes !== undefined && tier.estimatedBytes > cap) {
      set.dropped.push(`${tier.height}p (${formatBytes(tier.estimatedBytes)})`);
      continue;
    }
    add(set, `dl-keep-${tier.height}`, {
      url,
      dest: "keep",
      method: {
        type: "ytdlp",
        format: `bv*[height<=${tier.height}]+ba/b[height<=${tier.height}]/b`,
        subs: false,
      },
      estimatedBytes: tier.estimatedBytes,
    }, `save best audio+video at ≤${tier.height}p (${estimateLabel(tier.estimatedBytes)}) to ${cfg.download.keepDir} for the user`);
  }
  return set;
}

function add(set: OfferSet, token: string, plan: DownloadPlan, effect: string): void {
  set.offers[token] = { kind: "download", plan };
  set.tokens.push({ token, effect });
}

// ── direct-link identity helpers (pure, shared with the server) ─────

/** Container-name → file extension for URLs whose path carries none. */
const CONTAINER_EXT: Record<string, string> = {
  matroska: "mkv",
  webm: "webm",
  mov: "mp4",
  mp4: "mp4",
  mpegts: "ts",
  avi: "avi",
  flv: "flv",
  asf: "wmv",
  ogg: "ogg",
};

/** Best-effort extension from an ffprobe `format_name` (comma-list). */
export function extFromContainer(container: string | undefined): string {
  const first = container?.split(",")[0]?.trim() ?? "";
  return CONTAINER_EXT[first] ?? (/^[a-z0-9]{2,5}$/.test(first) ? first : "mp4");
}

/**
 * Build the identity of a direct media link: extension and a filename
 * fit for the keep directory, derived from the URL path with the probed
 * container as fallback.
 */
export function directInfo(url: string, probe: ProbeSummary): DirectUrlInfo {
  let base = "";
  try {
    base = decodeURIComponent(path.posix.basename(new URL(url).pathname));
  } catch {
    // unparseable URL — fall through to defaults
  }
  base = base.replace(/[/\\<>:"|?*\u0000-\u001f]/g, "").trim();
  const urlExt = path.posix.extname(base).slice(1).toLowerCase();
  const ext = /^[a-z0-9]{2,5}$/.test(urlExt) ? urlExt : extFromContainer(probe.container);
  const filename = base === "" || base === `.${urlExt}` ? `download.${ext}` : urlExt === "" ? `${base}.${ext}` : base;
  return { kind: "direct", probe, ext, filename };
}
