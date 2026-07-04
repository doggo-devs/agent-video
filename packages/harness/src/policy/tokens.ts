//! Consent token registry. A token is only honored against the plan
//! that issued it, for the source it was issued for — echoing a token
//! is informed consent to a specific consequence, so hallucinated or
//! stale tokens must never authorize anything.
//!
//! One offer may carry several tokens (e.g. `dl-analysis` next to
//! `dl-keep-1080`): they are alternatives over the same consequence
//! set, so redeeming any of them consumes the whole offer.
//!
//! In-memory by design: the registry lives exactly as long as the MCP
//! stdio session that issued the tokens.

import type { CoreInvocation } from "./plan.ts";

/** A decided download, self-contained: executing it needs no re-probe. */
export interface DownloadPlan {
  url: string;
  /** cache = evictable analysis rip; keep = user-owned file in keep_dir. */
  dest: "cache" | "keep";
  method:
    | {
        type: "direct";
        /** File extension for the cache path. */
        ext: string;
        /** Human filename used when `dest` is keep. */
        filename: string;
      }
    | {
        type: "ytdlp";
        /** yt-dlp `-f` format selector, decided at offer time. */
        format: string;
        /** Fetch subtitle sidecars alongside (analysis rips only). */
        subs: boolean;
      };
  /** Best-effort size estimate; undefined = unknown until download. */
  estimatedBytes?: number;
}

/** What a redeemed token authorizes. */
export type PendingAction =
  | { kind: "core"; invocation: CoreInvocation }
  | { kind: "download"; plan: DownloadPlan };

/** Storage surface the policy engine needs; hand-fake in tests. */
export interface PendingStore {
  /** Register an offer for `key`, replacing any previous offer wholesale. */
  offer(key: string, offers: Readonly<Record<string, PendingAction>>): void;
  /** Redeem: returns the action and consumes the entire offer, or undefined. */
  take(key: string, token: string): PendingAction | undefined;
}

export class TokenRegistry implements PendingStore {
  private pending = new Map<string, Record<string, PendingAction>>();

  offer(key: string, offers: Readonly<Record<string, PendingAction>>): void {
    this.pending.set(key, { ...offers });
  }

  take(key: string, token: string): PendingAction | undefined {
    const action = this.pending.get(key)?.[token];
    if (action === undefined) return undefined;
    this.pending.delete(key);
    return action;
  }
}
