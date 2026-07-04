//! Single source of truth for the `video` tool: input schema, argument
//! types, and the result family. MCP tool listings (R4) and plugin
//! manifests (R6) are generated from here, never hand-maintained.
//!
//! Design (docs/R4-IMPL.md): six params expressing real semantics only —
//! which video / where in it / how to look / raw kernel escape / consent
//! echo. Everything else is policy, decided server-side from probe data.

/** Intent-level presets; mapped to core knobs in policy/look.ts. */
export type Look = "overview" | "detailed" | "read_text" | "track_motion";

export const LOOKS: readonly Look[] = ["overview", "detailed", "read_text", "track_motion"];

/** Arguments of the `video` tool. */
export interface VideoArgs {
  /** Local file path or URL. */
  source: string;
  /** Window start: seconds or "MM:SS" / "H:MM:SS". Omitted = video start. */
  from?: string | number;
  /** Window end, same formats. Omitted = video end. */
  until?: string | number;
  /** How to look at the window. Omitted = balanced default. */
  look?: Look;
  /** Expert escape hatch: verbatim core CLI args, e.g. "process --strategy keyframes". */
  raw?: string;
  /** Echo of a server-issued consent token from a previous response. */
  confirm?: string;
}

/** JSON Schema for MCP `tools/list`. Kept in lockstep with VideoArgs. */
export const VIDEO_INPUT_SCHEMA = {
  type: "object",
  properties: {
    source: {
      type: "string",
      description: "Local video file path or URL. Call with only `source` first; the response tells you what to do next.",
    },
    from: {
      anyOf: [{ type: "string" }, { type: "number" }],
      description: 'Window start: seconds (83) or timestamp ("1:23", "1:02:03"). Omit to start from the beginning.',
    },
    until: {
      anyOf: [{ type: "string" }, { type: "number" }],
      description: "Window end, same formats. Omit to go to the end of the video.",
    },
    look: {
      type: "string",
      enum: LOOKS as Look[],
      description:
        "How to look: overview (fast triage), detailed (miss nothing), read_text (few large frames to read on-screen text/UI), track_motion (dense small frames for fast action). Subtitle text always rides along in metadata regardless; this only affects frame size/density. Omit for a balanced default.",
    },
    raw: {
      type: "string",
      description:
        'Expert escape hatch: verbatim agent-video core CLI arguments starting with the subcommand, e.g. "process --strategy keyframes --base-fps 0.2". Mutually exclusive with from/until/look. Input/output plumbing is injected by the server.',
    },
    confirm: {
      type: "string",
      description: "Echo a consent token issued by a previous response to authorize the pending action.",
    },
  },
  required: ["source"],
  additionalProperties: false,
} as const;

/** One-line tool description shown in tool listings. */
export const VIDEO_TOOL_DESCRIPTION =
  "Watch a video (local path or URL) as VLM-readable frame grids with timestamps and subtitles. Call with just `source` first and follow the returned guidance.";

// ── result family ────────────────────────────────────────────────────

import type { Metadata } from "../core/contract.ts";

/** Probe summary shown to the model in `Plan` responses. */
export interface ProbeSummary {
  /** Seconds. */
  duration: number;
  width: number;
  height: number;
  sizeBytes: number;
  container?: string;
  /** Human descriptors, e.g. "embedded #0 (eng, subrip)", "sidecar movie.srt". */
  subtitles: string[];
}

/** A literal argument object the model can copy verbatim, plus why. */
export interface RecommendedCall {
  call: Partial<VideoArgs>;
  note: string;
}

/** A consent token offered to the model, with its consequence spelled out. */
export interface OfferedToken {
  token: string;
  effect: string;
}

/** Grids delivered. `notices` carries degradation and confirmation notes. */
export interface Ready {
  status: "ready";
  metadata: Metadata;
  notices: string[];
}

/** Probe-only response: summary plus copy-pasteable next calls. */
export interface Plan {
  status: "plan";
  probe: ProbeSummary;
  guidance: string;
  recommended_calls: RecommendedCall[];
}

/** The request needs explicit consent; retry with `confirm`. */
export interface ConfirmRequired {
  status: "confirm_required";
  consequence: string;
  tokens: OfferedToken[];
}

/** Invalid input or environment failure. Message is model-presentable. */
export interface Failed {
  status: "failed";
  error: string;
}

export type VideoResult = Ready | Plan | ConfirmRequired | Failed;
