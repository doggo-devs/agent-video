//! Local-file probing via `ffprobe` on PATH. Interim implementation for
//! R4 PR3: PR5 replaces the internals with the core `probe` subcommand,
//! keeping this module's interface — callers never change.
//!
//! Output is the model-facing `ProbeSummary` from tool/schema.ts:
//! duration, dimensions, size, container, and subtitle descriptors
//! (embedded tracks + sidecar files with the same basename).

import { spawn } from "node:child_process";
import * as fs from "node:fs/promises";
import * as path from "node:path";
import type { ProbeSummary } from "../tool/schema.ts";

/** Probing failed. `message` is model-presentable. */
export class ProbeError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "ProbeError";
  }
}

/** Sidecar subtitle extensions scanned next to the video file. */
export const SIDECAR_EXTS = [".srt", ".ass", ".vtt"] as const;

// ── ffprobe document (the fields we read; everything else ignored) ──

interface FfprobeStream {
  codec_type?: string;
  codec_name?: string;
  width?: number;
  height?: number;
  duration?: string;
  tags?: { language?: string };
}

interface FfprobeDoc {
  format?: { duration?: string; size?: string; format_name?: string };
  streams?: FfprobeStream[];
}

/**
 * Reduce a raw ffprobe document to a ProbeSummary. Pure; exported for
 * direct testing. `sidecars` are basenames of subtitle files found next
 * to the video (see `scanSidecars`).
 *
 * @throws ProbeError when the document has no video stream or no
 * usable duration — such a file cannot be gridded anyway.
 */
export function summarize(
  doc: FfprobeDoc,
  sizeFallback: number,
  sidecars: readonly string[],
): ProbeSummary {
  const streams = doc.streams ?? [];
  const video = streams.find(
    (s) => s.codec_type === "video" && (s.width ?? 0) > 0 && (s.height ?? 0) > 0,
  );
  if (video === undefined) {
    throw new ProbeError("no video stream found — is this actually a video file?");
  }

  const duration = Number(doc.format?.duration ?? video.duration);
  if (!Number.isFinite(duration) || duration <= 0) {
    throw new ProbeError("could not determine video duration from ffprobe output");
  }

  const size = Number(doc.format?.size);

  const subtitles: string[] = [];
  let track = 0;
  for (const s of streams) {
    if (s.codec_type !== "subtitle") continue;
    const lang = s.tags?.language;
    const codec = s.codec_name ?? "unknown";
    subtitles.push(`embedded #${track} (${lang !== undefined ? `${lang}, ` : ""}${codec})`);
    track += 1;
  }
  for (const name of sidecars) subtitles.push(`sidecar ${name}`);

  return {
    duration,
    width: video.width ?? 0,
    height: video.height ?? 0,
    sizeBytes: Number.isFinite(size) && size > 0 ? size : sizeFallback,
    container: doc.format?.format_name,
    subtitles,
  };
}

/** Basenames of sidecar subtitle files sharing the video's basename. */
export async function scanSidecars(videoPath: string): Promise<string[]> {
  const dir = path.dirname(videoPath);
  const stem = path.basename(videoPath, path.extname(videoPath));
  const found: string[] = [];
  for (const ext of SIDECAR_EXTS) {
    const name = stem + ext;
    try {
      await fs.access(path.join(dir, name));
      found.push(name);
    } catch {
      // absent — not an error
    }
  }
  return found;
}

function runFfprobe(input: string, timeoutMs?: number): Promise<FfprobeDoc> {
  return new Promise((resolve, reject) => {
    const child = spawn(
      "ffprobe",
      ["-v", "error", "-print_format", "json", "-show_format", "-show_streams", input],
      { stdio: ["ignore", "pipe", "pipe"] },
    );
    let out = "";
    let err = "";
    let timedOut = false;
    const timer =
      timeoutMs === undefined
        ? undefined
        : setTimeout(() => {
            timedOut = true;
            child.kill("SIGKILL");
          }, timeoutMs);
    child.stdout.on("data", (chunk: Buffer) => (out += chunk));
    child.stderr.on("data", (chunk: Buffer) => (err += chunk));
    child.once("error", (e: NodeJS.ErrnoException) => {
      clearTimeout(timer);
      reject(
        e.code === "ENOENT"
          ? new ProbeError("ffprobe not found on PATH — install ffmpeg to analyze local files")
          : e,
      );
    });
    child.once("close", (code) => {
      clearTimeout(timer);
      if (timedOut) {
        reject(new ProbeError(`ffprobe timed out after ${timeoutMs}ms probing ${input}`));
        return;
      }
      if (code !== 0) {
        reject(new ProbeError(`ffprobe failed (exit ${code}): ${err.trim() || "no error output"}`));
        return;
      }
      try {
        resolve(JSON.parse(out) as FfprobeDoc);
      } catch {
        reject(new ProbeError("ffprobe emitted invalid JSON"));
      }
    });
  });
}

/**
 * Probe a local video file. Sidecar subtitles are scanned in the same
 * directory (same basename, .srt/.ass/.vtt).
 *
 * @throws ProbeError on missing ffprobe, ffprobe failure, or a file
 * with no video stream.
 */
export async function probeLocalFile(file: string): Promise<ProbeSummary> {
  const [doc, sidecars, stat] = await Promise.all([
    runFfprobe(file),
    scanSidecars(file),
    fs.stat(file),
  ]);
  return summarize(doc, stat.size, sidecars);
}

/** How long an over-http probe may hang before we call it a non-file. */
export const URL_PROBE_TIMEOUT_MS = 20_000;

/**
 * Probe a direct media URL over http(s): ffprobe reads the container
 * header (moov etc.) without downloading the payload. `sizeBytes` is 0
 * when the server reveals no length — the caller may fill it via HEAD.
 *
 * @throws ProbeError when the URL is not direct media (HTML pages land
 * here), ffprobe is missing, or the probe times out.
 */
export async function probeMediaUrl(url: string): Promise<ProbeSummary> {
  const doc = await runFfprobe(url, URL_PROBE_TIMEOUT_MS);
  return summarize(doc, 0, []);
}
