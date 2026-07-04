//! Direct-link downloads over http(s): zero external dependencies.
//! Streams to `<dest>.part`, counts bytes against the hard cap, and
//! renames on completion — a file at the final path is always whole.

import * as fs from "node:fs/promises";
import * as path from "node:path";

import { formatBytes } from "../policy/plan.ts";

/** A download failed or was refused. `message` is model-presentable. */
export class DownloadError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "DownloadError";
  }
}

/** Byte-level progress callback; `total` when the server revealed one. */
export type ProgressFn = (downloadedBytes: number, totalBytes?: number) => void;

/** Minimum interval between progress callbacks. */
const PROGRESS_INTERVAL_MS = 1000;

/** `Content-Length` via HEAD, or undefined when the server hides it. */
export async function headContentLength(url: string): Promise<number | undefined> {
  try {
    const res = await fetch(url, { method: "HEAD", redirect: "follow" });
    const len = Number(res.headers.get("content-length"));
    return res.ok && Number.isFinite(len) && len > 0 ? len : undefined;
  } catch {
    return undefined;
  }
}

/**
 * Download `url` to `dest`, refusing to exceed `maxBytes`: known
 * oversize is refused before the first byte, unknown-length streams are
 * cut off mid-flight. The `.part` file never survives a failure.
 *
 * @throws DownloadError on http errors or the size cap.
 */
export async function downloadDirect(
  url: string,
  dest: string,
  maxBytes: number,
  onProgress?: ProgressFn,
): Promise<{ bytes: number }> {
  const res = await fetch(url, { redirect: "follow" });
  if (!res.ok || res.body === null) {
    throw new DownloadError(`download failed: GET ${url} → HTTP ${res.status}`);
  }
  const lenHeader = Number(res.headers.get("content-length"));
  const total = Number.isFinite(lenHeader) && lenHeader > 0 ? lenHeader : undefined;
  if (total !== undefined && total > maxBytes) {
    await res.body.cancel();
    throw new DownloadError(
      `the file is ${formatBytes(total)}, past the ${formatBytes(maxBytes)} ceiling ` +
        "(download.max_bytes) — download it manually and pass the local path as source",
    );
  }

  await fs.mkdir(path.dirname(dest), { recursive: true });
  const part = `${dest}.part`;
  const handle = await fs.open(part, "w");
  // Explicit reader, not `for await`: Bun's fetch body lacks a working
  // async iterator (TypeError as of 1.3.14), and the error path must
  // cancel the stream so the peer stops sending.
  const reader = res.body.getReader();
  let bytes = 0;
  let lastReport = 0;
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      bytes += value.byteLength;
      if (bytes > maxBytes) {
        throw new DownloadError(
          `download aborted at ${formatBytes(bytes)}: the stream exceeded the ` +
            `${formatBytes(maxBytes)} ceiling (download.max_bytes) — download it manually`,
        );
      }
      await handle.write(value);
      const now = Date.now();
      if (now - lastReport >= PROGRESS_INTERVAL_MS) {
        lastReport = now;
        onProgress?.(bytes, total);
      }
    }
  } catch (err) {
    await reader.cancel().catch(() => {});
    await handle.close();
    await fs.rm(part, { force: true });
    throw err;
  }
  await handle.close();
  await fs.rename(part, dest);
  onProgress?.(bytes, total ?? bytes);
  return { bytes };
}
