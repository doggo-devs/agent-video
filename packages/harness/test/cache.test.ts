import { describe, expect, test } from "bun:test";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { Cache, PROBE_MAX_AGE_MS, urlKey } from "../src/store/cache.ts";

function tmpCache(maxBytes: number): { cache: Cache; dir: string } {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "av-cache-"));
  return { cache: new Cache({ dir, maxBytes }), dir };
}

describe("urlKey", () => {
  test("is_stable_and_url_sensitive", () => {
    expect(urlKey("https://a/v1")).toBe(urlKey("https://a/v1"));
    expect(urlKey("https://a/v1")).not.toBe(urlKey("https://a/v2"));
    expect(urlKey("x")).toHaveLength(32);
  });
});

describe("Cache probe documents", () => {
  test("roundtrips_probe_documents", async () => {
    const { cache } = tmpCache(1024);
    await cache.putProbe("abc", { duration: 12.5 });
    expect(await cache.getProbe("abc")).toEqual({ duration: 12.5 });
  });

  test("misses_return_undefined", async () => {
    const { cache } = tmpCache(1024);
    expect(await cache.getProbe("nope")).toBeUndefined();
  });

  test("drops_probe_docs_older_than_max_age", async () => {
    const { cache, dir } = tmpCache(1024);
    await cache.putProbe("old", {});
    const removed = await cache.evict(Date.now() + PROBE_MAX_AGE_MS + 1000);
    expect(removed).toContain(path.join(dir, "probe", "old.json"));
    expect(await cache.getProbe("old")).toBeUndefined();
  });
});

describe("Cache downloads", () => {
  test("reports_finished_download_with_size", async () => {
    const { cache } = tmpCache(1024);
    const p = cache.downloadPath(urlKey("https://a/v1"), "mp4");
    await cache.ensureDownloadsDir();
    fs.writeFileSync(p, Buffer.alloc(100));
    expect(await cache.getDownload(p)).toEqual({ size: 100 });
    expect(await cache.getDownload(cache.downloadPath("missing", "mp4"))).toBeUndefined();
  });

  test("finds_the_media_download_regardless_of_extension", async () => {
    const { cache } = tmpCache(1024);
    await cache.ensureDownloadsDir();
    const key = urlKey("https://a/v1");
    fs.writeFileSync(cache.downloadPath(key, "webm"), Buffer.alloc(10));
    fs.writeFileSync(cache.downloadPath(key, "srt"), "subs"); // sidecar, not media
    fs.writeFileSync(`${cache.downloadPath(key, "mkv")}.part`, Buffer.alloc(1)); // unfinished
    const found = await cache.findDownload(key);
    expect(found).toBe(cache.downloadPath(key, "webm"));
  });

  test("find_returns_undefined_for_unknown_keys_and_missing_dir", async () => {
    const { cache } = tmpCache(1024);
    expect(await cache.findDownload("nope")).toBeUndefined(); // dir absent
    await cache.ensureDownloadsDir();
    expect(await cache.findDownload("nope")).toBeUndefined();
  });

  test("evicts_oldest_downloads_until_under_cap", async () => {
    const { cache } = tmpCache(250);
    await cache.ensureDownloadsDir();
    const old = cache.downloadPath("old", "mp4");
    const mid = cache.downloadPath("mid", "mp4");
    const fresh = cache.downloadPath("new", "mp4");
    for (const [i, p] of [old, mid, fresh].entries()) {
      fs.writeFileSync(p, Buffer.alloc(100));
      const t = new Date(Date.now() - (3 - i) * 60_000);
      fs.utimesSync(p, t, t);
    }
    const removed = await cache.evict();
    expect(removed).toEqual([old]); // 300 -> 200 <= 250, oldest first
    expect(fs.existsSync(old)).toBe(false);
    expect(fs.existsSync(mid)).toBe(true);
    expect(fs.existsSync(fresh)).toBe(true);
  });
});
