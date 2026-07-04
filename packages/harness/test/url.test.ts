import { describe, expect, test } from "bun:test";

import { decideUrl, directInfo, extFromContainer, type SiteUrlInfo } from "../src/policy/url.ts";
import type { PendingAction, PendingStore } from "../src/policy/tokens.ts";
import { defaultConfig } from "../src/store/config.ts";
import type { ProbeSummary } from "../src/tool/schema.ts";

const cfg = defaultConfig("/home/u"); // maxBytes 2 GiB, analysisMaxHeight 480

class FakeStore implements PendingStore {
  offers: { key: string; offers: Record<string, PendingAction> }[] = [];
  offer(key: string, offers: Readonly<Record<string, PendingAction>>): void {
    this.offers.push({ key, offers: { ...offers } });
  }
  take(): PendingAction | undefined {
    return undefined;
  }
}

function probe(sizeBytes: number): ProbeSummary {
  return { duration: 300, width: 1280, height: 720, sizeBytes, subtitles: [] };
}

function site(overrides: Partial<SiteUrlInfo> = {}): SiteUrlInfo {
  return {
    kind: "site",
    title: "A Talk",
    duration: 1800,
    subtitleLangs: ["en"],
    keepTiers: [
      { height: 1080, estimatedBytes: 900 * 1024 ** 2 },
      { height: 720, estimatedBytes: 500 * 1024 ** 2 },
      { height: 480, estimatedBytes: 250 * 1024 ** 2 },
    ],
    analysisEstimatedBytes: 80 * 1024 ** 2,
    ...overrides,
  };
}

describe("decideUrl: direct links", () => {
  test("offers_analysis_and_single_keep_tier", () => {
    const store = new FakeStore();
    const info = directInfo("https://cdn.example.com/talk.mp4", probe(700 * 1024 ** 2));
    const d = decideUrl({ source: "https://cdn.example.com/talk.mp4" }, info, cfg, store);
    if (d.kind !== "confirm") throw new Error(`expected confirm, got ${d.kind}`);
    expect(d.tokens.map((t) => t.token)).toEqual(["dl-analysis", "dl-keep-720"]);
    expect(d.consequence).toContain("700 MiB");
    expect(store.offers).toHaveLength(1);
    expect(Object.keys(store.offers[0]?.offers ?? {})).toEqual(["dl-analysis", "dl-keep-720"]);
  });

  test("refuses_oversized_file_with_manual_hint", () => {
    const store = new FakeStore();
    const info = directInfo("https://cdn.example.com/big.mkv", probe(3 * 1024 ** 3));
    const d = decideUrl({ source: "https://cdn.example.com/big.mkv" }, info, cfg, store);
    if (d.kind !== "reject") throw new Error(`expected reject, got ${d.kind}`);
    expect(d.message).toContain("manually");
    expect(store.offers).toHaveLength(0);
  });

  test("unknown_size_is_offered_with_a_label", () => {
    const info = directInfo("https://cdn.example.com/talk.mp4", probe(0));
    const d = decideUrl({ source: "https://cdn.example.com/talk.mp4" }, info, cfg, new FakeStore());
    if (d.kind !== "confirm") throw new Error(`expected confirm, got ${d.kind}`);
    expect(d.tokens[0]?.effect).toContain("size unknown");
  });
});

describe("decideUrl: site pages", () => {
  test("offers_analysis_plus_keep_tiers_with_estimates", () => {
    const store = new FakeStore();
    const d = decideUrl({ source: "https://tube.example/w" }, site(), cfg, store);
    if (d.kind !== "confirm") throw new Error(`expected confirm, got ${d.kind}`);
    expect(d.tokens.map((t) => t.token)).toEqual([
      "dl-analysis",
      "dl-keep-1080",
      "dl-keep-720",
      "dl-keep-480",
    ]);
    expect(d.consequence).toContain("A Talk");
    expect(d.consequence).toContain("analysis rip");
    expect(d.tokens[0]?.effect).toContain("480");
  });

  test("drops_tiers_over_the_cap_and_says_so", () => {
    const tight = defaultConfig("/home/u");
    tight.download.maxBytes = 600 * 1024 ** 2;
    const d = decideUrl({ source: "https://tube.example/w" }, site(), tight, new FakeStore());
    if (d.kind !== "confirm") throw new Error(`expected confirm, got ${d.kind}`);
    expect(d.tokens.map((t) => t.token)).toEqual(["dl-analysis", "dl-keep-720", "dl-keep-480"]);
    expect(d.consequence).toContain("not offered");
  });

  test("rejects_when_every_tier_breaches_the_cap", () => {
    const tiny = defaultConfig("/home/u");
    tiny.download.maxBytes = 1024;
    const d = decideUrl({ source: "https://tube.example/w" }, site(), tiny, new FakeStore());
    if (d.kind !== "reject") throw new Error(`expected reject, got ${d.kind}`);
    expect(d.message).toContain("manually");
  });

  test("notes_that_intent_params_wait_for_the_download", () => {
    const d = decideUrl(
      { source: "https://tube.example/w", look: "overview" },
      site(),
      cfg,
      new FakeStore(),
    );
    if (d.kind !== "confirm") throw new Error(`expected confirm, got ${d.kind}`);
    expect(d.consequence).toContain("download comes first");
  });

  test("rejects_raw_on_an_undownloaded_url", () => {
    const d = decideUrl(
      { source: "https://tube.example/w", raw: "process" },
      site(),
      cfg,
      new FakeStore(),
    );
    expect(d.kind).toBe("reject");
  });
});

describe("direct-link identity", () => {
  test("takes_extension_and_filename_from_the_url_path", () => {
    const info = directInfo("https://cdn.example.com/a/b/Talk%202024.mp4?sig=x", probe(1));
    expect(info.ext).toBe("mp4");
    expect(info.filename).toBe("Talk 2024.mp4");
  });

  test("falls_back_to_container_when_the_path_has_no_extension", () => {
    const p = { ...probe(1), container: "matroska,webm" };
    const info = directInfo("https://cdn.example.com/stream", p);
    expect(info.ext).toBe("mkv");
    expect(info.filename).toBe("stream.mkv");
  });

  test("degenerate_urls_get_a_generic_filename", () => {
    const info = directInfo("https://cdn.example.com/", probe(1));
    expect(info.filename).toBe("download.mp4");
  });

  test("container_extension_mapping", () => {
    expect(extFromContainer("mov,mp4,m4a,3gp,3g2,mj2")).toBe("mp4");
    expect(extFromContainer("matroska,webm")).toBe("mkv");
    expect(extFromContainer(undefined)).toBe("mp4");
    expect(extFromContainer("avi")).toBe("avi");
  });
});
