import { describe, expect, test } from "bun:test";
import { defaultConfig } from "../src/store/config.ts";
import { decide, MAX_CHUNK_STRETCH_S, type CoreInvocation } from "../src/policy/plan.ts";
import type { DownloadPlan, PendingAction, PendingStore } from "../src/policy/tokens.ts";
import { TokenRegistry } from "../src/policy/tokens.ts";
import type { ProbeSummary } from "../src/tool/schema.ts";

const cfg = defaultConfig("/home/u"); // autoRun 2, max 8

function probe(duration: number, subtitles: string[] = []): ProbeSummary {
  return { duration, width: 1920, height: 1080, sizeBytes: 50 * 1024 * 1024, subtitles };
}

/** PendingStore fake that records offers and serves takes from a script. */
class FakeStore implements PendingStore {
  offers: { key: string; offers: Record<string, PendingAction> }[] = [];
  take_result: PendingAction | undefined;
  offer(key: string, offers: Readonly<Record<string, PendingAction>>): void {
    this.offers.push({ key, offers: { ...offers } });
  }
  take(): PendingAction | undefined {
    return this.take_result;
  }
}

describe("decide: fast path and planning", () => {
  test("runs_short_video_immediately_as_single_dense_grid", () => {
    const d = decide({ source: "a.mp4" }, probe(45), cfg, new FakeStore());
    if (d.kind !== "run") throw new Error(`expected run, got ${d.kind}`);
    expect(d.invocation.subcommand).toBe("extract");
    expect(d.invocation.flags).toContain("--range");
    expect(d.invocation.flags).toContain("0-45");
  });

  test("runs_two_minute_video_without_a_plan_round_trip", () => {
    const d = decide({ source: "a.mp4" }, probe(120), cfg, new FakeStore());
    if (d.kind !== "run") throw new Error(`expected run, got ${d.kind}`);
    expect(d.invocation.subcommand).toBe("process");
    expect(d.invocation.window).toBeUndefined();
    expect(d.invocation.predictedGrids).toBe(2);
  });

  test("returns_plan_for_bare_long_video", () => {
    const d = decide({ source: "a.mp4" }, probe(600, ["embedded #0 (eng)"]), cfg, new FakeStore());
    if (d.kind !== "plan") throw new Error(`expected plan, got ${d.kind}`);
    expect(d.guidance).toContain("10:00");
    expect(d.guidance).toContain("eng");
    const looks = d.recommendedCalls.map((r) => r.call.look);
    expect(looks).toContain("overview");
    expect(d.recommendedCalls.every((r) => r.call.source === "a.mp4")).toBe(true);
  });

  test("explicit_look_is_intent_and_skips_planning", () => {
    const d = decide({ source: "a.mp4", look: "overview" }, probe(420), cfg, new FakeStore());
    if (d.kind !== "run") throw new Error(`expected run, got ${d.kind}`);
    expect(d.invocation.predictedGrids).toBe(7);
    expect(d.invocation.flags).toContain("coarse");
  });
});

describe("decide: windows", () => {
  test("small_window_becomes_extract", () => {
    const d = decide({ source: "a.mp4", from: 60, until: 100 }, probe(600), cfg, new FakeStore());
    if (d.kind !== "run") throw new Error(`expected run, got ${d.kind}`);
    expect(d.invocation.subcommand).toBe("extract");
    expect(d.invocation.flags).toContain("60-100");
  });

  test("windowed_process_carries_the_clamp", () => {
    const d = decide(
      { source: "a.mp4", from: "10:00", until: "16:00" },
      probe(1200),
      cfg,
      new FakeStore(),
    );
    if (d.kind !== "run") throw new Error(`expected run, got ${d.kind}`);
    expect(d.invocation.subcommand).toBe("process");
    expect(d.invocation.window).toEqual({ start: 600, end: 960 });
  });

  test("rejects_inverted_window_with_readable_message", () => {
    const d = decide({ source: "a.mp4", from: 100, until: 50 }, probe(600), cfg, new FakeStore());
    expect(d.kind).toBe("reject");
  });
});

describe("decide: budget ladder", () => {
  test("degrades_over_budget_with_stretched_chunks_and_notice", () => {
    const d = decide({ source: "a.mp4", look: "detailed" }, probe(1200), cfg, new FakeStore());
    if (d.kind !== "run") throw new Error(`expected run, got ${d.kind}`);
    expect(d.invocation.flags).toContain("--chunk-seconds");
    expect(d.invocation.flags).toContain("150"); // 1200s / 8 grids
    expect(d.invocation.predictedGrids).toBe(8);
    expect(d.notices.some((n) => n.includes("degraded"))).toBe(true);
  });

  test("requires_consent_beyond_the_stretch_ceiling", () => {
    const store = new FakeStore();
    const duration = (MAX_CHUNK_STRETCH_S + 10) * cfg.budget.max;
    const d = decide({ source: "a.mp4", look: "detailed" }, probe(duration), cfg, store);
    if (d.kind !== "confirm") throw new Error(`expected confirm, got ${d.kind}`);
    expect(store.offers).toHaveLength(1);
    const offered = Object.keys(store.offers[0]?.offers ?? {});
    expect(d.tokens.map((t) => t.token)).toEqual(offered);
  });

  test("redeems_a_pending_token_and_runs_the_stored_plan", () => {
    const store = new FakeStore();
    const plan: CoreInvocation = { subcommand: "process", flags: [], predictedGrids: 8 };
    store.take_result = { kind: "core", invocation: plan };
    const d = decide({ source: "a.mp4", confirm: "force-stretch-8-grids" }, probe(9000), cfg, store);
    if (d.kind !== "run") throw new Error(`expected run, got ${d.kind}`);
    expect(d.invocation).toBe(plan);
    expect(d.notices[0]).toContain("confirmed");
  });

  test("redeemed_download_token_becomes_a_download_decision", () => {
    const store = new FakeStore();
    const dl: DownloadPlan = {
      url: "https://a/v",
      dest: "cache",
      method: { type: "ytdlp", format: "bv", subs: true },
    };
    store.take_result = { kind: "download", plan: dl };
    const d = decide({ source: "https://a/v", confirm: "dl-analysis" }, probe(600), cfg, store);
    if (d.kind !== "download") throw new Error(`expected download, got ${d.kind}`);
    expect(d.plan).toBe(dl);
  });

  test("rejects_stale_or_hallucinated_tokens", () => {
    const d = decide({ source: "a.mp4", confirm: "dl-keep-1080" }, probe(9000), cfg, new FakeStore());
    if (d.kind !== "reject") throw new Error(`expected reject, got ${d.kind}`);
    expect(d.message).toContain("stale");
  });

  test("rejects_confirm_combined_with_intent_params", () => {
    const d = decide(
      { source: "a.mp4", confirm: "t", look: "overview" },
      probe(600),
      cfg,
      new FakeStore(),
    );
    expect(d.kind).toBe("reject");
  });
});

describe("decide: raw escape hatch", () => {
  test("raw_extract_runs_directly", () => {
    const d = decide(
      { source: "a.mp4", raw: "extract --range 10-20 --base-fps 10" },
      probe(600),
      cfg,
      new FakeStore(),
    );
    if (d.kind !== "run") throw new Error(`expected run, got ${d.kind}`);
    expect(d.invocation.flags).toEqual(["--range", "10-20", "--base-fps", "10"]);
  });

  test("raw_process_over_budget_requires_consent", () => {
    const store = new FakeStore();
    const d = decide({ source: "a.mp4", raw: "process" }, probe(1200), cfg, store);
    if (d.kind !== "confirm") throw new Error(`expected confirm, got ${d.kind}`);
    expect(d.tokens[0]?.token).toBe("raw-full-20-grids");
  });

  test("rejects_raw_combined_with_look", () => {
    const d = decide(
      { source: "a.mp4", raw: "process", look: "overview" },
      probe(600),
      cfg,
      new FakeStore(),
    );
    expect(d.kind).toBe("reject");
  });
});

describe("TokenRegistry", () => {
  const core = (grids: number): PendingAction => ({
    kind: "core",
    invocation: { subcommand: "process", flags: [], predictedGrids: grids },
  });

  test("redeems_once_and_only_the_matching_token", () => {
    const reg = new TokenRegistry();
    const action = core(3);
    reg.offer("a.mp4", { "force-3": action });
    expect(reg.take("a.mp4", "wrong")).toBeUndefined();
    expect(reg.take("b.mp4", "force-3")).toBeUndefined();
    expect(reg.take("a.mp4", "force-3")).toBe(action);
    expect(reg.take("a.mp4", "force-3")).toBeUndefined(); // consumed
  });

  test("redeeming_one_token_consumes_its_siblings", () => {
    const reg = new TokenRegistry();
    const a = core(1);
    const b = core(2);
    reg.offer("u", { "dl-analysis": a, "dl-keep-1080": b });
    expect(reg.take("u", "dl-analysis")).toBe(a);
    expect(reg.take("u", "dl-keep-1080")).toBeUndefined();
  });

  test("new_offer_replaces_the_pending_one", () => {
    const reg = new TokenRegistry();
    const first = core(3);
    const second = core(1);
    reg.offer("a.mp4", { t1: first });
    reg.offer("a.mp4", { t2: second });
    expect(reg.take("a.mp4", "t1")).toBeUndefined();
    expect(reg.take("a.mp4", "t2")).toBe(second);
  });
});
