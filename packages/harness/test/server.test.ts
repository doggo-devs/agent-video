import { afterAll, beforeAll, describe, expect, test } from "bun:test";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";

import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { InMemoryTransport } from "@modelcontextprotocol/sdk/inMemory.js";
import type { CallToolResult } from "@modelcontextprotocol/sdk/types.js";

import { captureCore } from "../src/core/binary.ts";
import type { Metadata } from "../src/core/contract.ts";
import { TokenRegistry } from "../src/policy/tokens.ts";
import type { SiteUrlInfo } from "../src/policy/url.ts";
import { probeLocalFile, ProbeError } from "../src/probe/ffprobe.ts";
import { YtdlpError } from "../src/source/ytdlp.ts";
import { Cache, urlKey } from "../src/store/cache.ts";
import { defaultConfig } from "../src/store/config.ts";
import { createServer, handleVideoCall, type ServerDeps } from "../src/mcp/server.ts";
import type { ProbeSummary, VideoResult } from "../src/tool/schema.ts";

// ── fixtures ─────────────────────────────────────────────────────────

let tmp: string;
let videoPath: string;
let argsFile: string;
const savedEnv: Record<string, string | undefined> = {};

const CANNED_METADATA: Metadata = {
  source: "movie.mp4",
  duration: 45,
  settings: { max_edge: 2048, quality: "low", detail: "standard" },
  grids: [
    {
      base64: "aGVsbG8=",
      time_range: "00:00-00:45",
      total_frames: 12,
      key_frames: [{ timestamp: "00:12", score: 0.9, grid_position: "r1c2" }],
      layout: "3x4",
      subtitles: "00:01 hello world",
    },
  ],
};

const FFPROBE_DOC = {
  format: { duration: "45.0", size: "4096", format_name: "mov,mp4,m4a" },
  streams: [
    { codec_type: "video", codec_name: "h264", width: 1920, height: 1080 },
    { codec_type: "subtitle", codec_name: "subrip", tags: { language: "eng" } },
  ],
};

beforeAll(() => {
  tmp = fs.mkdtempSync(path.join(os.tmpdir(), "av-server-"));
  videoPath = path.join(tmp, "movie.mp4");
  fs.writeFileSync(videoPath, "not really a video, but bytes are bytes");

  // Fake ffprobe on a PATH prefix: emits the canned document.
  const binDir = path.join(tmp, "bin");
  fs.mkdirSync(binDir);
  const docFile = path.join(tmp, "ffprobe-doc.json");
  fs.writeFileSync(docFile, JSON.stringify(FFPROBE_DOC));
  const ffprobe = path.join(binDir, "ffprobe");
  fs.writeFileSync(ffprobe, `#!/bin/sh\ncat "${docFile}"\n`);
  fs.chmodSync(ffprobe, 0o755);

  // Fake core via AGENT_VIDEO_BIN: records argv, emits canned Metadata.
  argsFile = path.join(tmp, "core-args.txt");
  const metaFile = path.join(tmp, "core-meta.json");
  fs.writeFileSync(metaFile, JSON.stringify(CANNED_METADATA));
  const core = path.join(tmp, "fake-core");
  fs.writeFileSync(core, `#!/bin/sh\nprintf '%s\\n' "$@" > "${argsFile}"\ncat "${metaFile}"\n`);
  fs.chmodSync(core, 0o755);

  savedEnv["PATH"] = process.env["PATH"];
  savedEnv["AGENT_VIDEO_BIN"] = process.env["AGENT_VIDEO_BIN"];
  process.env["PATH"] = `${binDir}:${process.env["PATH"]}`;
  process.env["AGENT_VIDEO_BIN"] = core;
});

afterAll(() => {
  process.env["PATH"] = savedEnv["PATH"];
  if (savedEnv["AGENT_VIDEO_BIN"] === undefined) delete process.env["AGENT_VIDEO_BIN"];
  else process.env["AGENT_VIDEO_BIN"] = savedEnv["AGENT_VIDEO_BIN"];
});

function probeOf(duration: number): ProbeSummary {
  return { duration, width: 1920, height: 1080, sizeBytes: 4096, subtitles: [] };
}

/** Deps with throwing side effects and a fixed probe; override what the test owns. */
function fakeDeps(overrides: Partial<ServerDeps> = {}): ServerDeps {
  const config = defaultConfig("/home/u");
  config.cache.dir = fs.mkdtempSync(path.join(os.tmpdir(), "av-server-cache-"));
  config.download.keepDir = path.join(config.cache.dir, "keep");
  return {
    config,
    cache: new Cache(config.cache),
    tokens: new TokenRegistry(),
    probeFile: async () => probeOf(600),
    probeMediaUrl: async () => {
      throw new ProbeError("not a direct media link");
    },
    probeSite: async () => {
      throw new YtdlpError("yt-dlp not found on PATH — install it", true);
    },
    downloadDirect: async () => {
      throw new Error("no direct download expected in this test");
    },
    downloadSite: async () => {
      throw new Error("no site download expected in this test");
    },
    runCore: async () => {
      throw new Error("no core run expected in this test");
    },
    ...overrides,
  };
}

function bodyOf(result: CallToolResult): VideoResult {
  const first = result.content[0];
  if (first?.type !== "text") throw new Error("expected a text block first");
  return JSON.parse(first.text) as VideoResult;
}

// ── MCP wiring (real client ↔ server over linked transports) ─────────

async function connectedClient(deps: ServerDeps): Promise<Client> {
  const [clientTransport, serverTransport] = InMemoryTransport.createLinkedPair();
  const client = new Client({ name: "test", version: "0.0.0" });
  await Promise.all([
    createServer(deps).connect(serverTransport),
    client.connect(clientTransport),
  ]);
  return client;
}

describe("MCP wiring", () => {
  test("lists_the_video_tool_with_its_schema", async () => {
    const client = await connectedClient(fakeDeps());
    const { tools } = await client.listTools();
    expect(tools.map((t) => t.name)).toEqual(["video"]);
    expect(tools[0]?.inputSchema["required"]).toEqual(["source"]);
    await client.close();
  });

  test("delivers_grid_images_through_fake_ffprobe_and_fake_core", async () => {
    // Full stack: real probeLocalFile hits the PATH-prefix ffprobe, real
    // captureCore hits AGENT_VIDEO_BIN. 45s video → single-grid extract.
    const deps = fakeDeps({ probeFile: probeLocalFile, runCore: captureCore });
    const client = await connectedClient(deps);
    const result = (await client.callTool({
      name: "video",
      arguments: { source: videoPath },
    })) as CallToolResult;

    expect(result.isError).toBeUndefined();
    const kinds = result.content.map((c) => c.type);
    expect(kinds).toEqual(["text", "image"]);

    const text = result.content[0] as { type: "text"; text: string };
    expect(text.text).toContain("00:00-00:45");
    expect(text.text).toContain("00:12 @r1c2");
    expect(text.text).toContain("hello world");

    const image = result.content[1] as { type: "image"; data: string; mimeType: string };
    expect(image.data).toBe("aGVsbG8=");
    expect(image.mimeType).toBe("image/jpeg");

    const argv = fs.readFileSync(argsFile, "utf8").trim().split("\n");
    expect(argv[0]).toBe("extract");
    expect(argv[1]).toBe(videoPath);
    expect(argv).toContain("--json");
    expect(argv).toContain("--range");
    expect(argv).toContain("0-45");
    await client.close();
  });
});

// ── handler behavior (fakes injected directly) ───────────────────────

describe("handleVideoCall", () => {
  test("plans_on_bare_source_for_a_long_video", async () => {
    const result = await handleVideoCall({ source: videoPath }, fakeDeps());
    const body = bodyOf(result);
    if (body.status !== "plan") throw new Error(`expected plan, got ${body.status}`);
    expect(body.probe.duration).toBe(600);
    expect(body.recommended_calls.length).toBeGreaterThan(0);
  });

  test("rejects_multi_grid_windows_until_core_clamp_lands", async () => {
    const result = await handleVideoCall(
      { source: videoPath, from: 60, until: 360, look: "overview" },
      fakeDeps(),
    );
    expect(result.isError).toBe(true);
    const body = bodyOf(result);
    if (body.status !== "failed") throw new Error(`expected failed, got ${body.status}`);
    expect(body.error).toContain("not supported yet");
  });

  test("serves_cached_probe_on_second_call_for_same_content", async () => {
    let probes = 0;
    const deps = fakeDeps({
      probeFile: async () => {
        probes += 1;
        return probeOf(600);
      },
    });
    await handleVideoCall({ source: videoPath }, deps);
    await handleVideoCall({ source: videoPath }, deps);
    expect(probes).toBe(1);
  });

  test("fails_on_missing_file", async () => {
    const result = await handleVideoCall({ source: "/no/such/file.mp4" }, fakeDeps());
    expect(result.isError).toBe(true);
    const body = bodyOf(result);
    if (body.status !== "failed") throw new Error(`expected failed, got ${body.status}`);
    expect(body.error).toContain("file not found");
  });

  test("site_url_without_ytdlp_fails_with_install_hint", async () => {
    const result = await handleVideoCall(
      { source: "https://example.com/watch?v=1" },
      fakeDeps(),
    );
    expect(result.isError).toBe(true);
    const body = bodyOf(result);
    if (body.status !== "failed") throw new Error(`expected failed, got ${body.status}`);
    expect(body.error).toContain("yt-dlp");
  });

  test("rejects_unknown_arguments_by_name", async () => {
    const result = await handleVideoCall(
      { source: videoPath, zoom: true },
      fakeDeps(),
    );
    expect(result.isError).toBe(true);
    const body = bodyOf(result);
    if (body.status !== "failed") throw new Error(`expected failed, got ${body.status}`);
    expect(body.error).toContain('"zoom"');
  });

  test("rejects_invalid_look_with_the_valid_set", async () => {
    const result = await handleVideoCall(
      { source: videoPath, look: "cinematic" },
      fakeDeps(),
    );
    expect(result.isError).toBe(true);
    const body = bodyOf(result);
    if (body.status !== "failed") throw new Error(`expected failed, got ${body.status}`);
    expect(body.error).toContain("overview");
  });

  test("surfaces_core_failure_with_stderr_tail", async () => {
    const deps = fakeDeps({
      probeFile: async () => probeOf(45),
      runCore: async () => ({ code: 3, stdout: "", stderr: "boom: ffmpeg exploded\n" }),
    });
    const result = await handleVideoCall({ source: videoPath }, deps);
    expect(result.isError).toBe(true);
    const body = bodyOf(result);
    if (body.status !== "failed") throw new Error(`expected failed, got ${body.status}`);
    expect(body.error).toContain("ffmpeg exploded");
  });
});

// ── URL stack ─────────────────────────────────────────────────────────

const SITE_URL = "https://tube.example/watch?v=42";

function siteInfoFixture(): SiteUrlInfo {
  return {
    kind: "site",
    title: "A Talk",
    duration: 1800,
    subtitleLangs: ["en"],
    keepTiers: [{ height: 720, estimatedBytes: 500 * 1024 ** 2 }],
    analysisEstimatedBytes: 80 * 1024 ** 2,
  };
}

describe("handleVideoCall: URL stack", () => {
  test("first_contact_with_site_url_offers_download_tokens", async () => {
    const deps = fakeDeps({ probeSite: async () => siteInfoFixture() });
    const result = await handleVideoCall({ source: SITE_URL }, deps);
    const body = bodyOf(result);
    if (body.status !== "confirm_required") throw new Error(`expected confirm, got ${body.status}`);
    expect(body.tokens.map((t) => t.token)).toEqual(["dl-analysis", "dl-keep-720"]);
    expect(body.consequence).toContain("A Talk");
  });

  test("first_contact_with_direct_link_offers_download_tokens", async () => {
    const deps = fakeDeps({
      probeMediaUrl: async () => ({ ...probeOf(300), sizeBytes: 700 * 1024 ** 2 }),
    });
    const result = await handleVideoCall({ source: "https://cdn.example.com/talk.mp4" }, deps);
    const body = bodyOf(result);
    if (body.status !== "confirm_required") throw new Error(`expected confirm, got ${body.status}`);
    expect(body.tokens.map((t) => t.token)).toEqual(["dl-analysis", "dl-keep-1080"]);
  });

  test("confirmed_analysis_download_lands_in_cache_and_answers_with_a_plan", async () => {
    let requested: string | undefined;
    const progressReports: number[] = [];
    const deps = fakeDeps({
      probeSite: async () => siteInfoFixture(),
      downloadSite: async (req, onProgress) => {
        requested = req.format;
        onProgress?.(1024, 2048);
        const filepath = req.output.replace("%(ext)s", "webm");
        fs.writeFileSync(filepath, Buffer.alloc(2048, 1));
        fs.writeFileSync(req.subtitleOutput!.replace("%(lang)s.%(ext)s", "en.srt"), "1\nsubs");
        return { filepath };
      },
      probeFile: async () => probeOf(1800),
    });

    // 1: first contact issues tokens
    await handleVideoCall({ source: SITE_URL }, deps);
    // 2: redeem dl-analysis — downloads, then answers with a plan
    const result = await handleVideoCall(
      { source: SITE_URL, confirm: "dl-analysis" },
      deps,
      (p) => progressReports.push(p),
    );
    const body = bodyOf(result);
    if (body.status !== "plan") throw new Error(`expected plan, got ${body.status}`);
    expect(body.guidance).toContain("Analysis rip downloaded");
    expect(requested).toContain("height<=480");
    expect(progressReports).toEqual([1024]);
    // subtitle promoted to the exact video stem for sidecar pickup
    const key = urlKey(SITE_URL);
    expect(fs.existsSync(path.join(deps.config.cache.dir, "downloads", `${key}.srt`))).toBe(true);

    // 3: the URL now resolves to the cached rip and runs core on it.
    // The rip's probe (1800s) is already cached by content, so a bare
    // source would correctly answer with a plan — ask for one window.
    const argv: string[][] = [];
    deps.runCore = async (args) => {
      argv.push([...args]);
      return { code: 0, stdout: JSON.stringify(CANNED_METADATA), stderr: "" };
    };
    const ready = await handleVideoCall(
      { source: SITE_URL, from: 0, until: 45, look: "overview" },
      deps,
    );
    expect(ready.isError).toBeUndefined();
    expect(argv[0]?.[1]).toContain(key);
  });

  test("confirmed_keep_download_hands_the_file_over", async () => {
    const deps = fakeDeps({
      probeSite: async () => siteInfoFixture(),
      downloadSite: async (req) => {
        expect(req.subtitleOutput).toBeUndefined();
        const filepath = req.output.replace("%(title)s [%(id)s].%(ext)s", "A Talk [42].mp4");
        fs.writeFileSync(filepath, Buffer.alloc(1024, 1));
        return { filepath };
      },
      probeFile: async () => probeOf(1800),
    });

    await handleVideoCall({ source: SITE_URL }, deps);
    const result = await handleVideoCall({ source: SITE_URL, confirm: "dl-keep-720" }, deps);
    const body = bodyOf(result);
    if (body.status !== "plan") throw new Error(`expected plan, got ${body.status}`);
    expect(body.guidance).toContain("belongs to the user");
    expect(body.guidance).toContain("A Talk [42].mp4");
    expect(body.recommended_calls[0]?.call.source).toContain("A Talk [42].mp4");
  });

  test("hallucinated_download_token_is_rejected", async () => {
    const deps = fakeDeps();
    const result = await handleVideoCall({ source: SITE_URL, confirm: "dl-keep-4320" }, deps);
    expect(result.isError).toBe(true);
    const body = bodyOf(result);
    if (body.status !== "failed") throw new Error(`expected failed, got ${body.status}`);
    expect(body.error).toContain("stale");
  });

  test("download_failure_surfaces_as_failed_result", async () => {
    const deps = fakeDeps({
      probeSite: async () => siteInfoFixture(),
      downloadSite: async () => {
        throw new YtdlpError("yt-dlp download failed (exit 1): network sad");
      },
    });
    await handleVideoCall({ source: SITE_URL }, deps);
    const result = await handleVideoCall({ source: SITE_URL, confirm: "dl-analysis" }, deps);
    expect(result.isError).toBe(true);
    const body = bodyOf(result);
    if (body.status !== "failed") throw new Error(`expected failed, got ${body.status}`);
    expect(body.error).toContain("network sad");
  });
});
