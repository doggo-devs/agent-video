import { describe, expect, test } from "bun:test";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { ConfigError, defaultConfig, loadConfig } from "../src/store/config.ts";

function tmpConfig(json: string): string {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "av-config-"));
  const file = path.join(dir, "config.json");
  fs.writeFileSync(file, json);
  return file;
}

describe("loadConfig", () => {
  test("returns_defaults_without_file_or_env", () => {
    const cfg = loadConfig({ AGENT_VIDEO_CONFIG: "/nonexistent/config.json" }, "/home/u");
    expect(cfg).toEqual(defaultConfig("/home/u"));
    expect(cfg.budget.autoRun).toBe(2);
    expect(cfg.download.keepDir).toBe(path.join("/home/u", "Downloads"));
  });

  test("file_values_override_defaults", () => {
    const file = tmpConfig(
      JSON.stringify({ budget: { max: 16 }, download: { keep_dir: "/mnt/media" } }),
    );
    const cfg = loadConfig({ AGENT_VIDEO_CONFIG: file }, "/home/u");
    expect(cfg.budget.max).toBe(16);
    expect(cfg.budget.autoRun).toBe(2); // untouched default
    expect(cfg.download.keepDir).toBe("/mnt/media");
  });

  test("env_overrides_file", () => {
    const file = tmpConfig(JSON.stringify({ budget: { max: 16 } }));
    const cfg = loadConfig(
      { AGENT_VIDEO_CONFIG: file, AGENT_VIDEO_BUDGET_MAX: "4" },
      "/home/u",
    );
    expect(cfg.budget.max).toBe(4);
  });

  test("rejects_unknown_key", () => {
    const file = tmpConfig(JSON.stringify({ budget: { auto_ruun: 3 } }));
    expect(() => loadConfig({ AGENT_VIDEO_CONFIG: file }, "/home/u")).toThrow(ConfigError);
  });

  test("rejects_non_positive_int", () => {
    const file = tmpConfig(JSON.stringify({ cache: { max_bytes: -1 } }));
    expect(() => loadConfig({ AGENT_VIDEO_CONFIG: file }, "/home/u")).toThrow(ConfigError);
  });

  test("rejects_malformed_json", () => {
    const file = tmpConfig("{ not json");
    expect(() => loadConfig({ AGENT_VIDEO_CONFIG: file }, "/home/u")).toThrow(ConfigError);
  });

  test("rejects_invalid_env_int", () => {
    expect(() =>
      loadConfig(
        { AGENT_VIDEO_CONFIG: "/nonexistent/config.json", AGENT_VIDEO_BUDGET_MAX: "eight" },
        "/home/u",
      ),
    ).toThrow(ConfigError);
  });
});
