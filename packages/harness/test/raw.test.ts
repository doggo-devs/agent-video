import { describe, expect, test } from "bun:test";
import { ArgError } from "../src/policy/errors.ts";
import { parseRaw, tokenize } from "../src/policy/raw.ts";

describe("tokenize", () => {
  test("splits_on_whitespace_and_respects_quotes", () => {
    expect(tokenize('process --subtitle "my subs.srt" -j 4')).toEqual([
      "process",
      "--subtitle",
      "my subs.srt",
      "-j",
      "4",
    ]);
  });
});

describe("parseRaw", () => {
  test("requires_a_core_subcommand", () => {
    expect(() => parseRaw("--strategy keyframes")).toThrow(ArgError);
    expect(() => parseRaw("probe")).toThrow(ArgError);
  });

  test("rejects_plumbing_flags", () => {
    expect(() => parseRaw("process --json")).toThrow(ArgError);
    expect(() => parseRaw("process -o /tmp/out")).toThrow(ArgError);
    expect(() => parseRaw("process --output /tmp/out")).toThrow(ArgError);
  });

  test("extracts_chunk_seconds_for_budgeting", () => {
    expect(parseRaw("process").chunkSeconds).toBe(60);
    expect(parseRaw("process --profile fast").chunkSeconds).toBe(90);
    expect(parseRaw("process --profile quality").chunkSeconds).toBe(45);
    expect(parseRaw("process --profile fast --chunk-seconds 120").chunkSeconds).toBe(120);
  });

  test("passes_strategy_flags_through_verbatim", () => {
    const inv = parseRaw("process --strategy keyframes --base-fps 0.2");
    expect(inv.subcommand).toBe("process");
    expect(inv.flags).toEqual(["--strategy", "keyframes", "--base-fps", "0.2"]);
  });
});
