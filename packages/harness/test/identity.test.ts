import { describe, expect, test } from "bun:test";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import {
  SLICE_BYTES,
  digestParts,
  fingerprintFile,
  type SliceIo,
} from "../src/store/identity.ts";

function tmpFile(content: Buffer | string): string {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "av-identity-"));
  const file = path.join(dir, "video.bin");
  fs.writeFileSync(file, content);
  return file;
}

describe("digestParts", () => {
  test("digest_changes_with_size_alone", () => {
    const head = Buffer.from("same");
    expect(digestParts(head, Buffer.alloc(0), 100)).not.toBe(
      digestParts(head, Buffer.alloc(0), 101),
    );
  });
});

describe("fingerprintFile", () => {
  test("same_content_yields_same_fingerprint_across_paths", async () => {
    const content = Buffer.alloc(3 * SLICE_BYTES, 7);
    const a = await fingerprintFile(tmpFile(content));
    const b = await fingerprintFile(tmpFile(content));
    expect(a.hex).toBe(b.hex);
    expect(a.stable).toBe(true);
  });

  test("rename_preserves_fingerprint", async () => {
    const file = tmpFile(Buffer.alloc(3 * SLICE_BYTES, 9));
    const before = await fingerprintFile(file);
    const moved = path.join(path.dirname(file), "renamed.mp4");
    fs.renameSync(file, moved);
    const after = await fingerprintFile(moved);
    expect(after.hex).toBe(before.hex);
  });

  test("in_place_edit_changes_fingerprint", async () => {
    const file = tmpFile(Buffer.alloc(3 * SLICE_BYTES, 1));
    const before = await fingerprintFile(file);
    const fd = fs.openSync(file, "r+");
    fs.writeSync(fd, Buffer.from([2]), 0, 1, 0); // flip one head byte
    fs.closeSync(fd);
    const after = await fingerprintFile(file);
    expect(after.hex).not.toBe(before.hex);
  });

  test("hashes_small_files_in_full", async () => {
    const a = await fingerprintFile(tmpFile("tiny-a"));
    const b = await fingerprintFile(tmpFile("tiny-b"));
    expect(a.hex).not.toBe(b.hex);
    expect(a.size).toBe(6);
  });

  test("flags_unstable_when_file_changes_mid_read", async () => {
    let statCalls = 0;
    const fake: SliceIo = {
      async stat() {
        statCalls += 1;
        return { size: 10, mtimeMs: statCalls === 1 ? 1000 : 2000 };
      },
      async readSlice(_path, _offset, length) {
        return Buffer.alloc(Math.min(length, 10));
      },
    };
    const fp = await fingerprintFile("/fake/being-written.mp4", fake);
    expect(fp.stable).toBe(false);
  });
});
