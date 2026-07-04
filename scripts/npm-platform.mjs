#!/usr/bin/env node
// Generate a per-platform npm package (esbuild/Biome pattern) that carries
// the prebuilt core binary. The harness resolves these via optionalDependencies.
//
// Usage: node scripts/npm-platform.mjs <platform> <version> <binary> <out-dir>
//   platform  one of: linux-x64 linux-arm64 darwin-x64 darwin-arm64 win32-x64
//   version   package version, must match the core crate version
//   binary    path to the built agent-video[.exe]
//   out-dir   directory to write the package into (created/overwritten)

import { chmodSync, copyFileSync, mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";

const PLATFORMS = {
  "linux-x64": { os: "linux", cpu: "x64" },
  "linux-arm64": { os: "linux", cpu: "arm64" },
  "darwin-x64": { os: "darwin", cpu: "x64" },
  "darwin-arm64": { os: "darwin", cpu: "arm64" },
  "win32-x64": { os: "win32", cpu: "x64" },
};

const [platform, version, binary, outDir] = process.argv.slice(2);
const spec = PLATFORMS[platform];
if (!spec || !version || !binary || !outDir) {
  console.error(
    "usage: npm-platform.mjs <platform> <version> <binary> <out-dir>\n" +
      `  platform: ${Object.keys(PLATFORMS).join(" ")}`,
  );
  process.exit(2);
}

const binName = spec.os === "win32" ? "agent-video.exe" : "agent-video";

mkdirSync(join(outDir, "bin"), { recursive: true });

writeFileSync(
  join(outDir, "package.json"),
  JSON.stringify(
    {
      name: `@agent-video/${platform}`,
      version,
      description: `agent-video core binary for ${spec.os} ${spec.cpu}`,
      license: "MIT",
      repository: {
        type: "git",
        url: "git+https://github.com/durvia/agent-video.git",
      },
      preferUnplugged: true,
      os: [spec.os],
      cpu: [spec.cpu],
    },
    null,
    2,
  ) + "\n",
);

writeFileSync(
  join(outDir, "README.md"),
  `# @agent-video/${platform}\n\n` +
    `Prebuilt \`agent-video\` core binary for ${spec.os} ${spec.cpu}. ` +
    "Installed automatically as an optional dependency of " +
    "[@agent-video/harness](https://github.com/durvia/agent-video); " +
    "not meant to be installed directly.\n",
);

const dest = join(outDir, "bin", binName);
copyFileSync(binary, dest);
// upload-artifact drops the executable bit; restore it here.
chmodSync(dest, 0o755);

console.log(`${outDir}: @agent-video/${platform}@${version} (${binName})`);
