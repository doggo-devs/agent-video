#!/usr/bin/env bash
# Decode a sample with target/static/agent-video on an emulated CPU at the
# claimed baseline: instructions leaked from the build host's newer ISA
# (AVX-512, SVE) SIGILL here instead of on users' machines.
#
# Usage: scripts/smoke-static.sh <qemu CPU model>  (e.g. Haswell-noTSX, cortex-a72)
# Requirements: qemu-user and ffmpeg with libx264 on the host.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

ffmpeg -loglevel error -f lavfi -i testsrc2=s=640x360:r=25:d=3 \
  -c:v libx264 -pix_fmt yuv420p "$work/smoke.mp4"
"qemu-$(uname -m)" -cpu "$1" "$ROOT/target/static/agent-video" \
  process "$work/smoke.mp4" -o "$work/out"
