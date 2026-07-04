#!/usr/bin/env bash
# Build a fully static, zero-dependency agent-video.
#
# The `static-ffmpeg` feature compiles FFmpeg from source and links it in, but
# FFmpeg's native AV1 decoder is a hardware-only stub — software AV1 decoding
# needs libdav1d. This script builds dav1d as a static library first, exposes it
# via pkg-config, and then builds with `static-ffmpeg` so FFmpeg's configure
# picks up `--enable-libdav1d` and absorbs libdav1d.a into the binary.
#
# The result needs no FFmpeg installation at runtime and decodes AV1 in software.
# It has NO hardware acceleration (FFmpeg is configured --disable-autodetect);
# decoding is pure CPU (dav1d SIMD, dispatched at runtime).
#
# Requirements: git, nasm, meson, ninja, pkg-config, a C toolchain.
#
# Environment knobs:
#   DAV1D_VERSION      dav1d git tag to build            (default: 1.5.3)
#   DAV1D_PREFIX       install prefix for the static lib (default: target/dav1d-static)
#   FFMPEG_TARGET_CPU  if set, FFmpeg C code is compiled with -march=<value>
#                      -mtune=generic instead of the -march=native that
#                      ffmpeg-sys-next forces. Set this for distributable builds
#                      (e.g. x86-64-v3 on x86_64, armv8-a on aarch64) so the
#                      artifact runs on any CPU of that baseline. Leave unset for
#                      a local build tuned to this machine.
set -euo pipefail

DAV1D_VERSION="${DAV1D_VERSION:-1.5.3}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PREFIX="${DAV1D_PREFIX:-$ROOT/target/dav1d-static}"

# --- 1. Build dav1d as a static library (skip if already present) ------------
if [ -f "$PREFIX/lib/libdav1d.a" ]; then
  echo ">> reusing dav1d static lib at $PREFIX/lib/libdav1d.a"
else
  echo ">> building dav1d $DAV1D_VERSION (static) into $PREFIX"
  work="$(mktemp -d)"
  trap 'rm -rf "$work"' EXIT
  git clone --depth 1 --branch "$DAV1D_VERSION" \
    https://code.videolan.org/videolan/dav1d.git "$work/src"
  meson setup "$work/build" "$work/src" \
    --prefix="$PREFIX" \
    --libdir=lib \
    --default-library=static \
    --buildtype=release \
    -Denable_tools=false \
    -Denable_tests=false
  ninja -C "$work/build"
  ninja -C "$work/build" install
  rm -rf "$work"
  trap - EXIT
fi

# --- 2. Point FFmpeg's configure (via pkg-config) at the static dav1d --------
export PKG_CONFIG_PATH="$PREFIX/lib/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
export PKG_CONFIG_ALL_STATIC=1

# --- 3. Optionally pin the FFmpeg C baseline for distributable artifacts ------
# ffmpeg-sys-next hard-codes `-march=native` for host builds. A later -march on
# the compiler line wins, so a thin CC wrapper that appends the desired baseline
# neutralizes it without patching the crate.
if [ -n "${FFMPEG_TARGET_CPU:-}" ]; then
  wrapper="$ROOT/target/ffmpeg-cc-wrapper"
  real_cc="${CC:-cc}"
  cat > "$wrapper" <<EOF
#!/bin/sh
exec $real_cc "\$@" -march=$FFMPEG_TARGET_CPU -mtune=generic
EOF
  chmod +x "$wrapper"
  export CC="$wrapper"
  echo ">> pinning FFmpeg C baseline to -march=$FFMPEG_TARGET_CPU (via $real_cc)"
fi

# --- 4. Build ----------------------------------------------------------------
echo ">> cargo build --release --features static-ffmpeg"
exec cargo build --release --features static-ffmpeg "$@"
