#!/usr/bin/env bash
# Build a fully static, zero-dependency agent-video.
#
# FFmpeg is compiled and linked statically. One capability needs a lib it does
# not bundle: software AV1 (libdav1d). This builds it as a static lib, exposes
# it via pkg-config, then builds with `static-ffmpeg` so FFmpeg's configure
# picks up --enable-libdav1d. No hwaccel (--disable-autodetect) and no TLS
# backend: URL downloads happen in the harness (rustls), FFmpeg only opens
# local files.
#
# Requirements: git, nasm, meson, ninja, pkg-config, a C toolchain.
# For a musl target, also musl-tools + the rustup target.
#
# Environment knobs:
#   DAV1D_VERSION      dav1d git tag                     (default: 1.5.3)
#   DEPS_PREFIX        static-lib install prefix
#                      (default: target/static-deps-<libc>)
#   TARGET             Rust target triple. A *-musl triple cross-builds the C
#                      deps + Rust against musl for a no-dynamic-deps binary;
#                      unset = host glibc target.
#   FFMPEG_TARGET_CPU  -march baseline for distributable builds (e.g.
#                      x86-64-v3), overriding ffmpeg-sys-next's -march=native.
set -euo pipefail

DAV1D_VERSION="${DAV1D_VERSION:-1.5.3}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="${TARGET:-}"
JOBS="$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)"

# --- musl toolchain wiring ---------------------------------------------------
# A *-musl target builds the C deps + Rust against musl: CC=musl-gcc for the C
# side, the linker var for the Rust side. `link-self-contained=no` (set below)
# reuses that same distro musl instead of Rust's bundled copy — one libc for the
# whole binary.
LIBC="gnu"
if [[ "$TARGET" == *musl* ]]; then
  LIBC="musl"
  export CC="${CC:-musl-gcc}"
  # cc reads CC_<triple> (lowercase); cargo reads CARGO_TARGET_<TRIPLE>_*.
  triple_lower="${TARGET//-/_}"
  triple_upper="$(echo "$triple_lower" | tr '[:lower:]' '[:upper:]')"
  export "CC_${triple_lower}=${CC}"
  export "CARGO_TARGET_${triple_upper}_LINKER=${CC}"
fi

PREFIX="${DEPS_PREFIX:-$ROOT/target/static-deps-$LIBC}"

# --- 1. Build dav1d as a static library (skip if already present) ------------
if [ -f "$PREFIX/lib/libdav1d.a" ]; then
  echo ">> reusing dav1d static lib at $PREFIX/lib/libdav1d.a"
else
  echo ">> building dav1d $DAV1D_VERSION (static, $LIBC) into $PREFIX"
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

# --- 2. Point FFmpeg's configure (via pkg-config) at the static deps ---------
export PKG_CONFIG_PATH="$PREFIX/lib/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
export PKG_CONFIG_ALL_STATIC=1

# --- 3. Optionally pin the FFmpeg C baseline for distributable artifacts ------
# ffmpeg-sys-next hard-codes `-march=native` for host builds. A later -march on
# the compiler line wins, so a thin CC wrapper that appends the desired baseline
# neutralizes it without patching the crate. The wrapper also carries any
# musl-gcc CC set above.
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

# --- 4. musl link flags ------------------------------------------------------
# Env RUSTFLAGS replaces .cargo/config.toml's cfg rustflags wholesale, so a musl
# build must restate both: link-self-contained=no (reuse the distro musl) and
# the CPU baseline. The baseline is restated only on x86_64, where the gcc
# -march value (x86-64-v3) doubles as a valid rustc target-cpu; aarch64 -march
# values like armv8-a are not rustc CPU names, and rustc's default `generic`
# already is the armv8-a baseline.
if [[ "$TARGET" == *musl* ]]; then
  triple_upper="$(echo "${TARGET//-/_}" | tr '[:lower:]' '[:upper:]')"
  musl_rustflags="-C link-self-contained=no"
  if [[ -n "${FFMPEG_TARGET_CPU:-}" && "$TARGET" == x86_64* ]]; then
    musl_rustflags="$musl_rustflags -C target-cpu=$FFMPEG_TARGET_CPU"
  fi
  export "CARGO_TARGET_${triple_upper}_RUSTFLAGS=$musl_rustflags"
fi

# --- 5. Build ----------------------------------------------------------------
target_args=()
[ -n "$TARGET" ] && target_args=(--target "$TARGET")
echo ">> cargo build --release -p agent-video --features static-ffmpeg ${target_args[*]}"
exec cargo build --release -p agent-video --features static-ffmpeg "${target_args[@]}" "$@"
