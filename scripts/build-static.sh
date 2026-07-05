#!/usr/bin/env bash
# Build a fully static, zero-dependency agent-video.
#
# FFmpeg is compiled and linked statically. Two capabilities need libs it does
# not bundle: software AV1 (libdav1d) and https (OpenSSL). This builds both as
# static libs, exposes them via pkg-config, then builds with `static-ffmpeg` so
# FFmpeg's configure picks up --enable-libdav1d + --enable-openssl. No hwaccel
# (--disable-autodetect); TLS uses the system CA store at runtime.
#
# Requirements: git, nasm, meson, ninja, pkg-config, perl, a C toolchain.
# For a musl target, also musl-tools + the rustup target.
#
# Environment knobs:
#   DAV1D_VERSION      dav1d git tag                     (default: 1.5.3)
#   OPENSSL_VERSION    openssl git tag                   (default: openssl-3.5.1)
#   DEPS_PREFIX        static-lib install prefix
#                      (default: target/static-deps-<libc>)
#   TARGET             Rust target triple. A *-musl triple cross-builds the C
#                      deps + Rust against musl for a no-dynamic-deps binary;
#                      unset = host glibc target.
#   FFMPEG_TARGET_CPU  -march baseline for distributable builds (e.g.
#                      x86-64-v3), overriding ffmpeg-sys-next's -march=native.
set -euo pipefail

DAV1D_VERSION="${DAV1D_VERSION:-1.5.3}"
OPENSSL_VERSION="${OPENSSL_VERSION:-openssl-3.5.1}"
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

# --- 2. Build OpenSSL as a static library (skip if already present) ----------
# libssl + libcrypto only; no apps, no docs, no shared objects. `no-module`
# keeps the provider .so out so nothing needs to be found at runtime.
if [ -f "$PREFIX/lib/libssl.a" ] && [ -f "$PREFIX/lib/libcrypto.a" ]; then
  echo ">> reusing openssl static libs at $PREFIX/lib"
else
  echo ">> building openssl $OPENSSL_VERSION (static, $LIBC) into $PREFIX"
  work="$(mktemp -d)"
  trap 'rm -rf "$work"' EXIT
  git clone --depth 1 --branch "$OPENSSL_VERSION" \
    https://github.com/openssl/openssl.git "$work/src"
  (
    cd "$work/src"
    # Configure auto-detects the platform from CC; --libdir=lib matches the
    # pkgconfig path FFmpeg's configure searches.
    ./Configure \
      --prefix="$PREFIX" \
      --libdir=lib \
      no-shared \
      no-module \
      no-tests \
      no-docs
    make -j"$JOBS"
    # install_sw: libs + headers only, no man pages, no default cert store.
    make install_sw
  )
  rm -rf "$work"
  trap - EXIT
fi

# --- 3. Point FFmpeg's configure (via pkg-config) at the static deps ---------
export PKG_CONFIG_PATH="$PREFIX/lib/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
export PKG_CONFIG_ALL_STATIC=1

# --- 4. Optionally pin the FFmpeg C baseline for distributable artifacts ------
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

# --- 5. musl link flags ------------------------------------------------------
# Env RUSTFLAGS replaces .cargo/config.toml's cfg rustflags wholesale, so a musl
# build must restate both: link-self-contained=no (reuse the distro musl) and
# the CPU baseline (kept in sync with FFMPEG_TARGET_CPU).
if [[ "$TARGET" == *musl* ]]; then
  triple_upper="$(echo "${TARGET//-/_}" | tr '[:lower:]' '[:upper:]')"
  musl_rustflags="-C link-self-contained=no"
  [ -n "${FFMPEG_TARGET_CPU:-}" ] && musl_rustflags="$musl_rustflags -C target-cpu=$FFMPEG_TARGET_CPU"
  export "CARGO_TARGET_${triple_upper}_RUSTFLAGS=$musl_rustflags"
fi

# --- 6. Build ----------------------------------------------------------------
target_args=()
[ -n "$TARGET" ] && target_args=(--target "$TARGET")
echo ">> cargo build --release --features static-ffmpeg ${target_args[*]}"
exec cargo build --release --features static-ffmpeg "${target_args[@]}" "$@"
