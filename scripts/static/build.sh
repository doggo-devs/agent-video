#!/usr/bin/env bash
# The fully static build, as run inside the Alpine builder image. Invoke it
# through scripts/build-static.sh, which provides that image.
#
# Built without `local-asr`, so transcription here is the API route only.
# The prebuilt onnxruntime the local recognizer links is glibc-built and
# incompatible with musl: a static musl binary segfaults inside musl's
# allocator while loading any model — reproduced in a plain C program with no
# Rust involved, so it is upstream's constraint rather than this build's.
# Zero dynamic dependencies is the point of this variant; carrying a
# recognizer that crashes is not a trade worth making.
#
# FFmpeg is compiled and linked statically. One capability needs a lib it does
# not bundle: software AV1 (libdav1d). This builds it as a static lib, exposes
# it via pkg-config, then builds with `static-ffmpeg` so FFmpeg's configure
# picks up --enable-libdav1d. No hwaccel (--disable-autodetect) and no TLS
# backend: URL downloads happen in the harness (rustls), FFmpeg only opens
# local files.
#
# Output: target/static/agent-video, a static-pie for the container's arch,
# checked before it is placed there: no interpreter, no needed libraries.
#
# Environment knobs:
#   DAV1D_VERSION      dav1d git tag                     (default: 1.5.3)
#   FFMPEG_TARGET_CPU  -march baseline for distributable builds (e.g.
#                      x86-64-v3), overriding ffmpeg-sys-next's -march=native.
set -euo pipefail

DAV1D_VERSION="${DAV1D_VERSION:-1.5.3}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="$ROOT/target/static"
PREFIX="$OUT/deps"
TARGET="$(rustc -vV | sed -n 's/^host: //p')"
triple_lower="${TARGET//-/_}"
triple_upper="$(echo "$triple_lower" | tr '[:lower:]' '[:upper:]')"

# The image is discarded after each run; keep the crate cache with the build.
export CARGO_HOME="$OUT/cargo"
export CARGO_TARGET_DIR="$OUT"

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

# --- 2. Point FFmpeg's configure (via pkg-config) at the static deps ---------
export PKG_CONFIG_PATH="$PREFIX/lib/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
export PKG_CONFIG_ALL_STATIC=1

# --- 3. Optionally pin the FFmpeg C baseline for distributable artifacts ------
# ffmpeg-sys-next hard-codes `-march=native` for host builds. A later -march on
# the compiler line wins, so a thin CC wrapper that appends the desired baseline
# neutralizes it without patching the crate.
#
# Both names have to move. The `cc` crate resolves the compiler from
# `CC_<triple>` first and only falls back to `CC`, so exporting only one would
# let the baseline be announced here and stay absent from the object files.
#
# The wrapper keeps the real compiler's filename, in a directory of its own.
# FFmpeg's configure splits a compiler name into cross-prefix and tool and
# re-joins them as `${prefix}gcc`, so a wrapper named anything else is looked
# up under a name that does not exist.
if [ -n "${FFMPEG_TARGET_CPU:-}" ]; then
  real_cc="$(command -v cc)"
  shim_dir="$OUT/cc-shim"
  mkdir -p "$shim_dir"
  wrapper="$shim_dir/$(basename "$real_cc")"
  cat > "$wrapper" <<EOF
#!/bin/sh
exec $real_cc "\$@" -march=$FFMPEG_TARGET_CPU -mtune=generic
EOF
  chmod +x "$wrapper"
  export PATH="$shim_dir:$PATH"
  export CC="$wrapper"
  export "CC_${triple_lower}=$wrapper"
  echo ">> pinning FFmpeg C baseline to -march=$FFMPEG_TARGET_CPU (via $real_cc)"
fi

# --- 4. Link flags -----------------------------------------------------------
# link-self-contained=no hands crt objects and libc to Alpine's gcc instead of
# Rust's bundled musl, so the C deps and Rust share one libc and `-static-pie`
# gets the rcrt1.o it needs.
#
# The CPU baseline is restated only on x86_64, where the gcc -march value
# (x86-64-v3) doubles as a valid rustc target-cpu; aarch64 -march values like
# armv8-a are not rustc CPU names, and rustc's default `generic` already is the
# armv8-a baseline.
rustflags="-C link-self-contained=no"
if [[ -n "${FFMPEG_TARGET_CPU:-}" && "$TARGET" == x86_64* ]]; then
  rustflags="$rustflags -C target-cpu=$FFMPEG_TARGET_CPU"
fi
export "CARGO_TARGET_${triple_upper}_RUSTFLAGS=$rustflags"

# --- 5. Build ----------------------------------------------------------------
# The explicit --target is what lets the wrapper tell build scripts apart from
# the binary: they must stay dynamically linked, since bindgen dlopen()s
# libclang.
export RUSTC_WRAPPER="$ROOT/scripts/static/rustc-host-dynamic"
features="--no-default-features --features static-ffmpeg"
echo ">> cargo build --release --locked -p agent-video $features --target $TARGET"
# shellcheck disable=SC2086 # $features is two flag pairs, not one word.
cargo build --release --locked -p agent-video $features --target "$TARGET" "$@"

# --- 6. Check and publish ------------------------------------------------------
# Static means nothing for the kernel or a loader to resolve at exec time, and
# a DYN-type image is what lets the kernel randomize where it lands.
bin="$OUT/$TARGET/release/agent-video"
headers="$(readelf -lhdW "$bin")"
fail() { echo "error: $bin $1" >&2; exit 1; }
grep -q 'Requesting program interpreter' <<<"$headers" && fail "requests a program interpreter"
grep -q '(NEEDED)' <<<"$headers" && fail "needs shared libraries"
# ceiling: rustc's aarch64-unknown-linux-musl spec lacks static-position-independent-executables, so
# aarch64 links static non-PIE (no ASLR); require DYN there once rustc enables static-pie for it.
if [[ "$TARGET" != aarch64* ]]; then
  grep -q 'Type:[[:space:]]*DYN' <<<"$headers" || fail "is not position-independent"
fi
install -m 0755 "$bin" "$OUT/agent-video"
echo ">> built $OUT/agent-video (static-pie, $TARGET)"
