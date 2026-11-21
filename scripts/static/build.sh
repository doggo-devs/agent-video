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
# FFmpeg and its one external codec lib, dav1d (software AV1), are built here
# as static libs into target/static/deps, then linked through pkg-config by
# the `static-ffmpeg` feature. No hwaccel (--disable-autodetect) and no TLS
# backend: URL downloads happen in the harness (rustls), FFmpeg only opens
# local files. The deps prefix is reused while its STAMP matches, so CI can
# drop in a prebuilt one (see .github/workflows/ffmpeg-musl.yml).
#
# Output: target/static/agent-video, a static-pie for the container's arch,
# checked before it is placed there: no interpreter, no needed libraries.
#
# Environment knobs:
#   DAV1D_VERSION      dav1d git tag                     (default: 1.5.3)
#   FFMPEG_VERSION     FFmpeg git tag                    (default: n9.0.2)
#   FFMPEG_TARGET_CPU  -march baseline for distributable builds (e.g.
#                      x86-64-v3); unset leaves the compiler default.
set -euo pipefail

DAV1D_VERSION="${DAV1D_VERSION:-1.5.3}"
FFMPEG_VERSION="${FFMPEG_VERSION:-n9.0.2}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="$ROOT/target/static"
PREFIX="$OUT/deps"
TARGET="$(rustc -vV | sed -n 's/^host: //p')"
triple_upper="$(echo "${TARGET//-/_}" | tr '[:lower:]' '[:upper:]')"

# The image is discarded after each run; keep the crate cache with the build.
export CARGO_HOME="$OUT/cargo"
export CARGO_TARGET_DIR="$OUT"
export PKG_CONFIG_PATH="$PREFIX/lib/pkgconfig"
export PKG_CONFIG_ALL_STATIC=1

# --- 1. Static deps: dav1d, then FFmpeg against it ---------------------------
# Everything that decides the deps' bytes goes in the stamp.
stamp="$TARGET dav1d-$DAV1D_VERSION ffmpeg-$FFMPEG_VERSION march-${FFMPEG_TARGET_CPU:-default}"
if [ "$(cat "$PREFIX/STAMP" 2>/dev/null)" = "$stamp" ]; then
  echo ">> reusing static deps at $PREFIX ($stamp)"
else
  echo ">> building static deps into $PREFIX ($stamp)"
  rm -rf "$PREFIX"
  work="$(mktemp -d)"
  trap 'rm -rf "$work"' EXIT

  git clone --depth 1 --branch "$DAV1D_VERSION" \
    https://code.videolan.org/videolan/dav1d.git "$work/dav1d"
  meson setup "$work/dav1d-build" "$work/dav1d" \
    --prefix="$PREFIX" \
    --libdir=lib \
    --default-library=static \
    --buildtype=release \
    -Denable_tools=false \
    -Denable_tests=false
  ninja -C "$work/dav1d-build" install

  # Only the libraries ffmpeg-next is built with; avutil comes with them. The
  # -march baseline matters: plain C has no runtime CPU check, so the build
  # host's ISA (AVX-512 on GitHub's x86 runners) must never leak in.
  cpu_flags=()
  if [ -n "${FFMPEG_TARGET_CPU:-}" ]; then
    cpu_flags=(--extra-cflags="-march=$FFMPEG_TARGET_CPU -mtune=generic")
  fi
  git clone --depth 1 --branch "$FFMPEG_VERSION" \
    https://github.com/FFmpeg/FFmpeg.git "$work/ffmpeg"
  (cd "$work/ffmpeg" && ./configure \
    --prefix="$PREFIX" \
    --pkg-config-flags=--static \
    --enable-static --disable-shared --enable-pic --enable-pthreads \
    --disable-autodetect --disable-programs --disable-doc --disable-debug \
    --disable-avdevice --disable-avfilter \
    --enable-libdav1d \
    "${cpu_flags[@]}")
  make -C "$work/ffmpeg" -j"$(nproc)" install

  echo "$stamp" >"$PREFIX/STAMP"
  rm -rf "$work"
  trap - EXIT
fi

# --- 2. Link flags -----------------------------------------------------------
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

# --- 3. Build ----------------------------------------------------------------
# The explicit --target is what lets the wrapper tell build scripts apart from
# the binary: they must stay dynamically linked, since bindgen dlopen()s
# libclang.
export RUSTC_WRAPPER="$ROOT/scripts/static/rustc-host-dynamic"
features="--no-default-features --features static-ffmpeg"
echo ">> cargo build --release --locked -p agent-video $features --target $TARGET"
# shellcheck disable=SC2086 # $features is two flag pairs, not one word.
cargo build --release --locked -p agent-video $features --target "$TARGET" "$@"

# --- 4. Check and publish ------------------------------------------------------
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
