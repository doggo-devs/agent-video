#!/usr/bin/env bash
# Build the fully static, zero-dependency agent-video: FFmpeg, dav1d and musl
# all linked into one static-pie that loads on any Linux of the host's
# architecture. The work runs inside an Alpine builder image
# (scripts/static/Containerfile) whose musl-native gcc makes static-pie link
# correctly; the result lands at target/static/agent-video.
#
# Requirements: podman or docker.
#
# Environment knobs:
#   CONTAINER          runtime to use     (default: podman, else docker)
#   DAV1D_VERSION      forwarded to scripts/static/build.sh
#   FFMPEG_VERSION     forwarded to scripts/static/build.sh
#   FFMPEG_TARGET_CPU  forwarded to scripts/static/build.sh
# Extra arguments are passed through to `cargo build`.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="agent-video-static-builder"

runtime="${CONTAINER:-$(command -v podman || command -v docker || true)}"
[ -n "$runtime" ] || { echo "error: needs podman or docker" >&2; exit 1; }

echo ">> preparing builder image $IMAGE"
"$runtime" build -q -t "$IMAGE" -f "$ROOT/scripts/static/Containerfile" \
  "$ROOT/scripts/static" >/dev/null

# Run as the invoking user so everything written under target/ stays theirs.
case "$(basename "$runtime")" in
  podman) user_args=(--userns=keep-id) ;;
  *) user_args=(--user "$(id -u):$(id -g)") ;;
esac

exec "$runtime" run --rm "${user_args[@]}" \
  -v "$ROOT:/src" -w /src \
  -e HOME=/tmp -e DAV1D_VERSION -e FFMPEG_VERSION -e FFMPEG_TARGET_CPU \
  "$IMAGE" scripts/static/build.sh "$@"
