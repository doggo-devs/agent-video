#!/usr/bin/env bash
# Local test installer for agent-video.
#
# Unlike the public installer (which pulls a prebuilt binary from npm /
# GitHub releases), this script builds agent-video from THIS checkout with
# `cargo` and installs the result — the binary is our own local artifact.
#
# Runs interactively by default (prompts for each choice); pass --yes or set
# the matching env vars for a fully non-interactive run (CI, re-installs).
#
# Choices:
#   * local ASR weights — off by default (a ~240 MB download). Recognition
#                         runs on the CPU, so there is no backend to pick
#                         and no GPU for the build to match.
#   * yt-dlp            — the site-URL source adapter; checked on PATH, offered.
#
# Env overrides (each also has a flag; flag wins):
#   AV_ASR=yes|no              AV_BIN_DIR=<dir>
#   AV_YTDLP=yes|no
#   AV_BUILD=dynamic|static    AV_ASSUME_YES=1
#   AV_INTEGRATE=yes|no        (copy skill + register the video MCP server,
#                               pointing every entry at the LOCAL binary)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OS="$(uname -s)"
ARCH="$(uname -m)"

# ── defaults (env → flag can override) ───────────────────────────────────────
ASSUME_YES="${AV_ASSUME_YES:-0}"          # 1 = never prompt, take resolved values
NONINTERACTIVE=0                           # 1 = --non-interactive: prompt-free, no tty
INSTALL_ASR="${AV_ASR:-}"                  # yes|no|"" (ask)
INSTALL_YTDLP="${AV_YTDLP:-}"             # yes|no|"" (ask)
BIN_DIR="${AV_BIN_DIR:-$HOME/.local/bin}"
BUILD_MODE="${AV_BUILD:-dynamic}"          # dynamic|static
REUSE_BINARY=""                            # --binary <path>: skip build
DO_BUILD=1                                 # --no-build: reuse target/release
DO_INTEGRATE="${AV_INTEGRATE:-}"           # yes|no|"" (ask): skill + MCP wiring

usage() {
  # The header block, however long it grows: every line up to the first
  # non-comment. A line range would silently spill into code the next time
  # the header is edited.
  sed -n '2,/^[^#]/p' "${BASH_SOURCE[0]}" | sed '$d; s/^# \{0,1\}//'
  cat <<'EOF'

Usage: scripts/install-local.sh [options]
  -y, --yes                 Non-interactive; accept resolved defaults.
      --non-interactive     Like --yes but also refuse to read from a tty.
      --asr[=yes|no]        Install local ASR weights (default: ask / no).
      --yt-dlp[=yes|no]     Install/verify yt-dlp for site URLs.
      --bin-dir <dir>       Install directory (default: ~/.local/bin).
      --static | --dynamic  FFmpeg linkage (default: dynamic).
      --no-build            Reuse an existing target/release/agent-video.
      --binary <path>       Install this prebuilt local binary (no build).
      --integrate[=yes|no]  Copy the skill + register the `video` MCP server
                            into detected agents (pi/Claude/Codex/Cursor),
                            each pointed at the local binary.
  -h, --help                Show this help.
EOF
}

# ── arg parsing ──────────────────────────────────────────────────────────────
optval() { # split "--flag=value"; echo value or empty
  case "$1" in *=*) printf '%s' "${1#*=}";; *) printf '';; esac
}
while [ $# -gt 0 ]; do
  case "$1" in
    -y|--yes) ASSUME_YES=1 ;;
    --non-interactive) ASSUME_YES=1; NONINTERACTIVE=1 ;;
    --asr|--asr=*) INSTALL_ASR="$(optval "$1")"; INSTALL_ASR="${INSTALL_ASR:-yes}" ;;
    --no-asr) INSTALL_ASR=no ;;
    --yt-dlp|--yt-dlp=*) INSTALL_YTDLP="$(optval "$1")"; INSTALL_YTDLP="${INSTALL_YTDLP:-yes}" ;;
    --no-yt-dlp) INSTALL_YTDLP=no ;;
    --bin-dir) BIN_DIR="$2"; shift ;;
    --bin-dir=*) BIN_DIR="$(optval "$1")" ;;
    --static) BUILD_MODE=static ;;
    --dynamic) BUILD_MODE=dynamic ;;
    --no-build) DO_BUILD=0 ;;
    --binary) REUSE_BINARY="$2"; DO_BUILD=0; shift ;;
    --binary=*) REUSE_BINARY="$(optval "$1")"; DO_BUILD=0 ;;
    --integrate|--integrate=*) DO_INTEGRATE="$(optval "$1")"; DO_INTEGRATE="${DO_INTEGRATE:-yes}" ;;
    --no-integrate) DO_INTEGRATE=no ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

# ── ui helpers ───────────────────────────────────────────────────────────────
bold() { printf '\033[1m%s\033[0m\n' "$*"; }
info() { printf '  %s\n' "$*"; }
warn() { printf '\033[33mwarn:\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[31merror:\033[0m %s\n' "$*" >&2; exit 1; }

# ask <prompt> <default yes|no> → returns 0 for yes. Honors --yes/--non-interactive.
ask() {
  local prompt="$1" default="$2" reply
  if [ "$ASSUME_YES" = 1 ]; then
    [ "$default" = yes ] && return 0 || return 1
  fi
  if [ "$NONINTERACTIVE" = 1 ] || [ ! -t 0 ]; then
    [ "$default" = yes ] && return 0 || return 1
  fi
  local hint="[y/N]"; [ "$default" = yes ] && hint="[Y/n]"
  read -r -p "$prompt $hint " reply || reply=""
  reply="${reply:-$default}"
  case "$reply" in [Yy]*) return 0 ;; *) return 1 ;; esac
}

# ── config file location (mirrors the `dirs` crate the binary uses) ──────────
config_file() {
  if [ -n "${AGENT_VIDEO_CONFIG:-}" ]; then
    printf '%s' "$AGENT_VIDEO_CONFIG"; return
  fi
  case "$OS" in
    Darwin) printf '%s' "$HOME/Library/Application Support/agent-video/config.toml" ;;
    *)      printf '%s' "${XDG_CONFIG_HOME:-$HOME/.config}/agent-video/config.toml" ;;
  esac
}

# ── resolve choices ──────────────────────────────────────────────────────────
bold "agent-video local installer  (${OS}/${ARCH}, build: ${BUILD_MODE})"
echo

# 1. local ASR weights
if [ -z "$INSTALL_ASR" ]; then
  if ask "Install local ASR weights? (~240 MB, CPU recognition without an API key)" no; then
    INSTALL_ASR=yes
  else
    INSTALL_ASR=no
  fi
fi

# 2. yt-dlp
if [ -z "$INSTALL_YTDLP" ]; then
  if command -v yt-dlp >/dev/null 2>&1; then
    INSTALL_YTDLP=no  # already present; nothing to do
  elif ask "yt-dlp not found — install it (needed for site URLs)?" yes; then
    INSTALL_YTDLP=yes
  else
    INSTALL_YTDLP=no
  fi
fi

# 4. platform integrations — default: ask (interactive) / off (non-interactive),
# since it writes into other tools' config. Env/flag override wins.
if [ -z "$DO_INTEGRATE" ]; then
  if ask "Wire the skill + MCP server into your agents (pi/Claude/Codex/Cursor)?" yes; then
    DO_INTEGRATE=yes
  else
    DO_INTEGRATE=no
  fi
fi

echo
bold "Plan"
info "binary       : build from source (${BUILD_MODE})${REUSE_BINARY:+  [reuse: $REUSE_BINARY]}${DO_BUILD:+}"
[ "$DO_BUILD" = 0 ] && [ -z "$REUSE_BINARY" ] && info "binary       : reuse target/release/agent-video (--no-build)"
info "install to   : $BIN_DIR/agent-video"
info "local ASR    : $INSTALL_ASR"
info "yt-dlp       : $([ "$INSTALL_YTDLP" = yes ] && echo 'install' || (command -v yt-dlp >/dev/null 2>&1 && echo 'present' || echo 'skip'))"
info "integrations : $DO_INTEGRATE  (skill + MCP → local binary)"
info "config       : $(config_file)"
echo
ask "Proceed?" yes || { echo "aborted."; exit 0; }

# ── build ────────────────────────────────────────────────────────────────────
BIN_SRC=""
if [ -n "$REUSE_BINARY" ]; then
  [ -x "$REUSE_BINARY" ] || die "not an executable: $REUSE_BINARY"
  BIN_SRC="$REUSE_BINARY"
elif [ "$DO_BUILD" = 0 ]; then
  BIN_SRC="$ROOT/target/release/agent-video"
  [ -x "$BIN_SRC" ] || die "no prebuilt binary at $BIN_SRC — drop --no-build to build it"
else
  command -v cargo >/dev/null 2>&1 || die "cargo not found — install the Rust toolchain (https://rustup.rs)"
  bold "Building agent-video (${BUILD_MODE})..."
  if [ "$BUILD_MODE" = static ]; then
    ( cd "$ROOT" && ./scripts/build-static.sh )
  else
    ( cd "$ROOT" && cargo build --release -p agent-video )
  fi
  BIN_SRC="$ROOT/target/release/agent-video"
  [ -x "$BIN_SRC" ] || die "build reported success but $BIN_SRC is missing"
fi

# ── install ──────────────────────────────────────────────────────────────────
mkdir -p "$BIN_DIR"
install -m 0755 "$BIN_SRC" "$BIN_DIR/agent-video"
BIN="$BIN_DIR/agent-video"
bold "Installed $BIN"
case ":$PATH:" in
  *":$BIN_DIR:"*) ;;
  *) warn "$BIN_DIR is not on PATH — add it, e.g.  export PATH=\"$BIN_DIR:\$PATH\"" ;;
esac

# ── config.toml ──────────────────────────────────────────────────────────────
# Nothing to write. Which engine runs is discovered, not configured: a key
# in the environment selects the API, installed weights select local
# recognition, and neither is a decision this installer has to persist.
# The file stays the operator's to write.

# ── yt-dlp ───────────────────────────────────────────────────────────────────
# Prefer an isolated-tool manager over pip: uv and pipx sidestep PEP 668
# (Debian/Ubuntu's externally-managed-environment refusal) by owning their own
# venvs, and both land the executable in ~/.local/bin. uv is first — fastest,
# self-contained, and can provision Python itself when the host lacks one.
install_ytdlp() {
  if command -v uv   >/dev/null 2>&1; then uv tool install yt-dlp && return 0; fi
  if command -v pipx >/dev/null 2>&1; then pipx install yt-dlp     && return 0; fi
  if command -v brew >/dev/null 2>&1; then brew install yt-dlp     && return 0; fi
  if command -v pip3 >/dev/null 2>&1; then pip3 install --user -U yt-dlp && return 0; fi
  if command -v pip  >/dev/null 2>&1; then pip  install --user -U yt-dlp && return 0; fi
  return 1
}
if [ "$INSTALL_YTDLP" = yes ]; then
  bold "Installing yt-dlp..."
  if install_ytdlp; then
    if command -v yt-dlp >/dev/null 2>&1; then
      info "yt-dlp: $(command -v yt-dlp)"
    else
      # Installed, but its bin dir is not on PATH — agent-video resolves
      # yt-dlp via PATH at runtime, so this would still read as "missing".
      warn "yt-dlp installed but not on PATH; add its bin dir (usually ~/.local/bin) to PATH, or site URLs will report yt-dlp missing"
    fi
  else
    warn "could not install yt-dlp automatically (no uv/pipx/brew/pip found) — install it manually: https://github.com/yt-dlp/yt-dlp"
  fi
fi

# ── whisper weights ──────────────────────────────────────────────────────────
if [ "$INSTALL_ASR" = yes ]; then
  bold "Downloading local ASR weights..."
  "$BIN" install-asr
fi

# ── platform integrations (skill + MCP + pi extension) ───────────────────────
# Every wired surface points at the LOCAL binary ($BIN), not npm. The committed
# surfaces under surfaces/ target the public npm distribution, so we reuse only
# the skill payload verbatim and synthesize MCP entries against $BIN here.
SKILL_SRC="$ROOT/surfaces/skill/agent-video"

copy_skill() { # <dest-skills-dir> <label>
  local dir="$1" label="$2"
  [ -d "$SKILL_SRC" ] || { warn "skill source missing: $SKILL_SRC"; return; }
  mkdir -p "$dir"
  cp -R "$SKILL_SRC" "$dir/"
  info "skill → $dir/agent-video  ($label)"
}

# Merge a `video` server (→ $BIN mcp) into a JSON mcpServers config, in place.
mcp_merge_json() { # <config-file> <label>
  local cfg="$1" label="$2" tmp
  mkdir -p "$(dirname "$cfg")"
  if command -v jq >/dev/null 2>&1; then
    [ -f "$cfg" ] || echo '{}' > "$cfg"
    tmp="$(mktemp)"
    jq --arg bin "$BIN" '.mcpServers.video = {command: $bin, args: ["mcp"]}' "$cfg" > "$tmp" \
      && mv "$tmp" "$cfg" && info "mcp → $cfg  ($label)" \
      || { rm -f "$tmp"; warn "could not update $cfg ($label)"; }
  else
    warn "jq not found — add to $cfg ($label) under \"mcpServers\":"
    printf '\n  "video": { "command": "%s", "args": ["mcp"] }\n\n' "$BIN"
  fi
}

# node's `${platform}-${arch}` key, matching the extension's resolveBinary().
node_platform_key() {
  local os arch
  case "$OS" in
    Darwin) os=darwin ;; Linux) os=linux ;;
    MINGW*|MSYS*|CYGWIN*) os=win32 ;; *) os="$OS" ;;
  esac
  case "$ARCH" in
    x86_64|amd64) arch=x64 ;; aarch64|arm64) arch=arm64 ;; *) arch="$ARCH" ;;
  esac
  printf '%s-%s' "$os" "$arch"
}

# Build a self-contained LOCAL pi package whose video tool resolves to $BIN
# with no npm and no network. pi references local-path packages without
# running `npm install`, so the committed npm dependency never fires; we vendor
# a @agent-video/<platform> shim whose bin/ symlinks the installed binary —
# exactly what the extension's require.resolve('@agent-video/<key>/bin/…')
# returns. The package must live at a stable path: pi records the path in
# settings and loads from it in place, so a temp dir would vanish under it.
install_pi_extension() { # → 0 on success
  local key pkg bin_name
  key="$(node_platform_key)"
  bin_name=agent-video; [ "$OS" = win32 ] && bin_name=agent-video.exe
  pkg="${AV_PI_PKG_DIR:-$HOME/.local/share/agent-video/pi}"
  [ -d "$ROOT/surfaces/pi/extensions" ] || { warn "pi source missing: $ROOT/surfaces/pi"; return 1; }

  rm -rf "$pkg"
  mkdir -p "$pkg/node_modules/@agent-video/$key/bin"
  cp -R "$ROOT/surfaces/pi/extensions" "$ROOT/surfaces/pi/lib" "$pkg/"
  # No @agent-video/cli dependency: nothing for pi (or npm) to fetch. Peer deps
  # are bundled by pi; convention discovery picks up extensions/.
  cat > "$pkg/package.json" <<'EOF'
{
  "name": "@agent-video/pi-local",
  "version": "0.0.0-local",
  "private": true,
  "peerDependencies": {
    "@earendil-works/pi-ai": "*",
    "@earendil-works/pi-coding-agent": "*",
    "typebox": "*"
  },
  "files": ["extensions", "lib"]
}
EOF
  printf '{ "name": "@agent-video/%s", "version": "0.0.0-local" }\n' "$key" \
    > "$pkg/node_modules/@agent-video/$key/package.json"
  ln -sf "$BIN" "$pkg/node_modules/@agent-video/$key/bin/$bin_name"

  pi install "$pkg" && info "pi extension → $pkg  (binary: $BIN)"
}

# Codex uses TOML; append a [mcp_servers.video] block unless one exists.
mcp_merge_codex() { # <config.toml>
  local cfg="$1"
  if grep -q '^\[mcp_servers\.video\]' "$cfg" 2>/dev/null; then
    info "codex: [mcp_servers.video] already in $cfg (left as-is)"; return
  fi
  mkdir -p "$(dirname "$cfg")"
  printf '\n[mcp_servers.video]\ncommand = "%s"\nargs = ["mcp"]\n' "$BIN" >> "$cfg"
  info "mcp → $cfg  (codex)"
}

if [ "$DO_INTEGRATE" = yes ]; then
  echo
  bold "Platform integrations (all pointed at $BIN)"

  # pi — skill dir + a self-contained local extension wired to $BIN.
  if command -v pi >/dev/null 2>&1 || [ -d "$HOME/.pi" ]; then
    ask "pi: copy the skill?" yes && copy_skill "$HOME/.pi/agent/skills" pi
    if command -v pi >/dev/null 2>&1 \
       && ask "pi: install the video extension (wired to the local binary)?" yes; then
      install_pi_extension || warn "pi extension install failed; the skill path still works"
    fi
  fi

  # Claude — skill dir + Claude Code MCP via the official CLI + Desktop config.
  if command -v claude >/dev/null 2>&1 || [ -d "$HOME/.claude" ]; then
    ask "Claude: copy the skill?" yes && copy_skill "$HOME/.claude/skills" claude
    if command -v claude >/dev/null 2>&1 \
       && ask "Claude Code: register MCP via 'claude mcp add'?" yes; then
      claude mcp add video -- "$BIN" mcp \
        && info "claude mcp add video ✓" || warn "claude mcp add failed"
    fi
  fi
  case "$OS" in
    Darwin) CD_DESKTOP="$HOME/Library/Application Support/Claude/claude_desktop_config.json" ;;
    *)      CD_DESKTOP="${XDG_CONFIG_HOME:-$HOME/.config}/Claude/claude_desktop_config.json" ;;
  esac
  [ -d "$(dirname "$CD_DESKTOP")" ] && ask "Claude Desktop: register MCP server?" yes \
    && mcp_merge_json "$CD_DESKTOP" "Claude Desktop"

  # Cursor — JSON mcpServers.
  [ -d "$HOME/.cursor" ] && ask "Cursor: register MCP server?" yes \
    && mcp_merge_json "$HOME/.cursor/mcp.json" Cursor

  # Codex — skill dir + TOML mcp_servers.
  if [ -d "$HOME/.codex" ]; then
    ask "Codex: copy the skill?" yes && copy_skill "$HOME/.codex/skills" codex
    ask "Codex: register MCP server?" yes && mcp_merge_codex "$HOME/.codex/config.toml"
  fi
fi

echo
bold "Done."
info "Try:  $BIN process <video> -o out/"
if [ "$INSTALL_ASR" = yes ]; then
  info "Local transcription ready (SenseVoice on the CPU). Set WHISPER_API_KEY to prefer the API instead."
else
  info "Transcription: API route — set WHISPER_API_KEY (Groq-compatible endpoint by default), or run 'agent-video install-asr' for local recognition."
fi
