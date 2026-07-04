#!/usr/bin/env bash
# Bump the version across core and harness, refresh Cargo.lock, commit, tag.
#
# Usage: scripts/bump.sh [patch|minor|major]   (default: patch)
#
# Pushing the tag triggers .github/workflows/release.yml, which builds and
# uploads the release binaries: git push --follow-tags
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

if [ -n "$(git status --porcelain)" ]; then
    echo "error: working tree not clean; commit or stash first" >&2
    exit 1
fi

level="${1:-patch}"
current="$(sed -n 's/^version = "\(.*\)"$/\1/p' Cargo.toml | head -n1)"

# Core and harness release in lockstep; refuse to bump from a skewed state.
harness="$(sed -n 's/^  "version": "\(.*\)",$/\1/p' packages/harness/package.json | head -n1)"
if [ "$harness" != "$current" ]; then
    echo "error: harness version $harness != crate version $current" >&2
    exit 1
fi
IFS=. read -r major minor patch <<<"$current"
case "$level" in
    major) major=$((major + 1)); minor=0; patch=0 ;;
    minor) minor=$((minor + 1)); patch=0 ;;
    patch) patch=$((patch + 1)) ;;
    *) echo "usage: $0 [patch|minor|major]" >&2; exit 2 ;;
esac
next="$major.$minor.$patch"

sed -i "0,/^version = \"$current\"\$/s//version = \"$next\"/" Cargo.toml
sed -i "0,/^  \"version\": \"$current\",\$/s//  \"version\": \"$next\",/" \
    packages/harness/package.json
# Discovery surfaces embed the version (codex plugin manifest, pi package pin).
bun run gen:surfaces
# Sync Cargo.lock's own-package entry without touching dependency versions.
cargo update --workspace --quiet
# The tagged commit must at least build; catches a lockfile/toml mishap here
# rather than in the release workflow.
cargo check --quiet

git add Cargo.toml Cargo.lock packages/harness/package.json surfaces .agents
git commit -m "Release v$next"
git tag -a "v$next" -m "Release v$next"
echo "v$next tagged — publish with: git push --follow-tags"
