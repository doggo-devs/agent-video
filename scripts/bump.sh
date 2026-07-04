#!/usr/bin/env bash
# Bump the crate version, refresh Cargo.lock, commit, and tag.
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
IFS=. read -r major minor patch <<<"$current"
case "$level" in
    major) major=$((major + 1)); minor=0; patch=0 ;;
    minor) minor=$((minor + 1)); patch=0 ;;
    patch) patch=$((patch + 1)) ;;
    *) echo "usage: $0 [patch|minor|major]" >&2; exit 2 ;;
esac
next="$major.$minor.$patch"

sed -i "0,/^version = \"$current\"\$/s//version = \"$next\"/" Cargo.toml
# Sync Cargo.lock's own-package entry without touching dependency versions.
cargo update --workspace --quiet
# The tagged commit must at least build; catches a lockfile/toml mishap here
# rather than in the release workflow.
cargo check --quiet

git add Cargo.toml Cargo.lock
git commit -m "Release v$next"
git tag -a "v$next" -m "Release v$next"
echo "v$next tagged — publish with: git push --follow-tags"
