#!/usr/bin/env bash
set -euo pipefail
repo="$(cd "$(dirname "$0")/.." && pwd)"
target="${1:-/tmp/th_supermimic-server.tar.gz}"
name="$(basename "$repo")"
tar --exclude=__pycache__ --exclude='*.egg-info' --exclude=.pytest_cache \
  --anchored --exclude="$name/.git" --exclude="$name/.venv" \
  --exclude="$name/.runtime" --exclude="$name/assets" --exclude="$name/data" \
  --exclude="$name/runs" --exclude="$name/dist" --exclude="$name/build" \
  -czf "$target" -C "$(dirname "$repo")" "$name"
printf '%s\n' "$target"
