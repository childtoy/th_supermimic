#!/usr/bin/env bash
# Run inside the existing training environment; paths can be overridden with env vars.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-python}"
CHECKPOINT="${CHECKPOINT:-runs/k1_traffic_control2/last.pt}"
MOTION_DIR="${MOTION_DIR:-data/k1/traffic_control2}"
ASSETS_DIR="${ASSETS_DIR:-assets/k1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
exec "$PYTHON" -u tools/web_play.py --robot "${ROBOT:-k1}" \
  --checkpoint "$CHECKPOINT" --motion-dir "$MOTION_DIR" --assets-dir "$ASSETS_DIR" \
  --host "${HOST:-127.0.0.1}" --port "${PORT:-6006}" "$@"
