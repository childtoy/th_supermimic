#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${ROBOT:?Set ROBOT=k1 or ROBOT=g1}"
: "${MOTION_DIR:?Set MOTION_DIR to the converted corpus directory}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export CUDA_VISIBLE_DEVICES="${GPU_ID:-0}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
exec "${PYTHON:-python}" -u tools/train.py --robot "$ROBOT" \
  --motion-dir "$MOTION_DIR" --run-dir "${RUN_DIR:-runs/${ROBOT}_boxing}" \
  --backend "${BACKEND:-mjwarp}" "$@"
