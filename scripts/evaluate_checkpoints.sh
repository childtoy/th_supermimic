#!/usr/bin/env bash
# Compare early and final checkpoints without overwriting model weights.
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="${GPU_ID:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
for group in all boxing traffic_control1 traffic_control2; do
  run="runs/k1_$group"
  for ckpt in ckpt_002000.pt last.pt; do
    if [[ ! -f "$run/$ckpt" ]]; then
      echo "Missing $run/$ckpt; skipping" >&2
      continue
    fi
    echo "Evaluating $group / $ckpt"
    "${PYTHON:-python}" -u tools/train.py --robot k1 --motion-dir "data/k1/$group" \
      --backend "${BACKEND:-mjwarp}" --run-dir "$run" --eval --checkpoint "$run/$ckpt"
  done
done
