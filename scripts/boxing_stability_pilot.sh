#!/usr/bin/env bash
# Matched small-scale recovery arms; not a reproduction of the original full run.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${MOTION_DIR:?Set MOTION_DIR to the converted Boxing references}"
: "${INIT_CHECKPOINT:?Set INIT_CHECKPOINT to the Boxing 2000 checkpoint}"
ASSETS_DIR="${ASSETS_DIR:-assets/k1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-runs/boxing_stability_pilot}"
PYTHON="${PYTHON:-python}"
NUM_ENVS="${NUM_ENVS:-256}"
ITERATIONS="${ITERATIONS:-300}"
export CUDA_VISIBLE_DEVICES="${GPU_ID:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
mkdir -p "$OUTPUT_ROOT"
for variant in ${VARIANTS:-baseline no_critic bounded_critic}; do
  extra=()
  case "$variant" in
    baseline) ;;
    no_critic) extra+=(algo.no_critic=true) ;;
    bounded_critic) extra+=(algo.critic_upper_bound=true) ;;
    *) echo "Unknown arm: $variant" >&2; exit 2 ;;
  esac
  "$PYTHON" -u tools/train.py --robot k1 --motion-dir "$MOTION_DIR" \
    --assets-dir "$ASSETS_DIR" --backend mjwarp --run-dir "$OUTPUT_ROOT/$variant" \
    --warm-start "$INIT_CHECKPOINT" --num-envs "$NUM_ENVS" --iterations "$ITERATIONS" \
    --no-compile algo.buffer_steps=256 algo.pi_batch="$NUM_ENVS" \
    algo.wm_batch="$((NUM_ENVS/2))" algo.wm_only_iters=50 \
    algo.ckpt_every=100 algo.log_every=10 algo.log_backend=no_print \
    "${extra[@]}" > "$OUTPUT_ROOT/$variant.log" 2>&1
  for checkpoint in ckpt_000100.pt ckpt_000200.pt last.pt; do
    [[ -f "$OUTPUT_ROOT/$variant/$checkpoint" ]] || continue
    "$PYTHON" -u tools/train.py --robot k1 --motion-dir "$MOTION_DIR" \
      --assets-dir "$ASSETS_DIR" --backend mjwarp --run-dir "$OUTPUT_ROOT/$variant" \
      --eval --checkpoint "$OUTPUT_ROOT/$variant/$checkpoint" \
      > "$OUTPUT_ROOT/${variant}_${checkpoint%.pt}_eval.log" 2>&1
  done
done
