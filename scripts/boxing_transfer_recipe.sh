#!/usr/bin/env bash
# Experimental two-stage recovery recipe. Preserves original run directories.
set -euo pipefail
cd "$(dirname "$0")/.."
MOTION_DIR="${MOTION_DIR:-data/k1/boxing}"
ASSETS_DIR="${ASSETS_DIR:-assets/k1}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-runs/k1_all/last.pt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-runs/k1_boxing_transfer_recipe}"
PYTHON="${PYTHON:-python}"
export CUDA_VISIBLE_DEVICES="${GPU_ID:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
shared=(--robot k1 --motion-dir "$MOTION_DIR" --assets-dir "$ASSETS_DIR" --backend mjwarp
  --num-envs 256 --no-compile --eval-every 100 algo.buffer_steps=256 algo.pi_batch=256
  algo.wm_batch=128 algo.wm_only_iters=50 algo.ckpt_every=100 algo.log_every=10
  algo.critic_upper_bound=true algo.no_critic=false algo.lr=0.0002)
"$PYTHON" -u tools/train.py "${shared[@]}" \
  --run-dir "$OUTPUT_ROOT/transfer" --warm-start "$INIT_CHECKPOINT" \
  --allow-motion-transfer --iterations "${TRANSFER_ITERS:-300}" algo.policy_lr_scale=1.0
# Default matches the tested fresh-critic fine-tuning pilot.
"$PYTHON" -u tools/train.py "${shared[@]}" \
  --run-dir "$OUTPUT_ROOT/finetune" --warm-start "$OUTPUT_ROOT/transfer/best_coverage.pt" \
  --iterations "${FINETUNE_ITERS:-500}" algo.policy_lr_scale=0.25
