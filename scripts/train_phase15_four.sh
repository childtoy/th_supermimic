#!/usr/bin/env bash
# Four fresh runs: all / boxing / traffic_control1 / traffic_control2.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-python}"
MODE="${MODE:-sparse}" # sparse, dense, both (four simultaneous jobs per round)
GPU_IDS="${GPU_IDS:-0 1 2 3}"
OUTPUT_ROOT="${OUTPUT_ROOT:-runs/phase15_$(TZ=Asia/Seoul date +%Y%m%d_%H%M%S)}"
ASSETS_DIR="${ASSETS_DIR:-assets/k1}"
DATA_ROOT="${DATA_ROOT:-data/k1}"
ITERATIONS="${ITERATIONS:-3000}"
NUM_ENVS="${NUM_ENVS:-256}"
WM_ONLY_ITERS="${WM_ONLY_ITERS:-500}"
EVAL_EVERY="${EVAL_EVERY:-200}"
SEED="${SEED:-0}"
DRY_RUN="${DRY_RUN:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
read -r -a gpus <<< "$GPU_IDS"
groups=(all boxing traffic_control1 traffic_control2)
[[ ${#gpus[@]} == 4 ]] || { echo 'GPU_IDS must contain four distinct GPU indices' >&2; exit 2; }
case "$MODE" in sparse|dense) modes=("$MODE");; both) modes=(sparse dense);; *) echo 'MODE must be sparse, dense or both' >&2; exit 2;; esac
for index in 0 1 2 3; do
  [[ ${gpus[index]} =~ ^[0-9]+$ ]] || { echo 'GPU indices must be integers' >&2; exit 2; }
  for ((other=0; other<index; other++)); do
    [[ ${gpus[index]} != "${gpus[other]}" ]] || { echo 'Use distinct GPUs' >&2; exit 2; }
  done
done
for value in "$ITERATIONS" "$NUM_ENVS" "$EVAL_EVERY" "$SEED" "$WM_ONLY_ITERS"; do
  [[ $value =~ ^[0-9]+$ ]] || { echo 'Numeric settings must be nonnegative integers' >&2; exit 2; }
done
((NUM_ENVS >= 8 && ITERATIONS > WM_ONLY_ITERS && EVAL_EVERY > 0)) || {
  echo 'Require NUM_ENVS>=8, ITERATIONS>WM_ONLY_ITERS, EVAL_EVERY>0' >&2; exit 2;
}
[[ ! -e "$OUTPUT_ROOT" ]] || { echo "Refusing to overwrite $OUTPUT_ROOT" >&2; exit 2; }
split_root="$OUTPUT_ROOT/data"
shared=(--robot k1 --assets-dir "$ASSETS_DIR" --backend mjwarp --num-envs "$NUM_ENVS"
  --iterations "$ITERATIONS" --eval-every "$EVAL_EVERY" --no-compile
  algo.seed="$SEED" algo.buffer_steps=256 algo.pi_batch="$NUM_ENVS"
  algo.wm_batch="$((NUM_ENVS/2))" algo.wm_only_iters="$WM_ONLY_ITERS"
  algo.critic_upper_bound=true algo.no_critic=false algo.lr=0.0002
  algo.policy_lr_scale=0.25 algo.action_limit=20.0 algo.horizon=32
  algo.ckpt_every="$EVAL_EVERY" algo.log_every=20 algo.log_backend=no_print)
if [[ "$DRY_RUN" == 1 ]]; then
  printf 'Prepare splits: '
  printf '%q ' "$PYTHON" tools/split_transition_phases.py --data-root "$DATA_ROOT" --groups "${groups[@]}" --output-dir "$split_root"
  printf '\n'
  for mode in "${modes[@]}"; do
    subset=train; [[ "$mode" == dense ]] && subset=dense
    for index in 0 1 2 3; do
      group="${groups[index]}"
      printf '[%s/%s] CUDA_VISIBLE_DEVICES=%s ' "$mode" "$group" "${gpus[index]}"
      printf '%q ' "$PYTHON" -u tools/train.py "${shared[@]}" --motion-dir "$split_root/$group/$subset" --run-dir "$OUTPUT_ROOT/$mode/k1_$group"
      printf '\n'
    done
  done
  echo 'After each run: evaluate best_coverage.pt and last.pt on interpolation/extrapolation; no heldout checkpoint selection.'
  exit 0
fi
# Inspect the chosen GPUs in the existing environment; install/upgrade nothing.
printf -v gpu_mask '%s,' "${gpus[@]}"
gpu_mask="${gpu_mask%,}"
CUDA_VISIBLE_DEVICES="$gpu_mask" "$PYTHON" -c 'import torch, mujoco, mujoco_warp, warp; assert torch.cuda.is_available() and torch.cuda.device_count()==4, "Four CUDA GPUs required"; print("torch",torch.__version__,"mujoco",mujoco.__version__,"warp",warp.__version__)'
"$PYTHON" tools/split_transition_phases.py --data-root "$DATA_ROOT" --groups "${groups[@]}" --output-dir "$split_root"
mkdir -p "$OUTPUT_ROOT/logs"
pids=()
cleanup() {
  trap - INT TERM
  for pid in "${pids[@]}"; do kill "$pid" 2>/dev/null || true; done
  for pid in "${pids[@]}"; do wait "$pid" 2>/dev/null || true; done
  exit 130
}
trap cleanup INT TERM
worker() (
  local mode="$1" group="$2" gpu="$3" subset=train child=''
  [[ "$mode" == dense ]] && subset=dense
  export CUDA_VISIBLE_DEVICES="$gpu"
  trap 'if [[ -n "$child" ]]; then kill "$child" 2>/dev/null || true; wait "$child" 2>/dev/null || true; fi' EXIT
  trap 'exit 130' INT TERM
  local run="$OUTPUT_ROOT/$mode/k1_$group" log="$OUTPUT_ROOT/logs/${mode}_${group}.log"
  "$PYTHON" -u tools/train.py "${shared[@]}" --motion-dir "$split_root/$group/$subset" --run-dir "$run" >"$log" 2>&1 &
  child=$!; wait "$child"; child=''
  for filename in best_coverage.pt last.pt; do
    local extra=()
    [[ "$mode" == sparse ]] && extra+=(--expect-heldout)
    "$PYTHON" -u tools/evaluate_phase_holdout.py --checkpoint "$run/$filename" \
      --data-dir "$split_root/$group" --assets-dir "$ASSETS_DIR" \
      --output-dir "$run/heldout/${filename%.pt}" "${extra[@]}" >>"$log" 2>&1 &
    child=$!; wait "$child"; child=''
  done
  printf 'Completed %s/%s on GPU %s\n' "$mode" "$group" "$gpu"
)
for mode in "${modes[@]}"; do
  pids=()
  for index in 0 1 2 3; do
    worker "$mode" "${groups[index]}" "${gpus[index]}" &
    pids+=("$!")
  done
  failures=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then failures=$((failures+1)); fi
  done
  pids=()
  ((failures == 0)) || { echo "$failures jobs failed; inspect $OUTPUT_ROOT/logs" >&2; exit 1; }
done
"$PYTHON" tools/summarize_phase_experiment.py "$OUTPUT_ROOT"
printf 'Results: %s\n' "$OUTPUT_ROOT"
