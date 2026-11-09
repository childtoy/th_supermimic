#!/usr/bin/env bash
# Build three full references, retarget once per full clip, then train ONE policy.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-python}"
RMAI_ROOT="${RMAI_ROOT:-../RMAI}"
CORPUS_ROOT="${CORPUS_ROOT:-/workspace/primitive-motion-corpus}"
KIMODO_ROOT="${KIMODO_ROOT:-../kimodo}"
OUTPUT_ROOT="${OUTPUT_ROOT:-runs/long_full_base_$(TZ=Asia/Seoul date +%Y%m%d_%H%M%S)}"
ITERATIONS="${ITERATIONS:-12000}"
NUM_ENVS="${NUM_ENVS:-4096}"
MIN_SECONDS="${MIN_SECONDS:-30}"
GPU_ID="${GPU_ID:-0}"
RETARGET_PYTHON="${RETARGET_PYTHON:-$PYTHON}"
PREPARED_ROBOTS_DIR="${PREPARED_ROBOTS_DIR:-}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export TH_SUPERMIMIC_ROBOT=k1
export TH_SUPERMIMIC_ASSETS="$(pwd)/assets/k1"
if [[ -n "$PREPARED_ROBOTS_DIR" ]]; then
  [[ -f "$OUTPUT_ROOT/reference_build/manifest.json" && ! -e "$OUTPUT_ROOT/train/run_config.json" && ! -e "$OUTPUT_ROOT/data/manifest.json" ]] || {
    echo 'Prepared mode needs the generated manifest and a fresh conversion/training stage' >&2; exit 2;
  }
  robots_dir="$PREPARED_ROBOTS_DIR"
  "$PYTHON" -c 'from pathlib import Path; import sys; root=Path(sys.argv[1]); files=list(root.glob("*/long_*/k1/final/robot_motion.npz")); assert len(files)==3, f"Expected 3 full references, got {len(files)}"' "$robots_dir"
else
  [[ ! -e "$OUTPUT_ROOT" ]] || { echo "Refusing to overwrite $OUTPUT_ROOT" >&2; exit 2; }
  # Fail before expensive generation/DMR if the retarget interpreter is incompatible.
  "$RETARGET_PYTHON" -c 'import mujoco; assert mujoco.__version__ == "3.6.0", "RIMKit retarget requires a separate MuJoCo 3.6.0 interpreter; keep the training environment unchanged"'
  mkdir -p "$OUTPUT_ROOT"
  robots_dir="$OUTPUT_ROOT/robots"
fi
stage='generate'
status_file="$OUTPUT_ROOT/status.txt"
trap 'code=$?; if ((code)); then printf "failed stage=%s exit=%s\n" "$stage" "$code" > "$status_file"; fi' EXIT
printf 'running stage=%s\n' "$stage" > "$status_file"
if [[ -z "$PREPARED_ROBOTS_DIR" ]]; then
"$PYTHON" -u tools/build_long_full_references.py --rmai-root "$RMAI_ROOT" \
  --corpus-root "$CORPUS_ROOT" --kimodo-root "$KIMODO_ROOT" \
  --output-dir "$OUTPUT_ROOT/reference_build" --min-seconds "$MIN_SECONDS"
stage='retarget'; printf 'running stage=%s\n' "$stage" > "$status_file"
"$RETARGET_PYTHON" -u "$RMAI_ROOT/scripts/batch_retarget_corpus.py" \
  --input-dir "$OUTPUT_ROOT/reference_build/soma77" --pattern '*/*.npz' \
  --output-dir "$OUTPUT_ROOT/robots" --robots k1 --video-robots '' \
  --workers 3 --preserve-parent --final-only --fail-on-error
fi
stage='convert'; printf 'running stage=%s\n' "$stage" > "$status_file"
"$PYTHON" tools/convert_rmai.py --input-dir "$robots_dir" \
  --output-dir "$OUTPUT_ROOT/data" --robot k1 --groups boxing traffic_control1 traffic_control2
stage='train'; printf 'running stage=%s iterations=%s envs=%s\n' "$stage" "$ITERATIONS" "$NUM_ENVS" > "$status_file"
"$PYTHON" -u tools/train.py --robot k1 --assets-dir assets/k1 --motion-dir "$OUTPUT_ROOT/data" \
  --run-dir "$OUTPUT_ROOT/train" --backend mjwarp --num-envs "$NUM_ENVS" \
  --iterations "$ITERATIONS" --eval-every 500 \
  algo.seed=0 algo.buffer_steps=256 algo.pi_batch="$NUM_ENVS" algo.wm_batch="$((NUM_ENVS/2))" \
  algo.wm_only_iters=500 algo.critic_upper_bound=true algo.no_critic=false \
  algo.lr=0.0002 algo.policy_lr_scale=0.25 algo.action_limit=20.0 algo.horizon=32 \
  algo.ckpt_every=1000 algo.log_every=50 algo.log_backend=no_print
stage='video'; printf 'running stage=%s\n' "$stage" > "$status_file"
for group in boxing traffic_control1 traffic_control2; do
  "$PYTHON" -m th_supermimic.tasks.motion_tracking.humanoid.supermimic.render \
    --ckpt "$OUTPUT_ROOT/train/best_coverage.pt" --backend mjwarp \
    --clip "${group}__long_${group}" --seconds 120 \
    --output "$OUTPUT_ROOT/videos/$group.mp4" --scene assets/k1/scene.xml \
    --width 480 --height 288 --fps 10 --global-ghost
done
printf 'completed iterations=%s\n' "$ITERATIONS" > "$status_file"
