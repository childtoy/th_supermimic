#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-python}"
: "${OUTPUT_ROOT:?Set a fresh experiment root}"
ITERATIONS="${ITERATIONS:-12000}"
NUM_ENVS="${NUM_ENVS:-4096}"
export CUDA_VISIBLE_DEVICES="${GPU_ID:-0}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MUJOCO_GL=egl
export TH_SUPERMIMIC_ROBOT=k1 TH_SUPERMIMIC_ASSETS="$(pwd)/assets/k1"
[[ -f "$OUTPUT_ROOT/data/manifest.json" && ! -e "$OUTPUT_ROOT/train/run_config.json" ]]
stage=smoke
trap 'code=$?; if ((code)); then printf "failed stage=%s exit=%s\n" "$stage" "$code" > "$OUTPUT_ROOT/status.txt"; fi' EXIT
printf 'running stage=%s\n' "$stage" > "$OUTPUT_ROOT/status.txt"
"$PYTHON" -u tools/train.py --robot k1 --assets-dir assets/k1 --motion-dir "$OUTPUT_ROOT/data" --run-dir "$OUTPUT_ROOT/smoke" --backend mjwarp --smoke
stage=train
printf 'running stage=%s iterations=%s envs=%s\n' "$stage" "$ITERATIONS" "$NUM_ENVS" > "$OUTPUT_ROOT/status.txt"
"$PYTHON" -u tools/train.py --robot k1 --assets-dir assets/k1 --motion-dir "$OUTPUT_ROOT/data" --run-dir "$OUTPUT_ROOT/train" --backend mjwarp --num-envs "$NUM_ENVS" --iterations "$ITERATIONS" --eval-every 500 \
 algo.seed=0 algo.buffer_steps=256 algo.pi_batch="$NUM_ENVS" algo.wm_batch="$((NUM_ENVS/2))" \
 algo.wm_only_iters=500 algo.critic_upper_bound=true algo.no_critic=false \
 algo.lr=0.0002 algo.policy_lr_scale=0.25 algo.action_limit=20.0 algo.horizon=32 \
 algo.ckpt_every=1000 algo.log_every=50 algo.log_backend=no_print
stage=video
printf 'running stage=%s\n' "$stage" > "$OUTPUT_ROOT/status.txt"
mapfile -t clips < <("$PYTHON" - "$OUTPUT_ROOT/data/manifest.json" <<'PY'
import json,sys
for x in json.load(open(sys.argv[1]))['clips']:
 if x['kind']=='primitive' or x['transition']['cutback_seconds']==2.5:print(x['name'])
PY
)
for clip in "${clips[@]}"; do
 "$PYTHON" -m th_supermimic.tasks.motion_tracking.humanoid.supermimic.render --ckpt "$OUTPUT_ROOT/train/best_coverage.pt" --backend mjwarp --clip "$clip" --seconds 30 --output "$OUTPUT_ROOT/videos/$clip.mp4" --scene assets/k1/scene.xml --width 480 --height 288 --fps 10 --global-ghost
done
printf 'completed iterations=%s\n' "$ITERATIONS" > "$OUTPUT_ROOT/status.txt"
