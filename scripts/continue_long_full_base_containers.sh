#!/usr/bin/env bash
# Run on the Docker HOST. Existing 3.6 retarget + existing 3.8 Warp training.
set -euo pipefail
cd "$(dirname "$0")/.."
OUTPUT_ROOT="${OUTPUT_ROOT:-runs/long_full_base_20261103}"
RETARGET_CONTAINER="${RETARGET_CONTAINER:-th_u2204cuda128_kimodo}"
TRAIN_CONTAINER="${TRAIN_CONTAINER:-th_u2204cuda128_kimodo2}"
CONTAINER_REPO="${CONTAINER_REPO:-/workspace/workspace/childtoy/th_supermimic}"
PYTHON_BIN="${PYTHON_BIN:-/root/anaconda3/envs/env_kimodo_inb/bin/python}"
stage=preflight
failure() {
  code=$?
  if ((code)); then
    docker exec -w "$CONTAINER_REPO" "$TRAIN_CONTAINER" bash -c \
      'printf "failed stage=%s exit=%s\n" "$2" "$3" > "$1/status.txt"' bash "$OUTPUT_ROOT" "$stage" "$code" || true
  fi
}
trap failure EXIT
docker exec -w "$CONTAINER_REPO" "$RETARGET_CONTAINER" "$PYTHON_BIN" -c \
  'import mujoco,sys; from pathlib import Path; assert mujoco.__version__=="3.6.0"; root=Path(sys.argv[1]); assert len(list(root.glob("reference_build/soma77/*/*.npz")))==3; assert not (root/"robots_mj360").exists(); print("Retarget preflight: MuJoCo",mujoco.__version__)' "$OUTPUT_ROOT"
docker exec -w "$CONTAINER_REPO" "$TRAIN_CONTAINER" "$PYTHON_BIN" -c \
  'import torch,mujoco,mujoco_warp; assert torch.cuda.is_available(); assert mujoco.__version__=="3.8.1"; print("Training preflight:",torch.__version__,mujoco.__version__)'
stage=retarget_mj360
docker exec -w "$CONTAINER_REPO" -e OMP_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 -e MKL_NUM_THREADS=1 \
  "$RETARGET_CONTAINER" bash -c '
    set -e
    printf "running stage=retarget_mj360\n" > "$1/status.txt"
    exec "$2" -u ../RMAI/scripts/batch_retarget_corpus.py \
      --input-dir "$1/reference_build/soma77" --pattern "*/*.npz" \
      --output-dir "$1/robots_mj360" --robots k1 --video-robots "" \
      --workers 3 --preserve-parent --final-only --fail-on-error \
      > "$1/retarget_mj360.log" 2>&1
  ' bash "$OUTPUT_ROOT" "$PYTHON_BIN"
stage=train
docker exec -w "$CONTAINER_REPO" -e OUTPUT_ROOT="$OUTPUT_ROOT" \
  -e PREPARED_ROBOTS_DIR="$OUTPUT_ROOT/robots_mj360" -e PYTHON="$PYTHON_BIN" \
  -e ITERATIONS="${ITERATIONS:-12000}" -e NUM_ENVS="${NUM_ENVS:-4096}" -e GPU_ID="${GPU_ID:-0}" \
  "$TRAIN_CONTAINER" bash -c 'exec bash scripts/train_long_full_base.sh > "$OUTPUT_ROOT/continue_training.log" 2>&1'
