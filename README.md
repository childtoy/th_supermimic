# th_supermimic

K1 / G1 모션 tracking용 SuperMimic 파생 구현. **UniLab, UniSim, mjlab,
Isaac Lab 없이 MuJoCo 또는 MuJoCo Warp에 직접 연결**한다.
환경 생성, 상태 읽기/쓰기, reset, 데이터 수집, 평가를 이 레포에서 관리한다.

- 알고리즘 출처: Uni-SuperMimic의 PyTorch world model / BPTT policy / critic 코드.
- 참고: SuperMimic-Reproduced의 world-model, FK, reset·학습 구조.
  해당 레포도 mjlab에 의존하므로 환경 구현은 그대로 가져오지 않았다.
- 로봇: K1 23관절 / G1 29관절. 로봇별 MJCF·게인·관절 순서만 분리하고 학습기는 공유한다.
  각 로봇의 정책과 world model은 독립 학습한다.
- 학습: simulator replay → world model 업데이트 → imagined rollout 역전파 →
  actor/critic 업데이트. PPO 학습기가 아니다.
- 원본 revision과 license는 `UPSTREAM.json`, `NOTICE`, `LICENSES/`에 보존한다.

## 설치

**PyTorch 2.5.1에서 실행 검증했다. 2.9 업그레이드는 필요하지 않다.**
기본 의존성은 PyTorch >=2.5, MuJoCo 3.8.1, NumPy, Rich다.
GPU 물리는 선택 extra인 MuJoCo Warp 3.8.1 / Warp 1.13.0을 사용한다.
TensorBoard와 영상 출력도 선택 의존성이다. Python 3.10 이상을 사용한다.

```bash
# PyTorch 2.5.1이 이미 있는 환경을 활성화한 뒤, 별도 폴더에 설치 가능
python -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install -e '.[warp,logging,dev]'
```

이 명령은 이미 설치된 torch가 >=2.5이면 그 버전을 사용할 수 있다.
CPU만 쓰려면 `python -m pip install -e '.[dev]'`로 설치한다.
새 환경에 PyTorch를 직접 설치할 때는 서버 드라이버에 맞는 CUDA build를 사용한다.
50번 서버의 driver 535에서 Warp 실행은 별도 smoke 검증이 필요하다.
기존 IsaacLab/Kimodo 환경의 패키지를 직접 덮어쓰지 않는다.

## 1. 로봇 자산

```bash
python tools/prepare_assets.py --rimkit-root /path/to/RIMKit --robot all
python tools/preflight.py --robot k1 --require-cuda
```

RIMKit의 원본 기구학과 mesh를 `assets/k1`, `assets/g1`에 복사해
position-controlled 학습 MJCF로 만든다. K1 게인/기본 자세/action scale은
기존 th_beyondmimic K1 설정, G1 motor 설정은 Uni-SuperMimic에서 가져왔다.
로봇 license/source 파일도 함께 복사한다. 생성 자산은 Git에서 제외한다.

## 2. RMAI corpus 변환

```bash
python tools/convert_rmai.py \
  --input-dir /workspace/primitive-motion-corpus/runs/primitive_corpus_05/robots \
  --output-dir data/k1/boxing --robot k1 --groups boxing

python tools/convert_rmai.py \
  --input-dir /workspace/primitive-motion-corpus/runs/primitive_corpus_05/robots \
  --output-dir data/g1/boxing --robot g1 --groups boxing
```

입력은 `<group>/<motion>/<robot>/final/robot_motion.npz`다.
로봇 ID, 관절 이름, qpos layout을 검사하고 순서를 맞춘다. 순서를 추측하지 않는다.
RIMKit root Z-up/wxyz → CSV Z-up/xyzw로 변환하고 shortest-path quaternion
보간으로 50 Hz에 맞춘다. FK와 속도는 모션 라이브러리에서 계산한다.

`--groups`를 생략하면 세 그룹을 모두 변환한다. 현재 corpus는 원본 포함
boxing 52개 / traffic_control1 93개 / traffic_control2 67개, 총 212개다.
manifest의 절대 CSV 경로 대신 clip 이름으로 파일을 찾으므로 변환 폴더를 서버로
옮겨 사용할 수 있다. G1/K1 및 서로 다른 데이터셋은 출력 폴더를 분리한다.

## 3. 학습

```bash
# 먼저 실제 GPU 물리 + 학습 검증: 8 env, 작은 네트워크, H=4, 3 updates
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
  --robot k1 --motion-dir data/k1/boxing --run-dir runs/k1_warp_smoke --smoke

# 본 학습: 한 learner GPU. 기본 4096 env / 12000 iteration / H=32
ROBOT=k1 MOTION_DIR=data/k1/boxing GPU_ID=0 RUN_DIR=runs/k1_boxing \
  bash scripts/train_server.sh

# G1도 같은 코드
ROBOT=g1 MOTION_DIR=data/g1/boxing GPU_ID=1 RUN_DIR=runs/g1_boxing \
  bash scripts/train_server.sh

# CPU simulator 검증 (Warp 불필요)
python tools/train.py --robot k1 --motion-dir data/k1/boxing \
  --run-dir runs/k1_cpu_smoke --backend mujoco --smoke
```

설정은 dataclass `SuperMimicConfig`와 JSON 파일로 관리한다. Hydra는 사용하지 않는다.
`--config file.json`으로 learner 설정을 읽거나 `--num-envs`, `--iterations`,
`--no-compile`을 지정한다. 세부 설정은 `algo.horizon=32`, `algo.amp=none`,
`algo.wm_hidden=[1024,1024]`처럼 끝에 붙일 수 있다. 해당 값은 JSON 형식으로 읽는다.
`--config-only`는 simulator를 만들지 않고 최종 설정을 출력한다.
`--no-randomization`은 reset noise / encoder offset / friction / torso CoM 변화를 끈다.

```bash
# last.pt의 네트워크 구조·optimizer를 복원하고 총 iteration 수 연장
python tools/train.py --robot k1 --motion-dir data/k1/boxing \
  --run-dir runs/k1_boxing --resume runs/k1_boxing/last.pt --iterations 20000

# checkpoint의 구조를 읽어 deterministic 전체 clip 평가
python tools/train.py --robot k1 --motion-dir data/k1/boxing \
  --run-dir runs/k1_boxing --eval --checkpoint runs/k1_boxing/last.pt
```

checkpoint에 robot/joint/model/motion hash 계약을 저장하여 잘못된 로봇이나
데이터로 resume/eval하는 것을 거부한다. 모션·모델·게인을 바꾸면 새 run으로 시작한다.
종료 직전의 물리 상태를 replay에 저장한 후 해당 env만 자동 reset한다.
CPU와 Warp가 같은 관측·reward·termination·sampler 코드를 공유한다.

출력: `last.pt`, `ckpt_*.pt`, `train_log.csv`, `run_config.json`, 선택적 TensorBoard.
작은 `--smoke` 모델을 본 학습의 초기 checkpoint로 사용하지 않는다.
원래 UniLab 기반 checkpoint는 새 환경 계약으로 resume하지 않는다.

## 검증과 제한

검증 환경: Python 3.10 / torch 2.5.1+cu124 / MuJoCo 3.8.1 / Warp 1.13.0 /
MuJoCo Warp 3.8.1, RTX 3080 Ti Laptop, driver 555.42.02.

- 두 로봇의 FK 위치/회전/선속도/각속도와 MuJoCo 수치 비교.
- analytic reward → 4-step world-model rollout → action까지 gradient 확인.
- 실제 K1 reference로 직접 GPU Warp의 collector / world model / actor / critic
  3-update 실행 및 checkpoint 저장 완료.
- G1은 정적 fixture로 CPU/GPU 경로를 검증한다. 실제 G1 corpus 성능은 미검증.
- PyTorch 2.5 CPU 학습, checkpoint 재개, K1 평가 경로 검증.
- K1 GPU에서 H=32/BF16 및 별도 torch.compile 실행 검증.
- 동일 초기 상태에서 CPU/Warp의 20ms 물리 결과 비교: K1/G1 모두
  qpos 최대 오차 < 5e-9, qvel 최대 오차 < 3e-7 (해당 초기 상태 기준).
- reset 전 terminal transition 보존과 다른 env 상태 유지 검사.
- CPU 학습 테스트는 외부 환경 framework import를 차단한 상태로 실행한다.

```bash
python -m pytest tests -q
```

GPU physics 비교는 다음으로 재실행할 수 있다.

```bash
TH_SUPERMIMIC_ROBOT=k1 python tests/check_backend_parity.py
TH_SUPERMIMIC_ROBOT=g1 python tests/check_backend_parity.py
```

GPU 검증은 자동 기본 테스트와 별도로 `--backend mjwarp --smoke`로 실행한다.
첫 실행의 Warp kernel JIT compilation은 시간이 걸릴 수 있다.
이는 실행 검증이며 모션 fidelity/정책 성능 결과가 아니다. 전체 212개 본 학습,
4096 env 성능, DDP, 실기 배포와 플레이 명령 전환은 아직 검증/구현 범위 밖이다.

## 서버 전송

```bash
bash scripts/package_server.sh /tmp/th_supermimic-server.tar.gz
```

소스·설정·테스트만 묶고 환경·모션·생성 자산·로그는 제외한다.
서버에서는 자산 준비 → 이미 서버에 있는 retarget 결과 변환 → GPU smoke → 본 학습
순서로 실행한다. 런타임에 원본 두 참고 레포는 필요하지 않다.

## 학습 붕괴 조사와 기존 checkpoint 비교

TensorBoard의 step reward, completed-episode return, imagined return은 서로
다른 표본/가중치의 통계이므로 같은 값으로 해석하지 않는다. 전용 policy도
학습 후 critic/gradient가 불안정해질 수 있으므로 마지막 checkpoint만 평가하지 않는다.

```bash
GPU_ID=0 bash scripts/evaluate_checkpoints.sh
```

각 run의 2,000회 checkpoint와 last.pt를 전체 clip의 첫 프레임부터 평가하고
`*_eval.json`에 결과를 저장한다. model weights는 변경하지 않는다.

2026-10-28 수정: 기존 collector에는 action ±20 제한이 있었지만 imagined
rollout와 evaluator에는 제한이 없었다. 이제 policy 출력에서 공통으로 제한한다.
기존 state_dict는 로드할 수 있으며 기존 checkpoint 평가에도 collector와 같은
제한이 적용된다. 이 수정이 Boxing 붕괴의 주원인 또는 해결책인지는 아직
장기 학습/비교 실험으로 검증하지 않았다. 기존 checkpoint를 보존한 뒤 평가한다.

## 서버 결과 수집

```bash
python tools/collect_training_audit.py
```

`exports/k1_training_audit.tar.gz`에 네 run의 2,000회/4,000회/last checkpoint,
학습 로그·설정·TensorBoard events, 50 Hz CSV/manifest, K1 MJCF/meshes와 현재
소스 snapshot을 묶는다. checkpoint는 로드/수정하지 않고 원본 바이트를 보존한다.
각 파일 SHA256과 수집 당시 commit·패키지 버전을 기록한다. 수집 당시 commit은
학습 당시 commit의 증거가 아니므로 별도로 해석한다. `--groups boxing`으로
선택 수집할 수 있다. 압축 전 용량을 먼저 출력하며 전체 수집은 수 GB일 수 있다.
입력이 없거나 수집 중 변경되면 실패한다. 학습이 종료된 상태에서 실행한다.

## Boxing 안정성 복구 파일럿

`--warm-start checkpoint.pt`는 policy/world-model과 그 normalizer만 불러온다.
critic, optimizer, replay, iteration은 새로 시작한다. 전체 상태를 복원하는
`--resume`와 다르며 두 옵션을 함께 사용할 수 없다.

`algo.critic_upper_bound=true`는 actor bootstrap과 TD target에 사용하는 value를
`max_step_reward / (1-gamma)` 이하로 제한한다. critic의 raw prediction을 직접
clamp해서 학습을 막지 않고, 제한된 target을 향해 raw prediction을 회귀한다.
기본값은 false이며 실험 옵션이다. `algo.no_critic=true`는 bootstrap을 제거한다.

```bash
MOTION_DIR=data/k1/boxing \
INIT_CHECKPOINT=runs/k1_boxing/ckpt_002000.pt \
GPU_ID=0 bash scripts/boxing_stability_pilot.sh
```

동일한 초기 policy/world-model에서 fresh-critic control, no-critic, bounded-critic을
비교한다. 기본은 256 env / 300 updates / 첫 50회 WM-only인 소규모 복구 실험이다.
원래 4096-env 학습을 그대로 재현하는 실험이 아니며, 중간 checkpoint에 없던 critic을
재생성하므로 원래 붕괴의 원인까지 확정하지는 못한다. 결과는 실제 52개 clip 평가로
판단하며, 짧은 파일럿 성공을 장기 안정성으로 일반화하지 않는다.

### 동일 로봇의 모션 집합 변경

`--warm-start`에 `--allow-motion-transfer`를 추가하면 같은 robot / joint order /
MJCF hash를 유지하면서 reference 모션 집합만 변경할 수 있다. 예를 들어 통합
정책에서 Boxing 전용 정책으로 초기화할 때 사용한다. robot/model이 다르면
이 옵션이 있어도 거부한다. `--resume`와 평가의 계약 검사는 완화하지 않는다.

`--eval-every 250`은 학습 도중 실제 simulator에서 전체 reference를 처음부터
평가한다. `evals/eval_*.json`, 최대 return의 `best_return.pt`, 실패 수가 가장 적은
`best_coverage.pt`를 각각 저장한다. 후자는 동률일 때 return을 비교한다.
평가 중 RNG 사용은 학습 RNG와 분리하며, 두 best checkpoint 모두 optimizer를
포함해 재개할 수 있다. 추가 평가 환경·정책을 위한 GPU 메모리와 시간이 필요하다.

### 보수적인 Boxing 전이·미세조정 후보

```bash
GPU_ID=0 bash scripts/boxing_transfer_recipe.sh
```

통합 K1 모델의 policy/world-model에서 시작해 Boxing만 전이 학습하고, 실제 평가에서
실패 수가 가장 적고 그중 return이 높은 모델을 골라 두 번째 미세조정을 시작한다.
두 단계 모두 policy/world-model을 이어받고 critic/optimizer는 새로 시작한다. 두 단계 모두 critic bootstrap 상한을 사용한다. 첫 단계 policy LR은 2e-4,
두 번째는 `algo.policy_lr_scale=0.25`로 policy만 5e-5로 낮춘다. WM/critic LR은 유지한다.
256 env, 300+500 updates인 작은 실험이며 장기·여러 seed·서버 재현 전에는 확정된
최적 설정으로 해석하지 않는다. 기존의 학습 모션과 물리 모델을 그대로 사용한다.

## 새 full sequence 재생 (재학습 없음)

RIMKit으로 리타게팅한 `robot_motion.npz`를 직접 입력한다. 사람 SOMA motion은 먼저
선택 로봇으로 리타게팅해야 한다. 학습에 사용한 assets와 checkpoint를 지정한다.

```bash
python tools/play.py --robot k1 \
  --checkpoint runs/k1_traffic_control2/last.pt --assets-dir assets/k1 \
  --input /path/to/full_sequences/*/k1/final/robot_motion.npz \
  --output-dir runs/traffic_v2_full_play --video traffic_v5_final_01
```

50Hz 변환본과 manifest, 전체 clip 평가 결과 `evaluation.json`, 선택 clip의
reference/rollout NPZ 및 MP4를 저장한다. `--video all`은 전부 렌더링한다.
영상의 실체는 policy, 파란 ghost는 reference이며 조기 종료 시 영상도 종료된다.
기존 결과를 덮어쓰지 않도록 새 output 디렉터리를 사용한다.
새 reference 집합만 허용하며 robot/joint/schema/model hash 불일치는 거부한다.
이 예외는 별도 inference 명령에만 적용하고 기존 학습·resume·평가 검사는 유지한다.
checkpoint와 원본 모션은 수정하지 않는다.
