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
