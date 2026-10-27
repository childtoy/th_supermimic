# th_supermimic

RMAI로 만든 프리미티브/전환 모션을 K1 또는 G1에서 학습하기 위한
SuperMimic 파생 구현. 별도 로컬 Git 저장소이며, 원본 두 저장소는 수정하지 않는다.

- 학습 기반: `Uni-SuperMimic`의 world model, replay, BPTT policy/critic learner와 UniLab 환경.
- 대조 참고: `SuperMimic-Reproduced`의 world-model/FK/학습 구조. 두 구현을 섞어
  새 알고리즘으로 주장하지 않는다. 정확한 원본 revision은 `UPSTREAM.json`에 기록한다.
- K1 23관절 / G1 29관절을 **동일 코드에서 선택**한다. 각 로봇의 정책·world model은
  독립적으로 학습한다. K1 체크포인트를 G1에서 그대로 사용하지 않는다.
- PPO가 아닌 simulator replay → world model 업데이트 → imagined rollout 역전파 →
  actor/critic 업데이트 경로를 사용한다.

## 현재 검증 범위

- RIMKit의 K1/G1 MJCF에서 position-controlled 학습 모델 생성 및 MuJoCo 로드.
- 두 로봇의 differentiable FK 위치/회전/선속도/각속도와 MuJoCo 비교.
- 4-step world-model rollout에서 analytic reward를 거쳐 action까지 gradient 확인.
- 관절 이름 재정렬, quaternion shortest-path 보간, 50 Hz 모션 변환.
- 실제 K1 retarget 모션 1개로 CPU MuJoCo 3 iteration 학습, checkpoint 저장·평가 완료.
- G1은 정적 fixture로 CPU MuJoCo 3 iteration 학습·checkpoint 저장 완료.

이 결과는 실행 검증이며 학습 성능 또는 fidelity 결과가 아니다. 전체 212개 학습,
GPU mjwarp backend, 서버 드라이버 호환성, 실기 배포는 아직 검증하지 않았다.
현재 launcher는 한 프로세스/한 learner GPU를 사용한다. 여러 GPU를 쓰려면 우선
시나리오 또는 로봇별 독립 학습을 분배한다. DDP 학습을 구현했다고 가정하지 않는다.

## 환경

기존 `env_kimodo_inb`를 변경하지 말고 전용 Python 환경을 사용한다.
UniLab 1.5.1은 `torch>=2.9,<2.15`를 요구하므로 기존 서버의 torch 2.5.1 환경을
그대로 재사용할 수 없다. CUDA PyTorch 빌드는 서버 드라이버에 맞춰 설치하고
GPU runtime을 확인해야 한다. Python 3.12 환경을 권장한다.

```bash
# 별도 Python 3.12 환경을 활성화한 상태
python -m pip install -e '.[dev]'
```

주요 의존성은 `pyproject.toml`에 고정/범위 지정되어 있다. CPU backend에는
MuJoCo 3.11과 mjbatch-uni 0.2.5가 함께 필요하다. local 검증은 별도 `.runtime`
환경에서 torch 2.10.0, MuJoCo 3.11.0, UniLab 1.5.1, unisim-core 2.1.1,
mjbatch-uni 0.2.5를 사용했다. 이 환경 폴더는 전송하지 않는다.

## 1. 로봇 자산 준비

RIMKit 체크아웃에서 정확한 로봇과 mesh를 복사하여 portable MJCF를 만든다.
K1 게인·기본 자세·action scale은 기존 th_beyondmimic K1 설정에서 가져왔으며,
G1은 Uni-SuperMimic motor 설정을 사용한다. 기구학은 RIMKit 모델과 맞춘다.
원본 로봇 license/source 기록도 자산 폴더에 함께 보관한다.

```bash
python tools/prepare_assets.py --rimkit-root /path/to/RIMKit --robot all
python tools/preflight.py --robot k1 --require-cuda
python tools/preflight.py --robot g1 --require-cuda
```

생성 경로: `assets/k1/`, `assets/g1/`. meshes와 생성 자산은 Git에서 제외한다.

## 2. 서버의 RMAI retarget 결과 변환

입력은 `robots/<group>/<motion>/<robot>/final/robot_motion.npz` 구조다.
`qpos`, `joint_names`, `fps`를 읽고 로봇/관절 이름을 검증한다. 관절 이름이 없으면
순서를 추측하지 않는다. root는 Z-up, quaternion wxyz인 RIMKit 규약을 사용한다.
출력 CSV는 root xyz + quaternion xyzw + 선택한 로봇의 관절 순서이며 50 Hz다.
FK와 body/joint velocity는 학습 모션 라이브러리가 50 Hz에서 계산한다.

```bash
python tools/convert_rmai.py \
  --input-dir /workspace/primitive-motion-corpus/runs/primitive_corpus_05/robots \
  --output-dir data/k1/boxing --robot k1 --groups boxing

# 동일 원본 corpus에 G1 retarget 결과도 있는 경우
python tools/convert_rmai.py \
  --input-dir /workspace/primitive-motion-corpus/runs/primitive_corpus_05/robots \
  --output-dir data/g1/boxing --robot g1 --groups boxing
```

`--groups`를 생략하면 boxing / traffic_control1 / traffic_control2를 모두 변환한다.
기본 corpus는 boxing 52개, traffic 1 93개, traffic 2 67개로 원본 포함 총 212개다.
그룹별·로봇별로 다른 출력 폴더를 사용해 입력을 명확히 분리한다.

## 3. 실행 검증 후 학습

K1/G1 선택은 **프로세스 시작 시** 고정한다. launcher가 환경 모듈 import 전에
선택하여 MDP, action, replay, policy, world model의 크기를 함께 맞춘다.

```bash
# GPU backend의 짧은 학습 검증: 8 env, 3 iteration, 작은 network
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
  --robot k1 --motion-dir data/k1/boxing --run-dir runs/k1_smoke --smoke

# CPU simulator 검증; learner도 CPU에서 실행
CUDA_VISIBLE_DEVICES='' python tools/train.py \
  --robot k1 --motion-dir data/k1/boxing --run-dir runs/k1_cpu_smoke \
  --backend mujoco --smoke

# 실제 학습: 기본값 4096 env, 12000 iteration, H=32
ROBOT=k1 MOTION_DIR=data/k1/boxing GPU_ID=0 RUN_DIR=runs/k1_boxing \
  bash scripts/train_server.sh

# G1도 같은 진입점
ROBOT=g1 MOTION_DIR=data/g1/boxing GPU_ID=1 RUN_DIR=runs/g1_boxing \
  bash scripts/train_server.sh
```

Hydra 옵션을 끝에 붙여 조절한다. 예: `algo.num_envs=2048 algo.iterations=20000`.
`--smoke` checkpoint는 작은 network를 쓰므로 실제 학습의 초기 checkpoint로
사용하지 않는다. 실제 학습은 별도 run 폴더에서 시작한다.

```bash
# 동일 구조의 학습 재개
python tools/train.py --robot k1 --motion-dir data/k1/boxing \
  --run-dir runs/k1_boxing algo.resume=true algo.load_run=runs/k1_boxing \
  algo.iterations=20000

# 학습한 설정/구조와 같은 옵션으로 deterministic 전체 clip 평가
python tools/train.py --robot k1 --motion-dir data/k1/boxing \
  --run-dir runs/k1_boxing training.play_only=true algo.load_run=runs/k1_boxing
```

checkpoint는 `ckpt_*.pt`, `last.pt`; TensorBoard와 `metrics.csv`, 원본 config snapshot도
run 폴더에 저장한다. 학습 success는 원본 reference fidelity와 물리 rollout 평가로
별도로 판단한다. 런타임 명령 전환/edge 선택은 이 레포 초기 범위에 포함하지 않는다.

## 테스트

```bash
python -m pytest tests -q
```

준비된 assets가 필요하다. simulator 학습 테스트는 training runtime이 없으면 skip한다.
G1/K1 FK와 BPTT는 독립 프로세스로 검사하여 한 프로세스에서 robot 설정이 섞이지 않는다.

## 서버로 이동

`.git`, `.venv`, `.runtime`, `assets`, `data`, `runs`, cache를 제외한 소스 패키지를
만든다. 압축 해제 후 전용 학습 환경 설치 → 자산 준비 → 이미 서버에 있는 retarget
결과 변환 → GPU smoke → 본 학습 순서다.

```bash
bash scripts/package_server.sh /tmp/th_supermimic-server.tar.gz
```

license는 Apache-2.0이며 Uni-SuperMimic의 NOTICE와 원 저작권 고지를 보존한다.
