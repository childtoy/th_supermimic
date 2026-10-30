# th_supermimic 작업 인수인계

작성일: 2026-10-30 (Asia/Seoul)
기준 코드: `2076761` (렌더링/제어 분리 수정). 이 문서 이후 코드 변경 시 관련 내용도 갱신한다.

## 1. 지금 하려는 일

RMAI로 생성한 primitive 간 transition을 로봇으로 retarget한 reference bank에 대해
하나의 SuperMimic policy를 학습하고, 실행 중 사용자의 모션 명령에 따라 reference를
선택해 같은 policy로 계속 추종하는 시스템이다.

현재 가장 잘 검증된 것은 **K1 / Traffic Control V2**다. 웹에서 1–4 버튼으로 명령을
내리고 실제 policy + MuJoCo 물리를 실행한다. 동영상을 이어 붙여 재생하는 UI가 아니다.
웹 재생은 재학습하지 않는다. G1도 코드상 지원하지만 아래 실험 수치는 K1에 대한 것이다.

저장소: https://github.com/childtoy/th_supermimic.git

## 2. 다른 컴퓨터에서 시작하기

```bash
git clone https://github.com/childtoy/th_supermimic.git
cd th_supermimic
```

이미 받았으면 해당 repository에서 `git pull`한다. **Git에는 checkpoint, assets,
reference 데이터가 없다.** 아래 세 디렉터리는 기존 학습 서버에서 그대로 사용하거나
새 실행 컴퓨터로 별도로 복사해야 한다.

```text
runs/k1_traffic_control2/last.pt
data/k1/traffic_control2/manifest.json + *.csv
assets/k1/metadata.json + robot.xml + scene.xml + meshes/ 등 전체 assets
```

모션 CSV와 robot.xml이 학습 당시와 정확히 맞아야 한다. 실행기는 checkpoint의
robot / joint order / model hash / motion hash 계약을 검사한다. 기존 환경을 합칠 때
이 검사를 무조건 우회하거나 robot.xml을 새로 만들어 덮어쓰지 않는다.

기존 서버의 환경은 `env_kimodo_inb`, PyTorch `2.5.1+cu124`였다. MuJoCo Warp 실행에는
CUDA GPU가 필요하다. UniLab/UniSim/IsaacLab 런타임 의존성은 없다. 현재 dependency는
`pyproject.toml`을 기준으로 하고, 이 작업을 위해 PyTorch를 무조건 올리지 않는다.

기존 실행 환경에서 웹 관련 패키지 추가:

```bash
pip install -e '.[web]'
```

새 환경을 준비할 때의 Warp optional dependency:

```bash
pip install -e '.[warp,web,video]'
```

## 3. 서버·도커·포트: 사용자가 실제 쓰는 구성

- 서버: 50번 서버, 주소 `163.152.162.50`
- repository: 도커 안 `/workspace/workspace/childtoy/th_supermimic`
- **도커 내부 포트 6006 → 호스트 외부 포트 약 20031**로 사용자 설명을 받았다.
- 외부 포트는 실제 Docker 매핑으로 확인할 것. 내부 포트를 20031로 바꾸는 게 아니다.
- 기존 TensorBoard가 내부 6006을 사용하면 먼저 해당 TensorBoard를 종료해야 한다.
  실행 스크립트는 다른 프로세스를 자동으로 종료하지 않는다.

학습에 쓴 도커·conda 환경 안에서:

```bash
cd /workspace/workspace/childtoy/th_supermimic
git pull
pip install -e '.[web]'
HOST=0.0.0.0 PORT=6006 bash scripts/run_web_play.sh
```

브라우저: `http://163.152.162.50:20031`

서버를 켜 둔 터미널은 실행 중 그대로 둔다. 상태 조회는 다른 터미널 없이도 브라우저
새 탭에서 `http://163.152.162.50:20031/api/state`를 열면 된다.

업데이트 시 서버 터미널에서 Ctrl+C → `git pull` → 위 실행 명령 → 브라우저 새로고침.

경로가 다른 경우:

```bash
CHECKPOINT=/absolute/path/last.pt \
MOTION_DIR=/absolute/path/traffic_control2 \
ASSETS_DIR=/absolute/path/k1 \
HOST=0.0.0.0 PORT=6006 bash scripts/run_web_play.sh
```

스크립트 기본 robot은 K1, backend는 mjwarp다. `ROBOT=g1` 또는 CLI 옵션은 그 로봇의
정확한 assets/checkpoint/reference를 함께 준비했을 때만 사용한다.

## 4. 웹의 현재 동작

| 키 | 모션 |
|---|---|
| 1 | come_forward — 앞으로 오세요 |
| 2 | direct_left — 왼쪽으로 |
| 3 | firm_stop — 정지 |
| 4 | slow_down — 천천히 |
| R | Reset |

- 시작은 일시정지. 모션 버튼을 누르면 실행한다.
- primitive 재생 중 다른 모션 명령이 오면 가까운 source phase의 pair를 선택하고
  다음 제어 step에 reference를 교체한다. 다음 0.5초 grid까지 기다리는 방식이 아니다.
- 전이 도중 추가 명령은 **마지막 명령만 대기**시켜 target primitive 진입 직후 적용한다.
  전이 도중 전이를 즉시 끊는 기능은 아직 아니다.
- 전환 시 실제 로봇 상태, 이전 action, policy 관측 history를 유지한다.
- Reset만 실제 상태/history/대기 명령/시각/기록을 초기화하고 시작 자세에서 일시정지한다.
- reference 끝에 도달하면 **물리 시뮬레이션을 일시정지**한다. 마지막 프레임 화면은 유지된다.
  이것은 물리를 계속 돌리면서 policy가 끝 자세를 안정적으로 버틴다는 의미가 아니다.
  이전 실험에서는 마지막 reference만 계속 공급했을 때 넘어지는 경우가 있었다.
- 같은 모션 버튼은 새 A→A 전이나 자동 반복을 생성하지 않는다.
- 화면에는 모션 길이·진행률·남은 시간과 transition 남은 시간을 구분해 표시한다.
  시간은 simulation/reference 시간이다. 실제 처리 Hz가 낮으면 wall time은 더 길어진다.

## 5. 학습 데이터와 root 정렬을 정확히 설명하는 법

생성 corpus는 시나리오 내부의 서로 다른 primitive 방향 조합에 대해:

```text
A[:a_stop] + RMAI로 생성한 bridge + B[0:]
```

- A의 끝에서 0.5초씩 줄여 source phase를 만든다. 필요한 context 10프레임을 남긴다.
- B는 처음부터 진입한다 (`target-mode entry`).
- 한 clip에는 A→B 한 번의 전이만 있다. 원본 primitive도 따로 학습 데이터에 포함한다.
- A→A, 모든 연속 시점, B 중간 진입, 긴 다중 전이 시퀀스가 전부 학습된 것은 아니다.
- K1 corpus: 전체 212개 = Boxing 52 + Traffic V1 93 + Traffic V2 67.
- Boxing 52개는 원본 4개 + 전이 48개다. 원본이 1.6~2.6초로 짧아 개수가 작다.

명령 실행기는 bank의 source phase가 가장 가까운 A→B를 고른 뒤 전이 직전 frame으로
reference를 바꾼다. 기존 reference root를 기준으로 새 clip 전체에 같은 XY 이동과
같은 yaw 회전을 적용한다. 높이와 관절각은 유지한다. 실제 로봇을 teleport하지 않는다.
이 정렬은 매 전환 시 한 번 수행하며 실제 로봇 위치에 매 프레임 reference를 붙이지 않는다.

Policy 입력에 상대 좌표를 쓴다고 해서 global reference 불연속이 무해한 것은 아니다.
Root 정렬은 절대 위치/방향 차이를 처리할 뿐 관절 자세/속도 차이는 제거하지 않는다.
그 차이는 추종 controller가 흡수해야 한다. 모든 preceding 이력에 대한 성공을 입증한 것은 아니다.

데모에서 선택된 bridge는 모두 1.5초였다. 실행기가 강제로 1.5초 지연을 추가한 게 아니다.
명령 수신 → reference 교체 → 1.5초 연결 동작 → target primitive 진입이다.
RMAI 생성 설정에는 0.5/1.0/1.5초 후보가 있지만, bank에 실제로 어떤 후보가 저장됐는지
확인하지 않고 짧은 bridge를 언제든 고를 수 있다고 설명하면 안 된다.

현재 학습 설정: `hist=10`, `goal_frames=10`, `horizon=32`, `wm_window=8`.
32는 policy 입력 길이가 아닌 학습용 imagined rollout 길이이다. reset 시 history는
초기 관측 반복으로 채우고, reference 미래가 부족하면 마지막 frame을 반복한다.
imagination 시작점은 clip 끝까지 horizon이 남은 곳에서만 선택한다.

## 6. 검증 결과와 주장 범위

### 기존 Traffic V2 full sequence

RMAI의 완성된 긴 K1 시퀀스 10개(각 약 17.6초)를 50Hz로 변환해 기존 Traffic V2
`last.pt`로 재학습 없이 평가했다. 10/10 clip 끝까지 수행, 평균 normalized return 약 0.961.
이 실험은 완성된 긴 reference를 각각 넣은 평가다.

### 온라인 명령 10번 데모

별도로 명령마다 pair를 선택해 교체하는 데모를 실행했다. 10번 전환 모두 완료,
약 24.7초, mean normalized step reward 약 0.971. 실제 상태/history 리셋 없이 실행했다.
명령은 각 primitive 진입 후 0.55~1.15초 사이에 생성했고, 전이 중 연타는 이 데모에서
시험하지 않았다. 위 full sequence 평가와 서로 다른 실험이다.

### 단일 A→B cutback 비교

동일 `firm_stop → come_forward`를 A 끝에서 0.5/1.0/1.5초 전으로 각각 잘라 전환한
영상 3개를 생성했다. 모두 완료. bridge 길이는 모두 1.5초. 원본 Traffic V2가 약
2.4~2.9초라 끝에서 3초·5초 전 비교는 원본 그대로는 불가능하다.

### Boxing

기존 Boxing 단독 2000회: return 약 0.478, 20/52 실패.
기존 Boxing 단독 12000회: return 약 -0.009, 52/52 실패.
통합 policy/WM에서 Boxing으로 전이하고 낮은 actor LR로 fine-tune한 후보:
return 약 0.884, 0/52 실패. 이는 256 env·단일 seed recovery pilot이다.
critic 과대추정은 관측했지만 collapse의 유일한 원인으로 확정하지 않았다.
좋은 후보가 나왔다고 최적 학습법이나 원래 4096-env 장기 안정성이 입증된 건 아니다.

## 7. 현재 가장 중요한 미완료 확인: 서버 렌더링 성능

사용자가 보내준 서버 상태:

```text
backend: mjwarp
compute_ms: 11.28
render_ms: 382.63
simulated_steps_per_wall_second: 2.53
```

기존 구현은 제어와 렌더링이 같은 worker 루프여서 매 step이 느린 렌더링을 기다렸다.
`2076761`에서 수정한 내용:

- 별도 렌더링 thread가 GL context와 JPEG 인코딩을 소유한다.
- 제어 thread는 pose만 넘기고 진행한다. backlog를 만들지 않고 최신 pose만 유지한다.
- 시작/재개 시 측정 구간을 정리해 일시정지 시간이 Hz 계산에 섞이지 않게 한다.
- 기본 화면 해상도 640×384. API에 `render_hz`, `renderer_name`, `renderer_decoupled` 추가.
- 로컬에서 인코딩에 380ms의 인위적 지연을 넣어 검증: 화면 385.7ms / 2.6fps인 동안
  제어 약 49.9Hz. 일시정지/재개 후 측정도 확인했다.

**최신 수정 후 실제 서버가 50Hz로 회복됐는지는 아직 사용자 확인을 받지 못했다.**
실제 서버 그래픽 드라이버/자원 경쟁은 위 sleep 실험과 같지 않다. 서버의 383ms 원인은
소프트웨어 렌더링 등의 가능성을 의심했지만 확정하지 않았다. `renderer_name`을 먼저
확인할 것. 렌더링 분리는 제어 병목을 제거하는 조치이지 느린 렌더러 자체를 고친 건 아니다.

다음 작업 순서:

1. 서버에서 `git pull` 후 재시작. API의 `renderer_decoupled=true` 확인.
2. 움직이는 동안 `compute_ms`, 실제 Hz, `render_ms`, `render_hz`, `renderer_name` 수집.
3. 제어가 회복됐어도 화면이 느리면 실제 EGL/그래픽 경로를 확인한다.
4. 필요하면 `--render-width 480 --render-height 288`로 비교. 확인 전 원인을 단정하지 않는다.
5. 이후 전이 중 명령 처리 방식, 실제 자세·속도 기반 후보 선택, 빠른 bridge와 종료 자세
   유지/반복 재생 등을 별도 실험한다. 현재 기능처럼 설명하지 않는다.

## 8. 파일 지도

| 파일 | 역할 |
|---|---|
| `tools/web_play.py` | HTTP API, policy/physics worker, 별도 renderer worker |
| `web/index.html` | 1–4 버튼, 키보드, Reset, timer, 상태·성능 표시 |
| `scripts/run_web_play.sh` | 기존 환경/경로로 실행, HOST/PORT 등 override |
| `src/th_supermimic/data/command_bank.py` | bank timing 파싱, phase 선택, root 정렬, 표시용 시간 |
| `tools/command_demo.py` | 자동 명령 전환 및 지정 cutback 영상 생성 |
| `tools/play.py` | 새로운 retarget full sequence를 변환·평가·영상 출력 |
| `src/th_supermimic/data/playback.py` | 새 reference 평가 시 물리 계약 유지 검사 |
| `tools/inspect_training_audit.py` | checkpoint 상세 추종·critic/WM 진단 |
| `scripts/boxing_transfer_recipe.sh` | Boxing 전이/미세조정 pilot 후보 |
| `tests/test_command_bank.py` | bank timing, 정렬, 상태 비변경, 표시 시간 검사 |

최근 코드 변경 관련 단위 테스트 9개가 통과했고, 렌더링 분리 수정 후 관련 4개를
다시 실행해 통과했다. 느린 렌더링 API 실험도 수행했다. 전체 서버 통합 검증을
통과한 것으로 과장하지 않는다. 문서만 바꿀 때 학습이나 GPU 실험을 다시 돌릴 필요는 없다.

## 9. 원래 작업 컴퓨터에만 있는 분석 자료 (Git에 없음)

```text
/home/rilab/workspace/childtoy/th_supermimic
/home/rilab/workspace/childtoy/th_supermimic_audit_20261028/training_audit
/home/rilab/workspace/childtoy/th_supermimic_audit_20261028/results
/home/rilab/workspace/childtoy/th_supermimic_audit_20261028/pilots
/home/rilab/workspace/childtoy/RMAI/runs/main_figure_traffic_v5_final_k1
```

학습 audit 원본 압축: `/home/rilab/workspace/childtoy/k1_training_audit.tar.gz`.
원본 audit의 checkpoint/CSV/asset은 보존하고 recovery pilot은 별도 디렉터리에 만들었다.
Boxing의 검증한 후보는 `pilots/fidelity_finetune/best_coverage.pt` (fine-tune iteration 400).

이 repository의 ignored `runs/`에는 다음 결과가 있다:

```text
traffic_v2_full_play/             # evaluation.json, 01/10 비교 영상
traffic_v2_commands10/            # commands.json, commands.mp4, report.md
traffic_v2_cutbacks/cutback_0.5/  # commands.mp4 등
traffic_v2_cutbacks/cutback_1.0/
traffic_v2_cutbacks/cutback_1.5/
web_qa.json                      # 명령 대기/덮어쓰기/끝 일시정지 실험
web_slow_renderer_qa.json         # 느린 렌더링 분리 실험
```

이 절의 절대 경로는 원래 컴퓨터의 참고 위치다. 새 컴퓨터에 존재한다고 가정하지 않는다.
