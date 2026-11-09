# 세 full motion 장기 학습: 실행 기록

## 2026-11-04 확인: 학습·평가·영상 생성 완료

`status.txt`는 `completed iterations=12000`이다. 학습 시간은 기록상 3.43041시간, 물리 interaction은 123,142,144 steps다. 현재 GPU가 쉬는 이유는 학습이 종료됐기 때문이다.

학습 clip 완주율/return으로 선택한 best checkpoint는 11,500회이며 평균 return 0.933315, 실패 0/3이다.

| Full motion | 길이 | Best normalized return | 실패 |
|---|---:|---:|---|
| Boxing | 30.30초 | 0.836610 | 없음 |
| Traffic V1 | 33.00초 | 0.980942 | 없음 |
| Traffic V2 | 30.66초(50Hz 변환 후) | 0.982394 | 없음 |

12,000회 최종 checkpoint도 실패 0/3, 평균 return 0.932485다. `train/best_coverage.pt`와 `train/best_return.pt`는 11,500회 모델을 선택했다. 비교 영상 세 개는 `videos/{boxing,traffic_control1,traffic_control2}.mp4`에 저장됐다.

이 결과는 학습한 세 full reference의 추종 결과이며, cross-full 명령 전환이나 WM 기반 adaptive 추가 학습을 검증한 것은 아니다.

2026-11-03. 사용자가 세 full motion을 먼저 충분히 오래 학습한 뒤, 그 base에서 미학습 transition을 평가하도록 요청했다. 이전 3,000회 sparse pilot과 별개의 run이다.

## 데이터

기존 완성 sequence 파일은 현재 서버에서 찾지 못했다. 원래 학습 primitive의 human reference와 기존 RMAI alignment/inbetweening checkpoint는 존재하여, 새 full sequence 세 개를 생성했다.

| Full reference | Human reference 길이 |
|---|---:|
| Boxing | 30.30초 |
| Traffic V1 | 33.00초 |
| Traffic V2 | 30.67초 |

기존 short primitive들을 여러 번 사용하고 각 경계에 RMAI bridge를 생성했다. 단순한 frame 보간/재생 속도 변경이나 새로 촬영한 장기 모션이 아니다. 같은 primitive의 반복이 있으므로 모션 길이 증가와 동작 다양성 증가를 혼동하면 안 된다. 정확한 primitive 순서, segment/bridge 경계, source/weight hash, placement는 `reference_build/manifest.json`에 저장한다.

각 full sequence를 K1으로 한 번씩 retarget하고 기존 converter로 50Hz CSV를 만든다. 세 CSV를 **하나의 unified policy에 함께 학습**한다. Boxing/Traffic 사이를 연결하는 cross-full transition은 이 데이터에 넣지 않는다. 다만 각 full reference 내부에는 RMAI로 만든 연결이 포함된다. 따라서 이 base를 “어떤 bridge도 학습하지 않은 모델”이라고 부르면 안 된다.

## 학습 설정

- 출력 root: `runs/long_full_base_20261103/`.
- fresh policy/WM/critic/optimizer. 기존 sparse 또는 recovery checkpoint로 초기화하지 않는다.
- 4,096 env, **12,000 iterations**, GPU 0.
- WM-only 500회, imagined horizon 32, observation history 10, goal frames 10.
- WM/critic LR 2e-4, actor LR 5e-5, critic bootstrap 상한 적용, action bound 20.
- compile on, BF16. 기존 PyTorch/CUDA 버전을 유지한다.
- 매 500회 full reference 세 개를 처음부터 끝까지 평가한다. 평가 clip은 30초 이상이며 training imagination horizon 0.64초와 다른 개념이다.
- `best_coverage.pt`는 실패 수 우선, 그다음 normalized return으로 선택한다. `best_return.pt`, `last.pt`, 주기 checkpoint도 보존한다.
- 학습이 끝나면 best-coverage policy의 실제 physics 비교 영상 세 개를 자동 생성한다.

12,000회는 우선 실행하는 장기 학습 budget이다. 이 숫자만으로 충분히 학습됐다고 결론 내리지 않고, 완주율·추종 품질·최근 평가 곡선을 확인한다. 성능이 계속 오르거나 충분하지 않으면 이후 resume/추가 학습을 판단한다. 현재 script가 자동으로 12,000회를 넘어 학습하는 것은 아니다.

## 확인 경로

### 환경 호환성 수정

첫 pipeline은 retarget에도 학습용 MuJoCo 3.8.1을 사용해 RIMKit Stage 3의 버전 검사에서 실패했다. GPU 학습은 시작되지 않았다. 생성된 human reference는 보존하고 기존 환경을 분리해 재개했다. 패키지는 설치/변경하지 않았다.

- retarget: `th_u2204cuda128_kimodo`, `env_kimodo_inb`, MuJoCo **3.6.0**.
- 학습: `th_u2204cuda128_kimodo2`, `env_kimodo_inb`, PyTorch **2.5.1+cu124**, MuJoCo **3.8.1**, 기존 Warp 환경.
- 3.6.0 환경에서 실제 source의 12프레임을 사용해 retarget 전 단계를 통과하고 유한한 K1 qpos를 출력하는 smoke test에 성공했다.
- 긴 retarget을 `robots_mj360/`에 다시 생성하며 완료되면 기존 학습 환경에서 변환/12,000회 학습을 이어간다.
- host controller: `scripts/continue_long_full_base_containers.sh`, 실행 세션은 retarget과 학습을 순서대로 수행한다.
- 재개 후 retarget 로그: `retarget_mj360.log`, 재개 후 변환/학습 로그: `continue_training.log` (모두 output root 내부).
- 원래 `runs/long_full_base_20261103.log`는 실패한 첫 시도의 기록이다. 재개 이후 상태는 아래 status와 새 로그를 확인한다.

```bash
cat runs/long_full_base_20261103/status.txt
tail -f runs/long_full_base_20261103/retarget_mj360.log
# stage=train으로 바뀐 뒤:
tail -f runs/long_full_base_20261103/continue_training.log
tail runs/long_full_base_20261103/train/train_log.csv
```

| 경로 | 내용 |
|---|---|
| `reference_build/manifest.json` | human full sequence provenance와 bridge/primitive 경계 |
| `robots/` | K1 retarget 결과 |
| `data/manifest.json` | 50Hz full reference 세 개 |
| `train/run_config.json`, `train/args.json` | 실제 실행 설정 |
| `train/evals/eval_*.json` | 전체 clip 평가 |
| `train/best_coverage.pt` | 완주율 우선 모델 |
| `videos/` | 학습 종료 후 자동 생성할 비교 영상 |

위 표의 경로는 output root 기준이다. stage/status와 실제 파일로 진행을 확인하며, 아직 생성되지 않은 checkpoint나 영상을 완료한 것으로 취급하지 않는다.

## 다음 단계

1. base가 세 full motion을 충분히 추종하는지 확인한다. 특히 Boxing의 기존 reference/retarget 경계 문제가 새 full에서도 나타나는지 검사한다.
2. base checkpoint를 고정하고 A/B/C full motion 사이의 대표 전이를 추가 학습 없이 평가한다.
3. 실제 결과와 policy+WM 예상 bridge 점수를 비교한다. 기존 short-motion calibration 정확도가 이 데이터에도 유지된다고 가정하지 않는다.
4. 필요한 전이만 sparse/adaptive하게 추가 학습하는 방법을 동일 budget의 uniform 대조군과 비교한다.

현재 준비/학습 pipeline에는 cross-full transition 생성이나 adaptive 학습을 섞지 않았다.

## 실행 코드

- `tools/build_long_full_references.py`: 기존 RMAI 모델로 30초 이상 sequence 세 개 생성.
- `scripts/train_long_full_base.sh`: 생성 → K1 retarget → 50Hz 변환 → 장기 학습 → 영상.
- 기존 원본 reference, assets, checkpoint를 덮어쓰지 않는 새 run이다.
- 학습 Docker: `th_u2204cuda128_kimodo2`, 환경: `env_kimodo_inb`. retarget은 위의 기존 3.6.0 컨테이너를 사용한다.
