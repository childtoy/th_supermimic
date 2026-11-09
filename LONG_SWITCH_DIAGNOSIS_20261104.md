# 긴 모션 사이 전환: 실제 실행과 WM 예측 진단

> 후속 독립 검증에서는 source 처음부터 연속 실행하고 direct-cut/zero-action 대조군을 추가했다. 아래의 “성공/완주”는 높이·기울기 기준 hard termination이 없었다는 의미이며, 고품질 추종이나 bridge 필요성을 뜻하지 않는다. 최종 비교는 [LONG_SWITCH_AUDIT_20261104.md](LONG_SWITCH_AUDIT_20261104.md)를 우선 참고한다.

2026-11-04. 추가 학습 없이 고정한 long-base policy/WM으로 교차 전이를 진단했다. **18개 전이 모두 연결 구간과 B 진입 후 1초를 수행했고, 이후 원래 긴 B로 handoff한 추가 검증에서도 18개 모두 끝까지 수행했다.** 다만 Boxing에서 나오는 전이들의 연결 품질은 낮았고 reference 경계 불일치가 함께 관측됐다.

## 실험 조건

- 모델: `runs/long_full_base_20261103/train/best_coverage.pt`, iteration 11,500. policy/WM 모두 고정, critic 점수 사용 안 함.
- source: 학습한 Boxing/Traffic V1/V2 full motion. 각 source를 실제 수행하면서 5/15/25초의 실제 상태·이전 action·관측 history를 cache했다.
- 후보: 6개 directed pair × 3개 source phase = **18개**. 목표 B는 처음부터 진입한다.
- 각 후보는 RMAI로 새로 생성한 1.5초 bridge다. 해당 교차 reference는 base 학습에 포함되지 않았다.
- RMAI에는 source 마지막 10프레임 context와 target 처음 10프레임 context를 제공했다. 생성한 bridge와 target 첫 약 2초를 기존 MuJoCo 3.6.0/RIMKit에서 독립적으로 retarget했다. 긴 A/B 전체를 후보마다 다시 retarget하지 않았다.
- 물리 평가/WM은 기존 MuJoCo 3.8.1/Warp 학습 환경에서 수행했다.
- 후보 source 끝 reference를 원래 A reference의 XY/yaw에 한 번 정렬했다. 높이·관절각은 강제로 맞추지 않았다.
- 각 분기 환경의 실제 초기 상태는 cache의 실제 q/qvel로 복원했으며, reference 자세로 reset한 것이 아니다. 초기 상태와 이전 action 일치를 assertion으로 확인했다. cache의 실제 history도 유지했다.
- 대조군 9개: 같은 실제 시작 상태에서 원래 학습한 A를 그대로 이어간다. 이 대조군도 분기 환경에서 같은 상태 복원 절차를 사용했다.
- 여기서 reference 생성/retarget은 오프라인이다. 새 온라인 생성기나 adaptive 학습을 구현한 것은 아니다.

## 먼저 본 지표

| 구간 | 실제 평균 보상 | WM 예측 평균 보상 | 평균 절대 점수 오차 | 순위 상관(Spearman) |
|---|---:|---:|---:|---:|
| 전환 직후 32 step(0.64초) | 0.8569 | 0.8571 | 0.00091 | 0.996 |
| Bridge 전체(1.5초) | 0.8416 | 0.8424 | 0.00157 | 1.000 |
| B 진입 후 첫 1초 | 0.8663 | 0.8650 | 0.00754 | 0.961 |

점수는 해당 구간 reward를 max-step reward와 구간 길이로 정규화한 평균이다. 전체 30초 clip return과 다른 지표다. 각 WM rollout은 전환 직전 실제 state/history에서 시작하지만 그 후에는 **policy+WM을 closed loop로 전개하며 실제 상태로 재초기화하지 않는다.** B 첫 1초 예측에는 앞선 bridge의 예측 오차도 누적된다.

18개 모두 bridge와 target 첫 1초에서 실제/예측 hard termination이 없었다. 관측한 실패가 없으므로 **실제 넘어짐을 WM이 사전에 잘 탐지하는지는 이 실험으로 입증할 수 없다.** 여기서 검증한 것은 추종 품질 점수의 예측이다.

“bridge 점수 ≥ 0.8”이라는 탐색적 기준에서는 WM과 실제 실행이 동일한 13개를 통과시켰고, 낮은 품질 후보 5개도 일치했다. 이 기준은 통계적으로 보정된 confidence/safety 기준이 아니다. 18개·한 모델의 작은 표본이며 순위 상관 1.0을 임의 phase/새 motion에 일반화하지 않는다.

### 실제 bridge 점수

| 방향 | A의 5초 | A의 15초 | A의 25초 |
|---|---:|---:|---:|
| Boxing → Traffic V1 | 0.832 | 0.636 | 0.643 |
| Boxing → Traffic V2 | 0.773 | 0.638 | 0.667 |
| Traffic V1 → Boxing | 0.858 | 0.929 | 0.841 |
| Traffic V1 → Traffic V2 | 0.964 | 0.938 | 0.958 |
| Traffic V2 → Boxing | 0.917 | 0.898 | 0.859 |
| Traffic V2 → Traffic V1 | 0.944 | 0.913 | 0.939 |

![점수·WM 예측·경계 불일치](runs/long_switch_diagnosis_20261104/diagnosis.png)

## WM은 어떻게 반응했나

알려진 reference continuation 대조군의 첫 1.5초 실제 점수는 0.9324, WM 점수는 0.9328이었다. 새 bridge는 품질이 낮아졌지만 WM이 이를 낙관적으로 무시하지 않고 비슷한 크기로 예측했다.

새 bridge에서 실제 trajectory의 one-step 관절 prediction RMS는 평균 약 **0.00094rad**, closed-loop 여러 step의 관절 prediction RMS는 약 **0.0090rad**였다. 같은 구간의 알려진 continuation 대조군 multi-step RMS는 약 **0.0037rad**였다. 즉 새 연결에서 오차는 커졌으나 현재 표본에서는 bridge 점수를 크게 오판하지 않았다.

이 모델은 별도의 불확실성 출력을 제공하지 않는다. 위 오차는 simulator와 사후 비교해 구한 값이며, 온라인에서 정답 없이 바로 알 수 있는 confidence가 아니다. WM이 reference를 직접 입력받는 것도 아니다. reference는 policy에 입력되고, WM은 그 policy의 action과 예측 state를 받아 전개된다.

## 낮은 품질의 Boxing 출구와 reference 경계

| Source | 새 bridge 평균 점수 | 알려진 A continuation 점수 | Source reference 관절 불연속 RMS 평균 |
|---|---:|---:|---:|
| Boxing | 0.698 | 0.832 | 1.157rad |
| Traffic V1 | 0.915 | 0.978 | 0.037rad |
| Traffic V2 | 0.912 | 0.988 | 0.044rad |

Boxing source의 새 local reference가 원래 full Boxing reference와 약 1.0~1.45rad RMS까지 다르다. 실제 로봇은 원래 A 상태를 유지하므로 goal만 바뀌는 큰 경계가 생긴다. 이것이 낮은 점수와 함께 나타났지만 **인과관계나 유일한 원인으로 확정한 것은 아니다.** Boxing의 base 추종 품질 자체도 Traffic보다 낮다.

목표 B의 시작점 역시 독립 retarget 때문에 canonical B와 크게 다를 수 있었다. 다만 target 약 1초 지점에서는 canonical reference와의 관절 차이가 약 0.017~0.046rad로 작아져 그 지점에서 원래 긴 B로 handoff하는 검증을 했다.

Boxing 25초는 원래 full sequence 내부의 bridge 구간이다. 15초는 primitive 구간이다. 따라서 낮은 품질 후보가 단순히 “원래 bridge 중에 명령했기 때문”이라고 설명되지는 않는다. 각 row에 `source_segment_kind`를 기록했다.

## 추가 검증: 원래 긴 B로 이어서 끝까지 수행

첫 진단은 local retarget B의 첫 1초까지만 포함한다. 그것만으로 원래 30초 B를 계속 수행한다고 주장할 수 없어 추가 검증했다.

1. 같은 cache에서 bridge와 local B 첫 1초를 다시 수행한다.
2. 원래 학습한 full B의 1초 frame을 현재 reference의 XY/yaw에 맞춘다.
3. **환경의 reference index만 바꾸고 물리 상태·이전 action·history는 초기화하지 않는다.**
4. 원래 full B의 마지막 frame까지 실제 MuJoCo에서 실행한다.

결과: **18/18 완료, 실패 0개.** Handoff 후 남은 B 구간 평균 normalized return은 **0.9144**였다. 이 수치에 앞선 bridge 품질을 섞어 숨기지 않도록 별도로 보고한다. 알려진 A continuation 대조군 9개도 모두 완료했다. 원본 결과는 `result_full_follow.json`이다.

이 검증은 한 trial에 A→B 전환 한 번과 local→canonical B handoff 한 번을 수행했다. A→B→C를 반복하는 interactive command chain, 추가 명령 연타, 매 제어 step의 전환, 실제 로봇 deployment는 아직 검증하지 않았다. Wall-clock 실시간 성능 측정도 아니다.

## 지금의 판단과 다음 실험

현재 18개 표본에서는 이미 frozen base로 연결과 full B 수행이 가능했다. 따라서 실패 수만 기준으로 추가 학습하면 이번 표본에서는 추가할 전이가 없다. 더 높은 bridge fidelity를 목표로 하면 0.8 미만 5개가 후보인데, 모두 Boxing 출구이며 큰 reference 경계 차이가 동반된다.

다음 순서가 적절하다.

1. 낮은 품질 후보의 영상과 경계를 확인하고, 원래 A/B 관절 경계를 유지하는 reference 구성과 비교한다. 먼저 dataset/retarget 문제를 분리한다.
2. 경계를 맞춘 뒤에도 WM과 실제 모두 낮은 품질인 phase만 추가 학습 후보로 삼는다.
3. 더 많은 phase와 primitive/내부 bridge 구간으로 평가를 넓힌다. 현재 5/15/25초의 표본으로 연속 구간 전체의 성공을 보장하지 않는다.
4. 같은 비용의 uniform 추가 학습과 비교하고 기존 full-motion fidelity도 다시 평가한다.
5. 단일 전환이 안정적이면 여러 번의 상태/history 유지 명령 전환과 실제 latency를 검증한다.

## 결과와 영상

출력 root: `runs/long_switch_diagnosis_20261104/`.

- `result.json`, `results.csv`: 첫 18개 전이 + 9개 known-continuation 대조군 결과.
- `result_full_follow.json`: 원래 full B로 끝까지 이어간 추가 검증.
- `traces.npz`, `result_full_follow_traces.npz`, `full_follow_traces.npz`: 기록한 상태/reference와 점수.
- `diagnosis.png`, `diagnosis.pdf`: 진단 그림.
- [낮은 품질 예: Boxing 15초 → Traffic V1](runs/long_switch_diagnosis_20261104/videos/boxing__to__traffic_control1__t01500.mp4)
- [좋은 예: Traffic V1 5초 → Traffic V2](runs/long_switch_diagnosis_20261104/videos/traffic_control1__to__traffic_control2__t00500.mp4)
- [Boxing → Traffic V1: bridge부터 원래 긴 B 끝까지](runs/long_switch_diagnosis_20261104/videos_full/boxing__to__traffic_control1__t01500.mp4)
- [Traffic V1 → V2: bridge부터 원래 긴 B 끝까지](runs/long_switch_diagnosis_20261104/videos_full/traffic_control1__to__traffic_control2__t00500.mp4)

짧은 영상은 약 2.54초의 command 이후 실행이다. Full-follow 영상은 bridge부터 원래 B 끝까지를 기록하며 A의 명령 전 prefix는 영상에 포함하지 않는다. 그 prefix는 source cache를 만들 때 실제 수행했다. 녹화 trace를 렌더링했고 영상을 생성하려고 재시뮬레이션하지 않았다.

도구: `tools/diagnose_long_switching.py`, `tools/plot_long_switch_diagnosis.py`, `tools/render_long_switch_follow.py`.
