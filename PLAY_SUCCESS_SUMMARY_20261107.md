# 학습 구성부터 Play 결과까지: 실험 진행 정리

2026-11-07. 어떤 데이터를 어떻게 학습했고, 어떤 모션에 적용해서 무엇을 확인했는지를 실험 순서로 정리한다. 아래 모델들은 하나의 모델을 계속 이어 학습한 것이 아니라, 초기 학습·Boxing 회복·sparse fresh 학습·long-motion fresh 학습으로 구분된다.

## 1. 처음에는 짧은 primitive와 생성한 전이를 함께 학습했다

목표는 기존 SuperMimic 계열의 world-model 기반 tracking 학습기를 RMAI가 생성한 전이 reference에 적용해, K1이 여러 모션과 그 사이 연결을 수행하게 하는 것이었다.

데이터 준비는 **원본 human primitive → RMAI로 여러 source phase의 A→B 연결 생성 → SOMA export/RIMKit K1 retarget → 50Hz robot reference 변환** 순서였다. 실행할 때마다 새 bridge를 생성한 것이 아니라 오프라인 bank를 준비했다. 기존 0.5초 phase 간격의 reference bank에는 primitive와 A→bridge→B clip이 함께 들어 있었다.

| 적용 모션군 | 기본 primitive | primitive 수 | 학습 reference 수(전이 포함) |
|---|---|---:|---:|
| Boxing | straight_punch, double_uppercut, duck, weave | 4 | 52 |
| Traffic V1 | come_forward, direct_left, stop_a, turn_around, turn_right_90 | 5 | 93 |
| Traffic V2 | come_forward, direct_left, firm_stop, slow_down | 4 | 67 |
| 통합 | 위 세 모션군 전체 | 13 | 212 |

Boxing/V1/V2 각각 하나의 policy를 학습하고, 세 그룹을 묶은 통합 policy도 별도로 학습했다. primitive마다 controller를 만든 것이 아니라, 각 run의 policy가 그 run의 모든 reference를 추종하는 구조다. 초기 run은 12,000 iteration 예산, 50Hz 물리 제어, history 10, goal frames 10, imagined horizon 32로 학습했다.

학습은 simulator에서 state/action/next-state를 수집해 world model을 학습하고, 그 모델의 imagined rollout을 통해 policy/critic을 업데이트하는 방식이다. play에서는 관측 history와 reference를 policy에 넣고, 출력된 관절 목표를 MuJoCo의 position actuator로 실행했다. WM을 매 step 온라인 탐색기로 사용한 것은 아니다.

## 2. 초기 play에서는 Traffic은 잘 됐고 Boxing은 회복이 필요했다

12,000회 모델을 원본 primitive와 전이 clip 전체에 적용한 평가에서 Traffic V1은 평균 return 0.9624·실패 1/93, V2는 0.9610·실패 2/67이었다. 이후 Traffic V2의 실제 연속 명령 play에서도 10회 명령을 24.70초 동안 완료했고, 실행 평균 점수 0.9706·joint RMSE 0.0828rad를 기록했다. **학습한 bank에서 여러 primitive로 연속 전환하는 결과가 확보된 단계다.**

Boxing은 초기 단독 학습이 불안정해 마지막 12,000회 모델에서 52/52 실패했다. 통합 policy/WM을 출발점으로 critic/optimizer를 새로 만들고 bootstrap 상한·학습률 등을 조정해 Boxing bank를 재학습했다. 전이 학습 300회 뒤 미세조정을 수행했고, 미세조정 300회 모델이 전체 52개 clip에서 return 0.8815·실패 0개였다. 이 모델과 실제 pose/velocity 기반 전이 선택을 사용한 play는 seed 42/7에서 각각 10회 연속 명령을 완료했다. 다만 joint RMSE가 약 0.442rad여서 Traffic 수준의 정밀 추종이라고 볼 수는 없다.

## 3. 그다음에는 모든 phase를 학습할 필요가 있는지 확인했다

기존 0.5초 간격 bank에서 매 세 번째 phase, 즉 1.5초 간격만 선택했다. Boxing/V1/V2/통합 네 policy를 각각 새로 3,000회 학습했다. 학습 전이는 각각 21/36/24/81개였고 기본 primitive는 유지했다.

학습에서 빠진 중간 phase를 평가했을 때 네 모델 모두 해당 평가 clip을 완주했다. 중간 phase 평균 return은 Boxing 0.589, V1 0.860, V2 0.890, 통합 0.737이었다. **같은 primitive/pair 내 일부 phase를 제외해도 clip 실행이 가능한 사례**를 확인했다. 이것은 연속 interactive play나 브리지의 자연스러움을 검증한 결과와 구분한다.

## 4. 이후에는 긴 모션 세 개를 함께 학습했다

짧은 primitive를 반복하고 내부 경계마다 RMAI bridge를 넣어 Boxing 30.30초, Traffic V1 33.00초, V2 약 30.67초의 full human reference를 새로 만들었다. 각 sequence를 통째로 K1으로 retarget한 후, 세 full reference를 **하나의 새 policy/WM에 함께 12,000회** 학습했다. 그룹 사이 cross-full bridge는 넣지 않았지만 각 full 내부 bridge는 학습에 포함됐다.

11,500회 선택 모델의 play에서 세 모션 모두 완주했다. 전체 clip return은 Boxing 0.8366, V1 0.9809, V2 0.9824였다. **세 긴 학습 모션을 하나의 controller로 수행하는 base**가 확보됐다. 짧은 동작 반복으로 구성했으므로 길이 증가를 새로운 동작 다양성 증가와 동일시하지 않는다.

## 5. 최근에는 긴 모션 사이의 미학습 전환을 적용해 봤다

위 long-base의 가중치를 고정하고, 6개 cross-full 방향 × source phase 9개에 새 RMAI bridge를 주었다. 추가 전이 학습은 하지 않았다. 동일 policy에 B reference를 바로 주는 direct-cut도 비교했다.

둘 다 54/54 완주했지만 영상 재검토에서 Boxing이 bridge reference와 다른 중간 동작을 수행하는 사례가 확인됐다. Traffic에는 reference 자체가 정체하는 구간도 있었다. 따라서 이 단계는 **B로 넘어가서 버티는 능력의 확인**이며, **새 연결 동작의 자연스러운 재현 성공**으로 정리하지 않는다.

현재 성공한 play를 소개할 때 중심에 둘 것은 **기존 bank의 Traffic 연속 명령 실행, 회복한 Boxing의 연속 실행, 세 long-motion의 unified tracking**이다. sparse phase는 일반화의 보조 평가이고, 미학습 cross-full bridge 재현은 남은 문제다.

---

# 세부 결과와 영상 근거

2026-11-07 기준. 저장된 평가 JSON과 영상 파일을 확인하여 정리했다. 이번 정리는 새 play 실행이나 전체 영상의 재심사가 아니다. 모든 결과는 K1 MuJoCo 물리 시뮬레이션이며 하드웨어 deployment 검증은 아니다. 아래 실험은 서로 다른 checkpoint와 데이터로 수행했다.

## 1. 학습한 primitive/전이 bank로 연속 명령 실행

RMAI로 미리 생성하고 K1으로 retarget한 primitive/transition reference bank를 SuperMimic으로 학습했다. 실행 중 명령을 받아 bank에서 연결을 선택하고 실제 상태와 policy history를 유지하며 다음 모션으로 넘어갔다. 온라인 RMAI 생성은 아니다.

### Traffic V2: 가장 명확한 기존 연속 명령 결과

- 10회 연속 명령 완료, 24.70초.
- 실행 평균 normalized step reward 0.9706, 평균 joint RMSE 0.0828rad.
- phase 기준으로 bank에서 연결 선택.
- [영상](runs/review_commands_20261101/traffic2/commands.mp4), [원본 결과](runs/review_commands_20261101/traffic2/commands.json).
- 확인한 범위: 저장된 한 명령 시퀀스에서 안정적인 tracking과 연속 전환. 임의 명령/bridge 도중 재명령/하드웨어 일반화까지 검증한 것은 아니다.

### Boxing: 학습 회복 후 연속 명령 수행 가능

초기 모델은 불안정했다. 통합 policy/WM warm-start, fresh critic, bootstrap 상한, actor LR 조정 등을 적용해 학습을 회복했다. 그 뒤 실제 pose/velocity 기반으로 같은 pair의 연결 후보를 선택했다.

- 모델: `runs/boxing_recovery_20261101/finetune/ckpt_000300.pt`.
- 52개 clip 평가 실패 0/52, 평균 normalized return 0.8815.
- 명령 seed 42/7 각각 10회 완료, 실행 시간 24.68/23.60초.
- 평균 step reward 0.7123/0.7134, 평균 joint RMSE 0.4421/0.4417rad.
- [seed 42 영상](runs/review_commands_20261101/boxing_finetune300_pose_seed42/commands.mp4), [결과](runs/review_commands_20261101/boxing_finetune300_pose_seed42/commands.json).
- [seed 7 결과](runs/review_commands_20261101/boxing_finetune300_pose_seed7/commands.json). seed 7의 영상 파일은 현재 확인되지 않아 성공 영상 목록에 넣지 않았다.
- 확인한 범위: 기존 학습 bank를 사용한 연속 명령의 완료와 초기 모델 대비 회복. 큰 추종/retarget 경계 오차가 남으므로 완벽한 연결 재현·자연스러움으로 부르지 않는다. 여러 학습 설정을 함께 바꿔 어느 하나의 효과로 귀속할 수 없다.

## 2. 30초 이상의 세 full motion을 하나의 policy로 재생

기존 짧은 primitive를 반복하고 내부에 RMAI bridge를 넣어 세 full reference를 만들었다. 각 full sequence를 K1으로 retarget한 후, 세 개를 하나의 새 policy/WM에 함께 12,000 iteration 학습했다. 위 Boxing 회복 모델과 별개다.

선택 모델: `runs/long_full_base_20261103/train/best_coverage.pt`, iteration 11,500.

| 모션 | 길이 | normalized return | 완주 |
|---|---:|---:|---|
| Boxing | 30.30초 | 0.8366 | 완료 |
| Traffic V1 | 33.00초 | 0.9809 | 완료 |
| Traffic V2 | 30.66초 | 0.9824 | 완료 |

- [Boxing 영상](runs/long_full_base_20261103/videos/boxing.mp4)
- [Traffic V1 영상](runs/long_full_base_20261103/videos/traffic_control1.mp4)
- [Traffic V2 영상](runs/long_full_base_20261103/videos/traffic_control2.mp4)
- [평가 JSON](runs/long_full_base_20261103/train/evals/eval_011500.json)

확인한 범위: 학습한 세 긴 reference를 하나의 controller로 끝까지 수행한다. 특히 Traffic 두 개는 전체 clip 추종 점수가 높다. Boxing은 상대적으로 오차가 크다. 길이를 늘린 구성에는 짧은 primitive 반복이 있으므로 새 dynamics 다양성이나 임의 cross-full 전환까지 검증한 것은 아니다.

## 3. Sparse phase 평가: 완주 일반화는 확인, 자연스러운 play는 별도

기존 0.5초 phase 전이 bank에서 1.5초 간격의 전이만 선택해 Boxing/Traffic V1/V2/통합 네 모델을 각각 새로 학습했다. 학습에서 제외한 phase reference를 물리 평가했다.

| 모델 | 미학습 중간 phase 완주 | 범위 밖 phase 완주 | 중간 phase return |
|---|---:|---:|---:|
| Boxing | 18/18 | 9/9 | 0.589 |
| Traffic V1 | 32/32 | 20/20 | 0.860 |
| Traffic V2 | 24/24 | 15/15 | 0.890 |
| 통합 | 74/74 | 44/44 | 0.737 |

[조건과 원본 경로](RESEARCH_DISCUSSION_20261103.md). 통합 평가 clip은 개별 scenario 평가 clip을 합친 것이므로 독립된 새 표본으로 더하지 않는다. 같은 pair와 primitive를 공유한 한 seed의 평가다. 이 결과는 미학습 phase의 clip 완주이며, bridge 구간을 정확히 재현하거나 자연스러운 연속 명령을 수행했다는 증거로 쓰지 않는다. 앞의 영상 사례와 구분한 보조 결과다.

## 4. 최근 cross-full 전환은 성공 목록에서 분리

long-base는 세 full 내부 bridge를 학습했지만 새 cross-full bridge는 학습하지 않았다. frozen policy에 새 bridge 또는 direct-cut 목표를 주어 총 54조건을 비교했다. 양쪽 모두 54/54 완주했지만 Boxing의 실제 동작이 bridge reference와 다르게 진행하는 구간을 확인했다. Traffic 예시는 reference부터 정체하는 구간이 있다.

따라서 확인된 것은 **목표를 바꾼 뒤 넘어지지 않고 B까지 진행하는 능력**이며, **의도한 새 연결 동작을 자연스럽게 재현한 성공**으로 세지 않는다. 이전 문서의 “성공” 표현은 이 구분을 우선한다.

- [전체 비교 기록](reports/long_switch_20261104/README.md)
- [동작 흐름 재검토와 분리 영상](reports/long_switch_20261104/flow_review/README.md)

명령 직후 reference → policy 관절 목표 → 실제 관절각의 세 경로를 비교하는 후속 진단은 이 대화에서 제안했으나, 여기까지 확인한 산출물에는 완료 결과가 없다. 추가 transition 학습에 따른 cross-full 개선도 아직 확인된 성과가 아니다.

## 현재 확보한 것

1. 기존 학습 transition bank에서 Traffic V2의 높은 tracking 품질과 10회 연속 명령 실행.
2. Boxing 모델을 회복시켜 두 명령 시퀀스에서 각각 10회 완료한 실행 후보.
3. 세 30초 이상 reference를 하나의 policy로 완주하는 long-motion base와 영상.
4. sparse phase 학습 후 제외한 phase clip을 완주하는 제한된 평가 결과.

실제 시뮬레이션 완료, reference 추종 품질, 연결의 자연스러움은 서로 다른 주장이다. 지금 가장 강한 play 결과는 Traffic이고, 미학습 cross-full 연결의 재현은 남은 과제다.
