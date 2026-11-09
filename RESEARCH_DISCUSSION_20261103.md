# R2H2R 기반 실시간 모션 전환: 실험 결과와 논의

작성일: 2026-11-03, Asia/Seoul. 다른 사람과 연구 방향을 논의하기 위한 메모다. **확인된 결과, 연구 가설, 제안된 설계를 구분한다. 최종 방법을 결정한 문서는 아니다.**

> 후속 첨부 대화에서 의도를 더 명확히 했다. 핵심은 온라인 선택보다 **오프라인 adaptive transition 학습**이다. [후속 실험 설계와 실제 WM calibration 결과](ADAPTIVE_TRANSITION_STUDY_20261103.md)를 먼저 확인한다.

## 1. 지금 풀고 싶은 문제

두 primitive A와 B가 있을 때, 실행 중 명령에 맞춰 A에서 B로 전환하고 싶다. 모든 시작 시점의 A→B reference를 미리 만들어 로봇 policy에 학습시키는 방식은 데이터 준비와 학습에서 중복이 클 수 있다.

사용자가 제기한 질문은 다음 두 가지다.

1. **대표 전이만 학습한 로봇 controller가, 학습하지 않은 phase의 전이 reference도 실행할 수 있는가?**
2. **두 모션이 주어졌을 때 현재 로봇 상태에 따라 언제 연결을 시작하고, 얼마나 긴 연결을 만들 것인가?**

두 번째 문제까지 다루더라도 실시간 명령 응답이 중요하다. 매 명령마다 무거운 생성·retarget·다중 후보 rollout을 수행하는 설계는 사용자가 원하는 방향과 맞지 않는다.

## 2. 현재 구현과 구상 중인 R2H2R을 구분해야 한다

현재 검증한 실행기는 **미리 생성하고 로봇으로 retarget한 reference bank**에서 pair를 선택하고, 하나의 tracking policy로 MuJoCo 물리를 실행한다. 명령 시 reference의 XY/yaw를 정렬하고 실제 로봇 상태와 policy history를 유지한다. 실행 시 매번 human inbetweening을 새로 수행하는 시스템은 아니다.

사용자가 연구 대상으로 생각하는 R2H2R 구조는 다음과 같이 이해했다.

```text
현재 로봇 동작/상태
    → human 표현의 source context
    → H inbetweening으로 목표 모션까지 연결
    → 로봇 reference로 변환
    → tracking policy로 실행
```

이 전체 경로를 온라인에서 실행할지, 오프라인 teacher로 사용할지는 아직 결정하지 않았다. 현재 로봇 상태를 H로 바꾸는 방식과 왕복 변환의 경계 일치도 별도로 검증해야 한다.

현재 SuperMimic 계열 구현에서 역할은 다음과 같다.

| 구성 요소 | 입력과 역할 |
|---|---|
| World model | 로봇 상태·action·물리 context로 다음 상태를 예측한다. |
| Tracking policy | 관측 history와 미래 reference를 받아 action을 출력한다. |
| RMAI | 주어진 source/target context와 조건으로 연결 reference를 생성하는 역할이다. |
| 전이 선택기 | 어떤 pair/phase를 사용할지 결정한다. 현재 기본은 가까운 source phase 선택이다. |

**WM이 물리를 학습했다고 해서 policy가 임의의 새 reference를 추종한다는 보장은 없다.** 물리 예측의 일반화와 조건부 제어의 일반화를 구분해야 한다. 또한 현재 온라인 controller는 WM rollout을 매 제어 step 수행하는 MPC가 아니다.

## 3. 서로 다른 시간 변수를 혼동하지 말 것

| 변수 | 의미 | 이번 실험에서의 위치 |
|---|---|---|
| 학습 phase 간격 | A의 어느 시점에서 빠져나가는 전이를 학습할지 | 기존 0.5초 간격에서 매 세 번째 phase만 선택, 즉 1.5초 간격 |
| Bridge 길이 | A와 B 사이 연결 동작 자체의 지속 시간 | phase 간격과 별개의 변수. 기존 데모에서 관측한 bridge는 1.5초 |
| 명령 후 시작 지연 | 명령을 받고 연결을 시작할 때까지 기다리는 시간 | 앞으로 최적화할 수 있는 변수 |
| Reference 제어 간격 | reference/물리의 한 step | 기존 50Hz, 즉 0.02초를 유지 |
| 생성·추론 wall time | 실제 계산에 걸리는 시간 | 위 simulation 시간들과 별도로 측정해야 함 |

**1.5초 phase 간격으로 학습한 것이, 1.5초 길이의 연결을 선택하도록 학습했다는 뜻은 아니다.**

## 4. 완료한 실험: sparse phase 학습

사용자가 먼저 요청한 간단한 실험을 수행했다. 기존 reference에서 1.5초 cutback 간격만 학습하고, 빠진 phase의 전이 reference는 평가에만 사용했다. cutback=0은 각 source의 끝이며 그리드는 원래 corpus의 끝 기준을 유지한다.

두 학습 anchor 사이의 미학습 phase를 interpolation, 학습 범위 바깥을 extrapolation으로 나눴다. 짧은 source는 학습 anchor가 하나뿐일 수 있으므로 두 평가를 구분해야 한다.

| 데이터 | 학습 전이 | 학습 primitive | 중간 phase 평가 | 범위 밖 평가 |
|---|---:|---:|---:|---:|
| Boxing | 21 | 4 | 18 | 9 |
| Traffic V1 | 36 | 5 | 32 | 20 |
| Traffic V2 | 24 | 4 | 24 | 15 |
| 통합 | 81 | 13 | 74 | 44 |

통합의 평가 clip은 위 세 scenario 평가 clip의 합이다. 별개의 추가 모션 118개가 아니다. 통합 bank의 전체 전이는 199개였으며 sparse 조건은 81개를 학습했다.

### 학습 조건과 결과

- 출력: `runs/phase15_20261101_123724/`.
- 네 run 모두 256 env, 3,000 iterations, seed 0. 각 run의 기록된 물리 interaction은 1,936,384 steps.
- WM-only 500회, actor LR 5e-5, WM/critic LR 2e-4, critic upper bound 적용, compile off.
- policy/WM/critic 모두 새로 시작. 기존 dense/recovery checkpoint warm-start 없음.
- 학습 subset 평가로만 `best_coverage.pt`를 선택. 이후 heldout 평가.
- 학습 reference와 평가 reference의 이름 중복 0을 확인했다.
- 네 run 모두 학습과 heldout 평가 완료. 학습 기록상 run별 약 0.94~0.95시간.

| 모델 | 선택 iteration | 학습 clip return | 중간 phase return | 중간 실패 | 범위 밖 return | 범위 밖 실패 |
|---|---:|---:|---:|---:|---:|---:|
| Boxing | 2,800 | 0.601 | 0.589 | 0/18 | 0.602 | 0/9 |
| Traffic V1 | 3,000 | 0.878 | 0.860 | 0/32 | 0.867 | 0/20 |
| Traffic V2 | 3,000 | 0.889 | 0.890 | 0/24 | 0.843 | 0/15 |
| 통합 | 3,000 | 0.747 | 0.737 | 0/74 | 0.760 | 0/44 |

각 결과 원본은 `k1_<group>/heldout/best_coverage/{interpolation,extrapolation}.json`, 학습 평가는 `k1_<group>/evals/`에 있다. `last.pt`도 별도로 평가했고 모두 실패 0개였다. 중간 평가를 보고 checkpoint를 다시 고르지 않았다.

### 이 결과에서 말할 수 있는 것

**이번 seed와 corpus에서는 모든 0.5초 phase 전이를 학습하지 않아도, 제외한 phase reference를 끝까지 실행할 수 있었다.** Traffic V2는 학습 clip과 중간 phase 평가의 평균 return도 가까웠다. Boxing은 학습 clip 자체 점수도 낮아 전반적인 추종 품질이 남은 문제다.

다음은 아직 입증하지 않았다.

- 같은 개선 설정·예산의 dense 학습과 동등한 품질인지: 해당 대조군은 아직 실행하지 않았다. 예전 4,096-env/12,000회 모델과의 비교는 통제 실험이 아니다.
- 높은 전체 clip 점수가 bridge 구간 자체의 높은 품질을 뜻하는지: 현재 점수에는 이미 알려진 A/B 구간이 포함된다.
- WM 때문에 일반화가 가능했는지: WM의 기여를 분리한 비교가 없다.
- 완전히 새로운 상태나 primitive 조합에도 되는지: 원본 primitive는 학습했고, 같은 source→target pair의 일부 phase를 제외한 실험이다.
- RMAI가 온라인에서 새로 생성한 reference나 상태를 유지한 긴 연속 명령에서도 되는지: 이번 sparse 모델의 heldout 평가를 그런 실험과 동일시하면 안 된다.
- 데이터 감소가 학습 시간 감소로 이어지는지: 현재는 양쪽의 동일 목표 성능 도달 시간을 비교하지 않았다.
- 여러 seed에서도 재현되는지: 현재는 seed 0 pilot이다.

## 5. 기존 Boxing에서 발견한 두 문제

### 단독 장기 학습 붕괴

기존 Boxing 학습은 약 1,400회까지 좋아지다가 실제 보상은 떨어지고 actor 목적값은 비정상적으로 증가했다. 코드의 step reward 상한 0.1과 gamma 0.99에서 할인 수익 상한은 10인데, actor가 최적화하는 목적값은 이를 크게 초과했다. critic 과대추정과 함께 actor/WM 학습이 붕괴하는 패턴을 관측했다.

critic이 최초 원인인지, WM 오차가 먼저 촉발했는지는 분리하지 못했다. 이전 회복 실험은 통합 policy/WM 전이, fresh critic, critic 상한, actor LR 감소 등을 함께 사용했으므로 한 요소의 효과로 단정할 수 없다. 이번 sparse fresh run도 이 이전 recovery run과 다른 실험이다.

### 로봇 reference 경계 불일치

Boxing pair의 source A prefix는 원본 A와 관절각 RMS 평균 약 0.037rad로 가까웠다. 반면 target B suffix는 원본 B와 평균 약 0.588rad 차이가 났고, 48개 중 26개가 0.5rad를 초과했다. resampling의 작은 시간 정렬 오차가 있을 수 있는 진단이다.

생성 human feature의 B suffix는 root 위치/heading을 맞추면 48개 모두 원본 B와 최대 차이 약 9.54e-7이었다. 따라서 큰 관절 차이가 생기는 구간은 생성 이후 SOMA export/retarget 경로로 좁혀졌지만, 정확히 어느 단계가 최초 원인인지는 아직 분리하지 않았다.

이 때문에 A→B 이후의 B가 canonical B→C의 source와 다를 수 있다. 이전 온라인 rollout에서는 shoulder 관절을 중심으로 큰 reference jump를 관측했다. 원인은 모두 이 불일치 하나라고 단정할 수 없고 source phase 오차도 영향을 준다.

**B suffix만 원본 CSV로 갑자기 교체하는 것도 해결책이 아니다.** bridge 끝에 새 불연속이 생길 수 있으므로 q·qd 경계를 함께 맞춰야 한다. 이 문제는 이번 sparse 실험에서 수정하지 않았다.

## 6. Diffusion Policy를 어디에 넣을 것인가

사용자는 두 모션이 이미 있다면 남은 문제가 “건널목을 언제, 얼마나 만들지”가 아닌지 제안했다. 여기서 “얼마나”는 우선 bridge 지속 시간으로 해석했다.

구상 가능한 상위 planner의 출력은 다음과 같다.

```text
입력: 현재 로봇 상태/history, 진행 중인 모션, 목표 모션, 응답 요구
출력: 시작 지연 Δt, bridge 길이 T, 목표 진입 phase φB
      필요하면 bridge reference 자체
```

Diffusion을 넣는 안은 두 가지였으며 아직 구현하지 않았다.

| 안 | 기대하는 역할 | 남은 질문 |
|---|---|---|
| 시점·길이 선택을 생성 모델로 학습 | 같은 조건에서 여러 가능한 전이 선택을 표현 | 낮은 차원 출력에 diffusion이 필요한가? 작은 MLP/분류기보다 좋은가? |
| 조건부 bridge trajectory 생성 | 팔 회수/몸 낮추기 등 서로 다른 연결 경로를 표현 | 생성 latency, 로봇 실행 가능성, retarget 경계를 어떻게 처리하는가? |

Diffusion Policy는 action sequence를 생성하고 일부를 실행한 뒤 관측을 갱신하는 구조를 제시한다. reference sequence에 비슷한 아이디어를 적용하는 것은 **우리의 제안**이며 원래 action diffusion과 같은 방법은 아니다. [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/)

조건부 keyframe inbetweening 자체는 기존 연구가 있으므로, diffusion으로 연결한다는 것만으로 새 기여라고 보기 어렵다. 로봇 상태·접촉·실행 가능성과 연결 시점/길이를 어떻게 결합하는지가 연구 질문이다. [CondMDI](https://setarehc.github.io/CondMDI/)

## 7. 실시간 제약 때문에 온라인 후보 탐색안은 채택하지 않음

대화 중 다음 안을 제안했다.

```text
여러 시점/길이의 bridge 생성
    → 각 후보 retarget
    → tracking policy의 action을 WM으로 rollout
    → 점수가 좋은 후보 선택
```

WM에 reference만 넣어 실행 가능성을 평가할 수는 없다. reference를 policy에 넣어 action을 얻고 그 action으로 WM을 전개해야 한다.

사용자는 이 전체 루프가 실시간 요구에 맞지 않는다고 지적했다. **따라서 이 무거운 다중 후보 생성·변환·rollout 루프를 온라인 실행 경로의 기본안으로 삼지 않는다.** 특정 시스템에서 절대로 실시간이 불가능하다고 측정으로 증명한 것은 아니지만, 비용과 지연 위험 때문에 현 단계의 방향으로 적절하지 않다는 판단이다.

또한 기존 Boxing critic 과대추정 사례 때문에 learned score만 좋아졌다고 실행 품질이 보장되는 것도 아니다. 오프라인 teacher에 WM을 쓰더라도 실제 simulator에서 검증해야 한다.

## 8. 실시간을 유지하는 대안들 — 아직 선택하지 않음

### A. 준비된 bank에서 상태 기반 선택

온라인은 현재 상태와 목표를 보고 준비된 reference를 고른 뒤 tracking policy로 실행한다. 현재 구현과 가장 가깝다. 이전 회복 모델에서 관절 자세·속도 기반 선택을 시험한 적은 있으나, 이번 sparse 모델과 별개의 결과다.

장점은 온라인에 human 생성·retarget·다중 rollout을 추가하지 않는다는 것이다. 한계는 bank 밖의 bridge를 새로 생성하지 못하며 reference 경계 불일치는 그대로 남을 수 있다는 것이다.

### B. 무거운 모델을 오프라인 teacher로 사용

```text
오프라인:
R2H2R/RMAI로 여러 연결 생성 → 실제 실행 평가 → 좋은 선택/bridge 데이터 수집

온라인:
작은 planner 또는 student → reference 선택/생성 → tracking policy
```

작은 planner가 시점·길이만 출력하더라도 **온라인 bridge 생성과 retarget이 여전히 필요하면 지연 문제는 남는다.** 따라서 bank 선택으로 끝내거나, 로봇 reference를 직접 빠르게 생성하는 student까지 필요하다.

Diffusion을 teacher로 쓰는 것은 후보안이다. 같은 역할의 다른 생성기보다 좋은지, 충분한 학습 데이터가 있는지, distillation으로 품질이 유지되는지 아직 모른다. 작은 모델의 추론이 한 번이라는 이유만으로 전체 시스템의 실시간성을 보장할 수도 없다.

### C. 온라인 human 생성 유지 + 전체 경로 가속

R2H2R의 온라인 생성 자체를 중요하게 본다면 남는 선택지다. 생성과 retarget까지 포함한 latency를 실제로 측정하고, 비동기 처리·계획 선행 시간·빠른 생성기를 설계해야 한다. 아직 이 경로의 end-to-end 실시간 성능을 검증하지 않았다.

## 9. 다음 대화에서 정해야 할 질문

1. **연구의 중심은 무엇인가?** sparse 학습의 데이터 효율, 온라인 전이 시점/길이 결정, bank 밖 bridge 생성 중 어디에 우선 기여를 둘 것인가?
2. **R2H2R은 온라인 필수 경로인가, 오프라인 teacher인가?** 두 선택은 필요한 모델과 latency 조건이 다르다.
3. **연결을 즉시 시작해야 하는가?** 허용 가능한 명령 응답 지연과 목표 모션 진입 시간은 각각 얼마인가?
4. **B는 반드시 처음부터 진입하는가?** 현재 corpus는 target entry mode다. B 중간 진입까지 확장하면 다른 문제가 된다.
5. **실제 새 bridge가 필요한가?** 준비된 bank에서의 좋은 선택만으로 목표 성능이 나오면 더 단순한 방법이 충분할 수 있다.
6. **평가를 무엇으로 할 것인가?** bridge-only 추종 오차, 실패율, 명령 응답 지연, 목표 진입 시간, 계산 latency를 함께 봐야 한다.

실시간 측정은 50Hz 저수준 제어 deadline과 고수준 명령 처리 latency를 나눠야 한다. 평균뿐 아니라 긴 지연도 확인해야 한다. 영상 렌더링 속도와 물리/controller 실행 속도도 구분한다. 허용 latency 수치는 아직 정하지 않았다.

## 10. 바로 이어서 할 수 있는 검증

현재 sparse 결과를 먼저 해석하려면, 같은 조건의 dense 대조군·bridge-only 지표·heldout 영상·추가 seed가 필요하다. 이후 전이 의사결정을 다룬다면 tracker를 고정하고 고정 시점/길이 대비 상태 기반 선택을 비교하면 controller 재학습 효과와 분리하기 쉽다.

시점·길이 학습을 하려면 여러 길이와 선택 결과가 있는 데이터가 필요하다. 설정에 0.5/1.0/1.5초 후보가 있다고 해서 기존 bank에 모든 길이가 저장됐다는 뜻은 아니다. 실제 duration 분포를 먼저 확인해야 한다.

이 문서 시점에 새 diffusion planner, 새 student, 실시간 R2H2R 경로를 구현하거나 새 학습을 시작하지 않았다. **현재 성과는 sparse phase 추종의 첫 실험 결과이고, 온라인 생성/의사결정 방식은 논의 중이다.**

## 관련 자료

- [기존 재생·Boxing 회복·bank 진단 기록](REVIEW_20261101.md)
- [인수인계](HANDOFF.md)
- sparse 실험: `runs/phase15_20261101_123724/`
- 네 run 학습 스크립트: `scripts/train_phase15_four.sh`
- phase 분할: `tools/split_transition_phases.py`
- heldout 평가: `tools/evaluate_phase_holdout.py`
- 이전 온라인 비교 영상: `runs/review_commands_20261101/` — sparse 모델의 온라인 평가 영상은 아님
