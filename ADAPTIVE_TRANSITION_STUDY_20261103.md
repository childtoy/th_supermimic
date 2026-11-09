# 오프라인 적응형 전이 학습: 실험 설계와 WM 검증

작성일: 2026-11-03. 사용자 첨부 대화를 읽고 정리한 실험 설계와 즉시 수행한 calibration 결과다. 방법의 novelty나 학습 비용 절감이 입증됐다는 문서는 아니다.

## 사용자의 제안을 다시 정의

목표 B의 진입점은 고정하고, A의 유효한 전환 phase 구간을 양 끝부터 시작한다. 중간 phase를 현재 policy가 이미 실행할 수 있으면 추가 학습하지 않고, 어려운 중간 phase만 reference 학습에 추가한다. 추가한 구간을 다시 나누고 필요할 때만 반복한다.

이는 온라인에서 명령마다 전이 시점을 탐색하는 안과 다르다. **무거운 생성·선별·검증은 오프라인 학습 과정에서만 수행한다.** 온라인은 기존 reference 선택과 tracking 구조를 유지할 수 있다. 이분 구간 세분화에 가깝지만, 추종 오차는 phase에 대해 단조라는 보장이 없어 수학적인 root bisection 보장과는 다르다.

RMAI는 후보 reference 생성, policy는 실행, WM은 해당 policy가 후보를 추종할 때의 동역학 예측을 담당한다. WM 단독의 one-step prediction loss를 통과 기준으로 쓰지 않는다.

```text
후보 reference R + 현재 상태/history
    → policy의 action
    → WM의 다음 상태
    → 예측 상태에서 policy를 다시 실행
    → predicted bridge tracking score / predicted termination
```

## 즉시 확인한 것: WM 선별 점수 calibration

기존 sparse run `runs/phase15_20261101_123724`의 checkpoint를 그대로 사용했다. 새 학습이나 adaptive reference 추가는 하지 않았다.

- Traffic V2: 미학습 중간 phase 24개.
- Boxing: 미학습 중간 phase 18개.
- 각 600/1,400/3,000회 checkpoint로 policy와 WM을 함께 고정.
- 실제 MuJoCo에서 clip을 처음부터 수행해 state/action/history와 실패를 기록.
- bridge 직전 실제 state/history를 시작점으로 policy+WM closed-loop rollout 수행.
- 32-step prefix, 재초기화한 중간/끝 window, 그리고 **bridge 전체 75-step(1.5초) free rollout**을 별도로 평가.
- 전체 bridge free rollout은 내부에서 실제 state로 재초기화하지 않는다. 중간/끝 window는 각각 실제 상태에서 다시 시작하므로 전체 bridge rollout과 구분한다.
- WM 점수와 실제 점수는 같은 reward, 같은 고정 window 길이로 정규화한다. 종료 후 남은 step 보상은 0으로 처리한다.
- critic은 점수 계산에 사용하지 않는다. WM의 관절 prediction 오차와 reference tracking 오차를 따로 저장한다.

### 전체 bridge 비교

| 데이터 / checkpoint | bridge에 도달한 후보 수 | WM 평균 점수 | 실제 평균 점수 | 평균 절대 점수 오차 | 후보 순위 상관(Spearman) |
|---|---:|---:|---:|---:|---:|
| Traffic V2 / 600 | 10/24 | 0.1654 | 0.1978 | 0.03234 | 0.988 |
| Traffic V2 / 1,400 | 24/24 | 0.7532 | 0.7555 | 0.00247 | 0.991 |
| Traffic V2 / 3,000 | 24/24 | 0.8881 | 0.8871 | 0.00176 | 0.993 |
| Boxing / 600 | 12/18 | 0.0830 | 0.0878 | 0.00951 | 0.993 |
| Boxing / 1,400 | 12/18 | 0.2912 | 0.2913 | 0.00036 | 1.000 |
| Boxing / 3,000 | 18/18 | 0.6056 | 0.6142 | 0.01025 | 0.950 |

600/1,400회 일부 clip은 bridge 전에 실패했다. 해당 후보는 bridge 시작점의 평가에서 제외됐으며 통과로 간주하면 안 된다. 낮은 checkpoint에서 free rollout horizon은 75 step으로 설정했지만 실제/예측 종료 후 보상은 0이다. 성공한 75-step 예측 사례와 조기 종료한 사례를 동일한 장기 정확도 증거로 취급하지 않는다.

예시로 “예측 종료 없음 + bridge score ≥ 0.8”을 사용하면 Traffic V2 1,400회에서는 24개 중 1개, 3,000회에서는 24개 전부가 통과했다. 이 표본에서는 false pass가 없었다. Boxing 3,000회에서는 같은 품질 기준을 넘는 후보가 0개였다. 실제로 넘어지지 않는 것과 충분한 추종 품질은 다른 기준이다.

**해석:** 현재 corpus에서는 policy+WM의 bridge 점수가 실제 물리 점수와 가까워, 학습이 부족한 phase를 선별할 가능성이 있다. 하지만 표본이 작고 checkpoint들이 같은 clip을 반복 평가한다. 이 결과만으로 unseen motion, 새로운 접촉, arbitrary phase, 모델 업데이트 이후에도 gate가 정확하다고 보장하지 않는다.

### 중요한 한계: oracle 시작점

이번 calibration은 실제 simulator에서 얻은 시작 state/history를 사용했다. 따라서 **후보별 simulation을 생략하는 방법을 검증한 것은 아니다.** 우선 점수 예측이 가능한지를 확인한 실험이다.

실용적인 다음 단계는 source primitive를 수행한 trajectory/history를 cache하고 같은 source/phase의 여러 target에 재사용하는 것이다. 다만 다음을 확인해야 한다.

- pair source prefix와 canonical primitive가 retarget 후 정확히 같은가?
- global XY/yaw를 맞춘 실제 state/history를 올바르게 이전할 수 있는가?
- 미래 goal preview 때문에 bridge 직전 실제 state가 target별로 이미 달라지지는 않는가?
- cache 시작점 기반 WM 점수도 실제 candidate 실행과 잘 맞는가?

현재 boundary calibration이 좋았다는 이유로 cache 기반 평가도 자동으로 정확하다고 간주하지 않는다.

## 다음에 실행할 실제 adaptive 학습 pilot

첫 pilot은 상태가 안정적이고 phase별 reference가 준비된 Traffic V2로 제한하는 것이 적절하다. 새 diffusion 모델이나 온라인 탐색은 필요 없다.

1. 각 A→B pair의 유효한 최소/최대 source phase reference와 원본 primitive로 시작한다. A의 첫 frame은 필요한 context가 부족할 수 있으므로 유효 phase 끝점을 사용한다.
2. 새로운 policy/WM으로 초기 학습한다. 지금 사용한 sparse checkpoint는 calibration용이며 endpoint-only 학습 결과로 보고하지 않는다.
3. 각 interval의 중간 후보를 현재 policy+WM으로 평가한다. 기존 0.5초 bank만 쓰면 정확한 midpoint가 없을 수 있어 가까운 미평가 grid point를 선택하는 discrete refinement로 정의한다.
4. 예측 추종 품질이 낮거나 예측 실패/불확실성이 크면 그 후보를 학습 reference에 추가한다. 예측이 좋더라도 일부 후보는 실제 물리로 spot-check한다.
5. 추가 reference와 기존 primitive/selected transition replay를 함께 사용해 정해진 budget만큼 학습한다.
6. policy/WM이 바뀌었으므로 이전에 통과했던 후보도 다시 평가한다. 이전의 통과 상태를 영구적인 것으로 취급하지 않는다.
7. 정해진 종료 조건/budget에서 전체 평가 grid를 실제 MuJoCo로 평가하고 학습에 선택하지 않은 reference 결과를 별도로 보고한다.

현재 Traffic V2는 12개 directed pair다. 양 끝점을 각 2개씩 사용하면 초기 전이는 24개로, 기존 1.5초 sparse 전이 개수와 같지만 phase 배치는 다르다. 따라서 **같은 초기 개수에서 시작하는 endpoint-only / uniform / adaptive 비교**를 만들 수 있다.

### 대조군과 비용

| 방법 | 다음 학습 reference 결정 |
|---|---|
| Dense | 기존 63개 전이 전부 |
| Uniform sparse | 동일 간격 또는 동일 개수 |
| Random | 동일 개수/예산에서 무작위 선택 |
| Simulator-guided adaptive | 실제 실행 점수로 중간 후보를 선택; 비용 높은 teacher 대조군 |
| WM-guided adaptive | policy+WM 예측 점수로 중간 후보를 선택 |

실제 학습 budget이 adaptive round마다 늘어나므로 마지막 reference 개수만 맞추면 공정한 비교가 아니다. 초기 학습 + 추가 학습 + replay + gate/검증 비용을 합산하고, 고정 budget 성능과 목표 성능 도달 비용을 구분한다.

반드시 분리해서 기록할 비용:

- 생성/retarget한 candidate reference 수.
- 실제 학습에 추가한 reference 수.
- screening, spot-check, 최종 평가의 실제 physics steps.
- policy/WM 학습 physics steps와 GPU 시간.
- WM candidate rollout 수와 시간.

후보를 WM으로 평가하려면 그 후보 reference가 필요하다. **학습에서 제외한 reference도 이미 생성했다면 생성 비용 절감으로 세면 안 된다.** 기존 bank를 쓰는 첫 pilot은 selection 및 training 절감을 검증하며 새 generation 비용 절감을 직접 입증하지 않는다.

중간점 통과는 interval 전체 성공의 증명이 아니다. 접촉 전환이나 retarget branch 변화 때문에 off-center 실패가 생길 수 있다. quarter/random sentinel과 최종 미선택 phase 평가가 필요하다. threshold들은 현재 exploratory 예시이며, 이후 benchmark에서는 calibration용 자료와 최종 평가용 자료를 구분하고 기준을 미리 고정한다.

## Incremental motion integration으로 확장

사용자가 궁극적으로 원하는 상황은 motion library가 이미 학습됐고 새 D가 계속 추가되는 경우다.

```text
기존 A/B/C controller
    → D primitive 추가 학습 + 기존 데이터 replay
    → A/B/C ↔ D의 대표 전이만 추가
    → 어려운 phase만 adaptive refinement
    → 기존 모션/전이 retention과 새 D 연결을 함께 평가
```

현재 네 primitive 모두를 본 sparse checkpoint로는 “D를 새로 추가했다”는 실험을 할 수 없다. Traffic V2에서 primitive 하나를 D로 정하고 나머지 세 개와 그 사이 전이만 학습한 base model을 새로 만들어야 한다. base policy/WM의 데이터 노출 조건도 명시한다.

처음에는 replay로 retention을 유지하고, 새 continual-learning 알고리즘까지 동시에 제안하지 않아도 된다. 평가에는 D 추종, 양방향 새 전이, 이전 모션/전이 성능, 총 integration 비용을 포함한다.

이 방식은 phase 중복을 줄이는 것이며, 모든 directed pair가 필요하면 새 motion당 pair 수 증가 자체는 남는다. pair 조합의 증가를 해결했다고 과장하지 않는다. 현재 primitive는 짧아서 long-duration motion 확장성도 별도의 데이터로 평가해야 한다.

## 코드와 결과

- 실행 도구: `tools/calibrate_transition_wm.py`
- Traffic V2 결과: `runs/wm_gate_calibration_20261103/traffic2/`
- Boxing 결과: `runs/wm_gate_calibration_20261103/boxing/`
- checkpoint별 JSON과 `windows.csv`에 bridge 점수, 예측/실제 종료, 관절 prediction/tracking 오차가 있다.
- [이전 연구 논의](RESEARCH_DISCUSSION_20261103.md)

재현 예시(기존 학습 환경, 새 output directory 사용):

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python tools/calibrate_transition_wm.py \
  --run-dir runs/phase15_20261101_123724 \
  --group traffic_control2 \
  --output-dir runs/wm_gate_calibration_new/traffic2
```

이 시점에서 완료한 것은 offline oracle-start calibration이다. adaptive 학습 루프, reference 자동 추가, cache 기반 simulator-free screening, incremental D 통합은 아직 구현/실행하지 않았다.
