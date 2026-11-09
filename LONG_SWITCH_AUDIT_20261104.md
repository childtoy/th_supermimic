# Long-motion switching 독립 재검증

사용자의 “정말 된 것이 맞는가”라는 지적에 따라 기존 cache 분기 평가와 독립적인 연속 물리 실행으로 재검증했다. 추가 학습 없이 iteration 11,500의 동일 policy/WM을 고정했다.

## 검증 방법

- 6개 directed motion pair × source phase 9개(0.8, 3.7, 5, 8.3, 12.4, 15, 20.6, 25, 28.9초) = 54개 조건.
- 각 조건을 RMAI bridge, direct-cut, 명령 후 zero-action의 세 방식으로 실행: 총 162회.
- 매번 A의 0초에서 실제 물리 실행을 시작한다. 명령 시점의 cache 복원 없이 그대로 전환하고 원래 full B의 끝까지 실행한다. B 진입은 항상 0초이며 bridge 길이는 1.5초다.
- 명령 및 canonical B handoff에서 실제 state, 이전 action, observation history가 변하지 않는지 assertion으로 확인한다. simulator reset 호출도 감시하여 terminal lifecycle 외 reset이 있으면 중단한다. 정상 완주 후 자동 reset은 별개로 기록한다.
- XY/yaw reference 정렬은 명령/handoff 시 한 번 수행한다. 실제 로봇 자세나 관절을 reference로 옮기지 않는다.
- hard completion은 reference 대비 root 높이 오차 0.25m 및 tilt 행렬 성분 차이 0.8의 termination을 피하고 B 끝에 도달했다는 뜻이다. 정확한 추종이나 deployment 성공의 정의가 아니다.
- 탐색적 quality screen: hard completion AND 전환 점수≥0.8 AND 남은 B 점수≥0.8. 사전에 보정한 안전 기준이 아니다.
- bridge 전환 창은 명령 후 77step(1.54초), direct-cut은 75step(1.5초). 서로 다른 reference를 추종하므로 점수만으로 동일 동작의 우열을 뜻하지 않는다. body/joint raw 오차도 함께 저장한다.
- WM은 실제 명령 상태에서 출발한 뒤 실제 상태 피드백 없이 policy와 자유 전개한다. action=0 대조군과 상태를 고정한 reward 예측 baseline도 평가한다.

## 최종 결과

| 방식 | hard 완주 | quality 통과 | 전환 평균 점수 | 남은 B 점수 | 전환 body RMSE | 전환 joint RMSE |
|---|---:|---:|---:|---:|---:|---:|
| policy | 54/54 | 33/54 | 0.8267 | 0.9207 | 6.56cm | 0.383rad |
| direct_cut | 54/54 | 28/54 | 0.8077 | 0.9147 | 8.36cm | 0.307rad |
| zero_after_command | 0/54 | 0/54 | 0.4761 | 0.0000 | 17.63cm | 0.562rad |

**완주와 정밀 추종은 다른 결론이다.** bridge도 direct-cut도 54개 모두 완주했다. 따라서 완주율만으로 RMAI bridge나 추가 transition 학습의 필요성을 입증하지 못한다. bridge가 35/54 조건에서 전환 점수가 높았고 평균 body RMSE는 줄였지만, 평균 joint RMSE는 오히려 더 크다. 서로 다른 reference에 대한 오차라는 점까지 고려하면 모든 지표에서 개선됐다고 주장할 수 없다.

Boxing 출구는 bridge quality **1/18**, Traffic V1/V2 출구는 각각 **16/18**이다. Boxing 출구 평균 전환 점수는 bridge 0.695, direct-cut 0.732로 bridge가 더 낮다. 가장 낮은 Box→T2@12.4초는 0.589이며 reference 관절 경계 RMS 차이는 1.448rad다. 큰 reference 불일치가 여러 나쁜 조건과 함께 나타나지만 Box@0.8초처럼 경계 차이가 작아도 품질이 낮은 조건이 있어, 경계만을 유일 원인으로 단정하지 않는다. base Boxing tracking도 상대적으로 나쁘다.

명령/handoff assertion 216개에서 물리 state/action/history 변경이 없었고 unexpected reset은 0개였다. 종료 시 reset은 정상 기록됐다. action을 0으로 바꾼 대조군 54개는 모두 hard termination으로 종료되어, 무조건 완주 처리하는 평가기는 아니라는 점도 확인했다.

기존 18개를 54개 batch에 넣어 재실행했을 때 완주 판정은 전부 일치했다. 정상 policy 전환 점수 차이는 평균 0.00115 / 최대 0.00343, direct-cut 최대 0.00303이었다. zero-action 최대 차이는 0.02935였다. GPU/batch 실행은 bitwise 동일하지 않으며 정밀 점수를 과도하게 해석하지 않는다.

### WM 해석

정상 policy의 bridge free-rollout 점수 MAE는 **0.00182**, joint state 예측 RMSE는 평균 **0.00944rad**였다. 동일 실제 시작 자세를 고정하고 reference만 진행시킨 baseline의 점수 MAE는 **0.1432**여서, 단순히 비슷한 reference 점수를 읽는 것보다 실제 동역학 예측이 유효했다. B 첫 1초의 점수 MAE는 0.00483이다.

그러나 zero-action에서는 bridge MAE **0.09887**, joint 예측 RMSE **0.0580rad**, 점수 Spearman **0.558**로 악화됐다. bridge 창 안의 termination 수는 WM 35 / simulator 37이며 개별 사건 일치나 종료 시각 정확성을 보장하는 수치가 아니다. B 첫 1초까지는 둘 다 54개 종료를 예측/관측했다. 정상 추종 주변의 높은 WM 정확도를 분포 밖 실패 탐지 전반으로 확대하지 않는다.

### 다음 실험 우선순위

1. Boxing 원래 full reference와 새 bridge의 retarget 경계 및 실제 source tracking을 영상/관절별 오차로 분해한다. 경계를 수정한 reference를 같은 frozen policy로 재평가해 reference 문제와 controller 문제를 구분한다.
2. 그 뒤 남는 저품질 조건만 추가 학습하고 동일 54개 및 새로운 phase에서 direct-cut/frozen baseline과 비교한다. 추가 학습이 품질을 개선하는지 확인한 후 sparse/adaptive 효율 주장을 검토한다.
3. 임의 B 진입 phase와 연속 A→B→C 명령을 추가한다. 현재 모든 B 진입이 0초인 단일 전환이므로 실제 interactive long-horizon 검증은 아직 남아 있다.

![54개 조건의 대조군 비교](runs/long_switch_live_audit_54_20261104/audit.png)

## 재현 및 산출물

- 실행 코드: `tools/audit_live_long_switching.py`
- 기존 18개 독립 실행: `runs/long_switch_live_audit_original_v3_20261104/`
- 확장 54개 최종 실행: `runs/long_switch_live_audit_54_20261104/`
- 새 reference 36개: `runs/long_switch_expanded_20261104/`
- 각 실행 폴더에 실제 trial/event 로그, CSV, WM 예측, 원본 state/action trace와 영상용 pose를 보존한다.
- 기존 18개 영상은 source 처음부터 시작하는 연속 실행 기록이며 `long_switch_live_audit_original_v3_20261104/videos/`에 있다.

## 해석 범위

현재 full motion은 기존 짧은 primitive를 반복하고 내부 bridge로 연결한 구성이다. 길이는 늘었지만 고유 dynamics가 그만큼 증가한 것은 아니다. source phase가 달라도 비슷한 상태를 재방문한다. B의 임의 중간 phase, 연속 다중 명령, perturbation, 새로운 motion, 하드웨어 및 온라인 reference 생성 지연은 이 검증에 포함하지 않았다.

따라서 완주만으로 universal transition이나 adaptive 추가 학습의 필요성/효율을 주장할 수 없다. direct-cut 대비 품질 개선과 낮은 품질 구간을 먼저 확인해야 한다. WM의 정상 추종 주변 정확도와 실패 분포에서의 정확도도 구분해야 한다.
