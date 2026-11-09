# 전환 실험 기록 모음

> 후속: [동작 흐름 재검토](flow_review/README.md)에 ghost 없는 25fps reference/실제/direct-cut 비교 영상과 시간순 그림을 추가했다.

현재 상태를 보존한 결과 묶음이다. 추가 학습이나 새 물리 rollout 없이 기존 54개 조건 × 3방식의 결과와 저장된 연속 자세를 분석했다.

- [브라우저에서 그림·영상 한 번에 보기](index.html)
- [실험 전체 해석과 프로토콜](../../LONG_SWITCH_AUDIT_20261104.md)
- [완주·추종 품질·WM 그래프](audit.png), [PDF](audit.pdf)
- [실제 로봇 jerk 비교](actual_jerk_comparison.png), [PDF](actual_jerk_comparison.pdf)
- [실제/reference jerk 비교](jerk_windows.png), [PDF](jerk_windows.pdf)
- [전체 trial CSV](trials.csv), [전체 결과 JSON](result.json)
- [가속도·jerk 요약 CSV](smoothness.csv), [출처·hash·계산 조건](provenance.json)

## 현재 관찰

브리지 54/54 완주와 품질 33/54를 구분한다. direct-cut도 54/54 완주했다. Boxing 출구의 브리지 품질 통과는 1/18이다.

저장된 대표 4개 조건의 paired 8회에서 명령 주변 실제 jerk RMS는 브리지가 direct-cut보다 낮았다. 이 제한된 표본은 브리지가 추종 점수 외에 부드러움을 개선할 가능성을 보여주지만, 전체 조건의 결론은 아니다. Boxing 15초 브리지 전환도 명령 전보다 jerk가 증가한다. reference의 큰 피크는 명령에 따른 자세 불연속을 포함하며 실제 로봇이 똑같이 순간 이동했다는 뜻이 아니다.

## jerk 계산 조건

50Hz 관절각에서 Savitzky–Golay 11 sample / degree 5로 가속도와 jerk를 계산했다. 연속 trace 전체를 미분한 뒤 구간을 잘랐으며 raw third difference도 CSV에 보존했다. RMS 단위는 rad/s³, 관절 전체와 각 창의 시간에 대한 RMS다. 미분 추정은 필터 설정에 민감하고 0.22초 창이 경계 피크를 주변으로 퍼뜨리므로 절대값을 안전 기준으로 쓰지 않는다.

전환 전 창: −2~−0.25초. 명령 주변 창: −0.2~+0.5초. bridge 전환 창: 0~1.54초. canonical B handoff: +2.54초(분석 창 +2.34~2.74초). 전환 전 창은 실제 source prefix이며 별도로 실행한 무전환 대조군은 아니다.

| 조건 | 방식 | 전환 전 jerk RMS | 명령 주변 jerk RMS |
|---|---|---:|---:|
| boxing__to__traffic_control1__t00500 | policy | 262.5 | 275.8 |
| boxing__to__traffic_control1__t00500 | direct_cut | 264.2 | 631.3 |
| traffic_control1__to__traffic_control2__t00500 | policy | 46.9 | 80.9 |
| traffic_control1__to__traffic_control2__t00500 | direct_cut | 45.9 | 448.0 |
| boxing__to__traffic_control1__t01500 | policy | 71.6 | 349.3 |
| boxing__to__traffic_control1__t01500 | direct_cut | 71.7 | 740.6 |
| traffic_control1__to__traffic_control2__t01500 | policy | 340.6 | 188.1 |
| traffic_control1__to__traffic_control2__t01500 | direct_cut | 339.2 | 551.7 |

## 영상

영상의 solid는 실제 시뮬레이션 로봇, blue ghost는 reference다. window 영상은 명령 전 2초~후 4초, 원본 50Hz trace에서 5fps로 렌더했다. 정밀 피크 판단은 그래프/CSV를 사용한다. full 영상은 기존 독립 18조건 batch의 같은 case이며 확장 54조건 batch와 미세한 수치 차이가 있다.

- [boxing__to__traffic_control1__t01500__direct_cut__window.mp4](videos/boxing__to__traffic_control1__t01500__direct_cut__window.mp4)
- [boxing__to__traffic_control1__t01500__policy__window.mp4](videos/boxing__to__traffic_control1__t01500__policy__window.mp4)
- [full__boxing__to__traffic_control1__t01500__direct_cut.mp4](videos/full__boxing__to__traffic_control1__t01500__direct_cut.mp4)
- [full__boxing__to__traffic_control1__t01500__policy.mp4](videos/full__boxing__to__traffic_control1__t01500__policy.mp4)
- [traffic_control1__to__traffic_control2__t00500__direct_cut__window.mp4](videos/traffic_control1__to__traffic_control2__t00500__direct_cut__window.mp4)
- [traffic_control1__to__traffic_control2__t00500__policy__window.mp4](videos/traffic_control1__to__traffic_control2__t00500__policy__window.mp4)

## 재생성

```bash
python tools/archive_switch_results.py
python tools/render_live_switch_audit.py runs/long_switch_live_audit_54_20261104 --case boxing__to__traffic_control1__t01500 --window 2 4
```

기존 kimodo2 환경의 NumPy/SciPy/PyTorch/Matplotlib를 사용했다. 영상 렌더 명령은 출력이 이미 있으면 덮어쓰지 않는다. 실제 torque 기록은 없어 torque jerk나 하드웨어 충격을 측정했다고 주장하지 않는다. 54개 정상 전환의 첫 2.54초 state/action 원본은 source run의 `wm_audit_inputs.pt`, 대표 8회의 전체 pose는 `videos.npz`에 있다.
