# V8 primitive + transition SuperMimic 학습

## 완료 및 웹 확인 (2026-11-09)

12000회 학습과 후속 9개 rollout 영상 저장 완료. 마지막 eval_012000에서
75개 실패 0, norm_return=0.9651939. 이는 저장된 full clip 평가이며 임의 시점
온라인 전환 전체를 보장하지 않는다.

`best_coverage.pt`와 같은 run의 data75개로 기존 웹서버를 실행했다.
주소: `http://163.152.162.50:20031` (컨테이너6006 → 호스트20031).
Dance1/Dance2/Squat의 세 버튼으로 명령하며, 72개 bank에서 현재 시점과 가까운
연결을 선택한다. 임의의 75개 파일을 선택하는 UI가 아니라 primitive 명령 UI다.
로그: `runs/ai_sapiens_v8_transitions_20261108/web.log`.
새 primitive__ 이름을 CommandBank에서 인식하도록 확장했고 기존/신규4개 검사를 통과했다.
HTTP state/JPEG 및 Dance1 실행→Squat 전환 명령을 확인했다.
화면 렌더러는 llvmpipe여서 약2fps이며, 제어와 렌더링은 분리되어 있다.

사용자의 2026-11-08 학습 지시로 새 실험
`runs/ai_sapiens_v8_transitions_20261108/`를 시작했다.
이전 중단된 primitive-only `ai_sapiens_three_20261108`과 별개다.

- 입력: `../RMAI/runs/ai_sapiens_v8_transitions_05_20261108/robots/`.
- 같은 고정 H2R로 변환한 V8 primitive 3개 + 전환 72개 = 75 clip.
- 전환은 전체 A prefix → 생성 bridge → B를 학습한다. bridge만 분리하지 않는다.
- A 끝 기준 0.5초 간격 cutback 12개 × 6방향, B 진입은 frame 0 고정.
- 후보 0.5/1.0/1.5초 중 alignment가 고른 bridge는 모두 1.5초다.
- 원본 ai_sapiens CSV/ONNX가 아닌 V8 H2R 결과를 사용한다. fresh policy/WM/critic이며 warm start 없다.
- `prepare_ai_sapiens_transitions.py`에서 기존 `convert_file`로 관절 이름 순서 확인,
  root quaternion wxyz→xyzw 변환 및 50Hz resampling했다. source hash 및 bridge 시간은 data manifest에 기록했다.

실행 환경: Docker `th_u2204cuda128_kimodo2`,
`/root/anaconda3/envs/env_kimodo_inb/bin/python`, GPU 0, mjwarp.
8환경/3회 smoke를 통과했다. 이후 본 학습은 4096환경/12000회,
WM-only 500회, horizon32, seed0, WM/critic LR2e-4, actor LR5e-5,
critic upper bound를 사용한다. 기존 학습 설정을 유지했다.
500회마다 75개 full clip을 평가해 best coverage checkpoint를 저장한다.
종료 후 primitive 3개 + cutback2.5초의 6방향 전환에 대한 물리 rollout 영상을 자동 생성한다.
데이터의 경계 불연속이 남아 있으므로 학습 실행을 추종 성공으로 해석하지 않는다.

```bash
cat runs/ai_sapiens_v8_transitions_20261108/status.txt
tail -f runs/ai_sapiens_v8_transitions_20261108/pipeline.log
```

실제 진행은 `train/train_log.csv`, 평가 결과는 `train/evals/`,
선정 checkpoint는 `train/best_coverage.pt`, 영상은 `videos/`에 저장된다.
재현 script는 `scripts/train_ai_sapiens_transitions.sh`다.
