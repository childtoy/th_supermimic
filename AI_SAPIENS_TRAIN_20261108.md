# ai_sapiens K1 세 모션의 SuperMimic 학습

> 중단: 사용자는 robot→human 변환 및 transition 생성을 먼저 수행하려던 것이었다. 단독 세 모션 추종으로 잘못 시작한 run은 중단했고, 산출물은 보존했다. 이 run은 사용자가 의도한 transition 학습이 아니다.

2026-11-08. 사용자 요청에 따라 Dance 1 / Dance 2 / Squat을 하나의 새 policy/WM으로 함께 학습하기 시작했다. 세 모션 사이 transition reference는 이번 run에 포함하지 않았다. 기존 ai_sapiens ONNX policy를 변환하거나 warm-start한 것이 아니다.

## 입력 확인

- source: `../ai_sapiens/ai_sapiens_sim2real/assets/k1/mimic/{dance1,dance2,squat}/params/`.
- 각각 300/299/299 frame, 50Hz, 약 6초. 30열: root xyz + quaternion xyzw + 23 joints.
- ai_sapiens의 `motion_reference.cpp`와 `k1_config.yaml`을 확인했다. header 없는 CSV의 fallback `robot_joint_order`가 SuperMimic K1 순서와 정확히 일치한다. `policy_joints`는 ONNX 입출력 순서이며 이 CSV의 순서로 대신 쓰지 않는다.
- 유한값, 열 수, quaternion norm을 검사했다. norm 최대 오차는 2.3e-7 이하.
- 원본 bytes를 변경하지 않고 실험 data 폴더에 복사했다. 재리타겟팅/좌표변환/재샘플링 없음. 원본 ONNX와 CSV는 수정하지 않았다.
- source hash, joint names, root 높이 범위는 `data/manifest.json`에 보존했다. 이 구조 검사는 원본 reference의 물리 적합성을 보장하지 않으므로 full tracking 평가를 별도로 수행한다.

## 실행

- root: `runs/ai_sapiens_three_20261108/`.
- Docker `th_u2204cuda128_kimodo2`, Python `/root/anaconda3/envs/env_kimodo_inb/bin/python`, GPU 0.
- 먼저 작은 8-env / 3-update smoke가 완료됐고 본 학습 프로세스가 시작됐다.
- 본 학습: 4,096 env, fresh policy/WM/critic, 12,000 iteration, seed 0.
- WM-only 500회, horizon 32, history/goal frames 10, actor LR 5e-5, WM/critic LR 2e-4, critic upper bound, BF16/compile.
- 500회마다 세 clip을 처음부터 끝까지 평가하고 실패 수 우선의 best checkpoint를 선택한다.
- 초기 무학습 eval 0은 return 약 0.1722 / 실패 3개였다. 학습 성과로 해석하지 않는다.
- 종료 후 best checkpoint로 세 clip의 실제 물리/reference 비교 영상을 자동 생성한다.

```bash
cat runs/ai_sapiens_three_20261108/status.txt
tail -f runs/ai_sapiens_three_20261108/pipeline.log
```

단계는 smoke → train → video → completed 순서다. 실패하면 status에 stage/exit code가 기록된다. `train/best_coverage.pt`, `train/evals/`, `videos/`가 산출 위치다. 현재는 시작 확인이며 수렴/완주 성공을 보고한 것이 아니다.

재현 도구는 `tools/prepare_ai_sapiens.py`, `scripts/train_ai_sapiens.sh`다. 기존 결과를 덮어쓰지 않도록 새 실험 root와 data 폴더를 사용한다.
