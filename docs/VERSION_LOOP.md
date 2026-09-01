# ShadowCrafter-9B 95% version loop

이 워크플로는 현재 28,140건 확장 학습이 끝난 뒤 `v1.0`부터 순차적으로 다음을
수행합니다.

1. 체크포인트와 실행 manifest를 다시 해시하고 안전한 LoRA surface를 검증합니다.
2. 고정된 CTIBench 5,533건에서 accuracy, balanced accuracy, macro-F1을 전체 및
   과제별로 다시 계산합니다.
3. 실제 점수를 적은 private `Experimental Release`를
   `KaztoRay/ShadowCrafter-9B`의 `releases/vN.0/` 경로에 올리고 동일한 Git tag를
   생성합니다.
4. 모든 전체·과제별 지표가 0.95 이상이면 종료합니다. 미달이면 승인된 학습
   코퍼스만 사용해 다음 하이퍼파라미터 후보를 학습하고 `v(N+1).0`으로 반복합니다.

평가 정답, 원시 실패 문장, predictions는 어떤 재학습 입력에도 포함하지 않습니다.
반복 탐색은 사전에 고정된 epoch, learning-rate, LoRA rank/alpha, seed 조합만
사용합니다. 따라서 95%는 목표이지 보장이 아니며, 점수가 낮은 버전도 실제 점수와
`95% target met: no`를 표시합니다.

로컬 watcher는 Hugging Face 자격 증명을 원격 GPU에 보내지 않습니다. 원격의
manifest-검증 adapter를 메모리로 스트리밍해 private 저장소에 올린 뒤 Hub에서
체크섬을 다시 확인합니다. 로컬에는 평가 증거만 내려오며 `.safetensors` 등 모델
weight가 발견되면 즉시 중단합니다.

기본 자동화 상한은 32개 버전입니다. 이는 무제한 GPU 소비와 디스크 고갈을 막는
운영 안전장치이며 32개 모두 목표 미달이면 실제 결과를 검토한 뒤 새로운 독립
학습 데이터 또는 모델 설계 변경을 승인해야 합니다. 같은 평가 문항을 학습시키거나
임계값만 낮춰 재개해서는 안 됩니다.
