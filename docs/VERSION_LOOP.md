# ShadowCrafter-9B 단일 평가·릴리스

이 워크플로는 현재 28,140건 확장 학습이 끝난 뒤 `v1.0`에 대해 다음을 한 번만
수행합니다.

1. 체크포인트와 실행 manifest를 다시 해시하고 안전한 LoRA surface를 검증합니다.
2. 고정된 CTIBench 5,533건에서 accuracy, balanced accuracy, macro-F1을 전체 및
   과제별로 다시 계산합니다.
3. 실제 점수를 적은 public noncommercial `Official Release`를
   `KaztoRay/ShadowCrafter-9B`의 `releases/vN.0/` 경로에 올리고 동일한 Git tag를
   생성합니다.
4. 원격 프로젝트, 기반 모델, 체크포인트와 완료 가중치를 Git에서 제외된
   `local_mirror/`에 증분 동기화합니다.
5. 실제 정확도와 전체·과제별 지표 및 `95% target met` 상태를 기록한 뒤 종료합니다.

점수가 0.95보다 낮아도 결과를 숨기거나 자동 재학습하지 않습니다. `v2.0` 이후
candidate를 만드는 코드 경로는 비활성화되어 있으며, `v1.0`을 실제 점수와
`95% target met: no` 표기 그대로 public Official Release로 게시합니다.

로컬 watcher는 Hugging Face 자격 증명을 원격 GPU에 보내지 않습니다. 원격의
manifest-검증 adapter를 메모리로 스트리밍해 public 저장소에 올린 뒤 Hub에서
체크섬을 다시 확인합니다. 게시 경로와 별도로 전체 원격 프로젝트는
`local_mirror/remote-project/`에 내려받으며 Git 추적과 원격 재업로드에서 제외합니다.

자동화는 `v1.0` 한 개만 허용하며 `--start-version` 또는 `--max-version`을 다른 값으로
지정하면 fail-closed로 거부합니다. 추가 학습은 현재 운영 범위에 포함되지 않습니다.
