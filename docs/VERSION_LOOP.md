# ShadowCrafter-9B 평가·릴리스

v1.0은 28,140건 확장 학습 뒤 CTI-Bench 5,533건 평가와 public noncommercial
Official Release 게시를 완료했다. 측정 accuracy는 46.267847%, balanced accuracy는
11.099532%, macro-F1은 6.748866%였다.

V2는 다음 순서로 별도 immutable candidate를 만든다.

1. 공식 SHA-256으로 고정한 NIST Juliet 64,099건을 source-only로 canonicalize한다.
2. v1 corpus와 합친 92,239건에서 중복·비밀·CTI-Bench 오염 검사를 수행한다.
3. 깨끗한 detached Git snapshot, 고정 base model과 manifest로 새 QLoRA adapter를
   처음부터 학습한다. v1 평가 답안이나 예측은 학습 입력에 사용하지 않는다.
4. 완료 checkpoint와 실행 manifest를 다시 해시하고 안전한 LoRA surface를 검증한다.
5. 고정된 CTI-Bench 5,533건에서 accuracy, balanced accuracy, macro-F1을 전체 및
   과제별로 다시 계산한다.
6. 실제 점수를 적은 public noncommercial `v2.0` Official Release를
   `KaztoRay/ShadowCrafter-9B`에 올리고 동일한 Git tag를 생성한다.
7. 원격 프로젝트와 완료 가중치를 `local_mirror/`에 증분 동기화한다.

점수가 0.95보다 낮아도 결과를 숨기지 않으며 V2 점수는 학습 완료 후에만 측정한다.
같은 평가셋 점수를 보고 자동 반복 재학습하지 않는다.

로컬 watcher는 Hugging Face 자격 증명을 원격 GPU에 보내지 않습니다. 원격의
manifest-검증 adapter를 메모리로 스트리밍해 public 저장소에 올린 뒤 Hub에서
체크섬을 다시 확인합니다. 게시 경로와 별도로 전체 원격 프로젝트는
`local_mirror/remote-project/`에 내려받으며 Git 추적과 원격 재업로드에서 제외합니다.

V2 학습은 원격 detached worker의 `run-v2-training.py` 또는 공식 Google Colab
VS Code 확장에서 여는 `notebooks/ShadowCrafter_V2_Colab.ipynb`를 사용한다. Colab
경로는 동일 source/data/config hash에 묶인 Drive checkpoint만 재개하며, optimizer
state를 포함하므로 본인만 접근 가능한 private Drive를 신뢰 경계로 요구한다. 어느
경로에서도 Hugging Face 공개 게시 자격 증명은 학습 런타임에 전달하지 않고 로컬 게시
단계에만 유지한다.
