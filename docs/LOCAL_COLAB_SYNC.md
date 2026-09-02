# Colab 결과를 사용자 컴퓨터에서 동기화·게시

Colab은 학습, CTIBench 동결 평가와 export만 수행한다. GitHub와 Hugging Face 인증, 코드 push, 모델
Official Release 게시 및 게시 후 검증은 사용자 컴퓨터에서만 수행한다. 어떤 로컬
credential도 Colab notebook이나 export archive에 복사하지 않는다.

## 최초 1회 로컬 인증

GitHub private 저장소와 Hugging Face public 모델 저장소의 현재 계정을 확인한다.

    gh auth status
    hf auth whoami

Hugging Face token은 로컬 표준 credential 저장소에서만 읽는다. 명령행 인자, notebook,
환경 출력, Git 파일에 token 값을 넣지 않는다.

## Colab export 가져오기

Notebook 마지막 셀의 링크로 ShadowCrafter-V2-candidate-*.tar.gz 또는
ShadowCrafter-V2-checkpoints-*.tar.gz를 내려받은 뒤 실행한다.

    .venv/bin/shadowcrafter release import-colab-export \
      --archive ~/Downloads/ShadowCrafter-V2-candidate-<identity>.tar.gz \
      --destination-root local_mirror/colab-v2

가져오기 명령은 archive SHA-256, 경로 탈출, 링크·특수 파일, 압축 해제 크기,
run-manifest, 기반 모델 revision, LoRA safetensors/config SHA-256을 확인한다. 완료
candidate는 local_mirror/colab-v2/candidate 아래, 중간 checkpoint는 checkpoints 아래에
충돌 없는 해시 경로로 보관한다. 같은 archive 재실행은 기존 receipt를 검증하고
idempotent하게 종료한다. 측정 완료 archive에는 candidate 아래
`evaluation/<evaluation-id>/gate-report.json`과 `evidence/release-evidence.json`도 있으며,
각 지표는 원시 예측에서 재계산할 수 있다.

local_mirror는 Git 추적에서 제외되므로 모델·optimizer 파일이 private GitHub source
저장소에 섞이지 않는다.

## 코드 push

코드는 일반 Git 검토 흐름으로 private Odytssey/ShadowCrafter에 올린다.

    git status --short
    gh repo view Odytssey/ShadowCrafter --json visibility
    git push origin main

visibility가 PRIVATE가 아니거나 worktree에 예상하지 않은 변경이 있으면 push 전에
중단한다.

## 정확도 평가 후 로컬 모델 게시

candidate를 그대로 공개하지 않는다. Colab에서 생성한 CTIBench 고정 평가, 오염 검사,
artifact integrity, provenance, license, privacy, safety evidence와 measured gate report를
로컬에서 다시 검증한다.
정확도가 95% 미만인 것은 게시 차단 사유가 아니지만 실제 accuracy, balanced accuracy,
macro-F1과 quality_target_met=false를 모델 카드에 기록해야 한다.

검토된 release 파일, remote-release-manifest, release-evidence가 로컬에 준비되면 다음
명령으로 사용자 컴퓨터가 public Hugging Face commit을 생성하고 모든 파일을 다시
다운로드해 SHA-256을 검증한다.

    .venv/bin/shadowcrafter release publish-local-official \
      --manifest <remote-release-manifest.json> \
      --manifest-sha256 <manifest-sha256> \
      --artifact-root <local-reviewed-release-root> \
      --evidence <release-evidence.json>

게시기는 manifest의 parent commit이 현재 Hub main과 다르면 경합으로 중단하며,
저장소가 public이 아니거나 승인·평가 증거가 일치하지 않아도 중단한다. Hub token은
사용자 컴퓨터에서 Hugging Face API로만 전송되고 Colab, Vessl 또는 GitHub에는 전달하지
않는다.
