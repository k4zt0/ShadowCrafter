# VS Code에서 ShadowCrafter V2 Colab 학습

## 확장과 연결

공식 Google Colab VS Code 확장 ID는 `google.colab`이다. 프로젝트의
`.vscode/extensions.json`은 이 확장과 `ms-toolsai.jupyter`, `ms-python.python`을
추천한다.

1. VS Code에서 `notebooks/ShadowCrafter_V2_Colab.ipynb`를 연다.
2. 오른쪽 위 `Select Kernel`을 누른다.
3. `Colab` → `Assign New Server...` → `GPU`에서 A100 40GB 이상 서버를
   할당한다. `Auto Connect`가 CPU 서버를 선택했다면 현재 kernel을 누르고
   `Select Another Kernel...`에서 GPU 서버를 다시 할당한다.
4. 첫 셀이 `nvidia-smi`와 VRAM 38,000 MiB 이상을 확인하는지 본다.
5. Command Palette의 `Colab: Mount Google Drive to Server...`를 사용하거나 notebook의
   Drive mount 셀을 실행한다.

Colab GPU 종류와 사용 시간은 보장되지 않는다. 일반 runtime은 최대 12시간,
Pro+의 continuous execution도 compute unit이 충분할 때 최대 24시간이므로 전체 V2
학습은 여러 session에 걸쳐 재개될 수 있다.

## 비밀

Colab Secrets에 read-only `GITHUB_TOKEN`만 추가한다. fine-grained token은 private
`Odytssey/ShadowCrafter` repository contents read 권한으로 제한한다. token 값을
notebook, Drive, Git, 셀 출력 또는 URL에 기록하지 않는다. 학습 중 Hugging Face 업로드는
수행하지 않으므로 `HF_TOKEN`은 필요하지 않다.

## 최초 입력 업로드

다음 로컬 파일 두 개를 Google Drive의 `MyDrive/ShadowCrafterV2/inputs/` 아래에 한 번
업로드한다.

| Drive 경로 | 로컬 원본 |
|---|---|
| `v1/train.jsonl` | `local_mirror/remote-project/data/processed/security-expanded-20260901-v8-blackbox-train-only/train.jsonl` |
| `ctibench/cases.jsonl` | `local_mirror/remote-project/artifacts/evaluations/ctibench-9237e163/cases.jsonl` |

Notebook은 사용 전에 각각 고정 SHA-256과 레코드 수를 검증한다. CTIBench case는 오염
검사에만 사용하며 학습 데이터에 포함하지 않는다. NIST Juliet ZIP은 notebook이 NIST
공식 HTTPS 주소에서 내려받고 공식 SHA-256을 검증한다.

## 재개와 보관

최초 실행 시 GitHub `origin/main`의 exact commit을
`MyDrive/ShadowCrafterV2/SOURCE_REVISION`에 고정한다. 이후 session은 main이 바뀌어도
이 commit만 detached checkout한다.

Checkpoint는 `MyDrive/ShadowCrafterV2/checkpoints/<source>-<dataset>/`에 100 step마다
저장한다. 저장 완료 callback은 adapter, optimizer, scheduler, RNG, trainer state를 모두
해시한 `.shadowcrafter-complete.json`을 마지막에 기록한다. 마커가 없는 중단 파일은
`incomplete/`로 이동해 보존하고, 가장 최신의 hash-valid checkpoint만 재개한다.

Optimizer state는 PyTorch 직렬화를 포함하므로 checkpoint marker는 공격자에 대한
서명이 아니다. 공유 Drive나 제3자가 쓸 수 있는 디렉터리의 checkpoint를 재개하지 않는다.

학습 완료본은 `MyDrive/ShadowCrafterV2/candidates/v2.0-<identity>/`에 LoRA adapter와
`run-manifest.json`으로 원자적 승격된다. 이후 별도 검증 환경에서 CTIBench 5,533건을
평가하고 실제 점수를 적은 public `v2.0`만 게시한다.
