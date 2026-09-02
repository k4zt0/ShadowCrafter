# VS Code에서 ShadowCrafter V2 Colab 학습

## 계정 없는 실행

notebooks/ShadowCrafter_V2_Colab.ipynb는 다음 연결이나 비밀을 요구하지 않는다.

- Google Drive mount
- private GitHub clone 또는 GITHUB_TOKEN
- Hugging Face 로그인 또는 HF_TOKEN
- 학습 중 Hub 업로드, W&B, 외부 telemetry

공개 Ornith 기반 모델과 NIST Juliet 자료는 고정 revision/SHA-256을 사용해 익명 HTTPS로
내려받는다. 최소 실행 Git bundle, v1 28,140건 학습 입력, CTIBench 5,533건 평가 입력은
노트북 안에 압축되어 있으며 복원 전후 SHA-256을 확인한다. CTIBench는 오염 검사에만
사용하고 학습하지 않는다.

재현 가능한 노트북은 scripts/build-accountless-colab-notebook.py로 생성한다. 생성기는
고정된 로컬 입력 해시를 확인하고, 필요한 source/config만 담은 detached Git bundle을
만들어 notebook에 포함한다. token이나 credential은 포함하지 않는다.

## 확장과 GPU 연결

공식 Google Colab VS Code 확장 ID는 google.colab이다. 프로젝트의
.vscode/extensions.json은 이 확장과 ms-toolsai.jupyter, ms-python.python을 추천한다.

1. VS Code에서 notebooks/ShadowCrafter_V2_Colab.ipynb를 연다.
2. 오른쪽 위 Select Kernel을 누른다.
3. Colab → Assign New Server... → GPU에서 A100 40GB 이상 서버를 할당한다.
4. 첫 셀이 nvidia-smi, VRAM 38,000 MiB 이상, /content 여유 공간 45GiB 이상을
   확인하는지 본다.
5. 위에서 아래로 실행한다. 별도 계정 연결 셀이나 파일 업로드 단계는 없다.

Auto Connect는 CPU 서버를 선택할 수 있다. CPU 또는 VRAM 부족 오류가 나오면 현재
kernel을 누르고 Select Another Kernel...에서 GPU 서버를 다시 할당한다.

## 저장·재개·내보내기

모든 파일은 /content/ShadowCrafterV2에 저장된다. checkpoint는 100 step마다 생성하며
adapter, optimizer, scheduler, RNG, trainer state를 해시한 완전성 마커를 마지막에
기록한다. 같은 살아 있는 runtime에서는 최신 hash-valid checkpoint에서 재개한다.

계정 없는 방식은 Colab runtime이 완전히 종료되면 /content 파일도 사라진다. 마지막
export 셀은 완료된 candidate를 우선하고, 아직 학습 중이면 checkpoint 디렉터리를
tar.gz로 만든 뒤 VS Code notebook에 로컬 다운로드 링크를 표시한다. 세션 종료 전에
반드시 다운로드해야 한다.

Optimizer state는 PyTorch 직렬화를 포함하므로 본인이 생성하고 해시를 확인한
checkpoint만 재개한다. 완료 candidate는 별도 검증 환경에서 CTIBench 5,533건을 평가한
후 실제 점수와 함께 public v2.0으로 게시한다.
