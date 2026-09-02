#!/usr/bin/env python3
"""Build the accountless ShadowCrafter V2 Colab notebook with pinned embedded inputs."""

from __future__ import annotations

import base64
import gzip
import hashlib
import io
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
GIT = shutil.which("git")
if GIT is None:
    raise RuntimeError("git is required to build the Colab notebook")
OUTPUT = ROOT / "notebooks/ShadowCrafter_V2_Colab.ipynb"
V1_INPUT = (
    ROOT
    / "local_mirror/remote-project/data/processed/"
    "security-expanded-20260901-v8-blackbox-train-only/train.jsonl"
)
CTIBENCH_INPUT = (
    ROOT
    / "local_mirror/remote-project/artifacts/evaluations/"
    "ctibench-9237e163/cases.jsonl"
)
V1_SHA256 = "8b0be9434be7452bf8129650eec485a00d2ce3efabeb725dc2f81908e18b7c7f"
CTIBENCH_SHA256 = "2455b46b4851ed998ce3094ba7d9f796365bd0d71ce51264ff665f1c5203b423"
RUNTIME_PATHS = (
    "LICENSE",
    "README.md",
    "pyproject.toml",
    "src",
    "configs",
    "requirements/train-hf.lock.txt",
    "artifacts/manifests/ornith-1.5-9b.json",
)


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _read_pinned(path: Path, expected_sha256: str) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"missing regular input: {path}")
    content = path.read_bytes()
    observed = _sha256(content)
    if observed != expected_sha256:
        raise RuntimeError(f"input SHA-256 mismatch for {path}: {observed}")
    return content


def _git(*arguments: str, cwd: Path = ROOT, env: dict[str, str] | None = None) -> str:
    result = subprocess.run(  # noqa: S603 - git is resolved and callers are fixed below.
        [GIT, *arguments],
        cwd=cwd,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _runtime_bundle(temporary_root: Path) -> tuple[bytes, str, str]:
    upstream_revision = _git("rev-parse", "HEAD")
    archive = subprocess.run(  # noqa: S603 - git and archive paths are fixed.
        [GIT, "archive", "--format=tar", "HEAD", *RUNTIME_PATHS],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout
    runtime_root = temporary_root / "runtime"
    runtime_root.mkdir()
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as handle:
        handle.extractall(runtime_root, filter="data")
    provenance = {
        "schema_version": 1,
        "upstream_repository": "Odytssey/ShadowCrafter",
        "upstream_revision": upstream_revision,
        "purpose": "accountless Colab V2 training runtime",
    }
    (runtime_root / "BUNDLED_SOURCE.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _git("init", "--quiet", cwd=runtime_root)
    _git("config", "user.name", "Odytssey Runtime Builder", cwd=runtime_root)
    _git("config", "user.email", "runtime@odytssey.invalid", cwd=runtime_root)
    _git("config", "commit.gpgsign", "false", cwd=runtime_root)
    _git("add", "--all", cwd=runtime_root)
    commit_env = os.environ.copy()
    commit_env.update(
        {
            "GIT_AUTHOR_NAME": "Odytssey Runtime Builder",
            "GIT_AUTHOR_EMAIL": "runtime@odytssey.invalid",
            "GIT_COMMITTER_NAME": "Odytssey Runtime Builder",
            "GIT_COMMITTER_EMAIL": "runtime@odytssey.invalid",
            "GIT_AUTHOR_DATE": "2000-01-01T00:00:00Z",
            "GIT_COMMITTER_DATE": "2000-01-01T00:00:00Z",
        }
    )
    _git(
        "commit",
        "--quiet",
        "-m",
        "ShadowCrafter accountless Colab runtime",
        cwd=runtime_root,
        env=commit_env,
    )
    runtime_revision = _git("rev-parse", "HEAD", cwd=runtime_root)
    bundle_path = temporary_root / "shadowcrafter-runtime.bundle"
    _git("bundle", "create", str(bundle_path), "HEAD", cwd=runtime_root)
    return bundle_path.read_bytes(), runtime_revision, upstream_revision


def _deterministic_gzip(content: bytes) -> bytes:
    output = io.BytesIO()
    with gzip.GzipFile(fileobj=output, mode="wb", compresslevel=9, mtime=0) as handle:
        handle.write(content)
    return output.getvalue()


def _source(text: str) -> list[str]:
    return text.splitlines(keepends=True)


def _code_cell(text: str, *, hidden: bool = False) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    if hidden:
        metadata = {
            "jupyter": {"source_hidden": True},
            "tags": ["embedded-accountless-bootstrap"],
        }
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": metadata,
        "outputs": [],
        "source": _source(text.rstrip() + "\n"),
    }


def _markdown_cell(text: str) -> dict[str, Any]:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": _source(text.rstrip() + "\n"),
    }


def _b85_assignment(name: str, content: bytes) -> str:
    encoded = base64.b85encode(content).decode("ascii")
    chunks = (encoded[index : index + 65_536] for index in range(0, len(encoded), 65_536))
    return name + " = (\n" + "".join(f"    {chunk!r}\n" for chunk in chunks) + ")\n"


def _build_notebook(
    *,
    runtime_bundle: bytes,
    runtime_revision: str,
    upstream_revision: str,
    v1_gzip: bytes,
    ctibench_gzip: bytes,
) -> dict[str, Any]:
    runtime_compressed_sha = _sha256(runtime_bundle)
    v1_compressed_sha = _sha256(v1_gzip)
    ctibench_compressed_sha = _sha256(ctibench_gzip)
    embedded = (
        "# 2. 계정 없는 내장 runtime/data 복원 — source는 접혀 있어도 그대로 실행됩니다.\n"
        "import base64, gzip, hashlib, uuid\n\n"
        + _b85_assignment("RUNTIME_B85", runtime_bundle)
        + _b85_assignment("V1_B85", v1_gzip)
        + _b85_assignment("CTIBENCH_B85", ctibench_gzip)
        + f"""
RUNTIME_COMPRESSED_SHA256 = {runtime_compressed_sha!r}
V1_COMPRESSED_SHA256 = {v1_compressed_sha!r}
CTIBENCH_COMPRESSED_SHA256 = {ctibench_compressed_sha!r}
V1_SHA256 = {V1_SHA256!r}
CTI_SHA256 = {CTIBENCH_SHA256!r}
SOURCE_REVISION = {runtime_revision!r}
UPSTREAM_SOURCE_REVISION = {upstream_revision!r}

def decode_verified(encoded: str, expected_sha256: str, label: str) -> bytes:
    payload = base64.b85decode(encoded.encode('ascii'))
    observed = hashlib.sha256(payload).hexdigest()
    if observed != expected_sha256:
        raise RuntimeError(f'{{label}} embedded payload SHA-256 mismatch: {{observed}}')
    return payload

def write_atomic(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != content:
            raise RuntimeError(f'기존 bootstrap 파일이 검증값과 다릅니다: {{path}}')
        return
    temporary = path.parent / f'.{{path.name}}.tmp-{{uuid.uuid4().hex}}'
    with temporary.open('xb') as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)

BUNDLE_PATH = SESSION_ROOT / 'bootstrap/shadowcrafter-runtime.bundle'
runtime_payload = decode_verified(RUNTIME_B85, RUNTIME_COMPRESSED_SHA256, 'runtime')
write_atomic(BUNDLE_PATH, runtime_payload)
REPO_DIR = SESSION_ROOT / 'project'
if not REPO_DIR.exists():
    subprocess.run(['git', 'clone', '--quiet', str(BUNDLE_PATH), str(REPO_DIR)], check=True)
observed_revision = subprocess.run(
    ['git', '-C', str(REPO_DIR), 'rev-parse', 'HEAD'],
    check=True, capture_output=True, text=True,
).stdout.strip()
observed_status = subprocess.run(
    ['git', '-C', str(REPO_DIR), 'status', '--porcelain'],
    check=True, capture_output=True, text=True,
).stdout
if observed_revision != SOURCE_REVISION or observed_status:
    raise RuntimeError('내장 source checkout이 고정 detached revision과 다릅니다.')

def restore_gzip(encoded: str, compressed_sha256: str, raw_sha256: str, target: Path) -> None:
    compressed = decode_verified(encoded, compressed_sha256, target.name)
    raw = gzip.decompress(compressed)
    if hashlib.sha256(raw).hexdigest() != raw_sha256:
        raise RuntimeError(f'{{target.name}} 원본 SHA-256이 다릅니다.')
    write_atomic(target, raw)

V1_LOCAL = SESSION_ROOT / 'inputs/v1/train.jsonl'
CTI_LOCAL = SESSION_ROOT / 'inputs/ctibench/cases.jsonl'
restore_gzip(V1_B85, V1_COMPRESSED_SHA256, V1_SHA256, V1_LOCAL)
restore_gzip(CTIBENCH_B85, CTIBENCH_COMPRESSED_SHA256, CTI_SHA256, CTI_LOCAL)
del RUNTIME_B85, V1_B85, CTIBENCH_B85, runtime_payload
print('Bundled source revision:', SOURCE_REVISION)
print('Upstream private GitHub revision (연결하지 않음):', UPSTREAM_SOURCE_REVISION)
print('Restored inputs:', V1_LOCAL, CTI_LOCAL)
"""
    )
    cells = [
        _markdown_cell(
            """# ShadowCrafter-9B V2 — 계정 연결 없는 VS Code + Google Colab

이 노트북은 Google Drive 마운트, private GitHub clone, Hugging Face 로그인을 요구하지
않습니다. Select Kernel → Colab → Assign New Server... → GPU로 A100 40GB+ 서버만
연결한 뒤 위에서 아래로 실행하세요.

- 기반 모델: 공개 ornith-ai/Ornith-1.5-9B exact revision을 토큰 없이 익명 다운로드
- 내장 학습 자료: v1 28,140건 + 공개 NIST Juliet C/C++ 64,099건 = 92,239건
- 내장 평가 자료: CTIBench 5,533건은 오염 검사에만 사용하고 학습하지 않음
- 내장 실행 코드와 입력은 복원 전후 SHA-256으로 검증
- 학습 중 Hub 업로드, telemetry, W&B 보고를 사용하지 않음"""
        ),
        _markdown_cell(
            """## 저장 방식과 제한

모든 데이터, 체크포인트, candidate는 Colab의 /content/ShadowCrafterV2에 저장됩니다.
계정 연결이 없는 대신 런타임이 완전히 종료되면 이 파일은 사라집니다. 같은 살아 있는
세션에서는 무결성 마커가 있는 최신 checkpoint에서 재개할 수 있습니다.

학습 완료 또는 세션 종료 전에 마지막 export 셀을 실행해 candidate/checkpoint 압축본을
로컬로 다운로드하세요. 기반 모델 다운로드에는 공개 Hugging Face HTTPS가 필요하지만
HF_TOKEN, 로그인, repository 연동은 사용하지 않습니다."""
        ),
        _code_cell(
            """# 1. Colab GPU 및 임시 작업공간 확인
import json, os, shutil, subprocess, sys
from pathlib import Path

if not Path('/content').is_dir():
    raise RuntimeError('이 노트북은 Google Colab 런타임에서 실행해야 합니다.')
nvidia_smi = shutil.which('nvidia-smi')
if nvidia_smi is None:
    raise RuntimeError(
        '현재 Colab 서버는 CPU runtime입니다. VS Code 오른쪽 위의 현재 kernel을 누르고 '
        'Select Another Kernel... → Colab → Assign New Server... → GPU로 A100 40GB+ '
        '서버를 할당한 뒤 1번 셀부터 다시 실행하세요.'
    )
gpu = subprocess.run(
    [nvidia_smi, '--query-gpu=index,name,memory.total', '--format=csv,noheader,nounits'],
    check=True, capture_output=True, text=True,
).stdout.strip()
print('GPU:', gpu)
gpu_rows = [line.rsplit(',', 1) for line in gpu.splitlines() if line.strip()]
memory_mib = max(int(memory.strip()) for _, memory in gpu_rows)
if memory_mib < 38_000:
    raise RuntimeError(
        f'현재 GPU의 VRAM은 {memory_mib} MiB입니다. 공식 V2 설정은 A100 40GB급 '
        'GPU(VRAM 38,000 MiB 이상)가 필요합니다.'
    )
free_bytes = shutil.disk_usage('/content').free
if free_bytes < 45 * 1024**3:
    raise RuntimeError(f'/content 여유 공간이 부족합니다: {free_bytes / 1024**3:.1f} GiB')
for token_name in ('HF_TOKEN', 'HUGGING_FACE_HUB_TOKEN', 'GITHUB_TOKEN'):
    os.environ.pop(token_name, None)
os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN'] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
SESSION_ROOT = Path('/content/ShadowCrafterV2')
SESSION_ROOT.mkdir(mode=0o700, parents=True, exist_ok=True)
print('Accountless ephemeral root:', SESSION_ROOT)
print('Free disk GiB:', round(free_bytes / 1024**3, 1))"""
        ),
        _code_cell(embedded, hidden=True),
        _code_cell(
            """# 3. 감사된 학습 runtime 설치 및 버전 확인
subprocess.run(
    [
        sys.executable, '-m', 'pip', 'install', '--quiet', '-r',
        str(REPO_DIR / 'requirements/train-hf.lock.txt'),
    ],
    check=True,
)
subprocess.run([sys.executable, '-m', 'pip', 'install', '--quiet', '-e', str(REPO_DIR)], check=True)
from importlib.metadata import version
EXPECTED = {
    'accelerate': '1.14.0', 'bitsandbytes': '0.50.2', 'datasets': '4.8.5',
    'huggingface-hub': '1.29.0', 'peft': '0.19.1', 'safetensors': '0.8.0',
    'torch': '2.10.0', 'transformers': '5.12.1', 'trl': '0.29.1',
}
observed = {name: version(name) for name in EXPECTED}
mismatch = {
    name: {'expected': EXPECTED[name], 'actual': value}
    for name, value in observed.items() if value != EXPECTED[name]
}
if mismatch:
    raise RuntimeError(
        '패키지 버전 불일치입니다. Kernel을 재시작하고 1번 셀부터 다시 '
        f'실행하세요: {mismatch}'
    )
print(json.dumps(observed, indent=2, sort_keys=True))"""
        ),
        _code_cell(
            """# 4. 공개 Ornith base model을 로그인 없이 내려받고 18개 파일 검증
import hashlib
from huggingface_hub import snapshot_download

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

BASE_REVISION = '489cb97981b8654bcfcf30ce1f94ed1b62e07b53'
BASE_MANIFEST = REPO_DIR / 'artifacts/manifests/ornith-1.5-9b.json'
BASE_MANIFEST_SHA256 = '9a8c8c0c909311654a8ced2181b838cfc6d1db08d82f81b841cefa9030178f94'
if sha256_file(BASE_MANIFEST) != BASE_MANIFEST_SHA256:
    raise RuntimeError('base model manifest pin이 다릅니다.')
base_manifest = json.loads(BASE_MANIFEST.read_text())
allowed = [entry['path'] for entry in base_manifest['files']]
snapshot = Path(snapshot_download(
    'ornith-ai/Ornith-1.5-9B',
    revision=BASE_REVISION,
    allow_patterns=allowed,
    token=False,
))
BASE_MODEL = SESSION_ROOT / 'base-model/Ornith-1.5-9B'
BASE_MODEL.mkdir(parents=True, exist_ok=True)
for entry in base_manifest['files']:
    source = snapshot / entry['path']
    target = BASE_MODEL / entry['path']
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        shutil.copyfile(source, target)
    invalid = (
        target.is_symlink()
        or target.stat().st_size != entry['size']
        or sha256_file(target) != entry['sha256']
    )
    if invalid:
        raise RuntimeError(f'base model 파일 검증 실패: {entry["path"]}')
print(
    'Anonymous verified base model bytes:',
    sum(item['size'] for item in base_manifest['files']),
)"""
        ),
        _code_cell(
            """# 5. V2 92,239건 corpus 생성/검증
import urllib.request, uuid
from datetime import UTC, datetime
from shadowcrafter.data.adapters import canonicalize_nist_juliet
from shadowcrafter.data.ctibench import (
    find_ctibench_training_contamination,
    load_ctibench_eval_cases,
)
from shadowcrafter.data.prepare import SplitMode, prepare_jsonl_many
from shadowcrafter.schemas import SecurityRecord

JULIET_SHA256 = 'ada9d7e1c323d283446df3f55bdee0d00bda1fed786785fe98764d58688f38eb'
if sha256_file(V1_LOCAL) != V1_SHA256 or sha256_file(CTI_LOCAL) != CTI_SHA256:
    raise RuntimeError('내장 입력 파일 검증에 실패했습니다.')
DATA_CACHE = SESSION_ROOT / 'datasets' / f'v2.0-92239-{SOURCE_REVISION[:12]}'
LOCAL_DATA = SESSION_ROOT / 'processed-v2'
if DATA_CACHE.is_dir() and (DATA_CACHE / 'READY.json').is_file():
    ready = json.loads((DATA_CACHE / 'READY.json').read_text())
    if ready.get('record_count') != 92_239:
        raise RuntimeError('dataset cache record count가 다릅니다.')
    if not LOCAL_DATA.exists():
        shutil.copytree(DATA_CACHE / 'processed', LOCAL_DATA)
    if sha256_file(LOCAL_DATA / 'train.jsonl') != ready['train_sha256']:
        raise RuntimeError('dataset cache SHA-256 검증 실패')
    if sha256_file(LOCAL_DATA / 'manifest.json') != ready['manifest_sha256']:
        raise RuntimeError('dataset manifest SHA-256 검증 실패')
else:
    if LOCAL_DATA.exists():
        raise RuntimeError(f'검증되지 않은 기존 dataset 경로가 있습니다: {LOCAL_DATA}')
    WORK = SESSION_ROOT / f'v2-work-{uuid.uuid4().hex}'
    WORK.mkdir(parents=True, exist_ok=False)
    juliet_zip = WORK / 'juliet-cpp-1.3.zip'
    urllib.request.urlretrieve(
        'https://samate.nist.gov/SARD/downloads/test-suites/2017-10-01-juliet-test-suite-for-c-cplusplus-v1-3.zip',
        juliet_zip,
    )
    if sha256_file(juliet_zip) != JULIET_SHA256:
        raise RuntimeError('NIST Juliet 공식 ZIP SHA-256 검증 실패')
    juliet_jsonl = WORK / 'juliet.jsonl'
    canonicalize_nist_juliet(
        juliet_zip, juliet_jsonl,
        upstream_revision='nist-sard-suite-112-juliet-cpp-1.3',
        retrieved_at=datetime.now(UTC),
        registry_path=REPO_DIR / 'configs/data/sources.yaml',
    )
    cases = load_ctibench_eval_cases(CTI_LOCAL)
    juliet_records = [
        SecurityRecord.model_validate_json(line)
        for line in juliet_jsonl.read_text().splitlines() if line
    ]
    overlap = find_ctibench_training_contamination(juliet_records, cases)
    if overlap:
        raise RuntimeError(f'Juliet/CTIBench 오염 발견: {len(overlap)}')
    prepared = prepare_jsonl_many(
        [V1_LOCAL, juliet_jsonl], LOCAL_DATA,
        registry_path=REPO_DIR / 'configs/data/sources.yaml',
        split_mode=SplitMode.TRAIN_ONLY,
    )
    if prepared['record_count'] != 92_239 or prepared['split_counts']['train'] != 92_239:
        raise RuntimeError(f'V2 record count 불일치: {prepared["split_counts"]}')
    if prepared['exact_duplicate_count'] or prepared['normalized_duplicate_count']:
        raise RuntimeError('V2 corpus에 duplicate가 남았습니다.')
    staging = DATA_CACHE.parent / f'.{DATA_CACHE.name}.staging-{uuid.uuid4().hex}'
    staging.mkdir(parents=True, exist_ok=False)
    shutil.copytree(LOCAL_DATA, staging / 'processed')
    ready = {
        'schema_version': 1,
        'record_count': 92_239,
        'train_sha256': sha256_file(LOCAL_DATA / 'train.jsonl'),
        'manifest_sha256': sha256_file(LOCAL_DATA / 'manifest.json'),
        'dataset_sha256': prepared['dataset_sha256'],
        'ctibench_overlap': 0,
    }
    (staging / 'READY.json').write_text(json.dumps(ready, indent=2, sort_keys=True) + '\\n')
    DATA_CACHE.parent.mkdir(parents=True, exist_ok=True)
    staging.rename(DATA_CACHE)
print(json.dumps(ready, indent=2, sort_keys=True))"""
        ),
        _code_cell(
            """# 6. 계정 없는 임시 checkpoint에서 재개하며 V2 QLoRA 학습
from shadowcrafter.data.manifest import sha256_file as project_sha256_file
from shadowcrafter.data.registry import load_registry
from shadowcrafter.training.colab import train_resumable_colab
from shadowcrafter.training.training_safety import TrainingPins

CONFIG = REPO_DIR / 'configs/models/shadowcrafter-9b.yaml'
REGISTRY = REPO_DIR / 'configs/data/sources.yaml'
TRAIN = LOCAL_DATA / 'train.jsonl'
DATA_MANIFEST = LOCAL_DATA / 'manifest.json'
dataset_manifest = json.loads(DATA_MANIFEST.read_text())
identity = f'{SOURCE_REVISION[:12]}-{dataset_manifest["dataset_sha256"][:12]}'
CHECKPOINT_ROOT = SESSION_ROOT / 'checkpoints' / identity
FINAL_DIR = SESSION_ROOT / 'candidates' / f'v2.0-{identity}'
pins = TrainingPins(
    config_sha256=project_sha256_file(CONFIG),
    train_sha256=project_sha256_file(TRAIN),
    validation_sha256=None,
    dataset_manifest_sha256=project_sha256_file(DATA_MANIFEST),
    registry_sha256=load_registry(REGISTRY).canonical_sha256(),
    git_revision=SOURCE_REVISION,
)
if FINAL_DIR.exists():
    print('이미 완료된 immutable candidate가 있습니다:', FINAL_DIR)
    run_manifest = json.loads((FINAL_DIR / 'run-manifest.json').read_text())
    adapter_path = FINAL_DIR / 'adapter/adapter_model.safetensors'
    if sha256_file(adapter_path) != run_manifest['adapter']['adapter_weights_sha256']:
        raise RuntimeError('기존 final candidate adapter SHA-256 검증 실패')
else:
    run_manifest = train_resumable_colab(
        config_path=CONFIG,
        train_path=TRAIN,
        dataset_manifest_path=DATA_MANIFEST,
        registry_path=REGISTRY,
        base_model_path=BASE_MODEL,
        base_model_manifest_path=BASE_MANIFEST,
        base_model_manifest_sha256=BASE_MANIFEST_SHA256,
        checkpoint_root=CHECKPOINT_ROOT,
        final_dir=FINAL_DIR,
        pins=pins,
        save_steps=100,
        save_total_limit=3,
        checkpoint_storage='ephemeral',
    )
print(json.dumps({
    'candidate': str(FINAL_DIR),
    'global_step': run_manifest['training_observation']['global_step'],
    'train_loss': run_manifest['training_observation']['train_loss'],
    'adapter_sha256': run_manifest['adapter']['adapter_weights_sha256'],
}, indent=2, sort_keys=True))"""
        ),
        _code_cell(
            """# 7. candidate 또는 최신 checkpoint를 계정 연결 없이 로컬 다운로드
import os, tarfile, uuid
from IPython.display import FileLink, display

if 'FINAL_DIR' not in globals() or 'CHECKPOINT_ROOT' not in globals():
    raise RuntimeError('6번 학습 셀을 먼저 시작해 경로를 초기화하세요.')
if FINAL_DIR.is_dir():
    export_source = FINAL_DIR
    export_kind = 'candidate'
elif CHECKPOINT_ROOT.is_dir():
    export_source = CHECKPOINT_ROOT
    export_kind = 'checkpoints'
else:
    raise RuntimeError('내보낼 candidate 또는 checkpoint가 아직 없습니다.')
EXPORT = Path('/content') / f'ShadowCrafter-V2-{export_kind}-{identity}.tar.gz'
staging_export = EXPORT.parent / f'.{EXPORT.name}.tmp-{uuid.uuid4().hex}'
with tarfile.open(staging_export, 'w:gz') as archive:
    archive.add(export_source, arcname=export_source.name, recursive=True)
os.replace(staging_export, EXPORT)
print('Export:', EXPORT)
print('SHA-256:', sha256_file(EXPORT))
print('Size bytes:', EXPORT.stat().st_size)
display(FileLink(EXPORT.name))"""
        ),
    ]
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"gpuType": "A100", "provenance": []},
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.13"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    v1 = _read_pinned(V1_INPUT, V1_SHA256)
    ctibench = _read_pinned(CTIBENCH_INPUT, CTIBENCH_SHA256)
    with tempfile.TemporaryDirectory(prefix="shadowcrafter-accountless-") as temporary:
        runtime, runtime_revision, upstream_revision = _runtime_bundle(Path(temporary))
        notebook = _build_notebook(
            runtime_bundle=runtime,
            runtime_revision=runtime_revision,
            upstream_revision=upstream_revision,
            v1_gzip=_deterministic_gzip(v1),
            ctibench_gzip=_deterministic_gzip(ctibench),
        )
    content = json.dumps(notebook, indent=2, ensure_ascii=False) + "\n"
    temporary_output = OUTPUT.parent / f".{OUTPUT.name}.tmp"
    temporary_output.write_text(content, encoding="utf-8")
    os.replace(temporary_output, OUTPUT)
    print(f"wrote {OUTPUT} ({OUTPUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
