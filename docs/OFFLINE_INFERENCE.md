# Offline frozen inference

`scripts/remote/run-frozen-inference.py` is the only audited producer for CTIBench prediction
files. It runs on the approved remote GPU, never on the workstation. It scores nothing, selects
no checkpoint, contacts no Hub, and grants no publication right. CTIBench remains
`CC-BY-NC-SA-4.0`, evaluation-only, and restricted to noncommercial private research.

## Preconditions

Run from an exact clean, detached ShadowCrafter commit. The remote host must already contain:

- the complete pinned Ornith base snapshot and its immutable inventory;
- a completed LoRA-only training directory containing `adapter/` and `run-manifest.json`;
- a complete SHA-256 checkpoint-tree manifest created outside that training directory;
- the frozen CTIBench case, adapter-manifest, snapshot-manifest, and release-gate files;
- a fresh environment manifest created by the same Python environment that will run inference.

No token is needed or accepted. The runner sets the Hugging Face and Transformers offline flags,
removes proxy variables, denies Python socket connections, passes `local_files_only=True` and
`trust_remote_code=False`, and loads only the explicit Qwen 3.5 text class declared for that model
family. One visible CUDA device is required; CPU/disk model offload is rejected.

Create the environment observation once at a fresh, non-existing path:

```bash
.venv/bin/python scripts/remote/observe-inference-environment.py \
  --output /root/ShadowCrafter/artifacts/evaluations/environment-9b.json
```

Capture the SHA-256 printed by that command. Do not rewrite the file to reformat it.

## Immutable request

The supervisor writes one JSON request and pins its raw SHA-256. Every path is absolute and every
referenced evidence file has a raw SHA-256. `output.directory` must not exist, while its canonical,
non-symlink parent must already exist. A minimal 9B shape is shown below; replace every placeholder
with observed, independently checked evidence:

```json
{
  "schema_version": 1,
  "protocol": "shadowcrafter-frozen-release-evaluation-v1",
  "evaluation_id": "shadowcrafter-9b-<candidate>-ctibench-v1",
  "model": {
    "family": "ShadowCrafter-9B",
    "candidate_id": "shadowcrafter-9b-<candidate>",
    "model_id": "KaztoRay/ShadowCrafter-9B",
    "base_model_id": "ornith-ai/Ornith-1.5-9B",
    "base_model_revision": "489cb97981b8654bcfcf30ce1f94ed1b62e07b53",
    "text_model_class": "Qwen3_5ForCausalLM",
    "config": {"path": "/absolute/shadowcrafter-9b.yaml", "sha256": "<sha256>"},
    "base_model_path": "/absolute/Ornith-1.5-9B",
    "base_model_manifest": {"path": "/absolute/base-manifest.json", "sha256": "<sha256>"},
    "adapter_path": "/absolute/completed-run/adapter",
    "checkpoint_manifest": {"path": "/absolute/checkpoint-manifest.json", "sha256": "<sha256>"},
    "training_run_manifest": {"path": "/absolute/completed-run/run-manifest.json", "sha256": "<sha256>"}
  },
  "benchmark": {
    "benchmark_id": "ctibench",
    "repository_id": "AI4Sec/cti-bench",
    "upstream_revision": "9237e1636ee3e168fbe5ebdcc1c571de0525e568",
    "license_id": "CC-BY-NC-SA-4.0",
    "usage_scope": "noncommercial-private-research",
    "evaluation_only": true,
    "benchmark_holdout": true,
    "cases": {"path": "/absolute/ctibench-eval.jsonl", "sha256": "<sha256>", "record_count": 5533},
    "adapter_manifest": {"path": "/absolute/ctibench-eval.jsonl.manifest.json", "sha256": "<sha256>"},
    "snapshot_manifest": {"path": "/absolute/ctibench-snapshot/manifest.json", "sha256": "<sha256>"},
    "dataset_sha256": "<sha256>",
    "gate_config": {"path": "/absolute/release-gates.yaml", "sha256": "<sha256>"}
  },
  "decoding": {
    "seed": 20260901,
    "max_input_tokens": 4096,
    "max_new_tokens": 128,
    "per_case_seconds": 120,
    "total_seconds": 172800,
    "max_gpu_memory_gib": 72,
    "max_cpu_rss_gib": 192,
    "max_cpu_threads": 8
  },
  "source": {
    "git_revision": "<clean-40-character-commit>",
    "require_clean_git": true,
    "environment_manifest": {"path": "/absolute/environment-9b.json", "sha256": "<sha256>"}
  },
  "output": {
    "directory": "/absolute/evaluations/shadowcrafter-9b-<candidate>-ctibench-v1",
    "predictions_name": "predictions.jsonl",
    "manifest_name": "inference-manifest.json",
    "resume": false
  }
}
```

Run the exact request:

```bash
.venv/bin/python scripts/remote/run-frozen-inference.py \
  --request /absolute/inference-request.json \
  --request-sha256 <raw-request-sha256>
```

## Outputs and retry behavior

Success publishes an exclusively created, read-only directory containing only:

- `predictions.jsonl`: `FrozenPrediction` schema v1 records in the exact frozen case order;
- `inference-manifest.json`: hashes, identities, aggregate resource observations, and timestamps.

The prediction stream retains decoded output exactly. An EOS-only generation is represented by
the declared `<EMPTY_OUTPUT>` sentinel and will score as invalid downstream. The producer never
parses an answer or logs a benchmark example. Both complete artifact trees and every input,
configuration, environment, adapter, and manifest pin are revalidated after generation before
publication.

Existing output is always an error. There is no resume or “existing means success” mode. A failed
or interrupted attempt must use a new evaluation ID and a new output path after the supervisor has
recorded the failure. The strict evaluator later consumes `predictions.jsonl`; its measured metrics
are reporting-only for a private Experimental Release, and a score below 94% is retained honestly.
