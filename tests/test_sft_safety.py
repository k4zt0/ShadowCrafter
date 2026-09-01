from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from shadowcrafter.training.sft import (
    EXPECTED_LORA_TARGETS,
    EXPECTED_MODEL_ID,
    EXPECTED_REVISION,
    AdapterExpectation,
    DenseTrainingError,
    TrainingPins,
    _assert_lora_targets,
    _assert_training_invariants,
    _fresh_output_workspace,
    _load_dense_config,
    _sft_kwargs,
    _to_prompt_completion,
    _verify_completion_boundaries,
    _verify_local_model_tree,
    _verify_record_provenance,
    _verify_saved_adapter,
    train_sft,
)
from shadowcrafter.training.training_safety import VerifiedTrainingInputs


def test_messages_become_conversational_prompt_completion() -> None:
    messages = [
        {"role": "system", "content": "Answer defensively."},
        {"role": "user", "content": "Review this finding."},
        {"role": "assistant", "content": "Validate the evidence first."},
    ]

    converted = _to_prompt_completion({"messages": messages, "unused": "removed later"})

    assert converted == {
        "prompt": messages[:-1],
        "completion": [messages[-1]],
        "chat_template_kwargs": {"enable_thinking": False},
    }


@pytest.mark.parametrize(
    "messages",
    [
        [],
        [{"role": "assistant", "content": "alone"}],
        [{"role": "user", "content": "question"}, {"role": "user", "content": "again"}],
        [{"role": "user", "content": "question"}, {"role": "assistant", "content": "  "}],
    ],
)
def test_prompt_completion_rejects_ambiguous_or_empty_records(
    messages: list[dict[str, str]],
) -> None:
    with pytest.raises(ValueError):
        _to_prompt_completion({"messages": messages})


@dataclass
class _Weight:
    ndim: int


@dataclass
class _Module:
    weight: _Weight


class _FakeModel:
    def named_modules(self) -> list[tuple[str, _Module]]:
        return [
            ("layer.0.q_proj", _Module(_Weight(2))),
            ("layer.0.k_proj", _Module(_Weight(2))),
            ("layer.0.experts", _Module(_Weight(3))),
        ]


def test_lora_target_count_excludes_fused_three_dimensional_experts() -> None:
    _assert_lora_targets(
        _FakeModel(),
        {"target_modules": ["q_proj", "k_proj", "experts"], "expected_target_module_count": 2},
    )

    with pytest.raises(RuntimeError, match="target count mismatch"):
        _assert_lora_targets(
            _FakeModel(),
            {
                "target_modules": ["q_proj", "k_proj", "experts"],
                "expected_target_module_count": 3,
            },
        )


def test_training_invariants_reject_packing_or_ambiguous_loss() -> None:
    safe = {
        "completion_only_loss": True,
        "assistant_only_loss": False,
        "packing": False,
        "eval_packing": False,
        "padding_free": False,
        "enable_thinking": False,
    }
    _assert_training_invariants(safe)

    for key in safe:
        unsafe = {**safe, key: not safe[key]}
        with pytest.raises(RuntimeError, match="outside the audited path"):
            _assert_training_invariants(unsafe)


def test_local_model_tree_requires_exact_inventory(tmp_path: Path) -> None:
    import hashlib
    import json

    root = tmp_path / "model"
    root.mkdir()
    model_file = root / "config.json"
    model_file.write_text("{}")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "artifact_id": "example/model",
                "revision": "a" * 40,
                "files": [
                    {
                        "path": "config.json",
                        "size": 2,
                        "sha256": hashlib.sha256(b"{}").hexdigest(),
                    }
                ],
            }
        )
    )
    _verify_local_model_tree("example/model", "a" * 40, root, manifest)

    model_file.write_text("changed")
    with pytest.raises(RuntimeError, match="metadata mismatch"):
        _verify_local_model_tree("example/model", "a" * 40, root, manifest)


def test_dense_config_is_closed_and_max_steps_is_never_zero(tmp_path: Path) -> None:
    config = Path("configs/models/shadowcrafter-9b.yaml")
    observed = _load_dense_config(config, max_steps=1)
    assert observed["base_model"]["id"] == EXPECTED_MODEL_ID

    payload = config.read_text().replace("  optimizer: paged_adamw_8bit\n", "")
    drifted = tmp_path / "drifted.yaml"
    drifted.write_text(payload)
    with pytest.raises(DenseTrainingError, match="surface drifted"):
        _load_dense_config(drifted, max_steps=1)
    with pytest.raises(DenseTrainingError, match="max_steps"):
        _load_dense_config(config, max_steps=0)


def test_private_output_is_removed_on_failure_and_existing_output_is_refused(
    tmp_path: Path,
) -> None:
    output = tmp_path / "candidate"
    with pytest.raises(RuntimeError, match="simulated"), _fresh_output_workspace(output):
        raise RuntimeError("simulated failure")
    assert not output.exists()

    output.mkdir()
    with (
        pytest.raises(DenseTrainingError, match="refusing to reuse"),
        _fresh_output_workspace(output),
    ):
        pass


def test_record_provenance_rejects_holdouts_and_unknown_sources(tmp_path: Path) -> None:
    inputs = VerifiedTrainingInputs(
        train_path=tmp_path / "train.jsonl",
        validation_path=None,
        dataset_manifest_path=tmp_path / "manifest.json",
        registry_path=tmp_path / "registry.yaml",
        train_sha256="a" * 64,
        validation_sha256=None,
        dataset_manifest_sha256="b" * 64,
        dataset_sha256="c" * 64,
        registry_sha256="d" * 64,
        train_record_count=1,
        validation_record_count=None,
        source_licenses=(("safe-source", "Apache-2.0"),),
    )
    record = {
        "record_id": "record-1",
        "risk_tier": "low",
        "benchmark_holdout": False,
        "provenance": {"source_id": "safe-source", "license": "Apache-2.0"},
    }
    _verify_record_provenance({"train": [record]}, inputs)

    with pytest.raises(DenseTrainingError, match="benchmark holdout"):
        _verify_record_provenance({"train": [{**record, "benchmark_holdout": True}]}, inputs)
    with pytest.raises(DenseTrainingError, match="absent from the pinned manifest"):
        _verify_record_provenance(
            {
                "train": [
                    {
                        **record,
                        "provenance": {"source_id": "eval-only", "license": "unknown"},
                    }
                ]
            },
            inputs,
        )


def test_completion_boundary_and_no_hub_arguments() -> None:
    class Tokenizer:
        @staticmethod
        def apply_chat_template(
            _messages: list[dict[str, str]],
            *,
            add_generation_prompt: bool = False,
            return_dict: bool = False,
            **_kwargs: object,
        ) -> object:
            ids = [1, 2] if add_generation_prompt else [1, 2, 3]
            return {"input_ids": ids} if return_dict else ids

    dataset = {
        "train": [
            {
                "prompt": [{"role": "user", "content": "review"}],
                "completion": [{"role": "assistant", "content": "validate"}],
                "chat_template_kwargs": {"enable_thinking": False},
            }
        ]
    }
    _verify_completion_boundaries(dataset, Tokenizer(), 8)
    kwargs = _sft_kwargs(
        _load_dense_config(Path("configs/models/shadowcrafter-9b.yaml"), max_steps=1)["training"],
        output_dir=Path("private"),
        max_steps=1,
        has_validation=True,
    )
    assert kwargs["push_to_hub"] is False
    assert kwargs["report_to"] == "none"
    assert kwargs["resume_from_checkpoint"] is None
    assert kwargs["save_strategy"] == "no"
    assert kwargs["eval_strategy"] == "no"


def test_saved_adapter_is_reopened_as_finite_lora_only(tmp_path: Path) -> None:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(
        json.dumps(
            {
                "base_model_name_or_path": EXPECTED_MODEL_ID,
                "revision": EXPECTED_REVISION,
                "peft_type": "LORA",
                "bias": "none",
                "modules_to_save": None,
                "target_parameters": None,
                "use_dora": False,
                "lora_bias": False,
                "target_modules": list(EXPECTED_LORA_TARGETS),
            }
        )
    )
    (adapter / "adapter_model.safetensors").write_bytes(b"safe")

    class Scalar:
        def __init__(self, value: object) -> None:
            self.value = value

        def all(self) -> Scalar:
            return self

        def item(self) -> object:
            return self.value

    class Tensor:
        @staticmethod
        def numel() -> int:
            return 3

    class Handle:
        def __enter__(self) -> Handle:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        @staticmethod
        def keys() -> list[str]:
            return ["model.layer.q_proj.lora_A.weight", "model.layer.q_proj.lora_B.weight"]

        @staticmethod
        def get_tensor(_key: str) -> Tensor:
            return Tensor()

    class Torch:
        @staticmethod
        def isfinite(_tensor: Tensor) -> Scalar:
            return Scalar(True)

        @staticmethod
        def count_nonzero(_tensor: Tensor) -> Scalar:
            return Scalar(1)

    verification = _verify_saved_adapter(
        adapter,
        AdapterExpectation(tensor_count=2, parameter_count=6),
        safe_open_fn=lambda *_args, **_kwargs: Handle(),
        torch_module=Torch(),
    )
    assert verification.tensor_count == 2
    assert verification.nonzero_parameter_values == 2


def test_top_level_refuses_preexisting_output_before_loading_gpu(tmp_path: Path) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    pins = TrainingPins(
        config_sha256="a" * 64,
        train_sha256="b" * 64,
        validation_sha256=None,
        dataset_manifest_sha256="c" * 64,
        registry_sha256="d" * 64,
        git_revision="e" * 40,
    )
    with pytest.raises(DenseTrainingError, match="refusing to reuse"):
        train_sft(
            config_path=Path("missing"),
            train_path=Path("missing"),
            validation_path=None,
            dataset_manifest_path=Path("missing"),
            registry_path=Path("missing"),
            base_model_path=Path("missing"),
            base_model_manifest_path=Path("missing"),
            base_model_manifest_sha256="f" * 64,
            output_dir=output,
            pins=pins,
            max_steps=1,
        )
