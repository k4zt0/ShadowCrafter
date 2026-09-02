import base64
import gzip
import hashlib
import json
from pathlib import Path

NOTEBOOK = Path(__file__).parents[1] / "notebooks/ShadowCrafter_V2_Colab.ipynb"


def _notebook() -> dict[str, object]:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def test_accountless_colab_notebook_compiles_and_requests_a100() -> None:
    notebook = _notebook()
    metadata = notebook["metadata"]
    assert isinstance(metadata, dict)
    assert metadata["accelerator"] == "GPU"
    assert metadata["colab"] == {"gpuType": "A100", "provenance": []}
    cells = notebook["cells"]
    assert isinstance(cells, list)
    code_cells = [cell for cell in cells if cell["cell_type"] == "code"]
    assert len(code_cells) == 8
    for index, cell in enumerate(code_cells, start=1):
        compile("".join(cell["source"]), f"{NOTEBOOK}#code-{index}", "exec")


def test_accountless_colab_notebook_has_no_account_mount_or_private_clone() -> None:
    notebook = _notebook()
    cells = notebook["cells"]
    assert isinstance(cells, list)
    source = "\n".join(
        "".join(cell["source"]) for cell in cells if cell["cell_type"] == "code"
    )
    assert "drive.mount(" not in source
    assert "google.colab import drive" not in source
    assert "google.colab import userdata" not in source
    assert "github.com/Odytssey/ShadowCrafter.git" not in source
    assert "token=False" in source
    assert "checkpoint_storage='ephemeral'" in source
    assert "HF_HUB_DISABLE_IMPLICIT_TOKEN" in source
    assert "evaluate_colab_candidate" in source
    assert "evaluation_result.accuracy" in source
    assert "evaluation_result.balanced_accuracy" in source
    assert "evaluation_result.macro_f1" in source
    assert "evaluation_result.quality_target_met" in source
    assert "7번 정확도 평가를 먼저 완료" in source
    assert "derive_juliet_cwe_mapping_jsonl" in source
    assert "derive_attack_technique_id_jsonl" in source
    assert "173_977" in source


def test_accountless_colab_embedded_payloads_match_pins() -> None:
    notebook = _notebook()
    cells = notebook["cells"]
    assert isinstance(cells, list)
    embedded_cells = [
        cell
        for cell in cells
        if "embedded-accountless-bootstrap" in cell.get("metadata", {}).get("tags", [])
    ]
    assert len(embedded_cells) == 1
    embedded = "".join(embedded_cells[0]["source"])
    prefix = embedded.split("def decode_verified", maxsplit=1)[0]
    namespace: dict[str, object] = {}
    exec(  # noqa: S102 - the prefix contains only generated constant assignments.
        compile(prefix, f"{NOTEBOOK}#embedded-pins", "exec"),
        namespace,
    )

    for encoded_name, compressed_sha_name in (
        ("RUNTIME_B85", "RUNTIME_COMPRESSED_SHA256"),
        ("V1_B85", "V1_COMPRESSED_SHA256"),
        ("CTIBENCH_B85", "CTIBENCH_COMPRESSED_SHA256"),
    ):
        encoded = namespace[encoded_name]
        expected = namespace[compressed_sha_name]
        assert isinstance(encoded, str)
        assert isinstance(expected, str)
        content = base64.b85decode(encoded)
        assert hashlib.sha256(content).hexdigest() == expected

    runtime = base64.b85decode(str(namespace["RUNTIME_B85"]))
    assert runtime.startswith(b"# v2 git bundle\n")
    v1 = gzip.decompress(base64.b85decode(str(namespace["V1_B85"])))
    ctibench = gzip.decompress(base64.b85decode(str(namespace["CTIBENCH_B85"])))
    assert hashlib.sha256(v1).hexdigest() == namespace["V1_SHA256"]
    assert hashlib.sha256(ctibench).hexdigest() == namespace["CTI_SHA256"]
    assert v1.count(b"\n") == 28_140
    assert ctibench.count(b"\n") == 5_533
