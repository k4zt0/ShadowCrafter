"""Download only config/processors to verify current Transformers compatibility."""

from pathlib import Path

import transformers
import typer
import yaml
from transformers import AutoConfig, AutoTokenizer


def main(config_path: Path) -> None:
    config = yaml.safe_load(config_path.read_text())
    base = config["base_model"]
    kwargs = {
        "revision": base["revision"],
        "trust_remote_code": False,
    }
    resolved = AutoConfig.from_pretrained(base["id"], **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(base["id"], **kwargs)
    if getattr(resolved, "_commit_hash", None) != base["revision"]:
        raise RuntimeError("resolved model revision does not match the immutable pin")
    if getattr(transformers, base["text_model_class"], None) is None:
        raise RuntimeError(f"missing text-only class {base['text_model_class']}")
    if not tokenizer.chat_template:
        raise RuntimeError("tokenizer has no chat template")
    print(
        {
            "model": base["id"],
            "model_type": resolved.model_type,
            "resolved_revision": getattr(resolved, "_commit_hash", None),
            "tokenizer_class": tokenizer.__class__.__name__,
            "chat_template": bool(tokenizer.chat_template),
        }
    )


if __name__ == "__main__":
    typer.run(main)
