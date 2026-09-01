"""ShadowCrafter command-line interface."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
import yaml
from rich.console import Console

from shadowcrafter.evaluation.gate import load_and_evaluate, write_gate_report
from shadowcrafter.knowledge.database import initialize_database, search

app = typer.Typer(help="ShadowCrafter cybersecurity model engineering CLI")
config_app = typer.Typer(help="Configuration utilities")
knowledge_app = typer.Typer(help="Local security knowledge database")
eval_app = typer.Typer(help="Evaluation and release gates")
data_app = typer.Typer(help="Dataset and source snapshot utilities")
train_app = typer.Typer(help="Model training")
release_app = typer.Typer(help="Private Experimental Release publication")
assess_app = typer.Typer(help="Authorized evidence-grounded vulnerability assessment")
app.add_typer(config_app, name="config")
app.add_typer(knowledge_app, name="knowledge")
app.add_typer(eval_app, name="eval")
app.add_typer(data_app, name="data")
app.add_typer(train_app, name="train")
app.add_typer(release_app, name="release")
app.add_typer(assess_app, name="assess")
console = Console()


@config_app.command("validate")
def validate_config(path: Path) -> None:
    payload = yaml.safe_load(path.read_text())
    required = {"project", "base_model", "training"}
    missing = required - payload.keys()
    if missing:
        raise typer.BadParameter(f"missing keys: {sorted(missing)}")
    console.print(f"[green]valid[/green] {path}")


@knowledge_app.command("init")
def init_knowledge(path: Path = Path("artifacts/knowledge/shadowcrafter.db")) -> None:
    initialize_database(path)
    console.print(f"[green]initialized[/green] {path}")


@knowledge_app.command("search")
def search_knowledge(query: str, path: Path = Path("artifacts/knowledge/shadowcrafter.db")) -> None:
    console.print_json(json.dumps(search(path, query), ensure_ascii=False))


@assess_app.command("blackbox")
def assess_blackbox(
    scope: Annotated[Path, typer.Option("--scope")],
    authorization: Annotated[Path, typer.Option("--authorization")],
    target: Annotated[list[str], typer.Option("--target")],
    output: Annotated[Path, typer.Option("--output")],
    method: Annotated[list[str] | None, typer.Option("--method")] = None,
) -> None:
    """Run a passive assessment against exact URLs authorized by a hash-bound scope."""

    from shadowcrafter.blackbox import (
        AuthorizationError,
        NetworkSafetyError,
        read_blackbox_scope,
        run_authorized_assessment,
    )

    try:
        runtime_scope = read_blackbox_scope(scope)
        result = run_authorized_assessment(
            scope=runtime_scope,
            authorization_artifact=authorization,
            targets=tuple(target),
            methods=tuple(method or ("HEAD",)),
        )
        content = result.model_dump_json(indent=2).encode("utf-8") + b"\n"
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("xb") as handle:
            handle.write(content)
        output.chmod(0o600)
    except (AuthorizationError, NetworkSafetyError, OSError, ValueError):
        console.print("[red]black-box assessment failed closed[/red]; details withheld")
        raise typer.Exit(code=1) from None
    console.print(
        f"[green]assessment completed[/green] findings={len(result.findings)} "
        f"evidence={len(result.evidence)} output={output}"
    )


@eval_app.command("gate")
def run_gate(
    evidence: Path,
    config: Path = Path("configs/eval/release-gates.yaml"),
    report: Path | None = None,
) -> None:
    result = load_and_evaluate(evidence, config)
    if report is not None and result.report is not None:
        write_gate_report(result, report)
    if result.passed:
        quality_target_met = (
            result.report.get("quality_target_met") if result.report is not None else None
        )
        console.print(
            "[green]private Experimental Release integrity gate passed[/green]; "
            f"target_94_met={str(quality_target_met).lower()}"
        )
        return
    for failure in result.failures:
        console.print(f"[red]FAIL[/red] {failure}")
    raise typer.Exit(code=1)


@release_app.command("publish-remote-experimental")
def publish_remote_experimental(
    manifest: Annotated[Path, typer.Option("--manifest")],
    manifest_sha256: Annotated[str, typer.Option("--manifest-sha256")],
    ssh_key: Annotated[Path, typer.Option("--ssh-key")],
    evidence: Annotated[Path | None, typer.Option("--evidence")] = None,
    gate_config: Annotated[
        Path,
        typer.Option("--gate-config"),
    ] = Path("configs/eval/release-gates.yaml"),
) -> None:
    """Stream an exact remote bundle into one private Hugging Face commit."""

    from shadowcrafter.release.remote_huggingface import (
        publish_remote_experimental_release,
    )

    try:
        result = publish_remote_experimental_release(
            manifest,
            manifest_sha256=manifest_sha256,
            ssh_key=ssh_key,
            evidence_path=evidence,
            gate_config=gate_config,
        )
    except Exception as exc:
        # External client exceptions can contain request details. Keep credentials and
        # headers out of unattended logs; callers can reproduce with the Python API.
        console.print(
            "[red]publication failed closed[/red] "
            f"({type(exc).__name__}; sensitive details withheld)",
        )
        raise typer.Exit(code=1) from None
    console.print_json(json.dumps(result.as_dict(), ensure_ascii=False, sort_keys=True))


@data_app.command("prepare")
def prepare_data(input_path: Path, output_dir: Path = Path("data/processed/v1")) -> None:
    from shadowcrafter.data.prepare import prepare_jsonl

    console.print_json(json.dumps(prepare_jsonl(input_path, output_dir), ensure_ascii=False))


@data_app.command("snapshot")
def snapshot_data(
    config: Path = Path("configs/data/sources.yaml"),
    output_dir: Path = Path("data/raw/snapshots"),
) -> None:
    from shadowcrafter.data.snapshot import snapshot_http_sources

    console.print_json(json.dumps(snapshot_http_sources(config, output_dir), ensure_ascii=False))


@train_app.command("sft")
def train_model(
    config: Annotated[Path, typer.Option("--config")],
    config_sha256: Annotated[str, typer.Option("--config-sha256")],
    train: Annotated[Path, typer.Option("--train")],
    train_sha256: Annotated[str, typer.Option("--train-sha256")],
    dataset_manifest: Annotated[Path, typer.Option("--dataset-manifest")],
    dataset_manifest_sha256: Annotated[str, typer.Option("--dataset-manifest-sha256")],
    registry: Annotated[Path, typer.Option("--registry")],
    registry_sha256: Annotated[str, typer.Option("--registry-sha256")],
    base_model: Annotated[Path, typer.Option("--base-model")],
    base_model_manifest: Annotated[Path, typer.Option("--base-model-manifest")],
    base_model_manifest_sha256: Annotated[str, typer.Option("--base-model-manifest-sha256")],
    git_revision: Annotated[str, typer.Option("--git-revision")],
    output_dir: Annotated[Path, typer.Option("--output-dir")],
    validation: Annotated[Path | None, typer.Option("--validation")] = None,
    validation_sha256: Annotated[str | None, typer.Option("--validation-sha256")] = None,
    max_steps: Annotated[int, typer.Option("--max-steps")] = -1,
) -> None:
    from shadowcrafter.training.sft import TrainingPins, train_sft

    manifest = train_sft(
        config_path=config,
        train_path=train,
        validation_path=validation,
        dataset_manifest_path=dataset_manifest,
        registry_path=registry,
        base_model_path=base_model,
        base_model_manifest_path=base_model_manifest,
        base_model_manifest_sha256=base_model_manifest_sha256,
        output_dir=output_dir,
        pins=TrainingPins(
            config_sha256=config_sha256,
            train_sha256=train_sha256,
            validation_sha256=validation_sha256,
            dataset_manifest_sha256=dataset_manifest_sha256,
            registry_sha256=registry_sha256,
            git_revision=git_revision,
        ),
        max_steps=max_steps,
    )
    console.print_json(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    app()
