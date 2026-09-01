import hashlib
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

import shadowcrafter.whitebox.assessor as assessor_module
from shadowcrafter.integrations.contracts import AuthorizationEvidence, WhiteBoxScope
from shadowcrafter.whitebox.assessor import (
    AuthorizedWhiteBoxAssessor,
    WhiteBoxAuthorizationError,
    WhiteBoxLimitError,
    compute_python_source_snapshot_sha256,
)
from shadowcrafter.whitebox.models import WhiteBoxAuthorizationArtifact

REPOSITORY_URI = "git://internal/service"
REVISION = "a" * 40


def _authorization(
    root: Path,
    *,
    included_paths: tuple[str, ...] = ("src",),
    excluded_paths: tuple[str, ...] = ("src/vendor",),
    repository_uri: str = REPOSITORY_URI,
    revision: str = REVISION,
) -> tuple[WhiteBoxScope, bytes]:
    now = datetime.now(UTC)
    artifact = WhiteBoxAuthorizationArtifact(
        authorization_id="AUTH-WB-1",
        scope_id="wb-test",
        approved_by="owner@example.test",
        purpose="Authorized static Python source review for an owned repository.",
        repository_uri=repository_uri,
        revision=revision,
        included_paths=included_paths,
        excluded_paths=excluded_paths,
        python_source_snapshot_sha256=compute_python_source_snapshot_sha256(
            root,
            included_paths=included_paths,
            excluded_paths=excluded_paths,
        ),
        valid_from=now - timedelta(hours=1),
        valid_until=now + timedelta(hours=1),
        static_analysis_only=True,
        target_code_execution_allowed=False,
        exploit_execution_allowed=False,
    )
    artifact_bytes = artifact.model_dump_json().encode()
    scope = WhiteBoxScope(
        scope_id=artifact.scope_id,
        authorization=AuthorizationEvidence(
            authorization_id=artifact.authorization_id,
            approved_by=artifact.approved_by,
            evidence_uri="vault://AUTH-WB-1",
            evidence_sha256=hashlib.sha256(artifact_bytes).hexdigest(),
            valid_from=artifact.valid_from,
            valid_until=artifact.valid_until,
        ),
        repository_uris=(repository_uri,),
        revisions=(revision,),
        included_paths=included_paths,
        excluded_paths=excluded_paths,
    )
    return scope, artifact_bytes


def _source_tree(tmp_path: Path) -> Path:
    source = tmp_path / "src"
    source.mkdir()
    (source / "app.py").write_text("value = 1\n")
    return source


def test_static_assessor_detects_candidates_without_retaining_source(tmp_path: Path) -> None:
    source = _source_tree(tmp_path)
    source.joinpath("app.py").write_text(
        "import subprocess\npassword = 'not-a-real-secret'\nsubprocess.run(['tool'], shell=True)\n"
    )
    vendor = source / "vendor"
    vendor.mkdir()
    (vendor / "ignored.py").write_text("eval('ignored')\n")
    scope, artifact = _authorization(tmp_path)

    result = AuthorizedWhiteBoxAssessor(
        scope=scope,
        authorization_artifact=artifact,
        repository_root=tmp_path,
        repository_uri=REPOSITORY_URI,
        revision=REVISION,
    ).assess()

    assert result.files_reviewed == 1
    assert result.files_authorized == result.files_reviewed + result.files_skipped
    assert result.source_snapshot_verified
    assert result.analysis_complete
    assert {item.cwe_candidates[0] for item in result.findings} == {"CWE-78", "CWE-798"}
    assert all(not item.source_content_included for item in result.evidence)
    assert all(item.digest_kind == "authorization-keyed-hmac-sha256" for item in result.evidence)
    secret_evidence = next(item for item in result.evidence if item.rule_id == "SC-PY-006")
    assert (
        secret_evidence.evidence_digest
        != hashlib.sha256(b"password = 'not-a-real-secret'\n").hexdigest()
    )
    assert "not-a-real-secret" not in result.model_dump_json()
    assert REPOSITORY_URI not in {
        item.source_uri for finding in result.findings for item in finding.evidence
    }
    assert not result.target_code_executed


def test_whitebox_requires_strict_scope_bound_authorization(tmp_path: Path) -> None:
    _source_tree(tmp_path)
    scope, artifact = _authorization(tmp_path)

    with pytest.raises(WhiteBoxAuthorizationError, match="digest"):
        AuthorizedWhiteBoxAssessor(
            scope=scope,
            authorization_artifact=artifact + b" ",
            repository_root=tmp_path,
            repository_uri=REPOSITORY_URI,
            revision=REVISION,
        )

    duplicate_artifact = artifact.replace(
        b'{"schema_version":"1.0"',
        b'{"schema_version":"1.0","schema_version":"1.0"',
        1,
    )
    duplicate_proof = scope.authorization.model_copy(
        update={"evidence_sha256": hashlib.sha256(duplicate_artifact).hexdigest()}
    )
    duplicate_scope = scope.model_copy(update={"authorization": duplicate_proof})
    with pytest.raises(WhiteBoxAuthorizationError, match="strict JSON"):
        AuthorizedWhiteBoxAssessor(
            scope=duplicate_scope,
            authorization_artifact=duplicate_artifact,
            repository_root=tmp_path,
            repository_uri=REPOSITORY_URI,
            revision=REVISION,
        )

    expanded_scope = scope.model_copy(
        update={
            "repository_uris": (REPOSITORY_URI, "git://other/service"),
            "revisions": (REVISION, "b" * 40),
        }
    )
    with pytest.raises(WhiteBoxAuthorizationError, match="another source identity"):
        AuthorizedWhiteBoxAssessor(
            scope=expanded_scope,
            authorization_artifact=artifact,
            repository_root=tmp_path,
            repository_uri="git://other/service",
            revision="b" * 40,
        )


@pytest.mark.parametrize("included", ("../outside", "/absolute", "./src", "src/"))
def test_whitebox_rejects_noncanonical_or_traversing_paths(tmp_path: Path, included: str) -> None:
    _source_tree(tmp_path)
    scope, artifact = _authorization(tmp_path)
    unsafe_scope = scope.model_copy(update={"included_paths": (included,)})

    with pytest.raises(WhiteBoxAuthorizationError, match="relative|canonical"):
        AuthorizedWhiteBoxAssessor(
            scope=unsafe_scope,
            authorization_artifact=artifact,
            repository_root=tmp_path,
            repository_uri=REPOSITORY_URI,
            revision=REVISION,
        )


def test_source_snapshot_rejects_mutation_and_symlink_escape(tmp_path: Path) -> None:
    source = _source_tree(tmp_path)
    scope, artifact = _authorization(tmp_path)
    source.joinpath("app.py").write_text("value = 2\n")

    with pytest.raises(WhiteBoxAuthorizationError, match="snapshot"):
        AuthorizedWhiteBoxAssessor(
            scope=scope,
            authorization_artifact=artifact,
            repository_root=tmp_path,
            repository_uri=REPOSITORY_URI,
            revision=REVISION,
        )

    source.joinpath("app.py").write_text("value = 1\n")
    outside = tmp_path / "outside.py"
    outside.write_text("eval('outside')\n")
    source.joinpath("linked.py").symlink_to(outside)
    with pytest.raises(WhiteBoxAuthorizationError, match="symlink"):
        AuthorizedWhiteBoxAssessor(
            scope=scope,
            authorization_artifact=artifact,
            repository_root=tmp_path,
            repository_uri=REPOSITORY_URI,
            revision=REVISION,
        )

    source.joinpath("linked.py").unlink()
    outside_directory = tmp_path / "outside"
    outside_directory.mkdir()
    outside_directory.joinpath("outside.py").write_text("eval('outside')\n")
    source.joinpath("linked-directory").symlink_to(outside_directory, target_is_directory=True)
    with pytest.raises(WhiteBoxAuthorizationError, match="symlink directory"):
        AuthorizedWhiteBoxAssessor(
            scope=scope,
            authorization_artifact=artifact,
            repository_root=tmp_path,
            repository_uri=REPOSITORY_URI,
            revision=REVISION,
        )


def test_import_aliases_and_obvious_shell_deserialization_calls_are_detected(
    tmp_path: Path,
) -> None:
    source = _source_tree(tmp_path)
    source.joinpath("app.py").write_text(
        "import os as operating_system\n"
        "import subprocess as process\n"
        "from builtins import eval as dynamic_eval\n"
        "from pickle import loads as deserialize\n"
        "from yaml import load as yaml_load\n"
        "operating_system.system('fixed-tool')\n"
        "process.run(['fixed-tool'], shell=True)\n"
        "dynamic_eval('1 + 1')\n"
        "deserialize(b'data')\n"
        "yaml_load('key: value')\n"
    )
    scope, artifact = _authorization(tmp_path)
    result = AuthorizedWhiteBoxAssessor(
        scope=scope,
        authorization_artifact=artifact,
        repository_root=tmp_path,
        repository_uri=REPOSITORY_URI,
        revision=REVISION,
    ).assess()

    assert {finding.cwe_candidates[0] for finding in result.findings} >= {
        "CWE-78",
        "CWE-95",
        "CWE-502",
    }


def test_python_encoding_cookie_is_honored_without_leaking_literal(tmp_path: Path) -> None:
    source = _source_tree(tmp_path)
    secret_line = "secret_token = 'très-longue-valeur'\n"  # noqa: S105
    source.joinpath("app.py").write_bytes(("# coding: latin-1\n" + secret_line).encode("latin-1"))
    scope, artifact = _authorization(tmp_path)

    result = AuthorizedWhiteBoxAssessor(
        scope=scope,
        authorization_artifact=artifact,
        repository_root=tmp_path,
        repository_uri=REPOSITORY_URI,
        revision=REVISION,
    ).assess()

    assert "CWE-798" in {finding.cwe_candidates[0] for finding in result.findings}
    assert "très-longue-valeur" not in result.model_dump_json()


def test_limits_fail_closed_before_returning_partial_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _source_tree(tmp_path)
    source.joinpath("second.py").write_text("value = 2\n")
    monkeypatch.setattr(assessor_module, "MAX_FILES", 1)

    with pytest.raises(WhiteBoxLimitError, match="count"):
        compute_python_source_snapshot_sha256(
            tmp_path,
            included_paths=("src",),
            excluded_paths=(),
        )


def test_repository_uri_credentials_are_rejected(tmp_path: Path) -> None:
    _source_tree(tmp_path)
    scope, artifact = _authorization(tmp_path)

    with pytest.raises(WhiteBoxAuthorizationError, match="credentials"):
        AuthorizedWhiteBoxAssessor(
            scope=scope,
            authorization_artifact=artifact,
            repository_root=tmp_path,
            repository_uri="https://token@example.test/service",
            revision=REVISION,
        )


def test_result_contract_rejects_mismatched_evidence_location(tmp_path: Path) -> None:
    source = _source_tree(tmp_path)
    source.joinpath("app.py").write_text("secret_token = 'not-a-real-secret'\n")
    scope, artifact = _authorization(tmp_path)
    result = AuthorizedWhiteBoxAssessor(
        scope=scope,
        authorization_artifact=artifact,
        repository_root=tmp_path,
        repository_uri=REPOSITORY_URI,
        revision=REVISION,
    ).assess()
    payload = result.model_dump(mode="json")
    payload["evidence"][0]["path"] = "src/other.py"

    with pytest.raises(ValidationError, match="location or digest"):
        type(result).model_validate(payload)
