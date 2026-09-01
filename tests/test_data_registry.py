from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from shadowcrafter.data.registry import (
    LicenseStatus,
    PolicyClass,
    Purpose,
    SourceRegistry,
    load_registry,
)

REGISTRY = Path("configs/data/sources.yaml")


def test_main_registry_is_strict_and_covers_authoritative_sources() -> None:
    registry = load_registry(REGISTRY)
    expected = {
        "cve-list-v5",
        "nvd-cve-api",
        "cisa-kev",
        "mitre-cwe",
        "mitre-capec",
        "epss-daily-scores",
        "mitre-d3fend",
        "ocsf-schema",
        "osv-schema",
    }
    assert expected <= registry.by_id.keys()
    assert len(registry.canonical_sha256()) == 64


def test_registry_checksum_is_independent_of_source_order() -> None:
    payload = yaml.safe_load(REGISTRY.read_text())
    forward = SourceRegistry.model_validate(payload)
    payload["sources"].reverse()
    reversed_registry = SourceRegistry.model_validate(payload)
    assert forward.canonical_sha256() == reversed_registry.canonical_sha256()


def test_every_usable_source_has_verified_terms_and_no_raw_binaries() -> None:
    registry = load_registry(REGISTRY)
    usable = {PolicyClass.ALLOW_TRAIN, PolicyClass.RAG_ONLY, PolicyClass.EVAL_ONLY}
    for source in registry.sources:
        if source.policy_class in usable:
            assert source.license.status == LicenseStatus.VERIFIED
            assert not source.safety.raw_malware_binaries


def test_registry_enforces_train_rag_eval_purpose_boundaries() -> None:
    registry = load_registry(REGISTRY)
    assert registry.require_purpose("mitre-cwe", Purpose.TRAIN)
    assert registry.require_purpose("cisa-kev", Purpose.RAG)
    assert registry.require_purpose("ctibench", Purpose.EVALUATE)
    with pytest.raises(ValueError, match="does not permit train"):
        registry.require_purpose("cisa-kev", Purpose.TRAIN)
    with pytest.raises(ValueError, match="does not permit train"):
        registry.require_purpose("ctibench", Purpose.TRAIN)


def test_registry_rejects_policy_purpose_widening() -> None:
    payload = yaml.safe_load(REGISTRY.read_text())
    source = next(item for item in payload["sources"] if item["id"] == "cisa-kev")
    source["allowed_purposes"] = ["rag", "train"]
    with pytest.raises(ValidationError, match="requires purposes"):
        SourceRegistry.model_validate(payload)


def test_registry_rejects_raw_binary_source_unless_prohibited() -> None:
    payload = yaml.safe_load(REGISTRY.read_text())
    source = next(item for item in payload["sources"] if item["id"] == "ember2024")
    source["safety"]["raw_malware_binaries"] = True
    with pytest.raises(ValidationError, match="raw executable binaries are prohibited"):
        SourceRegistry.model_validate(payload)


def test_unknown_source_is_prohibited() -> None:
    with pytest.raises(ValueError, match="unregistered source is prohibited"):
        load_registry(REGISTRY).source("unknown-dump")
