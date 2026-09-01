from pathlib import Path

from shadowcrafter.automation import remote_worker


def test_detached_environment_pins_worker_to_its_source_tree(monkeypatch) -> None:
    monkeypatch.setenv("PYTHONPATH", "/untrusted/stale-source")
    monkeypatch.delenv("PYTHONNOUSERSITE", raising=False)

    environment = remote_worker._detached_environment()

    expected = Path(remote_worker.__file__).resolve().parents[2]
    assert environment["PYTHONPATH"] == str(expected)
    assert environment["PYTHONNOUSERSITE"] == "1"
    assert environment["PYTHONPATH"] != "/untrusted/stale-source"
