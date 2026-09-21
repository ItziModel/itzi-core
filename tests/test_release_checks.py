from pathlib import Path
from subprocess import CompletedProcess

import pytest

from scripts import release_checks


def mock_build_identifiers(monkeypatch: pytest.MonkeyPatch, identifiers: str) -> None:
    monkeypatch.setattr(
        release_checks.subprocess,
        "run",
        lambda *args, **kwargs: CompletedProcess(args, 0, identifiers),
    )


def test_verify_wheel_count_matches_cibuildwheel_builds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mock_build_identifiers(monkeypatch, "cp312-manylinux_x86_64\ncp313-manylinux_x86_64\n")
    (tmp_path / "itzi_core-0.6.0-cp312-cp312-manylinux_x86_64.whl").touch()
    (tmp_path / "itzi_core-0.6.0-cp313-cp313-manylinux_x86_64.whl").touch()

    release_checks.verify_wheel_count(tmp_path)


def test_verify_wheel_count_rejects_missing_wheel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mock_build_identifiers(monkeypatch, "cp312-manylinux_x86_64\ncp313-manylinux_x86_64\n")
    (tmp_path / "itzi_core-0.6.0-cp312-cp312-manylinux_x86_64.whl").touch()

    with pytest.raises(SystemExit, match="Expected 2 wheels"):
        release_checks.verify_wheel_count(tmp_path)


def test_verify_wheel_count_rejects_empty_build_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mock_build_identifiers(monkeypatch, "\n")

    with pytest.raises(SystemExit, match="did not select any wheel builds"):
        release_checks.verify_wheel_count(tmp_path)
