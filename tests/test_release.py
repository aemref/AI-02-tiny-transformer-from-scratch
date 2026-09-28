import json
from pathlib import Path

from tiny_transformer.release import main, run_release_checks


def test_repository_release_evidence_is_complete() -> None:
    checks = run_release_checks()

    assert {check.name for check in checks} == {
        "application_version",
        "corpus_license",
        "experiment_artifacts",
        "release_documents",
    }
    assert all(check.passed for check in checks), checks


def test_version_mismatch_blocks_release() -> None:
    checks = run_release_checks(expected_version="9.9.9")

    version = next(check for check in checks if check.name == "application_version")
    assert not version.passed
    assert "expected 9.9.9" in version.detail


def test_missing_repository_evidence_fails_closed(tmp_path: Path) -> None:
    checks = run_release_checks(root=tmp_path, expected_version="1.0.0")

    assert not all(check.passed for check in checks)
    assert next(
        check for check in checks if check.name == "experiment_artifacts"
    ).detail.startswith("missing")
    assert "missing README.md" in next(
        check for check in checks if check.name == "release_documents"
    ).detail


def test_cli_emits_machine_readable_gate_results(capsys) -> None:
    assert main([]) == 0

    report = json.loads(capsys.readouterr().out)
    assert report["release"] == "v1.0.0"
    assert report["ready"] is True
    assert all(check["passed"] for check in report["checks"])
