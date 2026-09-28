import json
from pathlib import Path

import pytest

from tiny_transformer.experiments.verify_training import (
    TrainingEvidenceMismatch,
    main,
    verify_training_evidence,
)


def test_evidence_comparison_accepts_platform_level_float_drift() -> None:
    expected = {"steps": 60, "curve": [12.0, 1.5], "name": "baseline"}
    actual = {"steps": 60, "curve": [12.000009, 1.499991], "name": "baseline"}

    verify_training_evidence(expected, actual)


@pytest.mark.parametrize(
    "actual",
    [
        {"steps": 61, "curve": [12.0, 1.5], "name": "baseline"},
        {"steps": 60, "curve": [12.0, 1.49], "name": "baseline"},
        {"steps": 60, "curve": [12.0], "name": "baseline"},
        {"steps": 60, "curve": [12.0, 1.5], "label": "baseline"},
    ],
)
def test_evidence_comparison_rejects_material_or_structural_changes(
    actual: dict[str, object],
) -> None:
    expected = {"steps": 60, "curve": [12.0, 1.5], "name": "baseline"}

    with pytest.raises(TrainingEvidenceMismatch):
        verify_training_evidence(expected, actual)


def test_verifier_cli_reports_the_tolerance(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    expected = tmp_path / "expected.json"
    actual = tmp_path / "actual.json"
    expected.write_text(json.dumps({"loss": 1.0}), encoding="utf-8")
    actual.write_text(json.dumps({"loss": 1.000001}), encoding="utf-8")

    assert main([str(expected), str(actual)]) == 0
    assert json.loads(capsys.readouterr().out) == {
        "absolute_float_tolerance": 1e-5,
        "matched": True,
    }
