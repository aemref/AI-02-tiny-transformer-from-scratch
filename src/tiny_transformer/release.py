"""Verify deterministic repository artifacts required for the v1 release."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from tiny_transformer import __version__

ROOT = Path(__file__).parents[2]
EXPECTED_VERSION = "1.0.0"
REQUIRED_DOCUMENTS = {
    Path("README.md"): ("Local demo", "Limitations", "Release verification"),
    Path("CHANGELOG.md"): ("1.0.0", "not suitable for"),
    Path("docs/architecture.md"): ("Trust and failure boundaries",),
    Path("docs/model-card.md"): ("not suitable for production",),
    Path("docs/quality-review-ai01.md"): ("Review decision",),
    Path("docs/release-checklist.md"): ("Clean-clone verification",),
    Path("docs/release-evidence-v1.0.0.md"): ("Release decision",),
}


@dataclass(frozen=True)
class ReleaseCheck:
    """One machine-readable release requirement and its observed result."""

    name: str
    passed: bool
    detail: str


def _check_corpus_license(root: Path) -> ReleaseCheck:
    license_path = root / "src/tiny_transformer/data/LICENSE.md"
    corpus_path = root / "src/tiny_transformer/data/tiny_corpus.txt"
    if not license_path.is_file() or not corpus_path.is_file():
        return ReleaseCheck(
            "corpus_license",
            False,
            "corpus or corpus license is missing",
        )
    license_text = license_path.read_text(encoding="utf-8")
    documents = [
        line
        for line in corpus_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    passed = "CC0 1.0" in license_text and len(documents) == 24
    detail = (
        "24 documents with CC0 1.0 metadata"
        if passed
        else f"expected CC0 metadata and 24 documents; found {len(documents)}"
    )
    return ReleaseCheck("corpus_license", passed, detail)


def _check_experiment_artifacts(root: Path) -> ReleaseCheck:
    json_path = root / "docs/experiments/training-results.json"
    svg_path = root / "docs/experiments/training-loss-curves.svg"
    missing = [
        str(path.relative_to(root))
        for path in (json_path, svg_path)
        if not path.is_file()
    ]
    if missing:
        return ReleaseCheck(
            "experiment_artifacts",
            False,
            f"missing {', '.join(missing)}",
        )
    try:
        report = json.loads(json_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
        return ReleaseCheck("experiment_artifacts", False, str(error))
    experiments = report.get("experiments")
    complete_curves = isinstance(experiments, list) and len(experiments) == 3 and all(
        isinstance(experiment, dict)
        and len(experiment.get("train_loss_curve", [])) == 61
        and len(experiment.get("validation_loss_curve", [])) == 61
        for experiment in experiments
    )
    valid_svg = svg_path.read_text(encoding="utf-8").lstrip().startswith("<svg")
    passed = complete_curves and valid_svg
    return ReleaseCheck(
        "experiment_artifacts",
        passed,
        "three complete 60-step curves and SVG are present"
        if passed
        else "training JSON or SVG does not match the release contract",
    )


def _check_documents(root: Path) -> ReleaseCheck:
    problems = []
    for relative_path, markers in REQUIRED_DOCUMENTS.items():
        document = root / relative_path
        if not document.is_file():
            problems.append(f"missing {relative_path}")
            continue
        text = document.read_text(encoding="utf-8")
        for marker in markers:
            if marker not in text:
                problems.append(f"{relative_path} lacks {marker!r}")
    return ReleaseCheck(
        "release_documents",
        not problems,
        "; ".join(problems) if problems else "required release documents are present",
    )


def run_release_checks(
    *, root: Path = ROOT, expected_version: str = EXPECTED_VERSION
) -> list[ReleaseCheck]:
    """Return repository-level checks that complement lint and behavioral tests."""
    return [
        ReleaseCheck(
            "application_version",
            __version__ == expected_version,
            f"expected {expected_version}; found {__version__}",
        ),
        _check_corpus_license(root),
        _check_experiment_artifacts(root),
        _check_documents(root),
    ]


def main(argv: Sequence[str] | None = None) -> int:
    """Print release checks as JSON and return nonzero when any gate fails."""
    if argv:
        raise ValueError("release verification does not accept arguments")
    checks = run_release_checks()
    ready = all(check.passed for check in checks)
    print(
        json.dumps(
            {
                "release": f"v{EXPECTED_VERSION}",
                "ready": ready,
                "checks": [asdict(check) for check in checks],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
