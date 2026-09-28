"""Compare regenerated training JSON with platform-aware numeric tolerance."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path

ABSOLUTE_TOLERANCE = 1e-5


class TrainingEvidenceMismatch(ValueError):
    """Raised when regenerated evidence violates the checked-in contract."""


def _compare(expected: object, actual: object, *, path: str) -> None:
    if isinstance(expected, bool) or isinstance(actual, bool):
        if expected != actual:
            raise TrainingEvidenceMismatch(f"{path}: {actual!r} != {expected!r}")
        return
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        if isinstance(expected, int) and isinstance(actual, int):
            if expected != actual:
                raise TrainingEvidenceMismatch(f"{path}: {actual} != {expected}")
            return
        if not math.isclose(
            float(expected),
            float(actual),
            rel_tol=0.0,
            abs_tol=ABSOLUTE_TOLERANCE,
        ):
            raise TrainingEvidenceMismatch(
                f"{path}: {actual} differs from {expected} by more than "
                f"{ABSOLUTE_TOLERANCE}"
            )
        return
    if isinstance(expected, Mapping) and isinstance(actual, Mapping):
        if expected.keys() != actual.keys():
            raise TrainingEvidenceMismatch(f"{path}: object keys differ")
        for key in expected:
            _compare(expected[key], actual[key], path=f"{path}.{key}")
        return
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            raise TrainingEvidenceMismatch(f"{path}: list lengths differ")
        for index, (expected_item, actual_item) in enumerate(
            zip(expected, actual, strict=True)
        ):
            _compare(expected_item, actual_item, path=f"{path}[{index}]")
        return
    if expected != actual:
        raise TrainingEvidenceMismatch(f"{path}: {actual!r} != {expected!r}")


def verify_training_evidence(expected: object, actual: object) -> None:
    """Require identical structure and bounded platform-level float drift."""
    _compare(expected, actual, path="training_evidence")


def main(argv: Sequence[str] | None = None) -> int:
    """Verify two JSON reports and print the applied numeric contract."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("expected", type=Path)
    parser.add_argument("actual", type=Path)
    args = parser.parse_args(argv)
    expected = json.loads(args.expected.read_text(encoding="utf-8"))
    actual = json.loads(args.actual.read_text(encoding="utf-8"))
    verify_training_evidence(expected, actual)
    print(
        json.dumps(
            {
                "matched": True,
                "absolute_float_tolerance": ABSOLUTE_TOLERANCE,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
