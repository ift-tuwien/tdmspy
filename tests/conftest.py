"""Shared fixtures: the stored test cases and real-file folders, and the `case_file` fixture."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.testcases import CASES, Built, Case

REAL_DIR = Path(__file__).parent / "data" / "real"
SYNTHETIC_DIR = Path(__file__).parent / "data" / "synthetic"


@pytest.fixture(scope="session")
def built() -> dict[str, Built]:
    """Every test case, built once."""
    return {case.id: case.build() for case in CASES}


@pytest.fixture
def case_file(tmp_path: Path, built: dict[str, Built]):
    """`case_file(case_or_id) -> Path` of the case's file: the stored one in tests/data/synthetic, or, for the long
    cases that are not stored, a copy built into the test's temporary folder."""

    def get(case: Case | str) -> Path:
        case_id = case if isinstance(case, str) else case.id
        stored = SYNTHETIC_DIR / f"{case_id}.tdms"
        if stored.is_file():
            return stored
        path = tmp_path / f"{case_id}.tdms"
        path.write_bytes(built[case_id].data)
        return path

    return get
