"""The stored synthetic files are exactly what the test writer writes for the test cases."""

from __future__ import annotations

import pytest

from tests.conftest import SYNTHETIC_DIR
from tests.testcases import CASES

STORED = [c for c in CASES if c.stored]


def test_every_stored_case_has_its_file_and_every_file_its_case():
    files = {p.name for p in SYNTHETIC_DIR.glob("*.tdms")}
    assert files == {f"{c.id}.tdms" for c in STORED}, "run `just testcases`"


@pytest.mark.parametrize("case", STORED, ids=lambda c: c.id)
def test_the_stored_file_is_what_the_builder_writes(case, built):
    assert (SYNTHETIC_DIR / f"{case.id}.tdms").read_bytes() == built[case.id].data, "run `just testcases`"


def test_the_stored_test_cases_stay_small():
    assert sum(p.stat().st_size for p in SYNTHETIC_DIR.glob("*.tdms")) < 2_000_000
