"""Real-world files: what they contain, and that tdmspy reads exactly that."""

from __future__ import annotations

import json

import pytest

import tdmspy
from tests.conftest import REAL_DIR
from tests.gen_real_expected import EXPECTED, summarize
from tests.testcases import REAL_FILES
from tests.truth import from_tdmspy

EXPECTED_VALUES = json.loads(EXPECTED.read_text(encoding="utf-8"))
NAMES = sorted(REAL_FILES)


def test_every_real_file_has_expected_values():
    assert sorted(EXPECTED_VALUES) == NAMES


@pytest.mark.parametrize("name", NAMES)
def test_the_real_files_have_the_shape_the_test_cases_rely_on(name: str):
    """Sanity of the data itself: every file has groups, and the claimed cases are visible in the structure."""
    summary = EXPECTED_VALUES[name]
    channels = [c for g in summary["groups"] for c in g["channels"]]
    assert channels and any(c["length"] > 1000 for c in channels)
    types = {c["type"] for c in channels}
    claimed = REAL_FILES[name]
    if "type.no_data_type" in claimed:
        assert None in types
    if "type.integers" in claimed:
        assert types & {"i8", "i16", "i32", "i64", "u8", "u16", "u32", "u64"}
    if "type.floats" in claimed:
        assert types & {"f32", "f64"}
    if "path.group_without_channels" in claimed:
        assert any(not g["channels"] for g in summary["groups"])
    if "path.reserved_looking_names" in claimed:
        assert any(g["name"] in {"info", "name", "groups", "properties"} for g in summary["groups"])
    if "layout.unequal_counts" in claimed:
        lengths = {c["length"] for c in channels if c["length"]}
        assert len(lengths) > 1


@pytest.mark.parametrize("name", NAMES)
def test_tdmspy_reads_the_real_files_as_expected(name: str):
    path = REAL_DIR / name
    file = tdmspy.read(path)
    assert not file.truncated
    assert summarize(from_tdmspy(file), path.stat().st_size) == EXPECTED_VALUES[name]
