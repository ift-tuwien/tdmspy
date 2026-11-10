"""`tdmspy.read` over all test cases: it must return what the Truth says, or raise what the case demands."""

from __future__ import annotations

import pytest

import tdmspy
from tests.testcases import CASES, Case
from tests.truth import differences, from_tdmspy

READ_CASES = [c for c in CASES if c.build().truth is not None]
ERROR_CASES = [c for c in CASES if c.build().error is not None]
STRICT_CASES = [c for c in CASES if c.build().strict_error is not None]


@pytest.mark.parametrize("case", READ_CASES, ids=lambda c: c.id)
def test_the_reader_returns_the_truth(case: Case, case_file, built):
    file = from_tdmspy(tdmspy.read(case_file(case)))
    problems = differences(built[case.id].truth, file, scaled_rtol=1e-12)
    assert not problems, "\n".join(problems)


@pytest.mark.parametrize("case", ERROR_CASES, ids=lambda c: c.id)
def test_a_broken_or_unsupported_file_raises_the_expected_error(case: Case, case_file, built):
    with pytest.raises(built[case.id].error) as caught:
        file = tdmspy.read(case_file(case))
        for group in file.groups:  # structure errors come at open; nothing may slip through to the data either
            for channel in group.channels:
                channel.read(scaled=False)
    assert isinstance(caught.value, tdmspy.TdmsError)
    assert str(caught.value), "the message says what is wrong"


@pytest.mark.parametrize("case", STRICT_CASES, ids=lambda c: c.id)
def test_strict_mode_refuses_what_lenient_mode_reads_partially(case: Case, case_file, built):
    with pytest.raises(built[case.id].strict_error):
        tdmspy.read(case_file(case), strict=True)
    assert tdmspy.read(case_file(case)).truncated is True


@pytest.mark.parametrize("case", [c for c in READ_CASES if not c.build().truth.truncated], ids=lambda c: c.id)
def test_strict_mode_reads_every_complete_file_too(case: Case, case_file, built):
    problems = differences(built[case.id].truth, from_tdmspy(tdmspy.read(case_file(case), strict=True)))
    assert not problems, "\n".join(problems)
