"""The test cases are measured against the catalogue: nothing listed goes unexercised, nothing is exercised unlisted."""

from __future__ import annotations

from pathlib import Path

from tests.catalogue import CATALOGUE, COVERED_BY_TESTS
from tests.gen_catalogue_doc import DOC, render
from tests.testcases import CASES, REAL_FILES

ROOT = Path(__file__).resolve().parent.parent


def test_every_special_case_is_exercised_by_a_synthetic_file_or_a_behaviour_test():
    synthetic = {tag for case in CASES for tag in case.tags}
    missing = [i for i in CATALOGUE if i not in synthetic and i not in COVERED_BY_TESTS]
    assert not missing, f"special cases with no synthetic file and no behaviour test: {missing}"


def test_nothing_is_tagged_that_the_catalogue_does_not_list():
    tagged = {tag for case in CASES for tag in case.tags} | {tag for tags in REAL_FILES.values() for tag in tags}
    assert not tagged - CATALOGUE.keys(), f"unknown ids: {sorted(tagged - CATALOGUE.keys())}"


def test_behaviour_tests_named_in_the_catalogue_exist():
    for identifier, location in COVERED_BY_TESTS.items():
        assert identifier in CATALOGUE
        path, _, name = location.partition("::")
        source = (ROOT / path).read_text(encoding="utf-8")
        assert f"def {name}(" in source, f"{identifier}: {location} does not exist"


def test_case_ids_are_unique_and_described():
    ids = [case.id for case in CASES]
    assert len(ids) == len(set(ids))
    assert all(case.description and case.tags for case in CASES)


def test_the_catalogue_has_substance():
    assert len(CATALOGUE) >= 70 and len(CASES) >= 70


def test_the_generated_documentation_is_current():
    assert DOC.read_text(encoding="utf-8") == render(), "run `just catalogue`"


def test_the_real_files_exist():
    for name in REAL_FILES:
        assert (ROOT / "tests" / "data" / "real" / name).is_file(), name
