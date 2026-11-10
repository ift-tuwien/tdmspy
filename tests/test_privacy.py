"""The test data must not say what was measured, where, by whom or on which device.

The real files were recorded by data acquisition software; their identifying strings were replaced by neutral
placeholders of the same length. These checks keep new test data (a new real file, a new case) from bringing
such strings back: they look for the kinds of identifier that give a measurement away, not for known names.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from tests.conftest import REAL_DIR, SYNTHETIC_DIR

ROOT = Path(__file__).resolve().parent.parent
PRINTABLE = re.compile(rb"[\x20-\x7e]{4,}")

MAC = re.compile(rb"\b[0-9A-Fa-f]{2}(?::[0-9A-Fa-f]{2}){5}\b")
DOCUMENTATION_MAC = re.compile(rb"^00:00:5[Ee]:00:53:[0-9A-Fa-f]{2}$")  # RFC 7042: reserved for documentation
PATTERNS = {
    "e-mail address": re.compile(
        rb"[\w.+-]{1,64}@[\w-]{1,63}\.[\w.-]{1,63}"
    ),  # bounded: a long string is not scanned slowly
    "IP address": re.compile(rb"\b\d{1,3}(?:\.\d{1,3}){3}\b"),
    "URL": re.compile(rb"https?://", re.IGNORECASE),
    "Windows path": re.compile(rb"\b[A-Za-z]:\\"),
    "user folder": re.compile(rb"(?:/home/|/Users/|\\Users\\)"),
}

DATA_FILES = sorted([*REAL_DIR.glob("*.tdms"), *SYNTHETIC_DIR.glob("*.tdms")])


@pytest.mark.parametrize("path", DATA_FILES, ids=lambda p: f"{p.parent.name}/{p.name}")
def test_no_identifiers_in_the_test_files(path: Path):
    data = path.read_bytes()
    for what, pattern in PATTERNS.items():
        assert not pattern.search(data), f"{path.name}: looks like it contains a {what}"
    for found in MAC.findall(data):
        assert DOCUMENTATION_MAC.match(found), (
            f"{path.name}: a device address ({found.decode()}) that is not a documentation address"
        )


STANDARD_KEYS = {
    "name",
    "wf_increment",
    "wf_start_time",
    "wf_start_offset",
    "unit_string",
}  # defined by the format itself
NEUTRAL_KEY = re.compile(r"^foo_\d\d ?$")
ALLOWED_NAMES = {  # neutral names, and the usual TDMS group names; nothing about what was measured
    "rawdata", "system", "quality_score", "grp1", "grp2", "grp1_Channel1", "grp2_Channel1", "Channel1", "Channel2",
    "Channel3", "external", "info", "extra_items", "group_a", "channel_a", "channel_b", "channel_c",
}  # fmt: skip


def real_summaries() -> dict:
    import json

    return json.loads((REAL_DIR / "expected.json").read_text(encoding="utf-8"))


def test_the_property_keys_of_the_real_files_are_neutral():
    """Only keys the format defines, NI keys and `foo_NN` (the odd trailing space is kept on purpose) may appear."""
    for name, summary in real_summaries().items():
        keys = set(summary["properties"])
        for group in summary["groups"]:
            keys |= set(group["properties"])
            for channel in group["channels"]:
                keys |= set(channel["properties"])
        odd = {k for k in keys if k not in STANDARD_KEYS and not k.startswith("NI_") and not NEUTRAL_KEY.match(k)}
        assert not odd, f"{name}: keys that say something: {sorted(odd)}"


def test_the_group_and_channel_names_of_the_real_files_are_the_allowed_ones():
    for name, summary in real_summaries().items():
        names = {g["name"] for g in summary["groups"]} | {c["name"] for g in summary["groups"] for c in g["channels"]}
        assert names <= ALLOWED_NAMES, (
            f"{name}: names that are not in the allowed list: {sorted(names - ALLOWED_NAMES)}"
        )


def test_the_measurement_names_of_the_real_files_are_placeholders():
    for name, summary in real_summaries().items():
        assert summary["properties"]["name"].startswith("anonymized_"), name
        for group in summary["groups"]:
            for value in group["properties"].values():
                if isinstance(value, str) and value:
                    assert value.isdigit() or value.startswith(
                        ("anonymized", "anon_", "Dev", "Device", "00:00:5E", "bar")
                    ), (name, group["name"], value)


def test_the_source_and_documentation_carry_no_such_strings():
    sources = [p for p in ROOT.glob("tests/*.py") if p.name != "test_privacy.py"]  # this file spells out the patterns
    texts = [
        *ROOT.glob("*.md"),
        *ROOT.glob("doc/*.md"),
        *sources,
        *ROOT.glob("tdmspy/*.py"),
        ROOT / "justfile",
        ROOT / "pyproject.toml",
    ]
    for path in texts:
        data = path.read_bytes()
        for what, pattern in PATTERNS.items():
            if what == "URL":
                continue  # links to documentation and licences are fine in text
            assert not pattern.search(data), f"{path.relative_to(ROOT)}: looks like it contains a {what}"
        for found in MAC.findall(data):
            assert DOCUMENTATION_MAC.match(found), f"{path.relative_to(ROOT)}: a device address ({found.decode()})"
