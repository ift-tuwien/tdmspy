"""Expected values of the real files: `python -m tests.gen_real_expected` (`just real-expected`).

For every file in tests/data/real: its size, properties and, per group and channel, the type, the number of
values and the SHA-256 of the raw values (little-endian bytes), so the JSON stays small and the comparison exact.
`test_real_files.py` checks that `tdmspy.read` reads exactly these values.

The stored values were produced once, with an independent reader, and checked by hand. Regenerate them only
when a real file changes, and then compare the result with the old file before trusting it: the reader under
test must not be the only judge of its own expected values.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

import tdmspy
from tdmspy.timestamp import Timestamp
from tests.truth import Truth, from_tdmspy

REAL_DIR = Path(__file__).resolve().parent / "data" / "real"
EXPECTED = REAL_DIR / "expected.json"


def plain(value: Any) -> Any:
    """A property value as JSON: timestamps as `{"us": ...}` (truncated to microseconds), odd floats as text."""
    if isinstance(value, Timestamp):
        value = value.to_datetime64("us")
    if isinstance(value, np.datetime64):
        return {"us": str(value.astype("datetime64[us]"))}
    if isinstance(value, float) and not math.isfinite(value):
        return {"float": str(value)}
    if isinstance(value, complex):
        return {"complex": [value.real, value.imag]}
    return value


def digest(values: np.ndarray) -> str:
    little = values.astype(values.dtype.newbyteorder("<"), copy=False)
    return hashlib.sha256(little.tobytes()).hexdigest()


def summarize(truth: Truth, size: int) -> dict[str, Any]:
    groups = []
    for name, group in truth.groups.items():
        channels = []
        for channel_name, channel in group.channels.items():
            channels.append(
                {
                    "name": channel_name,
                    "type": channel.type,
                    "length": 0 if channel.values is None else len(channel.values),
                    "sha256": None if channel.values is None else digest(channel.values),
                    "scaled": channel.scaled is not None,
                    "properties": {k: plain(v) for k, v in channel.props.items()},
                }
            )
        groups.append({"name": name, "properties": {k: plain(v) for k, v in group.props.items()}, "channels": channels})
    return {"size": size, "properties": {k: plain(v) for k, v in truth.props.items()}, "groups": groups}


def main() -> None:
    expected = {}
    for path in sorted(REAL_DIR.glob("*.tdms")):
        expected[path.name] = summarize(from_tdmspy(tdmspy.read(path)), path.stat().st_size)
        channels = sum(len(g["channels"]) for g in expected[path.name]["groups"])
        print(f"{path.name:28} {len(expected[path.name]['groups'])} groups, {channels} channels")
    EXPECTED.write_text(json.dumps(expected, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {EXPECTED}")


if __name__ == "__main__":
    main()
