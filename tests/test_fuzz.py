"""Damaged files: the reader may refuse them, but only with a `TdmsError`, quickly, and without allocating wildly."""

from __future__ import annotations

import time
import tracemalloc
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

import tdmspy
from tests.conftest import REAL_DIR
from tests.testcases import CASES

SECONDS_PER_FILE = 5.0
PEAK_BYTES = 300 * 1024 * 1024


def mutations(data: bytes, rng: np.random.Generator, count: int) -> Iterator[tuple[str, bytes]]:
    """Damaged copies of `data`: cut off, bytes changed, a block overwritten, inserted or removed."""
    n = len(data)
    for i in range(count):
        kind = i % 5
        if n < 2:
            yield "empty", data
            continue
        if kind == 0:
            yield f"truncate@{(at := int(rng.integers(0, n)))}", data[:at]
        elif kind == 1:
            out = bytearray(data)
            for _ in range(int(rng.choice([1, 3, 10]))):
                out[int(rng.integers(0, n))] = int(rng.integers(0, 256))
            yield "flip", bytes(out)
        elif kind == 2:
            at, length = int(rng.integers(0, n)), int(rng.integers(1, 64))
            yield (
                f"overwrite@{at}",
                data[:at] + bytes(rng.integers(0, 256, size=length, dtype=np.uint8)) + data[at + length :],
            )
        elif kind == 3:
            at = int(rng.integers(0, n))
            yield (
                f"insert@{at}",
                data[:at] + bytes(rng.integers(0, 256, size=int(rng.integers(1, 40)), dtype=np.uint8)) + data[at:],
            )
        else:
            at, length = int(rng.integers(0, n)), int(rng.integers(1, 64))
            yield f"remove@{at}", data[:at] + data[at + length :]


def read_everything(path: Path) -> None:
    file = tdmspy.read(path)
    for group in file.groups:
        for channel in group.channels:
            channel.read(scaled=False)
            try:
                channel.read()
            except tdmspy.UnsupportedScalingError:
                pass
            channel.read(start=1, stop=5)


def check(name: str, data: bytes, tmp_path: Path, count: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    target = tmp_path / "damaged.tdms"
    for description, damaged in mutations(data, rng, count):
        target.write_bytes(damaged)
        tracemalloc.start()
        started = time.perf_counter()
        try:
            read_everything(target)
        except tdmspy.TdmsError:
            pass  # refused cleanly
        except Exception as error:  # noqa: BLE001
            pytest.fail(f"{name}, {description}: raised {type(error).__name__}: {error}")
        finally:
            elapsed = time.perf_counter() - started
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
        assert elapsed < SECONDS_PER_FILE, f"{name}, {description}: took {elapsed:.1f} s"
        assert peak < PEAK_BYTES, f"{name}, {description}: allocated {peak / 1e6:.0f} MB"


def test_damaged_files_only_raise_tdms_errors(tmp_path, built):
    for case in CASES:
        data = built[case.id].data
        if len(data) < 200_000:  # the large synthetic files are covered by the slow run
            check(case.id, data, tmp_path, count=10, seed=hash(case.id) % 2**32)


def test_damaged_real_files_only_raise_tdms_errors(tmp_path):
    small = REAL_DIR / "three_int16_channels_short.tdms"
    check(small.name, small.read_bytes(), tmp_path, count=20, seed=7)


@pytest.mark.slow
def test_long_fuzz_run(tmp_path, built):
    for case in CASES:
        check(case.id, built[case.id].data, tmp_path, count=300, seed=hash(case.id) % 2**32)
    for path in sorted(REAL_DIR.glob("*.tdms")):
        check(path.name, path.read_bytes(), tmp_path, count=100, seed=11)
