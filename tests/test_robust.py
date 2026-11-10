"""Laziness and scale: opening reads the structure, a slice reads only its part, big files stay proportional."""

from __future__ import annotations

import time
import tracemalloc

import numpy as np
import pytest

import tdmspy
from tests.builder import Segment, tdms_file


def write_big(path, values: int) -> np.ndarray:
    data = np.arange(values, dtype="<f8")
    path.write_bytes(tdms_file(Segment().channel("g", "c", data)))
    return data


def test_opening_reads_the_structure_not_the_data(tmp_path):
    data = write_big(tmp_path / "big.tdms", 2_500_000)  # 20 MB
    tracemalloc.start()
    file = tdmspy.read(tmp_path / "big.tdms")
    channel = file["g"]["c"]
    assert len(channel) == len(data)
    _, opened = tracemalloc.get_traced_memory()
    part = channel.read(start=1_000_000, stop=1_001_000)
    _, sliced = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert np.array_equal(part, data[1_000_000:1_001_000])
    assert opened < 2 * 1024 * 1024, f"opening allocated {opened / 1e6:.1f} MB for a 20 MB file"
    assert sliced < 3 * 1024 * 1024, f"a 1000 value slice allocated {sliced / 1e6:.1f} MB"


def test_many_small_segments_open_in_reasonable_time(tmp_path, case_file):
    path = case_file("many_segments_5000")
    started = time.perf_counter()
    channel = tdmspy.read(path)["g"]["c"]
    elapsed = time.perf_counter() - started
    assert len(channel) == 50_000
    assert elapsed < 3.0, f"opening 5000 segments took {elapsed:.1f} s"
    assert np.array_equal(channel.read(), np.arange(50_000, dtype="<i4"))


@pytest.mark.slow
def test_a_hundred_megabyte_channel_reads_in_proportion(tmp_path):
    data = write_big(tmp_path / "huge.tdms", 12_500_000)  # 100 MB
    started = time.perf_counter()
    file = tdmspy.read(tmp_path / "huge.tdms")
    opened = time.perf_counter() - started
    started = time.perf_counter()
    values = file["g"]["c"].read()
    read = time.perf_counter() - started
    assert np.array_equal(values, data)
    assert opened < 1.0, f"opening took {opened:.2f} s"
    assert read < 5.0, f"reading 100 MB took {read:.2f} s"
