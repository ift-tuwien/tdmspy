"""The shape of the API (see doc/design.md): navigation, slices, scaling flags, types of what comes back, errors
that are not about the file's content."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import tdmspy
from tdmspy import Timestamp


def test_groups_and_channels_in_file_order_and_by_name(case_file):
    file = tdmspy.read(case_file("first_appearance_order"))
    assert [g.name for g in file.groups] == ["B", "A"]
    assert [c.name for c in file["B"].channels] == ["z", "a"]
    assert "A" in file and "nope" not in file
    assert file["A"]["x"].name == "x" and "y" in file["A"]
    assert [g.name for g in file] == ["B", "A"]  # iterating a file gives its groups
    with pytest.raises(KeyError):
        file["nope"]
    with pytest.raises(KeyError):
        file["A"]["nope"]


def test_the_path_may_be_a_string_or_a_path(case_file):
    path = case_file("minimal")
    assert tdmspy.read(str(path)).properties == tdmspy.read(Path(path)).properties
    assert Path(tdmspy.read(path).path) == path


def test_a_missing_file_is_the_operating_systems_error_not_a_tdms_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        tdmspy.read(tmp_path / "nothing.tdms")
    with pytest.raises(OSError):
        tdmspy.read(tmp_path)  # a folder


def test_a_channel_knows_its_type_and_length(case_file):
    file = tdmspy.read(case_file("integers"))
    channel = file["ints"]["i16"]
    assert channel.dtype == np.dtype("int16") and channel.dtype.isnative
    assert len(channel) == 3
    assert tdmspy.read(case_file("channel_without_data"))["system"]["quality_score"].dtype is None


def test_arrays_come_back_in_native_byte_order_whatever_the_file_has(case_file):
    for name, group, channel in (
        ("big_endian", "g", "ints"),
        ("big_endian", "g", "floats"),
        ("mixed_endianness", "g", "c"),
    ):
        assert tdmspy.read(case_file(name))[group][channel].read().dtype.isnative


def test_a_slice_reads_what_it_names_across_segments(case_file, built):
    file = tdmspy.read(case_file("many_segments"))
    channel = file["g"]["c"]
    everything = channel.read()
    assert len(everything) == 50_000
    for start, stop in ((0, 10), (95, 105), (250, 730), (49_990, 50_000), (49_999, 50_000), (0, 50_000), (7, 7)):
        assert np.array_equal(channel.read(start=start, stop=stop), everything[start:stop]), (start, stop)
    assert np.array_equal(channel.read(start=49_000), everything[49_000:])
    assert np.array_equal(channel.read(stop=3), everything[:3])
    assert len(channel.read(start=49_990, stop=60_000)) == 10  # beyond the end is cut, like a numpy slice
    assert len(channel.read(start=60_000)) == 0


def test_a_slice_of_interleaved_and_big_endian_data(case_file):
    for name, channel in (("interleaved_chunks", "b"), ("big_endian_interleaved", "a"), ("chunks", "a")):
        values = tdmspy.read(case_file(name))["g"][channel]
        full = values.read()
        for start, stop in ((1, 3), (0, 1), (2, len(full)), (len(full) - 1, len(full))):
            assert np.array_equal(values.read(start=start, stop=stop), full[start:stop]), (name, start, stop)


def test_negative_bounds_are_refused(case_file):
    channel = tdmspy.read(case_file("many_segments"))["g"]["c"]
    with pytest.raises(ValueError):
        channel.read(start=-1)
    with pytest.raises(ValueError):
        channel.read(stop=-1)


def test_scaled_and_raw_reads(case_file):
    channel = tdmspy.read(case_file("scale_linear"))["g"]["c"]
    assert channel.read(scaled=False).dtype == np.dtype("int16")
    scaled = channel.read()  # scaled is the default
    assert scaled.dtype == np.dtype("float64")
    assert np.array_equal(scaled, channel.read(scaled=False) * 2.0 + 10.0)
    plain = tdmspy.read(case_file("scale_none"))["g"]["c"]
    assert plain.read().dtype == np.dtype("int16")  # nothing to scale: the values as stored


def test_an_unsupported_scale_refuses_only_the_scaled_read(case_file):
    channel = tdmspy.read(case_file("scale_unsupported_type"))["g"]["c"]
    assert list(channel.read(scaled=False)) == [100, 110]
    with pytest.raises(tdmspy.UnsupportedScalingError):
        channel.read()
    assert issubclass(tdmspy.UnsupportedScalingError, tdmspy.UnsupportedFeatureError)


def test_properties_are_python_values(case_file):
    file = tdmspy.read(case_file("property_types"))
    group, channel = file["g"], file["g"]["c"]
    assert file.properties == {"i8": -8, "u8": 200, "i16": -300, "u16": 60000}
    assert type(group.properties["i32"]) is int and group.properties["u64"] == 2**63 + 5
    assert type(channel.properties["f32"]) is float and channel.properties["f32"] == float(np.float32(0.1))
    assert channel.properties["true"] is True and channel.properties["false"] is False
    assert channel.properties["text"] == "hello"
    assert isinstance(channel.properties["ts"], Timestamp)
    assert channel.properties["ts"] == Timestamp(3_900_000_000, 2**64 - 2**40)
    assert channel.properties["neg"] == Timestamp(-7, 1)


def test_properties_are_in_the_order_they_were_first_written(case_file):
    file = tdmspy.read(case_file("property_update"))
    assert list(file["g"]["c"].properties) == ["a", "b", "c"]
    assert list(file.properties) == ["name", "keep"]


def test_timestamp_channels_are_datetime64_in_microseconds(case_file):
    values = tdmspy.read(case_file("timestamps"))["g"]["t"].read()
    assert values.dtype == np.dtype("datetime64[us]")
    assert values[0] == np.datetime64("1904-01-01T00:00:00.000000")


def test_string_channels_are_object_arrays_of_str(case_file):
    values = tdmspy.read(case_file("strings"))["g"]["s"].read()
    assert values.dtype == object and all(isinstance(v, str) for v in values)
    assert values[2] == "héllo wörld" and values[1] == ""


def test_complex_and_bool_channels(case_file):
    assert tdmspy.read(case_file("complex_numbers"))["g"]["a"].read().dtype == np.dtype("complex64")
    assert tdmspy.read(case_file("complex_numbers"))["g"]["b"].read().dtype == np.dtype("complex128")
    assert tdmspy.read(case_file("booleans"))["g"]["flags"].read().dtype == np.dtype("bool")


def test_reads_are_independent_copies(case_file):
    channel = tdmspy.read(case_file("minimal"))["g"]["c"]
    first = channel.read()
    first[:] = -1
    assert list(channel.read()) == [1.5, -2.5, 0.0]


def test_a_complete_file_is_not_truncated(case_file):
    assert tdmspy.read(case_file("minimal")).truncated is False
    assert tdmspy.read(case_file("next_offset_unknown")).truncated is False


def test_a_file_that_is_read_stays_readable_after_other_files_are_opened(case_file):
    first = tdmspy.read(case_file("minimal"))
    tdmspy.read(case_file("integers"))
    assert list(first["g"]["c"].read()) == [1.5, -2.5, 0.0]
