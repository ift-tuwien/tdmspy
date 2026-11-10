"""The synthetic test cases: one small file per special case, each with what it must read as.

A case is `(id, tags, build)`. `build()` returns the bytes of the file and either its `Truth` (what the
reader must return) or the exception the reader must raise. `tags` are ids of `catalogue.CATALOGUE`: the
cases are how the catalogue is exercised, and `test_catalogue_coverage.py` insists on every id being used.

The files themselves are stored in `tests/data/synthetic` (`just testcases` rewrites them; a test fails if they
differ from what the builder writes), except the long ones (`stored=False`), which are built when needed.

Tests over the test cases:
- `test_reader_testcases.py` requires `tdmspy.read` to return the Truth, or raise the error the case demands;
- `test_fuzz.py` damages the files and requires only clean errors;
- `test_synthetic_files.py` checks the stored files.
"""

from __future__ import annotations

import struct
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

import tdmspy
from tdmspy.timestamp import Timestamp
from tests.builder import BIG, UNKNOWN_NEXT_OFFSET, ObjectSpec, Prop, Segment, pack_string, raw_segment, tdms_file
from tests.truth import Truth


@dataclass
class Built:
    data: bytes
    truth: Truth | None = None
    error: type[Exception] | None = None
    """The exception `tdmspy.read` must raise (for a file that is rejected)."""
    strict_error: type[Exception] | None = None
    """The exception `tdmspy.read(strict=True)` must raise, for a file that is only read leniently (truncated)."""


@dataclass(frozen=True)
class Case:
    id: str
    tags: tuple[str, ...]
    description: str
    build: Callable[[], Built]
    stored: bool = True
    """The file is stored in tests/data/synthetic (False for the long ones, which are built when a test needs them)."""


CASES: list[Case] = []


def case(id: str, tags: list[str], description: str, *, stored: bool = True):
    def register(build: Callable[[], Built]) -> Callable[[], Built]:
        CASES.append(Case(id, tuple(tags), description, build, stored))
        return build

    return register


def arr(values: Any, dtype: str) -> np.ndarray:
    return np.array(values, dtype=dtype)


# =====================================================================================================
# the file and its segments
# =====================================================================================================


@case(
    "minimal",
    ["file.minimal", "version.4713", "prop.root_group_channel"],
    "One segment: root, group and channel with properties.",
)
def _minimal() -> Built:
    values = arr([1.5, -2.5, 0.0], "<f8")
    props = (Prop("unit_string", "V"), Prop("gain", 2.5))
    seg = Segment().root(Prop("name", "minimal")).group("g", Prop("note", "hello")).channel("g", "c", values, *props)
    truth = Truth().root(Prop("name", "minimal")).group("g", Prop("note", "hello")).channel("g", "c", values, *props)
    return Built(tdms_file(seg), truth)


@case("version_4712", ["version.4712"], "A file written as TDMS 1.0 (version 4712).")
def _version_4712() -> Built:
    values = arr([1, 2, 3], "<i2")
    return Built(tdms_file(Segment(version=4712).channel("g", "c", values)), Truth().channel("g", "c", values))


@case("version_unknown", ["version.unknown_is_error"], "A segment of version 4714.")
def _version_unknown() -> Built:
    return Built(tdms_file(Segment(version=4714).channel("g", "c", arr([1], "<i2"))), error=tdmspy.TdmsFormatError)


@case("empty_file", ["file.empty_is_error"], "0 bytes.")
def _empty_file() -> Built:
    return Built(b"", error=tdmspy.TdmsFormatError)


@case("shorter_than_lead_in", ["file.empty_is_error"], "27 bytes of a valid file: not even a lead-in.")
def _short_file() -> Built:
    return Built(tdms_file(Segment().channel("g", "c", arr([1], "<i2")))[:27], error=tdmspy.TdmsFormatError)


@case("wrong_signature", ["file.wrong_signature_is_error"], "A file that does not start with TDSm.")
def _wrong_signature() -> Built:
    data = bytearray(tdms_file(Segment().channel("g", "c", arr([1], "<i2"))))
    data[:4] = b"TDSx"
    return Built(bytes(data), error=tdmspy.TdmsFormatError)


@case("index_file", ["file.index_file_is_error"], "A .tdms_index file (signature TDSh).")
def _index_file() -> Built:
    return Built(tdms_file(Segment(tag=b"TDSh").channel("g", "c", arr([1], "<i2"))), error=tdmspy.TdmsFormatError)


@case("garbage_after_segments", ["file.not_tdms_inside"], "Valid segments followed by 80 bytes of text.")
def _garbage_after() -> Built:
    return Built(
        tdms_file(Segment().channel("g", "c", arr([1], "<i2"))) + b"this is not a segment " * 4,
        error=tdmspy.TdmsFormatError,
    )


@case(
    "many_segments", ["segment.many", "index.same_as_previous"], "500 segments of one channel, the index written once."
)
def _many_segments() -> Built:
    segments, parts = [], []
    for i in range(500):
        values = arr(np.arange(100) + 100 * i, "<f4")
        parts.append(values)
        segments.append(Segment().channel("g", "c", values, index="explicit" if i == 0 else "same"))
    return Built(tdms_file(*segments), Truth().channel("g", "c", np.concatenate(parts)))


@case(
    "metadata_only_segment",
    ["segment.metadata_only", "index.none"],
    "A segment that only declares objects and properties; data comes later.",
)
def _metadata_only() -> Built:
    first = Segment().group("g", Prop("created", "today")).channel("g", "c", None, Prop("unit_string", "A"))
    second = Segment().channel("g", "c", arr([7, 8, 9, 10], "<i4"))
    truth = (
        Truth()
        .group("g", Prop("created", "today"))
        .channel("g", "c", arr([7, 8, 9, 10], "<i4"), Prop("unit_string", "A"))
    )
    return Built(tdms_file(first, second), truth)


@case("raw_only_segment", ["segment.raw_only"], "A segment without metadata continues the previous segment's layout.")
def _raw_only() -> Built:
    first = Segment().channel("g", "c", arr([1.0, 2.0, 3.0], "<f8")).channel("g", "d", arr([10.0, 20.0], "<f8"))
    second = (
        Segment(meta=False)
        .channel("g", "c", arr([4.0, 5.0, 6.0], "<f8"), index="same")
        .channel("g", "d", arr([30.0, 40.0], "<f8"), index="same")
    )
    truth = Truth().channel("g", "c", arr([1, 2, 3, 4, 5, 6], "<f8")).channel("g", "d", arr([10, 20, 30, 40], "<f8"))
    return Built(tdms_file(first, second), truth)


@case(
    "meta_without_new_list",
    ["segment.meta_without_new_list", "index.changed_count"],
    "Metadata that updates A only; B, left out of the list, still gets its data.",
)
def _meta_without_new_list() -> Built:
    first = Segment().channel("g", "a", arr([1, 2, 3, 4], "<i4")).channel("g", "b", arr([10, 20, 30, 40], "<i4"))
    second = (
        Segment(newlist=False)
        .channel("g", "a", arr([5, 6], "<i4"), Prop("seen", True))
        .channel("g", "b", arr([50, 60, 70, 80], "<i4"), index="same", listed=False)
    )
    truth = (
        Truth()
        .channel("g", "a", arr([1, 2, 3, 4, 5, 6], "<i4"), Prop("seen", True))
        .channel("g", "b", arr([10, 20, 30, 40, 50, 60, 70, 80], "<i4"))
    )
    return Built(tdms_file(first, second), truth)


@case(
    "new_list_replaces",
    ["segment.new_list_replaces", "layout.channel_added_later", "layout.channel_dropped"],
    "A new object list drops A, keeps B and adds C; a third brings A back.",
)
def _new_list_replaces() -> Built:
    first = Segment().channel("g", "a", arr([1, 2], "<i2")).channel("g", "b", arr([3, 4], "<i2"))
    second = Segment().channel("g", "b", arr([5, 6], "<i2"), index="same").channel("g", "c", arr([7, 8], "<i2"))
    third = Segment().channel("g", "a", arr([9], "<i2"))
    truth = (
        Truth()
        .channel("g", "a", arr([1, 2, 9], "<i2"))
        .channel("g", "b", arr([3, 4, 5, 6], "<i2"))
        .channel("g", "c", arr([7, 8], "<i2"))
    )
    return Built(tdms_file(first, second, third), truth)


@case("raw_offset_padding", ["segment.raw_offset_padding"], "The raw data starts 7 bytes after the metadata ends.")
def _raw_offset_padding() -> Built:
    values = arr([1.5, 2.5, 3.5], "<f8")
    return Built(tdms_file(Segment(padding=7).channel("g", "c", values)), Truth().channel("g", "c", values))


@case(
    "zero_values",
    ["segment.zero_values", "index.changed_count"],
    "A channel announced with 0 values, then with 3; a segment with no values at all.",
)
def _zero_values() -> Built:
    first = Segment().channel("g", "c", arr([], "<i2")).channel("g", "d", arr([1, 2], "<i2"))
    second = Segment().channel("g", "c", arr([5, 6, 7], "<i2")).channel("g", "d", arr([], "<i2"))
    third = Segment().group("g")
    truth = Truth().channel("g", "c", arr([5, 6, 7], "<i2")).channel("g", "d", arr([1, 2], "<i2"))
    return Built(tdms_file(first, second, third), truth)


@case(
    "next_offset_unknown",
    ["segment.next_offset_unknown"],
    "The last segment's next segment offset is 0xFFFFFFFFFFFFFFFF (still being written), its data complete.",
)
def _next_offset_unknown() -> Built:
    first = Segment().channel("g", "c", arr([1, 2, 3], "<f8"))
    last = Segment(next_offset=UNKNOWN_NEXT_OFFSET).channel("g", "c", arr([4, 5, 6], "<f8"), index="same")
    return Built(tdms_file(first, last), Truth().channel("g", "c", arr([1, 2, 3, 4, 5, 6], "<f8")))


@case("chunks", ["segment.chunks"], "One segment, three chunks of two channels.")
def _chunks() -> Built:
    seg = Segment().channel("g", "a", arr([1, 2], "<i2")).channel("g", "b", arr([0.5], "<f4"))
    seg.chunk_of({("g", "a"): arr([3, 4], "<i2"), ("g", "b"): arr([1.5], "<f4")})
    seg.chunk_of({("g", "a"): arr([5, 6], "<i2"), ("g", "b"): arr([2.5], "<f4")})
    truth = Truth().channel("g", "a", arr([1, 2, 3, 4, 5, 6], "<i2")).channel("g", "b", arr([0.5, 1.5, 2.5], "<f4"))
    return Built(tdms_file(seg), truth)


# -- truncated files ---------------------------------------------------------------------------------


def _lenient(truth: Truth) -> Truth:
    truth.truncated = True
    return truth


@case(
    "truncated_mid_values",
    ["segment.truncated_file", "segment.partial_chunk"],
    "One channel of 10 float64 values, the file cut after 7.5 of them.",
)
def _truncated_mid_values() -> Built:
    data = tdms_file(Segment().channel("g", "c", arr(np.arange(10.0), "<f8")))
    cut = data[: len(data) - 20]  # 80 bytes of data, 60 left: 7 complete values
    return Built(
        cut, _lenient(Truth().channel("g", "c", arr(np.arange(7.0), "<f8"))), strict_error=tdmspy.TdmsFormatError
    )


@case(
    "truncated_in_metadata",
    ["segment.truncated_file"],
    "A complete segment, then a second cut off inside its metadata.",
)
def _truncated_in_metadata() -> Built:
    first = tdms_file(Segment().channel("g", "c", arr([1, 2, 3], "<i2")))
    second = tdms_file(Segment().channel("g", "c", arr([4, 5, 6], "<i2"), Prop("late", 1)))
    return Built(
        first + second[:40],
        _lenient(Truth().channel("g", "c", arr([1, 2, 3], "<i2"))),
        strict_error=tdmspy.TdmsFormatError,
    )


@case(
    "truncated_in_lead_in", ["segment.truncated_file"], "A complete segment followed by 10 bytes of the next lead-in."
)
def _truncated_in_lead_in() -> Built:
    first = tdms_file(Segment().channel("g", "c", arr([1, 2, 3], "<i2")))
    second = tdms_file(Segment().channel("g", "c", arr([4], "<i2"), index="same"))
    return Built(
        first + second[:10],
        _lenient(Truth().channel("g", "c", arr([1, 2, 3], "<i2"))),
        strict_error=tdmspy.TdmsFormatError,
    )


@case(
    "truncated_interleaved_row",
    ["segment.truncated_file", "segment.partial_chunk", "layout.interleaved"],
    "Interleaved int16 pair, 10 rows, cut inside row 4: three whole rows remain.",
)
def _truncated_interleaved() -> Built:
    a, b = arr(np.arange(10), "<i2"), arr(np.arange(10) * 10, "<i2")
    data = tdms_file(Segment(interleaved=True).channel("g", "a", a).channel("g", "b", b))
    cut = data[: len(data) - (10 * 4 - (3 * 4 + 2))]
    truth = _lenient(Truth().channel("g", "a", a[:3]).channel("g", "b", b[:3]))
    return Built(cut, truth, strict_error=tdmspy.TdmsFormatError)


@case(
    "truncated_second_channel",
    ["segment.truncated_file", "segment.partial_chunk", "layout.contiguous_multi_channel"],
    "Two contiguous channels of 5 values; the cut leaves channel 1 whole and 3.5 values of channel 2.",
)
def _truncated_second_channel() -> Built:
    a, b = arr(np.arange(5), "<i2"), arr(np.arange(5) + 100, "<i2")
    data = tdms_file(Segment().channel("g", "a", a).channel("g", "b", b))
    cut = data[: len(data) - 3]  # 10 bytes of b: 5 values; minus 3 bytes: 3 whole values (6 bytes) + 1 byte
    return Built(
        cut, _lenient(Truth().channel("g", "a", a).channel("g", "b", b[:3])), strict_error=tdmspy.TdmsFormatError
    )


@case(
    "truncated_after_whole_chunks",
    ["segment.truncated_file", "segment.chunks"],
    "Three chunks announced by the lead-in, the file holds two and a half.",
)
def _truncated_chunks() -> Built:
    seg = Segment().channel("g", "a", arr([1, 2], "<i2"))
    seg.chunk_of({("g", "a"): arr([3, 4], "<i2")}).chunk_of({("g", "a"): arr([5, 6], "<i2")})
    data = tdms_file(seg)
    cut = data[: len(data) - 2]  # the last chunk loses its second value
    return Built(
        cut, _lenient(Truth().channel("g", "a", arr([1, 2, 3, 4, 5], "<i2"))), strict_error=tdmspy.TdmsFormatError
    )


@case(
    "next_offset_beyond_end",
    ["segment.next_offset_beyond_end"],
    "A last segment whose lead-in announces 1000 bytes more than the file has.",
)
def _next_offset_beyond_end() -> Built:
    values = arr([1, 2, 3], "<i2")
    seg = Segment().channel("g", "c", values)
    seg.next_offset = len(seg.build()) - 28 + 1000
    return Built(tdms_file(seg), _lenient(Truth().channel("g", "c", values)), strict_error=tdmspy.TdmsFormatError)


# =====================================================================================================
# the raw data index
# =====================================================================================================


@case(
    "index_none_then_data",
    ["index.none"],
    "B is in the list without data in the first segment and gets some in the second.",
)
def _index_none() -> Built:
    first = Segment().channel("g", "a", arr([1, 2, 3], "<i2")).channel("g", "b", None, Prop("unit_string", "Pa"))
    second = Segment().channel("g", "b", arr([9], "<i2"))
    truth = (
        Truth().channel("g", "a", arr([1, 2, 3], "<i2")).channel("g", "b", arr([9], "<i2"), Prop("unit_string", "Pa"))
    )
    return Built(tdms_file(first, second), truth)


@case(
    "index_same_after_change",
    ["index.same_as_previous", "index.changed_count"],
    "Counts 3, then an explicit 2, then 'same' meaning 2.",
)
def _index_same_after_change() -> Built:
    first = Segment().channel("g", "c", arr([1, 2, 3], "<i2"))
    second = Segment().channel("g", "c", arr([4, 5], "<i2"))
    third = Segment().channel("g", "c", arr([6, 7], "<i2"), index="same")
    return Built(tdms_file(first, second, third), Truth().channel("g", "c", arr([1, 2, 3, 4, 5, 6, 7], "<i2")))


@case("index_type_change", ["index.type_change_is_error"], "A channel that is int16 and then float32.")
def _index_type_change() -> Built:
    first = Segment().channel("g", "c", arr([1, 2], "<i2"))
    second = Segment().channel("g", "c", arr([1.0, 2.0], "<f4"))
    return Built(tdms_file(first, second), error=tdmspy.TdmsFormatError)


@case(
    "index_same_without_previous",
    ["index.same_without_previous_is_error"],
    "Index 0 for a channel that never had an explicit index.",
)
def _index_same_without_previous() -> Built:
    return Built(tdms_file(Segment().channel("g", "c", arr([1, 2], "<i2"), index="same")), error=tdmspy.TdmsFormatError)


# =====================================================================================================
# the layout of the raw data
# =====================================================================================================


@case(
    "contiguous_three_types",
    ["layout.contiguous_multi_channel"],
    "Three channels of different types, one after the other in each chunk.",
)
def _contiguous() -> Built:
    a, b, c = arr([1, 2, 3], "<i2"), arr([0.5, 1.5, 2.5], "<f4"), arr([-1.0, -2.0, -3.0], "<f8")
    seg = Segment().channel("g", "a", a).channel("g", "b", b).channel("g", "c", c)
    return Built(tdms_file(seg), Truth().channel("g", "a", a).channel("g", "b", b).channel("g", "c", c))


@case("unequal_counts", ["layout.unequal_counts"], "Channels of 4, 2 and 5 values in one segment.")
def _unequal_counts() -> Built:
    a, b, c = arr(np.arange(4), "<i4"), arr([7, 8], "<i4"), arr(np.arange(5) * 2, "<i4")
    seg = Segment().channel("g", "a", a).channel("g", "b", b).channel("g", "c", c)
    return Built(tdms_file(seg), Truth().channel("g", "a", a).channel("g", "b", b).channel("g", "c", c))


@case("interleaved_two_channels", ["layout.interleaved"], "Two int16 channels interleaved.")
def _interleaved() -> Built:
    a, b = arr(np.arange(6), "<i2"), arr(np.arange(6) * -3, "<i2")
    return Built(
        tdms_file(Segment(interleaved=True).channel("g", "a", a).channel("g", "b", b)),
        Truth().channel("g", "a", a).channel("g", "b", b),
    )


@case("interleaved_three_channels_floats", ["layout.interleaved"], "Three float32 channels interleaved.")
def _interleaved_three() -> Built:
    a, b, c = (arr(np.arange(5) + k / 10, "<f4") for k in range(3))
    seg = Segment(interleaved=True).channel("g", "a", a).channel("g", "b", b).channel("g", "c", c)
    return Built(tdms_file(seg), Truth().channel("g", "a", a).channel("g", "b", b).channel("g", "c", c))


@case(
    "interleaved_chunks",
    ["layout.interleaved_chunks", "segment.chunks"],
    "Interleaved data in three chunks, then a second segment.",
)
def _interleaved_chunks() -> Built:
    seg = Segment(interleaved=True).channel("g", "a", arr([1, 2], "<i2")).channel("g", "b", arr([10, 20], "<i2"))
    seg.chunk_of({("g", "a"): arr([3, 4], "<i2"), ("g", "b"): arr([30, 40], "<i2")})
    seg.chunk_of({("g", "a"): arr([5, 6], "<i2"), ("g", "b"): arr([50, 60], "<i2")})
    tail = (
        Segment(interleaved=True)
        .channel("g", "a", arr([7, 8], "<i2"), index="same")
        .channel("g", "b", arr([70, 80], "<i2"), index="same")
    )
    truth = Truth().channel("g", "a", arr(np.arange(1, 9), "<i2")).channel("g", "b", arr(np.arange(1, 9) * 10, "<i2"))
    return Built(tdms_file(seg, tail), truth)


@case(
    "interleaved_mixed_types",
    ["layout.interleaved_mixed_types"],
    "Interleaved channels of different sizes: int16 next to float64.",
)
def _interleaved_mixed() -> Built:
    a, b = arr([1, 2, 3], "<i2"), arr([0.25, 0.5, 0.75], "<f8")
    return Built(
        tdms_file(Segment(interleaved=True).channel("g", "a", a).channel("g", "b", b)),
        Truth().channel("g", "a", a).channel("g", "b", b),
    )


@case(
    "interleaved_unequal_counts",
    ["layout.interleaved_unequal_is_error"],
    "Interleaved channels whose indexes say 3 and 4 values.",
)
def _interleaved_unequal() -> Built:
    seg = Segment(interleaved=True).channel("g", "a", arr([1, 2, 3], "<i2")).channel("g", "b", arr([1, 2, 3, 4], "<i2"))
    return Built(tdms_file(seg), error=tdmspy.TdmsFormatError)


@case(
    "big_endian",
    ["layout.big_endian"],
    "A big-endian segment: lead-in, metadata (properties of several types) and int16, float64, string and timestamp data (timestamps store seconds before the fraction).",
)
def _big_endian() -> Built:
    ints, floats = arr([1, -2, 300], ">i2"), arr([0.5, -1.25], ">f8")
    strings = ["alpha", "", "γ"]
    stamps = [Timestamp(3_900_000_000, 1 << 60), Timestamp(-10, 5)]
    props = (Prop("i", 70000), Prop("f", 2.5), Prop("s", "grün"), Prop("t", Timestamp(5, 7)))
    seg = Segment(endian=BIG).group("g", *props)
    seg.channel("g", "ints", ints).channel("g", "floats", floats).channel("g", "strings", strings).channel(
        "g", "stamps", stamps
    )
    truth = Truth().group("g", *props)
    truth.channel("g", "ints", ints.astype("=i2")).channel("g", "floats", floats.astype("=f8")).channel(
        "g", "strings", strings
    ).channel("g", "stamps", stamps)
    return Built(tdms_file(seg), truth)


@case(
    "big_endian_interleaved",
    ["layout.big_endian", "layout.interleaved"],
    "Big-endian interleaved int32 and float32, two chunks.",
)
def _big_endian_interleaved() -> Built:
    a, b = arr([1, 2], ">i4"), arr([0.5, 1.5], ">f4")
    seg = Segment(endian=BIG, interleaved=True).channel("g", "a", a).channel("g", "b", b)
    seg.chunk_of({("g", "a"): arr([3, 4], ">i4"), ("g", "b"): arr([2.5, 3.5], ">f4")})
    truth = Truth().channel("g", "a", arr([1, 2, 3, 4], "=i4")).channel("g", "b", arr([0.5, 1.5, 2.5, 3.5], "=f4"))
    return Built(tdms_file(seg), truth)


@case(
    "mixed_endianness",
    ["layout.mixed_endianness", "index.same_as_previous"],
    "The same channel in a little-endian, a big-endian and a little-endian segment.",
)
def _mixed_endianness() -> Built:
    first = Segment().channel("g", "c", arr([1, 2], "<i4"))
    second = Segment(endian=BIG).channel("g", "c", arr([3, 4], ">i4"), index="same")
    third = Segment().channel("g", "c", arr([5, 6], "<i4"), index="same")
    return Built(tdms_file(first, second, third), Truth().channel("g", "c", arr([1, 2, 3, 4, 5, 6], "=i4")))


# =====================================================================================================
# data types
# =====================================================================================================


@case("integers", ["type.integers"], "All eight integer types with their extreme values.")
def _integers() -> Built:
    values = {
        "i8": arr([-128, 0, 127], "<i1"), "u8": arr([0, 1, 255], "<u1"), "i16": arr([-32768, 0, 32767], "<i2"),
        "u16": arr([0, 1, 65535], "<u2"), "i32": arr([-(2**31), 0, 2**31 - 1], "<i4"), "u32": arr([0, 1, 2**32 - 1], "<u4"),
        "i64": arr([-(2**63), 0, 2**63 - 1], "<i8"), "u64": arr([0, 1, 2**64 - 1], "<u8"),
    }  # fmt: skip
    seg, truth = Segment(), Truth()
    for name, v in values.items():
        seg.channel("ints", name, v)
        truth.channel("ints", name, v)
    return Built(tdms_file(seg), truth)


@case("floats", ["type.floats"], "float32 and float64 with NaN, infinities, negative zero and subnormals.")
def _floats() -> Built:
    f32 = arr([np.nan, np.inf, -np.inf, -0.0, 1e-45, 3.4028235e38, 0.1], "<f4")
    f64 = arr([np.nan, np.inf, -np.inf, -0.0, 5e-324, 1.7976931348623157e308, 0.1], "<f8")
    seg = Segment().channel("g", "f32", f32).channel("g", "f64", f64)
    return Built(tdms_file(seg), Truth().channel("g", "f32", f32).channel("g", "f64", f64))


@case("booleans", ["type.bool"], "A boolean channel.")
def _booleans() -> Built:
    values = np.array([True, False, True, True])
    return Built(tdms_file(Segment().channel("g", "flags", values)), Truth().channel("g", "flags", values))


@case(
    "strings", ["type.string", "index.string"], "String channels over two segments, with empty and non-ASCII strings."
)
def _strings() -> Built:
    first = ["alpha", "", "héllo wörld", "日本語", "x" * 300]
    second = ["tail", ""]
    segments = [Segment().channel("g", "s", first), Segment().channel("g", "s", second)]
    return Built(tdms_file(*segments), Truth().channel("g", "s", first + second))


@case("timestamps", ["type.timestamp"], "A timestamp channel, with a fraction of 0, 2**63 and negative seconds.")
def _timestamps() -> Built:
    values = [
        Timestamp(0, 0),
        Timestamp(3_900_000_000, 1 << 63),
        Timestamp(-5, 123456789012345),
        Timestamp(3_800_000_000, 2**64 - 2**40),
    ]
    return Built(tdms_file(Segment().channel("g", "t", values)), Truth().channel("g", "t", values))


@case(
    "timestamp_fraction_extremes",
    ["type.timestamp", "prop.timestamp_exact"],
    "Timestamps whose fraction is 2**64-1 and 2**64-2**20: the microsecond is truncated, never rounded up to the next second.",
)
def _timestamp_extremes() -> Built:
    values = [
        Timestamp(3_800_000_000, 2**64 - 1),
        Timestamp(3_800_000_000, 2**64 - 2**20),
        Timestamp(3_800_000_000, 2**64 - 2**44),
    ]
    props = (Prop("last", values[0]), Prop("almost", values[1]))
    seg = Segment().group("g", *props).channel("g", "t", values, *props)
    return Built(tdms_file(seg), Truth().group("g", *props).channel("g", "t", values, *props))


@case("complex_numbers", ["type.complex"], "Complex single and double channels.")
def _complex() -> Built:
    a, b = np.array([1 + 2j, -0.5j], dtype="<c8"), np.array([3.25 - 1j, 0j, 1e10 + 1e-10j], dtype="<c16")
    return Built(
        tdms_file(Segment().channel("g", "a", a).channel("g", "b", b)),
        Truth().channel("g", "a", a).channel("g", "b", b),
    )


@case("channel_without_data", ["type.no_data_type", "index.none"], "A channel that has properties but never data.")
def _no_data_type() -> Built:
    props = (Prop("wf_increment", 0.1), Prop("wf_start_time", Timestamp(3_900_000_000, 12345)))
    seg = (
        Segment().group("system").channel("system", "quality_score", None, *props).channel("g", "x", arr([1, 2], "<i2"))
    )
    truth = (
        Truth().group("system").channel("system", "quality_score", None, *props).channel("g", "x", arr([1, 2], "<i2"))
    )
    return Built(tdms_file(seg, Segment().channel("system", "quality_score", None)), truth)


@case("extended_float_channel", ["type.unsupported_is_error"], "A channel of 80-bit extended floats.")
def _extended_channel() -> Built:
    index = struct.pack("<IIIQ", 20, 11, 1, 2)
    seg = Segment(trailing_raw=bytes(32)).channel("g", "c", None, index_bytes=index)
    return Built(tdms_file(seg), error=tdmspy.UnsupportedFeatureError)


# =====================================================================================================
# properties
# =====================================================================================================


@case(
    "property_types",
    ["prop.all_types", "prop.root_group_channel", "prop.timestamp_exact"],
    "Properties of all types on the root, a group and a channel.",
)
def _property_types() -> Built:
    root = (Prop("i8", -8, "i8"), Prop("u8", 200, "u8"), Prop("i16", -300, "i16"), Prop("u16", 60000, "u16"))
    group = (
        Prop("i32", -70000),
        Prop("u32", 4_000_000_000, "u32"),
        Prop("i64", -(2**40), "i64"),
        Prop("u64", 2**63 + 5, "u64"),
    )
    channel = (
        Prop("f32", 0.1, "f32"), Prop("f64", 0.1), Prop("true", True), Prop("false", False), Prop("text", "hello"),
        Prop("ts", Timestamp(3_900_000_000, 2**64 - 2**40)), Prop("ts0", Timestamp(0, 0)), Prop("neg", Timestamp(-7, 1)),
    )  # fmt: skip
    seg = Segment().root(*root).group("g", *group).channel("g", "c", arr([1], "<i2"), *channel)
    truth = Truth().root(*root).group("g", *group).channel("g", "c", arr([1], "<i2"), *channel)
    return Built(tdms_file(seg), truth)


@case("property_complex", ["prop.all_types"], "Complex single and double properties.")
def _property_complex() -> Built:
    props = (Prop("c64", 1.5 - 2j, "c64"), Prop("c128", -0.25 + 8j))
    return Built(
        tdms_file(Segment().channel("g", "c", arr([1], "<i2"), *props)),
        Truth().channel("g", "c", arr([1], "<i2"), *props),
    )


@case(
    "property_unit_types",
    ["prop.all_types"],
    "Float properties with a unit (types 0x19 and 0x1A) read as plain floats.",
)
def _property_unit_types() -> Built:
    props = (Prop("single", 1.5, "f32u"), Prop("double", -2.25, "f64u"))
    return Built(
        tdms_file(Segment().channel("g", "c", arr([1], "<i2"), *props)),
        Truth().channel("g", "c", arr([1], "<i2"), *props),
    )


@case(
    "property_update",
    ["prop.update", "prop.added_later", "index.same_as_previous"],
    "A property rewritten and another added in a later segment, on the root, a group and a channel.",
)
def _property_update() -> Built:
    first = (
        Segment()
        .root(Prop("name", "run 1"), Prop("keep", 1))
        .group("g", Prop("a", 1))
        .channel("g", "c", arr([1], "<i2"), Prop("a", 1), Prop("b", "x"))
    )
    second = (
        Segment()
        .root(Prop("name", "run 2"))
        .group("g", Prop("late", "yes"))
        .channel("g", "c", arr([2], "<i2"), Prop("a", 2), Prop("c", 3.5), index="same")
    )
    truth = (
        Truth()
        .root(Prop("name", "run 2"), Prop("keep", 1))
        .group("g", Prop("a", 1), Prop("late", "yes"))
        .channel("g", "c", arr([1, 2], "<i2"), Prop("a", 2), Prop("b", "x"), Prop("c", 3.5))
    )
    return Built(tdms_file(first, second), truth)


@case(
    "property_unicode",
    ["prop.unicode", "path.unicode"],
    "Names and values with non-ASCII characters, in properties and in object names.",
)
def _property_unicode() -> Built:
    props = (Prop("größe ü", "Ünïcödé ✓ 日本"), Prop("日本語", 3))
    seg = Segment().group("Gruppé 日本", *props).channel("Gruppé 日本", "Kanal ß", arr([1.0], "<f8"), *props)
    truth = Truth().group("Gruppé 日本", *props).channel("Gruppé 日本", "Kanal ß", arr([1.0], "<f8"), *props)
    return Built(tdms_file(seg), truth)


@case(
    "property_odd_values",
    ["prop.empty_and_odd_names"],
    "An empty string, a name with a trailing space, a name with spaces and a 200 kB string.",
)
def _property_odd() -> Built:
    props = (
        Prop("empty", ""),
        Prop("trailing ", "x"),
        Prop("two  spaces", 1),
        Prop("long", "ab" * 100_000),
        Prop("", "empty name"),
    )
    seg = Segment().group("g", *props).channel("g", "c", arr([1], "<i2"))
    return Built(tdms_file(seg), Truth().group("g", *props).channel("g", "c", arr([1], "<i2")))


@case("many_properties", ["prop.many"], "5000 properties on one group.")
def _many_properties() -> Built:
    props = tuple(Prop(f"p{i}", i) for i in range(5000))
    seg = Segment().group("g", *props).channel("g", "c", arr([1], "<i2"))
    return Built(tdms_file(seg), Truth().group("g", *props).channel("g", "c", arr([1], "<i2")))


@case("property_extended_float", ["prop.unsupported_type_is_error"], "A property of the 80-bit extended float type.")
def _property_extended() -> Built:
    seg = Segment().group("g", Prop("ext", bytes(16), "ext"))
    return Built(tdms_file(seg), error=tdmspy.UnsupportedFeatureError)


# =====================================================================================================
# objects and paths
# =====================================================================================================


@case("group_without_channels", ["path.group_without_channels"], "A group that only has properties.")
def _group_without_channels() -> Built:
    seg = Segment().group("extra_items", Prop("wf_increment", 0.0)).group("g").channel("g", "c", arr([1], "<i2"))
    truth = Truth().group("extra_items", Prop("wf_increment", 0.0)).group("g").channel("g", "c", arr([1], "<i2"))
    return Built(tdms_file(seg), truth)


@case("implicit_group", ["path.implicit_group"], "A channel whose group object is never written.")
def _implicit_group() -> Built:
    return Built(
        tdms_file(Segment().channel("never_declared", "c", arr([1, 2], "<i2"))),
        Truth().channel("never_declared", "c", arr([1, 2], "<i2")),
    )


@case("first_appearance_order", ["path.order"], "Groups B then A, channels z, y, x declared in that order: not sorted.")
def _order() -> Built:
    seg = (
        Segment()
        .channel("B", "z", arr([1], "<i2"))
        .channel("A", "y", arr([2], "<i2"))
        .channel("B", "a", arr([3], "<i2"))
        .channel("A", "x", arr([4], "<i2"))
    )
    truth = (
        Truth()
        .channel("B", "z", arr([1], "<i2"))
        .channel("B", "a", arr([3], "<i2"))
        .channel("A", "y", arr([2], "<i2"))
        .channel("A", "x", arr([4], "<i2"))
    )
    return Built(tdms_file(seg), truth)


@case(
    "quotes_and_slashes",
    ["path.quote_and_slash"],
    "Names with an apostrophe, a slash, spaces and a doubled apostrophe.",
)
def _quotes_and_slashes() -> Built:
    names = [("it's", "a/b"), ("with space", "x''y"), ("/", "/'/"), ("g", "'")]
    seg, truth = Segment(), Truth()
    for i, (group, channel) in enumerate(names):
        values = arr([i], "<i2")
        seg.channel(group, channel, values)
        truth.channel(group, channel, values)
    return Built(tdms_file(seg), truth)


@case(
    "reserved_looking_names",
    ["path.reserved_looking_names"],
    "Groups and channels called info, name, groups, properties, data, read and dtype.",
)
def _reserved_names() -> Built:
    seg, truth = Segment(), Truth()
    for group in ("info", "name", "groups", "properties"):
        for channel in ("data", "read", "dtype", "name"):
            values = arr([len(group) + len(channel)], "<i2")
            seg.channel(group, channel, values)
            truth.channel(group, channel, values)
    return Built(tdms_file(seg), truth)


@case("root_only", ["path.root_only_properties"], "A file whose only object is the root.")
def _root_only() -> Built:
    return Built(
        tdms_file(Segment().root(Prop("name", "only root"), Prop("n", 3))),
        Truth().root(Prop("name", "only root"), Prop("n", 3)),
    )


@case("three_level_path", ["path.invalid_is_error"], "An object path with three names.")
def _three_level_path() -> Built:
    seg = Segment()
    seg.objects.append(ObjectSpec("/'a'/'b'/'c'", index="none"))
    return Built(tdms_file(seg), error=tdmspy.TdmsFormatError)


@case("path_without_slash", ["path.invalid_is_error"], "An object path that does not start with a slash.")
def _path_without_slash() -> Built:
    seg = Segment()
    seg.objects.append(ObjectSpec("group/channel", index="none"))
    return Built(tdms_file(seg), error=tdmspy.TdmsFormatError)


# =====================================================================================================
# scaling
# =====================================================================================================


def _linear(index: int, slope: float, intercept: float, source: int = 0xFFFF_FFFF) -> tuple[Prop, ...]:
    p = f"NI_Scale[{index}]_"
    return (
        Prop(p + "Scale_Type", "Linear"),
        Prop(p + "Linear_Slope", slope),
        Prop(p + "Linear_Y_Intercept", intercept),
        Prop(p + "Linear_Input_Source", source, "u32"),
    )


def _scales(count: int, *rest: Prop) -> tuple[Prop, ...]:
    return (Prop("NI_Number_Of_Scales", count), *rest)


@case("scale_linear", ["scale.linear"], "int16 values with one linear scale (2x + 10).")
def _scale_linear() -> Built:
    raw = arr([-3, 0, 5, 100], "<i2")
    props = _scales(1, *_linear(0, 2.0, 10.0))
    truth = Truth().channel("g", "c", raw, *props, scaled=raw.astype("f8") * 2.0 + 10.0)
    return Built(tdms_file(Segment().channel("g", "c", raw, *props)), truth)


@case("scale_polynomial", ["scale.polynomial"], "A polynomial scale 1 + 0.5x + 0.001x^2.")
def _scale_polynomial() -> Built:
    raw = arr([-400, -1, 0, 3, 480], "<i2")
    props = _scales(
        1,
        Prop("NI_Scale[0]_Scale_Type", "Polynomial"), Prop("NI_Scale[0]_Polynomial_Coefficients_Size", 3),
        Prop("NI_Scale[0]_Polynomial_Coefficients[0]", 1.0), Prop("NI_Scale[0]_Polynomial_Coefficients[1]", 0.5),
        Prop("NI_Scale[0]_Polynomial_Coefficients[2]", 0.001), Prop("NI_Scale[0]_Polynomial_Input_Source", 0xFFFF_FFFF, "u32"),
    )  # fmt: skip
    x = raw.astype("f8")
    truth = Truth().channel("g", "c", raw, *props, scaled=1.0 + 0.5 * x + 0.001 * x * x)
    return Built(tdms_file(Segment().channel("g", "c", raw, *props)), truth)


@case("scale_chain", ["scale.chain"], "A polynomial scale feeding a linear one (-x + 3).")
def _scale_chain() -> Built:
    raw = arr([10, 20, -30], "<i2")
    poly = (
        Prop("NI_Scale[0]_Scale_Type", "Polynomial"), Prop("NI_Scale[0]_Polynomial_Coefficients_Size", 2),
        Prop("NI_Scale[0]_Polynomial_Coefficients[0]", 1.0), Prop("NI_Scale[0]_Polynomial_Coefficients[1]", 0.5),
        Prop("NI_Scale[0]_Polynomial_Input_Source", 0xFFFF_FFFF, "u32"),
    )  # fmt: skip
    props = _scales(2, *poly, *_linear(1, -1.0, 3.0, source=0))
    truth = Truth().channel("g", "c", raw, *props, scaled=-(1.0 + 0.5 * raw.astype("f8")) + 3.0)
    return Built(tdms_file(Segment().channel("g", "c", raw, *props)), truth)


@case(
    "scale_already_scaled",
    ["scale.already_scaled"],
    "Scale properties with NI_Scaling_Status = scaled: the data is final.",
)
def _scale_already_scaled() -> Built:
    raw = arr([1, 2, 3], "<i2")
    props = _scales(1, Prop("NI_Scaling_Status", "scaled"), *_linear(0, 2.0, 10.0))
    return Built(tdms_file(Segment().channel("g", "c", raw, *props)), Truth().channel("g", "c", raw, *props))


@case("scale_group_level", ["scale.group_level"], "Scaling declared on the group applies to its channels.")
def _scale_group_level() -> Built:
    raw = arr([1, 2, 3], "<i2")
    props = _scales(1, *_linear(0, 4.0, -1.0))
    seg = Segment().group("g", *props).channel("g", "c", raw)
    return Built(tdms_file(seg), Truth().group("g", *props).channel("g", "c", raw, scaled=raw.astype("f8") * 4.0 - 1.0))


@case(
    "scale_file_level", ["scale.file_level"], "Scaling declared on the file applies to every channel without its own."
)
def _scale_file_level() -> Built:
    raw = arr([1, 2, 3], "<i2")
    props = _scales(1, *_linear(0, 0.5, 0.0))
    seg = Segment().root(*props).channel("g", "c", raw)
    return Built(tdms_file(seg), Truth().root(*props).channel("g", "c", raw, scaled=raw.astype("f8") * 0.5))


@case(
    "scale_channel_overrides_group",
    ["scale.channel_overrides_group", "scale.none"],
    "A channel's scale wins over its group's; a second channel without one uses the group's.",
)
def _scale_override() -> Built:
    raw = arr([1, 2, 3], "<i2")
    group = _scales(1, *_linear(0, 2.0, 10.0))
    own = _scales(1, *_linear(0, 3.0, 0.0))
    seg = Segment().group("g", *group).channel("g", "own", raw, *own).channel("g", "inherits", raw)
    truth = (
        Truth()
        .group("g", *group)
        .channel("g", "own", raw, *own, scaled=raw.astype("f8") * 3.0)
        .channel("g", "inherits", raw, scaled=raw.astype("f8") * 2.0 + 10.0)
    )
    return Built(tdms_file(seg), truth)


@case("scale_float32_input", ["scale.float32_input"], "Scaling float32 values: double precision result.")
def _scale_float32() -> Built:
    raw = arr([0.5, 1.5, -2.25], "<f4")
    props = _scales(1, *_linear(0, 0.1, 0.0))
    truth = Truth().channel("g", "c", raw, *props, scaled=raw.astype("f8") * 0.1)
    return Built(tdms_file(Segment().channel("g", "c", raw, *props)), truth)


@case(
    "scale_unsupported_type",
    ["scale.unsupported_is_error"],
    "A scale of type RTD: the raw read works, the scaled read refuses.",
)
def _scale_unsupported() -> Built:
    raw = arr([100, 110], "<i2")
    props = _scales(1, Prop("NI_Scale[0]_Scale_Type", "RTD"), Prop("NI_Scale[0]_RTD_R0_Nominal_Resistance", 100.0))
    return Built(
        tdms_file(Segment().channel("g", "c", raw, *props)), Truth().channel("g", "c", raw, *props, scaled_error=True)
    )


@case("scale_none", ["scale.none"], "A channel without scale properties.")
def _scale_none() -> Built:
    raw = arr([1, 2, 3], "<i2")
    return Built(tdms_file(Segment().channel("g", "c", raw)), Truth().channel("g", "c", raw))


# =====================================================================================================
# unsupported layouts and robustness
# =====================================================================================================


@case("daqmx_raw_data", ["daqmx.is_error"], "A channel with a DAQmx raw data index (0x1269).")
def _daqmx() -> Built:
    index = struct.pack("<I", 0x1269) + bytes(60)
    seg = Segment(trailing_raw=bytes(64)).channel("g", "c", None, index_bytes=index)
    return Built(tdms_file(seg), error=tdmspy.UnsupportedFeatureError)


def _metadata(*objects: bytes, count: int | None = None) -> bytes:
    return struct.pack("<I", len(objects) if count is None else count) + b"".join(objects)


def _object(path: str, index: bytes = struct.pack("<I", 0xFFFF_FFFF), props: bytes = struct.pack("<I", 0)) -> bytes:
    return pack_string(path, "<") + index + props


@case("huge_object_count", ["robust.huge_counts"], "Metadata announcing 4 billion objects.")
def _huge_object_count() -> Built:
    return Built(raw_segment(_metadata(_object("/'g'"), count=0xFFFF_FFFF)), error=tdmspy.TdmsFormatError)


@case("huge_property_count", ["robust.huge_counts"], "An object announcing 4 billion properties.")
def _huge_property_count() -> Built:
    return Built(
        raw_segment(_metadata(_object("/'g'", props=struct.pack("<I", 0xFFFF_FFFF)))), error=tdmspy.TdmsFormatError
    )


@case("huge_string_length", ["robust.huge_counts"], "A property string announcing 4 GB.")
def _huge_string() -> Built:
    prop = pack_string("p", "<") + struct.pack("<II", 0x20, 0xFFFF_FFF0)
    return Built(
        raw_segment(_metadata(_object("/'g'", props=struct.pack("<I", 1) + prop))), error=tdmspy.TdmsFormatError
    )


@case("huge_path_length", ["robust.huge_counts"], "An object path announcing 4 GB.")
def _huge_path() -> Built:
    return Built(raw_segment(struct.pack("<II", 1, 0xFFFF_FFF0) + b"/'g'"), error=tdmspy.TdmsFormatError)


@case(
    "huge_value_count",
    ["robust.huge_counts"],
    "A channel announcing 2**40 values in a segment that is followed by another.",
)
def _huge_value_count() -> Built:
    index = struct.pack("<IIIQ", 20, 2, 1, 2**40)
    first = raw_segment(_metadata(_object("/'g'/'c'", index)), bytes(8))
    second = tdms_file(Segment().channel("g", "c", arr([1], "<i2"), index="same"))
    return Built(first + second, error=tdmspy.TdmsFormatError)


@case(
    "many_segments_5000",
    ["robust.many_segments", "segment.many", "index.same_as_previous"],
    "5000 segments of 10 values: open time must not grow faster than the file.",
    stored=False,
)
def _many_segments_5000() -> Built:
    segments, parts = [], []
    for i in range(5000):
        values = arr(np.arange(10) + 10 * i, "<i4")
        parts.append(values)
        segments.append(Segment().channel("g", "c", values, index="explicit" if i == 0 else "same"))
    return Built(tdms_file(*segments), Truth().channel("g", "c", np.concatenate(parts)))


# =====================================================================================================
# real files
# =====================================================================================================

REAL_FILES: dict[str, list[str]] = {
    "one_int16_channel.tdms": [
        "version.4713", "segment.many", "index.same_as_previous", "type.integers", "type.floats", "prop.all_types",
        "prop.root_group_channel", "path.group_without_channels", "layout.contiguous_multi_channel",
    ],
    "three_int16_channels.tdms": [
        "version.4713", "segment.many", "index.same_as_previous", "type.integers", "type.floats", "prop.all_types",
        "prop.root_group_channel", "path.group_without_channels", "layout.contiguous_multi_channel",
    ],
    "three_int16_channels_short.tdms": [
        "version.4713", "segment.many", "index.same_as_previous", "type.integers", "type.floats", "prop.all_types",
        "path.group_without_channels", "layout.contiguous_multi_channel",
    ],
    "unequal_counts_untyped_channels.tdms": [
        "version.4713", "segment.many", "segment.meta_without_new_list", "index.same_as_previous", "type.floats", "type.no_data_type",
        "path.group_without_channels", "path.reserved_looking_names", "layout.unequal_counts", "prop.root_group_channel",
        "prop.empty_and_odd_names",
    ],
}  # fmt: skip
"""The real files in `tests/data/real` and the catalogue ids each one exercises (checked by `test_real_files.py`)."""
