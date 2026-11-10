"""The test writer itself: its bytes are what the format says (checked by hand here)."""

import struct

import numpy as np

from tdmspy.timestamp import Timestamp
from tests.builder import BIG, UNKNOWN_NEXT_OFFSET, Prop, Segment, object_path, pack_prop, pack_string, tdms_file


def lead_in(data: bytes, at: int = 0) -> tuple[bytes, int, int, int, int]:
    tag, toc = struct.unpack_from("<4sI", data, at)
    endian = ">" if toc & 64 else "<"
    version, next_offset, raw_offset = struct.unpack_from(endian + "IQQ", data, at + 8)
    return tag, toc, version, next_offset, raw_offset


def test_a_segment_has_the_lead_in_the_format_describes():
    data = Segment().channel("g", "c", np.array([1, 2, 3], dtype="<i2")).build()
    tag, toc, version, next_offset, raw_offset = lead_in(data)
    assert tag == b"TDSm" and version == 4713
    assert toc == 2 | 4 | 8  # metadata, new object list, raw data
    assert next_offset == len(data) - 28 and raw_offset == len(data) - 28 - 6


def test_flags_follow_the_segment():
    assert lead_in(Segment(meta=False).channel("g", "c", np.array([1], "<i2"), index="same").build())[1] == 8
    assert lead_in(Segment().group("g").build())[1] == 2 | 4  # no raw data
    assert lead_in(Segment(newlist=False).channel("g", "c", np.array([1], "<i2")).build())[1] == 2 | 8
    assert lead_in(Segment(interleaved=True).channel("g", "c", np.array([1], "<i2")).build())[1] == 2 | 4 | 8 | 32
    assert lead_in(Segment(endian=BIG).channel("g", "c", np.array([1], ">i2")).build())[1] == 2 | 4 | 8 | 64


def test_explicit_index_and_data_of_one_channel():
    data = Segment().channel("g", "c", np.array([1, 2, 3], dtype="<i2")).build()
    metadata = data[28:]
    objects = struct.unpack_from("<I", metadata)[0]
    path = pack_string(object_path("g", "c"), "<")
    index = struct.pack("<IIIQ", 20, 2, 1, 3)  # length 20, type int16, dimension 1, 3 values
    assert objects == 1
    assert metadata[4 : 4 + len(path) + len(index)] == path + index
    assert data[-6:] == np.array([1, 2, 3], dtype="<i2").tobytes()


def test_object_paths_double_their_quotes():
    assert object_path() == "/"
    assert object_path("it's") == "/'it''s'"
    assert object_path("g", "a/b") == "/'g'/'a/b'"


def test_interleaving_alternates_the_channels_row_by_row():
    seg = (
        Segment(interleaved=True)
        .channel("g", "a", np.array([1, 2], "<i2"))
        .channel("g", "b", np.array([10, 20], "<i2"))
    )
    assert seg.build()[-8:] == np.array([1, 10, 2, 20], dtype="<i2").tobytes()


def test_chunks_repeat_the_channel_set():
    seg = Segment().channel("g", "a", np.array([1], "<i2")).channel("g", "b", np.array([9], "<i2"))
    seg.chunk_of({("g", "a"): np.array([2], "<i2"), ("g", "b"): np.array([8], "<i2")})
    assert seg.build()[-8:] == np.array([1, 9, 2, 8], dtype="<i2").tobytes()


def test_big_endian_swaps_every_multi_byte_field():
    data = Segment(endian=BIG).channel("g", "c", np.array([258], ">i2"), Prop("p", 1)).build()
    assert data[8:12] == struct.pack(">I", 4713)
    assert data[-2:] == b"\x01\x02"
    assert pack_prop(Prop("n", 1), ">") == pack_string("n", ">") + struct.pack(">II", 3, 1)


def test_unknown_next_offset_and_padding():
    data = Segment(next_offset=UNKNOWN_NEXT_OFFSET, padding=3).channel("g", "c", np.array([1], "<i2")).build()
    _, _, _, next_offset, raw_offset = lead_in(data)
    assert next_offset == UNKNOWN_NEXT_OFFSET
    assert raw_offset == len(data) - 28 - 2 and data[-5:-2] == b"\x00\x00\x00"


def test_strings_are_offsets_then_text_with_the_total_in_the_index():
    data = Segment().channel("g", "s", ["ab", "", "cde"]).build()
    raw = np.array([2, 2, 5], "<u4").tobytes() + b"abcde"
    assert data[-len(raw) :] == raw
    assert struct.pack("<IIIQQ", 28, 0x20, 1, 3, len(raw)) in data


def test_timestamps_are_fraction_then_seconds():
    assert pack_prop(Prop("t", Timestamp(5, 7)), "<") == pack_string("t", "<") + struct.pack("<IQq", 0x44, 7, 5)


def test_a_file_is_its_segments_one_after_the_other():
    first, second = Segment().group("a"), Segment().group("b")
    assert tdms_file(first, second) == first.build() + second.build()
