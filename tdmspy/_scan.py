"""Scanning a file: every segment's lead-in and metadata become, per channel, a list of data blocks.

The result of a scan holds no channel data. A *block* says where a run of a channel's values lies in the file:
`[offset, per, repeat, stride, big_endian]` is `repeat` units of `per` consecutive values, the units `stride`
bytes apart, the first starting at `offset`. A segment of one chunk is one unit; several chunks, or interleaved
data, are many units; the identical segments of an acquisition are merged into one block.
"""

from __future__ import annotations

import mmap
import struct
from dataclasses import dataclass, field
from typing import Any

from tdmspy._format import (
    DATA_TYPES,
    INDEX_SIGNATURE,
    LEAD_IN_SIZE,
    SIGNATURE,
    TOC_BIG_ENDIAN,
    TOC_DAQMX,
    TOC_INTERLEAVED,
    TOC_METADATA,
    TOC_NEW_OBJECT_LIST,
    TOC_RAW_DATA,
    TYPE_STRING,
    UNKNOWN_NEXT_OFFSET,
    VERSIONS,
)
from tdmspy._metadata import NONE, SAME, Index, ObjectMeta, parse_metadata
from tdmspy.errors import TdmsFormatError, UnsupportedFeatureError


@dataclass(eq=False)
class ChannelIndex:
    """Everything the scan learned about one channel."""

    name: str
    properties: dict[str, Any] = field(default_factory=dict)
    type_code: int | None = None
    last: Index | None = None  # the last explicit raw data index
    current: Index | None = None  # the index in force now (None: no data in the current segments)
    blocks: list[list[int]] = field(default_factory=list)
    string_chunks: list[tuple[int, int, int, bool]] = field(default_factory=list)  # offset, count, bytes, big endian
    length: int = 0


@dataclass(eq=False)
class GroupIndex:
    name: str
    properties: dict[str, Any] = field(default_factory=dict)
    channels: dict[str, ChannelIndex] = field(default_factory=dict)


@dataclass
class Scan:
    properties: dict[str, Any] = field(default_factory=dict)
    groups: dict[str, GroupIndex] = field(default_factory=dict)
    truncated: bool = False


def _item_size(channel: ChannelIndex) -> int:
    assert channel.type_code is not None
    return DATA_TYPES[channel.type_code].size


class _State:
    def __init__(self) -> None:
        self.scan = Scan()
        self.active: list[ChannelIndex] = []
        self.layout: tuple[list[ChannelIndex], list[int], int] | None = None  # members, sizes, chunk size

    def group(self, name: str) -> GroupIndex:
        group = self.scan.groups.get(name)
        if group is None:
            group = self.scan.groups[name] = GroupIndex(name)
        return group

    def apply(self, objects: list[ObjectMeta], new_list: bool) -> None:
        listed: list[ChannelIndex] = []
        for obj in objects:
            if len(obj.path) == 0:
                self.scan.properties.update(obj.properties)
            elif len(obj.path) == 1:
                self.group(obj.path[0]).properties.update(obj.properties)
            else:
                group = self.group(obj.path[0])
                channel = group.channels.get(obj.path[1])
                if channel is None:
                    channel = group.channels[obj.path[1]] = ChannelIndex(obj.path[1])
                channel.properties.update(obj.properties)
                self._index(channel, obj.index, obj.path)
                if channel not in listed:
                    listed.append(channel)
        if new_list:
            self.active = listed
        else:
            known = {id(c) for c in self.active}
            self.active += [c for c in listed if id(c) not in known]
        self.layout = None

    @staticmethod
    def _index(channel: ChannelIndex, index: Index, path: tuple[str, ...]) -> None:
        name = "/".join(path)
        if index.kind == NONE:
            channel.current = None
        elif index.kind == SAME:
            if channel.last is None:
                raise TdmsFormatError(f"{name!r} has the raw data index 'same as before' but never had one")
            channel.current = channel.last
        else:
            if channel.type_code is not None and channel.type_code != index.type_code:
                raise TdmsFormatError(
                    f"the data type of {name!r} changes from {channel.type_code:#x} to {index.type_code:#x}"
                )
            channel.type_code = index.type_code
            channel.last = channel.current = index

    def members(self) -> tuple[list[ChannelIndex], list[int], int]:
        if self.layout is None:
            members = [c for c in self.active if c.current is not None]
            sizes = []
            for channel in members:
                index = channel.current
                assert index is not None
                sizes.append(index.total_size if index.type_code == TYPE_STRING else index.count * _item_size(channel))
            self.layout = (members, sizes, sum(sizes))
        return self.layout


def _add_block(channel: ChannelIndex, offset: int, per: int, repeat: int, stride: int, big: bool) -> None:
    """Append a block, merging it into the previous one when it continues it at a regular distance."""
    blocks = channel.blocks
    if blocks:
        last = blocks[-1]
        if repeat == 1 and last[1] == per and last[4] == int(big):
            if last[2] == 1:
                last[2], last[3] = 2, offset - last[0]
                channel.length += per
                return
            if offset - (last[0] + (last[2] - 1) * last[3]) == last[3]:
                last[2] += 1
                channel.length += per
                return
    blocks.append([offset, per, repeat, stride, int(big)])
    channel.length += per * repeat


def _segment_layout(
    state: _State, raw_start: int, raw_len: int, interleaved: bool, big: bool, partial_allowed: bool
) -> bool:
    """Add the blocks of one segment's raw data; returns whether the data was cut inside a chunk."""
    members, sizes, chunk = state.members()
    if chunk == 0:
        if raw_len > 0:
            raise TdmsFormatError("a segment has raw data but no channel accounts for it")
        return False
    whole, remainder = divmod(raw_len, chunk)
    if remainder and not partial_allowed or whole == 0 and not partial_allowed:
        raise TdmsFormatError(
            f"the raw data of a segment is {raw_len} bytes, which is not a whole number of chunks of {chunk} bytes"
        )
    if interleaved:
        counts = {c.current.count for c in members if c.current is not None}
        if len(counts) > 1 or any(c.type_code == TYPE_STRING for c in members):
            raise TdmsFormatError("interleaved channels must all have the same count and cannot be strings")
        count = counts.pop() if counts else 0
        row = sum(_item_size(c) for c in members)
        rows = whole * count + remainder // row
        offset = raw_start
        for channel in members:
            if rows:
                _add_block(channel, offset, 1, rows, row, big)
            offset += _item_size(channel)
        return remainder > 0
    offset = 0
    left = remainder
    for channel, size in zip(members, sizes, strict=True):
        index = channel.current
        assert index is not None
        if channel.type_code == TYPE_STRING:
            for c in range(whole):
                channel.string_chunks.append((raw_start + c * chunk + offset, index.count, size, big))
                channel.length += index.count
        else:
            if whole and index.count:
                _add_block(channel, raw_start + offset, index.count, whole, chunk, big)
            if left > 0:
                complete = min(left, size) // _item_size(channel)
                if complete:
                    _add_block(channel, raw_start + whole * chunk + offset, complete, 1, 0, big)
        left -= size
        offset += size
    return remainder > 0


def scan_file(path: str, *, strict: bool) -> Scan:
    with open(path, "rb") as handle:
        handle.seek(0, 2)
        size = handle.tell()
        if size < LEAD_IN_SIZE:
            raise TdmsFormatError(f"{path}: {size} bytes are too few for a TDMS file (a segment starts with 28 bytes)")
        with mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as data:
            return _scan(data, size, strict)


def _cut(strict: bool, message: str) -> None:
    if strict:
        raise TdmsFormatError(message)


def _scan(data: mmap.mmap, size: int, strict: bool) -> Scan:
    state = _State()
    scan = state.scan
    pos = 0
    segments = 0
    cache_key: tuple | None = None
    while pos < size:
        if size - pos < LEAD_IN_SIZE:
            _cut(strict, f"the file ends inside the lead-in of a segment (byte {pos})")
            scan.truncated = True
            break
        tag = data[pos : pos + 4]
        if tag != SIGNATURE:
            if tag == INDEX_SIGNATURE:
                raise TdmsFormatError("this is a TDMS index file (.tdms_index): it holds no data, open the .tdms file")
            raise TdmsFormatError(f"no TDMS segment at byte {pos} (found {bytes(tag)!r})")
        (toc,) = struct.unpack_from("<I", data, pos + 4)
        big = bool(toc & TOC_BIG_ENDIAN)
        version, next_offset, raw_offset = struct.unpack_from((">" if big else "<") + "IQQ", data, pos + 8)
        if version not in VERSIONS:
            raise TdmsFormatError(f"unknown TDMS version {version} in the segment at byte {pos} (known: {VERSIONS})")
        unknown_end = next_offset == UNKNOWN_NEXT_OFFSET
        end = size if unknown_end else pos + LEAD_IN_SIZE + next_offset
        cut = end > size
        if cut:
            end = size
        raw_start = pos + LEAD_IN_SIZE + raw_offset
        if raw_start > end:
            if cut:
                _cut(strict, f"the file ends inside the metadata of the segment at byte {pos}")
                scan.truncated = True
                break
            raise TdmsFormatError(f"the raw data offset of the segment at byte {pos} points beyond the segment")

        if toc & TOC_METADATA:
            metadata = bytes(data[pos + LEAD_IN_SIZE : raw_start])
            key = (metadata, toc & (TOC_NEW_OBJECT_LIST | TOC_BIG_ENDIAN))
            if key != cache_key:  # identical metadata twice in a row changes nothing the second time
                objects = parse_metadata(metadata, big)
                state.apply(objects, bool(toc & TOC_NEW_OBJECT_LIST))
                cache_key = key
        raw_len = end - raw_start
        if toc & TOC_DAQMX and raw_len:
            raise UnsupportedFeatureError("DAQmx raw data is not supported")
        partial = False
        if raw_len or toc & TOC_RAW_DATA:
            partial = _segment_layout(
                state, raw_start, raw_len, bool(toc & TOC_INTERLEAVED), big, partial_allowed=cut or unknown_end
            )
        if cut or partial:
            _cut(strict, f"the file ends inside the segment at byte {pos}")
            scan.truncated = True
        segments += 1
        pos = end
    if segments == 0:
        raise TdmsFormatError("the file has no complete segment")
    return scan
