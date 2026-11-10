"""The objects `tdmspy.read` returns: `TdmsFile`, `Group` and `Channel` (see doc/design.md)."""

from __future__ import annotations

import mmap
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

from tdmspy._format import DATA_TYPES, TYPE_BOOL, TYPE_STRING, TYPE_TIMESTAMP
from tdmspy._scaling import apply_scaling
from tdmspy._scan import ChannelIndex, GroupIndex, Scan, scan_file
from tdmspy.errors import TdmsFormatError
from tdmspy.timestamp import UNIX_EPOCH_IN_TDMS_SECONDS

_LE_STAMP = np.dtype([("fraction", "<u8"), ("seconds", "<i8")])
_BE_STAMP = np.dtype([("seconds", ">i8"), ("fraction", ">u8")])


def _microseconds(seconds: np.ndarray, fraction: np.ndarray) -> np.ndarray:
    """`datetime64[us]` from TDMS seconds and 2**-64 fractions: the fraction is truncated, exactly (no floats).

    floor(fraction * 10**6 / 2**64) with the fraction split in two 32 bit halves, so nothing overflows 64 bits.
    """
    thirty_two = np.uint64(32)
    high, low = fraction >> thirty_two, fraction & np.uint64(0xFFFF_FFFF)
    a, b = high * np.uint64(1_000_000), low * np.uint64(1_000_000)
    micro = ((a + (b >> thirty_two)) >> thirty_two).astype(np.int64)
    ticks = (seconds.astype(np.int64) - UNIX_EPOCH_IN_TDMS_SECONDS) * 1_000_000 + micro
    return ticks.astype("datetime64[us]")


def _decode(raw: bytes) -> str:
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        raise TdmsFormatError("a string value is not valid UTF-8") from None


class Channel:
    """A channel: its properties, data type, length, and `read` for its values."""

    def __init__(self, index: ChannelIndex, group: Group, file: TdmsFile) -> None:
        self.name = index.name
        self.properties: dict[str, Any] = index.properties
        self._group = group
        self._file = file
        self._type_code = index.type_code
        self._length = index.length
        self._strings = index.string_chunks
        self._blocks = np.array(index.blocks, dtype=np.int64).reshape(-1, 5)
        self._starts = np.concatenate(([0], np.cumsum(self._blocks[:, 1] * self._blocks[:, 2])))
        self._dtype = None if index.type_code is None else self._native_dtype(index.type_code)

    @staticmethod
    def _native_dtype(type_code: int) -> np.dtype:
        if type_code == TYPE_STRING:
            return np.dtype(object)
        if type_code == TYPE_BOOL:
            return np.dtype(bool)
        if type_code == TYPE_TIMESTAMP:
            return np.dtype("datetime64[us]")
        return np.dtype("=" + DATA_TYPES[type_code].numpy)

    @property
    def dtype(self) -> np.dtype | None:
        """The numpy dtype of the values (native byte order); `None` for a channel that never has a data type."""
        return self._dtype

    def __len__(self) -> int:
        return self._length

    def __repr__(self) -> str:
        return f"<Channel {self.name!r}: {self._dtype}, {self._length} values>"

    def read(self, start: int = 0, stop: int | None = None, *, scaled: bool = True) -> np.ndarray:
        """The values `start` up to `stop` (all by default) as a new array; NI scaling is applied unless `scaled=False`."""
        if start < 0 or (stop is not None and stop < 0):
            raise ValueError("start and stop must not be negative")
        stop = self._length if stop is None else min(stop, self._length)
        start = min(start, self._length)
        stop = max(stop, start)
        if self._dtype is None:
            return np.empty(0)
        raw = self._read_strings(start, stop) if self._type_code == TYPE_STRING else self._read_numbers(start, stop)
        if scaled and raw.dtype.kind in "iuf":
            return apply_scaling(raw, self.properties, self._group.properties, self._file.properties)
        return raw

    # -- numbers ---------------------------------------------------------------------------------

    def _read_numbers(self, start: int, stop: int) -> np.ndarray:
        assert self._dtype is not None
        out = np.empty(stop - start, dtype=self._dtype)
        if stop == start:
            return out
        first = int(np.searchsorted(self._starts, start, side="right")) - 1
        last = int(np.searchsorted(self._starts, stop, side="left"))
        with open(self._file.path, "rb") as handle, mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as data:
            for i in range(first, last):
                base = int(self._starts[i])
                lo, hi = max(start, base), min(stop, int(self._starts[i + 1]))
                if lo < hi:
                    offset, per, repeat, stride, big = (int(v) for v in self._blocks[i])
                    self._read_block(
                        data,
                        (offset, per, repeat, stride, bool(big)),
                        lo - base,
                        hi - base,
                        out[lo - start : hi - start],
                    )
        return out

    def _read_block(
        self, data: mmap.mmap, block: tuple[int, int, int, int, bool], v0: int, v1: int, out: np.ndarray
    ) -> None:
        """Values `v0` to `v1` of one block into `out`."""
        assert self._type_code is not None
        offset, per, repeat, stride, big = block
        info = DATA_TYPES[self._type_code]
        if self._type_code == TYPE_TIMESTAMP:
            dtype, size = (_BE_STAMP if big else _LE_STAMP), 16
        else:
            dtype, size = np.dtype(("<>"[big]) + info.numpy), info.size
        if offset + (repeat - 1) * stride + per * size > len(data):
            raise TdmsFormatError("the file is shorter than when it was opened")

        def copy(unit: int, count: int, within: int, into: np.ndarray, units: int = 1) -> None:
            view = np.ndarray(
                (units, count),
                dtype=dtype,
                buffer=data,
                offset=offset + unit * stride + within * size,
                strides=(stride, size),
            )
            self._convert(view, into.reshape(units, count))

        wanted = v1 - v0
        position = 0
        unit, within = divmod(v0, per)
        if within:  # the tail of a first unit that is only partly wanted
            count = min(per - within, wanted)
            copy(unit, count, within, out[:count])
            position, unit = count, unit + 1
        full = (v1 // per) - unit
        if full > 0 and position < wanted:
            copy(unit, per, 0, out[position : position + full * per], units=full)
            position, unit = position + full * per, unit + full
        if position < wanted:  # the head of a last unit that is only partly wanted
            copy(unit, wanted - position, 0, out[position:])

    def _convert(self, view: np.ndarray, into: np.ndarray) -> None:
        if self._type_code == TYPE_TIMESTAMP:
            into[...] = _microseconds(view["seconds"], view["fraction"].astype(np.uint64))
        elif self._type_code == TYPE_BOOL:
            into[...] = view != 0
        else:
            into[...] = view

    # -- strings ---------------------------------------------------------------------------------

    def _read_strings(self, start: int, stop: int) -> np.ndarray:
        out = np.empty(stop - start, dtype=object)
        position = 0
        with open(self._file.path, "rb") as handle:
            for offset, count, total, big in self._strings:
                lo, hi = max(start, position), min(stop, position + count)
                if lo < hi:
                    handle.seek(offset)
                    chunk = handle.read(total)
                    if len(chunk) < total or total < 4 * count:
                        raise TdmsFormatError("a string channel's data is shorter than its index says")
                    ends = np.frombuffer(chunk, dtype=("<>"[big]) + "u4", count=count)
                    text = chunk[4 * count :]
                    for i in range(lo - position, hi - position):
                        begin, end = (int(ends[i - 1]) if i else 0), int(ends[i])
                        if not begin <= end <= len(text):
                            raise TdmsFormatError("the offsets of a string channel are inconsistent")
                        out[lo - start + i - (lo - position)] = _decode(text[begin:end])
                position += count
                if position >= stop:
                    break
        return out


class Group:
    """A group: its properties and its channels, in file order."""

    def __init__(self, index: GroupIndex, file: TdmsFile) -> None:
        self.name = index.name
        self.properties: dict[str, Any] = index.properties
        self.channels: list[Channel] = [Channel(c, self, file) for c in index.channels.values()]
        self._by_name = {c.name: c for c in self.channels}

    def __getitem__(self, name: str) -> Channel:
        return self._by_name[name]

    def __contains__(self, name: object) -> bool:
        return name in self._by_name

    def __iter__(self) -> Iterator[Channel]:
        return iter(self.channels)

    def __repr__(self) -> str:
        return f"<Group {self.name!r}: {len(self.channels)} channels>"


class TdmsFile:
    """A TDMS file: its properties and groups. Channel data is read when asked for."""

    def __init__(self, path: str | os.PathLike[str], scan: Scan) -> None:
        self.path = Path(path)
        self.properties: dict[str, Any] = scan.properties
        self.truncated = scan.truncated
        self.groups: list[Group] = [Group(g, self) for g in scan.groups.values()]
        self._by_name = {g.name: g for g in self.groups}

    def __getitem__(self, name: str) -> Group:
        return self._by_name[name]

    def __contains__(self, name: object) -> bool:
        return name in self._by_name

    def __iter__(self) -> Iterator[Group]:
        return iter(self.groups)

    def __repr__(self) -> str:
        return f"<TdmsFile {self.path.name!r}: {len(self.groups)} groups>"


def read(path: str | os.PathLike[str], *, strict: bool = False) -> TdmsFile:
    """Open the TDMS file at `path`: its structure is read, the channel data is not (see doc/design.md)."""
    return TdmsFile(path, scan_file(os.fspath(path), strict=strict))
