"""Parsing the metadata section of a segment into objects, with every count and length checked against the bytes
that are there before anything is read or allocated."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Any

from tdmspy._format import (
    DATA_TYPES,
    INDEX_DAQMX,
    INDEX_NONE,
    INDEX_SAME,
    PROPERTY_STRUCT,
    TYPE_COMPLEX_DOUBLE,
    TYPE_COMPLEX_SINGLE,
    TYPE_EXTENDED,
    TYPE_STRING,
    TYPE_TIMESTAMP,
)
from tdmspy.errors import TdmsFormatError, UnsupportedFeatureError
from tdmspy.timestamp import Timestamp

_MIN_OBJECT = 12  # path length, raw data index, property count
_MIN_PROPERTY = 8  # name length, type

NONE = "none"
SAME = "same"


@dataclass(frozen=True)
class Index:
    """What an object says about its raw data in a segment."""

    kind: str  # NONE, SAME or "explicit"
    type_code: int = 0
    count: int = 0
    total_size: int = 0  # strings: bytes of the offsets and the text of one chunk


@dataclass
class ObjectMeta:
    path: tuple[str, ...]
    index: Index
    properties: list[tuple[str, Any]]


def parse_path(path: str) -> tuple[str, ...]:
    """`/` -> (), `/'g'` -> ('g',), `/'g'/'c'` -> ('g', 'c'); `''` inside a name is one apostrophe."""
    bad = TdmsFormatError(f"invalid object path {path!r}")
    if not path.startswith("/"):
        raise bad
    names: list[str] = []
    i, n = 1, len(path)
    while i < n:
        if path[i] != "'":
            raise bad
        i += 1
        name = []
        while True:
            if i >= n:
                raise bad
            if path[i] == "'":
                if i + 1 < n and path[i + 1] == "'":
                    name.append("'")
                    i += 2
                    continue
                i += 1
                break
            name.append(path[i])
            i += 1
        names.append("".join(name))
        if i < n:
            if path[i] != "/" or i + 1 >= n:
                raise bad
            i += 1
    if len(names) > 2:
        raise bad
    return tuple(names)


class _Reader:
    def __init__(self, buffer: bytes | memoryview, big_endian: bool) -> None:
        self.buffer = buffer
        self.pos = 0
        self.end = len(buffer)
        self.e = ">" if big_endian else "<"

    def remaining(self) -> int:
        return self.end - self.pos

    def need(self, count: int, what: str) -> None:
        if count > self.end - self.pos:
            raise TdmsFormatError(f"metadata ends inside {what}")

    def u32(self, what: str) -> int:
        self.need(4, what)
        (value,) = struct.unpack_from(self.e + "I", self.buffer, self.pos)
        self.pos += 4
        return value

    def u64(self, what: str) -> int:
        self.need(8, what)
        (value,) = struct.unpack_from(self.e + "Q", self.buffer, self.pos)
        self.pos += 8
        return value

    def string(self, what: str) -> str:
        length = self.u32(what)
        self.need(length, what)
        raw = bytes(self.buffer[self.pos : self.pos + length])
        self.pos += length
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError:
            raise TdmsFormatError(f"{what} is not valid UTF-8") from None


def _property_value(reader: _Reader, type_code: int, name: str) -> Any:
    if type_code in PROPERTY_STRUCT:
        code, size = PROPERTY_STRUCT[type_code]
        reader.need(size, f"property {name!r}")
        (value,) = struct.unpack_from(reader.e + code, reader.buffer, reader.pos)
        reader.pos += size
        return bool(value) if type_code == 0x21 else value
    if type_code == TYPE_STRING:
        return reader.string(f"property {name!r}")
    if type_code == TYPE_TIMESTAMP:
        reader.need(16, f"property {name!r}")
        if reader.e == ">":
            seconds, fraction = struct.unpack_from(">qQ", reader.buffer, reader.pos)
        else:
            fraction, seconds = struct.unpack_from("<Qq", reader.buffer, reader.pos)
        reader.pos += 16
        return Timestamp(seconds, fraction)
    if type_code in (TYPE_COMPLEX_SINGLE, TYPE_COMPLEX_DOUBLE):
        code, size = ("f", 8) if type_code == TYPE_COMPLEX_SINGLE else ("d", 16)
        reader.need(size, f"property {name!r}")
        real, imag = struct.unpack_from(reader.e + code * 2, reader.buffer, reader.pos)
        reader.pos += size
        return complex(real, imag)
    if type_code == 0:
        return None
    if type_code in (TYPE_EXTENDED, 0x1B):
        raise UnsupportedFeatureError(f"property {name!r} has the extended float type, which is not supported")
    raise TdmsFormatError(f"property {name!r} has the unknown type {type_code:#x}")


def _index(reader: _Reader, path: str) -> Index:
    value = reader.u32(f"the raw data index of {path!r}")
    if value == INDEX_NONE:
        return Index(NONE)
    if value == INDEX_SAME:
        return Index(SAME)
    if value in INDEX_DAQMX:
        raise UnsupportedFeatureError(f"{path!r} holds DAQmx raw data, which is not supported")
    if value < 20 or value > reader.remaining() + 4:
        raise TdmsFormatError(f"the raw data index of {path!r} has the length {value}, which does not fit the metadata")
    following = value - 4
    start = reader.pos
    type_code = reader.u32("a raw data index")
    dimension = reader.u32("a raw data index")
    count = reader.u64("a raw data index")
    total = 0
    if type_code == TYPE_STRING:
        total = reader.u64("a raw data index")
    if reader.pos - start > following:
        raise TdmsFormatError(f"the raw data index of {path!r} is longer than its length says")
    reader.pos = start + following
    if type_code == TYPE_STRING and total < 4 * count:
        raise TdmsFormatError(
            f"{path!r}: {count} strings need {4 * count} bytes of offsets, the index says {total} bytes in all"
        )
    if dimension != 1:
        raise TdmsFormatError(f"{path!r} has an array dimension of {dimension}; only 1 is valid")
    if type_code not in DATA_TYPES:
        if type_code == TYPE_EXTENDED or type_code == 0x1B:
            raise UnsupportedFeatureError(f"{path!r} has the extended float data type, which is not supported")
        raise TdmsFormatError(f"{path!r} has the unknown data type {type_code:#x}")
    return Index("explicit", type_code, count, total)


def parse_metadata(buffer: bytes | memoryview, big_endian: bool) -> list[ObjectMeta]:
    reader = _Reader(buffer, big_endian)
    count = reader.u32("the object count")
    if count * _MIN_OBJECT > reader.remaining():
        raise TdmsFormatError(f"the metadata announces {count} objects but is only {reader.remaining()} bytes long")
    objects: list[ObjectMeta] = []
    for _ in range(count):
        text = reader.string("an object path")
        path = parse_path(text)
        index = _index(reader, text)
        if len(path) < 2 and index.kind == "explicit":
            raise TdmsFormatError(f"{text!r} is not a channel but has raw data")
        n_props = reader.u32(f"the properties of {text!r}")
        if n_props * _MIN_PROPERTY > reader.remaining():
            raise TdmsFormatError(f"{text!r} announces {n_props} properties but the metadata ends sooner")
        properties = []
        for _ in range(n_props):
            name = reader.string("a property name")
            type_code = reader.u32(f"property {name!r}")
            properties.append((name, _property_value(reader, type_code, name)))
        objects.append(ObjectMeta(path, index, properties))
    return objects
