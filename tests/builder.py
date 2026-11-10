"""A byte-level TDMS writer for the tests.

It exists to produce files that no library writes for us: interleaved and big-endian segments, several
chunks per segment, segments without metadata, truncated and corrupt files, every property type. It writes
exactly what it is told (including things the format forbids, for the error cases), so it does no planning
of its own. It does not use tdmspy, so a bug in the reader cannot hide in here; the files it wrote are stored in
`tests/data/synthetic` and a test checks they are still what it writes.

    seg = Segment()
    seg.root(Prop("name", "test"))
    seg.channel("group", "channel", np.arange(5, dtype="<i2"), Prop("unit_string", "g"))
    data = tdms_file(seg)                      # bytes of a complete file

The format (https://www.ni.com/en/support/documentation/supplemental/07/tdms-file-format-internal-structure.html):
a file is a list of segments; a segment is a 28 byte lead-in ("TDSm", table of contents, version, next
segment offset, raw data offset), a metadata section (objects with their raw data index and properties) and
a raw data section.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from tdmspy.timestamp import Timestamp

LITTLE, BIG = "<", ">"
UNKNOWN_NEXT_OFFSET = 0xFFFF_FFFF_FFFF_FFFF

TYPE_CODES = {
    "void": 0, "i8": 1, "i16": 2, "i32": 3, "i64": 4, "u8": 5, "u16": 6, "u32": 7, "u64": 8,
    "f32": 9, "f64": 10, "ext": 11, "f32u": 0x19, "f64u": 0x1A, "str": 0x20, "bool": 0x21, "ts": 0x44,
    "c64": 0x0800_000C, "c128": 0x1000_000D,
}  # fmt: skip
CODE_NAMES = {code: name for name, code in TYPE_CODES.items()}

# type -> (struct code, size in bytes) for the types that are one fixed-size number
_STRUCT = {
    "i8": ("b", 1), "i16": ("h", 2), "i32": ("i", 4), "i64": ("q", 8), "u8": ("B", 1), "u16": ("H", 2),
    "u32": ("I", 4), "u64": ("Q", 8), "f32": ("f", 4), "f64": ("d", 8), "f32u": ("f", 4), "f64u": ("d", 8),
    "bool": ("B", 1),
}  # fmt: skip
FIXED_SIZES = {**{k: v[1] for k, v in _STRUCT.items()}, "ts": 16, "c64": 8, "c128": 16}

_NUMPY = {
    "i8": "i1", "i16": "i2", "i32": "i4", "i64": "i8", "u8": "u1", "u16": "u2", "u32": "u4", "u64": "u8",
    "f32": "f4", "f64": "f8", "bool": "u1", "c64": "c8", "c128": "c16",
}  # fmt: skip


@dataclass(frozen=True)
class Prop:
    """A property: `Prop("unit_string", "g")`. The type is inferred (`int` -> i32, `float` -> f64, ...) unless given."""

    name: str
    value: Any
    type: str | None = None

    def resolved_type(self) -> str:
        if self.type is not None:
            return self.type
        return infer_type(self.value)


def infer_type(value: Any) -> str:
    if isinstance(value, bool | np.bool_):
        return "bool"
    if isinstance(value, np.generic):
        kind = np.dtype(value.dtype).str[1:]
        for name, code in _NUMPY.items():
            if code == kind:
                return name
        raise TypeError(f"no TDMS type for {value.dtype}")
    if isinstance(value, int):
        return "i32" if -(2**31) <= value < 2**31 else "i64"
    if isinstance(value, float):
        return "f64"
    if isinstance(value, str):
        return "str"
    if isinstance(value, Timestamp):
        return "ts"
    if isinstance(value, complex):
        return "c128"
    raise TypeError(f"no TDMS type for {type(value).__name__}")


def data_type_of(values: Any) -> str:
    """The TDMS data type of a channel's values (a numpy array, a list of str, or a list of `Timestamp`)."""
    if isinstance(values, np.ndarray):
        if values.dtype.kind == "b":
            return "bool"
        key = f"{values.dtype.kind}{values.dtype.itemsize}"
        for name, code in _NUMPY.items():
            if code == key and name != "bool":
                return name
        raise TypeError(f"no TDMS data type for {values.dtype}")
    values = list(values)
    if values and all(isinstance(v, str) for v in values):
        return "str"
    if values and all(isinstance(v, Timestamp) for v in values):
        return "ts"
    raise TypeError("values must be a numpy array, a list of str or a list of Timestamp")


def pack_string(text: str, endian: str) -> bytes:
    raw = text.encode("utf-8")
    return struct.pack(endian + "I", len(raw)) + raw


def pack_value(type_name: str, value: Any, endian: str) -> bytes:
    """One property value (or one element of a fixed-size data channel)."""
    if type_name == "void":
        return b""
    if type_name in _STRUCT:
        code, _ = _STRUCT[type_name]
        return struct.pack(endian + code, int(value) if code not in "fd" else float(value))
    if type_name == "str":
        return pack_string(value, endian)
    if type_name == "ts":
        # a 128 bit value in the segment's byte order: fraction then seconds when little-endian, seconds then
        # fraction when big-endian (as one big-endian 128 bit number)
        if endian == ">":
            return struct.pack(">qQ", value.seconds, value.fraction)
        return struct.pack("<Qq", value.fraction, value.seconds)
    if type_name == "c64":
        return struct.pack(endian + "ff", value.real, value.imag)
    if type_name == "c128":
        return struct.pack(endian + "dd", value.real, value.imag)
    if type_name == "ext":
        return bytes(value)  # an 80 bit extended float, 16 bytes as LabVIEW stores it; the tests only need it as bytes
    raise ValueError(f"cannot pack a {type_name} value")


def pack_prop(prop: Prop, endian: str) -> bytes:
    type_name = prop.resolved_type()
    return (
        pack_string(prop.name, endian)
        + struct.pack(endian + "I", TYPE_CODES[type_name])
        + pack_value(type_name, prop.value, endian)
    )


def escape_name(name: str) -> str:
    return name.replace("'", "''")


def object_path(group: str | None = None, channel: str | None = None) -> str:
    if group is None:
        return "/"
    path = f"/'{escape_name(group)}'"
    return path if channel is None else f"{path}/'{escape_name(channel)}'"


def pack_values(type_name: str, values: Any, endian: str) -> bytes:
    """The raw bytes of a channel's values, all of them one after the other."""
    if type_name == "str":
        strings = [s.encode("utf-8") for s in values]
        ends = np.cumsum([len(s) for s in strings]).astype(endian + "u4") if strings else np.empty(0, endian + "u4")
        return ends.tobytes() + b"".join(strings)
    if type_name == "ts":
        return b"".join(pack_value("ts", v, endian) for v in values)
    if type_name == "bool":
        return np.asarray(values, dtype=bool).astype("u1").tobytes()
    return np.asarray(values, dtype=np.dtype(endian + _NUMPY[type_name])).tobytes()


@dataclass
class ObjectSpec:
    """One object of a segment's metadata (root, group or channel) and what the segment says about its data."""

    path: str
    props: tuple[Prop, ...] = ()
    values: Any = None
    """The channel's values in this segment's first chunk (None: the channel has no data here)."""
    index: Literal["explicit", "same", "none"] = "explicit"
    """`explicit`: a raw data index with type and count; `same`: index 0, "as in the previous segment"; `none`: no data."""
    data_type: str | None = None
    """Overrides the type taken from `values` (for the error cases)."""
    index_bytes: bytes | None = None
    """Replaces the whole raw data index (including its length), for the error cases (e.g. DAQmx)."""
    listed: bool = True
    """False: the object's raw data is written but the object is left out of the metadata (a segment without a
    new object list only lists the objects it changes; the others still get their share of the raw data)."""

    def has_data(self) -> bool:
        return self.values is not None and self.index != "none" and self.index_bytes is None

    def type_name(self) -> str:
        return self.data_type or data_type_of(self.values)

    def count(self) -> int:
        return len(self.values)


@dataclass
class Segment:
    """One segment. The flags default to what a writer normally produces; override them for the odd cases."""

    endian: str = LITTLE
    version: int = 4713
    meta: bool = True
    """Write a metadata section (table of contents flag kTocMetaData)."""
    newlist: bool = True
    """The object list is new (kTocNewObjList); without it the objects listed only update the previous ones."""
    interleaved: bool = False
    objects: list[ObjectSpec] = field(default_factory=list)
    chunks: list[dict[str, Any]] = field(default_factory=list)
    """Further chunks of raw data after the first: each maps a channel's object path to its values."""
    padding: int = 0
    """Bytes between the metadata and the raw data (the raw data offset says where the data starts)."""
    next_offset: int | None = None
    """None: computed. `UNKNOWN_NEXT_OFFSET`: a segment that is still being written. Anything else: as given."""
    toc: int | None = None
    """Replaces the table of contents entirely."""
    tag: bytes = b"TDSm"
    trailing_raw: bytes = b""
    """Extra bytes appended to the raw data that no object accounts for (a partial chunk, garbage)."""

    # -- building the object list ------------------------------------------------------------------

    def root(self, *props: Prop) -> Segment:
        self.objects.append(ObjectSpec(object_path(), props, index="none"))
        return self

    def group(self, name: str, *props: Prop) -> Segment:
        self.objects.append(ObjectSpec(object_path(name), props, index="none"))
        return self

    def channel(
        self,
        group: str,
        name: str,
        values: Any = None,
        *props: Prop,
        index: Literal["explicit", "same", "none"] | None = None,
        data_type: str | None = None,
        index_bytes: bytes | None = None,
        listed: bool = True,
    ) -> Segment:
        if index is None:
            index = "explicit" if values is not None else "none"
        self.objects.append(ObjectSpec(object_path(group, name), props, values, index, data_type, index_bytes, listed))
        return self

    def chunk_of(self, values_by_channel: dict[tuple[str, str], Any]) -> Segment:
        self.chunks.append({object_path(*key): values for key, values in values_by_channel.items()})
        return self

    # -- bytes -------------------------------------------------------------------------------------

    def _data_objects(self) -> list[ObjectSpec]:
        return [o for o in self.objects if o.has_data()]

    def _raw_chunk(self, values_of: dict[str, Any]) -> bytes:
        """One chunk of raw data: the channels one after the other, or row by row when interleaved."""
        e = self.endian
        channels = self._data_objects()
        if not self.interleaved:
            return b"".join(pack_values(o.type_name(), values_of[o.path], e) for o in channels)
        types = [o.type_name() for o in channels]
        columns = [values_of[o.path] for o in channels]
        rows = min(len(c) for c in columns) if columns else 0
        out = bytearray()
        for i in range(rows):
            for type_name, column in zip(types, columns, strict=True):
                if type_name == "str":
                    raise ValueError("string channels cannot be interleaved")
                out += pack_values(type_name, [column[i]], e)
        return bytes(out)

    def _metadata(self) -> bytes:
        e = self.endian
        listed = [o for o in self.objects if o.listed]
        out = bytearray(struct.pack(e + "I", len(listed)))
        for o in listed:
            out += pack_string(o.path, e)
            if o.index_bytes is not None:
                out += o.index_bytes
            elif o.index == "none" or not o.has_data():
                out += struct.pack(e + "I", 0xFFFF_FFFF)
            elif o.index == "same":
                out += struct.pack(e + "I", 0)
            else:
                name = o.type_name()
                if name == "str":
                    total = len(pack_values("str", o.values, e))
                    out += struct.pack(e + "IIIQQ", 28, TYPE_CODES["str"], 1, o.count(), total)
                else:
                    out += struct.pack(e + "IIIQ", 20, TYPE_CODES[name], 1, o.count())
            out += struct.pack(e + "I", len(o.props))
            for prop in o.props:
                out += pack_prop(prop, e)
        return bytes(out)

    def build(self) -> bytes:
        e = self.endian
        metadata = self._metadata() if self.meta else b""
        first = {o.path: o.values for o in self._data_objects()}
        raw = b"".join(self._raw_chunk(values) for values in [first, *self.chunks]) if self._data_objects() else b""
        raw += self.trailing_raw
        toc = self.toc
        if toc is None:
            toc = (2 if self.meta else 0) | (4 if self.meta and self.newlist else 0) | (8 if raw else 0)
            toc |= (32 if self.interleaved else 0) | (64 if e == BIG else 0)
        raw_offset = len(metadata) + self.padding
        nxt = raw_offset + len(raw) if self.next_offset is None else self.next_offset
        lead_in = self.tag + struct.pack("<I", toc) + struct.pack(e + "IQQ", self.version, nxt, raw_offset)
        return lead_in + metadata + b"\x00" * self.padding + raw


def tdms_file(*segments: Segment) -> bytes:
    """The bytes of a file with these segments."""
    return b"".join(s.build() for s in segments)


def raw_segment(
    metadata: bytes, raw: bytes = b"", *, endian: str = LITTLE, version: int = 4713, toc: int | None = None
) -> bytes:
    """A segment from hand-made metadata bytes, for the cases that need a metadata section no `Segment` would write."""
    if toc is None:
        toc = 2 | 4 | (8 if raw else 0) | (64 if endian == BIG else 0)
    lead_in = (
        b"TDSm" + struct.pack("<I", toc) + struct.pack(endian + "IQQ", version, len(metadata) + len(raw), len(metadata))
    )
    return lead_in + metadata + raw
