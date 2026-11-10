"""Constants and the data type table of the TDMS format (see doc/design.md for what is supported)."""

from __future__ import annotations

from dataclasses import dataclass

LEAD_IN_SIZE = 28
SIGNATURE = b"TDSm"
INDEX_SIGNATURE = b"TDSh"
VERSIONS = (4712, 4713)
UNKNOWN_NEXT_OFFSET = 0xFFFF_FFFF_FFFF_FFFF

# table of contents flags
TOC_METADATA = 1 << 1
TOC_NEW_OBJECT_LIST = 1 << 2
TOC_RAW_DATA = 1 << 3
TOC_INTERLEAVED = 1 << 5
TOC_BIG_ENDIAN = 1 << 6
TOC_DAQMX = 1 << 7

# raw data index values
INDEX_NONE = 0xFFFF_FFFF
INDEX_SAME = 0
INDEX_DAQMX = frozenset({0x1269, 0x126A, 0x1369, 0x136A})

TYPE_STRING = 0x20
TYPE_BOOL = 0x21
TYPE_TIMESTAMP = 0x44
TYPE_EXTENDED = 0x0B
TYPE_COMPLEX_SINGLE = 0x0800_000C
TYPE_COMPLEX_DOUBLE = 0x1000_000D


@dataclass(frozen=True)
class DataType:
    """A type a channel can have: the numpy type code of one element (without byte order) and its size in the file."""

    code: int
    name: str
    numpy: str
    size: int


_NUMERIC = {
    1: ("int8", "i1", 1), 2: ("int16", "i2", 2), 3: ("int32", "i4", 4), 4: ("int64", "i8", 8),
    5: ("uint8", "u1", 1), 6: ("uint16", "u2", 2), 7: ("uint32", "u4", 4), 8: ("uint64", "u8", 8),
    9: ("float32", "f4", 4), 10: ("float64", "f8", 8), 0x19: ("float32", "f4", 4), 0x1A: ("float64", "f8", 8),
    TYPE_BOOL: ("bool", "u1", 1), TYPE_TIMESTAMP: ("timestamp", "", 16),
    TYPE_COMPLEX_SINGLE: ("complex64", "c8", 8), TYPE_COMPLEX_DOUBLE: ("complex128", "c16", 16),
}  # fmt: skip

DATA_TYPES: dict[int, DataType] = {
    code: DataType(code, name, np_code, size) for code, (name, np_code, size) in _NUMERIC.items()
}
DATA_TYPES[TYPE_STRING] = DataType(TYPE_STRING, "string", "", 0)

# property value types: code -> (struct code, size); strings, timestamps and complex numbers are handled apart
PROPERTY_STRUCT = {
    1: ("b", 1), 2: ("h", 2), 3: ("i", 4), 4: ("q", 8), 5: ("B", 1), 6: ("H", 2), 7: ("I", 4), 8: ("Q", 8),
    9: ("f", 4), 10: ("d", 8), 0x19: ("f", 4), 0x1A: ("d", 8), TYPE_BOOL: ("B", 1),
}  # fmt: skip
