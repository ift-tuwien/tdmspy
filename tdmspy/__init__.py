"""tdmspy: a pure Python reader for National Instruments TDMS files (MIT licensed).

    import tdmspy

    file = tdmspy.read("measurement.tdms")
    channel = file["rawdata"]["Channel1"]
    values = channel.read()

The contract of the reader (what it reads, what it refuses, and which special cases of the format it
handles) is in `doc/design.md` and `doc/special-cases.md`; the tests in `tests/` check every case.
"""

from __future__ import annotations

from tdmspy.errors import TdmsError, TdmsFormatError, UnsupportedFeatureError, UnsupportedScalingError
from tdmspy.file import Channel, Group, TdmsFile, read
from tdmspy.timestamp import Timestamp

__all__ = [
    "Channel",
    "Group",
    "TdmsError",
    "TdmsFile",
    "TdmsFormatError",
    "Timestamp",
    "UnsupportedFeatureError",
    "UnsupportedScalingError",
    "read",
]

__implemented__ = True
