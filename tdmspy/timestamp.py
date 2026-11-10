"""The TDMS timestamp: seconds since 1904-01-01 00:00:00 UTC plus a fraction of a second.

A property or channel value of type "timestamp" keeps both parts exactly, as the file stores them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FRACTIONS_PER_SECOND = 1 << 64
"""The fraction counts units of 2**-64 seconds."""

UNIX_EPOCH_IN_TDMS_SECONDS = 2_082_844_800
"""Seconds between 1904-01-01 and 1970-01-01."""

_UNITS_PER_SECOND = {"s": 1, "ms": 1_000, "us": 1_000_000, "ns": 1_000_000_000}


@dataclass(frozen=True, order=True)
class Timestamp:
    """`seconds` since 1904-01-01 UTC (may be negative) and `fraction` of a second in units of 2**-64 s."""

    seconds: int
    fraction: int = 0

    def __post_init__(self) -> None:
        if not 0 <= self.fraction < FRACTIONS_PER_SECOND:
            raise ValueError(f"fraction must be in [0, 2**64), got {self.fraction}")

    def to_datetime64(self, unit: str = "us") -> np.datetime64:
        """UTC time as a `numpy.datetime64` of `unit` (`s`, `ms`, `us` or `ns`), truncated (never rounded up).

        The fraction is converted with integer arithmetic, so the result is exact.
        """
        try:
            per_second = _UNITS_PER_SECOND[unit]
        except KeyError:
            raise ValueError(f"unit must be one of {sorted(_UNITS_PER_SECOND)}, got {unit!r}") from None
        ticks = (
            self.seconds - UNIX_EPOCH_IN_TDMS_SECONDS
        ) * per_second + self.fraction * per_second // FRACTIONS_PER_SECOND
        return np.datetime64(ticks, unit)

    @classmethod
    def from_datetime64(cls, value: np.datetime64) -> Timestamp:
        """The timestamp of a `numpy.datetime64` (any unit up to `ns`), exact for the value given."""
        unit = np.datetime_data(value.dtype)[0]
        if unit not in _UNITS_PER_SECOND:
            raise ValueError(f"unsupported datetime64 unit {unit!r}")
        per_second = _UNITS_PER_SECOND[unit]
        ticks = int(value.astype(f"datetime64[{unit}]").astype(np.int64))
        seconds, remainder = divmod(ticks, per_second)
        # round up so that to_datetime64 of the result gives the same value back
        fraction = -(-remainder * FRACTIONS_PER_SECOND // per_second)
        return cls(seconds + UNIX_EPOCH_IN_TDMS_SECONDS, fraction)
