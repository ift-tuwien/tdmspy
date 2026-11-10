"""What a file must read as: the expected structure and values of a test case, and the means to compare it.

`Truth` is written down by hand next to each case (never derived from the bytes), so it is an independent
statement of what the file means. The same shape is built from what the reader returned (`from_tdmspy`),
and `differences` compares two of them and says in words what differs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from tdmspy.timestamp import Timestamp
from tests.builder import _NUMPY, Prop, data_type_of


def native(prop: Prop) -> Any:
    """The Python value a property reads as: numpy scalars become Python numbers (widths are not kept)."""
    type_name = prop.resolved_type()
    value = prop.value
    if type_name in _NUMPY and type_name != "bool":
        return np.dtype(_NUMPY[type_name]).type(value).item()
    if type_name in ("f32u", "f64u"):
        return np.dtype("f4" if type_name == "f32u" else "f8").type(value).item()
    return value


def props_of(*props: Prop) -> dict[str, Any]:
    return {p.name: native(p) for p in props}


def normalize_values(values: Any, type_name: str | None) -> Any:
    """Channel values as numpy: numbers as given, strings as an object array, timestamps as `datetime64[us]`."""
    if values is None:
        return None
    if type_name == "str":
        out = np.empty(len(values), dtype=object)
        out[:] = list(values)
        return out
    if type_name == "ts":
        return np.array([v.to_datetime64("us") for v in values], dtype="datetime64[us]")
    return np.asarray(values)


@dataclass
class ChannelTruth:
    values: Any
    """The raw values, or None for a channel without a data type (it has no data at all)."""
    type: str | None
    props: dict[str, Any] = field(default_factory=dict)
    scaled: Any = None
    """What `read(scaled=True)` gives if it differs from the raw values (float64), else None."""
    scaled_error: bool = False
    """`read(scaled=True)` raises `UnsupportedScalingError`."""


@dataclass
class GroupTruth:
    props: dict[str, Any] = field(default_factory=dict)
    channels: dict[str, ChannelTruth] = field(default_factory=dict)


@dataclass
class Truth:
    props: dict[str, Any] = field(default_factory=dict)
    groups: dict[str, GroupTruth] = field(default_factory=dict)
    truncated: bool = False

    # -- writing it down ---------------------------------------------------------------------------

    def root(self, *props: Prop) -> Truth:
        self.props.update(props_of(*props))
        return self

    def group(self, name: str, *props: Prop) -> Truth:
        self.groups.setdefault(name, GroupTruth()).props.update(props_of(*props))
        return self

    def channel(
        self,
        group: str,
        name: str,
        values: Any = None,
        *props: Prop,
        scaled: Any = None,
        scaled_error: bool = False,
    ) -> Truth:
        type_name = None if values is None else data_type_of(values)
        channels = self.groups.setdefault(group, GroupTruth()).channels
        channels[name] = ChannelTruth(
            normalize_values(values, type_name), type_name, props_of(*props), scaled, scaled_error
        )
        return self


# -- adapters: what a reader returned, in the same shape --------------------------------------------


def _type_name(dtype: np.dtype) -> str | None:
    if dtype.kind == "V":
        return None
    if dtype.kind == "O":
        return "str"
    if dtype.kind == "M":
        return "ts"
    if dtype.kind == "b":
        return "bool"
    key = f"{dtype.kind}{dtype.itemsize}"
    for name, code in _NUMPY.items():
        if code == f"{dtype.kind}{dtype.itemsize}" and name != "bool":
            return name
    raise AssertionError(f"unexpected dtype {dtype} ({key})")


def from_tdmspy(file: Any) -> Truth:
    truth = Truth(dict(file.properties), truncated=bool(getattr(file, "truncated", False)))
    for group in file.groups:
        g = truth.groups.setdefault(group.name, GroupTruth(dict(group.properties)))
        for channel in group.channels:
            raw = channel.read(scaled=False)
            type_name = None if channel.dtype is None else _type_name(channel.dtype)
            g.channels[channel.name] = ChannelTruth(
                None if type_name is None else raw, type_name, dict(channel.properties)
            )
            entry = g.channels[channel.name]
            try:
                scaled = channel.read(scaled=True)
            except Exception as error:
                entry.scaled_error = type(error).__name__ == "UnsupportedScalingError"
                if not entry.scaled_error:
                    raise
            else:
                if scaled.dtype.kind in "iufc" and not (
                    scaled.dtype == raw.dtype and scaled.tobytes() == raw.tobytes()
                ):
                    entry.scaled = scaled
    return truth


# -- comparing -------------------------------------------------------------------------------------


def _prop_equal(a: Any, b: Any) -> bool:
    if isinstance(a, Timestamp):
        a = a.to_datetime64("us")
    if isinstance(b, Timestamp):
        b = b.to_datetime64("us")
    if isinstance(a, np.datetime64) or isinstance(b, np.datetime64):
        return (
            isinstance(a, np.datetime64)
            and isinstance(b, np.datetime64)
            and a.astype("datetime64[us]") == b.astype("datetime64[us]")
        )
    if isinstance(a, bool) != isinstance(b, bool):
        return False
    if isinstance(a, float) and isinstance(b, float):
        return a == b and math.copysign(1, a) == math.copysign(1, b) or (math.isnan(a) and math.isnan(b))
    return type(a) is type(b) and a == b


def _props_differences(where: str, expected: dict[str, Any], actual: dict[str, Any]) -> list[str]:
    out = []
    for key in expected.keys() - actual.keys():
        out.append(f"{where}: property {key!r} is missing")
    for key in actual.keys() - expected.keys():
        out.append(f"{where}: unexpected property {key!r}")
    for key in expected.keys() & actual.keys():
        if not _prop_equal(expected[key], actual[key]):
            out.append(
                f"{where}: property {key!r} is {actual[key]!r} ({type(actual[key]).__name__}), expected {expected[key]!r} ({type(expected[key]).__name__})"
            )
    return out


def _array_equal(expected: np.ndarray, actual: np.ndarray) -> bool:
    if expected.dtype.kind in ("i", "u", "f", "c"):
        return expected.dtype == actual.dtype and expected.tobytes() == actual.tobytes()
    return expected.dtype == actual.dtype and expected.shape == actual.shape and bool(np.all(expected == actual))


def differences(
    expected: Truth, actual: Truth, *, scaled_rtol: float = 1e-12, check_truncated: bool = True
) -> list[str]:
    """Everything that differs, as sentences (an empty list means the same)."""
    out = _props_differences("file", expected.props, actual.props)
    if list(expected.groups) != list(actual.groups):
        out.append(f"groups are {list(actual.groups)}, expected {list(expected.groups)} (in this order)")
    if check_truncated and expected.truncated != actual.truncated:
        out.append(f"truncated is {actual.truncated}, expected {expected.truncated}")
    for name in expected.groups.keys() & actual.groups.keys():
        want, got = expected.groups[name], actual.groups[name]
        out += _props_differences(f"group {name!r}", want.props, got.props)
        if list(want.channels) != list(got.channels):
            out.append(
                f"group {name!r}: channels are {list(got.channels)}, expected {list(want.channels)} (in this order)"
            )
        for channel in want.channels.keys() & got.channels.keys():
            out += _channel_differences(
                f"channel {name!r}/{channel!r}", want.channels[channel], got.channels[channel], scaled_rtol
            )
    return out


def _channel_differences(where: str, want: ChannelTruth, got: ChannelTruth, rtol: float) -> list[str]:
    out = _props_differences(where, want.props, got.props)
    if want.type != got.type:
        return [*out, f"{where}: type is {got.type}, expected {want.type}"]
    if want.values is None:
        return out
    if len(want.values) != len(got.values):
        return [*out, f"{where}: {len(got.values)} values, expected {len(want.values)}"]
    if not _array_equal(want.values, got.values):
        bad = np.flatnonzero(want.values != got.values)[:3] if want.values.dtype.kind != "O" else []
        out.append(
            f"{where}: values differ (dtype {got.values.dtype}, expected {want.values.dtype}; first differing indexes {list(bad)})"
        )
    if want.scaled_error != got.scaled_error:
        out.append(f"{where}: scaled_error is {got.scaled_error}, expected {want.scaled_error}")
    if want.scaled is None and got.scaled is not None:
        out.append(f"{where}: reads scaled but should not")
    elif want.scaled is not None:
        if got.scaled is None:
            out.append(f"{where}: reads unscaled but should be scaled")
        elif got.scaled.shape != want.scaled.shape or not np.allclose(
            got.scaled, want.scaled, rtol=rtol, atol=0.0, equal_nan=True
        ):
            out.append(f"{where}: scaled values differ")
    return out
