"""NI scaling: the scales declared in the properties of a channel, its group or the file (see doc/design.md)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from tdmspy.errors import UnsupportedScalingError

RAW_DATA_INPUT_SOURCE = 0xFFFF_FFFF


def scale_count(properties: Mapping[str, Any]) -> int | None:
    """How many scales these properties declare, `None` if none or the data is already scaled."""
    count = properties.get("NI_Number_Of_Scales")
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        return None
    if properties.get("NI_Scaling_Status", "unscaled") == "scaled":
        return None
    return count


def _number(properties: Mapping[str, Any], key: str) -> float:
    value = properties.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise UnsupportedScalingError(f"the scaling property {key!r} is missing or not a number")
    return float(value)


def apply_scaling(raw: np.ndarray, *levels: Mapping[str, Any]) -> np.ndarray:
    """`raw` scaled by the first of `levels` (channel, group, file properties) that declares scales, else `raw` itself."""
    for properties in levels:
        count = scale_count(properties)
        if count is not None:
            return _scale(raw, properties, count)
    return raw


def _scale(raw: np.ndarray, properties: Mapping[str, Any], count: int) -> np.ndarray:
    def compute(index: int, depth: int) -> np.ndarray:
        if index == RAW_DATA_INPUT_SOURCE:
            return raw.astype(np.float64)
        if not 0 <= index < count or depth > count:
            raise UnsupportedScalingError(
                f"NI scale {index} does not exist or the scales refer to each other in a circle"
            )
        prefix = f"NI_Scale[{index}]_"
        kind = properties.get(prefix + "Scale_Type")
        if kind not in ("Linear", "Polynomial"):
            what = "a DAQmx scaler" if kind is None else f"the scale type {kind!r}"
            raise UnsupportedScalingError(f"{what} is not supported (only Linear and Polynomial scales are)")
        source = properties.get(f"{prefix}{kind}_Input_Source", RAW_DATA_INPUT_SOURCE)
        values = compute(int(source), depth + 1)
        if kind == "Linear":
            return values * _number(properties, prefix + "Linear_Slope") + _number(
                properties, prefix + "Linear_Y_Intercept"
            )
        size = properties.get(prefix + "Polynomial_Coefficients_Size", 4)
        if isinstance(size, bool) or not isinstance(size, int) or not 0 <= size <= 1024:
            raise UnsupportedScalingError(f"the polynomial of scale {index} has an invalid number of coefficients")
        coefficients = [_number(properties, f"{prefix}Polynomial_Coefficients[{k}]") for k in range(size)]
        result = np.zeros(len(values))
        for coefficient in reversed(coefficients):  # Horner's rule
            result = result * values + coefficient
        return result

    return compute(count - 1, 0)
