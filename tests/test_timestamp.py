"""`tdmspy.Timestamp`: seconds since 1904 plus a fraction of 2**-64 s, converted exactly."""

import numpy as np
import pytest

from tdmspy import Timestamp
from tdmspy.timestamp import FRACTIONS_PER_SECOND, UNIX_EPOCH_IN_TDMS_SECONDS


def test_the_unix_epoch():
    assert Timestamp(UNIX_EPOCH_IN_TDMS_SECONDS).to_datetime64("s") == np.datetime64(0, "s")
    assert Timestamp(UNIX_EPOCH_IN_TDMS_SECONDS + 86_400).to_datetime64("us") == np.datetime64(
        "1970-01-02T00:00:00.000000"
    )


def test_the_fraction_is_truncated_never_rounded_up():
    half = Timestamp(UNIX_EPOCH_IN_TDMS_SECONDS, FRACTIONS_PER_SECOND // 2)
    assert half.to_datetime64("us") == np.datetime64(500_000, "us")
    assert half.to_datetime64("ms") == np.datetime64(500, "ms")
    almost_one = Timestamp(UNIX_EPOCH_IN_TDMS_SECONDS, FRACTIONS_PER_SECOND - 1)
    assert almost_one.to_datetime64("us") == np.datetime64(999_999, "us")
    assert almost_one.to_datetime64("s") == np.datetime64(0, "s")


def test_negative_seconds_count_back_from_the_epoch():
    before = Timestamp(UNIX_EPOCH_IN_TDMS_SECONDS - 1, FRACTIONS_PER_SECOND // 4)  # 0.25 s into the last second of 1969
    assert before.to_datetime64("us") == np.datetime64(-750_000, "us")


def test_microseconds_survive_a_round_trip_through_datetime64():
    for text in ("2026-11-10T13:47:07.814238", "1969-12-31T23:59:59.999999", "2000-02-29T00:00:00.000001"):
        value = np.datetime64(text, "us")
        assert Timestamp.from_datetime64(value).to_datetime64("us") == value


def test_the_fraction_must_be_inside_a_second():
    with pytest.raises(ValueError):
        Timestamp(0, FRACTIONS_PER_SECOND)
    with pytest.raises(ValueError):
        Timestamp(0, -1)


def test_unknown_units_are_refused():
    with pytest.raises(ValueError):
        Timestamp(0).to_datetime64("h")


def test_timestamps_order_by_time():
    assert Timestamp(5, 0) < Timestamp(5, 1) < Timestamp(6, 0)
