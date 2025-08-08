from datetime import date, datetime, timedelta, timezone
import numpy as np
from pandas import NA, NaT, Timestamp

def test_constructor_ambiguous_dst() -> None:
    ts = Timestamp(1382835600000000000, tz='dateutil/Europe/London')
    expected = ts._value
    result = Timestamp(ts)._value
    assert result == expected

def test_constructor_before_dst_switch(epoch: int) -> None:
    ts = Timestamp(epoch, tz='dateutil/America/Los_Angeles')
    result = ts.tz.dst(ts)
    expected = timedelta(seconds=0)
    assert Timestamp(ts)._value == epoch
    assert result == expected

def test_timestamp_constructor_identity() -> None:
    expected = Timestamp('2017-01-01T12')
    result = Timestamp(expected)
    assert result is expected

def test_timestamp_nano_range(nano: int) -> None:
    with pytest.raises(ValueError, match='nanosecond must be in 0..999'):
        Timestamp(year=2022, month=1, day=1, nanosecond=nano)

def test_non_nano_value() -> None:
    result = Timestamp('1800-01-01', unit='s').value
    assert result == -5364662400000000000
    msg = "Cannot convert Timestamp to nanoseconds without overflow. Use `.asm8.view\\('i8'\\)` to cast represent Timestamp in its own unit \\(here, s\\).$"
    ts = Timestamp('0300-01-01')
    with pytest.raises(OverflowError, match=msg):
        ts.value
    result = ts.asm8.view('i8')
    assert result == -52700112000

def test_timestamp_constructor_na_value(na_value: Any) -> None:
    result = Timestamp(na_value)
    expected = NaT
    assert result is expected
