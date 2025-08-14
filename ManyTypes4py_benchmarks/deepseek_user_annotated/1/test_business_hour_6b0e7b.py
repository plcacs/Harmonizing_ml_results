"""
Tests for offsets.BusinessHour
"""

from __future__ import annotations

from datetime import (
    datetime,
    time as dt_time,
)
from typing import (
    Any,
    Dict,
    List,
    Tuple,
    Union,
)

import pytest

from pandas._libs.tslibs import (
    Timedelta,
    Timestamp,
)
from pandas._libs.tslibs.offsets import (
    BDay,
    BusinessHour,
    Nano,
)

from pandas import (
    DatetimeIndex,
    _testing as tm,
    date_range,
)
from pandas.tests.tseries.offsets.common import assert_offset_equal


@pytest.fixture
def dt() -> datetime:
    return datetime(2014, 7, 1, 10, 00)


@pytest.fixture
def _offset() -> type[BusinessHour]:
    return BusinessHour


@pytest.fixture
def offset1() -> BusinessHour:
    return BusinessHour()


@pytest.fixture
def offset2() -> BusinessHour:
    return BusinessHour(n=3)


@pytest.fixture
def offset3() -> BusinessHour:
    return BusinessHour(n=-1)


@pytest.fixture
def offset4() -> BusinessHour:
    return BusinessHour(n=-4)


@pytest.fixture
def offset5() -> BusinessHour:
    return BusinessHour(start=dt_time(11, 0), end=dt_time(14, 30))


@pytest.fixture
def offset6() -> BusinessHour:
    return BusinessHour(start="20:00", end="05:00")


@pytest.fixture
def offset7() -> BusinessHour:
    return BusinessHour(n=-2, start=dt_time(21, 30), end=dt_time(6, 30))


@pytest.fixture
def offset8() -> BusinessHour:
    return BusinessHour(start=["09:00", "13:00"], end=["12:00", "17:00"])


@pytest.fixture
def offset9() -> BusinessHour:
    return BusinessHour(n=3, start=["09:00", "22:00"], end=["13:00", "03:00"])


@pytest.fixture
def offset10() -> BusinessHour:
    return BusinessHour(n=-1, start=["23:00", "13:00"], end=["02:00", "17:00"])


class TestBusinessHour:
    @pytest.mark.parametrize(
        "start,end,match",
        [
            (
                dt_time(11, 0, 5),
                "17:00",
                "time data must be specified only with hour and minute",
            ),
            ("AAA", "17:00", "time data must match '%H:%M' format"),
            ("14:00:05", "17:00", "time data must match '%H:%M' format"),
            ([], "17:00", "Must include at least 1 start time"),
            ("09:00", [], "Must include at least 1 end time"),
            (
                ["09:00", "11:00"],
                "17:00",
                "number of starting time and ending time must be the same",
            ),
            (
                ["09:00", "11:00"],
                ["10:00"],
                "number of starting time and ending time must be the same",
            ),
            (
                ["09:00", "11:00"],
                ["12:00", "20:00"],
                r"invalid starting and ending time\(s\): opening hours should not "
                "touch or overlap with one another",
            ),
            (
                ["12:00", "20:00"],
                ["09:00", "11:00"],
                r"invalid starting and ending time\(s\): opening hours should not "
                "touch or overlap with one another",
            ),
        ],
    )
    def test_constructor_errors(self, start: Any, end: Any, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            BusinessHour(start=start, end=end)

    def test_different_normalize_equals(self, _offset: type[BusinessHour]) -> None:
        # GH#21404 changed __eq__ to return False when `normalize` does not match
        offset = _offset()
        offset2 = _offset(normalize=True)
        assert offset != offset2

    def test_repr(
        self,
        offset1: BusinessHour,
        offset2: BusinessHour,
        offset3: BusinessHour,
        offset4: BusinessHour,
        offset5: BusinessHour,
        offset6: BusinessHour,
        offset7: BusinessHour,
        offset8: BusinessHour,
        offset9: BusinessHour,
        offset10: BusinessHour,
    ) -> None:
        assert repr(offset1) == "<BusinessHour: bh=09:00-17:00>"
        assert repr(offset2) == "<3 * BusinessHours: bh=09:00-17:00>"
        assert repr(offset3) == "<-1 * BusinessHour: bh=09:00-17:00>"
        assert repr(offset4) == "<-4 * BusinessHours: bh=09:00-17:00>"

        assert repr(offset5) == "<BusinessHour: bh=11:00-14:30>"
        assert repr(offset6) == "<BusinessHour: bh=20:00-05:00>"
        assert repr(offset7) == "<-2 * BusinessHours: bh=21:30-06:30>"
        assert repr(offset8) == "<BusinessHour: bh=09:00-12:00,13:00-17:00>"
        assert repr(offset9) == "<3 * BusinessHours: bh=09:00-13:00,22:00-03:00>"
        assert repr(offset10) == "<-1 * BusinessHour: bh=13:00-17:00,23:00-02:00>"

    def test_with_offset(self, dt: datetime) -> None:
        expected = Timestamp("2014-07-01 13:00")

        assert dt + BusinessHour() * 3 == expected
        assert dt + BusinessHour(n=3) == expected

    @pytest.mark.parametrize(
        "offset_name",
        ["offset1", "offset2", "offset3", "offset4", "offset8", "offset9", "offset10"],
    )
    def test_eq_attribute(self, offset_name: str, request: Any) -> None:
        offset = request.getfixturevalue(offset_name)
        assert offset == offset

    @pytest.mark.parametrize(
        "offset1,offset2",
        [
            (BusinessHour(start="09:00"), BusinessHour()),
            (
                BusinessHour(start=["23:00", "13:00"], end=["12:00", "17:00"]),
                BusinessHour(start=["13:00", "23:00"], end=["17:00", "12:00"]),
            ),
        ],
    )
    def test_eq(self, offset1: BusinessHour, offset2: BusinessHour) -> None:
        assert offset1 == offset2

    @pytest.mark.parametrize(
        "offset1,offset2",
        [
            (BusinessHour(), BusinessHour(-1)),
            (BusinessHour(start="09:00"), BusinessHour(start="09:01")),
            (
                BusinessHour(start="09:00", end="17:00"),
                BusinessHour(start="17:00", end="09:01"),
            ),
            (
                BusinessHour(start=["13:00", "23:00"], end=["18:00", "07:00"]),
                BusinessHour(start=["13:00", "23:00"], end=["17:00", "12:00"]),
            ),
        ],
    )
    def test_neq(self, offset1: BusinessHour, offset2: BusinessHour) -> None:
        assert offset1 != offset2

    @pytest.mark.parametrize(
        "offset_name",
        ["offset1", "offset2", "offset3", "offset4", "offset8", "offset9", "offset10"],
    )
    def test_hash(self, offset_name: str, request: Any) -> None:
        offset = request.getfixturevalue(offset_name)
        assert offset == offset

    def test_add_datetime(
        self,
        dt: datetime,
        offset1: BusinessHour,
        offset2: BusinessHour,
        offset3: BusinessHour,
        offset4: BusinessHour,
        offset8: BusinessHour,
        offset9: BusinessHour,
        offset10: BusinessHour,
    ) -> None:
        assert offset1 + dt == datetime(2014, 7, 1, 11)
        assert offset2 + dt == datetime(2014, 7, 1, 13)
        assert offset3 + dt == datetime(2014, 6, 30, 17)
        assert offset4 + dt == datetime(2014, 6, 30, 14)
        assert offset8 + dt == datetime(2014, 7, 1, 11)
        assert offset9 + dt == datetime(2014, 7, 1, 22)
        assert offset10 + dt == datetime(2014, 7, 1, 1)

    def test_sub(self, dt: datetime, offset2: BusinessHour, _offset: type[BusinessHour]) -> None:
        off = offset2
        msg = "Cannot subtract datetime from offset"
        with pytest.raises(TypeError, match=msg):
            off - dt
        assert 2 * off - off == off

        assert dt - offset2 == dt + _offset(-3)

    def test_multiply_by_zero(self, dt: datetime, offset1: BusinessHour, offset2: BusinessHour) -> None:
        assert dt - 0 * offset1 == dt
        assert dt + 0 * offset1 == dt
        assert dt - 0 * offset2 == dt
        assert dt + 0 * offset2 == dt

    def testRollback1(
        self,
        dt: datetime,
        _offset: type[BusinessHour],
        offset1: BusinessHour,
        offset2: BusinessHour,
        offset3: BusinessHour,
        offset4: BusinessHour,
        offset5: BusinessHour,
        offset6: BusinessHour,
        offset7: BusinessHour,
        offset8: BusinessHour,
        offset9: BusinessHour,
        offset10: BusinessHour,
    ) -> None:
        assert offset1.rollback(dt) == dt
        assert offset2.rollback(dt) == dt
        assert offset3.rollback(dt) == dt
        assert offset4.rollback(dt) == dt
        assert offset5.rollback(dt) == datetime(2014, 6, 30, 14, 30)
        assert offset6.rollback(dt) == datetime(2014, 7, 1, 5, 0)
        assert offset7.rollback(dt) == datetime(2014, 7, 1, 6, 30)
        assert offset8.rollback(dt) == dt
        assert offset9.rollback(dt) == dt
        assert offset10.rollback(dt) == datetime(2014, 7, 1, 2)

        datet = datetime(2014, 7, 1, 0)
        assert offset1.rollback(datet) == datetime(2014, 6, 30, 17)
        assert offset2.rollback(datet) == datetime(2014, 6, 30, 17)
        assert offset3.rollback(datet) == datetime(2014, 6, 30, 17)
        assert offset4.rollback(datet) == datetime(2014, 6, 30, 17)
        assert offset5.rollback(datet) == datetime(2014, 6, 30, 14, 30)
        assert offset6.rollback(datet) == datet
        assert offset7.rollback(datet) == datet
        assert offset8.rollback(datet) == datetime(2014, 6, 30, 17)
        assert offset9.rollback(datet) == datet
        assert offset10.rollback(datet) == datet

        assert _offset(5).rollback(dt) == dt

    def testRollback2(self, _offset: type[BusinessHour]) -> None:
        assert _offset(-3).rollback(datetime(2014, 7, 5, 15, 0)) == datetime(
            2014, 7, 4, 17, 0
        )

    def testRollforward1(
        self,
        dt: datetime,
        _offset: type[BusinessHour],
        offset1: BusinessHour,
        offset2: BusinessHour,
        offset3: BusinessHour,
        offset4: BusinessHour,
        offset5: BusinessHour,
        offset6: BusinessHour,
        offset7: BusinessHour,
        offset8: BusinessHour,
        offset9: BusinessHour,
        offset10: BusinessHour,
    ) -> None:
        assert offset1.rollforward(dt) == dt
        assert offset2.rollforward(dt) == dt
        assert offset3.rollforward(dt) == dt
        assert offset4.rollforward(dt) == dt
        assert offset5.rollforward(dt) == datetime(2014, 7, 1, 11, 0)
        assert offset6.rollforward(dt) == datetime(2014, 7, 1, 20, 0)
        assert offset7.rollforward(dt) == datetime(2014, 7, 1, 21, 30)
        assert offset8.rollforward(dt) == dt
        assert offset9.rollforward(dt) == dt
        assert offset10.rollforward(dt) == datetime(2014, 7, 1, 13)

        datet = datetime(2014, 7, 1, 0)
        assert offset1.rollforward(datet) == datetime(2014, 7, 1, 9)
        assert offset2.rollforward(datet) == datetime(2014, 7, 1, 9)
        assert offset3.rollforward(datet) == datetime(2014, 7, 1, 9)
        assert offset4.rollforward(datet) == datetime(2014, 7, 1, 9)
        assert offset5.rollforward(datet) == datetime(2014, 7, 1, 11)
        assert offset6.rollforward(datet) == datet
        assert offset7.rollforward(datet) == datet
        assert offset8.rollforward(datet) == datetime(2014, 7, 1, 9)
        assert offset9.rollforward(datet) == datet
        assert offset10.rollforward(datet) == datet

        assert _offset(5).rollforward(dt) == dt

    def testRollforward2(self, _offset: type[BusinessHour]) -> None:
        assert _offset(-3).rollforward(datetime(2014, 7, 5, 16, 0)) == datetime(
            2014, 7, 7, 9
        )

    def test_roll_date_object(self) -> None:
        offset = BusinessHour()

        dt = datetime(2014, 7, 6, 15, 0)

        result = offset.rollback(dt)
        assert result == datetime(2014, 7, 4, 17)

        result = offset.rollforward(dt)
        assert result == datetime(2014, 7, 7, 9)

    normalize_cases: List[Tuple[BusinessHour, Dict[datetime, datetime]]] = []
    normalize_cases.append(
        (
            BusinessHour(normalize=True),
            {
                datetime(2014, 7, 1, 8): datetime(2014, 7, 1),
                datetime(2014, 7, 1, 17): datetime(2014, 7, 2),
                datetime(2014, 7, 1, 16): datetime(2014, 7, 2),
                datetime(2014, 7, 1, 23): datetime(2014, 7, 2),
                datetime(2014, 7, 1, 0): datetime(2014, 7, 1),
                datetime(2014, 7, 4, 15): datetime(2014, 7, 4),
                datetime(2014, 7, 4, 15, 59): datetime(2014, 7, 4),
                datetime(2014, 7, 4, 16, 30): datetime(2014, 7, 7),
                datetime(2014, 7, 5, 23): datetime(2014, 7, 7),
                datetime(2014, 7, 6, 10): datetime(2014, 7, 7),
            },
        )
    )

    normalize_cases.append(
        (
            BusinessHour(-1, normalize=True),
            {
                datetime(2014, 7, 1, 8): datetime(2014, 6, 30),
                datetime(2014, 7, 1, 17): datetime(2014, 7, 1),
                datetime(2014, 7, 1, 16): datetime(2014, 7, 1),
                datetime(2014, 7, 1, 10): datetime(2014, 6, 30),
                datetime(2014, 7, 1, 0): datetime(2014, 6, 30),
                datetime(2014, 7, 7, 10): datetime(2014, 7, 4),
                datetime(2014, 7, 7, 10, 1): datetime(2014, 7, 7),
                datetime(2014, 7, 5, 23): datetime(2014, 7, 4),
                datetime(2014, 7, 6, 10): datetime(2014, 7, 4),
            },
        )
    )

    normalize_cases.append(
        (
            BusinessHour(1, normalize=True, start="17:00", end="04:00"),
            {
                datetime(2014,