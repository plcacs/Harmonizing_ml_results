#!/usr/bin/env python3
"""
Tests for offsets.BusinessHour
"""
from __future__ import annotations
from datetime import datetime, time as dt_time
from typing import Any, Callable, Dict, List, Tuple
import pytest
from pandas._libs.tslibs import Timedelta, Timestamp
from pandas._libs.tslibs.offsets import BDay, BusinessHour, Nano
from pandas import DatetimeIndex, _testing as tm, date_range
from pandas.tests.tseries.offsets.common import assert_offset_equal


@pytest.fixture
def dt() -> datetime:
    return datetime(2014, 7, 1, 10, 0)


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
    return BusinessHour(start='20:00', end='05:00')


@pytest.fixture
def offset7() -> BusinessHour:
    return BusinessHour(n=-2, start=dt_time(21, 30), end=dt_time(6, 30))


@pytest.fixture
def offset8() -> BusinessHour:
    return BusinessHour(start=['09:00', '13:00'], end=['12:00', '17:00'])


@pytest.fixture
def offset9() -> BusinessHour:
    return BusinessHour(n=3, start=['09:00', '22:00'], end=['13:00', '03:00'])


@pytest.fixture
def offset10() -> BusinessHour:
    return BusinessHour(n=-1, start=['23:00', '13:00'], end=['02:00', '17:00'])


class TestBusinessHour:
    @pytest.mark.parametrize(
        'start,end,match',
        [
            (dt_time(11, 0, 5), '17:00', 'time data must be specified only with hour and minute'),
            ('AAA', '17:00', "time data must match '%H:%M' format"),
            ('14:00:05', '17:00', "time data must match '%H:%M' format"),
            ([], '17:00', 'Must include at least 1 start time'),
            ('09:00', [], 'Must include at least 1 end time'),
            (['09:00', '11:00'], '17:00', 'number of starting time and ending time must be the same'),
            (['09:00', '11:00'], ['10:00'], 'number of starting time and ending time must be the same'),
            (
                ['09:00', '11:00'],
                ['12:00', '20:00'],
                'invalid starting and ending time\\(s\\): opening hours should not touch or overlap with one another',
            ),
            (
                ['12:00', '20:00'],
                ['09:00', '11:00'],
                'invalid starting and ending time\\(s\\): opening hours should not touch or overlap with one another',
            ),
        ],
    )
    def test_constructor_errors(self, start: Any, end: Any, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            BusinessHour(start=start, end=end)

    def test_different_normalize_equals(self, _offset: type[BusinessHour]) -> None:
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
        assert repr(offset1) == '<BusinessHour: bh=09:00-17:00>'
        assert repr(offset2) == '<3 * BusinessHours: bh=09:00-17:00>'
        assert repr(offset3) == '<-1 * BusinessHour: bh=09:00-17:00>'
        assert repr(offset4) == '<-4 * BusinessHours: bh=09:00-17:00>'
        assert repr(offset5) == '<BusinessHour: bh=11:00-14:30>'
        assert repr(offset6) == '<BusinessHour: bh=20:00-05:00>'
        assert repr(offset7) == '<-2 * BusinessHours: bh=21:30-06:30>'
        assert repr(offset8) == '<BusinessHour: bh=09:00-12:00,13:00-17:00>'
        assert repr(offset9) == '<3 * BusinessHours: bh=09:00-13:00,22:00-03:00>'
        assert repr(offset10) == '<-1 * BusinessHour: bh=13:00-17:00,23:00-02:00>'

    def test_with_offset(self, dt: datetime) -> None:
        expected: Timestamp = Timestamp('2014-07-01 13:00')
        assert dt + BusinessHour() * 3 == expected
        assert dt + BusinessHour(n=3) == expected

    @pytest.mark.parametrize('offset_name', ['offset1', 'offset2', 'offset3', 'offset4', 'offset8', 'offset9', 'offset10'])
    def test_eq_attribute(self, offset_name: str, request: pytest.FixtureRequest) -> None:
        offset: BusinessHour = request.getfixturevalue(offset_name)
        assert offset == offset

    @pytest.mark.parametrize(
        'offset1,offset2',
        [
            (BusinessHour(start='09:00'), BusinessHour()),
            (
                BusinessHour(start=['23:00', '13:00'], end=['12:00', '17:00']),
                BusinessHour(start=['13:00', '23:00'], end=['17:00', '12:00']),
            ),
        ],
    )
    def test_eq(self, offset1: BusinessHour, offset2: BusinessHour) -> None:
        assert offset1 == offset2

    @pytest.mark.parametrize(
        'offset1,offset2',
        [
            (BusinessHour(), BusinessHour(-1)),
            (BusinessHour(start='09:00'), BusinessHour(start='09:01')),
            (
                BusinessHour(start='09:00', end='17:00'),
                BusinessHour(start='17:00', end='09:01'),
            ),
            (
                BusinessHour(start=['13:00', '23:00'], end=['18:00', '07:00']),
                BusinessHour(start=['13:00', '23:00'], end=['17:00', '12:00']),
            ),
        ],
    )
    def test_neq(self, offset1: BusinessHour, offset2: BusinessHour) -> None:
        assert offset1 != offset2

    @pytest.mark.parametrize('offset_name', ['offset1', 'offset2', 'offset3', 'offset4', 'offset8', 'offset9', 'offset10'])
    def test_hash(self, offset_name: str, request: pytest.FixtureRequest) -> None:
        offset: BusinessHour = request.getfixturevalue(offset_name)
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
        off: BusinessHour = offset2
        msg: str = 'Cannot subtract datetime from offset'
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
        datet: datetime = datetime(2014, 7, 1, 0)
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
        assert _offset(-3).rollback(datetime(2014, 7, 5, 15, 0)) == datetime(2014, 7, 4, 17, 0)

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
        datet: datetime = datetime(2014, 7, 1, 0)
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
        assert _offset(-3).rollforward(datetime(2014, 7, 5, 16, 0)) == datetime(2014, 7, 7, 9)

    def test_roll_date_object(self) -> None:
        offset: BusinessHour = BusinessHour()
        dt_obj: datetime = datetime(2014, 7, 6, 15, 0)
        result: datetime = offset.rollback(dt_obj)
        assert result == datetime(2014, 7, 4, 17)
        result = offset.rollforward(dt_obj)
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
            BusinessHour(1, normalize=True, start='17:00', end='04:00'),
            {
                datetime(2014, 7, 1, 8): datetime(2014, 7, 1),
                datetime(2014, 7, 1, 17): datetime(2014, 7, 1),
                datetime(2014, 7, 1, 23): datetime(2014, 7, 2),
                datetime(2014, 7, 2, 2): datetime(2014, 7, 2),
                datetime(2014, 7, 2, 3): datetime(2014, 7, 2),
                datetime(2014, 7, 4, 23): datetime(2014, 7, 5),
                datetime(2014, 7, 5, 2): datetime(2014, 7, 5),
                datetime(2014, 7, 7, 2): datetime(2014, 7, 7),
                datetime(2014, 7, 7, 17): datetime(2014, 7, 7),
            },
        )
    )

    @pytest.mark.parametrize('case', normalize_cases)
    def test_normalize(self, case: Tuple[BusinessHour, Dict[datetime, datetime]]) -> None:
        offset, cases = case
        for dt_val, expected in cases.items():
            assert offset._apply(dt_val) == expected

    on_offset_cases: List[Tuple[BusinessHour, Dict[datetime, bool]]] = []
    on_offset_cases.append(
        (
            BusinessHour(),
            {
                datetime(2014, 7, 1, 9): True,
                datetime(2014, 7, 1, 8, 59): False,
                datetime(2014, 7, 1, 8): False,
                datetime(2014, 7, 1, 17): True,
                datetime(2014, 7, 1, 17, 1): False,
                datetime(2014, 7, 1, 18): False,
                datetime(2014, 7, 5, 9): False,
                datetime(2014, 7, 6, 12): False,
            },
        )
    )
    on_offset_cases.append(
        (
            BusinessHour(start='10:00', end='15:00'),
            {
                datetime(2014, 7, 1, 9): False,
                datetime(2014, 7, 1, 10): True,
                datetime(2014, 7, 1, 15): True,
                datetime(2014, 7, 1, 15, 1): False,
                datetime(2014, 7, 5, 12): False,
                datetime(2014, 7, 6, 12): False,
            },
        )
    )
    on_offset_cases.append(
        (
            BusinessHour(start='19:00', end='05:00'),
            {
                datetime(2014, 7, 1, 9, 0): False,
                datetime(2014, 7, 1, 10, 0): False,
                datetime(2014, 7, 1, 15): False,
                datetime(2014, 7, 1, 15, 1): False,
                datetime(2014, 7, 5, 12, 0): False,
                datetime(2014, 7, 6, 12, 0): False,
                datetime(2014, 7, 1, 19, 0): True,
                datetime(2014, 7, 2, 0, 0): True,
                datetime(2014, 7, 4, 23): True,
                datetime(2014, 7, 5, 1): True,
                datetime(2014, 7, 5, 5, 0): True,
                datetime(2014, 7, 6, 23, 0): False,
                datetime(2014, 7, 7, 3, 0): False,
            },
        )
    )
    on_offset_cases.append(
        (
            BusinessHour(start=['09:00', '13:00'], end=['12:00', '17:00']),
            {
                datetime(2014, 7, 1, 9): True,
                datetime(2014, 7, 1, 8, 59): False,
                datetime(2014, 7, 1, 8): False,
                datetime(2014, 7, 1, 17): True,
                datetime(2014, 7, 1, 17, 1): False,
                datetime(2014, 7, 1, 18): False,
                datetime(2014, 7, 5, 9): False,
                datetime(2014, 7, 6, 12): False,
                datetime(2014, 7, 1, 12, 30): False,
            },
        )
    )
    on_offset_cases.append(
        (
            BusinessHour(start=['19:00', '23:00'], end=['21:00', '05:00']),
            {
                datetime(2014, 7, 1, 9, 0): False,
                datetime(2014, 7, 1, 10, 0): False,
                datetime(2014, 7, 1, 15): False,
                datetime(2014, 7, 1, 15, 1): False,
                datetime(2014, 7, 5, 12, 0): False,
                datetime(2014, 7, 6, 12, 0): False,
                datetime(2014, 7, 1, 19, 0): True,
                datetime(2014, 7, 2, 0, 0): True,
                datetime(2014, 7, 4, 23): True,
                datetime(2014, 7, 5, 1): True,
                datetime(2014, 7, 5, 5, 0): True,
                datetime(2014, 7, 6, 23, 0): False,
                datetime(2014, 7, 7, 3, 0): False,
                datetime(2014, 7, 4, 22): False,
            },
        )
    )

    @pytest.mark.parametrize('case', on_offset_cases)
    def test_is_on_offset(self, case: Tuple[BusinessHour, Dict[datetime, bool]]) -> None:
        offset, cases = case
        for dt_val, expected in cases.items():
            assert offset.is_on_offset(dt_val) == expected

    apply_cases: List[Tuple[BusinessHour, Dict[datetime, datetime]]] = [
        (
            BusinessHour(),
            {
                datetime(2014, 7, 1, 11): datetime(2014, 7, 1, 12),
                datetime(2014, 7, 1, 13): datetime(2014, 7, 1, 14),
                datetime(2014, 7, 1, 15): datetime(2014, 7, 1, 16),
                datetime(2014, 7, 1, 19): datetime(2014, 7, 2, 10),
                datetime(2014, 7, 1, 16): datetime(2014, 7, 2, 9),
                datetime(2014, 7, 1, 16, 30, 15): datetime(2014, 7, 2, 9, 30, 15),
                datetime(2014, 7, 1, 17): datetime(2014, 7, 2, 10),
                datetime(2014, 7, 2, 11): datetime(2014, 7, 2, 12),
                datetime(2014, 7, 2, 8): datetime(2014, 7, 2, 10),
                datetime(2014, 7, 2, 19): datetime(2014, 7, 3, 10),
                datetime(2014, 7, 2, 23): datetime(2014, 7, 3, 10),
                datetime(2014, 7, 3, 0): datetime(2014, 7, 3, 10),
                datetime(2014, 7, 5, 15): datetime(2014, 7, 7, 10),
                datetime(2014, 7, 4, 17): datetime(2014, 7, 7, 10),
                datetime(2014, 7, 4, 16, 30): datetime(2014, 7, 7, 9, 30),
                datetime(2014, 7, 4, 16, 30, 30): datetime(2014, 7, 7, 9, 30, 30),
            },
        ),
        (
            BusinessHour(4),
            {
                datetime(2014, 7, 1, 11): datetime(2014, 7, 1, 15),
                datetime(2014, 7, 1, 13): datetime(2014, 7, 2, 9),
                datetime(2014, 7, 1, 15): datetime(2014, 7, 2, 11),
                datetime(2014, 7, 1, 16): datetime(2014, 7, 2, 12),
                datetime(2014, 7, 1, 17): datetime(2014, 7, 2, 13),
                datetime(2014, 7, 2, 11): datetime(2014, 7, 2, 15),
                datetime(2014, 7, 2, 8): datetime(2014, 7, 2, 13),
                datetime(2014, 7, 2, 19): datetime(2014, 7, 3, 13),
                datetime(2014, 7, 2, 23): datetime(2014, 7, 3, 13),
                datetime(2014, 7, 3, 0): datetime(2014, 7, 3, 13),
                datetime(2014, 7, 5, 15): datetime(2014, 7, 7, 13),
                datetime(2014, 7, 4, 17): datetime(2014, 7, 7, 13),
                datetime(2014, 7, 4, 16, 30): datetime(2014, 7, 7, 12, 30),
                datetime(2014, 7, 4, 16, 30, 30): datetime(2014, 7, 7, 12, 30, 30),
            },
        ),
        (
            BusinessHour(-1),
            {
                datetime(2014, 7, 1, 11): datetime(2014, 7, 1, 10),
                datetime(2014, 7, 1, 13): datetime(2014, 7, 1, 12),
                datetime(2014, 7, 1, 15): datetime(2014, 7, 1, 14),
                datetime(2014, 7, 1, 16): datetime(2014, 7, 1, 15),
                datetime(2014, 7, 1, 10): datetime(2014, 6, 30, 17),
                datetime(2014, 7, 1, 16, 30, 15): datetime(2014, 7, 1, 15, 30, 15),
                datetime(2014, 7, 1, 9, 30, 15): datetime(2014, 6, 30, 16, 30, 15),
                datetime(2014, 7, 1, 17): datetime(2014, 7, 1, 16),
                datetime(2014, 7, 1, 5): datetime(2014, 6, 30, 16),
                datetime(2014, 7, 2, 11): datetime(2014, 7, 2, 10),
                datetime(2014, 7, 2, 8): datetime(2014, 7, 1, 16),
                datetime(2014, 7, 2, 19): datetime(2014, 7, 2, 16),
                datetime(2014, 7, 2, 23): datetime(2014, 7, 2, 16),
                datetime(2014, 7, 3, 0): datetime(2014, 7, 2, 16),
                datetime(2014, 7, 5, 15): datetime(2014, 7, 4, 16),
                datetime(2014, 7, 7, 9): datetime(2014, 7, 4, 16),
                datetime(2014, 7, 7, 9, 30): datetime(2014, 7, 4, 16, 30),
                datetime(2014, 7, 7, 9, 30, 30): datetime(2014, 7, 4, 16, 30, 30),
            },
        ),
        (
            BusinessHour(-4),
            {
                datetime(2014, 7, 1, 11): datetime(2014, 6, 30, 15),
                datetime(2014, 7, 1, 13): datetime(2014, 6, 30, 17),
                datetime(2014, 7, 1, 15): datetime(2014, 7, 1, 11),
                datetime(2014, 7, 1, 16): datetime(2014, 7, 1, 12),
                datetime(2014, 7, 1, 17): datetime(2014, 7, 1, 13),
                datetime(2014, 7, 2, 11): datetime(2014, 7, 1, 15),
                datetime(2014, 7, 2, 8): datetime(2014, 7, 1, 13),
                datetime(2014, 7, 2, 19): datetime(2014, 7, 2, 13),
                datetime(2014, 7, 2, 23): datetime(2014, 7, 2, 13),
                datetime(2014, 7, 3, 0): datetime(2014, 7, 2, 13),
                datetime(2014, 7, 5, 15): datetime(2014, 7, 4, 13),
                datetime(2014, 7, 4, 18): datetime(2014, 7, 4, 13),
                datetime(2014, 7, 7, 9, 30): datetime(2014, 7, 4, 13, 30),
                datetime(2014, 7, 7, 9, 30, 30): datetime(2014, 7, 4, 13, 30, 30),
            },
        ),
        (
            BusinessHour(start='13:00', end='16:00'),
            {
                datetime(2014, 7, 1, 11): datetime(2014, 7, 1, 14),
                datetime(2014, 7, 1, 13): datetime(2014, 7, 1, 14),
                datetime(2014, 7, 1, 15): datetime(2014, 7, 2, 13),
                datetime(2014, 7, 1, 19): datetime(2014, 7, 2, 14),
                datetime(2014, 7, 1, 16): datetime(2014, 7, 2, 14),
                datetime(2014, 7, 1, 15, 30, 15): datetime(2014, 7, 2, 13, 30, 15),
                datetime(2014, 7, 5, 15): datetime(2014, 7, 7, 14),
                datetime(2014, 7, 4, 17): datetime(2014, 7, 7, 14),
            },
        ),
        (
            BusinessHour(n=2, start='13:00', end='16:00'),
            {
                datetime(2014, 7, 1, 17): datetime(2014, 7, 2, 15),
                datetime(2014, 7, 2, 14): datetime(2014, 7, 3, 13),
                datetime(2014, 7, 2, 8): datetime(2014, 7, 2, 15),
                datetime(2014, 7, 2, 19): datetime(2014, 7, 3, 15),
                datetime(2014, 7, 2, 14, 30): datetime(2014, 7, 3, 13, 30),
                datetime(2014, 7, 3, 0): datetime(2014, 7, 3, 15),
                datetime(2014, 7, 5, 15): datetime(2014, 7, 7, 15),
                datetime(2014, 7, 4, 17): datetime(2014, 7, 7, 15),
                datetime(2014, 7, 4, 14, 30): datetime(2014, 7, 7, 13, 30),
                datetime(2014, 7, 4, 14, 30, 30): datetime(2014, 7, 7, 13, 30, 30),
            },
        ),
        (
            BusinessHour(n=-1, start='13:00', end='16:00'),
            {
                datetime(2014, 7, 2, 11): datetime(2014, 7, 1, 15),
                datetime(2014, 7, 2, 13): datetime(2014, 7, 1, 15),
                datetime(2014, 7, 2, 14): datetime(2014, 7, 1, 16),
                datetime(2014, 7, 2, 15): datetime(2014, 7, 2, 14),
                datetime(2014, 7, 2, 19): datetime(2014, 7, 2, 15),
                datetime(2014, 7, 2, 16): datetime(2014, 7, 2, 15),
                datetime(2014, 7, 2, 13, 30, 15): datetime(2014, 7, 1, 15, 30, 15),
                datetime(2014, 7, 5, 15): datetime(2014, 7, 4, 15),
                datetime(2014, 7, 7, 11): datetime(2014, 7, 4, 15),
            },
        ),
        (
            BusinessHour(n=-3, start='10:00', end='16:00'),
            {
                datetime(2014, 7, 1, 17): datetime(2014, 7, 1, 13),
                datetime(2014, 7, 2, 14): datetime(2014, 7, 2, 11),
                datetime(2014, 7, 2, 8): datetime(2014, 7, 1, 13),
                datetime(2014, 7, 2, 13): datetime(2014, 7, 1, 16),
                datetime(2014, 7, 2, 19): datetime(2014, 7, 2, 13),
                datetime(2014, 7, 2, 11, 30): datetime(2014, 7, 1, 14, 30),
                datetime(2014, 7, 3, 0): datetime(2014, 7, 2, 13),
                datetime(2014, 7, 4, 10): datetime(2014, 7, 3, 13),
                datetime(2014, 7, 5, 15): datetime(2014, 7, 4, 13),
                datetime(2014, 7, 4, 16): datetime(2014, 7, 4, 13),
                datetime(2014, 7, 4, 12, 30): datetime(2014, 7, 3, 15, 30),
                datetime(2014, 7, 4, 12, 30, 30): datetime(2014, 7, 3, 15, 30, 30),
            },
        ),
        (
            BusinessHour(start='19:00', end='05:00'),
            {
                datetime(2014, 7, 1, 17): datetime(2014, 7, 1, 20),
                datetime(2014, 7, 2, 14): datetime(2014, 7, 2, 20),
                datetime(2014, 7, 2, 8): datetime(2014, 7, 2, 20),
                datetime(2014, 7, 2, 13): datetime(2014, 7, 2, 20),
                datetime(2014, 7, 2, 19): datetime(2014, 7, 2, 20),
                datetime(2014, 7, 2, 4, 30): datetime(2014, 7, 2, 19, 30),
                datetime(2014, 7, 3, 0): datetime(2014, 7, 3, 1),
                datetime(2014, 7, 4, 10): datetime(2014, 7, 4, 20),
                datetime(2014, 7, 4, 23): datetime(2014, 7, 5, 0),
                datetime(2014, 7, 5, 0): datetime(2014, 7, 5, 1),
                datetime(2014, 7, 5, 4): datetime(2014, 7, 7, 19),
                datetime(2014, 7, 5, 4, 30): datetime(2014, 7, 7, 19, 30),
                datetime(2014, 7, 5, 4, 30, 30): datetime(2014, 7, 7, 19, 30, 30),
            },
        ),
        (
            BusinessHour(n=-1, start='19:00', end='05:00'),
            {
                datetime(2014, 7, 1, 17): datetime(2014, 7, 1, 4),
                datetime(2014, 7, 2, 14): datetime(2014, 7, 2, 4),
                datetime(2014, 7, 2, 8): datetime(2014, 7, 2, 4),
                datetime(2014, 7, 2, 13): datetime(2014, 7, 2, 4),
                datetime(2014, 7, 2, 20): datetime(2014, 7, 2, 5),
                datetime(2014, 7, 2, 19): datetime(2014, 7, 2, 4),
                datetime(2014, 7, 2, 19, 30): datetime(2014, 7, 2, 4, 30),
                datetime(2014, 7, 3, 0): datetime(2014, 7, 2, 23),
                datetime(2014, 7, 3, 6): datetime(2014, 7, 3, 4),
                datetime(2014, 7, 4, 23): datetime(2014, 7, 4, 22),
                datetime(2014, 7, 5, 0): datetime(2014, 7, 4, 23),
                datetime(2014, 7, 5, 4): datetime(2014, 7, 5, 3),
                datetime(2014, 7, 7, 19, 30): datetime(2014, 7, 5, 4, 30),
                datetime(2014, 7, 7, 19, 30, 30): datetime(2014, 7, 5, 4, 30, 30),
            },
        ),
        (
            BusinessHour(n=4, start='00:00', end='23:00'),
            {
                datetime(2014, 7, 3, 22): datetime(2014, 7, 4, 3),
                datetime(2014, 7, 4, 22): datetime(2014, 7, 7, 3),
                datetime(2014, 7, 3, 22, 30): datetime(2014, 7, 4, 3, 30),
                datetime(2014, 7, 3, 22, 20): datetime(2014, 7, 4, 3, 20),
                datetime(2014, 7, 4, 22, 30, 30): datetime(2014, 7, 7, 3, 30, 30),
                datetime(2014, 7, 4, 22, 30, 20): datetime(2014, 7, 7, 3, 30, 20),
            },
        ),
        (
            BusinessHour(n=-4, start='00:00', end='23:00'),
            {
                datetime(2014, 7, 4, 3): datetime(2014, 7, 3, 22),
                datetime(2014, 7, 7, 3): datetime(2014, 7, 4, 22),
                datetime(2014, 7, 4, 3, 30): datetime(2014, 7, 3, 22, 30),
                datetime(2014, 7, 4, 3, 20): datetime(2014, 7, 3, 22, 20),
                datetime(2014, 7, 7, 3, 30, 30): datetime(2014, 7, 4, 22, 30, 30),
                datetime(2014, 7, 7, 3, 30, 20): datetime(2014, 7, 4, 22, 30, 20),
            },
        ),
        (
            BusinessHour(start=['09:00', '14:00'], end=['12:00', '18:00']),
            {
                datetime(2014, 7, 1, 11): datetime(2014, 7, 1, 14),
                datetime(2014, 7, 1, 15): datetime(2014, 7, 1, 16),
                datetime(2014, 7, 1, 19): datetime(2014, 7, 2, 10),
                datetime(2014, 7, 1, 16): datetime(2014, 7, 1, 17),
                datetime(2014, 7, 1, 16, 30, 15): datetime(2014, 7, 1, 17, 30, 15),
                datetime(2014, 7, 1, 17): datetime(2014, 7, 2, 9),
                datetime(2014, 7, 2, 11): datetime(2014, 7, 2, 14),
                datetime(2014, 7, 1, 13): datetime(2014, 7, 1, 15),
                datetime(2014, 7, 2, 8): datetime(2014, 7, 2, 10),
                datetime(2014, 7, 2, 19): datetime(2014, 7, 3, 10),
                datetime(2014, 7, 2, 23): datetime(2014, 7, 3, 10),
                datetime(2014, 7, 3, 0): datetime(2014, 7, 3, 10),
                datetime(2014, 7, 5, 15): datetime(2014, 7, 7, 10),
                datetime(2014, 7, 4, 17): datetime(2014, 7, 7, 9),
                datetime(2014, 7, 4, 17, 30): datetime(2014, 7, 7, 9, 30),
                datetime(2014, 7, 4, 17, 30, 30): datetime(2014, 7, 7, 9, 30, 30),
            },
        ),
        (
            BusinessHour(n=4, start=['09:00', '14:00'], end=['12:00', '18:00']),
            {
                datetime(2014, 7, 1, 11): datetime(2014, 7, 1, 17),
                datetime(2014, 7, 1, 13): datetime(2014, 7, 2, 9),
                datetime(2014, 7, 1, 15): datetime(2014, 7, 2, 10),
                datetime(2014, 7, 1, 16): datetime(2014, 7, 2, 11),
                datetime(2014, 7, 1, 17): datetime(2014, 7, 2, 14),
                datetime(2014, 7, 2, 11): datetime(2014, 7, 2, 17),
                datetime(2014, 7, 2, 8): datetime(2014, 7, 2, 15),
                datetime(2014, 7, 2, 19): datetime(2014, 7, 3, 15),
                datetime(2014, 7, 2, 23): datetime(2014, 7, 3, 15),
                datetime(2014, 7, 3, 0): datetime(2014, 7, 3, 15),
                datetime(2014, 7, 5, 15): datetime(2014, 7, 7, 15),
                datetime(2014, 7, 4, 17): datetime(2014, 7, 7, 14),
                datetime(2014, 7, 4, 16, 30): datetime(2014, 7, 7, 11, 30),
                datetime(2014, 7, 4, 16, 30, 30): datetime(2014, 7, 7, 11, 30, 30),
            },
        ),
        (
            BusinessHour(n=-4, start=['09:00', '14:00'], end=['12:00', '18:00']),
            {
                datetime(2014, 7, 1, 11): datetime(2014, 6, 30, 16),
                datetime(2014, 7, 1, 13): datetime(2014, 6, 30, 17),
                datetime(2014, 7, 1, 15): datetime(2014, 6, 30, 18),
                datetime(2014, 7, 1, 16): datetime(2014, 7, 1, 10),
                datetime(2014, 7, 1, 17): datetime(2014, 7, 1, 11),
                datetime(2014, 7, 2, 11): datetime(2014, 7, 1, 16),
                datetime(2014, 7, 2, 8): datetime(2014, 7, 1, 12),
                datetime(2014, 7, 2, 19): datetime(2014, 7, 2, 12),
                datetime(2014, 7, 2, 23): datetime(2014, 7, 2, 12),
                datetime(2014, 7, 3, 0): datetime(2014, 7, 2, 12),
                datetime(2014, 7, 5, 15): datetime(2014, 7, 4, 12),
                datetime(2014, 7, 4, 18): datetime(2014, 7, 4, 12),
                datetime(2014, 7, 7, 9, 30): datetime(2014, 7, 4, 14, 30),
                datetime(2014, 7, 7, 9, 30, 30): datetime(2014, 7, 4, 14, 30, 30),
            },
        ),
        (
            BusinessHour(n=-1, start=['19:00', '03:00'], end=['01:00', '04:00']),
            {
                datetime(2014, 7, 1, 17): datetime(2014, 7, 1, 4),
                datetime(2014, 7, 2, 14): datetime(2014, 7, 2, 4),
                datetime(2014, 7, 2, 8): datetime(2014, 7, 2, 4),
                datetime(2014, 7, 2, 13): datetime(2014, 7, 2, 4),
                datetime(2014, 7, 2, 20): datetime(2014, 7, 2, 5),
                datetime(2014, 7, 2, 19): datetime(2014, 7, 2, 4),
                datetime(2014, 7, 2, 4): datetime(2014, 7, 2, 1),
                datetime(2014, 7, 2, 19, 30): datetime(2014, 7, 2, 4, 30),
                datetime(2014, 7, 3, 0): datetime(2014, 7, 2, 23),
                datetime(2014, 7, 3, 6): datetime(2014, 7, 3, 4),
                datetime(2014, 7, 4, 23): datetime(2014, 7, 4, 22),
                datetime(2014, 7, 5, 0): datetime(2014, 7, 4, 23),
                datetime(2014, 7, 5, 4): datetime(2014, 7, 5, 0),
                datetime(2014, 7, 7, 3, 30): datetime(2014, 7, 5, 0, 30),
                datetime(2014, 7, 7, 19, 30): datetime(2014, 7, 7, 4, 30),
                datetime(2014, 7, 7, 19, 30, 30): datetime(2014, 7, 7, 4, 30, 30),
            },
        ),
    ]

    @pytest.mark.parametrize('case', apply_cases)
    def test_apply(self, case: Tuple[BusinessHour, Dict[datetime, datetime]]) -> None:
        offset, cases = case
        for base, expected in cases.items():
            assert_offset_equal(offset, base, expected)

    apply_large_n_cases: List[Tuple[BusinessHour, Dict[datetime, datetime]]] = [
        (
            BusinessHour(40),
            {
                datetime(2014, 7, 1, 11): datetime(2014, 7, 8, 11),
                datetime(2014, 7, 1, 13): datetime(2014, 7, 8, 13),
                datetime(2014, 7, 1, 15): datetime(2014, 7, 8, 15),
                datetime(2014, 7, 1, 16): datetime(2014, 7, 8, 16),
                datetime(2014, 7, 1, 17): datetime(2014, 7, 9, 9),
                datetime(2014, 7, 2, 11): datetime(2014, 7, 9, 11),
                datetime(2014, 7, 2, 8): datetime(2014, 7, 9, 9),
                datetime(2014, 7, 2, 19): datetime(2014, 7, 10, 9),
                datetime(2014, 7, 2, 23): datetime(2014, 7, 10, 9),
                datetime(2014, 7, 3, 0): datetime(2014, 7, 10, 9),
                datetime(2014, 7, 5, 15): datetime(2014, 7, 14, 9),
                datetime(2014, 7, 4, 18): datetime(2014, 7, 14, 9),
                datetime(2014, 7, 7, 9, 30): datetime(2014, 7, 14, 9, 30),
                datetime(2014, 7, 7, 9, 30, 30): datetime(2014, 7, 14, 9, 30, 30),
            },
        ),
        (
            BusinessHour(-25),
            {
                datetime(2014, 7, 1, 11): datetime(2014, 6, 26, 10),
                datetime(2014, 7, 1, 13): datetime(2014, 6, 26, 12),
                datetime(2014, 7, 1, 9): datetime(2014, 6, 25, 16),
                datetime(2014, 7, 1, 10): datetime(2014, 6, 25, 17),
                datetime(2014, 7, 3, 11): datetime(2014, 6, 30, 10),
                datetime(2014, 7, 3, 8): datetime(2014, 6, 27, 16),
                datetime(2014, 7, 3, 19): datetime(2014, 6, 30, 16),
                datetime(2014, 7, 3, 23): datetime(2014, 6, 30, 16),
                datetime(2014, 7, 4, 9): datetime(2014, 6, 30, 16),
                datetime(2014, 7, 5, 15): datetime(2014, 7, 1, 16),
                datetime(2014, 7, 6, 18): datetime(2014, 7, 1, 16),
                datetime(2014, 7, 7, 9, 30): datetime(2014, 7, 1, 16, 30),
                datetime(2014, 7, 7, 10, 30, 30): datetime(2014, 7, 2, 9, 30, 30),
            },
        ),
        (
            BusinessHour(28, start='21:00', end='02:00'),
            {
                datetime(2014, 7, 1, 11): datetime(2014, 7, 9, 0),
                datetime(2014, 7, 1, 22): datetime(2014, 7, 9, 1),
                datetime(2014, 7, 1, 23): datetime(2014, 7, 9, 21),
                datetime(2014, 7, 2, 2): datetime(2014, 7, 10, 0),
                datetime(2014, 7, 3, 21): datetime(2014, 7, 11, 0),
                datetime(2014, 7, 4, 1): datetime(2014, 7, 11, 23),
                datetime(2014, 7, 4, 2): datetime(2014, 7, 12, 0),
                datetime(2014, 7, 4, 3): datetime(2014, 7, 12, 0),
                datetime(2014, 7, 5, 1): datetime(2014, 7, 14, 23),
                datetime(2014, 7, 5, 15): datetime(2014, 7, 15, 0),
                datetime(2014, 7, 6, 18): datetime(2014, 7, 15, 0),
                datetime(2014, 7, 7, 1): datetime(2014, 7, 15, 0),
                datetime(2014, 7, 7, 23, 30): datetime(2014, 7, 15, 21, 30),
            },
        ),
        (
            BusinessHour(n=-25, start=['09:00', '14:00'], end=['12:00', '19:00']),
            {
                datetime(2014, 7, 1, 11): datetime(2014, 6, 26, 10),
                datetime(2014, 7, 1, 13): datetime(2014, 6, 26, 11),
                datetime(2014, 7, 1, 9): datetime(2014, 6, 25, 18),
                datetime(2014, 7, 1, 10): datetime(2014, 6, 25, 19),
                datetime(2014, 7, 3, 11): datetime(2014, 6, 30, 10),
                datetime(2014, 7, 3, 8): datetime(2014, 6, 27, 18),
                datetime(2014, 7, 3, 19): datetime(2014, 6, 30, 18),
                datetime(2014, 7, 3, 23): datetime(2014, 6, 30, 18),
                datetime(2014, 7, 4, 9): datetime(2014, 6, 30, 18),
                datetime(2014, 7, 5, 15): datetime(2014, 7, 1, 18),
                datetime(2014, 7, 6, 18): datetime(2014, 7, 1, 18),
                datetime(2014, 7, 7, 9, 30): datetime(2014, 7, 1, 18, 30),
                datetime(2014, 7, 7, 10, 30, 30): datetime(2014, 7, 2, 9, 30, 30),
            },
        ),
        (
            BusinessHour(28, start=['21:00', '03:00'], end=['01:00', '04:00']),
            {
                datetime(2014, 7, 1, 11): datetime(2014, 7, 9, 0),
                datetime(2014, 7, 1, 22): datetime(2014, 7, 9, 3),
                datetime(2014, 7, 1, 23): datetime(2014, 7, 9, 21),
                datetime(2014, 7, 2, 2): datetime(2014, 7, 9, 23),
                datetime(2014, 7, 3, 21): datetime(2014, 7, 11, 0),
                datetime(2014, 7, 4, 1): datetime(2014, 7, 11, 23),
                datetime(2014, 7, 4, 2): datetime(2014, 7, 11, 23),
                datetime(2014, 7, 4, 3): datetime(2014, 7, 11, 23),
                datetime(2014, 7, 4, 21): datetime(2014, 7, 12, 0),
                datetime(2014, 7, 5, 0): datetime(2014, 7, 14, 22),
                datetime(2014, 7, 5, 1): datetime(2014, 7, 14, 23),
                datetime(2014, 7, 5, 15): datetime(2014, 7, 14, 23),
                datetime(2014, 7, 6, 18): datetime(2014, 7, 14, 23),
                datetime(2014, 7, 7, 1): datetime(2014, 7, 14, 23),
                datetime(2014, 7, 7, 23, 30): datetime(2014, 7, 15, 21, 30),
            },
        ),
    ]

    @pytest.mark.parametrize('case', apply_large_n_cases)
    def test_apply_large_n(self, case: Tuple[BusinessHour, Dict[datetime, datetime]]) -> None:
        offset, cases = case
        for base, expected in cases.items():
            assert_offset_equal(offset, base, expected)

    def test_apply_nanoseconds(self) -> None:
        tests: List[Tuple[BusinessHour, Dict[Timestamp, Timestamp]]] = [
            (
                BusinessHour(),
                {
                    Timestamp('2014-07-04 15:00') + Nano(5): Timestamp('2014-07-04 16:00') + Nano(5),
                    Timestamp('2014-07-04 16:00') + Nano(5): Timestamp('2014-07-07 09:00') + Nano(5),
                    Timestamp('2014-07-04 16:00') - Nano(5): Timestamp('2014-07-04 17:00') - Nano(5),
                },
            ),
            (
                BusinessHour(-1),
                {
                    Timestamp('2014-07-04 15:00') + Nano(5): Timestamp('2014-07-04 14:00') + Nano(5),
                    Timestamp('2014-07-04 10:00') + Nano(5): Timestamp('2014-07-04 09:00') + Nano(5),
                    Timestamp('2014-07-04 10:00') - Nano(5): Timestamp('2014-07-03 17:00') - Nano(5),
                },
            ),
        ]
        for offset, cases in tests:
            for base, expected in cases.items():
                assert_offset_equal(offset, base, expected)

    @pytest.mark.parametrize('td_unit', ['s', 'ms', 'us', 'ns'])
    def test_bday_ignores_timedeltas(self, unit: str, td_unit: str) -> None:
        idx: DatetimeIndex = date_range('2010/02/01', '2010/02/10', freq='12h', unit=unit)
        td = Timedelta(3, unit='h').as_unit(td_unit)
        off = BDay(offset=td)
        t1: DatetimeIndex = idx + off
        exp_unit = tm.get_finest_unit(td.unit, idx.unit)
        expected: DatetimeIndex = DatetimeIndex(
            [
                '2010-02-02 03:00:00',
                '2010-02-02 15:00:00',
                '2010-02-03 03:00:00',
                '2010-02-03 15:00:00',
                '2010-02-04 03:00:00',
                '2010-02-04 15:00:00',
                '2010-02-05 03:00:00',
                '2010-02-05 15:00:00',
                '2010-02-08 03:00:00',
                '2010-02-08 15:00:00',
                '2010-02-08 03:00:00',
                '2010-02-08 15:00:00',
                '2010-02-08 03:00:00',
                '2010-02-08 15:00:00',
                '2010-02-09 03:00:00',
                '2010-02-09 15:00:00',
                '2010-02-10 03:00:00',
                '2010-02-10 15:00:00',
                '2010-02-11 03:00:00',
            ],
            freq=None,
        ).as_unit(exp_unit)
        tm.assert_index_equal(t1, expected)
        pointwise: DatetimeIndex = DatetimeIndex([x + off for x in idx]).as_unit(exp_unit)
        tm.assert_index_equal(pointwise, expected)

    def test_add_bday_offset_nanos(self) -> None:
        idx: DatetimeIndex = date_range('2010/02/01', '2010/02/10', freq='12h', unit='ns')
        off = BDay(offset=Timedelta(3, unit='ns'))
        result: DatetimeIndex = idx + off
        expected: DatetimeIndex = DatetimeIndex([x + off for x in idx])
        tm.assert_index_equal(result, expected)


class TestOpeningTimes:
    opening_time_cases: List[
        Tuple[List[BusinessHour], Dict[datetime, Tuple[datetime, datetime]]]
    ] = []
    opening_time_cases.append(
        (
            [
                BusinessHour(),
                BusinessHour(n=2),
                BusinessHour(n=4),
                BusinessHour(end='10:00'),
                BusinessHour(n=2, end='4:00'),
                BusinessHour(n=4, end='15:00'),
            ],
            {
                datetime(2014, 7, 1, 11): (datetime(2014, 7, 2, 9), datetime(2014, 7, 1, 9)),
                datetime(2014, 7, 1, 18): (datetime(2014, 7, 2, 9), datetime(2014, 7, 1, 9)),
                datetime(2014, 7, 1, 23): (datetime(2014, 7, 2, 9), datetime(2014, 7, 1, 9)),
                datetime(2014, 7, 2, 8): (datetime(2014, 7, 2, 9), datetime(2014, 7, 1, 9)),
                datetime(2014, 7, 2, 9): (datetime(2014, 7, 2, 9), datetime(2014, 7, 2, 9)),
                datetime(2014, 7, 2, 10): (datetime(2014, 7, 3, 9), datetime(2014, 7, 2, 9)),
                datetime(2014, 7, 5, 10): (datetime(2014, 7, 7, 9), datetime(2014, 7, 4, 9)),
                datetime(2014, 7, 4, 10): (datetime(2014, 7, 7, 9), datetime(2014, 7, 4, 9)),
                datetime(2014, 7, 4, 23): (datetime(2014, 7, 7, 9), datetime(2014, 7, 4, 9)),
                datetime(2014, 7, 6, 10): (datetime(2014, 7, 7, 9), datetime(2014, 7, 4, 9)),
                datetime(2014, 7, 7, 5): (datetime(2014, 7, 7, 9), datetime(2014, 7, 4, 9)),
                datetime(2014, 7, 7, 9, 1): (datetime(2014, 7, 8, 9), datetime(2014, 7, 7, 9)),
            },
        )
    )
    opening_time_cases.append(
        (
            [
                BusinessHour(start='11:15'),
                BusinessHour(n=2, start='11:15'),
                BusinessHour(n=3, start='11:15'),
                BusinessHour(start='11:15', end='10:00'),
                BusinessHour(n=2, start='11:15', end='4:00'),
                BusinessHour(n=3, start='11:15', end='15:00'),
            ],
            {
                datetime(2014, 7, 1, 11): (datetime(2014, 7, 1, 11, 15), datetime(2014, 6, 30, 11, 15)),
                datetime(2014, 7, 1, 18): (datetime(2014, 7, 2, 11, 15), datetime(2014, 7, 1, 11, 15)),
                datetime(2014, 7, 1, 23): (datetime(2014, 7, 2, 11, 15), datetime(2014, 7, 1, 11, 15)),
                datetime(2014, 7, 2, 8): (datetime(2014, 7, 2, 11, 15), datetime(2014, 7, 1, 11, 15)),
                datetime(2014, 7, 2, 9): (datetime(2014, 7, 2, 11, 15), datetime(2014, 7, 1, 11, 15)),
                datetime(2014, 7, 2, 10): (datetime(2014, 7, 2, 11, 15), datetime(2014, 7, 1, 11, 15)),
                datetime(2014, 7, 2, 11, 15): (datetime(2014, 7, 2, 11, 15), datetime(2014, 7, 2, 11, 15)),
                datetime(2014, 7, 2, 11, 15, 1): (datetime(2014, 7, 3, 11, 15), datetime(2014, 7, 2, 11, 15)),
                datetime(2014, 7, 5, 10): (datetime(2014, 7, 7, 11, 15), datetime(2014, 7, 4, 11, 15)),
                datetime(2014, 7, 4, 10): (datetime(2014, 7, 4, 11, 15), datetime(2014, 7, 3, 11, 15)),
                datetime(2014, 7, 4, 23): (datetime(2014, 7, 7, 11, 15), datetime(2014, 7, 4, 11, 15)),
                datetime(2014, 7, 6, 10): (datetime(2014, 7, 7, 11, 15), datetime(2014, 7, 4, 11, 15)),
                datetime(2014, 7, 7, 5): (datetime(2014, 7, 7, 11, 15), datetime(2014, 7, 4, 11, 15)),
                datetime(2014, 7, 7, 9, 1): (datetime(2014, 7, 7, 11, 15), datetime(2014, 7, 4, 11, 15)),
            },
        )
    )
    opening_time_cases.append(
        (
            [
                BusinessHour(-1),
                BusinessHour(n=-2),
                BusinessHour(n=-4),
                BusinessHour(n=-1, end='10:00'),
                BusinessHour(n=-2, end='4:00'),
                BusinessHour(n=-4, end='15:00'),
            ],
            {
                datetime(2014, 7, 1, 11): (datetime(2014, 7, 1, 9), datetime(2014, 7, 2, 9)),
                datetime(2014, 7, 1, 18): (datetime(2014, 7, 1, 9), datetime(2014, 7, 2, 9)),
                datetime(2014, 7, 1, 23): (datetime(2014, 7, 1, 9), datetime(2014, 7, 2, 9)),
                datetime(2014, 7, 2, 8): (datetime(2014, 7, 1, 9), datetime(2014, 7, 2, 9)),
                datetime(2014, 7, 2, 9): (datetime(2014, 7, 2, 9), datetime(2014, 7, 2, 9)),
                datetime(2014, 7, 2, 10): (datetime(2014, 7, 2, 9), datetime(2014, 7, 3, 9)),
                datetime(2014, 7, 5, 10): (datetime(2014, 7, 4, 9), datetime(2014, 7, 7, 9)),
                datetime(2014, 7, 4, 10): (datetime(2014, 7, 4, 9), datetime(2014, 7, 7, 9)),
                datetime(2014, 7, 4, 23): (datetime(2014, 7, 4, 9), datetime(2014, 7, 7, 9)),
                datetime(2014, 7, 6, 10): (datetime(2014, 7, 4, 9), datetime(2014, 7, 7, 9)),
                datetime(2014, 7, 7, 5): (datetime(2014, 7, 4, 9), datetime(2014, 7, 7, 9)),
                datetime(2014, 7, 7, 9): (datetime(2014, 7, 7, 9), datetime(2014, 7, 7, 9)),
                datetime(2014, 7, 7, 9, 1): (datetime(2014, 7, 7, 9), datetime(2014, 7, 8, 9)),
            },
        )
    )
    opening_time_cases.append(
        (
            [
                BusinessHour(start='17:00', end='05:00'),
                BusinessHour(n=3, start='17:00', end='03:00'),
            ],
            {
                datetime(2014, 7, 1, 11): (datetime(2014, 7, 1, 17), datetime(2014, 6, 30, 17)),
                datetime(2014, 7, 1, 18): (datetime(2014, 7, 2, 17), datetime(2014, 7, 1, 17)),
                datetime(2014, 7, 1, 23): (datetime(2014, 7, 2, 17), datetime(2014, 7, 1, 17)),
                datetime(2014, 7, 2, 8): (datetime(2014, 7, 2, 17), datetime(2014, 7, 1, 17)),
                datetime(2014, 7, 2, 9): (datetime(2014, 7, 2, 17), datetime(2014, 7, 1, 17)),
                datetime(2014, 7, 4, 17): (datetime(2014, 7, 4, 17), datetime(2014, 7, 4, 17)),
                datetime(2014, 7, 5, 10): (datetime(2014, 7, 7, 17), datetime(2014, 7, 4, 17)),
                datetime(2014, 7, 4, 10): (datetime(2014, 7, 4, 17), datetime(2014, 7, 3, 17)),
                datetime(2014, 7, 4, 23): (datetime(2014, 7, 7, 17), datetime(2014, 7, 4, 17)),
                datetime(2014, 7, 6, 10): (datetime(2014, 7, 7, 17), datetime(2014, 7, 4, 17)),
                datetime(2014, 7, 7, 5): (datetime(2014, 7, 7, 17), datetime(2014, 7, 4, 17)),
                datetime(2014, 7, 7, 17, 1): (datetime(2014, 7, 8, 17), datetime(2014, 7, 7, 17)),
            },
        )
    )
    opening_time_cases.append(
        (
            [
                BusinessHour(-1, start='17:00', end='05:00'),
                BusinessHour(n=-2, start='17:00', end='03:00'),
            ],
            {
                datetime(2014, 7, 1, 11): (datetime(2014, 6, 30, 17), datetime(2014, 7, 1, 17)),
                datetime(2014, 7, 1, 18): (datetime(2014, 7, 1, 17), datetime(2014, 7, 2, 17)),
                datetime(2014, 7, 1, 23): (datetime(2014, 7, 1, 17), datetime(2014, 7, 2, 17)),
                datetime(2014, 7, 2, 8): (datetime(2014, 7, 1, 17), datetime(2014, 7, 2, 17)),
                datetime(2014, 7, 2, 9): (datetime(2014, 7, 1, 17), datetime(2014, 7, 2, 17)),
                datetime(2014, 7, 2, 16, 59): (datetime(2014, 7, 1, 17), datetime(2014, 7, 2, 17)),
                datetime(2014, 7, 5, 10): (datetime(2014, 7, 4, 17), datetime(2014, 7, 7, 17)),
                datetime(2014, 7, 4, 10): (datetime(2014, 7, 3, 17), datetime(2014, 7, 4, 17)),
                datetime(2014, 7, 4, 23): (datetime(2014, 7, 4, 17), datetime(2014, 7, 7, 17)),
                datetime(2014, 7, 6, 10): (datetime(2014, 7, 4, 17), datetime(2014, 7, 7, 17)),
                datetime(2014, 7, 7, 5): (datetime(2014, 7, 4, 17), datetime(2014, 7, 7, 17)),
                datetime(2014, 7, 7, 18): (datetime(2014, 7, 7, 17), datetime(2014, 7, 8, 17)),
            },
        )
    )
    opening_time_cases.append(
        (
            [
                BusinessHour(start=['11:15', '15:00'], end=['13:00', '20:00']),
                BusinessHour(n=3, start=['11:15', '15:00'], end=['12:00', '20:00']),
                BusinessHour(start=['11:15', '15:00'], end=['13:00', '17:00']),
                BusinessHour(n=2, start=['11:15', '15:00'], end=['12:00', '03:00']),
                BusinessHour(n=3, start=['11:15', '15:00'], end=['13:00', '16:00']),
            ],
            {
                datetime(2014, 7, 1, 11): (datetime(2014, 7, 1, 11, 15), datetime(2014, 6, 30, 15)),
                datetime(2014, 7, 1, 18): (datetime(2014, 7, 2, 11, 15), datetime(2014, 7, 1, 15)),
                datetime(2014, 7, 1, 23): (datetime(2014, 7, 2, 11, 15), datetime(2014, 7, 1, 15)),
                datetime(2014, 7, 2, 8): (datetime(2014, 7, 2, 11, 15), datetime(2014, 7, 1, 15)),
                datetime(2014, 7, 2, 9): (datetime(2014, 7, 2, 11, 15), datetime(2014, 7, 1, 15)),
                datetime(2014, 7, 2, 10): (datetime(2014, 7, 2, 11, 15), datetime(2014, 7, 1, 15)),
                datetime(2014, 7, 2, 11, 15): (datetime(2014, 7, 2, 11, 15), datetime(2014, 7, 2, 11, 15)),
                datetime(2014, 7, 2, 11, 15, 1): (datetime(2014, 7, 2, 15), datetime(2014, 7, 2, 11, 15)),
                datetime(2014, 7, 5, 10): (datetime(2014, 7, 7, 11, 15), datetime(2014, 7, 4, 15)),
                datetime(2014, 7, 4, 10): (datetime(2014, 7, 4, 11, 15), datetime(2014, 7, 3, 15)),
                datetime(2014, 7, 4, 23): (datetime(2014, 7, 7, 11, 15), datetime(2014, 7, 4, 15)),
                datetime(2014, 7, 6, 10): (datetime(2014, 7, 7, 11, 15), datetime(2014, 7, 4, 15)),
                datetime(2014, 7, 7, 5): (datetime(2014, 7, 7, 11, 15), datetime(2014, 7, 4, 15)),
                datetime(2014, 7, 7, 9, 1): (datetime(2014, 7, 7, 11, 15), datetime(2014, 7, 4, 15)),
                datetime(2014, 7, 7, 12): (datetime(2014, 7, 7, 15), datetime(2014, 7, 7, 11, 15)),
            },
        )
    )
    opening_time_cases.append(
        (
            [
                BusinessHour(n=-1, start=['17:00', '08:00'], end=['05:00', '10:00']),
                BusinessHour(n=-2, start=['08:00', '17:00'], end=['10:00', '03:00']),
            ],
            {
                datetime(2014, 7, 1, 11): (datetime(2014, 7, 1, 8), datetime(2014, 7, 1, 17)),
                datetime(2014, 7, 1, 18): (datetime(2014, 7, 1, 17), datetime(2014, 7, 2, 8)),
                datetime(2014, 7, 1, 23): (datetime(2014, 7, 1, 17), datetime(2014, 7, 2, 8)),
                datetime(2014, 7, 2, 8): (datetime(2014, 7, 2, 8), datetime(2014, 7, 2, 8)),
                datetime(2014, 7, 2, 9): (datetime(2014, 7, 2, 8), datetime(2014, 7, 2, 17)),
                datetime(2014, 7, 2, 16, 59): (datetime(2014, 7, 2, 8), datetime(2014, 7, 2, 17)),
                datetime(2014, 7, 5, 10): (datetime(2014, 7, 4, 17), datetime(2014, 7, 7, 8)),
                datetime(2014, 7, 4, 10): (datetime(2014, 7, 4, 8), datetime(2014, 7, 4, 17)),
                datetime(2014, 7, 4, 23): (datetime(2014, 7, 4, 17), datetime(2014, 7, 7, 8)),
                datetime(2014, 7, 6, 10): (datetime(2014, 7, 4, 17), datetime(2014, 7, 7, 8)),
                datetime(2014, 7, 7, 5): (datetime(2014, 7, 4, 17), datetime(2014, 7, 7, 8)),
                datetime(2014, 7, 7, 18): (datetime(2014, 7, 7, 17), datetime(2014, 7, 8, 8)),
            },
        )
    )

    @pytest.mark.parametrize('case', opening_time_cases)
    def test_opening_time(self, case: Tuple[List[BusinessHour], Dict[datetime, Tuple[datetime, datetime]]]) -> None:
        _offsets, cases = case
        for offset in _offsets:
            for dt_val, (exp_next, exp_prev) in cases.items():
                assert offset._next_opening_time(dt_val) == exp_next
                assert offset._prev_opening_time(dt_val) == exp_prev
