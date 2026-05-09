"""
Stub file for test_business_hour_6b0e7b.py
"""

from datetime import datetime, time as dt_time
from typing import Any, Dict, List, Optional, Tuple, Type, Union
import pytest
from pandas._libs.tslibs import Timedelta, Timestamp
from pandas._libs.tslibs.offsets import BDay, BusinessHour, Nano
from pandas import DatetimeIndex

@pytest.fixture
def dt() -> datetime:
    ...

@pytest.fixture
def _offset() -> Type[BusinessHour]:
    ...

@pytest.fixture
def offset1() -> BusinessHour:
    ...

@pytest.fixture
def offset2() -> BusinessHour:
    ...

@pytest.fixture
def offset3() -> BusinessHour:
    ...

@pytest.fixture
def offset4() -> BusinessHour:
    ...

@pytest.fixture
def offset5() -> BusinessHour:
    ...

@pytest.fixture
def offset6() -> BusinessHour:
    ...

@pytest.fixture
def offset7() -> BusinessHour:
    ...

@pytest.fixture
def offset8() -> BusinessHour:
    ...

@pytest.fixture
def offset9() -> BusinessHour:
    ...

@pytest.fixture
def offset10() -> BusinessHour:
    ...

class TestBusinessHour:
    @pytest.mark.parametrize('start,end,match', [
        (dt_time, str, str),
        (str, str, str),
        (str, str, str),
        (List[Any], str, str),
        (str, List[Any], str),
        (List[str], str, str),
        (List[str], List[str], str),
        (List[str], List[str], str),
        (List[str], List[str], str),
        (List[str], List[str], str)
    ])
    def test_constructor_errors(self, start: Union[List[str], str, dt_time], end: Union[List[str], str], match: str) -> None:
        ...

    def test_different_normalize_equals(self, _offset: Type[BusinessHour]) -> None:
        ...

    def test_repr(self, offset1: BusinessHour, offset2: BusinessHour, offset3: BusinessHour, offset4: BusinessHour, offset5: BusinessHour, offset6: BusinessHour, offset7: BusinessHour, offset8: BusinessHour, offset9: BusinessHour, offset10: BusinessHour) -> None:
        ...

    def test_with_offset(self, dt: datetime) -> None:
        ...

    @pytest.mark.parametrize('offset_name', ['offset1', 'offset2', 'offset3', 'offset4', 'offset8', 'offset9', 'offset10'])
    def test_eq_attribute(self, offset_name: str, request: pytest.FixtureRequest) -> None:
        ...

    @pytest.mark.parametrize('offset1,offset2', [
        (BusinessHour, BusinessHour),
        (BusinessHour, BusinessHour)
    ])
    def test_eq(self, offset1: BusinessHour, offset2: BusinessHour) -> None:
        ...

    @pytest.mark.parametrize('offset1,offset2', [
        (BusinessHour, BusinessHour),
        (BusinessHour, BusinessHour),
        (BusinessHour, BusinessHour),
        (BusinessHour, BusinessHour)
    ])
    def test_neq(self, offset1: BusinessHour, offset2: BusinessHour) -> None:
        ...

    @pytest.mark.parametrize('offset_name', ['offset1', 'offset2', 'offset3', 'offset4', 'offset8', 'offset9', 'offset10'])
    def test_hash(self, offset_name: str, request: pytest.FixtureRequest) -> None:
        ...

    def test_add_datetime(self, dt: datetime, offset1: BusinessHour, offset2: BusinessHour, offset3: BusinessHour, offset4: BusinessHour, offset8: BusinessHour, offset9: BusinessHour, offset10: BusinessHour) -> None:
        ...

    def test_sub(self, dt: datetime, offset2: BusinessHour, _offset: Type[BusinessHour]) -> None:
        ...

    def test_multiply_by_zero(self, dt: datetime, offset1: BusinessHour, offset2: BusinessHour) -> None:
        ...

    def testRollback1(self, dt: datetime, _offset: Type[BusinessHour], offset1: BusinessHour, offset2: BusinessHour, offset3: BusinessHour, offset4: BusinessHour, offset5: BusinessHour, offset6: BusinessHour, offset7: BusinessHour, offset8: BusinessHour, offset9: BusinessHour, offset10: BusinessHour) -> None:
        ...

    def testRollback2(self, _offset: Type[BusinessHour]) -> None:
        ...

    def testRollforward1(self, dt: datetime, _offset: Type[BusinessHour], offset1: BusinessHour, offset2: BusinessHour, offset3: BusinessHour, offset4: BusinessHour, offset5: BusinessHour, offset6: BusinessHour, offset7: BusinessHour, offset8: BusinessHour, offset9: BusinessHour, offset10: BusinessHour) -> None:
        ...

    def testRollforward2(self, _offset: Type[BusinessHour]) -> None:
        ...

    def test_roll_date_object(self) -> None:
        ...

    @pytest.mark.parametrize('case', List[Tuple[BusinessHour, Dict[datetime, datetime]]])
    def test_normalize(self, case: Tuple[BusinessHour, Dict[datetime, datetime]]) -> None:
        ...

    @pytest.mark.parametrize('case', List[Tuple[BusinessHour, Dict[datetime, bool]]])
    def test_is_on_offset(self, case: Tuple[BusinessHour, Dict[datetime, bool]]) -> None:
        ...

    @pytest.mark.parametrize('case', List[Tuple[BusinessHour, Dict[datetime, datetime]]])
    def test_apply(self, case: Tuple[BusinessHour, Dict[datetime, datetime]]) -> None:
        ...

    @pytest.mark.parametrize('case', List[Tuple[BusinessHour, Dict[datetime, datetime]]])
    def test_apply_large_n(self, case: Tuple[BusinessHour, Dict[datetime, datetime]]) -> None:
        ...

    def test_apply_nanoseconds(self) -> None:
        ...

    @pytest.mark.parametrize('td_unit', ['s', 'ms', 'us', 'ns'])
    def test_bday_ignores_timedeltas(self, unit: str, td_unit: str) -> None:
        ...

    def test_add_bday_offset_nanos(self) -> None:
        ...

class TestOpeningTimes:
    @pytest.mark.parametrize('case', List[Tuple[List[BusinessHour], Dict[datetime, Tuple[datetime, datetime]]]])
    def test_opening_time(self, case: Tuple[List[BusinessHour], Dict[datetime, Tuple[datetime, datetime]]]) -> None:
        ...