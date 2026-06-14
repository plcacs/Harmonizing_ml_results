from __future__ import annotations

from datetime import datetime, time as dt_time
from typing import Any

import pytest

from pandas._libs.tslibs.offsets import BusinessHour


@pytest.fixture
def dt() -> datetime: ...

@pytest.fixture
def _offset() -> type[BusinessHour]: ...

@pytest.fixture
def offset1() -> BusinessHour: ...

@pytest.fixture
def offset2() -> BusinessHour: ...

@pytest.fixture
def offset3() -> BusinessHour: ...

@pytest.fixture
def offset4() -> BusinessHour: ...

@pytest.fixture
def offset5() -> BusinessHour: ...

@pytest.fixture
def offset6() -> BusinessHour: ...

@pytest.fixture
def offset7() -> BusinessHour: ...

@pytest.fixture
def offset8() -> BusinessHour: ...

@pytest.fixture
def offset9() -> BusinessHour: ...

@pytest.fixture
def offset10() -> BusinessHour: ...


class TestBusinessHour:
    @pytest.mark.parametrize("start,end,match", [])
    def test_constructor_errors(self, start: Any, end: Any, match: str) -> None: ...
    def test_different_normalize_equals(self, _offset: type[BusinessHour]) -> None: ...
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
    ) -> None: ...
    def test_with_offset(self, dt: datetime) -> None: ...
    @pytest.mark.parametrize("offset_name", [])
    def test_eq_attribute(self, offset_name: str, request: pytest.FixtureRequest) -> None: ...
    @pytest.mark.parametrize("offset1,offset2", [])
    def test_eq(self, offset1: BusinessHour, offset2: BusinessHour) -> None: ...
    @pytest.mark.parametrize("offset1,offset2", [])
    def test_neq(self, offset1: BusinessHour, offset2: BusinessHour) -> None: ...
    @pytest.mark.parametrize("offset_name", [])
    def test_hash(self, offset_name: str, request: pytest.FixtureRequest) -> None: ...
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
    ) -> None: ...
    def test_sub(self, dt: datetime, offset2: BusinessHour, _offset: type[BusinessHour]) -> None: ...
    def test_multiply_by_zero(self, dt: datetime, offset1: BusinessHour, offset2: BusinessHour) -> None: ...
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
    ) -> None: ...
    def testRollback2(self, _offset: type[BusinessHour]) -> None: ...
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
    ) -> None: ...
    def testRollforward2(self, _offset: type[BusinessHour]) -> None: ...
    def test_roll_date_object(self) -> None: ...

    normalize_cases: list[tuple[BusinessHour, dict[datetime, datetime]]]

    @pytest.mark.parametrize("case", normalize_cases)
    def test_normalize(self, case: tuple[BusinessHour, dict[datetime, datetime]]) -> None: ...

    on_offset_cases: list[tuple[BusinessHour, dict[datetime, bool]]]

    @pytest.mark.parametrize("case", on_offset_cases)
    def test_is_on_offset(self, case: tuple[BusinessHour, dict[datetime, bool]]) -> None: ...

    apply_cases: list[tuple[BusinessHour, dict[datetime, datetime]]]

    @pytest.mark.parametrize("case", apply_cases)
    def test_apply(self, case: tuple[BusinessHour, dict[datetime, datetime]]) -> None: ...

    apply_large_n_cases: list[tuple[BusinessHour, dict[datetime, datetime]]]

    @pytest.mark.parametrize("case", apply_large_n_cases)
    def test_apply_large_n(self, case: tuple[BusinessHour, dict[datetime, datetime]]) -> None: ...
    def test_apply_nanoseconds(self) -> None: ...
    @pytest.mark.parametrize("td_unit", [])
    def test_bday_ignores_timedeltas(self, unit: str, td_unit: str) -> None: ...
    def test_add_bday_offset_nanos(self) -> None: ...


class TestOpeningTimes:
    opening_time_cases: list[tuple[list[BusinessHour], dict[datetime, tuple[datetime, datetime]]]]

    @pytest.mark.parametrize("case", opening_time_cases)
    def test_opening_time(
        self, case: tuple[list[BusinessHour], dict[datetime, tuple[datetime, datetime]]]
    ) -> None: ...