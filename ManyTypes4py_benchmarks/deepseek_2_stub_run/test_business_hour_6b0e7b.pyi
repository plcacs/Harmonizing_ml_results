```python
from __future__ import annotations
from typing import Any, ClassVar, Dict, List, Tuple, Union
from datetime import datetime, time as dt_time
from pandas._libs.tslibs import Timedelta, Timestamp
from pandas._libs.tslibs.offsets import BDay, BusinessHour, Nano
from pandas import DatetimeIndex

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
    @pytest.mark.parametrize
    def test_constructor_errors(
        self,
        start: Any,
        end: Any,
        match: str
    ) -> None: ...
    
    def test_different_normalize_equals(
        self,
        _offset: type[BusinessHour]
    ) -> None: ...
    
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
        offset10: BusinessHour
    ) -> None: ...
    
    def test_with_offset(
        self,
        dt: datetime
    ) -> None: ...
    
    @pytest.mark.parametrize
    def test_eq_attribute(
        self,
        offset_name: str,
        request: Any
    ) -> None: ...
    
    @pytest.mark.parametrize
    def test_eq(
        self,
        offset1: BusinessHour,
        offset2: BusinessHour
    ) -> None: ...
    
    @pytest.mark.parametrize
    def test_neq(
        self,
        offset1: BusinessHour,
        offset2: BusinessHour
    ) -> None: ...
    
    @pytest.mark.parametrize
    def test_hash(
        self,
        offset_name: str,
        request: Any
    ) -> None: ...
    
    def test_add_datetime(
        self,
        dt: datetime,
        offset1: BusinessHour,
        offset2: BusinessHour,
        offset3: BusinessHour,
        offset4: BusinessHour,
        offset8: BusinessHour,
        offset9: BusinessHour,
        offset10: BusinessHour
    ) -> None: ...
    
    def test_sub(
        self,
        dt: datetime,
        offset2: BusinessHour,
        _offset: type[BusinessHour]
    ) -> None: ...
    
    def test_multiply_by_zero(
        self,
        dt: datetime,
        offset1: BusinessHour,
        offset2: BusinessHour
    ) -> None: ...
    
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
        offset10: BusinessHour
    ) -> None: ...
    
    def testRollback2(
        self,
        _offset: type[BusinessHour]
    ) -> None: ...
    
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
        offset10: BusinessHour
    ) -> None: ...
    
    def testRollforward2(
        self,
        _offset: type[BusinessHour]
    ) -> None: ...
    
    def test_roll_date_object(self) -> None: ...
    
    @pytest.mark.parametrize
    def test_normalize(
        self,
        case: Any
    ) -> None: ...
    
    @pytest.mark.parametrize
    def test_is_on_offset(
        self,
        case: Any
    ) -> None: ...
    
    @pytest.mark.parametrize
    def test_apply(
        self,
        case: Any
    ) -> None: ...
    
    @pytest.mark.parametrize
    def test_apply_large_n(
        self,
        case: Any
    ) -> None: ...
    
    def test_apply_nanoseconds(self) -> None: ...
    
    @pytest.mark.parametrize
    def test_bday_ignores_timedeltas(
        self,
        unit: Any,
        td_unit: str
    ) -> None: ...
    
    def test_add_bday_offset_nanos(self) -> None: ...

class TestOpeningTimes:
    @pytest.mark.parametrize
    def test_opening_time(
        self,
        case: Any
    ) -> None: ...

def assert_offset_equal(
    offset: Any,
    base: Any,
    expected: Any
) -> None: ...

normalize_cases: List[Any] = ...
on_offset_cases: List[Any] = ...
apply_cases: List[Any] = ...
apply_large_n_cases: List[Any] = ...
opening_time_cases: List[Any] = ...
```