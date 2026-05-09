from datetime import datetime, timedelta
import numpy as np
import pytest
from pandas._libs import index as libindex
from pandas import DatetimeIndex, Index, Timestamp, bdate_range, date_range, notna
import pandas as pd
from pandas.tseries.frequencies import to_offset
from pandas._testing import tm

class TestGetItem:
    def test_getitem_slice_keeps_name(self) -> None:
        ...
    @pytest.mark.parametrize('tz', [None, 'Asia/Tokyo'])
    def test_getitem(self, tz: str) -> None:
        ...
    def test_getitem_int_list(self) -> None:
        ...

class TestWhere:
    def test_where_doesnt_retain_freq(self) -> None:
        ...
    def test_where_other(self) -> None:
        ...

class TestTake:
    def test_take_dont_lose_meta(self) -> None:
        ...
    def test_take_nan_first_datetime(self) -> None:
        ...
    def test_take(self, tz: str) -> None:
        ...
    def test_take_fill_value(self) -> None:
        ...

class TestGetLoc:
    def test_get_loc_key_unit_mismatch(self) -> None:
        ...
    def test_get_loc_key_unit_mismatch_not_castable(self) -> None:
        ...
    def test_get_loc_time_obj(self) -> None:
        ...
    def test_get_loc_time_nat(self) -> None:
        ...

class TestContains:
    def test_dti_contains_with_duplicates(self) -> None:
        ...
    @pytest.mark.parametrize('vals', [[0, 1, 0], [0, 0, -1], [0, -1, -1], ['2015', '2015', '2016'], ['2015', '2015', '2014']])
    def test_contains_nonunique(self, vals: list) -> None:
        ...

class TestGetIndexer:
    def test_get_indexer_date_objs(self) -> None:
        ...
    def test_get_indexer(self) -> None:
        ...

class TestMaybeCastSliceBound:
    def test_maybe_cast_slice_bounds_empty(self) -> None:
        ...
    def test_maybe_cast_slice_duplicate_monotonic(self) -> None:
        ...

class TestGetSliceBounds:
    @pytest.mark.parametrize('box', [date, datetime, Timestamp])
    @pytest.mark.parametrize('side, expected', [('left', 4), ('right', 5)])
    def test_get_slice_bounds_datetime_within(self, box: type, side: str, expected: int, tz_aware_fixture: str) -> None:
        ...
    @pytest.mark.parametrize('box', [datetime, Timestamp])
    @pytest.mark.parametrize('side', ['left', 'right'])
    @pytest.mark.parametrize('year, expected', [(1999, 0), (2020, 30)])
    def test_get_slice_bounds_datetime_outside(self, box: type, side: str, year: int, expected: int, tz_aware_fixture: str) -> None:
        ...
    @pytest.mark.parametrize('box', [datetime, Timestamp])
    def test_slice_datetime_locs(self, box: type, tz_aware_fixture: str) -> None:
        ...

class TestIndexerBetweenTime:
    def test_indexer_between_time(self) -> None:
        ...
    @pytest.mark.parametrize('unit', ['us', 'ms', 's'])
    def test_indexer_between_time_non_nano(self, unit: str) -> None:
        ...
