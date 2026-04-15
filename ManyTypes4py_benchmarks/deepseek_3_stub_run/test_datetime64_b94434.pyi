import datetime
from typing import Any, Callable, List, Optional, Sequence, Tuple, Type, Union

import numpy as np
import pandas as pd
from pandas import (
    DataFrame,
    DatetimeIndex,
    Index,
    NaTType,
    Period,
    Series,
    Timedelta,
    TimedeltaIndex,
    Timestamp,
)
from pandas._libs.tslibs.offsets import BaseOffset, Tick
from pandas.core.arrays import DatetimeArray
from pandas.core.ops import roperator

import pandas._testing as tm

def localize_pydatetime(
    dt: datetime.datetime, tz: Optional[datetime.tzinfo]
) -> datetime.datetime: ...

def shift_months(
    stamps: np.ndarray, months: int, reso: int
) -> np.ndarray: ...

def assert_cannot_add(
    left: Any, right: Any, msg: Optional[str] = None
) -> None: ...

def assert_invalid_addsub_type(
    left: Any, right: Any, msg: Optional[str] = None
) -> None: ...

def assert_invalid_comparison(
    left: Any, right: Any, box: Any
) -> None: ...

def get_upcast_box(
    left: Any, right: Any, is_cmp: bool = False
) -> Type: ...

class TestDatetime64ArrayLikeComparisons:
    def test_compare_zerodim(
        self,
        tz_naive_fixture: Optional[datetime.tzinfo],
        box_with_array: Type,
    ) -> None: ...

    @pytest.mark.parametrize("other", ["foo", -1, 99, 4.0, object(), timedelta(days=2), datetime(2001, 1, 1).date(), None, np.nan])
    def test_dt64arr_cmp_scalar_invalid(
        self,
        other: Any,
        tz_naive_fixture: Optional[datetime.tzinfo],
        box_with_array: Type,
    ) -> None: ...

    @pytest.mark.parametrize("other", [list(range(10)), np.arange(10), np.arange(10).astype(np.float32), np.arange(10).astype(object), pd.timedelta_range('1ns', periods=10).array, np.array(pd.timedelta_range('1ns', periods=10)), list(pd.timedelta_range('1ns', periods=10)), pd.timedelta_range('1 Day', periods=10).astype(object), pd.period_range('1971-01-01', freq='D', periods=10).array, pd.period_range('1971-01-01', freq='D', periods=10).astype(object)])
    def test_dt64arr_cmp_arraylike_invalid(
        self,
        other: Any,
        tz_naive_fixture: Optional[datetime.tzinfo],
        box_with_array: Type,
    ) -> None: ...

    def test_dt64arr_cmp_mixed_invalid(
        self, tz_naive_fixture: Optional[datetime.tzinfo]
    ) -> None: ...

    def test_dt64arr_nat_comparison(
        self,
        tz_naive_fixture: Optional[datetime.tzinfo],
        box_with_array: Type,
    ) -> None: ...

class TestDatetime64SeriesComparison:
    @pytest.mark.parametrize("pair", [([Timestamp('2011-01-01'), NaT, Timestamp('2011-01-03')], [NaT, NaT, Timestamp('2011-01-03')]), ([Timedelta('1 days'), NaT, Timedelta('3 days')], [NaT, NaT, Timedelta('3 days')]), ([Period('2011-01', freq='M'), NaT, Period('2011-03', freq='M')], [NaT, NaT, Period('2011-03', freq='M')])])
    @pytest.mark.parametrize("reverse", [True, False])
    @pytest.mark.parametrize("dtype", [None, object])
    @pytest.mark.parametrize("op, expected", [(operator.eq, [False, False, True]), (operator.ne, [True, True, False]), (operator.lt, [False, False, False]), (operator.gt, [False, False, False]), (operator.ge, [False, False, True]), (operator.le, [False, False, True])])
    def test_nat_comparisons(
        self,
        dtype: Optional[Type],
        index_or_series: Type,
        reverse: bool,
        pair: Tuple[List[Any], List[Any]],
        op: Callable[[Any, Any], Any],
        expected: List[bool],
    ) -> None: ...

    @pytest.mark.parametrize("data", [[Timestamp('2011-01-01'), NaT, Timestamp('2011-01-03')], [Timedelta('1 days'), NaT, Timedelta('3 days')], [Period('2011-01', freq='M'), NaT, Period('2011-03', freq='M')]])
    @pytest.mark.parametrize("dtype", [None, object])
    def test_nat_comparisons_scalar(
        self,
        dtype: Optional[Type],
        data: List[Any],
        box_with_array: Type,
    ) -> None: ...

    @pytest.mark.parametrize("val", [datetime(2000, 1, 4), datetime(2000, 1, 5)])
    def test_series_comparison_scalars(self, val: datetime.datetime) -> None: ...

    @pytest.mark.parametrize("left,right", [("lt", "gt"), ("le", "ge"), ("eq", "eq"), ("ne", "ne")])
    def test_timestamp_compare_series(self, left: str, right: str) -> None: ...

    def test_dt64arr_timestamp_equality(self, box_with_array: Type) -> None: ...

    @pytest.mark.parametrize("datetimelike", [Timestamp('20130101'), datetime(2013, 1, 1), np.datetime64('2013-01-01T00:00', 'ns')])
    @pytest.mark.parametrize("op,expected", [(operator.lt, [True, False, False, False]), (operator.le, [True, True, False, False]), (operator.eq, [False, True, False, False]), (operator.gt, [False, False, False, True])])
    def test_dt64_compare_datetime_scalar(
        self,
        datetimelike: Union[Timestamp, datetime.datetime, np.datetime64],
        op: Callable[[Any, Any], Any],
        expected: List[bool],
    ) -> None: ...

    def test_ts_series_numpy_maximum(self) -> None: ...

class TestDatetimeIndexComparisons:
    def test_comparators(self, comparison_op: Callable[[Any, Any], Any]) -> None: ...

    @pytest.mark.parametrize("other", [datetime(2016, 1, 1), Timestamp('2016-01-01'), np.datetime64('2016-01-01')])
    def test_dti_cmp_datetimelike(
        self,
        other: Union[datetime.datetime, Timestamp, np.datetime64],
        tz_naive_fixture: Optional[datetime.tzinfo],
    ) -> None: ...

    @pytest.mark.parametrize("dtype", [None, object])
    def test_dti_cmp_nat(
        self,
        dtype: Optional[Type],
        box_with_array: Type,
    ) -> None: ...

    def test_dti_cmp_nat_behaves_like_float_cmp_nan(self) -> None: ...

    def test_comparison_tzawareness_compat(
        self,
        comparison_op: Callable[[Any, Any], Any],
        box_with_array: Type,
    ) -> None: ...

    def test_comparison_tzawareness_compat_scalars(
        self,
        comparison_op: Callable[[Any, Any], Any],
        box_with_array: Type,
    ) -> None: ...

    @pytest.mark.parametrize("other", [datetime(2016, 1, 1), Timestamp('2016-01-01'), np.datetime64('2016-01-01')])
    @pytest.mark.filterwarnings("ignore:elementwise comp:DeprecationWarning")
    def test_scalar_comparison_tzawareness(
        self,
        comparison_op: Callable[[Any, Any], Any],
        other: Union[datetime.datetime, Timestamp, np.datetime64],
        tz_aware_fixture: datetime.tzinfo,
        box_with_array: Type,
    ) -> None: ...

    def test_nat_comparison_tzawareness(
        self, comparison_op: Callable[[Any, Any], Any]
    ) -> None: ...

    def test_dti_cmp_str(
        self, tz_naive_fixture: Optional[datetime.tzinfo]
    ) -> None: ...

    def test_dti_cmp_list(self) -> None: ...

    @pytest.mark.parametrize("other", [pd.timedelta_range('1D', periods=10), pd.timedelta_range('1D', periods=10).to_series(), pd.timedelta_range('1D', periods=10).asi8.view('m8[ns]')], ids=lambda x: type(x).__name__)
    def test_dti_cmp_tdi_tzawareness(self, other: Any) -> None: ...

    def test_dti_cmp_object_dtype(self) -> None: ...

class TestDatetime64Arithmetic:
    @pytest.mark.arm_slow
    def test_dt64arr_add_timedeltalike_scalar(
        self,
        tz_naive_fixture: Optional[datetime.tzinfo],
        two_hours: Union[Timedelta, np.timedelta64, datetime.timedelta],
        box_with_array: Type,
    ) -> None: ...

    def test_dt64arr_sub_timedeltalike_scalar(
        self,
        tz_naive_fixture: Optional[datetime.tzinfo],
        two_hours: Union[Timedelta, np.timedelta64, datetime.timedelta],
        box_with_array: Type,
    ) -> None: ...

    def test_dt64_array_sub_dt_with_different_timezone(
        self, box_with_array: Type
    ) -> None: ...

    def test_dt64_array_sub_dt64_array_with_different_timezone(
        self, box_with_array: Type
    ) -> None: ...

    def test_dt64arr_add_sub_td64_nat(
        self,
        box_with_array: Type,
        tz_naive_fixture: Optional[datetime.tzinfo],
    ) -> None: ...

    def test_dt64arr_add_sub_td64ndarray(
        self,
        tz_naive_fixture: Optional[datetime.tzinfo],
        box_with_array: Type,
    ) -> None: ...

    @pytest.mark.parametrize("ts", [Timestamp('2013-01-01'), Timestamp('2013-01-01').to_pydatetime(), Timestamp('2013-01-01').to_datetime64(), np.datetime64('2013-01-01', 'D')])
    def test_dt64arr_sub_dtscalar(
        self,
        box_with_array: Type,
        ts: Union[Timestamp, datetime.datetime, np.datetime64],
    ) -> None: ...

    def test_dt64arr_sub_timestamp_tzaware(
        self, box_with_array: Type
    ) -> None: ...

    def test_dt64arr_sub_NaT(
        self, box_with_array: Type, unit: str
    ) -> None: ...

    def test_dt64arr_sub_dt64object_array(
        self,
        performance_warning: Any,
        box_with_array: Type,
        tz_naive_fixture: Optional[datetime.tzinfo],
    ) -> None: ...

    def test_dt64arr_naive_sub_dt64ndarray(
        self, box_with_array: Type
    ) -> None: ...

    def test_dt64arr_aware_sub_dt64ndarray_raises(
        self,
        tz_aware_fixture: datetime.tzinfo,
        box_with_array: Type,
    ) -> None: ...

    def test_dt64arr_add_dtlike_raises(
        self,
        tz_naive_fixture: Optional[datetime.tzinfo],
        box_with_array: Type,
    ) -> None: ...

    @pytest.mark.parametrize("freq", ["h", "D", "W", "2ME", "MS", "QE", "B", None])
    @pytest.mark.parametrize("dtype", [None, "uint8"])
    def test_dt64arr_addsub_intlike(
        self,
        dtype: Optional[str],
        index_or_series_or_array: Type,
        freq: Optional[str],
        tz_naive_fixture: Optional[datetime.tzinfo],
    ) -> None: ...

    @pytest.mark.parametrize("other", [3.14, np.array([2.0, 3.0]), Period('2011-01-01', freq='D'), time(1, 2, 3)])
    @pytest.mark.parametrize("dti_freq", [None, "D"])
    def test_dt64arr_add_sub_invalid(
        self,
        dti_freq: Optional[str],
        other: Any,
        box_with_array: Type,
    ) -> None: ...

    @pytest.mark.parametrize("pi_freq", ["D", "W", "Q", "h"])
    @pytest.mark.parametrize("dti_freq", [None, "D"])
    def test_dt64arr_add_sub_parr(
        self,
        dti_freq: Optional[str],
        pi_freq: str,
        box_with_array: Type,
        box_with_array2: Type,
    ) -> None: ...

    @pytest.mark.filterwarnings("ignore::pandas.errors.PerformanceWarning")
    def test_dt64arr_addsub_time_objects_raises(
        self,
        box_with_array: Type,
        tz_naive_fixture: Optional[datetime.tzinfo],
    ) -> None: ...

    @pytest.mark.parametrize("dt64_series", [Series([Timestamp('19900315'), Timestamp('19900315')]), Series([NaT, Timestamp('19900315')]), Series([NaT, NaT], dtype='datetime64[ns]')])
    @pytest.mark.parametrize("one", [1, 1.0, np.array(1)])
    def test_dt64_mul_div_numeric_invalid(
        self,
        one: Union[int, float, np.ndarray],
        dt64_series: Series,
        box_with_array: Type,
    ) -> None: ...

class TestDatetime64DateOffsetArithmetic:
    def test_dt64arr_series_add_tick_DateOffset(
        self, box_with_array: Type, unit: str
    ) -> None: ...

    def test_dt64arr_series_sub_tick_DateOffset(
        self, box_with_array: Type
    ) -> None: ...

    @pytest.mark.parametrize("cls_name", ["Day", "Hour", "Minute", "Second", "Milli", "Micro", "Nano"])
    def test_dt64arr_add_sub_tick_DateOffset_smoke(
        self, cls_name: str, box_with_array: Type
    ) -> None: ...

    def test_dti_add_tick_tzaware(
        self,
        tz_aware_fixture: datetime.tzinfo,
        box_with_array: Type,
    ) -> None: ...

    def test_dt64arr_add_sub_relativedelta_offsets(
        self, box_with_array: Type, unit: str
    ) -> None: ...

    @pytest.mark.filterwarnings("ignore::pandas.errors.PerformanceWarning")
    @pytest.mark.parametrize("cls_and_kwargs", ["YearBegin", ("YearBegin", {"month": 5}), "YearEnd", ("YearEnd", {"month": 5}), "MonthBegin", "MonthEnd", "SemiMonthEnd", "SemiMonthBegin", "Week", ("Week", {"weekday": 3}), ("Week", {"weekday": 6}), "BusinessDay", "BDay", "QuarterEnd", "QuarterBegin", "CustomBusinessDay", "CDay", "CBMonthEnd", "CBMonthBegin", "BMonthBegin", "BMonthEnd", "BusinessHour", "BYearBegin", "BYearEnd", "BQuarterBegin", ("LastWeekOfMonth", {"weekday": 2}), ("FY5253Quarter", {"qtr_with_extra_week": 1, "startingMonth": 1, "weekday": 2, "variation": "nearest"}), ("FY5253", {"weekday": 0, "startingMonth": 2, "variation": "nearest"}), ("WeekOfMonth", {"weekday": 2, "week": 2}), "Easter", ("DateOffset", {"day": 4}), ("DateOffset", {"month": 5})])
    @pytest.mark.parametrize("normalize", [True, False])
    @pytest.mark.parametrize("n", [0, 5])
    @pytest.mark.parametrize("tz", [None, "US/Central"])
    def test_dt64arr_add_sub_DateOffsets(
        self,
        box_with_array: Type,
        n: int,
        normalize: bool,
        cls_and_kwargs: Union[str, Tuple[str, dict]],
        unit: str,
        tz: Optional[str],
    ) -> None: ...

    @pytest.mark.parametrize("other", [[pd.offsets.MonthEnd(), pd.offsets.Day(n=2)], [pd.offsets.DateOffset(years=1), pd.offsets.MonthEnd()], [