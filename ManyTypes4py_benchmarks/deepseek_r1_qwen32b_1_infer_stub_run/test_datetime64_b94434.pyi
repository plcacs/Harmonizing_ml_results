from datetime import datetime, time, timedelta, timezone
from itertools import product, starmap
import operator
import numpy as np
import pytest
from pandas._libs.tslibs.conversion import localize_pydatetime
from pandas._libs.tslibs.offsets import shift_months
import pandas as pd
from pandas import (
    DateOffset,
    DatetimeIndex,
    NaT,
    Period,
    Series,
    Timedelta,
    TimedeltaIndex,
    Timestamp,
    date_range,
)
import pandas._testing as tm
from pandas.core import roperator
from pandas.tests.arithmetic.common import (
    assert_cannot_add,
    assert_invalid_addsub_type,
    assert_invalid_comparison,
    get_upcast_box,
)

class TestDatetime64ArrayLikeComparisons:
    def test_compare_zerodim(
        self, tz_naive_fixture: timezone, box_with_array: pd.DataFrame | pd.Series | np.ndarray
    ) -> None:
        ...
    
    @pytest.mark.parametrize('other', [str, int, float, object, timedelta, datetime.date, None, np.nan])
    def test_dt64arr_cmp_scalar_invalid(
        self, other: Union[str, int, float, object, timedelta, datetime.date, None, np.nan], 
        tz_naive_fixture: timezone, 
        box_with_array: pd.DataFrame | pd.Series | np.ndarray
    ) -> None:
        ...

    @pytest.mark.parametrize('other', [list, np.ndarray, pd.timedelta_range])
    def test_dt64arr_cmp_arraylike_invalid(
        self, other: Union[list, np.ndarray, pd.timedelta_range], 
        tz_naive_fixture: timezone, 
        box_with_array: pd.DataFrame | pd.Series | np.ndarray
    ) -> None:
        ...

    def test_dt64arr_cmp_mixed_invalid(self, tz_naive_fixture: timezone) -> None:
        ...

    def test_dt64arr_nat_comparison(
        self, tz_naive_fixture: timezone, box_with_array: pd.DataFrame | pd.Series | np.ndarray
    ) -> None:
        ...

class TestDatetime64SeriesComparison:
    @pytest.mark.parametrize('pair', [list, np.ndarray])
    @pytest.mark.parametrize('reverse', [bool])
    @pytest.mark.parametrize('dtype', [type, None])
    @pytest.mark.parametrize('op', [operator.eq, operator.ne, operator.lt, operator.gt, operator.ge, operator.le])
    def test_nat_comparisons(
        self, dtype: type | None, 
        index_or_series: pd.Index | pd.Series, 
        reverse: bool, 
        pair: list, 
        op: operator
    ) -> None:
        ...

    @pytest.mark.parametrize('data', [list])
    @pytest.mark.parametrize('dtype', [type, None])
    def test_nat_comparisons_scalar(
        self, dtype: type | None, 
        data: list, 
        box_with_array: pd.DataFrame | pd.Series | np.ndarray
    ) -> None:
        ...

    @pytest.mark.parametrize('val', [datetime])
    def test_series_comparison_scalars(self, val: datetime) -> None:
        ...

    @pytest.mark.parametrize('left,right', [('lt', 'gt'), ('le', 'ge'), ('eq', 'eq'), ('ne', 'ne')])
    def test_timestamp_compare_series(self, left: str, right: str) -> None:
        ...

    def test_dt64arr_timestamp_equality(
        self, box_with_array: pd.DataFrame | pd.Series | np.ndarray
    ) -> None:
        ...

    @pytest.mark.parametrize('datetimelike', [Timestamp, datetime, np.datetime64])
    @pytest.mark.parametrize('op', [operator.lt, operator.le, operator.eq, operator.gt])
    def test_dt64_compare_datetime_scalar(
        self, datetimelike: Union[Timestamp, datetime, np.datetime64], 
        op: operator, 
        expected: list
    ) -> None:
        ...

    def test_ts_series_numpy_maximum(self) -> None:
        ...

class TestDatetimeIndexComparisons:
    def test_comparators(self, comparison_op: operator) -> None:
        ...

    @pytest.mark.parametrize('other', [datetime, Timestamp, np.datetime64])
    def test_dti_cmp_datetimelike(
        self, other: Union[datetime, Timestamp, np.datetime64], 
        tz_naive_fixture: timezone
    ) -> None:
        ...

    @pytest.mark.parametrize('dtype', [type, None])
    def test_dti_cmp_nat(
        self, dtype: type | None, 
        box_with_array: pd.DataFrame | pd.Series | np.ndarray
    ) -> None:
        ...

    def test_dti_cmp_nat_behaves_like_float_cmp_nan(self) -> None:
        ...

    def test_comparison_tzawareness_compat(
        self, comparison_op: operator, 
        box_with_array: pd.DataFrame | pd.Series | np.ndarray
    ) -> None:
        ...

    def test_comparison_tzawareness_compat_scalars(
        self, comparison_op: operator, 
        box_with_array: pd.DataFrame | pd.Series | np.ndarray
    ) -> None:
        ...

    @pytest.mark.parametrize('other', [datetime, Timestamp, np.datetime64])
    def test_scalar_comparison_tzawareness(
        self, comparison_op: operator, 
        other: Union[datetime, Timestamp, np.datetime64], 
        tz_aware_fixture: timezone, 
        box_with_array: pd.DataFrame | pd.Series | np.ndarray
    ) -> None:
        ...

    def test_nat_comparison_tzawareness(self, comparison_op: operator) -> None:
        ...

    def test_dti_cmp_str(self, tz_naive_fixture: timezone) -> None:
        ...

    def test_dti_cmp_list(self) -> None:
        ...

    @pytest.mark.parametrize('other', [pd.timedelta_range, pd.Series, np.ndarray])
    def test_dti_cmp_tdi_tzawareness(self, other: Union[pd.timedelta_range, pd.Series, np.ndarray]) -> None:
        ...

    def test_dti_cmp_object_dtype(self) -> None:
        ...

class TestDatetime64Arithmetic:
    def test_dt64arr_add_timedeltalike_scalar(
        self, tz_naive_fixture: timezone, 
        two_hours: timedelta, 
        box_with_array: pd.DataFrame | pd.Series | np.ndarray
    ) -> None:
        ...

    def test_dt64arr_sub_timedeltalike_scalar(
        self, tz_naive_fixture: timezone, 
        two_hours: timedelta, 
        box_with_array: pd.DataFrame | pd.Series | np.ndarray
    ) -> None:
        ...

    def test_dt64_array_sub_dt_with_different_timezone(
        self, box_with_array: pd.DataFrame | pd.Series | np.ndarray
    ) -> None:
        ...

    def test_dt64_array_sub_dt64_array_with_different_timezone(
        self, box_with_array: pd.DataFrame | pd.Series | np.ndarray
    ) -> None:
        ...

    def test_dt64arr_add_sub_td64_nat(
        self, box_with_array: pd.DataFrame | pd.Series | np.ndarray, 
        tz_naive_fixture: timezone
    ) -> None:
        ...

    def test_dt64arr_add_sub_td64ndarray(
        self, tz_naive_fixture: timezone, 
        box_with_array: pd.DataFrame | pd.Series | np.ndarray
    ) -> None:
        ...

    @pytest.mark.parametrize('ts', [Timestamp, datetime, np.datetime64])
    def test_dt64arr_sub_dtscalar(
        self, box_with_array: pd.DataFrame | pd.Series | np.ndarray, 
        ts: Union[Timestamp, datetime, np.datetime64]
    ) -> None:
        ...

    def test_dt64arr_sub_timestamp_tzaware(
        self, box_with_array: pd.DataFrame | pd.Series | np.ndarray
    ) -> None:
        ...

    def test_dt64arr_sub_NaT(
        self, box_with_array: pd.DataFrame | pd.Series | np.ndarray, 
        unit: str
    ) -> None:
        ...

    def test_dt64arr_sub_dt64object_array(
        self, performance_warning: pytest.Warning, 
        box_with_array: pd.DataFrame | pd.Series | np.ndarray, 
        tz_naive_fixture: timezone
    ) -> None:
        ...

    def test_dt64arr_naive_sub_dt64ndarray(
        self, box_with_array: pd.DataFrame | pd.Series | np.ndarray
    ) -> None:
        ...

    def test_dt64arr_aware_sub_dt64ndarray_raises(
        self, tz_aware_fixture: timezone, 
        box_with_array: pd.DataFrame | pd.Series | np.ndarray
    ) -> None:
        ...

    def test_dt64arr_add_dtlike_raises(
        self, tz_naive_fixture: timezone, 
        box_with_array: pd.DataFrame | pd.Series | np.ndarray
    ) -> None:
        ...

    @pytest.mark.parametrize('freq', ['h', 'D', 'W', '2ME', 'MS', 'QE', 'B', None])
    @pytest.mark.parametrize('dtype', [type, None])
    def test_dt64arr_addsub_intlike(
        self, dtype: type | None, 
        index_or_series_or_array: pd.Index | pd.Series | np.ndarray, 
        freq: str | None, 
        tz_naive_fixture: timezone
    ) -> None:
        ...

    @pytest.mark.parametrize('other', [float, np.ndarray, Period, time])
    @pytest.mark.parametrize('dti_freq', [None, 'D'])
    def test_dt64arr_add_sub_invalid(
        self, dti_freq: str | None, 
        other: Union[float, np.ndarray, Period, time], 
        box_with_array: pd.DataFrame | pd.Series | np.ndarray
    ) -> None:
        ...

    @pytest.mark.parametrize('pi_freq', ['D', 'W', 'Q', 'h'])
    @pytest.mark.parametrize('dti_freq', [None, 'D'])
    def test_dt64arr_add_sub_parr(
        self, dti_freq: str | None, 
        pi_freq: str, 
        box_with_array: pd.DataFrame | pd.Series | np.ndarray
    ) -> None:
        ...

    def test_dt64arr_addsub_time_objects_raises(
        self, box_with_array: pd.DataFrame | pd.Series | np.ndarray, 
        tz_naive_fixture: timezone
    ) -> None:
        ...

    @pytest.mark.parametrize('dt64_series', [pd.Series, pd.Series, pd.Series])
    @pytest.mark.parametrize('one', [int, float, np.ndarray])
    def test_dt64_mul_div_numeric_invalid(
        self, one: Union[int, float, np.ndarray], 
        dt64_series: pd.Series, 
        box_with_array: pd.DataFrame | pd.Series | np.ndarray
    ) -> None:
        ...

class TestDatetime64DateOffsetArithmetic:
    def test_dt64arr_series_add_tick_DateOffset(
        self, box_with_array: pd.DataFrame | pd.Series | np.ndarray, 
        unit: str
    ) -> None:
        ...

    def test_dt64arr_series_sub_tick_DateOffset(
        self, box_with_array: pd.DataFrame | pd.Series | np.ndarray
    ) -> None:
        ...

    @pytest.mark.parametrize('cls_name', ['Day', 'Hour', 'Minute', 'Second', 'Milli', 'Micro', 'Nano'])
    def test_dt64arr_add_sub_tick_DateOffset_smoke(
        self, cls_name: str, 
        box_with_array: pd.DataFrame | pd.Series | np.ndarray
    ) -> None:
        ...

    def test_dti_add_tick_tzaware(
        self, tz_aware_fixture: timezone, 
        box_with_array: pd.DataFrame | pd.Series | np.ndarray
    ) -> None:
        ...

    def test_dt64arr_add_sub_relativedelta_offsets(
        self, box_with_array: pd.DataFrame | pd.Series | np.ndarray, 
        unit: str
    ) -> None:
        ...

    @pytest.mark.filterwarnings('ignore::pandas.errors.PerformanceWarning')
    @pytest.mark.parametrize('cls_and_kwargs', ['YearBegin', ('YearBegin', {'month': 5}), 'YearEnd', ('YearEnd', {'month': 5}), 'MonthBegin', 'MonthEnd', 'SemiMonthEnd', 'SemiMonthBegin', 'Week', ('Week', {'weekday': 3}), ('Week', {'weekday': 6}), 'BusinessDay', 'BDay', 'QuarterEnd', 'QuarterBegin', 'CustomBusinessDay', 'CDay', 'CBMonthEnd', 'CBMonthBegin', 'BMonthBegin', 'BMonthEnd', 'BusinessHour', 'BYearBegin', 'BYearEnd', 'BQuarterBegin', ('LastWeekOfMonth', {'weekday': 2}), ('FY5253Quarter', {'qtr_with_extra_week': 1, 'startingMonth': 1, 'weekday': 2, 'variation': 'nearest'}), ('FY5253', {'weekday': 0, 'startingMonth': 2, 'variation': 'nearest'}), ('WeekOfMonth', {'weekday': 2, 'week': 2}), 'Easter', ('DateOffset', {'day': 4}), ('DateOffset', {'month': 5})])
    @pytest.mark.parametrize('normalize', [bool])
    @pytest.mark.parametrize('n', [int])
    @pytest.mark.parametrize('tz', [str, None])
    def test_dt64arr_add_sub_DateOffsets(
        self, box_with_array: pd.DataFrame | pd.Series | np.ndarray, 
        n: int, 
        normalize: bool, 
        cls_and_kwargs: str | tuple, 
        unit: str, 
        tz: str | None
    ) -> None:
        ...

    @pytest.mark.parametrize('other', [list, np.ndarray])
    @pytest.mark.parametrize('op', [operator.add, roperator.radd, operator.sub])
    def test_dt64arr_add_sub_offset_array(
        self, performance_warning: pytest.Warning, 
        tz_naive_fixture: timezone, 
        box_with_array: pd.DataFrame | pd.Series | np.ndarray, 
        op: operator, 
        other: Union[list, np.ndarray]
    ) -> None:
        ...

    @pytest.mark.parametrize('op, offset, exp, exp_freq', [('__add__', DateOffset, [Timestamp, Timestamp, Timestamp, Timestamp], None), ('__add__', DateOffset, [Timestamp, Timestamp, Timestamp, Timestamp], 'YS-APR'), ('__sub__', DateOffset, [Timestamp, Timestamp, Timestamp, Timestamp], None), ('__sub__', DateOffset, [Timestamp, Timestamp, Timestamp, Timestamp], 'YS-OCT')])
    def test_dti_add_sub_nonzero_mth_offset(
        self, op: str, 
        offset: DateOffset, 
        exp: list, 
        exp_freq: str | None, 
        tz_aware_fixture: timezone, 
        box_with_array: pd.DataFrame | pd.Series | np.ndarray
    ) -> None:
        ...

    def test_dt64arr_series_add_DateOffset_with_milli(self) -> None:
        ...

    def test_datetimeindex_sub_timestamp_overflow(self) -> None:
        ...

    def test_datetimeindex_sub_datetimeindex_overflow(self) -> None:
        ...

class TestTimestampSeriesArithmetic:
    def test_empty_series_add_sub(
        self, box_with_array: pd.DataFrame | pd.Series | np.ndarray
    ) -> None:
        ...

    def test_operators_datetimelike(self) -> None:
        ...

    def test_operators_datetimelike_invalid(
        self, left: list, 
        right: list, 
        op_fail: list, 
        all_arithmetic_operators: operator
    ) -> None:
        ...

    def test_sub_single_tz(self, unit: str) -> None:
        ...

    def test_dt64tz_series_sub_dtitz(self) -> None:
        ...

    def test_sub_datetime_compat(self, unit: str) -> None:
        ...

    def test_dt64tz_series_sub_dtitz(self) -> None:
        ...

    def test_sub_datetime_compat(self, unit: str) -> None:
        ...

    def test_dt64_series_add_mixed_tick_DateOffset(self) -> None:
        ...

    def test_datetime64_ops_nat(self, unit: str) -> None:
        ...

    def test_operators_datetimelike_with_timezones(self) -> None:
        ...

class TestDatetimeIndexArithmetic:
    def test_dti_add_tdi(
        self, tz_naive_fixture: timezone
    ) -> None:
        ...

    def test_dti_iadd_tdi(
        self, tz_naive_fixture: timezone
    ) -> None:
        ...

    def test_dti_sub_tdi(
        self, tz_naive_fixture: timezone
    ) -> None:
        ...

    def test_dti_isub_tdi(
        self, tz_naive_fixture: timezone, 
        unit: str
    ) -> None:
        ...

    def test_dta_add_sub_index(
        self, tz_naive_fixture: timezone
    ) -> None:
        ...

    def test_sub_dti_dti(
        self, unit: str
    ) -> None:
        ...

    @pytest.mark.parametrize('op', [operator.add, operator.sub])
    def test_timedelta64_equal_timedelta_supported_ops(
        self, op: operator, 
        box_with_array: pd.DataFrame | pd.Series | np.ndarray
    ) -> None:
        ...

    def test_ops_nat_mixed_datetime64_timedelta64(self) -> None:
        ...

    def test_ufunc_coercions(self, unit: str) -> None:
        ...

    def test_dti_add_series(
        self, tz_naive_fixture: timezone, 
        names: list
    ) -> None:
        ...

    @pytest.mark.parametrize('op', [operator.add, roperator.radd, operator.sub])
    def test_dti_addsub_offset_arraylike(
        self, performance_warning: pytest.Warning, 
        tz_naive_fixture: timezone, 
        names: list, 
        op: operator, 
        index_or_series: pd.Index | pd.Series
    ) -> None:
        ...

    @pytest.mark.parametrize('other_box', [pd.Index, np.array])
    def test_dti_addsub_object_arraylike(
        self, performance_warning: pytest.Warning, 
        tz_naive_fixture: timezone, 
        box_with_array: pd.DataFrame | pd.Series | np.ndarray, 
        other_box: type
    ) -> None:
        ...

@pytest.mark.parametrize('years', [-1, 0, 1])
@pytest.mark.parametrize('months', [-2, 0, 2])
def test_shift_months(years: int, months: int, unit: str) -> None:
    ...

def test_dt64arr_addsub_object_dtype_2d(performance_warning: pytest.Warning) -> None:
    ...

def test_non_nano_dt64_addsub_np_nat_scalars() -> None:
    ...

def test_non_nano_dt64_addsub_np_nat_scalars_unitless() -> None:
    ...

def test_non_nano_dt64_addsub_np_nat_scalars_unsupported_unit() -> None:
    ...