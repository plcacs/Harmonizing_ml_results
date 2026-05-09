"""
Stub file for test_index_new_74deaa module.
"""

from datetime import datetime, timedelta, timezone
from decimal import Decimal
import numpy as np
import pytest
from pandas._libs.tslibs.timezones import maybe_get_tz
from pandas import (
    NA,
    Categorical,
    CategoricalIndex,
    DatetimeIndex,
    Index,
    IntervalIndex,
    MultiIndex,
    NaT,
    PeriodIndex,
    Series,
    TimedeltaIndex,
    Timestamp,
    array,
    date_range,
    period_range,
    timedelta_range,
)
import pandas._testing as tm

class TestIndexConstructorInference:
    def test_object_all_bools(self) -> None:
        ...

    def test_object_all_complex(self) -> None:
        ...

    @pytest.mark.parametrize('val', [NaT, None, np.nan, float('nan')])
    def test_infer_nat(self, val: Any) -> None:
        ...

    @pytest.mark.parametrize('na_value', [None, np.nan])
    @pytest.mark.parametrize('vtype', [list, tuple, iter])
    def test_construction_list_tuples_nan(self, na_value: Optional[Any], vtype: Any) -> None:
        ...

    def test_constructor_int_dtype_float(self, any_int_numpy_dtype: np.dtype) -> None:
        ...

    @pytest.mark.parametrize('cast_index', [True, False])
    @pytest.mark.parametrize('vals', [[True, False, True], np.array([True, False, True], dtype=bool)])
    def test_constructor_dtypes_to_object(self, cast_index: bool, vals: Union[List[bool], np.ndarray]) -> None:
        ...

    def test_constructor_categorical_to_object(self) -> None:
        ...

    def test_constructor_infer_periodindex(self, xp: PeriodIndex) -> None:
        ...

    def test_from_list_of_periods(self, rng: PeriodIndex) -> None:
        ...

    @pytest.mark.parametrize('pos', [0, 1])
    @pytest.mark.parametrize('klass,dtype,ctor', [(DatetimeIndex, 'datetime64[ns]', np.datetime64('nat')), (TimedeltaIndex, 'timedelta64[ns]', np.timedelta64('nat'))])
    def test_constructor_infer_nat_dt_like(self, pos: int, klass: Union[Type[DatetimeIndex], Type[TimedeltaIndex]], dtype: str, ctor: Union[np.datetime64, np.timedelta64], nulls_fixture: Any, request: pytest.FixtureRequest) -> None:
        ...

    @pytest.mark.parametrize('swap_objs', [True, False])
    def test_constructor_mixed_nat_objs_infers_object(self, swap_objs: bool) -> None:
        ...

    @pytest.mark.parametrize('swap_objs', [True, False])
    def test_constructor_datetime_and_datetime64(self, swap_objs: bool) -> None:
        ...

    def test_constructor_datetimes_mixed_tzs(self, tz: timezone) -> None:
        ...

class TestDtypeEnforced:
    def test_constructor_object_dtype_with_ea_data(self, any_numeric_ea_dtype: np.dtype) -> None:
        ...

    @pytest.mark.parametrize('dtype', [object, 'float64', 'uint64', 'category'])
    def test_constructor_range_values_mismatched_dtype(self, dtype: Union[type, str]) -> None:
        ...

    @pytest.mark.parametrize('dtype', [object, 'float64', 'uint64', 'category'])
    def test_constructor_categorical_values_mismatched_non_ea_dtype(self, dtype: Union[type, str]) -> None:
        ...

    def test_constructor_categorical_values_mismatched_dtype(self) -> None:
        ...

    def test_constructor_ea_values_mismatched_categorical_dtype(self) -> None:
        ...

    def test_constructor_period_values_mismatched_dtype(self) -> None:
        ...

    def test_constructor_timedelta64_values_mismatched_dtype(self) -> None:
        ...

    def test_constructor_interval_values_mismatched_dtype(self) -> None:
        ...

    def test_constructor_datetime64_values_mismatched_period_dtype(self) -> None:
        ...

    @pytest.mark.parametrize('dtype', ['int64', 'uint64'])
    def test_constructor_int_dtype_nan_raises(self, dtype: str) -> None:
        ...

    @pytest.mark.parametrize('vals', [[1, 2, 3], np.array([1, 2, 3]), np.array([1, 2, 3], dtype=int), [1.0, 2.0, 3.0], np.array([1.0, 2.0, 3.0], dtype=float)])
    def test_constructor_dtypes_to_int(self, vals: Union[List[int], np.ndarray], any_int_numpy_dtype: np.dtype) -> None:
        ...

    @pytest.mark.parametrize('vals', [[1, 2, 3], [1.0, 2.0, 3.0], np.array([1.0, 2.0, 3.0]), np.array([1, 2, 3], dtype=int), np.array([1.0, 2.0, 3.0], dtype=float)])
    def test_constructor_dtypes_to_float(self, vals: Union[List[float], np.ndarray], float_numpy_dtype: np.dtype) -> None:
        ...

    @pytest.mark.parametrize('vals', [[1, 2, 3], np.array([1, 2, 3], dtype=int), np.array(['2011-01-01', '2011-01-02'], dtype='datetime64[ns]'), [datetime(2011, 1, 1), datetime(2011, 1, 2)]])
    def test_constructor_dtypes_to_categorical(self, vals: Union[List[int], np.ndarray, List[datetime]]) -> None:
        ...

    @pytest.mark.parametrize('cast_index', [True, False])
    @pytest.mark.parametrize('vals', [np.array([np.datetime64('2011-01-01'), np.datetime64('2011-01-02')]), [datetime(2011, 1, 1), datetime(2011, 1, 2)]])
    def test_constructor_dtypes_to_datetime(self, cast_index: bool, vals: Union[np.ndarray, List[datetime]]) -> None:
        ...

    @pytest.mark.parametrize('cast_index', [True, False])
    @pytest.mark.parametrize('vals', [np.array([np.timedelta64(1, 'D'), np.timedelta64(1, 'D')]), [timedelta(1), timedelta(1)]])
    def test_constructor_dtypes_to_timedelta(self, cast_index: bool, vals: Union[np.ndarray, List[timedelta]]) -> None:
        ...

    def test_pass_timedeltaindex_to_index(self, rng: TimedeltaIndex) -> None:
        ...

    def test_pass_datetimeindex_to_index(self, rng: DatetimeIndex) -> None:
        ...

class TestIndexConstructorUnwrapping:
    @pytest.mark.parametrize('klass', [Index, DatetimeIndex])
    def test_constructor_from_series_dt64(self, klass: Union[Type[Index], Type[DatetimeIndex]]) -> None:
        ...

    def test_constructor_no_pandas_array(self) -> None:
        ...

    @pytest.mark.parametrize('array', [np.arange(5), np.array(['a', 'b', 'c']), date_range('2000-01-01', periods=3).values])
    def test_constructor_ndarray_like(self, array: Union[np.ndarray, List[datetime]]) -> None:
        ...

class TestIndexConstructionErrors:
    def test_constructor_overflow_int64(self) -> None:
        ...