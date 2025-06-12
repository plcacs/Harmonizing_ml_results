"""
Tests for the Index constructor conducting inference.
"""
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Iterable, List, Tuple, Type
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
        arr: np.ndarray = np.array([True, False], dtype=object)
        res: Index = Index(arr)
        assert res.dtype == object
        assert Series(arr).dtype == object

    def test_object_all_complex(self) -> None:
        arr: np.ndarray = np.array([complex(1), complex(2)], dtype=object)
        res: Index = Index(arr)
        assert res.dtype == object
        assert Series(arr).dtype == object

    @pytest.mark.parametrize('val', [NaT, None, np.nan, float('nan')])
    def test_infer_nat(self, val: Any) -> None:
        values: List[Any] = [NaT, val]
        idx: Index = Index(values)
        assert idx.dtype == 'datetime64[s]' and idx.isna().all()
        idx = Index(values[::-1])
        assert idx.dtype == 'datetime64[s]' and idx.isna().all()
        idx = Index(np.array(values, dtype=object))
        assert idx.dtype == 'datetime64[s]' and idx.isna().all()
        idx = Index(np.array(values, dtype=object)[::-1])
        assert idx.dtype == 'datetime64[s]' and idx.isna().all()

    @pytest.mark.parametrize('na_value', [None, np.nan])
    @pytest.mark.parametrize('vtype', [list, tuple, iter])
    def test_construction_list_tuples_nan(self, na_value: Any, vtype: Type[Iterable[Any]]) -> None:
        values: List[Any] = [(1, 'two'), (3.0, na_value)]
        result: Index = Index(vtype(values))
        expected: MultiIndex = MultiIndex.from_tuples(values)
        tm.assert_index_equal(result, expected)

    def test_constructor_int_dtype_float(self, any_int_numpy_dtype: np.dtype) -> None:
        expected: Index = Index([0, 1, 2, 3], dtype=any_int_numpy_dtype)
        result: Index = Index([0.0, 1.0, 2.0, 3.0], dtype=any_int_numpy_dtype)
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize('cast_index', [True, False])
    @pytest.mark.parametrize('vals', [[True, False, True], np.array([True, False, True], dtype=bool)])
    def test_constructor_dtypes_to_object(self, cast_index: bool, vals: Any) -> None:
        if cast_index:
            index: Index = Index(vals, dtype=bool)
        else:
            index = Index(vals)
        assert type(index) is Index
        assert index.dtype == bool

    def test_constructor_categorical_to_object(self) -> None:
        ci: CategoricalIndex = CategoricalIndex(range(5))
        result: Index = Index(ci, dtype=object)
        assert not isinstance(result, CategoricalIndex)

    def test_constructor_infer_periodindex(self) -> None:
        xp: PeriodIndex = period_range('2012-1-1', freq='M', periods=3)
        rs: Index = Index(xp)
        tm.assert_index_equal(rs, xp)
        assert isinstance(rs, PeriodIndex)

    def test_from_list_of_periods(self) -> None:
        rng: PeriodIndex = period_range('1/1/2000', periods=20, freq='D')
        periods: List[Timestamp] = list(rng)
        result: Index = Index(periods)
        assert isinstance(result, PeriodIndex)

    @pytest.mark.parametrize('pos', [0, 1])
    @pytest.mark.parametrize(
        'klass,dtype,ctor',
        [
            (DatetimeIndex, 'datetime64[ns]', np.datetime64('nat')),
            (TimedeltaIndex, 'timedelta64[ns]', np.timedelta64('nat')),
        ],
    )
    def test_constructor_infer_nat_dt_like(
        self,
        pos: int,
        klass: Type[Index],
        dtype: str,
        ctor: Any,
        nulls_fixture: Any,
        request: pytest.FixtureRequest,
    ) -> None:
        if isinstance(nulls_fixture, Decimal):
            pytest.skip(f"We don't cast {type(nulls_fixture).__name__} to datetime64/timedelta64")
        expected: Index = klass([NaT, NaT])
        if dtype[0] == 'd':
            expected = expected.astype('M8[ns]')
        assert expected.dtype == dtype
        data: List[Any] = [ctor]
        data.insert(pos, nulls_fixture)
        if nulls_fixture is NA:
            expected = Index([NA, NaT])
            mark = pytest.mark.xfail(reason='Broken with np.NaT ctor; see GH 31884')
            request.applymarker(mark)
        result: Index = Index(data)
        tm.assert_index_equal(result, expected)
        result = Index(np.array(data, dtype=object))
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize('swap_objs', [True, False])
    def test_constructor_mixed_nat_objs_infers_object(self, swap_objs: bool) -> None:
        data: List[Any] = [np.datetime64('nat'), np.timedelta64('nat')]
        if swap_objs:
            data = data[::-1]
        expected: Index = Index(data, dtype=object)
        tm.assert_index_equal(Index(data), expected)
        tm.assert_index_equal(Index(np.array(data, dtype=object)), expected)

    @pytest.mark.parametrize('swap_objs', [True, False])
    def test_constructor_datetime_and_datetime64(self, swap_objs: bool) -> None:
        data: List[Any] = [Timestamp(2021, 6, 8, 9, 42), np.datetime64('now')]
        if swap_objs:
            data = data[::-1]
        expected: DatetimeIndex = DatetimeIndex(data)
        tm.assert_index_equal(Index(data), expected)
        tm.assert_index_equal(Index(np.array(data, dtype=object)), expected)

    def test_constructor_datetimes_mixed_tzs(self) -> None:
        tz = maybe_get_tz('US/Central')
        dt1: datetime = datetime(2020, 1, 1, tzinfo=tz)
        dt2: datetime = datetime(2020, 1, 1, tzinfo=timezone.utc)
        result: Index = Index([dt1, dt2])
        expected: Index = Index([dt1, dt2], dtype=object)
        tm.assert_index_equal(result, expected)


class TestDtypeEnforced:

    def test_constructor_object_dtype_with_ea_data(self, any_numeric_ea_dtype: np.dtype) -> None:
        arr: Any = array([0], dtype=any_numeric_ea_dtype)
        idx: Index = Index(arr, dtype=object)
        assert idx.dtype == object

    @pytest.mark.parametrize('dtype', [object, 'float64', 'uint64', 'category'])
    def test_constructor_range_values_mismatched_dtype(self, dtype: Any) -> None:
        rng: Index = Index(range(5))
        result: Index = Index(rng, dtype=dtype)
        assert result.dtype == dtype
        result = Index(range(5), dtype=dtype)
        assert result.dtype == dtype

    @pytest.mark.parametrize('dtype', [object, 'float64', 'uint64', 'category'])
    def test_constructor_categorical_values_mismatched_non_ea_dtype(self, dtype: Any) -> None:
        cat: Categorical = Categorical([1, 2, 3])
        result: Index = Index(cat, dtype=dtype)
        assert result.dtype == dtype

    def test_constructor_categorical_values_mismatched_dtype(self) -> None:
        dti: DatetimeIndex = date_range('2016-01-01', periods=3)
        cat: Categorical = Categorical(dti)
        result: Index = Index(cat, dti.dtype)
        tm.assert_index_equal(result, dti)
        dti2: DatetimeIndex = dti.tz_localize('Asia/Tokyo')
        cat2: Categorical = Categorical(dti2)
        result = Index(cat2, dti2.dtype)
        tm.assert_index_equal(result, dti2)
        ii: IntervalIndex = IntervalIndex.from_breaks(range(5))
        cat3: Categorical = Categorical(ii)
        result = Index(cat3, dtype=ii.dtype)
        tm.assert_index_equal(result, ii)

    def test_constructor_ea_values_mismatched_categorical_dtype(self) -> None:
        dti: DatetimeIndex = date_range('2016-01-01', periods=3)
        result: Index = Index(dti, dtype='category')
        expected: CategoricalIndex = CategoricalIndex(dti)
        tm.assert_index_equal(result, expected)
        dti2: DatetimeIndex = date_range('2016-01-01', periods=3, tz='US/Pacific')
        result = Index(dti2, dtype='category')
        expected = CategoricalIndex(dti2)
        tm.assert_index_equal(result, expected)

    def test_constructor_period_values_mismatched_dtype(self) -> None:
        pi: PeriodIndex = period_range('2016-01-01', periods=3, freq='D')
        result: Index = Index(pi, dtype='category')
        expected: CategoricalIndex = CategoricalIndex(pi)
        tm.assert_index_equal(result, expected)

    def test_constructor_timedelta64_values_mismatched_dtype(self) -> None:
        tdi: TimedeltaIndex = timedelta_range('4 Days', periods=5)
        result: Index = Index(tdi, dtype='category')
        expected: CategoricalIndex = CategoricalIndex(tdi)
        tm.assert_index_equal(result, expected)

    def test_constructor_interval_values_mismatched_dtype(self) -> None:
        dti: DatetimeIndex = date_range('2016-01-01', periods=3)
        ii: IntervalIndex = IntervalIndex.from_breaks(dti)
        result: Index = Index(ii, dtype='category')
        expected: CategoricalIndex = CategoricalIndex(ii)
        tm.assert_index_equal(result, expected)

    def test_constructor_datetime64_values_mismatched_period_dtype(self) -> None:
        dti: DatetimeIndex = date_range('2016-01-01', periods=3)
        result: Index = Index(dti, dtype='Period[D]')
        expected: PeriodIndex = dti.to_period('D')
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize('dtype', ['int64', 'uint64'])
    def test_constructor_int_dtype_nan_raises(self, dtype: str) -> None:
        data: List[float] = [np.nan]
        msg: str = 'cannot convert'
        with pytest.raises(ValueError, match=msg):
            Index(data, dtype=dtype)

    @pytest.mark.parametrize(
        'vals',
        [
            [1, 2, 3],
            np.array([1, 2, 3]),
            np.array([1, 2, 3], dtype=int),
            [1.0, 2.0, 3.0],
            np.array([1.0, 2.0, 3.0], dtype=float),
        ],
    )
    def test_constructor_dtypes_to_int(self, vals: Any, any_int_numpy_dtype: np.dtype) -> None:
        dtype: np.dtype = any_int_numpy_dtype
        index: Index = Index(vals, dtype=dtype)
        assert index.dtype == dtype

    @pytest.mark.parametrize(
        'vals',
        [
            [1, 2, 3],
            [1.0, 2.0, 3.0],
            np.array([1.0, 2.0, 3.0]),
            np.array([1, 2, 3], dtype=int),
            np.array([1.0, 2.0, 3.0], dtype=float),
        ],
    )
    def test_constructor_dtypes_to_float(self, vals: Any, float_numpy_dtype: np.dtype) -> None:
        dtype: np.dtype = float_numpy_dtype
        index: Index = Index(vals, dtype=dtype)
        assert index.dtype == dtype

    @pytest.mark.parametrize(
        'vals',
        [
            [1, 2, 3],
            np.array([1, 2, 3], dtype=int),
            np.array(['2011-01-01', '2011-01-02'], dtype='datetime64[ns]'),
            [datetime(2011, 1, 1), datetime(2011, 1, 2)],
        ],
    )
    def test_constructor_dtypes_to_categorical(self, vals: Any) -> None:
        index: CategoricalIndex = Index(vals, dtype='category')
        assert isinstance(index, CategoricalIndex)

    @pytest.mark.parametrize('cast_index', [True, False])
    @pytest.mark.parametrize(
        'vals',
        [
            np.array([np.datetime64('2011-01-01'), np.datetime64('2011-01-02')]),
            [datetime(2011, 1, 1), datetime(2011, 1, 2)],
        ],
    )
    def test_constructor_dtypes_to_datetime(self, cast_index: bool, vals: Any) -> None:
        vals = Index(vals)
        if cast_index:
            index: Index = Index(vals, dtype=object)
            assert isinstance(index, Index)
            assert index.dtype == object
        else:
            index = Index(vals)
            assert isinstance(index, DatetimeIndex)

    @pytest.mark.parametrize('cast_index', [True, False])
    @pytest.mark.parametrize(
        'vals',
        [
            np.array([np.timedelta64(1, 'D'), np.timedelta64(1, 'D')]),
            [timedelta(1), timedelta(1)],
        ],
    )
    def test_constructor_dtypes_to_timedelta(self, cast_index: bool, vals: Any) -> None:
        if cast_index:
            index: Index = Index(vals, dtype=object)
            assert isinstance(index, Index)
            assert index.dtype == object
        else:
            index = Index(vals)
            assert isinstance(index, TimedeltaIndex)

    def test_pass_timedeltaindex_to_index(self) -> None:
        rng: TimedeltaIndex = timedelta_range('1 days', '10 days')
        idx: Index = Index(rng, dtype=object)
        expected: Index = Index(rng.to_pytimedelta(), dtype=object)
        tm.assert_numpy_array_equal(idx.values, expected.values)

    def test_pass_datetimeindex_to_index(self) -> None:
        rng: DatetimeIndex = date_range('1/1/2000', '3/1/2000')
        idx: Index = Index(rng, dtype=object)
        expected: Index = Index(rng.to_pydatetime(), dtype=object)
        tm.assert_numpy_array_equal(idx.values, expected.values)


class TestIndexConstructorUnwrapping:

    @pytest.mark.parametrize('klass', [Index, DatetimeIndex])
    def test_constructor_from_series_dt64(self, klass: Type[Index]) -> None:
        stamps: List[Timestamp] = [
            Timestamp('20110101'),
            Timestamp('20120101'),
            Timestamp('20130101'),
        ]
        expected: DatetimeIndex = DatetimeIndex(stamps)
        ser: Series = Series(stamps)
        result: Index = klass(ser)
        tm.assert_index_equal(result, expected)

    def test_constructor_no_pandas_array(self) -> None:
        ser: Series = Series([1, 2, 3])
        result: Index = Index(ser.array)
        expected: Index = Index([1, 2, 3])
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize(
        'array',
        [
            np.arange(5),
            np.array(['a', 'b', 'c']),
            date_range('2000-01-01', periods=3).values,
        ],
    )
    def test_constructor_ndarray_like(self, array: Any) -> None:
        class ArrayLike:

            def __init__(self, array: Any) -> None:
                self.array = array

            def __array__(self, dtype: Any = None, copy: Any = None) -> Any:
                return self.array

        expected: Index = Index(array)
        result: Index = Index(ArrayLike(array))
        tm.assert_index_equal(result, expected)


class TestIndexConstructionErrors:

    def test_constructor_overflow_int64(self) -> None:
        msg: str = 'The elements provided in the data cannot all be casted to the dtype int64'
        with pytest.raises(OverflowError, match=msg):
            Index([np.iinfo(np.uint64).max - 1], dtype='int64')
