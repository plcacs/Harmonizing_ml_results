from collections import defaultdict
from datetime import datetime
from functools import partial
import math
import operator
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Type, TypeVar, Union, cast
import numpy as np
import pytest
from pandas.compat import IS64
from pandas.errors import InvalidIndexError
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_any_real_numeric_dtype, is_numeric_dtype, is_object_dtype
import pandas as pd
from pandas import CategoricalIndex, DataFrame, DatetimeIndex, IntervalIndex, PeriodIndex, RangeIndex, Series, TimedeltaIndex, date_range, period_range, timedelta_range
import pandas._testing as tm
from pandas.core.indexes.api import Index, MultiIndex, _get_combined_index, ensure_index, ensure_index_from_sequences

T = TypeVar('T')

class TestIndex:

    @pytest.fixture
    def simple_index(self) -> Index:
        return Index(list('abcde'))

    def test_can_hold_identifiers(self, simple_index: Index) -> None:
        index = simple_index
        key = index[0]
        assert index._can_hold_identifiers_and_holds_name(key) is True

    @pytest.mark.parametrize('index', ['datetime'], indirect=True)
    def test_new_axis(self, index: Index) -> None:
        with pytest.raises(ValueError, match='Multi-dimensional indexing'):
            index[None, :]

    def test_constructor_regular(self, index: Index) -> None:
        tm.assert_contains_all(index, index)

    @pytest.mark.parametrize('index', ['string'], indirect=True)
    def test_constructor_casting(self, index: Index) -> None:
        arr = np.array(index)
        new_index = Index(arr)
        tm.assert_contains_all(arr, new_index)
        tm.assert_index_equal(index, new_index)

    def test_constructor_copy(self, using_infer_string: bool) -> None:
        index = Index(list('abc'), name='name')
        arr = np.array(index)
        new_index = Index(arr, copy=True, name='name')
        assert isinstance(new_index, Index)
        assert new_index.name == 'name'
        if using_infer_string:
            tm.assert_extension_array_equal(new_index.values, pd.array(arr, dtype='str'))
        else:
            tm.assert_numpy_array_equal(arr, new_index.values)
        arr[0] = 'SOMEBIGLONGSTRING'
        assert new_index[0] != 'SOMEBIGLONGSTRING'

    @pytest.mark.parametrize('cast_as_obj', [True, False])
    @pytest.mark.parametrize('index', [date_range('2015-01-01 10:00', freq='D', periods=3, tz='US/Eastern', name='Green Eggs & Ham'), date_range('2015-01-01 10:00', freq='D', periods=3), timedelta_range('1 days', freq='D', periods=3), period_range('2015-01-01', freq='D', periods=3)])
    def test_constructor_from_index_dtlike(self, cast_as_obj: bool, index: Index) -> None:
        if cast_as_obj:
            result = Index(index.astype(object))
            assert result.dtype == np.dtype(object)
            if isinstance(index, DatetimeIndex):
                index += pd.Timedelta(nanoseconds=50)
                result = Index(index, dtype=object)
                assert result.dtype == np.object_
                assert list(result) == list(index)
        else:
            result = Index(index)
            tm.assert_index_equal(result, index)

    @pytest.mark.parametrize('index,has_tz', [(date_range('2015-01-01 10:00', freq='D', periods=3, tz='US/Eastern'), True), (timedelta_range('1 days', freq='D', periods=3), False), (period_range('2015-01-01', freq='D', periods=3), False)])
    def test_constructor_from_series_dtlike(self, index: Index, has_tz: bool) -> None:
        result = Index(Series(index))
        tm.assert_index_equal(result, index)
        if has_tz:
            assert result.tz == index.tz

    def test_constructor_from_series_freq(self) -> None:
        dts = ['1-1-1990', '2-1-1990', '3-1-1990', '4-1-1990', '5-1-1990']
        expected = DatetimeIndex(dts, freq='MS')
        s = Series(pd.to_datetime(dts))
        result = DatetimeIndex(s, freq='MS')
        tm.assert_index_equal(result, expected)

    def test_constructor_from_frame_series_freq(self, using_infer_string: bool) -> None:
        dts = ['1-1-1990', '2-1-1990', '3-1-1990', '4-1-1990', '5-1-1990']
        expected = DatetimeIndex(dts, freq='MS')
        df = DataFrame(np.random.default_rng(2).random((5, 3)))
        df['date'] = dts
        result = DatetimeIndex(df['date'], freq='MS')
        dtype = object if not using_infer_string else 'str'
        assert df['date'].dtype == dtype
        expected.name = 'date'
        tm.assert_index_equal(result, expected)
        expected = Series(dts, name='date')
        tm.assert_series_equal(df['date'], expected)
        if not using_infer_string:
            freq = pd.infer_freq(df['date'])
            assert freq == 'MS'

    def test_constructor_int_dtype_nan(self) -> None:
        data = [np.nan]
        expected = Index(data, dtype=np.float64)
        result = Index(data, dtype='float')
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize('klass,dtype,na_val', [(Index, np.float64, np.nan), (DatetimeIndex, 'datetime64[s]', pd.NaT)])
    def test_index_ctor_infer_nan_nat(self, klass: Type[Index], dtype: str, na_val: Any) -> None:
        na_list = [na_val, na_val]
        expected = klass(na_list)
        assert expected.dtype == dtype
        result = Index(na_list)
        tm.assert_index_equal(result, expected)
        result = Index(np.array(na_list))
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize('vals,dtype', [([1, 2, 3, 4, 5], 'int'), ([1.1, np.nan, 2.2, 3.0], 'float'), (['A', 'B', 'C', np.nan], 'obj')])
    def test_constructor_simple_new(self, vals: List[Any], dtype: str) -> None:
        index = Index(vals, name=dtype)
        result = index._simple_new(index.values, dtype)
        tm.assert_index_equal(result, index)

    @pytest.mark.parametrize('attr', ['values', 'asi8'])
    @pytest.mark.parametrize('klass', [Index, DatetimeIndex])
    def test_constructor_dtypes_datetime(self, tz_naive_fixture: Any, attr: str, klass: Type[Index]) -> None:
        index = date_range('2011-01-01', periods=5)
        arg = getattr(index, attr)
        index = index.tz_localize(tz_naive_fixture)
        dtype = index.dtype
        err = tz_naive_fixture is not None
        msg = 'Cannot use .astype to convert from timezone-naive dtype to'
        if attr == 'asi8':
            result = DatetimeIndex(arg).tz_localize(tz_naive_fixture)
            tm.assert_index_equal(result, index)
        elif klass is Index:
            with pytest.raises(TypeError, match='unexpected keyword'):
                klass(arg, tz=tz_naive_fixture)
        else:
            result = klass(arg, tz=tz_naive_fixture)
            tm.assert_index_equal(result, index)
        if attr == 'asi8':
            if err:
                with pytest.raises(TypeError, match=msg):
                    DatetimeIndex(arg).astype(dtype)
            else:
                result = DatetimeIndex(arg).astype(dtype)
                tm.assert_index_equal(result, index)
        else:
            result = klass(arg, dtype=dtype)
            tm.assert_index_equal(result, index)
        if attr == 'asi8':
            result = DatetimeIndex(list(arg)).tz_localize(tz_naive_fixture)
            tm.assert_index_equal(result, index)
        elif klass is Index:
            with pytest.raises(TypeError, match='unexpected keyword'):
                klass(arg, tz=tz_naive_fixture)
        else:
            result = klass(list(arg), tz=tz_naive_fixture)
            tm.assert_index_equal(result, index)
        if attr == 'asi8':
            if err:
                with pytest.raises(TypeError, match=msg):
                    DatetimeIndex(list(arg)).astype(dtype)
            else:
                result = DatetimeIndex(list(arg)).astype(dtype)
                tm.assert_index_equal(result, index)
        else:
            result = klass(list(arg), dtype=dtype)
            tm.assert_index_equal(result, index)

    @pytest.mark.parametrize('attr', ['values', 'asi8'])
    @pytest.mark.parametrize('klass', [Index, TimedeltaIndex])
    def test_constructor_dtypes_timedelta(self, attr: str, klass: Type[Index]) -> None:
        index = timedelta_range('1 days', periods=5)
        index = index._with_freq(None)
        dtype = index.dtype
        values = getattr(index, attr)
        result = klass(values, dtype=dtype)
        tm.assert_index_equal(result, index)
        result = klass(list(values), dtype=dtype)
        tm.assert_index_equal(result, index)

    @pytest.mark.parametrize('value', [[], iter([]), (_ for _ in [])])
    @pytest.mark.parametrize('klass', [Index, CategoricalIndex, DatetimeIndex, TimedeltaIndex])
    def test_constructor_empty(self, value: Any, klass: Type[Index]) -> None:
        empty = klass(value)
        assert isinstance(empty, klass)
        assert not len(empty)

    @pytest.mark.parametrize('empty,klass', [(PeriodIndex([], freq='D'), PeriodIndex), (PeriodIndex(iter([]), freq='D'), PeriodIndex), (PeriodIndex((_ for _ in []), freq='D'), PeriodIndex), (RangeIndex(step=1), RangeIndex), (MultiIndex(levels=[[1, 2], ['blue', 'red']], codes=[[], []]), MultiIndex)])
    def test_constructor_empty_special(self, empty: Index, klass: Type[Index]) -> None:
        assert isinstance(empty, klass)
        assert not len(empty)

    @pytest.mark.parametrize('index', ['datetime', 'float64', 'float32', 'int64', 'int32', 'period', 'range', 'repeats', 'timedelta', 'tuples', 'uint64', 'uint32'], indirect=True)
    def test_view_with_args(self, index: Index) -> None:
        index.view('i8')

    @pytest.mark.parametrize('index', ['string', pytest.param('categorical', marks=pytest.mark.xfail(reason='gh-25464')), 'bool-object', 'bool-dtype', 'empty'], indirect=True)
    def test_view_with_args_object_array_raises(self, index: Index) -> None:
        if index.dtype == bool:
            msg = 'When changing to a larger dtype'
            with pytest.raises(ValueError, match=msg):
                index.view('i8')
        else:
            msg = 'Cannot change data-type for array of references\\.|Cannot change data-type for object array\\.|Cannot change data-type for array of strings\\.|'
            with pytest.raises(TypeError, match=msg):
                index.view('i8')

    @pytest.mark.parametrize('index', ['int64', 'int32', 'range'], indirect=True)
    def test_astype(self, index: Index) -> None:
        casted = index.astype('i8')
        casted.get_loc(5)
        index.name = 'foobar'
        casted = index.astype('i8')
        assert casted.name == 'foobar'

    def test_equals_object(self) -> None:
        assert Index(['a', 'b', 'c']).equals(Index(['a', 'b', 'c']))

    @pytest.mark.parametrize('comp', [Index(['a', 'b']), Index(['a', 'b', 'd']), ['a', 'b', 'c']])
    def test_not_equals_object(self, comp: Any) -> None:
        assert not Index(['a', 'b', 'c']).equals(comp)

    def test_identical(self) -> None:
        i1 = Index(['a', 'b', 'c'])
        i2 = Index(['a', 'b', 'c'])
        assert i1.identical(i2)
        i1 = i1.rename('foo')
        assert i1.equals(i2)
        assert not i1.identical(i2)
        i2 = i2.rename('foo')
        assert i1.identical(i2)
        i3 = Index([('a', 'a'), ('a', 'b'), ('b', 'a')])
        i4 = Index([('a', 'a'), ('a', 'b'), ('b', 'a')], tupleize_cols=False)
        assert not i3.identical(i4)

    def test_is_(self) -> None:
        ind = Index(range(10))
        assert ind.is_(ind)
        assert ind.is_(ind.view().view().view().view())
        assert not ind.is_(Index(range(10)))
        assert not ind.is_(ind.copy())
        assert not ind.is_(ind.copy(deep=False))
        assert not ind.is_(ind[:])
        assert not ind.is_(np.array(range(10)))
        assert ind.is_(ind.view())
        ind2 = ind.view()
        ind2.name = 'bob'
        assert ind.is_(ind2)
        assert ind2.is_(ind)
        assert not ind.is_(Index(ind.values))
        arr = np.array(range(1, 11))
        ind1 = Index(arr, copy=False)
        ind2 = Index(arr, copy=False)
        assert not ind1.is_(ind2)

    def test_asof_numeric_vs_bool_raises(self) -> None:
        left = Index([1, 2, 3])
        right = Index([True, False], dtype=object)
        msg = 'Cannot compare dtypes int64 and bool'
        with pytest.raises(TypeError, match=msg):
            left.asof(right[0])
        with pytest.raises(InvalidIndexError, match=re.escape(str(right))):
            left.asof(right)
        with pytest.raises(InvalidIndexError, match=re.escape(str(left))):
            right.asof(left)

    @pytest.mark.parametrize('index', ['string'], indirect=True)
    def test_booleanindex(self, index: Index) -> None:
        bool_index = np.ones(len(index), dtype=bool)
        bool_index[5:30:2] = False
        sub_index = index[bool_index]
        for i, val in enumerate(sub_index):
            assert sub_index.get_loc(val) == i
        sub_index = index[list(bool_index)]
        for i, val in enumerate(sub_index):
            assert sub_index.get_loc(val) == i

    def test_fancy(self, simple_index: Index) -> None:
        index = simple_index
        sl = index[[1, 2, 3]]
        for i in sl:
            assert i == sl[sl.get_loc(i)]

    @pytest.mark.parametrize('index', ['string', 'int64', 'int32', 'uint64', 'uint32', 'float64', 'float32'], indirect=True)
    @pytest.mark.parametrize('dtype', [int, np.bool_])
    def test_empty_fancy(self, index: Index, dtype: Any, request: Any, using_infer_string: bool) -> None:
        if dtype is np.bool_ and using_infer_string and (index.dtype == 'string'):
            request.applymarker(pytest.mark.xfail(reason='numpy behavior is buggy'))
        empty_arr = np.array([], dtype=dtype)
        empty_index = type(index)([], dtype=index.dtype)
        assert index[[]].identical(empty_index)
        if dtype == np.bool_:
            with pytest.raises(ValueError, match='length of the boolean indexer'):
                assert index[empty_arr].identical(empty_index)
        else:
            assert index[empty_arr].identical(empty_index)

    @pytest.mark.parametrize('index', ['string', 'int64', 'int32', 'uint64', 'uint32', 'float64', 'float32'], indirect=True)
    def test_empty_fancy_raises(self, index: Index) -> None:
        empty_farr = np.array([], dtype=np.float64)
        empty_index = type(index)([], dtype=index.dtype)
        assert index[[]].identical(empty_index)
        msg = 'arrays used as indices must be of integer'
        with pytest.raises(IndexError, match=msg):
            index[empty_farr]

    def test_union_dt_as_obj(self, simple_index: Index) -> None:
        index = simple_index
        date_index = date_range('2019-01-01', periods=10)
        first_cat = index.union(date_index)
        second_cat = index.union(index)
        appended = Index(np.append(index, date_index.astype('O')))
        tm.assert_index_equal(first_cat, appended)
        tm.assert_index_equal(second_cat, index)
        tm.assert_contains_all(index, first_c