from collections import defaultdict
from datetime import datetime
from functools import partial
import math
import operator
import re
import numpy as np
import pytest
from pandas.compat import IS64
from pandas.errors import InvalidIndexError
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_any_real_numeric_dtype, is_numeric_dtype, is_object_dtype
import pandas as pd
from pandas import (
    CategoricalIndex,
    DataFrame,
    DatetimeIndex,
    IntervalIndex,
    PeriodIndex,
    RangeIndex,
    Series,
    TimedeltaIndex,
    date_range,
    period_range,
    timedelta_range,
)
import pandas._testing as tm
from pandas.core.indexes.api import (
    Index,
    MultiIndex,
    _get_combined_index,
    ensure_index,
    ensure_index_from_sequences,
)
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union


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
    @pytest.mark.parametrize(
        'index',
        [
            date_range('2015-01-01 10:00', freq='D', periods=3, tz='US/Eastern', name='Green Eggs & Ham'),
            date_range('2015-01-01 10:00', freq='D', periods=3),
            timedelta_range('1 days', freq='D', periods=3),
            period_range('2015-01-01', freq='D', periods=3),
        ],
    )
    def test_constructor_from_index_dtlike(
        self, cast_as_obj: bool, index: Union[DatetimeIndex, TimedeltaIndex, PeriodIndex]
    ) -> None:
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

    @pytest.mark.parametrize(
        'index,has_tz',
        [
            (
                date_range('2015-01-01 10:00', freq='D', periods=3, tz='US/Eastern'),
                True,
            ),
            (
                timedelta_range('1 days', freq='D', periods=3),
                False,
            ),
            (
                period_range('2015-01-01', freq='D', periods=3),
                False,
            ),
        ],
    )
    def test_constructor_from_series_dtlike(
        self, index: Union[DatetimeIndex, TimedeltaIndex, PeriodIndex], has_tz: bool
    ) -> None:
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
        dtype: Union[None, type] = object if not using_infer_string else 'str'
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

    @pytest.mark.parametrize(
        'klass,dtype,na_val',
        [
            (Index, np.float64, np.nan),
            (DatetimeIndex, 'datetime64[s]', pd.NaT),
        ],
    )
    def test_index_ctor_infer_nan_nat(
        self, klass: Callable[..., Index], dtype: Union[np.dtype, str], na_val: Any
    ) -> None:
        na_list = [na_val, na_val]
        expected = klass(na_list)
        assert expected.dtype == dtype
        result = Index(na_list)
        tm.assert_index_equal(result, expected)
        result = Index(np.array(na_list))
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize(
        'vals,dtype',
        [
            ([1, 2, 3, 4, 5], 'int'),
            ([1.1, np.nan, 2.2, 3.0], 'float'),
            (['A', 'B', 'C', np.nan], 'obj'),
        ],
    )
    def test_constructor_simple_new(
        self, vals: List[Any], dtype: str
    ) -> None:
        index = Index(vals, name=dtype)
        result = index._simple_new(index.values, dtype)
        tm.assert_index_equal(result, index)

    @pytest.mark.parametrize('attr', ['values', 'asi8'])
    @pytest.mark.parametrize('klass', [Index, DatetimeIndex])
    def test_constructor_dtypes_datetime(
        self, tz_naive_fixture: Optional[str], attr: str, klass: Callable[..., Index]
    ) -> None:
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
    def test_constructor_dtypes_timedelta(self, attr: str, klass: Callable[..., Index]) -> None:
        index = timedelta_range('1 days', periods=5)
        index = index._with_freq(None)
        dtype = index.dtype
        values = getattr(index, attr)
        result = klass(values, dtype=dtype)
        tm.assert_index_equal(result, index)
        result = klass(list(values), dtype=dtype)
        tm.assert_index_equal(result, index)

    @pytest.mark.parametrize(
        'value', [[], iter([]), (_ for _ in [])]
    )
    @pytest.mark.parametrize(
        'klass',
        [Index, CategoricalIndex, DatetimeIndex, TimedeltaIndex],
    )
    def test_constructor_empty(
        self, value: Iterable[Any], klass: Callable[..., Index]
    ) -> None:
        empty = klass(value)
        assert isinstance(empty, klass)
        assert not len(empty)

    @pytest.mark.parametrize(
        'empty,klass',
        [
            (PeriodIndex([], freq='D'), PeriodIndex),
            (PeriodIndex(iter([]), freq='D'), PeriodIndex),
            (PeriodIndex((_ for _ in []), freq='D'), PeriodIndex),
            (RangeIndex(step=1), RangeIndex),
            (
                MultiIndex(
                    levels=[[1, 2], ['blue', 'red']],
                    codes=[[], []],
                ),
                MultiIndex,
            ),
        ],
    )
    def test_constructor_empty_special(
        self, empty: Index, klass: Callable[..., Index]
    ) -> None:
        assert isinstance(empty, klass)
        assert not len(empty)

    @pytest.mark.parametrize(
        'index',
        [
            'datetime',
            'float64',
            'float32',
            'int64',
            'int32',
            'period',
            'range',
            'repeats',
            'timedelta',
            'tuples',
            'uint64',
            'uint32',
        ],
        indirect=True,
    )
    def test_view_with_args(self, index: Index) -> None:
        index.view('i8')

    @pytest.mark.parametrize(
        'index',
        [
            'string',
            pytest.param('categorical', marks=pytest.mark.xfail(reason='gh-25464')),
            'bool-object',
            'bool-dtype',
            'empty',
        ],
        indirect=True,
    )
    def test_view_with_args_object_array_raises(
        self, index: Index
    ) -> None:
        if index.dtype == bool:
            msg = 'When changing to a larger dtype'
            with pytest.raises(ValueError, match=msg):
                index.view('i8')
        else:
            msg = (
                'Cannot change data-type for array of references\\.|'
                'Cannot change data-type for object array\\.|'
                'Cannot change data-type for array of strings\\.|'
            )
            with pytest.raises(TypeError, match=msg):
                index.view('i8')

    @pytest.mark.parametrize(
        'index',
        [
            'int64',
            'int32',
            'range',
        ],
        indirect=True,
    )
    @pytest.mark.parametrize('dtype', [int, np.bool_])
    def test_astype(
        self, index: Index, dtype: Union[type, np.dtype]
    ) -> None:
        casted = index.astype('i8')
        casted.get_loc(5)
        index.name = 'foobar'
        casted = index.astype('i8')
        assert casted.name == 'foobar'

    def test_equals_object(self) -> None:
        assert Index(['a', 'b', 'c']).equals(Index(['a', 'b', 'c']))

    @pytest.mark.parametrize(
        'comp',
        [
            Index(['a', 'b']),
            Index(['a', 'b', 'd']),
            ['a', 'b', 'c'],
        ],
    )
    def test_not_equals_object(self, comp: Union[Index, List[str]]) -> None:
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

    @pytest.mark.parametrize(
        'index',
        [
            'string',
            'int64',
            'int32',
            'uint64',
            'uint32',
            'float64',
            'float32',
        ],
        indirect=True,
    )
    @pytest.mark.parametrize('dtype', [int, np.bool_])
    def test_empty_fancy(
        self, index: Index, dtype: Union[type, np.dtype], request: pytest.FixtureRequest, using_infer_string: bool
    ) -> None:
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

    @pytest.mark.parametrize(
        'index',
        [
            'string',
            'int64',
            'int32',
            'uint64',
            'uint32',
            'float64',
            'float32',
        ],
        indirect=True,
    )
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
        tm.assert_contains_all(index, first_cat)
        tm.assert_contains_all(index, second_cat)
        tm.assert_contains_all(date_index, first_cat)

    def test_map_with_tuples(self) -> None:
        index = Index(np.arange(3), dtype=np.int64)
        result = index.map(lambda x: (x,))
        expected = Index([(i,) for i in index])
        tm.assert_index_equal(result, expected)
        result = index.map(lambda x: (x, x == 1))
        expected = MultiIndex.from_tuples([(i, i == 1) for i in index])
        tm.assert_index_equal(result, expected)

    def test_map_with_tuples_mi(self) -> None:
        first_level = ['foo', 'bar', 'baz']
        multi_index = MultiIndex.from_tuples(zip(first_level, [1, 2, 3]))
        reduced_index = multi_index.map(lambda x: x[0])
        tm.assert_index_equal(reduced_index, Index(first_level))

    @pytest.mark.parametrize(
        'index',
        [
            date_range('2020-01-01', freq='D', periods=10),
            period_range('2020-01-01', freq='D', periods=10),
            timedelta_range('1 day', periods=10),
        ],
        indirect=False,
    )
    def test_map_tseries_indices_return_index(
        self, index: Union[DatetimeIndex, PeriodIndex, TimedeltaIndex]
    ) -> None:
        expected = Index([1] * 10)
        result = index.map(lambda x: 1)
        tm.assert_index_equal(expected, result)

    def test_map_tseries_indices_accsr_return_index(self) -> None:
        date_index = DatetimeIndex(date_range('2020-01-01', periods=24, freq='h'), name='hourly')
        result = date_index.map(lambda x: x.hour)
        expected = Index(np.arange(24, dtype='int64'), name='hourly')
        tm.assert_index_equal(result, expected, exact=True)

    @pytest.mark.parametrize(
        'mapper',
        [
            lambda values, index: {i: e for e, i in zip(values, index)},
            lambda values, index: Series(values, index=index),
        ],
    )
    def test_map_dictlike_simple(
        self, mapper: Callable[[Any, Index], Union[Dict[Any, Any], Series[Any]]]
    ) -> None:
        expected = Index(['foo', 'bar', 'baz'])
        index = Index(np.arange(3), dtype=np.int64)
        result = index.map(mapper(expected.values, index))
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize(
        'mapper',
        [
            lambda values, index: {i: e for e, i in zip(values, index)},
            lambda values, index: Series(values, index=index),
        ],
    )
    @pytest.mark.filterwarnings('ignore:PeriodDtype\\[B\\] is deprecated:FutureWarning')
    def test_map_dictlike(
        self, index: Index, mapper: Callable[[Index, Index], Union[Dict[Any, Any], Series[Any]]], request: pytest.FixtureRequest
    ) -> None:
        if isinstance(index, CategoricalIndex):
            pytest.skip('Tested in test_categorical')
        elif not index.is_unique:
            pytest.skip('Cannot map duplicated index')
        rng = np.arange(len(index), 0, -1, dtype=np.int64)
        if index.empty:
            expected = Index([])
        elif is_numeric_dtype(index.dtype):
            expected = index._constructor(rng, dtype=index.dtype)
        elif type(index) is Index and index.dtype != object:
            expected = Index(rng, dtype=index.dtype)
        else:
            expected = Index(rng)
        result = index.map(mapper(expected, index))
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize(
        'mapper',
        [
            Series(['foo', 2.0, 'baz'], index=[0, 2, -1]),
            {0: 'foo', 2: 2.0, -1: 'baz'},
        ],
    )
    def test_map_with_non_function_missing_values(
        self, mapper: Union[Dict[int, Any], Series[Any]]
    ) -> None:
        expected = Index([2.0, np.nan, 'foo'])
        result = Index([2, 1, 0]).map(mapper)
        tm.assert_index_equal(result, expected)

    def test_map_na_exclusion(self) -> None:
        index = Index([1.5, np.nan, 3, np.nan, 5])
        result = index.map(lambda x: x * 2, na_action='ignore')
        expected = index * 2
        tm.assert_index_equal(result, expected)

    def test_map_defaultdict(self) -> None:
        index = Index([1, 2, 3])
        default_dict = defaultdict(lambda: 'blank')
        default_dict[1] = 'stuff'
        result = index.map(default_dict)
        expected = Index(['stuff', 'blank', 'blank'])
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize(
        'name,expected',
        [
            ('foo', 'foo'),
            ('bar', None),
        ],
    )
    def test_append_empty_preserve_name(
        self, name: Optional[str], expected: Optional[str]
    ) -> None:
        left = Index([], name='foo')
        right = Index([1, 2, 3], name=name)
        result = left.append(right)
        assert result.name == expected

    @pytest.mark.parametrize(
        'index, expected',
        [
            ('string', False),
            ('bool-object', False),
            ('bool-dtype', False),
            ('categorical', False),
            ('int64', True),
            ('int32', True),
            ('uint64', True),
            ('uint32', True),
            ('datetime', False),
            ('float64', True),
            ('float32', True),
        ],
        indirect=['index'],
    )
    def test_is_numeric(
        self, index: Index, expected: bool
    ) -> None:
        assert is_any_real_numeric_dtype(index) is expected

    @pytest.mark.parametrize(
        'index, expected',
        [
            ('string', True),
            ('bool-object', True),
            ('bool-dtype', False),
            ('categorical', False),
            ('int64', False),
            ('int32', False),
            ('uint64', False),
            ('uint32', False),
            ('datetime', False),
            ('float64', False),
            ('float32', False),
        ],
        indirect=['index'],
    )
    def test_is_object(
        self, index: Index, expected: bool, using_infer_string: bool
    ) -> None:
        if using_infer_string and index.dtype == 'string' and expected:
            expected = False
        assert is_object_dtype(index) is expected

    def test_summary(self, index: Index) -> None:
        index._summary()

    def test_logical_compat(
        self, all_boolean_reductions: str, simple_index: Index
    ) -> None:
        index = simple_index
        left = getattr(index, all_boolean_reductions)()
        assert left == getattr(index.values, all_boolean_reductions)()
        right = getattr(index.to_series(), all_boolean_reductions)()
        assert bool(left) == bool(right)

    @pytest.mark.parametrize(
        'index',
        [
            'string',
            'int64',
            'int32',
            'uint64',
            'uint32',
            'float64',
            'float32',
        ],
        indirect=True,
    )
    def test_drop_by_str_label(
        self, index: Index
    ) -> None:
        n = len(index)
        drop = index[list(range(5, 10))]
        dropped = index.drop(drop)
        expected = index[list(range(5)) + list(range(10, n))]
        tm.assert_index_equal(dropped, expected)
        dropped = index.drop(index[0])
        expected = index[1:]
        tm.assert_index_equal(dropped, expected)

    @pytest.mark.parametrize(
        'index',
        [
            'string',
            'int64',
            'int32',
            'uint64',
            'uint32',
            'float64',
            'float32',
        ],
        indirect=True,
    )
    @pytest.mark.parametrize(
        'keys',
        [
            ['foo', 'bar'],
            ['1', 'bar'],
        ],
    )
    def test_drop_by_str_label_raises_missing_keys(
        self, index: Index, keys: List[str]
    ) -> None:
        with pytest.raises(KeyError, match=''):
            index.drop(keys)

    @pytest.mark.parametrize(
        'index',
        [
            'string',
            'int64',
            'int32',
            'uint64',
            'uint32',
            'float64',
            'float32',
        ],
        indirect=True,
    )
    def test_drop_by_str_label_errors_ignore(
        self, index: Index
    ) -> None:
        n = len(index)
        drop = index[list(range(5, 10))]
        mixed = drop.tolist() + ['foo']
        dropped = index.drop(mixed, errors='ignore')
        expected = index[list(range(5)) + list(range(10, n))]
        tm.assert_index_equal(dropped, expected)
        dropped = index.drop(['foo', 'bar'], errors='ignore')
        expected = index[list(range(n))]
        tm.assert_index_equal(dropped, expected)

    def test_drop_by_numeric_label_loc(self) -> None:
        index = Index([1, 2, 3])
        dropped = index.drop(1)
        expected = Index([2, 3])
        tm.assert_index_equal(dropped, expected)

    def test_drop_by_numeric_label_raises_missing_keys(self) -> None:
        index = Index([1, 2, 3])
        with pytest.raises(KeyError, match=''):
            index.drop([3, 4])

    @pytest.mark.parametrize(
        'key,expected',
        [
            (4, Index([1, 2, 3])),
            ([3, 4, 5], Index([1, 2])),
        ],
    )
    def test_drop_by_numeric_label_errors_ignore(
        self, key: Union[int, List[int]], expected: Index
    ) -> None:
        index = Index([1, 2, 3])
        dropped = index.drop(key, errors='ignore')
        tm.assert_index_equal(dropped, expected)

    @pytest.mark.parametrize(
        'values, to_drop',
        [
            (['a', 'b', ('c', 'd')], [('c', 'd'), 'a']),
            (['a', 'b', ('c', 'd')], ['a', ('c', 'd')]),
        ],
    )
    def test_drop_tuple(
        self, values: List[Union[str, Tuple[str, str]]], to_drop: List[Union[str, Tuple[str, str]]]
    ) -> None:
        index = Index(values)
        expected = Index(['b'], dtype=object)
        result = index.drop(to_drop)
        tm.assert_index_equal(result, expected)
        removed = index.drop(to_drop[0])
        for drop_me in (to_drop[1], [to_drop[1]]):
            result = removed.drop(drop_me)
            tm.assert_index_equal(result, expected)
        removed = index.drop(to_drop[1])
        msg = f'\\"\\[{re.escape(to_drop[1].__repr__())}\\] not found in axis\\"'
        for drop_me in (to_drop[1], [to_drop[1]]):
            with pytest.raises(KeyError, match=msg):
                removed.drop(drop_me)

    @pytest.mark.filterwarnings('ignore:PeriodDtype\\[B\\] is deprecated:FutureWarning')
    def test_drop_with_duplicates_in_index(self, index: Index) -> None:
        if len(index) == 0 or isinstance(index, MultiIndex):
            pytest.skip("Test doesn't make sense for empty MultiIndex")
        if isinstance(index, IntervalIndex) and (not IS64):
            pytest.skip('Cannot test IntervalIndex with int64 dtype on 32 bit platform')
        index = index.unique().repeat(2)
        expected = index[2:]
        result = index.drop(index[0])
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize(
        'attr',
        [
            'is_monotonic_increasing',
            'is_monotonic_decreasing',
            '_is_strictly_monotonic_increasing',
            '_is_strictly_monotonic_decreasing',
        ],
    )
    def test_is_monotonic_incomparable(
        self, attr: str
    ) -> None:
        index = Index([5, datetime.now(), 7])
        assert not getattr(index, attr)

    @pytest.mark.parametrize(
        'values',
        [
            ['foo', 'bar', 'quux'],
            {'foo', 'bar', 'quux'},
        ],
    )
    @pytest.mark.parametrize(
        'index,expected',
        [
            (['qux', 'baz', 'foo', 'bar'], [False, False, True, True]),
            ([], []),
        ],
    )
    def test_isin(
        self, values: Union[List[str], set], index: Union[List[str], np.ndarray], expected: List[bool]
    ) -> None:
        index = Index(index)
        result = index.isin(values)
        expected_array = np.array(expected, dtype=bool)
        tm.assert_numpy_array_equal(result, expected_array)

    def test_isin_nan_common_object(
        self, nulls_fixture: Any, nulls_fixture2: Any, using_infer_string: bool
    ) -> None:
        idx = Index(['a', nulls_fixture])
        if (
            isinstance(nulls_fixture, float)
            and isinstance(nulls_fixture2, float)
            and math.isnan(nulls_fixture)
            and math.isnan(nulls_fixture2)
        ):
            tm.assert_numpy_array_equal(idx.isin([nulls_fixture2]), np.array([False, True]))
        elif nulls_fixture is nulls_fixture2:
            tm.assert_numpy_array_equal(idx.isin([nulls_fixture2]), np.array([False, True]))
        elif using_infer_string and idx.dtype == 'string':
            tm.assert_numpy_array_equal(idx.isin([nulls_fixture2]), np.array([False, True]))
        else:
            tm.assert_numpy_array_equal(idx.isin([nulls_fixture2]), np.array([False, False]))

    def test_isin_nan_common_float64(
        self, nulls_fixture: Any, float_numpy_dtype: np.dtype
    ) -> None:
        dtype = float_numpy_dtype
        if nulls_fixture is pd.NaT or nulls_fixture is pd.NA:
            msg = f'float\\(\\) argument must be a string or a (real )?number, not {type(nulls_fixture).__name__!r}'
            with pytest.raises(TypeError, match=msg):
                Index([1.0, nulls_fixture], dtype=dtype)
            idx = Index([1.0, np.nan], dtype=dtype)
            assert not idx.isin([nulls_fixture]).any()
            return
        idx = Index([1.0, nulls_fixture], dtype=dtype)
        res = idx.isin([np.nan])
        tm.assert_numpy_array_equal(res, np.array([False, True]))
        res = idx.isin([pd.NaT])
        tm.assert_numpy_array_equal(res, np.array([False, False]))

    @pytest.mark.parametrize(
        'level',
        [0, -1],
    )
    @pytest.mark.parametrize(
        'index,has_tz',
        [
            (['qux', 'baz', 'foo', 'bar'], [False, False, True, True]),
            (np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64), [False, False, True, True]),
        ],
    )
    def test_isin_level_kwarg(
        self, level: int, index: Union[List[str], np.ndarray], expected: List[bool]
    ) -> None:
        index = Index(index)
        values = index.tolist()[-2:] + ['nonexisting']
        expected_array = np.array([False, False, True, True], dtype=bool)
        tm.assert_numpy_array_equal(expected_array, index.isin(values, level=level))
        index.name = 'foobar'
        tm.assert_numpy_array_equal(expected_array, index.isin(values, level='foobar'))

    def test_isin_level_kwarg_bad_level_raises(self, index: Index) -> None:
        for level in [10, index.nlevels, -(index.nlevels + 1)]:
            with pytest.raises(IndexError, match='Too many levels'):
                index.isin([], level=level)

    @pytest.mark.parametrize(
        'label',
        [
            1.0,
            'foobar',
            'xyzzy',
            np.nan,
        ],
    )
    def test_isin_level_kwarg_bad_label_raises(
        self, label: Union[str, float], index: Index
    ) -> None:
        if isinstance(index, MultiIndex):
            index = index.rename(['foo', 'bar'] + index.names[2:])
            msg = f"'Level {label} not found'"
        else:
            index = index.rename('foo')
            msg = f'Requested level \\({label}\\) does not match index name \\(foo\\)'
        with pytest.raises(KeyError, match=msg):
            index.isin([], level=label)

    @pytest.mark.parametrize(
        'empty',
        [
            [],
            Series(dtype=object),
            np.array([]),
        ],
    )
    def test_isin_empty(
        self, empty: Union[List[Any], Series[Any], np.ndarray]
    ) -> None:
        index = Index(['a', 'b'])
        expected = np.array([False, False])
        result = index.isin(empty)
        tm.assert_numpy_array_equal(expected, result)

    def test_isin_string_null(
        self, string_dtype_no_object: str
    ) -> None:
        index = Index(['a', 'b'], dtype=string_dtype_no_object)
        result = index.isin([None])
        expected = np.array([False, False])
        tm.assert_numpy_array_equal(result, expected)

    @pytest.mark.parametrize(
        'values',
        [
            [1, 2, 3, 4],
            [1.0, 2.0, 3.0, 4.0],
            [True, True, True, True],
            ['foo', 'bar', 'baz', 'qux'],
            date_range('2018-01-01', freq='D', periods=4),
        ],
    )
    def test_boolean_cmp(
        self, values: List[Any]
    ) -> None:
        index = Index(values)
        result = index == values
        expected = np.array([True, True, True, True], dtype=bool)
        tm.assert_numpy_array_equal(result, expected)

    @pytest.mark.parametrize(
        'index',
        [
            MultiIndex.from_tuples([(1, 2), (4, 5)]),
            Index(['foo', 'bar', 'baz']),
        ],
    )
    def test_equals_op_multiindex(
        self, mi: Index, expected_results: np.ndarray
    ) -> None:
        df = DataFrame([3, 6], columns=['c'], index=MultiIndex.from_arrays([[1, 4], [2, 5]], names=['a', 'b']))
        result = df.index == mi
        tm.assert_numpy_array_equal(result, expected_results)

    def test_equals_op_multiindex_identify(self) -> None:
        df = DataFrame([3, 6], columns=['c'], index=MultiIndex.from_arrays([[1, 4], [2, 5]], names=['a', 'b']))
        result = df.index == df.index
        expected = np.array([True, True])
        tm.assert_numpy_array_equal(result, expected)

    @pytest.mark.parametrize(
        'index',
        [
            MultiIndex.from_tuples([(1, 2), (4, 5), (8, 9)]),
            Index(['foo', 'bar', 'baz']),
        ],
    )
    def test_equals_op_mismatched_multiindex_raises(
        self, index: Index
    ) -> None:
        df = DataFrame([3, 6], columns=['c'], index=MultiIndex.from_arrays([[1, 4], [2, 5]], names=['a', 'b']))
        with pytest.raises(ValueError, match='Lengths must match'):
            df.index == index

    def test_equals_op_index_vs_mi_same_length(
        self, using_infer_string: bool
    ) -> None:
        mi = MultiIndex.from_tuples([(1, 2), (4, 5), (8, 9)])
        index = Index(['foo', 'bar', 'baz'])
        result = mi == index
        expected = np.array([False, False, False])
        tm.assert_numpy_array_equal(result, expected)

    @pytest.mark.parametrize(
        'dt_conv, arg',
        [
            (pd.to_datetime, ['2000-01-01', '2000-01-02']),
            (pd.to_timedelta, ['01:02:03', '01:02:04']),
        ],
    )
    def test_dt_conversion_preserves_name(
        self, dt_conv: Callable[[List[str]], pd.Series], arg: List[str]
    ) -> None:
        index = Index(arg, name='label')
        assert index.name == dt_conv(index).name

    def test_cached_properties_not_settable(self) -> None:
        index = Index([1, 2, 3])
        with pytest.raises(AttributeError, match="Can't set attribute"):
            index.is_unique = False

    def test_tab_complete_warning(self, ip: Any) -> None:
        pytest.importorskip('IPython', minversion='6.0.0')
        from IPython.core.completer import provisionalcompleter
        code = 'import pandas as pd; idx = pd.Index([1, 2])'
        ip.run_cell(code)
        with tm.assert_produces_warning(None, raise_on_extra_warnings=False):
            with provisionalcompleter('ignore'):
                list(ip.Completer.completions('idx.', 4))

    def test_contains_method_removed(self, index: Index) -> None:
        if isinstance(index, IntervalIndex):
            index.contains(1)
        else:
            msg = f"'{type(index).__name__}' object has no attribute 'contains'"
            with pytest.raises(AttributeError, match=msg):
                index.contains(1)

    def test_sortlevel(self) -> None:
        index = Index([5, 4, 3, 2, 1])
        with pytest.raises(Exception, match='ascending must be a single bool value or'):
            index.sortlevel(ascending='True')
        with pytest.raises(Exception, match='ascending must be a list of bool values of length 1'):
            index.sortlevel(ascending=[True, True])
        with pytest.raises(Exception, match='ascending must be a bool value'):
            index.sortlevel(ascending=['True'])
        expected = Index([1, 2, 3, 4, 5])
        result = index.sortlevel(ascending=[True])
        tm.assert_index_equal(result[0], expected)
        expected = Index([1, 2, 3, 4, 5])
        result = index.sortlevel(ascending=True)
        tm.assert_index_equal(result[0], expected)
        expected = Index([5, 4, 3, 2, 1])
        result = index.sortlevel(ascending=False)
        tm.assert_index_equal(result[0], expected)

    def test_sortlevel_na_position(self) -> None:
        idx = Index([1, np.nan])
        result = idx.sortlevel(na_position='first')[0]
        expected = Index([np.nan, 1])
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize(
        'periods, expected_results',
        [
            (1, [np.nan, 10, 10, 10, 10]),
            (2, [np.nan, np.nan, 20, 20, 20]),
            (3, [np.nan, np.nan, np.nan, 30, 30]),
        ],
    )
    def test_index_diff(
        self, periods: int, expected_results: List[Optional[float]]
    ) -> None:
        idx = Index([10, 20, 30, 40, 50])
        result = idx.diff(periods)
        expected = Index(expected_results)
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize(
        'decimals, expected_results',
        [
            (0, [1.0, 2.0, 3.0]),
            (1, [1.2, 2.3, 3.5]),
            (2, [1.23, 2.35, 3.46]),
        ],
    )
    def test_index_round(
        self, decimals: int, expected_results: List[float]
    ) -> None:
        idx = Index([1.234, 2.345, 3.456])
        result = idx.round(decimals)
        expected = Index(expected_results)
        tm.assert_index_equal(result, expected)


class TestMixedIntIndex:

    @pytest.fixture
    def simple_index(self) -> Index:
        return Index([0, 'a', 1, 'b', 2, 'c'])

    def test_argsort(self, simple_index: Index) -> None:
        index = simple_index
        with pytest.raises(TypeError, match="'>|<' not supported"):
            index.argsort()

    def test_numpy_argsort(self, simple_index: Index) -> None:
        index = simple_index
        with pytest.raises(TypeError, match="'>|<' not supported"):
            np.argsort(index)

    def test_copy_name(self, simple_index: Index) -> None:
        index = simple_index
        first = type(index)(index, copy=True, name='mario')
        second = type(first)(first, copy=False)
        assert first is not second
        tm.assert_index_equal(first, second)
        assert first.name == 'mario'
        assert second.name == 'mario'
        s1 = Series(2, index=first)
        s2 = Series(3, index=second[:-1])
        s3 = s1 * s2
        assert s3.index.name == 'mario'

    def test_copy_name2(self) -> None:
        index = Index([1, 2], name='MyName')
        index1 = index.copy()
        tm.assert_index_equal(index, index1)
        index2 = index.copy(name='NewName')
        tm.assert_index_equal(index, index2, check_names=False)
        assert index.name == 'MyName'
        assert index2.name == 'NewName'

    def test_unique_na(self) -> None:
        idx = Index([2, np.nan, 2, 1], name='my_index')
        expected = Index([2, np.nan, 1], name='my_index')
        result = idx.unique()
        tm.assert_index_equal(result, expected)

    def test_logical_compat(
        self, simple_index: Index
    ) -> None:
        index = simple_index
        assert index.all() == index.values.all()
        assert index.any() == index.values.any()

    @pytest.mark.parametrize(
        'how,dtype,vals,expected',
        [
            (
                'any',
                None,
                [1, 2, 3],
                [1, 2, 3],
            ),
            (
                'any',
                None,
                [1.0, 2.0, 3.0],
                [1.0, 2.0, 3.0],
            ),
            (
                'any',
                None,
                [1.0, 2.0, np.nan, 3.0],
                [1.0, 2.0, 3.0],
            ),
            (
                'any',
                None,
                ['A', 'B', 'C'],
                ['A', 'B', 'C'],
            ),
            (
                'any',
                None,
                ['A', np.nan, 'B', 'C'],
                ['A', 'B', 'C'],
            ),
        ],
    )
    def test_dropna(
        self,
        how: str,
        dtype: Optional[Union[type, str]],
        vals: List[Any],
        expected: List[Any],
    ) -> None:
        index = Index(vals, dtype=dtype)
        result = index.dropna(how=how)
        expected_index = Index(expected, dtype=dtype)
        tm.assert_index_equal(result, expected_index)

    @pytest.mark.parametrize(
        'how,index,expected',
        [
            (
                'any',
                DatetimeIndex(['2011-01-01', '2011-01-02', '2011-01-03']),
                DatetimeIndex(['2011-01-01', '2011-01-02', '2011-01-03']),
            ),
            (
                'any',
                DatetimeIndex(['2011-01-01', '2011-01-02', '2011-01-03', pd.NaT]),
                DatetimeIndex(['2011-01-01', '2011-01-02', '2011-01-03']),
            ),
            (
                'any',
                TimedeltaIndex(['1 days', '2 days', '3 days']),
                TimedeltaIndex(['1 days', '2 days', '3 days']),
            ),
            (
                'any',
                TimedeltaIndex([pd.NaT, '1 days', '2 days', '3 days', pd.NaT]),
                TimedeltaIndex(['1 days', '2 days', '3 days']),
            ),
            (
                'any',
                PeriodIndex(['2012-02', '2012-04', '2012-05'], freq='M'),
                PeriodIndex(['2012-02', '2012-04', '2012-05'], freq='M'),
            ),
            (
                'any',
                PeriodIndex(['2012-02', '2012-04', 'NaT', '2012-05'], freq='M'),
                PeriodIndex(['2012-02', '2012-04', '2012-05'], freq='M'),
            ),
        ],
    )
    def test_dropna_dt_like(
        self,
        how: str,
        index: Union[DatetimeIndex, TimedeltaIndex, PeriodIndex],
        expected: Union[DatetimeIndex, TimedeltaIndex, PeriodIndex],
    ) -> None:
        result = index.dropna(how=how)
        tm.assert_index_equal(result, expected)

    def test_dropna_invalid_how_raises(self) -> None:
        msg = 'invalid how option: xxx'
        with pytest.raises(ValueError, match=msg):
            Index([1, 2, 3]).dropna(how='xxx')

    @pytest.mark.parametrize(
        'dtype',
        ['f8', 'm8[ns]', 'M8[us]'],
    )
    @pytest.mark.parametrize(
        'unique_first',
        [True, False],
    )
    def test_is_monotonic_unique_na(
        self, dtype: str, unique_first: bool
    ) -> None:
        index = Index([None, 1, 1], dtype=dtype)
        if unique_first:
            assert index.is_unique is False
            assert index.is_monotonic_increasing is False
            assert index.is_monotonic_decreasing is False
        else:
            assert index.is_monotonic_increasing is False
            assert index.is_monotonic_decreasing is False
            assert index.is_unique is False

    @pytest.mark.parametrize(
        'dtype',
        ['f8', 'm8[ns]', 'M8[us]'],
    )
    @pytest.mark.parametrize(
        'unique_first',
        [True, False],
    )
    def test_is_monotonic_unique_na(
        self, dtype: str, unique_first: bool
    ) -> None:
        index = Index([None, 1, 1], dtype=dtype)
        if unique_first:
            assert index.is_unique is False
            assert index.is_monotonic_increasing is False
            assert index.is_monotonic_decreasing is False
        else:
            assert index.is_monotonic_increasing is False
            assert index.is_monotonic_decreasing is False
            assert index.is_unique is False

    @pytest.mark.parametrize(
        'dtype',
        ['f8', 'm8[ns]', 'M8[us]'],
    )
    @pytest.mark.parametrize(
        'unique_first',
        [True, False],
    )
    def test_is_monotonic_unique_na(
        self, dtype: str, unique_first: bool
    ) -> None:
        index = Index([None, 1, 1], dtype=dtype)
        if unique_first:
            assert index.is_unique is False
            assert index.is_monotonic_increasing is False
            assert index.is_monotonic_decreasing is False
        else:
            assert index.is_monotonic_increasing is False
            assert index.is_monotonic_decreasing is False
            assert index.is_unique is False

    def test_int_name_format(
        self, frame_or_series: Callable[..., Union[DataFrame, Series]]
    ) -> None:
        index = Index(['a', 'b', 'c'], name=0)
        result = frame_or_series(list(range(3)), index=index)
        assert '0' in repr(result)

    def test_str_to_bytes_raises(self) -> None:
        index = Index([str(x) for x in range(10)])
        msg = "^'str' object cannot be interpreted as an integer$"
        with pytest.raises(TypeError, match=msg):
            bytes(index)

    @pytest.mark.filterwarnings('ignore:elementwise comparison failed:FutureWarning')
    def test_index_with_tuple_bool(self) -> None:
        idx = Index([('a', 'b'), ('b', 'c'), ('c', 'a')])
        result = idx == ('c', 'a')
        expected = np.array([False, False, True])
        tm.assert_numpy_array_equal(result, expected)

class TestIndexUtils:

    @pytest.mark.parametrize(
        'data, names, expected',
        [
            ([[1, 2, 4]], None, Index([1, 2, 4])),
            ([[1, 2, 4]], ['name'], Index([1, 2, 4], name='name')),
            ([[1, 2, 3]], None, RangeIndex(1, 4)),
            ([[1, 2, 3]], ['name'], RangeIndex(1, 4, name='name')),
            ([['a', 'a'], ['c', 'd']], None, MultiIndex([['a'], ['c', 'd']], [[0, 0], [0, 1]])),
            (
                [['a', 'a'], ['c', 'd']],
                ['L1', 'L2'],
                MultiIndex([['a'], ['c', 'd']], [[0, 0], [0, 1]], names=['L1', 'L2']),
            ),
        ],
    )
    def test_ensure_index_from_sequences(
        self,
        data: List[List[Any]],
        names: Optional[List[str]],
        expected: Index,
    ) -> None:
        result = ensure_index_from_sequences(data, names)
        tm.assert_index_equal(result, expected, exact=True)

    def test_ensure_index_mixed_closed_intervals(self) -> None:
        intervals = [
            pd.Interval(0, 1, closed='left'),
            pd.Interval(1, 2, closed='right'),
            pd.Interval(2, 3, closed='neither'),
            pd.Interval(3, 4, closed='both'),
        ]
        result = ensure_index(intervals)
        expected = Index(intervals, dtype=object)
        tm.assert_index_equal(result, expected)

    def test_ensure_index_uint64(self) -> None:
        values = [0, np.iinfo(np.uint64).max]
        result = ensure_index(values)
        assert list(result) == values
        expected = Index(values, dtype='uint64')
        tm.assert_index_equal(result, expected)

    def test_get_combined_index(self) -> None:
        result = _get_combined_index([])
        expected = RangeIndex(0)
        tm.assert_index_equal(result, expected)


@pytest.mark.parametrize(
    'opname',
    [
        'eq',
        'ne',
        'le',
        'lt',
        'ge',
        'gt',
        'add',
        'radd',
        'sub',
        'rsub',
        'mul',
        'rmul',
        'truediv',
        'rtruediv',
        'floordiv',
        'rfloordiv',
        'pow',
        'rpow',
        'mod',
        'divmod',
    ],
)
def test_generated_op_names(
    opname: str, index: Index
) -> None:
    opname = f'__{opname}__'
    method = getattr(index, opname)
    assert method.__name__ == opname


@pytest.mark.parametrize(
    'klass, extra_kwargs',
    [
        (Index, {}),
        *[
            (lambda x: Index(x, dtype=dtyp), {})
            for dtyp in tm.ALL_REAL_NUMPY_DTYPES
        ],
        (DatetimeIndex, {}),
        (TimedeltaIndex, {}),
        (PeriodIndex, {'freq': 'Y'}),
        (RangeIndex, {'start': range(1)}),
        (IntervalIndex, {'data': [pd.Interval(0, 1)]}),
        (Index, {'data': ['a'], 'dtype': object}),
        (MultiIndex, {'levels': [[1], [2]], 'codes': [[0], [0]]}),
    ],
)
def test_index_subclass_constructor_wrong_kwargs(
    klass: Callable[..., Index], extra_kwargs: Dict[str, Any]
) -> None:
    with pytest.raises(TypeError, match='unexpected keyword argument'):
        klass(foo='bar')


def test_deprecated_fastpath() -> None:
    msg = '[Uu]nexpected keyword argument'
    with pytest.raises(TypeError, match=msg):
        Index(np.array(['a', 'b'], dtype=object), name='test', fastpath=True)
    with pytest.raises(TypeError, match=msg):
        Index(np.array([1, 2, 3], dtype='int64'), name='test', fastpath=True)
    with pytest.raises(TypeError, match=msg):
        RangeIndex(0, 5, 2, name='test', fastpath=True)
    with pytest.raises(TypeError, match=msg):
        CategoricalIndex(['a', 'b', 'c'], name='test', fastpath=True)


def test_shape_of_invalid_index() -> None:
    idx = Index([0, 1, 2, 3])
    with pytest.raises(ValueError, match='Multi-dimensional indexing'):
        idx[:, None]


@pytest.mark.parametrize(
    'dtype',
    [None, np.int64, np.uint64, np.float64],
)
def test_validate_1d_input(
    dtype: Optional[Union[type, np.dtype]]
) -> None:
    msg = 'Index data must be 1-dimensional'
    arr = np.arange(8).reshape(2, 2, 2)
    with pytest.raises(ValueError, match=msg):
        Index(arr, dtype=dtype)
    df = DataFrame(arr.reshape(4, 2))
    with pytest.raises(ValueError, match=msg):
        Index(df, dtype=dtype)
    ser = Series(0, range(4))
    with pytest.raises(ValueError, match=msg):
        ser.index = np.array([[2, 3]] * 4, dtype=dtype)


@pytest.mark.parametrize(
    'klass, extra_kwargs',
    [
        (Index, {}),
        *[
            (partial(CategoricalIndex, data=[1]), {}),
            (partial(DatetimeIndex, data=['2020-01-01']), {}),
            (partial(PeriodIndex, data=['2020-01-01']), {}),
            (partial(TimedeltaIndex, data=['1 day']), {}),
            (partial(RangeIndex, start=range(1)), {}),
            (partial(IntervalIndex, data=[pd.Interval(0, 1)]), {}),
            (partial(Index, data=['a'], dtype=object), {}),
            (partial(MultiIndex, levels=[1], codes=[0]), {}),
        ],
    ],
)
def test_construct_from_memoryview(
    klass: Callable[..., Index], extra_kwargs: Dict[str, Any]
) -> None:
    result = klass(memoryview(np.arange(2000, 2005)), **extra_kwargs)
    expected = klass(list(range(2000, 2005)), **extra_kwargs)
    tm.assert_index_equal(result, expected, exact=True)


@pytest.mark.parametrize(
    'op',
    [
        operator.lt,
        operator.gt,
    ],
)
def test_nan_comparison_same_object(
    op: Callable[[Any, Any], Any], index: Index
) -> None:
    idx = Index([np.nan])
    expected = np.array([False])
    result = op(idx, idx)
    tm.assert_numpy_array_equal(result, expected)
    result = op(idx, idx.copy())
    tm.assert_numpy_array_equal(result, expected)


@td.skip_if_no('pyarrow')
def test_is_monotonic_pyarrow_list_type() -> None:
    import pyarrow as pa
    idx = Index([[1], [2, 3]], dtype=pd.ArrowDtype(pa.list_(pa.int64())))
    assert not idx.is_monotonic_increasing
    assert not idx.is_monotonic_decreasing
