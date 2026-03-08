import array
from collections import OrderedDict, abc, defaultdict, namedtuple
from collections.abc import Iterator
from dataclasses import make_dataclass
from datetime import date, datetime, timedelta
import functools
import re
import zoneinfo
import numpy as np
from numpy import ma
from numpy.ma import mrecords
import pytest
from pandas._config import using_string_dtype
from pandas._libs import lib
from pandas.compat.numpy import np_version_gt2
from pandas.errors import IntCastingNaNError
from pandas.core.dtypes.common import is_integer_dtype
from pandas.core.dtypes.dtypes import DatetimeTZDtype, IntervalDtype, NumpyEADtype, PeriodDtype
import pandas as pd
from pandas import Categorical, CategoricalIndex, DataFrame, DatetimeIndex, Index, Interval, MultiIndex, Period, RangeIndex, Series, Timedelta, Timestamp, cut, date_range, isna
import pandas._testing as tm
from pandas.arrays import DatetimeArray, IntervalArray, PeriodArray, SparseArray, TimedeltaArray
from typing import Any, Callable, Dict, Generator, Iterable, Iterator as TypingIterator, List, Mapping, Optional, Sequence, Tuple, Type, Union, cast, overload

MIXED_FLOAT_DTYPES: List[str] = ['float16', 'float32', 'float64']
MIXED_INT_DTYPES: List[str] = ['uint8', 'uint16', 'uint32', 'uint64', 'int8', 'int16', 'int32', 'int64']


class TestDataFrameConstructors:

    def test_constructor_from_ndarray_with_str_dtype(self) -> None:
        arr: np.ndarray = np.arange(12).reshape(4, 3)
        df = DataFrame(arr, dtype=str)
        expected = DataFrame(arr.astype(str), dtype='str')
        tm.assert_frame_equal(df, expected)

    def test_constructor_from_2d_datetimearray(self) -> None:
        dti = date_range('2016-01-01', periods=6, tz='US/Pacific')
        dta = dti._data.reshape(3, 2)
        df = DataFrame(dta)
        expected = DataFrame({0: dta[:, 0], 1: dta[:, 1]})
        tm.assert_frame_equal(df, expected)
        assert len(df._mgr.blocks) == 1

    def test_constructor_dict_with_tzaware_scalar(self) -> None:
        dt = Timestamp('2019-11-03 01:00:00-0700').tz_convert('America/Los_Angeles')
        dt = dt.as_unit('ns')
        df = DataFrame({'dt': dt}, index=[0])
        expected = DataFrame({'dt': [dt]})
        tm.assert_frame_equal(df, expected, check_index_type=False)
        df = DataFrame({'dt': dt, 'value': [1]})
        expected = DataFrame({'dt': [dt], 'value': [1]})
        tm.assert_frame_equal(df, expected)

    def test_construct_ndarray_with_nas_and_int_dtype(self) -> None:
        arr = np.array([[1, np.nan], [2, 3]])
        msg = 'Cannot convert non-finite values \\(NA or inf\\) to integer'
        with pytest.raises(IntCastingNaNError, match=msg):
            DataFrame(arr, dtype='i8')
        with pytest.raises(IntCastingNaNError, match=msg):
            Series(arr[0], dtype='i8', name=0)

    def test_construct_from_list_of_datetimes(self) -> None:
        df = DataFrame([datetime.now(), datetime.now()])
        assert df[0].dtype == np.dtype('M8[us]')

    def test_constructor_from_tzaware_datetimeindex(self) -> None:
        naive = DatetimeIndex(['2013-1-1 13:00', '2013-1-2 14:00'], name='B')
        idx = naive.tz_localize('US/Pacific')
        expected = Series(np.array(idx.tolist(), dtype='object'), name='B')
        assert expected.dtype == idx.dtype
        result = Series(idx)
        tm.assert_series_equal(result, expected)

    def test_columns_with_leading_underscore_work_with_to_dict(self) -> None:
        col_underscore = '_b'
        df = DataFrame({'a': [1, 2], col_underscore: [3, 4]})
        d = df.to_dict(orient='records')
        ref_d = [{'a': 1, col_underscore: 3}, {'a': 2, col_underscore: 4}]
        assert ref_d == d

    def test_columns_with_leading_number_and_underscore_work_with_to_dict(self) -> None:
        col_with_num = '1_b'
        df = DataFrame({'a': [1, 2], col_with_num: [3, 4]})
        d = df.to_dict(orient='records')
        ref_d = [{'a': 1, col_with_num: 3}, {'a': 2, col_with_num: 4}]
        assert ref_d == d

    def test_array_of_dt64_nat_with_td64dtype_raises(self, frame_or_series: Type[Union[DataFrame, Series]]) -> None:
        nat = np.datetime64('NaT', 'ns')
        arr: np.ndarray = np.array([nat], dtype=object)
        if frame_or_series is DataFrame:
            arr = arr.reshape(1, 1)
        msg = "Invalid type for timedelta scalar: <class 'numpy.datetime64'>"
        with pytest.raises(TypeError, match=msg):
            frame_or_series(arr, dtype='m8[ns]')

    @pytest.mark.parametrize('kind', ['m', 'M'])
    def test_datetimelike_values_with_object_dtype(self, kind: str, frame_or_series: Type[Union[DataFrame, Series]]) -> None:
        if kind == 'M':
            dtype = 'M8[ns]'
            scalar_type = Timestamp
        else:
            dtype = 'm8[ns]'
            scalar_type = Timedelta
        arr = np.arange(6, dtype='i8').view(dtype).reshape(3, 2)
        if frame_or_series is Series:
            arr = arr[:, 0]
        obj = frame_or_series(arr, dtype=object)
        assert obj._mgr.blocks[0].values.dtype == object
        assert isinstance(obj._mgr.blocks[0].values.ravel()[0], scalar_type)
        obj = frame_or_series(frame_or_series(arr), dtype=object)
        assert obj._mgr.blocks[0].values.dtype == object
        assert isinstance(obj._mgr.blocks[0].values.ravel()[0], scalar_type)
        obj = frame_or_series(frame_or_series(arr), dtype=NumpyEADtype(object))
        assert obj._mgr.blocks[0].values.dtype == object
        assert isinstance(obj._mgr.blocks[0].values.ravel()[0], scalar_type)
        if frame_or_series is DataFrame:
            sers = [Series(x) for x in arr]
            obj = frame_or_series(sers, dtype=object)
            assert obj._mgr.blocks[0].values.dtype == object
            assert isinstance(obj._mgr.blocks[0].values.ravel()[0], scalar_type)

    def test_series_with_name_not_matching_column(self) -> None:
        x = Series(range(5), name=1)
        y = Series(range(5), name=0)
        result = DataFrame(x, columns=[0])
        expected = DataFrame([], columns=[0])
        tm.assert_frame_equal(result, expected)
        result = DataFrame(y, columns=[1])
        expected = DataFrame([], columns=[1])
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('constructor', [lambda: DataFrame(), lambda: DataFrame(None), lambda: DataFrame(()), lambda: DataFrame([]), lambda: DataFrame((_ for _ in [])), lambda: DataFrame(range(0)), lambda: DataFrame(data=None), lambda: DataFrame(data=()), lambda: DataFrame(data=[]), lambda: DataFrame(data=(_ for _ in [])), lambda: DataFrame(data=range(0))])
    def test_empty_constructor(self, constructor: Callable[[], DataFrame]) -> None:
        expected = DataFrame()
        result = constructor()
        assert len(result.index) == 0
        assert len(result.columns) == 0
        tm.assert_frame_equal(result, expected)

    def test_empty_constructor_object_index(self) -> None:
        expected = DataFrame(index=RangeIndex(0), columns=RangeIndex(0))
        result = DataFrame({})
        assert len(result.index) == 0
        assert len(result.columns) == 0
        tm.assert_frame_equal(result, expected, check_index_type=True)

    @pytest.mark.parametrize('emptylike,expected_index,expected_columns', [([[]], RangeIndex(1), RangeIndex(0)), ([[], []], RangeIndex(2), RangeIndex(0)), ([(_ for _ in [])], RangeIndex(1), RangeIndex(0))])
    def test_emptylike_constructor(self, emptylike: Any, expected_index: Index, expected_columns: Index) -> None:
        expected = DataFrame(index=expected_index, columns=expected_columns)
        result = DataFrame(emptylike)
        tm.assert_frame_equal(result, expected)

    def test_constructor_mixed(self, float_string_frame: DataFrame, using_infer_string: bool) -> None:
        dtype: Any = 'str' if using_infer_string else np.object_
        assert float_string_frame['foo'].dtype == dtype

    def test_constructor_cast_failure(self) -> None:
        msg = 'could not convert string to float'
        with pytest.raises(ValueError, match=msg):
            DataFrame({'a': ['a', 'b', 'c']}, dtype=np.float64)
        df = DataFrame(np.ones((4, 2)))
        df['foo'] = np.ones((4, 2)).tolist()
        msg = 'Expected a 1D array, got an array with shape \\(4, 2\\)'
        with pytest.raises(ValueError, match=msg):
            df['test'] = np.ones((4, 2))
        df['foo2'] = np.ones((4, 2)).tolist()

    def test_constructor_dtype_copy(self) -> None:
        orig_df = DataFrame({'col1': [1.0], 'col2': [2.0], 'col3': [3.0]})
        new_df = DataFrame(orig_df, dtype=float, copy=True)
        new_df['col1'] = 200.0
        assert orig_df['col1'][0] == 1.0

    def test_constructor_dtype_nocast_view_dataframe(self) -> None:
        df = DataFrame([[1, 2]])
        should_be_view = DataFrame(df, dtype=df[0].dtype)
        should_be_view.iloc[0, 0] = 99
        assert df.values[0, 0] == 1

    def test_constructor_dtype_nocast_view_2d_array(self) -> None:
        df = DataFrame([[1, 2], [3, 4]], dtype='int64')
        df2 = DataFrame(df.values, dtype=df[0].dtype)
        assert df2._mgr.blocks[0].values.flags.c_contiguous

    def test_1d_object_array_does_not_copy(self, using_infer_string: bool) -> None:
        arr = np.array(['a', 'b'], dtype='object')
        df = DataFrame(arr, copy=False)
        if using_infer_string:
            if df[0].dtype.storage == 'pyarrow':
                pass
            else:
                assert np.shares_memory(df[0].to_numpy(), arr)
        else:
            assert np.shares_memory(df.values, arr)
        df = DataFrame(arr, dtype=object, copy=False)
        assert np.shares_memory(df.values, arr)

    def test_2d_object_array_does_not_copy(self, using_infer_string: bool) -> None:
        arr = np.array([['a', 'b'], ['c', 'd']], dtype='object')
        df = DataFrame(arr, copy=False)
        if using_infer_string:
            if df[0].dtype.storage == 'pyarrow':
                pass
            else:
                assert np.shares_memory(df[0].to_numpy(), arr)
        else:
            assert np.shares_memory(df.values, arr)
        df = DataFrame(arr, dtype=object, copy=False)
        assert np.shares_memory(df.values, arr)

    def test_constructor_dtype_list_data(self) -> None:
        df = DataFrame([[1, '2'], [None, 'a']], dtype=object)
        assert df.loc[1, 0] is None
        assert df.loc[0, 1] == '2'

    def test_constructor_list_of_2d_raises(self) -> None:
        a = DataFrame()
        b = np.empty((0, 0))
        with pytest.raises(ValueError, match='shape=\\(1, 0, 0\\)'):
            DataFrame([a])
        with pytest.raises(ValueError, match='shape=\\(1, 0, 0\\)'):
            DataFrame([b])
        a = DataFrame({'A': [1, 2]})
        with pytest.raises(ValueError, match='shape=\\(2, 2, 1\\)'):
            DataFrame([a, a])

    @pytest.mark.parametrize('typ, ad', [['float', {}], ['float', {'A': 1, 'B': 'foo', 'C': 'bar'}], ['int', {}]])
    def test_constructor_mixed_dtypes(self, typ: str, ad: Dict[str, Any]) -> None:
        if typ == 'int':
            dtypes = MIXED_INT_DTYPES
            arrays = [np.array(np.random.default_rng(2).random(10), dtype=d) for d in dtypes]
        elif typ == 'float':
            dtypes = MIXED_FLOAT_DTYPES
            arrays = [np.array(np.random.default_rng(2).integers(10, size=10), dtype=d) for d in dtypes]
        for d, a in zip(dtypes, arrays):
            assert a.dtype == d
        ad.update(dict(zip(dtypes, arrays)))
        df = DataFrame(ad)
        dtypes = MIXED_FLOAT_DTYPES + MIXED_INT_DTYPES
        for d in dtypes:
            if d in df:
                assert df.dtypes[d] == d

    def test_constructor_complex_dtypes(self) -> None:
        a = np.random.default_rng(2).random(10).astype(np.complex64)
        b = np.random.default_rng(2).random(10).astype(np.complex128)
        df = DataFrame({'a': a, 'b': b})
        assert a.dtype == df.a.dtype
        assert b.dtype == df.b.dtype

    def test_constructor_dtype_str_na_values(self, string_dtype: Any) -> None:
        df = DataFrame({'A': ['x', None]}, dtype=string_dtype)
        result = df.isna()
        expected = DataFrame({'A': [False, True]})
        tm.assert_frame_equal(result, expected)
        assert df.iloc[1, 0] is None
        df = DataFrame({'A': ['x', np.nan]}, dtype=string_dtype)
        assert np.isnan(df.iloc[1, 0])

    def test_constructor_rec(self, float_frame: DataFrame) -> None:
        rec = float_frame.to_records(index=False)
        rec.dtype.names = list(rec.dtype.names)[::-1]
        index = float_frame.index
        df = DataFrame(rec)
        tm.assert_index_equal(df.columns, Index(rec.dtype.names))
        df2 = DataFrame(rec, index=index)
        tm.assert_index_equal(df2.columns, Index(rec.dtype.names))
        tm.assert_index_equal(df2.index, index)
        rng = np.arange(len(rec))[::-1]
        df3 = DataFrame(rec, index=rng, columns=['C', 'B'])
        expected = DataFrame(rec, index=rng).reindex(columns=['C', 'B'])
        tm.assert_frame_equal(df3, expected)

    def test_constructor_bool(self) -> None:
        df = DataFrame({0: np.ones(10, dtype=bool), 1: np.zeros(10, dtype=bool)})
        assert df.values.dtype == np.bool_

    def test_constructor_overflow_int64(self) -> None:
        values = np.array([2 ** 64 - i for i in range(1, 10)], dtype=np.uint64)
        result = DataFrame({'a': values})
        assert result['a'].dtype == np.uint64
        data_scores: List[Tuple[int, int]] = [(6311132704823138710, 273), (2685045978526272070, 23), (8921811264899370420, 45), (17019687244989530680, 270), (9930107427299601010, 273)]
        dtype = [('uid', 'u8'), ('score', 'u8')]
        data = np.zeros((len(data_scores),), dtype=dtype)
        data[:] = data_scores
        df_crawls = DataFrame(data)
        assert df_crawls['uid'].dtype == np.uint64

    @pytest.mark.parametrize('values', [np.array([2 ** 64], dtype=object), np.array([2 ** 65]), [2 ** 64 + 1], np.array([-2 ** 63 - 4], dtype=object), np.array([-2 ** 64 - 1]), [-2 ** 65 - 2]])
    def test_constructor_int_overflow(self, values: Any) -> None:
        value = values[0]
        result = DataFrame(values)
        assert result[0].dtype == object
        assert result[0][0] == value

    @pytest.mark.parametrize('values', [np.array([1], dtype=np.uint16), np.array([1], dtype=np.uint32), np.array([1], dtype=np.uint64), [np.uint16(1)], [np.uint32(1)], [np.uint64(1)]])
    def test_constructor_numpy_uints(self, values: Any) -> None:
        value = values[0]
        result = DataFrame(values)
        assert result[0].dtype == value.dtype
        assert result[0][0] == value

    def test_constructor_ordereddict(self) -> None:
        nitems = 100
        nums = list(range(nitems))
        np.random.default_rng(2).shuffle(nums)
        expected = [f'A{i:d}' for i in nums]
        df = DataFrame(OrderedDict(zip(expected, [[0]] * nitems)))
        assert expected == list(df.columns)

    def test_constructor_dict(self) -> None:
        datetime_series = Series(np.arange(30, dtype=np.float64), index=date_range('2020-01-01', periods=30))
        datetime_series_short = datetime_series[5:]
        frame = DataFrame({'col1': datetime_series, 'col2': datetime_series_short})
        assert len(datetime_series) == 30
        assert len(datetime_series_short) == 25
        tm.assert_series_equal(frame['col1'], datetime_series.rename('col1'))
        exp = Series(np.concatenate([[np.nan] * 5, datetime_series_short.values]), index=datetime_series.index, name='col2')
        tm.assert_series_equal(exp, frame['col2'])
        frame = DataFrame({'col1': datetime_series, 'col2': datetime_series_short}, columns=['col2', 'col3', 'col4'])
        assert len(frame) == len(datetime_series_short)
        assert 'col1' not in frame
        assert isna(frame['col3']).all()
        assert len(DataFrame()) == 0
        msg = 'Mixing dicts with non-Series may lead to ambiguous ordering.'
        with pytest.raises(ValueError, match=msg):
            DataFrame({'A': {'a': 'a', 'b': 'b'}, 'B': ['a', 'b', 'c']})

    def test_constructor_dict_length1(self) -> None:
        frame = DataFrame({'A': {'1': 1, '2': 2}})
        tm.assert_index_equal(frame.index, Index(['1', '2']))

    def test_constructor_dict_with_index(self) -> None:
        idx = Index([0, 1, 2])
        frame = DataFrame({}, index=idx)
        assert frame.index is idx

    def test_constructor_dict_with_index_and_columns(self) -> None:
        idx = Index([0, 1, 2])
        frame = DataFrame({}, index=idx, columns=idx)
        assert frame.index is idx
        assert frame.columns is idx
        assert len(frame._series) == 3

    def test_constructor_dict_of_empty_lists(self) -> None:
        frame = DataFrame({'A': [], 'B': []}, columns=['A', 'B'])
        tm.assert_index_equal(frame.index, RangeIndex(0), exact=True)

    def test_constructor_dict_with_none(self) -> None:
        frame_none = DataFrame({'a': None}, index=[0])
        frame_none_list = DataFrame({'a': [None]}, index=[0])
        assert frame_none._get_value(0, 'a') is None
        assert frame_none_list._get_value(0, 'a') is None
        tm.assert_frame_equal(frame_none, frame_none_list)

    def test_constructor_dict_errors(self) -> None:
        msg = 'If using all scalar values, you must pass an index'
        with pytest.raises(ValueError, match=msg):
            DataFrame({'a': 0.7})
        with pytest.raises(ValueError, match=msg):
            DataFrame({'a': 0.7}, columns=['a'])

    @pytest.mark.parametrize('scalar', [2, np.nan, None, 'D'])
    def test_constructor_invalid_items_unused(self, scalar: Any) -> None:
        result = DataFrame({'a': scalar}, columns=['b'])
        expected = DataFrame(columns=['b'])
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('value', [4, np.nan, None, float('nan')])
    def test_constructor_dict_nan_key(self, value: Any) -> None:
        cols = [1, value, 3]
        idx = ['a', value]
        values = [[0, 3], [1, 4], [2, 5]]
        data = {cols[c]: Series(values[c], index=idx) for c in range(3)}
        result = DataFrame(data).sort_values(1).sort_values('a', axis=1)
        expected = DataFrame(np.arange(6, dtype='int64').reshape(2, 3), index=idx, columns=cols)
        tm.assert_frame_equal(result, expected)
        result = DataFrame(data, index=idx).sort_values('a', axis=1)
        tm.assert_frame_equal(result, expected)
        result = DataFrame(data, index=idx, columns=cols)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('value', [np.nan, None, float('nan')])
    def test_constructor_dict_nan_tuple_key(self, value: Any) -> None:
        cols = Index([(11, 21), (value, 22), (13, value)])
        idx = Index([('a', value), (value, 2)])
        values = [[0, 3], [1, 4], [2, 5]]
        data = {cols[c]: Series(values[c], index=idx) for c in range(3)}
        result = DataFrame(data).sort_values((11, 21)).sort_values(('a', value), axis=1)
        expected = DataFrame(np.arange(6, dtype='int64').reshape(2, 3), index=idx, columns=cols)
        tm.assert_frame_equal(result, expected)
        result = DataFrame(data, index=idx).sort_values(('a', value), axis=1)
        tm.assert_frame_equal(result, expected)
        result = DataFrame(data, index=idx, columns=cols)
        tm.assert_frame_equal(result, expected)

    def test_constructor_dict_order_insertion(self) -> None:
        datetime_series = Series(np.arange(10, dtype=np.float64), index=date_range('2020-01-01', periods=10))
        datetime_series_short = datetime_series[:5]
        d = {'b': datetime_series_short, 'a': datetime_series}
        frame = DataFrame(data=d)
        expected = DataFrame(data=d, columns=list('ba'))
        tm.assert_frame_equal(frame, expected)

    def test_constructor_dict_nan_key_and_columns(self) -> None:
        result = DataFrame({np.nan: [1, 2], 2: [2, 3]}, columns=[np.nan, 2])
        expected = DataFrame([[1, 2], [2, 3]], columns=[np.nan, 2])
        tm.assert_frame_equal(result, expected)

    def test_constructor_multi_index(self) -> None:
        tuples = [(2, 3), (3, 3), (3, 3)]
        mi = MultiIndex.from_tuples(tuples)
        df = DataFrame(index=mi, columns=mi)
        assert isna(df).values.ravel().all()
        tuples = [(3, 3), (2, 3), (3, 3)]
        mi = MultiIndex.from_tuples(tuples)
        df = DataFrame(index=mi, columns=mi)
        assert isna(df).values.ravel().all()

    def test_constructor_2d_index(self) -> None:
        df = DataFrame([[1]], columns=[[1]], index=[1, 2])
        expected = DataFrame([1, 1], index=Index([1, 2], dtype='int64'), columns=MultiIndex(levels=[[1]], codes=[[0]]))
        tm.assert_frame_equal(df, expected)
        df = DataFrame([[1]], columns=[[1]], index=[[1, 2]])
        expected = DataFrame([1, 1], index=MultiIndex(levels=[[1, 2]], codes=[[0, 1]]), columns=MultiIndex(levels=[[1]], codes=[[0]]))
        tm.assert_frame_equal(df, expected)

    def test_constructor_error_msgs(self) -> None:
        msg = 'Empty data passed with indices specified.'
        with pytest.raises(ValueError, match=msg):
            DataFrame(np.empty(0), index=[1])
        msg = 'Mixing dicts with non-Series may lead to ambiguous ordering.'
        with pytest.raises(ValueError, match=msg):
            DataFrame({'A': {'a': 'a', 'b': 'b'}, 'B': ['a', 'b', 'c']})
        msg = 'Shape of passed values is \\(4, 3\\), indices imply \\(3, 3\\)'
        with pytest.raises(ValueError, match=msg):
            DataFrame(np.arange(12).reshape((4, 3)), columns=['foo', 'bar', 'baz'], index=date_range('2000-01-01', periods=3))
        arr = np.array([[4, 5, 6]])
        msg = 'Shape of passed values is \\(1, 3\\), indices imply \\(1, 4\\)'
        with pytest.raises(ValueError, match=msg):
            DataFrame(index=[0], columns=range(4), data=arr)
        arr = np.array([4, 5, 6])
        msg = 'Shape of passed values is \\(3, 1\\), indices imply \\(1, 4\\)'
        with pytest.raises(ValueError, match=msg):
            DataFrame(index=[0], columns=range(4), data=arr)
        with pytest.raises(ValueError, match='Must pass 2-d input'):
            DataFrame(np.zeros((3, 3, 3)), columns=['A', 'B', 'C'], index=[1])
        msg = 'Shape of passed values is \\(2, 3\\), indices imply \\(1, 3\\)'
        with pytest.raises(ValueError, match=msg):
            DataFrame(np.random.default_rng(2).random((2, 3)), columns=['A', 'B', 'C'], index=[1])
        msg = 'Shape of passed values is \\(2, 3\\), indices imply \\(2, 2\\)'
        with pytest.raises(ValueError, match=msg):
            DataFrame(np.random.default_rng(2).random((2, 3)), columns=['A', 'B'], index=[1, 2])
        msg = '2 columns passed, passed data had 10 columns'
        with pytest.raises(ValueError, match=msg):
            DataFrame((range(10), range(10, 20)), columns=('ones', 'twos'))
        msg = 'If using all scalar values, you must pass an index'
        with pytest.raises(ValueError, match=msg):
            DataFrame({'a': False, 'b': True})

    def test_constructor_subclass_dict(self, dict_subclass: Type[dict]) -> None:
        data = {'col1': dict_subclass(((x, 10.0 * x) for x in range(10))), 'col2': dict_subclass(((x, 20.0 * x) for x in range(10)))}
        df = DataFrame(data)
        refdf = DataFrame({col: dict(val.items()) for col, val in data.items()})
        tm.assert_frame_equal(refdf, df)
        data2 = dict_subclass(data.items())
        df2 = DataFrame(data2)
        tm.assert_frame_equal(refdf, df2)

    def test_constructor_defaultdict(self, float_frame: DataFrame) -> None:
        data: Dict[Any, defaultdict] = {}
        float_frame.loc[:float_frame.index[10], 'B'] = np.nan
        for k, v in float_frame.items():
            dct: defaultdict = defaultdict(dict)
            dct.update(v.to_dict())
            data[k] = dct
        frame = DataFrame(data)
        expected = frame.reindex(index=float_frame.index)
        tm.assert_frame_equal(float_frame, expected)

    def test_constructor_dict_block(self) -> None:
        expected = np.array([[4.0, 3.0, 2.0, 1.0]])
        df = DataFrame({'d': [4.0], 'c': [3.0], 'b': [2.0], 'a': [1.0]}, columns=['d', 'c', 'b', 'a'])
        tm.assert_numpy_array_equal(df.values, expected)

    def test_constructor_dict_cast(self, using_infer_string: bool) -> None:
        test_data = {'A': {'1': 1, '2': 2}, 'B': {'1': '1', '2': '2', '3': '3'}}
        frame = DataFrame(test_data, dtype=float)
        assert len(frame) == 3
        assert frame['B'].dtype == np.float64
        assert frame['A'].dtype == np.float64
        frame = DataFrame(test_data)
        assert len(frame) == 3
        assert frame['B'].dtype == np.object_ if not using_infer_string else 'str'
        assert frame['A'].dtype == np.float64

    def test_constructor_dict_cast2(self) -> None:
        test_data = {'A': dict(zip(range(20), [f'word_{i}' for i in range(20)])), 'B': dict(zip(range(15), np.random.default_rng(2).standard_normal(15)))}
        with pytest.raises(ValueError, match='could not convert string'):
            DataFrame(test_data, dtype=float)

    def test_constructor_dict_dont_upcast(self) -> None:
        d = {'Col1': {'Row1': 'A String', 'Row2': np.nan}}
        df = DataFrame(d)
        assert isinstance(df['Col1']['Row2'], float)

    def test_constructor_dict_dont_upcast2(self) -> None:
        dm = DataFrame([[1, 2], ['a', 'b']], index=[1, 2], columns=[1, 2])
        assert isinstance(dm[1][1], int)

    def test_constructor_dict_of_tuples(self) -> None:
        data = {'a': (1, 2, 3), 'b': (4, 5, 6)}
        result = DataFrame(data)
        expected = DataFrame({k: list(v) for k, v in data.items()})
        tm.assert_frame_equal(result, expected, check_dtype=False)

    def test_constructor_dict_of_ranges(self) -> None:
        data = {'a': range(3), 'b': range(3, 6)}
        result = DataFrame(data)
        expected = DataFrame({'a': [0, 1, 2], 'b': [3, 4, 5]})
        tm.assert_frame_equal(result, expected)

    def test_constructor_dict_of_iterators(self) -> None:
        data = {'a': iter(range(3)), 'b': reversed(range(3))}
        result = DataFrame(data)
        expected = DataFrame({'a': [0, 1, 2], 'b': [2, 1, 0]})
        tm.assert_frame_equal(result, expected)

    def test_constructor_dict_of_generators(self) -> None:
        data = {'a': (i for i in range(3)), 'b': (i for i in reversed(range(3)))}
        result = DataFrame(data)
        expected = DataFrame({'a': [0, 1, 2], 'b': [2, 1, 0]})
        tm.assert_frame_equal(result, expected)

    def test_constructor_dict_multiindex(self) -> None:
        d: Dict[Union[Tuple[str, str], str], Dict[Union[Tuple[str, str], str], Union[int, float]]] = {
            ('a', 'a'): {('i', 'i'): 0, ('i', 'j'): 1, ('j', 'i'): 2},
            ('b', 'a'): {('i', 'i'): 6, ('i', 'j'): 5, ('j', 'i'): 4},
            ('b', 'c'): {('i', 'i'): 7, ('i', 'j'): 8, ('j', 'i'): 9},
        }
        _d = sorted(d.items())
        df = DataFrame(d)
        expected = DataFrame([x[1] for x in _d], index=MultiIndex.from_tuples([x[0] for x in _d])).T
        expected.index = MultiIndex.from_tuples(expected.index)
        tm.assert_frame_equal(df, expected)
        d['z'] = {'y': 123.0, ('i', 'i'): 111, ('i', 'j'): 111, ('j', 'i'): 111}
        _d.insert(0, ('z', d['z']))
        expected = DataFrame([x[1] for x in _d], index=Index([x[0] for x in _d], tupleize_cols=False)).T
        expected.index = Index(expected.index, tupleize_cols=False)
        df = DataFrame(d)
        df = df.reindex(columns=expected.columns, index=expected.index)
        tm.assert_frame_equal(df, expected)

    def test_constructor_dict_datetime64_index(self) -> None:
        dates_as_str = ['1984-02-19', '1988-11-06', '1989-12-03', '1990-03-15']

        def create_data(constructor: Callable[[str], Any]) -> Dict[int, Dict[Any, int]]:
            return {i: {constructor(s): 2 * i} for i, s in enumerate(dates_as_str)}
        data_datetime64 = create_data(np.datetime64)
        data_datetime = create_data(lambda x: datetime.strptime(x, '%Y-%m-%d'))
        data_Timestamp = create_data(Timestamp)
        expected = DataFrame([[0, None, None, None], [None, 2, None, None], [None, None, 4, None], [None, None, None, 6]], index=[Timestamp(dt) for dt in dates_as_str])
        result_datetime64 = DataFrame(data_datetime64)
        result_datetime = DataFrame(data_datetime)
        assert result_datetime.index.unit == 'us'
        result_datetime.index = result_datetime.index.as_unit('s')
        result_Timestamp = DataFrame(data_Timestamp)
        tm.assert_frame_equal(result_datetime64, expected)
        tm.assert_frame_equal(result_datetime, expected)
        tm.assert_frame_equal(result_Timestamp, expected)

    @pytest.mark.parametrize('klass,name', [(lambda x: np.timedelta64(x, 'D'), 'timedelta64'), (lambda x: timedelta(days=x), 'pytimedelta'), (lambda x: Timedelta(x, 'D'), 'Timedelta[ns]'), (lambda x: Timedelta(x, 'D').as_unit('s'), 'Timedelta[s]')])
    def test_constructor_dict_timedelta64_index(self, klass: Callable[[int], Any], name: str) -> None:
        td_as_int = [1, 2, 3, 4]
        data = {i: {klass(s): 2 * i} for i, s in enumerate(td_as_int)}
        expected = DataFrame([{0: 0, 1: None, 2: None, 3: None}, {0: None, 1: 2, 2: None, 3: None}, {0: None, 1: None, 2: 4, 3: None}, {0: None, 1: None, 2: None, 3: 6}], index=[Timedelta(td, 'D') for td in td_as_int])
        result = DataFrame(data)
        tm.assert_frame_equal(result, expected)

    def test_constructor_period_dict(self) -> None:
        a = pd.PeriodIndex(['2012-01', 'NaT', '2012-04'], freq='M')
        b = pd.PeriodIndex(['2012-02-01', '2012-03-01', 'NaT'], freq='D')
        df = DataFrame({'a': a, 'b': b})
        assert df['a'].dtype == a.dtype
        assert df['b'].dtype == b.dtype
        df = DataFrame({'a': a.astype(object).tolist(), 'b': b.astype(object).tolist()})
        assert df['a'].dtype == a.dtype
        assert df['b'].dtype == b.dtype

    def test_constructor_dict_extension_scalar(self, ea_scalar_and_dtype: Tuple[Any, Any]) -> None:
        ea_scalar, ea_dtype = ea_scalar_and_dtype
        df = DataFrame({'a': ea_scalar}, index=[0])
        assert df['a'].dtype == ea_dtype
        expected = DataFrame(index=[0], columns=['a'], data=ea_scalar)
        tm.assert_frame_equal(df, expected)

    @pytest.mark.parametrize('data,dtype', [(Period('2020-01'), PeriodDtype('M')), (Interval(left=0, right=5), IntervalDtype('int64', 'right')), (Timestamp('2011-01-01', tz='US/Eastern'), DatetimeTZDtype(unit='s', tz='US/Eastern'))])
    def test_constructor_extension_scalar_data(self, data: Any, dtype: Any) -> None:
        df = DataFrame(index=range(2), columns=['a', 'b'], data=data)
        assert df['a'].dtype == dtype
        assert df['b'].dtype == dtype
        arr = pd.array([data] * 2, dtype=dtype)
        expected = DataFrame({'a': arr, 'b': arr})
        tm.assert_frame_equal(df, expected)

    def test_nested_dict_frame_constructor(self) -> None:
        rng = pd.period_range('1/1/2000', periods=5)
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 5)), columns=rng)
        data: Dict[Any, Dict[Any, float]] = {}
        for col in df.columns:
            for row in df.index:
                data.setdefault(col, {})[row] = df._get_value(row, col)
        result = DataFrame(data, columns=rng)
        tm.assert_frame_equal(result, df)
        data = {}
        for col in df.columns:
            for row in df.index:
                data.setdefault(row, {})[col] = df._get_value(row, col)
        result = DataFrame(data, index=rng).T
        tm.assert_frame_equal(result, df)

    def _check_basic_constructor(self, empty: Callable[[Tuple[int, int]], np.ndarray]) -> None:
        mat = empty((2, 3), dtype=float)  # type: ignore[call-arg]
        frame = DataFrame(mat, columns=['A', 'B', 'C'], index=[1, 2])
        assert len(frame.index) == 2
        assert len(frame.columns) == 3
        frame = DataFrame(empty((3,)), columns=['A'], index=[1, 2, 3])  # type: ignore[arg-type]
        assert len(frame.index) == 3
        assert len(frame.columns) == 1
        if empty is not np.ones:
            msg = 'Cannot convert non-finite values \\(NA or inf\\) to integer'
            with pytest.raises(IntCastingNaNError, match=msg):
                DataFrame(mat, columns=['A', 'B', 'C'], index=[1, 2], dtype=np.int64)
            return
        else:
            frame = DataFrame(mat, columns=['A', 'B', 'C'], index=[1, 2], dtype=np.int64)
            assert frame.values.dtype == np.int64
        msg = 'Shape of passed values is \\(2, 3\\), indices imply \\(1, 3\\)'
        with pytest.raises(ValueError, match=msg):
            DataFrame(mat, columns=['A', 'B', 'C'], index=[1])
        msg = 'Shape of passed values is \\(2, 3\\), indices imply \\(2, 2\\)'
        with pytest.raises(ValueError, match=msg):
            DataFrame(mat, columns=['A', 'B'], index=[1, 2])
        with pytest.raises(ValueError, match='Must pass 2-d input'):
            DataFrame(empty((3, 3, 3)), columns=['A', 'B', 'C'], index=[1])  # type: ignore[arg-type]
        frame = DataFrame(mat)
        tm.assert_index_equal(frame.index, Index(range(2)), exact=True)
        tm.assert_index_equal(frame.columns, Index(range(3)), exact=True)
        frame = DataFrame(mat, index=[1, 2])
        tm.assert_index_equal(frame.columns, Index(range(3)), exact=True)
        frame = DataFrame(mat, columns=['A', 'B', 'C'])
        tm.assert_index_equal(frame.index, Index(range(2)), exact=True)
        frame = DataFrame(empty((0, 3)))  # type: ignore[call-arg]
        assert len(frame.index) == 0
        frame = DataFrame(empty((3, 0)))  # type: ignore[call-arg]
        assert len(frame.columns) == 0

    def test_constructor_ndarray(self) -> None:
        self._check_basic_constructor(np.ones)
    ...
    # (The rest of the code remains unchanged up to the sections requiring modification)
    ...

    def test_constructor_with_datetimes(self, using_infer_string: bool) -> None:
        intname = np.dtype(int).name
        floatname = np.dtype(np.float64).name
        objectname = np.dtype(np.object_).name
        df = DataFrame({'A': 1, 'B': 'foo', 'C': 'bar', 'D': Timestamp('20010101'), 'E': datetime(2001, 1, 2, 0, 0)}, index=np.arange(10))
        result = df.dtypes
        idx_ABCDE: Sequence[Any] = list('ABCDE')
        expected = Series([np.dtype('int64')] + [np.dtype(objectname) if not using_infer_string else pd.StringDtype(na_value=np.nan)] * 2 + [np.dtype('M8[s]'), np.dtype('M8[us]')], index=idx_ABCDE)
        tm.assert_series_equal(result, expected)
        df = DataFrame({'a': 1.0, 'b': 2, 'c': 'foo', floatname: np.array(1.0, dtype=floatname), intname: np.array(1, dtype=intname)}, index=np.arange(10))
        result = df.dtypes
        idx_labels1: Sequence[Any] = ['a', 'b', 'c', floatname, intname]
        expected = Series([np.dtype('float64')] + [np.dtype('int64')] + [np.dtype('object') if not using_infer_string else pd.StringDtype(na_value=np.nan)] + [np.dtype('float64')] + [np.dtype(intname)], index=idx_labels1)
        tm.assert_series_equal(result, expected)
        df = DataFrame({'a': 1.0, 'b': 2, 'c': 'foo', floatname: np.array([1.0] * 10, dtype=floatname), intname: np.array([1] * 10, dtype=intname)}, index=np.arange(10))
        result = df.dtypes
        idx_labels2: Sequence[Any] = ['a', 'b', 'c', floatname, intname]
        expected = Series([np.dtype('float64')] + [np.dtype('int64')] + [np.dtype('object') if not using_infer_string else pd.StringDtype(na_value=np.nan)] + [np.dtype('float64')] + [np.dtype(intname)], index=idx_labels2)
        tm.assert_series_equal(result, expected)

    def test_constructor_with_datetimes1(self) -> None:
        ind = date_range(start='2000-01-01', freq='D', periods=10)
        datetimes = [ts.to_pydatetime() for ts in ind]
        datetime_s = Series(datetimes)
        assert datetime_s.dtype == 'M8[us]'

    def test_constructor_with_datetimes2(self) -> None:
        ind = date_range(start='2000-01-01', freq='D', periods=10)
        datetimes = [ts.to_pydatetime() for ts in ind]
        dates = [ts.date() for ts in ind]
        df = DataFrame(datetimes, columns=['datetimes'])
        df['dates'] = dates
        result = df.dtypes
        idx_dd: Sequence[Any] = ['datetimes', 'dates']
        expected = Series([np.dtype('datetime64[us]'), np.dtype('object')], index=idx_dd)
        tm.assert_series_equal(result, expected)

    def test_constructor_with_datetimes3(self) -> None:
        dt = datetime(2012, 1, 1, tzinfo=zoneinfo.ZoneInfo('US/Eastern'))
        df = DataFrame({'End Date': dt}, index=[0])
        assert df.iat[0, 0] == dt
        tm.assert_series_equal(df.dtypes, Series({'End Date': 'datetime64[us, US/Eastern]'}, dtype=object))
        df = DataFrame([{'End Date': dt}])
        assert df.iat[0, 0] == dt
        tm.assert_series_equal(df.dtypes, Series({'End Date': 'datetime64[us, US/Eastern]'}, dtype=object))
    ...
    # (Unchanged code continues)
    ...

    def test_for_list_with_dtypes(self, using_infer_string: bool) -> None:
        df = DataFrame([np.arange(5) for x in range(5)])
        result = df.dtypes
        expected = Series([np.dtype('int')] * 5)
        tm.assert_series_equal(result, expected)
        df = DataFrame([np.array(np.arange(5), dtype='int32') for x in range(5)])
        result = df.dtypes
        expected = Series([np.dtype('int32')] * 5)
        tm.assert_series_equal(result, expected)
        df = DataFrame({'a': [2 ** 31, 2 ** 31 + 1]})
        assert df.dtypes.iloc[0] == np.dtype('int64')
        df = DataFrame([1, 2])
        assert df.dtypes.iloc[0] == np.dtype('int64')
        df = DataFrame([1.0, 2.0])
        assert df.dtypes.iloc[0] == np.dtype('float64')
        df = DataFrame({'a': [1, 2]})
        assert df.dtypes.iloc[0] == np.dtype('int64')
        df = DataFrame({'a': [1.0, 2.0]})
        assert df.dtypes.iloc[0] == np.dtype('float64')
        df = DataFrame({'a': 1}, index=range(3))
        assert df.dtypes.iloc[0] == np.dtype('int64')
        df = DataFrame({'a': 1.0}, index=range(3))
        assert df.dtypes.iloc[0] == np.dtype('float64')
        df = DataFrame({'a': [1, 2, 4, 7], 'b': [1.2, 2.3, 5.1, 6.3], 'c': list('abcd'), 'd': [datetime(2000, 1, 1) for i in range(4)], 'e': [1.0, 2, 4.0, 7]})
        result = df.dtypes
        idx_abcde: Sequence[Any] = list('abcde')
        expected = Series([np.dtype('int64'), np.dtype('float64'), np.dtype('object') if not using_infer_string else pd.StringDtype(na_value=np.nan), np.dtype('datetime64[us]'), np.dtype('float64')], index=idx_abcde)
        tm.assert_series_equal(result, expected)
    ...
    # (Rest of code remains unchanged)
    ...