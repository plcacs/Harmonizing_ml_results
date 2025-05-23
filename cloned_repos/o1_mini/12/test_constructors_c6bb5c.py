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
from pandas import (
    Categorical,
    CategoricalIndex,
    DataFrame,
    DatetimeIndex,
    Index,
    Interval,
    MultiIndex,
    Period,
    RangeIndex,
    Series,
    Timedelta,
    Timestamp,
    cut,
    date_range,
    isna,
)
import pandas._testing as tm
from pandas.arrays import (
    DatetimeArray,
    IntervalArray,
    PeriodArray,
    SparseArray,
    TimedeltaArray,
)
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

MIXED_FLOAT_DTYPES: List[str] = ['float16', 'float32', 'float64']
MIXED_INT_DTYPES: List[str] = ['uint8', 'uint16', 'uint32', 'uint64', 'int8', 'int16', 'int32', 'int64']


class TestDataFrameConstructors:

    def test_constructor_from_ndarray_with_str_dtype(self) -> None:
        arr = np.arange(12).reshape(4, 3)
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
        msg = r'Cannot convert non-finite values \(NA or inf\) to integer'
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

    def test_array_of_dt64_nat_with_td64dtype_raises(self, frame_or_series: Type[DataFrame] | Type[Series]) -> None:
        nat = np.datetime64('NaT', 'ns')
        arr = np.array([nat], dtype=object)
        if frame_or_series is DataFrame:
            arr = arr.reshape(1, 1)
        msg = "Invalid type for timedelta scalar: <class 'numpy.datetime64'>"
        with pytest.raises(TypeError, match=msg):
            frame_or_series(arr, dtype='m8[ns]')

    @pytest.mark.parametrize('kind', ['m', 'M'])
    def test_datetimelike_values_with_object_dtype(
        self, kind: str, frame_or_series: Type[DataFrame] | Type[Series]
    ) -> None:
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

    @pytest.mark.parametrize(
        'constructor',
        [
            lambda: DataFrame(),
            lambda: DataFrame(None),
            lambda: DataFrame(()),
            lambda: DataFrame([]),
            lambda: DataFrame((_ for _ in [])),
            lambda: DataFrame(range(0)),
            lambda: DataFrame(data=None),
            lambda: DataFrame(data=()),
            lambda: DataFrame(data=[]),
            lambda: DataFrame(data=(_ for _ in [])),
            lambda: DataFrame(data=range(0)),
        ],
    )
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

    @pytest.mark.parametrize(
        'emptylike,expected_index,expected_columns',
        [
            ([[]], RangeIndex(1), RangeIndex(0)),
            ([[], []], RangeIndex(2), RangeIndex(0)),
            ([(_ for _ in [])], RangeIndex(1), RangeIndex(0)),
        ],
    )
    def test_emptylike_constructor(
        self,
        emptylike: List[List[Any]] | List[Iterator[Any]],
        expected_index: RangeIndex,
        expected_columns: RangeIndex,
    ) -> None:
        expected = DataFrame(index=expected_index, columns=expected_columns)
        result = DataFrame(emptylike)
        tm.assert_frame_equal(result, expected)

    def test_constructor_mixed(self, float_string_frame: DataFrame, using_infer_string: bool) -> None:
        dtype: Union[str, type] = 'str' if using_infer_string else object
        assert float_string_frame['foo'].dtype == dtype

    def test_constructor_cast_failure(self) -> None:
        msg = 'could not convert string to float'
        with pytest.raises(ValueError, match=msg):
            DataFrame({'a': ['a', 'b', 'c']}, dtype=np.float64)
        df = DataFrame(np.ones((4, 2)))
        df['foo'] = np.ones((4, 2)).tolist()
        msg = r'Expected a 1D array, got an array with shape \(4, 2\)'
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
        with pytest.raises(ValueError, match=r'shape=\(1, 0, 0\)'):
            DataFrame([a])
        with pytest.raises(ValueError, match=r'shape=\(1, 0, 0\)'):
            DataFrame([b])
        a = DataFrame({'A': [1, 2]})
        with pytest.raises(ValueError, match=r'shape=\(2, 2, 1\)'):
            DataFrame([a, a])

    @pytest.mark.parametrize(
        'typ, ad',
        [
            ['float', {}],
            ['float', {'A': 1, 'B': 'foo', 'C': 'bar'}],
            ['int', {}],
        ],
    )
    def test_constructor_mixed_dtypes(
        self, typ: str, ad: Dict[str, Any]
    ) -> None:
        if typ == 'int':
            dtypes = MIXED_INT_DTYPES
            arrays = [np.array(np.random.default_rng(2).random(10), dtype=d) for d in dtypes]
        elif typ == 'float':
            dtypes = MIXED_FLOAT_DTYPES
            arrays = [
                np.array(np.random.default_rng(2).integers(10, size=10), dtype=d) for d in dtypes
            ]
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

    def test_constructor_dtype_str_na_values(self, string_dtype: str) -> None:
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
        data_scores: List[Tuple[int, int]] = [
            (6311132704823138710, 273),
            (2685045978526272070, 23),
            (8921811264899370420, 45),
            (17019687244989530680, 270),
            (9930107427299601010, 273),
        ]
        dtype = [('uid', 'u8'), ('score', 'u8')]
        data = np.zeros((len(data_scores),), dtype=dtype)
        data[:] = data_scores
        df_crawls = DataFrame(data)
        assert df_crawls['uid'].dtype == np.uint64

    @pytest.mark.parametrize(
        'values',
        [
            np.array([2 ** 64], dtype=object),
            np.array([2 ** 65]),
            [2 ** 64 + 1],
            np.array([-2 ** 63 - 4], dtype=object),
            np.array([-2 ** 64 - 1]),
            [-2 ** 65 - 2],
        ],
    )
    def test_constructor_int_overflow(self, values: List[int] | np.ndarray) -> None:
        value = values[0]
        result = DataFrame(values)
        assert result[0].dtype == object
        assert result[0][0] == value

    @pytest.mark.parametrize(
        'values',
        [
            np.array([1], dtype=np.uint16),
            np.array([1], dtype=np.uint32),
            np.array([1], dtype=np.uint64),
            [np.uint16(1)],
            [np.uint32(1)],
            [np.uint64(1)],
        ],
    )
    def test_constructor_numpy_uints(self, values: List[np.uint16] | List[np.uint32] | List[np.uint64] | np.ndarray) -> None:
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
        exp = Series(
            np.concatenate([[np.nan] * 5, datetime_series_short.values]),
            index=datetime_series.index,
            name='col2',
        )
        tm.assert_series_equal(exp, frame['col2'])
        frame = DataFrame(
            {'col1': datetime_series, 'col2': datetime_series_short},
            columns=['col2', 'col3', 'col4'],
        )
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
    def test_constructor_invalid_items_unused(self, scalar: Union[int, float, None, str]) -> None:
        result = DataFrame({'a': scalar}, columns=['b'])
        expected = DataFrame(columns=['b'])
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('value', [4, np.nan, None, float('nan')])
    def test_constructor_dict_nan_key(self, value: Union[int, float, None]) -> None:
        cols = [1, value, 3]
        idx = ['a', value]
        values = [[0, 3], [1, 4], [2, 5]]
        data: Dict[Union[int, float, None], Series] = {cols[c]: Series(values[c], index=idx) for c in range(3)}
        result = DataFrame(data).sort_values(1).sort_values('a', axis=1)
        expected = DataFrame(np.arange(6, dtype='int64').reshape(2, 3), index=idx, columns=cols)
        tm.assert_frame_equal(result, expected)
        result = DataFrame(data, index=idx).sort_values('a', axis=1)
        tm.assert_frame_equal(result, expected)
        result = DataFrame(data, index=idx, columns=cols)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('value', [np.nan, None, float('nan')])
    def test_constructor_dict_nan_tuple_key(self, value: Union[float, None]) -> None:
        cols = Index([(11, 21), (value, 22), (13, value)])
        idx = Index([('a', value), (value, 2)])
        values = [[0, 3], [1, 4], [2, 5]]
        data: Dict[Tuple[Union[float, None], int], Series] = {cols[c]: Series(values[c], index=idx) for c in range(3)}
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
        expected = DataFrame(
            [1, 1], index=Index([1, 2], dtype='int64'), columns=MultiIndex(levels=[[1]], codes=[[0]])
        )
        tm.assert_frame_equal(df, expected)
        df = DataFrame([[1]], columns=[[1]], index=[[1, 2]])
        expected = DataFrame(
            [1, 1],
            index=MultiIndex(levels=[[1, 2]], codes=[[0, 1]]),
            columns=MultiIndex(levels=[[1]], codes=[[0]]),
        )
        tm.assert_frame_equal(df, expected)

    def test_constructor_error_msgs(self) -> None:
        msg = 'Empty data passed with indices specified.'
        with pytest.raises(ValueError, match=msg):
            DataFrame(np.empty(0), index=[1])
        msg = 'Mixing dicts with non-Series may lead to ambiguous ordering.'
        with pytest.raises(ValueError, match=msg):
            DataFrame({'A': {'a': 'a', 'b': 'b'}, 'B': ['a', 'b', 'c']})
        msg = r'Shape of passed values is \(4, 3\), indices imply \(3, 3\)'
        with pytest.raises(ValueError, match=msg):
            DataFrame(
                np.arange(12).reshape((4, 3)),
                columns=['foo', 'bar', 'baz'],
                index=date_range('2000-01-01', periods=3),
            )
        arr = np.array([[4, 5, 6]])
        msg = r'Shape of passed values is \(1, 3\), indices imply \(1, 4\)'
        with pytest.raises(ValueError, match=msg):
            DataFrame(index=[0], columns=range(4), data=arr)
        arr = np.array([4, 5, 6])
        msg = r'Shape of passed values is \(3, 1\), indices imply \(1, 4\)'
        with pytest.raises(ValueError, match=msg):
            DataFrame(index=[0], columns=range(4), data=arr)
        with pytest.raises(ValueError, match='Must pass 2-d input'):
            DataFrame(np.zeros((3, 3, 3)), columns=['A', 'B', 'C'], index=[1])
        msg = r'Shape of passed values is \(2, 3\), indices imply \(1, 3\)'
        with pytest.raises(ValueError, match=msg):
            DataFrame(np.random.default_rng(2).random((2, 3)), columns=['A', 'B', 'C'], index=[1])
        msg = r'Shape of passed values is \(2, 3\), indices imply \(2, 2\)'
        with pytest.raises(ValueError, match=msg):
            DataFrame(np.random.default_rng(2).random((2, 3)), columns=['A', 'B'], index=[1, 2])
        msg = '2 columns passed, passed data had 10 columns'
        with pytest.raises(ValueError, match=msg):
            DataFrame((range(10), range(10, 20)), columns=('ones', 'twos'))
        msg = 'If using all scalar values, you must pass an index'
        with pytest.raises(ValueError, match=msg):
            DataFrame({'a': False, 'b': True})

    def test_constructor_subclass_dict(self, dict_subclass: Type[Dict[Any, Any]]) -> None:
        data = {
            'col1': dict_subclass(((x, 10.0 * x) for x in range(10))),
            'col2': dict_subclass(((x, 20.0 * x) for x in range(10))),
        }
        df = DataFrame(data)
        refdf = DataFrame({col: dict(val.items()) for col, val in data.items()})
        tm.assert_frame_equal(refdf, df)
        data = dict_subclass(data.items())
        df = DataFrame(data)
        tm.assert_frame_equal(refdf, df)

    def test_constructor_defaultdict(self, float_frame: DataFrame) -> None:
        data: Dict[str, defaultdict[Any, Any]] = {}
        float_frame.loc[: float_frame.index[10], 'B'] = np.nan
        for k, v in float_frame.items():
            dct = defaultdict(dict)
            dct.update(v.to_dict())
            data[k] = dct
        frame = DataFrame(data)
        expected = frame.reindex(index=float_frame.index)
        tm.assert_frame_equal(float_frame, expected)

    def test_constructor_dict_block(self) -> None:
        expected = np.array([[4.0, 3.0, 2.0, 1.0]])
        df = DataFrame(
            {'d': [4.0], 'c': [3.0], 'b': [2.0], 'a': [1.0]},
            columns=['d', 'c', 'b', 'a'],
        )
        tm.assert_numpy_array_equal(df.values, expected)

    def test_constructor_dict_cast(self, using_infer_string: bool) -> None:
        test_data = {'A': {'1': 1, '2': 2}, 'B': {'1': '1', '2': '2', '3': '3'}}
        frame = DataFrame(test_data, dtype=float)
        assert len(frame) == 3
        assert frame['B'].dtype == np.float64
        assert frame['A'].dtype == np.float64
        frame = DataFrame(test_data)
        assert len(frame) == 3
        assert frame['B'].dtype == (object if not using_infer_string else 'str')
        assert frame['A'].dtype == np.float64

    def test_constructor_dict_cast2(self) -> None:
        test_data = {
            'A': dict(zip(range(20), [f'word_{i}' for i in range(20)])),
            'B': dict(zip(range(15), np.random.default_rng(2).standard_normal(15))),
        }
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
        d = {
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

        def create_data(constructor: Callable[[str], Any]) -> Dict[str, Any]:
            return {i: {constructor(s): 2 * i} for i, s in enumerate(dates_as_str)}

        data_datetime64 = create_data(np.datetime64)
        data_datetime = create_data(lambda x: datetime.strptime(x, '%Y-%m-%d'))
        data_Timestamp = create_data(Timestamp)
        expected = DataFrame(
            [
                [0, None, None, None],
                [None, 2, None, None],
                [None, None, 4, None],
                [None, None, None, 6],
            ],
            index=[Timestamp(dt) for dt in dates_as_str],
        )
        result_datetime64 = DataFrame(data_datetime64)
        result_datetime = DataFrame(data_datetime)
        assert result_datetime.index.unit == 'us'
        result_datetime.index = result_datetime.index.as_unit('s')
        result_Timestamp = DataFrame(data_Timestamp)
        tm.assert_frame_equal(result_datetime64, expected)
        tm.assert_frame_equal(result_datetime, expected)
        tm.assert_frame_equal(result_Timestamp, expected)

    @pytest.mark.parametrize(
        'klass,name',
        [
            (lambda x: np.timedelta64(x, 'D'), 'timedelta64'),
            (lambda x: timedelta(days=x), 'pytimedelta'),
            (lambda x: Timedelta(x, 'D'), 'Timedelta[ns]'),
            (lambda x: Timedelta(x, 'D').as_unit('s'), 'Timedelta[s]'),
        ],
    )
    def test_constructor_dict_timedelta64_index(
        self, klass: Callable[[int], Any], name: str
    ) -> None:
        td_as_int = [1, 2, 3, 4]
        data = {i: {klass(s): 2 * i} for i, s in enumerate(td_as_int)}
        expected = DataFrame(
            [
                {'0': 0, '1': None, '2': None, '3': None},
                {'0': None, '1': 2, '2': None, '3': None},
                {'0': None, '1': None, '2': 4, '3': None},
                {'0': None, '1': None, '2': None, '3': 6},
            ],
            index=[Timedelta(td, 'D') for td in td_as_int],
        )
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

    def test_constructor_dict_extension_scalar(
        self, ea_scalar_and_dtype: Tuple[Any, NumpyEADtype]
    ) -> None:
        ea_scalar, ea_dtype = ea_scalar_and_dtype
        df = DataFrame({'a': ea_scalar}, index=[0])
        assert df['a'].dtype == ea_dtype
        expected = DataFrame(index=[0], columns=['a'], data=ea_scalar)
        tm.assert_frame_equal(df, expected)

    @pytest.mark.parametrize(
        'data,dtype',
        [
            (Period('2020-01'), PeriodDtype('M')),
            (Interval(left=0, right=5), IntervalDtype('int64', 'right')),
            (Timestamp('2011-01-01', tz='US/Eastern'), DatetimeTZDtype(unit='s', tz='US/Eastern')),
        ],
    )
    def test_constructor_extension_scalar_data(
        self, data: Any, dtype: NumpyEADtype
    ) -> None:
        df = DataFrame(index=range(2), columns=['a', 'b'], data=data)
        assert df['a'].dtype == dtype
        assert df['b'].dtype == dtype
        arr = pd.array([data] * 2, dtype=dtype)
        expected = DataFrame({'a': arr, 'b': arr})
        tm.assert_frame_equal(df, expected)

    def test_nested_dict_frame_constructor(self) -> None:
        rng = pd.period_range('1/1/2000', periods=5)
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 5)), columns=rng)
        data: Dict[Any, Any] = {}
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

    def _check_basic_constructor(
        self, empty: Callable[[Tuple[int, ...], dtype], np.ndarray]
    ) -> None:
        mat = empty((2, 3), dtype=float)
        frame = DataFrame(mat, columns=['A', 'B', 'C'], index=[1, 2])
        assert len(frame.index) == 2
        assert len(frame.columns) == 3
        frame = DataFrame(empty((3,), dtype=float), columns=['A'], index=[1, 2, 3])
        assert len(frame.index) == 3
        assert len(frame.columns) == 1
        if empty is not np.ones:
            msg = r'Cannot convert non-finite values \(NA or inf\) to integer'
            with pytest.raises(IntCastingNaNError, match=msg):
                DataFrame(mat, columns=['A', 'B', 'C'], index=[1, 2], dtype=np.int64)
            return
        else:
            frame = DataFrame(mat, columns=['A', 'B', 'C'], index=[1, 2], dtype=np.int64)
            assert frame.values.dtype == np.int64
        msg = r'Shape of passed values is \(2, 3\), indices imply \(1, 3\)'
        with pytest.raises(ValueError, match=msg):
            DataFrame(mat, columns=['A', 'B', 'C'], index=[1])
        msg = r'Shape of passed values is \(2, 3\), indices imply \(2, 2\)'
        with pytest.raises(ValueError, match=msg):
            DataFrame(mat, columns=['A', 'B'], index=[1, 2])
        with pytest.raises(ValueError, match='Must pass 2-d input'):
            DataFrame(empty((3, 3, 3), dtype=float), columns=['A', 'B', 'C'], index=[1])
        frame = DataFrame(mat)
        tm.assert_index_equal(frame.index, Index(range(2)), exact=True)
        tm.assert_index_equal(frame.columns, Index(range(3)), exact=True)
        frame = DataFrame(mat, index=[1, 2])
        tm.assert_index_equal(frame.columns, Index(range(3)), exact=True)
        frame = DataFrame(mat, columns=['A', 'B', 'C'])
        tm.assert_index_equal(frame.index, Index(range(2)), exact=True)
        frame = DataFrame(empty((0, 3), dtype=float), dtype=float)
        assert len(frame.index) == 0
        frame = DataFrame(empty((3, 0), dtype=float), dtype=float)
        assert len(frame.columns) == 0

    def test_constructor_ndarray(self) -> None:
        self._check_basic_constructor(np.ones)
        frame = DataFrame(['foo', 'bar'], index=[0, 1], columns=['A'])
        assert len(frame) == 2

    def test_constructor_maskedarray(self) -> None:
        self._check_basic_constructor(ma.masked_all)
        mat = ma.masked_all((2, 3), dtype=float)
        mat[0, 0] = 1.0
        mat[1, 2] = 2.0
        frame = DataFrame(mat, columns=['A', 'B', 'C'], index=[1, 2])
        assert 1.0 == frame['A'][1]
        assert 2.0 == frame['C'][2]
        mat = ma.masked_all((2, 3), dtype=float)
        frame = DataFrame(mat, columns=['A', 'B', 'C'], index=[1, 2])
        assert np.all(~np.asarray(frame == frame))

    @pytest.mark.filterwarnings('ignore:elementwise comparison failed:DeprecationWarning')
    def test_constructor_maskedarray_nonfloat(self) -> None:
        mat = ma.masked_all((2, 3), dtype=int)
        frame = DataFrame(mat, columns=['A', 'B', 'C'], index=[1, 2])
        assert len(frame.index) == 2
        assert len(frame.columns) == 3
        assert np.all(~np.asarray(frame == frame))
        frame = DataFrame(mat, columns=['A', 'B', 'C'], index=[1, 2], dtype=np.float64)
        assert frame.values.dtype == np.float64
        mat2 = ma.copy(mat)
        mat2[0, 0] = 1
        mat2[1, 2] = 2
        frame = DataFrame(mat2, columns=['A', 'B', 'C'], index=[1, 2])
        assert 1 == frame['A'][1]
        assert 2 == frame['C'][2]
        mat = ma.masked_all((2, 3), dtype='M8[ns]')
        frame = DataFrame(mat, columns=['A', 'B', 'C'], index=[1, 2])
        assert len(frame.index) == 2
        assert len(frame.columns) == 3
        assert isna(frame).values.all()
        msg = r'datetime64\[ns\] values and dtype=int64 is not supported'
        with pytest.raises(TypeError, match=msg):
            DataFrame(mat, columns=['A', 'B', 'C'], index=[1, 2], dtype=np.int64)
        mat2 = ma.copy(mat)
        mat2[0, 0] = 1
        mat2[1, 2] = 2
        frame = DataFrame(mat2, columns=['A', 'B', 'C'], index=[1, 2])
        assert 1 == frame['A'].astype('i8')[1]
        assert 2 == frame['C'].astype('i8')[2]
        mat = ma.masked_all((2, 3), dtype=bool)
        frame = DataFrame(mat, columns=['A', 'B', 'C'], index=[1, 2])
        assert len(frame.index) == 2
        assert len(frame.columns) == 3
        assert np.all(~np.asarray(frame == frame))
        frame = DataFrame(mat, columns=['A', 'B', 'C'], index=[1, 2], dtype=object)
        assert frame.values.dtype == object
        mat2 = ma.copy(mat)
        mat2[0, 0] = True
        mat2[1, 2] = False
        frame = DataFrame(mat2, columns=['A', 'B', 'C'], index=[1, 2])
        assert frame['A'][1] is True
        assert frame['C'][2] is False

    def test_constructor_maskedarray_hardened(self) -> None:
        mat_hard = ma.masked_all((2, 2), dtype=float).harden_mask()
        result = DataFrame(mat_hard, columns=['A', 'B'], index=[1, 2])
        expected = DataFrame(
            {'A': [np.nan, np.nan], 'B': [np.nan, np.nan]},
            columns=['A', 'B'],
            index=[1, 2],
            dtype=float,
        )
        tm.assert_frame_equal(result, expected)
        mat_hard = ma.ones((2, 2), dtype=float).harden_mask()
        result = DataFrame(mat_hard, columns=['A', 'B'], index=[1, 2])
        expected = DataFrame(
            {'A': [1.0, 1.0], 'B': [1.0, 1.0]},
            columns=['A', 'B'],
            index=[1, 2],
            dtype=float,
        )
        tm.assert_frame_equal(result, expected)

    def test_constructor_maskedrecarray_dtype(self) -> None:
        data = np.ma.array(
            np.ma.zeros(5, dtype=[('date', '<f8'), ('price', '<f8')]),
            mask=[False] * 5,
        )
        data = data.view(mrecords.mrecarray)
        with pytest.raises(TypeError, match=r'Pass \{name: data\[name\]'):
            DataFrame(data, dtype=int)

    def test_constructor_corner_shape(self) -> None:
        df = DataFrame(index=[])
        assert df.values.shape == (0, 0)

    @pytest.mark.parametrize(
        'data,index,columns,dtype,expected',
        [
            (None, list(range(10)), ['a', 'b'], object, np.object_),
            (None, None, ['a', 'b'], 'int64', np.dtype('int64')),
            (None, list(range(10)), ['a', 'b'], int, np.dtype('float64')),
            ({}, None, ['foo', 'bar'], None, np.object_),
            ({'b': 1}, list(range(10)), list('abc'), int, np.dtype('float64')),
        ],
    )
    def test_constructor_dtype(
        self,
        data: Any,
        index: Optional[List[int]] | Index,
        columns: List[str] | Index,
        dtype: Optional[Union[str, type]],
        expected: np.dtype,
    ) -> None:
        df = DataFrame(data, index, columns, dtype)
        assert df.values.dtype == expected

    @pytest.mark.parametrize(
        'data,input_dtype,expected_dtype',
        [
            ([True, False, None], 'boolean', pd.BooleanDtype),
            ([1.0, 2.0, None], 'Float64', pd.Float64Dtype),
            ([1, 2, None], 'Int64', pd.Int64Dtype),
            (['a', 'b', 'c'], 'string', pd.StringDtype),
        ],
    )
    def test_constructor_dtype_nullable_extension_arrays(
        self,
        data: List[Any],
        input_dtype: str,
        expected_dtype: Type[pd.api.extensions.ExtensionDtype],
    ) -> None:
        df = DataFrame({'a': data}, dtype=input_dtype)
        assert df['a'].dtype == expected_dtype()

    def test_constructor_scalar_inference(self, using_infer_string: bool) -> None:
        data = {'int': 1, 'bool': True, 'float': 3.0, 'complex': 4j, 'object': 'foo'}
        df = DataFrame(data, index=np.arange(10))
        assert df['int'].dtype == np.int64
        assert df['bool'].dtype == np.bool_
        assert df['float'].dtype == np.float64
        assert df['complex'].dtype == np.complex128
        assert df['object'].dtype == (object if not using_infer_string else 'str')

    def test_constructor_arrays_and_scalars(self) -> None:
        df = DataFrame({'a': np.random.default_rng(2).standard_normal(10), 'b': True})
        exp = DataFrame({'a': df['a'].values, 'b': [True] * 10})
        tm.assert_frame_equal(df, exp)
        with pytest.raises(ValueError, match='must pass an index'):
            DataFrame({'a': False, 'b': True})

    def test_constructor_DataFrame(self, float_frame: DataFrame) -> None:
        df = DataFrame(float_frame)
        tm.assert_frame_equal(df, float_frame)
        df_casted = DataFrame(float_frame, dtype=np.int64)
        assert df_casted.values.dtype == np.int64

    def test_constructor_empty_dataframe(self) -> None:
        actual = DataFrame(DataFrame(), dtype='object')
        expected = DataFrame([], dtype='object')
        tm.assert_frame_equal(actual, expected)

    def test_constructor_more(self, float_frame: DataFrame) -> None:
        arr = np.random.default_rng(2).standard_normal(10)
        dm = DataFrame(arr, columns=['A'], index=np.arange(10))
        assert dm.values.ndim == 2
        arr = np.random.default_rng(2).standard_normal(0)
        dm = DataFrame(arr)
        assert dm.values.ndim == 2
        assert dm.values.ndim == 2
        dm = DataFrame(columns=['A', 'B'], index=np.arange(10))
        assert dm.values.shape == (10, 2)
        dm = DataFrame(columns=['A', 'B'])
        assert dm.values.shape == (0, 2)
        dm = DataFrame(index=np.arange(10))
        assert dm.values.shape == (10, 0)
        mat = np.array(['foo', 'bar'], dtype=object).reshape(2, 1)
        msg = "could not convert string to float: 'foo'"
        with pytest.raises(ValueError, match=msg):
            DataFrame(mat, index=[0, 1], columns=[0], dtype=float)
        dm = DataFrame(DataFrame(float_frame._series))
        tm.assert_frame_equal(dm, float_frame)
        dm = DataFrame(
            {'A': np.ones(10, dtype=int), 'B': np.ones(10, dtype=np.float64)},
            index=np.arange(10),
        )
        assert len(dm.columns) == 2
        assert dm.values.dtype == np.float64

    def test_constructor_empty_list(self) -> None:
        df = DataFrame([], index=[])
        expected = DataFrame(index=[])
        tm.assert_frame_equal(df, expected)
        df = DataFrame([], columns=['A', 'B'])
        expected = DataFrame({}, columns=['A', 'B'])
        tm.assert_frame_equal(df, expected)

        def empty_gen() -> Iterator[Any]:
            yield from ()

        df = DataFrame(empty_gen(), columns=['A', 'B'])
        tm.assert_frame_equal(df, expected)

    def test_constructor_list_of_lists(self, using_infer_string: bool) -> None:
        df = DataFrame(data=[[1, 'a'], [2, 'b']], columns=['num', 'str'])
        assert is_integer_dtype(df['num'])
        assert df['str'].dtype == (object if not using_infer_string else 'str')
        expected = DataFrame(np.arange(10))
        data = [np.array(x) for x in range(10)]
        result = DataFrame(data)
        tm.assert_frame_equal(result, expected)

    def test_nested_pandasarray_matches_nested_ndarray(self) -> None:
        ser = Series([1, 2])
        arr = np.array([None, None], dtype=object)
        arr[0] = ser
        arr[1] = ser * 2
        df = DataFrame(arr)
        expected = DataFrame(pd.array(arr))
        tm.assert_frame_equal(df, expected)
        assert df.shape == (2, 1)
        tm.assert_numpy_array_equal(df[0].values, arr)

    def test_constructor_list_like_data_nested_list_column(self) -> None:
        arrays = [list('abcd'), list('cdef')]
        result = DataFrame([[1, 2, 3, 4], [4, 5, 6, 7]], columns=arrays)
        mi = MultiIndex.from_arrays(arrays)
        expected = DataFrame([[1, 2, 3, 4], [4, 5, 6, 7]], columns=mi)
        tm.assert_frame_equal(result, expected)

    def test_constructor_wrong_length_nested_list_column(self) -> None:
        arrays = [list('abc'), list('cde')]
        msg = '3 columns passed, passed data had 4'
        with pytest.raises(ValueError, match=msg):
            DataFrame([[1, 2, 3, 4], [4, 5, 6, 7]], columns=arrays)

    def test_constructor_unequal_length_nested_list_column(self) -> None:
        arrays = [list('abcd'), list('cde')]
        msg = 'all arrays must be same length'
        with pytest.raises(ValueError, match=msg):
            DataFrame([[1, 2, 3, 4], [4, 5, 6, 7]], columns=arrays)

    @pytest.mark.parametrize(
        'data',
        [
            [Timestamp('2021-01-01')],
            [{'x': Timestamp('2021-01-01')}],
            {'x': [Timestamp('2021-01-01')]},
            {'x': Timestamp('2021-01-01')},
        ],
    )
    def test_constructor_one_element_data_list(
        self, data: Any
    ) -> None:
        result = DataFrame(data, index=range(3), columns=['x'])
        expected = DataFrame({'x': [Timestamp('2021-01-01')] * 3})
        tm.assert_frame_equal(result, expected)

    def test_constructor_sequence_like(self) -> None:

        class DummyContainer(abc.Sequence):
            def __init__(self, lst: List[Any]) -> None:
                self._lst = lst

            def __getitem__(self, n: int) -> Any:
                return self._lst.__getitem__(n)

            def __len__(self) -> int:
                return self._lst.__len__()

        lst_containers = [DummyContainer([1, 'a']), DummyContainer([2, 'b'])]
        columns = ['num', 'str']
        result = DataFrame(lst_containers, columns=columns)
        expected = DataFrame([[1, 'a'], [2, 'b']], columns=columns)
        tm.assert_frame_equal(result, expected, check_dtype=False)

    def test_constructor_stdlib_array(self) -> None:
        result = DataFrame({'A': array.array('i', range(10))})
        expected = DataFrame({'A': list(range(10))})
        tm.assert_frame_equal(result, expected, check_dtype=False)
        expected = DataFrame([list(range(10)), list(range(10))])
        result = DataFrame([array.array('i', range(10)), array.array('i', range(10))])
        tm.assert_frame_equal(result, expected, check_dtype=False)

    def test_constructor_range(self) -> None:
        result = DataFrame(range(10))
        expected = DataFrame(list(range(10)))
        tm.assert_frame_equal(result, expected)

    def test_constructor_list_of_ranges(self) -> None:
        result = DataFrame([range(10), range(10)])
        expected = DataFrame([list(range(10)), list(range(10))])
        tm.assert_frame_equal(result, expected)

    def test_constructor_iterable(self) -> None:

        class Iter:
            def __iter__(self) -> Iterator[List[Any]]:
                for i in range(10):
                    yield [1, 2, 3]

        expected = DataFrame([[1, 2, 3]] * 10)
        result = DataFrame(Iter())
        tm.assert_frame_equal(result, expected)

    def test_constructor_iterator(self) -> None:
        result = DataFrame(iter(range(10)))
        expected = DataFrame(list(range(10)))
        tm.assert_frame_equal(result, expected)

    def test_constructor_list_of_iterators(self) -> None:
        result = DataFrame([iter(range(10)), iter(range(10))])
        expected = DataFrame([list(range(10)), list(range(10))])
        tm.assert_frame_equal(result, expected)

    def test_constructor_generator(self) -> None:
        gen1 = (i for i in range(10))
        gen2 = (i for i in range(10))
        expected = DataFrame([list(range(10)), list(range(10))])
        result = DataFrame([gen1, gen2])
        tm.assert_frame_equal(result, expected)
        gen = ([i, 'a'] for i in range(10))
        result = DataFrame(gen)
        expected = DataFrame({0: range(10), 1: 'a'})
        tm.assert_frame_equal(result, expected, check_dtype=False)

    def test_constructor_list_of_dicts(self) -> None:
        result = DataFrame([{}])
        expected = DataFrame(index=RangeIndex(1), columns=[])
        tm.assert_frame_equal(result, expected)

    def test_constructor_ordered_dict_nested_preserve_order(self) -> None:
        nested1 = OrderedDict([('b', 1), ('a', 2)])
        nested2 = OrderedDict([('b', 2), ('a', 5)])
        data = OrderedDict([('col2', nested1), ('col1', nested2)])
        result = DataFrame(data)
        data = {'col2': [1, 2], 'col1': [2, 5]}
        expected = DataFrame(data=data, columns=['b', 'a'])
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('dict_type', [dict, OrderedDict])
    def test_constructor_ordered_dict_preserve_order(self, dict_type: Type[dict]) -> None:
        expected = DataFrame([[2, 1]], columns=['b', 'a'])
        data = dict_type()
        data['b'] = [2]
        data['a'] = [1]
        result = DataFrame(data)
        tm.assert_frame_equal(result, expected)
        data = dict_type()
        data['b'] = 2
        data['a'] = 1
        result = DataFrame([data])
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('dict_type', [dict, OrderedDict])
    def test_constructor_ordered_dict_conflicting_orders(
        self, dict_type: Type[dict]
    ) -> None:
        row_one: Dict[str, Any] = dict_type()
        row_one['b'] = 2
        row_one['a'] = 1
        row_two: Dict[str, Any] = dict_type()
        row_two['a'] = 1
        row_two['b'] = 2
        row_three: Dict[str, Any] = {'b': 2, 'a': 1}
        expected = DataFrame([[2, 1], [2, 1]], columns=['b', 'a'])
        result = DataFrame([row_one, row_two])
        tm.assert_frame_equal(result, expected)
        expected = DataFrame([[2, 1], [2, 1], [2, 1]], columns=['b', 'a'])
        result = DataFrame([row_one, row_two, row_three])
        tm.assert_frame_equal(result, expected)

    def test_constructor_list_of_series_aligned_index(
        self
    ) -> None:
        series = [Series(i, index=['b', 'a', 'c'], name=str(i)) for i in range(3)]
        result = DataFrame(series)
        expected = DataFrame(
            {'b': [0, 1, 2], 'a': [0, 1, 2], 'c': [0, 1, 2]},
            columns=['b', 'a', 'c'],
            index=['0', '1', '2'],
        )
        tm.assert_frame_equal(result, expected)

    def test_constructor_list_of_derived_dicts(self) -> None:
        class CustomDict(dict):
            pass

        d = {'a': 1.5, 'b': 3}
        data_custom = [CustomDict(d)]
        data = [d]
        result_custom = DataFrame(data_custom)
        result = DataFrame(data)
        tm.assert_frame_equal(result, result_custom)

    def test_constructor_ragged(self) -> None:
        data = {'A': np.random.default_rng(2).standard_normal(10), 'B': np.random.default_rng(2).standard_normal(8)}
        with pytest.raises(ValueError, match='All arrays must be of the same length'):
            DataFrame(data)

    def test_constructor_scalar(self) -> None:
        idx = Index(range(3))
        df = DataFrame({'a': 0}, index=idx)
        expected = DataFrame({'a': [0, 0, 0]}, index=idx)
        tm.assert_frame_equal(df, expected, check_dtype=False)

    def test_constructor_Series_copy_bug(self, float_frame: DataFrame) -> None:
        df = DataFrame(float_frame['A'], index=float_frame.index, columns=['A'])
        df.copy()

    def test_constructor_mixed_dict_nonseries(self) -> None:
        data = {}
        data['A'] = {'foo': 1, 'bar': 2, 'baz': 3}
        data['B'] = Series([4, 3, 2, 1], index=['bar', 'qux', 'baz', 'foo'])
        result = DataFrame(data)
        assert result.index.is_monotonic_increasing
        with pytest.raises(ValueError, match='ambiguous ordering'):
            DataFrame({'A': ['a', 'b'], 'B': {'a': 'a', 'b': 'b'}})
        result = DataFrame({'A': ['a', 'b'], 'B': Series(['a', 'b'], index=['a', 'b'])})
        expected = DataFrame({'A': ['a', 'b'], 'B': ['a', 'b']}, index=['a', 'b'])
        tm.assert_frame_equal(result, expected)

    def test_constructor_mixed_type_rows(self) -> None:
        data = [[1, 2], (3, 4)]
        result = DataFrame(data)
        expected = DataFrame([[1, 2], [3, 4]])
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        'tuples,lists',
        [
            ((), []),
            (((),), [[]]),
            (((), ()), [(), ()]),
            (((), ()), [[], []]),
            (([], []), [[], []]),
            (([1], [2]), [[1], [2]]),
            (([1, 2, 3], [4, 5, 6]), [[1, 2, 3], [4, 5, 6]]),
        ],
    )
    def test_constructor_tuple(
        self,
        tuples: Tuple[Any, ...],
        lists: List[Any],
    ) -> None:
        result = DataFrame(tuples)
        expected = DataFrame(lists)
        tm.assert_frame_equal(result, expected)

    def test_constructor_list_of_tuples(self) -> None:
        result = DataFrame({'A': [(1, 2), (3, 4)]})
        expected = DataFrame({'A': Series([(1, 2), (3, 4)])})
        tm.assert_frame_equal(result, expected)

    def test_constructor_list_of_namedtuples(self) -> None:
        named_tuple = namedtuple('Pandas', list('ab'))
        tuples = [named_tuple(1, 3), named_tuple(2, 4)]
        expected = DataFrame({'a': [1, 2], 'b': [3, 4]})
        result = DataFrame(tuples)
        tm.assert_frame_equal(result, expected)
        expected = DataFrame({'y': [1, 2], 'z': [3, 4]})
        result = DataFrame(tuples, columns=['y', 'z'])
        tm.assert_frame_equal(result, expected)

    def test_constructor_list_of_dataclasses(self) -> None:
        Point = make_dataclass('Point', [('x', int), ('y', int)])
        data = [Point(0, 3), Point(1, 3)]
        expected = DataFrame({'x': [0, 1], 'y': [3, 3]})
        result = DataFrame(data)
        tm.assert_frame_equal(result, expected)

    def test_constructor_list_of_dataclasses_with_varying_types(self) -> None:
        Point = make_dataclass('Point', [('x', int), ('y', int)])
        HLine = make_dataclass('HLine', [('x0', int), ('x1', int), ('y', int)])
        data = [Point(0, 3), HLine(1, 3, 3)]
        expected = DataFrame(
            {'x': [0, np.nan], 'y': [3, 3], 'x0': [np.nan, 1], 'x1': [np.nan, 3]}
        )
        result = DataFrame(data)
        tm.assert_frame_equal(result, expected)

    def test_constructor_list_of_dataclasses_error_thrown(self) -> None:
        Point = make_dataclass('Point', [('x', int), ('y', int)])
        msg = 'asdict() should be called on dataclass instances'
        with pytest.raises(TypeError, match=re.escape(msg)):
            DataFrame([Point(0, 0), {'x': 1, 'y': 0}])

    def test_constructor_list_of_dict_order(self) -> None:
        data = [
            {'First': 1, 'Second': 4, 'Third': 7, 'Fourth': 10},
            {'Second': 5, 'First': 2, 'Fourth': 11, 'Third': 8},
            {'Second': 6, 'First': 3, 'Fourth': 12, 'Third': 9, 'YYY': 14, 'XXX': 13},
        ]
        expected = DataFrame(
            {'First': [1, 2, 3], 'Second': [4, 5, 6], 'Third': [7, 8, 9], 'Fourth': [10, 11, 12], 'YYY': [None, None, 14], 'XXX': [None, None, 13]}
        )
        result = DataFrame(data)
        tm.assert_frame_equal(result, expected)

    def test_constructor_Series_named(self) -> None:
        a = Series([1, 2, 3], name=0)
        df = DataFrame(a)
        assert df.columns[0] == 0
        tm.assert_index_equal(df.index, a.index)
        arr = np.random.default_rng(2).standard_normal(10)
        s = Series(arr, name='x')
        df = DataFrame(s)
        expected = DataFrame({'x': s})
        tm.assert_frame_equal(df, expected)
        s = Series(arr, index=range(3, 13))
        df = DataFrame(s)
        expected = DataFrame({0: s})
        tm.assert_frame_equal(df, expected, check_column_type=False)
        msg = r'Shape of passed values is \(10, 1\), indices imply \(10, 2\)'
        with pytest.raises(ValueError, match=msg):
            DataFrame(s, columns=[1, 2])
        a = Series([], name='x', dtype=object)
        df = DataFrame(a)
        assert df.columns[0] == 'x'
        s1 = Series(arr, name='x')
        df = DataFrame([s1, arr]).T
        expected = DataFrame({'x': s1, 'Unnamed 0': arr}, columns=['x', 'Unnamed 0'])
        tm.assert_frame_equal(df, expected)
        df = DataFrame([arr, s1]).T
        expected = DataFrame({1: s1, 0: arr}, columns=range(2))
        tm.assert_frame_equal(df, expected)

    def test_constructor_Series_named_and_columns(self) -> None:
        s0 = Series(range(5), name=0)
        s1 = Series(range(5), name=1)
        tm.assert_frame_equal(DataFrame(s0, columns=[0]), s0.to_frame())
        tm.assert_frame_equal(DataFrame(s1, columns=[1]), s1.to_frame())
        assert DataFrame(s0, columns=[1]).empty
        assert DataFrame(s1, columns=[0]).empty

    def test_constructor_Series_differently_indexed(self) -> None:
        s1 = Series([1, 2, 3], index=['a', 'b', 'c'], name='x')
        s2 = Series([1, 2, 3], index=['a', 'b', 'c'])
        other_index = Index(['a', 'b'])
        df1 = DataFrame(s1, index=other_index)
        exp1 = DataFrame(s1.reindex(other_index))
        assert df1.columns[0] == 'x'
        tm.assert_frame_equal(df1, exp1)
        df2 = DataFrame(s2, index=other_index)
        exp2 = DataFrame(s2.reindex(other_index))
        assert df2.columns[0] == 0
        tm.assert_index_equal(df2.index, other_index)
        tm.assert_frame_equal(df2, exp2)

    @pytest.mark.parametrize('name_in1,name_in2,name_in3,name_out', [
        ('idx', 'idx', 'idx', 'idx'),
        ('idx', 'idx', None, None),
        ('idx', None, None, None),
        ('idx1', 'idx2', None, None),
        ('idx1', 'idx1', 'idx2', None),
        ('idx1', 'idx2', 'idx3', None),
        (None, None, None, None),
    ])
    def test_constructor_index_names(
        self,
        name_in1: Optional[str],
        name_in2: Optional[str],
        name_in3: Optional[str],
        name_out: Optional[str],
    ) -> None:
        indices = [
            Index(['a', 'a', 'b', 'b'], name=name_in1),
            Index(['x', 'y', 'x', 'y'], name=name_in2),
        ]
        if name_in3 is not None:
            multi_index = MultiIndex.from_product([indices[0], indices[1]], names=[name_in1, name_in2])
            indices.append(Index(['c', 'd', 'e'], name=name_in3))
        series = {
            c: Series([0, 1, 2], index=mi)
            for mi, c in zip(indices, ['A', 'B', 'C'])
            if c in indices
        }
        result = DataFrame(series)
        if name_out is None:
            exp_index = Index(list(series['A'].index) + list(series['B'].index), dtype=object)
        else:
            exp_index = Index(list(series['A'].index) + list(series['B'].index), name=name_out)
        tm.assert_frame_equal(result, expected=DataFrame({'A': [0, 1, 2, np.nan, np.nan], 'B': [np.nan, 0, 1, 2, np.nan], 'C': [np.nan, np.nan, np.nan, np.nan, np.nan]}))

    def test_constructor_manager_resize(
        self, float_frame: DataFrame
    ) -> None:
        index = list(float_frame.index[:5])
        columns = list(float_frame.columns[:3])
        msg = 'Passing a BlockManager to DataFrame'
        with tm.assert_produces_warning(DeprecationWarning, match=msg, check_stacklevel=False):
            result = DataFrame(float_frame._mgr, index=index, columns=columns)
        tm.assert_index_equal(result.index, Index(index))
        tm.assert_index_equal(result.columns, Index(columns))

    def test_constructor_mix_series_nonseries(self) -> None:
        df = DataFrame({'A': float_frame['A'], 'B': list(float_frame['B'])}, columns=['A', 'B'])
        tm.assert_frame_equal(df, float_frame.loc[:, ['A', 'B']])
        msg = 'does not match index length'
        with pytest.raises(ValueError, match=msg):
            DataFrame({'A': float_frame['A'], 'B': list(float_frame['B'])[:-2]})

    def test_constructor_miscast_na_int_dtype(self) -> None:
        msg = r'Cannot convert non-finite values \(NA or inf\) to integer'
        with pytest.raises(IntCastingNaNError, match=msg):
            DataFrame([[np.nan, 1], [1, 0]], dtype=np.int64)

    def test_constructor_column_duplicates(self) -> None:
        df = DataFrame([[8, 5]], columns=['a', 'a'])
        edf = DataFrame([[8, 5]])
        edf.columns = ['a', 'a']
        tm.assert_frame_equal(df, edf)
        idf = DataFrame.from_records([(8, 5)], columns=['a', 'a'])
        tm.assert_frame_equal(idf, edf)

    def test_constructor_empty_with_string_dtype(
        self, using_infer_string: bool
    ) -> None:
        expected = DataFrame(index=[0, 1], columns=[0, 1], dtype=object)
        expected_str = DataFrame(index=[0, 1], columns=[0, 1], dtype=pd.StringDtype(na_value=np.nan))
        df = DataFrame(index=[0, 1], columns=[0, 1], dtype=str)
        if using_infer_string:
            tm.assert_frame_equal(df, expected_str)
        else:
            tm.assert_frame_equal(df, expected)
        df = DataFrame(index=[0, 1], columns=[0, 1], dtype=np.str_)
        tm.assert_frame_equal(df, expected)
        df = DataFrame(index=[0, 1], columns=[0, 1], dtype='U5')
        tm.assert_frame_equal(df, expected)

    def test_constructor_empty_with_string_extension(
        self, nullable_string_dtype: pd.StringDtype
    ) -> None:
        expected = DataFrame(columns=['c1'], dtype=nullable_string_dtype)
        df = DataFrame(columns=['c1'], dtype=nullable_string_dtype)
        tm.assert_frame_equal(df, expected)

    def test_constructor_single_value(self) -> None:
        df = DataFrame(0.0, index=[1, 2, 3], columns=['a', 'b', 'c'])
        tm.assert_frame_equal(
            df,
            DataFrame(np.zeros(df.shape).astype('float64'), df.index, df.columns),
        )
        df = DataFrame(0, index=[1, 2, 3], columns=['a', 'b', 'c'])
        tm.assert_frame_equal(
            df,
            DataFrame(np.zeros(df.shape).astype('int64'), df.index, df.columns),
        )
        df = DataFrame('a', index=[1, 2], columns=['a', 'c'])
        tm.assert_frame_equal(
            df,
            DataFrame(
                np.array([['a', 'a'], ['a', 'a']], dtype=object),
                index=[1, 2],
                columns=['a', 'c'],
            ),
        )
        msg = r'Construction DataFrame constructor not properly called!'
        with pytest.raises(ValueError, match=msg):
            DataFrame('a', [1, 2])
        with pytest.raises(ValueError, match=msg):
            DataFrame('a', columns=['a', 'c'])
        msg = r'incompatible data and dtype'
        with pytest.raises(TypeError, match=msg):
            DataFrame('a', [1, 2], ['a', 'c'], float)

    def test_constructor_with_datetimes(self, using_infer_string: bool) -> None:
        intname = np.dtype(int).name
        floatname = np.dtype(np.float64).name
        objectname = np.dtype(np.object_).name
        df = DataFrame(
            {
                'A': 'foo',
                'B': 'bar',
                'C': 'baz',
                'D': Timestamp('20010101'),
                'E': datetime(2001, 1, 2, 0, 0)
            },
            index=np.arange(10),
        )
        result = df.dtypes
        expected = Series(
            [
                np.dtype('int64'),
                np.dtype(objectname) if not using_infer_string else pd.StringDtype(na_value=np.nan),
                np.dtype(objectname) if not using_infer_string else pd.StringDtype(na_value=np.nan),
                np.dtype(objectname) if not using_infer_string else pd.StringDtype(na_value=np.nan),
                np.dtype('M8[s]'),
                np.dtype('M8[us]'),
            ],
            index=list('ABCDE'),
        )
        tm.assert_series_equal(result, expected)
        df = DataFrame(
            {
                'a': [1, 2, 4, 7],
                'b': [1.2, 2.3, 5.1, 6.3],
                'c': list('abcd'),
                'd': [datetime(2000, 1, 1) for _ in range(4)],
                'e': [1.0, 2, 4.0, 7]
            }
        )
        result = df.dtypes
        expected = Series(
            [
                np.dtype('float64'),
                np.dtype('float64'),
                np.dtype('object') if not using_infer_string else pd.StringDtype(na_value=np.nan),
                np.dtype('datetime64[us]'),
                np.dtype('float64'),
            ],
            index=['a', 'b', 'c', 'd', 'e'],
        )
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
        expected = Series({'datetimes': np.dtype('datetime64[us]'), 'dates': np.dtype('object')})
        tm.assert_series_equal(result, expected)

    def test_constructor_with_datetimes3(self) -> None:
        dt = datetime(2012, 1, 1, tzinfo=zoneinfo.ZoneInfo('US/Eastern'))
        df = DataFrame({'End Date': dt}, index=[0])
        assert df.iat[0, 0] == dt
        tm.assert_series_equal(
            df.dtypes, Series({'End Date': 'datetime64[us, US/Eastern]'}, dtype=object)
        )
        df = DataFrame([{'End Date': dt}])
        assert df.iat[0, 0] == dt
        tm.assert_series_equal(
            df.dtypes, Series({'End Date': 'datetime64[us, US/Eastern]'}, dtype=object)
        )

    def test_frame_from_dict_with_mixed_tzaware_indexes(self) -> None:
        dti = date_range('2016-01-01', periods=3, tz='US/Pacific')
        ser1 = Series(range(3), index=dti)
        ser2 = Series(range(3), index=dti.tz_localize('UTC'))
        ser3 = Series(range(3), index=dti.tz_localize('US/Central'))
        ser4 = Series(range(3))
        df1 = DataFrame({'A': ser2, 'B': ser3, 'C': ser4})
        assert df1.index.is_monotonic_increasing
        with pytest.raises(TypeError, match=r'^Cannot join tz-naive with tz-aware DatetimeIndex$'):
            DataFrame({'A': ser2, 'B': ser3, 'C': ser4, 'D': ser1})
        with pytest.raises(TypeError, match=r'^Cannot join tz-naive with tz-aware DatetimeIndex$'):
            DataFrame({'A': ser2, 'B': ser3, 'D': ser1})
        result = DataFrame({'D': ser1, 'A': ser2, 'B': ser3})
        with pytest.raises(TypeError, match=r'^Cannot join tz-naive with tz-aware DatetimeIndex$'):
            DataFrame({'D': ser1, 'A': ser2, 'B': ser3})

    @pytest.mark.parametrize('klass,name', [
        (lambda x: np.timedelta64(x, 'D'), 'timedelta64'),
        (lambda x: timedelta(days=x), 'pytimedelta'),
        (lambda x: Timedelta(x, 'D'), 'Timedelta[ns]'),
        (lambda x: Timedelta(x, 'D').as_unit('s'), 'Timedelta[s]'),
    ])
    def test_from_out_of_bounds_ns_datetime(
        self,
        klass: Callable[[int], Any],
        name: str,
        constructor: Callable[[Any], DataFrame],
    ) -> None:
        scalar = klass(4)
        result = constructor(scalar)
        assert isinstance(result.dtypes.iloc[0], DatetimeTZDtype)
        assert result.dtypes.iloc[0].unit == 's'
        assert isinstance(result.iloc[0, 0], Timedelta)

    def test_constructor_datetime64_mixed_index_ctor_1681(self) -> None:
        dr = date_range('20130110', periods=3)
        ser = Series(dr)
        d = DataFrame({'A': 'foo', 'B': ser}, index=dr)
        assert d['B'].isna().all()

    def test_frame_timeseries_column(self) -> None:
        dr = date_range(start='20130101T10:00:00', periods=3, freq='min', tz='US/Eastern')
        result = DataFrame(dr, columns=['timestamps'])
        expected = DataFrame(
            {
                'timestamps': [
                    Timestamp('20130101T10:00:00', tz='US/Eastern'),
                    Timestamp('20130101T10:01:00', tz='US/Eastern'),
                    Timestamp('20130101T10:02:00', tz='US/Eastern'),
                ]
            },
            dtype='M8[ns, US/Eastern]',
        )
        tm.assert_frame_equal(result, expected)

    def test_constructor_with_timelines(self) -> None:
        # Placeholder for additional datetime test cases
        pass

    def test_constructor_with_datetimes4(self) -> None:
        i = date_range('1/1/2011', periods=5, freq='10s', tz='US/Eastern')
        i_no_tz = date_range('1/1/2011', periods=5, freq='10s')
        df = DataFrame({'a': i, 'b': i_no_tz})
        expected = DataFrame({'a': i.to_series().reset_index(drop=True), 'b': i_no_tz})
        tm.assert_frame_equal(df, expected)

    @pytest.mark.arm_slow
    @pytest.mark.parametrize('dtype', [None, 'uint8', 'category'])
    def test_constructor_range_dtype(
        self, dtype: Optional[str]
    ) -> None:
        expected = DataFrame({'A': [0, 1, 2, 3, 4]}, dtype=dtype or 'int64')
        result = DataFrame(range(5), columns=['A'], dtype=dtype)
        tm.assert_frame_equal(result, expected)
        result = DataFrame({'A': range(5)}, dtype=dtype)
        tm.assert_frame_equal(result, expected)

    def test_frame_from_list_subclass(self) -> None:
        class ListSubclass(list):
            pass

        expected = DataFrame([[1, 2, 3], [4, 5, 6]])
        result = DataFrame(ListSubclass([ListSubclass([1, 2, 3]), ListSubclass([4, 5, 6])]))
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        'extension_arr',
        [
            Categorical(list('aabbc')),
            SparseArray([1, np.nan, np.nan, np.nan]),
            IntervalArray([Interval(0, 1), Interval(1, 5)]),
            PeriodArray(pd.period_range(start='1/1/2017', end='1/1/2018', freq='M')),
        ],
    )
    def test_constructor_with_extension_array(
        self, extension_arr: pd.api.extensions.ExtensionArray
    ) -> None:
        expected = DataFrame(Series(extension_arr))
        result = DataFrame(extension_arr)
        tm.assert_frame_equal(result, expected)

    def test_datetime_date_tuple_columns_from_dict(self) -> None:
        v = date.today()
        tup = (v, v)
        result = DataFrame({tup: Series(range(3), index=range(3))}, columns=[tup])
        expected = DataFrame(Series([tup] * 3, name=0))
        tm.assert_frame_equal(result, expected)

    def test_construct_with_two_categoricalindex_series(
        self,
    ) -> None:
        s1 = Series([39, 6, 4], index=CategoricalIndex(['female', 'male', 'unknown']))
        s2 = Series(
            [2, 152, 2, 242, 150],
            index=CategoricalIndex(['f', 'female', 'm', 'male', 'unknown']),
        )
        result = DataFrame([s1, s2])
        expected = DataFrame(
            np.array([[39, 6, 4, np.nan, np.nan], [152.0, 242.0, 150.0, 2.0, 2.0]]),
            columns=['female', 'male', 'unknown', 'f', 'm'],
        )
        tm.assert_frame_equal(result, expected)

    @pytest.mark.filterwarnings('ignore:invalid value encountered in cast:RuntimeWarning')
    def test_constructor_series_nonexact_categoricalindex(self) -> None:
        ser = Series(range(100))
        ser1 = cut(ser, 10).value_counts().head(5)
        ser2 = cut(ser, 10).value_counts().tail(5)
        result = DataFrame({'1': ser1, '2': ser2})
        expected = DataFrame(
            [
                [10] * 5 + [np.nan] * 5,
                [np.nan] * 5 + [10] * 5,
            ],
            columns=['1', '2'],
        )
        tm.assert_frame_equal(expected, result)

    def test_from_M8_structured(self) -> None:
        dates = [(datetime(2012, 9, 9, 0, 0), datetime(2012, 9, 8, 15, 10))]
        arr = np.array(dates, dtype=[('Date', 'M8[us]'), ('Forecasting', 'M8[us]')])
        df = DataFrame(arr)
        assert df['Date'][0] == dates[0][0]
        assert df['Forecasting'][0] == dates[0][1]
        s = Series(arr['Date'])
        assert isinstance(s[0], Timestamp)
        assert s[0] == dates[0][0]

    def test_from_datetime_subclass(self) -> None:
        class DatetimeSubclass(datetime):
            pass

        df = DataFrame({'datetime': [DatetimeSubclass(2020, 1, 1, 1, 1)]})
        assert df['datetime'].dtype == 'datetime64[ns]'

    def test_with_mismatched_index_length_raises(self) -> None:
        dti = date_range('2016-01-01', periods=3, tz='US/Pacific')
        msg = r'Shape of passed values is \(3, 1\), indices imply \(4, 1\)'
        with pytest.raises(ValueError, match=msg):
            DataFrame(dti, index=range(4))

    def test_frame_ctor_datetime64_column(self) -> None:
        rng = date_range('1/1/2000 00:00:00', '1/1/2000 1:59:50', freq='10s')
        dates = np.asarray(rng)
        df = DataFrame({'A': np.random.default_rng(2).standard_normal(len(rng)), 'B': dates})
        assert np.issubdtype(df['B'].dtype, np.dtype('M8[ns]'))

    def test_dataframe_constructor_infer_multiindex(self) -> None:
        index_lists = [['a', 'a', 'b', 'b'], ['x', 'y', 'x', 'y']]
        multi = DataFrame(
            np.random.default_rng(2).standard_normal((4, 4)),
            index=[np.array(x) for x in index_lists],
        )
        assert isinstance(multi.index, MultiIndex)
        assert not isinstance(multi.columns, MultiIndex)
        multi = DataFrame(
            np.random.default_rng(2).standard_normal((4, 4)),
            columns=index_lists,
        )
        assert isinstance(multi.columns, MultiIndex)

    @pytest.mark.parametrize(
        'col_a, col_b, col_type',
        [
            ('3', ['3', '4'], 'utf8'),
            (3, [3, 4], 'int8'),
        ],
    )
    def test_dict_data_arrow_column_expansion(
        self, col_a: str | int, col_b: List[str | int], col_type: str
    ) -> None:
        pa = pytest.importorskip('pyarrow')
        cols = pd.arrays.ArrowExtensionArray(
            pa.array(col_b, type=pa.dictionary(pa.int8(), getattr(pa, col_type)()))
        )
        result = DataFrame({col_a: [1, 2]}, columns=cols)
        expected = DataFrame([[1, np.nan], [2, np.nan]], columns=cols)
        expected.isetitem(1, expected.iloc[:, 1].astype(object))
        tm.assert_frame_equal(result, expected)

    def test_from_dict_with_columns_na_scalar(self) -> None:
        result = DataFrame({'a': pd.NaT}, columns=['a'], index=range(2))
        expected = DataFrame({'a': Series([pd.NaT, pd.NaT])})
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        'extension_arr',
        [
            Categorical(list('aabbc')),
            SparseArray([1, np.nan, np.nan, np.nan]),
            IntervalArray([Interval(0, 1), Interval(1, 5)]),
            PeriodArray(pd.period_range(start='1/1/2017', end='1/1/2018', freq='M')),
        ],
    )
    def test_constructor_with_extension_array(
        self, extension_arr: pd.api.extensions.ExtensionArray
    ) -> None:
        expected = DataFrame(Series(extension_arr))
        result = DataFrame(extension_arr)
        tm.assert_frame_equal(result, expected)

    def test_datetime_date_tuple_columns_from_dict(self) -> None:
        v = date.today()
        tup = (v, v)
        result = DataFrame({tup: Series(range(3), index=range(3))}, columns=[tup])
        expected = DataFrame(Series([tup, tup, tup], name=tup))
        tm.assert_frame_equal(result, expected)

    def test_construct_with_two_categoricalindex_series(
        self,
    ) -> None:
        s1 = Series([39, 6, 4], index=CategoricalIndex(['female', 'male', 'unknown']))
        s2 = Series(
            [2, 152, 2, 242, 150],
            index=CategoricalIndex(['f', 'female', 'm', 'male', 'unknown']),
        )
        result = DataFrame([s1, s2])
        expected = DataFrame(
            np.array(
                [
                    [39, 6, 4, np.nan, np.nan],
                    [152.0, 242.0, 150.0, 2.0, 2.0],
                ]
            ),
            columns=['female', 'male', 'unknown', 'f', 'm'],
        )
        tm.assert_frame_equal(result, expected)

    @pytest.mark.filterwarnings('ignore:invalid value encountered in cast:RuntimeWarning')
    def test_constructor_series_nonexact_categoricalindex(self) -> None:
        ser = Series(range(100))
        ser1 = cut(ser, 10).value_counts().head(5)
        ser2 = cut(ser, 10).value_counts().tail(5)
        result = DataFrame({'1': ser1, '2': ser2})
        expected = DataFrame(
            [
                [10] * 5 + [np.nan] * 5,
                [np.nan] * 5 + [10] * 5,
            ],
            columns=['1', '2'],
        )
        tm.assert_frame_equal(expected, result)

    def test_from_M8_structured(self) -> None:
        dates = [(datetime(2012, 9, 9, 0, 0), datetime(2012, 9, 8, 15, 10))]
        arr = np.array(dates, dtype=[('Date', 'M8[us]'), ('Forecasting', 'M8[us]')])
        df = DataFrame(arr)
        assert df['Date'][0] == dates[0][0]
        assert df['Forecasting'][0] == dates[0][1]
        s = Series(arr['Date'])
        assert isinstance(s[0], Timestamp)
        assert s[0] == dates[0][0]

    def test_from_datetime_subclass(self) -> None:
        class DatetimeSubclass(datetime):
            pass

        df = DataFrame({'datetime': [DatetimeSubclass(2020, 1, 1, 1, 1)]})
        assert df['datetime'].dtype == 'datetime64[ns]'

    def test_with_mismatched_index_length_raises(self) -> None:
        dti = date_range('2016-01-01', periods=3, tz='US/Pacific')
        msg = r'Shape of passed values is \(3, 1\), indices imply \(4, 1\)'
        with pytest.raises(ValueError, match=msg):
            DataFrame(dti, index=range(4))

    def test_frame_ctor_datetime64_column(self) -> None:
        rng = date_range('1/1/2000 00:00:00', '1/1/2000 1:59:50', freq='10s')
        dates = np.asarray(rng)
        df = DataFrame({'A': np.random.default_rng(2).standard_normal(len(rng)), 'B': dates})
        assert np.issubdtype(df['B'].dtype, np.dtype('M8[ns]'))

    def test_dataframe_constructor_infer_multiindex(self) -> None:
        index_lists = [['a', 'a', 'b', 'b'], ['x', 'y', 'x', 'y']]
        multi = DataFrame(
            np.random.default_rng(2).standard_normal((4, 4)),
            index=[np.array(x) for x in index_lists],
        )
        assert isinstance(multi.index, MultiIndex)
        assert not isinstance(multi.columns, MultiIndex)
        multi = DataFrame(
            np.random.default_rng(2).standard_normal((4, 4)),
            columns=index_lists,
        )
        assert isinstance(multi.columns, MultiIndex)

    @pytest.mark.parametrize(
        'input_vals,columns,lists',
        [
            ([1, 2], ['a', 'b'], [[1, 2], [1, 2]]),
            (['1', '2'], ['a', 'b'], [['1', '2'], ['1', '2']]),
            (list(date_range('1/1/2011', periods=2, freq='h')), ['a', 'b'], [list(date_range('1/1/2011', periods=2, freq='h')), list(date_range('1/1/2011', periods=2, freq='h'))]),
            (
                list(date_range('1/1/2011', periods=2, freq='h', tz='US/Eastern')),
                ['a', 'b'],
                [list(date_range('1/1/2011', periods=2, freq='h', tz='US/Eastern')), list(date_range('1/1/2011', periods=2, freq='h', tz='US/Eastern'))],
            ),
            (
                [Interval(left=0, right=5)],
                ['a', 'b'],
                [[Interval(left=0, right=5), Interval(left=0, right=5)]],
            ),
        ],
    )
    def test_construction_with_iterations(
        self,
        input_vals: List[Union[int, str, pd.Timestamp, pd.Interval]],
        columns: List[str],
        lists: List[List[Union[int, str, pd.Timestamp, pd.Interval]]],
    ) -> None:
        result = DataFrame(input_vals, columns=columns)
        expected = DataFrame(lists, columns=columns)
        tm.assert_frame_equal(result, expected)

    def test_constructor_with_datetimes5(self) -> None:
        i = date_range('1/1/2011', periods=5, freq='10s', tz='US/Eastern')
        expected = DataFrame({'a': pd.array(i, dtype='datetime64[ns, US/Eastern]')})
        with pytest.option_context('future.infer_string', True):
            df = DataFrame({'a': i})
        tm.assert_frame_equal(df, expected)

    def test_constructor_with_datetimes6(self) -> None:
        i = date_range('1/1/2011', periods=5, freq='10s', tz='US/Eastern')
        i_no_tz = date_range('1/1/2011', periods=5, freq='10s')
        df = DataFrame({'a': i, 'b': i_no_tz})
        expected = DataFrame(
            {'a': i.to_series().reset_index(drop=True), 'b': i_no_tz},
            dtype='datetime64[us, US/Eastern]',
        )
        tm.assert_frame_equal(df, expected)

    @pytest.mark.parametrize(
        'arr',
        [
            np.array([None, None, None, None, datetime.now(), None]),
            np.array([None, None, datetime.now(), None]),
            [[np.datetime64('NaT')], [None]],
            [[np.datetime64('NaT')], [pd.NaT]],
            [[None], [np.datetime64('NaT')]],
            [[None], [pd.NaT]],
            [[pd.NaT], [np.datetime64('NaT')]],
            [[pd.NaT], [None]],
        ],
    )
    def test_constructor_datetimes_with_nulls(
        self,
        arr: Union[np.ndarray, List[List[Optional[datetime]]]],
    ) -> None:
        result = DataFrame(arr).dtypes
        unit = 'ns'
        if isinstance(arr, np.ndarray):
            unit = 'us'
        elif not any(isinstance(x, np.datetime64) for y in arr for x in y):
            unit = 's'
        expected = Series([np.dtype(f'datetime64[{unit}]')])
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        'order,unit',
        [
            ('K', 'M'),
            ('K', 'D'),
            ('K', 'h'),
            ('K', 'm'),
            ('K', 's'),
            ('K', 'ms'),
            ('K', 'us'),
            ('K', 'ns'),
            ('A', 'M'),
            ('A', 'D'),
            ('A', 'h'),
            ('A', 'm'),
            ('A', 's'),
            ('A', 'ms'),
            ('A', 'us'),
            ('A', 'ns'),
            ('C', 'M'),
            ('C', 'D'),
            ('C', 'h'),
            ('C', 'm'),
            ('C', 's'),
            ('C', 'ms'),
            ('C', 'us'),
            ('C', 'ns'),
            ('F', 'M'),
            ('F', 'D'),
            ('F', 'h'),
            ('F', 'm'),
            ('F', 's'),
            ('F', 'ms'),
            ('F', 'us'),
            ('F', 'ns'),
        ],
    )
    def test_constructor_datetimes_non_ns(
        self, order: str, unit: str
    ) -> None:
        dtype = f'datetime64[{unit}]'
        na = np.array(
            [
                ['2015-01-01', '2015-01-02', '2015-01-03'],
                ['2017-01-01', '2017-01-02', '2017-01-03'],
            ],
            dtype=dtype,
            order=order,
        )
        df = DataFrame(na)
        expected = DataFrame(na.astype('M8[ns]'))
        if unit in ['M', 'D', 'h', 'm']:
            with pytest.raises(TypeError, match=r'Cannot cast'):
                expected.astype(dtype)
            expected = expected.astype('datetime64[s]')
        else:
            expected = expected.astype(dtype=dtype)
        tm.assert_frame_equal(df, expected)

    @pytest.mark.parametrize(
        'order,unit',
        [
            ('K', 'D'),
            ('K', 'h'),
            ('K', 'm'),
            ('K', 's'),
            ('K', 'ms'),
            ('K', 'us'),
            ('K', 'ns'),
            ('A', 'D'),
            ('A', 'h'),
            ('A', 'm'),
            ('A', 's'),
            ('A', 'ms'),
            ('A', 'us'),
            ('A', 'ns'),
            ('C', 'D'),
            ('C', 'h'),
            ('C', 'm'),
            ('C', 's'),
            ('C', 'ms'),
            ('C', 'us'),
            ('C', 'ns'),
            ('F', 'D'),
            ('F', 'h'),
            ('F', 'm'),
            ('F', 's'),
            ('F', 'ms'),
            ('F', 'us'),
            ('F', 'ns'),
        ],
    )
    def test_constructor_timedelta_non_ns(
        self, order: str, unit: str
    ) -> None:
        dtype = f'timedelta64[{unit}]'
        na = np.array(
            [
                [np.timedelta64(1, 'D'), np.timedelta64(2, 'D')],
                [np.timedelta64(4, 'D'), np.timedelta64(5, 'D')],
            ],
            dtype=dtype,
            order=order,
        )
        df = DataFrame(na)
        if unit in ['D', 'h', 'm']:
            exp_unit = 's'
        else:
            exp_unit = unit
        exp_dtype = np.dtype(f'm8[{exp_unit}]')
        expected = DataFrame(
            [[Timedelta(1, 'D'), Timedelta(2, 'D')], [Timedelta(4, 'D'), Timedelta(5, 'D')]],
            dtype=exp_dtype,
        )
        tm.assert_frame_equal(df, expected)

    def test_constructor_for_list_with_dtypes(self, using_infer_string: bool) -> None:
        df = DataFrame([np.arange(5) for _ in range(5)])
        result = df.dtypes
        expected = Series([np.dtype('int')] * 5)
        tm.assert_series_equal(result, expected)
        df = DataFrame([np.array(np.arange(5), dtype='int32') for _ in range(5)])
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
        df = DataFrame(
            {
                'a': [1, 2, 4, 7],
                'b': [1.2, 2.3, 5.1, 6.3],
                'c': list('abcd'),
                'd': [datetime(2000, 1, 1) for _ in range(4)],
                'e': [1.0, 2, 4.0, 7],
            }
        )
        result = df.dtypes
        expected = Series(
            [
                np.dtype('float64'),
                np.dtype('float64'),
                np.dtype('object') if not using_infer_string else pd.StringDtype(na_value=np.nan),
                np.dtype('datetime64[us]'),
                np.dtype('float64'),
            ],
            index=['a', 'b', 'c', 'd', 'e'],
        )
        tm.assert_series_equal(result, expected)

    def test_constructor_frame_copy(self, float_frame: DataFrame) -> None:
        cop = DataFrame(float_frame, copy=True)
        cop['A'] = 5
        assert (cop['A'] == 5).all()
        assert not (float_frame['A'] == 5).all()

    def test_constructor_frame_shallow_copy(self, float_frame: DataFrame) -> None:
        orig = float_frame.copy()
        cop = DataFrame(float_frame)
        assert cop._mgr is not float_frame._mgr
        cop.index = np.arange(len(cop))
        tm.assert_frame_equal(float_frame, orig)

    def test_constructor_ndarray_copy(self, float_frame: DataFrame) -> None:
        arr = float_frame.values.copy()
        df = DataFrame(arr)
        arr[5] = 5
        assert not (df.values[5] == 5).all()
        df = DataFrame(arr, copy=True)
        arr[6] = 6
        assert not (df.values[6] == 6).all()

    def test_constructor_series_copy(
        self, float_frame: DataFrame
    ) -> None:
        series = float_frame._series
        df = DataFrame({'A': series['A']}, copy=True)
        df.loc[df.index[0]:df.index[-1], 'A'] = 5
        assert not (series['A'] == 5).all()

    @pytest.mark.parametrize(
        'df',
        [
            DataFrame([[1, 2, 3], [4, 5, 6]], index=[1, np.nan]),
            DataFrame([[1, 2, 3], [4, 5, 6]], columns=[1.1, 2.2, np.nan]),
            DataFrame([[0, 1, 2, 3], [4, 5, 6, 7]], columns=[np.nan, 1.1, 2.2, np.nan]),
            DataFrame([[0.0, 1, 2, 3.0], [4, 5, 6, 7]], columns=[np.nan, 1.1, 2.2, np.nan]),
            DataFrame([[0.0, 1, 2, 3.0], [4, 5, 6, 7]], columns=[np.nan, 1, 2, 2]),
        ],
    )
    def test_constructor_with_nas(self, df: DataFrame) -> None:
        for i in range(len(df.columns)):
            df.iloc[:, i]
        indexer = np.arange(len(df.columns))[isna(df.columns)]
        if len(indexer) == 0:
            with pytest.raises(KeyError, match=r'^nan$'):
                df.loc[:, np.nan]
        elif len(indexer) == 1:
            tm.assert_series_equal(df.iloc[:, indexer[0]], df.loc[:, np.nan])
        else:
            tm.assert_frame_equal(df.iloc[:, indexer], df.loc[:, np.nan])

    def test_constructor_lists_to_object_dtype(self) -> None:
        d = DataFrame({'a': [np.nan, False]})
        assert d['a'].dtype == object
        assert not d['a'][1]

    def test_constructor_ndarray_categorical_dtype(self) -> None:
        cat = Categorical(['A', 'B', 'C'])
        arr = np.array(cat).reshape(-1, 1)
        arr = np.broadcast_to(arr, (3, 4))
        result = DataFrame(arr, dtype=cat.dtype)
        expected = DataFrame({0: cat, 1: cat, 2: cat, 3: cat})
        tm.assert_frame_equal(result, expected)

    def test_constructor_categorical(self) -> None:
        df = DataFrame({'A': list('abc')}, dtype='category')
        expected = Series(list('abc'), dtype='category', name='A')
        tm.assert_series_equal(df['A'], expected)
        s = Series(list('abc'), dtype='category')
        result = s.to_frame()
        expected = Series(list('abc'), dtype='category', name=0)
        tm.assert_series_equal(result[0], expected)
        result = s.to_frame(name='foo')
        expected = Series(list('abc'), dtype='category', name='foo')
        tm.assert_series_equal(result['foo'], expected)
        df = DataFrame(list('abc'), dtype='category')
        expected = Series(list('abc'), dtype='category', name=0)
        tm.assert_series_equal(df[0], expected)

    def test_construct_from_1item_list_of_categorical(self) -> None:
        cat = Categorical(list('abc'))
        df = DataFrame([cat])
        expected = DataFrame(pd.Series([cat.astype(object)]))
        tm.assert_frame_equal(df, expected)

    def test_construct_from_list_of_categoricals(self) -> None:
        df = DataFrame([Categorical(list('abc')), Categorical(list('abd'))])
        expected = DataFrame(
            [['a', 'b', 'c'], ['a', 'b', 'd']],
        )
        tm.assert_frame_equal(df, expected)

    def test_from_list_like_mixed_types(self) -> None:
        df = DataFrame([Categorical(list('abc')), list('def')])
        expected = DataFrame(
            [['a', 'b', 'c'], ['d', 'e', 'f']],
        )
        tm.assert_frame_equal(df, expected)

    def test_construct_from_listlikes_mismatched_lengths(self) -> None:
        df = DataFrame([Categorical(list('abc')), Categorical(list('abdefg'))])
        expected = DataFrame(
            [[1, 2, 3], [1, 2, 3, 4, 5, 6]],
        )
        tm.assert_frame_equal(df, expected)

    def test_constructor_series_with_name_with_columns(self) -> None:
        result = DataFrame(Series(1, name='foo'), columns=['bar'])
        expected = DataFrame(columns=['bar'])
        tm.assert_frame_equal(result, expected)

    def test_constructor_series_with_name_with_columns(
        self, 
        float_frame: DataFrame
    ) -> None:
        df = DataFrame(float_frame)
        tm.assert_frame_equal(df, float_frame)

    def test_constructor_empty_dataframe(self):
        actual = DataFrame(DataFrame(), dtype='object')
        expected = DataFrame([], dtype='object')
        tm.assert_frame_equal(actual, expected)

    def test_frame_from_list_subclass(self) -> None:
        class ListSubclass(list):
            pass
        expected = DataFrame([[1, 2, 3], [4, 5, 6]])
        result = DataFrame(ListSubclass([ListSubclass([1, 2, 3]), ListSubclass([4, 5, 6])]))
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        'extension_arr',
        [
            Categorical(list('aabbc')),
            SparseArray([1, np.nan, np.nan, np.nan]),
            IntervalArray([Interval(0, 1), Interval(1, 5)]),
            PeriodArray(pd.period_range(start='1/1/2017', end='1/1/2018', freq='M')),
        ],
    )
    def test_constructor_with_extension_array(
        self, extension_arr: pd.api.extensions.ExtensionArray
    ) -> None:
        expected = DataFrame(Series(extension_arr))
        result = DataFrame(extension_arr)
        tm.assert_frame_equal(result, expected)

    def test_datetime_date_tuple_columns_from_dict(self) -> None:
        v = date.today()
        tup = (v, v)
        result = DataFrame({tup: Series(range(3), index=range(3))}, columns=[tup])
        expected = DataFrame(Series([tup, tup, tup], name=tup))
        tm.assert_frame_equal(result, expected)

    def test_construct_with_two_categoricalindex_series(
        self,
    ) -> None:
        s1 = Series([39, 6, 4], index=CategoricalIndex(['female', 'male', 'unknown']))
        s2 = Series(
            [2, 152, 2, 242, 150],
            index=CategoricalIndex(['f', 'female', 'm', 'male', 'unknown']),
        )
        result = DataFrame([s1, s2])
        expected = DataFrame(
            np.array([[39, 6, 4, np.nan, np.nan], [152.0, 242.0, 150.0, 2.0, 2.0]]),
            columns=['female', 'male', 'unknown', 'f', 'm'],
        )
        tm.assert_frame_equal(result, expected)

    def test_constructor_with_extension_arrays(self, extension_arr: pd.api.extensions.ExtensionArray) -> None:
        result = DataFrame(extension_arr)
        expected = DataFrame(Series(extension_arr))
        tm.assert_frame_equal(result, expected)

    def test_constructor_additional_cases(self) -> None:
        # Placeholder for additional test cases
        pass

    def test_constructor_complex_case(self):
        # Placeholder for additional test cases
        pass


def get1(obj: Union[Series, DataFrame]) -> Any:
    if isinstance(obj, Series):
        return obj.iloc[0]
    else:
        return obj.iloc[0, 0]


class TestFromScalar:

    @pytest.fixture(params=[list, dict, None])
    def box(self, request: pytest.FixtureRequest) -> Optional[Type[Any]]:
        return request.param

    @pytest.fixture
    def constructor(
        self,
        frame_or_series: Type[DataFrame] | Type[Series],
        box: Optional[Type[Any]],
    ) -> Callable[[Any], DataFrame | Series]:
        extra: Dict[str, Any] = {'index': range(2)}
        if frame_or_series is DataFrame:
            extra['columns'] = ['A']
        if box is None:
            return functools.partial(frame_or_series, **extra)
        elif box is dict:
            if frame_or_series is Series:
                return lambda x, **kwargs: frame_or_series({0: x, 1: x}, **extra, **kwargs)
            else:
                return lambda x, **kwargs: frame_or_series({'A': x}, **extra, **kwargs)
        elif frame_or_series is Series:
            return lambda x, **kwargs: frame_or_series([x, x], **extra, **kwargs)
        else:
            return lambda x, **kwargs: frame_or_series({'A': [x, x]}, **extra, **kwargs)

    @pytest.mark.parametrize(
        'dtype',
        ['M8[ns]', 'm8[ns]'],
    )
    def test_from_nat_scalar(
        self,
        dtype: str,
        constructor: Callable[[Any], DataFrame | Series],
    ) -> None:
        obj = constructor(pd.NaT, dtype=dtype)
        assert np.all(obj.dtypes == dtype)
        assert np.all(obj.isna())

    def test_from_timedelta_scalar_preserves_nanos(
        self, constructor: Callable[[Any], DataFrame | Series]
    ) -> None:
        td = Timedelta(1)
        obj = constructor(td, dtype='m8[ns]')
        assert get1(obj) == td

    def test_from_timestamp_scalar_preserves_nanos(
        self, constructor: Callable[[Any], DataFrame | Series], fixed_now_ts: pd.Timestamp
    ) -> None:
        ts = fixed_now_ts + Timedelta(1)
        obj = constructor(ts, dtype='M8[ns]')
        assert get1(obj) == ts

    def test_from_timedelta64_scalar_object(
        self, constructor: Callable[[Any], DataFrame | Series]
    ) -> None:
        td = Timedelta(1)
        td64 = td.to_timedelta64()
        obj = constructor(td64, dtype=object)
        assert isinstance(get1(obj), np.timedelta64)

    @pytest.mark.parametrize(
        'cls',
        [np.datetime64, np.timedelta64],
    )
    def test_from_scalar_datetimelike_mismatched(
        self, constructor: Callable[[Any], DataFrame | Series], cls: Type[Any]
    ) -> None:
        scalar = cls('NaT', 'ns')
        dtype = {np.datetime64: 'm8[ns]', np.timedelta64: 'M8[ns]'}[cls]
        if cls is np.datetime64:
            msg1 = r"Invalid type for timedelta scalar: <class 'numpy.datetime64'>"
        else:
            msg1 = r"<class 'numpy.timedelta64'> is not convertible to datetime"
        msg = f'Cannot cast|{msg1}'
        with pytest.raises(TypeError, match=msg):
            constructor(scalar, dtype=dtype)
        scalar = cls(4, 'ns')
        with pytest.raises(TypeError, match=msg):
            constructor(scalar, dtype=dtype)

    @pytest.mark.parametrize(
        'cls',
        [datetime, np.datetime64],
    )
    def test_from_out_of_bounds_ns_datetime(
        self,
        constructor: Callable[[Any], DataFrame | Series],
        cls: Type[Any],
    ) -> None:
        scalar = cls('9999-01-01T00:00:00Z')
        result = constructor(scalar)
        item = get1(result)
        assert isinstance(item, Timestamp)
        dtype = tm.get_dtype(result)
        assert dtype == 'M8[us]'

    @pytest.mark.parametrize(
        'cls',
        [timedelta, np.timedelta64],
    )
    def test_from_out_of_bounds_ns_timedelta(
        self,
        constructor: Callable[[Any], DataFrame | Series],
        cls: Type[Any],
    ) -> None:
        scalar = cls('9999-01-01T00:00:00') - cls('1970-01-01T00:00:00')
        dtype = 'm8[us]'
        result = constructor(scalar, dtype=dtype)
        item = get1(result)
        assert isinstance(item, Timedelta)
        assert item.asm8.dtype == dtype
        assert dtype == 'm8[us]'

    def test_tzaware_data_tznaive_dtype(
        self,
        constructor: Callable[[Any], DataFrame | Series],
        box: Optional[Type[Any]],
        frame_or_series: Type[DataFrame] | Type[Series],
    ) -> None:
        tz = 'US/Eastern'
        ts = Timestamp('2019', tz=tz)
        if box is None or (frame_or_series is DataFrame and box is dict):
            msg = r'Cannot unbox tzaware Timestamp to tznaive dtype'
            err = TypeError
        else:
            msg = (
                r'Cannot convert timezone-aware data to timezone-naive dtype. '
                r'Use pd.Series\(values\).dt.tz_localize\(None\) instead.'
            )
            err = ValueError
        with pytest.raises(err, match=msg):
            constructor(ts, dtype='M8[ns]')


class TestAllowNonNano:

    @pytest.fixture(params=[True, False])
    def as_td(self, request: pytest.FixtureRequest) -> bool:
        return request.param

    @pytest.fixture
    def arr(self, as_td: bool) -> Any:
        values = np.arange(5).astype(np.int64).view('M8[s]')
        if as_td:
            values = values - values[0]
            return TimedeltaArray._simple_new(values, dtype=values.dtype)
        else:
            return DatetimeArray._simple_new(values, dtype=values.dtype)

    def test_index_allow_non_nano(self, arr: Any) -> None:
        idx = Index(arr)
        assert idx.dtype == arr.dtype

    def test_dti_tdi_allow_non_nano(self, arr: Any, as_td: bool) -> None:
        if as_td:
            idx = pd.TimedeltaIndex(arr)
        else:
            idx = DatetimeIndex(arr)
        assert idx.dtype == arr.dtype

    def test_series_allow_non_nano(self, arr: Any) -> None:
        ser = Series(arr)
        assert ser.dtype == arr.dtype

    def test_frame_allow_non_nano(self, arr: Any) -> None:
        df = DataFrame(arr)
        assert df.dtypes[0] == arr.dtype

    def test_frame_from_dict_allow_non_nano(self, arr: Any) -> None:
        df = DataFrame({0: arr})
        assert df.dtypes[0] == arr.dtype
