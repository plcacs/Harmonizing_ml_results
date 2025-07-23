import array
from collections import OrderedDict, defaultdict, namedtuple
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
from typing import Any, Callable, Dict, Generator, Iterable, List, Optional, Tuple, Union

MIXED_FLOAT_DTYPES: List[str] = ['float16', 'float32', 'float64']
MIXED_INT_DTYPES: List[str] = [
    'uint8',
    'uint16',
    'uint32',
    'uint64',
    'int8',
    'int16',
    'int32',
    'int64',
]

class TestDataFrameConstructors:

    def test_constructor_from_ndarray_with_str_dtype(self) -> None:
        arr: np.ndarray = np.arange(12).reshape(4, 3)
        df: DataFrame = DataFrame(arr, dtype=str)
        expected: DataFrame = DataFrame(arr.astype(str), dtype='str')
        tm.assert_frame_equal(df, expected)

    def test_constructor_from_2d_datetimearray(self) -> None:
        dti: DatetimeIndex = date_range('2016-01-01', periods=6, tz='US/Pacific')
        dta: np.ndarray = dti._data.reshape(3, 2)
        df: DataFrame = DataFrame(dta)
        expected: DataFrame = DataFrame({0: dta[:, 0], 1: dta[:, 1]})
        tm.assert_frame_equal(df, expected)
        assert len(df._mgr.blocks) == 1

    def test_constructor_dict_with_tzaware_scalar(self) -> None:
        dt: Timestamp = Timestamp('2019-11-03 01:00:00-0700').tz_convert('America/Los_Angeles')
        dt = dt.as_unit('ns')
        df: DataFrame = DataFrame({'dt': dt}, index=[0])
        expected: DataFrame = DataFrame({'dt': [dt]})
        tm.assert_frame_equal(df, expected, check_index_type=False)
        df = DataFrame({'dt': dt, 'value': [1]})
        expected = DataFrame({'dt': [dt], 'value': [1]})
        tm.assert_frame_equal(df, expected)

    def test_construct_ndarray_with_nas_and_int_dtype(self) -> None:
        arr: np.ndarray = np.array([[1, np.nan], [2, 3]])
        msg: str = 'Cannot convert non-finite values \\(NA or inf\\) to integer'
        with pytest.raises(IntCastingNaNError, match=msg):
            DataFrame(arr, dtype='i8')
        with pytest.raises(IntCastingNaNError, match=msg):
            Series(arr[0], dtype='i8', name=0)

    def test_construct_from_list_of_datetimes(self) -> None:
        df: DataFrame = DataFrame([datetime.now(), datetime.now()])
        assert df[0].dtype == np.dtype('M8[us]')

    def test_constructor_from_tzaware_datetimeindex(self) -> None:
        naive: DatetimeIndex = DatetimeIndex(['2013-1-1 13:00', '2013-1-2 14:00'], name='B')
        idx: DatetimeIndex = naive.tz_localize('US/Pacific')
        expected: Series = Series(np.array(idx.tolist(), dtype='object'), name='B')
        assert expected.dtype == idx.dtype
        result: Series = Series(idx)
        tm.assert_series_equal(result, expected)

    def test_columns_with_leading_underscore_work_with_to_dict(self) -> None:
        col_underscore: str = '_b'
        df: DataFrame = DataFrame({'a': [1, 2], col_underscore: [3, 4]})
        d: List[Dict[str, Any]] = df.to_dict(orient='records')
        ref_d: List[Dict[str, Any]] = [{'a': 1, col_underscore: 3}, {'a': 2, col_underscore: 4}]
        assert ref_d == d

    def test_columns_with_leading_number_and_underscore_work_with_to_dict(self) -> None:
        col_with_num: str = '1_b'
        df: DataFrame = DataFrame({'a': [1, 2], col_with_num: [3, 4]})
        d: List[Dict[str, Any]] = df.to_dict(orient='records')
        ref_d: List[Dict[str, Any]] = [{'a': 1, col_with_num: 3}, {'a': 2, col_with_num: 4}]
        assert ref_d == d

    def test_array_of_dt64_nat_with_td64dtype_raises(self, frame_or_series: Callable[..., Union[DataFrame, Series]]) -> None:
        nat: np.datetime64 = np.datetime64('NaT', 'ns')
        arr: np.ndarray = np.array([nat], dtype=object)
        if frame_or_series is DataFrame:
            arr = arr.reshape(1, 1)
        msg: str = "Invalid type for timedelta scalar: <class 'numpy.datetime64'>"
        with pytest.raises(TypeError, match=msg):
            frame_or_series(arr, dtype='m8[ns]')

    @pytest.mark.parametrize('kind', ['m', 'M'])
    def test_datetimelike_values_with_object_dtype(
        self, kind: str, frame_or_series: Callable[..., Union[DataFrame, Series]]
    ) -> None:
        if kind == 'M':
            dtype: str = 'M8[ns]'
            scalar_type: type = Timestamp
        else:
            dtype = 'm8[ns]'
            scalar_type = Timedelta
        arr: np.ndarray = np.arange(6, dtype='i8').view(dtype).reshape(3, 2)
        if frame_or_series is Series:
            arr = arr[:, 0]
        obj: Union[DataFrame, Series] = frame_or_series(arr, dtype=object)
        assert obj._mgr.blocks[0].values.dtype == object
        assert isinstance(obj._mgr.blocks[0].values.ravel()[0], scalar_type)
        obj = frame_or_series(frame_or_series(arr), dtype=object)
        assert obj._mgr.blocks[0].values.dtype == object
        assert isinstance(obj._mgr.blocks[0].values.ravel()[0], scalar_type)
        obj = frame_or_series(frame_or_series(arr), dtype=NumpyEADtype(object))
        assert obj._mgr.blocks[0].values.dtype == object
        assert isinstance(obj._mgr.blocks[0].values.ravel()[0], scalar_type)
        if frame_or_series is DataFrame:
            sers: List[Series] = [Series(x) for x in arr]
            obj = frame_or_series(sers, dtype=object)
            assert obj._mgr.blocks[0].values.dtype == object
            assert isinstance(obj._mgr.blocks[0].values.ravel()[0], scalar_type)

    def test_series_with_name_not_matching_column(self) -> None:
        x: Series = Series(range(5), name=1)
        y: Series = Series(range(5), name=0)
        result: DataFrame = DataFrame(x, columns=[0])
        expected: DataFrame = DataFrame([], columns=[0])
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
    def test_empty_constructor(self, constructor: Callable[[], Union[DataFrame, Series]]) -> None:
        expected: DataFrame = DataFrame()
        result: Union[DataFrame, Series] = constructor()
        assert len(result.index) == 0
        assert len(result.columns) == 0
        tm.assert_frame_equal(result, expected)

    def test_empty_constructor_object_index(self) -> None:
        expected: DataFrame = DataFrame(index=RangeIndex(0), columns=RangeIndex(0))
        result: DataFrame = DataFrame({})
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
        emptylike: Iterable[Iterable[Any]],
        expected_index: RangeIndex,
        expected_columns: RangeIndex,
    ) -> None:
        expected: DataFrame = DataFrame(index=expected_index, columns=expected_columns)
        result: DataFrame = DataFrame(emptylike)
        tm.assert_frame_equal(result, expected)

    def test_constructor_mixed(
        self, float_string_frame: DataFrame, using_infer_string: bool
    ) -> None:
        dtype: Union[str, Any] = 'str' if using_infer_string else np.object_
        assert float_string_frame['foo'].dtype == dtype

    def test_constructor_cast_failure(self) -> None:
        msg: str = 'could not convert string to float'
        with pytest.raises(ValueError, match=msg):
            DataFrame({'a': ['a', 'b', 'c']}, dtype=np.float64)
        df: DataFrame = DataFrame(np.ones((4, 2)))
        df['foo'] = np.ones((4, 2)).tolist()
        msg = 'Expected a 1D array, got an array with shape \\(4, 2\\)'
        with pytest.raises(ValueError, match=msg):
            df['test'] = np.ones((4, 2))
        df['foo2'] = np.ones((4, 2)).tolist()

    def test_constructor_dtype_copy(self) -> None:
        orig_df: DataFrame = DataFrame({'col1': [1.0], 'col2': [2.0], 'col3': [3.0]})
        new_df: DataFrame = DataFrame(orig_df, dtype=float, copy=True)
        new_df['col1'] = 200.0
        assert orig_df['col1'][0] == 1.0

    def test_constructor_dtype_nocast_view_dataframe(self) -> None:
        df: DataFrame = DataFrame([[1, 2]])
        should_be_view: DataFrame = DataFrame(df, dtype=df[0].dtype)
        should_be_view.iloc[0, 0] = 99
        assert df.values[0, 0] == 1

    def test_constructor_dtype_nocast_view_2d_array(self) -> None:
        df: DataFrame = DataFrame([[1, 2], [3, 4]], dtype='int64')
        df2: DataFrame = DataFrame(df.values, dtype=df[0].dtype)
        assert df2._mgr.blocks[0].values.flags.c_contiguous

    def test_1d_object_array_does_not_copy(
        self, using_infer_string: bool
    ) -> None:
        arr: np.ndarray = np.array(['a', 'b'], dtype='object')
        df: DataFrame = DataFrame(arr, copy=False)
        if using_infer_string:
            if df[0].dtype.storage == 'pyarrow':
                pass
            else:
                assert np.shares_memory(df[0].to_numpy(), arr)
        else:
            assert np.shares_memory(df.values, arr)
        df = DataFrame(arr, dtype=object, copy=False)
        assert np.shares_memory(df.values, arr)

    def test_2d_object_array_does_not_copy(
        self, using_infer_string: bool
    ) -> None:
        arr: np.ndarray = np.array([['a', 'b'], ['c', 'd']], dtype='object')
        df: DataFrame = DataFrame(arr, copy=False)
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
        df: DataFrame = DataFrame([[1, '2'], [None, 'a']], dtype=object)
        assert df.loc[1, 0] is None
        assert df.loc[0, 1] == '2'

    def test_constructor_list_of_2d_raises(self) -> None:
        a: DataFrame = DataFrame()
        b: np.ndarray = np.empty((0, 0))
        with pytest.raises(ValueError, match='shape=\\(1, 0, 0\\)'):
            DataFrame([a])
        with pytest.raises(ValueError, match='shape=\\(1, 0, 0\\)'):
            DataFrame([b])
        a = DataFrame({'A': [1, 2]})
        with pytest.raises(ValueError, match='shape=\\(2, 2, 1\\)'):
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
            arrays: List[np.ndarray] = [
                np.array(np.random.default_rng(2).integers(10, size=10), dtype=d)
                for d in dtypes
            ]
        elif typ == 'float':
            dtypes = MIXED_FLOAT_DTYPES
            arrays = [
                np.array(np.random.default_rng(2).integers(10, size=10), dtype=d)
                for d in dtypes
            ]
        for d, a in zip(dtypes, arrays):
            assert a.dtype == d
        ad.update(dict(zip(dtypes, arrays)))
        df: DataFrame = DataFrame(ad)
        dtypes_combined = MIXED_FLOAT_DTYPES + MIXED_INT_DTYPES
        for d in dtypes_combined:
            if d in df:
                assert df.dtypes[d] == d

    def test_constructor_complex_dtypes(self) -> None:
        a: np.ndarray = np.random.default_rng(2).random(10).astype(np.complex64)
        b: np.ndarray = np.random.default_rng(2).random(10).astype(np.complex128)
        df: DataFrame = DataFrame({'a': a, 'b': b})
        assert a.dtype == df.a.dtype
        assert b.dtype == df.b.dtype

    def test_constructor_dtype_str_na_values(self, string_dtype: Any) -> None:
        df: DataFrame = DataFrame({'A': ['x', None]}, dtype=string_dtype)
        result: DataFrame = df.isna()
        expected: DataFrame = DataFrame({'A': [False, True]})
        tm.assert_frame_equal(result, expected)
        assert df.iloc[1, 0] is None
        df = DataFrame({'A': ['x', np.nan]}, dtype=string_dtype)
        assert np.isnan(df.iloc[1, 0])

    def test_constructor_rec(self, float_frame: DataFrame) -> None:
        rec: np.recarray = float_frame.to_records(index=False)
        rec.dtype.names = list(rec.dtype.names)[::-1]
        index: RangeIndex = float_frame.index
        df: DataFrame = DataFrame(rec)
        tm.assert_index_equal(df.columns, Index(rec.dtype.names))
        df2: DataFrame = DataFrame(rec, index=index)
        tm.assert_index_equal(df2.columns, Index(rec.dtype.names))
        tm.assert_index_equal(df2.index, index)
        rng: np.ndarray = np.arange(len(rec))[::-1]
        df3: DataFrame = DataFrame(rec, index=rng, columns=['C', 'B'])
        expected: DataFrame = DataFrame(rec, index=rng).reindex(columns=['C', 'B'])
        tm.assert_frame_equal(df3, expected)

    def test_constructor_bool(self) -> None:
        df: DataFrame = DataFrame(
            {0: np.ones(10, dtype=bool), 1: np.zeros(10, dtype=bool)}
        )
        assert df.values.dtype == np.bool_

    def test_constructor_overflow_int64(self) -> None:
        values: np.ndarray = np.array(
            [2 ** 64 - i for i in range(1, 10)], dtype=np.uint64
        )
        result: DataFrame = DataFrame({'a': values})
        assert result['a'].dtype == np.uint64
        data_scores: List[Tuple[int, int]] = [
            (6311132704823138710, 273),
            (2685045978526272070, 23),
            (8921811264899370420, 45),
            (17019687244989530680, 270),
            (9930107427299601010, 273),
        ]
        dtype = [('uid', 'u8'), ('score', 'u8')]
        data: np.ndarray = np.zeros((len(data_scores),), dtype=dtype)
        data[:] = data_scores
        df_crawls: DataFrame = DataFrame(data)
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
        value: Any = values[0]
        result: DataFrame = DataFrame(values)
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
    def test_constructor_numpy_uints(self, values: List[Any] | np.ndarray) -> None:
        value: Any = values[0]
        result: DataFrame = DataFrame(values)
        assert result[0].dtype == value.dtype
        assert result[0][0] == value

    def test_constructor_ordereddict(self) -> None:
        nitems: int = 100
        nums: List[int] = list(range(nitems))
        np.random.default_rng(2).shuffle(nums)
        expected: List[str] = [f'A{i:d}' for i in nums]
        df: DataFrame = DataFrame(OrderedDict(zip(expected, [[0]] * nitems)))
        assert expected == list(df.columns)

    def test_constructor_dict(self) -> None:
        datetime_series: Series = Series(
            np.arange(30, dtype=np.float64),
            index=date_range('2020-01-01', periods=30),
        )
        datetime_series_short: Series = datetime_series[5:]
        frame: DataFrame = DataFrame({'col1': datetime_series, 'col2': datetime_series_short})
        assert len(datetime_series) == 30
        assert len(datetime_series_short) == 25
        tm.assert_series_equal(
            frame['col1'], datetime_series.rename('col1')
        )
        exp: Series = Series(
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
        msg: str = 'Mixing dicts with non-Series may lead to ambiguous ordering.'
        with pytest.raises(ValueError, match=msg):
            DataFrame({'A': {'a': 'a', 'b': 'b'}, 'B': ['a', 'b', 'c']})

    def test_constructor_dict_length1(self) -> None:
        frame: DataFrame = DataFrame({'A': {'1': 1, '2': 2}})
        tm.assert_index_equal(frame.index, Index(['1', '2']))

    def test_constructor_dict_with_index(self) -> None:
        idx: Index = Index([0, 1, 2])
        frame: DataFrame = DataFrame({}, index=idx)
        assert frame.index is idx

    def test_constructor_dict_with_index_and_columns(self) -> None:
        idx: Index = Index([0, 1, 2])
        frame: DataFrame = DataFrame({}, index=idx, columns=idx)
        assert frame.index is idx
        assert frame.columns is idx
        assert len(frame._series) == 3

    def test_constructor_dict_of_empty_lists(self) -> None:
        frame: DataFrame = DataFrame({'A': [], 'B': []}, columns=['A', 'B'])
        tm.assert_index_equal(frame.index, RangeIndex(0), exact=True)

    def test_constructor_dict_with_none(self) -> None:
        frame_none: DataFrame = DataFrame({'a': None}, index=[0])
        frame_none_list: DataFrame = DataFrame({'a': [None]}, index=[0])
        assert frame_none._get_value(0, 'a') is None
        assert frame_none_list._get_value(0, 'a') is None
        tm.assert_frame_equal(frame_none, frame_none_list)

    def test_constructor_dict_errors(self) -> None:
        msg: str = 'If using all scalar values, you must pass an index'
        with pytest.raises(ValueError, match=msg):
            DataFrame({'a': 0.7})
        with pytest.raises(ValueError, match=msg):
            DataFrame({'a': 0.7}, columns=['a'])

    @pytest.mark.parametrize('scalar', [2, np.nan, None, 'D'])
    def test_constructor_invalid_items_unused(self, scalar: Any) -> None:
        result: DataFrame = DataFrame({'a': scalar}, columns=['b'])
        expected: DataFrame = DataFrame(columns=['b'])
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('value', [4, np.nan, None, float('nan')])
    def test_constructor_dict_nan_key(self, value: Any) -> None:
        cols: List[Any] = [1, value, 3]
        idx: List[Any] = ['a', value]
        values: List[List[int]] = [[0, 3], [1, 4], [2, 5]]
        data: Dict[Any, Series] = {
            cols[c]: Series(values[c], index=idx) for c in range(3)
        }
        result: DataFrame = DataFrame(data).sort_values(1).sort_values('a', axis=1)
        expected: DataFrame = DataFrame(np.arange(6, dtype='int64').reshape(2, 3), index=idx, columns=cols)
        tm.assert_frame_equal(result, expected)
        result = DataFrame(data, index=idx).sort_values('a', axis=1)
        tm.assert_frame_equal(result, expected)
        result = DataFrame(data, index=idx, columns=cols)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('value', [np.nan, None, float('nan')])
    def test_constructor_dict_nan_tuple_key(self, value: Any) -> None:
        cols: Index = Index([(11, 21), (value, 22), (13, value)])
        idx: Index = Index([('a', value), (value, 2)])
        values: List[List[int]] = [[0, 3], [1, 4], [2, 5]]
        data: Dict[Any, Series] = {
            cols[c]: Series(values[c], index=idx) for c in range(3)
        }
        result: DataFrame = DataFrame(data).sort_values((11, 21)).sort_values(('a', value), axis=1)
        expected: DataFrame = DataFrame(np.arange(6, dtype='int64').reshape(2, 3), index=idx, columns=cols)
        tm.assert_frame_equal(result, expected)
        result = DataFrame(data, index=idx).sort_values(('a', value), axis=1)
        tm.assert_frame_equal(result, expected)
        result = DataFrame(data, index=idx, columns=cols)
        tm.assert_frame_equal(result, expected)

    def test_constructor_dict_order_insertion(self) -> None:
        datetime_series: Series = Series(
            np.arange(10, dtype=np.float64),
            index=date_range('2020-01-01', periods=10),
        )
        datetime_series_short: Series = datetime_series[:5]
        d: Dict[str, Dict[Any, Any]] = {'b': datetime_series_short, 'a': datetime_series}
        frame: DataFrame = DataFrame(data=d)
        expected: DataFrame = DataFrame(data=d, columns=list('ba'))
        tm.assert_frame_equal(frame, expected)

    def test_constructor_dict_nan_key_and_columns(self) -> None:
        result: DataFrame = DataFrame({np.nan: [1, 2], 2: [2, 3]}, columns=[np.nan, 2])
        expected: DataFrame = DataFrame([[1, 2], [2, 3]], columns=[np.nan, 2])
        tm.assert_frame_equal(result, expected)

    def test_constructor_multi_index(self) -> None:
        tuples: List[Tuple[int, int]] = [(2, 3), (3, 3), (3, 3)]
        mi: MultiIndex = MultiIndex.from_tuples(tuples)
        df: DataFrame = DataFrame(index=mi, columns=mi)
        assert isna(df).values.ravel().all()
        tuples = [(3, 3), (2, 3), (3, 3)]
        mi = MultiIndex.from_tuples(tuples)
        df = DataFrame(index=mi, columns=mi)
        assert isna(df).values.ravel().all()

    def test_constructor_2d_index(self) -> None:
        df: DataFrame = DataFrame([[1]], columns=[[1]], index=[1, 2])
        expected: DataFrame = DataFrame(
            [1, 1],
            index=Index([1, 2], dtype='int64'),
            columns=MultiIndex(levels=[[1]], codes=[[0]]),
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
        msg: str = 'Empty data passed with indices specified.'
        with pytest.raises(ValueError, match=msg):
            DataFrame(np.empty(0), index=[1])
        msg = 'Mixing dicts with non-Series may lead to ambiguous ordering.'
        with pytest.raises(ValueError, match=msg):
            DataFrame({'A': {'a': 'a', 'b': 'b'}, 'B': ['a', 'b', 'c']})
        msg = 'Shape of passed values is \\(4, 3\\), indices imply \\(3, 3\\)'
        with pytest.raises(ValueError, match=msg):
            DataFrame(
                np.arange(12).reshape((4, 3)), columns=['foo', 'bar', 'baz'], index=date_range('2000-01-01', periods=3)
            )
        arr: np.ndarray = np.array([[4, 5, 6]])
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
            DataFrame(
                np.random.default_rng(2).random((2, 3)), columns=['A', 'B', 'C'], index=[1]
            )
        msg = 'Shape of passed values is \\(2, 3\\), indices imply \\(2, 2\\)'
        with pytest.raises(ValueError, match=msg):
            DataFrame(
                np.random.default_rng(2).random((2, 3)), columns=['A', 'B'], index=[1, 2]
            )
        msg = '2 columns passed, passed data had 10 columns'
        with pytest.raises(ValueError, match=msg):
            DataFrame((range(10), range(10, 20)), columns=('ones', 'twos'))
        msg = 'If using all scalar values, you must pass an index'
        with pytest.raises(ValueError, match=msg):
            DataFrame({'a': False, 'b': True})

    def test_constructor_subclass_dict(
        self, dict_subclass: Callable[..., Dict[Any, Any]]
    ) -> None:
        data: Dict[str, Dict[Any, float]] = {
            'col1': dict_subclass(((x, 10.0 * x) for x in range(10))),
            'col2': dict_subclass(((x, 20.0 * x) for x in range(10))),
        }
        df: DataFrame = DataFrame(data)
        refdf: DataFrame = DataFrame(
            {col: dict(val.items()) for col, val in data.items()}
        )
        tm.assert_frame_equal(refdf, df)
        data = dict_subclass(data.items())
        df = DataFrame(data)
        tm.assert_frame_equal(refdf, df)

    def test_constructor_defaultdict(self, float_frame: DataFrame) -> None:
        data: Dict[str, defaultdict] = {}
        float_frame.loc[: float_frame.index[10], 'B'] = np.nan
        for k, v in float_frame.items():
            dct: defaultdict = defaultdict(dict)
            dct.update(v.to_dict())
            data[k] = dct
        frame: DataFrame = DataFrame(data)
        expected: DataFrame = frame.reindex(index=float_frame.index)
        tm.assert_frame_equal(float_frame, expected)

    def test_constructor_dict_block(self) -> None:
        expected: np.ndarray = np.array([[4.0, 3.0, 2.0, 1.0]])
        df: DataFrame = DataFrame(
            {'d': [4.0], 'c': [3.0], 'b': [2.0], 'a': [1.0]}, columns=['d', 'c', 'b', 'a']
        )
        tm.assert_numpy_array_equal(df.values, expected)

    def test_constructor_dict_cast(self, using_infer_string: bool) -> None:
        test_data: Dict[str, Dict[str, Any]] = {
            'A': {'1': 1, '2': 2},
            'B': {'1': '1', '2': '2', '3': '3'},
        }
        frame: DataFrame = DataFrame(test_data, dtype=float)
        assert len(frame) == 3
        assert frame['B'].dtype == np.float64
        assert frame['A'].dtype == np.float64
        frame = DataFrame(test_data)
        assert len(frame) == 3
        assert frame['B'].dtype == (np.object_ if not using_infer_string else 'str')
        assert frame['A'].dtype == np.float64

    def test_constructor_dict_cast2(self) -> None:
        test_data: Dict[str, Dict[int, Union[str, float]]] = {
            'A': dict(zip(range(20), [f'word_{i}' for i in range(20)])),
            'B': dict(zip(range(15), np.random.default_rng(2).standard_normal(15))),
        }
        with pytest.raises(ValueError, match='could not convert string'):
            DataFrame(test_data, dtype=float)

    def test_constructor_dict_dont_upcast(self) -> None:
        d: Dict[str, Any] = {'Col1': {'Row1': 'A String', 'Row2': np.nan}}
        df: DataFrame = DataFrame(d)
        assert isinstance(df['Col1']['Row2'], float)

    def test_constructor_dict_dont_upcast2(self) -> None:
        dm: DataFrame = DataFrame([[1, 2], ['a', 'b']], index=[1, 2], columns=[1, 2])
        assert isinstance(dm[1][1], int)

    def test_constructor_dict_of_tuples(self) -> None:
        data: Dict[str, Tuple[int, int, int]] = {'a': (1, 2, 3), 'b': (4, 5, 6)}
        result: DataFrame = DataFrame(data)
        expected: DataFrame = DataFrame({k: list(v) for k, v in data.items()})
        tm.assert_frame_equal(result, expected, check_dtype=False)

    def test_constructor_dict_of_ranges(self) -> None:
        data: Dict[str, range] = {'a': range(3), 'b': range(3, 6)}
        result: DataFrame = DataFrame(data)
        expected: DataFrame = DataFrame({'a': [0, 1, 2], 'b': [3, 4, 5]})
        tm.assert_frame_equal(result, expected)

    def test_constructor_dict_of_iterators(self) -> None:
        data: Dict[str, Iterator[int]] = {'a': iter(range(3)), 'b': reversed(range(3))}
        result: DataFrame = DataFrame(data)
        expected: DataFrame = DataFrame({'a': [0, 1, 2], 'b': [2, 1, 0]})
        tm.assert_frame_equal(result, expected)

    def test_constructor_dict_of_generators(self) -> None:
        data: Dict[str, Generator[int, None, None]] = {
            'a': (i for i in range(3)),
            'b': (i for i in reversed(range(3))),
        }
        result: DataFrame = DataFrame(data)
        expected: DataFrame = DataFrame({'a': [0, 1, 2], 'b': [2, 1, 0]})
        tm.assert_frame_equal(result, expected)

    def test_constructor_dict_multiindex(self) -> None:
        d: Dict[Tuple[str, str], Dict[Tuple[str, str], int]] = {
            ('a', 'a'): {('i', 'i'): 0, ('i', 'j'): 1, ('j', 'i'): 2},
            ('b', 'a'): {('i', 'i'): 6, ('i', 'j'): 5, ('j', 'i'): 4},
            ('b', 'c'): {('i', 'i'): 7, ('i', 'j'): 8, ('j', 'i'): 9},
        }
        _d: List[Tuple[Tuple[str, str], Dict[Tuple[str, str], int]]] = sorted(d.items())
        df: DataFrame = DataFrame(d)
        expected: DataFrame = DataFrame([x[1] for x in _d], index=MultiIndex.from_tuples([x[0] for x in _d])).T
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
        dates_as_str: List[str] = ['1984-02-19', '1988-11-06', '1989-12-03', '1990-03-15']

        def create_data(constructor: Callable[[str], datetime]) -> Dict[int, datetime]:
            return {i: constructor(s) for i, s in enumerate(dates_as_str)}

        data_datetime64: Dict[int, datetime] = create_data(np.datetime64)
        data_datetime: Dict[int, datetime] = create_data(lambda x: datetime.strptime(x, '%Y-%m-%d'))
        data_Timestamp: Dict[int, Timestamp] = create_data(Timestamp)
        expected: DataFrame = DataFrame(
            [[0, np.nan, np.nan, np.nan], [1, 2, np.nan, np.nan], [np.nan, 3, 4, np.nan], [np.nan, np.nan, 5, 6]],
            index=[Timestamp(dt) for dt in dates_as_str],
        )
        result_datetime64: DataFrame = DataFrame(data_datetime64)
        result_datetime: DataFrame = DataFrame(data_datetime)
        assert result_datetime.index.unit == 'us'
        result_datetime.index = result_datetime.index.as_unit('s')
        result_Timestamp: DataFrame = DataFrame(data_Timestamp)
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
        td_as_int: List[int] = [1, 2, 3, 4]
        data: Dict[int, Any] = {i: klass(s) for i, s in enumerate(td_as_int)}
        expected: DataFrame = DataFrame(
            [
                {0: 0, 1: None, 2: None, 3: None},
                {0: None, 1: 2, 2: None, 3: None},
                {0: None, 1: None, 2: 4, 3: None},
                {0: None, 1: None, 2: None, 3: 6},
            ],
            index=[Timedelta(td, 'D') for td in td_as_int],
        )
        result: DataFrame = DataFrame(data)
        tm.assert_frame_equal(result, expected)

    def test_constructor_period_dict(self) -> None:
        a: pd.PeriodIndex = pd.PeriodIndex(['2012-01', 'NaT', '2012-04'], freq='M')
        b: pd.PeriodIndex = pd.PeriodIndex(['2012-02-01', '2012-03-01', 'NaT'], freq='D')
        df: DataFrame = DataFrame({'a': a, 'b': b})
        assert df['a'].dtype == a.dtype
        assert df['b'].dtype == b.dtype
        df = DataFrame({'a': a.astype(object).tolist(), 'b': b.astype(object).tolist()})
        assert df['a'].dtype == a.dtype
        assert df['b'].dtype == b.dtype

    def test_constructor_dict_extension_scalar(
        self, ea_scalar_and_dtype: Tuple[Any, Any]
    ) -> None:
        ea_scalar, ea_dtype = ea_scalar_and_dtype
        df: DataFrame = DataFrame({'a': ea_scalar}, index=[0])
        assert df['a'].dtype == ea_dtype
        expected: DataFrame = DataFrame(index=[0], columns=['a'], data=ea_scalar)
        tm.assert_frame_equal(df, expected)

    @pytest.mark.parametrize(
        'data,dtype',
        [
            (Period('2020-01'), PeriodDtype('M')),
            (Interval(left=0, right=5), IntervalDtype('int64', 'right')),
            (
                Timestamp('2011-01-01', tz='US/Eastern'),
                DatetimeTZDtype(unit='s', tz='US/Eastern'),
            ),
        ],
    )
    def test_constructor_extension_scalar_data(
        self, data: Any, dtype: Any
    ) -> None:
        df: DataFrame = DataFrame(index=range(2), columns=['a', 'b'], data=data)
        assert df['a'].dtype == dtype
        assert df['b'].dtype == dtype
        arr: Any = pd.array([data] * 2, dtype=dtype)
        expected: DataFrame = DataFrame({'a': arr, 'b': arr})
        tm.assert_frame_equal(df, expected)

    def test_nested_dict_frame_constructor(self) -> None:
        rng: pd.period_range = pd.period_range('1/1/2000', periods=5)
        df: DataFrame = DataFrame(
            np.random.default_rng(2).standard_normal((10, 5)), columns=rng
        )
        data: Dict[Any, Dict[Any, float]] = {}
        for col in df.columns:
            for row in df.index:
                data.setdefault(col, {})[row] = df._get_value(row, col)
        result: DataFrame = DataFrame(data, columns=rng)
        tm.assert_frame_equal(result, df)
        data = {}
        for col in df.columns:
            for row in df.index:
                data.setdefault(row, {})[col] = df._get_value(row, col)
        result = DataFrame(data, index=rng).T
        tm.assert_frame_equal(result, df)

    def _check_basic_constructor(
        self, empty: Callable[[Tuple[int, ...], ...], Union[np.ndarray, Any]]
    ) -> None:
        mat: np.ndarray = empty((2, 3), dtype=float)
        frame: DataFrame = DataFrame(mat, columns=['A', 'B', 'C'], index=[1, 2])
        assert len(frame.index) == 2
        assert len(frame.columns) == 3
        frame = DataFrame(empty((3,), dtype=float), columns=['A'], index=[1, 2, 3])
        assert len(frame.index) == 3
        assert len(frame.columns) == 1
        if empty is not np.ones:
            msg: str = 'Cannot convert non-finite values \\(NA or inf\\) to integer'
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
            DataFrame(empty((3, 3, 3)), columns=['A', 'B', 'C'], index=[1])
        frame = DataFrame(mat)
        tm.assert_index_equal(frame.index, Index(range(2)), exact=True)
        tm.assert_index_equal(frame.columns, Index(range(3)), exact=True)
        frame = DataFrame(mat, index=[1, 2])
        tm.assert_index_equal(frame.columns, Index(range(3)), exact=True)
        frame = DataFrame(mat, columns=['A', 'B', 'C'])
        tm.assert_index_equal(frame.index, Index(range(2)), exact=True)
        frame = DataFrame(empty((0, 3), dtype=float))
        assert len(frame.index) == 0
        frame = DataFrame(empty((3, 0), dtype=float))
        assert len(frame.columns) == 0

    def test_constructor_ndarray(self) -> None:
        self._check_basic_constructor(np.ones)
        frame: DataFrame = DataFrame(['foo', 'bar'], index=[0, 1], columns=['A'])
        assert len(frame) == 2

    def test_constructor_maskedarray(self) -> None:
        self._check_basic_constructor(ma.masked_all)
        mat: np.ma.MaskedArray = ma.masked_all((2, 3), dtype=float)
        mat[0, 0] = 1.0
        mat[1, 2] = 2.0
        frame: DataFrame = DataFrame(mat, columns=['A', 'B', 'C'], index=[1, 2])
        assert 1.0 == frame['A'][1]
        assert 2.0 == frame['C'][2]
        mat = ma.masked_all((2, 3), dtype=float)
        frame = DataFrame(mat, columns=['A', 'B', 'C'], index=[1, 2])
        assert np.all(~np.asarray(frame == frame))

    @pytest.mark.filterwarnings('ignore:elementwise comparison failed:DeprecationWarning')
    def test_constructor_maskedarray_nonfloat(self) -> None:
        mat: np.ma.MaskedArray = ma.masked_all((2, 3), dtype=int)
        frame: DataFrame = DataFrame(mat, columns=['A', 'B', 'C'], index=[1, 2])
        assert len(frame.index) == 2
        assert len(frame.columns) == 3
        assert np.all(~np.asarray(frame == frame))
        frame = DataFrame(mat, columns=['A', 'B', 'C'], index=[1, 2], dtype=np.float64)
        assert frame.values.dtype == np.float64
        mat2: np.ma.MaskedArray = ma.copy(mat)
        mat2[0, 0] = 1
        mat2[1, 2] = 2
        frame = DataFrame(mat2, columns=['A', 'B', 'C'], index=[1, 2])
        assert 1 == frame['A'].astype('i8')[1]
        assert 2 == frame['C'].astype('i8')[2]
        mat = ma.masked_all((2, 3), dtype='M8[ns]')
        frame = DataFrame(mat, columns=['A', 'B', 'C'], index=[1, 2])
        assert len(frame.index) == 2
        assert len(frame.columns) == 3
        assert isna(frame).values.all()
        msg: str = 'datetime64\\[ns\\] values and dtype=int64 is not supported'
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
        mat_hard: np.ma.MaskedArray = ma.masked_all((2, 2), dtype=float).harden_mask()
        result: DataFrame = DataFrame(mat_hard, columns=['A', 'B'], index=[1, 2])
        expected: DataFrame = DataFrame(
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
        data: np.ma.MaskedArray = np.ma.array(
            np.ma.zeros(5, dtype=[('date', '<f8'), ('price', '<f8')]),
            mask=[False] * 5,
        )
        data = data.view(mrecords.mrecarray)
        with pytest.raises(TypeError, match='Pass \\{name: data\\[name\\]'):
            DataFrame(data, dtype=int)

    def test_constructor_corner_shape(self) -> None:
        df: DataFrame = DataFrame(index=[])
        assert df.values.shape == (0, 0)

    @pytest.mark.parametrize(
        'data, index, columns, dtype, expected',
        [
            (None, list(range(10)), ['a', 'b'], object, np.object_),
            (None, None, ['a', 'b'], 'int64', np.dtype('int64')),
            (None, list(range(10)), ['a', 'b'], int, np.dtype('float64')),
            ({}, None, ['foo', 'bar'], None, np.object_),
            (
                {'b': 1},
                list(range(10)),
                list('abc'),
                int,
                np.dtype('float64'),
            ),
        ],
    )
    def test_constructor_dtype(
        self,
        data: Optional[Union[List[Any], Dict[str, Any]]],
        index: Optional[Iterable[Any]],
        columns: List[str],
        dtype: Optional[Union[str, type]],
        expected: np.dtype,
    ) -> None:
        df: DataFrame = DataFrame(data, index, columns, dtype)
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
        expected_dtype: Callable[[], Any],
    ) -> None:
        df: DataFrame = DataFrame({'a': data}, dtype=input_dtype)
        assert df['a'].dtype == expected_dtype()

    def test_constructor_scalar_inference(self, using_infer_string: bool) -> None:
        data: Dict[str, Any] = {
            'int': 1,
            'bool': True,
            'float': 3.0,
            'complex': 4j,
            'object': 'foo',
        }
        df: DataFrame = DataFrame(data, index=np.arange(10))
        assert df['int'].dtype == np.int64
        assert df['bool'].dtype == np.bool_
        assert df['float'].dtype == np.float64
        assert df['complex'].dtype == np.complex128
        assert df['object'].dtype == (np.object_ if not using_infer_string else 'str')

    def test_constructor_arrays_and_scalars(self) -> None:
        df: DataFrame = DataFrame(
            {'a': np.random.default_rng(2).standard_normal(10), 'b': True}
        )
        exp: DataFrame = DataFrame({'a': df['a'].values, 'b': [True] * 10})
        tm.assert_frame_equal(df, exp)
        with pytest.raises(ValueError, match='must pass an index'):
            DataFrame({'a': False, 'b': True})

    def test_constructor_DataFrame(self, float_frame: DataFrame) -> None:
        df: DataFrame = DataFrame(float_frame)
        tm.assert_frame_equal(df, float_frame)
        df_casted: DataFrame = DataFrame(float_frame, dtype=np.int64)
        assert df_casted.values.dtype == np.int64

    def test_constructor_empty_dataframe(self) -> None:
        actual: DataFrame = DataFrame(DataFrame(), dtype='object')
        expected: DataFrame = DataFrame([], dtype='object')
        tm.assert_frame_equal(actual, expected)

    def test_constructor_more(self, float_frame: DataFrame) -> None:
        arr: np.ndarray = np.random.default_rng(2).standard_normal(10)
        dm: DataFrame = DataFrame(arr, columns=['A'], index=np.arange(10))
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
        mat: np.ndarray = np.array(['foo', 'bar'], dtype=object).reshape(2, 1)
        msg: str = "could not convert string to float: 'foo'"
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
        df: DataFrame = DataFrame([], index=[])
        expected: DataFrame = DataFrame(index=[])
        tm.assert_frame_equal(df, expected)
        df = DataFrame([], columns=['A', 'B'])
        expected = DataFrame({}, columns=['A', 'B'])
        tm.assert_frame_equal(df, expected)

        def empty_gen() -> Generator[Any, None, None]:
            yield from ()

        df = DataFrame(empty_gen(), columns=['A', 'B'])
        tm.assert_frame_equal(df, expected)

    def test_constructor_list_of_lists(
        self, using_infer_string: bool
    ) -> None:
        df: DataFrame = DataFrame(data=[[1, 'a'], [2, 'b']], columns=['num', 'str'])
        assert is_integer_dtype(df['num'])
        assert df['str'].dtype == (
            np.object_ if not using_infer_string else 'str'
        )
        expected: DataFrame = DataFrame(np.arange(10))
        data: List[np.ndarray] = [np.array(x) for x in range(10)]
        result: DataFrame = DataFrame(data)
        tm.assert_frame_equal(result, expected)

    def test_nested_pandasarray_matches_nested_ndarray(self) -> None:
        ser: Series = Series([1, 2])
        arr: np.ndarray = np.array([None, None], dtype=object)
        arr[0] = ser
        arr[1] = ser * 2
        df: DataFrame = DataFrame(arr)
        expected: DataFrame = DataFrame(pd.array(arr))
        tm.assert_frame_equal(df, expected)
        assert df.shape == (2, 1)
        tm.assert_numpy_array_equal(df[0].values, arr)

    def test_constructor_list_like_data_nested_list_column(self) -> None:
        arrays: List[List[str]] = [list('abcd'), list('cdef')]
        result: DataFrame = DataFrame([[1, 2, 3, 4], [4, 5, 6, 7]], columns=arrays)
        mi: MultiIndex = MultiIndex.from_arrays(arrays)
        expected: DataFrame = DataFrame(
            [[1, 2, 3, 4], [4, 5, 6, 7]], columns=mi
        )
        tm.assert_frame_equal(result, expected)

    def test_constructor_wrong_length_nested_list_column(self) -> None:
        arrays: List[List[Any]] = [list('abc'), list('cde')]
        msg: str = '3 columns passed, passed data had 4'
        with pytest.raises(ValueError, match=msg):
            DataFrame([[1, 2, 3, 4], [4, 5, 6, 7]], columns=arrays)

    def test_constructor_unequal_length_nested_list_column(self) -> None:
        arrays: List[List[Any]] = [list('abcd'), list('cde')]
        msg: str = 'all arrays must be same length'
        with pytest.raises(ValueError, match=msg):
            DataFrame([[1, 2, 3, 4], [4, 5, 6, 7]], columns=arrays)

    @pytest.mark.parametrize('data', [[Timestamp('2021-01-01')], [{'x': Timestamp('2021-01-01')}], {'x': [Timestamp('2021-01-01')]}, {'x': Timestamp('2021-01-01')}])
    def test_constructor_one_element_data_list(self, data: Any) -> None:
        result: DataFrame = DataFrame(data, index=range(3), columns=['x'])
        expected: DataFrame = DataFrame({'x': [Timestamp('2021-01-01')] * 3})
        tm.assert_frame_equal(result, expected)

    def test_constructor_sequence_like(self) -> None:

        class DummyContainer(Iterable[Any]):
            def __init__(self, lst: List[Any]) -> None:
                self._lst = lst

            def __getitem__(self, n: Union[int, slice]) -> Any:
                return self._lst.__getitem__(n)

            def __len__(self) -> int:
                return self._lst.__len__()

        lst_containers: List[DummyContainer] = [
            DummyContainer([1, 'a']),
            DummyContainer([2, 'b']),
        ]
        columns: List[str] = ['num', 'str']
        result: DataFrame = DataFrame(lst_containers, columns=columns)
        expected: DataFrame = DataFrame([[1, 'a'], [2, 'b']], columns=columns)
        tm.assert_frame_equal(result, expected, check_dtype=False)

    def test_constructor_stdlib_array(self) -> None:
        result: DataFrame = DataFrame({'A': array.array('i', range(10))})
        expected: DataFrame = DataFrame({'A': list(range(10))})
        tm.assert_frame_equal(result, expected, check_dtype=False)
        expected = DataFrame([list(range(10)), list(range(10))])
        result = DataFrame([array.array('i', range(10)), array.array('i', range(10))])
        tm.assert_frame_equal(result, expected, check_dtype=False)

    def test_constructor_range(self) -> None:
        result: DataFrame = DataFrame(range(10))
        expected: DataFrame = DataFrame(list(range(10)))
        tm.assert_frame_equal(result, expected)

    def test_constructor_list_of_ranges(self) -> None:
        result: DataFrame = DataFrame([range(10), range(10)])
        expected: DataFrame = DataFrame([list(range(10)), list(range(10))])
        tm.assert_frame_equal(result, expected)

    def test_constructor_iterable(self) -> None:

        class Iter:
            def __iter__(self) -> Iterator[List[Any]]:
                for i in range(10):
                    yield [1, 2, 3]

        expected: DataFrame = DataFrame([[1, 2, 3]] * 10)
        result: DataFrame = DataFrame(Iter())
        tm.assert_frame_equal(result, expected)

    def test_constructor_iterator(self) -> None:
        result: DataFrame = DataFrame(iter(range(10)))
        expected: DataFrame = DataFrame(list(range(10)))
        tm.assert_frame_equal(result, expected)

    def test_constructor_list_of_iterators(self) -> None:
        result: DataFrame = DataFrame([iter(range(10)), iter(range(10))])
        expected: DataFrame = DataFrame([list(range(10)), list(range(10))])
        tm.assert_frame_equal(result, expected)

    def test_constructor_generator(self) -> None:
        gen1: Generator[int, None, None] = (i for i in range(10))
        gen2: Generator[int, None, None] = (i for i in range(10))
        expected: DataFrame = DataFrame([list(range(10)), list(range(10))])
        result: DataFrame = DataFrame([gen1, gen2])
        tm.assert_frame_equal(result, expected)
        gen: Generator[List[Any], None, None] = ([i, 'a'] for i in range(10))
        result = DataFrame(gen)
        expected = DataFrame({0: range(10), 1: 'a'})
        tm.assert_frame_equal(result, expected, check_dtype=False)

    def test_constructor_list_of_dicts(self) -> None:
        result: DataFrame = DataFrame([{}])
        expected: DataFrame = DataFrame(index=RangeIndex(1), columns=[])
        tm.assert_frame_equal(result, expected)

    def test_constructor_ordered_dict_nested_preserve_order(self) -> None:
        nested1: OrderedDict[str, int] = OrderedDict([('b', 1), ('a', 2)])
        nested2: OrderedDict[str, int] = OrderedDict([('b', 2), ('a', 5)])
        data: OrderedDict[str, OrderedDict[str, int]] = OrderedDict(
            [('col2', nested1), ('col1', nested2)]
        )
        result: DataFrame = DataFrame(data)
        data = {'col2': [1, 2], 'col1': [2, 5]}
        expected: DataFrame = DataFrame(data=data, columns=list('ba'))
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('dict_type', [dict, OrderedDict])
    def test_constructor_ordered_dict_preserve_order(
        self, dict_type: Callable[..., Dict[Any, Any]]
    ) -> None:
        expected: DataFrame = DataFrame([[2, 1]], columns=['b', 'a'])
        data: Dict[str, List[int]] = dict_type()
        data['b'] = [2]
        data['a'] = [1]
        result: DataFrame = DataFrame(data)
        tm.assert_frame_equal(result, expected)
        data = dict_type()
        data['b'] = 2
        data['a'] = 1
        result = DataFrame([data])
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('dict_type', [dict, OrderedDict])
    def test_constructor_ordered_dict_conflicting_orders(
        self, dict_type: Callable[..., Dict[Any, Any]]
    ) -> None:
        row_one: Dict[str, int] = dict_type()
        row_one['b'] = 2
        row_one['a'] = 1
        row_two: Dict[str, int] = dict_type()
        row_two['a'] = 1
        row_two['b'] = 2
        row_three: Dict[str, int] = {'b': 2, 'a': 1}
        expected: DataFrame = DataFrame([[2, 1], [2, 1]], columns=['b', 'a'])
        result: DataFrame = DataFrame([row_one, row_two])
        tm.assert_frame_equal(result, expected)
        expected = DataFrame([[2, 1], [2, 1], [2, 1]], columns=['b', 'a'])
        result = DataFrame([row_one, row_two, row_three])
        tm.assert_frame_equal(result, expected)

    def test_constructor_list_of_series_aligned_index(self) -> None:
        series: List[Series] = [Series(i, index=['b', 'a', 'c'], name=str(i)) for i in range(3)]
        result: DataFrame = DataFrame(series)
        expected: DataFrame = DataFrame(
            {'b': [0, 1, 2], 'a': [0, 1, 2], 'c': [0, 1, 2]},
            columns=['b', 'a', 'c'],
            index=['0', '1', '2'],
        )
        tm.assert_frame_equal(result, expected)

    def test_constructor_list_of_derived_dicts(self) -> None:
        class CustomDict(dict):
            pass

        d: Dict[str, Any] = {'a': 1.5, 'b': 3}
        data_custom: List[CustomDict] = [CustomDict(d)]
        data: List[Dict[str, Any]] = [d]
        result_custom: DataFrame = DataFrame(data_custom)
        result: DataFrame = DataFrame(data)
        tm.assert_frame_equal(result, result_custom)

    def test_constructor_ragged(self) -> None:
        data: Dict[str, List[Any]] = {
            'A': np.random.default_rng(2).standard_normal(10),
            'B': np.random.default_rng(2).standard_normal(8),
        }
        with pytest.raises(ValueError, match='All arrays must be of the same length'):
            DataFrame(data)

    def test_constructor_scalar(self) -> None:
        idx: Index = Index(range(3))
        df: DataFrame = DataFrame({'a': 0}, index=idx)
        expected: DataFrame = DataFrame({'a': [0, 0, 0]}, index=idx)
        tm.assert_frame_equal(df, expected, check_dtype=False)

    def test_constructor_Series_copy_bug(self, float_frame: DataFrame) -> None:
        df: DataFrame = DataFrame(float_frame['A'], index=float_frame.index, columns=['A'])
        df.copy()

    def test_constructor_mixed_dict_nonseries(self) -> None:
        data: Dict[str, Union[Series, List[Any]]] = {}
        data['A'] = {'foo': 1, 'bar': 2, 'baz': 3}
        data['B'] = Series([4, 3, 2, 1], index=['bar', 'qux', 'baz', 'foo'])
        result: DataFrame = DataFrame(data)
        assert result.index.is_monotonic_increasing
        with pytest.raises(ValueError, match='ambiguous ordering'):
            DataFrame({'A': ['a', 'b'], 'B': {'a': 'a', 'b': 'b'}})
        result = DataFrame({'A': ['a', 'b'], 'B': Series(['a', 'b'], index=['a', 'b'])})
        expected: DataFrame = DataFrame(
            {'A': ['a', 'b'], 'B': ['a', 'b']}, index=['a', 'b']
        )
        tm.assert_frame_equal(result, expected)

    def test_constructor_mixed_type_rows(self) -> None:
        data: List[Union[List[Any], Tuple[Any, Any]]] = [[1, 2], (3, 4)]
        result: DataFrame = DataFrame(data)
        expected: DataFrame = DataFrame([[1, 2], [3, 4]])
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
        self, tuples: Tuple[Any, ...], lists: List[List[Any]]
    ) -> None:
        result: DataFrame = DataFrame(tuples)
        expected: DataFrame = DataFrame(lists)
        tm.assert_frame_equal(result, expected)

    def test_constructor_list_of_tuples(self) -> None:
        data: Dict[str, Tuple[int, int]] = {'A': (1, 2), 'B': (3, 4)}
        result: DataFrame = DataFrame({'A': [(1, 2), (3, 4)]})
        expected: DataFrame = DataFrame({'A': Series([(1, 2), (3, 4)])})
        tm.assert_frame_equal(result, expected)

    def test_constructor_list_of_namedtuples(self) -> None:
        named_tuple = namedtuple('Pandas', list('ab'))
        tuples: List[Any] = [named_tuple(1, 3), named_tuple(2, 4)]
        expected: DataFrame = DataFrame({'a': [1, 2], 'b': [3, 4]})
        result: DataFrame = DataFrame(tuples)
        tm.assert_frame_equal(result, expected)
        expected = DataFrame({'y': [1, 2], 'z': [3, 4]})
        result = DataFrame(tuples, columns=['y', 'z'])
        tm.assert_frame_equal(result, expected)

    def test_constructor_list_of_dataclasses(self) -> None:
        Point = make_dataclass('Point', [('x', int), ('y', int)])
        data: List[Any] = [Point(0, 3), Point(1, 3)]
        expected: DataFrame = DataFrame({'x': [0, 1], 'y': [3, 3]})
        result: DataFrame = DataFrame(data)
        tm.assert_frame_equal(result, expected)

    def test_constructor_list_of_dataclasses_with_varying_types(self) -> None:
        Point = make_dataclass('Point', [('x', int), ('y', int)])
        HLine = make_dataclass('HLine', [('x0', int), ('x1', int), ('y', int)])
        data: List[Any] = [Point(0, 3), HLine(1, 3, 3)]
        expected: DataFrame = DataFrame(
            {'x': [0, np.nan], 'y': [3, 3], 'x0': [np.nan, 1], 'x1': [np.nan, 3]}
        )
        result: DataFrame = DataFrame(data)
        tm.assert_frame_equal(result, expected)

    def test_constructor_list_of_dataclasses_error_thrown(self) -> None:
        Point = make_dataclass('Point', [('x', int), ('y', int)])
        msg: str = 'asdict() should be called on dataclass instances'
        with pytest.raises(TypeError, match=re.escape(msg)):
            DataFrame([Point(0, 0), {'x': 1, 'y': 0}])

    def test_constructor_list_of_dict_order(self) -> None:
        data: List[Dict[str, Any]] = [
            {'First': 1, 'Second': 4, 'Third': 7, 'Fourth': 10},
            {'Second': 5, 'First': 2, 'Fourth': 11, 'Third': 8},
            {'Second': 6, 'First': 3, 'Fourth': 12, 'Third': 9, 'YYY': 14, 'XXX': 13},
        ]
        expected: DataFrame = DataFrame(
            {'First': [1, 2, 3], 'Second': [4, 5, 6], 'Third': [7, 8, 9], 'Fourth': [10, 11, 12], 'YYY': [None, None, 14], 'XXX': [None, None, 13]}
        )
        result: DataFrame = DataFrame(data)
        tm.assert_frame_equal(result, expected)

    def test_constructor_Series_named(self) -> None:
        a: Series = Series([1, 2, 3], name=0)
        df: DataFrame = DataFrame(a)
        assert df.columns[0] == 0
        tm.assert_index_equal(df.index, a.index)
        arr: np.ndarray = np.random.default_rng(2).standard_normal(10)
        s: Series = Series(arr, name='x')
        df = DataFrame(s)
        expected: Series = Series(arr, dtype=np.float64, name='x')
        tm.assert_series_equal(df['x'], expected)
        s = Series(arr, index=range(3, 13))
        df = DataFrame(s)
        expected = Series(arr, index=range(3, 13), name='x')
        tm.assert_series_equal(df[0], Series(expected, name=0))
        msg: str = 'Shape of passed values is \\(10, 1\\), indices imply \\(10, 2\\)'
        with pytest.raises(ValueError, match=msg):
            DataFrame(s, columns=[1, 2])
        a = Series([], name='x', dtype=object)
        df = DataFrame(a)
        assert df.columns[0] == 'x'
        s1: Series = Series(arr, name='x')
        df = DataFrame([s1, arr]).T
        expected = DataFrame({'x': s1, 'Unnamed 0': arr}, columns=['x', 'Unnamed 0'])
        tm.assert_series_equal(df['x'], expected['x'])
        df = DataFrame([arr, s1]).T
        expected = DataFrame({'1': s1, '0': arr}, columns=range(2))
        tm.assert_frame_equal(df, expected)

    def test_constructor_Series_named_and_columns(self) -> None:
        s0: Series = Series(range(5), name=0)
        s1: Series = Series(range(5), name=1)
        tm.assert_frame_equal(DataFrame(s0, columns=[0]), s0.to_frame())
        tm.assert_frame_equal(DataFrame(s1, columns=[1]), s1.to_frame())
        assert DataFrame(s0, columns=[1]).empty
        assert DataFrame(s1, columns=[0]).empty

    def test_constructor_Series_differently_indexed(self) -> None:
        s1: Series = Series([1, 2, 3], index=['a', 'b', 'c'], name='x')
        s2: Series = Series([1, 2, 3], index=['a', 'b', 'c'])
        other_index: Index = Index(['a', 'b'])
        df1: DataFrame = DataFrame(s1, index=other_index)
        exp1: DataFrame = DataFrame(s1.reindex(other_index))
        assert df1.columns[0] == 'x'
        tm.assert_frame_equal(df1, exp1)
        df2: DataFrame = DataFrame(s2, index=other_index)
        exp2: DataFrame = DataFrame(s2.reindex(other_index))
        assert df2.columns[0] == 0
        tm.assert_index_equal(df2.index, other_index)
        tm.assert_frame_equal(df2, exp2)

    @pytest.mark.parametrize(
        'name_in1,name_in2,name_in3,name_out',
        [
            ('idx', 'idx', 'idx', 'idx'),
            ('idx', 'idx', None, None),
            ('idx', None, None, None),
            ('idx1', 'idx2', None, None),
            ('idx1', 'idx1', 'idx2', None),
            ('idx1', 'idx2', 'idx3', None),
            (None, None, None, None),
        ],
    )
    def test_constructor_index_names(
        self,
        name_in1: Optional[str],
        name_in2: Optional[str],
        name_in3: Optional[str],
        name_out: Optional[str],
    ) -> None:
        indices: List[Index] = [
            Index(['a', 'a', 'b', 'b'], name=name_in1),
            Index(['x', 'y', 'x', 'y'], name=name_in2),
        ]
        multi: DataFrame = DataFrame(
            np.random.default_rng(2).standard_normal((4, 4)),
            index=[np.array(x) for x in indices],
        )
        assert isinstance(multi.index, MultiIndex)
        assert not isinstance(multi.columns, MultiIndex)
        multi = DataFrame(
            np.random.default_rng(2).standard_normal((4, 4)),
            columns=['A', 'A', 'A', 'A'],
        )
        assert isinstance(multi.columns, MultiIndex)

    @pytest.mark.arm_slow
    @pytest.mark.parametrize('dtype', [None, 'uint8', 'category'])
    def test_constructor_range_dtype(
        self, dtype: Optional[str]
    ) -> None:
        expected: DataFrame = DataFrame({'A': [0, 1, 2, 3, 4]}, dtype=dtype or 'int64')
        result: DataFrame = DataFrame(range(5), columns=['A'], dtype=dtype)
        tm.assert_frame_equal(result, expected)
        result = DataFrame({'A': range(5)}, dtype=dtype)
        tm.assert_frame_equal(result, expected)

    def test_frame_from_list_subclass(self) -> None:
        class ListSubclass(list):
            pass

        expected: DataFrame = DataFrame([[1, 2, 3], [4, 5, 6]])
        result: DataFrame = DataFrame(ListSubclass([ListSubclass([1, 2, 3]), ListSubclass([4, 5, 6])]))
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
        self, extension_arr: Union[Categorical, SparseArray, IntervalArray, PeriodArray]
    ) -> None:
        expected: DataFrame = DataFrame(Series(extension_arr))
        result: DataFrame = DataFrame(extension_arr)
        tm.assert_frame_equal(result, expected)

    def test_datetime_date_tuple_columns_from_dict(self) -> None:
        v: date = date.today()
        tup: Tuple[date, date] = (v, v)
        result: DataFrame = DataFrame({'a': tup}, columns=[tup])
        expected: DataFrame = DataFrame(
            [0, 1], columns=Index(Series([tup]))
        )
        tm.assert_frame_equal(result, expected)

    def test_construct_with_two_categoricalindex_series(self) -> None:
        s1: Series = Series(
            [39, 6, 4], index=CategoricalIndex(['female', 'male', 'unknown'])
        )
        s2: Series = Series(
            [2, 152, 2, 242, 150],
            index=CategoricalIndex(['f', 'female', 'm', 'male', 'unknown']),
        )
        result: DataFrame = DataFrame([s1, s2])
        expected: DataFrame = DataFrame(
            np.array([[39, 6, 4, np.nan, np.nan], [152.0, 242.0, 150.0, 2.0, 2.0]]),
            columns=['female', 'male', 'unknown', 'f', 'm'],
        )
        tm.assert_frame_equal(result, expected)

    @pytest.mark.filterwarnings('ignore:invalid value encountered in cast:RuntimeWarning')
    def test_constructor_series_nonexact_categoricalindex(self) -> None:
        ser: Series = Series(range(100))
        ser1: Series = cut(ser, 10).value_counts().head(5)
        ser2: Series = cut(ser, 10).value_counts().tail(5)
        result: DataFrame = DataFrame({'1': ser1, '2': ser2})
        index: pd.IntervalIndex = pd.interval_range(start=-0.099, end=99, periods=10, closed='right')
        expected: DataFrame = DataFrame(
            {'1': [10] * 5 + [np.nan] * 5, '2': [np.nan] * 5 + [10] * 5},
            index=index,
        )
        tm.assert_frame_equal(expected, result)

    def test_from_M8_structured(self) -> None:
        dates: List[Tuple[float, float]] = [(datetime(2012, 9, 9, 0, 0), datetime(2012, 9, 8, 15, 10))]
        arr: np.recarray = np.array(dates, dtype=[('Date', 'M8[us]'), ('Forecasting', 'M8[us]')]).view(np.recarray)
        df: DataFrame = DataFrame(arr)
        assert df['Date'][0] == dates[0][0]
        assert df['Forecasting'][0] == dates[0][1]
        s: Series = Series(arr['Date'])
        assert isinstance(s[0], Timestamp)
        assert s[0] == dates[0][0]

    def test_from_datetime_subclass(self) -> None:
        class DatetimeSubclass(datetime):
            pass

        datetimes: List[DatetimeSubclass] = [DatetimeSubclass(2020, 1, 1, 1, 1)]
        df: DataFrame = DataFrame({'datetime': datetimes})
        assert df['datetime'].dtype == 'datetime64[ns]'

    def test_with_mismatched_index_length_raises(self) -> None:
        dti: DatetimeIndex = date_range('2016-01-01', periods=3, tz='US/Pacific')
        msg: str = 'Shape of passed values|Passed arrays should have the same length'
        with pytest.raises(ValueError, match=msg):
            DataFrame(dti, index=range(4))

    def test_frame_ctor_datetime64_column(self) -> None:
        rng: pd.DatetimeIndex = date_range('1/1/2000 00:00:00', '1/1/2000 1:59:50', freq='10s')
        dates: np.ndarray = np.asarray(rng)
        df: DataFrame = DataFrame({'A': np.random.default_rng(2).standard_normal(len(rng)), 'B': dates})
        assert np.issubdtype(df['B'].dtype, np.dtype('M8[ns]'))

    def test_dataframe_constructor_infer_multiindex(self) -> None:
        index_lists: List[List[Any]] = [['a', 'a', 'b', 'b'], ['x', 'y', 'x', 'y']]
        multi: DataFrame = DataFrame(
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
        'input_vals',
        [
            [[1, 2], ['1', '2']],
            [['1', '2'], [1, 2]],
            [list(date_range('1/1/2011', periods=2, freq='h')),
             list(date_range('1/1/2011', periods=2, freq='h', tz='US/Eastern'))],
            [list(date_range('1/1/2011', periods=2, freq='h', tz='US/Eastern')),
             list(date_range('1/1/2011', periods=2, freq='h'))],
            [[Interval(left=0, right=5)]],
        ],
    )
    def test_constructor_list_str(
        self, input_vals: List[Any], string_dtype: Any
    ) -> None:
        result: DataFrame = DataFrame({'A': input_vals}, dtype=string_dtype)
        expected: DataFrame = DataFrame({'A': input_vals}).astype({'A': string_dtype})
        tm.assert_frame_equal(result, expected)

    def test_constructor_list_str_na(self, string_dtype: Any) -> None:
        result: DataFrame = DataFrame({'A': [1.0, 2.0, None]}, dtype=string_dtype)
        expected: DataFrame = DataFrame({'A': ['1.0', '2.0', None]}, dtype=object)
        tm.assert_frame_equal(result, expected)
        expected = DataFrame({'A': ['1.0', '2.0', None]}, dtype=object)
        tm.assert_frame_equal(result, expected)
        result = DataFrame({'A': [1.0, 2.0, None]}, dtype=string_dtype)
        tm.assert_frame_equal(result, expected)
        result = DataFrame({'A': [1.0, 2.0, None]}, dtype='object')
        expected = DataFrame({'A': ['1.0', '2.0', None]}, dtype=object)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        'copy', [False, True]
    )
    def test_dict_nocopy(
        self,
        copy: bool,
        any_numeric_ea_dtype: Any,
        any_numpy_dtype: Any,
    ) -> None:
        a: np.ndarray = np.array([1, 2], dtype=any_numpy_dtype)
        b: np.ndarray = np.array([3, 4], dtype=any_numpy_dtype)
        if b.dtype.kind in ['S', 'U']:
            pytest.skip(f'{b.dtype} get cast, making the checks below more cumbersome')
        c: Any = pd.array([1, 2], dtype=any_numeric_ea_dtype)
        c_orig: Any = c.copy()
        df: DataFrame = DataFrame({'a': a, 'b': b, 'c': c}, copy=copy)

        def get_base(obj: Any) -> Any:
            if isinstance(obj, np.ndarray):
                return obj.base
            elif isinstance(obj.dtype, np.dtype):
                return obj._ndarray.base
            else:
                raise TypeError

        def check_views(c_only: bool = False) -> None:
            assert sum(
                (x.values is c for x in df._mgr.blocks)
            ) == 1
            if c_only:
                return
            assert sum(
                (get_base(x.values) is a for x in df._mgr.blocks if isinstance(x.values.dtype, np.dtype))
            ) == 1
            assert sum(
                (get_base(x.values) is b for x in df._mgr.blocks if isinstance(x.values.dtype, np.dtype))
            ) == 1

        if not copy:
            check_views()
        should_raise: bool = not lib.is_np_dtype(df.dtypes.iloc[0], 'fciuO')
        if should_raise:
            with pytest.raises(TypeError, match='Invalid value'):
                df.iloc[0, 0] = 0
                df.iloc[0, 1] = 0
            return
        else:
            df.iloc[0, 0] = 0
            df.iloc[0, 1] = 0
        if not copy:
            check_views(True)
        df.iloc[:, 2] = pd.array([45, 46], dtype=c.dtype)
        assert df.dtypes.iloc[2] == c.dtype
        if copy:
            if a.dtype.kind == 'M':
                assert a[0] == a.dtype.type(1, 'ns')
                assert b[0] == b.dtype.type(3, 'ns')
            else:
                assert a[0] == a.dtype.type(1)
                assert b[0] == b.dtype.type(3)
            assert c[0] == c_orig[0]

    def test_construct_from_dict_ea_series(self) -> None:
        ser: Series = Series([1, 2, 3], dtype='Int64')
        df: DataFrame = DataFrame({'a': ser})
        assert not np.shares_memory(ser.values._data, df['a'].values._data)

    def test_from_series_with_name_with_columns(self) -> None:
        result: DataFrame = DataFrame(Series(1, name='foo'), columns=['bar'])
        expected: DataFrame = DataFrame(columns=['bar'])
        tm.assert_frame_equal(result, expected)

    def test_nested_list_columns(self) -> None:
        result: DataFrame = DataFrame(
            [[1, 2, 3], [4, 5, 6]], columns=[['A', 'A', 'A'], ['a', 'b', 'c']]
        )
        expected: DataFrame = DataFrame(
            [[1, 2, 3], [4, 5, 6]],
            columns=MultiIndex.from_tuples([('A', 'a'), ('A', 'b'), ('A', 'c')]),
        )
        tm.assert_frame_equal(result, expected)

    def test_from_2d_ndarray_with_dtype(self) -> None:
        array_dim2: np.ndarray = np.arange(10).reshape((5, 2))
        df: DataFrame = DataFrame(array_dim2, dtype='datetime64[ns, UTC]')
        expected: DataFrame = DataFrame(array_dim2).astype('datetime64[ns, UTC]')
        tm.assert_frame_equal(df, expected)

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
        td_as_int: List[int] = [1, 2, 3, 4]
        data: Dict[int, Any] = {i: klass(s) for i, s in enumerate(td_as_int)}
        expected: DataFrame = DataFrame(
            [
                {0: 0, 1: None, 2: None, 3: None},
                {0: None, 1: 2, 2: None, 3: None},
                {0: None, 1: None, 2: 4, 3: None},
                {0: None, 1: None, 2: None, 3: 6},
            ],
            index=[Timedelta(td, 'D') for td in td_as_int],
        )
        result: DataFrame = DataFrame(data)
        tm.assert_frame_equal(result, expected)

    def test_constructor_period_dict(self) -> None:
        a: pd.PeriodIndex = pd.PeriodIndex(['2012-01', 'NaT', '2012-04'], freq='M')
        b: pd.PeriodIndex = pd.PeriodIndex(['2012-02-01', '2012-03-01', 'NaT'], freq='D')
        df: DataFrame = DataFrame({'a': a, 'b': b})
        assert df['a'].dtype == a.dtype
        assert df['b'].dtype == b.dtype
        df = DataFrame({'a': a.astype(object).tolist(), 'b': b.astype(object).tolist()})
        assert df['a'].dtype == a.dtype
        assert df['b'].dtype == b.dtype

    def test_constructor_dict_extension_scalar(
        self, ea_scalar_and_dtype: Tuple[Any, Any]
    ) -> None:
        ea_scalar, ea_dtype = ea_scalar_and_dtype
        df: DataFrame = DataFrame({'a': ea_scalar}, index=[0])
        assert df['a'].dtype == ea_dtype
        expected: DataFrame = DataFrame(index=[0], columns=['a'], data=ea_scalar)
        tm.assert_frame_equal(df, expected)

    @pytest.mark.parametrize(
        'data,dtype',
        [
            (Period('2020-01'), PeriodDtype('M')),
            (Interval(left=0, right=5), IntervalDtype('int64', 'right')),
            (
                Timestamp('2011-01-01', tz='US/Eastern'),
                DatetimeTZDtype(unit='s', tz='US/Eastern'),
            ),
        ],
    )
    def test_constructor_extension_scalar_data(
        self, data: Any, dtype: Any
    ) -> None:
        df: DataFrame = DataFrame(index=range(2), columns=['a', 'b'], data=data)
        assert df['a'].dtype == dtype
        assert df['b'].dtype == dtype
        arr: Any = pd.array([data] * 2, dtype=dtype)
        expected: DataFrame = DataFrame({'a': arr, 'b': arr})
        tm.assert_frame_equal(df, expected)

    def test_nested_dict_frame_constructor(self) -> None:
        rng: pd.period_range = pd.period_range('1/1/2000', periods=5)
        df: DataFrame = DataFrame(
            np.random.default_rng(2).standard_normal((10, 5)), columns=rng
        )
        data: Dict[Any, Dict[Any, float]] = {}
        for col in df.columns:
            for row in df.index:
                data.setdefault(col, {})[row] = df._get_value(row, col)
        result: DataFrame = DataFrame(data, columns=rng)
        tm.assert_frame_equal(result, df)
        data = {}
        for col in df.columns:
            for row in df.index:
                data.setdefault(row, {})[col] = df._get_value(row, col)
        result = DataFrame(data, index=rng).T
        tm.assert_frame_equal(result, df)

    def _check_basic_constructor(
        self, empty: Callable[[Tuple[int, ...], ...], Union[np.ndarray, Any]]
    ) -> None:
        mat: np.ndarray = empty((2, 3), dtype=float)
        frame: DataFrame = DataFrame(mat, columns=['A', 'B', 'C'], index=[1, 2])
        assert len(frame.index) == 2
        assert len(frame.columns) == 3
        frame = DataFrame(empty((3,), dtype=float), columns=['A'], index=[1, 2, 3])
        assert len(frame.index) == 3
        assert len(frame.columns) == 1
        if empty is not np.ones:
            msg: str = 'Cannot convert non-finite values \\(NA or inf\\) to integer'
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
            DataFrame(empty((3, 3, 3), dtype=float), columns=['A', 'B', 'C'], index=[1])
        frame = DataFrame(mat)
        tm.assert_index_equal(frame.index, Index(range(2)), exact=True)
        tm.assert_index_equal(frame.columns, Index(range(3)), exact=True)
        frame = DataFrame(mat, index=[1, 2])
        tm.assert_index_equal(frame.columns, Index(range(3)), exact=True)
        frame = DataFrame(mat, columns=['A', 'B', 'C'])
        tm.assert_index_equal(frame.index, Index(range(2)), exact=True)
        frame = DataFrame(empty((0, 3), dtype=float))
        assert len(frame.index) == 0
        frame = DataFrame(empty((3, 0), dtype=float))
        assert len(frame.columns) == 0

    def test_constructor_ndarray(self) -> None:
        self._check_basic_constructor(np.ones)
        frame: DataFrame = DataFrame(['foo', 'bar'], index=[0, 1], columns=['A'])
        assert len(frame) == 2

    def test_constructor_maskedarray(self) -> None:
        self._check_basic_constructor(ma.masked_all)
        mat: np.ma.MaskedArray = ma.masked_all((2, 3), dtype=float)
        mat[0, 0] = 1.0
        mat[1, 2] = 2.0
        frame: DataFrame = DataFrame(mat, columns=['A', 'B', 'C'], index=[1, 2])
        assert 1.0 == frame['A'][1]
        assert 2.0 == frame['C'][2]
        mat = ma.masked_all((2, 3), dtype=float)
        frame = DataFrame(mat, columns=['A', 'B', 'C'], index=[1, 2])
        assert np.all(~np.asarray(frame == frame))

    @pytest.mark.filterwarnings('ignore:elementwise comparison failed:DeprecationWarning')
    def test_constructor_maskedarray_nonfloat(self) -> None:
        mat: np.ma.MaskedArray = ma.masked_all((2, 3), dtype=int)
        frame: DataFrame = DataFrame(mat, columns=['A', 'B', 'C'], index=[1, 2])
        assert len(frame.index) == 2
        assert len(frame.columns) == 3
        assert np.all(~np.asarray(frame == frame))
        frame = DataFrame(mat, columns=['A', 'B', 'C'], index=[1, 2], dtype=np.float64)
        assert frame.values.dtype == np.float64
        mat2: np.ma.MaskedArray = ma.copy(mat)
        mat2[0, 0] = 1
        mat2[1, 2] = 2
        frame = DataFrame(mat2, columns=['A', 'B', 'C'], index=[1, 2])
        assert 1 == frame['A'].astype('i8')[1]
        assert 2 == frame['C'].astype('i8')[2]
        mat = ma.masked_all((2, 3), dtype='M8[ns]')
        frame = DataFrame(mat, columns=['A', 'B', 'C'], index=[1, 2])
        assert len(frame.index) == 2
        assert len(frame.columns) == 3
        assert isna(frame).values.all()
        msg: str = 'datetime64\\[ns\\] values and dtype=int64 is not supported'
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
        mat_hard: np.ma.MaskedArray = ma.masked_all((2, 2), dtype=float).harden_mask()
        result: DataFrame = DataFrame(mat_hard, columns=['A', 'B'], index=[1, 2])
        expected: DataFrame = DataFrame(
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
        data: np.ma.MaskedArray = np.ma.array(
            np.ma.zeros(5, dtype=[('date', '<f8'), ('price', '<f8')]),
            mask=[False] * 5,
        )
        data = data.view(mrecords.mrecarray)
        with pytest.raises(TypeError, match='Pass \\{name: data\\[name\\]'):
            DataFrame(data, dtype=int)

    def test_constructor_corner_shape(self) -> None:
        df: DataFrame = DataFrame(index=[])
        assert df.values.shape == (0, 0)

    @pytest.mark.parametrize(
        'data,index,columns,dtype,expected',
        [
            (None, list(range(10)), ['a', 'b'], object, np.object_),
            (None, None, ['a', 'b'], 'int64', np.dtype('int64')),
            (None, list(range(10)), ['a', 'b'], int, np.dtype('float64')),
            ({}, None, ['foo', 'bar'], None, np.object_),
            (
                {'b': 1},
                list(range(10)),
                list('abc'),
                int,
                np.dtype('float64'),
            ),
        ],
    )
    def test_constructor_dtype(
        self,
        data: Optional[Union[List[Any], Dict[str, Any]]],
        index: Optional[Iterable[Any]],
        columns: List[str],
        dtype: Optional[Union[str, type]],
        expected: np.dtype,
    ) -> None:
        df: DataFrame = DataFrame(data, index, columns, dtype)
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
        expected_dtype: Callable[[], Any],
    ) -> None:
        df: DataFrame = DataFrame({'a': data}, dtype=input_dtype)
        assert df['a'].dtype == expected_dtype()

    def test_constructor_scalar_inference(self, using_infer_string: bool) -> None:
        data: Dict[str, Any] = {
            'int': 1,
            'bool': True,
            'float': 3.0,
            'complex': 4j,
            'object': 'foo',
        }
        df: DataFrame = DataFrame(data, index=np.arange(10))
        assert df['int'].dtype == np.int64
        assert df['bool'].dtype == np.bool_
        assert df['float'].dtype == np.float64
        assert df['complex'].dtype == np.complex128
        assert df['object'].dtype == (
            np.object_ if not using_infer_string else 'str'
        )

    def test_constructor_arrays_and_scalars(self) -> None:
        df: DataFrame = DataFrame(
            {'a': np.random.default_rng(2).standard_normal(10), 'b': True}
        )
        exp: DataFrame = DataFrame({'a': df['a'].values, 'b': [True] * 10})
        tm.assert_frame_equal(df, exp)
        with pytest.raises(ValueError, match='must pass an index'):
            DataFrame({'a': False, 'b': True})

    def test_constructor_DataFrame(self, float_frame: DataFrame) -> None:
        df: DataFrame = DataFrame(float_frame)
        tm.assert_frame_equal(df, float_frame)
        df_casted: DataFrame = DataFrame(float_frame, dtype=np.int64)
        assert df_casted.values.dtype == np.int64

    def test_constructor_empty_dataframe(self) -> None:
        actual: DataFrame = DataFrame(DataFrame(), dtype='object')
        expected: DataFrame = DataFrame([], dtype='object')
        tm.assert_frame_equal(actual, expected)

    def test_constructor_more(self, float_frame: DataFrame) -> None:
        arr: np.ndarray = np.random.default_rng(2).standard_normal(10)
        dm: DataFrame = DataFrame(arr, columns=['A'], index=np.arange(10))
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
        mat: np.ndarray = np.array(['foo', 'bar'], dtype=object).reshape(2, 1)
        msg: str = "could not convert string to float: 'foo'"
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
        df: DataFrame = DataFrame([], index=[])
        expected: DataFrame = DataFrame(index=[])
        tm.assert_frame_equal(df, expected)
        df = DataFrame([], columns=['A', 'B'])
        expected = DataFrame({}, columns=['A', 'B'])
        tm.assert_frame_equal(df, expected)

        def empty_gen() -> Generator[Any, None, None]:
            yield from ()

        df = DataFrame(empty_gen(), columns=['A', 'B'])
        tm.assert_frame_equal(df, expected)

    def test_constructor_list_of_lists(
        self, using_infer_string: bool
    ) -> None:
        df: DataFrame = DataFrame(data=[[1, 'a'], [2, 'b']], columns=['num', 'str'])
        assert is_integer_dtype(df['num'])
        assert df['str'].dtype == (
            np.object_ if not using_infer_string else 'str'
        )
        expected: DataFrame = DataFrame(np.arange(10))
        data: List[np.ndarray] = [np.array(x) for x in range(10)]
        result: DataFrame = DataFrame(data)
        tm.assert_frame_equal(result, expected)

    def test_nested_pandasarray_matches_nested_ndarray(self) -> None:
        ser: Series = Series([1, 2])
        arr: np.ndarray = np.array([None, None], dtype=object)
        arr[0] = ser
        arr[1] = ser * 2
        df: DataFrame = DataFrame(arr)
        expected: DataFrame = DataFrame(pd.array(arr))
        tm.assert_frame_equal(df, expected)
        assert df.shape == (2, 1)
        tm.assert_numpy_array_equal(df[0].values, arr)

    def test_constructor_list_like_data_nested_list_column(self) -> None:
        arrays: List[List[str]] = [list('abcd'), list('cdef')]
        result: DataFrame = DataFrame([[1, 2, 3, 4], [4, 5, 6, 7]], columns=arrays)
        mi: MultiIndex = MultiIndex.from_arrays(arrays)
        expected: DataFrame = DataFrame(
            [[1, 2, 3, 4], [4, 5, 6, 7]],
            columns=mi,
        )
        tm.assert_frame_equal(result, expected)

    def test_constructor_wrong_length_nested_list_column(self) -> None:
        arrays: List[List[Any]] = [list('abc'), list('cde')]
        msg: str = '3 columns passed, passed data had 4'
        with pytest.raises(ValueError, match=msg):
            DataFrame([[1, 2, 3, 4], [4, 5, 6, 7]], columns=arrays)

    def test_constructor_unequal_length_nested_list_column(self) -> None:
        arrays: List[List[Any]] = [list('abcd'), list('cde')]
        msg: str = 'all arrays must be same length'
        with pytest.raises(ValueError, match=msg):
            DataFrame([[1, 2, 3, 4], [4, 5, 6, 7]], columns=arrays)

    @pytest.mark.parametrize(
        'data',
        [
            [{'a': 1.5, 'b': 3}],
            [{'a': {'foo': 1, 'bar': 2, 'baz': 3}, 'b': Series([4, 3, 2, 1], index=['bar', 'qux', 'baz', 'foo'])}],
        ],
    )
    def test_constructor_mixed_dict_nonseries(
        self, data: List[Dict[str, Any]]
    ) -> None:
        data_dict: Dict[str, Any] = data[0]
        df: DataFrame = DataFrame(data_dict)
        tm.assert_frame_equal(df, DataFrame({k: list(v) for k, v in data_dict.items()}))

    def test_frame_from_dict_of_series_overlapping_monthly_period_indexes(
        self,
    ) -> None:
        rng1: pd.period_range = pd.period_range('1/1/1999', '1/1/2012', freq='M')
        s1: Series = Series(
            np.random.default_rng(2).standard_normal(len(rng1)), rng1
        )
        rng2: pd.period_range = pd.period_range('1/1/1980', '12/1/2001', freq='M')
        s2: Series = Series(
            np.random.default_rng(2).standard_normal(len(rng2)), rng2
        )
        df: DataFrame = DataFrame({'s1': s1, 's2': s2})
        exp: pd.PeriodIndex = pd.period_range('1/1/1980', '1/1/2012', freq='M')
        tm.assert_index_equal(df.index, exp)

    def test_frame_from_dict_with_mixed_tzaware_indexes(self) -> None:
        dti: DatetimeIndex = date_range('2016-01-01', periods=3, tz='US/Pacific')
        ser1: Series = Series(range(3), index=dti)
        ser2: Series = Series(range(3), index=dti.tz_localize('UTC'))
        ser3: Series = Series(range(3), index=dti.tz_localize('US/Central'))
        ser4: Series = Series(range(3))
        df1: DataFrame = DataFrame({'A': ser2, 'B': ser3, 'C': ser4})
        exp_index: Index = Index(list(ser2.index) + list(ser3.index) + list(ser4.index), dtype=object)
        tm.assert_index_equal(df1.index, exp_index)
        df2: DataFrame = DataFrame({'A': ser2, 'C': ser4, 'B': ser3})
        exp_index3: Index = Index(list(ser2.index) + list(ser4.index) + list(ser3.index), dtype=object)
        tm.assert_index_equal(df2.index, exp_index3)
        df3: DataFrame = DataFrame({'B': ser3, 'A': ser2, 'C': ser4})
        exp_index3 = Index(list(ser3.index) + list(ser2.index) + list(ser4.index), dtype=object)
        tm.assert_index_equal(df3.index, exp_index3)
        df4: DataFrame = DataFrame({'C': ser4, 'B': ser3, 'A': ser2})
        exp_index4: Index = Index(list(ser4.index) + list(ser3.index) + list(ser2.index), dtype=object)
        tm.assert_index_equal(df4.index, exp_index4)
        msg: str = 'Cannot join tz-naive with tz-aware DatetimeIndex'
        with pytest.raises(TypeError, match=msg):
            DataFrame({'A': ser2, 'B': ser3, 'C': ser4, 'D': ser1})
        with pytest.raises(TypeError, match=msg):
            DataFrame({'A': ser2, 'B': ser3, 'D': ser1})
        with pytest.raises(TypeError, match=msg):
            DataFrame({'D': ser1, 'A': ser2, 'B': ser3})

    def test_frame_ctor_datetime64_column(self) -> None:
        rng: pd.DatetimeIndex = date_range('1/1/2000 00:00:00', '1/1/2000 1:59:50', freq='10s')
        dates: np.ndarray = np.asarray(rng)
        df: DataFrame = DataFrame({'A': np.random.default_rng(2).standard_normal(len(rng)), 'B': dates})
        assert np.issubdtype(df['B'].dtype, np.dtype('M8[ns]'))

    def test_dataframe_constructor_infer_multiindex(self) -> None:
        index_lists: List[List[Any]] = [['a', 'a', 'b', 'b'], ['x', 'y', 'x', 'y']]
        multi: DataFrame = DataFrame(
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
        'input_vals',
        [
            [[1, 2], ['1', '2']],
            [['1', '2'], [1, 2]],
            [
                list(date_range('1/1/2011', periods=2, freq='h')),
                list(date_range('1/1/2011', periods=2, freq='h', tz='US/Eastern')),
            ],
            [
                list(date_range('1/1/2011', periods=2, freq='h', tz='US/Eastern')),
                list(date_range('1/1/2011', periods=2, freq='h')),
            ],
            [[Interval(left=0, right=5)]],
        ],
    )
    def test_constructor_list_str(
        self,
        input_vals: List[Any],
        string_dtype: Any,
    ) -> None:
        result: DataFrame = DataFrame({'A': input_vals}, dtype=string_dtype)
        expected: DataFrame = DataFrame({'A': input_vals}).astype({'A': string_dtype})
        tm.assert_frame_equal(result, expected)

    def test_constructor_list_str_na(self, string_dtype: Any) -> None:
        result: DataFrame = DataFrame({'A': [1.0, 2.0, None]}, dtype=string_dtype)
        expected: DataFrame = DataFrame({'A': ['1.0', '2.0', None]}, dtype=object)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        'copy', [False, True]
    )
    def test_dict_nocopy(
        self,
        copy: bool,
        any_numeric_ea_dtype: Any,
        any_numpy_dtype: Any,
    ) -> None:
        a: np.ndarray = np.array([1, 2], dtype=any_numpy_dtype)
        b: np.ndarray = np.array([3, 4], dtype=any_numpy_dtype)
        if b.dtype.kind in ['S', 'U']:
            pytest.skip(f'{b.dtype} get cast, making the checks below more cumbersome')
        c: Any = pd.array([1, 2], dtype=any_numeric_ea_dtype)
        c_orig: Any = c.copy()
        df: DataFrame = DataFrame({'a': a, 'b': b, 'c': c}, copy=copy)

        def get_base(obj: Any) -> Any:
            if isinstance(obj, np.ndarray):
                return obj.base
            elif isinstance(obj.dtype, np.dtype):
                return obj._ndarray.base
            else:
                raise TypeError

        def check_views(c_only: bool = False) -> None:
            assert sum(
                (x.values is c for x in df._mgr.blocks)
            ) == 1
            if c_only:
                return
            assert sum(
                (get_base(x.values) is a for x in df._mgr.blocks if isinstance(x.values.dtype, np.dtype))
            ) == 1
            assert sum(
                (get_base(x.values) is b for x in df._mgr.blocks if isinstance(x.values.dtype, np.dtype))
            ) == 1

        if not copy:
            check_views()
        should_raise: bool = not lib.is_np_dtype(df.dtypes.iloc[0], 'fciuO')
        if should_raise:
            with pytest.raises(TypeError, match='Invalid value'):
                df.iloc[0, 0] = 0
                df.iloc[0, 1] = 0
            return
        else:
            df.iloc[0, 0] = 0
            df.iloc[0, 1] = 0
        if not copy:
            check_views(True)
        df.iloc[:, 2] = pd.array([45, 46], dtype=c.dtype)
        assert df.dtypes.iloc[2] == c.dtype
        if copy:
            if a.dtype.kind == 'M':
                assert a[0] == a.dtype.type(1, 'ns')
                assert b[0] == b.dtype.type(3, 'ns')
            else:
                assert a[0] == a.dtype.type(1)
                assert b[0] == b.dtype.type(3)
            assert c[0] == c_orig[0]

    def test_construct_from_dict_ea_series(self) -> None:
        ser: Series = Series([1, 2, 3], dtype='Int64')
        df: DataFrame = DataFrame({'a': ser})
        assert not np.shares_memory(ser.values._data, df['a'].values._data)

    def test_from_series_with_name_with_columns(self) -> None:
        result: DataFrame = DataFrame(Series(1, name='foo'), columns=['bar'])
        expected: DataFrame = DataFrame(columns=['bar'])
        tm.assert_frame_equal(result, expected)

    def test_nested_list_columns(self) -> None:
        result: DataFrame = DataFrame(
            [[1, 2, 3], [4, 5, 6]],
            columns=[['A', 'A', 'A'], ['a', 'b', 'c']],
        )
        expected: DataFrame = DataFrame(
            [[1, 2, 3], [4, 5, 6]],
            columns=MultiIndex.from_tuples([('A', 'a'), ('A', 'b'), ('A', 'c')]),
        )
        tm.assert_frame_equal(result, expected)

    def test_from_2d_ndarray_with_dtype(self) -> None:
        array_dim2: np.ndarray = np.arange(10).reshape((5, 2))
        df: DataFrame = DataFrame(array_dim2, dtype='datetime64[ns, UTC]')
        expected: DataFrame = DataFrame(array_dim2).astype('datetime64[ns, UTC]')
        tm.assert_frame_equal(df, expected)

    def test_frame_from_dict_of_series_overlapping_monthly_period_indexes(
        self,
    ) -> None:
        rng1: pd.period_range = pd.period_range('1/1/1999', '1/1/2012', freq='M')
        s1: Series = Series(
            np.random.default_rng(2).standard_normal(len(rng1)), rng1
        )
        rng2: pd.period_range = pd.period_range('1/1/1980', '12/1/2001', freq='M')
        s2: Series = Series(
            np.random.default_rng(2).standard_normal(len(rng2)), rng2
        )
        df: DataFrame = DataFrame({'s1': s1, 's2': s2})
        exp: pd.PeriodIndex = pd.period_range('1/1/1980', '1/1/2012', freq='M')
        tm.assert_index_equal(df.index, exp)

    def test_frame_from_dict_with_mixed_tzaware_indexes(self) -> None:
        dti: DatetimeIndex = date_range('2016-01-01', periods=3, tz='US/Pacific')
        ser1: Series = Series(range(3), index=dti)
        ser2: Series = Series(range(3), index=dti.tz_localize('UTC'))
        ser3: Series = Series(range(3), index=dti.tz_localize('US/Central'))
        ser4: Series = Series(range(3))
        df1: DataFrame = DataFrame({'A': ser2, 'B': ser3, 'C': ser4})
        exp_index: Index = Index(
            list(ser2.index) + list(ser3.index) + list(ser4.index), dtype=object
        )
        tm.assert_index_equal(df1.index, exp_index)
        df2: DataFrame = DataFrame({'A': ser2, 'C': ser4, 'B': ser3})
        exp_index3: Index = Index(
            list(ser2.index) + list(ser4.index) + list(ser3.index), dtype=object
        )
        tm.assert_index_equal(df2.index, exp_index3)
        df3: DataFrame = DataFrame({'B': ser3, 'A': ser2, 'C': ser4})
        exp_index3 = Index(
            list(ser3.index) + list(ser2.index) + list(ser4.index), dtype=object
        )
        tm.assert_index_equal(df3.index, exp_index3)
        df4: DataFrame = DataFrame({'C': ser4, 'B': ser3, 'A': ser2})
        exp_index4: Index = Index(
            list(ser4.index) + list(ser3.index) + list(ser2.index), dtype=object
        )
        tm.assert_index_equal(df4.index, exp_index4)
        msg: str = 'Cannot join tz-naive with tz-aware DatetimeIndex'
        with pytest.raises(TypeError, match=msg):
            DataFrame({'A': ser2, 'B': ser3, 'C': ser4, 'D': ser1})
        with pytest.raises(TypeError, match=msg):
            DataFrame({'A': ser2, 'B': ser3, 'D': ser1})
        with pytest.raises(TypeError, match=msg):
            DataFrame({'D': ser1, 'A': ser2, 'B': ser3})

    def test_frame_ctor_datetime64_column(self) -> None:
        rng: pd.DatetimeIndex = date_range('1/1/2000 00:00:00', '1/1/2000 1:59:50', freq='10s')
        dates: np.ndarray = np.asarray(rng)
        df: DataFrame = DataFrame({'A': np.random.default_rng(2).standard_normal(len(rng)), 'B': dates})
        assert np.issubdtype(df['B'].dtype, np.dtype('M8[ns]'))

    def test_dataframe_constructor_infer_multiindex(self) -> None:
        index_lists: List[List[Any]] = [['a', 'a', 'b', 'b'], ['x', 'y', 'x', 'y']]
        multi: DataFrame = DataFrame(
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
        'input_vals',
        [
            [[1, 2], ['1', '2']],
            [['1', '2'], [1, 2]],
            [
                list(date_range('1/1/2011', periods=2, freq='h')),
                list(date_range('1/1/2011', periods=2, freq='h', tz='US/Eastern')),
            ],
            [
                list(date_range('1/1/2011', periods=2, freq='h', tz='US/Eastern')),
                list(date_range('1/1/2011', periods=2, freq='h')),
            ],
            [[Interval(left=0, right=5)]],
        ],
    )
    def test_constructor_list_str(
        self,
        input_vals: List[Any],
        string_dtype: Any,
    ) -> None:
        result: DataFrame = DataFrame({'A': input_vals}, dtype=string_dtype)
        expected: DataFrame = DataFrame({'A': input_vals}).astype({'A': string_dtype})
        tm.assert_frame_equal(result, expected)

    def test_constructor_list_str_na(self, string_dtype: Any) -> None:
        result: DataFrame = DataFrame({'A': [1.0, 2.0, None]}, dtype=string_dtype)
        expected: DataFrame = DataFrame({'A': ['1.0', '2.0', None]}, dtype=object)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        'copy', [False, True]
    )
    def test_dict_nocopy(
        self,
        copy: bool,
        any_numeric_ea_dtype: Any,
        any_numpy_dtype: Any,
    ) -> None:
        a: np.ndarray = np.array([1, 2], dtype=any_numpy_dtype)
        b: np.ndarray = np.array([3, 4], dtype=any_numpy_dtype)
        if b.dtype.kind in ['S', 'U']:
            pytest.skip(f'{b.dtype} get cast, making the checks below more cumbersome')
        c: Any = pd.array([1, 2], dtype=any_numeric_ea_dtype)
        c_orig: Any = c.copy()
        df: DataFrame = DataFrame({'a': a, 'b': b, 'c': c}, copy=copy)

        def get_base(obj: Any) -> Any:
            if isinstance(obj, np.ndarray):
                return obj.base
            elif isinstance(obj.dtype, np.dtype):
                return obj._ndarray.base
            else:
                raise TypeError

        def check_views(c_only: bool = False) -> None:
            assert sum(
                (x.values is c for x in df._mgr.blocks)
            ) == 1
            if c_only:
                return
            assert sum(
                (get_base(x.values) is a for x in df._mgr.blocks if isinstance(x.values.dtype, np.dtype))
            ) == 1
            assert sum(
                (get_base(x.values) is b for x in df._mgr.blocks if isinstance(x.values.dtype, np.dtype))
            ) == 1

        if not copy:
            check_views()
        should_raise: bool = not lib.is_np_dtype(df.dtypes.iloc[0], 'fciuO')
        if should_raise:
            with pytest.raises(TypeError, match='Invalid value'):
                df.iloc[0, 0] = 0
                df.iloc[0, 1] = 0
            return
        else:
            df.iloc[0, 0] = 0
            df.iloc[0, 1] = 0
        if not copy:
            check_views(True)
        df.iloc[:, 2] = pd.array([45, 46], dtype=c.dtype)
        assert df.dtypes.iloc[2] == c.dtype
        if copy:
            if a.dtype.kind == 'M':
                assert a[0] == a.dtype.type(1, 'ns')
                assert b[0] == b.dtype.type(3, 'ns')
            else:
                assert a[0] == a.dtype.type(1)
                assert b[0] == b.dtype.type(3)
            assert c[0] == c_orig[0]

    def test_construct_from_dict_ea_series(self) -> None:
        ser: Series = Series([1, 2, 3], dtype='Int64')
        df: DataFrame = DataFrame({'a': ser})
        assert not np.shares_memory(ser.values._data, df['a'].values._data)

    def test_from_series_with_name_with_columns(self) -> None:
        result: DataFrame = DataFrame(Series(1, name='foo'), columns=['bar'])
        expected: DataFrame = DataFrame(columns=['bar'])
        tm.assert_frame_equal(result, expected)

    def test_nested_list_columns(self) -> None:
        result: DataFrame = DataFrame(
            [[1, 2, 3], [4, 5, 6]],
            columns=[['A', 'A', 'A'], ['a', 'b', 'c']],
        )
        expected: DataFrame = DataFrame(
            [[1, 2, 3], [4, 5, 6]],
            columns=MultiIndex.from_tuples([('A', 'a'), ('A', 'b'), ('A', 'c')]),
        )
        tm.assert_frame_equal(result, expected)

    def test_from_2d_ndarray_with_dtype(self) -> None:
        array_dim2: np.ndarray = np.arange(10).reshape((5, 2))
        df: DataFrame = DataFrame(array_dim2, dtype='datetime64[ns, UTC]')
        expected: DataFrame = DataFrame(array_dim2).astype('datetime64[ns, UTC]')
        tm.assert_frame_equal(df, expected)

    def test_frame_from_dict_of_series_overlapping_monthly_period_indexes(
        self,
    ) -> None:
        rng1: pd.period_range = pd.period_range('1/1/1999', '1/1/2012', freq='M')
        s1: Series = Series(
            np.random.default_rng(2).standard_normal(len(rng1)), rng1
        )
        rng2: pd.period_range = pd.period_range('1/1/1980', '12/1/2001', freq='M')
        s2: Series = Series(
            np.random.default_rng(2).standard_normal(len(rng2)), rng2
        )
        df: DataFrame = DataFrame({'s1': s1, 's2': s2})
        exp: pd.PeriodIndex = pd.period_range('1/1/1980', '1/1/2012', freq='M')
        tm.assert_index_equal(df.index, exp)

    def test_frame_from_dict_with_mixed_tzaware_indexes(self) -> None:
        dti: DatetimeIndex = date_range('2016-01-01', periods=3, tz='US/Pacific')
        ser1: Series = Series(range(3), index=dti)
        ser2: Series = Series(range(3), index=dti.tz_localize('UTC'))
        ser3: Series = Series(range(3), index=dti.tz_localize('US/Central'))
        ser4: Series = Series(range(3))
        df1: DataFrame = DataFrame({'A': ser2, 'B': ser3, 'C': ser4})
        exp_index: Index = Index(
            list(ser2.index) + list(ser3.index) + list(ser4.index), dtype=object
        )
        tm.assert_index_equal(df1.index, exp_index)
        df2: DataFrame = DataFrame({'A': ser2, 'C': ser4, 'B': ser3})
        exp_index3: Index = Index(
            list(ser2.index) + list(ser4.index) + list(ser3.index), dtype=object
        )
        tm.assert_index_equal(df2.index, exp_index3)
        df3: DataFrame = DataFrame({'B': ser3, 'A': ser2, 'C': ser4})
        exp_index3 = Index(
            list(ser3.index) + list(ser2.index) + list(ser4.index), dtype=object
        )
        tm.assert_index_equal(df3.index, exp_index3)
        df4: DataFrame = DataFrame({'C': ser4, 'B': ser3, 'A': ser2})
        exp_index4: Index = Index(
            list(ser4.index) + list(ser3.index) + list(ser2.index), dtype=object
        )
        tm.assert_index_equal(df4.index, exp_index4)
        msg: str = 'Cannot join tz-naive with tz-aware DatetimeIndex'
        with pytest.raises(TypeError, match=msg):
            DataFrame({'A': ser2, 'B': ser3, 'C': ser4, 'D': ser1})
        with pytest.raises(TypeError, match=msg):
            DataFrame({'A': ser2, 'B': ser3, 'D': ser1})
        with pytest.raises(TypeError, match=msg):
            DataFrame({'D': ser1, 'A': ser2, 'B': ser3})

    def test_frame_ctor_datetime64_column(self) -> None:
        rng: pd.DatetimeIndex = date_range('1/1/2000 00:00:00', '1/1/2000 1:59:50', freq='10s')
        dates: np.ndarray = np.asarray(rng)
        df: DataFrame = DataFrame({'A': np.random.default_rng(2).standard_normal(len(rng)), 'B': dates})
        assert np.issubdtype(df['B'].dtype, np.dtype('M8[ns]'))

    def test_dataframe_constructor_infer_multiindex(self) -> None:
        index_lists: List[List[Any]] = [['a', 'a', 'b', 'b'], ['x', 'y', 'x', 'y']]
        multi: DataFrame = DataFrame(
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
        'input_vals',
        [
            [[1, 2], ['1', '2']],
            [['1', '2'], [1, 2]],
            [
                list(date_range('1/1/2011', periods=2, freq='h')),
                list(date_range('1/1/2011', periods=2, freq='h', tz='US/Eastern')),
            ],
            [
                list(date_range('1/1/2011', periods=2, freq='h', tz='US/Eastern')),
                list(date_range('1/1/2011', periods=2, freq='h')),
            ],
            [[Interval(left=0, right=5)]],
        ],
    )
    def test_constructor_list_str(
        self,
        input_vals: List[Any],
        string_dtype: Any,
    ) -> None:
        result: DataFrame = DataFrame({'A': input_vals}, dtype=string_dtype)
        expected: DataFrame = DataFrame({'A': input_vals}).astype({'A': string_dtype})
        tm.assert_frame_equal(result, expected)

    def test_constructor_list_str_na(
        self,
        string_dtype: Any,
    ) -> None:
        result: DataFrame = DataFrame({'A': [1.0, 2.0, None]}, dtype=string_dtype)
        expected: DataFrame = DataFrame({'A': ['1.0', '2.0', None]}, dtype=object)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        'copy', [False, True]
    )
    def test_dict_nocopy(
        self,
        copy: bool,
        any_numeric_ea_dtype: Any,
        any_numpy_dtype: Any,
    ) -> None:
        a: np.ndarray = np.array([1, 2], dtype=any_numpy_dtype)
        b: np.ndarray = np.array([3, 4], dtype=any_numpy_dtype)
        if b.dtype.kind in ['S', 'U']:
            pytest.skip(f'{b.dtype} get cast, making the checks below more cumbersome')
        c: Any = pd.array([1, 2], dtype=any_numeric_ea_dtype)
        c_orig: Any = c.copy()
        df: DataFrame = DataFrame({'a': a, 'b': b, 'c': c}, copy=copy)

        def get_base(obj: Any) -> Any:
            if isinstance(obj, np.ndarray):
                return obj.base
            elif isinstance(obj.dtype, np.dtype):
                return obj._ndarray.base
            else:
                raise TypeError

        def check_views(c_only: bool = False) -> None:
            assert sum(
                (x.values is c for x in df._mgr.blocks)
            ) == 1
            if c_only:
                return
            assert sum(
                (get_base(x.values) is a for x in df._mgr.blocks if isinstance(x.values.dtype, np.dtype))
            ) == 1
            assert sum(
                (get_base(x.values) is b for x in df._mgr.blocks if isinstance(x.values.dtype, np.dtype))
            ) == 1

        if not copy:
            check_views()
        should_raise: bool = not lib.is_np_dtype(df.dtypes.iloc[0], 'fciuO')
        if should_raise:
            with pytest.raises(TypeError, match='Invalid value'):
                df.iloc[0, 0] = 0
                df.iloc[0, 1] = 0
            return
        else:
            df.iloc[0, 0] = 0
            df.iloc[0, 1] = 0
        if not copy:
            check_views(True)
        df.iloc[:, 2] = pd.array([45, 46], dtype=c.dtype)
        assert df.dtypes.iloc[2] == c.dtype
        if copy:
            if a.dtype.kind == 'M':
                assert a[0] == a.dtype.type(1, 'ns')
                assert b[0] == b.dtype.type(3, 'ns')
            else:
                assert a[0] == a.dtype.type(1)
                assert b[0] == b.dtype.type(3)
            assert c[0] == c_orig[0]

    def test_construct_from_dict_ea_series(self) -> None:
        ser: Series = Series([1, 2, 3], dtype='Int64')
        df: DataFrame = DataFrame({'a': ser})
        assert not np.shares_memory(ser.values._data, df['a'].values._data)

    def test_from_series_with_name_with_columns(self) -> None:
        result: DataFrame = DataFrame(Series(1, name='foo'), columns=['bar'])
        expected: DataFrame = DataFrame(columns=['bar'])
        tm.assert_frame_equal(result, expected)

    def test_nested_list_columns(self) -> None:
        result: DataFrame = DataFrame(
            [[1, 2, 3], [4, 5, 6]],
            columns=[['A', 'A', 'A'], ['a', 'b', 'c']],
        )
        expected: DataFrame = DataFrame(
            [[1, 2, 3], [4, 5, 6]],
            columns=MultiIndex.from_tuples([('A', 'a'), ('A', 'b'), ('A', 'c')]),
        )
        tm.assert_frame_equal(result, expected)

    def test_from_2d_ndarray_with_dtype(self) -> None:
        array_dim2: np.ndarray = np.arange(10).reshape((5, 2))
        df: DataFrame = DataFrame(array_dim2, dtype='datetime64[ns, UTC]')
        expected: DataFrame = DataFrame(array_dim2).astype('datetime64[ns, UTC]')
        tm.assert_frame_equal(df, expected)

    def test_constructor_mixed_dict_nonseries(self) -> None:
        data: Dict[str, Any] = {'A': {'foo': 1, 'bar': 2, 'baz': 3}, 'B': Series([4, 3, 2, 1], index=['bar', 'qux', 'baz', 'foo'])}
        result: DataFrame = DataFrame(data)
        assert result.index.is_monotonic_increasing
        with pytest.raises(ValueError, match='ambiguous ordering'):
            DataFrame({'A': ['a', 'b'], 'B': {'a': 'a', 'b': 'b'}})
        result = DataFrame({'A': ['a', 'b'], 'B': Series(['a', 'b'], index=['a', 'b'])})
        expected: DataFrame = DataFrame({'A': ['a', 'b'], 'B': ['a', 'b']}, index=['a', 'b'])
        tm.assert_frame_equal(result, expected)

    def test_constructor_mixed_type_rows(self) -> None:
        data: List[Union[List[Any], Tuple[Any, Any]]] = [[1, 2], (3, 4)]
        result: DataFrame = DataFrame(data)
        expected: DataFrame = DataFrame([[1, 2], [3, 4]])
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
        lists: List[List[Any]],
    ) -> None:
        result: DataFrame = DataFrame(tuples)
        expected: DataFrame = DataFrame(lists)
        tm.assert_frame_equal(result, expected)

    def test_constructor_list_of_tuples(self) -> None:
        data: Dict[str, Tuple[int, int]] = {'A': (1, 2), 'B': (3, 4)}
        result: DataFrame = DataFrame({'A': [(1, 2), (3, 4)]})
        expected: DataFrame = DataFrame({'A': Series([(1, 2), (3, 4)])})
        tm.assert_frame_equal(result, expected)

    def test_constructor_list_of_namedtuples(self) -> None:
        named_tuple = namedtuple('Pandas', list('ab'))
        tuples: List[Any] = [named_tuple(1, 3), named_tuple(2, 4)]
        expected: DataFrame = DataFrame({'a': [1, 2], 'b': [3, 4]})
        result: DataFrame = DataFrame(tuples)
        tm.assert_frame_equal(result, expected)
        expected = DataFrame({'y': [1, 2], 'z': [3, 4]})
        result = DataFrame(tuples, columns=['y', 'z'])
        tm.assert_frame_equal(result, expected)

    def test_constructor_list_of_dataclasses(self) -> None:
        Point = make_dataclass('Point', [('x', int), ('y', int)])
        data: List[Any] = [Point(0, 3), Point(1, 3)]
        expected: DataFrame = DataFrame({'x': [0, 1], 'y': [3, 3]})
        result: DataFrame = DataFrame(data)
        tm.assert_frame_equal(result, expected)

    def test_constructor_list_of_dataclasses_with_varying_types(self) -> None:
        Point = make_dataclass('Point', [('x', int), ('y', int)])
        HLine = make_dataclass('HLine', [('x0', int), ('x1', int), ('y', int)])
        data: List[Any] = [Point(0, 3), HLine(1, 3, 3)]
        expected: DataFrame = DataFrame(
            {'x': [0, np.nan], 'y': [3, 3], 'x0': [np.nan, 1], 'x1': [np.nan, 3]}
        )
        result: DataFrame = DataFrame(data)
        tm.assert_frame_equal(result, expected)

    def test_constructor_list_of_dataclasses_error_thrown(self) -> None:
        Point = make_dataclass('Point', [('x', int), ('y', int)])
        msg: str = 'asdict() should be called on dataclass instances'
        with pytest.raises(TypeError, match=re.escape(msg)):
            DataFrame([Point(0, 0), {'x': 1, 'y': 0}])

    def test_constructor_list_of_dict_order(self) -> None:
        data: List[Dict[str, Any]] = [
            {'First': 1, 'Second': 4, 'Third': 7, 'Fourth': 10},
            {'Second': 5, 'First': 2, 'Fourth': 11, 'Third': 8},
            {'Second': 6, 'First': 3, 'Fourth': 12, 'Third': 9, 'YYY': 14, 'XXX': 13},
        ]
        expected: DataFrame = DataFrame(
            {'First': [1, 2, 3], 'Second': [4, 5, 6], 'Third': [7, 8, 9], 'Fourth': [10, 11, 12], 'YYY': [None, None, 14], 'XXX': [None, None, 13]}
        )
        result: DataFrame = DataFrame(data)
        tm.assert_frame_equal(result, expected)

    def test_constructor_Series_named(self) -> None:
        a: Series = Series([1, 2, 3], name=0)
        df: DataFrame = DataFrame(a)
        assert df.columns[0] == 0
        tm.assert_index_equal(df.index, a.index)
        arr: np.ndarray = np.random.default_rng(2).standard_normal(10)
        s: Series = Series(arr, name='x')
        df = DataFrame(s)
        expected: Series = Series(arr, dtype=np.float64, name='x')
        tm.assert_series_equal(df['x'], expected)
        s = Series(arr, index=range(3, 13))
        df = DataFrame(s)
        expected = Series(arr, index=range(3, 13), name='x')
        tm.assert_series_equal(df[0], Series(expected, name=0))
        msg: str = 'Shape of passed values is \\(10, 1\\), indices imply \\(10, 2\\)'
        with pytest.raises(ValueError, match=msg):
            DataFrame(s, columns=[1, 2])
        a = Series([], name='x', dtype=object)
        df = DataFrame(a)
        assert df.columns[0] == 'x'
        s1: Series = Series(arr, name='x')
        df = DataFrame([s1, arr]).T
        expected: DataFrame = DataFrame({'x': s1, 'Unnamed 0': arr}, columns=['x', 'Unnamed 0'])
        tm.assert_series_equal(df['x'], expected['x'])
        df = DataFrame([arr, s1]).T
        expected = DataFrame({'1': s1, '0': arr}, columns=range(2))
        tm.assert_frame_equal(df, expected)

    def test_constructor_Series_named_and_columns(self) -> None:
        s0: Series = Series(range(5), name=0)
        s1: Series = Series(range(5), name=1)
        tm.assert_frame_equal(DataFrame(s0, columns=[0]), s0.to_frame())
        tm.assert_frame_equal(DataFrame(s1, columns=[1]), s1.to_frame())
        assert DataFrame(s0, columns=[1]).empty
        assert DataFrame(s1, columns=[0]).empty

    def test_constructor_Series_differently_indexed(self) -> None:
        s1: Series = Series([1, 2, 3], index=['a', 'b', 'c'], name='x')
        s2: Series = Series([1, 2, 3], index=['a', 'b', 'c'])
        other_index: Index = Index(['a', 'b'])
        df1: DataFrame = DataFrame(s1, index=other_index)
        exp1: DataFrame = DataFrame(s1.reindex(other_index))
        assert df1.columns[0] == 'x'
        tm.assert_frame_equal(df1, exp1)
        df2: DataFrame = DataFrame(s2, index=other_index)
        exp2: DataFrame = DataFrame(s2.reindex(other_index))
        assert df2.columns[0] == 0
        tm.assert_index_equal(df2.index, other_index)
        tm.assert_frame_equal(df2, exp2)

    @pytest.mark.parametrize(
        'name_in1,name_in2,name_in3,name_out',
        [
            ('idx', 'idx', 'idx', 'idx'),
            ('idx', 'idx', None, None),
            ('idx', None, None, None),
            ('idx1', 'idx2', None, None),
            ('idx1', 'idx1', 'idx2', None),
            ('idx1', 'idx2', 'idx3', None),
            (None, None, None, None),
        ],
    )
    def test_constructor_index_names(
        self,
        name_in1: Optional[str],
        name_in2: Optional[str],
        name_in3: Optional[str],
        name_out: Optional[str],
    ) -> None:
        indices: List[Index] = [
            Index(['a', 'a', 'b', 'b'], name=name_in1),
            Index(['x', 'y', 'x', 'y'], name=name_in2),
        ]
        multi: DataFrame = DataFrame(
            np.random.default_rng(2).standard_normal((4, 4)),
            index=[np.array(x) for x in index_lists],
        )
        assert isinstance(multi.index, MultiIndex)
        assert not isinstance(multi.columns, MultiIndex)
        index_lists: List[List[Any]] = [['a', 'a', 'b', 'b'], ['x', 'y', 'x', 'y']]
        multi = DataFrame(
            np.random.default_rng(2).standard_normal((4, 4)),
            index=[np.array(x) for x in index_lists],
        )
        assert isinstance(multi.index, MultiIndex)
        assert not isinstance(multi.columns, MultiIndex)

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
        td_as_int: List[int] = [1, 2, 3, 4]
        data: Dict[int, Any] = {i: klass(s) for i, s in enumerate(td_as_int)}
        expected: DataFrame = DataFrame(
            [
                {0: 0, 1: None, 2: None, 3: None},
                {0: None, 1: 2, 2: None, 3: None},
                {0: None, 1: None, 2: 4, 3: None},
                {0: None, 1: None, 2: None, 3: 6},
            ],
            index=[Timedelta(td, 'D') for td in td_as_int],
        )
        result: DataFrame = DataFrame(data)
        tm.assert_frame_equal(result, expected)

    def test_constructor_period_dict(self) -> None:
        a: pd.PeriodIndex = pd.PeriodIndex(['2012-01', 'NaT', '2012-04'], freq='M')
        b: pd.PeriodIndex = pd.PeriodIndex(['2012-02-01', '2012-03-01', 'NaT'], freq='D')
        df: DataFrame = DataFrame({'a': a, 'b': b})
        assert df['a'].dtype == a.dtype
        assert df['b'].dtype == b.dtype
        df = DataFrame({'a': a.astype(object).tolist(), 'b': b.astype(object).tolist()})
        assert df['a'].dtype == a.dtype
        assert df['b'].dtype == b.dtype

    def test_constructor_dict_extension_scalar(
        self, ea_scalar_and_dtype: Tuple[Any, Any]
    ) -> None:
        ea_scalar, ea_dtype = ea_scalar_and_dtype
        df: DataFrame = DataFrame({'a': ea_scalar}, index=[0])
        assert df['a'].dtype == ea_dtype
        expected: DataFrame = DataFrame(index=[0], columns=['a'], data=ea_scalar)
        tm.assert_frame_equal(df, expected)

    @pytest.mark.parametrize(
        'data,dtype',
        [
            (Period('2020-01'), PeriodDtype('M')),
            (Interval(left=0, right=5), IntervalDtype('int64', 'right')),
            (
                Timestamp('2011-01-01', tz='US/Eastern'),
                DatetimeTZDtype(unit='s', tz='US/Eastern'),
            ),
        ],
    )
    def test_constructor_extension_scalar_data(
        self, data: Any, dtype: Any
    ) -> None:
        df: DataFrame = DataFrame(index=range(2), columns=['a', 'b'], data=data)
        assert df['a'].dtype == dtype
        assert df['b'].dtype == dtype
        arr: Any = pd.array([data] * 2, dtype=dtype)
        expected: DataFrame = DataFrame({'a': arr, 'b': arr})
        tm.assert_frame_equal(df, expected)

    def test_nested_dict_frame_constructor(self) -> None:
        rng: pd.period_range = pd.period_range('1/1/2000', periods=5)
        df: DataFrame = DataFrame(
            np.random.default_rng(2).standard_normal((10, 5)), columns=rng
        )
        data: Dict[Any, Dict[Any, float]] = {}
        for col in df.columns:
            for row in df.index:
                data.setdefault(col, {})[row] = df._get_value(row, col)
        result: DataFrame = DataFrame(data, columns=rng)
        tm.assert_frame_equal(result, df)
        data = {}
        for col in df.columns:
            for row in df.index:
                data.setdefault(row, {})[col] = df._get_value(row, col)
        result = DataFrame(data, index=rng).T
        tm.assert_frame_equal(result, df)

    def _check_basic_constructor(
        self, empty: Callable[[Tuple[int, ...], ...], Union[np.ndarray, Any]]
    ) -> None:
        mat: np.ndarray = empty((2, 3), dtype=float)
        frame: DataFrame = DataFrame(mat, columns=['A', 'B', 'C'], index=[1, 2])
        assert len(frame.index) == 2
        assert len(frame.columns) == 3
        frame = DataFrame(empty((3,), dtype=float), columns=['A'], index=[1, 2, 3])
        assert len(frame.index) == 3
        assert len(frame.columns) == 1
        if empty is not np.ones:
            msg: str = 'Cannot convert non-finite values \\(NA or inf\\) to integer'
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
            DataFrame(empty((3, 3, 3), dtype=float), columns=['A', 'B', 'C'], index=[1])
        frame = DataFrame(mat)
        tm.assert_index_equal(frame.index, Index(range(2)), exact=True)
        tm.assert_index_equal(frame.columns, Index(range(3)), exact=True)
        frame = DataFrame(mat, index=[1, 2])
        tm.assert_index_equal(frame.columns, Index(range(3)), exact=True)
        frame = DataFrame(mat, columns=['A', 'B', 'C'])
        tm.assert_index_equal(frame.index, Index(range(2)), exact=True)
        frame = DataFrame(empty((0, 3), dtype=float))
        assert len(frame.index) == 0
        frame = DataFrame(empty((3, 0), dtype=float))
        assert len(frame.columns) == 0

    def test_constructor_ndarray(self) -> None:
        self._check_basic_constructor(np.ones)
        frame: DataFrame = DataFrame(['foo', 'bar'], index=[0, 1], columns=['A'])
        assert len(frame) == 2

    def test_constructor_maskedarray(self) -> None:
        self._check_basic_constructor(ma.masked_all)
        mat: np.ma.MaskedArray = ma.masked_all((2, 3), dtype=float)
        mat[0, 0] = 1.0
        mat[1, 2] = 2.0
        frame: DataFrame = DataFrame(mat, columns=['A', 'B', 'C'], index=[1, 2])
        assert 1.0 == frame['A'][1]
        assert 2.0 == frame['C'][2]
        mat = ma.masked_all((2, 3), dtype=float)
        frame = DataFrame(mat, columns=['A', 'B', 'C'], index=[1, 2])
        assert np.all(~np.asarray(frame == frame))

    @pytest.mark.filterwarnings('ignore:elementwise comparison failed:DeprecationWarning')
    def test_constructor_maskedarray_nonfloat(self) -> None:
        mat: np.ma.MaskedArray = ma.masked_all((2, 3), dtype=int)
        frame: DataFrame = DataFrame(mat, columns=['A', 'B', 'C'], index=[1, 2])
        assert len(frame.index) == 2
        assert len(frame.columns) == 3
        assert np.all(~np.asarray(frame == frame))
        frame = DataFrame(mat, columns=['A', 'B', 'C'], index=[1, 2], dtype=np.float64)
        assert frame.values.dtype == np.float64
        mat2: np.ma.MaskedArray = ma.copy(mat)
        mat2[0, 0] = 1
        mat2[1, 2] = 2
        frame = DataFrame(mat2, columns=['A', 'B', 'C'], index=[1, 2])
        assert 1 == frame['A'].astype('i8')[1]
        assert 2 == frame['C'].astype('i8')[2]
        mat = ma.masked_all((2, 3), dtype='M8[ns]')
        frame = DataFrame(mat, columns=['A', 'B', 'C'], index=[1, 2])
        assert len(frame.index) == 2
        assert len(frame.columns) == 3
        assert isna(frame).values.all()
        msg: str = 'datetime64\\[ns\\] values and dtype=int64 is not supported'
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
        mat_hard: np.ma.MaskedArray = ma.masked_all((2, 2), dtype=float).harden_mask()
        result: DataFrame = DataFrame(mat_hard, columns=['A', 'B'], index=[1, 2])
        expected: DataFrame = DataFrame(
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
        data: np.ma.MaskedArray = np.ma.array(
            np.ma.zeros(5, dtype=[('date', '<f8'), ('price', '<f8')]),
            mask=[False] * 5,
        )
        data = data.view(mrecords.mrecarray)
        with pytest.raises(TypeError, match='Pass \\{name: data\\[name\\]'):
            DataFrame(data, dtype=int)

    def test_constructor_corner_shape(self) -> None:
        df: DataFrame = DataFrame(index=[])
        assert df.values.shape == (0, 0)

    @pytest.mark.parametrize(
        'copy',
        [False, True],
    )
    def test_dict_nocopy(
        self,
        copy: bool,
        any_numeric_ea_dtype: Any,
        any_numpy_dtype: Any,
    ) -> None:
        a: np.ndarray = np.array([1, 2], dtype=any_numpy_dtype)
        b: np.ndarray = np.array([3, 4], dtype=any_numpy_dtype)
        if b.dtype.kind in ['S', 'U']:
            pytest.skip(f'{b.dtype} get cast, making the checks below more cumbersome')
        c: Any = pd.array([1, 2], dtype=any_numeric_ea_dtype)
        c_orig: Any = c.copy()
        df: DataFrame = DataFrame({'a': a, 'b': b, 'c': c}, copy=copy)

        def get_base(obj: Any) -> Any:
            if isinstance(obj, np.ndarray):
                return obj.base
            elif isinstance(obj.dtype, np.dtype):
                return obj._ndarray.base
            else:
                raise TypeError

        def check_views(c_only: bool = False) -> None:
            assert sum(
                (x.values is c for x in df._mgr.blocks)
            ) == 1
            if c_only:
                return
            assert sum(
                (get_base(x.values) is a for x in df._mgr.blocks if isinstance(x.values.dtype, np.dtype))
            ) == 1
            assert sum(
                (get_base(x.values) is b for x in df._mgr.blocks if isinstance(x.values.dtype, np.dtype))
            ) == 1

        if not copy:
            check_views()
        should_raise: bool = not lib.is_np_dtype(df.dtypes.iloc[0], 'fciuO')
        if should_raise:
            with pytest.raises(TypeError, match='Invalid value'):
                df.iloc[0, 0] = 0
                df.iloc[0, 1] = 0
            return
        else:
            df.iloc[0, 0] = 0
            df.iloc[0, 1] = 0
        if not copy:
            check_views(True)
        df.iloc[:, 2] = pd.array([45, 46], dtype=c.dtype)
        assert df.dtypes.iloc[2] == c.dtype
        if copy:
            if a.dtype.kind == 'M':
                assert a[0] == a.dtype.type(1, 'ns')
                assert b[0] == b.dtype.type(3, 'ns')
            else:
                assert a[0] == a.dtype.type(1)
                assert b[0] == b.dtype.type(3)
            assert c[0] == c_orig[0]

    def test_construct_from_dict_ea_series(self) -> None:
        ser: Series = Series([1, 2, 3], dtype='Int64')
        df: DataFrame = DataFrame({'a': ser})
        assert not np.shares_memory(ser.values._data, df['a'].values._data)

    def test_from_series_with_name_with_columns(self) -> None:
        result: DataFrame = DataFrame(Series(1, name='foo'), columns=['bar'])
        expected: DataFrame = DataFrame(columns=['bar'])
        tm.assert_frame_equal(result, expected)

    def test_nested_list_columns(self) -> None:
        result: DataFrame = DataFrame(
            [[1, 2, 3], [4, 5, 6]], columns=[['A', 'A', 'A'], ['a', 'b', 'c']]
        )
        expected: DataFrame = DataFrame(
            [[1, 2, 3], [4, 5, 6]],
            columns=MultiIndex.from_tuples([('A', 'a'), ('A', 'b'), ('A', 'c')]),
        )
        tm.assert_frame_equal(result, expected)

    def test_from_2d_ndarray_with_dtype(self) -> None:
        array_dim2: np.ndarray = np.arange(10).reshape((5, 2))
        df: DataFrame = DataFrame(array_dim2, dtype='datetime64[ns, UTC]')
        expected: DataFrame = DataFrame(array_dim2).astype('datetime64[ns, UTC]')
        tm.assert_frame_equal(df, expected)

    def test_constructor_mixed_dict_nonseries(self) -> None:
        data: Dict[str, Any] = {'A': {'foo': 1, 'bar': 2, 'baz': 3}, 'B': Series([4, 3, 2, 1], index=['bar', 'qux', 'baz', 'foo'])}
        result: DataFrame = DataFrame(data)
        tm.assert_frame_equal(result, DataFrame({'A': [1, 2, 3], 'B': [4, 3, 2]}))

    def test_constructor_mixed_type_rows(self) -> None:
        data: List[Union[List[Any], Tuple[Any, Any]]] = [[1, 2], (3, 4)]
        result: DataFrame = DataFrame(data)
        expected: DataFrame = DataFrame([[1, 2], [3, 4]])
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
        lists: List[List[Any]],
    ) -> None:
        result: DataFrame = DataFrame(tuples)
        expected: DataFrame = DataFrame(lists)
        tm.assert_frame_equal(result, expected)

    def test_constructor_list_of_tuples(self) -> None:
        data: Dict[str, Tuple[int, int]] = {'A': (1, 2), 'B': (3, 4)}
        result: DataFrame = DataFrame({'A': [(1, 2), (3, 4)]})
        expected: DataFrame = DataFrame({'A': Series([(1, 2), (3, 4)])})
        tm.assert_frame_equal(result, expected)

    def test_constructor_list_of_namedtuples(self) -> None:
        named_tuple = namedtuple('Pandas', list('ab'))
        tuples: List[Any] = [named_tuple(1, 3), named_tuple(2, 4)]
        expected: DataFrame = DataFrame({'a': [1, 2], 'b': [3, 4]})
        result: DataFrame = DataFrame(tuples)
        tm.assert_frame_equal(result, expected)
        expected = DataFrame({'y': [1, 2], 'z': [3, 4]})
        result = DataFrame(tuples, columns=['y', 'z'])
        tm.assert_frame_equal(result, expected)

    def test_constructor_list_of_dataclasses(self) -> None:
        Point = make_dataclass('Point', [('x', int), ('y', int)])
        data: List[Any] = [Point(0, 3), Point(1, 3)]
        expected: DataFrame = DataFrame({'x': [0, 1], 'y': [3, 3]})
        result: DataFrame = DataFrame(data)
        tm.assert_frame_equal(result, expected)

    def test_constructor_list_of_dataclasses_with_varying_types(self) -> None:
        Point = make_dataclass('Point', [('x', int), ('y', int)])
        HLine = make_dataclass('HLine', [('x0', int), ('x1', int), ('y', int)])
        data: List[Any] = [Point(0, 3), HLine(1, 3, 3)]
        expected: DataFrame = DataFrame(
            {'x': [0, np.nan], 'y': [3, 3], 'x0': [np.nan, 1], 'x1': [np.nan, 3]}
        )
        result: DataFrame = DataFrame(data)
        tm.assert_frame_equal(result, expected)

    def test_constructor_list_of_dataclasses_error_thrown(self) -> None:
        Point = make_dataclass('Point', [('x', int), ('y', int)])
        msg: str = 'asdict() should be called on dataclass instances'
        with pytest.raises(TypeError, match=re.escape(msg)):
            DataFrame([Point(0, 0), {'x': 1, 'y': 0}])

    def test_constructor_list_of_dict_order(self) -> None:
        data: List[Dict[str, Any]] = [
            {'First': 1, 'Second': 4, 'Third': 7, 'Fourth': 10},
            {'Second': 5, 'First': 2, 'Fourth': 11, 'Third': 8},
            {'Second': 6, 'First': 3, 'Fourth': 12, 'Third': 9, 'YYY': 14, 'XXX': 13},
        ]
        expected: DataFrame = DataFrame(
            {'First': [1, 2, 3], 'Second': [4, 5, 6], 'Third': [7, 8, 9], 'Fourth': [10, 11, 12], 'YYY': [None, None, 14], 'XXX': [None, None, 13]}
        )
        result: DataFrame = DataFrame(data)
        tm.assert_frame_equal(result, expected)

    def test_constructor_Series_named(self) -> None:
        a: Series = Series([1, 2, 3], name=0)
        df: DataFrame = DataFrame(a)
        assert df.columns[0] == 0
        tm.assert_index_equal(df.index, a.index)
        arr: np.ndarray = np.random.default_rng(2).standard_normal(10)
        s: Series = Series(arr, name='x')
        df = DataFrame(s)
        expected: Series = Series(arr, dtype=np.float64, name='x')
        tm.assert_series_equal(df['x'], expected)
        s = Series(arr, index=range(3, 13))
        df = DataFrame(s)
        expected = Series(arr, index=range(3, 13), name='x')
        tm.assert_series_equal(df[0], Series(expected, name=0))
        msg: str = 'Shape of passed values is \\(10, 1\\), indices imply \\(10, 2\\)'
        with pytest.raises(ValueError, match=msg):
            DataFrame(s, columns=[1, 2])
        a = Series([], name='x', dtype=object)
        df = DataFrame(a)
        assert df.columns[0] == 'x'
        s1: Series = Series(arr, name='x')
        df = DataFrame([s1, arr]).T
        expected: DataFrame = DataFrame({'x': s1, 'Unnamed 0': arr}, columns=['x', 'Unnamed 0'])
        tm.assert_series_equal(df['x'], expected['x'])
        df = DataFrame([arr, s1]).T
        expected = DataFrame({'1': s1, '0': arr}, columns=range(2))
        tm.assert_frame_equal(df, expected)

    def test_constructor_Series_named_and_columns(self) -> None:
        s0: Series = Series(range(5), name=0)
        s1: Series = Series(range(5), name=1)
        tm.assert_frame_equal(DataFrame(s0, columns=[0]), s0.to_frame())
        tm.assert_frame_equal(DataFrame(s1, columns=[1]), s1.to_frame())
        assert DataFrame(s0, columns=[1]).empty
        assert DataFrame(s1, columns=[0]).empty

    def test_constructor_Series_differently_indexed(self) -> None:
        s1: Series = Series([1, 2, 3], index=['a', 'b', 'c'], name='x')
        s2: Series = Series([1, 2, 3], index=['a', 'b', 'c'])
        other_index: Index = Index(['a', 'b'])
        df1: DataFrame = DataFrame(s1, index=other_index)
        exp1: DataFrame = DataFrame(s1.reindex(other_index))
        assert df1.columns[0] == 'x'
        tm.assert_frame_equal(df1, exp1)
        df2: DataFrame = DataFrame(s2, index=other_index)
        exp2: DataFrame = DataFrame(s2.reindex(other_index))
        assert df2.columns[0] == 0
        tm.assert_index_equal(df2.index, other_index)
        tm.assert_frame_equal(df2, exp2)

    @pytest.mark.parametrize(
        'name_in1,name_in2,name_in3,name_out',
        [
            ('idx', 'idx', 'idx', 'idx'),
            ('idx', 'idx', None, None),
            ('idx', None, None, None),
            ('idx1', 'idx2', None, None),
            ('idx1', 'idx1', 'idx2', None),
            ('idx1', 'idx2', 'idx3', None),
            (None, None, None, None),
        ],
    )
    def test_constructor_index_names(
        self,
        name_in1: Optional[str],
        name_in2: Optional[str],
        name_in3: Optional[str],
        name_out: Optional[str],
    ) -> None:
        indices: List[Index] = [
            Index(['a', 'a', 'b', 'b'], name=name_in1),
            Index(['x', 'y', 'x', 'y'], name=name_in2),
        ]
        multi: DataFrame = DataFrame(
            np.random.default_rng(2).standard_normal((4, 4)),
            index=[np.array(x) for x in indices],
        )
        assert isinstance(multi.index, MultiIndex)
        assert not isinstance(multi.columns, MultiIndex)
        multi = DataFrame(
            np.random.default_rng(2).standard_normal((4, 4)),
            columns=['A', 'A', 'A', 'A'],
        )
        assert isinstance(multi.columns, MultiIndex)

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
        td_as_int: List[int] = [1, 2, 3, 4]
        data: Dict[int, Any] = {i: klass(s) for i, s in enumerate(td_as_int)}
        expected: DataFrame = DataFrame(
            [
                {0: 0, 1: None, 2: None, 3: None},
                {0: None, 1: 2, 2: None, 3: None},
                {0: None, 1: None, 2: 4, 3: None},
                {0: None, 1: None, 2: None, 3: 6},
            ],
            index=[Timedelta(td, 'D') for td in td_as_int],
        )
        result: DataFrame = DataFrame(data)
        tm.assert_frame_equal(result, expected)

    def test_constructor_period_dict(self) -> None:
        a: pd.PeriodIndex = pd.PeriodIndex(['2012-01', 'NaT', '2012-04'], freq='M')
        b: pd.PeriodIndex = pd.PeriodIndex(['2012-02-01', '2012-03-01', 'NaT'], freq='D')
        df: DataFrame = DataFrame({'a': a, 'b': b})
        assert df['a'].dtype == a.dtype
        assert df['b'].dtype == b.dtype
        df = DataFrame({'a': a.astype(object).tolist(), 'b': b.astype(object).tolist()})
        assert df['a'].dtype == a.dtype
        assert df['b'].dtype == b.dtype

    def test_constructor_dict_extension_scalar(
        self, ea_scalar_and_dtype: Tuple[Any, Any]
    ) -> None:
        ea_scalar, ea_dtype = ea_scalar_and_dtype
        df: DataFrame = DataFrame({'a': ea_scalar}, index=[0])
        assert df['a'].dtype == ea_dtype
        expected: DataFrame = DataFrame(index=[0], columns=['a'], data=ea_scalar)
        tm.assert_frame_equal(df, expected)

    @pytest.mark.parametrize(
        'data,dtype',
        [
            (Period('2020-01'), PeriodDtype('M')),
            (Interval(left=0, right=5), IntervalDtype('int64', 'right')),
            (
                Timestamp('2011-01-01', tz='US/Eastern'),
                DatetimeTZDtype(unit='s', tz='US/Eastern'),
            ),
        ],
    )
    def test_constructor_extension_scalar_data(
        self, data: Any, dtype: Any
    ) -> None:
        df: DataFrame = DataFrame(index=range(2), columns=['a', 'b'], data=data)
        assert df['a'].dtype == dtype
        assert df['b'].dtype == dtype
        arr: Any = pd.array([data] * 2, dtype=dtype)
        expected: DataFrame = DataFrame({'a': arr, 'b': arr})
        tm.assert_frame_equal(df, expected)

    def test_nested_dict_frame_constructor(self) -> None:
        rng: pd.period_range = pd.period_range('1/1/2000', periods=5)
        df: DataFrame = DataFrame(
            np.random.default_rng(2).standard_normal((10, 5)), columns=rng
        )
        data: Dict[Any, Dict[Any, float]] = {}
        for col in df.columns:
            for row in df.index:
                data.setdefault(col, {})[row] = df._get_value(row, col)
        result: DataFrame = DataFrame(data, columns=rng)
        tm.assert_frame_equal(result, df)
        data = {}
        for col in df.columns:
            for row in df.index:
                data.setdefault(row, {})[col] = df._get_value(row, col)
        result = DataFrame(data, index=rng).T
        tm.assert_frame_equal(result, df)

    def test_ctor_with_datetimes(self, using_infer_string: bool) -> None:
        dtype = pd.StringDtype() if using_infer_string else object
        df: DataFrame = DataFrame({'A': [1, 2, 3], 'B': 'foo', 'C': 'bar'}, dtype=dtype)
        assert df.dtypes['A'] == np.float64
        assert df.dtypes['B'] == dtype
        assert df.dtypes['C'] == dtype

    def test_frame_string_inference(self, using_infer_string: bool) -> None:
        dtype = pd.StringDtype(na_value=np.nan)
        expected: DataFrame = DataFrame({'a': ['a', 'b']}, dtype=dtype, columns=Index(['a'], dtype=dtype))
        with pd.option_context('future.infer_string', True):
            df: DataFrame = DataFrame({'a': ['a', 'b']})
        tm.assert_frame_equal(df, expected)
        expected = DataFrame({'a': ['a', 'b']}, dtype=dtype, columns=Index(['a'], dtype=dtype), index=Index(['x', 'y'], dtype=dtype))
        with pd.option_context('future.infer_string', True):
            df = DataFrame({'a': ['a', 'b']}, index=['x', 'y'])
        tm.assert_frame_equal(df, expected)
        expected = DataFrame({'a': ['a', 1]}, dtype='object', columns=Index(['a'], dtype=dtype))
        with pd.option_context('future.infer_string', True):
            df: DataFrame = DataFrame({'a': ['a', 1]})
        tm.assert_frame_equal(df, expected)
        expected = DataFrame({'a': ['a', 'b']}, dtype='object', columns=Index(['a'], dtype=dtype))
        with pd.option_context('future.infer_string', True):
            df = DataFrame({'a': ['a', 'b']}, dtype='object')
        tm.assert_frame_equal(df, expected)

    def test_frame_string_inference_array_string_dtype(
        self, using_infer_string: bool
    ) -> None:
        dtype = pd.StringDtype(na_value=np.nan)
        expected: DataFrame = DataFrame({'a': ['a', 'b']}, dtype=dtype, columns=Index(['a'], dtype=dtype))
        with pd.option_context('future.infer_string', True):
            df: DataFrame = DataFrame({'a': np.array(['a', 'b'])})
        tm.assert_frame_equal(df, expected)
        expected = DataFrame({'0': ['a', 'b'], '1': ['c', 'd']}, dtype=dtype)
        with pd.option_context('future.infer_string', True):
            df: DataFrame = DataFrame(np.array([['a', 'c'], ['b', 'd']]))
        tm.assert_frame_equal(df, expected)
        expected = DataFrame(
            {'a': ['a', 'b'], 'b': ['c', 'd']},
            dtype=dtype,
            columns=Index(['a', 'b'], dtype=dtype),
        )
        with pd.option_context('future.infer_string', True):
            df = DataFrame(np.array([['a', 'c'], ['b', 'd']]), columns=['a', 'b'])
        tm.assert_frame_equal(df, expected)

    def test_frame_string_inference_block_dim(self) -> None:
        with pd.option_context('future.infer_string', True):
            df: DataFrame = DataFrame(np.array([['hello', 'goodbye'], ['hello', 'Hello']]))
        assert df._mgr.blocks[0].ndim == 2

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
        td_as_int: List[int] = [1, 2, 3, 4]
        data: Dict[int, Any] = {i: klass(s) for i, s in enumerate(td_as_int)}
        expected: DataFrame = DataFrame(
            [
                {0: 0, 1: None, 2: None, 3: None},
                {0: None, 1: 2, 2: None, 3: None},
                {0: None, 1: None, 2: 4, 3: None},
                {0: None, 1: None, 2: None, 3: 6},
            ],
            index=[Timedelta(td, 'D') for td in td_as_int],
        )
        result: DataFrame = DataFrame(data)
        tm.assert_frame_equal(result, expected)

    def test_constructor_period_dict(self) -> None:
        a: pd.PeriodIndex = pd.PeriodIndex(['2012-01', 'NaT', '2012-04'], freq='M')
        b: pd.PeriodIndex = pd.PeriodIndex(['2012-02-01', '2012-03-01', 'NaT'], freq='D')
        df: DataFrame = DataFrame({'a': a, 'b': b})
        assert df['a'].dtype == a.dtype
        assert df['b'].dtype == b.dtype
        df = DataFrame({'a': a.astype(object).tolist(), 'b': b.astype(object).tolist()})
        assert df['a'].dtype == a.dtype
        assert df['b'].dtype == b.dtype

    def test_nested_dict_frame_constructor(self) -> None:
        rng: pd.period_range = pd.period_range('1/1/2000', periods=5)
        df: DataFrame = DataFrame(
            np.random.default_rng(2).standard_normal((10, 5)), columns=rng
        )
        data: Dict[Any, Dict[Any, float]] = {}
        for col in df.columns:
            for row in df.index:
                data.setdefault(col, {})[row] = df._get_value(row, col)
        result: DataFrame = DataFrame(data, columns=rng)
        tm.assert_frame_equal(result, df)
        data = {}
        for col in df.columns:
            for row in df.index:
                data.setdefault(row, {})[col] = df._get_value(row, col)
        result = DataFrame(data, index=rng).T
        tm.assert_frame_equal(result, df)

    def test_constructor_dict_extension_scalar(
        self, ea_scalar_and_dtype: Tuple[Any, Any]
    ) -> None:
        ea_scalar, ea_dtype = ea_scalar_and_dtype
        df: DataFrame = DataFrame({'a': ea_scalar}, index=[0])
        assert df['a'].dtype == ea_dtype
        expected: DataFrame = DataFrame(index=[0], columns=['a'], data=ea_scalar)
        tm.assert_frame_equal(df, expected)

    def test_from_series_with_name_with_columns(self) -> None:
        result: DataFrame = DataFrame(Series(1, name='foo'), columns=['bar'])
        expected: DataFrame = DataFrame(columns=['bar'])
        tm.assert_frame_equal(result, expected)

    def test_nested_list_columns(self) -> None:
        result: DataFrame = DataFrame(
            [[1, 2, 3], [4, 5, 6]],
            columns=[['A', 'A', 'A'], ['a', 'b', 'c']],
        )
        expected: DataFrame = DataFrame(
            [[1, 2, 3], [4, 5, 6]],
            columns=MultiIndex.from_tuples([('A', 'a'), ('A', 'b'), ('A', 'c')]),
        )
        tm.assert_frame_equal(result, expected)

    def test_from_2d_ndarray_with_dtype(self) -> None:
        array_dim2: np.ndarray = np.arange(10).reshape((5, 2))
        df: DataFrame = DataFrame(array_dim2, dtype='datetime64[ns, UTC]')
        expected: DataFrame = DataFrame(array_dim2).astype('datetime64[ns, UTC]')
        tm.assert_frame_equal(df, expected)

    def test_constructor_mixed_dict_nonseries(self) -> None:
        data: Dict[str, Any] = {'A': {'foo': 1, 'bar': 2, 'baz': 3}, 'B': Series([4, 3, 2, 1], index=['bar', 'qux', 'baz', 'foo'])}
        result: DataFrame = DataFrame(data)
        tm.assert_frame_equal(result, DataFrame({'A': [1, 2, 3], 'B': [4, 3, 2, 1]}))

    def test_constructor_mixed_type_rows(self) -> None:
        data: List[Union[List[Any], Tuple[Any, Any]]] = [[1, 2], (3, 4)]
        result: DataFrame = DataFrame(data)
        expected: DataFrame = DataFrame([[1, 2], [3, 4]])
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
        lists: List[List[Any]],
    ) -> None:
        result: DataFrame = DataFrame(tuples)
        expected: DataFrame = DataFrame(lists)
        tm.assert_frame_equal(result, expected)

    def test_constructor_list_of_tuples(self) -> None:
        data: Dict[str, Tuple[int, int]] = {'A': (1, 2), 'B': (3, 4)}
        result: DataFrame = DataFrame({'A': [(1, 2), (3, 4)]})
        expected: DataFrame = DataFrame({'A': Series([(1, 2), (3, 4)])})
        tm.assert_frame_equal(result, expected)

    def test_constructor_list_of_namedtuples(self) -> None:
        named_tuple = namedtuple('Pandas', list('ab'))
        tuples: List[Any] = [named_tuple(1, 3), named_tuple(2, 4)]
        expected: DataFrame = DataFrame({'a': [1, 2], 'b': [3, 4]})
        result: DataFrame = DataFrame(tuples)
        tm.assert_frame_equal(result, expected)
        expected = DataFrame({'y': [1, 2], 'z': [3, 4]})
        result = DataFrame(tuples, columns=['y', 'z'])
        tm.assert_frame_equal(result, expected)

    def test_constructor_list_of_dataclasses(self) -> None:
        Point = make_dataclass('Point', [('x', int), ('y', int)])
        data: List[Any] = [Point(0, 3), Point(1, 3)]
        expected: DataFrame = DataFrame({'x': [0, 1], 'y': [3, 3]})
        result: DataFrame = DataFrame(data)
        tm.assert_frame_equal(result, expected)

    def test_constructor_list_of_dataclasses_with_varying_types(self) -> None:
        Point = make_dataclass('Point', [('x', int), ('y', int)])
        HLine = make_dataclass('HLine', [('x0', int), ('x1', int), ('y', int)])
        data: List[Any] = [Point(0, 3), HLine(1, 3, 3)]
        expected: DataFrame = DataFrame(
            {'x': [0, np.nan], 'y': [3, 3], 'x0': [np.nan, 1], 'x1': [np.nan, 3]}
        )
        result: DataFrame = DataFrame(data)
        tm.assert_frame_equal(result, expected)

    def test_constructor_list_of_dataclasses_error_thrown(self) -> None:
        Point = make_dataclass('Point', [('x', int), ('y', int)])
        msg: str = 'asdict() should be called on dataclass instances'
        with pytest.raises(TypeError, match=re.escape(msg)):
            DataFrame([Point(0, 0), {'x': 1, 'y': 0}])

    def test_constructor_list_of_dict_order(self) -> None:
        data: List[Dict[str, Any]] = [
            {'First': 1, 'Second': 4, 'Third': 7, 'Fourth': 10},
            {'Second': 5, 'First': 2, 'Fourth': 11, 'Third': 8},
            {'Second': 6, 'First': 3, 'Fourth': 12, 'Third': 9, 'YYY': 14, 'XXX': 13},
        ]
        expected: DataFrame = DataFrame(
            {'First': [1, 2, 3], 'Second': [4, 5, 6], 'Third': [7, 8, 9], 'Fourth': [10, 11, 12], 'YYY': [None, None, 14], 'XXX': [None, None, 13]}
        )
        result: DataFrame = DataFrame(data)
        tm.assert_frame_equal(result, expected)

    def test_constructor_Series_named(self) -> None:
        a: Series = Series([1, 2, 3], name=0)
        df: DataFrame = DataFrame(a)
        assert df.columns[0] == 0
        tm.assert_index_equal(df.index, a.index)
        arr: np.ndarray = np.random.default_rng(2).standard_normal(10)
        s: Series = Series(arr, name='x')
        df = DataFrame(s)
        expected: Series = Series(arr, dtype=np.float64, name='x')
        tm.assert_series_equal(df['x'], expected)
        s = Series(arr, index=range(3, 13))
        df = DataFrame(s)
        expected = Series(arr, index=range(3, 13), name='x')
        tm.assert_series_equal(df[0], Series(expected, name=0))
        msg: str = 'Shape of passed values is \\(10, 1\\), indices imply \\(10, 2\\)'
        with pytest.raises(ValueError, match=msg):
            DataFrame(s, columns=[1, 2])
        a = Series([], name='x', dtype=object)
        df = DataFrame(a)
        assert df.columns[0] == 'x'
        s1: Series = Series(arr, name='x')
        df = DataFrame([s1, arr]).T
        expected: DataFrame = DataFrame({'x': s1, 'Unnamed 0': arr}, columns=['x', 'Unnamed 0'])
        tm.assert_series_equal(df['x'], expected['x'])
        df = DataFrame([arr, s1]).T
        expected = DataFrame({'1': s1, '0': arr}, columns=range(2))
        tm.assert_frame_equal(df, expected)

    def test_constructor_Series_named_and_columns(self) -> None:
        s0: Series = Series(range(5), name=0)
        s1: Series = Series(range(5), name=1)
        tm.assert_frame_equal(DataFrame(s0, columns=[0]), s0.to_frame())
        tm.assert_frame_equal(DataFrame(s1, columns=[1]), s1.to_frame())
        assert DataFrame(s0, columns=[1]).empty
        assert DataFrame(s1, columns=[0]).empty

    def test_constructor_Series_differently_indexed(self) -> None:
        s1: Series = Series([1, 2, 3], index=['a', 'b', 'c'], name='x')
        s2: Series = Series([1, 2, 3], index=['a', 'b', 'c'])
        other_index: Index = Index(['a', 'b'])
        df1: DataFrame = DataFrame(s1, index=other_index)
        exp1: DataFrame = DataFrame(s1.reindex(other_index))
        assert df1.columns[0] == 'x'
        tm.assert_frame_equal(df1, exp1)
        df2: DataFrame = DataFrame(s2, index=other_index)
        exp2: DataFrame = DataFrame(s2.reindex(other_index))
        assert df2.columns[0] == 0
        tm.assert_index_equal(df2.index, other_index)
        tm.assert_frame_equal(df2, exp2)

    @pytest.mark.parametrize(
        'name_in1,name_in2,name_in3,name_out',
        [
            ('idx', 'idx', 'idx', 'idx'),
            ('idx', 'idx', None, None),
            ('idx', None, None, None),
            ('idx1', 'idx2', None, None),
            ('idx1', 'idx1', 'idx2', None),
            ('idx1', 'idx2', 'idx3', None),
            (None, None, None, None),
        ],
    )
    def test_constructor_index_names(
        self,
        name_in1: Optional[str],
        name_in2: Optional[str],
        name_in3: Optional[str],
        name_out: Optional[str],
    ) -> None:
        indices: List[Index] = [
            Index(['a', 'a', 'b', 'b'], name=name_in1),
            Index(['x', 'y', 'x', 'y'], name=name_in2),
        ]
        multi: DataFrame = DataFrame(
            np.random.default_rng(2).standard_normal((4, 4)),
            index=[np.array(x) for x in indices],
        )
        assert isinstance(multi.index, MultiIndex)
        assert not isinstance(multi.columns, MultiIndex)
        multi = DataFrame(
            np.random.default_rng(2).standard_normal((4, 4)),
            columns=['A', 'A', 'A', 'A'],
        )
        assert isinstance(multi.columns, MultiIndex)

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
        td_as_int: List[int] = [1, 2, 3, 4]
        data: Dict[int, Any] = {i: klass(s) for i, s in enumerate(td_as_int)}
        expected: DataFrame = DataFrame(
            [
                {0: 0, 1: None, 2: None, 3: None},
                {0: None, 1: 2, 2: None, 3: None},
                {0: None, 1: None, 2: 4, 3: None},
                {0: None, 1: None, 2: None, 3: 6},
            ],
            index=[Timedelta(td, 'D') for td in td_as_int],
        )
        result: DataFrame = DataFrame(data)
        tm.assert_frame_equal(result, expected)

    def test_constructor_period_dict(self) -> None:
        a: pd.PeriodIndex = pd.PeriodIndex(['2012-01', 'NaT', '2012-04'], freq='M')
        b: pd.PeriodIndex = pd.PeriodIndex(['2012-02-01', '2012-03-01', 'NaT'], freq='D')
        df: DataFrame = DataFrame({'a': a, 'b': b})
        assert df['a'].dtype == a.dtype
        assert df['b'].dtype == b.dtype
        df = DataFrame({'a': a.astype(object).tolist(), 'b': b.astype(object).tolist()})
        assert df['a'].dtype == a.dtype
        assert df['b'].dtype == b.dtype

    def test_nested_dict_frame_constructor(self) -> None:
        rng: pd.period_range = pd.period_range('1/1/2000', periods=5)
        df: DataFrame = DataFrame(
            np.random.default_rng(2).standard_normal((10, 5)), columns=rng
        )
        data: Dict[Any, Dict[Any, float]] = {}
        for col in df.columns:
            for row in df.index:
                data.setdefault(col, {})[row] = df._get_value(row, col)
        result: DataFrame = DataFrame(data, columns=rng)
        tm.assert_frame_equal(result, df)
        data = {}
        for col in df.columns:
            for row in df.index:
                data.setdefault(row, {})[col] = df._get_value(row, col)
        result = DataFrame(data, index=rng).T
        tm.assert_frame_equal(result, df)

    def test_ctor_with_datetimes(self, using_infer_string: bool) -> None:
        dtype = pd.StringDtype() if using_infer_string else object
        df: DataFrame = DataFrame({'A': [1, 2, 3], 'B': 'foo', 'C': 'bar'}, dtype=dtype)
        assert df.dtypes['A'] == np.float64
        assert df.dtypes['B'] == dtype
        assert df.dtypes['C'] == dtype

    def test_frame_string_inference(self, using_infer_string: bool) -> None:
        dtype: Any = pd.StringDtype(na_value=np.nan)
        expected: DataFrame = DataFrame({'a': ['a', 'b']}, dtype=dtype, columns=Index(['a'], dtype=dtype))
        with pd.option_context('future.infer_string', True):
            df: DataFrame = DataFrame({'a': ['a', 'b']})
        tm.assert_frame_equal(df, expected)
        expected = DataFrame(
            {'a': ['a', 'b']},
            dtype=dtype,
            columns=Index(['a'], dtype=dtype),
            index=Index(['x', 'y'], dtype=dtype),
        )
        with pd.option_context('future.infer_string', True):
            df = DataFrame({'a': ['a', 'b']}, index=['x', 'y'])
        tm.assert_frame_equal(df, expected)
        expected = DataFrame(
            {'a': ['a', 1]},
            dtype='object',
            columns=Index(['a'], dtype=dtype),
        )
        with pd.option_context('future.infer_string', True):
            df: DataFrame = DataFrame({'a': ['a', 1]})
        tm.assert_frame_equal(df, expected)
        expected = DataFrame(
            {'a': ['a', 'b']},
            dtype='object',
            columns=Index(['a'], dtype=dtype),
        )
        with pd.option_context('future.infer_string', True):
            df = DataFrame({'a': ['a', 'b']}, dtype='object')
        tm.assert_frame_equal(df, expected)

    def test_frame_string_inference_array_string_dtype(
        self, using_infer_string: bool
    ) -> None:
        dtype: Any = pd.StringDtype(na_value=np.nan)
        expected: DataFrame = DataFrame({'a': ['a', 'b']}, dtype=dtype, columns=Index(['a'], dtype=dtype))
        with pd.option_context('future.infer_string', True):
            df: DataFrame = DataFrame({'a': np.array(['a', 'b'])})
        tm.assert_frame_equal(df, expected)
        expected = DataFrame(
            {'0': ['a', 'b'], '1': ['c', 'd']},
            dtype=dtype,
        )
        with pd.option_context('future.infer_string', True):
            df = DataFrame(np.array([['a', 'c'], ['b', 'd']]))
        tm.assert_frame_equal(df, expected)
        expected = DataFrame(
            {'a': ['a', 'b'], 'b': ['c', 'd']},
            dtype=dtype,
            columns=Index(['a', 'b'], dtype=dtype),
        )
        with pd.option_context('future.infer_string', True):
            df = DataFrame(np.array([['a', 'c'], ['b', 'd']]), columns=['a', 'b'])
        tm.assert_frame_equal(df, expected)

    def test_frame_string_inference_block_dim(self) -> None:
        with pd.option_context('future.infer_string', True):
            df: DataFrame = DataFrame(np.array([['hello', 'goodbye'], ['hello', 'Hello']]))
        assert df._mgr.blocks[0].ndim == 2

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
        td_as_int: List[int] = [1, 2, 3, 4]
        data: Dict[int, Any] = {i: klass(s) for i, s in enumerate(td_as_int)}
        expected: DataFrame = DataFrame(
            [
                {0: 0, 1: None, 2: None, 3: None},
                {0: None, 1: 2, 2: None, 3: None},
                {0: None, 1: None, 2: 4, 3: None},
                {0: None, 1: None, 2: None, 3: 6},
            ],
            index=[Timedelta(td, 'D') for td in td_as_int],
        )
        result: DataFrame = DataFrame(data)
        tm.assert_frame_equal(result, expected)

    def test_constructor_period_dict(self) -> None:
        a: pd.PeriodIndex = pd.PeriodIndex(['2012-01', 'NaT', '2012-04'], freq='M')
        b: pd.PeriodIndex = pd.PeriodIndex(['2012-02-01', '2012-03-01', 'NaT'], freq='D')
        df: DataFrame = DataFrame({'a': a, 'b': b})
        assert df['a'].dtype == a.dtype
        assert df['b'].dtype == b.dtype
        df = DataFrame({'a': a.astype(object).tolist(), 'b': b.astype(object).tolist()})
        assert df['a'].dtype == a.dtype
        assert df['b'].dtype == b.dtype

    def test_nested_dict_frame_constructor(self) -> None:
        rng: pd.period_range = pd.period_range('1/1/2000', periods=5)
        df: DataFrame = DataFrame(
            np.random.default_rng(2).standard_normal((10, 5)), columns=rng
        )
        data: Dict[Any, Dict[Any, float]] = {}
        for col in df.columns:
            for row in df.index:
                data.setdefault(col, {})[row] = df._get_value(row, col)
        result: DataFrame = DataFrame(data, columns=rng)
        tm.assert_frame_equal(result, df)
        data = {}
        for col in df.columns:
            for row in df.index:
                data.setdefault(row, {})[col] = df._get_value(row, col)
        result = DataFrame(data, index=rng).T
        tm.assert_frame_equal(result, df)

    def test_constructor_dict_extension_scalar(
        self, ea_scalar_and_dtype: Tuple[Any, Any]
    ) -> None:
        ea_scalar, ea_dtype = ea_scalar_and_dtype
        df: DataFrame = DataFrame({'a': ea_scalar}, index=[0])
        assert df['a'].dtype == ea_dtype
        expected: DataFrame = DataFrame(index=[0], columns=['a'], data=ea_scalar)
        tm.assert_frame_equal(df, expected)

    @pytest.mark.parametrize(
        'data,dtype',
        [
            (Period('2020-01'), PeriodDtype('M')),
            (Interval(left=0, right=5), IntervalDtype('int64', 'right')),
            (
                Timestamp('2011-01-01', tz='US/Eastern'),
                DatetimeTZDtype(unit='s', tz='US/Eastern'),
            ),
        ],
    )
    def test_constructor_extension_scalar_data(
        self, data: Any, dtype: Any
    ) -> None:
        df: DataFrame = DataFrame(index=range(2), columns=['a', 'b'], data=data)
        assert df['a'].dtype == dtype
        assert df['b'].dtype == dtype
        arr: Any = pd.array([data] * 2, dtype=dtype)
        expected: DataFrame = DataFrame({'a': arr, 'b': arr})
        tm.assert_frame_equal(df, expected)

    def test_constructor_datetimes3(self) -> None:
        dt: Timestamp = Timestamp('2019-12-31 01:00:00-0500', tz='US/Eastern')
        df: DataFrame = DataFrame({'End Date': dt}, index=[0])
        assert df.iat[0, 0] == dt
        tm.assert_series_equal(
            df.dtypes,
            Series({'End Date': 'datetime64[us, US/Eastern]'}, dtype=object),
        )
        df = DataFrame([{'End Date': dt}])
        assert df.iat[0, 0] == dt
        tm.assert_series_equal(
            df.dtypes,
            Series({'End Date': 'datetime64[us, US/Eastern]'}, dtype=object),
        )

    @pytest.mark.parametrize(
        'tz',
        ['US/Eastern', 'dateutil/US/Eastern'],
    )
    def test_construction_preserves_tzaware_dtypes(self, tz: str) -> None:
        dr: pd.DatetimeIndex = date_range('2011/1/1', periods=3, tz=tz)
        df: DataFrame = DataFrame({'A': 'foo', 'B': dr}, index=dr)
        assert df['B'].dtype == DatetimeTZDtype('ns', pd.Timestamp(tz=tz).tz)

    def test_construct_with_two_categoricalindex_series(self) -> None:
        s1: Series = Series(
            [39, 6, 4], index=CategoricalIndex(['female', 'male', 'unknown'])
        )
        s2: Series = Series(
            [2, 152, 2, 242, 150],
            index=CategoricalIndex(['f', 'female', 'm', 'male', 'unknown']),
        )
        result: DataFrame = DataFrame([s1, s2])
        expected: DataFrame = DataFrame(
            np.array(
                [[39, 6, 4, np.nan, np.nan], [152.0, 242.0, 150.0, 2.0, 2.0]]
            ),
            columns=['female', 'male', 'unknown', 'f', 'm'],
        )
        tm.assert_frame_equal(result, expected)

    def test_from_series_with_name_with_columns(self) -> None:
        result: DataFrame = DataFrame(
            Series([1], name='foo'), columns=['bar']
        )
        expected: DataFrame = DataFrame(columns=['bar'])
        tm.assert_frame_equal(result, expected)

    def test_frame_string_inference_array_string_dtype(
        self, using_infer_string: bool
    ) -> None:
        dtype: Any = pd.StringDtype(na_value=np.nan)
        expected: DataFrame = DataFrame(
            {'a': ['a', 'b']},
            dtype=dtype,
            columns=Index(['a'], dtype=dtype),
        )
        with pd.option_context('future.infer_string', True):
            df: DataFrame = DataFrame({'a': np.array(['a', 'b'])})
        tm.assert_frame_equal(df, expected)
        expected = DataFrame(
            {'0': ['a', 'b'], '1': ['c', 'd']},
            dtype=dtype,
        )
        with pd.option_context('future.infer_string', True):
            df = DataFrame(np.array([['a', 'c'], ['b', 'd']]))
        tm.assert_frame_equal(df, expected)
        expected = DataFrame(
            {'a': ['a', 'b'], 'b': ['c', 'd']},
            dtype=dtype,
            columns=Index(['a', 'b'], dtype=dtype),
        )
        with pd.option_context('future.infer_string', True):
            df = DataFrame(np.array([['a', 'c'], ['b', 'd']]), columns=['a', 'b'])
        tm.assert_frame_equal(df, expected)

    def test_frame_string_inference_block_dim(self) -> None:
        with pd.option_context('future.infer_string', True):
            df: DataFrame = DataFrame(
                np.array([['hello', 'goodbye'], ['hello', 'Hello']])
            )
        assert df._mgr.blocks[0].ndim == 2

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
        td_as_int: List[int] = [1, 2, 3, 4]
        data: Dict[int, Any] = {i: klass(s) for i, s in enumerate(td_as_int)}
        expected: DataFrame = DataFrame(
            [
                {0: 0, 1: None, 2: None, 3: None},
                {0: None, 1: 2, 2: None, 3: None},
                {0: None, 1: None, 2: 4, 3: None},
                {0: None, 1: None, 2: None, 3: 6},
            ],
            index=[Timedelta(td, 'D') for td in td_as_int],
        )
        result: DataFrame = DataFrame(data)
        tm.assert_frame_equal(result, expected)

    def test_constructor_period_dict(self) -> None:
        a: pd.PeriodIndex = pd.PeriodIndex(['2012-01', 'NaT', '2012-04'], freq='M')
        b: pd.PeriodIndex = pd.PeriodIndex(['2012-02-01', '2012-03-01', 'NaT'], freq='D')
        df: DataFrame = DataFrame({'a': a, 'b': b})
        assert df['a'].dtype == a.dtype
        assert df['b'].dtype == b.dtype
        df = DataFrame({'a': a.astype(object).tolist(), 'b': b.astype(object).tolist()})
        assert df['a'].dtype == a.dtype
        assert df['b'].dtype == b.dtype

    def test_nested_dict_frame_constructor(self) -> None:
        rng: pd.period_range = pd.period_range('1/1/2000', periods=5)
        df: DataFrame = DataFrame(
            np.random.default_rng(2).standard_normal((10, 5)), columns=rng
        )
        data: Dict[Any, Dict[Any, float]] = {}
        for col in df.columns:
            for row in df.index:
                data.setdefault(col, {})[row] = df._get_value(row, col)
        result: DataFrame = DataFrame(data, columns=rng)
        tm.assert_frame_equal(result, df)
        data = {}
        for col in df.columns:
            for row in df.index:
                data.setdefault(row, {})[col] = df._get_value(row, col)
        result = DataFrame(data, index=rng).T
        tm.assert_frame_equal(result, df)

    def test_constructor_dtype_copy(self) -> None:
        orig_df: DataFrame = DataFrame({'col1': [1.0], 'col2': [2.0], 'col3': [3.0]})
        new_df: DataFrame = DataFrame(orig_df, dtype=float, copy=True)
        new_df['col1'] = 200.0
        assert orig_df['col1'][0] == 1.0

    def test_constructor_dtype_nocast_view_dataframe(self) -> None:
        df: DataFrame = DataFrame([[1, 2]])
        should_be_view: DataFrame = DataFrame(df, dtype=df[0].dtype)
        should_be_view.iloc[0, 0] = 99
        assert df.values[0, 0] == 1

    def test_constructor_dtype_nocast_view_2d_array(self) -> None:
        df: DataFrame = DataFrame([[1, 2], [3, 4]], dtype='int64')
        df2: DataFrame = DataFrame(df.values, dtype=df[0].dtype)
        assert df2._mgr.blocks[0].values.flags.c_contiguous

    def test_1d_object_array_does_not_copy(
        self, using_infer_string: bool
    ) -> None:
        arr: np.ndarray = np.array(['a', 'b'], dtype='object')
        df: DataFrame = DataFrame(arr, copy=False)
        if using_infer_string:
            if df[0].dtype.storage == 'pyarrow':
                pass
            else:
                assert np.shares_memory(df[0].to_numpy(), arr)
        else:
            assert np.shares_memory(df.values, arr)
        df = DataFrame(arr, dtype=object, copy=False)
        assert np.shares_memory(df.values, arr)

    def test_2d_object_array_does_not_copy(
        self, using_infer_string: bool
    ) -> None:
        arr: np.ndarray = np.array([['a', 'b'], ['c', 'd']], dtype='object')
        df: DataFrame = DataFrame(arr, copy=False)
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
        df: DataFrame = DataFrame([[1, '2'], [None, 'a']], dtype=object)
        assert df.loc[1, 0] is None
        assert df.loc[0, 1] == '2'

    def test_constructor_list_of_2d_raises(self) -> None:
        a: DataFrame = DataFrame()
        b: np.ndarray = np.empty((0, 0))
        with pytest.raises(ValueError, match='shape=\\(1, 0, 0\\)'):
            DataFrame([a])
        with pytest.raises(ValueError, match='shape=\\(1, 0, 0\\)'):
            DataFrame([b])
        a = DataFrame({'A': [1, 2]})
        with pytest.raises(ValueError, match='shape=\\(2, 2, 1\\)'):
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
            arrays: List[np.ndarray] = [
                np.array(np.random.default_rng(2).integers(10, size=10), dtype=d)
                for d in dtypes
            ]
        elif typ == 'float':
            dtypes = MIXED_FLOAT_DTYPES
            arrays = [
                np.array(np.random.default_rng(2).integers(10, size=10), dtype=d)
                for d in dtypes
            ]
        for d, a in zip(dtypes, arrays):
            assert a.dtype == d
        ad.update(dict(zip(dtypes, arrays)))
        df: DataFrame = DataFrame(ad)
        dtypes_combined = MIXED_FLOAT_DTYPES + MIXED_INT_DTYPES
        for d in dtypes_combined:
            if d in df:
                assert df.dtypes[d] == d

    def test_constructor_complex_dtypes(self) -> None:
        a: np.ndarray = np.random.default_rng(2).random(10).astype(np.complex64)
        b: np.ndarray = np.random.default_rng(2).random(10).astype(np.complex128)
        df: DataFrame = DataFrame({'a': a, 'b': b})
        assert a.dtype == df.a.dtype
        assert b.dtype == df.b.dtype

    def test_constructor_dtype_str_na_values(
        self, string_dtype: Any
    ) -> None:
        df: DataFrame = DataFrame({'A': ['x', None]}, dtype=string_dtype)
        result: DataFrame = df.isna()
        expected: DataFrame = DataFrame({'A': [False, True]})
        tm.assert_frame_equal(result, expected)
        assert df.iloc[1, 0] is None
        df = DataFrame({'A': ['x', np.nan]}, dtype=string_dtype)
        assert np.isnan(df.iloc[1, 0])

    def test_constructor_rec(self, float_frame: DataFrame) -> None:
        rec: np.recarray = float_frame.to_records(index=False)
        rec.dtype.names = list(rec.dtype.names)[::-1]
        index: RangeIndex = float_frame.index
        df: DataFrame = DataFrame(rec)
        tm.assert_index_equal(df.columns, Index(rec.dtype.names))
        df2: DataFrame = DataFrame(rec, index=index)
        tm.assert_index_equal(df2.columns, Index(rec.dtype.names))
        tm.assert_index_equal(df2.index, index)
        rng: np.ndarray = np.arange(len(rec))[::-1]
        df3: DataFrame = DataFrame(rec, index=rng, columns=['C', 'B'])
        expected: DataFrame = DataFrame(rec, index=rng).reindex(columns=['C', 'B'])
        tm.assert_frame_equal(df3, expected)

    def test_constructor_bool(self) -> None:
        df: DataFrame = DataFrame({0: np.ones(10, dtype=bool), 1: np.zeros(10, dtype=bool)})
        assert df.values.dtype == np.bool_

    def test_constructor_overflow_int64(self) -> None:
        values: np.ndarray = np.array(
            [2 ** 64 - i for i in range(1, 10)], dtype=np.uint64
        )
        result: DataFrame = DataFrame({'a': values})
        assert result['a'].dtype == np.uint64
        data_scores: List[Tuple[int, int]] = [
            (6311132704823138710, 273),
            (2685045978526272070, 23),
            (8921811264899370420, 45),
            (17019687244989530680, 270),
            (9930107427299601010, 273),
        ]
        dtype = [('uid', 'u8'), ('score', 'u8')]
        data: np.ndarray = np.zeros((len(data_scores),), dtype=dtype)
        data[:] = data_scores
        df_crawls: DataFrame = DataFrame(data)
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
        value: Any = values[0]
        result: DataFrame = DataFrame(values)
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
    def test_constructor_numpy_uints(self, values: List[Any] | np.ndarray) -> None:
        value: Any = values[0]
        result: DataFrame = DataFrame(values)
        assert result[0].dtype == value.dtype
        assert result[0][0] == value

    def test_constructor_ordereddict(self) -> None:
        nitems: int = 100
        nums: List[int] = list(range(nitems))
        np.random.default_rng(2).shuffle(nums)
        expected: List[str] = [f'A{i:d}' for i in nums]
        df: DataFrame = DataFrame(OrderedDict(zip(expected, [[0]] * nitems)))
        assert expected == list(df.columns)

    def test_constructor_dict(self) -> None:
        datetime_series: Series = Series(
            np.arange(30, dtype=np.float64),
            index=date_range('2020-01-01', periods=30),
        )
        datetime_series_short: Series = datetime_series[5:]
        frame: DataFrame = DataFrame({'col1': datetime_series, 'col2': datetime_series_short})
        assert len(datetime_series) == 30
        assert len(datetime_series_short) == 25
        tm.assert_series_equal(
            frame['col1'], datetime_series.rename('col1')
        )
        exp: Series = Series(
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
        msg: str = 'Mixing dicts with non-Series may lead to ambiguous ordering.'
        with pytest.raises(ValueError, match=msg):
            DataFrame({'A': {'a': 'a', 'b': 'b'}, 'B': ['a', 'b', 'c']})

    def test_constructor_dict_length1(self) -> None:
        frame: DataFrame = DataFrame({'A': {'1': 1, '2': 2}})
        tm.assert_index_equal(frame.index, Index(['1', '2']))

    def test_constructor_dict_with_index(self) -> None:
        idx: Index = Index([0, 1, 2])
        frame: DataFrame = DataFrame({}, index=idx)
        assert frame.index is idx

    def test_constructor_dict_with_index_and_columns(self) -> None:
        idx: Index = Index([0, 1, 2])
        frame: DataFrame = DataFrame({}, index=idx, columns=idx)
        assert frame.index is idx
        assert frame.columns is idx
        assert len(frame._series) == 3

    def test_constructor_dict_of_empty_lists(self) -> None:
        frame: DataFrame = DataFrame({'A': [], 'B': []}, columns=['A', 'B'])
        tm.assert_index_equal(frame.index, RangeIndex(0), exact=True)

    def test_constructor_dict_with_none(self) -> None:
        frame_none: DataFrame = DataFrame({'a': None}, index=[0])
        frame_none_list: DataFrame = DataFrame({'a': [None]}, index=[0])
        assert frame_none._get_value(0, 'a') is None
        assert frame_none_list._get_value(0, 'a') is None
        tm.assert_frame_equal(frame_none, frame_none_list)

    def test_constructor_dict_errors(self) -> None:
        msg: str = 'If using all scalar values, you must pass an index'
        with pytest.raises(ValueError, match=msg):
            DataFrame({'a': 0.7})
        with pytest.raises(ValueError, match=msg):
            DataFrame({'a': 0.7}, columns=['a'])

    @pytest.mark.parametrize(
        'scalar', [2, np.nan, None, 'D']
    )
    def test_constructor_invalid_items_unused(
        self, scalar: Any
    ) -> None:
        result: DataFrame = DataFrame({'a': scalar}, columns=['b'])
        expected: DataFrame = DataFrame(columns=['b'])
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        'value',
        [4, np.nan, None, float('nan')],
    )
    def test_constructor_dict_nan_key(
        self, value: Any
    ) -> None:
        cols: List[Any] = [1, value, 3]
        idx: List[Any] = ['a', value]
        values: List[List[int]] = [[0, 3], [1, 4], [2, 5]]
        data: Dict[Any, Series] = {
            cols[c]: Series(values[c], index=idx) for c in range(3)
        }
        result: DataFrame = DataFrame(data).sort_values(1).sort_values('a', axis=1)
        expected: DataFrame = DataFrame(np.arange(6, dtype='int64').reshape(2, 3), index=idx, columns=cols)
        tm.assert_frame_equal(result, expected)
        result = DataFrame(data, index=idx).sort_values('a', axis=1)
        tm.assert_frame_equal(result, expected)
        result = DataFrame(data, index=idx, columns=cols)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        'value',
        [np.nan, None, float('nan')],
    )
    def test_constructor_dict_nan_tuple_key(
        self, value: Any
    ) -> None:
        cols: Index = Index([(11, 21), (value, 22), (13, value)])
        idx: Index = Index([('a', value), (value, 2)])
        values: List[List[int]] = [[0, 3], [1, 4], [2, 5]]
        data: Dict[Any, Series] = {
            cols[c]: Series(values[c], index=idx) for c in range(3)
        }
        result: DataFrame = DataFrame(data).sort_values((11, 21)).sort_values(('a', value), axis=1)
        expected: DataFrame = DataFrame(np.arange(6, dtype='int64').reshape(2, 3), index=idx, columns=cols)
        tm.assert_frame_equal(result, expected)
        result = DataFrame(data, index=idx).sort_values(('a', value), axis=1)
        tm.assert_frame_equal(result, expected)
        result = DataFrame(data, index=idx, columns=cols)
        tm.assert_frame_equal(result, expected)

    def test_constructor_order_insertion(self) -> None:
        datetime_series: Series = Series(
            np.arange(10, dtype=np.float64),
            index=date_range('2020-01-01', periods=10),
        )
        datetime_series_short: Series = datetime_series[:5]
        d: Dict[str, Dict[Any, Any]] = {'b': datetime_series_short, 'a': datetime_series}
        frame: DataFrame = DataFrame(data=d)
        expected: DataFrame = DataFrame(data=d, columns=list('ba'))
        tm.assert_frame_equal(frame, expected)

    def test_constructor_Series_named(self) -> None:
        a: Series = Series([1, 2, 3], name=0)
        df: DataFrame = DataFrame(a)
        assert df.columns[0] == 0
        tm.assert_index_equal(df.index, a.index)
        arr: np.ndarray = np.random.default_rng(2).standard_normal(10)
        s: Series = Series(arr, name='x')
        df = DataFrame(s)
        expected: Series = Series(arr, dtype=np.float64, name='x')
        tm.assert_series_equal(df['x'], expected)
        s = Series(arr, index=range(3, 13))
        df = DataFrame(s)
        expected = Series(arr, index=range(3, 13), name='x')
        tm.assert_series_equal(df[0], Series(expected, name=0))
        msg: str = 'Shape of passed values is \\(10, 1\\), indices imply \\(10, 2\\)'
        with pytest.raises(ValueError, match=msg):
            DataFrame(s, columns=[1, 2])
        a = Series([], name='x', dtype=object)
        df = DataFrame(a)
        assert df.columns[0] == 'x'
        s1: Series = Series(arr, name='x')
        df = DataFrame([s1, arr]).T
        expected: DataFrame = DataFrame({'x': s1, 'Unnamed 0': arr}, columns=['x', 'Unnamed 0'])
        tm.assert_series_equal(df['x'], expected['x'])
        df = DataFrame([arr, s1]).T
        expected = DataFrame({'1': s1, '0': arr}, columns=range(2))
        tm.assert_frame_equal(df, expected)

    @pytest.mark.parametrize(
        'copy',
        [False, True],
    )
    def test_dict_nocopy(
        self,
        copy: bool,
        any_numeric_ea_dtype: Any,
        any_numpy_dtype: Any,
    ) -> None:
        a: np.ndarray = np.array([1, 2], dtype=any_numpy_dtype)
        b: np.ndarray = np.array([3, 4], dtype=any_numpy_dtype)
        if b.dtype.kind in ['S', 'U']:
            pytest.skip(f'{b.dtype} get cast, making the checks below more cumbersome')
        c: Any = pd.array([1, 2], dtype=any_numeric_ea_dtype)
        c_orig: Any = c.copy()
        df: DataFrame = DataFrame({'a': a, 'b': b, 'c': c}, copy=copy)

        def get_base(obj: Any) -> Any:
            if isinstance(obj, np.ndarray):
                return obj.base
            elif isinstance(obj.dtype, np.dtype):
                return obj._ndarray.base
            else:
                raise TypeError

        def check_views(c_only: bool = False) -> None:
            assert sum(
                (x.values is c for x in df._mgr.blocks)
            ) == 1
            if c_only:
                return
            assert sum(
                (get_base(x.values) is a for x in df._mgr.blocks if isinstance(x.values.dtype, np.dtype))
            ) == 1
            assert sum(
                (get_base(x.values) is b for x in df._mgr.blocks if isinstance(x.values.dtype, np.dtype))
            ) == 1

        if not copy:
            check_views()
        should_raise: bool = not lib.is_np_dtype(df.dtypes.iloc[0], 'fciuO')
        if should_raise:
            with pytest.raises(TypeError, match='Invalid value'):
                df.iloc[0, 0] = 0
                df.iloc[0, 1] = 0
            return
        else:
            df.iloc[0, 0] = 0
            df.iloc[0, 1] = 0
        if not copy:
            check_views(True)
        df.iloc[:, 2] = pd.array([45, 46], dtype=c.dtype)
        assert df.dtypes.iloc[2] == c.dtype
        if copy:
            if a.dtype.kind == 'M':
                assert a[0] == a.dtype.type(1, 'ns')
                assert b[0] == b.dtype.type(3, 'ns')
            else:
                assert a[0] == a.dtype.type(1)
                assert b[0] == b.dtype.type(3)
            assert c[0] == c_orig[0]

    def test_construct_from_dict_ea_series(self) -> None:
        ser: Series = Series([1, 2, 3], dtype='Int64')
        df: DataFrame = DataFrame({'a': ser})
        assert not np.shares_memory(ser.values._data, df['a'].values._data)

    def test_from_series_with_name_with_columns(self) -> None:
        result: DataFrame = DataFrame(Series(1, name='foo'), columns=['bar'])
        expected: DataFrame = DataFrame(columns=['bar'])
        tm.assert_frame_equal(result, expected)

    def test_nested_list_columns(self) -> None:
        result: DataFrame = DataFrame(
            [[1, 2, 3], [4, 5, 6]],
            columns=[['A', 'A', 'A'], ['a', 'b', 'c']],
        )
        expected: DataFrame = DataFrame(
            [[1, 2, 3], [4, 5, 6]],
            columns=MultiIndex.from_tuples([('A', 'a'), ('A', 'b'), ('A', 'c')]),
        )
        tm.assert_frame_equal(result, expected)

    def test_from_2d_ndarray_with_dtype(self) -> None:
        array_dim2: np.ndarray = np.arange(10).reshape((5, 2))
        df: DataFrame = DataFrame(array_dim2, dtype='datetime64[ns, UTC]')
        expected: DataFrame = DataFrame(array_dim2).astype('datetime64[ns, UTC]')
        tm.assert_frame_equal(df, expected)

    def test_constructor_mixed_dict_nonseries(self) -> None:
        data: Dict[str, Any] = {'A': {'foo': 1, 'bar': 2, 'baz': 3}, 'B': Series([4, 3, 2, 1], index=['bar', 'qux', 'baz', 'foo'])}
        result: DataFrame = DataFrame(data)
        tm.assert_frame_equal(result, DataFrame({'A': [1, 2, 3], 'B': [4, 3, 2, 1]}))

    def test_constructor_mixed_type_rows(self) -> None:
        data: List[Union[List[Any], Tuple[Any, Any]]] = [[1, 2], (3, 4)]
        result: DataFrame = DataFrame(data)
        expected: DataFrame = DataFrame([[1, 2], [3, 4]])
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
        lists: List[List[Any]],
    ) -> None:
        result: DataFrame = DataFrame(tuples)
        expected: DataFrame = DataFrame(lists)
        tm.assert_frame_equal(result, expected)

    def test_constructor_list_of_tuples(self) -> None:
        data: Dict[str, Tuple[int, int]] = {'A': (1, 2), 'B': (3, 4)}
        result: DataFrame = DataFrame({'A': [(1, 2), (3, 4)]})
        expected: DataFrame = DataFrame({'A': Series([(1, 2), (3, 4)])})
        tm.assert_frame_equal(result, expected)

    def test_constructor_list_of_namedtuples(self) -> None:
        named_tuple = namedtuple('Pandas', list('ab'))
        tuples: List[Any] = [named_tuple(1, 3), named_tuple(2, 4)]
        expected: DataFrame = DataFrame({'a': [1, 2], 'b': [3, 4]})
        result: DataFrame = DataFrame(tuples)
        tm.assert_frame_equal(result, expected)
        expected = DataFrame({'y': [1, 2], 'z': [3, 4]})
        result = DataFrame(tuples, columns=['y', 'z'])
        tm.assert_frame_equal(result, expected)

    def test_constructor_list_of_dataclasses(self) -> None:
        Point = make_dataclass('Point', [('x', int), ('y', int)])
        data: List[Any] = [Point(0, 3), Point(1, 3)]
        expected: DataFrame = DataFrame({'x': [0, 1], 'y': [3, 3]})
        result: DataFrame = DataFrame(data)
        tm.assert_frame_equal(result, expected)

    def test_constructor_list_of_dataclasses_with_varying_types(self) -> None:
        Point = make_dataclass('Point', [('x', int), ('y', int)])
        HLine = make_dataclass('HLine', [('x0', int), ('x1', int), ('y', int)])
        data: List[Any] = [Point(0, 3), HLine(1, 3, 3)]
        expected: DataFrame = DataFrame(
            {'x': [0, np.nan], 'y': [3, 3], 'x0': [np.nan, 1], 'x1': [np.nan, 3]}
        )
        result: DataFrame = DataFrame(data)
        tm.assert_frame_equal(result, expected)

    def test_constructor_list_of_dataclasses_error_thrown(self) -> None:
        Point = make_dataclass('Point', [('x', int), ('y', int)])
        msg: str = 'asdict() should be called on dataclass instances'
        with pytest.raises(TypeError, match=re.escape(msg)):
            DataFrame([Point(0, 0), {'x': 1, 'y': 0}])

    def test_constructor_list_of_dict_order(self) -> None:
        data: List[Dict[str, Any]] = [
            {'First': 1, 'Second': 4, 'Third': 7, 'Fourth': 10},
            {'Second': 5, 'First': 2, 'Fourth': 11, 'Third': 8},
            {'Second': 6, 'First': 3, 'Fourth': 12, 'Third': 9, 'YYY': 14, 'XXX': 13},
        ]
        expected: DataFrame = DataFrame(
            {'First': [1, 2, 3], 'Second': [4, 5, 6], 'Third': [7, 8, 9], 'Fourth': [10, 11, 12], 'YYY': [None, None, 14], 'XXX': [None, None, 13]}
        )
        result: DataFrame = DataFrame(data)
        tm.assert_frame_equal(result, expected)

    def test_constructor_Series_named(
        self, float_frame: DataFrame
    ) -> None:
        ser: Series = Series(float_frame['A'], index=float_frame.index, name='A')
        df: DataFrame = DataFrame({'A': ser}, copy=True)
        df['A'] = 5
        assert (df['A'] == 5).all()
        assert not (float_frame['A'] == 5).all()

    def test_from_series_with_name_with_columns(self) -> None:
        result: DataFrame = DataFrame(Series(1, name='foo'), columns=['bar'])
        expected: DataFrame = DataFrame(columns=['bar'])
        tm.assert_frame_equal(result, expected)

    def test_constructor_Series_named_and_columns(self) -> None:
        s0: Series = Series(range(5), name=0)
        s1: Series = Series(range(5), name=1)
        tm.assert_frame_equal(DataFrame(s0, columns=[0]), s0.to_frame())
        tm.assert_frame_equal(DataFrame(s1, columns=[1]), s1.to_frame())
        assert DataFrame(s0, columns=[1]).empty
        assert DataFrame(s1, columns=[0]).empty

    def test_constructor_Series_differently_indexed(self) -> None:
        s1: Series = Series([1, 2, 3], index=['a', 'b', 'c'], name='x')
        s2: Series = Series([1, 2, 3], index=['a', 'b', 'c'])
        other_index: Index = Index(['a', 'b'])
        df1: DataFrame = DataFrame(s1, index=other_index)
        exp1: DataFrame = DataFrame(s1.reindex(other_index))
        assert df1.columns[0] == 'x'
        tm.assert_frame_equal(df1, exp1)
        df2: DataFrame = DataFrame(s2, index=other_index)
        exp2: DataFrame = DataFrame(s2.reindex(other_index))
        assert df2.columns[0] == 0
        tm.assert_index_equal(df2.index, other_index)
        tm.assert_frame_equal(df2, exp2)

    @pytest.mark.parametrize(
        'scalar', [2, np.nan, None, 'D']
    )
    def test_constructor_invalid_items_unused(
        self, scalar: Any
    ) -> None:
        result: DataFrame = DataFrame({'a': scalar}, columns=['b'])
        expected: DataFrame = DataFrame(columns=['b'])
        tm.assert_frame_equal(result, expected)

    def test_constructor_Series_named_and_columns(
        self, float_frame: DataFrame
    ) -> None:
        ser: Series = Series(float_frame['A'], index=float_frame.index, name='A')
        df: DataFrame = DataFrame({'A': ser}, copy=True)
        df['A'] = 5
        assert (df['A'] == 5).all()
        assert not (float_frame['A'] == 5).all()

    def test_constructor_Series_copy_bug(
        self, float_frame: DataFrame
    ) -> None:
        df: DataFrame = DataFrame(float_frame['A'], index=float_frame.index, columns=['A'])
        df.copy()

    def test_constructor_mixed_dict_nonseries(
        self, float_frame: DataFrame
    ) -> None:
        data: Dict[str, Union[Series, List[Any]]] = {}
        data['A'] = {'foo': 1, 'bar': 2, 'baz': 3}
        data['B'] = Series([4, 3, 2, 1], index=['bar', 'qux', 'baz', 'foo'])
        df: DataFrame = DataFrame(data)
        tm.assert_frame_equal(df, DataFrame({'A': [1, 2, 3], 'B': [4, 3, 2, 1]}))

    def test_constructor_mixed_type_rows(
        self, float_frame: DataFrame
    ) -> None:
        data: List[Union[List[Any], Tuple[Any, Any]]] = [[1, 2], (3, 4)]
        df: DataFrame = DataFrame(data)
        tm.assert_frame_equal(df, DataFrame([[1, 2], [3, 4]]))

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
        lists: List[List[Any]],
    ) -> None:
        result: DataFrame = DataFrame(tuples)
        expected: DataFrame = DataFrame(lists)
        tm.assert_frame_equal(result, expected)

    def test_constructor_list_of_tuples(self) -> None:
        data: Dict[str, Tuple[int, int]] = {'A': (1, 2), 'B': (3, 4)}
        result: DataFrame = DataFrame({'A': [(1, 2), (3, 4)]})
        expected: DataFrame = DataFrame({'A': Series([(1, 2), (3, 4)])})
        tm.assert_frame_equal(result, expected)

    def test_constructor_list_of_namedtuples(self) -> None:
        named_tuple = namedtuple('Pandas', list('ab'))
        tuples: List[Any] = [named_tuple(1, 3), named_tuple(2, 4)]
        expected: DataFrame = DataFrame({'a': [1, 2], 'b': [3, 4]})
        result: DataFrame = DataFrame(tuples)
        tm.assert_frame_equal(result, expected)
        expected = DataFrame({'y': [1, 2], 'z': [3, 4]})
        result = DataFrame(tuples, columns=['y', 'z'])
        tm.assert_frame_equal(result, expected)

    def test_constructor_list_of_dataclasses(self) -> None:
        Point = make_dataclass('Point', [('x', int), ('y', int)])
        data: List[Any] = [Point(0, 3), Point(1, 3)]
        expected: DataFrame = DataFrame({'x': [0, 1], 'y': [3, 3]})
        result: DataFrame = DataFrame(data)
        tm.assert_frame_equal(result, expected)

    def test_constructor_list_of_dataclasses_with_varying_types(self) -> None:
        Point = make_dataclass('Point', [('x', int), ('y', int)])
        HLine = make_dataclass('HLine', [('x0', int), ('x1', int), ('y', int)])
        data: List[Any] = [Point(0, 3), HLine(1, 3, 3)]
        expected: DataFrame = DataFrame(
            {'x': [0, np.nan], 'y': [3, 3], 'x0': [np.nan, 1], 'x1': [np.nan, 3]}
        )
        result: DataFrame = DataFrame(data)
        tm.assert_frame_equal(result, expected)

    def test_constructor_list_of_dataclasses_error_thrown(self) -> None:
        Point = make_dataclass('Point', [('x', int), ('y', int)])
        msg: str = 'asdict() should be called on dataclass instances'
        with pytest.raises(TypeError, match=re.escape(msg)):
            DataFrame([Point(0, 0), {'x': 1, 'y': 0}])

    def test_constructor_list_of_dict_order(self) -> None:
        data: List[Dict[str, Any]] = [
            {'First': 1, 'Second': 4, 'Third': 7, 'Fourth': 10},
            {'Second': 5, 'First': 2, 'Fourth': 11, 'Third': 8},
            {'Second': 6, 'First': 3, 'Fourth': 12, 'Third': 9, 'YYY': 14, 'XXX': 13},
        ]
        expected: DataFrame = DataFrame(
            {'First': [1, 2, 3], 'Second': [4, 5, 6], 'Third': [7, 8, 9], 'Fourth': [10, 11, 12], 'YYY': [None, None, 14], 'XXX': [None, None, 13]}
        )
        result: DataFrame = DataFrame(data)
        tm.assert_frame_equal(result, expected)

    def test_constructor_Series_named_and_columns(
        self, float_frame: DataFrame
    ) -> None:
        s0: Series = Series(range(5), name=0)
        s1: Series = Series(range(5), name=1)
        tm.assert_frame_equal(DataFrame(s0, columns=[0]), s0.to_frame())
        tm.assert_frame_equal(DataFrame(s1, columns=[1]), s1.to_frame())
        assert DataFrame(s0, columns=[1]).empty
        assert DataFrame(s1, columns=[0]).empty

    def test_constructor_Series_differently_indexed(
        self, float_frame: DataFrame
    ) -> None:
        s1: Series = Series([1, 2, 3], index=['a', 'b', 'c'], name='x')
        s2: Series = Series([1, 2, 3], index=['a', 'b', 'c'])
        other_index: Index = Index(['a', 'b'])
        df1: DataFrame = DataFrame(s1, index=other_index)
        exp1: DataFrame = DataFrame(s1.reindex(other_index))
        assert df1.columns[0] == 'x'
        tm.assert_frame_equal(df1, exp1)
        df2: DataFrame = DataFrame(s2, index=other_index)
        exp2: DataFrame = DataFrame(s2.reindex(other_index))
        assert df2.columns[0] == 0
        tm.assert_index_equal(df2.index, other_index)
        tm.assert_frame_equal(df2, exp2)

    @pytest.mark.parametrize(
        'scalar', [2, np.nan, None, 'D']
    )
    def test_constructor_invalid_items_unused(
        self, scalar: Any
    ) -> None:
        result: DataFrame = DataFrame({'a': scalar}, columns=['b'])
        expected: DataFrame = DataFrame(columns=['b'])
        tm.assert_frame_equal(result, expected)

    def test_constructor_Series_named_and_columns(
        self, float_frame: DataFrame
    ) -> None:
        s0: Series = Series(range(5), name=0)
        s1: Series = Series(range(5), name=1)
        tm.assert_frame_equal(DataFrame(s0, columns=[0]), s0.to_frame())
        tm.assert_frame_equal(DataFrame(s1, columns=[1]), s1.to_frame())
        assert DataFrame(s0, columns=[1]).empty
        assert DataFrame(s1, columns=[0]).empty

    def test_constructor_Series_differently_indexed(
        self, float_frame: DataFrame
    ) -> None:
        s1: Series = Series([1, 2, 3], index=['a', 'b', 'c'], name='x')
        s2: Series = Series([1, 2, 3], index=['a', 'b', 'c'])
        other_index: Index = Index(['a', 'b'])
        df1: DataFrame = DataFrame(s1, index=other_index)
        exp1: DataFrame = DataFrame(s1.reindex(other_index))
        assert df1.columns[0] == 'x'
        tm.assert_frame_equal(df1, exp1)
        df2: DataFrame = DataFrame(s2, index=other_index)
        exp2: DataFrame = DataFrame(s2.reindex(other_index))
        assert df2.columns[0] == 0
        tm.assert_index_equal(df2.index, other_index)
        tm.assert_frame_equal(df2, exp2)

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
        td_as_int: List[int] = [1, 2, 3, 4]
        data: Dict[int, Any] = {i: klass(s) for i, s in enumerate(td_as_int)}
        expected: DataFrame = DataFrame(
            [
                {0: 0, 1: None, 2: None, 3: None},
                {0: None, 1: 2, 2: None, 3: None},
                {0: None, 1: None, 2: 4, 3: None},
                {0: None, 1: None, 2: None, 3: 6},
            ],
            index=[Timedelta(td, 'D') for td in td_as_int],
        )
        result: DataFrame = DataFrame(data)
        tm.assert_frame_equal(result, expected)

    def test_constructor_period_dict(self) -> None:
        a: pd.PeriodIndex = pd.PeriodIndex(['2012-01', 'NaT', '2012-04'], freq='M')
        b: pd.PeriodIndex = pd.PeriodIndex(['2012-02-01', '2012-03-01', 'NaT'], freq='D')
        df: DataFrame = DataFrame({'a': a, 'b': b})
        assert df['a'].dtype == a.dtype
        assert df['b'].dtype == b.dtype
        df = DataFrame({'a': a.astype(object).tolist(), 'b': b.astype(object).tolist()})
        assert df['a'].dtype == a.dtype
        assert df['b'].dtype == b.dtype

    def test_nested_dict_frame_constructor(self) -> None:
        rng: pd.period_range = pd.period_range('1/1/2000', periods=5)
        df: DataFrame = DataFrame(
            np.random.default_rng(2).standard_normal((10, 5)), columns=rng
        )
        data: Dict[Any, Dict[Any, float]] = {}
        for col in df.columns:
            for row in df.index:
                data.setdefault(col, {})[row] = df._get_value(row, col)
        result: DataFrame = DataFrame(data, columns=rng)
        tm.assert_frame_equal(result, df)
        data = {}
        for col in df.columns:
            for row in df.index:
                data.setdefault(row, {})[col] = df._get_value(row, col)
        result = DataFrame(data, index=rng).T
        tm.assert_frame_equal(result, df)

    def test_constructor_dtype_copy(self) -> None:
        orig_df: DataFrame = DataFrame({'col1': [1.0], 'col2': [2.0], 'col3': [3.0]})
        new_df: DataFrame = DataFrame(orig_df, dtype=float, copy=True)
        new_df['col1'] = 200.0
        assert orig_df['col1'][0] == 1.0

    def test_constructor_dtype_nocast_view_dataframe(self) -> None:
        df: DataFrame = DataFrame([[1, 2]])
        should_be_view: DataFrame = DataFrame(df, dtype=df[0].dtype)
        should_be_view.iloc[0, 0] = 99
        assert df.values[0, 0] == 1

    def test_constructor_dtype_nocast_view_2d_array(self) -> None:
        df: DataFrame = DataFrame([[1, 2], [3, 4]], dtype='int64')
        df2: DataFrame = DataFrame(df.values, dtype=df[0].dtype)
        assert df2._mgr.blocks[0].values.flags.c_contiguous

    def test_1d_object_array_does_not_copy(
        self, using_infer_string: bool
    ) -> None:
        arr: np.ndarray = np.array(['a', 'b'], dtype='object')
        df: DataFrame = DataFrame(arr, copy=False)
        if using_infer_string:
            if df[0].dtype.storage == 'pyarrow':
                pass
            else:
                assert np.shares_memory(df[0].to_numpy(), arr)
        else:
            assert np.shares_memory(df.values, arr)
        df = DataFrame(arr, dtype=object, copy=False)
        assert np.shares_memory(df.values, arr)

    def test_2d_object_array_does_not_copy(
        self, using_infer_string: bool
    ) -> None:
        arr: np.ndarray = np.array([['a', 'b'], ['c', 'd']], dtype='object')
        df: DataFrame = DataFrame(arr, copy=False)
        if using_infer_string:
            if df[0].dtype.storage == 'pyarrow':
                pass
            else:
                assert np.shares_memory(df[0].to_numpy(), arr)
        else:
            assert np.shares_memory(df.values, arr)
        df = DataFrame(arr, dtype=object, copy=False)
        assert np.shares_memory(df.values, arr)

    def test_to_frame_with_falsey_names(self) -> None:
        result: DataFrame = Series(name=0, dtype=object).to_frame().dtypes
        expected: Series = Series({0: object})
        tm.assert_series_equal(result, expected)
        result = DataFrame(Series(name=0, dtype=object)).dtypes
        tm.assert_series_equal(result, expected)

    @pytest.mark.arm_slow
    @pytest.mark.parametrize(
        'dtype',
        [None, 'uint8', 'category'],
    )
    def test_constructor_range_dtype(
        self, dtype: Optional[str]
    ) -> None:
        expected: DataFrame = DataFrame({'A': [0, 1, 2, 3, 4]}, dtype=dtype or 'int64')
        result: DataFrame = DataFrame(range(5), columns=['A'], dtype=dtype)
        tm.assert_frame_equal(result, expected)
        result = DataFrame({'A': range(5)}, dtype=dtype)
        tm.assert_frame_equal(result, expected)

    def test_frame_from_list_subclass(self) -> None:
        class ListSubclass(list):
            pass

        expected: DataFrame = DataFrame([[1, 2, 3], [4, 5, 6]])
        result: DataFrame = DataFrame(
            ListSubclass([ListSubclass([1, 2, 3]), ListSubclass([4, 5, 6])])
        )
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
        self,
        extension_arr: Union[Categorical, SparseArray, IntervalArray, PeriodArray],
    ) -> None:
        expected: DataFrame = DataFrame(Series(extension_arr))
        result: DataFrame = DataFrame(extension_arr)
        tm.assert_frame_equal(result, expected)

    def test_datetime_date_tuple_columns_from_dict(self) -> None:
        v: date = date.today()
        tup: Tuple[date, date] = (v, v)
        result: DataFrame = DataFrame({'a': tup}, columns=[tup])
        expected: DataFrame = DataFrame(
            [[0, 1]],
            columns=Index(Series([tup]))
        )
        tm.assert_frame_equal(result, expected)

    def test_construct_with_two_categoricalindex_series(self) -> None:
        s1: Series = Series(
            [39, 6, 4], index=CategoricalIndex(['female', 'male', 'unknown'])
        )
        s2: Series = Series(
            [2, 152, 2, 242, 150],
            index=CategoricalIndex(['f', 'female', 'm', 'male', 'unknown']),
        )
        result: DataFrame = DataFrame([s1, s2])
        expected: DataFrame = DataFrame(
            np.array(
                [[39, 6, 4, np.nan, np.nan], [152.0, 242.0, 150.0, 2.0, 2.0]]
            ),
            columns=['female', 'male', 'unknown', 'f', 'm'],
        )
        tm.assert_frame_equal(result, expected)

    @pytest.mark.filterwarnings('ignore:invalid value encountered in cast:RuntimeWarning')
    def test_constructor_series_nonexact_categoricalindex(self) -> None:
        ser: Series = Series(range(100))
        ser1: Series = cut(ser, 10).value_counts().head(5)
        ser2: Series = cut(ser, 10).value_counts().tail(5)
        result: DataFrame = DataFrame({'1': ser1, '2': ser2})
        index: pd.IntervalIndex = pd.interval_range(start=-0.099, end=99, periods=10, closed='right')
        expected: DataFrame = DataFrame(
            {'1': [10] * 5 + [np.nan] * 5, '2': [np.nan] * 5 + [10] * 5},
            index=index,
        )
        tm.assert_frame_equal(expected, result)

    def test_from_M8_structured(self) -> None:
        dates: List[Tuple[float, float]] = [(datetime(2012, 9, 9, 0, 0), datetime(2012, 9, 8, 15, 10))]
        arr: np.recarray = np.array(dates, dtype=[('Date', 'M8[us]'), ('Forecasting', 'M8[us]')]).view(np.recarray)
        df: DataFrame = DataFrame(arr)
        assert df['Date'][0] == dates[0][0]
        assert df['Forecasting'][0] == dates[0][1]
        s: Series = Series(arr['Date'])
        assert isinstance(s[0], Timestamp)
        assert s[0] == dates[0][0]

    def test_from_datetime_subclass(self) -> None:
        class DatetimeSubclass(datetime):
            pass

        datetimes: List[DatetimeSubclass] = [DatetimeSubclass(2020, 1, 1, 1, 1)]
        df: DataFrame = DataFrame({'datetime': datetimes})
        assert df['datetime'].dtype == 'datetime64[ns]'

    def test_with_mismatched_index_length_raises(
        self,
    ) -> None:
        dti: DatetimeIndex = date_range('2016-01-01', periods=3, tz='US/Pacific')
        msg: str = 'Shape of passed values|Passed arrays should have the same length'
        with pytest.raises(ValueError, match=msg):
            DataFrame(dti, index=range(4))

    def test_frame_ctor_datetime64_column(self) -> None:
        rng: pd.DatetimeIndex = date_range('1/1/2000 00:00:00', '1/1/2000 1:59:50', freq='10s')
        dates: np.ndarray = np.asarray(rng)
        df: DataFrame = DataFrame({'A': np.random.default_rng(2).standard_normal(len(rng)), 'B': dates})
        assert np.issubdtype(df['B'].dtype, np.dtype('M8[ns]'))

    def test_dataframe_constructor_infer_multiindex(self) -> None:
        index_lists: List[List[Any]] = [['a', 'a', 'b', 'b'], ['x', 'y', 'x', 'y']]
        multi: DataFrame = DataFrame(
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
        'input_vals',
        [
            [[1, 2], ['1', '2']],
            [['1', '2'], [1, 2]],
            [
                list(date_range('1/1/2011', periods=2, freq='h')),
                list(date_range('1/1/2011', periods=2, freq='h', tz='US/Eastern')),
            ],
            [
                list(date_range('1/1/2011', periods=2, freq='h', tz='US/Eastern')),
                list(date_range('1/1/2011', periods=2, freq='h')),
            ],
            [[Interval(left=0, right=5)]],
        ],
    )
    def test_constructor_list_str(
        self,
        input_vals: List[Any],
        string_dtype: Any,
    ) -> None:
        result: DataFrame = DataFrame({'A': input_vals}, dtype=string_dtype)
        expected: DataFrame = DataFrame({'A': input_vals}).astype({'A': string_dtype})
        tm.assert_frame_equal(result, expected)

    def test_constructor_list_str_na(
        self,
        string_dtype: Any,
    ) -> None:
        result: DataFrame = DataFrame({'A': [1.0, 2.0, None]}, dtype=string_dtype)
        expected: DataFrame = DataFrame({'A': ['1.0', '2.0', None]}, dtype=object)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        'copy',
        [False, True],
    )
    def test_dict_nocopy(
        self,
        copy: bool,
        any_numeric_ea_dtype: Any,
        any_numpy_dtype: Any,
    ) -> None:
        a: np.ndarray = np.array([1, 2], dtype=any_numpy_dtype)
        b: np.ndarray = np.array([3, 4], dtype=any_numpy_dtype)
        if b.dtype.kind in ['S', 'U']:
            pytest.skip(f'{b.dtype} get cast, making the checks below more cumbersome')
        c: Any = pd.array([1, 2], dtype=any_numeric_ea_dtype)
        c_orig: Any = c.copy()
        df: DataFrame = DataFrame({'a': a, 'b': b, 'c': c}, copy=copy)

        def get_base(obj: Any) -> Any:
            if isinstance(obj, np.ndarray):
                return obj.base
            elif isinstance(obj.dtype, np.dtype):
                return obj._ndarray.base
            else:
                raise TypeError

        def check_views(c_only: bool = False) -> None:
            assert sum(
                (x.values is c for x in df._mgr.blocks)
            ) == 1
            if c_only:
                return
            assert sum(
                (get_base(x.values) is a for x in df._mgr.blocks if isinstance(x.values.dtype, np.dtype))
            ) == 1
            assert sum(
                (get_base(x.values) is b for x in df._mgr.blocks if isinstance(x.values.dtype, np.dtype))
            ) == 1

        if not copy:
            check_views()
        should_raise: bool = not lib.is_np_dtype(df.dtypes.iloc[0], 'fciuO')
        if should_raise:
            with pytest.raises(TypeError, match='Invalid value'):
                df.iloc[0, 0] = 0
                df.iloc[0, 1] = 0
            return
        else:
            df.iloc[0, 0] = 0
            df.iloc[0, 1] = 0
        if not copy:
            check_views(True)
        df.iloc[:, 2] = pd.array([45, 46], dtype=c.dtype)
        assert df.dtypes.iloc[2] == c.dtype
        if copy:
            if a.dtype.kind == 'M':
                assert a[0] == a.dtype.type(1, 'ns')
                assert b[0] == b.dtype.type(3, 'ns')
            else:
                assert a[0] == a.dtype.type(1)
                assert b[0] == b.dtype.type(3)
            assert c[0] == c_orig[0]

    def test_construct_from_dict_ea_series(self) -> None:
        ser: Series = Series([1, 2, 3], dtype='Int64')
        df: DataFrame = DataFrame({'a': ser})
        assert not np.shares_memory(ser.values._data, df['a'].values._data)

    def test_from_series_with_name_with_columns(self) -> None:
        result: DataFrame = DataFrame(Series(1, name='foo'), columns=['bar'])
        expected: DataFrame = DataFrame(columns=['bar'])
        tm.assert_frame_equal(result, expected)

    def test_nested_list_columns(self) -> None:
        result: DataFrame = DataFrame(
            [[1, 2, 3], [4, 5, 6]],
            columns=[['A', 'A', 'A'], ['a', 'b', 'c']],
        )
        expected: DataFrame = DataFrame(
            [[1, 2, 3], [4, 5, 6]],
            columns=MultiIndex.from_tuples([('A', 'a'), ('A', 'b'), ('A', 'c')]),
        )
        tm.assert_frame_equal(result, expected)

    def test_from_2d_ndarray_with_dtype(self) -> None:
        array_dim2: np.ndarray = np.arange(10).reshape((5, 2))
        df: DataFrame = DataFrame(array_dim2, dtype='datetime64[ns, UTC]')
        expected: DataFrame = DataFrame(array_dim2).astype('datetime64[ns, UTC]')
        tm.assert_frame_equal(df, expected)

    def test_constructor_mixed_dict_nonseries(self) -> None:
        data: Dict[str, Any] = {'A': {'foo': 1, 'bar': 2, 'baz': 3}, 'B': Series([4, 3, 2, 1], index=['bar', 'qux', 'baz', 'foo'])}
        result: DataFrame = DataFrame(data)
        tm.assert_frame_equal(result, DataFrame({'A': [1, 2, 3], 'B': [4, 3, 2, 1]}))

    def test_constructor_mixed_type_rows(self) -> None:
        data: List[Union[List[Any], Tuple[Any, Any]]] = [[1, 2], (3, 4)]
        result: DataFrame = DataFrame(data)
        expected: DataFrame = DataFrame([[1, 2], [3, 4]])
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
        lists: List[List[Any]],
    ) -> None:
        result: DataFrame = DataFrame(tuples)
        expected: DataFrame = DataFrame(lists)
        tm.assert_frame_equal(result, expected)

    def test_constructor_list_of_tuples(self) -> None:
        data: Dict[str, Tuple[int, int]] = {'A': (1, 2), 'B': (3, 4)}
        result: DataFrame = DataFrame({'A': [(1, 2), (3, 4)]})
        expected: DataFrame = DataFrame({'A': Series([(1, 2), (3, 4)])})
        tm.assert_frame_equal(result, expected)

    def test_constructor_list_of_namedtuples(self) -> None:
        named_tuple = namedtuple('Pandas', list('ab'))
        tuples: List[Any] = [named_tuple(1, 3), named_tuple(2, 4)]
        expected: DataFrame = DataFrame({'a': [1, 2], 'b': [3, 4]})
        result: DataFrame = DataFrame(tuples)
        tm.assert_frame_equal(result, expected)
        expected = DataFrame({'y': [1, 2], 'z': [3, 4]})
        result = DataFrame(tuples, columns=['y', 'z'])
        tm.assert_frame_equal(result, expected)

    def test_constructor_list_of_dataclasses(self) -> None:
        Point = make_dataclass('Point', [('x', int), ('y', int)])
        data: List[Any] = [Point(0, 3), Point(1, 3)]
        expected: DataFrame = DataFrame({'x': [0, 1], 'y': [3, 3]})
        result: DataFrame = DataFrame(data)
        tm.assert_frame_equal(result, expected)

    def test_constructor_list_of_dataclasses_with_varying_types(self) -> None:
        Point = make_dataclass('Point', [('x', int), ('y', int)])
        HLine = make_dataclass('HLine', [('x0', int), ('x1', int), ('y', int)])
        data: List[Any] = [Point(0, 3), HLine(1, 3, 3)]
        expected: DataFrame = DataFrame(
            {'x': [0, np.nan], 'y': [3, 3], 'x0': [np.nan, 1], 'x1': [np.nan, 3]}
        )
        result: DataFrame = DataFrame(data)
        tm.assert_frame_equal(result, expected)

    def test_constructor_list_of_dataclasses_error_thrown(
        self, float_frame: DataFrame
    ) -> None:
        Point = make_dataclass('Point', [('x', int), ('y', int)])
        msg: str = 'asdict() should be called on dataclass instances'
        with pytest.raises(TypeError, match=re.escape(msg)):
            DataFrame([Point(0, 0), {'x': 1, 'y': 0}])

    def test_constructor_list_of_dict_order(self) -> None:
        data: List[Dict[str, Any]] = [
            {'First': 1, 'Second': 4, 'Third': 7, 'Fourth': 10},
            {'Second': 5, 'First': 2, 'Fourth': 11, 'Third': 8},
            {'Second': 6, 'First': 3, 'Fourth': 12, 'Third': 9, 'YYY': 14, 'XXX': 13},
        ]
        expected: DataFrame = DataFrame(
            {'First': [1, 2, 3], 'Second': [4, 5, 6], 'Third': [7, 8, 9], 'Fourth': [10, 11, 12], 'YYY': [None, None, 14], 'XXX': [None, None, 13]}
        )
        result: DataFrame = DataFrame(data)
        tm.assert_frame_equal(result, expected)

    def test_constructor_Series_named_and_columns(
        self, float_frame: DataFrame
    ) -> None:
        s0: Series = Series(range(5), name=0)
        s1: Series = Series(range(5), name=1)
        tm.assert_frame_equal(DataFrame(s0, columns=[0]), s0.to_frame())
        tm.assert_frame_equal(DataFrame(s1, columns=[1]), s1.to_frame())
        assert DataFrame(s0, columns=[1]).empty
        assert DataFrame(s1, columns=[0]).empty

    def test_constructor_Series_differently_indexed(
        self, float_frame: DataFrame
    ) -> None:
        s1: Series = Series([1, 2, 3], index=['a', 'b', 'c'], name='x')
        s2: Series = Series([1, 2, 3], index=['a', 'b', 'c'])
        other_index: Index = Index(['a', 'b'])
        df1: DataFrame = DataFrame(s1, index=other_index)
        exp1: DataFrame = DataFrame(s1.reindex(other_index))
        assert df1.columns[0] == 'x'
        tm.assert_frame_equal(df1, exp1)
        df2: DataFrame = DataFrame(s2, index=other_index)
        exp2: DataFrame = DataFrame(s2.reindex(other_index))
        assert df2.columns[0] == 0
        tm.assert_index_equal(df2.index, other_index)
        tm.assert_frame_equal(df2, exp2)

    def test_1d_object_array_does_not_copy(
        self, using_infer_string: bool
    ) -> None:
        arr: np.ndarray = np.array(['a', 'b'], dtype='object')
        df: DataFrame = DataFrame(arr, copy=False)
        if using_infer_string:
            if df[0].dtype.storage == 'pyarrow':
                pass
            else:
                assert np.shares_memory(df[0].to_numpy(), arr)
        else:
            assert np.shares_memory(df.values, arr)
        df = DataFrame(arr, dtype=object, copy=False)
        assert np.shares_memory(df.values, arr)

    def test_2d_object_array_does_not_copy(
        self, using_infer_string: bool
    ) -> None:
        arr: np.ndarray = np.array([['a', 'b'], ['c', 'd']], dtype='object')
        df: DataFrame = DataFrame(arr, copy=False)
        if using_infer_string:
            if df[0].dtype.storage == 'pyarrow':
                pass
            else:
                assert np.shares_memory(df[0].to_numpy(), arr)
        else:
            assert np.shares_memory(df.values, arr)
        df = DataFrame(arr, dtype=object, copy=False)
        assert np.shares_memory(df.values, arr)
