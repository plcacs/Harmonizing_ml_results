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
from typing import Any, Dict, List, Optional, Tuple, Union

MIXED_FLOAT_DTYPES: List[str] = ['float16', 'float32', 'float64']
MIXED_INT_DTYPES: List[str] = ['uint8', 'uint16', 'uint32', 'uint64', 'int8', 'int16', 'int32', 'int64']

class TestDataFrameConstructors:

    def test_constructor_from_ndarray_with_str_dtype(self) -> None:
        arr: np.ndarray = np.arange(12).reshape(4, 3)
        df: DataFrame = DataFrame(arr, dtype=str)
        expected: DataFrame = DataFrame(arr.astype(str), dtype='str')
        tm.assert_frame_equal(df, expected)

    def test_constructor_from_2d_datetimearray(self) -> None:
        dti: DatetimeIndex = date_range('2016-01-01', periods=6, tz='US/Pacific')
        dta: DatetimeArray = dti._data.reshape(3, 2)
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

    def test_array_of_dt64_nat_with_td64dtype_raises(self, frame_or_series: Any) -> None:
        nat: np.datetime64 = np.datetime64('NaT', 'ns')
        arr: np.ndarray = np.array([nat], dtype=object)
        if frame_or_series is DataFrame:
            arr = arr.reshape(1, 1)
        msg: str = "Invalid type for timedelta scalar: <class 'numpy.datetime64'>"
        with pytest.raises(TypeError, match=msg):
            frame_or_series(arr, dtype='m8[ns]')

    @pytest.mark.parametrize('kind', ['m', 'M'])
    def test_datetimelike_values_with_object_dtype(self, kind: str, frame_or_series: Any) -> None:
        if kind == 'M':
            dtype: str = 'M8[ns]'
            scalar_type: Any = Timestamp
        else:
            dtype = 'm8[ns]'
            scalar_type = Timedelta
        arr: np.ndarray = np.arange(6, dtype='i8').view(dtype).reshape(3, 2)
        if frame_or_series is Series:
            arr = arr[:, 0]
        obj: Any = frame_or_series(arr, dtype=object)
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

    @pytest.mark.parametrize('constructor', [lambda: DataFrame(), lambda: DataFrame(None), lambda: DataFrame(()), lambda: DataFrame([]), lambda: DataFrame((_ for _ in [])), lambda: DataFrame(range(0)), lambda: DataFrame(data=None), lambda: DataFrame(data=()), lambda: DataFrame(data=[]), lambda: DataFrame(data=(_ for _ in [])), lambda: DataFrame(data=range(0))])
    def test_empty_constructor(self, constructor: Any) -> None:
        expected: DataFrame = DataFrame()
        result: DataFrame = constructor()
        assert len(result.index) == 0
        assert len(result.columns) == 0
        tm.assert_frame_equal(result, expected)

    def test_empty_constructor_object_index(self) -> None:
        expected: DataFrame = DataFrame(index=RangeIndex(0), columns=RangeIndex(0))
        result: DataFrame = DataFrame({})
        assert len(result.index) == 0
        assert len(result.columns) == 0
        tm.assert_frame_equal(result, expected, check_index_type=True)

    @pytest.mark.parametrize('emptylike,expected_index,expected_columns', [([[]], RangeIndex(1), RangeIndex(0)), ([[], []], RangeIndex(2), RangeIndex(0)), ([(_ for _ in [])], RangeIndex(1), RangeIndex(0))])
    def test_emptylike_constructor(self, emptylike: Any, expected_index: Any, expected_columns: Any) -> None:
        expected: DataFrame = DataFrame(index=expected_index, columns=expected_columns)
        result: DataFrame = DataFrame(emptylike)
        tm.assert_frame_equal(result, expected)

    def test_constructor_mixed(self, float_string_frame: DataFrame, using_infer_string: bool) -> None:
        dtype: str = 'str' if using_infer_string else np.object_
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

    def test_1d_object_array_does_not_copy(self, using_infer_string: bool) -> None:
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

    def test_2d_object_array_does_not_copy(self, using_infer_string: bool) -> None:
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

    @pytest.mark.parametrize('typ, ad', [['float', {}], ['float', {'A': 1, 'B': 'foo', 'C': 'bar'}], ['int', {}]])
    def test_constructor_mixed_dtypes(self, typ: str, ad: Dict[str, Any]) -> None:
        if typ == 'int':
            dtypes: List[str] = MIXED_INT_DTYPES
            arrays: List[np.ndarray] = [np.array(np.random.default_rng(2).random(10), dtype=d) for d in dtypes]
        elif typ == 'float':
            dtypes = MIXED_FLOAT_DTYPES
            arrays = [np.array(np.random.default_rng(2).integers(10, size=10), dtype=d) for d in dtypes]
        for d, a in zip(dtypes, arrays):
            assert a.dtype == d
        ad.update(dict(zip(dtypes, arrays)))
        df: DataFrame = DataFrame(ad)
        dtypes = MIXED_FLOAT_DTYPES + MIXED_INT_DTYPES
        for d in dtypes:
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
        index: Index = float_frame.index
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
        values: np.ndarray = np.array([2 ** 64 - i for i in range(1, 10)], dtype=np.uint64)
        result: DataFrame = DataFrame({'a': values})
        assert result['a'].dtype == np.uint64
        data_scores: List[Tuple[int, int]] = [(6311132704823138710, 273), (2685045978526272070, 23), (8921811264899370420, 45), (17019687244989530680, 270), (9930107427299601010, 273)]
        dtype: List[Tuple[str, str]] = [('uid', 'u