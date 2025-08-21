from __future__ import annotations

import datetime
from datetime import timedelta
from io import StringIO
import json
import os
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import pytest
from pandas._config import using_string_dtype
from pandas.compat import IS64
import pandas.util._test_decorators as td
import pandas as pd
from pandas import NA, DataFrame, DatetimeIndex, Index, RangeIndex, Series, Timestamp, date_range, read_json
import pandas._testing as tm
from pandas.io.json import ujson_dumps


def test_literal_json_raises() -> None:
    jsonl = '{"a": 1, "b": 2}\n        {"a": 3, "b": 4}\n        {"a": 5, "b": 6}\n        {"a": 7, "b": 8}'
    msg = '.* does not exist'
    with pytest.raises(FileNotFoundError, match=msg):
        read_json(jsonl, lines=False)
    with pytest.raises(FileNotFoundError, match=msg):
        read_json('{"a": 1, "b": 2}\n{"b":2, "a" :1}\n', lines=True)
    with pytest.raises(FileNotFoundError, match=msg):
        read_json('{"a\\\\":"foo\\\\","b":"bar"}\n{"a\\\\":"foo\\"","b":"bar"}\n', lines=False)
    with pytest.raises(FileNotFoundError, match=msg):
        read_json('{"a": 1, "b": 2}\n{"b":2, "a" :1}\n', lines=False)


def assert_json_roundtrip_equal(result: DataFrame, expected: DataFrame, orient: str) -> None:
    if orient in ('records', 'values'):
        expected = expected.reset_index(drop=True)
    if orient == 'values':
        expected.columns = range(len(expected.columns))
    tm.assert_frame_equal(result, expected)


class TestPandasContainer:

    @pytest.fixture
    def datetime_series(self) -> Series:
        ser = Series(1.1 * np.arange(10, dtype=np.float64), index=date_range('2020-01-01', periods=10), name='ts')
        ser.index = ser.index._with_freq(None)
        return ser

    @pytest.fixture
    def datetime_frame(self) -> DataFrame:
        df = DataFrame(np.random.default_rng(2).standard_normal((30, 4)), columns=Index(list('ABCD')), index=date_range('2000-01-01', periods=30, freq='B'))
        df.index = df.index._with_freq(None)
        return df

    def test_frame_double_encoded_labels(self, orient: str) -> None:
        df = DataFrame([['a', 'b'], ['c', 'd']], index=['index " 1', 'index / 2'], columns=['a \\ b', 'y / z'])
        data = StringIO(df.to_json(orient=orient))
        result = read_json(data, orient=orient)
        expected = df.copy()
        assert_json_roundtrip_equal(result, expected, orient)

    @pytest.mark.parametrize('orient', ['split', 'records', 'values'])
    def test_frame_non_unique_index(self, orient: str) -> None:
        df = DataFrame([['a', 'b'], ['c', 'd']], index=[1, 1], columns=['x', 'y'])
        data = StringIO(df.to_json(orient=orient))
        result = read_json(data, orient=orient)
        expected = df.copy()
        assert_json_roundtrip_equal(result, expected, orient)

    @pytest.mark.parametrize('orient', ['index', 'columns'])
    def test_frame_non_unique_index_raises(self, orient: str) -> None:
        df = DataFrame([['a', 'b'], ['c', 'd']], index=[1, 1], columns=['x', 'y'])
        msg = f"DataFrame index must be unique for orient='{orient}'"
        with pytest.raises(ValueError, match=msg):
            df.to_json(orient=orient)

    @pytest.mark.parametrize('orient', ['split', 'values'])
    @pytest.mark.parametrize(
        'data',
        [
            [['a', 'b'], ['c', 'd']],
            [[1.5, 2.5], [3.5, 4.5]],
            [[1, 2.5], [3, 4.5]],
            [[Timestamp('20130101'), 3.5], [Timestamp('20130102'), 4.5]],
        ],
    )
    def test_frame_non_unique_columns(self, orient: str, data: List[List[Any]], request: pytest.FixtureRequest) -> None:
        if isinstance(data[0][0], Timestamp) and orient == 'split':
            mark = pytest.mark.xfail(reason='GH#55827 non-nanosecond dt64 fails to round-trip')
            request.applymarker(mark)
        df = DataFrame(data, index=[1, 2], columns=['x', 'x'])
        expected_warning: Optional[Type[Warning]] = None
        msg = "The default 'epoch' date format is deprecated and will be removed in a future version, please use 'iso' date format instead."
        if df.iloc[:, 0].dtype == 'datetime64[s]':
            expected_warning = FutureWarning
        with tm.assert_produces_warning(expected_warning, match=msg):
            result = read_json(StringIO(df.to_json(orient=orient)), orient=orient, convert_dates=['x'])
        if orient == 'values':
            expected = DataFrame(data)
            if expected.iloc[:, 0].dtype == 'datetime64[s]':
                expected.isetitem(0, expected.iloc[:, 0].astype(np.int64) // 1000000)
        elif orient == 'split':
            expected = df
            expected.columns = ['x', 'x.1']
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('orient', ['index', 'columns', 'records'])
    def test_frame_non_unique_columns_raises(self, orient: str) -> None:
        df = DataFrame([['a', 'b'], ['c', 'd']], index=[1, 2], columns=['x', 'x'])
        msg = f"DataFrame columns must be unique for orient='{orient}'"
        with pytest.raises(ValueError, match=msg):
            df.to_json(orient=orient)

    def test_frame_default_orient(self, float_frame: DataFrame) -> None:
        assert float_frame.to_json() == float_frame.to_json(orient='columns')

    @pytest.mark.parametrize('dtype', [False, float])
    @pytest.mark.parametrize('convert_axes', [True, False])
    def test_roundtrip_simple(self, orient: str, convert_axes: bool, dtype: Any, float_frame: DataFrame) -> None:
        data = StringIO(float_frame.to_json(orient=orient))
        result = read_json(data, orient=orient, convert_axes=convert_axes, dtype=dtype)
        expected = float_frame
        assert_json_roundtrip_equal(result, expected, orient)

    @pytest.mark.parametrize('dtype', [False, np.int64])
    @pytest.mark.parametrize('convert_axes', [True, False])
    def test_roundtrip_intframe(self, orient: str, convert_axes: bool, dtype: Any, int_frame: DataFrame) -> None:
        data = StringIO(int_frame.to_json(orient=orient))
        result = read_json(data, orient=orient, convert_axes=convert_axes, dtype=dtype)
        expected = int_frame
        assert_json_roundtrip_equal(result, expected, orient)

    @pytest.mark.parametrize('dtype', [None, np.float64, int, 'U3'])
    @pytest.mark.parametrize('convert_axes', [True, False])
    def test_roundtrip_str_axes(self, orient: str, convert_axes: bool, dtype: Any) -> None:
        df = DataFrame(np.zeros((200, 4)), columns=[str(i) for i in range(4)], index=[str(i) for i in range(200)], dtype=dtype)
        data = StringIO(df.to_json(orient=orient))
        result = read_json(data, orient=orient, convert_axes=convert_axes, dtype=dtype)
        expected = df.copy()
        if not dtype:
            expected = expected.astype(np.int64)
        if convert_axes and orient in ('index', 'columns'):
            expected.columns = expected.columns.astype(np.int64)
            expected.index = expected.index.astype(np.int64)
        elif orient == 'records' and convert_axes:
            expected.columns = expected.columns.astype(np.int64)
        elif convert_axes and orient == 'split':
            expected.columns = expected.columns.astype(np.int64)
        assert_json_roundtrip_equal(result, expected, orient)

    @pytest.mark.parametrize('convert_axes', [True, False])
    def test_roundtrip_categorical(self, request: pytest.FixtureRequest, orient: str, convert_axes: bool, using_infer_string: bool) -> None:
        if orient in ('index', 'columns'):
            request.applymarker(pytest.mark.xfail(reason=f"Can't have duplicate index values for orient '{orient}')"))
        data: Dict[str, Any] = {c: np.random.default_rng(i).standard_normal(30) for i, c in enumerate(list('ABCD'))}
        cat = ['bah'] * 5 + ['bar'] * 5 + ['baz'] * 5 + ['foo'] * 15
        data['E'] = list(reversed(cat))
        data['sort'] = np.arange(30, dtype='int64')
        categorical_frame = DataFrame(data, index=pd.CategoricalIndex(cat, name='E'))
        data_io = StringIO(categorical_frame.to_json(orient=orient))
        result = read_json(data_io, orient=orient, convert_axes=convert_axes)
        expected = categorical_frame.copy()
        expected.index = expected.index.astype(str if not using_infer_string else 'str')
        expected.index.name = None
        assert_json_roundtrip_equal(result, expected, orient)

    @pytest.mark.parametrize('convert_axes', [True, False])
    def test_roundtrip_empty(self, orient: str, convert_axes: bool) -> None:
        empty_frame = DataFrame()
        data = StringIO(empty_frame.to_json(orient=orient))
        result = read_json(data, orient=orient, convert_axes=convert_axes)
        if orient == 'split':
            idx = Index([], dtype=float if convert_axes else object)
            expected = DataFrame(index=idx, columns=idx)
        elif orient in ['index', 'columns']:
            expected = DataFrame()
        else:
            expected = empty_frame.copy()
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('convert_axes', [True, False])
    def test_roundtrip_timestamp(self, orient: str, convert_axes: bool, datetime_frame: DataFrame) -> None:
        data = StringIO(datetime_frame.to_json(orient=orient))
        result = read_json(data, orient=orient, convert_axes=convert_axes)
        expected = datetime_frame.copy()
        if not convert_axes:
            idx = expected.index.view(np.int64) // 1000000
            if orient != 'split':
                idx = idx.astype(str)
            expected.index = idx
        assert_json_roundtrip_equal(result, expected, orient)

    @pytest.mark.parametrize('convert_axes', [True, False])
    def test_roundtrip_mixed(self, orient: str, convert_axes: bool) -> None:
        index = Index(['a', 'b', 'c', 'd', 'e'])
        values: Dict[str, List[Any]] = {'A': [0.0, 1.0, 2.0, 3.0, 4.0], 'B': [0.0, 1.0, 0.0, 1.0, 0.0], 'C': ['foo1', 'foo2', 'foo3', 'foo4', 'foo5'], 'D': [True, False, True, False, True]}
        df = DataFrame(data=values, index=index)
        data = StringIO(df.to_json(orient=orient))
        result = read_json(data, orient=orient, convert_axes=convert_axes)
        expected = df.copy()
        expected = expected.assign(**expected.select_dtypes('number').astype(np.int64))
        assert_json_roundtrip_equal(result, expected, orient)

    @pytest.mark.xfail(reason='#50456 Column multiindex is stored and loaded differently', raises=AssertionError)
    @pytest.mark.parametrize('columns', [[['2022', '2022'], ['JAN', 'FEB']], [['2022', '2023'], ['JAN', 'JAN']], [['2022', '2022'], ['JAN', 'JAN']]])
    def test_roundtrip_multiindex(self, columns: List[List[str]]) -> None:
        df = DataFrame([[1, 2], [3, 4]], columns=pd.MultiIndex.from_arrays(columns))
        data = StringIO(df.to_json(orient='split'))
        result = read_json(data, orient='split')
        tm.assert_frame_equal(result, df)

    @pytest.mark.parametrize(
        'data,msg,orient',
        [
            ('{"key":b:a:d}', 'Expected object or value', 'columns'),
            (
                '{"columns":["A","B"],"index":["2","3"],"data":[[1.0,"1"],[2.0,"2"],[null,"3"]]}',
                '|'.join(['Length of values \\(3\\) does not match length of index \\(2\\)']),
                'split',
            ),
            (
                '{"columns":["A","B","C"],"index":["1","2","3"],"data":[[1.0,"1"],[2.0,"2"],[null,"3"]]}',
                '3 columns passed, passed data had 2 columns',
                'split',
            ),
            (
                '{"badkey":["A","B"],"index":["2","3"],"data":[[1.0,"1"],[2.0,"2"],[null,"3"]]}',
                'unexpected key\\(s\\): badkey',
                'split',
            ),
        ],
    )
    def test_frame_from_json_bad_data_raises(self, data: str, msg: str, orient: str) -> None:
        with pytest.raises(ValueError, match=msg):
            read_json(StringIO(data), orient=orient)

    @pytest.mark.parametrize('dtype', [True, False])
    @pytest.mark.parametrize('convert_axes', [True, False])
    def test_frame_from_json_missing_data(self, orient: str, convert_axes: bool, dtype: Union[bool, type, None]) -> None:
        num_df = DataFrame([[1, 2], [4, 5, 6]])
        result = read_json(StringIO(num_df.to_json(orient=orient)), orient=orient, convert_axes=convert_axes, dtype=dtype)
        assert np.isnan(result.iloc[0, 2])
        obj_df = DataFrame([['1', '2'], ['4', '5', '6']])
        result = read_json(StringIO(obj_df.to_json(orient=orient)), orient=orient, convert_axes=convert_axes, dtype=dtype)
        assert np.isnan(result.iloc[0, 2])

    @pytest.mark.parametrize('dtype', [True, False])
    def test_frame_read_json_dtype_missing_value(self, dtype: Union[bool, None]) -> None:
        result = read_json(StringIO('[null]'), dtype=dtype)
        expected = DataFrame([np.nan], dtype=object if not dtype else None)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('inf', [np.inf, -np.inf])
    @pytest.mark.parametrize('dtype', [True, False])
    def test_frame_infinity(self, inf: float, dtype: Union[bool, None]) -> None:
        df = DataFrame([[1, 2], [4, 5, 6]])
        df.loc[0, 2] = inf
        data = StringIO(df.to_json())
        result = read_json(data, dtype=dtype)
        assert np.isnan(result.iloc[0, 2])

    @pytest.mark.skipif(not IS64, reason='not compliant on 32-bit, xref #15865')
    @pytest.mark.parametrize('value,precision,expected_val', [(0.95, 1, 1.0), (1.95, 1, 2.0), (-1.95, 1, -2.0), (0.995, 2, 1.0), (0.9995, 3, 1.0), (0.9999999999999994, 15, 1.0)])
    def test_frame_to_json_float_precision(self, value: float, precision: int, expected_val: float) -> None:
        df = DataFrame([{'a_float': value}])
        encoded = df.to_json(double_precision=precision)
        assert encoded == f'{{"a_float":{{"0":{expected_val}}}}}'

    def test_frame_to_json_except(self) -> None:
        df = DataFrame([1, 2, 3])
        msg = "Invalid value 'garbage' for option 'orient'"
        with pytest.raises(ValueError, match=msg):
            df.to_json(orient='garbage')

    def test_frame_empty(self) -> None:
        df = DataFrame(columns=['jim', 'joe'])
        assert not df._is_mixed_type
        data = StringIO(df.to_json())
        result = read_json(data, dtype=dict(df.dtypes))
        tm.assert_frame_equal(result, df, check_index_type=False)

    def test_frame_empty_to_json(self) -> None:
        df = DataFrame({'test': []}, index=[])
        result = df.to_json(orient='columns')
        expected = '{"test":{}}'
        assert result == expected

    def test_frame_empty_mixedtype(self) -> None:
        df = DataFrame(columns=['jim', 'joe'])
        df['joe'] = df['joe'].astype('i8')
        assert df._is_mixed_type
        data = df.to_json()
        tm.assert_frame_equal(read_json(StringIO(data), dtype=dict(df.dtypes)), df, check_index_type=False)

    def test_frame_mixedtype_orient(self) -> None:
        vals = [[10, 1, 'foo', 0.1, 0.01], [20, 2, 'bar', 0.2, 0.02], [30, 3, 'baz', 0.3, 0.03], [40, 4, 'qux', 0.4, 0.04]]
        df = DataFrame(vals, index=list('abcd'), columns=['1st', '2nd', '3rd', '4th', '5th'])
        assert df._is_mixed_type
        right = df.copy()
        for orient in ['split', 'index', 'columns']:
            inp = StringIO(df.to_json(orient=orient))
            left = read_json(inp, orient=orient, convert_axes=False)
            tm.assert_frame_equal(left, right)
        right.index = RangeIndex(len(df))
        inp = StringIO(df.to_json(orient='records'))
        left = read_json(inp, orient='records', convert_axes=False)
        tm.assert_frame_equal(left, right)
        right.columns = RangeIndex(df.shape[1])
        inp = StringIO(df.to_json(orient='values'))
        left = read_json(inp, orient='values', convert_axes=False)
        tm.assert_frame_equal(left, right)

    def test_v12_compat(self, datapath: Callable[..., str]) -> None:
        dti = date_range('2000-01-03', '2000-01-07')
        dti = DatetimeIndex(np.asarray(dti), freq=None)
        df = DataFrame([[1.56808523, 0.65727391, 1.81021139, -0.17251653], [-0.2550111, -0.08072427, -0.03202878, -0.17581665], [1.51493992, 0.11805825, 1.629455, -1.31506612], [-0.02765498, 0.44679743, 0.33192641, -0.27885413], [0.05951614, -2.69652057, 1.28163262, 0.34703478]], columns=['A', 'B', 'C', 'D'], index=dti)
        df['date'] = Timestamp('19920106 18:21:32.12').as_unit('ns')
        df.iloc[3, df.columns.get_loc('date')] = Timestamp('20130101')
        df['modified'] = df['date']
        df.iloc[1, df.columns.get_loc('modified')] = pd.NaT
        dirpath = datapath('io', 'json', 'data')
        v12_json = os.path.join(dirpath, 'tsframe_v012.json')
        df_unser = read_json(v12_json)
        tm.assert_frame_equal(df, df_unser)
        df_iso = df.drop(['modified'], axis=1)
        v12_iso_json = os.path.join(dirpath, 'tsframe_iso_v012.json')
        df_unser_iso = read_json(v12_iso_json)
        tm.assert_frame_equal(df_iso, df_unser_iso, check_column_type=False)

    def test_blocks_compat_GH9037(self, using_infer_string: bool) -> None:
        index = date_range('20000101', periods=10, freq='h')
        index = DatetimeIndex(list(index), freq=None)
        df_mixed = DataFrame({'float_1': [-0.92077639, 0.77434435, 1.25234727, 0.61485564, -0.60316077, 0.24653374, 0.28668979, -2.51969012, 0.95748401, -1.02970536], 'int_1': [19680418, 75337055, 99973684, 65103179, 79373900, 40314334, 21290235, 4991321, 41903419, 16008365], 'str_1': ['78c608f1', '64a99743', '13d2ff52', 'ca7f4af2', '97236474', 'bde7e214', '1a6bde47', 'b1190be5', '7a669144', '8d64d068'], 'float_2': [-0.0428278, -1.80872357, 3.36042349, -0.7573685, -0.48217572, 0.86229683, 1.08935819, 0.93898739, -0.03030452, 1.43366348], 'str_2': ['14f04af9', 'd085da90', '4bcfac83', '81504caf', '2ffef4a9', '08e2f5c4', '07e1af03', 'addbd4a7', '1f6a09ba', '4bfc4d87'], 'int_2': [86967717, 98098830, 51927505, 20372254, 12601730, 20884027, 34193846, 10561746, 24867120, 76131025]}, index=index)
        df_mixed.columns = df_mixed.columns.astype(np.str_ if not using_infer_string else 'str')
        data = StringIO(df_mixed.to_json(orient='split'))
        df_roundtrip = read_json(data, orient='split')
        tm.assert_frame_equal(df_mixed, df_roundtrip, check_index_type=True, check_column_type=True, by_blocks=True, check_exact=True)

    def test_frame_nonprintable_bytes(self) -> None:

        class BinaryThing:

            def __init__(self, hexed: str) -> None:
                self.hexed = hexed
                self.binary = bytes.fromhex(hexed)

            def __str__(self) -> str:
                return self.hexed
        hexed = '574b4454ba8c5eb4f98a8f45'
        binthing = BinaryThing(hexed)
        df_printable = DataFrame({'A': [binthing.hexed]})
        assert df_printable.to_json() == f'{{"A":{{"0":"{hexed}"}}}}'
        df_nonprintable = DataFrame({'A': [binthing]})
        msg = 'Unsupported UTF-8 sequence length when encoding string'
        with pytest.raises(OverflowError, match=msg):
            df_nonprintable.to_json()
        df_mixed = DataFrame({'A': [binthing], 'B': [1]}, columns=['A', 'B'])
        with pytest.raises(OverflowError, match=msg):
            df_mixed.to_json()
        result = df_nonprintable.to_json(default_handler=str)
        expected = f'{{"A":{{"0":"{hexed}"}}}}'
        assert result == expected
        assert df_mixed.to_json(default_handler=str) == f'{{"A":{{"0":"{hexed}"}}},"B":{{"0":1}}}}'

    def test_label_overflow(self) -> None:
        result = DataFrame({'bar' * 100000: [1], 'foo': [1337]}).to_json()
        expected = f'{{"{'bar' * 100000}":{{"0":1}},"foo":{{"0":1337}}}}'
        assert result == expected

    def test_series_non_unique_index(self) -> None:
        s = Series(['a', 'b'], index=[1, 1])
        msg = "Series index must be unique for orient='index'"
        with pytest.raises(ValueError, match=msg):
            s.to_json(orient='index')
        tm.assert_series_equal(s, read_json(StringIO(s.to_json(orient='split')), orient='split', typ='series'))
        unserialized = read_json(StringIO(s.to_json(orient='records')), orient='records', typ='series')
        tm.assert_equal(s.values, unserialized.values)

    def test_series_default_orient(self, string_series: Series) -> None:
        assert string_series.to_json() == string_series.to_json(orient='index')

    def test_series_roundtrip_simple(self, orient: str, string_series: Series, using_infer_string: bool) -> None:
        data = StringIO(string_series.to_json(orient=orient))
        result = read_json(data, typ='series', orient=orient)
        expected = string_series
        if using_infer_string and orient in ('split', 'index', 'columns'):
            expected.index = expected.index.astype('str')
        if orient in ('values', 'records'):
            expected = expected.reset_index(drop=True)
        if orient != 'split':
            expected.name = None
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('dtype', [False, None])
    def test_series_roundtrip_object(self, orient: str, dtype: Union[bool, None], object_series: Series) -> None:
        data = StringIO(object_series.to_json(orient=orient))
        result = read_json(data, typ='series', orient=orient, dtype=dtype)
        expected = object_series
        if orient in ('values', 'records'):
            expected = expected.reset_index(drop=True)
        if orient != 'split':
            expected.name = None
        if using_string_dtype():
            expected = expected.astype('str')
        tm.assert_series_equal(result, expected)

    def test_series_roundtrip_empty(self, orient: str) -> None:
        empty_series = Series([], index=[], dtype=np.float64)
        data = StringIO(empty_series.to_json(orient=orient))
        result = read_json(data, typ='series', orient=orient)
        expected = empty_series.reset_index(drop=True)
        if orient in 'split':
            expected.index = expected.index.astype(np.float64)
        tm.assert_series_equal(result, expected)

    def test_series_roundtrip_timeseries(self, orient: str, datetime_series: Series) -> None:
        data = StringIO(datetime_series.to_json(orient=orient))
        result = read_json(data, typ='series', orient=orient)
        expected = datetime_series
        if orient in ('values', 'records'):
            expected = expected.reset_index(drop=True)
        if orient != 'split':
            expected.name = None
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('dtype', [np.float64, int])
    def test_series_roundtrip_numeric(self, orient: str, dtype: Any) -> None:
        s = Series(range(6), index=['a', 'b', 'c', 'd', 'e', 'f'])
        data = StringIO(s.to_json(orient=orient))
        result = read_json(data, typ='series', orient=orient)
        expected = s.copy()
        if orient in ('values', 'records'):
            expected = expected.reset_index(drop=True)
        tm.assert_series_equal(result, expected)

    def test_series_to_json_except(self) -> None:
        s = Series([1, 2, 3])
        msg = "Invalid value 'garbage' for option 'orient'"
        with pytest.raises(ValueError, match=msg):
            s.to_json(orient='garbage')

    def test_series_from_json_precise_float(self) -> None:
        s = Series([4.56, 4.56, 4.56])
        result = read_json(StringIO(s.to_json()), typ='series', precise_float=True)
        tm.assert_series_equal(result, s, check_index_type=False)

    def test_series_with_dtype(self) -> None:
        s = Series([4.56, 4.56, 4.56])
        result = read_json(StringIO(s.to_json()), typ='series', dtype=np.int64)
        expected = Series([4] * 3)
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('dtype,expected', [(True, Series(['2000-01-01'], dtype='datetime64[ns]')), (False, Series([946684800000]))])
    def test_series_with_dtype_datetime(self, dtype: bool, expected: Series) -> None:
        s = Series(['2000-01-01'], dtype='datetime64[ns]')
        msg = "The default 'epoch' date format is deprecated and will be removed in a future version, please use 'iso' date format instead."
        with tm.assert_produces_warning(FutureWarning, match=msg):
            data = StringIO(s.to_json())
        result = read_json(data, typ='series', dtype=dtype)
        tm.assert_series_equal(result, expected)

    def test_frame_from_json_precise_float(self) -> None:
        df = DataFrame([[4.56, 4.56, 4.56], [4.56, 4.56, 4.56]])
        result = read_json(StringIO(df.to_json()), precise_float=True)
        tm.assert_frame_equal(result, df)

    def test_typ(self) -> None:
        s = Series(range(6), index=['a', 'b', 'c', 'd', 'e', 'f'], dtype='int64')
        result = read_json(StringIO(s.to_json()), typ='series')
        tm.assert_series_equal(result, s)

    def test_reconstruction_index(self) -> None:
        df = DataFrame([[1, 2, 3], [4, 5, 6]])
        result = read_json(StringIO(df.to_json()))
        tm.assert_frame_equal(result, df)
        df = DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]}, index=['A', 'B', 'C'])
        result = read_json(StringIO(df.to_json()))
        tm.assert_frame_equal(result, df)

    def test_path(self, float_frame: DataFrame, int_frame: DataFrame, datetime_frame: DataFrame) -> None:
        with tm.ensure_clean('test.json') as path:
            for df in [float_frame, int_frame, datetime_frame]:
                df.to_json(path)
                read_json(path)

    def test_axis_dates(self, datetime_series: Series, datetime_frame: DataFrame) -> None:
        json_io = StringIO(datetime_frame.to_json())
        result = read_json(json_io)
        tm.assert_frame_equal(result, datetime_frame)
        json_io = StringIO(datetime_series.to_json())
        result = read_json(json_io, typ='series')
        tm.assert_series_equal(result, datetime_series, check_names=False)
        assert result.name is None

    def test_convert_dates(self, datetime_series: Series, datetime_frame: DataFrame) -> None:
        df = datetime_frame
        df['date'] = Timestamp('20130101').as_unit('ns')
        msg = "The default 'epoch' date format is deprecated and will be removed in a future version, please use 'iso' date format instead."
        with tm.assert_produces_warning(FutureWarning, match=msg):
            json_io = StringIO(df.to_json())
        result = read_json(json_io)
        tm.assert_frame_equal(result, df)
        df['foo'] = 1.0
        with tm.assert_produces_warning(FutureWarning, match=msg):
            json_io = StringIO(df.to_json(date_unit='ns'))
        result = read_json(json_io, convert_dates=False)
        expected = df.copy()
        expected['date'] = expected['date'].values.view('i8')
        expected['foo'] = expected['foo'].astype('int64')
        tm.assert_frame_equal(result, expected)
        ts = Series(Timestamp('20130101').as_unit('ns'), index=datetime_series.index)
        with tm.assert_produces_warning(FutureWarning, match=msg):
            json_io = StringIO(ts.to_json())
        result = read_json(json_io, typ='series')
        tm.assert_series_equal(result, ts)

    @pytest.mark.parametrize('date_format', ['epoch', 'iso'])
    @pytest.mark.parametrize('as_object', [True, False])
    @pytest.mark.parametrize('date_typ', [datetime.date, datetime.datetime, Timestamp])
    def test_date_index_and_values(self, date_format: str, as_object: bool, date_typ: type) -> None:
        data = [date_typ(year=2020, month=1, day=1), pd.NaT]
        if as_object:
            data.append('a')
        ser = Series(data, index=data)
        if not as_object:
            ser = ser.astype('M8[ns]')
            if isinstance(ser.index, DatetimeIndex):
                ser.index = ser.index.as_unit('ns')
        expected_warning: Optional[Type[Warning]] = None
        if date_format == 'epoch':
            expected = '{"1577836800000":1577836800000,"null":null}'
            expected_warning = FutureWarning
        else:
            expected = '{"2020-01-01T00:00:00.000":"2020-01-01T00:00:00.000","null":null}'
        msg = "'epoch' date format is deprecated and will be removed in a future version, please use 'iso' date format instead."
        with tm.assert_produces_warning(expected_warning, match=msg):
            result = ser.to_json(date_format=date_format)
        if as_object:
            expected = expected.replace('}', ',"a":"a"}')
        assert result == expected

    @pytest.mark.parametrize('infer_word', ['trade_time', 'date', 'datetime', 'sold_at', 'modified', 'timestamp', 'timestamps'])
    def test_convert_dates_infer(self, infer_word: str) -> None:
        data = [{'id': 1, infer_word: 1036713600000}, {'id': 2}]
        expected = DataFrame([[1, Timestamp('2002-11-08')], [2, pd.NaT]], columns=['id', infer_word])
        expected[infer_word] = expected[infer_word].astype('M8[ns]')
        result = read_json(StringIO(ujson_dumps(data)))[['id', infer_word]]
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('date,date_unit', [('20130101 20:43:42.123', None), ('20130101 20:43:42', 's'), ('20130101 20:43:42.123', 'ms'), ('20130101 20:43:42.123456', 'us'), ('20130101 20:43:42.123456789', 'ns')])
    def test_date_format_frame(self, date: str, date_unit: Optional[str], datetime_frame: DataFrame) -> None:
        df = datetime_frame
        df['date'] = Timestamp(date).as_unit('ns')
        df.iloc[1, df.columns.get_loc('date')] = pd.NaT
        df.iloc[5, df.columns.get_loc('date')] = pd.NaT
        if date_unit:
            json_str = df.to_json(date_format='iso', date_unit=date_unit)
        else:
            json_str = df.to_json(date_format='iso')
        result = read_json(StringIO(json_str))
        expected = df.copy()
        tm.assert_frame_equal(result, expected)

    def test_date_format_frame_raises(self, datetime_frame: DataFrame) -> None:
        df = datetime_frame
        msg = "Invalid value 'foo' for option 'date_unit'"
        with pytest.raises(ValueError, match=msg):
            df.to_json(date_format='iso', date_unit='foo')

    @pytest.mark.parametrize('date,date_unit', [('20130101 20:43:42.123', None), ('20130101 20:43:42', 's'), ('20130101 20:43:42.123', 'ms'), ('20130101 20:43:42.123456', 'us'), ('20130101 20:43:42.123456789', 'ns')])
    def test_date_format_series(self, date: str, date_unit: Optional[str], datetime_series: Series) -> None:
        ts = Series(Timestamp(date).as_unit('ns'), index=datetime_series.index)
        ts.iloc[1] = pd.NaT
        ts.iloc[5] = pd.NaT
        if date_unit:
            json_str = ts.to_json(date_format='iso', date_unit=date_unit)
        else:
            json_str = ts.to_json(date_format='iso')
        result = read_json(StringIO(json_str), typ='series')
        expected = ts.copy()
        tm.assert_series_equal(result, expected)

    def test_date_format_series_raises(self, datetime_series: Series) -> None:
        ts = Series(Timestamp('20130101 20:43:42.123'), index=datetime_series.index)
        msg = "Invalid value 'foo' for option 'date_unit'"
        with pytest.raises(ValueError, match=msg):
            ts.to_json(date_format='iso', date_unit='foo')

    def test_date_unit(self, unit: str, datetime_frame: DataFrame) -> None:
        df = datetime_frame
        df['date'] = Timestamp('20130101 20:43:42').as_unit('ns')
        dl = df.columns.get_loc('date')
        df.iloc[1, dl] = Timestamp('19710101 20:43:42')
        df.iloc[2, dl] = Timestamp('21460101 20:43:42')
        df.iloc[4, dl] = pd.NaT
        msg = "'epoch' date format is deprecated and will be removed in a future version, please use 'iso' date format instead."
        with tm.assert_produces_warning(FutureWarning, match=msg):
            json_str = df.to_json(date_format='epoch', date_unit=unit)
        result = read_json(StringIO(json_str), date_unit=unit)
        tm.assert_frame_equal(result, df)
        result = read_json(StringIO(json_str), date_unit=None)
        tm.assert_frame_equal(result, df)

    @pytest.mark.parametrize('df, warn', [(DataFrame({'A': ['a', 'b', 'c'], 'B': np.arange(3)}), None), (DataFrame({'A': [True, False, False]}), None), (DataFrame({'A': ['a', 'b', 'c'], 'B': pd.to_timedelta(np.arange(3), unit='D')}), FutureWarning), (DataFrame({'A': pd.to_datetime(['2020-01-01', '2020-02-01', '2020-03-01'])}), FutureWarning)])
    def test_default_epoch_date_format_deprecated(self, df: DataFrame, warn: Optional[Type[Warning]]) -> None:
        msg = "The default 'epoch' date format is deprecated and will be removed in a future version, please use 'iso' date format instead."
        with tm.assert_produces_warning(warn, match=msg):
            df.to_json()

    @pytest.mark.parametrize('unit', ['s', 'ms', 'us'])
    def test_iso_non_nano_datetimes(self, unit: str) -> None:
        index = DatetimeIndex([np.datetime64('2023-01-01T11:22:33.123456', unit)], dtype=f'datetime64[{unit}]')
        df = DataFrame({'date': Series([np.datetime64('2022-01-01T11:22:33.123456', unit)], dtype=f'datetime64[{unit}]', index=index), 'date_obj': Series([np.datetime64('2023-01-01T11:22:33.123456', unit)], dtype=object, index=index)})
        buf = StringIO()
        df.to_json(buf, date_format='iso', date_unit=unit)
        buf.seek(0)
        tm.assert_frame_equal(read_json(buf, convert_dates=['date', 'date_obj']), df, check_index_type=False, check_dtype=False)

    def test_weird_nested_json(self) -> None:
        s = '{\n        "status": "success",\n        "data": {\n        "posts": [\n            {\n            "id": 1,\n            "title": "A blog post",\n            "body": "Some useful content"\n            },\n            {\n            "id": 2,\n            "title": "Another blog post",\n            "body": "More content"\n            }\n           ]\n          }\n        }'
        read_json(StringIO(s))

    def test_doc_example(self) -> None:
        dfj2 = DataFrame(np.random.default_rng(2).standard_normal((5, 2)), columns=list('AB'))
        dfj2['date'] = Timestamp('20130101')
        dfj2['ints'] = range(5)
        dfj2['bools'] = True
        dfj2.index = date_range('20130101', periods=5)
        msg = "The default 'epoch' date format is deprecated and will be removed in a future version, please use 'iso' date format instead."
        with tm.assert_produces_warning(FutureWarning, match=msg):
            json_io = StringIO(dfj2.to_json())
        result = read_json(json_io, dtype={'ints': np.int64, 'bools': np.bool_})
        tm.assert_frame_equal(result, result)

    def test_round_trip_exception(self, datapath: Callable[..., str]) -> None:
        path = datapath('io', 'json', 'data', 'teams.csv')
        df = pd.read_csv(path)
        s = df.to_json()
        result = read_json(StringIO(s))
        res = result.reindex(index=df.index, columns=df.columns)
        res = res.fillna(np.nan)
        tm.assert_frame_equal(res, df)

    @pytest.mark.network
    @pytest.mark.single_cpu
    @pytest.mark.parametrize('field,dtype', [['created_at', pd.DatetimeTZDtype(tz='UTC')], ['closed_at', 'datetime64[ns]'], ['updated_at', pd.DatetimeTZDtype(tz='UTC')]])
    def test_url(self, field: str, dtype: Any, httpserver: Any) -> None:
        data = '{"created_at": ["2023-06-23T18:21:36Z"], "closed_at": ["2023-06-23T18:21:36"], "updated_at": ["2023-06-23T18:21:36Z"]}\n'
        httpserver.serve_content(content=data)
        result = read_json(httpserver.url, convert_dates=True)
        assert result[field].dtype == dtype

    def test_timedelta(self) -> None:
        converter = lambda x: pd.to_timedelta(x, unit='ms')
        ser = Series([timedelta(23), timedelta(seconds=5)])
        assert ser.dtype == 'timedelta64[ns]'
        msg = "The default 'epoch' date format is deprecated and will be removed in a future version, please use 'iso' date format instead."
        with tm.assert_produces_warning(FutureWarning, match=msg):
            result = read_json(StringIO(ser.to_json()), typ='series').apply(converter)
        tm.assert_series_equal(result, ser)
        ser = Series([timedelta(23), timedelta(seconds=5)], index=Index([0, 1]))
        assert ser.dtype == 'timedelta64[ns]'
        with tm.assert_produces_warning(FutureWarning, match=msg):
            result = read_json(StringIO(ser.to_json()), typ='series').apply(converter)
        tm.assert_series_equal(result, ser)
        frame = DataFrame([timedelta(23), timedelta(seconds=5)])
        assert frame[0].dtype == 'timedelta64[ns]'
        with tm.assert_produces_warning(FutureWarning, match=msg):
            json_str = frame.to_json()
        tm.assert_frame_equal(frame, read_json(StringIO(json_str)).apply(converter))

    def test_timedelta2(self) -> None:
        frame = DataFrame({'a': [timedelta(days=23), timedelta(seconds=5)], 'b': [1, 2], 'c': date_range(start='20130101', periods=2)})
        msg = "The default 'epoch' date format is deprecated and will be removed in a future version, please use 'iso' date format instead."
        with tm.assert_produces_warning(FutureWarning, match=msg):
            data = StringIO(frame.to_json(date_unit='ns'))
        result = read_json(data)
        result['a'] = pd.to_timedelta(result.a, unit='ns')
        result['c'] = pd.to_datetime(result.c)
        tm.assert_frame_equal(frame, result)

    def test_mixed_timedelta_datetime(self) -> None:
        td = timedelta(23)
        ts = Timestamp('20130101')
        frame = DataFrame({'a': [td, ts]}, dtype=object)
        expected = DataFrame({'a': [pd.Timedelta(td).as_unit('ns')._value, ts.as_unit('ns')._value]})
        data = StringIO(frame.to_json(date_unit='ns'))
        result = read_json(data, dtype={'a': 'int64'})
        tm.assert_frame_equal(result, expected, check_index_type=False)

    @pytest.mark.parametrize('as_object', [True, False])
    @pytest.mark.parametrize('date_format', ['iso', 'epoch'])
    @pytest.mark.parametrize('timedelta_typ', [pd.Timedelta, timedelta])
    def test_timedelta_to_json(self, as_object: bool, date_format: str, timedelta_typ: type) -> None:
        data = [timedelta_typ(days=1), timedelta_typ(days=2), pd.NaT]
        if as_object:
            data.append('a')
        ser = Series(data, index=data)
        expected_warning: Optional[Type[Warning]] = None
        if date_format == 'iso':
            expected = '{"P1DT0H0M0S":"P1DT0H0M0S","P2DT0H0M0S":"P2DT0H0M0S","null":null}'
        else:
            expected_warning = FutureWarning
            expected = '{"86400000":86400000,"172800000":172800000,"null":null}'
        if as_object:
            expected = expected.replace('}', ',"a":"a"}')
        msg = "'epoch' date format is deprecated and will be removed in a future version, please use 'iso' date format instead."
        with tm.assert_produces_warning(expected_warning, match=msg):
            result = ser.to_json(date_format=date_format)
        assert result == expected

    @pytest.mark.parametrize('as_object', [True, False])
    @pytest.mark.parametrize('timedelta_typ', [pd.Timedelta, timedelta])
    def test_timedelta_to_json_fractional_precision(self, as_object: bool, timedelta_typ: type) -> None:
        data = [timedelta_typ(milliseconds=42)]
        ser = Series(data, index=data)
        warn: Optional[Type[Warning]] = FutureWarning
        if as_object:
            ser = ser.astype(object)
            warn = None
        msg = "The default 'epoch' date format is deprecated and will be removed in a future version, please use 'iso' date format instead."
        with tm.assert_produces_warning(warn, match=msg):
            result = ser.to_json()
        expected = '{"42":42}'
        assert result == expected

    def test_default_handler(self) -> None:
        value = object()
        frame = DataFrame({'a': [7, value]})
        expected = DataFrame({'a': [7, str(value)]})
        result = read_json(StringIO(frame.to_json(default_handler=str)))
        tm.assert_frame_equal(expected, result, check_index_type=False)

    def test_default_handler_indirect(self) -> None:

        def default(obj: Any) -> Any:
            if isinstance(obj, complex):
                return [('mathjs', 'Complex'), ('re', obj.real), ('im', obj.imag)]
            return str(obj)
        df_list: List[Any] = [9, DataFrame({'a': [1, 'STR', complex(4, -5)], 'b': [float('nan'), None, 'N/A']}, columns=['a', 'b'])]
        expected = '[9,[[1,null],["STR",null],[[["mathjs","Complex"],["re",4.0],["im",-5.0]],"N\\/A"]]]'
        assert ujson_dumps(df_list, default_handler=default, orient='values') == expected

    def test_default_handler_numpy_unsupported_dtype(self) -> None:
        df = DataFrame({'a': [1, 2.3, complex(4, -5)], 'b': [float('nan'), None, complex(1.2, 0)]}, columns=['a', 'b'])
        expected = '[["(1+0j)","(nan+0j)"],["(2.3+0j)","(nan+0j)"],["(4-5j)","(1.2+0j)"]]'
        assert df.to_json(default_handler=str, orient='values') == expected

    def test_default_handler_raises(self) -> None:
        msg = 'raisin'

        def my_handler_raises(obj: Any) -> Any:
            raise TypeError(msg)
        with pytest.raises(TypeError, match=msg):
            DataFrame({'a': [1, 2, object()]}).to_json(default_handler=my_handler_raises)
        with pytest.raises(TypeError, match=msg):
            DataFrame({'a': [1, 2, complex(4, -5)]}).to_json(default_handler=my_handler_raises)

    def test_categorical(self) -> None:
        df = DataFrame({'A': ['a', 'b', 'c', 'a', 'b', 'b', 'a']})
        df['B'] = df['A']
        expected = df.to_json()
        df['B'] = df['A'].astype('category')
        assert expected == df.to_json()
        s = df['A']
        sc = df['B']
        assert s.to_json() == sc.to_json()

    def test_datetime_tz(self) -> None:
        tz_range = date_range('20130101', periods=3, tz='US/Eastern')
        tz_naive = tz_range.tz_convert('utc').tz_localize(None)
        df = DataFrame({'A': tz_range, 'B': date_range('20130101', periods=3)})
        df_naive = df.copy()
        df_naive['A'] = tz_naive
        msg = "The default 'epoch' date format is deprecated and will be removed in a future version, please use 'iso' date format instead."
        with tm.assert_produces_warning(FutureWarning, match=msg):
            expected = df_naive.to_json()
            assert expected == df.to_json()
        stz = Series(tz_range)
        s_naive = Series(tz_naive)
        with tm.assert_produces_warning(FutureWarning, match=msg):
            assert stz.to_json() == s_naive.to_json()

    def test_sparse(self) -> None:
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 4)))
        df.loc[:8] = np.nan
        sdf = df.astype('Sparse')
        expected = df.to_json()
        assert expected == sdf.to_json()
        s = Series(np.random.default_rng(2).standard_normal(10))
        s.loc[:8] = np.nan
        ss = s.astype('Sparse')
        expected = s.to_json()
        assert expected == ss.to_json()

    @pytest.mark.parametrize('ts', [Timestamp('2013-01-10 05:00:00Z'), Timestamp('2013-01-10 00:00:00', tz='US/Eastern'), Timestamp('2013-01-10 00:00:00-0500')])
    def test_tz_is_utc(self, ts: Timestamp) -> None:
        exp = '"2013-01-10T05:00:00.000Z"'
        assert ujson_dumps(ts, iso_dates=True) == exp
        dt = ts.to_pydatetime()
        assert ujson_dumps(dt, iso_dates=True) == exp

    def test_tz_is_naive(self) -> None:
        ts = Timestamp('2013-01-10 05:00:00')
        exp = '"2013-01-10T05:00:00.000"'
        assert ujson_dumps(ts, iso_dates=True) == exp
        dt = ts.to_pydatetime()
        assert ujson_dumps(dt, iso_dates=True) == exp

    @pytest.mark.parametrize('tz_range', [date_range('2013-01-01 05:00:00Z', periods=2), date_range('2013-01-01 00:00:00', periods=2, tz='US/Eastern'), date_range('2013-01-01 00:00:00-0500', periods=2)])
    def test_tz_range_is_utc(self, tz_range: DatetimeIndex) -> None:
        exp = '["2013-01-01T05:00:00.000Z","2013-01-02T05:00:00.000Z"]'
        dfexp = '{"DT":{"0":"2013-01-01T05:00:00.000Z","1":"2013-01-02T05:00:00.000Z"}}'
        assert ujson_dumps(tz_range, iso_dates=True) == exp
        dti = DatetimeIndex(tz_range)
        assert ujson_dumps(dti, iso_dates=True) == exp
        assert ujson_dumps(dti.astype(object), iso_dates=True) == exp
        df = DataFrame({'DT': dti})
        result = ujson_dumps(df, iso_dates=True)
        assert result == dfexp
        assert ujson_dumps(df.astype({'DT': object}), iso_dates=True)

    def test_tz_range_is_naive(self) -> None:
        dti = date_range('2013-01-01 05:00:00', periods=2)
        exp = '["2013-01-01T05:00:00.000","2013-01-02T05:00:00.000"]'
        dfexp = '{"DT":{"0":"2013-01-01T05:00:00.000","1":"2013-01-02T05:00:00.000"}}'
        assert ujson_dumps(dti, iso_dates=True) == exp
        assert ujson_dumps(dti.astype(object), iso_dates=True) == exp
        df = DataFrame({'DT': dti})
        result = ujson_dumps(df, iso_dates=True)
        assert result == dfexp
        assert ujson_dumps(df.astype({'DT': object}), iso_dates=True)

    def test_read_inline_jsonl(self) -> None:
        result = read_json(StringIO('{"a": 1, "b": 2}\n{"b":2, "a" :1}\n'), lines=True)
        expected = DataFrame([[1, 2], [1, 2]], columns=['a', 'b'])
        tm.assert_frame_equal(result, expected)

    @pytest.mark.single_cpu
    @pytest.mark.network
    @td.skip_if_not_us_locale
    def test_read_s3_jsonl(self, s3_public_bucket_with_data: Any, s3so: Any) -> None:
        result = read_json(f's3n://{s3_public_bucket_with_data.name}/items.jsonl', lines=True, storage_options=s3so)
        expected = DataFrame([[1, 2], [1, 2]], columns=['a', 'b'])
        tm.assert_frame_equal(result, expected)

    def test_read_local_jsonl(self) -> None:
        with tm.ensure_clean('tmp_items.json') as path:
            with open(path, 'w', encoding='utf-8') as infile:
                infile.write('{"a": 1, "b": 2}\n{"b":2, "a" :1}\n')
            result = read_json(path, lines=True)
            expected = DataFrame([[1, 2], [1, 2]], columns=['a', 'b'])
            tm.assert_frame_equal(result, expected)

    def test_read_jsonl_unicode_chars(self) -> None:
        json_str = '{"a": "foo”", "b": "bar"}\n{"a": "foo", "b": "bar"}\n'
        json_io = StringIO(json_str)
        result = read_json(json_io, lines=True)
        expected = DataFrame([['foo”', 'bar'], ['foo', 'bar']], columns=['a', 'b'])
        tm.assert_frame_equal(result, expected)
        json_io = StringIO('{"a": "foo”", "b": "bar"}\n{"a": "foo", "b": "bar"}\n')
        result = read_json(json_io, lines=True)
        expected = DataFrame([['foo”', 'bar'], ['foo', 'bar']], columns=['a', 'b'])
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('bigNum', [sys.maxsize + 1, -(sys.maxsize + 2)])
    def test_to_json_large_numbers(self, bigNum: int) -> None:
        series = Series(bigNum, dtype=object, index=['articleId'])
        json_str = series.to_json()
        expected = '{"articleId":' + str(bigNum) + '}'
        assert json_str == expected
        df = DataFrame(bigNum, dtype=object, index=['articleId'], columns=[0])
        json_str = df.to_json()
        expected = '{"0":{"articleId":' + str(bigNum) + '}}'
        assert json_str == expected

    @pytest.mark.parametrize('bigNum', [-2 ** 63 - 1, 2 ** 64])
    def test_read_json_large_numbers(self, bigNum: int) -> None:
        json_io = StringIO('{"articleId":' + str(bigNum) + '}')
        msg = 'Value is too small|Value is too big'
        with pytest.raises(ValueError, match=msg):
            read_json(json_io)
        json_io = StringIO('{"0":{"articleId":' + str(bigNum) + '}}')
        with pytest.raises(ValueError, match=msg):
            read_json(json_io)

    def test_read_json_large_numbers2(self) -> None:
        json_io = '{"articleId": "1404366058080022500245"}'
        json_io = StringIO(json_io)
        result = read_json(json_io, typ='series')
        expected = Series(1.404366e+21, index=['articleId'])
        tm.assert_series_equal(result, expected)
        json_io = '{"0": {"articleId": "1404366058080022500245"}}'
        json_io = StringIO(json_io)
        result = read_json(json_io)
        expected = DataFrame(1.404366e+21, index=['articleId'], columns=[0])
        tm.assert_frame_equal(result, expected)

    def test_to_jsonl(self) -> None:
        df = DataFrame([[1, 2], [1, 2]], columns=['a', 'b'])
        result = df.to_json(orient='records', lines=True)
        expected = '{"a":1,"b":2}\n{"a":1,"b":2}\n'
        assert result == expected
        df = DataFrame([['foo}', 'bar'], ['foo"', 'bar']], columns=['a', 'b'])
        result = df.to_json(orient='records', lines=True)
        expected = '{"a":"foo}","b":"bar"}\n{"a":"foo\\"","b":"bar"}\n'
        assert result == expected
        tm.assert_frame_equal(read_json(StringIO(result), lines=True), df)
        df = DataFrame([['foo\\', 'bar'], ['foo"', 'bar']], columns=['a\\', 'b'])
        result = df.to_json(orient='records', lines=True)
        expected = '{"a\\\\":"foo\\\\","b":"bar"}\n{"a\\\\":"foo\\"","b":"bar"}\n'
        assert result == expected
        tm.assert_frame_equal(read_json(StringIO(result), lines=True), df)

    @pytest.mark.xfail(reason='GH#13774 encoding kwarg not supported', raises=TypeError)
    @pytest.mark.parametrize('val', [[b'E\xc9, 17', b'', b'a', b'b', b'c'], [b'E\xc9, 17', b'a', b'b', b'c'], [b'EE, 17', b'', b'a', b'b', b'c'], [b'E\xc9, 17', b'\xf8\xfc', b'a', b'b', b'c'], [b'', b'a', b'b', b'c'], [b'\xf8\xfc', b'a', b'b', b'c'], [b'A\xf8\xfc', b'', b'a', b'b', b'c'], [np.nan, b'', b'b', b'c'], [b'A\xf8\xfc', np.nan, b'', b'b', b'c']])
    @pytest.mark.parametrize('dtype', ['category', object])
    def test_latin_encoding(self, dtype: Union[str, type], val: List[Any]) -> None:
        ser = Series([x.decode('latin-1') if isinstance(x, bytes) else x for x in val], dtype=dtype)
        encoding = 'latin-1'
        with tm.ensure_clean('test.json') as path:
            ser.to_json(path, encoding=encoding)
            retr = read_json(StringIO(path), encoding=encoding)
            tm.assert_series_equal(ser, retr, check_categorical=False)

    def test_data_frame_size_after_to_json(self) -> None:
        df = DataFrame({'a': [str(1)]})
        size_before = df.memory_usage(index=True, deep=True).sum()
        df.to_json()
        size_after = df.memory_usage(index=True, deep=True).sum()
        assert size_before == size_after

    @pytest.mark.parametrize('index', [None, [1, 2], [1.0, 2.0], ['a', 'b'], ['1', '2'], ['1.', '2.']])
    @pytest.mark.parametrize('columns', [['a', 'b'], ['1', '2'], ['1.', '2.']])
    def test_from_json_to_json_table_index_and_columns(self, index: Optional[List[Any]], columns: List[str]) -> None:
        expected = DataFrame([[1, 2], [3, 4]], index=index, columns=columns)
        dfjson = expected.to_json(orient='table')
        result = read_json(StringIO(dfjson), orient='table')
        tm.assert_frame_equal(result, expected)

    def test_from_json_to_json_table_dtypes(self) -> None:
        expected = DataFrame({'a': [1, 2], 'b': [3.0, 4.0], 'c': ['5', '6']})
        dfjson = expected.to_json(orient='table')
        result = read_json(StringIO(dfjson), orient='table')
        tm.assert_frame_equal(result, expected)

    @pytest.mark.xfail(using_string_dtype(), reason='incorrect na conversion')
    @pytest.mark.parametrize('orient', ['split', 'records', 'index', 'columns'])
    def test_to_json_from_json_columns_dtypes(self, orient: str) -> None:
        expected = DataFrame.from_dict({'Integer': Series([1, 2, 3], dtype='int64'), 'Float': Series([None, 2.0, 3.0], dtype='float64'), 'Object': Series([None, '', 'c'], dtype='object'), 'Bool': Series([True, False, True], dtype='bool'), 'Category': Series(['a', 'b', None], dtype='category'), 'Datetime': Series(['2020-01-01', None, '2020-01-03'], dtype='datetime64[ns]')})
        msg = "The default 'epoch' date format is deprecated and will be removed in a future version, please use 'iso' date format instead."
        with tm.assert_produces_warning(FutureWarning, match=msg):
            dfjson = expected.to_json(orient=orient)
        result = read_json(StringIO(dfjson), orient=orient, dtype={'Integer': 'int64', 'Float': 'float64', 'Object': 'object', 'Bool': 'bool', 'Category': 'category', 'Datetime': 'datetime64[ns]'})
        tm.assert_frame_equal(result, expected)

    def test_to_json_with_index_as_a_column_name(self) -> None:
        df = DataFrame(data={'index': [1, 2], 'a': [2, 3]})
        with pytest.raises(ValueError, match='Overlapping names between the index and columns'):
            df.to_json(orient='table')

    @pytest.mark.parametrize('dtype', [True, {'b': int, 'c': int}])
    def test_read_json_table_dtype_raises(self, dtype: Union[bool, Dict[str, type]]) -> None:
        df = DataFrame({'a': [1, 2], 'b': [3.0, 4.0], 'c': ['5', '6']})
        dfjson = df.to_json(orient='table')
        msg = "cannot pass both dtype and orient='table'"
        with pytest.raises(ValueError, match=msg):
            read_json(dfjson, orient='table', dtype=dtype)

    @pytest.mark.parametrize('orient', ['index', 'columns', 'records', 'values'])
    def test_read_json_table_empty_axes_dtype(self, orient: str) -> None:
        expected = DataFrame()
        result = read_json(StringIO('{}'), orient=orient, convert_axes=True)
        tm.assert_index_equal(result.index, expected.index)
        tm.assert_index_equal(result.columns, expected.columns)

    def test_read_json_table_convert_axes_raises(self) -> None:
        df = DataFrame([[1, 2], [3, 4]], index=[1.0, 2.0], columns=['1.', '2.'])
        dfjson = df.to_json(orient='table')
        msg = "cannot pass both convert_axes and orient='table'"
        with pytest.raises(ValueError, match=msg):
            read_json(dfjson, orient='table', convert_axes=True)

    @pytest.mark.parametrize('data, expected', [(DataFrame([[1, 2], [4, 5]], columns=['a', 'b']), {'columns': ['a', 'b'], 'data': [[1, 2], [4, 5]]}), (DataFrame([[1, 2], [4, 5]], columns=['a', 'b']).rename_axis('foo'), {'columns': ['a', 'b'], 'data': [[1, 2], [4, 5]]}), (DataFrame([[1, 2], [4, 5]], columns=['a', 'b'], index=[['a', 'b'], ['c', 'd']]), {'columns': ['a', 'b'], 'data': [[1, 2], [4, 5]]}), (Series([1, 2, 3], name='A'), {'name': 'A', 'data': [1, 2, 3]}), (Series([1, 2, 3], name='A').rename_axis('foo'), {'name': 'A', 'data': [1, 2, 3]}), (Series([1, 2], name='A', index=[['a', 'b'], ['c', 'd']]), {'name': 'A', 'data': [1, 2]})])
    def test_index_false_to_json_split(self, data: Union[DataFrame, Series], expected: Dict[str, Any]) -> None:
        result = data.to_json(orient='split', index=False)
        result_dict = json.loads(result)
        assert result_dict == expected

    @pytest.mark.parametrize('data', [DataFrame([[1, 2], [4, 5]], columns=['a', 'b']), DataFrame([[1, 2], [4, 5]], columns=['a', 'b']).rename_axis('foo'), DataFrame([[1, 2], [4, 5]], columns=['a', 'b'], index=[['a', 'b'], ['c', 'd']]), Series([1, 2, 3], name='A'), Series([1, 2, 3], name='A').rename_axis('foo'), Series([1, 2], name='A', index=[['a', 'b'], ['c', 'd']])])
    def test_index_false_to_json_table(self, data: Union[DataFrame, Series]) -> None:
        result = data.to_json(orient='table', index=False)
        result_dict = json.loads(result)
        expected = {'schema': pd.io.json.build_table_schema(data, index=False), 'data': DataFrame(data).to_dict(orient='records')}
        assert result_dict == expected

    @pytest.mark.parametrize('orient', ['index', 'columns'])
    def test_index_false_error_to_json(self, orient: str) -> None:
        df = DataFrame([[1, 2], [4, 5]], columns=['a', 'b'])
        msg = "'index=False' is only valid when 'orient' is 'split', 'table', 'records', or 'values'"
        with pytest.raises(ValueError, match=msg):
            df.to_json(orient=orient, index=False)

    @pytest.mark.parametrize('orient', ['records', 'values'])
    def test_index_true_error_to_json(self, orient: str) -> None:
        df = DataFrame([[1, 2], [4, 5]], columns=['a', 'b'])
        msg = "'index=True' is only valid when 'orient' is 'split', 'table', 'index', or 'columns'"
        with pytest.raises(ValueError, match=msg):
            df.to_json(orient=orient, index=True)

    @pytest.mark.parametrize('orient', ['split', 'table'])
    @pytest.mark.parametrize('index', [True, False])
    def test_index_false_from_json_to_json(self, orient: str, index: bool) -> None:
        expected = DataFrame({'a': [1, 2], 'b': [3, 4]})
        dfjson = expected.to_json(orient=orient, index=index)
        result = read_json(StringIO(dfjson), orient=orient)
        tm.assert_frame_equal(result, expected)

    def test_read_timezone_information(self) -> None:
        result = read_json(StringIO('{"2019-01-01T11:00:00.000Z":88}'), typ='series', orient='index')
        exp_dti = DatetimeIndex(['2019-01-01 11:00:00'], dtype='M8[ns, UTC]')
        expected = Series([88], index=exp_dti)
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('url', ['s3://example-fsspec/', 'gcs://another-fsspec/file.json', 'https://example-site.com/data', 'some-protocol://data.txt'])
    def test_read_json_with_url_value(self, url: str) -> None:
        result = read_json(StringIO(f'{{"url":{{"0":"{url}"}}}}'))
        expected = DataFrame({'url': [url]})
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('compression', ['', '.gz', '.bz2', '.tar'])
    def test_read_json_with_very_long_file_path(self, compression: str) -> None:
        long_json_path = f'{'a' * 1000}.json{compression}'
        with pytest.raises(FileNotFoundError, match=f'File {long_json_path} does not exist'):
            read_json(long_json_path)

    @pytest.mark.parametrize('date_format,key', [('epoch', 86400000), ('iso', 'P1DT0H0M0S')])
    def test_timedelta_as_label(self, date_format: str, key: Union[int, str]) -> None:
        df = DataFrame([[1]], columns=[pd.Timedelta('1D')])
        expected = f'{{"{key}":{{"0":1}}}}'
        expected_warning: Optional[Type[Warning]] = None
        if date_format == 'epoch':
            expected_warning = FutureWarning
        msg = "'epoch' date format is deprecated and will be removed in a future version, please use 'iso' date format instead."
        with tm.assert_produces_warning(expected_warning, match=msg):
            result = df.to_json(date_format=date_format)
        assert result == expected

    @pytest.mark.parametrize('orient,expected', [('index', '{"(\'a\', \'b\')":{"(\'c\', \'d\')":1}}'), ('columns', '{"(\'c\', \'d\')":{"(\'a\', \'b\')":1}}'), pytest.param('split', '', marks=pytest.mark.xfail(reason='Produces JSON but not in a consistent manner')), pytest.param('table', '', marks=pytest.mark.xfail(reason='Produces JSON but not in a consistent manner'))])
    def test_tuple_labels(self, orient: str, expected: str) -> None:
        df = DataFrame([[1]], index=[('a', 'b')], columns=[('c', 'd')])
        result = df.to_json(orient=orient)
        assert result == expected

    @pytest.mark.parametrize('indent', [1, 2, 4])
    def test_to_json_indent(self, indent: int) -> None:
        df = DataFrame([['foo', 'bar'], ['baz', 'qux']], columns=['a', 'b'])
        result = df.to_json(indent=indent)
        spaces = ' ' * indent
        expected = f'{{\n{spaces}"a":{{\n{spaces}{spaces}"0":"foo",\n{spaces}{spaces}"1":"baz"\n{spaces}}},\n{spaces}"b":{{\n{spaces}{spaces}"0":"bar",\n{spaces}{spaces}"1":"qux"\n{spaces}}}\n}}'
        assert result == expected

    @pytest.mark.skipif(using_string_dtype(), reason='Adjust expected when infer_string is default, no bug here, just a complicated parametrization')
    @pytest.mark.parametrize('orient,expected', [('split', '{\n    "columns":[\n        "a",\n        "b"\n    ],\n    "index":[\n        0,\n        1\n    ],\n    "data":[\n        [\n            "foo",\n            "bar"\n        ],\n        [\n            "baz",\n            "qux"\n        ]\n    ]\n}'), ('records', '[\n    {\n        "a":"foo",\n        "b":"bar"\n    },\n    {\n        "a":"baz",\n        "b":"qux"\n    }\n]'), ('index', '{\n    "0":{\n        "a":"foo",\n        "b":"bar"\n    },\n    "1":{\n        "a":"baz",\n        "b":"qux"\n    }\n}'), ('columns', '{\n    "a":{\n        "0":"foo",\n        "1":"baz"\n    },\n    "b":{\n        "0":"bar",\n        "1":"qux"\n    }\n}'), ('values', '[\n    [\n        "foo",\n        "bar"\n    ],\n    [\n        "baz",\n        "qux"\n    ]\n]'), ('table', '{\n    "schema":{\n        "fields":[\n            {\n                "name":"index",\n                "type":"integer"\n            },\n            {\n                "name":"a",\n                "type":"string"\n            },\n            {\n                "name":"b",\n                "type":"string"\n            }\n        ],\n        "primaryKey":[\n            "index"\n        ],\n        "pandas_version":"1.4.0"\n    },\n    "data":[\n        {\n            "index":0,\n            "a":"foo",\n            "b":"bar"\n        },\n        {\n            "index":1,\n            "a":"baz",\n            "b":"qux"\n        }\n    ]\n}')])
    def test_json_indent_all_orients(self, orient: str, expected: str) -> None:
        df = DataFrame([['foo', 'bar'], ['baz', 'qux']], columns=['a', 'b'])
        result = df.to_json(orient=orient, indent=4)
        assert result == expected

    def test_json_negative_indent_raises(self) -> None:
        with pytest.raises(ValueError, match='must be a nonnegative integer'):
            DataFrame().to_json(indent=-1)

    def test_emca_262_nan_inf_support(self) -> None:
        data = StringIO('["a", NaN, "NaN", Infinity, "Infinity", -Infinity, "-Infinity"]')
        result = read_json(data)
        expected = DataFrame(['a', None, 'NaN', np.inf, 'Infinity', -np.inf, '-Infinity'])
        tm.assert_frame_equal(result, expected)

    def test_frame_int_overflow(self) -> None:
        encoded_json = json.dumps([{'col': '31900441201190696999'}, {'col': 'Text'}])
        expected = DataFrame({'col': ['31900441201190696999', 'Text']})
        result = read_json(StringIO(encoded_json))
        tm.assert_frame_equal(result, expected)

    def test_json_multiindex(self) -> None:
        dataframe = DataFrame({'x': [1, 2, 3], 'y': ['a', 'b', 'c']})
        expected = '{"(0, \'x\')":1,"(0, \'y\')":"a","(1, \'x\')":2,"(1, \'y\')":"b","(2, \'x\')":3,"(2, \'y\')":"c"}'
        series = dataframe.stack()
        result = series.to_json(orient='index')
        assert result == expected

    @pytest.mark.single_cpu
    @pytest.mark.network
    def test_to_s3(self, s3_public_bucket: Any, s3so: Any) -> None:
        mock_bucket_name, target_file = (s3_public_bucket.name, 'test.json')
        df = DataFrame({'x': [1, 2, 3], 'y': [2, 4, 6]})
        df.to_json(f's3://{mock_bucket_name}/{target_file}', storage_options=s3so)
        timeout = 5
        while True:
            if target_file in (obj.key for obj in s3_public_bucket.objects.all()):
                break
            time.sleep(0.1)
            timeout -= 0.1
            assert timeout > 0, 'Timed out waiting for file to appear on moto'

    def test_json_pandas_nulls(self, nulls_fixture: Any) -> None:
        expected_warning: Optional[Type[Warning]] = None
        msg = "The default 'epoch' date format is deprecated and will be removed in a future version, please use 'iso' date format instead."
        if nulls_fixture is pd.NaT:
            expected_warning = FutureWarning
        with tm.assert_produces_warning(expected_warning, match=msg):
            result = DataFrame([[nulls_fixture]]).to_json()
        assert result == '{"0":{"0":null}}'

    def test_readjson_bool_series(self) -> None:
        result = read_json(StringIO('[true, true, false]'), typ='series')
        expected = Series([True, True, False])
        tm.assert_series_equal(result, expected)

    def test_to_json_multiindex_escape(self) -> None:
        df = DataFrame(True, index=date_range('2017-01-20', '2017-01-23'), columns=['foo', 'bar']).stack()
        result = df.to_json()
        expected = '{"(Timestamp(\'2017-01-20 00:00:00\'), \'foo\')":true,"(Timestamp(\'2017-01-20 00:00:00\'), \'bar\')":true,"(Timestamp(\'2017-01-21 00:00:00\'), \'foo\')":true,"(Timestamp(\'2017-01-21 00:00:00\'), \'bar\')":true,"(Timestamp(\'2017-01-22 00:00:00\'), \'foo\')":true,"(Timestamp(\'2017-01-22 00:00:00\'), \'bar\')":true,"(Timestamp(\'2017-01-23 00:00:00\'), \'foo\')":true,"(Timestamp(\'2017-01-23 00:00:00\'), \'bar\')":true}'
        assert result == expected

    def test_to_json_series_of_objects(self) -> None:

        class _TestObject:

            def __init__(self, a: int, b: int, _c: int, d: int) -> None:
                self.a = a
                self.b = b
                self._c = _c
                self.d = d

            def e(self) -> int:
                return 5
        series = Series([_TestObject(a=1, b=2, _c=3, d=4)])
        assert json.loads(series.to_json()) == {'0': {'a': 1, 'b': 2, 'd': 4}}

    @pytest.mark.parametrize('data,expected', [(Series({0: -6 + 8j, 1: 0 + 1j, 2: 9 - 5j}), '{"0":{"imag":8.0,"real":-6.0},"1":{"imag":1.0,"real":0.0},"2":{"imag":-5.0,"real":9.0}}'), (Series({0: -9.39 + 0.66j, 1: 3.95 + 9.32j, 2: 4.03 - 0.17j}), '{"0":{"imag":0.66,"real":-9.39},"1":{"imag":9.32,"real":3.95},"2":{"imag":-0.17,"real":4.03}}'), (DataFrame([[-2 + 3j, -1 - 0j], [4 - 3j, -0 - 10j]]), '{"0":{"0":{"imag":3.0,"real":-2.0},"1":{"imag":-3.0,"real":4.0}},"1":{"0":{"imag":0.0,"real":-1.0},"1":{"imag":-10.0,"real":0.0}}}'), (DataFrame([[-0.28 + 0.34j, -1.08 - 0.39j], [0.41 - 0.34j, -0.78 - 1.35j]]), '{"0":{"0":{"imag":0.34,"real":-0.28},"1":{"imag":-0.34,"real":0.41}},"1":{"0":{"imag":-0.39,"real":-1.08},"1":{"imag":-1.35,"real":-0.78}}}')])
    def test_complex_data_tojson(self, data: Union[Series, DataFrame], expected: str) -> None:
        result = data.to_json()
        assert result == expected

    def test_json_uint64(self) -> None:
        expected = '{"columns":["col1"],"index":[0,1],"data":[[13342205958987758245],[12388075603347835679]]}'
        df = DataFrame(data={'col1': [13342205958987758245, 12388075603347835679]})
        result = df.to_json(orient='split')
        assert result == expected

    def test_read_json_dtype_backend(self, string_storage: str, dtype_backend: str, orient: str, using_infer_string: bool) -> None:
        df = DataFrame({'a': Series([1, np.nan, 3], dtype='Int64'), 'b': Series([1, 2, 3], dtype='Int64'), 'c': Series([1.5, np.nan, 2.5], dtype='Float64'), 'd': Series([1.5, 2.0, 2.5], dtype='Float64'), 'e': [True, False, None], 'f': [True, False, True], 'g': ['a', 'b', 'c'], 'h': ['a', 'b', None]})
        out = df.to_json(orient=orient)
        with pd.option_context('mode.string_storage', string_storage):
            result = read_json(StringIO(out), dtype_backend=dtype_backend, orient=orient)
        if dtype_backend == 'pyarrow':
            pa = pytest.importorskip('pyarrow')
            string_dtype = pd.ArrowDtype(pa.string())
        else:
            string_dtype = pd.StringDtype(string_storage)
        expected = DataFrame({'a': Series([1, np.nan, 3], dtype='Int64'), 'b': Series([1, 2, 3], dtype='Int64'), 'c': Series([1.5, np.nan, 2.5], dtype='Float64'), 'd': Series([1.5, 2.0, 2.5], dtype='Float64'), 'e': Series([True, False, NA], dtype='boolean'), 'f': Series([True, False, True], dtype='boolean'), 'g': Series(['a', 'b', 'c'], dtype=string_dtype), 'h': Series(['a', 'b', None], dtype=string_dtype)})
        if dtype_backend == 'pyarrow':
            pa = pytest.importorskip('pyarrow')
            from pandas.arrays import ArrowExtensionArray
            expected = DataFrame({col: ArrowExtensionArray(pa.array(expected[col], from_pandas=True)) for col in expected.columns})
        if orient == 'values':
            expected.columns = list(range(8))
        tm.assert_frame_equal(result, expected, check_column_type=False)

    @pytest.mark.parametrize('orient', ['split', 'records', 'index'])
    def test_read_json_nullable_series(self, string_storage: str, dtype_backend: str, orient: str) -> None:
        pa = pytest.importorskip('pyarrow')
        ser = Series([1, np.nan, 3], dtype='Int64')
        out = ser.to_json(orient=orient)
        with pd.option_context('mode.string_storage', string_storage):
            result = read_json(StringIO(out), dtype_backend=dtype_backend, orient=orient, typ='series')
        expected = Series([1, np.nan, 3], dtype='Int64')
        if dtype_backend == 'pyarrow':
            from pandas.arrays import ArrowExtensionArray
            expected = Series(ArrowExtensionArray(pa.array(expected, from_pandas=True)))
        tm.assert_series_equal(result, expected)

    def test_invalid_dtype_backend(self) -> None:
        msg = "dtype_backend numpy is invalid, only 'numpy_nullable' and 'pyarrow' are allowed."
        with pytest.raises(ValueError, match=msg):
            read_json('test', dtype_backend='numpy')


def test_invalid_engine() -> None:
    ser = Series(range(1))
    out = ser.to_json()
    with pytest.raises(ValueError, match='The engine type foo'):
        read_json(out, engine='foo')


def test_pyarrow_engine_lines_false() -> None:
    ser = Series(range(1))
    out = ser.to_json()
    with pytest.raises(ValueError, match='currently pyarrow engine only supports'):
        read_json(out, engine='pyarrow', lines=False)


def test_json_roundtrip_string_inference(orient: str) -> None:
    df = DataFrame([['a', 'b'], ['c', 'd']], index=['row 1', 'row 2'], columns=['col 1', 'col 2'])
    out = df.to_json()
    with pd.option_context('future.infer_string', True):
        result = read_json(StringIO(out))
    dtype = pd.StringDtype(na_value=np.nan)
    expected = DataFrame([['a', 'b'], ['c', 'd']], dtype=dtype, index=Index(['row 1', 'row 2'], dtype=dtype), columns=Index(['col 1', 'col 2'], dtype=dtype))
    tm.assert_frame_equal(result, expected)


@td.skip_if_no('pyarrow')
def test_to_json_ea_null() -> None:
    df = DataFrame({'a': Series([1, NA], dtype='int64[pyarrow]'), 'b': Series([2, NA], dtype='Int64')})
    result = df.to_json(orient='records', lines=True)
    expected = '{"a":1,"b":2}\n{"a":null,"b":null}\n'
    assert result == expected


def test_read_json_lines_rangeindex() -> None:
    data = '\n{"a": 1, "b": 2}\n{"a": 3, "b": 4}\n'
    result = read_json(StringIO(data), lines=True).index
    expected = RangeIndex(2)
    tm.assert_index_equal(result, expected, exact=True)


def test_large_number() -> None:
    result = read_json(StringIO('["9999999999999999"]'), orient='values', typ='series', convert_dates=False)
    expected = Series([9999999999999999])
    tm.assert_series_equal(result, expected)