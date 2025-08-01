#!/usr/bin/env python3
from datetime import datetime, date
import re
from pathlib import Path
from typing import Any, Callable, Union

import numpy as np
import pytest
from pandas._libs.tslibs import Timestamp
from pandas.compat import is_platform_windows
import pandas as pd
from pandas import DataFrame, DatetimeIndex, Index, Series, _testing as tm, bdate_range, date_range, read_hdf
from pandas.tests.io.pytables.common import _maybe_remove, ensure_clean_store
from pandas.util import _test_decorators as td

pytestmark = [pytest.mark.single_cpu]

def test_conv_read_write() -> None:
    with tm.ensure_clean() as path:
        def roundtrip(key: str, obj: Any, **kwargs: Any) -> Any:
            obj.to_hdf(path, key=key, **kwargs)
            return read_hdf(path, key)
        o: Series = Series(np.arange(10, dtype=np.float64), index=date_range('2020-01-01', periods=10))
        tm.assert_series_equal(o, roundtrip('series', o))
        o = Series(range(10), dtype='float64', index=[f'i_{i}' for i in range(10)])
        tm.assert_series_equal(o, roundtrip('string_series', o))
        o = DataFrame(1.1 * np.arange(120).reshape((30, 4)),
                      columns=Index(list('ABCD')),
                      index=Index([f'i-{i}' for i in range(30)]))
        tm.assert_frame_equal(o, roundtrip('frame', o))
        df: DataFrame = DataFrame({'A': range(5), 'B': range(5)})
        df.to_hdf(path, key='table', append=True)
        result: DataFrame = read_hdf(path, 'table', where=['index>2'])
        tm.assert_frame_equal(df[df.index > 2], result)

def test_long_strings(setup_path: Union[str, Path]) -> None:
    data: list[str] = ['a' * 50] * 10
    df: DataFrame = DataFrame({'a': data}, index=data)
    with ensure_clean_store(setup_path) as store:
        store.append('df', df, data_columns=['a'])
        result: DataFrame = store.select('df')
        tm.assert_frame_equal(df, result)

def test_api(tmp_path: Path, setup_path: Union[str, Path]) -> None:
    path: Path = tmp_path / setup_path
    df: DataFrame = DataFrame(range(20))
    df.iloc[:10].to_hdf(path, key='df', append=True, format='table')
    df.iloc[10:].to_hdf(path, key='df', append=True, format='table')
    tm.assert_frame_equal(read_hdf(path, 'df'), df)
    df.iloc[:10].to_hdf(path, key='df', append=False, format='table')
    df.iloc[10:].to_hdf(path, key='df', append=True, format='table')
    tm.assert_frame_equal(read_hdf(path, 'df'), df)

def test_api_append(tmp_path: Path, setup_path: Union[str, Path]) -> None:
    path: Path = tmp_path / setup_path
    df: DataFrame = DataFrame(range(20))
    df.iloc[:10].to_hdf(path, key='df', append=True)
    df.iloc[10:].to_hdf(path, key='df', append=True, format='table')
    tm.assert_frame_equal(read_hdf(path, 'df'), df)
    df.iloc[:10].to_hdf(path, key='df', append=False, format='table')
    df.iloc[10:].to_hdf(path, key='df', append=True)
    tm.assert_frame_equal(read_hdf(path, 'df'), df)

def test_api_2(tmp_path: Path, setup_path: Union[str, Path]) -> None:
    path: Path = tmp_path / setup_path
    df: DataFrame = DataFrame(range(20))
    df.to_hdf(path, key='df', append=False, format='fixed')
    tm.assert_frame_equal(read_hdf(path, 'df'), df)
    df.to_hdf(path, key='df', append=False, format='f')
    tm.assert_frame_equal(read_hdf(path, 'df'), df)
    df.to_hdf(path, key='df', append=False)
    tm.assert_frame_equal(read_hdf(path, 'df'), df)
    df.to_hdf(path, key='df')
    tm.assert_frame_equal(read_hdf(path, 'df'), df)
    with ensure_clean_store(setup_path) as store:
        df = DataFrame(range(20))
        _maybe_remove(store, 'df')
        store.append('df', df.iloc[:10], append=True, format='table')
        store.append('df', df.iloc[10:], append=True, format='table')
        tm.assert_frame_equal(store.select('df'), df)
        _maybe_remove(store, 'df')
        store.append('df', df.iloc[:10], append=False, format='table')
        store.append('df', df.iloc[10:], append=True, format='table')
        tm.assert_frame_equal(store.select('df'), df)
        _maybe_remove(store, 'df')
        store.append('df', df.iloc[:10], append=False, format='table')
        store.append('df', df.iloc[10:], append=True, format='table')
        tm.assert_frame_equal(store.select('df'), df)
        _maybe_remove(store, 'df')
        store.append('df', df.iloc[:10], append=False, format='table')
        store.append('df', df.iloc[10:], append=True, format=None)
        tm.assert_frame_equal(store.select('df'), df)

def test_api_invalid(tmp_path: Path, setup_path: Union[str, Path]) -> None:
    path: Path = tmp_path / setup_path
    df: DataFrame = DataFrame(1.1 * np.arange(120).reshape((30, 4)),
                              columns=Index(list('ABCD')),
                              index=Index([f'i-{i}' for i in range(30)]))
    msg: str = 'Can only append to Tables'
    with pytest.raises(ValueError, match=msg):
        df.to_hdf(path, key='df', append=True, format='f')
    with pytest.raises(ValueError, match=msg):
        df.to_hdf(path, key='df', append=True, format='fixed')
    msg = 'invalid HDFStore format specified \\[foo\\]'
    with pytest.raises(TypeError, match=msg):
        df.to_hdf(path, key='df', append=True, format='foo')
    with pytest.raises(TypeError, match=msg):
        df.to_hdf(path, key='df', append=False, format='foo')
    path_str: str = ''
    msg = f'File {path_str} does not exist'
    with pytest.raises(FileNotFoundError, match=msg):
        read_hdf(path_str, 'df')

def test_get(setup_path: Union[str, Path]) -> None:
    with ensure_clean_store(setup_path) as store:
        store['a'] = Series(np.arange(10, dtype=np.float64), index=date_range('2020-01-01', periods=10))
        left: Series = store.get('a')
        right: Series = store['a']
        tm.assert_series_equal(left, right)
        left = store.get('/a')
        right = store['/a']
        tm.assert_series_equal(left, right)
        with pytest.raises(KeyError, match="'No object named b in the file'"):
            store.get('b')

def test_put_integer(setup_path: Union[str, Path]) -> None:
    df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((50, 100)))
    _check_roundtrip(df, tm.assert_frame_equal, setup_path)

def test_table_values_dtypes_roundtrip(setup_path: Union[str, Path], using_infer_string: bool) -> None:
    with ensure_clean_store(setup_path) as store:
        df1: DataFrame = DataFrame({'a': [1, 2, 3]}, dtype='f8')
        store.append('df_f8', df1)
        tm.assert_series_equal(df1.dtypes, store['df_f8'].dtypes)
        df2: DataFrame = DataFrame({'a': [1, 2, 3]}, dtype='i8')
        store.append('df_i8', df2)
        tm.assert_series_equal(df2.dtypes, store['df_i8'].dtypes)
        msg: str = re.escape('Cannot serialize the column [a] because its data contents are not [float] but [integer] object dtype')
        with pytest.raises(ValueError, match=msg):
            store.append('df_i8', df1)
        df1 = DataFrame(np.array([[1], [2], [3]], dtype='f4'), columns=['A'])
        store.append('df_f4', df1)
        tm.assert_series_equal(df1.dtypes, store['df_f4'].dtypes)
        assert df1.dtypes.iloc[0] == 'float32'
        df1 = DataFrame({c: Series(np.random.default_rng(2).integers(5), dtype=c)
                         for c in ['float32', 'float64', 'int32', 'int64', 'int16', 'int8']})
        df1['string'] = 'foo'
        df1['float322'] = 1.0
        df1['float322'] = df1['float322'].astype('float32')
        df1['bool'] = df1['float32'] > 0
        df1['time_s_1'] = Timestamp('20130101')
        df1['time_s_2'] = Timestamp('20130101 00:00:00')
        df1['time_ms'] = Timestamp('20130101 00:00:00.000')
        df1['time_ns'] = Timestamp('20130102 00:00:00.000000000')
        store.append('df_mixed_dtypes1', df1)
        result: Series = store.select('df_mixed_dtypes1').dtypes.value_counts()
        result.index = [str(i) for i in result.index]
        str_dtype: str = 'str' if using_infer_string else 'object'
        expected: Series = Series({'float32': 2, 'float64': 1, 'int32': 1, 'bool': 1,
                                     'int16': 1, 'int8': 1, 'int64': 1, str_dtype: 1,
                                     'datetime64[s]': 2, 'datetime64[ms]': 1, 'datetime64[ns]': 1},
                                    name='count')
        result = result.sort_index()
        expected = expected.sort_index()
        tm.assert_series_equal(result, expected)

@pytest.mark.filterwarnings('ignore::pandas.errors.PerformanceWarning')
def test_series(setup_path: Union[str, Path]) -> None:
    s: Series = Series(range(10), dtype='float64', index=[f'i_{i}' for i in range(10)])
    _check_roundtrip(s, tm.assert_series_equal, path=setup_path)
    ts: Series = Series(np.arange(10, dtype=np.float64), index=date_range('2020-01-01', periods=10))
    _check_roundtrip(ts, tm.assert_series_equal, path=setup_path)
    ts2: Series = Series(ts.index, Index(ts.index))
    _check_roundtrip(ts2, tm.assert_series_equal, path=setup_path)
    ts3: Series = Series(ts.values, Index(np.asarray(ts.index)))
    _check_roundtrip(ts3, tm.assert_series_equal, path=setup_path, check_index_type=False)

def test_float_index(setup_path: Union[str, Path]) -> None:
    index: np.ndarray = np.random.default_rng(2).standard_normal(10)
    s: Series = Series(np.random.default_rng(2).standard_normal(10), index=index)
    _check_roundtrip(s, tm.assert_series_equal, path=setup_path)

def test_tuple_index(setup_path: Union[str, Path], performance_warning: Any) -> None:
    col: np.ndarray = np.arange(10)
    idx: list[tuple[float, float]] = [(0.0, 1.0), (2.0, 3.0), (4.0, 5.0)]
    data: np.ndarray = np.random.default_rng(2).standard_normal(30).reshape((3, 10))
    DF: DataFrame = DataFrame(data, index=idx, columns=col)
    with tm.assert_produces_warning(performance_warning):
        _check_roundtrip(DF, tm.assert_frame_equal, path=setup_path)

@pytest.mark.filterwarnings('ignore::pandas.errors.PerformanceWarning')
def test_index_types(setup_path: Union[str, Path]) -> None:
    values: np.ndarray = np.random.default_rng(2).standard_normal(2)
    func: Callable[[Any, Any], None] = lambda lhs, rhs: tm.assert_series_equal(lhs, rhs, check_index_type=True)
    ser: Series = Series(values, [0, 'y'])
    _check_roundtrip(ser, func, path=setup_path)
    ser = Series(values, [datetime.today(), 0])
    _check_roundtrip(ser, func, path=setup_path)
    ser = Series(values, ['y', 0])
    _check_roundtrip(ser, func, path=setup_path)
    ser = Series(values, [date.today(), 'a'])
    _check_roundtrip(ser, func, path=setup_path)
    ser = Series(values, [0, 'y'])
    _check_roundtrip(ser, func, path=setup_path)
    ser = Series(values, [datetime.today(), 0])
    _check_roundtrip(ser, func, path=setup_path)
    ser = Series(values, ['y', 0])
    _check_roundtrip(ser, func, path=setup_path)
    ser = Series(values, [date.today(), 'a'])
    _check_roundtrip(ser, func, path=setup_path)
    ser = Series(values, [1.23, 'b'])
    _check_roundtrip(ser, func, path=setup_path)
    ser = Series(values, [1, 1.53])
    _check_roundtrip(ser, func, path=setup_path)
    ser = Series(values, [1, 5])
    _check_roundtrip(ser, func, path=setup_path)
    dti: DatetimeIndex = DatetimeIndex(['2012-01-01', '2012-01-02'], dtype='M8[ns]')
    ser = Series(values, index=dti)
    _check_roundtrip(ser, func, path=setup_path)
    ser.index = ser.index.as_unit('s')
    _check_roundtrip(ser, func, path=setup_path)

def test_timeseries_preepoch(setup_path: Union[str, Path], request: pytest.FixtureRequest) -> None:
    dr = bdate_range('1/1/1940', '1/1/1960')
    ts: Series = Series(np.random.default_rng(2).standard_normal(len(dr)), index=dr)
    try:
        _check_roundtrip(ts, tm.assert_series_equal, path=setup_path)
    except OverflowError:
        if is_platform_windows():
            request.applymarker(pytest.mark.xfail('known failure on some windows platforms'))
        raise

@pytest.mark.parametrize('compression', [False, pytest.param(True, marks=td.skip_if_windows)])
def test_frame(compression: bool, setup_path: Union[str, Path]) -> None:
    df: DataFrame = DataFrame(1.1 * np.arange(120).reshape((30, 4)),
                               columns=Index(list('ABCD')),
                               index=Index([f'i-{i}' for i in range(30)]))
    df.iloc[0, 0] = np.nan
    df.iloc[5, 3] = np.nan
    _check_roundtrip_table(df, tm.assert_frame_equal, path=setup_path, compression=compression)
    _check_roundtrip(df, tm.assert_frame_equal, path=setup_path, compression=compression)
    tdf: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((10, 4)),
                                 columns=Index(list('ABCD')),
                                 index=date_range('2000-01-01', periods=10, freq='B'))
    _check_roundtrip(tdf, tm.assert_frame_equal, path=setup_path, compression=compression)
    with ensure_clean_store(setup_path) as store:
        df['foo'] = np.random.default_rng(2).standard_normal(len(df))
        store['df'] = df
        recons: DataFrame = store['df']
        assert recons._mgr.is_consolidated()
    df2: DataFrame = df[:0]
    df2.index = Index([])
    _check_roundtrip(df2[:0], tm.assert_frame_equal, path=setup_path)

def test_empty_series_frame(setup_path: Union[str, Path]) -> None:
    s0: Series = Series(dtype=object)
    s1: Series = Series(name='myseries', dtype=object)
    df0: DataFrame = DataFrame()
    df1: DataFrame = DataFrame(index=['a', 'b', 'c'])
    df2: DataFrame = DataFrame(columns=['d', 'e', 'f'])
    _check_roundtrip(s0, tm.assert_series_equal, path=setup_path)
    _check_roundtrip(s1, tm.assert_series_equal, path=setup_path)
    _check_roundtrip(df0, tm.assert_frame_equal, path=setup_path)
    _check_roundtrip(df1, tm.assert_frame_equal, path=setup_path)
    _check_roundtrip(df2, tm.assert_frame_equal, path=setup_path)

@pytest.mark.parametrize('dtype', [np.int64, np.float64, object, 'm8[ns]', 'M8[ns]'])
def test_empty_series(dtype: Any, setup_path: Union[str, Path]) -> None:
    s: Series = Series(dtype=dtype)
    _check_roundtrip(s, tm.assert_series_equal, path=setup_path)

def test_can_serialize_dates(setup_path: Union[str, Path]) -> None:
    rng = [x.date() for x in bdate_range('1/1/2000', '1/30/2000')]
    frame: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((len(rng), 4)), index=rng)
    _check_roundtrip(frame, tm.assert_frame_equal, path=setup_path)

def test_store_hierarchical(setup_path: Union[str, Path], using_infer_string: bool, multiindex_dataframe_random_data: DataFrame) -> None:
    frame: DataFrame = multiindex_dataframe_random_data
    if using_infer_string:
        msg: str = 'Saving a MultiIndex with an extension dtype is not supported.'
        with pytest.raises(NotImplementedError, match=msg):
            _check_roundtrip(frame, tm.assert_frame_equal, path=setup_path)
        return
    _check_roundtrip(frame, tm.assert_frame_equal, path=setup_path)
    _check_roundtrip(frame.T, tm.assert_frame_equal, path=setup_path)
    _check_roundtrip(frame['A'], tm.assert_series_equal, path=setup_path)
    with ensure_clean_store(setup_path) as store:
        store['frame'] = frame
        recons: DataFrame = store['frame']
        tm.assert_frame_equal(recons, frame)

@pytest.mark.parametrize('compression', [False, pytest.param(True, marks=td.skip_if_windows)])
def test_store_mixed(compression: bool, setup_path: Union[str, Path]) -> None:
    def _make_one() -> DataFrame:
        df: DataFrame = DataFrame(1.1 * np.arange(120).reshape((30, 4)),
                                   columns=Index(list('ABCD')),
                                   index=Index([f'i-{i}' for i in range(30)]))
        df['obj1'] = 'foo'
        df['obj2'] = 'bar'
        df['bool1'] = df['A'] > 0
        df['bool2'] = df['B'] > 0
        df['int1'] = 1
        df['int2'] = 2
        return df._consolidate()
    df1: DataFrame = _make_one()
    df2: DataFrame = _make_one()
    _check_roundtrip(df1, tm.assert_frame_equal, path=setup_path)
    _check_roundtrip(df2, tm.assert_frame_equal, path=setup_path)
    with ensure_clean_store(setup_path) as store:
        store['obj'] = df1
        tm.assert_frame_equal(store['obj'], df1)
        store['obj'] = df2
        tm.assert_frame_equal(store['obj'], df2)
    _check_roundtrip(df1['obj1'], tm.assert_series_equal, path=setup_path, compression=compression)
    _check_roundtrip(df1['bool1'], tm.assert_series_equal, path=setup_path, compression=compression)
    _check_roundtrip(df1['int1'], tm.assert_series_equal, path=setup_path, compression=compression)

def _check_roundtrip(obj: Any, comparator: Callable[[Any, Any], None], path: Union[str, Path],
                       compression: bool = False, **kwargs: Any) -> None:
    options: dict[str, Any] = {}
    if compression:
        options['complib'] = 'blosc'
    with ensure_clean_store(path, 'w', **options) as store:
        store['obj'] = obj
        retrieved: Any = store['obj']
        comparator(retrieved, obj, **kwargs)

def _check_roundtrip_table(obj: Any, comparator: Callable[[Any, Any], None], path: Union[str, Path],
                           compression: bool = False) -> None:
    options: dict[str, Any] = {}
    if compression:
        options['complib'] = 'blosc'
    with ensure_clean_store(path, 'w', **options) as store:
        store.put('obj', obj, format='table')
        retrieved: Any = store['obj']
        comparator(retrieved, obj)

def test_unicode_index(setup_path: Union[str, Path]) -> None:
    unicode_values: list[str] = ['σ', 'σσ']
    s: Series = Series(np.random.default_rng(2).standard_normal(len(unicode_values)), unicode_values)
    _check_roundtrip(s, tm.assert_series_equal, path=setup_path)

def test_unicode_longer_encoded(setup_path: Union[str, Path]) -> None:
    char: str = 'Δ'
    df: DataFrame = DataFrame({'A': [char]})
    with ensure_clean_store(setup_path) as store:
        store.put('df', df, format='table', encoding='utf-8')
        result: DataFrame = store.get('df')
        tm.assert_frame_equal(result, df)
    df = DataFrame({'A': ['a', char], 'B': ['b', 'b']})
    with ensure_clean_store(setup_path) as store:
        store.put('df', df, format='table', encoding='utf-8')
        result = store.get('df')
        tm.assert_frame_equal(result, df)

def test_store_datetime_mixed(setup_path: Union[str, Path]) -> None:
    df: DataFrame = DataFrame({'a': [1, 2, 3],
                               'b': [1.0, 2.0, 3.0],
                               'c': ['a', 'b', 'c']})
    ts: Series = Series(np.arange(10, dtype=np.float64), index=date_range('2020-01-01', periods=10))
    df['d'] = ts.index[:3]
    _check_roundtrip(df, tm.assert_frame_equal, path=setup_path)

def test_round_trip_equals(tmp_path: Path, setup_path: Union[str, Path]) -> None:
    df: DataFrame = DataFrame({'B': [1, 2], 'A': ['x', 'y']})
    path: Path = tmp_path / setup_path
    df.to_hdf(path, key='df', format='table')
    other: DataFrame = read_hdf(path, 'df')
    tm.assert_frame_equal(df, other)
    assert df.equals(other)
    assert other.equals(df)

def test_infer_string_columns(tmp_path: Path, setup_path: Union[str, Path]) -> None:
    pytest.importorskip('pyarrow')
    path: Path = tmp_path / setup_path
    with pd.option_context('future.infer_string', True):
        df: DataFrame = DataFrame(1, columns=list('ABCD'), index=list(range(10))).set_index(['A', 'B'])
        expected: DataFrame = df.copy()
        df.to_hdf(path, key='df', format='table')
        result: DataFrame = read_hdf(path, 'df')
        tm.assert_frame_equal(result, expected)