#!/usr/bin/env python
from __future__ import annotations
import contextlib
import datetime as dt
import hashlib
import tempfile
import time
import numpy as np
import pytest
from pandas._config import using_string_dtype
from pandas.compat import PY312
import pandas as pd
from pandas import DataFrame, DatetimeIndex, Index, MultiIndex, Series, Timestamp, concat, date_range, period_range, timedelta_range
import pandas._testing as tm
from pandas.conftest import has_pyarrow
from pandas.tests.io.pytables.common import _maybe_remove, ensure_clean_store
from pandas.io.pytables import HDFStore, read_hdf
from pathlib import Path
from typing import Any, Callable, Dict, List, Set, Tuple

pytestmark = [pytest.mark.single_cpu]
tables = pytest.importorskip('tables')


def test_context(setup_path: str) -> None:
    with tm.ensure_clean(setup_path) as path:
        try:
            with HDFStore(path) as tbl:
                raise ValueError('blah')
        except ValueError:
            pass
    with tm.ensure_clean(setup_path) as path:
        with HDFStore(path) as tbl:
            tbl['a'] = DataFrame(
                1.1 * np.arange(120).reshape((30, 4)),
                columns=Index(list('ABCD'), dtype=object),
                index=Index([f'i-{i}' for i in range(30)], dtype=object),
            )
            assert len(tbl) == 1
            assert type(tbl['a']) == DataFrame


def test_no_track_times(tmp_path: Path, setup_path: str) -> None:
    def checksum(
        filename: str,
        hash_factory: Callable[[], Any] = hashlib.md5,
        chunk_num_blocks: int = 128,
    ) -> bytes:
        h = hash_factory()
        with open(filename, 'rb') as f:
            for chunk in iter(lambda: f.read(chunk_num_blocks * h.block_size), b''):
                h.update(chunk)
        return h.digest()

    def create_h5_and_return_checksum(tmp_path: Path, track_times: bool) -> bytes:
        path = tmp_path / setup_path
        df = DataFrame({'a': [1]})
        with HDFStore(path, mode='w') as hdf:
            hdf.put('table', df, format='table', data_columns=True, index=None, track_times=track_times)
        return checksum(str(path))

    checksum_0_tt_false: bytes = create_h5_and_return_checksum(tmp_path, track_times=False)
    checksum_0_tt_true: bytes = create_h5_and_return_checksum(tmp_path, track_times=True)
    time.sleep(1)
    checksum_1_tt_false: bytes = create_h5_and_return_checksum(tmp_path, track_times=False)
    checksum_1_tt_true: bytes = create_h5_and_return_checksum(tmp_path, track_times=True)
    assert checksum_0_tt_false == checksum_1_tt_false
    assert checksum_0_tt_true != checksum_1_tt_true


def test_iter_empty(setup_path: str) -> None:
    with ensure_clean_store(setup_path) as store:
        assert list(store) == []


def test_repr(setup_path: str, performance_warning: Any, using_infer_string: bool) -> None:
    with ensure_clean_store(setup_path) as store:
        repr(store)
        store.info()
        store['a'] = Series(
            np.arange(10, dtype=np.float64),
            index=date_range('2020-01-01', periods=10),
        )
        store['b'] = Series(
            range(10),
            dtype='float64',
            index=[f'i_{i}' for i in range(10)],
        )
        store['c'] = DataFrame(
            1.1 * np.arange(120).reshape((30, 4)),
            columns=Index(list('ABCD'), dtype=object),
            index=Index([f'i-{i}' for i in range(30)], dtype=object),
        )
        df = DataFrame(
            1.1 * np.arange(120).reshape((30, 4)),
            columns=Index(list('ABCD'), dtype=object),
            index=Index([f'i-{i}' for i in range(30)], dtype=object),
        )
        df['obj1'] = 'foo'
        df['obj2'] = 'bar'
        df['bool1'] = df['A'] > 0
        df['bool2'] = df['B'] > 0
        df['bool3'] = True
        df['int1'] = 1
        df['int2'] = 2
        df['timestamp1'] = Timestamp('20010102')
        df['timestamp2'] = Timestamp('20010103')
        df['datetime1'] = dt.datetime(2001, 1, 2, 0, 0)
        df['datetime2'] = dt.datetime(2001, 1, 3, 0, 0)
        df.loc[df.index[3:6], ['obj1']] = np.nan
        df = df._consolidate()
        warning: Any = None if using_infer_string else performance_warning
        msg: str = "cannot\nmap directly to c-types .* dtype='object'"
        with tm.assert_produces_warning(warning, match=msg):
            store['df'] = df
        store._handle.create_group(store._handle.root, 'bah')
        assert store.filename in repr(store)
        assert store.filename in str(store)
        store.info()
    with ensure_clean_store(setup_path) as store:
        df = DataFrame(
            1.1 * np.arange(120).reshape((30, 4)),
            columns=Index(list('ABCD'), dtype=object),
            index=Index([f'i-{i}' for i in range(30)], dtype=object),
        )
        store.append('df', df)
        s = store.get_storer('df')
        repr(s)
        str(s)


def test_contains(setup_path: str) -> None:
    with ensure_clean_store(setup_path) as store:
        store['a'] = Series(
            np.arange(10, dtype=np.float64),
            index=date_range('2020-01-01', periods=10),
        )
        store['b'] = DataFrame(
            1.1 * np.arange(120).reshape((30, 4)),
            columns=Index(list('ABCD'), dtype=object),
            index=Index([f'i-{i}' for i in range(30)], dtype=object),
        )
        store['foo/bar'] = DataFrame(
            1.1 * np.arange(120).reshape((30, 4)),
            columns=Index(list('ABCD'), dtype=object),
            index=Index([f'i-{i}' for i in range(30)], dtype=object),
        )
        assert 'a' in store
        assert 'b' in store
        assert 'c' not in store
        assert 'foo/bar' in store
        assert '/foo/bar' in store
        assert '/foo/b' not in store
        assert 'bar' not in store
        with tm.assert_produces_warning(tables.NaturalNameWarning, check_stacklevel=False):
            store['node())'] = DataFrame(
                1.1 * np.arange(120).reshape((30, 4)),
                columns=Index(list('ABCD'), dtype=object),
                index=Index([f'i-{i}' for i in range(30)], dtype=object),
            )
        assert 'node())' in store


def test_versioning(setup_path: str) -> None:
    with ensure_clean_store(setup_path) as store:
        store['a'] = Series(
            np.arange(10, dtype=np.float64),
            index=date_range('2020-01-01', periods=10),
        )
        store['b'] = DataFrame(
            1.1 * np.arange(120).reshape((30, 4)),
            columns=Index(list('ABCD'), dtype=object),
            index=Index([f'i-{i}' for i in range(30)], dtype=object),
        )
        df = DataFrame(
            np.random.default_rng(2).standard_normal((20, 4)),
            columns=Index(list('ABCD'), dtype=object),
            index=date_range('2000-01-01', periods=20, freq='B'),
        )
        _maybe_remove(store, 'df1')
        store.append('df1', df[:10])
        store.append('df1', df[10:])
        assert store.root.a._v_attrs.pandas_version == '0.15.2'
        assert store.root.b._v_attrs.pandas_version == '0.15.2'
        assert store.root.df1._v_attrs.pandas_version == '0.15.2'
        _maybe_remove(store, 'df2')
        store.append('df2', df)
        store.get_node('df2')._v_attrs.pandas_version = None
        msg: str = "'NoneType' object has no attribute 'startswith'"
        with pytest.raises(Exception, match=msg):
            store.select('df2')


@pytest.mark.parametrize(
    'where, expected',
    [
        (
            '/',
            {
                '': ({'first_group', 'second_group'}, set()),
                '/first_group': (set(), {'df1', 'df2'}),
                '/second_group': ({'third_group'}, {'df3', 's1'}),
                '/second_group/third_group': (set(), {'df4'}),
            },
        ),
        (
            '/second_group',
            {
                '/second_group': (set(), {'df3', 's1'}),
                '/second_group/third_group': (set(), {'df4'}),
            },
        ),
    ],
)
def test_walk(where: str, expected: Dict[str, Tuple[Set[str], Set[str]]]) -> None:
    objs: Dict[str, Any] = {
        'df1': DataFrame([1, 2, 3]),
        'df2': DataFrame([4, 5, 6]),
        'df3': DataFrame([6, 7, 8]),
        'df4': DataFrame([9, 10, 11]),
        's1': Series([10, 9, 8]),
        'a1': np.array([[1, 2, 3], [4, 5, 6]]),
        'tb1': np.array([(1, 2, 3), (4, 5, 6)], dtype='i,i,i'),
        'tb2': np.array([(7, 8, 9), (10, 11, 12)], dtype='i,i,i'),
    }
    with ensure_clean_store('walk_groups.hdf', mode='w') as store:
        store.put('/first_group/df1', objs['df1'])
        store.put('/first_group/df2', objs['df2'])
        store.put('/second_group/df3', objs['df3'])
        store.put('/second_group/s1', objs['s1'])
        store.put('/second_group/third_group/df4', objs['df4'])
        store._handle.create_array('/first_group', 'a1', objs['a1'])
        store._handle.create_table('/first_group', 'tb1', obj=objs['tb1'])
        store._handle.create_table('/second_group', 'tb2', obj=objs['tb2'])
        assert len(list(store.walk(where=where))) == len(expected)
        for path, groups, leaves in store.walk(where=where):
            assert path in expected
            expected_groups, expected_frames = expected[path]
            assert expected_groups == set(groups)
            assert expected_frames == set(leaves)
            for leaf in leaves:
                frame_path: str = '/'.join([path, leaf])
                obj = store.get(frame_path)
                if 'df' in leaf:
                    tm.assert_frame_equal(obj, objs[leaf])
                else:
                    tm.assert_series_equal(obj, objs[leaf])


def test_getattr(setup_path: str) -> None:
    with ensure_clean_store(setup_path) as store:
        s: Series = Series(
            np.arange(10, dtype=np.float64),
            index=date_range('2020-01-01', periods=10),
        )
        store['a'] = s
        result: Series = store.a
        tm.assert_series_equal(result, s)
        result = store.a
        tm.assert_series_equal(result, s)
        df: DataFrame = DataFrame(
            np.random.default_rng(2).standard_normal((10, 4)),
            columns=Index(list('ABCD')),
            index=date_range('2000-01-01', periods=10, freq='B'),
        )
        store['df'] = df
        result = store.df
        tm.assert_frame_equal(result, df)
        for x in ['d', 'mode', 'path', 'handle', 'complib']:
            msg: str = f"'HDFStore' object has no attribute '{x}'"
            with pytest.raises(AttributeError, match=msg):
                getattr(store, x)
        for x in ['mode', 'path', 'handle', 'complib']:
            getattr(store, f'_{x}')


def test_store_dropna(tmp_path: Path, setup_path: str) -> None:
    df_with_missing: DataFrame = DataFrame(
        {'col1': [0.0, np.nan, 2.0], 'col2': [1.0, np.nan, np.nan]},
        index=list('abc'),
    )
    df_without_missing: DataFrame = DataFrame(
        {'col1': [0.0, 2.0], 'col2': [1.0, np.nan]},
        index=list('ac'),
    )
    path: Path = tmp_path / setup_path
    df_with_missing.to_hdf(str(path), key='df', format='table')
    reloaded: DataFrame = read_hdf(str(path), 'df')
    tm.assert_frame_equal(df_with_missing, reloaded)
    path = tmp_path / setup_path
    df_with_missing.to_hdf(str(path), key='df', format='table', dropna=False)
    reloaded = read_hdf(str(path), 'df')
    tm.assert_frame_equal(df_with_missing, reloaded)
    path = tmp_path / setup_path
    df_with_missing.to_hdf(str(path), key='df', format='table', dropna=True)
    reloaded = read_hdf(str(path), 'df')
    tm.assert_frame_equal(df_without_missing, reloaded)


def test_to_hdf_with_min_itemsize(tmp_path: Path, setup_path: str) -> None:
    path: Path = tmp_path / setup_path
    df: DataFrame = DataFrame({
        'A': [0.0, 1.0, 2.0, 3.0, 4.0],
        'B': [0.0, 1.0, 0.0, 1.0, 0.0],
        'C': Index(['foo1', 'foo2', 'foo3', 'foo4', 'foo5']),
        'D': date_range('20130101', periods=5)
    }).set_index('C')
    df.to_hdf(str(path), key='ss3', format='table', min_itemsize={'index': 6})
    df2: DataFrame = df.copy().reset_index().assign(C='longer').set_index('C')
    df2.to_hdf(str(path), key='ss3', append=True, format='table')
    tm.assert_frame_equal(read_hdf(str(path), 'ss3'), concat([df, df2]))
    df['B'].to_hdf(str(path), key='ss4', format='table', min_itemsize={'index': 6})
    df2['B'].to_hdf(str(path), key='ss4', append=True, format='table')
    tm.assert_series_equal(read_hdf(str(path), 'ss4'), concat([df['B'], df2['B']]))


@pytest.mark.xfail(using_string_dtype() and has_pyarrow, reason="TODO(infer_string): can't encode '\ud800': surrogates not allowed")
@pytest.mark.parametrize('format', ['fixed', 'table'])
def test_to_hdf_errors(tmp_path: Path, format: str, setup_path: str) -> None:
    data: List[str] = ['\ud800foo']
    ser: Series = Series(data, index=Index(data))
    path: Path = tmp_path / setup_path
    ser.to_hdf(str(path), key='table', format=format, errors='surrogatepass')
    result: Series = read_hdf(str(path), 'table', errors='surrogatepass')
    tm.assert_series_equal(result, ser)


def test_create_table_index(setup_path: str) -> None:
    with ensure_clean_store(setup_path) as store:

        def col(t: str, column: str) -> Any:
            return getattr(store.get_storer(t).table.cols, column)

        df: DataFrame = DataFrame(
            np.random.default_rng(2).standard_normal((10, 4)),
            columns=Index(list('ABCD')),
            index=date_range('2000-01-01', periods=10, freq='B'),
        )
        df['string'] = 'foo'
        df['string2'] = 'bar'
        store.append('f', df, data_columns=['string', 'string2'])
        assert col('f', 'index').is_indexed is True
        assert col('f', 'string').is_indexed is True
        assert col('f', 'string2').is_indexed is True
        store.append('f2', df, index=['string'], data_columns=['string', 'string2'])
        assert col('f2', 'index').is_indexed is False
        assert col('f2', 'string').is_indexed is True
        assert col('f2', 'string2').is_indexed is False
        _maybe_remove(store, 'f2')
        store.put('f2', df)
        msg: str = 'cannot create table index on a Fixed format store'
        with pytest.raises(TypeError, match=msg):
            store.create_table_index('f2')


def test_create_table_index_data_columns_argument(setup_path: str) -> None:
    with ensure_clean_store(setup_path) as store:

        def col(t: str, column: str) -> Any:
            return getattr(store.get_storer(t).table.cols, column)

        df: DataFrame = DataFrame(
            np.random.default_rng(2).standard_normal((10, 4)),
            columns=Index(list('ABCD')),
            index=date_range('2000-01-01', periods=10, freq='B'),
        )
        df['string'] = 'foo'
        df['string2'] = 'bar'
        store.append('f', df, data_columns=['string'])
        assert col('f', 'index').is_indexed is True
        assert col('f', 'string').is_indexed is True
        msg: str = "'Cols' object has no attribute 'string2'"
        with pytest.raises(AttributeError, match=msg):
            col('f', 'string2').is_indexed
        msg = (
            'column string2 is not a data_column.\nIn order to read column string2 you must reload the dataframe \n'
            'into HDFStore and include string2 with the data_columns argument.'
        )
        with pytest.raises(AttributeError, match=msg):
            store.create_table_index('f', columns=['string2'])


def test_mi_data_columns(setup_path: str) -> None:
    idx = MultiIndex.from_arrays(
        [date_range('2000-01-01', periods=5), range(5)],
        names=['date', 'id'],
    )
    df: DataFrame = DataFrame({'a': [1.1, 1.2, 1.3, 1.4, 1.5]}, index=idx)
    with ensure_clean_store(setup_path) as store:
        store.append('df', df, data_columns=True)
        actual: DataFrame = store.select('df', where='id == 1')
        expected: DataFrame = df.iloc[[1], :]
        tm.assert_frame_equal(actual, expected)


def test_table_mixed_dtypes(setup_path: str) -> None:
    df: DataFrame = DataFrame(
        1.1 * np.arange(120).reshape((30, 4)),
        columns=Index(list('ABCD')),
        index=Index([f'i-{i}' for i in range(30)]),
    )
    df['obj1'] = 'foo'
    df['obj2'] = 'bar'
    df['bool1'] = df['A'] > 0
    df['bool2'] = df['B'] > 0
    df['bool3'] = True
    df['int1'] = 1
    df['int2'] = 2
    df['timestamp1'] = Timestamp('20010102').as_unit('ns')
    df['timestamp2'] = Timestamp('20010103').as_unit('ns')
    df['datetime1'] = Timestamp('20010102').as_unit('ns')
    df['datetime2'] = Timestamp('20010103').as_unit('ns')
    df.loc[df.index[3:6], ['obj1']] = np.nan
    df = df._consolidate()
    with ensure_clean_store(setup_path) as store:
        store.append('df1_mixed', df)
        tm.assert_frame_equal(store.select('df1_mixed'), df)


def test_calendar_roundtrip_issue(setup_path: str) -> None:
    weekmask_egypt: str = 'Sun Mon Tue Wed Thu'
    holidays = ['2012-05-01', dt.datetime(2013, 5, 1), np.datetime64('2014-05-01')]
    bday_egypt = pd.offsets.CustomBusinessDay(holidays=holidays, weekmask=weekmask_egypt)
    mydt = dt.datetime(2013, 4, 30)
    dts = date_range(mydt, periods=5, freq=bday_egypt)
    s: Series = Series(dts.weekday, dts).map(Series('Mon Tue Wed Thu Fri Sat Sun'.split()))
    with ensure_clean_store(setup_path) as store:
        store.put('fixed', s)
        result: Series = store.select('fixed')
        tm.assert_series_equal(result, s)
        store.append('table', s)
        result = store.select('table')
        tm.assert_series_equal(result, s)


def test_remove(setup_path: str) -> None:
    with ensure_clean_store(setup_path) as store:
        ts: Series = Series(
            np.arange(10, dtype=np.float64),
            index=date_range('2020-01-01', periods=10),
        )
        df: DataFrame = DataFrame(
            1.1 * np.arange(120).reshape((30, 4)),
            columns=Index(list('ABCD')),
            index=Index([f'i-{i}' for i in range(30)]),
        )
        store['a'] = ts
        store['b'] = df
        _maybe_remove(store, 'a')
        assert len(store) == 1
        tm.assert_frame_equal(df, store['b'])
        _maybe_remove(store, 'b')
        assert len(store) == 0
        with pytest.raises(KeyError, match="'No object named a_nonexistent_store in the file'"):
            store.remove('a_nonexistent_store')
        store['a'] = ts
        store['b/foo'] = df
        _maybe_remove(store, 'foo')
        _maybe_remove(store, 'b/foo')
        assert len(store) == 1
        store['a'] = ts
        store['b/foo'] = df
        _maybe_remove(store, 'b')
        assert len(store) == 1
        store['a'] = ts
        store['b'] = df
        del store['a']
        del store['b']
        assert len(store) == 0


def test_same_name_scoping(setup_path: str) -> None:
    with ensure_clean_store(setup_path) as store:
        df: DataFrame = DataFrame(
            np.random.default_rng(2).standard_normal((20, 2)),
            index=date_range('20130101', periods=20),
        )
        store.put('df', df, format='table')
        expected: DataFrame = df[df.index > Timestamp('20130105')]
        result: DataFrame = store.select('df', 'index>datetime.datetime(2013,1,5)')
        tm.assert_frame_equal(result, expected)
        result = store.select('df', 'index>datetime.datetime(2013,1,5)')
        tm.assert_frame_equal(result, expected)
        result = store.select('df', 'index>datetime(2013,1,5)')
        tm.assert_frame_equal(result, expected)


def test_store_index_name(setup_path: str) -> None:
    df: DataFrame = DataFrame(
        1.1 * np.arange(120).reshape((30, 4)),
        columns=Index(list('ABCD')),
        index=Index([f'i-{i}' for i in range(30)]),
    )
    df.index.name = 'foo'
    with ensure_clean_store(setup_path) as store:
        store['frame'] = df
        recons: DataFrame = store['frame']
        tm.assert_frame_equal(recons, df)


@pytest.mark.parametrize('tz', [None, 'US/Pacific'])
@pytest.mark.parametrize('table_format', ['table', 'fixed'])
def test_store_index_name_numpy_str(tmp_path: Path, table_format: str, setup_path: str, unit: Any, tz: Any) -> None:
    idx: DatetimeIndex = DatetimeIndex(
        [dt.date(2000, 1, 1), dt.date(2000, 1, 2)], name='colsג'
    ).tz_localize(tz).as_unit(unit)
    idx1: DatetimeIndex = DatetimeIndex(
        [dt.date(2010, 1, 1), dt.date(2010, 1, 2)], name='rowsא'
    ).as_unit(unit).tz_localize(tz)
    df: DataFrame = DataFrame(
        np.arange(4).reshape(2, 2), columns=idx, index=idx1
    )
    path: Path = tmp_path / setup_path
    df.to_hdf(str(path), key='df', format=table_format)
    df2: DataFrame = read_hdf(str(path), 'df')
    tm.assert_frame_equal(df, df2, check_names=True)
    assert isinstance(df2.index.name, str)
    assert isinstance(df2.columns.name, str)


def test_store_series_name(setup_path: str) -> None:
    df: DataFrame = DataFrame(
        1.1 * np.arange(120).reshape((30, 4)),
        columns=Index(list('ABCD')),
        index=Index([f'i-{i}' for i in range(30)]),
    )
    series: Series = df['A']
    with ensure_clean_store(setup_path) as store:
        store['series'] = series
        recons: Series = store['series']
        tm.assert_series_equal(recons, series)


def test_overwrite_node(setup_path: str) -> None:
    with ensure_clean_store(setup_path) as store:
        store['a'] = DataFrame(
            np.random.default_rng(2).standard_normal((10, 4)),
            columns=Index(list('ABCD')),
            index=date_range('2000-01-01', periods=10, freq='B'),
        )
        ts: Series = Series(
            np.arange(10, dtype=np.float64),
            index=date_range('2020-01-01', periods=10),
        )
        store['a'] = ts
        tm.assert_series_equal(store['a'], ts)


def test_coordinates(setup_path: str) -> None:
    df: DataFrame = DataFrame(
        np.random.default_rng(2).standard_normal((10, 4)),
        columns=Index(list('ABCD')),
        index=date_range('2000-01-01', periods=10, freq='B'),
    )
    with ensure_clean_store(setup_path) as store:
        _maybe_remove(store, 'df')
        store.append('df', df)
        c: Any = store.select_as_coordinates('df')
        assert (c.values == np.arange(len(df.index))).all()
        _maybe_remove(store, 'df')
        df = DataFrame({'A': range(5), 'B': range(5)})
        store.append('df', df)
        c = store.select_as_coordinates('df', ['index<3'])
        assert (c.values == np.arange(3)).all()
        result: DataFrame = store.select('df', where=c)
        expected: DataFrame = df.loc[0:2, :]
        tm.assert_frame_equal(result, expected)
        c = store.select_as_coordinates('df', ['index>=3', 'index<=4'])
        assert (c.values == np.arange(2) + 3).all()
        result = store.select('df', where=c)
        expected = df.loc[3:4, :]
        tm.assert_frame_equal(result, expected)
        assert isinstance(c, Index)
        _maybe_remove(store, 'df1')
        _maybe_remove(store, 'df2')
        df1: DataFrame = DataFrame(
            np.random.default_rng(2).standard_normal((10, 4)),
            columns=Index(list('ABCD')),
            index=date_range('2000-01-01', periods=10, freq='B'),
        )
        df2: DataFrame = df1.copy().rename(columns='{}_2'.format)
        store.append('df1', df1, data_columns=['A', 'B'])
        store.append('df2', df2)
        c = store.select_as_coordinates('df1', ['A>0', 'B>0'])
        df1_result: DataFrame = store.select('df1', c)
        df2_result: DataFrame = store.select('df2', c)
        result = concat([df1_result, df2_result], axis=1)
        expected = concat([df1, df2], axis=1)
        expected = expected[(expected.A > 0) & (expected.B > 0)]
        tm.assert_frame_equal(result, expected, check_freq=False)
    with ensure_clean_store(setup_path) as store:
        df = DataFrame(
            np.random.default_rng(2).standard_normal((1000, 2)),
            index=date_range('20000101', periods=1000),
        )
        store.append('df', df)
        c = store.select_column('df', 'index')
        where = DatetimeIndex(c)[DatetimeIndex(c).month == 5].index
        expected = df.iloc[where]
        result = store.select('df', where=where)
        tm.assert_frame_equal(result, expected)
        result = store.select('df', where=where)
        tm.assert_frame_equal(result, expected)
        msg: str = 'where must be passed as a string, PyTablesExpr, or list-like of PyTablesExpr'
        with pytest.raises(TypeError, match=msg):
            store.select('df', where=np.arange(len(df), dtype='float64'))
        with pytest.raises(TypeError, match=msg):
            store.select('df', where=np.arange(len(df) + 1))
        with pytest.raises(TypeError, match=msg):
            store.select('df', where=np.arange(len(df)), start=5)
        with pytest.raises(TypeError, match=msg):
            store.select('df', where=np.arange(len(df)), start=5, stop=10)
        selection = date_range('20000101', periods=500)
        result = store.select('df', where='index in selection')
        expected = df[df.index.isin(selection)]
        tm.assert_frame_equal(result, expected)
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 2)))
        store.append('df2', df)
        result = store.select('df2', where=[0, 3, 5])
        expected = df.iloc[[0, 3, 5]]
        tm.assert_frame_equal(result, expected)
        where = [True] * 10
        where[-2] = False
        result = store.select('df2', where=where)
        expected = df.loc[where]
        tm.assert_frame_equal(result, expected)
        result = store.select('df2', start=5, stop=10)
        expected = df[5:10]
        tm.assert_frame_equal(result, expected)


def test_start_stop_table(setup_path: str) -> None:
    with ensure_clean_store(setup_path) as store:
        df: DataFrame = DataFrame({
            'A': np.random.default_rng(2).random(20),
            'B': np.random.default_rng(2).random(20),
        })
        store.append('df', df)
        result: DataFrame = store.select('df', "columns=['A']", start=0, stop=5)
        expected: DataFrame = df.loc[0:4, ['A']]
        tm.assert_frame_equal(result, expected)
        result = store.select('df', "columns=['A']", start=30, stop=40)
        assert len(result) == 0
        expected = df.loc[30:40, ['A']]
        tm.assert_frame_equal(result, expected)


def test_start_stop_multiple(setup_path: str) -> None:
    with ensure_clean_store(setup_path) as store:
        df: DataFrame = DataFrame({'foo': [1, 2], 'bar': [1, 2]})
        store.append_to_multiple({'selector': ['foo'], 'data': None}, df, selector='selector')
        result: DataFrame = store.select_as_multiple(['selector', 'data'], selector='selector', start=0, stop=1)
        expected: DataFrame = df.loc[[0], ['foo', 'bar']]
        tm.assert_frame_equal(result, expected)


def test_start_stop_fixed(setup_path: str) -> None:
    with ensure_clean_store(setup_path) as store:
        df: DataFrame = DataFrame({
            'A': np.random.default_rng(2).random(20),
            'B': np.random.default_rng(2).random(20),
        }, index=date_range('20130101', periods=20))
        store.put('df', df)
        result: DataFrame = store.select('df', start=0, stop=5)
        expected: DataFrame = df.iloc[0:5, :]
        tm.assert_frame_equal(result, expected)
        result = store.select('df', start=5, stop=10)
        expected = df.iloc[5:10, :]
        tm.assert_frame_equal(result, expected)
        df = DataFrame(
            1.1 * np.arange(120).reshape((30, 4)),
            columns=Index(list('ABCD')),
            index=Index([f'i-{i}' for i in range(30)]),
        )
        df.iloc[3:5, 1:3] = np.nan
        df.iloc[8:10, -2] = np.nan


def test_select_filter_corner(setup_path: str, request: Any) -> None:
    df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((50, 100)))
    df.index = [f'{c:3d}' for c in df.index]
    df.columns = [f'{c:3d}' for c in df.columns]
    with ensure_clean_store(setup_path) as store:
        store.put('frame', df, format='table')
        request.applymarker(pytest.mark.xfail(PY312, reason='AST change in PY312', raises=ValueError))
        crit: str = 'columns=df.columns[:75]'
        result: DataFrame = store.select('frame', [crit])
        tm.assert_frame_equal(result, df.loc[:, df.columns[:75]])
        crit = 'columns=df.columns[:75:2]'
        result = store.select('frame', [crit])
        tm.assert_frame_equal(result, df.loc[:, df.columns[:75:2]])


def test_path_pathlib() -> None:
    df: DataFrame = DataFrame(
        1.1 * np.arange(120).reshape((30, 4)),
        columns=Index(list('ABCD')),
        index=Index([f'i-{i}' for i in range(30)]),
    )
    result: DataFrame = tm.round_trip_pathlib(lambda p: df.to_hdf(p, key='df'), lambda p: read_hdf(p, 'df'))
    tm.assert_frame_equal(df, result)


@pytest.mark.parametrize('start, stop', [(0, 2), (1, 2), (None, None)])
def test_contiguous_mixed_data_table(start: Any, stop: Any, setup_path: str) -> None:
    df: DataFrame = DataFrame({'a': Series([20111010, 20111011, 20111012]), 'b': Series(['ab', 'cd', 'ab'])})
    with ensure_clean_store(setup_path) as store:
        store.append('test_dataset', df)
        result: DataFrame = store.select('test_dataset', start=start, stop=stop)
        tm.assert_frame_equal(df[start:stop], result)


def test_path_pathlib_hdfstore() -> None:
    df: DataFrame = DataFrame(
        1.1 * np.arange(120).reshape((30, 4)),
        columns=Index(list('ABCD')),
        index=Index([f'i-{i}' for i in range(30)]),
    )

    def writer(path: Path) -> None:
        with HDFStore(str(path)) as store:
            df.to_hdf(store, key='df')

    def reader(path: Path) -> DataFrame:
        with HDFStore(str(path)) as store:
            return read_hdf(store, 'df')

    result: DataFrame = tm.round_trip_pathlib(writer, reader)
    tm.assert_frame_equal(df, result)


def test_pickle_path_localpath() -> None:
    df: DataFrame = DataFrame(
        1.1 * np.arange(120).reshape((30, 4)),
        columns=Index(list('ABCD')),
        index=Index([f'i-{i}' for i in range(30)]),
    )
    result: DataFrame = tm.round_trip_pathlib(lambda p: df.to_hdf(p, key='df'), lambda p: read_hdf(p, 'df'))
    tm.assert_frame_equal(df, result)


@pytest.mark.parametrize('propindexes', [True, False])
def test_copy(propindexes: bool) -> None:
    df: DataFrame = DataFrame(
        1.1 * np.arange(120).reshape((30, 4)),
        columns=Index(list('ABCD')),
        index=Index([f'i-{i}' for i in range(30)]),
    )
    with tm.ensure_clean() as path:
        with HDFStore(str(path)) as st:
            st.append('df', df, data_columns=['A'])
        with tempfile.NamedTemporaryFile() as new_f:
            with HDFStore(str(path)) as store:
                with contextlib.closing(store.copy(new_f.name, keys=None, propindexes=propindexes)) as tstore:
                    keys = store.keys()
                    assert set(keys) == set(tstore.keys())
                    for k in tstore.keys():
                        if tstore.get_storer(k).is_table:
                            new_t = tstore.get_storer(k)
                            orig_t = store.get_storer(k)
                            assert orig_t.nrows == new_t.nrows
                            if propindexes:
                                for a in orig_t.axes:
                                    if a.is_indexed:
                                        assert new_t[a.name].is_indexed


def test_duplicate_column_name(tmp_path: Path, setup_path: str) -> None:
    df: DataFrame = DataFrame(columns=['a', 'a'], data=[[0, 0]])
    path: Path = tmp_path / setup_path
    msg: str = 'Columns index has to be unique for fixed format'
    with pytest.raises(ValueError, match=msg):
        df.to_hdf(str(path), key='df', format='fixed')
    df.to_hdf(str(path), key='df', format='table')
    other: DataFrame = read_hdf(str(path), 'df')
    tm.assert_frame_equal(df, other)
    assert df.equals(other)
    assert other.equals(df)


def test_preserve_timedeltaindex_type(setup_path: str) -> None:
    df: DataFrame = DataFrame(np.random.default_rng(2).normal(size=(10, 5)))
    df.index = timedelta_range(start='0s', periods=10, freq='1s', name='example')
    with ensure_clean_store(setup_path) as store:
        store['df'] = df
        tm.assert_frame_equal(store['df'], df)


def test_columns_multiindex_modified(tmp_path: Path, setup_path: str) -> None:
    df: DataFrame = DataFrame(
        np.random.default_rng(2).random((4, 5)),
        index=list('abcd'),
        columns=list('ABCDE'),
    )
    df.index.name = 'letters'
    df = df.set_index(keys='E', append=True)
    data_columns: List[Any] = df.index.names + df.columns.tolist()
    path: Path = tmp_path / setup_path
    df.to_hdf(str(path), key='df', mode='a', append=True, data_columns=data_columns, index=False)
    cols2load: List[str] = list('BCD')
    cols2load_original: List[str] = list(cols2load)
    read_hdf(str(path), 'df', columns=cols2load)
    assert cols2load_original == cols2load


@pytest.mark.filterwarnings('ignore:PeriodDtype\\[B\\] is deprecated:FutureWarning')
@pytest.mark.parametrize(
    'columns',
    [
        Index([0, 1], dtype=np.int64),
        Index([0.0, 1.0], dtype=np.float64),
        date_range('2020-01-01', periods=2),
        timedelta_range('1 day', periods=2),
        period_range('2020-01-01', periods=2, freq='D'),
    ],
)
def test_to_hdf_with_object_column_names_should_fail(tmp_path: Path, setup_path: str, columns: Index) -> None:
    df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((10, 2)), columns=columns)
    path: Path = tmp_path / setup_path
    msg: str = 'cannot have non-object label DataIndexableCol'
    with pytest.raises(ValueError, match=msg):
        df.to_hdf(str(path), key='df', format='table', data_columns=True)


@pytest.mark.parametrize('dtype', [None, 'category'])
def test_to_hdf_with_object_column_names_should_run(tmp_path: Path, setup_path: str, dtype: Any) -> None:
    df: DataFrame = DataFrame(
        np.random.default_rng(2).standard_normal((10, 2)),
        columns=Index(['a', 'b'], dtype=dtype),
    )
    path: Path = tmp_path / setup_path
    df.to_hdf(str(path), key='df', format='table', data_columns=True)
    result: DataFrame = read_hdf(str(path), 'df', where=f'index = [{df.index[0]}]')
    assert len(result)


def test_hdfstore_strides(setup_path: str) -> None:
    df: DataFrame = DataFrame({'a': [1, 2, 3, 4], 'b': [5, 6, 7, 8]})
    with ensure_clean_store(setup_path) as store:
        store.put('df', df)
        assert df['a'].values.strides == store['df']['a'].values.strides


def test_store_bool_index(tmp_path: Path, setup_path: str) -> None:
    df: DataFrame = DataFrame([[1]], columns=[True], index=Index([False], dtype='bool'))
    expected: DataFrame = df.copy()
    path: Path = tmp_path / setup_path
    df.to_hdf(str(path), key='a')
    result: DataFrame = read_hdf(str(path), 'a')
    tm.assert_frame_equal(expected, result)
