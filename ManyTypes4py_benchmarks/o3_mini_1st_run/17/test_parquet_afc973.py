#!/usr/bin/env python3
"""test parquet compat"""
import datetime
from decimal import Decimal
from io import BytesIO
import os
import pathlib
from typing import Any, Dict, Iterable, Optional, Type, Union

import numpy as np
import pytest
from pandas._config import using_string_dtype
from pandas.compat import is_platform_windows
from pandas.compat.pyarrow import (
    pa_version_under11p0,
    pa_version_under13p0,
    pa_version_under15p0,
    pa_version_under17p0,
    pa_version_under19p0,
)
import pandas as pd
import pandas._testing as tm
from pandas.util.version import Version
from pandas.io.parquet import FastParquetImpl, PyArrowImpl, get_engine, read_parquet, to_parquet

try:
    import pyarrow
    _HAVE_PYARROW = True
except ImportError:
    _HAVE_PYARROW = False
try:
    import fastparquet
    _HAVE_FASTPARQUET = True
except ImportError:
    _HAVE_FASTPARQUET = False

pytestmark = [
    pytest.mark.filterwarnings(
        'ignore:DataFrame._data is deprecated:FutureWarning'
    ),
    pytest.mark.filterwarnings(
        'ignore:Passing a BlockManager to DataFrame:DeprecationWarning'
    ),
]


@pytest.fixture(
    params=[
        pytest.param(
            'fastparquet',
            marks=[
                pytest.mark.skipif(
                    not _HAVE_FASTPARQUET, reason='fastparquet is not installed'
                ),
                pytest.mark.xfail(
                    using_string_dtype(), reason='TODO(infer_string) fastparquet', strict=False
                ),
            ],
        ),
        pytest.param(
            'pyarrow',
            marks=pytest.mark.skipif(
                not _HAVE_PYARROW, reason='pyarrow is not installed'
            ),
        ),
    ]
)
def engine(request: pytest.FixtureRequest) -> str:
    return request.param


@pytest.fixture
def pa() -> str:
    if not _HAVE_PYARROW:
        pytest.skip('pyarrow is not installed')
    return 'pyarrow'


@pytest.fixture
def fp(request: pytest.FixtureRequest) -> str:
    if not _HAVE_FASTPARQUET:
        pytest.skip('fastparquet is not installed')
    if using_string_dtype():
        request.applymarker(pytest.mark.xfail(reason='TODO(infer_string) fastparquet', strict=False))
    return 'fastparquet'


@pytest.fixture
def df_compat() -> pd.DataFrame:
    return pd.DataFrame({'A': [1, 2, 3], 'B': 'foo'}, columns=pd.Index(['A', 'B']))


@pytest.fixture
def df_cross_compat() -> pd.DataFrame:
    df = pd.DataFrame(
        {
            'a': list('abc'),
            'b': list(range(1, 4)),
            'd': np.arange(4.0, 7.0, dtype='float64'),
            'e': [True, False, True],
            'f': pd.date_range('20130101', periods=3),
        }
    )
    return df


@pytest.fixture
def df_full() -> pd.DataFrame:
    return pd.DataFrame({
        'string': list('abc'),
        'string_with_nan': ['a', np.nan, 'c'],
        'string_with_none': ['a', None, 'c'],
        'bytes': [b'foo', b'bar', b'baz'],
        'unicode': ['foo', 'bar', 'baz'],
        'int': list(range(1, 4)),
        'uint': np.arange(3, 6).astype('u1'),
        'float': np.arange(4.0, 7.0, dtype='float64'),
        'float_with_nan': [2.0, np.nan, 3.0],
        'bool': [True, False, True],
        'datetime': pd.date_range('20130101', periods=3),
        'datetime_with_nat': [pd.Timestamp('20130101'), pd.NaT, pd.Timestamp('20130103')]
    })


@pytest.fixture(
    params=[
        datetime.datetime.now(datetime.timezone.utc),
        datetime.datetime.now(datetime.timezone.min),
        datetime.datetime.now(datetime.timezone.max),
        datetime.datetime.strptime('2019-01-04T16:41:24+0200', '%Y-%m-%dT%H:%M:%S%z'),
        datetime.datetime.strptime('2019-01-04T16:41:24+0215', '%Y-%m-%dT%H:%M:%S%z'),
        datetime.datetime.strptime('2019-01-04T16:41:24-0200', '%Y-%m-%dT%H:%M:%S%z'),
        datetime.datetime.strptime('2019-01-04T16:41:24-0215', '%Y-%m-%dT%H:%M:%S%z')
    ]
)
def timezone_aware_date_list(request: pytest.FixtureRequest) -> datetime.datetime:
    return request.param


def check_round_trip(
    df: pd.DataFrame,
    engine: Optional[str] = None,
    path: Optional[Union[str, pathlib.Path]] = None,
    write_kwargs: Optional[Dict[str, Any]] = None,
    read_kwargs: Optional[Dict[str, Any]] = None,
    expected: Optional[pd.DataFrame] = None,
    check_names: bool = True,
    check_like: bool = False,
    check_dtype: bool = True,
    repeat: int = 2,
) -> None:
    """Verify parquet serializer and deserializer produce the same results.

    Performs a pandas to disk and disk to pandas round trip,
    then compares the 2 resulting DataFrames to verify equality.
    """
    write_kwargs = write_kwargs or {'compression': None}
    read_kwargs = read_kwargs or {}
    if expected is None:
        expected = df

    if engine:
        write_kwargs['engine'] = engine
        read_kwargs['engine'] = engine

    def compare(repeat_times: int) -> None:
        for _ in range(repeat_times):
            df.to_parquet(path, **write_kwargs)
            actual: pd.DataFrame = read_parquet(path, **read_kwargs)
            if 'string_with_nan' in expected:
                expected.loc[1, 'string_with_nan'] = None
            tm.assert_frame_equal(
                expected, actual, check_names=check_names, check_like=check_like, check_dtype=check_dtype
            )
    if path is None:
        with tm.ensure_clean() as temp_path:
            compare(repeat)
    else:
        compare(repeat)


def check_partition_names(path: str, expected: Iterable[str]) -> None:
    """Check partitions of a parquet file are as expected.
    """
    import pyarrow.dataset as ds
    dataset = ds.dataset(path, partitioning='hive')
    assert dataset.partitioning.schema.names == list(expected)


def test_invalid_engine(df_compat: pd.DataFrame) -> None:
    msg = "engine must be one of 'pyarrow', 'fastparquet'"
    with pytest.raises(ValueError, match=msg):
        check_round_trip(df_compat, 'foo', 'bar')


def test_options_py(df_compat: pd.DataFrame, pa: str, using_infer_string: bool) -> None:
    if using_infer_string and (not pa_version_under19p0):
        df_compat.columns = df_compat.columns.astype('str')
    with pd.option_context('io.parquet.engine', 'pyarrow'):
        check_round_trip(df_compat)


def test_options_fp(df_compat: pd.DataFrame, fp: str) -> None:
    with pd.option_context('io.parquet.engine', 'fastparquet'):
        check_round_trip(df_compat)


def test_options_auto(df_compat: pd.DataFrame, fp: str, pa: str) -> None:
    with pd.option_context('io.parquet.engine', 'auto'):
        check_round_trip(df_compat)


def test_options_get_engine(fp: str, pa: str) -> None:
    assert isinstance(get_engine('pyarrow'), PyArrowImpl)
    assert isinstance(get_engine('fastparquet'), FastParquetImpl)
    with pd.option_context('io.parquet.engine', 'pyarrow'):
        assert isinstance(get_engine('auto'), PyArrowImpl)
        assert isinstance(get_engine('pyarrow'), PyArrowImpl)
        assert isinstance(get_engine('fastparquet'), FastParquetImpl)
    with pd.option_context('io.parquet.engine', 'fastparquet'):
        assert isinstance(get_engine('auto'), FastParquetImpl)
        assert isinstance(get_engine('pyarrow'), PyArrowImpl)
        assert isinstance(get_engine('fastparquet'), FastParquetImpl)
    with pd.option_context('io.parquet.engine', 'auto'):
        assert isinstance(get_engine('auto'), PyArrowImpl)
        assert isinstance(get_engine('pyarrow'), PyArrowImpl)
        assert isinstance(get_engine('fastparquet'), FastParquetImpl)


def test_get_engine_auto_error_message() -> None:
    from pandas.compat._optional import VERSIONS
    pa_min_ver: str = VERSIONS.get('pyarrow')
    fp_min_ver: str = VERSIONS.get('fastparquet')
    have_pa_bad_version: bool = False if not _HAVE_PYARROW else Version(pyarrow.__version__) < Version(pa_min_ver)
    have_fp_bad_version: bool = False if not _HAVE_FASTPARQUET else Version(fastparquet.__version__) < Version(fp_min_ver)
    have_usable_pa: bool = _HAVE_PYARROW and (not have_pa_bad_version)
    have_usable_fp: bool = _HAVE_FASTPARQUET and (not have_fp_bad_version)
    if not have_usable_pa and (not have_usable_fp):
        if have_pa_bad_version:
            match: str = f'Pandas requires version .{pa_min_ver}. or newer of .pyarrow.'
            with pytest.raises(ImportError, match=match):
                get_engine('auto')
        else:
            match = 'Missing optional dependency .pyarrow.'
            with pytest.raises(ImportError, match=match):
                get_engine('auto')
        if have_fp_bad_version:
            match = f'Pandas requires version .{fp_min_ver}. or newer of .fastparquet.'
            with pytest.raises(ImportError, match=match):
                get_engine('auto')
        else:
            match = 'Missing optional dependency .fastparquet.'
            with pytest.raises(ImportError, match=match):
                get_engine('auto')


def test_cross_engine_pa_fp(df_cross_compat: pd.DataFrame, pa: str, fp: str) -> None:
    df = df_cross_compat
    with tm.ensure_clean() as path:
        df.to_parquet(path, engine=pa, compression=None)
        result: pd.DataFrame = read_parquet(path, engine=fp)
        tm.assert_frame_equal(result, df)
        result = read_parquet(path, engine=fp, columns=['a', 'd'])
        tm.assert_frame_equal(result, df[['a', 'd']])


def test_cross_engine_fp_pa(df_cross_compat: pd.DataFrame, pa: str, fp: str) -> None:
    df = df_cross_compat
    with tm.ensure_clean() as path:
        df.to_parquet(path, engine=fp, compression=None)
        result: pd.DataFrame = read_parquet(path, engine=pa)
        tm.assert_frame_equal(result, df)
        result = read_parquet(path, engine=pa, columns=['a', 'd'])
        tm.assert_frame_equal(result, df[['a', 'd']])


class Base:
    def check_error_on_write(
        self, df: Any, engine: str, exc: Type[BaseException], err_msg: str
    ) -> None:
        with tm.ensure_clean() as path:
            with pytest.raises(exc, match=err_msg):
                to_parquet(df, path, engine, compression=None)

    def check_external_error_on_write(
        self, df: Any, engine: str, exc: Type[BaseException]
    ) -> None:
        with tm.ensure_clean() as path:
            with tm.external_error_raised(exc):
                to_parquet(df, path, engine, compression=None)


class TestBasic(Base):
    def test_error(self, engine: str) -> None:
        for obj in [pd.Series([1, 2, 3]), 1, 'foo', pd.Timestamp('20130101'), np.array([1, 2, 3])]:
            msg: str = 'to_parquet only supports IO with DataFrames'
            self.check_error_on_write(obj, engine, ValueError, msg)

    def test_columns_dtypes(self, engine: str) -> None:
        df = pd.DataFrame({'string': list('abc'), 'int': list(range(1, 4))})
        df.columns = ['foo', 'bar']
        check_round_trip(df, engine)

    @pytest.mark.parametrize('compression', [None, 'gzip', 'snappy', 'brotli'])
    def test_compression(self, engine: str, compression: Optional[str]) -> None:
        df = pd.DataFrame({'A': [1, 2, 3]})
        check_round_trip(df, engine, write_kwargs={'compression': compression})

    def test_read_columns(self, engine: str) -> None:
        df = pd.DataFrame({'string': list('abc'), 'int': list(range(1, 4))})
        expected = pd.DataFrame({'string': list('abc')})
        check_round_trip(df, engine, expected=expected, read_kwargs={'columns': ['string']})

    def test_read_filters(self, engine: str, tmp_path: pathlib.Path) -> None:
        df = pd.DataFrame({'int': list(range(4)), 'part': list('aabb')})
        expected = pd.DataFrame({'int': [0, 1]})
        check_round_trip(
            df,
            engine,
            path=str(tmp_path),
            expected=expected,
            write_kwargs={'partition_cols': ['part']},
            read_kwargs={'filters': [('part', '==', 'a')], 'columns': ['int']},
            repeat=1,
        )

    def test_write_index(self) -> None:
        pytest.importorskip('pyarrow')
        df = pd.DataFrame({'A': [1, 2, 3]})
        check_round_trip(df, 'pyarrow')
        indexes = [[2, 3, 4], pd.date_range('20130101', periods=3), list('abc'), [1, 3, 4]]
        for index in indexes:
            df.index = index
            if isinstance(index, pd.DatetimeIndex):
                df.index = df.index._with_freq(None)
            check_round_trip(df, 'pyarrow')
        df.index = [0, 1, 2]
        df.index.name = 'foo'
        check_round_trip(df, 'pyarrow')

    def test_write_multiindex(self, pa: str) -> None:
        engine: str = pa
        df = pd.DataFrame({'A': [1, 2, 3]})
        index = pd.MultiIndex.from_tuples([('a', 1), ('a', 2), ('b', 1)])
        df.index = index
        check_round_trip(df, engine)

    def test_multiindex_with_columns(self, pa: str) -> None:
        engine: str = pa
        dates = pd.date_range('01-Jan-2018', '01-Dec-2018', freq='MS')
        df = pd.DataFrame(np.random.default_rng(2).standard_normal((2 * len(dates), 3)), columns=list('ABC'))
        index1 = pd.MultiIndex.from_product([['Level1', 'Level2'], dates], names=['level', 'date'])
        index2 = index1.copy()
        for index in [index1, index2]:
            df.index = index
            check_round_trip(df, engine)
            check_round_trip(df, engine, read_kwargs={'columns': ['A', 'B']}, expected=df[['A', 'B']])

    def test_write_ignoring_index(self, engine: str) -> None:
        df = pd.DataFrame({'a': [1, 2, 3], 'b': ['q', 'r', 's']})
        write_kwargs: Dict[str, Any] = {'compression': None, 'index': False}
        expected = df.reset_index(drop=True)
        check_round_trip(df, engine, write_kwargs=write_kwargs, expected=expected)
        df = pd.DataFrame({'a': [1, 2, 3], 'b': ['q', 'r', 's']}, index=['zyx', 'wvu', 'tsr'])
        check_round_trip(df, engine, write_kwargs=write_kwargs, expected=expected)
        arrays = [['bar', 'bar', 'baz', 'baz', 'foo', 'foo', 'qux', 'qux'],
                  ['one', 'two', 'one', 'two', 'one', 'two', 'one', 'two']]
        df = pd.DataFrame({'one': list(range(8)), 'two': [-i for i in range(8)]}, index=arrays)
        expected = df.reset_index(drop=True)
        check_round_trip(df, engine, write_kwargs=write_kwargs, expected=expected)

    def test_write_column_multiindex(self, engine: str) -> None:
        mi_columns = pd.MultiIndex.from_tuples([('a', 1), ('a', 2), ('b', 1)])
        df = pd.DataFrame(np.random.default_rng(2).standard_normal((4, 3)), columns=mi_columns)
        if engine == 'fastparquet':
            self.check_error_on_write(df, engine, TypeError, 'Column name must be a string')
        elif engine == 'pyarrow':
            check_round_trip(df, engine)

    def test_write_column_multiindex_nonstring(self, engine: str) -> None:
        arrays = [['bar', 'bar', 'baz', 'baz', 'foo', 'foo', 'qux', 'qux'], [1, 2, 1, 2, 1, 2, 1, 2]]
        df = pd.DataFrame(np.random.default_rng(2).standard_normal((8, 8)), columns=arrays)
        df.columns.names = ['Level1', 'Level2']
        if engine == 'fastparquet':
            self.check_error_on_write(df, engine, ValueError, 'Column name')
        elif engine == 'pyarrow':
            check_round_trip(df, engine)

    def test_write_column_multiindex_string(self, pa: str) -> None:
        engine: str = pa
        arrays = [['bar', 'bar', 'baz', 'baz', 'foo', 'foo', 'qux', 'qux'],
                  ['one', 'two', 'one', 'two', 'one', 'two', 'one', 'two']]
        df = pd.DataFrame(np.random.default_rng(2).standard_normal((8, 8)), columns=arrays)
        df.columns.names = ['ColLevel1', 'ColLevel2']
        check_round_trip(df, engine)

    def test_write_column_index_string(self, pa: str) -> None:
        engine: str = pa
        arrays = ['bar', 'baz', 'foo', 'qux']
        df = pd.DataFrame(np.random.default_rng(2).standard_normal((8, 4)), columns=arrays)
        df.columns.name = 'StringCol'
        check_round_trip(df, engine)

    def test_write_column_index_nonstring(self, engine: str) -> None:
        arrays = [1, 2, 3, 4]
        df = pd.DataFrame(np.random.default_rng(2).standard_normal((8, 4)), columns=arrays)
        df.columns.name = 'NonStringCol'
        if engine == 'fastparquet':
            self.check_error_on_write(df, engine, TypeError, 'Column name must be a string')
        else:
            check_round_trip(df, engine)

    def test_dtype_backend(self, engine: str, request: pytest.FixtureRequest) -> None:
        pq = pytest.importorskip('pyarrow.parquet')
        if engine == 'fastparquet':
            mark = pytest.mark.xfail(reason='Fastparquet nullable dtype support is disabled')
            request.applymarker(mark)
        import pyarrow  # type: ignore
        table = pyarrow.table({
            'a': pyarrow.array([1, 2, 3, None], 'int64'),
            'b': pyarrow.array([1, 2, 3, None], 'uint8'),
            'c': pyarrow.array(['a', 'b', 'c', None]),
            'd': pyarrow.array([True, False, True, None]),
            'e': pyarrow.array([1, 2, 3, 4], 'int64'),
            'f': pyarrow.array([1.0, 2.0, 3.0, None], 'float32'),
            'g': pyarrow.array([1.0, 2.0, 3.0, None], 'float64'),
        })
        with tm.ensure_clean() as path:
            pq.write_table(table, path)
            result1: pd.DataFrame = read_parquet(path, engine=engine)
            result2: pd.DataFrame = read_parquet(path, engine=engine, dtype_backend='numpy_nullable')
        assert result1['a'].dtype == np.dtype('float64')
        expected = pd.DataFrame({
            'a': pd.array([1, 2, 3, None], dtype='Int64'),
            'b': pd.array([1, 2, 3, None], dtype='UInt8'),
            'c': pd.array(['a', 'b', 'c', None], dtype='string'),
            'd': pd.array([True, False, True, None], dtype='boolean'),
            'e': pd.array([1, 2, 3, 4], dtype='Int64'),
            'f': pd.array([1.0, 2.0, 3.0, None], dtype='Float32'),
            'g': pd.array([1.0, 2.0, 3.0, None], dtype='Float64')
        })
        if engine == 'fastparquet':
            result2 = result2.drop('c', axis=1)
            expected = expected.drop('c', axis=1)
        tm.assert_frame_equal(result2, expected)

    @pytest.mark.parametrize('dtype', ['Int64', 'UInt8', 'boolean', 'object', 'datetime64[ns, UTC]', 'float', 'period[D]', 'Float64', 'string'])
    def test_read_empty_array(self, pa: str, dtype: str) -> None:
        df = pd.DataFrame({'value': pd.array([], dtype=dtype)})
        pytest.importorskip('pyarrow', '11.0.0')
        expected: Optional[pd.DataFrame] = None
        if dtype == 'float':
            expected = pd.DataFrame({'value': pd.array([], dtype='Float64')})
        check_round_trip(df, pa, read_kwargs={'dtype_backend': 'numpy_nullable'}, expected=expected)
    
    @pytest.mark.network
    @pytest.mark.single_cpu
    def test_parquet_read_from_url(self, httpserver: Any, datapath: Any, df_compat: pd.DataFrame, engine: str) -> None:
        if engine != 'auto':
            pytest.importorskip(engine)
        with open(datapath('io', 'data', 'parquet', 'simple.parquet'), mode='rb') as f:
            httpserver.serve_content(content=f.read())
            df: pd.DataFrame = read_parquet(httpserver.url, engine=engine)
        expected = df_compat
        if pa_version_under19p0:
            expected.columns = expected.columns.astype(object)
        tm.assert_frame_equal(df, expected)


class TestParquetPyArrow(Base):
    @pytest.mark.xfail(reason="datetime_with_nat unit doesn't round-trip")
    def test_basic(self, pa: str, df_full: pd.DataFrame) -> None:
        df = df_full
        pytest.importorskip('pyarrow', '11.0.0')
        dti: pd.DatetimeIndex = pd.date_range('20130101', periods=3, tz='Europe/Brussels')
        dti = dti._with_freq(None)
        df['datetime_tz'] = dti
        df['bool_with_none'] = [True, None, True]
        check_round_trip(df, pa)

    def test_basic_subset_columns(self, pa: str, df_full: pd.DataFrame) -> None:
        df = df_full
        df['datetime_tz'] = pd.date_range('20130101', periods=3, tz='Europe/Brussels')
        check_round_trip(
            df, pa,
            expected=df[['string', 'int']],
            read_kwargs={'columns': ['string', 'int']}
        )

    def test_to_bytes_without_path_or_buf_provided(self, pa: str, df_full: pd.DataFrame) -> None:
        buf_bytes: bytes = df_full.to_parquet(engine=pa)  # type: ignore
        assert isinstance(buf_bytes, bytes)
        buf_stream = BytesIO(buf_bytes)
        res: pd.DataFrame = read_parquet(buf_stream)
        expected = df_full.copy()
        expected.loc[1, 'string_with_nan'] = None
        if pa_version_under11p0:
            expected['datetime_with_nat'] = expected['datetime_with_nat'].astype('M8[ns]')
        else:
            expected['datetime_with_nat'] = expected['datetime_with_nat'].astype('M8[ms]')
        tm.assert_frame_equal(res, expected)

    def test_duplicate_columns(self, pa: str) -> None:
        df = pd.DataFrame(np.arange(12).reshape(4, 3), columns=list('aaa')).copy()
        self.check_error_on_write(df, pa, ValueError, 'Duplicate column names found')

    def test_timedelta(self, pa: str) -> None:
        df = pd.DataFrame({'a': pd.timedelta_range('1 day', periods=3)})
        check_round_trip(df, pa)

    def test_unsupported(self, pa: str) -> None:
        df = pd.DataFrame({'a': ['a', 1, 2.0]})
        self.check_external_error_on_write(df, pa, pyarrow.ArrowException)

    def test_unsupported_float16(self, pa: str) -> None:
        data = np.arange(2, 10, dtype=np.float16)
        df = pd.DataFrame(data=data, columns=['fp16'])
        if pa_version_under15p0:
            self.check_external_error_on_write(df, pa, pyarrow.ArrowException)
        else:
            check_round_trip(df, pa)

    @pytest.mark.xfail(is_platform_windows(), reason='PyArrow does not cleanup of partial files dumps when unsupported dtypes are passed to_parquet function in windows')
    @pytest.mark.skipif(not pa_version_under15p0, reason='float16 works on 15')
    @pytest.mark.parametrize('path_type', [str, pathlib.Path], ids=['string', 'pathlib.Path'])
    def test_unsupported_float16_cleanup(self, pa: str, path_type: Union[Type[str], Type[pathlib.Path]]) -> None:
        data = np.arange(2, 10, dtype=np.float16)
        df = pd.DataFrame(data=data, columns=['fp16'])
        with tm.ensure_clean() as path_str:
            path = path_type(path_str)  # type: ignore
            with tm.external_error_raised(pyarrow.ArrowException):
                df.to_parquet(path=path, engine=pa)
            assert not os.path.isfile(str(path))

    def test_categorical(self, pa: str) -> None:
        df = pd.DataFrame({
            'a': pd.Categorical(list('abcdef')),
            'b': pd.Categorical(
                ['bar', 'foo', 'foo', 'bar', None, 'bar'], 
                dtype=pd.CategoricalDtype(['foo', 'bar', 'baz'])
            ),
            'c': pd.Categorical(
                ['a', 'b', 'c', 'a', 'c', 'b'], 
                categories=['b', 'c', 'd'], ordered=True
            )
        })
        check_round_trip(df, pa)

    @pytest.mark.single_cpu
    def test_s3_roundtrip_explicit_fs(self, df_compat: pd.DataFrame, s3_public_bucket: Any, pa: str, s3so: Dict[str, Any]) -> None:
        s3fs = pytest.importorskip('s3fs')
        s3 = s3fs.S3FileSystem(**s3so)
        kw: Dict[str, Any] = {'filesystem': s3}
        check_round_trip(df_compat, pa, path=f'{s3_public_bucket.name}/pyarrow.parquet', read_kwargs=kw, write_kwargs=kw)

    @pytest.mark.single_cpu
    def test_s3_roundtrip(self, df_compat: pd.DataFrame, s3_public_bucket: Any, pa: str, s3so: Dict[str, Any]) -> None:
        s3so_opts: Dict[str, Any] = {'storage_options': s3so}
        check_round_trip(df_compat, pa, path=f's3://{s3_public_bucket.name}/pyarrow.parquet', read_kwargs=s3so_opts, write_kwargs=s3so_opts)

    @pytest.mark.single_cpu
    @pytest.mark.parametrize('partition_col', [['A'], []])
    def test_s3_roundtrip_for_dir(
        self, df_compat: pd.DataFrame, s3_public_bucket: Any, pa: str, partition_col: Iterable[Any], s3so: Dict[str, Any]
    ) -> None:
        pytest.importorskip('s3fs')
        expected_df = df_compat.copy()
        if partition_col:
            expected_df = expected_df.astype(dict.fromkeys(partition_col, np.int32))
            partition_col_type: str = 'category'
            expected_df[partition_col] = expected_df[partition_col].astype(partition_col_type)
        check_round_trip(
            df_compat,
            pa,
            expected=expected_df,
            path=f's3://{s3_public_bucket.name}/parquet_dir',
            read_kwargs={'storage_options': s3so},
            write_kwargs={'partition_cols': partition_col, 'compression': None, 'storage_options': s3so},
            check_like=True,
            repeat=1
        )

    def test_read_file_like_obj_support(self, df_compat: pd.DataFrame, using_infer_string: bool) -> None:
        pytest.importorskip('pyarrow')
        buffer: BytesIO = BytesIO()
        df_compat.to_parquet(buffer)
        df_from_buf: pd.DataFrame = read_parquet(buffer)
        if using_infer_string and (not pa_version_under19p0):
            df_compat.columns = df_compat.columns.astype('str')
        tm.assert_frame_equal(df_compat, df_from_buf)

    def test_expand_user(self, df_compat: pd.DataFrame, monkeypatch: Any) -> None:
        pytest.importorskip('pyarrow')
        monkeypatch.setenv('HOME', 'TestingUser')
        monkeypatch.setenv('USERPROFILE', 'TestingUser')
        with pytest.raises(OSError, match='.*TestingUser.*'):
            read_parquet('~/file.parquet')
        with pytest.raises(OSError, match='.*TestingUser.*'):
            df_compat.to_parquet('~/file.parquet')

    def test_partition_cols_supported(self, tmp_path: pathlib.Path, pa: str, df_full: pd.DataFrame) -> None:
        partition_cols = ['bool', 'int']
        df = df_full
        df.to_parquet(tmp_path, partition_cols=partition_cols, compression=None)
        check_partition_names(str(tmp_path), partition_cols)
        assert read_parquet(tmp_path).shape == df.shape

    def test_partition_cols_string(self, tmp_path: pathlib.Path, pa: str, df_full: pd.DataFrame) -> None:
        partition_cols = 'bool'
        partition_cols_list = [partition_cols]
        df = df_full
        df.to_parquet(tmp_path, partition_cols=partition_cols, compression=None)
        check_partition_names(str(tmp_path), partition_cols_list)
        assert read_parquet(tmp_path).shape == df.shape

    @pytest.mark.parametrize('path_type', [str, lambda x: x], ids=['string', 'pathlib.Path'])
    def test_partition_cols_pathlib(self, tmp_path: pathlib.Path, pa: str, df_compat: pd.DataFrame, path_type: Any) -> None:
        partition_cols = 'B'
        partition_cols_list = [partition_cols]
        df = df_compat
        path = path_type(tmp_path)
        df.to_parquet(path, partition_cols=partition_cols_list)
        assert read_parquet(path).shape == df.shape

    def test_empty_dataframe(self, pa: str) -> None:
        df = pd.DataFrame(index=[], columns=[])
        check_round_trip(df, pa)

    def test_write_with_schema(self, pa: str) -> None:
        import pyarrow  # type: ignore
        df = pd.DataFrame({'x': [0, 1]})
        schema = pyarrow.schema([pyarrow.field('x', type=pyarrow.bool_())])
        out_df = df.astype(bool)
        check_round_trip(df, pa, write_kwargs={'schema': schema}, expected=out_df)

    def test_additional_extension_arrays(self, pa: str, using_infer_string: bool) -> None:
        pytest.importorskip('pyarrow')
        df = pd.DataFrame({
            'a': pd.Series([1, 2, 3], dtype='Int64'),
            'b': pd.Series([1, 2, 3], dtype='UInt32'),
            'c': pd.Series(['a', None, 'c'], dtype='string')
        })
        if using_infer_string and pa_version_under19p0:
            check_round_trip(df, pa, expected=df.astype({'c': 'str'}))
        else:
            check_round_trip(df, pa)
        df = pd.DataFrame({'a': pd.Series([1, 2, 3, None], dtype='Int64')})
        check_round_trip(df, pa)

    def test_pyarrow_backed_string_array(self, pa: str, string_storage: str, using_infer_string: bool) -> None:
        pytest.importorskip('pyarrow')
        df = pd.DataFrame({'a': pd.Series(['a', None, 'c'], dtype='string[pyarrow]')})
        with pd.option_context('string_storage', string_storage):
            if using_infer_string:
                if pa_version_under19p0:
                    expected = df.astype('str')
                else:
                    expected = df.astype(f'string[{string_storage}]')
                expected.columns = expected.columns.astype('str')
            else:
                expected = df.astype(f'string[{string_storage}]')
            check_round_trip(df, pa, expected=expected)

    def test_additional_extension_types(self, pa: str) -> None:
        pytest.importorskip('pyarrow')
        df = pd.DataFrame({
            'c': pd.IntervalIndex.from_tuples([(0, 1), (1, 2), (3, 4)]),
            'd': pd.period_range('2012-01-01', periods=3, freq='D'),
            'e': pd.IntervalIndex.from_breaks(pd.date_range('2012-01-01', periods=4, freq='D'))
        })
        check_round_trip(df, pa)

    def test_timestamp_nanoseconds(self, pa: str) -> None:
        ver: str = '2.6'
        df = pd.DataFrame({'a': pd.date_range('2017-01-01', freq='1ns', periods=10)})
        check_round_trip(df, pa, write_kwargs={'version': ver})

    def test_timezone_aware_index(self, pa: str, timezone_aware_date_list: datetime.datetime) -> None:
        pytest.importorskip('pyarrow', '11.0.0')
        idx = 5 * [timezone_aware_date_list]
        df = pd.DataFrame(index=idx, data={'index_as_col': idx})
        expected = df.copy()
        if pa_version_under11p0:
            expected.index = expected.index.as_unit('ns')
        if timezone_aware_date_list.tzinfo != datetime.timezone.utc:
            try:
                import pytz
            except ImportError:
                pass
            else:
                offset = df.index.tz.utcoffset(timezone_aware_date_list)
                tz = pytz.FixedOffset(int(offset.total_seconds() / 60))
                expected.index = expected.index.tz_convert(tz)
                expected['index_as_col'] = expected['index_as_col'].dt.tz_convert(tz)
        check_round_trip(df, pa, check_dtype=False, expected=expected)

    def test_filter_row_groups(self, pa: str) -> None:
        pytest.importorskip('pyarrow')
        df = pd.DataFrame({'a': list(range(3))})
        with tm.ensure_clean() as path:
            df.to_parquet(path, engine=pa)
            result: pd.DataFrame = read_parquet(path, pa, filters=[('a', '==', 0)])
        assert len(result) == 1

    @pytest.mark.filterwarnings('ignore:make_block is deprecated:DeprecationWarning')
    def test_read_dtype_backend_pyarrow_config(self, pa: str, df_full: pd.DataFrame) -> None:
        import pyarrow
        df = df_full
        dti = pd.date_range('20130101', periods=3, tz='Europe/Brussels')
        dti = dti._with_freq(None)
        df['datetime_tz'] = dti
        df['bool_with_none'] = [True, None, True]
        pa_table = pyarrow.Table.from_pandas(df)
        expected = pa_table.to_pandas(types_mapper=pd.ArrowDtype)
        if pa_version_under13p0:
            expected['datetime'] = expected['datetime'].astype('timestamp[us][pyarrow]')
            expected['datetime_tz'] = expected['datetime_tz'].astype(pd.ArrowDtype(pyarrow.timestamp(unit='us', tz='Europe/Brussels')))
        expected['datetime_with_nat'] = expected['datetime_with_nat'].astype('timestamp[ms][pyarrow]')
        check_round_trip(df, engine=pa, read_kwargs={'dtype_backend': 'pyarrow'}, expected=expected)

    def test_read_dtype_backend_pyarrow_config_index(self, pa: str) -> None:
        df = pd.DataFrame({'a': [1, 2]}, index=pd.Index([3, 4], name='test'), dtype='int64[pyarrow]')
        expected = df.copy()
        import pyarrow
        if Version(pyarrow.__version__) > Version('11.0.0'):
            expected.index = expected.index.astype('int64[pyarrow]')
        check_round_trip(df, engine=pa, read_kwargs={'dtype_backend': 'pyarrow'}, expected=expected)

    @pytest.mark.xfail(pa_version_under17p0, reason="pa.pandas_compat passes 'datetime64' to .astype")
    def test_columns_dtypes_not_invalid(self, pa: str) -> None:
        df = pd.DataFrame({'string': list('abc'), 'int': list(range(1, 4))})
        df.columns = [0, 1]
        check_round_trip(df, pa)
        df.columns = [b'foo', b'bar']
        with pytest.raises(NotImplementedError, match='|S3'):
            check_round_trip(df, pa)
        df.columns = [datetime.datetime(2011, 1, 1, 0, 0), datetime.datetime(2011, 1, 1, 1, 1)]
        check_round_trip(df, pa)

    def test_empty_columns(self, pa: str) -> None:
        df = pd.DataFrame(index=pd.Index(['a', 'b', 'c'], name='custom name'))
        check_round_trip(df, pa)

    def test_df_attrs_persistence(self, tmp_path: pathlib.Path, pa: str) -> None:
        path = tmp_path / 'test_df_metadata.p'
        df = pd.DataFrame(data={1: [1]})
        df.attrs = {'test_attribute': 1}
        df.to_parquet(path, engine=pa)
        new_df: pd.DataFrame = read_parquet(path, engine=pa)
        assert new_df.attrs == df.attrs

    def test_string_inference(self, tmp_path: pathlib.Path, pa: str, using_infer_string: bool) -> None:
        path = tmp_path / 'test_string_inference.p'
        df = pd.DataFrame(data={'a': ['x', 'y']}, index=['a', 'b'])
        df.to_parquet(path, engine=pa)
        with pd.option_context('future.infer_string', True):
            result: pd.DataFrame = read_parquet(path, engine=pa)
        dtype = pd.StringDtype(na_value=np.nan)
        expected = pd.DataFrame(
            data={'a': ['x', 'y']},
            dtype=dtype,
            index=pd.Index(['a', 'b'], dtype=dtype),
            columns=pd.Index(['a'], dtype=object if pa_version_under19p0 and (not using_infer_string) else dtype)
        )
        tm.assert_frame_equal(result, expected)

    @pytest.mark.skipif(pa_version_under11p0, reason='not supported before 11.0')
    def test_roundtrip_decimal(self, tmp_path: pathlib.Path, pa: str) -> None:
        import pyarrow as pa_mod
        path = tmp_path / 'decimal.p'
        df = pd.DataFrame({'a': [Decimal('123.00')]}, dtype='string[pyarrow]')
        df.to_parquet(path, schema=pa_mod.schema([('a', pa_mod.decimal128(5))]))
        result: pd.DataFrame = read_parquet(path)
        if pa_version_under19p0:
            expected = pd.DataFrame({'a': ['123']}, dtype='string[python]')
        else:
            expected = pd.DataFrame({'a': [Decimal('123.00')]}, dtype='object')
        tm.assert_frame_equal(result, expected)

    def test_infer_string_large_string_type(self, tmp_path: pathlib.Path, pa: str) -> None:
        import pyarrow as pa_mod
        import pyarrow.parquet as pq
        path = tmp_path / 'large_string.p'
        table = pa_mod.table({'a': pa_mod.array([None, 'b', 'c'], pa_mod.large_string())})
        pq.write_table(table, path)
        with pd.option_context('future.infer_string', True):
            result: pd.DataFrame = read_parquet(path)
        expected = pd.DataFrame(
            data={'a': [None, 'b', 'c']},
            dtype=pd.StringDtype(na_value=np.nan),
            columns=pd.Index(['a'], dtype=pd.StringDtype(na_value=np.nan))
        )
        tm.assert_frame_equal(result, expected)

    def test_non_nanosecond_timestamps(self, temp_file: str) -> None:
        pa_mod = pytest.importorskip('pyarrow', '11.0.0')
        pq = pytest.importorskip('pyarrow.parquet')
        arr = pa_mod.array([datetime.datetime(1600, 1, 1)], type=pa_mod.timestamp('us'))
        table = pa_mod.table([arr], names=['timestamp'])
        pq.write_table(table, temp_file)
        result: pd.DataFrame = read_parquet(temp_file)
        expected = pd.DataFrame(data={'timestamp': [datetime.datetime(1600, 1, 1)]}, dtype='datetime64[us]')
        tm.assert_frame_equal(result, expected)

    def test_maps_as_pydicts(self, pa: str) -> None:
        pyarrow_mod = pytest.importorskip('pyarrow', '13.0.0')
        schema = pyarrow_mod.schema([('foo', pyarrow_mod.map_(pyarrow_mod.string(), pyarrow_mod.int64()))])
        df = pd.DataFrame([{'foo': {'A': 1}}, {'foo': {'B': 2}}])
        check_round_trip(
            df,
            pa,
            write_kwargs={'schema': schema},
            read_kwargs={'to_pandas_kwargs': {'maps_as_pydicts': 'strict'}}
        )


class TestParquetFastParquet(Base):
    def test_basic(self, fp: str, df_full: pd.DataFrame, request: pytest.FixtureRequest) -> None:
        pytz = pytest.importorskip('pytz')
        import fastparquet
        if Version(fastparquet.__version__) < Version('2024.11.0'):
            request.applymarker(pytest.mark.xfail(reason='datetime_with_nat gets incorrect values'))
        tz = pytz.timezone('US/Eastern')
        df = df_full
        dti = pd.date_range('20130101', periods=3, tz=tz)
        dti = dti._with_freq(None)
        df['datetime_tz'] = dti
        df['timedelta'] = pd.timedelta_range('1 day', periods=3)
        check_round_trip(df, fp)

    def test_columns_dtypes_invalid(self, fp: str) -> None:
        df = pd.DataFrame({'string': list('abc'), 'int': list(range(1, 4))})
        err: Type[BaseException] = TypeError
        msg: str = 'Column name must be a string'
        df.columns = [0, 1]
        self.check_error_on_write(df, fp, err, msg)
        df.columns = [b'foo', b'bar']
        self.check_error_on_write(df, fp, err, msg)
        df.columns = [datetime.datetime(2011, 1, 1, 0, 0), datetime.datetime(2011, 1, 1, 1, 1)]
        self.check_error_on_write(df, fp, err, msg)

    def test_duplicate_columns(self, fp: str) -> None:
        df = pd.DataFrame(np.arange(12).reshape(4, 3), columns=list('aaa')).copy()
        msg: str = 'Cannot create parquet dataset with duplicate column names'
        self.check_error_on_write(df, fp, ValueError, msg)

    def test_bool_with_none(self, fp: str, request: pytest.FixtureRequest) -> None:
        import fastparquet
        if Version(fastparquet.__version__) < Version('2024.11.0') and Version(np.__version__) >= Version('2.0.0'):
            request.applymarker(pytest.mark.xfail(reason='fastparquet uses np.float_ in numpy2'))
        df = pd.DataFrame({'a': [True, None, False]})
        expected = pd.DataFrame({'a': [1.0, np.nan, 0.0]}, dtype='float16')
        check_round_trip(df, fp, expected=expected, check_dtype=False)

    def test_unsupported(self, fp: str) -> None:
        df = pd.DataFrame({'a': pd.period_range('2013', freq='M', periods=3)})
        self.check_error_on_write(df, fp, ValueError, None)
        df = pd.DataFrame({'a': ['a', 1, 2.0]})
        msg: str = "Can't infer object conversion type"
        self.check_error_on_write(df, fp, ValueError, msg)

    def test_categorical(self, fp: str) -> None:
        df = pd.DataFrame({'a': pd.Categorical(list('abc'))})
        check_round_trip(df, fp)

    def test_filter_row_groups(self, fp: str) -> None:
        d = {'a': list(range(3))}
        df = pd.DataFrame(d)
        with tm.ensure_clean() as path:
            df.to_parquet(path, engine=fp, compression=None, row_group_offsets=1)
            result: pd.DataFrame = read_parquet(path, fp, filters=[('a', '==', 0)])
        assert len(result) == 1

    @pytest.mark.single_cpu
    def test_s3_roundtrip(self, df_compat: pd.DataFrame, s3_public_bucket: Any, fp: str, s3so: Dict[str, Any]) -> None:
        check_round_trip(
            df_compat,
            fp,
            path=f's3://{s3_public_bucket.name}/fastparquet.parquet',
            read_kwargs={'storage_options': s3so},
            write_kwargs={'compression': None, 'storage_options': s3so}
        )

    def test_partition_cols_supported(self, tmp_path: pathlib.Path, fp: str, df_full: pd.DataFrame) -> None:
        partition_cols = ['bool', 'int']
        df = df_full
        df.to_parquet(tmp_path, engine='fastparquet', partition_cols=partition_cols, compression=None)
        assert os.path.exists(tmp_path)
        import fastparquet
        actual_partition_cols = fastparquet.ParquetFile(str(tmp_path), False).cats
        assert len(actual_partition_cols) == 2

    def test_partition_cols_string(self, tmp_path: pathlib.Path, fp: str, df_full: pd.DataFrame) -> None:
        partition_cols = 'bool'
        df = df_full
        df.to_parquet(tmp_path, engine='fastparquet', partition_cols=partition_cols, compression=None)
        assert os.path.exists(tmp_path)
        import fastparquet
        actual_partition_cols = fastparquet.ParquetFile(str(tmp_path), False).cats
        assert len(actual_partition_cols) == 1

    def test_partition_on_supported(self, tmp_path: pathlib.Path, fp: str, df_full: pd.DataFrame) -> None:
        partition_cols = ['bool', 'int']
        df = df_full
        df.to_parquet(tmp_path, engine='fastparquet', compression=None, partition_on=partition_cols)
        assert os.path.exists(tmp_path)
        import fastparquet
        actual_partition_cols = fastparquet.ParquetFile(str(tmp_path), False).cats
        assert len(actual_partition_cols) == 2

    def test_error_on_using_partition_cols_and_partition_on(self, tmp_path: pathlib.Path, fp: str, df_full: pd.DataFrame) -> None:
        partition_cols = ['bool', 'int']
        df = df_full
        msg: str = 'Cannot use both partition_on and partition_cols. Use partition_cols for partitioning data'
        with pytest.raises(ValueError, match=msg):
            df.to_parquet(tmp_path, engine='fastparquet', compression=None, partition_on=partition_cols, partition_cols=partition_cols)

    def test_empty_dataframe(self, fp: str) -> None:
        df = pd.DataFrame()
        expected = df.copy()
        check_round_trip(df, fp, expected=expected)

    def test_timezone_aware_index(self, fp: str, timezone_aware_date_list: datetime.datetime, request: pytest.FixtureRequest) -> None:
        import fastparquet
        if Version(fastparquet.__version__) < Version('2024.11.0'):
            request.applymarker(pytest.mark.xfail(reason='fastparquet bug, see https://github.com/dask/fastparquet/issues/929'))
        idx = 5 * [timezone_aware_date_list]
        df = pd.DataFrame(index=idx, data={'index_as_col': idx})
        expected = df.copy()
        expected.index.name = 'index'
        check_round_trip(df, fp, expected=expected)

    def test_close_file_handle_on_read_error(self) -> None:
        with tm.ensure_clean('test.parquet') as path:
            pathlib.Path(path).write_bytes(b'breakit')
            with tm.external_error_raised(Exception):
                read_parquet(path, engine='fastparquet')
            pathlib.Path(path).unlink(missing_ok=False)

    def test_bytes_file_name(self, engine: str) -> None:
        df = pd.DataFrame(data={'A': [0, 1], 'B': [1, 0]})
        with tm.ensure_clean('test.parquet') as path:
            with open(path.encode(), 'wb') as f:
                df.to_parquet(f)
            result: pd.DataFrame = read_parquet(path, engine=engine)
        tm.assert_frame_equal(result, df)

    def test_filesystem_notimplemented(self) -> None:
        pytest.importorskip('fastparquet')
        df = pd.DataFrame(data={'A': [0, 1], 'B': [1, 0]})
        with tm.ensure_clean() as path:
            with pytest.raises(NotImplementedError, match='filesystem is not implemented'):
                df.to_parquet(path, engine='fastparquet', filesystem='foo')
        with tm.ensure_clean() as path:
            pathlib.Path(path).write_bytes(b'foo')
            with pytest.raises(NotImplementedError, match='filesystem is not implemented'):
                read_parquet(path, engine='fastparquet', filesystem='foo')

    def test_invalid_filesystem(self) -> None:
        pytest.importorskip('pyarrow')
        df = pd.DataFrame(data={'A': [0, 1], 'B': [1, 0]})
        with tm.ensure_clean() as path:
            with pytest.raises(ValueError, match='filesystem must be a pyarrow or fsspec FileSystem'):
                df.to_parquet(path, engine='pyarrow', filesystem='foo')
        with tm.ensure_clean() as path:
            pathlib.Path(path).write_bytes(b'foo')
            with pytest.raises(ValueError, match='filesystem must be a pyarrow or fsspec FileSystem'):
                read_parquet(path, engine='pyarrow', filesystem='foo')

    def test_unsupported_pa_filesystem_storage_options(self) -> None:
        pa_fs = pytest.importorskip('pyarrow.fs')
        df = pd.DataFrame(data={'A': [0, 1], 'B': [1, 0]})
        with tm.ensure_clean() as path:
            with pytest.raises(NotImplementedError, match='storage_options not supported with a pyarrow FileSystem.'):
                df.to_parquet(path, engine='pyarrow', filesystem=pa_fs.LocalFileSystem(), storage_options={'foo': 'bar'})
        with tm.ensure_clean() as path:
            pathlib.Path(path).write_bytes(b'foo')
            with pytest.raises(NotImplementedError, match='storage_options not supported with a pyarrow FileSystem.'):
                read_parquet(path, engine='pyarrow', filesystem=pa_fs.LocalFileSystem(), storage_options={'foo': 'bar'})

    def test_invalid_dtype_backend(self, engine: str) -> None:
        msg: str = "dtype_backend numpy is invalid, only 'numpy_nullable' and 'pyarrow' are allowed."
        df = pd.DataFrame({'int': list(range(1, 4))})
        with tm.ensure_clean('tmp.parquet') as path:
            df.to_parquet(path)
            with pytest.raises(ValueError, match=msg):
                read_parquet(path, dtype_backend='numpy')