from __future__ import annotations
from datetime import datetime, time
from functools import partial
from io import BytesIO
import os
from pathlib import Path
import platform
import re
from urllib.error import URLError
import uuid
from zipfile import BadZipFile
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union

import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import DataFrame, Index, MultiIndex, Series, read_csv
import pandas._testing as tm

read_ext_params: List[str] = ['.xls', '.xlsx', '.xlsm', '.xlsb', '.ods']
engine_params: List[pytest.param] = [
    pytest.param('xlrd', marks=[td.skip_if_no('xlrd')]),
    pytest.param('openpyxl', marks=[td.skip_if_no('openpyxl')]),
    pytest.param(None, marks=[td.skip_if_no('xlrd')]),
    pytest.param('pyxlsb', marks=td.skip_if_no('pyxlsb')),
    pytest.param('odf', marks=td.skip_if_no('odf')),
    pytest.param('calamine', marks=td.skip_if_no('python_calamine'))
]


def func_8c5a9y9x(engine: pytest.param, read_ext: str) -> bool:
    """
    Filter out invalid (engine, ext) pairs instead of skipping, as that
    produces 500+ pytest.skips.
    """
    engine = engine.values[0]
    if engine == 'openpyxl' and read_ext == '.xls':
        return False
    if engine == 'odf' and read_ext != '.ods':
        return False
    if read_ext == '.ods' and engine not in {'odf', 'calamine'}:
        return False
    if engine == 'pyxlsb' and read_ext != '.xlsb':
        return False
    if read_ext == '.xlsb' and engine not in {'pyxlsb', 'calamine'}:
        return False
    if engine == 'xlrd' and read_ext != '.xls':
        return False
    return True


def func_jds5urnz(engine: pytest.param, read_ext: str) -> pytest.param:
    """
    engine gives us a pytest.param object with some marks, read_ext is just
    a string.  We need to generate a new pytest.param inheriting the marks.
    """
    values = engine.values + (read_ext,)
    new_param = pytest.param(values, marks=engine.marks)
    return new_param


@pytest.fixture(params=[func_jds5urnz(eng, ext) for eng in engine_params for
    ext in read_ext_params if func_8c5a9y9x(eng, ext)], ids=str)
def func_kghfv52i(request: pytest.FixtureRequest) -> Tuple[Optional[str], str]:
    """
    Fixture for Excel reader engine and read_ext, only including valid pairs.
    """
    return request.param


@pytest.fixture
def func_d7yn5ojf(engine_and_read_ext: Tuple[Optional[str], str]) -> Optional[str]:
    engine, _ = engine_and_read_ext
    return engine


@pytest.fixture
def func_xzsbpzmy(engine_and_read_ext: Tuple[Optional[str], str]) -> str:
    _, read_ext = engine_and_read_ext
    return read_ext


@pytest.fixture
def func_hn75yqc5(read_ext: str, tmp_path: Path) -> str:
    tmp = tmp_path / f'{uuid.uuid4()}{read_ext}'
    tmp.touch()
    return str(tmp)


@pytest.fixture
def func_u8i3bhxt(datapath: Callable[[str, str, str, str], str]) -> DataFrame:
    """
    Obtain the reference data from read_csv with the Python engine.
    """
    filepath = datapath('io', 'data', 'csv', 'test1.csv')
    df_ref = read_csv(filepath, index_col=0, parse_dates=True, engine='python')
    return df_ref


def func_h5xd31uv(read_ext: str, engine: Optional[str]) -> str:
    unit = 'us'
    if (read_ext == '.ods') ^ (engine == 'calamine'):
        unit = 's'
    return unit


def func_r8n3x9o9(expected: DataFrame, read_ext: str, engine: Optional[str]) -> None:
    expected.index.name = None
    unit = func_h5xd31uv(read_ext, engine)
    expected.index = expected.index.as_unit(unit)


def func_2tw9ycnx(engine: Optional[str], request: pytest.FixtureRequest) -> None:
    if engine == 'pyxlsb':
        request.applymarker(pytest.mark.xfail(reason=
            'Sheets containing datetimes not supported by pyxlsb'))


class TestReaders:

    @pytest.mark.parametrize('col', [[True, None, False], [True], [True, False]])
    def func_9jcm22ji(self, col: List[Optional[bool]], tmp_excel: str, read_ext: str) -> None:
        if read_ext in ('.xlsb', '.xls'):
            pytest.skip(f"No engine for filetype: '{read_ext}'")
        df = DataFrame({'bool_column': col}, dtype='boolean')
        df.to_excel(tmp_excel, index=False)
        df2 = pd.read_excel(tmp_excel, dtype={'bool_column': 'boolean'})
        tm.assert_frame_equal(df, df2)

    def func_rbbtqw94(self, datapath: Callable[[str, str, str, str], str]) -> None:
        f_path = datapath('io', 'data', 'excel', 'test_none_type.xlsx')
        with pd.ExcelFile(f_path) as excel:
            parsed = pd.read_excel(excel, sheet_name='Sheet1',
                keep_default_na=True, na_values=['nan', 'None', 'abcd'],
                dtype='boolean', engine='openpyxl')
        expected = DataFrame({'Test': [True, None, False, None, False, None,
            True]}, dtype='boolean')
        tm.assert_frame_equal(parsed, expected)

    @pytest.fixture(autouse=True)
    def func_pqpvebzn(self, engine: Optional[str], datapath: Callable[[str, str, str, str], str], monkeypatch: pytest.MonkeyPatch) -> None:
        """
        Change directory and set engine for read_excel calls.
        """
        func = partial(pd.read_excel, engine=engine)
        monkeypatch.chdir(datapath('io', 'data', 'excel'))
        monkeypatch.setattr(pd, 'read_excel', func)

    def func_xvopw874(self, read_ext: str, engine: Optional[str], monkeypatch: pytest.MonkeyPatch) -> None:

        def parser(self: Any, *args: Any, **kwargs: Any) -> Any:
            return self.engine

        monkeypatch.setattr(pd.ExcelFile, 'parse', parser)
        expected_defaults: Dict[str, Optional[str]] = {
            'xlsx': 'openpyxl',
            'xlsm': 'openpyxl',
            'xlsb': 'pyxlsb',
            'xls': 'xlrd',
            'ods': 'odf'
        }
        with open('test1' + read_ext, 'rb') as f:
            result = pd.read_excel(f)
        if engine is not None:
            expected = engine
        else:
            expected = expected_defaults[read_ext[1:]]
        assert result == expected

    def func_h5dyecrz(self, read_ext: str, engine: Optional[str]) -> None:
        expected_defaults: Dict[str, Dict[str, Any]] = {
            'xlsx': {'foo': 'abcd'},
            'xlsm': {'foo': 123},
            'xlsb': {'foo': 'True'},
            'xls': {'foo': True},
            'ods': {'foo': 'abcd'}
        }
        if engine in {'xlrd', 'pyxlsb'}:
            msg = re.escape(
                "open_workbook() got an unexpected keyword argument 'foo'")
        elif engine == 'odf':
            msg = re.escape("load() got an unexpected keyword argument 'foo'")
        else:
            msg = re.escape(
                "load_workbook() got an unexpected keyword argument 'foo'")
        if engine is not None:
            with pytest.raises(TypeError, match=msg):
                pd.read_excel('test1' + read_ext, sheet_name='Sheet1',
                    index_col=0, engine_kwargs=expected_defaults[read_ext[1:]])

    def func_oj5as2b3(self, read_ext: str) -> None:
        msg = 'Passing an integer for `usecols`'
        with pytest.raises(ValueError, match=msg):
            pd.read_excel('test1' + read_ext, sheet_name='Sheet1',
                index_col=0, usecols=3)
        with pytest.raises(ValueError, match=msg):
            pd.read_excel('test1' + read_ext, sheet_name='Sheet2', skiprows
                =[1], index_col=0, usecols=3)

    def func_z3koxpzd(self, request: pytest.FixtureRequest, engine: Optional[str], read_ext: str, df_ref: DataFrame) -> None:
        func_2tw9ycnx(engine, request)
        expected = df_ref[['B', 'C']]
        func_r8n3x9o9(expected, read_ext, engine)
        df1 = pd.read_excel('test1' + read_ext, sheet_name='Sheet1',
            index_col=0, usecols=[0, 2, 3])
        df2 = pd.read_excel('test1' + read_ext, sheet_name='Sheet2',
            skiprows=[1], index_col=0, usecols=[0, 2, 3])
        tm.assert_frame_equal(df1, expected)
        tm.assert_frame_equal(df2, expected)

    def func_yxcwh5jq(self, request: pytest.FixtureRequest, engine: Optional[str], read_ext: str, df_ref: DataFrame) -> None:
        func_2tw9ycnx(engine, request)
        expected = df_ref[['A', 'B', 'C']]
        func_r8n3x9o9(expected, read_ext, engine)
        df2 = pd.read_excel('test1' + read_ext, sheet_name='Sheet1',
            index_col=0, usecols='A:D')
        df3 = pd.read_excel('test1' + read_ext, sheet_name='Sheet2',
            skiprows=[1], index_col=0, usecols='A:D')
        tm.assert_frame_equal(df2, expected)
        tm.assert_frame_equal(df3, expected)
        expected = df_ref[['B', 'C']]
        func_r8n3x9o9(expected, read_ext, engine)
        df2 = pd.read_excel('test1' + read_ext, sheet_name='Sheet1',
            index_col=0, usecols='A,C,D')
        df3 = pd.read_excel('test1' + read_ext, sheet_name='Sheet2',
            skiprows=[1], index_col=0, usecols='A,C,D')
        tm.assert_frame_equal(df2, expected)
        tm.assert_frame_equal(df3, expected)
        df2 = pd.read_excel('test1' + read_ext, sheet_name='Sheet1',
            index_col=0, usecols='A,C:D')
        df3 = pd.read_excel('test1' + read_ext, sheet_name='Sheet2',
            skiprows=[1], index_col=0, usecols='A,C:D')
        tm.assert_frame_equal(df2, expected)
        tm.assert_frame_equal(df3, expected)

    @pytest.mark.parametrize('usecols', [[0, 1, 3], [0, 3, 1], [1, 0, 3], [
        1, 3, 0], [3, 0, 1], [3, 1, 0]])
    def func_gamky2lo(self, request: pytest.FixtureRequest, engine: Optional[str], read_ext: str, usecols: List[int], df_ref: DataFrame) -> None:
        func_2tw9ycnx(engine, request)
        expected = df_ref[['A', 'C']]
        func_r8n3x9o9(expected, read_ext, engine)
        result = pd.read_excel('test1' + read_ext, sheet_name='Sheet1',
            index_col=0, usecols=usecols)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('usecols', [['B', 'D'], ['D', 'B']])
    def func_0t8qtwg1(self, read_ext: str, usecols: List[str], df_ref: DataFrame) -> None:
        expected = df_ref[['B', 'D']]
        expected.index = range(len(expected))
        result = pd.read_excel('test1' + read_ext, sheet_name='Sheet1',
            usecols=usecols)
        tm.assert_frame_equal(result, expected)

    def func_prw3pvl9(self, request: pytest.FixtureRequest, engine: Optional[str], read_ext: str, df_ref: DataFrame) -> None:
        func_2tw9ycnx(engine, request)
        expected = df_ref
        func_r8n3x9o9(expected, read_ext, engine)
        result = pd.read_excel('test1' + read_ext, sheet_name='Sheet1',
            index_col=0)
        tm.assert_frame_equal(result, expected)

    def func_ot2ryb70(self, request: pytest.FixtureRequest, engine: Optional[str], read_ext: str, df_ref: DataFrame) -> None:
        func_2tw9ycnx(engine, request)
        expected = df_ref[['C', 'D']]
        func_r8n3x9o9(expected, read_ext, engine)
        result = pd.read_excel('test1' + read_ext, sheet_name='Sheet1',
            index_col=0, usecols='A,D:E')
        tm.assert_frame_equal(result, expected)

    def func_odz2bv7c(self, read_ext: str) -> None:
        msg = 'Invalid column name: E1'
        with pytest.raises(ValueError, match=msg):
            pd.read_excel('test1' + read_ext, sheet_name='Sheet1', usecols=
                'D:E1')

    def func_e7xv9x7m(self, read_ext: str) -> None:
        msg = 'list indices must be integers.*, not str'
        with pytest.raises(TypeError, match=msg):
            pd.read_excel('test1' + read_ext, sheet_name='Sheet1',
                index_col=['A'], usecols=['A', 'C'])

    def func_8u4hi162(self, read_ext: str) -> None:
        result = pd.read_excel('test1' + read_ext, sheet_name='Sheet3',
            index_col='A')
        expected = DataFrame(columns=['B', 'C', 'D', 'E', 'F'], index=Index
            ([], name='A'))
        tm.assert_frame_equal(result, expected)

    def func_btud0p8t(self, read_ext: str) -> None:
        result = pd.read_excel('test1' + read_ext, sheet_name='Sheet3',
            index_col=['A', 'B', 'C'])
        expected = DataFrame(columns=['D', 'E', 'F'], index=MultiIndex(
            levels=[[]] * 3, codes=[[]] * 3, names=['A', 'B', 'C']))
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('index_col', [None, 2])
    def func_fc989g5u(self, read_ext: str, index_col: Optional[int]) -> None:
        result = pd.read_excel('test1' + read_ext, sheet_name='Sheet4',
            index_col=index_col)
        expected = DataFrame([['i1', 'a', 'x'], ['i2', 'b', 'y']], columns=
            ['Unnamed: 0', 'col1', 'col2'])
        if index_col is not None:
            expected = expected.set_index(expected.columns[index_col])
        tm.assert_frame_equal(result, expected)

    def func_1skx9sju(self, read_ext: str) -> None:
        msg = (
            "Usecols do not match columns, columns expected but not found: \\['E'\\]"
            )
        with pytest.raises(ValueError, match=msg):
            pd.read_excel('test1' + read_ext, usecols=['E'])

    def func_tlzzjw1a(self, read_ext: str) -> None:
        msg = (
            "'usecols' must either be list-like of all strings, all unicode, all integers or a callable."
            )
        with pytest.raises(ValueError, match=msg):
            pd.read_excel('test1' + read_ext, usecols=['E1', 0])

    def func_lhnpaax8(self, read_ext: str) -> None:
        parsed = pd.read_excel('test2' + read_ext, sheet_name='Sheet1')
        expected = DataFrame([['aaaa', 'bbbbb']], columns=['Test', 'Test1'])
        tm.assert_frame_equal(parsed, expected)

    def func_xd4a5cnq(self, request: pytest.FixtureRequest, engine: Optional[str], read_ext: str) -> None:
        func_2tw9ycnx(engine, request)
        if engine == 'calamine' and read_ext == '.ods':
            request.applymarker(pytest.mark.xfail(reason=
                "Calamine can't extract error from ods files"))
        parsed = pd.read_excel('test3' + read_ext, sheet_name='Sheet1')
        expected = DataFrame([[np.nan]], columns=['Test'])
        tm.assert_frame_equal(parsed, expected)

    def func_dvjwgrg7(self, request: pytest.FixtureRequest, engine: Optional[str], read_ext: str, df_ref: DataFrame) -> None:
        func_2tw9ycnx(engine, request)
        expected = df_ref
        func_r8n3x9o9(expected, read_ext, engine)
        df1 = pd.read_excel('test1' + read_ext, sheet_name='Sheet1',
            index_col=0)
        df2 = pd.read_excel('test1' + read_ext, sheet_name='Sheet2',
            skiprows=[1], index_col=0)
        tm.assert_frame_equal(df1, expected)
        tm.assert_frame_equal(df2, expected)
        df3 = pd.read_excel('test1' + read_ext, sheet_name='Sheet1',
            index_col=0, skipfooter=1)
        tm.assert_frame_equal(df3, df1.iloc[:-1])

    def func_2pkvsvn9(self, request: pytest.FixtureRequest, engine: Optional[str], read_ext: str) -> None:
        func_2tw9ycnx(engine, request)
        unit = func_h5xd31uv(read_ext, engine)
        expected = DataFrame.from_dict({'IntCol': [1, 2, -3, 4, 0],
            'FloatCol': [1.25, 2.25, 1.83, 1.92, 5e-10], 'BoolCol': [True, 
            False, True, True, False], 'StrCol': [1, 2, 3, 4, 5], 'Str2Col':
            ['a', 3, 'c', 'd', 'e'], 'DateCol': Index([datetime(2013, 10, 
            30), datetime(2013, 10, 31), datetime(1905, 1, 1), datetime(
            2013, 12, 14), datetime(2015, 3, 14)], dtype=f'M8[{unit}]')})
        basename = 'test_types'
        actual = pd.read_excel(basename + read_ext, sheet_name='Sheet1')
        tm.assert_frame_equal(actual, expected)
        float_expected = expected.copy()
        float_expected.loc[float_expected.index[1], 'Str2Col'] = 3.0
        actual = pd.read_excel(basename + read_ext, sheet_name='Sheet1')
        tm.assert_frame_equal(actual, float_expected)
        for icol, name in enumerate(expected.columns):
            actual = pd.read_excel(basename + read_ext, sheet_name='Sheet1',
                index_col=icol)
            exp = expected.set_index(name)
            tm.assert_frame_equal(actual, exp)
        expected['StrCol'] = expected['StrCol'].apply(str)
        actual = pd.read_excel(basename + read_ext, sheet_name='Sheet1',
            converters={'StrCol': str})
        tm.assert_frame_equal(actual, expected)

    def func_87zqxtpf(self, read_ext: str) -> None:
        basename = 'test_converters'
        expected = DataFrame.from_dict({'IntCol': [1, 2, -3, -1000, 0],
            'FloatCol': [12.5, np.nan, 18.3, 19.2, 5e-09], 'BoolCol': [
            'Found', 'Found', 'Found', 'Not found', 'Found'], 'StrCol': [
            '1', np.nan, '3', '4', '5']})
        converters: Dict[Union[str, int], Callable[[Any], Any]] = {
            'IntCol': lambda x: int(x) if x != '' else -1000,
            'FloatCol': lambda x: 10 * x if x else np.nan,
            2: lambda x: 'Found' if x != '' else 'Not found',
            3: lambda x: str(x) if x else ''
        }
        actual = pd.read_excel(basename + read_ext, sheet_name='Sheet1',
            converters=converters)
        tm.assert_frame_equal(actual, expected)

    def func_5a7nwomg(self, read_ext: str) -> None:
        basename = 'testdtype'
        actual = pd.read_excel(basename + read_ext)
        expected = DataFrame({'a': [1, 2, 3, 4], 'b': [2.5, 3.5, 4.5, 5.5],
            'c': [1, 2, 3, 4], 'd': [1.0, 2.0, np.nan, 4.0]})
        tm.assert_frame_equal(actual, expected)
        actual = pd.read_excel(basename + read_ext, dtype={'a': 'float64',
            'b': 'float32', 'c': str})
        expected['a'] = expected['a'].astype('float64')
        expected['b'] = expected['b'].astype('float32')
        expected['c'] = Series(['001', '002', '003', '004'], dtype='str')
        tm.assert_frame_equal(actual, expected)
        msg = 'Unable to convert column d to type int64'
        with pytest.raises(ValueError, match=msg):
            pd.read_excel(basename + read_ext, dtype={'d': 'int64'})

    @pytest.mark.parametrize('dtype,expected', [
        (
            None,
            {
                'a': [1, 2, 3, 4],
                'b': [2.5, 3.5, 4.5, 5.5],
                'c': [1, 2, 3, 4],
                'd': [1.0, 2.0, np.nan, 4.0]
            }
        ),
        (
            {'a': 'float64', 'b': 'float32', 'c': str, 'd': str},
            {
                'a': Series([1, 2, 3, 4], dtype='float64'),
                'b': Series([2.5, 3.5, 4.5, 5.5], dtype='float32'),
                'c': Series(['001', '002', '003', '004'], dtype='str'),
                'd': Series(['1', '2', np.nan, '4'], dtype='str')
            }
        )
    ])
    def func_gfrjq93o(self, read_ext: str, dtype: Optional[Dict[str, Any]], expected: Dict[str, Any]) -> None:
        basename = 'testdtype'
        actual = pd.read_excel(basename + read_ext, dtype=dtype)
        expected_df = DataFrame(expected)
        tm.assert_frame_equal(actual, expected_df)

    def func_6qav95dx(self, read_ext: str, dtype_backend: str, engine: Optional[str], tmp_excel: str) -> None:
        if read_ext in ('.xlsb', '.xls'):
            pytest.skip(f"No engine for filetype: '{read_ext}'")
        df = DataFrame({
            'a': Series([1, 3], dtype='Int64'),
            'b': Series([2.5, 4.5], dtype='Float64'),
            'c': Series([True, False], dtype='boolean'),
            'd': Series(['a', 'b'], dtype='string'),
            'e': Series([pd.NA, 6], dtype='Int64'),
            'f': Series([pd.NA, 7.5], dtype='Float64'),
            'g': Series([pd.NA, True], dtype='boolean'),
            'h': Series([pd.NA, 'a'], dtype='string'),
            'i': Series([pd.Timestamp('2019-12-31')] * 2),
            'j': Series([pd.NA, pd.NA], dtype='Int64')
        })
        df.to_excel(tmp_excel, sheet_name='test', index=False)
        result = pd.read_excel(tmp_excel, sheet_name='test', dtype_backend=dtype_backend)
        if dtype_backend == 'pyarrow':
            import pyarrow as pa
            from pandas.arrays import ArrowExtensionArray
            expected = DataFrame({
                col: ArrowExtensionArray(pa.array(df[col], from_pandas=True)) for col in df.columns
            })
            expected['i'] = ArrowExtensionArray(expected['i'].array._pa_array.cast(pa.timestamp(unit='us')))
            expected['j'] = ArrowExtensionArray(pa.array([None, None]))
        else:
            expected = df
            unit = func_h5xd31uv(read_ext, engine)
            expected['i'] = expected['i'].astype(f'M8[{unit}]')
        tm.assert_frame_equal(result, expected)

    def func_hr5mddk1(self, read_ext: str, tmp_excel: str) -> None:
        if read_ext in ('.xlsb', '.xls'):
            pytest.skip(f"No engine for filetype: '{read_ext}'")
        df = DataFrame({'a': [np.nan, 1.0], 'b': [2.5, np.nan]})
        df.to_excel(tmp_excel, sheet_name='test', index=False)
        result = pd.read_excel(tmp_excel, sheet_name='test', dtype_backend='numpy_nullable', dtype='float64')
        tm.assert_frame_equal(result, df)

    def func_2pxrdduy(self, read_ext: str, string_storage: str, tmp_excel: str) -> None:
        if read_ext in ('.xlsb', '.xls'):
            pytest.skip(f"No engine for filetype: '{read_ext}'")
        df = DataFrame({
            'a': np.array(['a', 'b'], dtype=object),
            'b': np.array(['x', pd.NA], dtype=object)
        })
        df.to_excel(tmp_excel, sheet_name='test', index=False)
        with pd.option_context('mode.string_storage', string_storage):
            result = pd.read_excel(tmp_excel, sheet_name='test',
                dtype_backend='numpy_nullable')
        expected = DataFrame({
            'a': Series(['a', 'b'], dtype=pd.StringDtype(string_storage)),
            'b': Series(['x', None], dtype=pd.StringDtype(string_storage))
        })
        tm.assert_frame_equal(result, expected, check_column_type=False)

    @pytest.mark.parametrize('dtypes, exp_value', [
        ({}, 1),
        ({'a.1': 'int64'}, 1)
    ])
    def func_vvj9ao3r(self, read_ext: str, dtypes: Dict[str, Any], exp_value: int) -> None:
        basename = 'df_mangle_dup_col_dtypes'
        dtype_dict: Dict[str, Any] = {'a': object, **dtypes}
        dtype_dict_copy = dtype_dict.copy()
        result = pd.read_excel(basename + read_ext, dtype=dtype_dict)
        expected = DataFrame({
            'a': Series([1], dtype=object), 
            'a.1': Series([exp_value], dtype=object if not dtypes else None)
        })
        assert dtype_dict == dtype_dict_copy, 'dtype dict changed'
        tm.assert_frame_equal(result, expected)

    def func_2h3l68t3(self, read_ext: str) -> None:
        msg = 'Passing a bool to header is invalid'
        for arg in [True, False]:
            with pytest.raises(TypeError, match=msg):
                pd.read_excel('test1' + read_ext, header=arg)

    def func_v1wa7qzy(self, request: pytest.FixtureRequest, engine: Optional[str], read_ext: str) -> None:
        func_2tw9ycnx(engine, request)
        unit = func_h5xd31uv(read_ext, engine)
        expected = DataFrame([
            [1, 2.5, pd.Timestamp('2015-01-01'), True],
            [2, 3.5, pd.Timestamp('2015-01-02'), False],
            [3, 4.5, pd.Timestamp('2015-01-03'), False],
            [4, 5.5, pd.Timestamp('2015-01-04'), True]
        ], columns=['a', 'b', 'c', 'd'])
        expected['c'] = expected['c'].astype(f'M8[{unit}]')
        tm.assert_frame_equal(actual := pd.read_excel('testskiprows' + read_ext, sheet_name='skiprows_list', skiprows=[0, 2]), expected)
        tm.assert_frame_equal(actual := pd.read_excel('testskiprows' + read_ext, sheet_name='skiprows_list', skiprows=np.array([0, 2])), expected)
        tm.assert_frame_equal(actual := pd.read_excel('testskiprows' + read_ext, sheet_name='skiprows_list', skiprows=lambda x: x in [0, 2]), expected)
        expected_new = DataFrame([
            [2, 3.5, pd.Timestamp('2015-01-02'), False],
            [3, 4.5, pd.Timestamp('2015-01-03'), False],
            [4, 5.5, pd.Timestamp('2015-01-04'), True]
        ], columns=['a', 'b', 'c', 'd'])
        expected_new['c'] = expected_new['c'].astype(f'M8[{unit}]')
        tm.assert_frame_equal(actual := pd.read_excel('testskiprows' + read_ext, sheet_name='skiprows_list', skiprows=3, names=['a', 'b', 'c', 'd']), expected_new)

    def func_h06071fy(self, request: pytest.FixtureRequest, engine: Optional[str], read_ext: str) -> None:
        func_2tw9ycnx(engine, request)
        unit = func_h5xd31uv(read_ext, engine)
        expected = DataFrame([
            [1, 2.5, pd.Timestamp('2015-01-01'), True],
            [3, 4.5, pd.Timestamp('2015-01-03'), False]
        ], columns=['a', 'b', 'c', 'd'])
        expected['c'] = expected['c'].astype(f'M8[{unit}]')
        tm.assert_frame_equal(actual := pd.read_excel('testskiprows' + read_ext, sheet_name='skiprows_list', skiprows=lambda x: x not in [1, 3, 5]), expected)

    def func_6jynh7cx(self, read_ext: str) -> None:
        num_rows_to_pull = 5
        actual = pd.read_excel('test1' + read_ext, nrows=num_rows_to_pull)
        expected = pd.read_excel('test1' + read_ext)
        expected = expected[:num_rows_to_pull]
        tm.assert_frame_equal(actual, expected)

    def func_0on7x3fy(self, read_ext: str) -> None:
        expected = pd.read_excel('test1' + read_ext)
        num_records_in_file = len(expected)
        num_rows_to_pull = num_records_in_file + 10
        actual = pd.read_excel('test1' + read_ext, nrows=num_rows_to_pull)
        tm.assert_frame_equal(actual, expected)

    def func_m072iuyh(self, read_ext: str) -> None:
        msg = "'nrows' must be an integer >=0"
        with pytest.raises(ValueError, match=msg):
            pd.read_excel('test1' + read_ext, nrows='5')

    @pytest.mark.parametrize('filename,sheet_name,header,index_col,skiprows', [
        ('testmultiindex', 'mi_column', [0, 1], 0, None),
        ('testmultiindex', 'mi_index', None, [0, 1], None),
        ('testmultiindex', 'both', [0, 1], [0, 1], None),
        ('testmultiindex', 'mi_column_name', [0, 1], 0, None),
        ('testskiprows', 'skiprows_list', None, None, [0, 2]),
        ('testskiprows', 'skiprows_list', None, None, lambda x: x in (0, 2))
    ])
    def func_jfsn0t2f(self, read_ext: str, filename: str, sheet_name: Union[str, List[Union[str, int]]], header: Optional[Union[int, List[int]]], index_col: Optional[Union[int, List[int]]], skiprows: Union[None, List[int], Callable[[int], bool]]) -> None:
        """
        For various parameters, we should get the same result whether we
        limit the rows during load (nrows=3) or after (df.iloc[:3]).
        """
        expected = pd.read_excel(filename + read_ext, sheet_name=sheet_name,
            header=header, index_col=index_col, skiprows=skiprows).iloc[:3]
        actual = pd.read_excel(filename + read_ext, sheet_name=sheet_name,
            header=header, index_col=index_col, skiprows=skiprows, nrows=3)
        tm.assert_frame_equal(actual, expected)

    def func_m40ma66s(self, request: pytest.FixtureRequest, engine: Optional[str], read_ext: str) -> None:
        if read_ext == '.ods':
            pytest.skip('chartsheets do not exist in the ODF format')
        if engine == 'pyxlsb':
            request.applymarker(pytest.mark.xfail(reason=
                "pyxlsb can't distinguish chartsheets from worksheets"))
        with pytest.raises(ValueError, match=
            "Worksheet named 'Chart1' not found"):
            pd.read_excel('chartsheet' + read_ext, sheet_name='Chart1')

    def func_ed9oh69s(self, request: pytest.FixtureRequest, engine: Optional[str], read_ext: str) -> None:
        if read_ext == '.ods':
            pytest.skip('chartsheets do not exist in the ODF format')
        if engine == 'pyxlsb':
            request.applymarker(pytest.mark.xfail(reason=
                "pyxlsb can't distinguish chartsheets from worksheets"))
        with pytest.raises(ValueError, match=
            'Worksheet index 1 is invalid, 1 worksheets found'):
            pd.read_excel('chartsheet' + read_ext, sheet_name=1)

    def func_8i3tgebh(self, read_ext: str) -> None:
        result = pd.read_excel('test_decimal' + read_ext, decimal=',',
            skiprows=1)
        expected = DataFrame([
            [1, 1521.1541, 187101.9543, 'ABC', 'poi', 4.738797819],
            [2, 121.12, 14897.76, 'DEF', 'uyt', 0.377320872],
            [3, 878.158, 108013.434, 'GHI', 'rez', 2.735694704]
        ], columns=['Id', 'Number1', 'Number2', 'Text1', 'Text2', 'Number3'])
        tm.assert_frame_equal(result, expected)


class TestExcelFileRead:

    def func_v03b6exe(self, engine: Optional[str], read_ext: str) -> None:
        msg = 'Expected file path name or file-like object'
        with pytest.raises(TypeError, match=msg):
            with open('test1' + read_ext, 'rb') as f:
                pd.read_excel(f.read(), engine=engine)

    @pytest.fixture(autouse=True)
    def func_pqpvebzn(self, engine: Optional[str], datapath: Callable[[str, str, str, str], str], monkeypatch: pytest.MonkeyPatch) -> None:
        """
        Change directory and set engine for ExcelFile objects.
        """
        func = partial(pd.ExcelFile, engine=engine)
        monkeypatch.chdir(datapath('io', 'data', 'excel'))
        monkeypatch.setattr(pd, 'ExcelFile', func)

    def func_xvopw874(self, read_ext: str, engine: Optional[str]) -> None:
        expected_defaults: Dict[str, Optional[str]] = {
            'xlsx': 'openpyxl',
            'xlsm': 'openpyxl',
            'xlsb': 'pyxlsb',
            'xls': 'xlrd',
            'ods': 'odf'
        }
        with pd.ExcelFile('test1' + read_ext) as excel:
            result = excel.engine
        if engine is not None:
            expected = engine
        else:
            expected = expected_defaults[read_ext[1:]]
        assert result == expected

    def func_mluoiufw(self, read_ext: str) -> None:
        with pd.ExcelFile('test4' + read_ext) as excel:
            parsed = pd.read_excel(excel, sheet_name='Sheet1',
                keep_default_na=False, na_values=['apple'])
        expected = DataFrame([['NA'], [1], ['NA'], [np.nan], ['rabbit']],
            columns=['Test'])
        tm.assert_frame_equal(parsed, expected)
        with pd.ExcelFile('test4' + read_ext) as excel:
            parsed = pd.read_excel(excel, sheet_name='Sheet1',
                keep_default_na=True, na_values=['apple'])
        expected = DataFrame([[np.nan], [1], [np.nan], [np.nan], ['rabbit']
            ], columns=['Test'])
        tm.assert_frame_equal(parsed, expected)
        with pd.ExcelFile('test5' + read_ext) as excel:
            parsed = pd.read_excel(excel, sheet_name='Sheet1',
                keep_default_na=False, na_values=['apple'])
        expected = DataFrame([['1.#QNAN'], [1], ['nan'], [np.nan], [
            'rabbit']], columns=['Test'])
        tm.assert_frame_equal(parsed, expected)
        with pd.ExcelFile('test5' + read_ext) as excel:
            parsed = pd.read_excel(excel, sheet_name='Sheet1',
                keep_default_na=True, na_values=['apple'])
        expected = DataFrame([[np.nan], [1], [np.nan], [np.nan], ['rabbit']
            ], columns=['Test'])
        tm.assert_frame_equal(parsed, expected)

    @pytest.mark.parametrize('na_filter', [None, True, False])
    def func_qm14tkmz(self, read_ext: str, na_filter: Optional[bool]) -> None:
        kwargs: Dict[str, Any] = {}
        if na_filter is not None:
            kwargs['na_filter'] = na_filter
        with pd.ExcelFile('test5' + read_ext) as excel:
            parsed = pd.read_excel(excel, sheet_name='Sheet1',
                keep_default_na=True, na_values=['apple'], **kwargs)
        if na_filter is False:
            expected = [['1.#QNAN'], [1], ['nan'], ['apple'], ['rabbit']]
        else:
            expected = [[np.nan], [1], [np.nan], [np.nan], ['rabbit']]
        expected_df = DataFrame(expected, columns=['Test'])
        tm.assert_frame_equal(parsed, expected_df)

    def func_k07ibrmj(self, request: pytest.FixtureRequest, engine: Optional[str], read_ext: str, df_ref: DataFrame) -> None:
        func_2tw9ycnx(engine, request)
        expected = df_ref
        func_r8n3x9o9(expected, read_ext, engine)
        with pd.ExcelFile('test1' + read_ext) as excel:
            df1 = pd.read_excel(excel, sheet_name=0, index_col=0)
            df2 = pd.read_excel(excel, sheet_name=1, skiprows=[1], index_col=0)
        tm.assert_frame_equal(df1, expected)
        tm.assert_frame_equal(df2, expected)
        with pd.ExcelFile('test1' + read_ext) as excel:
            df1 = excel.parse(0, index_col=0)
            df2 = excel.parse(1, skiprows=[1], index_col=0)
        tm.assert_frame_equal(df1, expected)
        tm.assert_frame_equal(df2, expected)
        with pd.ExcelFile('test1' + read_ext) as excel:
            df3 = pd.read_excel(excel, sheet_name=0, index_col=0, skipfooter=1)
        tm.assert_frame_equal(df3, df1.iloc[:-1])
        with pd.ExcelFile('test1' + read_ext) as excel:
            df3 = excel.parse(0, index_col=0, skipfooter=1)
        tm.assert_frame_equal(df3, df1.iloc[:-1])

    def func_x29f0xa4(self, request: pytest.FixtureRequest, engine: Optional[str], read_ext: str, df_ref: DataFrame) -> None:
        func_2tw9ycnx(engine, request)
        expected = df_ref
        func_r8n3x9o9(expected, read_ext, engine)
        filename = 'test1'
        sheet_name = 'Sheet1'
        with pd.ExcelFile(filename + read_ext) as excel:
            df1_parse = pd.read_excel(excel, sheet_name=sheet_name, index_col=0)
            df2_parse = pd.read_excel(excel, index_col=0, sheet_name=sheet_name)
        tm.assert_frame_equal(df1_parse, expected)
        tm.assert_frame_equal(df2_parse, expected)

    @pytest.mark.parametrize('sheet_name', [3, [0, 3], [3, 0], 'Sheet4', [
        'Sheet1', 'Sheet4'], ['Sheet4', 'Sheet1']])
    def func_b2xk3vtr(self, read_ext: str, sheet_name: Union[str, List[Union[str, int]]]) -> None:
        msg = "Worksheet index 3 is invalid|Worksheet named 'Sheet4' not found"
        with pytest.raises(ValueError, match=msg):
            with pd.ExcelFile('blank' + read_ext) as excel:
                excel.parse(sheet_name=sheet_name)

    def func_370zx14s(self, request: pytest.FixtureRequest, engine: Optional[str], read_ext: str) -> None:
        pth = 'test1' + read_ext
        expected = pd.read_excel(pth, sheet_name='Sheet1', index_col=0)
        with open(pth, 'rb') as f:
            with pd.ExcelFile(f) as xls:
                actual = pd.read_excel(xls, sheet_name='Sheet1', index_col=0)
        tm.assert_frame_equal(expected, actual)

    def func_tfxs9u18(self, engine: Optional[str], read_ext: str) -> None:
        with open('test1' + read_ext, 'rb') as f:
            with pd.ExcelFile(f) as xlsx:
                pd.read_excel(xlsx, sheet_name='Sheet1', index_col=0,
                    engine=engine)
            assert f.closed

    def func_7kdf83v8(self, read_ext: str) -> None:
        msg = 'Engine should not be specified when passing an ExcelFile'
        with pd.ExcelFile('test1' + read_ext) as xl:
            with pytest.raises(ValueError, match=msg):
                pd.read_excel(xl, engine='foo')

    def func_hszzaef4(self, engine: Optional[str], read_ext: str) -> None:
        expected = pd.read_excel('test1' + read_ext, engine=engine)
        with open('test1' + read_ext, 'rb') as f:
            data = f.read()
        actual = pd.read_excel(BytesIO(data), engine=engine)
        tm.assert_frame_equal(expected, actual)

    def func_7ikmkc9m(self, read_ext: str, engine: Optional[str]) -> None:
        with open('test1' + read_ext, 'rb') as f:
            result = pd.read_excel(f, engine=engine)
        expected = pd.read_excel('test1' + read_ext, engine=engine)
        tm.assert_frame_equal(result, expected)

    def func_t6ju4ctv(self, engine: Optional[str]) -> None:
        with open('df_header_oob.xlsx', 'rb') as f:
            with pytest.raises(ValueError, match='exceeds maximum'):
                pd.read_excel(f, header=[0, 1])

    @pytest.mark.parametrize('filename', ['df_empty.xlsx', 'df_equals.xlsx'])
    def func_4sqi4ln2(self, filename: str) -> None:
        idx = Index(['Z'], name='I2')
        cols = MultiIndex.from_tuples([('A', 'B'), ('A', 'B.1')], names=[
            'I11', 'I12'])
        expected = DataFrame([[1, 3]], index=idx, columns=cols, dtype='int64')
        result = pd.read_excel(filename, sheet_name='Sheet1', index_col=0,
            header=[0, 1])
        tm.assert_frame_equal(expected, result)

    def func_hg41h2x8(self, request: pytest.FixtureRequest, engine: Optional[str], read_ext: str) -> None:
        func_2tw9ycnx(engine, request)
        f = 'test_datetime_mi' + read_ext
        with pd.ExcelFile(f) as excel:
            actual = pd.read_excel(excel, header=[0, 1], index_col=0,
                engine=engine)
        unit = func_h5xd31uv(read_ext, engine)
        expected_column_index = MultiIndex.from_arrays([
            pd.to_datetime(['2020-02-29', '2020-03-01'], unit='us').tolist(),
            pd.to_datetime(['2020-02-29', '2020-03-01'], unit='us').tolist()
        ], names=[
            pd.to_datetime(['2020-02-29', '2020-03-01'], unit='us')[0].to_pydatetime(),
            pd.to_datetime(['2020-02-29', '2020-03-01'], unit='us')[1].to_pydatetime()
        ])
        expected = DataFrame([], index=[], columns=expected_column_index)
        tm.assert_frame_equal(expected, actual)

    def func_rpca9dli(self, read_ext: str) -> None:
        with pytest.raises(ValueError, match='Value must be one of *'):
            with pd.option_context(f'io.excel{read_ext}.reader', 'abc'):
                pass

    def func_oviszlf1(self, request: pytest.FixtureRequest, engine: Optional[str], read_ext: str) -> None:
        if read_ext == '.ods':
            pytest.skip('chartsheets do not exist in the ODF format')
        if engine == 'pyxlsb':
            request.applymarker(pytest.mark.xfail(reason=
                "pyxlsb can't distinguish chartsheets from worksheets"))
        with pd.ExcelFile('chartsheet' + read_ext) as excel:
            assert excel.sheet_names == ['Sheet1']

    def func_n4h0gny4(self, engine: Optional[str], tmp_excel: str) -> None:
        errors: Tuple[Any, ...]
        if engine is None:
            pytest.skip(f'Invalid test for engine={engine}')
        elif engine == 'xlrd':
            import xlrd
            errors = (BadZipFile, xlrd.biffh.XLRDError)
        elif engine == 'calamine':
            from python_calamine import CalamineError
            errors = (CalamineError,)
        else:
            errors = (BadZipFile,)
        Path(tmp_excel).write_text('corrupt', encoding='utf-8')
        with tm.assert_produces_warning(False):
            try:
                pd.ExcelFile(tmp_excel, engine=engine)
            except errors:
                pass


class TestReaders:
    # Continued from previous section due to duplication
    pass
