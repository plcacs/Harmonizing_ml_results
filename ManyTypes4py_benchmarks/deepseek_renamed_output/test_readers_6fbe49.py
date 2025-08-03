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
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, cast
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
    engine_val: str = engine.values[0]
    if engine_val == 'openpyxl' and read_ext == '.xls':
        return False
    if engine_val == 'odf' and read_ext != '.ods':
        return False
    if read_ext == '.ods' and engine_val not in {'odf', 'calamine'}:
        return False
    if engine_val == 'pyxlsb' and read_ext != '.xlsb':
        return False
    if read_ext == '.xlsb' and engine_val not in {'pyxlsb', 'calamine'}:
        return False
    if engine_val == 'xlrd' and read_ext != '.xls':
        return False
    return True

def func_jds5urnz(engine: pytest.param, read_ext: str) -> pytest.param:
    """
    engine gives us a pytest.param object with some marks, read_ext is just
    a string.  We need to generate a new pytest.param inheriting the marks.
    """
    values: Tuple[Any, ...] = engine.values + (read_ext,)
    new_param: pytest.param = pytest.param(values, marks=engine.marks)
    return new_param

@pytest.fixture(params=[func_jds5urnz(eng, ext) for eng in engine_params for ext in read_ext_params if func_8c5a9y9x(eng, ext)], ids=str)
def func_kghfv52i(request: pytest.FixtureRequest) -> Any:
    """
    Fixture for Excel reader engine and read_ext, only including valid pairs.
    """
    return request.param

@pytest.fixture
def func_d7yn5ojf(engine_and_read_ext: Tuple[str, str]) -> str:
    engine, read_ext = engine_and_read_ext
    return engine

@pytest.fixture
def func_xzsbpzmy(engine_and_read_ext: Tuple[str, str]) -> str:
    engine, read_ext = engine_and_read_ext
    return read_ext

@pytest.fixture
def func_hn75yqc5(read_ext: str, tmp_path: Path) -> str:
    tmp: Path = tmp_path / f'{uuid.uuid4()}{read_ext}'
    tmp.touch()
    return str(tmp)

@pytest.fixture
def func_u8i3bhxt(datapath: Callable[..., str]) -> DataFrame:
    """
    Obtain the reference data from read_csv with the Python engine.
    """
    filepath: str = datapath('io', 'data', 'csv', 'test1.csv')
    df_ref: DataFrame = read_csv(filepath, index_col=0, parse_dates=True, engine='python')
    return df_ref

def func_h5xd31uv(read_ext: str, engine: str) -> str:
    unit: str = 'us'
    if (read_ext == '.ods') ^ (engine == 'calamine'):
        unit = 's'
    return unit

def func_r8n3x9o9(expected: DataFrame, read_ext: str, engine: str) -> None:
    expected.index.name = None
    unit: str = func_h5xd31uv(read_ext, engine)
    expected.index = expected.index.as_unit(unit)

def func_2tw9ycnx(engine: str, request: pytest.FixtureRequest) -> None:
    if engine == 'pyxlsb':
        request.applymarker(pytest.mark.xfail(reason='Sheets containing datetimes not supported by pyxlsb'))

class TestReaders:
    @pytest.mark.parametrize('col', [[True, None, False], [True], [True, False]])
    def func_9jcm22ji(self, col: List[Optional[bool]], tmp_excel: str, read_ext: str) -> None:
        if read_ext in ('.xlsb', '.xls'):
            pytest.skip(f"No engine for filetype: '{read_ext}'")
        df: DataFrame = DataFrame({'bool_column': col}, dtype='boolean')
        df.to_excel(tmp_excel, index=False)
        df2: DataFrame = pd.read_excel(tmp_excel, dtype={'bool_column': 'boolean'})
        tm.assert_frame_equal(df, df2)

    def func_rbbtqw94(self, datapath: Callable[..., str]) -> None:
        f_path: str = datapath('io', 'data', 'excel', 'test_none_type.xlsx')
        with pd.ExcelFile(f_path) as excel:
            parsed: DataFrame = pd.read_excel(excel, sheet_name='Sheet1',
                keep_default_na=True, na_values=['nan', 'None', 'abcd'],
                dtype='boolean', engine='openpyxl')
        expected: DataFrame = DataFrame({'Test': [True, None, False, None, False, None, True]}, dtype='boolean')
        tm.assert_frame_equal(parsed, expected)

    @pytest.fixture(autouse=True)
    def func_pqpvebzn(self, engine: str, datapath: Callable[..., str], monkeypatch: pytest.MonkeyPatch) -> None:
        """
        Change directory and set engine for read_excel calls.
        """
        func: partial[DataFrame] = partial(pd.read_excel, engine=engine)
        monkeypatch.chdir(datapath('io', 'data', 'excel'))
        monkeypatch.setattr(pd, 'read_excel', func)

    def func_xvopw874(self, read_ext: str, engine: str, monkeypatch: pytest.MonkeyPatch) -> None:
        def func_u2bfvezs(self: Any, *args: Any, **kwargs: Any) -> str:
            return self.engine
        monkeypatch.setattr(pd.ExcelFile, 'parse', func_u2bfvezs)
        expected_defaults: Dict[str, str] = {'xlsx': 'openpyxl', 'xlsm': 'openpyxl', 'xlsb': 'pyxlsb', 'xls': 'xlrd', 'ods': 'odf'}
        with open('test1' + read_ext, 'rb') as f:
            result: str = pd.read_excel(f)
        expected: str = engine if engine is not None else expected_defaults[read_ext[1:]]
        assert result == expected

    def func_h5dyecrz(self, read_ext: str, engine: str) -> None:
        expected_defaults: Dict[str, Dict[str, Union[str, int, bool]] = {
            'xlsx': {'foo': 'abcd'}, 
            'xlsm': {'foo': 123},
            'xlsb': {'foo': 'True'}, 
            'xls': {'foo': True}, 
            'ods': {'foo': 'abcd'}
        }
        if engine in {'xlrd', 'pyxlsb'}:
            msg: str = re.escape("open_workbook() got an unexpected keyword argument 'foo'")
        elif engine == 'odf':
            msg = re.escape("load() got an unexpected keyword argument 'foo'")
        else:
            msg = re.escape("load_workbook() got an unexpected keyword argument 'foo'")
        if engine is not None:
            with pytest.raises(TypeError, match=msg):
                pd.read_excel('test1' + read_ext, sheet_name='Sheet1',
                    index_col=0, engine_kwargs=expected_defaults[read_ext[1:]])

    def func_oj5as2b3(self, read_ext: str) -> None:
        msg: str = 'Passing an integer for `usecols`'
        with pytest.raises(ValueError, match=msg):
            pd.read_excel('test1' + read_ext, sheet_name='Sheet1',
                index_col=0, usecols=3)
        with pytest.raises(ValueError, match=msg):
            pd.read_excel('test1' + read_ext, sheet_name='Sheet2', skiprows=[1], index_col=0, usecols=3)

    def func_z3koxpzd(self, request: pytest.FixtureRequest, engine: str, read_ext: str, df_ref: DataFrame) -> None:
        func_2tw9ycnx(engine, request)
        expected: DataFrame = df_ref[['B', 'C']]
        func_r8n3x9o9(expected, read_ext, engine)
        df1: DataFrame = pd.read_excel('test1' + read_ext, sheet_name='Sheet1', index_col=0, usecols=[0, 2, 3])
        df2: DataFrame = pd.read_excel('test1' + read_ext, sheet_name='Sheet2', skiprows=[1], index_col=0, usecols=[0, 2, 3])
        tm.assert_frame_equal(df1, expected)
        tm.assert_frame_equal(df2, expected)

    def func_yxcwh5jq(self, request: pytest.FixtureRequest, engine: str, read_ext: str, df_ref: DataFrame) -> None:
        func_2tw9ycnx(engine, request)
        expected: DataFrame = df_ref[['A', 'B', 'C']]
        func_r8n3x9o9(expected, read_ext, engine)
        df2: DataFrame = pd.read_excel('test1' + read_ext, sheet_name='Sheet1', index_col=0, usecols='A:D')
        df3: DataFrame = pd.read_excel('test1' + read_ext, sheet_name='Sheet2', skiprows=[1], index_col=0, usecols='A:D')
        tm.assert_frame_equal(df2, expected)
        tm.assert_frame_equal(df3, expected)
        expected = df_ref[['B', 'C']]
        func_r8n3x9o9(expected, read_ext, engine)
        df2 = pd.read_excel('test1' + read_ext, sheet_name='Sheet1', index_col=0, usecols='A,C,D')
        df3 = pd.read_excel('test1' + read_ext, sheet_name='Sheet2', skiprows=[1], index_col=0, usecols='A,C,D')
        tm.assert_frame_equal(df2, expected)
        tm.assert_frame_equal(df3, expected)
        df2 = pd.read_excel('test1' + read_ext, sheet_name='Sheet1', index_col=0, usecols='A,C:D')
        df3 = pd.read_excel('test1' + read_ext, sheet_name='Sheet2', skiprows=[1], index_col=0, usecols='A,C:D')
        tm.assert_frame_equal(df2, expected)
        tm.assert_frame_equal(df3, expected)

    @pytest.mark.parametrize('usecols', [[0, 1, 3], [0, 3, 1], [1, 0, 3], [1, 3, 0], [3, 0, 1], [3, 1, 0]])
    def func_gamky2lo(self, request: pytest.FixtureRequest, engine: str, read_ext: str, usecols: List[int], df_ref: DataFrame) -> None:
        func_2tw9ycnx(engine, request)
        expected: DataFrame = df_ref[['A', 'C']]
        func_r8n3x9o9(expected, read_ext, engine)
        result: DataFrame = pd.read_excel('test1' + read_ext, sheet_name='Sheet1', index_col=0, usecols=usecols)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('usecols', [['B', 'D'], ['D', 'B']])
    def func_0t8qtwg1(self, read_ext: str, usecols: List[str], df_ref: DataFrame) -> None:
        expected: DataFrame = df_ref[['B', 'D']]
        expected.index = range(len(expected))
        result: DataFrame = pd.read_excel('test1' + read_ext, sheet_name='Sheet1', usecols=usecols)
        tm.assert_frame_equal(result, expected)

    def func_prw3pvl9(self, request: pytest.FixtureRequest, engine: str, read_ext: str, df_ref: DataFrame) -> None:
        func_2tw9ycnx(engine, request)
        expected: DataFrame = df_ref
        func_r8n3x9o9(expected, read_ext, engine)
        result: DataFrame = pd.read_excel('test1' + read_ext, sheet_name='Sheet1', index_col=0)
        tm.assert_frame_equal(result, expected)

    def func_ot2ryb70(self, request: pytest.FixtureRequest, engine: str, read_ext: str, df_ref: DataFrame) -> None:
        func_2tw9ycnx(engine, request)
        expected: DataFrame = df_ref[['C', 'D']]
        func_r8n3x9o9(expected, read_ext, engine)
        result: DataFrame = pd.read_excel('test1' + read_ext, sheet_name='Sheet1', index_col=0, usecols='A,D:E')
        tm.assert_frame_equal(result, expected)

    def func_odz2bv7c(self, read_ext: str) -> None:
        msg: str = 'Invalid column name: E1'
        with pytest.raises(ValueError, match=msg):
            pd.read_excel('test1' + read_ext, sheet_name='Sheet1', usecols='D:E1')

    def func_e7xv9x7m(self, read_ext: str) -> None:
        msg: str = 'list indices must be integers.*, not str'
        with pytest.raises(TypeError, match=msg):
            pd.read_excel('test1' + read_ext, sheet_name='Sheet1', index_col=['A'], usecols=['A', 'C'])

    def func_8u4hi162(self, read_ext: str) -> None:
        result: DataFrame = pd.read_excel('test1' + read_ext, sheet_name='Sheet3', index_col='A')
        expected: DataFrame = DataFrame(columns=['B', 'C', 'D', 'E', 'F'], index=Index([], name='A'))
        tm.assert_frame_equal(result, expected)

    def func_btud0p8t(self, read_ext: str) -> None:
        result: DataFrame = pd.read_excel('test1' + read_ext, sheet_name='Sheet3', index_col=['A', 'B', 'C'])
        expected: DataFrame = DataFrame(columns=['D', 'E', 'F'], index=MultiIndex(
            levels=[[]] * 3, codes=[[]] * 3, names=['A', 'B', 'C']))
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('index_col', [None, 2])
    def func_fc989g5u(self, read_ext: str, index_col: Optional[int]) -> None:
        result: DataFrame = pd.read_excel('test1' + read_ext, sheet_name='Sheet4', index_col=index_col)
        expected: DataFrame = DataFrame([['i1', 'a', 'x'], ['i2', 'b', 'y']], columns=['Unnamed: 0', 'col1', 'col2'])
        if index_col:
            expected = expected.set_index(expected.columns[index_col])
        tm.assert_frame_equal(result, expected)

    def func_1skx9sju(self, read_ext: str) -> None:
        msg: str = "Usecols do not match columns, columns expected but not found: \\['E'\\]"
        with pytest.raises(ValueError, match=msg):
            pd.read_excel('test1' + read_ext, usecols=['E'])

    def func_tlzzjw1a(self, read_ext: str) -> None:
        msg: str = "'usecols' must either be list-like of all strings, all unicode, all integers or a callable."
        with pytest.raises(ValueError, match=msg):
            pd.read_excel('test1' + read_ext, usecols=['E1', 0])

    def func_lhnpaax8(self, read_ext: str) -> None:
        parsed: DataFrame = pd.read_excel('test2' + read_ext, sheet_name='Sheet1')
        expected: DataFrame = DataFrame([['aaaa', 'bbbbb']], columns=['Test', 'Test1'])
        tm.assert_frame_equal(parsed, expected)

    def func_xd4a5cnq(self, request: pytest.FixtureRequest, engine: str, read_ext: str) -> None:
        func_2tw9ycnx(engine, request)
        if engine == 'calamine' and read_ext == '.ods':
            request.applymarker(pytest.mark.xfail(reason="Calamine can't extract error from ods files"))
        parsed: DataFrame = pd.read_excel('test3