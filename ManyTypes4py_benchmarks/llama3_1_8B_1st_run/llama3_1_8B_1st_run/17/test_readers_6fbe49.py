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
import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import DataFrame, Index, MultiIndex, Series, read_csv
import pandas._testing as tm

read_ext_params: list[str] = ['.xls', '.xlsx', '.xlsm', '.xlsb', '.ods']
engine_params: list[pytest.Param] = [
    pytest.param('xlrd', marks=[td.skip_if_no('xlrd')]),
    pytest.param('openpyxl', marks=[td.skip_if_no('openpyxl')]),
    pytest.param(None, marks=[td.skip_if_no('xlrd')]),
    pytest.param('pyxlsb', marks=td.skip_if_no('pyxlsb')),
    pytest.param('odf', marks=td.skip_if_no('odf')),
    pytest.param('calamine', marks=td.skip_if_no('python_calamine')),
]

def _is_valid_engine_ext_pair(engine: pytest.Param, read_ext: str) -> bool:
    """
    Filter out invalid (engine, ext) pairs instead of skipping, as that
    produces 500+ pytest.skips.
    """
    engine_value: str = engine.values[0]
    if engine_value == 'openpyxl' and read_ext == '.xls':
        return False
    if engine_value == 'odf' and read_ext != '.ods':
        return False
    if read_ext == '.ods' and engine_value not in {'odf', 'calamine'}:
        return False
    if engine_value == 'pyxlsb' and read_ext != '.xlsb':
        return False
    if read_ext == '.xlsb' and engine_value not in {'pyxlsb', 'calamine'}:
        return False
    if engine_value == 'xlrd' and read_ext != '.xls':
        return False
    return True

def _transfer_marks(engine: pytest.Param, read_ext: str) -> pytest.Param:
    """
    engine gives us a pytest.param object with some marks, read_ext is just
    a string.  We need to generate a new pytest.param inheriting the marks.
    """
    values: tuple = engine.values + (read_ext,)
    new_param: pytest.Param = pytest.param(values, marks=engine.marks)
    return new_param

@pytest.fixture(params=[_transfer_marks(eng, ext) for eng in engine_params for ext in read_ext_params if _is_valid_engine_ext_pair(eng, ext)], ids=str)
def engine_and_read_ext(request: pytest.FixtureRequest) -> tuple[str, str]:
    """
    Fixture for Excel reader engine and read_ext, only including valid pairs.
    """
    return request.param

@pytest.fixture
def engine(engine_and_read_ext: tuple[str, str]) -> str:
    engine_value: str
    read_ext: str
    engine_value, read_ext = engine_and_read_ext
    return engine_value

@pytest.fixture
def read_ext(engine_and_read_ext: tuple[str, str]) -> str:
    engine_value: str
    read_ext: str
    engine_value, read_ext = engine_and_read_ext
    return read_ext

@pytest.fixture
def tmp_excel(read_ext: str, tmp_path: Path) -> str:
    tmp: Path = tmp_path / f'{uuid.uuid4()}{read_ext}'
    tmp.touch()
    return str(tmp)

@pytest.fixture
def df_ref(datapath: Path) -> DataFrame:
    """
    Obtain the reference data from read_csv with the Python engine.
    """
    filepath: Path = datapath('io', 'data', 'csv', 'test1.csv')
    df_ref: DataFrame = read_csv(filepath, index_col=0, parse_dates=True, engine='python')
    return df_ref

def get_exp_unit(read_ext: str, engine: str) -> str:
    """
    Determine the expected unit for datetime columns.
    """
    unit: str = 'us'
    if (read_ext == '.ods') ^ (engine == 'calamine'):
        unit = 's'
    return unit

def adjust_expected(expected: DataFrame, read_ext: str, engine: str) -> DataFrame:
    """
    Adjust the expected DataFrame to match the engine and file type.
    """
    expected.index.name = None
    unit: str = get_exp_unit(read_ext, engine)
    expected.index = expected.index.as_unit(unit)

class TestReaders:
    @pytest.mark.parametrize('col', [[True, None, False], [True], [True, False]])
    def test_read_excel_type_check(self, col: list, tmp_excel: str, read_ext: str) -> None:
        if read_ext in ('.xlsb', '.xls'):
            pytest.skip(f"No engine for filetype: '{read_ext}'")
        df: DataFrame = DataFrame({'bool_column': col}, dtype='boolean')
        df.to_excel(tmp_excel, index=False)
        df2: DataFrame = pd.read_excel(tmp_excel, dtype={'bool_column': 'boolean'})
        tm.assert_frame_equal(df, df2)

    def test_pass_none_type(self, datapath: Path) -> None:
        f_path: Path = datapath('io', 'data', 'excel', 'test_none_type.xlsx')
        with pd.ExcelFile(f_path) as excel:
            parsed: DataFrame = pd.read_excel(excel, sheet_name='Sheet1', keep_default_na=True, na_values=['nan', 'None', 'abcd'], dtype='boolean', engine='openpyxl')
        expected: DataFrame = DataFrame({'Test': [True, None, False, None, False, None, True]}, dtype='boolean')
        tm.assert_frame_equal(parsed, expected)

    @pytest.fixture(autouse=True)
    def cd_and_set_engine(self, engine: str, datapath: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """
        Change directory and set engine for read_excel calls.
        """
        func: callable = partial(pd.read_excel, engine=engine)
        monkeypatch.chdir(datapath('io', 'data', 'excel'))
        monkeypatch.setattr(pd, 'read_excel', func)

    def test_engine_used(self, read_ext: str, engine: str, monkeypatch: pytest.MonkeyPatch) -> None:
        expected_defaults: dict[str, str] = {'xlsx': 'openpyxl', 'xlsm': 'openpyxl', 'xlsb': 'pyxlsb', 'xls': 'xlrd', 'ods': 'odf'}
        with open('test1' + read_ext, 'rb') as f:
            result: str = pd.read_excel(f)
        if engine is not None:
            expected: str = engine
        else:
            expected: str = expected_defaults[read_ext[1:]]
        assert result == expected

    def test_engine_kwargs(self, read_ext: str, engine: str) -> None:
        expected_defaults: dict[str, dict[str, str]] = {'xlsx': {'foo': 'abcd'}, 'xlsm': {'foo': 123}, 'xlsb': {'foo': 'True'}, 'xls': {'foo': True}, 'ods': {'foo': 'abcd'}}
        if engine in {'xlrd', 'pyxlsb'}:
            msg: str = re.escape("open_workbook() got an unexpected keyword argument 'foo'")
        elif engine == 'odf':
            msg: str = re.escape("load() got an unexpected keyword argument 'foo'")
        else:
            msg: str = re.escape("load_workbook() got an unexpected keyword argument 'foo'")
        if engine is not None:
            with pytest.raises(TypeError, match=msg):
                pd.read_excel('test1' + read_ext, sheet_name='Sheet1', index_col=0, engine_kwargs=expected_defaults[read_ext[1:]])

    def test_usecols_int(self, read_ext: str) -> None:
        msg: str = 'Passing an integer for `usecols`'
        with pytest.raises(ValueError, match=msg):
            pd.read_excel('test1' + read_ext, sheet_name='Sheet1', index_col=0, usecols=3)
        with pytest.raises(ValueError, match=msg):
            pd.read_excel('test1' + read_ext, sheet_name='Sheet2', skiprows=[1], index_col=0, usecols=3)

    def test_usecols_list(self, request: pytest.FixtureRequest, engine: str, read_ext: str, df_ref: DataFrame) -> None:
        xfail_datetimes_with_pyxlsb(engine, request)
        expected: DataFrame = df_ref[['B', 'C']]
        adjust_expected(expected, read_ext, engine)
        df1: DataFrame = pd.read_excel('test1' + read_ext, sheet_name='Sheet1', index_col=0, usecols=[0, 2, 3])
        df2: DataFrame = pd.read_excel('test1' + read_ext, sheet_name='Sheet2', skiprows=[1], index_col=0, usecols=[0, 2, 3])
        tm.assert_frame_equal(df1, expected)
        tm.assert_frame_equal(df2, expected)

    def test_usecols_str(self, request: pytest.FixtureRequest, engine: str, read_ext: str, df_ref: DataFrame) -> None:
        xfail_datetimes_with_pyxlsb(engine, request)
        expected: DataFrame = df_ref[['A', 'B', 'C']]
        adjust_expected(expected, read_ext, engine)
        df2: DataFrame = pd.read_excel('test1' + read_ext, sheet_name='Sheet1', index_col=0, usecols='A:D')
        df3: DataFrame = pd.read_excel('test1' + read_ext, sheet_name='Sheet2', skiprows=[1], index_col=0, usecols='A:D')
        tm.assert_frame_equal(df2, expected)
        tm.assert_frame_equal(df3, expected)
        expected: DataFrame = df_ref[['B', 'C']]
        adjust_expected(expected, read_ext, engine)
        df2: DataFrame = pd.read_excel('test1' + read_ext, sheet_name='Sheet1', index_col=0, usecols='A,C,D')
        df3: DataFrame = pd.read_excel('test1' + read_ext, sheet_name='Sheet2', skiprows=[1], index_col=0, usecols='A,C,D')
        tm.assert_frame_equal(df2, expected)
        tm.assert_frame_equal(df3, expected)
        df2: DataFrame = pd.read_excel('test1' + read_ext, sheet_name='Sheet1', index_col=0, usecols='A,C:D')
        df3: DataFrame = pd.read_excel('test1' + read_ext, sheet_name='Sheet2', skiprows=[1], index_col=0, usecols='A,C:D')
        tm.assert_frame_equal(df2, expected)
        tm.assert_frame_equal(df3, expected)

    # ... rest of the code remains the same ...
