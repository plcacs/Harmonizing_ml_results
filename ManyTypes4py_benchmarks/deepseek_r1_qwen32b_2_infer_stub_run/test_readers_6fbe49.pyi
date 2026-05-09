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
from pytest import Param, fixture, mark, raises

read_ext_params: list[str] = ['.xls', '.xlsx', '.xlsm', '.xlsb', '.ods']
engine_params: list[Param] = [
    pytest.param('xlrd', marks=[td.skip_if_no('xlrd')]),
    pytest.param('openpyxl', marks=[td.skip_if_no('openpyxl')]),
    pytest.param(None, marks=[td.skip_if_no('xlrd')]),
    pytest.param('pyxlsb', marks=td.skip_if_no('pyxlsb')),
    pytest.param('odf', marks=td.skip_if_no('odf')),
    pytest.param('calamine', marks=td.skip_if_no('python_calamine'))
]

def _is_valid_engine_ext_pair(engine: str, read_ext: str) -> bool:
    ...

def _transfer_marks(engine: pytest.param, read_ext: str) -> Param:
    ...

@fixture(params=[_transfer_marks(eng, ext) for eng in engine_params for ext in read_ext_params if _is_valid_engine_ext_pair(eng, ext)], ids=str)
def engine_and_read_ext(request: pytest.FixtureRequest) -> tuple[str, str]:
    ...

@fixture
def engine(engine_and_read_ext: tuple[str, str]) -> str:
    ...

@fixture
def read_ext(engine_and_read_ext: tuple[str, str]) -> str:
    ...

@fixture
def tmp_excel(read_ext: str, tmp_path: Path) -> str:
    ...

@fixture
def df_ref(datapath: str) -> DataFrame:
    ...

def get_exp_unit(read_ext: str, engine: str) -> str:
    ...

def adjust_expected(expected: DataFrame, read_ext: str, engine: str) -> None:
    ...

def xfail_datetimes_with_pyxlsb(engine: str, request: pytest.FixtureRequest) -> None:
    ...

class TestReaders:
    @mark.parametrize('col', [[True, None, False], [True], [True, False]])
    def test_read_excel_type_check(self, col: list[bool], tmp_excel: str, read_ext: str) -> None:
        ...

    def test_pass_none_type(self, datapath: str) -> None:
        ...

    @fixture(autouse=True)
    def cd_and_set_engine(self, engine: str, datapath: str, monkeypatch: pytest.MonkeyPatch) -> None:
        ...

    def test_engine_used(self, read_ext: str, engine: str, monkeypatch: pytest.MonkeyPatch) -> None:
        ...

    def test_engine_kwargs(self, read_ext: str, engine: str) -> None:
        ...

    def test_usecols_int(self, read_ext: str) -> None:
        ...

    def test_usecols_list(self, request: pytest.FixtureRequest, engine: str, read_ext: str, df_ref: DataFrame) -> None:
        ...

    def test_usecols_str(self, request: pytest.FixtureRequest, engine: str, read_ext: str, df_ref: DataFrame) -> None:
        ...

    @mark.parametrize('usecols', [[0, 1, 3], [0, 3, 1], [1, 0, 3], [1, 3, 0], [3, 0, 1], [3, 1, 0]])
    def test_usecols_diff_positional_int_columns_order(self, request: pytest.FixtureRequest, engine: str, read_ext: str, usecols: list[int], df_ref: DataFrame) -> None:
        ...

    @mark.parametrize('usecols', [['B', 'D'], ['D', 'B']])
    def test_usecols_diff_positional_str_columns_order(self, read_ext: str, usecols: list[str], df_ref: DataFrame) -> None:
        ...

    def test_read_excel_without_slicing(self, request: pytest.FixtureRequest, engine: str, read_ext: str, df_ref: DataFrame) -> None:
        ...

    def test_usecols_excel_range_str(self, request: pytest.FixtureRequest, engine: str, read_ext: str, df_ref: DataFrame) -> None:
        ...

    def test_usecols_excel_range_str_invalid(self, read_ext: str) -> None:
        ...

    def test_index_col_label_error(self, read_ext: str) -> None:
        ...

    def test_index_col_str(self, read_ext: str) -> None:
        ...

    def test_index_col_empty(self, read_ext: str) -> None:
        ...

    @mark.parametrize('index_col', [None, 2])
    def test_index_col_with_unnamed(self, read_ext: str, index_col: int | None) -> None:
        ...

    def test_usecols_pass_non_existent_column(self, read_ext: str) -> None:
        ...

    def test_usecols_wrong_type(self, read_ext: str) -> None:
        ...

    def test_excel_stop_iterator(self, read_ext: str) -> None:
        ...

    def test_excel_cell_error_na(self, request: pytest.FixtureRequest, engine: str, read_ext: str) -> None:
        ...

    def test_excel_table(self, request: pytest.FixtureRequest, engine: str, read_ext: str, df_ref: DataFrame) -> None:
        ...

    def test_reader_special_dtypes(self, request: pytest.FixtureRequest, engine: str, read_ext: str) -> None:
        ...

    def test_reader_converters(self, read_ext: str) -> None:
        ...

    def test_reader_dtype(self, read_ext: str) -> None:
        ...

    @mark.parametrize('dtype,expected', [(None, {'a': [1, 2, 3, 4], 'b': [2.5, 3.5, 4.5, 5.5], 'c': [1, 2, 3, 4], 'd': [1.0, 2.0, np.nan, 4.0]}), ({'a': 'float64', 'b': 'float32', 'c': str, 'd': str}, {'a': Series([1, 2, 3, 4], dtype='float64'), 'b': Series([2.5, 3.5, 4.5, 5.5], dtype='float32'), 'c': Series(['001', '002', '003', '004'], dtype='str'), 'd': Series(['1', '2', np.nan, '4'], dtype='str')})])
    def test_reader_dtype_str(self, read_ext: str, dtype: dict | None, expected: dict) -> None:
        ...

    def test_dtype_backend(self, read_ext: str, dtype_backend: str, engine: str, tmp_excel: str) -> None:
        ...

    def test_dtype_backend_and_dtype(self, read_ext: str, tmp_excel: str) -> None:
        ...

    def test_dtype_backend_string(self, read_ext: str, string_storage: str, tmp_excel: str) -> None:
        ...

    @mark.parametrize('dtypes, exp_value', [({}, 1), ({'a.1': 'int64'}, 1)])
    def test_dtype_mangle_dup_cols(self, read_ext: str, dtypes: dict, exp_value: int) -> None:
        ...

    def test_reader_spaces(self, read_ext: str) -> None:
        ...

    @mark.parametrize('basename,expected', [('gh-35802', DataFrame({'COLUMN': ['Test (1)']})), ('gh-36122', DataFrame(columns=['got 2nd sa']))])
    def test_read_excel_ods_nested_xml(self, engine: str, read_ext: str, basename: str, expected: DataFrame) -> None:
        ...

    def test_reading_all_sheets(self, read_ext: str) -> None:
        ...

    def test_reading_multiple_specific_sheets(self, read_ext: str) -> None:
        ...

    def test_reading_all_sheets_with_blank(self, read_ext: str) -> None:
        ...

    def test_read_excel_blank(self, read_ext: str) -> None:
        ...

    def test_read_excel_blank_with_header(self, read_ext: str) -> None:
        ...

    def test_exception_message_includes_sheet_name(self, read_ext: str) -> None:
        ...

    @mark.filterwarnings('ignore:Cell A4 is marked:UserWarning:openpyxl')
    def test_date_conversion_overflow(self, request: pytest.FixtureRequest, engine: str, read_ext: str) -> None:
        ...

    def test_sheet_name(self, request: pytest.FixtureRequest, read_ext: str, engine: str, df_ref: DataFrame) -> None:
        ...

    def test_excel_read_buffer(self, read_ext: str) -> None:
        ...

    def test_bad_engine_raises(self) -> None:
        ...

    @mark.parametrize('sheet_name', [3, [0, 3], [3, 0], 'Sheet4', ['Sheet1', 'Sheet4'], ['Sheet4', 'Sheet1']])
    def test_bad_sheetname_raises(self, read_ext: str, sheet_name: int | str | list[int | str]) -> None:
        ...

    def test_missing_file_raises(self, read_ext: str) -> None:
        ...

    def test_corrupt_bytes_raises(self, engine: str) -> None:
        ...

    @mark.network
    @mark.single_cpu
    def test_read_from_http_url(self, httpserver: pytest.fixture, read_ext: str) -> None:
        ...

    @td.skip_if_not_us_locale
    @mark.single_cpu
    def test_read_from_s3_url(self, read_ext: str, s3_public_bucket: pytest.fixture, s3so: dict) -> None:
        ...

    @mark.single_cpu
    def test_read_from_s3_object(self, read_ext: str, s3_public_bucket: pytest.fixture, s3so: dict) -> None:
        ...

    @mark.slow
    def test_read_from_file_url(self, read_ext: str, datapath: str) -> None:
        ...

    def test_read_from_pathlib_path(self, read_ext: str) -> None:
        ...

    def test_close_from_py_localpath(self, read_ext: str) -> None:
        ...

    def test_reader_seconds(self, request: pytest.FixtureRequest, engine: str, read_ext: str) -> None:
        ...

    def test_read_excel_multiindex(self, request: pytest.FixtureRequest, engine: str, read_ext: str) -> None:
        ...

    @mark.parametrize('sheet_name,idx_lvl2', [('both_name_blank_after_mi_name', [np.nan, 'b', 'a', 'b']), ('both_name_multiple_blanks', [np.nan] * 4)])
    def test_read_excel_multiindex_blank_after_name(self, request: pytest.FixtureRequest, engine: str, read_ext: str, sheet_name: str, idx_lvl2: list) -> None:
        ...

    def test_read_excel_multiindex_header_only(self, read_ext: str) -> None:
        ...

    def test_excel_old_index_format(self, read_ext: str) -> None:
        ...

    def test_read_excel_bool_header_arg(self, read_ext: str) -> None:
        ...

    def test_read_excel_skiprows(self, request: pytest.FixtureRequest, engine: str, read_ext: str) -> None:
        ...

    def test_read_excel_skiprows_callable_not_in(self, request: pytest.FixtureRequest, engine: str, read_ext: str) -> None:
        ...

    def test_read_excel_nrows(self, read_ext: str) -> None:
        ...

    def test_read_excel_nrows_greater_than_nrows_in_file(self, read_ext: str) -> None:
        ...

    def test_read_excel_nrows_non_integer_parameter(self, read_ext: str) -> None:
        ...

    @mark.parametrize('filename,sheet_name,header,index_col,skiprows', [('testmultiindex', 'mi_column', [0, 1], 0, None), ('testmultiindex', 'mi_index', None, [0, 1], None), ('testmultiindex', 'both', [0, 1], [0, 1], None), ('testmultiindex', 'mi_column_name', [0, 1], 0, None), ('testskiprows', 'skiprows_list', None, None, [0, 2]), ('testskiprows', 'skiprows_list', None, None, lambda x: x in (0, 2))])
    def test_read_excel_nrows_params(self, read_ext: str, filename: str, sheet_name: str, header: list[int], index_col: int | list[int], skiprows: int | list[int] | callable) -> None:
        ...

    def test_deprecated_kwargs(self, read_ext: str) -> None:
        ...

    def test_no_header_with_list_index_col(self, read_ext: str) -> None:
        ...

    def test_one_col_noskip_blank_line(self, read_ext: str) -> None:
        ...

    def test_multiheader_two_blank_lines(self, read_ext: str) -> None:
        ...

    def test_trailing_blanks(self, read_ext: str) -> None:
        ...

    def test_ignore_chartsheets_by_str(self, request: pytest.FixtureRequest, engine: str, read_ext: str) -> None:
        ...

    def test_ignore_chartsheets_by_int(self, request: pytest.FixtureRequest, engine: str, read_ext: str) -> None:
        ...

    def test_euro_decimal_format(self, read_ext: str) -> None:
        ...

class TestExcelFileRead:
    def test_raises_bytes_input(self, engine: str, read_ext: str) -> None:
        ...

    @fixture(autouse=True)
    def cd_and_set_engine(self, engine: str, datapath: str, monkeypatch: pytest.MonkeyPatch) -> None:
        ...

    def test_engine_used(self, read_ext: str, engine: str) -> None:
        ...

    def test_excel_passes_na(self, read_ext: str) -> None:
        ...

    @mark.parametrize('na_filter', [None, True, False])
    def test_excel_passes_na_filter(self, read_ext: str, na_filter: bool | None) -> None:
        ...

    def test_excel_table_sheet_by_index(self, request: pytest.FixtureRequest, engine: str, read_ext: str, df_ref: DataFrame) -> None:
        ...

    def test_sheet_name(self, request: pytest.FixtureRequest, engine: str, read_ext: str, df_ref: DataFrame) -> None:
        ...

    @mark.parametrize('sheet_name', [3, [0, 3], [3, 0], 'Sheet4', ['Sheet1', 'Sheet4'], ['Sheet4', 'Sheet1']])
    def test_bad_sheetname_raises(self, read_ext: str, sheet_name: int | str | list[int | str]) -> None:
        ...

    def test_excel_read_buffer(self, engine: str, read_ext: str) -> None:
        ...

    def test_reader_closes_file(self, engine: str, read_ext: str) -> None:
        ...

    def test_conflicting_excel_engines(self, read_ext: str) -> None:
        ...

    def test_excel_read_binary(self, engine: str, read_ext: str) -> None:
        ...

    def test_excel_read_binary_via_read_excel(self, read_ext: str, engine: str) -> None:
        ...

    def test_read_excel_header_index_out_of_range(self, engine: str) -> None:
        ...

    @mark.parametrize('filename', ['df_empty.xlsx', 'df_equals.xlsx'])
    def test_header_with_index_col(self, filename: str) -> None:
        ...

    def test_read_datetime_multiindex(self, request: pytest.FixtureRequest, engine: str, read_ext: str) -> None:
        ...

    def test_engine_invalid_option(self, read_ext: str) -> None:
        ...

    def test_ignore_chartsheets(self, request: pytest.FixtureRequest, engine: str, read_ext: str) -> None:
        ...

    def test_corrupt_files_closed(self, engine: str, tmp_excel: str) -> None:
        ...