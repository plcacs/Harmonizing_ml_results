"""
Stub file for 'test_common_basic_a8eb4c.py'
"""

from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Any, Callable, List, Optional, Union

import numpy as np
import pytest
from pandas import DataFrame, Index
from pandas._testing import tm
from pytest importFixture

@pytest.fixture
def all_parsers() -> Any:
    ...

@pytest.fixture
def csv1() -> Path:
    ...

@pytest.fixture
def csv_dir_path() -> Path:
    ...

@pytest.fixture
def xfail_pyarrow() -> Callable:
    ...

@pytest.fixture
def skip_pyarrow() -> Callable:
    ...

def test_read_csv_local(all_parsers:Fixture[Any], csv1:Path) -> None:
    ...

def test_1000_sep(all_parsers:Fixture[Any]) -> None:
    ...

def test_unnamed_columns(all_parsers:Fixture[Any]) -> None:
    ...

def test_csv_mixed_type(all_parsers:Fixture[Any]) -> None:
    ...

def test_read_csv_low_memory_no_rows_with_index(all_parsers:Fixture[Any]) -> None:
    ...

def test_read_csv_dataframe(all_parsers:Fixture[Any], csv1:Path) -> None:
    ...

@pytest.mark.parametrize('nrows', [3, 3.0])
def test_read_nrows(all_parsers:Fixture[Any], nrows:int) -> None:
    ...

@pytest.mark.parametrize('nrows', [1.2, 'foo', -1])
def test_read_nrows_bad(all_parsers:Fixture[Any], nrows:Union[float, str, int]) -> None:
    ...

def test_nrows_skipfooter_errors(all_parsers:Fixture[Any]) -> None:
    ...

def test_missing_trailing_delimiters(all_parsers:Fixture[Any]) -> None:
    ...

def test_skip_initial_space(all_parsers:Fixture[Any]) -> None:
    ...

def test_trailing_delimiters(all_parsers:Fixture[Any]) -> None:
    ...

def test_escapechar(all_parsers:Fixture[Any]) -> None:
    ...

def test_ignore_leading_whitespace(all_parsers:Fixture[Any]) -> None:
    ...

@pytest.mark.parametrize('usecols', [None, [0, 1], ['a', 'b']])
def test_uneven_lines_with_usecols(all_parsers:Fixture[Any], usecols:Optional[List[Union[int, str]]]) -> None:
    ...

@pytest.mark.parametrize('data,kwargs,expected', [('', {}, None), ('', {'usecols': ['X']}, None), (',,', {'names': ['Dummy', 'X', 'Dummy_2'], 'usecols': ['X']}, DataFrame(columns=['X'], index=[0], dtype=np.float64)), ('', {'names': ['Dummy', 'X', 'Dummy_2'], 'usecols': ['X']}, DataFrame(columns=['X']))])
def test_read_empty_with_usecols(all_parsers:Fixture[Any], data:str, kwargs:dict, expected:Optional[DataFrame]) -> None:
    ...

@pytest.mark.parametrize('kwargs,expected_data', [({'header': None, 'sep': '\\s+', 'skiprows': [0, 1, 2, 3, 5, 6], 'skip_blank_lines': True}, [[1.0, 2.0, 4.0], [5.1, np.nan, 10.0]]), ({'sep': '\\s+', 'skiprows': [1, 2, 3, 5, 6], 'skip_blank_lines': True}, {'A': [1.0, 5.1], 'B': [2.0, np.nan], 'C': [4.0, 10]})])
def test_trailing_spaces(all_parsers:Fixture[Any], kwargs:dict, expected_data:Union[List[List[float]], dict]) -> None:
    ...

def test_read_filepath_or_buffer(all_parsers:Fixture[Any]) -> None:
    ...

def test_single_char_leading_whitespace(all_parsers:Fixture[Any]) -> None:
    ...

@pytest.mark.parametrize('sep,skip_blank_lines,exp_data', [(',', True, [[1.0, 2.0, 4.0], [5.0, np.nan, 10.0], [-70.0, 0.4, 1.0]]), ('\\s+', True, [[1.0, 2.0, 4.0], [5.0, np.nan, 10.0], [-70.0, 0.4, 1.0]]), (',', False, [[1.0, 2.0, 4.0], [np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan], [5.0, np.nan, 10.0], [np.nan, np.nan, np.nan], [-70.0, 0.4, 1.0]])])
def test_empty_lines(all_parsers:Fixture[Any], sep:str, skip_blank_lines:bool, exp_data:List[List[float]], request:Any) -> None:
    ...

def test_whitespace_lines(all_parsers:Fixture[Any]) -> None:
    ...

@pytest.mark.parametrize('data,expected', [('   A   B   C   D\na   1   2   3   4\nb   1   2   3   4\nc   1   2   3   4\n', DataFrame([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]], columns=['A', 'B', 'C', 'D'], index=['a', 'b', 'c'])), ('    a b c\n1 2 3 \n4 5  6\n 7 8 9', DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]], columns=['a', 'b', 'c']))])
def test_whitespace_regex_separator(all_parsers:Fixture[Any], data:str, expected:DataFrame) -> None:
    ...

def test_sub_character(all_parsers:Fixture[Any], csv_dir_path:Path) -> None:
    ...

@pytest.mark.parametrize('filename', ['sé-es-vé.csv', 'ru-sй.csv', '中文文件名.csv'])
def test_filename_with_special_chars(all_parsers:Fixture[Any], filename:str) -> None:
    ...

def test_read_table_same_signature_as_read_csv(all_parsers:Fixture[Any]) -> None:
    ...

def test_read_table_equivalency_to_read_csv(all_parsers:Fixture[Any]) -> None:
    ...

@pytest.mark.parametrize('read_func', ['read_csv', 'read_table'])
def test_read_csv_and_table_sys_setprofile(all_parsers:Fixture[Any], read_func:str) -> None:
    ...

def test_first_row_bom(all_parsers:Fixture[Any]) -> None:
    ...

def test_first_row_bom_unquoted(all_parsers:Fixture[Any]) -> None:
    ...

@pytest.mark.parametrize('nrows', range(1, 6))
def test_blank_lines_between_header_and_data_rows(all_parsers:Fixture[Any], nrows:int) -> None:
    ...

def test_no_header_two_extra_columns(all_parsers:Fixture[Any]) -> None:
    ...

def test_read_csv_names_not_accepting_sets(all_parsers:Fixture[Any]) -> None:
    ...

def test_read_csv_delimiter_and_sep_no_default(all_parsers:Fixture[Any]) -> None:
    ...

@pytest.mark.parametrize('kwargs', [{'delimiter': '\n'}, {'sep': '\n'}])
def test_read_csv_line_break_as_separator(kwargs:dict, all_parsers:Fixture[Any]) -> None:
    ...

def test_dict_keys_as_names(all_parsers:Fixture[Any]) -> None:
    ...

def test_encoding_surrogatepass(all_parsers:Fixture[Any]) -> None:
    ...

def test_malformed_second_line(all_parsers:Fixture[Any]) -> None:
    ...

def test_short_single_line(all_parsers:Fixture[Any]) -> None:
    ...

def test_short_multi_line(all_parsers:Fixture[Any]) -> None:
    ...

def test_read_seek(all_parsers:Fixture[Any]) -> None:
    ...