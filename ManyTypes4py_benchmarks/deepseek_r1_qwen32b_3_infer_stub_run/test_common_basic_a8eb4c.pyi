"""
Stub file for 'test_common_basic_a8eb4c.py'
"""

from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)
import pytest
from pandas import DataFrame, Index
import pandas._testing as tm

pytestmark = pytest.mark.filterwarnings('ignore:Passing a BlockManager to DataFrame:DeprecationWarning')
xfail_pyarrow = pytest.mark.usefixtures('pyarrow_xfail')
skip_pyarrow = pytest.mark.usefixtures('pyarrow_skip')

def test_read_csv_local(all_parsers: Any, csv1: Union[str, Path]) -> None:
    ...

def test_1000_sep(all_parsers: Any) -> None:
    ...

def test_unnamed_columns(all_parsers: Any) -> None:
    ...

def test_csv_mixed_type(all_parsers: Any) -> None:
    ...

def test_read_csv_dataframe(all_parsers: Any, csv1: Union[str, Path]) -> None:
    ...

@pytest.mark.parametrize('nrows', [3, 3.0])
def test_read_nrows(all_parsers: Any, nrows: Union[int, float]) -> None:
    ...

@pytest.mark.parametrize('nrows', [1.2, 'foo', -1])
def test_read_nrows_bad(all_parsers: Any, nrows: Union[int, float, str]) -> None:
    ...

def test_nrows_skipfooter_errors(all_parsers: Any) -> None:
    ...

@skip_pyarrow
def test_missing_trailing_delimiters(all_parsers: Any) -> None:
    ...

def test_skip_initial_space(all_parsers: Any) -> None:
    ...

@skip_pyarrow
def test_trailing_delimiters(all_parsers: Any) -> None:
    ...

def test_escapechar(all_parsers: Any) -> None:
    ...

def test_ignore_leading_whitespace(all_parsers: Any) -> None:
    ...

@skip_pyarrow
@pytest.mark.parametrize('usecols', [None, [0, 1], ['a', 'b']])
def test_uneven_lines_with_usecols(all_parsers: Any, usecols: Optional[List[Union[int, str]]]) -> None:
    ...

@pytest.mark.parametrize('data,kwargs,expected', [('', {}, None), ('', {'usecols': ['X']}, None), (',,', {'names': ['Dummy', 'X', 'Dummy_2'], 'usecols': ['X']}, DataFrame(columns=['X'], index=[0], dtype=np.float64)), ('', {'names': ['Dummy', 'X', 'Dummy_2'], 'usecols': ['X']}, DataFrame(columns=['X']))])
def test_read_empty_with_usecols(all_parsers: Any, data: str, kwargs: Dict[str, Any], expected: Optional[DataFrame]) -> None:
    ...

@pytest.mark.parametrize('kwargs,expected_data', [({'header': None, 'sep': '\\s+', 'skiprows': [0, 1, 2, 3, 5, 6], 'skip_blank_lines': True}, [[1.0, 2.0, 4.0], [5.1, np.nan, 10.0]]), ({'sep': '\\s+', 'skiprows': [1, 2, 3, 5, 6], 'skip_blank_lines': True}, {'A': [1.0, 5.1], 'B': [2.0, np.nan], 'C': [4.0, 10]})])
def test_trailing_spaces(all_parsers: Any, kwargs: Dict[str, Any], expected_data: Union[List[List[float]], Dict[str, List[float]]]) -> None:
    ...

def test_read_filepath_or_buffer(all_parsers: Any) -> None:
    ...

def test_single_char_leading_whitespace(all_parsers: Any) -> None:
    ...

@pytest.mark.parametrize('sep,skip_blank_lines,exp_data', [(',', True, [[1.0, 2.0, 4.0], [5.0, np.nan, 10.0], [-70.0, 0.4, 1.0]]), ('\\s+', True, [[1.0, 2.0, 4.0], [5.0, np.nan, 10.0], [-70.0, 0.4, 1.0]]), (',', False, [[1.0, 2.0, 4.0], [np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan], [5.0, np.nan, 10.0], [np.nan, np.nan, np.nan], [-70.0, 0.4, 1.0]])])
def test_empty_lines(all_parsers: Any, sep: str, skip_blank_lines: bool, exp_data: List[List[Union[float, np.nan]]]) -> None:
    ...

@skip_pyarrow
def test_whitespace_lines(all_parsers: Any) -> None:
    ...

@skip_pyarrow
def test_trailing_delimiters(all_parsers: Any) -> None:
    ...

@skip_pyarrow
def test_whitespace_regex_separator(all_parsers: Any, data: str, expected: DataFrame) -> None:
    ...

def test_sub_character(all_parsers: Any, csv_dir_path: str) -> None:
    ...

@pytest.mark.parametrize('filename', ['sé-es-vé.csv', 'ru-sй.csv', '中文文件名.csv'])
def test_filename_with_special_chars(all_parsers: Any, filename: str) -> None:
    ...

def test_read_table_same_signature_as_read_csv(all_parsers: Any) -> None:
    ...

def test_read_table_equivalency_to_read_csv(all_parsers: Any) -> None:
    ...

@pytest.mark.parametrize('read_func', ['read_csv', 'read_table'])
def test_read_csv_and_table_sys_setprofile(all_parsers: Any, read_func: str) -> None:
    ...

@skip_pyarrow
def test_first_row_bom(all_parsers: Any) -> None:
    ...

@skip_pyarrow
def test_first_row_bom_unquoted(all_parsers: Any) -> None:
    ...

@pytest.mark.parametrize('nrows', range(1, 6))
def test_blank_lines_between_header_and_data_rows(all_parsers: Any, nrows: int) -> None:
    ...

@skip_pyarrow
def test_no_header_two_extra_columns(all_parsers: Any) -> None:
    ...

def test_read_csv_names_not_accepting_sets(all_parsers: Any) -> None:
    ...

def test_read_csv_delimiter_and_sep_no_default(all_parsers: Any) -> None:
    ...

@pytest.mark.parametrize('kwargs', [{'delimiter': '\n'}, {'sep': '\n'}])
def test_read_csv_line_break_as_separator(kwargs: Dict[str, str], all_parsers: Any) -> None:
    ...

@skip_pyarrow
def test_dict_keys_as_names(all_parsers: Any) -> None:
    ...

@pytest.mark.xfail(using_string_dtype() and HAS_PYARROW, reason='TODO(infer_string)')
@xfail_pyarrow
def test_encoding_surrogatepass(all_parsers: Any) -> None:
    ...

def test_malformed_second_line(all_parsers: Any) -> None:
    ...

@skip_pyarrow
def test_short_single_line(all_parsers: Any) -> None:
    ...

@xfail_pyarrow
def test_short_multi_line(all_parsers: Any) -> None:
    ...

def test_read_seek(all_parsers: Any) -> None:
    ...