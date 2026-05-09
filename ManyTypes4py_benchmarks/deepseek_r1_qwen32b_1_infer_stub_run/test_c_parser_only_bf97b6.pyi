"""
Stub file for test_c_parser_only_bf97b6 module
"""

from decimal import Decimal
from io import BytesIO, StringIO, TextIOWrapper
import mmap
import os
import tarfile
import numpy as np
import pytest
from pandas import DataFrame
from pandas.compat import WASM
from pandas.errors import ParserError, ParserWarning
from pandas._testing import tm

CParserWrapper = ...  # Type alias for the parser wrapper

@pytest.mark.parametrize('malformed', ['1\r1\r1\r 1\r 1\r', '1\r1\r1\r 1\r 1\r11\r', '1\r1\r1\r 1\r 1\r11\r1\r'], ids=['words pointer', 'stream pointer', 'lines pointer'])
def test_buffer_overflow(c_parser_only: CParserWrapper, malformed: str) -> None:
    ...

def test_delim_whitespace_custom_terminator(c_parser_only: CParserWrapper) -> DataFrame:
    ...

def test_dtype_and_names_error(c_parser_only: CParserWrapper) -> DataFrame:
    ...

@pytest.mark.parametrize('match,kwargs', [('the dtype datetime64 is not supported for parsing, pass this column using parse_dates instead', {'dtype': {'A': 'datetime64', 'B': 'float64'}}), ('the dtype datetime64 is not supported for parsing, pass this column using parse_dates instead', {'dtype': {'A': 'datetime64', 'B': 'float64'}, 'parse_dates': ['B']}), ('the dtype timedelta64 is not supported for parsing', {'dtype': {'A': 'timedelta64', 'B': 'float64'}}), (f'the dtype {tm.ENDIAN}U8 is not supported for parsing', {'dtype': {'A': 'U8'}})], ids=['dt64-0', 'dt64-1', 'td64', f'{tm.ENDIAN}U8'])
def test_unsupported_dtype(c_parser_only: CParserWrapper, match: str, kwargs: dict) -> None:
    ...

@td.skip_if_32bit
@pytest.mark.slow
@pytest.mark.parametrize('num', np.linspace(1.0, 2.0, num=21))
def test_precise_conversion(c_parser_only: CParserWrapper, num: float) -> None:
    ...

def test_usecols_dtypes(c_parser_only: CParserWrapper, using_infer_string: bool) -> None:
    ...

def test_disable_bool_parsing(c_parser_only: CParserWrapper) -> None:
    ...

def test_custom_lineterminator(c_parser_only: CParserWrapper) -> None:
    ...

def test_parse_ragged_csv(c_parser_only: CParserWrapper) -> None:
    ...

def test_tokenize_CR_with_quoting(c_parser_only: CParserWrapper) -> None:
    ...

@pytest.mark.slow
@pytest.mark.parametrize('count', [3 * 2 ** n for n in range(6)])
def test_grow_boundary_at_cap(c_parser_only: CParserWrapper, count: int) -> None:
    ...

@pytest.mark.slow
@pytest.mark.parametrize('encoding', [None, 'utf-8'])
def test_parse_trim_buffers(c_parser_only: CParserWrapper, encoding: str | None) -> None:
    ...

def test_internal_null_byte(c_parser_only: CParserWrapper) -> None:
    ...

def test_read_nrows_large(c_parser_only: CParserWrapper) -> None:
    ...

def test_float_precision_round_trip_with_text(c_parser_only: CParserWrapper) -> None:
    ...

def test_large_difference_in_columns(c_parser_only: CParserWrapper) -> None:
    ...

def test_data_after_quote(c_parser_only: CParserWrapper) -> None:
    ...

def test_comment_whitespace_delimited(c_parser_only: CParserWrapper) -> None:
    ...

def test_file_like_no_next(c_parser_only: CParserWrapper) -> None:
    ...

def test_buffer_rd_bytes_bad_unicode(c_parser_only: CParserWrapper) -> None:
    ...

@pytest.mark.parametrize('tar_suffix', ['.tar', '.tar.gz'])
def test_read_tarfile(c_parser_only: CParserWrapper, csv_dir_path: str, tar_suffix: str) -> None:
    ...

def test_chunk_whitespace_on_boundary(c_parser_only: CParserWrapper) -> None:
    ...

@pytest.mark.skipif(WASM, reason='limited file system access on WASM')
def test_file_handles_mmap(c_parser_only: CParserWrapper, csv1: str) -> None:
    ...

def test_file_binary_mode(c_parser_only: CParserWrapper) -> None:
    ...

def test_unix_style_breaks(c_parser_only: CParserWrapper) -> None:
    ...

@pytest.mark.parametrize('float_precision', [None, 'legacy', 'high', 'round_trip'])
@pytest.mark.parametrize('data,thousands,decimal', [('A|B|C\n1|2,334.01|5\n10|13|10.\n', ',', '.'), ('A|B|C\n1|2.334,01|5\n10|13|10,\n', '.', ',')])
def test_1000_sep_with_decimal(c_parser_only: CParserWrapper, data: str, thousands: str, decimal: str, float_precision: str | None) -> None:
    ...

def test_float_precision_options(c_parser_only: CParserWrapper) -> None:
    ...