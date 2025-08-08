from decimal import Decimal
from io import BytesIO, StringIO
import mmap
import os
import tarfile
from typing import Any
import numpy as np
import pytest
from pandas import DataFrame, concat
from pandas.errors import ParserError, ParserWarning
import pandas.util._test_decorators as td
import pandas._testing as tm

def test_buffer_overflow(c_parser_only: Any, malformed: str) -> None:
    ...

def test_delim_whitespace_custom_terminator(c_parser_only: Any) -> None:
    ...

def test_dtype_and_names_error(c_parser_only: Any) -> None:
    ...

def test_unsupported_dtype(c_parser_only: Any, match: str, kwargs: dict) -> None:
    ...

def test_precise_conversion(c_parser_only: Any, num: float) -> None:
    ...

def test_usecols_dtypes(c_parser_only: Any, using_infer_string: bool) -> None:
    ...

def test_disable_bool_parsing(c_parser_only: Any) -> None:
    ...

def test_custom_lineterminator(c_parser_only: Any) -> None:
    ...

def test_parse_ragged_csv(c_parser_only: Any) -> None:
    ...

def test_tokenize_CR_with_quoting(c_parser_only: Any) -> None:
    ...

def test_grow_boundary_at_cap(c_parser_only: Any, count: int) -> None:
    ...

def test_parse_trim_buffers(c_parser_only: Any, encoding: str) -> None:
    ...

def test_internal_null_byte(c_parser_only: Any) -> None:
    ...

def test_read_nrows_large(c_parser_only: Any) -> None:
    ...

def test_float_precision_round_trip_with_text(c_parser_only: Any) -> None:
    ...

def test_large_difference_in_columns(c_parser_only: Any) -> None:
    ...

def test_data_after_quote(c_parser_only: Any) -> None:
    ...

def test_comment_whitespace_delimited(c_parser_only: Any) -> None:
    ...

def test_file_like_no_next(c_parser_only: Any) -> None:
    ...

def test_buffer_rd_bytes_bad_unicode(c_parser_only: Any) -> None:
    ...

def test_read_tarfile(c_parser_only: Any, csv_dir_path: str, tar_suffix: str) -> None:
    ...

def test_chunk_whitespace_on_boundary(c_parser_only: Any) -> None:
    ...

def test_file_handles_mmap(c_parser_only: Any, csv1: str) -> None:
    ...

def test_file_binary_mode(c_parser_only: Any) -> None:
    ...

def test_unix_style_breaks(c_parser_only: Any) -> None:
    ...

def test_1000_sep_with_decimal(c_parser_only: Any, data: str, thousands: str, decimal: str, float_precision: str) -> None:
    ...

def test_float_precision_options(c_parser_only: Any) -> None:
    ...
