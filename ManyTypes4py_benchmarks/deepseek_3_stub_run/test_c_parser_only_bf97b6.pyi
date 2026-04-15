import decimal
import io
import mmap
import os
import tarfile
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
import pytest
from pandas import DataFrame
from pandas._testing import TestCase

@pytest.mark.parametrize("malformed", ...)
def test_buffer_overflow(
    c_parser_only: Any,
    malformed: str,
) -> None: ...

def test_delim_whitespace_custom_terminator(
    c_parser_only: Any,
) -> None: ...

def test_dtype_and_names_error(
    c_parser_only: Any,
) -> None: ...

@pytest.mark.parametrize("match,kwargs", ...)
def test_unsupported_dtype(
    c_parser_only: Any,
    match: str,
    kwargs: dict[str, Any],
) -> None: ...

@td.skip_if_32bit
@pytest.mark.slow
@pytest.mark.parametrize("num", ...)
def test_precise_conversion(
    c_parser_only: Any,
    num: float,
) -> None: ...

def test_usecols_dtypes(
    c_parser_only: Any,
    using_infer_string: bool,
) -> None: ...

def test_disable_bool_parsing(
    c_parser_only: Any,
) -> None: ...

def test_custom_lineterminator(
    c_parser_only: Any,
) -> None: ...

def test_parse_ragged_csv(
    c_parser_only: Any,
) -> None: ...

def test_tokenize_CR_with_quoting(
    c_parser_only: Any,
) -> None: ...

@pytest.mark.slow
@pytest.mark.parametrize("count", ...)
def test_grow_boundary_at_cap(
    c_parser_only: Any,
    count: int,
) -> None: ...

@pytest.mark.slow
@pytest.mark.parametrize("encoding", ...)
def test_parse_trim_buffers(
    c_parser_only: Any,
    encoding: Optional[str],
) -> None: ...

def test_internal_null_byte(
    c_parser_only: Any,
) -> None: ...

def test_read_nrows_large(
    c_parser_only: Any,
) -> None: ...

def test_float_precision_round_trip_with_text(
    c_parser_only: Any,
) -> None: ...

def test_large_difference_in_columns(
    c_parser_only: Any,
) -> None: ...

def test_data_after_quote(
    c_parser_only: Any,
) -> None: ...

def test_comment_whitespace_delimited(
    c_parser_only: Any,
) -> None: ...

def test_file_like_no_next(
    c_parser_only: Any,
) -> None: ...

def test_buffer_rd_bytes_bad_unicode(
    c_parser_only: Any,
) -> None: ...

@pytest.mark.parametrize("tar_suffix", ...)
def test_read_tarfile(
    c_parser_only: Any,
    csv_dir_path: str,
    tar_suffix: str,
) -> None: ...

def test_chunk_whitespace_on_boundary(
    c_parser_only: Any,
) -> None: ...

@pytest.mark.skipif(WASM, reason="limited file system access on WASM")
def test_file_handles_mmap(
    c_parser_only: Any,
    csv1: str,
) -> None: ...

def test_file_binary_mode(
    c_parser_only: Any,
) -> None: ...

def test_unix_style_breaks(
    c_parser_only: Any,
) -> None: ...

@pytest.mark.parametrize("float_precision", ...)
@pytest.mark.parametrize("data,thousands,decimal", ...)
def test_1000_sep_with_decimal(
    c_parser_only: Any,
    data: str,
    thousands: str,
    decimal: str,
    float_precision: Optional[str],
) -> None: ...

def test_float_precision_options(
    c_parser_only: Any,
) -> None: ...