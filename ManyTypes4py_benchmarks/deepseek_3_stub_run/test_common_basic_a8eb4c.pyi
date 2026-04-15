import sys
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pytest
from pandas import DataFrame, Index
from pandas.errors import EmptyDataError, ParserError, ParserWarning

pytestmark: Any = ...

xfail_pyarrow: Any = ...
skip_pyarrow: Any = ...


def test_read_csv_local(
    all_parsers: Any, csv1: Union[str, Path]
) -> None: ...


def test_1000_sep(all_parsers: Any) -> None: ...


def test_unnamed_columns(all_parsers: Any) -> None: ...


def test_csv_mixed_type(all_parsers: Any) -> None: ...


def test_read_csv_low_memory_no_rows_with_index(all_parsers: Any) -> None: ...


def test_read_csv_dataframe(
    all_parsers: Any, csv1: Union[str, Path]
) -> None: ...


def test_read_nrows(
    all_parsers: Any, nrows: Union[int, float]
) -> None: ...


def test_read_nrows_bad(
    all_parsers: Any, nrows: Union[str, float, int]
) -> None: ...


def test_nrows_skipfooter_errors(all_parsers: Any) -> None: ...


def test_missing_trailing_delimiters(all_parsers: Any) -> None: ...


def test_skip_initial_space(all_parsers: Any) -> None: ...


def test_trailing_delimiters(all_parsers: Any) -> None: ...


def test_escapechar(all_parsers: Any) -> None: ...


def test_ignore_leading_whitespace(all_parsers: Any) -> None: ...


def test_uneven_lines_with_usecols(
    all_parsers: Any, usecols: Optional[Union[List[int], List[str]]]
) -> None: ...


def test_read_empty_with_usecols(
    all_parsers: Any,
    data: str,
    kwargs: Dict[str, Any],
    expected: Optional[DataFrame],
) -> None: ...


def test_trailing_spaces(
    all_parsers: Any,
    kwargs: Dict[str, Any],
    expected_data: Union[List[List[float]], Dict[str, List[Optional[float]]]],
) -> None: ...


def test_read_filepath_or_buffer(all_parsers: Any) -> None: ...


def test_single_char_leading_whitespace(all_parsers: Any) -> None: ...


def test_empty_lines(
    all_parsers: Any,
    sep: str,
    skip_blank_lines: bool,
    exp_data: List[List[Optional[float]]],
) -> None: ...


def test_whitespace_lines(all_parsers: Any) -> None: ...


def test_whitespace_regex_separator(
    all_parsers: Any, data: str, expected: DataFrame
) -> None: ...


def test_sub_character(
    all_parsers: Any, csv_dir_path: Union[str, Path]
) -> None: ...


def test_filename_with_special_chars(
    all_parsers: Any, filename: str
) -> None: ...


def test_read_table_same_signature_as_read_csv(all_parsers: Any) -> None: ...


def test_read_table_equivalency_to_read_csv(all_parsers: Any) -> None: ...


def test_read_csv_and_table_sys_setprofile(
    all_parsers: Any, read_func: str
) -> None: ...


def test_first_row_bom(all_parsers: Any) -> None: ...


def test_first_row_bom_unquoted(all_parsers: Any) -> None: ...


def test_blank_lines_between_header_and_data_rows(
    all_parsers: Any, nrows: int
) -> None: ...


def test_no_header_two_extra_columns(all_parsers: Any) -> None: ...


def test_read_csv_names_not_accepting_sets(all_parsers: Any) -> None: ...


def test_read_csv_delimiter_and_sep_no_default(all_parsers: Any) -> None: ...


def test_read_csv_line_break_as_separator(
    all_parsers: Any, kwargs: Dict[str, str]
) -> None: ...


def test_dict_keys_as_names(all_parsers: Any) -> None: ...


def test_encoding_surrogatepass(all_parsers: Any) -> None: ...


def test_malformed_second_line(all_parsers: Any) -> None: ...


def test_short_single_line(all_parsers: Any) -> None: ...


def test_short_multi_line(all_parsers: Any) -> None: ...


def test_read_seek(all_parsers: Any) -> None: ...