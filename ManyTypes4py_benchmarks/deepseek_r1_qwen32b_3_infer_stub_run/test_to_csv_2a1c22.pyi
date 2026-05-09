import io
import os
import sys
from zipfile import ZipFile
from _csv import Error
import numpy as np
import pytest
import pandas as pd
from pandas import DataFrame, Index, compat
from pandas._testing import tm

class TestToCSV:
    def test_to_csv_with_single_column(self) -> None:
        ...

    def test_to_csv_default_encoding(self) -> None:
        ...

    def test_to_csv_quotechar(self) -> None:
        ...

    def test_to_csv_doublequote(self) -> None:
        ...

    def test_to_csv_escapechar(self) -> None:
        ...

    def test_csv_to_string(self) -> None:
        ...

    def test_to_csv_decimal(self) -> None:
        ...

    def test_to_csv_float_format(self) -> None:
        ...

    def test_to_csv_na_rep(self) -> None:
        ...

    def test_to_csv_na_rep_nullable_string(self, nullable_string_dtype: Any) -> None:
        ...

    def test_to_csv_date_format(self) -> None:
        ...

    def test_to_csv_different_datetime_formats(self) -> None:
        ...

    def test_to_csv_date_format_in_categorical(self) -> None:
        ...

    def test_to_csv_float_ea_float_format(self) -> None:
        ...

    def test_to_csv_float_ea_no_float_format(self) -> None:
        ...

    def test_to_csv_multi_index(self) -> None:
        ...

    @pytest.mark.parametrize('ind,expected', [(pd.MultiIndex(levels=[[1.0]], codes=[[0]], names=['x']), 'x,data\n1.0,1\n'), (pd.MultiIndex(levels=[[1.0], [2.0]], codes=[[0], [0]], names=['x', 'y']), 'x,y,data\n1.0,2.0,1\n')])
    def test_to_csv_single_level_multi_index(self, ind: pd.MultiIndex, expected: str, frame_or_series: Any) -> None:
        ...

    def test_to_csv_string_array_ascii(self) -> None:
        ...

    def test_to_csv_string_array_utf8(self) -> None:
        ...

    def test_to_csv_string_with_lf(self) -> None:
        ...

    def test_to_csv_string_with_crlf(self) -> None:
        ...

    def test_to_csv_stdout_file(self, capsys: Any) -> None:
        ...

    @pytest.mark.xfail(compat.is_platform_windows(), reason="Especially in Windows, file stream should not be passed to csv writer without newline='' option.(https://docs.python.org/3/library/csv.html#csv.writer)")
    def test_to_csv_write_to_open_file(self) -> None:
        ...

    def test_to_csv_write_to_open_file_with_newline_py3(self) -> None:
        ...

    @pytest.mark.parametrize('to_infer', [True, False])
    @pytest.mark.parametrize('read_infer', [True, False])
    def test_to_csv_compression(self, compression_only: str, read_infer: bool, to_infer: bool, compression_to_extension: dict) -> None:
        ...

    def test_to_csv_compression_dict(self, compression_only: str) -> None:
        ...

    def test_to_csv_compression_dict_no_method_raises(self) -> None:
        ...

    @pytest.mark.parametrize('compression', ['zip', 'infer'])
    @pytest.mark.parametrize('archive_name', ['test_to_csv.csv', 'test_to_csv.zip'])
    def test_to_csv_zip_arguments(self, compression: str, archive_name: str) -> None:
        ...

    @pytest.mark.parametrize('filename,expected_arcname', [('archive.csv', 'archive.csv'), ('archive.tsv', 'archive.tsv'), ('archive.csv.zip', 'archive.csv'), ('archive.tsv.zip', 'archive.tsv'), ('archive.zip', 'archive')])
    def test_to_csv_zip_infer_name(self, tmp_path: Any, filename: str, expected_arcname: str) -> None:
        ...

    @pytest.mark.parametrize('df_new_type', ['Int64'])
    def test_to_csv_na_rep_long_string(self, df_new_type: str) -> None:
        ...

    def test_to_csv_timedelta_precision(self) -> None:
        ...

    def test_na_rep_truncated(self) -> None:
        ...

    @pytest.mark.parametrize('errors', ['surrogatepass', 'ignore', 'replace'])
    def test_to_csv_errors(self, errors: str) -> None:
        ...

    @pytest.mark.parametrize('mode', ['wb', 'w'])
    def test_to_csv_binary_handle(self, mode: str) -> None:
        ...

    @pytest.mark.parametrize('mode', ['wb', 'w'])
    def test_to_csv_encoding_binary_handle(self, mode: str) -> None:
        ...

def test_to_csv_iterative_compression_name(compression: str = 'infer') -> None:
    ...

def test_to_csv_iterative_compression_buffer(compression: str = 'infer') -> None:
    ...