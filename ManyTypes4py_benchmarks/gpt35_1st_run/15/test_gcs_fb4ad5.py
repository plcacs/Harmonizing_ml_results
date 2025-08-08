from io import BytesIO
import os
import pathlib
import tarfile
import zipfile
from typing import Any
import numpy as np
import pytest
from pandas import DataFrame, Index, date_range, read_csv, read_excel, read_json, read_parquet
import pandas._testing as tm
from pandas.util import _test_decorators as td

@pytest.fixture
def gcs_buffer() -> BytesIO:
    ...

def test_to_read_gcs(gcs_buffer: BytesIO, format: str, monkeypatch: Any, capsys: Any, request: Any) -> None:
    ...

def assert_equal_zip_safe(result: Any, expected: Any, compression: str) -> None:
    ...

def test_to_csv_compression_encoding_gcs(gcs_buffer: BytesIO, compression_only: str, encoding: str, compression_to_extension: dict) -> None:
    ...

def test_to_parquet_gcs_new_file(monkeypatch: Any, tmpdir: Any) -> None:
    ...

@td.skip_if_installed('gcsfs')
def test_gcs_not_present_exception() -> None:
    ...
