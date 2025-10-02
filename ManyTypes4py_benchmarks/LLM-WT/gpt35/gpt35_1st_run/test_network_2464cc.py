from io import BytesIO
import logging
import re
import numpy as np
import pytest
from pandas import DataFrame
from pandas.io.feather_format import read_feather
from pandas.io.parsers import read_csv
from pandas._testing import assert_frame_equal
from pandas.util._test_decorators import skip_if_not_us_locale
from typing import Any

def test_compressed_urls(httpserver: Any, datapath: Any, salaries_table: DataFrame, mode: str, engine: str, compression_only: str, compression_to_extension: dict) -> None:
    ...

def test_url_encoding_csv(httpserver: Any, datapath: Any) -> None:
    ...

def test_parse_public_s3_bucket(self, s3_public_bucket_with_data: Any, tips_df: DataFrame, s3so: Any) -> None:
    ...

def test_parse_private_s3_bucket(self, s3_private_bucket_with_data: Any, tips_df: DataFrame, s3so: Any) -> None:
    ...

def test_parse_public_s3_bucket_nrows(self, s3_public_bucket_with_data: Any, tips_df: DataFrame, s3so: Any) -> None:
    ...

def test_parse_public_s3_bucket_chunked(self, s3_public_bucket_with_data: Any, tips_df: DataFrame, s3so: Any) -> None:
    ...

def test_parse_public_s3_bucket_chunked_python(self, s3_public_bucket_with_data: Any, tips_df: DataFrame, s3so: Any) -> None:
    ...

def test_parse_public_s3_bucket_python(self, s3_public_bucket_with_data: Any, tips_df: DataFrame, s3so: Any) -> None:
    ...

def test_infer_s3_compression(self, s3_public_bucket_with_data: Any, tips_df: DataFrame, s3so: Any) -> None:
    ...

def test_parse_public_s3_bucket_nrows_python(self, s3_public_bucket_with_data: Any, tips_df: DataFrame, s3so: Any) -> None:
    ...

def test_read_s3_fails(self, s3so: Any) -> None:
    ...

def test_read_s3_fails_private(self, s3_private_bucket: Any, s3so: Any) -> None:
    ...

def test_write_s3_csv_fails(self, tips_df: DataFrame, s3so: Any) -> None:
    ...

def test_write_s3_parquet_fails(self, tips_df: DataFrame, s3so: Any) -> None:
    ...

def test_read_csv_handles_boto_s3_object(self, s3_public_bucket_with_data: Any, tips_file: Any) -> None:
    ...

def test_read_csv_chunked_download(self, s3_public_bucket: Any, caplog: Any, s3so: Any) -> None:
    ...

def test_read_s3_with_hash_in_key(self, s3_public_bucket_with_data: Any, tips_df: DataFrame, s3so: Any) -> None:
    ...

def test_read_feather_s3_file_path(self, s3_public_bucket_with_data: Any, feather_file: Any, s3so: Any) -> None:
    ...
