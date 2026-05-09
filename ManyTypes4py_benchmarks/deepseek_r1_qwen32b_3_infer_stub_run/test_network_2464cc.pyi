"""
Tests parsers ability to read and parse non-local files
and hence require a network connection to be read.
"""

from io import BytesIO
import logging
import re
import numpy as np
import pytest
from pandas import DataFrame
from pandas.io.feather_format import read_feather
from pandas.io.parsers import read_csv

pytestmark: pytest.MarkDecorator = pytest.mark.filterwarnings('ignore:Passing a BlockManager to DataFrame:DeprecationWarning')

@pytest.mark.network
@pytest.mark.single_cpu
@pytest.mark.parametrize('mode', ['explicit', 'infer'])
@pytest.mark.parametrize('engine', ['python', 'c'])
def test_compressed_urls(httpserver, datapath, salaries_table: DataFrame, mode: str, engine: str, compression_only: str, compression_to_extension: dict[str, str]) -> None:
    ...

@pytest.mark.network
@pytest.mark.single_cpu
def test_url_encoding_csv(httpserver, datapath) -> None:
    ...

@pytest.fixture
def tips_df(datapath) -> DataFrame:
    ...

@pytest.mark.single_cpu
@pytest.mark.network
@pytest.mark.usefixtures('s3_resource')
@td.skip_if_not_us_locale()
class TestS3:
    def test_parse_public_s3_bucket(self, s3_public_bucket_with_data, tips_df: DataFrame, s3so: dict) -> None:
        ...

    def test_parse_private_s3_bucket(self, s3_private_bucket_with_data, tips_df: DataFrame, s3so: dict) -> None:
        ...

    def test_parse_public_s3n_bucket(self, s3_public_bucket_with_data, tips_df: DataFrame, s3so: dict) -> None:
        ...

    def test_parse_public_s3a_bucket(self, s3_public_bucket_with_data, tips_df: DataFrame, s3so: dict) -> None:
        ...

    def test_parse_public_s3_bucket_nrows(self, s3_public_bucket_with_data, tips_df: DataFrame, s3so: dict) -> None:
        ...

    def test_parse_public_s3_bucket_chunked(self, s3_public_bucket_with_data, tips_df: DataFrame, s3so: dict) -> None:
        ...

    def test_parse_public_s3_bucket_chunked_python(self, s3_public_bucket_with_data, tips_df: DataFrame, s3so: dict) -> None:
        ...

    def test_parse_public_s3_bucket_python(self, s3_public_bucket_with_data, tips_df: DataFrame, s3so: dict) -> None:
        ...

    def test_infer_s3_compression(self, s3_public_bucket_with_data, tips_df: DataFrame, s3so: dict) -> None:
        ...

    def test_parse_public_s3_bucket_nrows_python(self, s3_public_bucket_with_data, tips_df: DataFrame, s3so: dict) -> None:
        ...

    def test_read_s3_fails(self, s3so: dict) -> None:
        ...

    def test_read_s3_fails_private(self, s3_private_bucket, s3so: dict) -> None:
        ...

    @pytest.mark.xfail(reason='GH#39155 s3fs upgrade', strict=False)
    def test_write_s3_csv_fails(self, tips_df: DataFrame, s3so: dict) -> None:
        ...

    @pytest.mark.xfail(reason='GH#39155 s3fs upgrade', strict=False)
    def test_write_s3_parquet_fails(self, tips_df: DataFrame, s3so: dict) -> None:
        ...

    @pytest.mark.single_cpu
    def test_read_csv_handles_boto_s3_object(self, s3_public_bucket_with_data, tips_file) -> None:
        ...

    @pytest.mark.single_cpu
    def test_read_csv_chunked_download(self, s3_public_bucket, caplog, s3so: dict) -> None:
        ...

    def test_read_s3_with_hash_in_key(self, s3_public_bucket_with_data, tips_df: DataFrame, s3so: dict) -> None:
        ...

    def test_read_feather_s3_file_path(self, s3_public_bucket_with_data, feather_file, s3so: dict) -> None:
        ...