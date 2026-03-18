```python
import pytest
from pandas import DataFrame
from typing import Any, Iterator, Union
from _pytest.fixtures import FixtureRequest

def test_compressed_urls(
    httpserver: Any,
    datapath: Any,
    salaries_table: Any,
    mode: str,
    engine: str,
    compression_only: str,
    compression_to_extension: Any
) -> None: ...

def test_url_encoding_csv(httpserver: Any, datapath: Any) -> None: ...

@pytest.fixture
def tips_df(datapath: Any) -> DataFrame: ...

class TestS3:
    def test_parse_public_s3_bucket(
        self,
        s3_public_bucket_with_data: Any,
        tips_df: DataFrame,
        s3so: Any
    ) -> None: ...

    def test_parse_private_s3_bucket(
        self,
        s3_private_bucket_with_data: Any,
        tips_df: DataFrame,
        s3so: Any
    ) -> None: ...

    def test_parse_public_s3n_bucket(
        self,
        s3_public_bucket_with_data: Any,
        tips_df: DataFrame,
        s3so: Any
    ) -> None: ...

    def test_parse_public_s3a_bucket(
        self,
        s3_public_bucket_with_data: Any,
        tips_df: DataFrame,
        s3so: Any
    ) -> None: ...

    def test_parse_public_s3_bucket_nrows(
        self,
        s3_public_bucket_with_data: Any,
        tips_df: DataFrame,
        s3so: Any
    ) -> None: ...

    def test_parse_public_s3_bucket_chunked(
        self,
        s3_public_bucket_with_data: Any,
        tips_df: DataFrame,
        s3so: Any
    ) -> None: ...

    def test_parse_public_s3_bucket_chunked_python(
        self,
        s3_public_bucket_with_data: Any,
        tips_df: DataFrame,
        s3so: Any
    ) -> None: ...

    def test_parse_public_s3_bucket_python(
        self,
        s3_public_bucket_with_data: Any,
        tips_df: DataFrame,
        s3so: Any
    ) -> None: ...

    def test_infer_s3_compression(
        self,
        s3_public_bucket_with_data: Any,
        tips_df: DataFrame,
        s3so: Any
    ) -> None: ...

    def test_parse_public_s3_bucket_nrows_python(
        self,
        s3_public_bucket_with_data: Any,
        tips_df: DataFrame,
        s3so: Any
    ) -> None: ...

    def test_read_s3_fails(self, s3so: Any) -> None: ...

    def test_read_s3_fails_private(
        self,
        s3_private_bucket: Any,
        s3so: Any
    ) -> None: ...

    def test_write_s3_csv_fails(
        self,
        tips_df: DataFrame,
        s3so: Any
    ) -> None: ...

    def test_write_s3_parquet_fails(
        self,
        tips_df: DataFrame,
        s3so: Any
    ) -> None: ...

    def test_read_csv_handles_boto_s3_object(
        self,
        s3_public_bucket_with_data: Any,
        tips_file: Any
    ) -> None: ...

    def test_read_csv_chunked_download(
        self,
        s3_public_bucket: Any,
        caplog: Any,
        s3so: Any
    ) -> None: ...

    def test_read_s3_with_hash_in_key(
        self,
        s3_public_bucket_with_data: Any,
        tips_df: DataFrame,
        s3so: Any
    ) -> None: ...

    def test_read_feather_s3_file_path(
        self,
        s3_public_bucket_with_data: Any,
        feather_file: Any,
        s3so: Any
    ) -> None: ...

pytestmark: Any = ...
```