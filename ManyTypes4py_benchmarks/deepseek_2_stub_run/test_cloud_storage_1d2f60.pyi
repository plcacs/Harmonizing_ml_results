```python
import os
from typing import Any, Optional, Union, List, BinaryIO, IO
from pathlib import Path, PurePosixPath
import pandas as pd
from prefect_gcp.cloud_storage import GcsBucket
from prefect_gcp.credentials import GcpCredentials

def test_cloud_storage_create_bucket(gcp_credentials: Any) -> None: ...

def test_cloud_storage_download_blob_to_file(
    tmp_path: Any,
    path: Union[str, Path],
    gcp_credentials: Any
) -> None: ...

def test_cloud_storage_download_blob_as_bytes(gcp_credentials: Any) -> None: ...

def test_cloud_storage_upload_blob_from_file(
    file: Union[str, BinaryIO],
    gcp_credentials: Any
) -> None: ...

def test_cloud_storage_upload_blob_from_string(
    data: Union[str, bytes],
    blob: Optional[str],
    gcp_credentials: Any
) -> None: ...

def test_cloud_storage_copy_blob(
    dest_blob: Optional[str],
    gcp_credentials: Any
) -> None: ...

class TestGcsBucket:
    @pytest.fixture
    def gcs_bucket(self, gcp_credentials: Any, request: Any) -> GcsBucket: ...
    
    def test_bucket_folder_suffix(self, gcs_bucket: GcsBucket) -> None: ...
    
    def test_resolve_path(self, gcs_bucket: GcsBucket, path: str) -> None: ...
    
    def test_read_path(self, gcs_bucket: GcsBucket) -> None: ...
    
    def test_write_path(self, gcs_bucket: GcsBucket) -> None: ...
    
    def test_get_directory(
        self,
        gcs_bucket: GcsBucket,
        tmp_path: Any,
        from_path: Optional[str],
        local_path: Optional[str]
    ) -> None: ...
    
    def test_put_directory(
        self,
        gcs_bucket: GcsBucket,
        tmp_path: Any,
        to_path: Optional[str],
        ignore: bool
    ) -> None: ...
    
    def test_get_bucket(self, gcs_bucket: GcsBucket) -> None: ...
    
    def test_create_bucket(self, gcs_bucket: GcsBucket) -> None: ...
    
    @pytest.fixture
    def gcs_bucket_no_bucket_folder(self, gcp_credentials: Any) -> GcsBucket: ...
    
    @pytest.fixture
    def gcs_bucket_with_bucket_folder(
        self,
        gcp_credentials: Any,
        request: Any
    ) -> GcsBucket: ...
    
    @pytest.fixture
    def pandas_dataframe(self) -> pd.DataFrame: ...
    
    def test_list_folders_root_folder(
        self,
        gcs_bucket_no_bucket_folder: GcsBucket
    ) -> None: ...
    
    def test_list_folders_with_root_only(
        self,
        gcs_bucket_with_bucket_folder: GcsBucket
    ) -> None: ...
    
    def test_list_folders_with_sub_folders(
        self,
        gcs_bucket_with_bucket_folder: GcsBucket
    ) -> None: ...
    
    def test_list_folders_with_dotted_folders(
        self,
        gcs_bucket_no_bucket_folder: GcsBucket
    ) -> None: ...
    
    def test_list_blobs(
        self,
        gcs_bucket_no_bucket_folder: GcsBucket
    ) -> None: ...
    
    def test_list_blobs_with_bucket_folder(
        self,
        gcs_bucket_with_bucket_folder: GcsBucket
    ) -> None: ...
    
    def test_list_blobs_root_folder(
        self,
        gcs_bucket_no_bucket_folder: GcsBucket
    ) -> None: ...
    
    def test_download_object_to_path_default(
        self,
        gcs_bucket_with_bucket_folder: GcsBucket,
        tmp_path: Any
    ) -> None: ...
    
    def test_download_object_to_path_set_to_path(
        self,
        gcs_bucket_with_bucket_folder: GcsBucket,
        tmp_path: Any,
        type_: Any
    ) -> None: ...
    
    def test_download_object_to_file_object_bytesio(
        self,
        gcs_bucket_with_bucket_folder: GcsBucket,
        tmp_path: Any
    ) -> None: ...
    
    def test_download_object_to_file_object_bufferedwriter(
        self,
        gcs_bucket_with_bucket_folder: GcsBucket,
        tmp_path: Any
    ) -> None: ...
    
    def test_download_folder_to_path_default_no_bucket_folder(
        self,
        gcs_bucket_no_bucket_folder: GcsBucket,
        tmp_path: Any
    ) -> None: ...
    
    def test_download_folder_to_path_default_with_bucket_folder(
        self,
        gcs_bucket_with_bucket_folder: GcsBucket,
        tmp_path: Any
    ) -> None: ...
    
    def test_download_folder_to_path_no_bucket_folder(
        self,
        gcs_bucket_no_bucket_folder: GcsBucket,
        tmp_path: Any
    ) -> None: ...
    
    def test_download_folder_to_path_nested(
        self,
        gcs_bucket_no_bucket_folder: GcsBucket,
        tmp_path: Any
    ) -> None: ...
    
    def test_download_folder_to_path_nested_with_bucket_folder(
        self,
        gcs_bucket_with_bucket_folder: GcsBucket,
        tmp_path: Any
    ) -> None: ...
    
    def test_upload_from_path_default_with_bucket_folder(
        self,
        gcs_bucket_with_bucket_folder: GcsBucket,
        tmp_path: Any
    ) -> None: ...
    
    def test_upload_from_path_set_with_bucket_folder(
        self,
        gcs_bucket_with_bucket_folder: GcsBucket,
        tmp_path: Any
    ) -> None: ...
    
    def test_upload_from_file_object_bytesio_with_bucket_folder(
        self,
        gcs_bucket_with_bucket_folder: GcsBucket
    ) -> None: ...
    
    def test_upload_from_file_object_bufferedwriter_with_bucket_folder(
        self,
        gcs_bucket_with_bucket_folder: GcsBucket,
        tmp_path: Any
    ) -> None: ...
    
    def test_upload_from_folder_default_no_bucket_folder(
        self,
        gcs_bucket_no_bucket_folder: GcsBucket,
        tmp_path: Any
    ) -> None: ...
    
    def test_upload_from_folder_default_with_bucket_folder(
        self,
        gcs_bucket_with_bucket_folder: GcsBucket,
        tmp_path: Any
    ) -> None: ...
    
    def test_upload_from_folder_set_to_path_no_bucket_folder(
        self,
        gcs_bucket_no_bucket_folder: GcsBucket,
        tmp_path: Any
    ) -> None: ...
    
    def test_upload_from_folder_set_to_path_with_bucket_folder(
        self,
        gcs_bucket_with_bucket_folder: GcsBucket,
        tmp_path: Any
    ) -> None: ...
    
    def test_upload_from_dataframe_with_default_options(
        self,
        gcs_bucket_with_bucket_folder: GcsBucket,
        pandas_dataframe: pd.DataFrame
    ) -> None: ...
    
    def test_upload_from_dataframe_with_parquet_output(
        self,
        gcs_bucket_with_bucket_folder: GcsBucket,
        pandas_dataframe: pd.DataFrame
    ) -> None: ...
    
    def test_upload_from_dataframe_with_parquet_snappy_output(
        self,
        gcs_bucket_with_bucket_folder: GcsBucket,
        pandas_dataframe: pd.DataFrame
    ) -> None: ...
    
    def test_upload_from_dataframe_with_parquet_gzip_output(
        self,
        gcs_bucket_with_bucket_folder: GcsBucket,
        pandas_dataframe: pd.DataFrame
    ) -> None: ...
    
    def test_upload_from_dataframe_with_csv_output(
        self,
        gcs_bucket_with_bucket_folder: GcsBucket,
        pandas_dataframe: pd.DataFrame
    ) -> None: ...
    
    def test_upload_from_dataframe_with_csv_gzip_output(
        self,
        gcs_bucket_with_bucket_folder: GcsBucket,
        pandas_dataframe: pd.DataFrame
    ) -> None: ...
    
    def test_upload_from_dataframe_with_invalid_serialization_should_raise_key_error(
        self,
        gcs_bucket_with_bucket_folder: GcsBucket,
        pandas_dataframe: pd.DataFrame
    ) -> None: ...
```