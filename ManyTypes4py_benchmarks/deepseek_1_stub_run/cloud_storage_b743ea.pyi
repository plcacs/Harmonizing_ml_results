```python
from typing import Any, Optional, Union, List, Dict, Tuple, BinaryIO
from enum import Enum
from pathlib import Path, PurePosixPath
from pydantic import Field
from prefect.blocks.abstract import ObjectStorageBlock
from prefect.filesystems import WritableDeploymentStorage, WritableFileSystem
from prefect_gcp.credentials import GcpCredentials

try:
    from pandas import DataFrame
except ModuleNotFoundError:
    DataFrame = Any

try:
    from google.cloud.storage import Bucket
    from google.cloud.storage.blob import Blob
except ModuleNotFoundError:
    Bucket = Any
    Blob = Any

def cloud_storage_create_bucket(
    bucket: Any,
    gcp_credentials: Any,
    project: Any = ...,
    location: Any = ...,
    **create_kwargs: Any
) -> Any: ...

def cloud_storage_download_blob_as_bytes(
    bucket: Any,
    blob: Any,
    gcp_credentials: Any,
    chunk_size: Any = ...,
    encryption_key: Any = ...,
    timeout: Any = ...,
    project: Any = ...,
    **download_kwargs: Any
) -> Any: ...

def cloud_storage_download_blob_to_file(
    bucket: Any,
    blob: Any,
    path: Any,
    gcp_credentials: Any,
    chunk_size: Any = ...,
    encryption_key: Any = ...,
    timeout: Any = ...,
    project: Any = ...,
    **download_kwargs: Any
) -> Any: ...

def cloud_storage_upload_blob_from_string(
    data: Any,
    bucket: Any,
    blob: Any,
    gcp_credentials: Any,
    content_type: Any = ...,
    chunk_size: Any = ...,
    encryption_key: Any = ...,
    timeout: Any = ...,
    project: Any = ...,
    **upload_kwargs: Any
) -> Any: ...

def cloud_storage_upload_blob_from_file(
    file: Any,
    bucket: Any,
    blob: Any,
    gcp_credentials: Any,
    content_type: Any = ...,
    chunk_size: Any = ...,
    encryption_key: Any = ...,
    timeout: Any = ...,
    project: Any = ...,
    **upload_kwargs: Any
) -> Any: ...

def cloud_storage_copy_blob(
    source_bucket: Any,
    dest_bucket: Any,
    source_blob: Any,
    gcp_credentials: Any,
    dest_blob: Any = ...,
    timeout: Any = ...,
    project: Any = ...,
    **copy_kwargs: Any
) -> Any: ...

class DataFrameSerializationFormat(Enum):
    CSV: Any = ...
    CSV_GZIP: Any = ...
    PARQUET: Any = ...
    PARQUET_SNAPPY: Any = ...
    PARQUET_GZIP: Any = ...
    
    @property
    def format(self) -> Any: ...
    
    @property
    def compression(self) -> Any: ...
    
    @property
    def content_type(self) -> Any: ...
    
    @property
    def suffix(self) -> Any: ...
    
    def fix_extension_with(self, gcs_blob_path: Any) -> Any: ...

class GcsBucket(WritableDeploymentStorage, WritableFileSystem, ObjectStorageBlock):
    bucket: Any = ...
    gcp_credentials: Any = ...
    bucket_folder: Any = ...
    
    @property
    def basepath(self) -> Any: ...
    
    @classmethod
    def _bucket_folder_suffix(cls, value: Any) -> Any: ...
    
    def _resolve_path(self, path: Any) -> Any: ...
    
    def get_directory(
        self,
        from_path: Any = ...,
        local_path: Any = ...
    ) -> Any: ...
    
    def put_directory(
        self,
        local_path: Any = ...,
        to_path: Any = ...,
        ignore_file: Any = ...
    ) -> Any: ...
    
    def read_path(self, path: Any) -> Any: ...
    
    def write_path(self, path: Any, content: Any) -> Any: ...
    
    def _join_bucket_folder(self, bucket_path: Any = ...) -> Any: ...
    
    def create_bucket(
        self,
        location: Any = ...,
        **create_kwargs: Any
    ) -> Any: ...
    
    def get_bucket(self) -> Any: ...
    
    def list_blobs(self, folder: Any = ...) -> Any: ...
    
    def list_folders(self, folder: Any = ...) -> Any: ...
    
    def download_object_to_path(
        self,
        from_path: Any,
        to_path: Any = ...,
        **download_kwargs: Any
    ) -> Any: ...
    
    def download_object_to_file_object(
        self,
        from_path: Any,
        to_file_object: Any,
        **download_kwargs: Any
    ) -> Any: ...
    
    def download_folder_to_path(
        self,
        from_folder: Any,
        to_folder: Any = ...,
        **download_kwargs: Any
    ) -> Any: ...
    
    def upload_from_path(
        self,
        from_path: Any,
        to_path: Any = ...,
        **upload_kwargs: Any
    ) -> Any: ...
    
    def upload_from_file_object(
        self,
        from_file_object: Any,
        to_path: Any,
        **upload_kwargs: Any
    ) -> Any: ...
    
    def upload_from_folder(
        self,
        from_folder: Any,
        to_folder: Any = ...,
        **upload_kwargs: Any
    ) -> Any: ...
    
    def upload_from_dataframe(
        self,
        df: Any,
        to_path: Any,
        serialization_format: Any = ...,
        **upload_kwargs: Any
    ) -> Any: ...
```