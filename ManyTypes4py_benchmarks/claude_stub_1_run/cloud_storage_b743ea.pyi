```python
import asyncio
from enum import Enum
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Dict, List, Optional, Tuple, Union

from pydantic import Field, field_validator
from prefect.blocks.abstract import ObjectStorageBlock
from prefect.filesystems import WritableDeploymentStorage, WritableFileSystem
from prefect_gcp.credentials import GcpCredentials

try:
    from pandas import DataFrame
except ModuleNotFoundError:
    DataFrame: Any

try:
    from google.cloud.storage import Bucket
    from google.cloud.storage.blob import Blob
except ModuleNotFoundError:
    Bucket: Any
    Blob: Any

async def cloud_storage_create_bucket(
    bucket: str,
    gcp_credentials: GcpCredentials,
    project: Optional[str] = None,
    location: Optional[str] = None,
    **create_kwargs: Any
) -> str: ...

async def _get_bucket_async(
    bucket: str,
    gcp_credentials: GcpCredentials,
    project: Optional[str] = None
) -> Any: ...

def _get_bucket(
    bucket: str,
    gcp_credentials: GcpCredentials,
    project: Optional[str] = None
) -> Any: ...

async def cloud_storage_download_blob_as_bytes(
    bucket: str,
    blob: str,
    gcp_credentials: GcpCredentials,
    chunk_size: Optional[int] = None,
    encryption_key: Optional[Any] = None,
    timeout: Union[int, Tuple[int, int]] = 60,
    project: Optional[str] = None,
    **download_kwargs: Any
) -> Union[bytes, str]: ...

async def cloud_storage_download_blob_to_file(
    bucket: str,
    blob: str,
    path: Union[str, Path],
    gcp_credentials: GcpCredentials,
    chunk_size: Optional[int] = None,
    encryption_key: Optional[Any] = None,
    timeout: Union[int, Tuple[int, int]] = 60,
    project: Optional[str] = None,
    **download_kwargs: Any
) -> Union[str, Path]: ...

async def cloud_storage_upload_blob_from_string(
    data: Union[str, bytes],
    bucket: str,
    blob: str,
    gcp_credentials: GcpCredentials,
    content_type: Optional[str] = None,
    chunk_size: Optional[int] = None,
    encryption_key: Optional[Any] = None,
    timeout: Union[int, Tuple[int, int]] = 60,
    project: Optional[str] = None,
    **upload_kwargs: Any
) -> str: ...

async def cloud_storage_upload_blob_from_file(
    file: Union[str, Path, BytesIO],
    bucket: str,
    blob: str,
    gcp_credentials: GcpCredentials,
    content_type: Optional[str] = None,
    chunk_size: Optional[int] = None,
    encryption_key: Optional[Any] = None,
    timeout: Union[int, Tuple[int, int]] = 60,
    project: Optional[str] = None,
    **upload_kwargs: Any
) -> str: ...

def cloud_storage_copy_blob(
    source_bucket: str,
    dest_bucket: str,
    source_blob: str,
    gcp_credentials: GcpCredentials,
    dest_blob: Optional[str] = None,
    timeout: Union[int, Tuple[int, int]] = 60,
    project: Optional[str] = None,
    **copy_kwargs: Any
) -> str: ...

class DataFrameSerializationFormat(Enum):
    CSV: Tuple[str, None, str, str]
    CSV_GZIP: Tuple[str, str, str, str]
    PARQUET: Tuple[str, None, str, str]
    PARQUET_SNAPPY: Tuple[str, str, str, str]
    PARQUET_GZIP: Tuple[str, str, str, str]
    
    @property
    def format(self) -> str: ...
    
    @property
    def compression(self) -> Optional[str]: ...
    
    @property
    def content_type(self) -> str: ...
    
    @property
    def suffix(self) -> str: ...
    
    def fix_extension_with(self, gcs_blob_path: str) -> str: ...

class GcsBucket(WritableDeploymentStorage, WritableFileSystem, ObjectStorageBlock):
    _logo_url: str
    _block_type_name: str
    _documentation_url: str
    bucket: str
    gcp_credentials: GcpCredentials
    bucket_folder: str
    
    @property
    def basepath(self) -> str: ...
    
    @field_validator('bucket_folder')
    @classmethod
    def _bucket_folder_suffix(cls, value: str) -> str: ...
    
    def _resolve_path(self, path: str) -> Optional[str]: ...
    
    async def get_directory(
        self,
        from_path: Optional[str] = None,
        local_path: Optional[str] = None
    ) -> List[str]: ...
    
    async def put_directory(
        self,
        local_path: Optional[str] = None,
        to_path: Optional[str] = None,
        ignore_file: Optional[str] = None
    ) -> int: ...
    
    async def read_path(self, path: str) -> Union[bytes, str]: ...
    
    async def write_path(self, path: str, content: Union[str, bytes]) -> str: ...
    
    def _join_bucket_folder(self, bucket_path: str = '') -> Optional[str]: ...
    
    async def create_bucket(
        self,
        location: Optional[str] = None,
        **create_kwargs: Any
    ) -> Any: ...
    
    async def get_bucket(self) -> Any: ...
    
    async def list_blobs(self, folder: str = '') -> List[Any]: ...
    
    async def list_folders(self, folder: str = '') -> List[str]: ...
    
    async def download_object_to_path(
        self,
        from_path: str,
        to_path: Optional[str] = None,
        **download_kwargs: Any
    ) -> Path: ...
    
    async def download_object_to_file_object(
        self,
        from_path: str,
        to_file_object: BinaryIO,
        **download_kwargs: Any
    ) -> BinaryIO: ...
    
    async def download_folder_to_path(
        self,
        from_folder: str,
        to_folder: Optional[str] = None,
        **download_kwargs: Any
    ) -> Path: ...
    
    async def upload_from_path(
        self,
        from_path: str,
        to_path: Optional[str] = None,
        **upload_kwargs: Any
    ) -> str: ...
    
    async def upload_from_file_object(
        self,
        from_file_object: BinaryIO,
        to_path: str,
        **upload_kwargs: Any
    ) -> str: ...
    
    async def upload_from_folder(
        self,
        from_folder: str,
        to_folder: Optional[str] = None,
        **upload_kwargs: Any
    ) -> str: ...
    
    async def upload_from_dataframe(
        self,
        df: Any,
        to_path: str,
        serialization_format: Union[DataFrameSerializationFormat, str] = DataFrameSerializationFormat.CSV_GZIP,
        **upload_kwargs: Any
    ) -> str: ...
```