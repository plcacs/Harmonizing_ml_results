"""Tasks for interacting with GCP Cloud Storage."""
import asyncio
from enum import Enum
from io import BytesIO
from pathlib import Path
from typing import Any, BinaryIO, List, Optional, Tuple, Union, overload

from prefect_gcp.credentials import GcpCredentials
from prefect.blocks.abstract import ObjectStorageBlock
from prefect.filesystems import WritableDeploymentStorage, WritableFileSystem

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

async def cloud_storage_create_bucket(
    bucket: str,
    gcp_credentials: GcpCredentials,
    project: Optional[str] = ...,
    location: Optional[str] = ...,
    **create_kwargs: Any,
) -> str: ...

async def _get_bucket_async(
    bucket: str, gcp_credentials: GcpCredentials, project: Optional[str] = ...
) -> Bucket: ...

def _get_bucket(
    bucket: str, gcp_credentials: GcpCredentials, project: Optional[str] = ...
) -> Bucket: ...

async def cloud_storage_download_blob_as_bytes(
    bucket: str,
    blob: str,
    gcp_credentials: GcpCredentials,
    chunk_size: Optional[int] = ...,
    encryption_key: Optional[bytes] = ...,
    timeout: Union[int, float, Tuple[float, float]] = 60,
    project: Optional[str] = ...,
    **download_kwargs: Any,
) -> Union[bytes, str]: ...

async def cloud_storage_download_blob_to_file(
    bucket: str,
    blob: str,
    path: Union[str, Path],
    gcp_credentials: GcpCredentials,
    chunk_size: Optional[int] = ...,
    encryption_key: Optional[bytes] = ...,
    timeout: Union[int, float, Tuple[float, float]] = 60,
    project: Optional[str] = ...,
    **download_kwargs: Any,
) -> Union[str, Path]: ...

async def cloud_storage_upload_blob_from_string(
    data: Union[str, bytes],
    bucket: str,
    blob: str,
    gcp_credentials: GcpCredentials,
    content_type: Optional[str] = ...,
    chunk_size: Optional[int] = ...,
    encryption_key: Optional[bytes] = ...,
    timeout: Union[int, float, Tuple[float, float]] = 60,
    project: Optional[str] = ...,
    **upload_kwargs: Any,
) -> str: ...

async def cloud_storage_upload_blob_from_file(
    file: Union[str, Path, BinaryIO],
    bucket: str,
    blob: str,
    gcp_credentials: GcpCredentials,
    content_type: Optional[str] = ...,
    chunk_size: Optional[int] = ...,
    encryption_key: Optional[bytes] = ...,
    timeout: Union[int, float, Tuple[float, float]] = 60,
    project: Optional[str] = ...,
    **upload_kwargs: Any,
) -> str: ...

def cloud_storage_copy_blob(
    source_bucket: str,
    dest_bucket: str,
    source_blob: str,
    gcp_credentials: GcpCredentials,
    dest_blob: Optional[str] = ...,
    timeout: Union[int, float, Tuple[float, float]] = 60,
    project: Optional[str] = ...,
    **copy_kwargs: Any,
) -> str: ...

class DataFrameSerializationFormat(Enum):
    CSV = ('csv', None, 'text/csv', '.csv')
    CSV_GZIP = ('csv', 'gzip', 'application/x-gzip', '.csv.gz')
    PARQUET = ('parquet', None, 'application/octet-stream', '.parquet')
    PARQUET_SNAPPY = ('parquet', 'snappy', 'application/octet-stream', '.snappy.parquet')
    PARQUET_GZIP = ('parquet', 'gzip', 'application/octet-stream', '.gz.parquet')

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
    bucket: str
    gcp_credentials: GcpCredentials
    bucket_folder: str
    _logo_url: str
    _block_type_name: str
    _documentation_url: str

    @property
    def basepath(self) -> str: ...

    def _resolve_path(self, path: str) -> Optional[str]: ...

    async def get_directory(
        self, from_path: Optional[str] = ..., local_path: Optional[str] = ...
    ) -> List[str]: ...

    async def put_directory(
        self,
        local_path: Optional[str] = ...,
        to_path: Optional[str] = ...,
        ignore_file: Optional[str] = ...,
    ) -> int: ...

    async def read_path(self, path: str) -> Union[bytes, str]: ...

    async def write_path(self, path: str, content: Union[str, bytes]) -> Optional[str]: ...

    def _join_bucket_folder(self, bucket_path: str = '') -> Optional[str]: ...

    async def create_bucket(self, location: Optional[str] = ..., **create_kwargs: Any) -> Bucket: ...

    async def get_bucket(self) -> Bucket: ...

    async def list_blobs(self, folder: str = '') -> List[Blob]: ...

    async def list_folders(self, folder: str = '') -> List[str]: ...

    async def download_object_to_path(
        self, from_path: str, to_path: Optional[str] = ..., **download_kwargs: Any
    ) -> Path: ...

    async def download_object_to_file_object(
        self, from_path: str, to_file_object: BinaryIO, **download_kwargs: Any
    ) -> BinaryIO: ...

    async def download_folder_to_path(
        self, from_folder: str, to_folder: Optional[str] = ..., **download_kwargs: Any
    ) -> Path: ...

    async def upload_from_path(
        self, from_path: str, to_path: Optional[str] = ..., **upload_kwargs: Any
    ) -> Optional[str]: ...

    async def upload_from_file_object(
        self, from_file_object: BinaryIO, to_path: str, **upload_kwargs: Any
    ) -> Optional[str]: ...

    async def upload_from_folder(
        self, from_folder: str, to_folder: Optional[str] = ..., **upload_kwargs: Any
    ) -> str: ...

    async def upload_from_dataframe(
        self,
        df: DataFrame,
        to_path: str,
        serialization_format: Union[DataFrameSerializationFormat, str] = ...,
        **upload_kwargs: Any,
    ) -> Optional[str]: ...