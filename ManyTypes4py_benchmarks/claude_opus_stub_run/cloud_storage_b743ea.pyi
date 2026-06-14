import asyncio
from enum import Enum
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Dict, List, Optional, Tuple, Union

from pydantic import Field
from prefect.blocks.abstract import ObjectStorageBlock
from prefect.filesystems import WritableDeploymentStorage, WritableFileSystem
from prefect_gcp.credentials import GcpCredentials

try:
    from pandas import DataFrame
except ModuleNotFoundError:
    pass

try:
    from google.cloud.storage import Bucket
    from google.cloud.storage.blob import Blob
except ModuleNotFoundError:
    pass


def cloud_storage_create_bucket(
    bucket: str,
    gcp_credentials: GcpCredentials,
    project: Optional[str] = None,
    location: Optional[str] = None,
    **create_kwargs: Any,
) -> str: ...


async def _get_bucket_async(
    bucket: str,
    gcp_credentials: GcpCredentials,
    project: Optional[str] = None,
) -> Bucket: ...


def _get_bucket(
    bucket: str,
    gcp_credentials: GcpCredentials,
    project: Optional[str] = None,
) -> Bucket: ...


def cloud_storage_download_blob_as_bytes(
    bucket: str,
    blob: str,
    gcp_credentials: GcpCredentials,
    chunk_size: Optional[int] = None,
    encryption_key: Optional[str] = None,
    timeout: Union[float, Tuple[float, float]] = 60,
    project: Optional[str] = None,
    **download_kwargs: Any,
) -> bytes: ...


def cloud_storage_download_blob_to_file(
    bucket: str,
    blob: str,
    path: Union[str, Path],
    gcp_credentials: GcpCredentials,
    chunk_size: Optional[int] = None,
    encryption_key: Optional[str] = None,
    timeout: Union[float, Tuple[float, float]] = 60,
    project: Optional[str] = None,
    **download_kwargs: Any,
) -> Union[str, Path]: ...


def cloud_storage_upload_blob_from_string(
    data: Union[str, bytes],
    bucket: str,
    blob: str,
    gcp_credentials: GcpCredentials,
    content_type: Optional[str] = None,
    chunk_size: Optional[int] = None,
    encryption_key: Optional[str] = None,
    timeout: Union[float, Tuple[float, float]] = 60,
    project: Optional[str] = None,
    **upload_kwargs: Any,
) -> str: ...


def cloud_storage_upload_blob_from_file(
    file: Union[str, Path, BytesIO, BinaryIO],
    bucket: str,
    blob: str,
    gcp_credentials: GcpCredentials,
    content_type: Optional[str] = None,
    chunk_size: Optional[int] = None,
    encryption_key: Optional[str] = None,
    timeout: Union[float, Tuple[float, float]] = 60,
    project: Optional[str] = None,
    **upload_kwargs: Any,
) -> str: ...


def cloud_storage_copy_blob(
    source_bucket: str,
    dest_bucket: str,
    source_blob: str,
    gcp_credentials: GcpCredentials,
    dest_blob: Optional[str] = None,
    timeout: Union[float, Tuple[float, float]] = 60,
    project: Optional[str] = None,
    **copy_kwargs: Any,
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
    @field_validator("bucket_folder")
    @classmethod
    def _bucket_folder_suffix(cls, value: str) -> str: ...
    def _resolve_path(self, path: str) -> Optional[str]: ...
    def get_directory(
        self,
        from_path: Optional[str] = None,
        local_path: Optional[str] = None,
    ) -> List[str]: ...
    def put_directory(
        self,
        local_path: Optional[str] = None,
        to_path: Optional[str] = None,
        ignore_file: Optional[str] = None,
    ) -> int: ...
    def read_path(self, path: str) -> bytes: ...
    def write_path(self, path: str, content: Union[str, bytes]) -> str: ...
    def _join_bucket_folder(self, bucket_path: str = "") -> Optional[str]: ...
    def create_bucket(
        self,
        location: Optional[str] = None,
        **create_kwargs: Any,
    ) -> Bucket: ...
    def get_bucket(self) -> Bucket: ...
    def list_blobs(self, folder: str = "") -> List[Blob]: ...
    def list_folders(self, folder: str = "") -> List[str]: ...
    def download_object_to_path(
        self,
        from_path: str,
        to_path: Optional[Union[str, Path]] = None,
        **download_kwargs: Any,
    ) -> Path: ...
    def download_object_to_file_object(
        self,
        from_path: str,
        to_file_object: BinaryIO,
        **download_kwargs: Any,
    ) -> BinaryIO: ...
    def download_folder_to_path(
        self,
        from_folder: str,
        to_folder: Optional[Union[str, Path]] = None,
        **download_kwargs: Any,
    ) -> Path: ...
    def upload_from_path(
        self,
        from_path: Union[str, Path],
        to_path: Optional[str] = None,
        **upload_kwargs: Any,
    ) -> str: ...
    def upload_from_file_object(
        self,
        from_file_object: BinaryIO,
        to_path: str,
        **upload_kwargs: Any,
    ) -> str: ...
    def upload_from_folder(
        self,
        from_folder: Union[str, Path],
        to_folder: Optional[str] = None,
        **upload_kwargs: Any,
    ) -> str: ...
    def upload_from_dataframe(
        self,
        df: DataFrame,
        to_path: str,
        serialization_format: Union[DataFrameSerializationFormat, str] = DataFrameSerializationFormat.CSV_GZIP,
        **upload_kwargs: Any,
    ) -> str: ...