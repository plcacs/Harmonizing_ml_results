from __future__ import annotations

from enum import Enum
from io import BytesIO
import os
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, ClassVar, List, Optional, Tuple, Union

from prefect.blocks.abstract import ObjectStorageBlock
from prefect.filesystems import WritableDeploymentStorage, WritableFileSystem
from prefect_gcp.credentials import GcpCredentials


async def cloud_storage_create_bucket(
    bucket: str,
    gcp_credentials: GcpCredentials,
    project: Optional[str] = ...,
    location: Optional[str] = ...,
    **create_kwargs: Any,
) -> str: ...


async def _get_bucket_async(
    bucket: str,
    gcp_credentials: GcpCredentials,
    project: Optional[str] = ...,
) -> Any: ...


def _get_bucket(
    bucket: str,
    gcp_credentials: GcpCredentials,
    project: Optional[str] = ...,
) -> Any: ...


async def cloud_storage_download_blob_as_bytes(
    bucket: str,
    blob: str,
    gcp_credentials: GcpCredentials,
    chunk_size: Optional[int] = ...,
    encryption_key: Optional[Any] = ...,
    timeout: Union[float, Tuple[float, float]] = ...,
    project: Optional[str] = ...,
    **download_kwargs: Any,
) -> Any: ...


async def cloud_storage_download_blob_to_file(
    bucket: str,
    blob: str,
    path: Union[str, Path],
    gcp_credentials: GcpCredentials,
    chunk_size: Optional[int] = ...,
    encryption_key: Optional[Any] = ...,
    timeout: Union[float, Tuple[float, float]] = ...,
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
    encryption_key: Optional[Any] = ...,
    timeout: Union[float, Tuple[float, float]] = ...,
    project: Optional[str] = ...,
    **upload_kwargs: Any,
) -> str: ...


async def cloud_storage_upload_blob_from_file(
    file: Union[str, os.PathLike[str], BinaryIO, BytesIO],
    bucket: str,
    blob: str,
    gcp_credentials: GcpCredentials,
    content_type: Optional[str] = ...,
    chunk_size: Optional[int] = ...,
    encryption_key: Optional[Any] = ...,
    timeout: Union[float, Tuple[float, float]] = ...,
    project: Optional[str] = ...,
    **upload_kwargs: Any,
) -> str: ...


def cloud_storage_copy_blob(
    source_bucket: str,
    dest_bucket: str,
    source_blob: str,
    gcp_credentials: GcpCredentials,
    dest_blob: Optional[str] = ...,
    timeout: Union[float, Tuple[float, float]] = ...,
    project: Optional[str] = ...,
    **copy_kwargs: Any,
) -> str: ...


class DataFrameSerializationFormat(Enum):
    CSV: ClassVar["DataFrameSerializationFormat"]
    CSV_GZIP: ClassVar["DataFrameSerializationFormat"]
    PARQUET: ClassVar["DataFrameSerializationFormat"]
    PARQUET_SNAPPY: ClassVar["DataFrameSerializationFormat"]
    PARQUET_GZIP: ClassVar["DataFrameSerializationFormat"]

    @property
    def format(self) -> str: ...
    @property
    def compression(self) -> Optional[str]: ...
    @property
    def content_type(self) -> str: ...
    @property
    def suffix(self) -> str: ...
    def fix_extension_with(self, gcs_blob_path: Union[str, PurePosixPath]) -> str: ...


class GcsBucket(WritableDeploymentStorage, WritableFileSystem, ObjectStorageBlock):
    _logo_url: ClassVar[str]
    _block_type_name: ClassVar[str]
    _documentation_url: ClassVar[str]
    bucket: str
    gcp_credentials: GcpCredentials
    bucket_folder: str

    @property
    def basepath(self) -> str: ...

    @classmethod
    def _bucket_folder_suffix(cls, value: str) -> str: ...

    def _resolve_path(self, path: str) -> Optional[str]: ...

    async def get_directory(
        self,
        from_path: Optional[str] = ...,
        local_path: Optional[Union[str, os.PathLike[str]]] = ...,
    ) -> List[Union[str, Path]]: ...

    async def put_directory(
        self,
        local_path: Optional[Union[str, os.PathLike[str]]] = ...,
        to_path: Optional[str] = ...,
        ignore_file: Optional[Union[str, os.PathLike[str]]] = ...,
    ) -> int: ...

    async def read_path(self, path: str) -> Any: ...

    async def write_path(self, path: str, content: Union[str, bytes]) -> str: ...

    def _join_bucket_folder(self, bucket_path: str = ...) -> Optional[str]: ...

    async def create_bucket(self, location: Optional[str] = ..., **create_kwargs: Any) -> Any: ...

    async def get_bucket(self) -> Any: ...

    async def list_blobs(self, folder: str = ...) -> List[Any]: ...

    async def list_folders(self, folder: str = ...) -> List[str]: ...

    async def download_object_to_path(
        self,
        from_path: str,
        to_path: Optional[Union[str, os.PathLike[str]]] = ...,
        **download_kwargs: Any,
    ) -> Path: ...

    async def download_object_to_file_object(
        self,
        from_path: str,
        to_file_object: BinaryIO,
        **download_kwargs: Any,
    ) -> BinaryIO: ...

    async def download_folder_to_path(
        self,
        from_folder: str,
        to_folder: Optional[Union[str, os.PathLike[str]]] = ...,
        **download_kwargs: Any,
    ) -> Path: ...

    async def upload_from_path(
        self,
        from_path: Union[str, os.PathLike[str]],
        to_path: Optional[str] = ...,
        **upload_kwargs: Any,
    ) -> str: ...

    async def upload_from_file_object(
        self,
        from_file_object: BinaryIO,
        to_path: str,
        **upload_kwargs: Any,
    ) -> str: ...

    async def upload_from_folder(
        self,
        from_folder: Union[str, os.PathLike[str]],
        to_folder: Optional[str] = ...,
        **upload_kwargs: Any,
    ) -> str: ...

    async def upload_from_dataframe(
        self,
        df: Any,
        to_path: str,
        serialization_format: Union[DataFrameSerializationFormat, str] = ...,
        **upload_kwargs: Any,
    ) -> str: ...