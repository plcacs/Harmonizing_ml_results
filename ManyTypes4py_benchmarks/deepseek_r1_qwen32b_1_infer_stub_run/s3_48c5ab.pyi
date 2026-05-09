"""Stub file for 's3_48c5ab' module."""

from __future__ import annotations
from typing import (
    Any,
    Optional,
    List,
    Dict,
    Union,
    Tuple,
    Callable,
    Iterable,
    Iterator,
    overload,
)
from pathlib import Path
from uuid import UUID
from io import BytesIO
from botocore.response import StreamingBody
from pydantic import BaseModel
from prefect import task
from prefect_aws import AwsCredentials, MinIOCredentials
from prefect_aws.client_parameters import AwsClientParameters

class S3Bucket:
    """Stub for S3Bucket class."""

    def __init__(
        self,
        bucket_name: str,
        credentials: Union[AwsCredentials, MinIOCredentials],
        bucket_folder: str = '',
    ) -> None:
        ...

    async def aread_path(self, path: str) -> bytes:
        ...

    def read_path(self, path: str) -> bytes:
        ...

    async def awrite_path(self, path: str, content: bytes) -> str:
        ...

    def write_path(self, path: str, content: bytes) -> str:
        ...

    async def alist_objects(
        self,
        folder: str = '',
        delimiter: str = '',
        page_size: Optional[int] = None,
        max_items: Optional[int] = None,
        jmespath_query: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        ...

    def list_objects(
        self,
        folder: str = '',
        delimiter: str = '',
        page_size: Optional[int] = None,
        max_items: Optional[int] = None,
        jmespath_query: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        ...

    async def adownload_object_to_path(
        self,
        from_path: str,
        to_path: Optional[str] = None,
        **download_kwargs: Any,
    ) -> Path:
        ...

    def download_object_to_path(
        self,
        from_path: str,
        to_path: Optional[str] = None,
        **download_kwargs: Any,
    ) -> Path:
        ...

    async def adownload_object_to_file_object(
        self,
        from_path: str,
        to_file_object: BinaryIO,
        **download_kwargs: Any,
    ) -> BinaryIO:
        ...

    def download_object_to_file_object(
        self,
        from_path: str,
        to_file_object: BinaryIO,
        **download_kwargs: Any,
    ) -> BinaryIO:
        ...

    async def adownload_folder_to_path(
        self,
        from_folder: str,
        to_folder: Optional[str] = None,
        **download_kwargs: Any,
    ) -> Path:
        ...

    def download_folder_to_path(
        self,
        from_folder: str,
        to_folder: Optional[str] = None,
        **download_kwargs: Any,
    ) -> Path:
        ...

    async def aupload_from_path(
        self,
        from_path: str,
        to_path: Optional[str] = None,
        **upload_kwargs: Any,
    ) -> str:
        ...

    def upload_from_path(
        self,
        from_path: str,
        to_path: Optional[str] = None,
        **upload_kwargs: Any,
    ) -> str:
        ...

    async def aupload_from_file_object(
        self,
        from_file_object: BinaryIO,
        to_path: str,
        **upload_kwargs: Any,
    ) -> str:
        ...

    def upload_from_file_object(
        self,
        from_file_object: BinaryIO,
        to_path: str,
        **upload_kwargs: Any,
    ) -> str:
        ...

    async def aupload_from_folder(
        self,
        from_folder: str,
        to_folder: Optional[str] = None,
        **upload_kwargs: Any,
    ) -> Optional[str]:
        ...

    def upload_from_folder(
        self,
        from_folder: str,
        to_folder: Optional[str] = None,
        **upload_kwargs: Any,
    ) -> Optional[str]:
        ...

    def copy_object(
        self,
        from_path: str,
        to_path: str,
        to_bucket: Optional[Union[str, S3Bucket]] = None,
        **copy_kwargs: Any,
    ) -> str:
        ...

    async def amove_object(
        self,
        from_path: str,
        to_path: str,
        to_bucket: Optional[Union[str, S3Bucket]] = None,
    ) -> str:
        ...

    def move_object(
        self,
        from_path: str,
        to_path: str,
        to_bucket: Optional[Union[str, S3Bucket]] = None,
    ) -> str:
        ...

@task
async def adownload_from_bucket(
    bucket: str,
    key: str,
    aws_credentials: AwsCredentials,
    aws_client_parameters: AwsClientParameters = AwsClientParameters(),
) -> bytes:
    ...

@task
def download_from_bucket(
    bucket: str,
    key: str,
    aws_credentials: AwsCredentials,
    aws_client_parameters: AwsClientParameters = AwsClientParameters(),
) -> bytes:
    ...

@task
async def aupload_to_bucket(
    data: bytes,
    bucket: str,
    aws_credentials: AwsCredentials,
    aws_client_parameters: AwsClientParameters = AwsClientParameters(),
    key: Optional[str] = None,
) -> str:
    ...

@task
def upload_to_bucket(
    data: bytes,
    bucket: str,
    aws_credentials: AwsCredentials,
    aws_client_parameters: AwsClientParameters = AwsClientParameters(),
    key: Optional[str] = None,
) -> str:
    ...

@task
async def acopy_objects(
    source_path: Union[str, Path],
    target_path: Union[str, Path],
    source_bucket_name: str,
    aws_credentials: AwsCredentials,
    target_bucket_name: Optional[str] = None,
    **copy_kwargs: Any,
) -> str:
    ...

@task
def copy_objects(
    source_path: Union[str, Path],
    target_path: Union[str, Path],
    source_bucket_name: str,
    aws_credentials: AwsCredentials,
    target_bucket_name: Optional[str] = None,
    **copy_kwargs: Any,
) -> str:
    ...

@task
async def amove_objects(
    source_path: Union[str, Path],
    target_path: Union[str, Path],
    source_bucket_name: str,
    aws_credentials: AwsCredentials,
    target_bucket_name: Optional[str] = None,
) -> str:
    ...

@task
def move_objects(
    source_path: Union[str, Path],
    target_path: Union[str, Path],
    source_bucket_name: str,
    aws_credentials: AwsCredentials,
    target_bucket_name: Optional[str] = None,
) -> str:
    ...

@task
async def alist_objects(
    bucket: str,
    aws_credentials: AwsCredentials,
    aws_client_parameters: AwsClientParameters = AwsClientParameters(),
    prefix: str = '',
    delimiter: str = '',
    page_size: Optional[int] = None,
    max_items: Optional[int] = None,
    jmespath_query: Optional[str] = None,
) -> List[Dict[str, Any]]:
    ...

@task
def list_objects(
    bucket: str,
    aws_credentials: AwsCredentials,
    aws_client_parameters: AwsClientParameters = AwsClientParameters(),
    prefix: str = '',
    delimiter: str = '',
    page_size: Optional[int] = None,
    max_items: Optional[int] = None,
    jmespath_query: Optional[str] = None,
) -> List[Dict[str, Any]]:
    ...