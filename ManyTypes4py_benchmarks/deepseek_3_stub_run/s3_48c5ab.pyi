import asyncio
import io
import os
import uuid
from pathlib import Path
from typing import (
    Any,
    BinaryIO,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
    get_args,
    overload,
)
from botocore.paginate import PageIterator
from botocore.response import StreamingBody
from pydantic import Field, field_validator
from prefect import task
from prefect._internal.compatibility.async_dispatch import async_dispatch
from prefect.blocks.abstract import CredentialsBlock, ObjectStorageBlock
from prefect.filesystems import WritableDeploymentStorage, WritableFileSystem
from prefect.logging import get_run_logger
from prefect.utilities.asyncutils import run_sync_in_worker_thread
from prefect.utilities.filesystem import filter_files
from prefect.utilities.pydantic import lookup_type
from prefect_aws import AwsCredentials, MinIOCredentials
from prefect_aws.client_parameters import AwsClientParameters

@task
async def adownload_from_bucket(
    bucket: str,
    key: str,
    aws_credentials: Union[AwsCredentials, MinIOCredentials],
    aws_client_parameters: AwsClientParameters = ...,
) -> bytes: ...

@async_dispatch(adownload_from_bucket)
@task
def download_from_bucket(
    bucket: str,
    key: str,
    aws_credentials: Union[AwsCredentials, MinIOCredentials],
    aws_client_parameters: AwsClientParameters = ...,
) -> bytes: ...

s3_download = download_from_bucket

@task
async def aupload_to_bucket(
    data: bytes,
    bucket: str,
    aws_credentials: Union[AwsCredentials, MinIOCredentials],
    aws_client_parameters: AwsClientParameters = ...,
    key: Optional[str] = ...,
) -> str: ...

@async_dispatch(aupload_to_bucket)
@task
def upload_to_bucket(
    data: bytes,
    bucket: str,
    aws_credentials: Union[AwsCredentials, MinIOCredentials],
    aws_client_parameters: AwsClientParameters = ...,
    key: Optional[str] = ...,
) -> str: ...

s3_upload = upload_to_bucket

@task
async def acopy_objects(
    source_path: Union[str, Path],
    target_path: Union[str, Path],
    source_bucket_name: str,
    aws_credentials: Union[AwsCredentials, MinIOCredentials],
    target_bucket_name: Optional[str] = ...,
    **copy_kwargs: Any,
) -> str: ...

@async_dispatch(acopy_objects)
@task
def copy_objects(
    source_path: Union[str, Path],
    target_path: Union[str, Path],
    source_bucket_name: str,
    aws_credentials: Union[AwsCredentials, MinIOCredentials],
    target_bucket_name: Optional[str] = ...,
    **copy_kwargs: Any,
) -> str: ...

s3_copy = copy_objects

@task
async def amove_objects(
    source_path: Union[str, Path],
    target_path: Union[str, Path],
    source_bucket_name: str,
    aws_credentials: Union[AwsCredentials, MinIOCredentials],
    target_bucket_name: Optional[str] = ...,
) -> str: ...

@async_dispatch(amove_objects)
@task
def move_objects(
    source_path: Union[str, Path],
    target_path: Union[str, Path],
    source_bucket_name: str,
    aws_credentials: Union[AwsCredentials, MinIOCredentials],
    target_bucket_name: Optional[str] = ...,
) -> str: ...

s3_move = move_objects

def _list_objects_sync(page_iterator: PageIterator) -> List[Dict[str, Any]]: ...

@task
async def alist_objects(
    bucket: str,
    aws_credentials: Union[AwsCredentials, MinIOCredentials],
    aws_client_parameters: AwsClientParameters = ...,
    prefix: str = ...,
    delimiter: str = ...,
    page_size: Optional[int] = ...,
    max_items: Optional[int] = ...,
    jmespath_query: Optional[str] = ...,
) -> List[Dict[str, Any]]: ...

@async_dispatch(alist_objects)
@task
def list_objects(
    bucket: str,
    aws_credentials: Union[AwsCredentials, MinIOCredentials],
    aws_client_parameters: AwsClientParameters = ...,
    prefix: str = ...,
    delimiter: str = ...,
    page_size: Optional[int] = ...,
    max_items: Optional[int] = ...,
    jmespath_query: Optional[str] = ...,
) -> List[Dict[str, Any]]: ...

s3_list_objects = list_objects

class S3Bucket(WritableFileSystem, WritableDeploymentStorage, ObjectStorageBlock):
    _logo_url: str = ...
    _block_type_name: str = ...
    _documentation_url: str = ...
    bucket_name: str = Field(...)
    credentials: Union[AwsCredentials, MinIOCredentials] = Field(...)
    bucket_folder: str = Field(...)
    
    @field_validator('credentials', mode='before')
    @classmethod
    def validate_credentials(cls, value: Any, field: Any) -> Union[AwsCredentials, MinIOCredentials]: ...
    
    @property
    def basepath(self) -> str: ...
    
    @basepath.setter
    def basepath(self, value: str) -> None: ...
    
    def _resolve_path(self, path: str) -> str: ...
    
    def _get_s3_client(self) -> Any: ...
    
    def _get_bucket_resource(self) -> Any: ...
    
    async def aget_directory(
        self,
        from_path: Optional[str] = ...,
        local_path: Optional[str] = ...,
    ) -> None: ...
    
    @async_dispatch(aget_directory)
    def get_directory(
        self,
        from_path: Optional[str] = ...,
        local_path: Optional[str] = ...,
    ) -> None: ...
    
    async def aput_directory(
        self,
        local_path: Optional[str] = ...,
        to_path: Optional[str] = ...,
        ignore_file: Optional[str] = ...,
    ) -> int: ...
    
    @async_dispatch(aput_directory)
    def put_directory(
        self,
        local_path: Optional[str] = ...,
        to_path: Optional[str] = ...,
        ignore_file: Optional[str] = ...,
    ) -> int: ...
    
    def _read_sync(self, key: str) -> bytes: ...
    
    async def aread_path(self, path: str) -> bytes: ...
    
    @async_dispatch(aread_path)
    def read_path(self, path: str) -> bytes: ...
    
    def _write_sync(self, key: str, data: bytes) -> None: ...
    
    async def awrite_path(self, path: str, content: bytes) -> str: ...
    
    @async_dispatch(awrite_path)
    def write_path(self, path: str, content: bytes) -> str: ...
    
    @staticmethod
    def _list_objects_sync(page_iterator: PageIterator) -> List[Dict[str, Any]]: ...
    
    def _join_bucket_folder(self, bucket_path: str = ...) -> str: ...
    
    def _list_objects_setup(
        self,
        folder: str = ...,
        delimiter: str = ...,
        page_size: Optional[int] = ...,
        max_items: Optional[int] = ...,
        jmespath_query: Optional[str] = ...,
    ) -> Tuple[PageIterator, str]: ...
    
    async def alist_objects(
        self,
        folder: str = ...,
        delimiter: str = ...,
        page_size: Optional[int] = ...,
        max_items: Optional[int] = ...,
        jmespath_query: Optional[str] = ...,
    ) -> List[Dict[str, Any]]: ...
    
    @async_dispatch(alist_objects)
    def list_objects(
        self,
        folder: str = ...,
        delimiter: str = ...,
        page_size: Optional[int] = ...,
        max_items: Optional[int] = ...,
        jmespath_query: Optional[str] = ...,
    ) -> List[Dict[str, Any]]: ...
    
    async def adownload_object_to_path(
        self,
        from_path: str,
        to_path: Optional[str] = ...,
        **download_kwargs: Any,
    ) -> Path: ...
    
    @async_dispatch(adownload_object_to_path)
    def download_object_to_path(
        self,
        from_path: str,
        to_path: Optional[str] = ...,
        **download_kwargs: Any,
    ) -> Path: ...
    
    async def adownload_object_to_file_object(
        self,
        from_path: str,
        to_file_object: BinaryIO,
        **download_kwargs: Any,
    ) -> BinaryIO: ...
    
    @async_dispatch(adownload_object_to_file_object)
    def download_object_to_file_object(
        self,
        from_path: str,
        to_file_object: BinaryIO,
        **download_kwargs: Any,
    ) -> BinaryIO: ...
    
    async def adownload_folder_to_path(
        self,
        from_folder: str,
        to_folder: Optional[str] = ...,
        **download_kwargs: Any,
    ) -> Path: ...
    
    @async_dispatch(adownload_folder_to_path)
    def download_folder_to_path(
        self,
        from_folder: str,
        to_folder: Optional[str] = ...,
        **download_kwargs: Any,
    ) -> Path: ...
    
    async def astream_from(
        self,
        bucket: "S3Bucket",
        from_path: str,
        to_path: Optional[str] = ...,
        **upload_kwargs: Any,
    ) -> str: ...
    
    @async_dispatch(astream_from)
    def stream_from(
        self,
        bucket: "S3Bucket",
        from_path: str,
        to_path: Optional[str] = ...,
        **upload_kwargs: Any,
    ) -> str: ...
    
    async def aupload_from_path(
        self,
        from_path: str,
        to_path: Optional[str] = ...,
        **upload_kwargs: Any,
    ) -> str: ...
    
    @async_dispatch(aupload_from_path)
    def upload_from_path(
        self,
        from_path: str,
        to_path: Optional[str] = ...,
        **upload_kwargs: Any,
    ) -> str: ...
    
    async def aupload_from_file_object(
        self,
        from_file_object: BinaryIO,
        to_path: str,
        **upload_kwargs: Any,
    ) -> str: ...
    
    @async_dispatch(aupload_from_file_object)
    def upload_from_file_object(
        self,
        from_file_object: BinaryIO,
        to_path: str,
        **upload_kwargs: Any,
    ) -> str: ...
    
    async def aupload_from_folder(
        self,
        from_folder: str,
        to_folder: Optional[str] = ...,
        **upload_kwargs: Any,
    ) -> Optional[str]: ...
    
    @async_dispatch(aupload_from_folder)
    def upload_from_folder(
        self,
        from_folder: str,
        to_folder: Optional[str] = ...,
        **upload_kwargs: Any,
    ) -> Optional[str]: ...
    
    def copy_object(
        self,
        from_path: str,
        to_path: str,
        to_bucket: Optional[Union[str, "S3Bucket"]] = ...,
        **copy_kwargs: Any,
    ) -> str: ...
    
    def _move_object_setup(
        self,
        from_path: str,
        to_path: str,
        to_bucket: Optional[Union[str, "S3Bucket"]] = ...,
    ) -> Tuple[str, str, str, str]: ...
    
    async def amove_object(
        self,
        from_path: str,
        to_path: str,
        to_bucket: Optional[Union[str, "S3Bucket"]] = ...,
    ) -> str: ...
    
    @async_dispatch(amove_object)
    def move_object(
        self,
        from_path: str,
        to_path: str,
        to_bucket: Optional[Union[str, "S3Bucket"]] = ...,
    ) -> str: ...