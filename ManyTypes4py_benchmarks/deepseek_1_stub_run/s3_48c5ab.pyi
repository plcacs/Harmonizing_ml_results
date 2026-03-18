```python
import asyncio
import io
import os
import uuid
from pathlib import Path
from typing import Any, BinaryIO, Dict, List, Optional, Tuple, Union
from botocore.paginate import PageIterator
from botocore.response import StreamingBody
from pydantic import Field, field_validator
from prefect import task
from prefect.blocks.abstract import CredentialsBlock, ObjectStorageBlock
from prefect.filesystems import WritableDeploymentStorage, WritableFileSystem
from prefect.logging import get_run_logger
from prefect_aws import AwsCredentials, MinIOCredentials
from prefect_aws.client_parameters import AwsClientParameters

@task
async def adownload_from_bucket(
    bucket: Any,
    key: Any,
    aws_credentials: Any,
    aws_client_parameters: AwsClientParameters = ...
) -> bytes: ...

@async_dispatch(adownload_from_bucket)
@task
def download_from_bucket(
    bucket: Any,
    key: Any,
    aws_credentials: Any,
    aws_client_parameters: AwsClientParameters = ...
) -> bytes: ...

s3_download: Any = ...

@task
async def aupload_to_bucket(
    data: Any,
    bucket: Any,
    aws_credentials: Any,
    aws_client_parameters: AwsClientParameters = ...,
    key: Optional[Any] = None
) -> str: ...

@async_dispatch(aupload_to_bucket)
@task
def upload_to_bucket(
    data: Any,
    bucket: Any,
    aws_credentials: Any,
    aws_client_parameters: AwsClientParameters = ...,
    key: Optional[Any] = None
) -> str: ...

s3_upload: Any = ...

@task
async def acopy_objects(
    source_path: Any,
    target_path: Any,
    source_bucket_name: Any,
    aws_credentials: Any,
    target_bucket_name: Optional[Any] = None,
    **copy_kwargs: Any
) -> Any: ...

@async_dispatch(acopy_objects)
@task
def copy_objects(
    source_path: Any,
    target_path: Any,
    source_bucket_name: Any,
    aws_credentials: Any,
    target_bucket_name: Optional[Any] = None,
    **copy_kwargs: Any
) -> Any: ...

s3_copy: Any = ...

@task
async def amove_objects(
    source_path: Any,
    target_path: Any,
    source_bucket_name: Any,
    aws_credentials: Any,
    target_bucket_name: Optional[Any] = None
) -> Any: ...

@async_dispatch(amove_objects)
@task
def move_objects(
    source_path: Any,
    target_path: Any,
    source_bucket_name: Any,
    aws_credentials: Any,
    target_bucket_name: Optional[Any] = None
) -> Any: ...

s3_move: Any = ...

def _list_objects_sync(page_iterator: PageIterator) -> List[Dict[Any, Any]]: ...

@task
async def alist_objects(
    bucket: Any,
    aws_credentials: Any,
    aws_client_parameters: AwsClientParameters = ...,
    prefix: str = '',
    delimiter: str = '',
    page_size: Optional[int] = None,
    max_items: Optional[int] = None,
    jmespath_query: Optional[str] = None
) -> List[Dict[Any, Any]]: ...

@async_dispatch(alist_objects)
@task
def list_objects(
    bucket: Any,
    aws_credentials: Any,
    aws_client_parameters: AwsClientParameters = ...,
    prefix: str = '',
    delimiter: str = '',
    page_size: Optional[int] = None,
    max_items: Optional[int] = None,
    jmespath_query: Optional[str] = None
) -> List[Dict[Any, Any]]: ...

s3_list_objects: Any = ...

class S3Bucket(WritableFileSystem, WritableDeploymentStorage, ObjectStorageBlock):
    _logo_url: str = ...
    _block_type_name: str = ...
    _documentation_url: str = ...
    bucket_name: str = Field(...)
    credentials: Union[AwsCredentials, MinIOCredentials] = Field(...)
    bucket_folder: str = Field(...)
    
    @field_validator('credentials', mode='before')
    @classmethod
    def validate_credentials(cls, value: Any, field: Any) -> Any: ...
    
    @property
    def basepath(self) -> str: ...
    
    @basepath.setter
    def basepath(self, value: str) -> None: ...
    
    def _resolve_path(self, path: Any) -> str: ...
    
    def _get_s3_client(self) -> Any: ...
    
    def _get_bucket_resource(self) -> Any: ...
    
    async def aget_directory(
        self,
        from_path: Optional[Any] = None,
        local_path: Optional[Any] = None
    ) -> None: ...
    
    @async_dispatch(aget_directory)
    def get_directory(
        self,
        from_path: Optional[Any] = None,
        local_path: Optional[Any] = None
    ) -> None: ...
    
    async def aput_directory(
        self,
        local_path: Optional[Any] = None,
        to_path: Optional[Any] = None,
        ignore_file: Optional[Any] = None
    ) -> int: ...
    
    @async_dispatch(aput_directory)
    def put_directory(
        self,
        local_path: Optional[Any] = None,
        to_path: Optional[Any] = None,
        ignore_file: Optional[Any] = None
    ) -> int: ...
    
    def _read_sync(self, key: str) -> bytes: ...
    
    async def aread_path(self, path: Any) -> bytes: ...
    
    @async_dispatch(aread_path)
    def read_path(self, path: Any) -> bytes: ...
    
    def _write_sync(self, key: str, data: bytes) -> None: ...
    
    async def awrite_path(self, path: Any, content: bytes) -> str: ...
    
    @async_dispatch(awrite_path)
    def write_path(self, path: Any, content: bytes) -> str: ...
    
    @staticmethod
    def _list_objects_sync(page_iterator: PageIterator) -> List[Dict[Any, Any]]: ...
    
    def _join_bucket_folder(self, bucket_path: str = '') -> str: ...
    
    def _list_objects_setup(
        self,
        folder: str = '',
        delimiter: str = '',
        page_size: Optional[int] = None,
        max_items: Optional[int] = None,
        jmespath_query: Optional[str] = None
    ) -> Tuple[Any, str]: ...
    
    async def alist_objects(
        self,
        folder: str = '',
        delimiter: str = '',
        page_size: Optional[int] = None,
        max_items: Optional[int] = None,
        jmespath_query: Optional[str] = None
    ) -> List[Dict[Any, Any]]: ...
    
    @async_dispatch(alist_objects)
    def list_objects(
        self,
        folder: str = '',
        delimiter: str = '',
        page_size: Optional[int] = None,
        max_items: Optional[int] = None,
        jmespath_query: Optional[str] = None
    ) -> List[Dict[Any, Any]]: ...
    
    async def adownload_object_to_path(
        self,
        from_path: Any,
        to_path: Optional[Any] = None,
        **download_kwargs: Any
    ) -> Path: ...
    
    @async_dispatch(adownload_object_to_path)
    def download_object_to_path(
        self,
        from_path: Any,
        to_path: Optional[Any] = None,
        **download_kwargs: Any
    ) -> Path: ...
    
    async def adownload_object_to_file_object(
        self,
        from_path: Any,
        to_file_object: Any,
        **download_kwargs: Any
    ) -> Any: ...
    
    @async_dispatch(adownload_object_to_file_object)
    def download_object_to_file_object(
        self,
        from_path: Any,
        to_file_object: Any,
        **download_kwargs: Any
    ) -> Any: ...
    
    async def adownload_folder_to_path(
        self,
        from_folder: Any,
        to_folder: Optional[Any] = None,
        **download_kwargs: Any
    ) -> Path: ...
    
    @async_dispatch(adownload_folder_to_path)
    def download_folder_to_path(
        self,
        from_folder: Any,
        to_folder: Optional[Any] = None,
        **download_kwargs: Any
    ) -> Path: ...
    
    async def astream_from(
        self,
        bucket: Any,
        from_path: Any,
        to_path: Optional[Any] = None,
        **upload_kwargs: Any
    ) -> str: ...
    
    @async_dispatch(astream_from)
    def stream_from(
        self,
        bucket: Any,
        from_path: Any,
        to_path: Optional[Any] = None,
        **upload_kwargs: Any
    ) -> str: ...
    
    async def aupload_from_path(
        self,
        from_path: Any,
        to_path: Optional[Any] = None,
        **upload_kwargs: Any
    ) -> str: ...
    
    @async_dispatch(aupload_from_path)
    def upload_from_path(
        self,
        from_path: Any,
        to_path: Optional[Any] = None,
        **upload_kwargs: Any
    ) -> str: ...
    
    async def aupload_from_file_object(
        self,
        from_file_object: Any,
        to_path: Any,
        **upload_kwargs: Any
    ) -> str: ...
    
    @async_dispatch(aupload_from_file_object)
    def upload_from_file_object(
        self,
        from_file_object: Any,
        to_path: Any,
        **upload_kwargs: Any
    ) -> str: ...
    
    async def aupload_from_folder(
        self,
        from_folder: Any,
        to_folder: Optional[Any] = None,
        **upload_kwargs: Any
    ) -> Optional[str]: ...
    
    @async_dispatch(aupload_from_folder)
    def upload_from_folder(
        self,
        from_folder: Any,
        to_folder: Optional[Any] = None,
        **upload_kwargs: Any
    ) -> Optional[str]: ...
    
    def copy_object(
        self,
        from_path: Any,
        to_path: Any,
        to_bucket: Optional[Any] = None,
        **copy_kwargs: Any
    ) -> str: ...
    
    def _move_object_setup(
        self,
        from_path: Any,
        to_path: Any,
        to_bucket: Optional[Any] = None
    ) -> Tuple[str, str, str, str]: ...
    
    async def amove_object(
        self,
        from_path: Any,
        to_path: Any,
        to_bucket: Optional[Any] = None
    ) -> str: ...
    
    @async_dispatch(amove_object)
    def move_object(
        self,
        from_path: Any,
        to_path: Any,
        to_bucket: Optional[Any] = None
    ) -> str: ...
```