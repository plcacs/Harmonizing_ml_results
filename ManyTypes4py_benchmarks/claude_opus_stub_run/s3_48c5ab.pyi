from pathlib import Path
from typing import Any, BinaryIO, Dict, List, Optional, Tuple, Union

from botocore.paginate import PageIterator
from pydantic import Field

from prefect.blocks.abstract import ObjectStorageBlock
from prefect.filesystems import WritableDeploymentStorage, WritableFileSystem
from prefect_aws import AwsCredentials, MinIOCredentials
from prefect_aws.client_parameters import AwsClientParameters


async def adownload_from_bucket(
    bucket: str,
    key: str,
    aws_credentials: AwsCredentials,
    aws_client_parameters: AwsClientParameters = ...,
) -> bytes: ...

def download_from_bucket(
    bucket: str,
    key: str,
    aws_credentials: AwsCredentials,
    aws_client_parameters: AwsClientParameters = ...,
) -> bytes: ...

s3_download = download_from_bucket

async def aupload_to_bucket(
    data: bytes,
    bucket: str,
    aws_credentials: AwsCredentials,
    aws_client_parameters: AwsClientParameters = ...,
    key: Optional[str] = ...,
) -> str: ...

def upload_to_bucket(
    data: bytes,
    bucket: str,
    aws_credentials: AwsCredentials,
    aws_client_parameters: AwsClientParameters = ...,
    key: Optional[str] = ...,
) -> str: ...

s3_upload = upload_to_bucket

async def acopy_objects(
    source_path: Union[str, Path],
    target_path: Union[str, Path],
    source_bucket_name: str,
    aws_credentials: AwsCredentials,
    target_bucket_name: Optional[str] = ...,
    **copy_kwargs: Any,
) -> Union[str, Path]: ...

def copy_objects(
    source_path: Union[str, Path],
    target_path: Union[str, Path],
    source_bucket_name: str,
    aws_credentials: AwsCredentials,
    target_bucket_name: Optional[str] = ...,
    **copy_kwargs: Any,
) -> Union[str, Path]: ...

s3_copy = copy_objects

async def amove_objects(
    source_path: Union[str, Path],
    target_path: Union[str, Path],
    source_bucket_name: str,
    aws_credentials: AwsCredentials,
    target_bucket_name: Optional[str] = ...,
) -> Union[str, Path]: ...

def move_objects(
    source_path: Union[str, Path],
    target_path: Union[str, Path],
    source_bucket_name: str,
    aws_credentials: AwsCredentials,
    target_bucket_name: Optional[str] = ...,
) -> Union[str, Path]: ...

s3_move = move_objects

def _list_objects_sync(page_iterator: PageIterator) -> List[Dict[str, Any]]: ...

async def alist_objects(
    bucket: str,
    aws_credentials: AwsCredentials,
    aws_client_parameters: AwsClientParameters = ...,
    prefix: str = ...,
    delimiter: str = ...,
    page_size: Optional[int] = ...,
    max_items: Optional[int] = ...,
    jmespath_query: Optional[str] = ...,
) -> List[Dict[str, Any]]: ...

def list_objects(
    bucket: str,
    aws_credentials: AwsCredentials,
    aws_client_parameters: AwsClientParameters = ...,
    prefix: str = ...,
    delimiter: str = ...,
    page_size: Optional[int] = ...,
    max_items: Optional[int] = ...,
    jmespath_query: Optional[str] = ...,
) -> List[Dict[str, Any]]: ...

s3_list_objects = list_objects

class S3Bucket(WritableFileSystem, WritableDeploymentStorage, ObjectStorageBlock):
    _logo_url: str
    _block_type_name: str
    _documentation_url: str

    bucket_name: str
    credentials: Union[AwsCredentials, MinIOCredentials]
    bucket_folder: str

    @classmethod
    def validate_credentials(cls, value: Any, field: Any) -> Any: ...

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

    def put_directory(
        self,
        local_path: Optional[str] = ...,
        to_path: Optional[str] = ...,
        ignore_file: Optional[str] = ...,
    ) -> int: ...

    def _read_sync(self, key: str) -> bytes: ...

    async def aread_path(self, path: str) -> bytes: ...
    def read_path(self, path: str) -> bytes: ...

    def _write_sync(self, key: str, data: bytes) -> None: ...

    async def awrite_path(self, path: str, content: bytes) -> str: ...
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
    ) -> Tuple[Any, str]: ...

    async def alist_objects(
        self,
        folder: str = ...,
        delimiter: str = ...,
        page_size: Optional[int] = ...,
        max_items: Optional[int] = ...,
        jmespath_query: Optional[str] = ...,
    ) -> List[Dict[str, Any]]: ...

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
        to_path: Optional[Union[str, Path]],
        **download_kwargs: Any,
    ) -> Path: ...

    def download_object_to_path(
        self,
        from_path: str,
        to_path: Optional[Union[str, Path]],
        **download_kwargs: Any,
    ) -> Path: ...

    async def adownload_object_to_file_object(
        self,
        from_path: str,
        to_file_object: BinaryIO,
        **download_kwargs: Any,
    ) -> BinaryIO: ...

    def download_object_to_file_object(
        self,
        from_path: str,
        to_file_object: BinaryIO,
        **download_kwargs: Any,
    ) -> BinaryIO: ...

    async def adownload_folder_to_path(
        self,
        from_folder: str,
        to_folder: Optional[Union[str, Path]] = ...,
        **download_kwargs: Any,
    ) -> Path: ...

    def download_folder_to_path(
        self,
        from_folder: str,
        to_folder: Optional[Union[str, Path]] = ...,
        **download_kwargs: Any,
    ) -> Path: ...

    async def astream_from(
        self,
        bucket: S3Bucket,
        from_path: str,
        to_path: Optional[str] = ...,
        **upload_kwargs: Any,
    ) -> str: ...

    def stream_from(
        self,
        bucket: S3Bucket,
        from_path: str,
        to_path: Optional[str] = ...,
        **upload_kwargs: Any,
    ) -> str: ...

    async def aupload_from_path(
        self,
        from_path: Union[str, Path],
        to_path: Optional[str] = ...,
        **upload_kwargs: Any,
    ) -> str: ...

    def upload_from_path(
        self,
        from_path: Union[str, Path],
        to_path: Optional[str] = ...,
        **upload_kwargs: Any,
    ) -> str: ...

    async def aupload_from_file_object(
        self,
        from_file_object: BinaryIO,
        to_path: str,
        **upload_kwargs: Any,
    ) -> str: ...

    def upload_from_file_object(
        self,
        from_file_object: BinaryIO,
        to_path: str,
        **upload_kwargs: Any,
    ) -> str: ...

    async def aupload_from_folder(
        self,
        from_folder: Union[str, Path],
        to_folder: Optional[str] = ...,
        **upload_kwargs: Any,
    ) -> Optional[str]: ...

    def upload_from_folder(
        self,
        from_folder: Union[str, Path],
        to_folder: Optional[str] = ...,
        **upload_kwargs: Any,
    ) -> Optional[str]: ...

    def copy_object(
        self,
        from_path: Union[str, Path],
        to_path: Union[str, Path],
        to_bucket: Optional[Union[S3Bucket, str]] = ...,
        **copy_kwargs: Any,
    ) -> str: ...

    def _move_object_setup(
        self,
        from_path: Union[str, Path],
        to_path: Union[str, Path],
        to_bucket: Optional[Union[S3Bucket, str]] = ...,
    ) -> Tuple[str, str, str, str]: ...

    async def amove_object(
        self,
        from_path: Union[str, Path],
        to_path: Union[str, Path],
        to_bucket: Optional[Union[S3Bucket, str]] = ...,
    ) -> str: ...

    def move_object(
        self,
        from_path: Union[str, Path],
        to_path: Union[str, Path],
        to_bucket: Optional[Union[S3Bucket, str]] = ...,
    ) -> str: ...