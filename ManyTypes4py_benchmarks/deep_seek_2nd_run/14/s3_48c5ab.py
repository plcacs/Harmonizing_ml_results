"""Tasks for interacting with AWS S3"""
import asyncio
import io
import os
import uuid
from pathlib import Path
from typing import Any, BinaryIO, Dict, List, Optional, Tuple, Union, get_args, cast
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
    aws_client_parameters: AwsClientParameters = AwsClientParameters()
) -> bytes:
    """
    Downloads an object with a given key from a given S3 bucket.

    Added in prefect-aws==0.5.3.

    Args:
        bucket: Name of bucket to download object from. Required if a default value was
            not supplied when creating the task.
        key: Key of object to download. Required if a default value was not supplied
            when creating the task.
        aws_credentials: Credentials to use for authentication with AWS.
        aws_client_parameters: Custom parameter for the boto3 client initialization.

    Returns:
        A `bytes` representation of the downloaded object.
    """
    logger = get_run_logger()
    logger.info('Downloading object from bucket %s with key %s', bucket, key)
    s3_client = aws_credentials.get_boto3_session().client('s3', **aws_client_parameters.get_params_override())
    stream = io.BytesIO()
    await run_sync_in_worker_thread(s3_client.download_fileobj, Bucket=bucket, Key=key, Fileobj=stream)
    stream.seek(0)
    output = stream.read()
    return output

@async_dispatch(adownload_from_bucket)
@task
def download_from_bucket(
    bucket: str,
    key: str,
    aws_credentials: Union[AwsCredentials, MinIOCredentials],
    aws_client_parameters: AwsClientParameters = AwsClientParameters()
) -> bytes:
    """
    Downloads an object with a given key from a given S3 bucket.
    """
    logger = get_run_logger()
    logger.info('Downloading object from bucket %s with key %s', bucket, key)
    s3_client = aws_credentials.get_boto3_session().client('s3', **aws_client_parameters.get_params_override())
    stream = io.BytesIO()
    s3_client.download_fileobj(Bucket=bucket, Key=key, Fileobj=stream)
    stream.seek(0)
    output = stream.read()
    return output

s3_download = download_from_bucket

@task
async def aupload_to_bucket(
    data: bytes,
    bucket: str,
    aws_credentials: Union[AwsCredentials, MinIOCredentials],
    aws_client_parameters: AwsClientParameters = AwsClientParameters(),
    key: Optional[str] = None
) -> str:
    """
    Asynchronously uploads data to an S3 bucket.
    """
    logger = get_run_logger()
    key = key or str(uuid.uuid4())
    logger.info('Uploading object to bucket %s with key %s', bucket, key)
    s3_client = aws_credentials.get_boto3_session().client('s3', **aws_client_parameters.get_params_override())
    stream = io.BytesIO(data)
    await run_sync_in_worker_thread(s3_client.upload_fileobj, stream, Bucket=bucket, Key=key)
    return key

@async_dispatch(aupload_to_bucket)
@task
def upload_to_bucket(
    data: bytes,
    bucket: str,
    aws_credentials: Union[AwsCredentials, MinIOCredentials],
    aws_client_parameters: AwsClientParameters = AwsClientParameters(),
    key: Optional[str] = None
) -> str:
    """
    Uploads data to an S3 bucket.
    """
    logger = get_run_logger()
    key = key or str(uuid.uuid4())
    logger.info('Uploading object to bucket %s with key %s', bucket, key)
    s3_client = aws_credentials.get_boto3_session().client('s3', **aws_client_parameters.get_params_override())
    stream = io.BytesIO(data)
    s3_client.upload_fileobj(stream, Bucket=bucket, Key=key)
    return key

s3_upload = upload_to_bucket

@task
async def acopy_objects(
    source_path: Union[str, Path],
    target_path: Union[str, Path],
    source_bucket_name: str,
    aws_credentials: Union[AwsCredentials, MinIOCredentials],
    target_bucket_name: Optional[str] = None,
    **copy_kwargs: Any
) -> str:
    """Asynchronously uses S3's internal CopyObject to copy objects within or between buckets."""
    logger = get_run_logger()
    s3_client = aws_credentials.get_s3_client()
    target_bucket_name = target_bucket_name or source_bucket_name
    logger.info('Copying object from bucket %s with key %s to bucket %s with key %s', source_bucket_name, source_path, target_bucket_name, target_path)
    await run_sync_in_worker_thread(
        s3_client.copy_object,
        CopySource={'Bucket': source_bucket_name, 'Key': str(source_path)},
        Bucket=target_bucket_name,
        Key=str(target_path),
        **copy_kwargs
    )
    return str(target_path)

@async_dispatch(acopy_objects)
@task
def copy_objects(
    source_path: Union[str, Path],
    target_path: Union[str, Path],
    source_bucket_name: str,
    aws_credentials: Union[AwsCredentials, MinIOCredentials],
    target_bucket_name: Optional[str] = None,
    **copy_kwargs: Any
) -> str:
    """Uses S3's internal CopyObject to copy objects within or between buckets."""
    logger = get_run_logger()
    s3_client = aws_credentials.get_s3_client()
    target_bucket_name = target_bucket_name or source_bucket_name
    logger.info('Copying object from bucket %s with key %s to bucket %s with key %s', source_bucket_name, source_path, target_bucket_name, target_path)
    s3_client.copy_object(
        CopySource={'Bucket': source_bucket_name, 'Key': str(source_path)},
        Bucket=target_bucket_name,
        Key=str(target_path),
        **copy_kwargs
    )
    return str(target_path)

s3_copy = copy_objects

@task
async def amove_objects(
    source_path: Union[str, Path],
    target_path: Union[str, Path],
    source_bucket_name: str,
    aws_credentials: Union[AwsCredentials, MinIOCredentials],
    target_bucket_name: Optional[str] = None
) -> str:
    """
    Asynchronously moves an object from one S3 location to another.
    """
    logger = get_run_logger()
    s3_client = aws_credentials.get_s3_client()
    target_bucket_name = target_bucket_name or source_bucket_name
    logger.info('Moving object from s3://%s/%s s3://%s/%s', source_bucket_name, source_path, target_bucket_name, target_path)
    await run_sync_in_worker_thread(
        s3_client.copy_object,
        Bucket=target_bucket_name,
        CopySource={'Bucket': source_bucket_name, 'Key': str(source_path)},
        Key=str(target_path)
    )
    await run_sync_in_worker_thread(
        s3_client.delete_object,
        Bucket=source_bucket_name,
        Key=str(source_path)
    )
    return str(target_path)

@async_dispatch(amove_objects)
@task
def move_objects(
    source_path: Union[str, Path],
    target_path: Union[str, Path],
    source_bucket_name: str,
    aws_credentials: Union[AwsCredentials, MinIOCredentials],
    target_bucket_name: Optional[str] = None
) -> str:
    """
    Move an object from one S3 location to another.
    """
    logger = get_run_logger()
    s3_client = aws_credentials.get_s3_client()
    target_bucket_name = target_bucket_name or source_bucket_name
    logger.info('Moving object from s3://%s/%s s3://%s/%s', source_bucket_name, source_path, target_bucket_name, target_path)
    s3_client.copy_object(
        Bucket=target_bucket_name,
        CopySource={'Bucket': source_bucket_name, 'Key': str(source_path)},
        Key=str(target_path)
    )
    s3_client.delete_object(Bucket=source_bucket_name, Key=str(source_path))
    return str(target_path)

s3_move = move_objects

def _list_objects_sync(page_iterator: PageIterator) -> List[Dict[str, Any]]:
    """
    Synchronous method to collect S3 objects into a list
    """
    return [content for page in page_iterator for content in page.get('Contents', [])]

@task
async def alist_objects(
    bucket: str,
    aws_credentials: Union[AwsCredentials, MinIOCredentials],
    aws_client_parameters: AwsClientParameters = AwsClientParameters(),
    prefix: str = '',
    delimiter: str = '',
    page_size: Optional[int] = None,
    max_items: Optional[int] = None,
    jmespath_query: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Asynchronously lists details of objects in a given S3 bucket.
    """
    logger = get_run_logger()
    logger.info('Listing objects in bucket %s with prefix %s', bucket, prefix)
    s3_client = aws_credentials.get_boto3_session().client('s3', **aws_client_parameters.get_params_override())
    paginator = s3_client.get_paginator('list_objects_v2')
    page_iterator = paginator.paginate(
        Bucket=bucket,
        Prefix=prefix,
        Delimiter=delimiter,
        PaginationConfig={'PageSize': page_size, 'MaxItems': max_items}
    )
    if jmespath_query:
        page_iterator = page_iterator.search(f'{jmespath_query} | {{Contents: @}}')
    return await run_sync_in_worker_thread(_list_objects_sync, page_iterator)

@async_dispatch(alist_objects)
@task
def list_objects(
    bucket: str,
    aws_credentials: Union[AwsCredentials, MinIOCredentials],
    aws_client_parameters: AwsClientParameters = AwsClientParameters(),
    prefix: str = '',
    delimiter: str = '',
    page_size: Optional[int] = None,
    max_items: Optional[int] = None,
    jmespath_query: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Lists details of objects in a given S3 bucket.
    """
    logger = get_run_logger()
    logger.info('Listing objects in bucket %s with prefix %s', bucket, prefix)
    s3_client = aws_credentials.get_boto3_session().client('s3', **aws_client_parameters.get_params_override())
    paginator = s3_client.get_paginator('list_objects_v2')
    page_iterator = paginator.paginate(
        Bucket=bucket,
        Prefix=prefix,
        Delimiter=delimiter,
        PaginationConfig={'PageSize': page_size, 'MaxItems': max_items}
    )
    if jmespath_query:
        page_iterator = page_iterator.search(f'{jmespath_query} | {{Contents: @}}')
    return _list_objects_sync(page_iterator)

s3_list_objects = list_objects

class S3Bucket(WritableFileSystem, WritableDeploymentStorage, ObjectStorageBlock):
    """
    Block used to store data using AWS S3 or S3-compatible object storage like MinIO.
    """
    _logo_url: str = 'https://cdn.sanity.io/images/3ugk85nk/production/d74b16fe84ce626345adf235a47008fea2869a60-225x225.png'
    _block_type_name: str = 'S3 Bucket'
    _documentation_url: str = 'https://docs.prefect.io/integrations/prefect-aws'
    bucket_name: str = Field(default=..., description='Name of your bucket.')
    credentials: Union[AwsCredentials, MinIOCredentials] = Field(
        default_factory=AwsCredentials,
        description='A block containing your credentials to AWS or MinIO.'
    )
    bucket_folder: str = Field(
        default='',
        description='A default path to a folder within the S3 bucket to use for reading and writing objects.'
    )

    @field_validator('credentials', mode='before')
    def validate_credentials(cls, value: Any, field: Any) -> Union[AwsCredentials, MinIOCredentials]:
        if isinstance(value, dict):
            block_type_slug = value.pop('block_type_slug', None)
            if block_type_slug:
                credential_classes = (lookup_type(CredentialsBlock, dispatch_key=block_type_slug),)
            else:
                credential_classes = get_args(cls.model_fields['credentials'].annotation)
            for credentials_cls in credential_classes:
                try:
                    return credentials_cls(**value)
                except ValueError:
                    pass
            valid_classes = ', '.join((c.__name__ for c in credential_classes))
            raise ValueError(f'Invalid credentials data: does not match any credential type. Valid types: {valid_classes}')
        return cast(Union[AwsCredentials, MinIOCredentials], value)

    @property
    def basepath(self) -> str:
        """
        The base path of the S3 bucket.
        """
        return self.bucket_folder

    @basepath.setter
    def basepath(self, value: str) -> None:
        self.bucket_folder = value

    def _resolve_path(self, path: Union[str, Path]) -> str:
        """
        A helper function used in write_path to join `self.basepath` and `path`.
        """
        path = (Path(self.bucket_folder) / path).as_posix() if self.bucket_folder else str(path)
        return path

    def _get_s3_client(self) -> Any:
        """
        Authenticate MinIO credentials or AWS credentials and return an S3 client.
        """
        return self.credentials.get_client('s3')

    def _get_bucket_resource(self) -> Any:
        """
        Retrieves boto3 resource object for the configured bucket
        """
        params_override = self.credentials.aws_client_parameters.get_params_override()
        bucket = self.credentials.get_boto3_session().resource('s3', **params_override).Bucket(self.bucket_name)
        return bucket

    async def aget_directory(self, from_path: Optional[str] = None, local_path: Optional[str] = None) -> None:
        """
        Asynchronously copies a folder from the configured S3 bucket to a local directory.
        """
        bucket_folder = self.bucket_folder
        if from_path is None:
            from_path = str(bucket_folder) if bucket_folder else ''
        if local_path is None:
            local_path = str(Path('.').absolute())
        else:
            local_path = str(Path(local_path).expanduser())
        bucket = self._get_bucket_resource()
        for obj in bucket.objects.filter(Prefix=from_path):
            if obj.key[-1] == '/':
                continue
            target = os.path.join(local_path, os.path.relpath(obj.key, from_path))
            os.makedirs(os.path.dirname(target), exist_ok=True)
            await run_sync_in_worker_thread(bucket.download_file, obj.key, target)

    @async_dispatch(aget_directory)
    def get_directory(self, from_path: Optional[str] = None, local_path: Optional[str] = None) -> None:
        """
        Copies a folder from the configured S3 bucket to a local directory.
        """
        bucket_folder = self.bucket_folder
        if from_path is None:
            from_path = str(bucket_folder) if bucket_folder else ''
        if local_path is None:
            local_path = str(Path('.').absolute())
        else:
            local_path = str(Path(local_path).expanduser())
        bucket = self._get_bucket_resource()
        for obj in bucket.objects.filter(Prefix=from_path):
            if obj.key[-1] == '/':
                continue
            target = os.path.join(local_path, os.path.relpath(obj.key, from_path))
            os.makedirs(os.path.dirname(target), exist_ok=True)
            bucket.download_file(obj.key, target)

    async def aput_directory(
        self,
        local_path: Optional[str] = None,
        to_path: Optional[str] = None,
        ignore_file: Optional[str] = None
    ) -> int:
        """
        Asynchronously uploads a directory from a given local path to the configured S3 bucket.
        """
        to_path = '' if to_path is None else to_path
        if local_path is None:
            local_path = '.'
        included_files = None
        if ignore_file:
            with open(ignore_file, 'r') as f:
                ignore_patterns = f.readlines()
            included_files = filter_files(local_path, ignore_patterns)
        uploaded_file_count = 0
        for local_file_path in Path(local_path).expanduser().rglob('*'):
            if included_files is not None and str(local_file_path.relative_to(local_path)) not in included_files:
                continue
            elif not local_file_path.is_dir():
                remote_file_path = Path(to_path) / local_file_path.