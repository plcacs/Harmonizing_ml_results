```python
"""Tasks for interacting with AWS S3"""
import asyncio
import io
import os
import uuid
from pathlib import Path
from typing import Any, BinaryIO, Dict, List, Optional, Tuple, Union, get_args
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
    aws_credentials: AwsCredentials,
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

    Example:
        Download a file from an S3 bucket:

        ```python
        from prefect import flow
        from prefect_aws import AwsCredentials
        from prefect_aws.s3 import adownload_from_bucket

        @flow
        async def example_download_from_bucket_flow():
            aws_credentials = AwsCredentials(
                aws_access_key_id="acccess_key_id",
                aws_secret_access_key="secret_access_key"
            )
            data = await adownload_from_bucket(
                bucket="bucket",
                key="key",
                aws_credentials=aws_credentials,
            )

        await example_download_from_bucket_flow()
        ```
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
    aws_credentials: AwsCredentials,
    aws_client_parameters: AwsClientParameters = AwsClientParameters()
) -> bytes:
    """
    Downloads an object with a given key from a given S3 bucket.

    Args:
        bucket: Name of bucket to download object from. Required if a default value was
            not supplied when creating the task.
        key: Key of object to download. Required if a default value was not supplied
            when creating the task.
        aws_credentials: Credentials to use for authentication with AWS.
        aws_client_parameters: Custom parameter for the boto3 client initialization.


    Returns:
        A `bytes` representation of the downloaded object.

    Example:
        Download a file from an S3 bucket:

        ```python
        from prefect import flow
        from prefect_aws import AwsCredentials
        from prefect_aws.s3 import download_from_bucket


        @flow
        async def example_download_from_bucket_flow():
            aws_credentials = AwsCredentials(
                aws_access_key_id="acccess_key_id",
                aws_secret_access_key="secret_access_key"
            )
            data = download_from_bucket(
                bucket="bucket",
                key="key",
                aws_credentials=aws_credentials,
            )

        example_download_from_bucket_flow()
        ```
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
    aws_credentials: AwsCredentials,
    aws_client_parameters: AwsClientParameters = AwsClientParameters(),
    key: Optional[str] = None
) -> str:
    """
    Asynchronously uploads data to an S3 bucket.

    Added in prefect-aws==0.5.3.

    Args:
        data: Bytes representation of data to upload to S3.
        bucket: Name of bucket to upload data to. Required if a default value was not
            supplied when creating the task.
        aws_credentials: Credentials to use for authentication with AWS.
        aws_client_parameters: Custom parameter for the boto3 client initialization..
        key: Key of object to download. Defaults to a UUID string.

    Returns:
        The key of the uploaded object

    Example:
        Read and upload a file to an S3 bucket:

        ```python
        from prefect import flow
        from prefect_aws import AwsCredentials
        from prefect_aws.s3 import aupload_to_bucket


        @flow
        async def example_s3_upload_flow():
            aws_credentials = AwsCredentials(
                aws_access_key_id="acccess_key_id",
                aws_secret_access_key="secret_access_key"
            )
            with open("data.csv", "rb") as file:
                key = await aupload_to_bucket(
                    bucket="bucket",
                    key="data.csv",
                    data=file.read(),
                    aws_credentials=aws_credentials,
                )

        await example_s3_upload_flow()
        ```
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
    aws_credentials: AwsCredentials,
    aws_client_parameters: AwsClientParameters = AwsClientParameters(),
    key: Optional[str] = None
) -> str:
    """
    Uploads data to an S3 bucket.

    Args:
        data: Bytes representation of data to upload to S3.
        bucket: Name of bucket to upload data to. Required if a default value was not
            supplied when creating the task.
        aws_credentials: Credentials to use for authentication with AWS.
        aws_client_parameters: Custom parameter for the boto3 client initialization..
        key: Key of object to download. Defaults to a UUID string.

    Returns:
        The key of the uploaded object

    Example:
        Read and upload a file to an S3 bucket:

        ```python
        from prefect import flow
        from prefect_aws import AwsCredentials
        from prefect_aws.s3 import upload_to_bucket


        @flow
        async def example_s3_upload_flow():
            aws_credentials = AwsCredentials(
                aws_access_key_id="acccess_key_id",
                aws_secret_access_key="secret_access_key"
            )
            with open("data.csv", "rb") as file:
                key = upload_to_bucket(
                    bucket="bucket",
                    key="data.csv",
                    data=file.read(),
                    aws_credentials=aws_credentials,
                )

        example_s3_upload_flow()
        ```
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
    aws_credentials: AwsCredentials,
    target_bucket_name: Optional[str] = None,
    **copy_kwargs: Any
) -> str:
    """Asynchronously uses S3's internal
    [CopyObject](https://docs.aws.amazon.com/AmazonS3/latest/API/API_CopyObject.html)
    to copy objects within or between buckets. To copy objects between buckets, the
    credentials must have permission to read the source object and write to the target
    object. If the credentials do not have those permissions, try using
    `S3Bucket.stream_from`.

    Added in prefect-aws==0.5.3.

    Args:
        source_path: The path to the object to copy. Can be a string or `Path`.
        target_path: The path to copy the object to. Can be a string or `Path`.
        source_bucket_name: The bucket to copy the object from.
        aws_credentials: Credentials to use for authentication with AWS.
        target_bucket_name: The bucket to copy the object to. If not provided, defaults
            to `source_bucket`.
        **copy_kwargs: Additional keyword arguments to pass to `S3Client.copy_object`.

    Returns:
        The path that the object was copied to. Excludes the bucket name.

    Examples:

        Copy notes.txt from s3://my-bucket/my_folder/notes.txt to
        s3://my-bucket/my_folder/notes_copy.txt.

        ```python
        from prefect import flow
        from prefect_aws import AwsCredentials
        from prefect_aws.s3 import acopy_objects

        aws_credentials = AwsCredentials.load("my-creds")

        @flow
        async def example_copy_flow():
            await acopy_objects(
                source_path="my_folder/notes.txt",
                target_path="my_folder/notes_copy.txt",
                source_bucket_name="my-bucket",
                aws_credentials=aws_credentials,
            )

        await example_copy_flow()
        ```

        Copy notes.txt from s3://my-bucket/my_folder/notes.txt to
        s3://other-bucket/notes_copy.txt.

        ```python
        from prefect import flow
        from prefect_aws import AwsCredentials
        from prefect_aws.s3 import acopy_objects

        aws_credentials = AwsCredentials.load("shared-creds")

        @flow
        async def example_copy_flow():
            await acopy_objects(
                source_path="my_folder/notes.txt",
                target_path="notes_copy.txt",
                source_bucket_name="my-bucket",
                aws_credentials=aws_credentials,
                target_bucket_name="other-bucket",
            )

        await example_copy_flow()
        ```

    """
    logger = get_run_logger()
    s3_client = aws_credentials.get_s3_client()
    target_bucket_name = target_bucket_name or source_bucket_name
    logger.info('Copying object from bucket %s with key %s to bucket %s with key %s', source_bucket_name, source_path, target_bucket_name, target_path)
    await run_sync_in_worker_thread(s3_client.copy_object, CopySource={'Bucket': source_bucket_name, 'Key': source_path}, Bucket=target_bucket_name, Key=target_path, **copy_kwargs)
    return target_path

@async_dispatch(acopy_objects)
@task
def copy_objects(
    source_path: Union[str, Path],
    target_path: Union[str, Path],
    source_bucket_name: str,
    aws_credentials: AwsCredentials,
    target_bucket_name: Optional[str] = None,
    **copy_kwargs: Any
) -> str:
    """Uses S3's internal
    [CopyObject](https://docs.aws.amazon.com/AmazonS3/latest/API/API_CopyObject.html)
    to copy objects within or between buckets. To copy objects between buckets, the
    credentials must have permission to read the source object and write to the target
    object. If the credentials do not have those permissions, try using
    `S3Bucket.stream_from`.

    Args:
        source_path: The path to the object to copy. Can be a string or `Path`.
        target_path: The path to copy the object to. Can be a string or `Path`.
        source_bucket_name: The bucket to copy the object from.
        aws_credentials: Credentials to use for authentication with AWS.
        target_bucket_name: The bucket to copy the object to. If not provided, defaults
            to `source_bucket`.
        **copy_kwargs: Additional keyword arguments to pass to `S3Client.copy_object`.

    Returns:
        The path that the object was copied to. Excludes the bucket name.

    Examples:

        Copy notes.txt from s3://my-bucket/my_folder/notes.txt to
        s3://my-bucket/my_folder/notes_copy.txt.

        ```python
        from prefect import flow
        from prefect_aws import AwsCredentials
        from prefect_aws.s3 import copy_objects

        aws_credentials = AwsCredentials.load("my-creds")

        @flow
        def example_copy_flow():
            copy_objects(
                source_path="my_folder/notes.txt",
                target_path="my_folder/notes_copy.txt",
                source_bucket_name="my-bucket",
                aws_credentials=aws_credentials,
            )

        example_copy_flow()
        ```

        Copy notes.txt from s3://my-bucket/my_folder/notes.txt to
        s3://other-bucket/notes_copy.txt.

        ```python
        from prefect import flow
        from prefect_aws import AwsCredentials
        from prefect_aws.s3 import copy_objects

        aws_credentials = AwsCredentials.load("shared-creds")

        @flow
        def example_copy_flow():
            copy_objects(
                source_path="my_folder/notes.txt",
                target_path="notes_copy.txt",
                source_bucket_name="my-bucket",
                aws_credentials=aws_credentials,
                target_bucket_name="other-bucket",
            )

        example_copy_flow()
        ```

    """
    logger = get_run_logger()
    s3_client = aws_credentials.get_s3_client()
    target_bucket_name = target_bucket_name or source_bucket_name
    logger.info('Copying object from bucket %s with key %s to bucket %s with key %s', source_bucket_name, source_path, target_bucket_name, target_path)
    s3_client.copy_object(CopySource={'Bucket': source_bucket_name, 'Key': source_path}, Bucket=target_bucket_name, Key=target_path, **copy_kwargs)
    return target_path
s3_copy = copy_objects

@task
async def amove_objects(
    source_path: Union[str, Path],
    target_path: Union[str, Path],
    source_bucket_name: str,
    aws_credentials: AwsCredentials,
    target_bucket_name: Optional[str] = None
) -> str:
    """
    Asynchronously moves an object from one S3 location to another. To move objects
    between buckets, the credentials must have permission to read and delete the source
    object and write to the target object. If the credentials do not have those
    permissions, this method will raise an error. If the credentials have permission to
    read the source object but not delete it, the object will be copied but not deleted.

    Added in prefect-aws==0.5.3.

    Args:
        source_path: The path of the object to move
        target_path: The path to move the object to
        source_bucket_name: The name of the bucket containing the source object
        aws_credentials: Credentials to use for authentication with AWS.
        target_bucket_name: The bucket to copy the object to. If not provided, defaults
            to `source_bucket`.

    Returns:
        The path that the object was moved to. Excludes the bucket name.
    """
    logger = get_run_logger()
    s3_client = aws_credentials.get_s3_client()
    target_bucket_name = target_bucket_name or source_bucket_name
    logger.info('Moving object from s3://%s/%s s3://%s/%s', source_bucket_name, source_path, target_bucket_name, target_path)
    await run_sync_in_worker_thread(s3_client.copy_object, Bucket=target_bucket_name, CopySource={'Bucket': source_bucket_name, 'Key': source_path}, Key=target_path)
    await run_sync_in_worker_thread(s3_client.delete_object, Bucket=source_bucket_name, Key=source_path)
    return target_path

@async_dispatch(amove_objects)
@task
def move_objects(
    source_path: Union[str, Path],
    target_path: Union[str, Path],
    source_bucket_name: str,
    aws_credentials: AwsCredentials,
    target_bucket_name: Optional[str] = None
) -> str:
    """
    Move an object from one S3 location to another. To move objects between buckets,
    the credentials must have permission to read and delete the source object and write
    to the target object. If the