#!/usr/bin/env python3
"""
Tasks for interacting with AWS S3
"""
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
    aws_client_parameters: AwsClientParameters = AwsClientParameters(),
) -> bytes:
    """
    Downloads an object with a given key from a given S3 bucket.
    """
    logger = get_run_logger()
    logger.info("Downloading object from bucket %s with key %s", bucket, key)
    s3_client = aws_credentials.get_boto3_session().client(
        "s3", **aws_client_parameters.get_params_override()
    )
    stream = io.BytesIO()
    await run_sync_in_worker_thread(s3_client.download_fileobj, Bucket=bucket, Key=key, Fileobj=stream)
    stream.seek(0)
    output: bytes = stream.read()
    return output


@async_dispatch(adownload_from_bucket)
@task
def download_from_bucket(
    bucket: str,
    key: str,
    aws_credentials: AwsCredentials,
    aws_client_parameters: AwsClientParameters = AwsClientParameters(),
) -> bytes:
    """
    Downloads an object with a given key from a given S3 bucket (synchronous version).
    """
    logger = get_run_logger()
    logger.info("Downloading object from bucket %s with key %s", bucket, key)
    s3_client = aws_credentials.get_boto3_session().client(
        "s3", **aws_client_parameters.get_params_override()
    )
    stream = io.BytesIO()
    s3_client.download_fileobj(Bucket=bucket, Key=key, Fileobj=stream)
    stream.seek(0)
    output: bytes = stream.read()
    return output


s3_download = download_from_bucket


@task
async def aupload_to_bucket(
    data: bytes,
    bucket: str,
    aws_credentials: AwsCredentials,
    aws_client_parameters: AwsClientParameters = AwsClientParameters(),
    key: Optional[str] = None,
) -> str:
    """
    Asynchronously uploads data to an S3 bucket.
    """
    logger = get_run_logger()
    key = key or str(uuid.uuid4())
    logger.info("Uploading object to bucket %s with key %s", bucket, key)
    s3_client = aws_credentials.get_boto3_session().client(
        "s3", **aws_client_parameters.get_params_override()
    )
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
    key: Optional[str] = None,
) -> str:
    """
    Uploads data to an S3 bucket (synchronous version).
    """
    logger = get_run_logger()
    key = key or str(uuid.uuid4())
    logger.info("Uploading object to bucket %s with key %s", bucket, key)
    s3_client = aws_credentials.get_boto3_session().client(
        "s3", **aws_client_parameters.get_params_override()
    )
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
    **copy_kwargs: Any,
) -> str:
    """
    Asynchronously copies objects within or between buckets.
    """
    logger = get_run_logger()
    s3_client = aws_credentials.get_s3_client()
    target_bucket_name = target_bucket_name or source_bucket_name
    logger.info(
        "Copying object from bucket %s with key %s to bucket %s with key %s",
        source_bucket_name,
        source_path,
        target_bucket_name,
        target_path,
    )
    await run_sync_in_worker_thread(
        s3_client.copy_object,
        CopySource={"Bucket": source_bucket_name, "Key": source_path},
        Bucket=target_bucket_name,
        Key=target_path,
        **copy_kwargs,
    )
    return str(target_path)


@async_dispatch(acopy_objects)
@task
def copy_objects(
    source_path: Union[str, Path],
    target_path: Union[str, Path],
    source_bucket_name: str,
    aws_credentials: AwsCredentials,
    target_bucket_name: Optional[str] = None,
    **copy_kwargs: Any,
) -> str:
    """
    Copies objects within or between buckets (synchronous version).
    """
    logger = get_run_logger()
    s3_client = aws_credentials.get_s3_client()
    target_bucket_name = target_bucket_name or source_bucket_name
    logger.info(
        "Copying object from bucket %s with key %s to bucket %s with key %s",
        source_bucket_name,
        source_path,
        target_bucket_name,
        target_path,
    )
    s3_client.copy_object(
        CopySource={"Bucket": source_bucket_name, "Key": source_path},
        Bucket=target_bucket_name,
        Key=target_path,
        **copy_kwargs,
    )
    return str(target_path)


s3_copy = copy_objects


@task
async def amove_objects(
    source_path: Union[str, Path],
    target_path: Union[str, Path],
    source_bucket_name: str,
    aws_credentials: AwsCredentials,
    target_bucket_name: Optional[str] = None,
) -> str:
    """
    Asynchronously moves an object from one S3 location to another.
    """
    logger = get_run_logger()
    s3_client = aws_credentials.get_s3_client()
    target_bucket_name = target_bucket_name or source_bucket_name
    logger.info(
        "Moving object from s3://%s/%s s3://%s/%s",
        source_bucket_name,
        source_path,
        target_bucket_name,
        target_path,
    )
    await run_sync_in_worker_thread(
        s3_client.copy_object,
        Bucket=target_bucket_name,
        CopySource={"Bucket": source_bucket_name, "Key": source_path},
        Key=target_path,
    )
    await run_sync_in_worker_thread(s3_client.delete_object, Bucket=source_bucket_name, Key=source_path)
    return str(target_path)


@async_dispatch(amove_objects)
@task
def move_objects(
    source_path: Union[str, Path],
    target_path: Union[str, Path],
    source_bucket_name: str,
    aws_credentials: AwsCredentials,
    target_bucket_name: Optional[str] = None,
) -> str:
    """
    Moves an object from one S3 location to another (synchronous version).
    """
    logger = get_run_logger()
    s3_client = aws_credentials.get_s3_client()
    target_bucket_name = target_bucket_name or source_bucket_name
    logger.info(
        "Moving object from s3://%s/%s s3://%s/%s",
        source_bucket_name,
        source_path,
        target_bucket_name,
        target_path,
    )
    s3_client.copy_object(
        Bucket=target_bucket_name,
        CopySource={"Bucket": source_bucket_name, "Key": source_path},
        Key=target_path,
    )
    s3_client.delete_object(Bucket=source_bucket_name, Key=source_path)
    return str(target_path)


s3_move = move_objects


def _list_objects_sync(page_iterator: PageIterator) -> List[Dict[str, Any]]:
    """
    Synchronous method to collect S3 objects into a list.
    """
    return [content for page in page_iterator for content in page.get("Contents", [])]


@task
async def alist_objects(
    bucket: str,
    aws_credentials: AwsCredentials,
    aws_client_parameters: AwsClientParameters = AwsClientParameters(),
    prefix: str = "",
    delimiter: str = "",
    page_size: Optional[int] = None,
    max_items: Optional[int] = None,
    jmespath_query: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Asynchronously lists details of objects in a given S3 bucket.
    """
    logger = get_run_logger()
    logger.info("Listing objects in bucket %s with prefix %s", bucket, prefix)
    s3_client = aws_credentials.get_boto3_session().client(
        "s3", **aws_client_parameters.get_params_override()
    )
    paginator = s3_client.get_paginator("list_objects_v2")
    page_iterator = paginator.paginate(
        Bucket=bucket,
        Prefix=prefix,
        Delimiter=delimiter,
        PaginationConfig={"PageSize": page_size, "MaxItems": max_items},
    )
    if jmespath_query:
        page_iterator = page_iterator.search(f"{jmespath_query} | {{Contents: @}}")
    return await run_sync_in_worker_thread(_list_objects_sync, page_iterator)


@async_dispatch(alist_objects)
@task
def list_objects(
    bucket: str,
    aws_credentials: AwsCredentials,
    aws_client_parameters: AwsClientParameters = AwsClientParameters(),
    prefix: str = "",
    delimiter: str = "",
    page_size: Optional[int] = None,
    max_items: Optional[int] = None,
    jmespath_query: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Lists details of objects in a given S3 bucket (synchronous version).
    """
    logger = get_run_logger()
    logger.info("Listing objects in bucket %s with prefix %s", bucket, prefix)
    s3_client = aws_credentials.get_boto3_session().client(
        "s3", **aws_client_parameters.get_params_override()
    )
    paginator = s3_client.get_paginator("list_objects_v2")
    page_iterator = paginator.paginate(
        Bucket=bucket,
        Prefix=prefix,
        Delimiter=delimiter,
        PaginationConfig={"PageSize": page_size, "MaxItems": max_items},
    )
    if jmespath_query:
        page_iterator = page_iterator.search(f"{jmespath_query} | {{Contents: @}}")
    return _list_objects_sync(page_iterator)


s3_list_objects = list_objects


class S3Bucket(WritableFileSystem, WritableDeploymentStorage, ObjectStorageBlock):
    """
    Block used to store data using AWS S3 or S3-compatible object storage like MinIO.
    """
    _logo_url: str = "https://cdn.sanity.io/images/3ugk85nk/production/d74b16fe84ce626345adf235a47008fea2869a60-225x225.png"
    _block_type_name: str = "S3 Bucket"
    _documentation_url: str = "https://docs.prefect.io/integrations/prefect-aws"
    bucket_name: str = Field(..., description="Name of your bucket.")
    credentials: Union[AwsCredentials, MinIOCredentials] = Field(
        default_factory=AwsCredentials,
        description="A block containing your credentials to AWS or MinIO.",
    )
    bucket_folder: str = Field(default="", description="A default path to a folder within the S3 bucket to use for reading and writing objects.")

    @field_validator("credentials", mode="before")
    def validate_credentials(cls, value: Any, field: Any) -> Any:
        if isinstance(value, dict):
            block_type_slug = value.pop("block_type_slug", None)
            if block_type_slug:
                credential_classes = (lookup_type(CredentialsBlock, dispatch_key=block_type_slug),)
            else:
                credential_classes = get_args(cls.model_fields["credentials"].annotation)
            for credentials_cls in credential_classes:
                try:
                    return credentials_cls(**value)
                except ValueError:
                    pass
            valid_classes = ", ".join((c.__name__ for c in credential_classes))
            raise ValueError(f"Invalid credentials data: does not match any credential type. Valid types: {valid_classes}")
        return value

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
        Helper function to join bucket_folder and path.
        """
        path_str: str = str(path)
        if self.bucket_folder:
            return (Path(self.bucket_folder) / path_str).as_posix()
        return path_str

    def _get_s3_client(self) -> Any:
        """
        Authenticate credentials and return an S3 client.
        """
        return self.credentials.get_client("s3")

    def _get_bucket_resource(self) -> Any:
        """
        Retrieves boto3 resource object for the configured bucket.
        """
        params_override = self.credentials.aws_client_parameters.get_params_override()
        bucket = self.credentials.get_boto3_session().resource("s3", **params_override).Bucket(self.bucket_name)
        return bucket

    async def aget_directory(self, from_path: Optional[str] = None, local_path: Optional[str] = None) -> None:
        """
        Asynchronously copies a folder from the configured S3 bucket to a local directory.
        """
        bucket_folder = self.bucket_folder
        if from_path is None:
            from_path = str(bucket_folder) if bucket_folder else ""
        if local_path is None:
            local_path = str(Path(".").absolute())
        else:
            local_path = str(Path(local_path).expanduser())
        bucket = self._get_bucket_resource()
        for obj in bucket.objects.filter(Prefix=from_path):
            if obj.key[-1] == "/":
                continue
            target = os.path.join(local_path, os.path.relpath(obj.key, from_path))
            os.makedirs(os.path.dirname(target), exist_ok=True)
            await run_sync_in_worker_thread(bucket.download_file, obj.key, target)

    @async_dispatch(aget_directory)
    def get_directory(self, from_path: Optional[str] = None, local_path: Optional[str] = None) -> None:
        """
        Copies a folder from the configured S3 bucket to a local directory (synchronous version).
        """
        bucket_folder = self.bucket_folder
        if from_path is None:
            from_path = str(bucket_folder) if bucket_folder else ""
        if local_path is None:
            local_path = str(Path(".").absolute())
        else:
            local_path = str(Path(local_path).expanduser())
        bucket = self._get_bucket_resource()
        for obj in bucket.objects.filter(Prefix=from_path):
            if obj.key[-1] == "/":
                continue
            target = os.path.join(local_path, os.path.relpath(obj.key, from_path))
            os.makedirs(os.path.dirname(target), exist_ok=True)
            bucket.download_file(obj.key, target)

    async def aput_directory(
        self, local_path: Optional[str] = None, to_path: Optional[str] = None, ignore_file: Optional[str] = None
    ) -> int:
        """
        Asynchronously uploads a directory from a local path to the configured S3 bucket.
        """
        to_path = "" if to_path is None else to_path
        if local_path is None:
            local_path = "."
        included_files: Optional[List[str]] = None
        if ignore_file:
            with open(ignore_file, "r") as f:
                ignore_patterns = f.readlines()
            included_files = filter_files(local_path, ignore_patterns)
        uploaded_file_count = 0
        for local_file_path in Path(local_path).expanduser().rglob("*"):
            if included_files is not None and str(local_file_path.relative_to(local_path)) not in included_files:
                continue
            elif not local_file_path.is_dir():
                remote_file_path = Path(to_path) / local_file_path.relative_to(local_path)
                with open(local_file_path, "rb") as local_file:
                    local_file_content = local_file.read()
                await self.awrite_path(remote_file_path.as_posix(), content=local_file_content)
                uploaded_file_count += 1
        return uploaded_file_count

    @async_dispatch(aput_directory)
    def put_directory(
        self, local_path: Optional[str] = None, to_path: Optional[str] = None, ignore_file: Optional[str] = None
    ) -> int:
        """
        Uploads a directory from a local path to the configured S3 bucket (synchronous version).
        """
        to_path = "" if to_path is None else to_path
        if local_path is None:
            local_path = "."
        included_files: Optional[List[str]] = None
        if ignore_file:
            with open(ignore_file, "r") as f:
                ignore_patterns = f.readlines()
            included_files = filter_files(local_path, ignore_patterns)
        uploaded_file_count = 0
        for local_file_path in Path(local_path).expanduser().rglob("*"):
            if included_files is not None and str(local_file_path.relative_to(local_path)) not in included_files:
                continue
            elif not local_file_path.is_dir():
                remote_file_path = Path(to_path) / local_file_path.relative_to(local_path)
                with open(local_file_path, "rb") as local_file:
                    local_file_content = local_file.read()
                self.write_path(remote_file_path.as_posix(), content=local_file_content)
                uploaded_file_count += 1
        return uploaded_file_count

    def _read_sync(self, key: str) -> bytes:
        """
        Called by read_path. Retrieves the contents from a specified path.
        """
        s3_client = self._get_s3_client()
        with io.BytesIO() as stream:
            s3_client.download_fileobj(Bucket=self.bucket_name, Key=key, Fileobj=stream)
            stream.seek(0)
            output: bytes = stream.read()
            return output

    async def aread_path(self, path: str) -> bytes:
        """
        Asynchronously reads the contents of a specified path from the S3 bucket.
        """
        path = self._resolve_path(path)
        return await run_sync_in_worker_thread(self._read_sync, path)

    @async_dispatch(aread_path)
    def read_path(self, path: str) -> bytes:
        """
        Reads a specified path from the S3 bucket (synchronous version).
        """
        path = self._resolve_path(path)
        return self._read_sync(path)

    def _write_sync(self, key: str, data: bytes) -> None:
        """
        Called by write_path. Creates an S3 client and uploads the file object.
        """
        s3_client = self._get_s3_client()
        with io.BytesIO(data) as stream:
            s3_client.upload_fileobj(Fileobj=stream, Bucket=self.bucket_name, Key=key)

    async def awrite_path(self, path: str, content: bytes) -> str:
        """
        Asynchronously writes to an S3 bucket.
        """
        path = self._resolve_path(path)
        await run_sync_in_worker_thread(self._write_sync, path, content)
        return path

    @async_dispatch(awrite_path)
    def write_path(self, path: str, content: bytes) -> str:
        """
        Writes to an S3 bucket (synchronous version).
        """
        path = self._resolve_path(path)
        self._write_sync(path, content)
        return path

    @staticmethod
    def _list_objects_sync(page_iterator: PageIterator) -> List[Dict[str, Any]]:
        """
        Synchronous method to collect S3 objects into a list.
        """
        return [content for page in page_iterator for content in page.get("Contents", [])]

    def _join_bucket_folder(self, bucket_path: str = "") -> str:
        """
        Joins the base bucket folder to the bucket path.
        """
        if not self.bucket_folder and not bucket_path:
            return ""
        bucket_path_str = str(bucket_path)
        if self.bucket_folder != "" and bucket_path_str.startswith(self.bucket_folder):
            self.logger.info(
                f"Bucket path {bucket_path_str!r} is already prefixed with bucket folder {self.bucket_folder!r}; is this intentional?"
            )
        return (Path(self.bucket_folder) / bucket_path_str).as_posix() + ("" if not bucket_path_str.endswith("/") else "/")

    def _list_objects_setup(
        self,
        folder: str = "",
        delimiter: str = "",
        page_size: Optional[int] = None,
        max_items: Optional[int] = None,
        jmespath_query: Optional[str] = None,
    ) -> Tuple[PageIterator, str]:
        bucket_path: str = self._join_bucket_folder(folder)
        client = self.credentials.get_s3_client()
        paginator = client.get_paginator("list_objects_v2")
        page_iterator = paginator.paginate(
            Bucket=self.bucket_name,
            Prefix=bucket_path,
            Delimiter=delimiter,
            PaginationConfig={"PageSize": page_size, "MaxItems": max_items},
        )
        if jmespath_query:
            page_iterator = page_iterator.search(f"{jmespath_query} | {{Contents: @}}")
        return page_iterator, bucket_path

    async def alist_objects(
        self,
        folder: str = "",
        delimiter: str = "",
        page_size: Optional[int] = None,
        max_items: Optional[int] = None,
        jmespath_query: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Asynchronously lists objects in the S3 bucket.
        """
        page_iterator, bucket_path = self._list_objects_setup(folder, delimiter, page_size, max_items, jmespath_query)
        self.logger.info(f"Listing objects in bucket {bucket_path}.")
        objects = await run_sync_in_worker_thread(self._list_objects_sync, page_iterator)
        return objects

    @async_dispatch(alist_objects)
    def list_objects(
        self,
        folder: str = "",
        delimiter: str = "",
        page_size: Optional[int] = None,
        max_items: Optional[int] = None,
        jmespath_query: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Lists objects in the S3 bucket (synchronous version).
        """
        page_iterator, bucket_path = self._list_objects_setup(folder, delimiter, page_size, max_items, jmespath_query)
        self.logger.info(f"Listing objects in bucket {bucket_path}.")
        return self._list_objects_sync(page_iterator)

    async def adownload_object_to_path(
        self, from_path: str, to_path: Optional[str] = None, **download_kwargs: Any
    ) -> Path:
        """
        Asynchronously downloads an object from the S3 bucket to a local path.
        """
        if to_path is None:
            to_path = Path(from_path).name
        to_path_absolute: Path = Path(to_path).absolute()
        bucket_path: str = self._join_bucket_folder(from_path)
        client = self.credentials.get_s3_client()
        self.logger.debug(f"Preparing to download object from bucket {self.bucket_name!r} path {bucket_path!r} to {str(to_path_absolute)!r}.")
        await run_sync_in_worker_thread(
            client.download_file, Bucket=self.bucket_name, Key=bucket_path, Filename=str(to_path_absolute), **download_kwargs
        )
        self.logger.info(f"Downloaded object from bucket {self.bucket_name!r} path {bucket_path!r} to {str(to_path_absolute)!r}.")
        return to_path_absolute

    @async_dispatch(adownload_object_to_path)
    def download_object_to_path(
        self, from_path: str, to_path: Optional[str] = None, **download_kwargs: Any
    ) -> Path:
        """
        Downloads an object from the S3 bucket to a local path (synchronous version).
        """
        if to_path is None:
            to_path = Path(from_path).name
        to_path_absolute: Path = Path(to_path).absolute()
        bucket_path: str = self._join_bucket_folder(from_path)
        client = self.credentials.get_s3_client()
        self.logger.debug(f"Preparing to download object from bucket {self.bucket_name!r} path {bucket_path!r} to {str(to_path_absolute)!r}.")
        client.download_file(Bucket=self.bucket_name, Key=bucket_path, Filename=str(to_path_absolute), **download_kwargs)
        self.logger.info(f"Downloaded object from bucket {self.bucket_name!r} path {bucket_path!r} to {str(to_path_absolute)!r}.")
        return to_path_absolute

    async def adownload_object_to_file_object(
        self, from_path: str, to_file_object: BinaryIO, **download_kwargs: Any
    ) -> BinaryIO:
        """
        Asynchronously downloads an object to a file-like object.
        """
        client = self.credentials.get_s3_client()
        bucket_path: str = self._join_bucket_folder(from_path)
        self.logger.debug(f"Preparing to download object from bucket {self.bucket_name!r} path {bucket_path!r} to file object.")
        await run_sync_in_worker_thread(
            client.download_fileobj, Bucket=self.bucket_name, Key=bucket_path, Fileobj=to_file_object, **download_kwargs
        )
        self.logger.info(f"Downloaded object from bucket {self.bucket_name!r} path {bucket_path!r} to file object.")
        return to_file_object

    @async_dispatch(adownload_object_to_file_object)
    def download_object_to_file_object(
        self, from_path: str, to_file_object: BinaryIO, **download_kwargs: Any
    ) -> BinaryIO:
        """
        Downloads an object to a file-like object (synchronous version).
        """
        client = self.credentials.get_s3_client()
        bucket_path: str = self._join_bucket_folder(from_path)
        self.logger.debug(f"Preparing to download object from bucket {self.bucket_name!r} path {bucket_path!r} to file object.")
        client.download_fileobj(Bucket=self.bucket_name, Key=bucket_path, Fileobj=to_file_object, **download_kwargs)
        self.logger.info(f"Downloaded object from bucket {self.bucket_name!r} path {bucket_path!r} to file object.")
        return to_file_object

    async def adownload_folder_to_path(
        self, from_folder: str, to_folder: Optional[str] = None, **download_kwargs: Any
    ) -> Path:
        """
        Asynchronously downloads objects within a folder from the S3 bucket to a local folder.
        """
        if to_folder is None:
            to_folder = ""
        to_folder_path: Path = Path(to_folder).absolute()
        client = self.credentials.get_s3_client()
        objects: List[Dict[str, Any]] = await self.list_objects(folder=from_folder)
        bucket_folder: str = self._join_bucket_folder(from_folder)
        async_coros: List[Any] = []
        for obj in objects:
            bucket_path_relative = Path(obj["Key"]).relative_to(bucket_folder)
            if bucket_path_relative.is_dir():
                continue
            to_path_full: Path = to_folder_path / bucket_path_relative
            to_path_full.parent.mkdir(parents=True, exist_ok=True)
            async_coros.append(
                run_sync_in_worker_thread(
                    client.download_file, Bucket=self.bucket_name, Key=obj["Key"], Filename=str(to_path_full), **download_kwargs
                )
            )
        await asyncio.gather(*async_coros)
        return to_folder_path

    @async_dispatch(adownload_folder_to_path)
    def download_folder_to_path(
        self, from_folder: str, to_folder: Optional[str] = None, **download_kwargs: Any
    ) -> Path:
        """
        Downloads objects within a folder from the S3 bucket to a local folder (synchronous version).
        """
        if to_folder is None:
            to_folder = ""
        to_folder_path: Path = Path(to_folder).absolute()
        client = self.credentials.get_s3_client()
        objects: List[Dict[str, Any]] = self.list_objects(folder=from_folder)
        bucket_folder: str = self._join_bucket_folder(from_folder)
        for obj in objects:
            bucket_path_relative = Path(obj["Key"]).relative_to(bucket_folder)
            if bucket_path_relative.is_dir():
                continue
            to_path_full: Path = to_folder_path / bucket_path_relative
            to_path_full.parent.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Downloading object from bucket {self.bucket_name!r} path {bucket_path_relative.as_posix()!r} to {str(to_path_full)!r}.")
            client.download_file(Bucket=self.bucket_name, Key=obj["Key"], Filename=str(to_path_full), **download_kwargs)
        return to_folder_path

    async def astream_from(
        self, bucket: "S3Bucket", from_path: str, to_path: Optional[str] = None, **upload_kwargs: Any
    ) -> str:
        """
        Asynchronously streams an object from another bucket to this bucket.
        """
        if to_path is None:
            to_path = Path(from_path).name
        _from_path: str = bucket._join_bucket_folder(from_path)
        from_client = bucket.credentials.get_s3_client()
        obj: Dict[str, Any] = await run_sync_in_worker_thread(from_client.get_object, Bucket=bucket.bucket_name, Key=_from_path)
        body: StreamingBody = obj["Body"]
        bucket_path: str = str(self._join_bucket_folder(to_path))
        to_client = self.credentials.get_s3_client()
        await run_sync_in_worker_thread(
            to_client.upload_fileobj, Fileobj=body, Bucket=self.bucket_name, Key=bucket_path, **upload_kwargs
        )
        self.logger.info(f"Streamed s3://{bucket.bucket_name}/{_from_path} to the bucket {self.bucket_name!r} path {bucket_path!r}.")
        return bucket_path

    @async_dispatch(astream_from)
    def stream_from(
        self, bucket: "S3Bucket", from_path: str, to_path: Optional[str] = None, **upload_kwargs: Any
    ) -> str:
        """
        Streams an object from another bucket to this bucket (synchronous version).
        """
        if to_path is None:
            to_path = Path(from_path).name
        _from_path: str = bucket._join_bucket_folder(from_path)
        from_client = bucket.credentials.get_s3_client()
        obj: Dict[str, Any] = from_client.get_object(Bucket=bucket.bucket_name, Key=_from_path)
        body: StreamingBody = obj["Body"]
        bucket_path: str = str(self._join_bucket_folder(to_path))
        to_client = self.credentials.get_s3_client()
        to_client.upload_fileobj(Fileobj=body, Bucket=self.bucket_name, Key=bucket_path, **upload_kwargs)
        self.logger.info(f"Streamed s3://{bucket.bucket_name}/{_from_path} to the bucket {self.bucket_name!r} path {bucket_path!r}.")
        return bucket_path

    async def aupload_from_path(
        self, from_path: str, to_path: Optional[str] = None, **upload_kwargs: Any
    ) -> str:
        """
        Asynchronously uploads an object from a local path to the S3 bucket.
        """
        from_path_abs: str = str(Path(from_path).absolute())
        if to_path is None:
            to_path = Path(from_path).name
        bucket_path: str = str(self._join_bucket_folder(to_path))
        client = self.credentials.get_s3_client()
        await run_sync_in_worker_thread(
            client.upload_file, Filename=from_path_abs, Bucket=self.bucket_name, Key=bucket_path, **upload_kwargs
        )
        self.logger.info(f"Uploaded from {from_path_abs!r} to the bucket {self.bucket_name!r} path {bucket_path!r}.")
        return bucket_path

    @async_dispatch(aupload_from_path)
    def upload_from_path(
        self, from_path: str, to_path: Optional[str] = None, **upload_kwargs: Any
    ) -> str:
        """
        Uploads an object from a local path to the S3 bucket (synchronous version).
        """
        from_path_abs: str = str(Path(from_path).absolute())
        if to_path is None:
            to_path = Path(from_path).name
        bucket_path: str = str(self._join_bucket_folder(to_path))
        client = self.credentials.get_s3_client()
        client.upload_file(Filename=from_path_abs, Bucket=self.bucket_name, Key=bucket_path, **upload_kwargs)
        self.logger.info(f"Uploaded from {from_path_abs!r} to the bucket {self.bucket_name!r} path {bucket_path!r}.")
        return bucket_path

    async def aupload_from_file_object(
        self, from_file_object: BinaryIO, to_path: str, **upload_kwargs: Any
    ) -> str:
        """
        Asynchronously uploads an object to the S3 bucket from a file-like object.
        """
        bucket_path: str = str(self._join_bucket_folder(to_path))
        client = self.credentials.get_s3_client()
        await run_sync_in_worker_thread(
            client.upload_fileobj, Fileobj=from_file_object, Bucket=self.bucket_name, Key=bucket_path, **upload_kwargs
        )
        self.logger.info(f"Uploaded from file object to the bucket {self.bucket_name!r} path {bucket_path!r}.")
        return bucket_path

    @async_dispatch(aupload_from_file_object)
    def upload_from_file_object(
        self, from_file_object: BinaryIO, to_path: str, **upload_kwargs: Any
    ) -> str:
        """
        Uploads an object to the S3 bucket from a file-like object (synchronous version).
        """
        bucket_path: str = str(self._join_bucket_folder(to_path))
        client = self.credentials.get_s3_client()
        client.upload_fileobj(Fileobj=from_file_object, Bucket=self.bucket_name, Key=bucket_path, **upload_kwargs)
        self.logger.info(f"Uploaded from file object to the bucket {self.bucket_name!r} path {bucket_path!r}.")
        return bucket_path

    async def aupload_from_folder(
        self, from_folder: Union[str, Path], to_folder: Optional[str] = None, **upload_kwargs: Any
    ) -> str:
        """
        Asynchronously uploads files within a folder to the S3 bucket.
        """
        from_folder_path: Path = Path(from_folder)
        bucket_folder: str = self._join_bucket_folder(to_folder or "")
        num_uploaded = 0
        client = self.credentials.get_s3_client()
        async_coros: List[Any] = []
        for from_path in from_folder_path.rglob("**/*"):
            if from_path.is_dir():
                continue
            bucket_path = (Path(bucket_folder) / from_path.relative_to(from_folder_path)).as_posix()
            self.logger.info(f"Uploading from {str(from_path)!r} to the bucket {self.bucket_name!r} path {bucket_path!r}.")
            async_coros.append(
                run_sync_in_worker_thread(
                    client.upload_file, Filename=str(from_path), Bucket=self.bucket_name, Key=bucket_path, **upload_kwargs
                )
            )
            num_uploaded += 1
        await asyncio.gather(*async_coros)
        if num_uploaded == 0:
            self.logger.warning(f"No files were uploaded from {str(from_folder)!r}.")
        else:
            self.logger.info(f"Uploaded {num_uploaded} files from {str(from_folder)!r} to the bucket {self.bucket_name!r} path {bucket_folder!r}")
        return to_folder if to_folder is not None else ""

    @async_dispatch(aupload_from_folder)
    def upload_from_folder(
        self, from_folder: Union[str, Path], to_folder: Optional[str] = None, **upload_kwargs: Any
    ) -> str:
        """
        Uploads files within a folder to the S3 bucket (synchronous version).
        """
        from_folder_path: Path = Path(from_folder)
        bucket_folder: str = self._join_bucket_folder(to_folder or "")
        num_uploaded = 0
        client = self.credentials.get_s3_client()
        for from_path in from_folder_path.rglob("**/*"):
            if from_path.is_dir():
                continue
            bucket_path = (Path(bucket_folder) / from_path.relative_to(from_folder_path)).as_posix()
            self.logger.info(f"Uploading from {str(from_path)!r} to the bucket {self.bucket_name!r} path {bucket_path!r}.")
            client.upload_file(Filename=str(from_path), Bucket=self.bucket_name, Key=bucket_path, **upload_kwargs)
            num_uploaded += 1
        if num_uploaded == 0:
            self.logger.warning(f"No files were uploaded from {str(from_folder)!r}.")
        else:
            self.logger.info(f"Uploaded {num_uploaded} files from {str(from_folder)!r} to the bucket {self.bucket_name!r} path {bucket_folder!r}")
        return to_folder if to_folder is not None else ""

    def copy_object(
        self,
        from_path: str,
        to_path: str,
        to_bucket: Optional[Union[str, "S3Bucket"]] = None,
        **copy_kwargs: Any,
    ) -> str:
        """
        Copies an object using S3's CopyObject.
        """
        s3_client = self.credentials.get_s3_client()
        source_bucket_name: str = self.bucket_name
        source_path: str = self._resolve_path(Path(from_path).as_posix())
        to_bucket_obj: Union["S3Bucket", str] = to_bucket or self
        if isinstance(to_bucket_obj, S3Bucket):
            target_bucket_name: str = to_bucket_obj.bucket_name
            target_path: str = to_bucket_obj._resolve_path(Path(to_path).as_posix())
        elif isinstance(to_bucket_obj, str):
            target_bucket_name = to_bucket_obj
            target_path = Path(to_path).as_posix()
        else:
            raise TypeError(f"to_bucket must be a string or S3Bucket, not {type(to_bucket_obj)}")
        self.logger.info(
            "Copying object from bucket %s with key %s to bucket %s with key %s",
            source_bucket_name,
            source_path,
            target_bucket_name,
            target_path,
        )
        s3_client.copy_object(
            CopySource={"Bucket": source_bucket_name, "Key": source_path},
            Bucket=target_bucket_name,
            Key=target_path,
            **copy_kwargs,
        )
        return target_path

    def _move_object_setup(
        self, from_path: str, to_path: str, to_bucket: Optional[Union[str, "S3Bucket"]] = None
    ) -> Tuple[str, str, str, str]:
        source_bucket_name: str = self.bucket_name
        source_path: str = self._resolve_path(Path(from_path).as_posix())
        to_bucket_obj: Union["S3Bucket", str] = to_bucket or self
        if isinstance(to_bucket_obj, S3Bucket):
            target_bucket_name: str = to_bucket_obj.bucket_name
            target_path: str = to_bucket_obj._resolve_path(Path(to_path).as_posix())
        elif isinstance(to_bucket_obj, str):
            target_bucket_name = to_bucket_obj
            target_path = Path(to_path).as_posix()
        else:
            raise TypeError(f"to_bucket must be a string or S3Bucket, not {type(to_bucket_obj)}")
        self.logger.info("Moving object from s3://%s/%s to s3://%s/%s", source_bucket_name, source_path, target_bucket_name, target_path)
        return (source_bucket_name, source_path, target_bucket_name, target_path)

    async def amove_object(
        self, from_path: str, to_path: str, to_bucket: Optional[Union[str, "S3Bucket"]] = None
    ) -> str:
        """
        Asynchronously moves an object using S3's CopyObject and DeleteObject.
        """
        s3_client = self.credentials.get_s3_client()
        source_bucket_name, source_path, target_bucket_name, target_path = self._move_object_setup(from_path, to_path, to_bucket)
        await run_sync_in_worker_thread(
            s3_client.copy, CopySource={"Bucket": source_bucket_name, "Key": source_path}, Bucket=target_bucket_name, Key=target_path
        )
        s3_client.delete_object(Bucket=source_bucket_name, Key=source_path)
        return target_path

    @async_dispatch(amove_object)
    def move_object(
        self, from_path: str, to_path: str, to_bucket: Optional[Union[str, "S3Bucket"]] = None
    ) -> str:
        """
        Moves an object using S3's CopyObject and DeleteObject (synchronous version).
        """
        s3_client = self.credentials.get_s3_client()
        source_bucket_name, source_path, target_bucket_name, target_path = self._move_object_setup(from_path, to_path, to_bucket)
        s3_client.copy(CopySource={"Bucket": source_bucket_name, "Key": source_path}, Bucket=target_bucket_name, Key=target_path)
        s3_client.delete_object(Bucket=source_bucket_name, Key=source_path)
        return target_path
