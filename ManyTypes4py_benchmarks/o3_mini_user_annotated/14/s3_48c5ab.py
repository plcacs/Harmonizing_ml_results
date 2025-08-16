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
    logger = get_run_logger()
    logger.info("Downloading object from bucket %s with key %s", bucket, key)

    s3_client = aws_credentials.get_boto3_session().client(
        "s3", **aws_client_parameters.get_params_override()
    )
    stream = io.BytesIO()
    await run_sync_in_worker_thread(
        s3_client.download_fileobj, Bucket=bucket, Key=key, Fileobj=stream
    )
    stream.seek(0)
    output = stream.read()

    return output


@async_dispatch(adownload_from_bucket)
@task
def download_from_bucket(
    bucket: str,
    key: str,
    aws_credentials: AwsCredentials,
    aws_client_parameters: AwsClientParameters = AwsClientParameters(),
) -> bytes:
    logger = get_run_logger()
    logger.info("Downloading object from bucket %s with key %s", bucket, key)

    s3_client = aws_credentials.get_boto3_session().client(
        "s3", **aws_client_parameters.get_params_override()
    )
    stream = io.BytesIO()
    s3_client.download_fileobj(Bucket=bucket, Key=key, Fileobj=stream)
    stream.seek(0)
    output = stream.read()

    return output


s3_download = download_from_bucket  # backward compatibility


@task
async def aupload_to_bucket(
    data: bytes,
    bucket: str,
    aws_credentials: AwsCredentials,
    aws_client_parameters: AwsClientParameters = AwsClientParameters(),
    key: Optional[str] = None,
) -> str:
    logger = get_run_logger()

    key = key or str(uuid.uuid4())

    logger.info("Uploading object to bucket %s with key %s", bucket, key)

    s3_client = aws_credentials.get_boto3_session().client(
        "s3", **aws_client_parameters.get_params_override()
    )
    stream = io.BytesIO(data)
    await run_sync_in_worker_thread(
        s3_client.upload_fileobj, stream, Bucket=bucket, Key=key
    )

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
    logger = get_run_logger()

    key = key or str(uuid.uuid4())

    logger.info("Uploading object to bucket %s with key %s", bucket, key)

    s3_client = aws_credentials.get_boto3_session().client(
        "s3", **aws_client_parameters.get_params_override()
    )
    stream = io.BytesIO(data)
    s3_client.upload_fileobj(stream, Bucket=bucket, Key=key)
    return key


s3_upload = upload_to_bucket  # backward compatibility


@task
async def acopy_objects(
    source_path: str,
    target_path: str,
    source_bucket_name: str,
    aws_credentials: AwsCredentials,
    target_bucket_name: Optional[str] = None,
    **copy_kwargs: Any,
) -> str:
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

    return target_path


@async_dispatch(acopy_objects)
@task
def copy_objects(
    source_path: str,
    target_path: str,
    source_bucket_name: str,
    aws_credentials: AwsCredentials,
    target_bucket_name: Optional[str] = None,
    **copy_kwargs: Any,
) -> str:
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

    return target_path


s3_copy = copy_objects  # backward compatibility


@task
async def amove_objects(
    source_path: str,
    target_path: str,
    source_bucket_name: str,
    aws_credentials: AwsCredentials,
    target_bucket_name: Optional[str] = None,
) -> str:
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

    await run_sync_in_worker_thread(
        s3_client.delete_object, Bucket=source_bucket_name, Key=source_path
    )

    return target_path


@async_dispatch(amove_objects)
@task
def move_objects(
    source_path: str,
    target_path: str,
    source_bucket_name: str,
    aws_credentials: AwsCredentials,
    target_bucket_name: Optional[str] = None,
) -> str:
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

    return target_path


s3_move = move_objects  # backward compatibility


def _list_objects_sync(page_iterator: PageIterator) -> List[Dict[str, Any]]:
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
    logger = get_run_logger()
    logger.info("Listing objects in bucket %s with prefix %s", bucket, prefix)

    s3_client = aws_credentials.get_boto3_session().client(
        "s3", **aws_client_parameters.get_params_override()
    )
    paginator = s3_client.get_paginator("list_objects_v2")
    page_iterator: PageIterator = paginator.paginate(
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
    logger = get_run_logger()
    logger.info("Listing objects in bucket %s with prefix %s", bucket, prefix)

    s3_client = aws_credentials.get_boto3_session().client(
        "s3", **aws_client_parameters.get_params_override()
    )
    paginator = s3_client.get_paginator("list_objects_v2")
    page_iterator: PageIterator = paginator.paginate(
        Bucket=bucket,
        Prefix=prefix,
        Delimiter=delimiter,
        PaginationConfig={"PageSize": page_size, "MaxItems": max_items},
    )
    if jmespath_query:
        page_iterator = page_iterator.search(f"{jmespath_query} | {{Contents: @}}")
    return _list_objects_sync(page_iterator)  # type: ignore


s3_list_objects = list_objects  # backward compatibility


class S3Bucket(WritableFileSystem, WritableDeploymentStorage, ObjectStorageBlock):
    _logo_url: str = "https://cdn.sanity.io/images/3ugk85nk/production/d74b16fe84ce626345adf235a47008fea2869a60-225x225.png"
    _block_type_name: str = "S3 Bucket"
    _documentation_url: str = "https://docs.prefect.io/integrations/prefect-aws"

    bucket_name: str = Field(..., description="Name of your bucket.")

    credentials: Union[MinIOCredentials, AwsCredentials] = Field(
        default_factory=AwsCredentials,
        description="A block containing your credentials to AWS or MinIO.",
    )

    bucket_folder: str = Field(
        default="",
        description=(
            "A default path to a folder within the S3 bucket to use "
            "for reading and writing objects."
        ),
    )

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
                    return credentials_cls(**value)  # type: ignore
                except ValueError:
                    pass
            valid_classes = ", ".join(c.__name__ for c in credential_classes)
            raise ValueError(
                f"Invalid credentials data: does not match any credential type. Valid types: {valid_classes}"
            )
        return value

    @property
    def basepath(self) -> str:
        return self.bucket_folder

    @basepath.setter
    def basepath(self, value: str) -> None:
        self.bucket_folder = value

    def _resolve_path(self, path: str) -> str:
        path = (Path(self.bucket_folder) / path).as_posix() if self.bucket_folder else path
        return path

    def _get_s3_client(self) -> Any:
        return self.credentials.get_client("s3")

    def _get_bucket_resource(self) -> Any:
        params_override: Dict[str, Any] = self.credentials.aws_client_parameters.get_params_override()
        bucket = (
            self.credentials.get_boto3_session()
            .resource("s3", **params_override)
            .Bucket(self.bucket_name)
        )
        return bucket

    async def aget_directory(
        self, from_path: Optional[str] = None, local_path: Optional[str] = None
    ) -> None:
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
            target: str = os.path.join(
                local_path,
                os.path.relpath(obj.key, from_path),
            )
            os.makedirs(os.path.dirname(target), exist_ok=True)
            await run_sync_in_worker_thread(bucket.download_file, obj.key, target)

    @async_dispatch(aget_directory)
    def get_directory(
        self, from_path: Optional[str] = None, local_path: Optional[str] = None
    ) -> None:
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
            target: str = os.path.join(
                local_path,
                os.path.relpath(obj.key, from_path),
            )
            os.makedirs(os.path.dirname(target), exist_ok=True)
            bucket.download_file(obj.key, target)

    async def aput_directory(
        self,
        local_path: Optional[str] = None,
        to_path: Optional[str] = None,
        ignore_file: Optional[str] = None,
    ) -> int:
        to_path = "" if to_path is None else to_path

        if local_path is None:
            local_path = "."

        included_files: Optional[List[str]] = None
        if ignore_file:
            with open(ignore_file, "r") as f:
                ignore_patterns = f.readlines()
            included_files = filter_files(local_path, ignore_patterns)

        uploaded_file_count: int = 0
        for local_file_path in Path(local_path).expanduser().rglob("*"):
            if (included_files is not None and
                str(local_file_path.relative_to(local_path)) not in included_files):
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
        self,
        local_path: Optional[str] = None,
        to_path: Optional[str] = None,
        ignore_file: Optional[str] = None,
    ) -> int:
        to_path = "" if to_path is None else to_path

        if local_path is None:
            local_path = "."

        included_files: Optional[List[str]] = None
        if ignore_file:
            with open(ignore_file, "r") as f:
                ignore_patterns = f.readlines()
            included_files = filter_files(local_path, ignore_patterns)

        uploaded_file_count: int = 0
        for local_file_path in Path(local_path).expanduser().rglob("*"):
            if (included_files is not None and
                str(local_file_path.relative_to(local_path)) not in included_files):
                continue
            elif not local_file_path.is_dir():
                remote_file_path = Path(to_path) / local_file_path.relative_to(local_path)
                with open(local_file_path, "rb") as local_file:
                    local_file_content = local_file.read()
                self.write_path(remote_file_path.as_posix(), content=local_file_content, _sync=True)
                uploaded_file_count += 1

        return uploaded_file_count

    def _read_sync(self, key: str) -> bytes:
        s3_client = self._get_s3_client()
        with io.BytesIO() as stream:
            s3_client.download_fileobj(Bucket=self.bucket_name, Key=key, Fileobj=stream)
            stream.seek(0)
            output = stream.read()
            return output

    async def aread_path(self, path: str) -> bytes:
        path = self._resolve_path(path)
        return await run_sync_in_worker_thread(self._read_sync, path)

    @async_dispatch(aread_path)
    def read_path(self, path: str) -> bytes:
        path = self._resolve_path(path)
        return self._read_sync(path)

    def _write_sync(self, key: str, data: bytes) -> None:
        s3_client = self._get_s3_client()
        with io.BytesIO(data) as stream:
            s3_client.upload_fileobj(Fileobj=stream, Bucket=self.bucket_name, Key=key)

    async def awrite_path(self, path: str, content: bytes) -> str:
        path = self._resolve_path(path)
        await run_sync_in_worker_thread(self._write_sync, path, content)
        return path

    @async_dispatch(awrite_path)
    def write_path(self, path: str, content: bytes) -> str:
        path = self._resolve_path(path)
        self._write_sync(path, content)
        return path

    @staticmethod
    def _list_objects_sync(page_iterator: PageIterator) -> List[Dict[str, Any]]:
        return [content for page in page_iterator for content in page.get("Contents", [])]

    def _join_bucket_folder(self, bucket_path: str = "") -> str:
        if not self.bucket_folder and not bucket_path:
            return ""
        bucket_path = str(bucket_path)
        if self.bucket_folder != "" and bucket_path.startswith(self.bucket_folder):
            self.logger.info(
                f"Bucket path {bucket_path!r} is already prefixed with "
                f"bucket folder {self.bucket_folder!r}; is this intentional?"
            )
        return (Path(self.bucket_folder) / bucket_path).as_posix() + ("" if not bucket_path.endswith("/") else "/")

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
        page_iterator: PageIterator = paginator.paginate(
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
        page_iterator, bucket_path = self._list_objects_setup(folder, delimiter, page_size, max_items, jmespath_query)
        self.logger.info(f"Listing objects in bucket {bucket_path}.")
        objects: List[Dict[str, Any]] = await run_sync_in_worker_thread(self._list_objects_sync, page_iterator)
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
        page_iterator, bucket_path = self._list_objects_setup(folder, delimiter, page_size, max_items, jmespath_query)
        self.logger.info(f"Listing objects in bucket {bucket_path}.")
        return self._list_objects_sync(page_iterator)

    async def adownload_object_to_path(
        self,
        from_path: str,
        to_path: Optional[Union[str, Path]],
        **download_kwargs: Any,
    ) -> Path:
        if to_path is None:
            to_path = Path(from_path).name
        to_path_abs: str = str(Path(to_path).absolute())
        bucket_path: str = self._join_bucket_folder(from_path)
        client = self.credentials.get_s3_client()
        self.logger.debug(
            f"Preparing to download object from bucket {self.bucket_name!r} "
            f"path {bucket_path!r} to {to_path_abs!r}."
        )
        await run_sync_in_worker_thread(
            client.download_file,
            Bucket=self.bucket_name,
            Key=bucket_path,
            Filename=to_path_abs,
            **download_kwargs,
        )
        self.logger.info(
            f"Downloaded object from bucket {self.bucket_name!r} path {bucket_path!r} "
            f"to {to_path_abs!r}."
        )
        return Path(to_path_abs)

    @async_dispatch(adownload_object_to_path)
    def download_object_to_path(
        self,
        from_path: str,
        to_path: Optional[Union[str, Path]],
        **download_kwargs: Any,
    ) -> Path:
        if to_path is None:
            to_path = Path(from_path).name
        to_path_abs: str = str(Path(to_path).absolute())
        bucket_path: str = self._join_bucket_folder(from_path)
        client = self.credentials.get_s3_client()
        self.logger.debug(
            f"Preparing to download object from bucket {self.bucket_name!r} "
            f"path {bucket_path!r} to {to_path_abs!r}."
        )
        client.download_file(
            Bucket=self.bucket_name,
            Key=bucket_path,
            Filename=to_path_abs,
            **download_kwargs,
        )
        self.logger.info(
            f"Downloaded object from bucket {self.bucket_name!r} path {bucket_path!r} "
            f"to {to_path_abs!r}."
        )
        return Path(to_path_abs)

    async def adownload_object_to_file_object(
        self,
        from_path: str,
        to_file_object: BinaryIO,
        **download_kwargs: Any,
    ) -> BinaryIO:
        client = self.credentials.get_s3_client()
        bucket_path: str = self._join_bucket_folder(from_path)
        self.logger.debug(
            f"Preparing to download object from bucket {self.bucket_name!r} "
            f"path {bucket_path!r} to file object."
        )
        await run_sync_in_worker_thread(
            client.download_fileobj,
            Bucket=self.bucket_name,
            Key=bucket_path,
            Fileobj=to_file_object,
            **download_kwargs,
        )
        self.logger.info(
            f"Downloaded object from bucket {self.bucket_name!r} path {bucket_path!r} "
            "to file object."
        )
        return to_file_object

    @async_dispatch(adownload_object_to_file_object)
    def download_object_to_file_object(
        self,
        from_path: str,
        to_file_object: BinaryIO,
        **download_kwargs: Any,
    ) -> BinaryIO:
        client = self.credentials.get_s3_client()
        bucket_path: str = self._join_bucket_folder(from_path)
        self.logger.debug(
            f"Preparing to download object from bucket {self.bucket_name!r} "
            f"path {bucket_path!r} to file object."
        )
        client.download_fileobj(
            Bucket=self.bucket_name,
            Key=bucket_path,
            Fileobj=to_file_object,
            **download_kwargs,
        )
        self.logger.info(
            f"Downloaded object from bucket {self.bucket_name!r} path {bucket_path!r} "
            "to file object."
        )
        return to_file_object

    async def adownload_folder_to_path(
        self,
        from_folder: str,
        to_folder: Optional[Union[str, Path]] = None,
        **download_kwargs: Any,
    ) -> Path:
        if to_folder is None:
            to_folder = ""
        to_folder_path: Path = Path(to_folder).absolute()
        client = self.credentials.get_s3_client()
        objects: List[Dict[str, Any]] = await self.list_objects(folder=from_folder)
        bucket_folder: str = self._join_bucket_folder(from_folder)
        async_coros: List[Any] = []
        for obj in objects:
            bucket_relative = Path(obj["Key"]).relative_to(bucket_folder)
            if bucket_relative.is_dir():
                continue
            target_path: Path = to_folder_path / bucket_relative
            target_path.parent.mkdir(parents=True, exist_ok=True)
            target_str: str = str(target_path)
            self.logger.info(
                f"Downloading object from bucket {self.bucket_name!r} path "
                f"{bucket_relative.as_posix()!r} to {target_str!r}."
            )
            async_coros.append(
                run_sync_in_worker_thread(
                    client.download_file,
                    Bucket=self.bucket_name,
                    Key=obj["Key"],
                    Filename=target_str,
                    **download_kwargs,
                )
            )
        await asyncio.gather(*async_coros)
        return to_folder_path

    @async_dispatch(adownload_folder_to_path)
    def download_folder_to_path(
        self,
        from_folder: str,
        to_folder: Optional[Union[str, Path]] = None,
        **download_kwargs: Any,
    ) -> Path:
        if to_folder is None:
            to_folder = ""
        to_folder_path: Path = Path(to_folder).absolute()
        client = self.credentials.get_s3_client()
        objects: List[Dict[str, Any]] = self.list_objects(folder=from_folder)
        bucket_folder: str = self._join_bucket_folder(from_folder)
        for obj in objects:
            bucket_relative = Path(obj["Key"]).relative_to(bucket_folder)
            if bucket_relative.is_dir():
                continue
            target_path: Path = to_folder_path / bucket_relative
            target_path.parent.mkdir(parents=True, exist_ok=True)
            target_str: str = str(target_path)
            self.logger.info(
                f"Downloading object from bucket {self.bucket_name!r} path "
                f"{bucket_relative.as_posix()!r} to {target_str!r}."
            )
            client.download_file(
                Bucket=self.bucket_name,
                Key=obj["Key"],
                Filename=target_str,
                **download_kwargs,
            )
        return to_folder_path

    async def astream_from(
        self,
        bucket: "S3Bucket",
        from_path: str,
        to_path: Optional[str] = None,
        **upload_kwargs: Any,
    ) -> str:
        if to_path is None:
            to_path = Path(from_path).name
        _from_path: str = bucket._join_bucket_folder(from_path)
        from_client = bucket.credentials.get_s3_client()
        obj: Dict[str, Any] = await run_sync_in_worker_thread(
            from_client.get_object, Bucket=bucket.bucket_name, Key=_from_path
        )
        body: StreamingBody = obj["Body"]
        bucket_path: str = str(self._join_bucket_folder(to_path))
        to_client = self.credentials.get_s3_client()
        await run_sync_in_worker_thread(
            to_client.upload_fileobj,
            Fileobj=body,
            Bucket=self.bucket_name,
            Key=bucket_path,
            **upload_kwargs,
        )
        self.logger.info(
            f"Streamed s3://{bucket.bucket_name}/{_from_path} to the bucket "
            f"{self.bucket_name!r} path {bucket_path!r}."
        )
        return bucket_path

    @async_dispatch(astream_from)
    def stream_from(
        self,
        bucket: "S3Bucket",
        from_path: str,
        to_path: Optional[str] = None,
        **upload_kwargs: Any,
    ) -> str:
        if to_path is None:
            to_path = Path(from_path).name
        _from_path: str = bucket._join_bucket_folder(from_path)
        from_client = bucket.credentials.get_s3_client()
        obj: Dict[str, Any] = from_client.get_object(Bucket=bucket.bucket_name, Key=_from_path)
        body: StreamingBody = obj["Body"]
        bucket_path: str = str(self._join_bucket_folder(to_path))
        to_client = self.credentials.get_s3_client()
        to_client.upload_fileobj(
            Fileobj=body,
            Bucket=self.bucket_name,
            Key=bucket_path,
            **upload_kwargs,
        )
        self.logger.info(
            f"Streamed s3://{bucket.bucket_name}/{_from_path} to the bucket "
            f"{self.bucket_name!r} path {bucket_path!r}."
        )
        return bucket_path

    async def aupload_from_path(
        self,
        from_path: Union[str, Path],
        to_path: Optional[str] = None,
        **upload_kwargs: Any,
    ) -> str:
        from_path_str: str = str(Path(from_path).absolute())
        if to_path is None:
            to_path = Path(from_path_str).name
        bucket_path: str = str(self._join_bucket_folder(to_path))
        client = self.credentials.get_s3_client()
        await run_sync_in_worker_thread(
            client.upload_file,
            Filename=from_path_str,
            Bucket=self.bucket_name,
            Key=bucket_path,
            **upload_kwargs,
        )
        self.logger.info(
            f"Uploaded from {from_path_str!r} to the bucket "
            f"{self.bucket_name!r} path {bucket_path!r}."
        )
        return bucket_path

    @async_dispatch(aupload_from_path)
    def upload_from_path(
        self,
        from_path: Union[str, Path],
        to_path: Optional[str] = None,
        **upload_kwargs: Any,
    ) -> str:
        from_path_str: str = str(Path(from_path).absolute())
        if to_path is None:
            to_path = Path(from_path_str).name
        bucket_path: str = str(self._join_bucket_folder(to_path))
        client = self.credentials.get_s3_client()
        client.upload_file(
            Filename=from_path_str,
            Bucket=self.bucket_name,
            Key=bucket_path,
            **upload_kwargs,
        )
        self.logger.info(
            f"Uploaded from {from_path_str!r} to the bucket "
            f"{self.bucket_name!r} path {bucket_path!r}."
        )
        return bucket_path

    async def aupload_from_file_object(
        self, from_file_object: BinaryIO, to_path: str, **upload_kwargs: Any
    ) -> str:
        bucket_path: str = str(self._join_bucket_folder(to_path))
        client = self.credentials.get_s3_client()
        await run_sync_in_worker_thread(
            client.upload_fileobj,
            Fileobj=from_file_object,
            Bucket=self.bucket_name,
            Key=bucket_path,
            **upload_kwargs,
        )
        self.logger.info(
            "Uploaded from file object to the bucket "
            f"{self.bucket_name!r} path {bucket_path!r}."
        )
        return bucket_path

    @async_dispatch(aupload_from_file_object)
    def upload_from_file_object(
        self, from_file_object: BinaryIO, to_path: str, **upload_kwargs: Any
    ) -> str:
        bucket_path: str = str(self._join_bucket_folder(to_path))
        client = self.credentials.get_s3_client()
        client.upload_fileobj(
            Fileobj=from_file_object,
            Bucket=self.bucket_name,
            Key=bucket_path,
            **upload_kwargs,
        )
        self.logger.info(
            "Uploaded from file object to the bucket "
            f"{self.bucket_name!r} path {bucket_path!r}."
        )
        return bucket_path

    async def aupload_from_folder(
        self,
        from_folder: Union[str, Path],
        to_folder: Optional[str] = None,
        **upload_kwargs: Any,
    ) -> Union[str, None]:
        from_folder_path: Path = Path(from_folder)
        bucket_folder: str = self._join_bucket_folder(to_folder or "")
        num_uploaded: int = 0
        client = self.credentials.get_s3_client()
        async_coros: List[Any] = []
        for from_path in from_folder_path.rglob("**/*"):
            if from_path.is_dir():
                continue
            bucket_path: str = (Path(bucket_folder) / from_path.relative_to(from_folder_path)).as_posix()
            self.logger.info(
                f"Uploading from {str(from_path)!r} to the bucket "
                f"{self.bucket_name!r} path {bucket_path!r}."
            )
            async_coros.append(
                run_sync_in_worker_thread(
                    client.upload_file,
                    Filename=str(from_path),
                    Bucket=self.bucket_name,
                    Key=bucket_path,
                    **upload_kwargs,
                )
            )
            num_uploaded += 1
        await asyncio.gather(*async_coros)
        if num_uploaded == 0:
            self.logger.warning(f"No files were uploaded from {str(from_folder_path)!r}.")
        else:
            self.logger.info(
                f"Uploaded {num_uploaded} files from {str(from_folder_path)!r} to "
                f"the bucket {self.bucket_name!r} path {bucket_folder!r}"
            )
        return to_folder

    @async_dispatch(aupload_from_folder)
    def upload_from_folder(
        self,
        from_folder: Union[str, Path],
        to_folder: Optional[str] = None,
        **upload_kwargs: Any,
    ) -> Union[str, None]:
        from_folder_path: Path = Path(from_folder)
        bucket_folder: str = self._join_bucket_folder(to_folder or "")
        num_uploaded: int = 0
        client = self.credentials.get_s3_client()
        for from_path in from_folder_path.rglob("**/*"):
            if from_path.is_dir():
                continue
            bucket_path: str = (Path(bucket_folder) / from_path.relative_to(from_folder_path)).as_posix()
            self.logger.info(
                f"Uploading from {str(from_path)!r} to the bucket "
                f"{self.bucket_name!r} path {bucket_path!r}."
            )
            client.upload_file(
                Filename=str(from_path),
                Bucket=self.bucket_name,
                Key=bucket_path,
                **upload_kwargs,
            )
            num_uploaded += 1
        if num_uploaded == 0:
            self.logger.warning(f"No files were uploaded from {str(from_folder_path)!r}.")
        else:
            self.logger.info(
                f"Uploaded {num_uploaded} files from {str(from_folder_path)!r} to "
                f"the bucket {self.bucket_name!r} path {bucket_folder!r}"
            )
        return to_folder

    def copy_object(
        self,
        from_path: Union[str, Path],
        to_path: Union[str, Path],
        to_bucket: Optional[Union["S3Bucket", str]] = None,
        **copy_kwargs: Any,
    ) -> str:
        s3_client = self.credentials.get_s3_client()

        source_bucket_name: str = self.bucket_name
        source_path: str = self._resolve_path(Path(from_path).as_posix())

        to_bucket = to_bucket or self

        if isinstance(to_bucket, S3Bucket):
            target_bucket_name: str = to_bucket.bucket_name
            target_path: str = to_bucket._resolve_path(Path(to_path).as_posix())
        elif isinstance(to_bucket, str):
            target_bucket_name = to_bucket
            target_path = Path(to_path).as_posix()
        else:
            raise TypeError(f"to_bucket must be a string or S3Bucket, not {type(to_bucket)}")

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
        self,
        from_path: Union[str, Path],
        to_path: Union[str, Path],
        to_bucket: Optional[Union["S3Bucket", str]] = None,
    ) -> Tuple[str, str, str, str]:
        source_bucket_name: str = self.bucket_name
        source_path: str = self._resolve_path(Path(from_path).as_posix())

        to_bucket = to_bucket or self

        if isinstance(to_bucket, S3Bucket):
            target_bucket_name: str = to_bucket.bucket_name
            target_path: str = to_bucket._resolve_path(Path(to_path).as_posix())
        elif isinstance(to_bucket, str):
            target_bucket_name = to_bucket
            target_path = Path(to_path).as_posix()
        else:
            raise TypeError(f"to_bucket must be a string or S3Bucket, not {type(to_bucket)}")

        self.logger.info(
            "Moving object from s3://%s/%s to s3://%s/%s",
            source_bucket_name,
            source_path,
            target_bucket_name,
            target_path,
        )

        return source_bucket_name, source_path, target_bucket_name, target_path

    async def amove_object(
        self,
        from_path: Union[str, Path],
        to_path: Union[str, Path],
        to_bucket: Optional[Union["S3Bucket", str]] = None,
    ) -> str:
        s3_client = self.credentials.get_s3_client()

        (
            source_bucket_name,
            source_path,
            target_bucket_name,
            target_path,
        ) = self._move_object_setup(from_path, to_path, to_bucket)

        await run_sync_in_worker_thread(
            s3_client.copy,
            CopySource={"Bucket": source_bucket_name, "Key": source_path},
            Bucket=target_bucket_name,
            Key=target_path,
        )
        s3_client.delete_object(Bucket=source_bucket_name, Key=source_path)
        return target_path

    @async_dispatch(amove_object)
    def move_object(
        self,
        from_path: Union[str, Path],
        to_path: Union[str, Path],
        to_bucket: Optional[Union["S3Bucket", str]] = None,
    ) -> str:
        s3_client = self.credentials.get_s3_client()

        (
            source_bucket_name,
            source_path,
            target_bucket_name,
            target_path,
        ) = self._move_object_setup(from_path, to_path, to_bucket)

        s3_client.copy(
            CopySource={"Bucket": source_bucket_name, "Key": source_path},
            Bucket=target_bucket_name,
            Key=target_path,
        )
        s3_client.delete_object(Bucket=source_bucket_name, Key=source_path)
        return target_path
