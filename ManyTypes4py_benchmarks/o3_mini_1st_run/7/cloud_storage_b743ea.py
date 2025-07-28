"""Tasks for interacting with GCP Cloud Storage."""
import asyncio
import os
from enum import Enum
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Dict, List, Optional, Tuple, Union

from pydantic import Field, field_validator
from prefect import task
from prefect.blocks.abstract import ObjectStorageBlock
from prefect.filesystems import WritableDeploymentStorage, WritableFileSystem
from prefect.logging import disable_run_logger, get_run_logger
from prefect.utilities.asyncutils import run_sync_in_worker_thread, sync_compatible
from prefect.utilities.filesystem import filter_files
from prefect_gcp.credentials import GcpCredentials
try:
    from pandas import DataFrame
except ModuleNotFoundError:
    DataFrame = None  # type: ignore
try:
    from google.cloud.storage import Bucket
    from google.cloud.storage.blob import Blob
except ModuleNotFoundError:
    Bucket, Blob = None, None  # type: ignore


@task
@sync_compatible
async def cloud_storage_create_bucket(
    bucket: str,
    gcp_credentials: GcpCredentials,
    project: Optional[str] = None,
    location: Optional[str] = None,
    **create_kwargs: Any
) -> str:
    logger = get_run_logger()
    logger.info('Creating %s bucket', bucket)
    client = gcp_credentials.get_cloud_storage_client(project=project)
    await run_sync_in_worker_thread(client.create_bucket, bucket, location=location, **create_kwargs)
    return bucket


async def _get_bucket_async(
    bucket: str,
    gcp_credentials: GcpCredentials,
    project: Optional[str] = None
) -> Bucket:
    client = gcp_credentials.get_cloud_storage_client(project=project)
    bucket_obj = await run_sync_in_worker_thread(client.get_bucket, bucket)
    return bucket_obj


def _get_bucket(
    bucket: str,
    gcp_credentials: GcpCredentials,
    project: Optional[str] = None
) -> Bucket:
    client = gcp_credentials.get_cloud_storage_client(project=project)
    bucket_obj = client.get_bucket(bucket)
    return bucket_obj


@task
@sync_compatible
async def cloud_storage_download_blob_as_bytes(
    bucket: str,
    blob: str,
    gcp_credentials: GcpCredentials,
    chunk_size: Optional[int] = None,
    encryption_key: Optional[Any] = None,
    timeout: Union[int, Tuple[int, int]] = 60,
    project: Optional[str] = None,
    **download_kwargs: Any
) -> bytes:
    logger = get_run_logger()
    logger.info('Downloading blob named %s from the %s bucket', blob, bucket)
    bucket_obj = await _get_bucket_async(bucket, gcp_credentials, project=project)
    blob_obj = bucket_obj.blob(blob, chunk_size=chunk_size, encryption_key=encryption_key)
    contents = await run_sync_in_worker_thread(blob_obj.download_as_bytes, timeout=timeout, **download_kwargs)
    return contents


@task
@sync_compatible
async def cloud_storage_download_blob_to_file(
    bucket: str,
    blob: str,
    path: Union[str, Path],
    gcp_credentials: GcpCredentials,
    chunk_size: Optional[int] = None,
    encryption_key: Optional[Any] = None,
    timeout: Union[int, Tuple[int, int]] = 60,
    project: Optional[str] = None,
    **download_kwargs: Any
) -> Union[str, Path]:
    logger = get_run_logger()
    logger.info('Downloading blob named %s from the %s bucket to %s', blob, bucket, path)
    bucket_obj = await _get_bucket_async(bucket, gcp_credentials, project=project)
    blob_obj = bucket_obj.blob(blob, chunk_size=chunk_size, encryption_key=encryption_key)
    if os.path.isdir(str(path)):
        if isinstance(path, Path):
            path = path.joinpath(blob)
        else:
            path = os.path.join(str(path), blob)
    await run_sync_in_worker_thread(blob_obj.download_to_filename, path, timeout=timeout, **download_kwargs)
    return path


@task
@sync_compatible
async def cloud_storage_upload_blob_from_string(
    data: Union[str, bytes],
    bucket: str,
    blob: str,
    gcp_credentials: GcpCredentials,
    content_type: Optional[str] = None,
    chunk_size: Optional[int] = None,
    encryption_key: Optional[Any] = None,
    timeout: Union[int, Tuple[int, int]] = 60,
    project: Optional[str] = None,
    **upload_kwargs: Any
) -> str:
    logger = get_run_logger()
    logger.info('Uploading blob named %s to the %s bucket', blob, bucket)
    bucket_obj = await _get_bucket_async(bucket, gcp_credentials, project=project)
    blob_obj = bucket_obj.blob(blob, chunk_size=chunk_size, encryption_key=encryption_key)
    await run_sync_in_worker_thread(blob_obj.upload_from_string, data, content_type=content_type, timeout=timeout, **upload_kwargs)
    return blob


@task
@sync_compatible
async def cloud_storage_upload_blob_from_file(
    file: Union[str, BinaryIO],
    bucket: str,
    blob: str,
    gcp_credentials: GcpCredentials,
    content_type: Optional[str] = None,
    chunk_size: Optional[int] = None,
    encryption_key: Optional[Any] = None,
    timeout: Union[int, Tuple[int, int]] = 60,
    project: Optional[str] = None,
    **upload_kwargs: Any
) -> str:
    logger = get_run_logger()
    logger.info('Uploading blob named %s to the %s bucket', blob, bucket)
    bucket_obj = await _get_bucket_async(bucket, gcp_credentials, project=project)
    blob_obj = bucket_obj.blob(blob, chunk_size=chunk_size, encryption_key=encryption_key)
    if isinstance(file, BytesIO) or hasattr(file, "read"):
        await run_sync_in_worker_thread(blob_obj.upload_from_file, file, content_type=content_type, timeout=timeout, **upload_kwargs)
    else:
        await run_sync_in_worker_thread(blob_obj.upload_from_filename, file, content_type=content_type, timeout=timeout, **upload_kwargs)
    return blob


@task
def cloud_storage_copy_blob(
    source_bucket: str,
    dest_bucket: str,
    source_blob: str,
    gcp_credentials: GcpCredentials,
    dest_blob: Optional[str] = None,
    timeout: Union[int, Tuple[int, int]] = 60,
    project: Optional[str] = None,
    **copy_kwargs: Any
) -> str:
    logger = get_run_logger()
    logger.info('Copying blob named %s from the %s bucket to the %s bucket', source_blob, source_bucket, dest_bucket)
    source_bucket_obj = _get_bucket(source_bucket, gcp_credentials, project=project)
    dest_bucket_obj = _get_bucket(dest_bucket, gcp_credentials, project=project)
    if dest_blob is None:
        dest_blob = source_blob
    source_blob_obj = source_bucket_obj.blob(source_blob)
    source_bucket_obj.copy_blob(blob=source_blob_obj, destination_bucket=dest_bucket_obj, new_name=dest_blob, timeout=timeout, **copy_kwargs)
    return dest_blob


class DataFrameSerializationFormat(Enum):
    CSV = ('csv', None, 'text/csv', '.csv')
    CSV_GZIP = ('csv', 'gzip', 'application/x-gzip', '.csv.gz')
    PARQUET = ('parquet', None, 'application/octet-stream', '.parquet')
    PARQUET_SNAPPY = ('parquet', 'snappy', 'application/octet-stream', '.snappy.parquet')
    PARQUET_GZIP = ('parquet', 'gzip', 'application/octet-stream', '.gz.parquet')

    @property
    def format(self) -> str:
        return self.value[0]

    @property
    def compression(self) -> Optional[str]:
        return self.value[1]

    @property
    def content_type(self) -> str:
        return self.value[2]

    @property
    def suffix(self) -> str:
        return self.value[3]

    def fix_extension_with(self, gcs_blob_path: str) -> str:
        gcs_blob_path = PurePosixPath(gcs_blob_path)
        folder = gcs_blob_path.parent
        filename = PurePosixPath(gcs_blob_path.stem).with_suffix(self.suffix)
        return str(folder.joinpath(filename))


class GcsBucket(WritableDeploymentStorage, WritableFileSystem, ObjectStorageBlock):
    _logo_url: str = 'https://cdn.sanity.io/images/3ugk85nk/production/10424e311932e31c477ac2b9ef3d53cefbaad708-250x250.png'
    _block_type_name: str = 'GCS Bucket'
    _documentation_url: str = 'https://docs.prefect.io/integrations/prefect-gcp'
    bucket: str = Field(..., description='Name of the bucket.')
    gcp_credentials: GcpCredentials = Field(default_factory=GcpCredentials, description='The credentials to authenticate with GCP.')
    bucket_folder: str = Field(default='', description='A default path to a folder within the GCS bucket to use for reading and writing objects.')

    @property
    def basepath(self) -> str:
        return self.bucket_folder

    @field_validator('bucket_folder')
    @classmethod
    def _bucket_folder_suffix(cls, value: str) -> str:
        if value != '' and (not value.endswith('/')):
            value = f'{value}/'
        return value

    def _resolve_path(self, path: str) -> Optional[str]:
        path = str(PurePosixPath(self.bucket_folder, path)) if self.bucket_folder else path
        if path in ['', '.', '/']:
            path = None
        return path

    @sync_compatible
    async def get_directory(
        self,
        from_path: Optional[str] = None,
        local_path: Optional[str] = None
    ) -> List[str]:
        from_path = self.bucket_folder if from_path is None else self._resolve_path(from_path)
        if local_path is None:
            local_path = os.path.abspath('.')
        else:
            local_path = os.path.abspath(os.path.expanduser(local_path))
        project: Optional[str] = self.gcp_credentials.project
        client = self.gcp_credentials.get_cloud_storage_client(project=project)
        blobs = await run_sync_in_worker_thread(client.list_blobs, self.bucket, prefix=from_path)
        file_paths: List[str] = []
        for blob in blobs:
            blob_path: str = blob.name
            if blob_path[-1] == '/':
                continue
            relative_blob_path: str = os.path.relpath(blob_path, from_path)  # type: ignore
            local_file_path: str = os.path.join(local_path, relative_blob_path)
            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
            with disable_run_logger():
                file_path = await cloud_storage_download_blob_to_file.fn(bucket=self.bucket, blob=blob_path, path=local_file_path, gcp_credentials=self.gcp_credentials)
                file_paths.append(file_path if isinstance(file_path, str) else str(file_path))
        return file_paths

    @sync_compatible
    async def put_directory(
        self,
        local_path: Optional[str] = None,
        to_path: Optional[str] = None,
        ignore_file: Optional[str] = None
    ) -> int:
        if local_path is None:
            local_path = os.path.abspath('.')
        else:
            local_path = os.path.expanduser(local_path)
        to_path = self.bucket_folder if to_path is None else self._resolve_path(to_path)
        included_files: Optional[List[str]] = None
        if ignore_file:
            with open(ignore_file, 'r') as f:
                ignore_patterns = f.readlines()
            included_files = filter_files(local_path, ignore_patterns)
        uploaded_file_count: int = 0
        for local_file_path in Path(local_path).rglob('*'):
            relative_str = str(local_file_path.relative_to(local_path))
            if included_files is not None and relative_str not in included_files:
                continue
            elif not local_file_path.is_dir():
                remote_file_path = str(PurePosixPath(to_path, local_file_path.relative_to(local_path)))
                local_file_content: bytes = local_file_path.read_bytes()
                with disable_run_logger():
                    await cloud_storage_upload_blob_from_string.fn(data=local_file_content, bucket=self.bucket, blob=remote_file_path, gcp_credentials=self.gcp_credentials)
                uploaded_file_count += 1
        return uploaded_file_count

    @sync_compatible
    async def read_path(self, path: str) -> bytes:
        path = self._resolve_path(path)
        with disable_run_logger():
            contents = await cloud_storage_download_blob_as_bytes.fn(bucket=self.bucket, blob=path, gcp_credentials=self.gcp_credentials)
        return contents

    @sync_compatible
    async def write_path(self, path: str, content: Union[str, bytes]) -> str:
        path = self._resolve_path(path)
        with disable_run_logger():
            await cloud_storage_upload_blob_from_string.fn(data=content, bucket=self.bucket, blob=path, gcp_credentials=self.gcp_credentials)
        return path

    def _join_bucket_folder(self, bucket_path: str = '') -> Optional[str]:
        bucket_path = str(bucket_path)
        if self.bucket_folder != '' and bucket_path.startswith(self.bucket_folder):
            self.logger.info(f'Bucket path {bucket_path!r} is already prefixed with bucket folder {self.bucket_folder!r}; is this intentional?')
        bucket_path = str(PurePosixPath(self.bucket_folder) / bucket_path)
        if bucket_path in ['', '.', '/']:
            bucket_path = None
        return bucket_path

    @sync_compatible
    async def create_bucket(self, location: Optional[str] = None, **create_kwargs: Any) -> Bucket:
        self.logger.info(f'Creating bucket {self.bucket!r}.')
        client = self.gcp_credentials.get_cloud_storage_client()
        bucket_obj = await run_sync_in_worker_thread(client.create_bucket, self.bucket, location=location, **create_kwargs)
        return bucket_obj

    @sync_compatible
    async def get_bucket(self) -> Bucket:
        self.logger.info(f'Getting bucket {self.bucket!r}.')
        client = self.gcp_credentials.get_cloud_storage_client()
        bucket_obj = await run_sync_in_worker_thread(client.get_bucket, self.bucket)
        return bucket_obj

    @sync_compatible
    async def list_blobs(self, folder: str = '') -> List[Blob]:
        client = self.gcp_credentials.get_cloud_storage_client()
        bucket_path = self._join_bucket_folder(folder)
        if bucket_path is None:
            self.logger.info(f'Listing blobs in bucket {self.bucket!r}.')
        else:
            self.logger.info(f'Listing blobs in folder {bucket_path!r} in bucket {self.bucket!r}.')
        blobs = await run_sync_in_worker_thread(client.list_blobs, self.bucket, prefix=bucket_path)
        return [blob for blob in blobs if not blob.name.endswith('/')]

    @sync_compatible
    async def list_folders(self, folder: str = '') -> List[str]:
        bucket_path = self._join_bucket_folder(folder)
        if bucket_path is None:
            self.logger.info(f'Listing folders in bucket {self.bucket!r}.')
        else:
            self.logger.info(f'Listing folders in {bucket_path!r} in bucket {self.bucket!r}.')
        blobs = await self.list_blobs(folder)
        folders = {str(PurePosixPath(blob.name).parent) for blob in blobs}
        return [folder for folder in folders if folder != '.']

    @sync_compatible
    async def download_object_to_path(
        self,
        from_path: str,
        to_path: Optional[str] = None,
        **download_kwargs: Any
    ) -> Path:
        if to_path is None:
            to_path = Path(from_path).name
        to_path = Path(to_path).absolute()
        bucket_obj = await self.get_bucket()
        bucket_path = self._join_bucket_folder(from_path)
        blob = bucket_obj.blob(bucket_path)
        self.logger.info(f'Downloading blob from bucket {self.bucket!r} path {bucket_path!r}to {str(to_path)!r}.')
        await run_sync_in_worker_thread(blob.download_to_filename, filename=str(to_path), **download_kwargs)
        return to_path

    @sync_compatible
    async def download_object_to_file_object(
        self,
        from_path: str,
        to_file_object: BinaryIO,
        **download_kwargs: Any
    ) -> BinaryIO:
        bucket_obj = await self.get_bucket()
        bucket_path = self._join_bucket_folder(from_path)
        blob = bucket_obj.blob(bucket_path)
        self.logger.info(f'Downloading blob from bucket {self.bucket!r} path {bucket_path!r}to file object.')
        await run_sync_in_worker_thread(blob.download_to_file, file_obj=to_file_object, **download_kwargs)
        return to_file_object

    @sync_compatible
    async def download_folder_to_path(
        self,
        from_folder: str,
        to_folder: Optional[str] = None,
        **download_kwargs: Any
    ) -> Path:
        if to_folder is None:
            to_folder = ''
        to_folder_path = Path(to_folder).absolute()
        blobs = await self.list_blobs(folder=from_folder)
        if len(blobs) == 0:
            self.logger.warning(f'No blobs were downloaded from bucket {self.bucket!r} path {from_folder!r}.')
            return to_folder_path
        bucket_folder = self._join_bucket_folder(from_folder)
        async_coros: List[Any] = []
        for blob in blobs:
            bucket_path = PurePosixPath(blob.name).relative_to(bucket_folder)  # type: ignore
            if str(bucket_path).endswith('/'):
                continue
            to_path = to_folder_path / bucket_path
            to_path.parent.mkdir(parents=True, exist_ok=True)
            self.logger.info(f'Downloading blob from bucket {self.bucket!r} path {str(bucket_path)!r} to {to_path}.')
            async_coros.append(run_sync_in_worker_thread(blob.download_to_filename, filename=str(to_path), **download_kwargs))
        await asyncio.gather(*async_coros)
        return to_folder_path

    @sync_compatible
    async def upload_from_path(
        self,
        from_path: Union[str, Path],
        to_path: Optional[str] = None,
        **upload_kwargs: Any
    ) -> str:
        if to_path is None:
            to_path = Path(from_path).name  # type: ignore
        bucket_path = self._join_bucket_folder(to_path)
        bucket_obj = await self.get_bucket()
        blob = bucket_obj.blob(bucket_path)
        self.logger.info(f'Uploading from {from_path!r} to the bucket {self.bucket!r} path {bucket_path!r}.')
        await run_sync_in_worker_thread(blob.upload_from_filename, filename=str(from_path), **upload_kwargs)
        return bucket_path

    @sync_compatible
    async def upload_from_file_object(
        self,
        from_file_object: BinaryIO,
        to_path: str,
        **upload_kwargs: Any
    ) -> str:
        bucket_obj = await self.get_bucket()
        bucket_path = self._join_bucket_folder(to_path)
        blob = bucket_obj.blob(bucket_path)
        self.logger.info(f'Uploading from file object to the bucket {self.bucket!r} path {bucket_path!r}.')
        await run_sync_in_worker_thread(blob.upload_from_file, from_file_object, **upload_kwargs)
        return bucket_path

    @sync_compatible
    async def upload_from_folder(
        self,
        from_folder: Union[str, Path],
        to_folder: Optional[str] = None,
        **upload_kwargs: Any
    ) -> str:
        from_folder_path = Path(from_folder)
        bucket_folder = self._join_bucket_folder(to_folder or '') or ''
        num_uploaded: int = 0
        bucket_obj = await self.get_bucket()
        async_coros: List[Any] = []
        for from_path in from_folder_path.rglob('**/*'):
            if from_path.is_dir():
                continue
            bucket_path = str(Path(bucket_folder) / from_path.relative_to(from_folder_path))
            self.logger.info(f'Uploading from {str(from_path)!r} to the bucket {self.bucket!r} path {bucket_path!r}.')
            blob = bucket_obj.blob(bucket_path)
            async_coros.append(run_sync_in_worker_thread(blob.upload_from_filename, filename=str(from_path), **upload_kwargs))
            num_uploaded += 1
        await asyncio.gather(*async_coros)
        if num_uploaded == 0:
            self.logger.warning(f'No files were uploaded from {from_folder}.')
        return bucket_folder

    @sync_compatible
    async def upload_from_dataframe(
        self,
        df: "DataFrame",
        to_path: str,
        serialization_format: Union[DataFrameSerializationFormat, str] = DataFrameSerializationFormat.CSV_GZIP,
        **upload_kwargs: Any
    ) -> str:
        if isinstance(serialization_format, str):
            serialization_format = DataFrameSerializationFormat[serialization_format.upper()]
        with BytesIO() as bytes_buffer:
            if serialization_format.format == 'parquet':
                df.to_parquet(path=bytes_buffer, compression=serialization_format.compression, index=False)
            elif serialization_format.format == 'csv':
                df.to_csv(path_or_buf=bytes_buffer, compression=serialization_format.compression, index=False)
            bytes_buffer.seek(0)
            to_path = serialization_format.fix_extension_with(gcs_blob_path=to_path)
            return await self.upload_from_file_object(from_file_object=bytes_buffer, to_path=to_path, **{'content_type': serialization_format.content_type, **upload_kwargs})
