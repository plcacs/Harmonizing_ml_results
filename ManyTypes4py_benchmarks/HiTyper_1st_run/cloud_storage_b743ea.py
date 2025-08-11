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
    pass
try:
    from google.cloud.storage import Bucket
    from google.cloud.storage.blob import Blob
except ModuleNotFoundError:
    pass

@task
@sync_compatible
async def cloud_storage_create_bucket(bucket, gcp_credentials, project=None, location=None, **create_kwargs):
    """
    Creates a bucket.

    Args:
        bucket: Name of the bucket.
        gcp_credentials: Credentials to use for authentication with GCP.
        project: Name of the project to use; overrides the
            gcp_credentials project if provided.
        location: Location of the bucket.
        **create_kwargs: Additional keyword arguments to pass to `client.create_bucket`.

    Returns:
        The bucket name.

    Example:
        Creates a bucket named "prefect".
        ```python
        from prefect import flow
        from prefect_gcp import GcpCredentials
        from prefect_gcp.cloud_storage import cloud_storage_create_bucket

        @flow()
        def example_cloud_storage_create_bucket_flow():
            gcp_credentials = GcpCredentials(
                service_account_file="/path/to/service/account/keyfile.json")
            bucket = cloud_storage_create_bucket("prefect", gcp_credentials)

        example_cloud_storage_create_bucket_flow()
        ```
    """
    logger = get_run_logger()
    logger.info('Creating %s bucket', bucket)
    client = gcp_credentials.get_cloud_storage_client(project=project)
    await run_sync_in_worker_thread(client.create_bucket, bucket, location=location, **create_kwargs)
    return bucket

async def _get_bucket_async(bucket, gcp_credentials, project=None):
    """
    Helper function to retrieve a bucket.
    """
    client = gcp_credentials.get_cloud_storage_client(project=project)
    bucket_obj = await run_sync_in_worker_thread(client.get_bucket, bucket)
    return bucket_obj

def _get_bucket(bucket: Union[str, pathlib.Path, None], gcp_credentials: Union[str, None], project: Union[None, str]=None):
    """
    Helper function to retrieve a bucket.
    """
    client = gcp_credentials.get_cloud_storage_client(project=project)
    bucket_obj = client.get_bucket(bucket)
    return bucket_obj

@task
@sync_compatible
async def cloud_storage_download_blob_as_bytes(bucket, blob, gcp_credentials, chunk_size=None, encryption_key=None, timeout=60, project=None, **download_kwargs):
    """
    Downloads a blob as bytes.

    Args:
        bucket: Name of the bucket.
        blob: Name of the Cloud Storage blob.
        gcp_credentials: Credentials to use for authentication with GCP.
        chunk_size (int, optional): The size of a chunk of data whenever
            iterating (in bytes). This must be a multiple of 256 KB
            per the API specification.
        encryption_key: An encryption key.
        timeout: The number of seconds the transport should wait
            for the server response. Can also be passed as a tuple
            (connect_timeout, read_timeout).
        project: Name of the project to use; overrides the
            gcp_credentials project if provided.
        **download_kwargs: Additional keyword arguments to pass to
            `Blob.download_as_bytes`.

    Returns:
        A bytes or string representation of the blob object.

    Example:
        Downloads blob from bucket.
        ```python
        from prefect import flow
        from prefect_gcp import GcpCredentials
        from prefect_gcp.cloud_storage import cloud_storage_download_blob_as_bytes

        @flow()
        def example_cloud_storage_download_blob_flow():
            gcp_credentials = GcpCredentials(
                service_account_file="/path/to/service/account/keyfile.json")
            contents = cloud_storage_download_blob_as_bytes(
                "bucket", "blob", gcp_credentials)
            return contents

        example_cloud_storage_download_blob_flow()
        ```
    """
    logger = get_run_logger()
    logger.info('Downloading blob named %s from the %s bucket', blob, bucket)
    bucket_obj = await _get_bucket_async(bucket, gcp_credentials, project=project)
    blob_obj = bucket_obj.blob(blob, chunk_size=chunk_size, encryption_key=encryption_key)
    contents = await run_sync_in_worker_thread(blob_obj.download_as_bytes, timeout=timeout, **download_kwargs)
    return contents

@task
@sync_compatible
async def cloud_storage_download_blob_to_file(bucket, blob, path, gcp_credentials, chunk_size=None, encryption_key=None, timeout=60, project=None, **download_kwargs):
    """
    Downloads a blob to a file path.

    Args:
        bucket: Name of the bucket.
        blob: Name of the Cloud Storage blob.
        path: Downloads the contents to the provided file path;
            if the path is a directory, automatically joins the blob name.
        gcp_credentials: Credentials to use for authentication with GCP.
        chunk_size (int, optional): The size of a chunk of data whenever
            iterating (in bytes). This must be a multiple of 256 KB
            per the API specification.
        encryption_key: An encryption key.
        timeout: The number of seconds the transport should wait
            for the server response. Can also be passed as a tuple
            (connect_timeout, read_timeout).
        project: Name of the project to use; overrides the
            gcp_credentials project if provided.
        **download_kwargs: Additional keyword arguments to pass to
            `Blob.download_to_filename`.

    Returns:
        The path to the blob object.

    Example:
        Downloads blob from bucket.
        ```python
        from prefect import flow
        from prefect_gcp import GcpCredentials
        from prefect_gcp.cloud_storage import cloud_storage_download_blob_to_file

        @flow()
        def example_cloud_storage_download_blob_flow():
            gcp_credentials = GcpCredentials(
                service_account_file="/path/to/service/account/keyfile.json")
            path = cloud_storage_download_blob_to_file(
                "bucket", "blob", "file_path", gcp_credentials)
            return path

        example_cloud_storage_download_blob_flow()
        ```
    """
    logger = get_run_logger()
    logger.info('Downloading blob named %s from the %s bucket to %s', blob, bucket, path)
    bucket_obj = await _get_bucket_async(bucket, gcp_credentials, project=project)
    blob_obj = bucket_obj.blob(blob, chunk_size=chunk_size, encryption_key=encryption_key)
    if os.path.isdir(path):
        if isinstance(path, Path):
            path = path.joinpath(blob)
        else:
            path = os.path.join(path, blob)
    await run_sync_in_worker_thread(blob_obj.download_to_filename, path, timeout=timeout, **download_kwargs)
    return path

@task
@sync_compatible
async def cloud_storage_upload_blob_from_string(data, bucket, blob, gcp_credentials, content_type=None, chunk_size=None, encryption_key=None, timeout=60, project=None, **upload_kwargs):
    """
    Uploads a blob from a string or bytes representation of data.

    Args:
        data: String or bytes representation of data to upload.
        bucket: Name of the bucket.
        blob: Name of the Cloud Storage blob.
        gcp_credentials: Credentials to use for authentication with GCP.
        content_type: Type of content being uploaded.
        chunk_size: The size of a chunk of data whenever
            iterating (in bytes). This must be a multiple of 256 KB
            per the API specification.
        encryption_key: An encryption key.
        timeout: The number of seconds the transport should wait
            for the server response. Can also be passed as a tuple
            (connect_timeout, read_timeout).
        project: Name of the project to use; overrides the
            gcp_credentials project if provided.
        **upload_kwargs: Additional keyword arguments to pass to
            `Blob.upload_from_string`.

    Returns:
        The blob name.

    Example:
        Uploads blob to bucket.
        ```python
        from prefect import flow
        from prefect_gcp import GcpCredentials
        from prefect_gcp.cloud_storage import cloud_storage_upload_blob_from_string

        @flow()
        def example_cloud_storage_upload_blob_from_string_flow():
            gcp_credentials = GcpCredentials(
                service_account_file="/path/to/service/account/keyfile.json")
            blob = cloud_storage_upload_blob_from_string(
                "data", "bucket", "blob", gcp_credentials)
            return blob

        example_cloud_storage_upload_blob_from_string_flow()
        ```
    """
    logger = get_run_logger()
    logger.info('Uploading blob named %s to the %s bucket', blob, bucket)
    bucket_obj = await _get_bucket_async(bucket, gcp_credentials, project=project)
    blob_obj = bucket_obj.blob(blob, chunk_size=chunk_size, encryption_key=encryption_key)
    await run_sync_in_worker_thread(blob_obj.upload_from_string, data, content_type=content_type, timeout=timeout, **upload_kwargs)
    return blob

@task
@sync_compatible
async def cloud_storage_upload_blob_from_file(file, bucket, blob, gcp_credentials, content_type=None, chunk_size=None, encryption_key=None, timeout=60, project=None, **upload_kwargs):
    """
    Uploads a blob from file path or file-like object. Usage for passing in
    file-like object is if the data was downloaded from the web;
    can bypass writing to disk and directly upload to Cloud Storage.

    Args:
        file: Path to data or file like object to upload.
        bucket: Name of the bucket.
        blob: Name of the Cloud Storage blob.
        gcp_credentials: Credentials to use for authentication with GCP.
        content_type: Type of content being uploaded.
        chunk_size: The size of a chunk of data whenever
            iterating (in bytes). This must be a multiple of 256 KB
            per the API specification.
        encryption_key: An encryption key.
        timeout: The number of seconds the transport should wait
            for the server response. Can also be passed as a tuple
            (connect_timeout, read_timeout).
        project: Name of the project to use; overrides the
            gcp_credentials project if provided.
        **upload_kwargs: Additional keyword arguments to pass to
            `Blob.upload_from_file` or `Blob.upload_from_filename`.

    Returns:
        The blob name.

    Example:
        Uploads blob to bucket.
        ```python
        from prefect import flow
        from prefect_gcp import GcpCredentials
        from prefect_gcp.cloud_storage import cloud_storage_upload_blob_from_file

        @flow()
        def example_cloud_storage_upload_blob_from_file_flow():
            gcp_credentials = GcpCredentials(
                service_account_file="/path/to/service/account/keyfile.json")
            blob = cloud_storage_upload_blob_from_file(
                "/path/somewhere", "bucket", "blob", gcp_credentials)
            return blob

        example_cloud_storage_upload_blob_from_file_flow()
        ```
    """
    logger = get_run_logger()
    logger.info('Uploading blob named %s to the %s bucket', blob, bucket)
    bucket_obj = await _get_bucket_async(bucket, gcp_credentials, project=project)
    blob_obj = bucket_obj.blob(blob, chunk_size=chunk_size, encryption_key=encryption_key)
    if isinstance(file, BytesIO):
        await run_sync_in_worker_thread(blob_obj.upload_from_file, file, content_type=content_type, timeout=timeout, **upload_kwargs)
    else:
        await run_sync_in_worker_thread(blob_obj.upload_from_filename, file, content_type=content_type, timeout=timeout, **upload_kwargs)
    return blob

@task
def cloud_storage_copy_blob(source_bucket: Union[str, typing.BinaryIO, list[str]], dest_bucket: Union[str, typing.BinaryIO], source_blob: Union[str, typing.BinaryIO], gcp_credentials: Union[str, None, dict[str, str]], dest_blob: Union[None, str, bytes]=None, timeout: int=60, project: Union[None, str, dict[str, str]]=None, **copy_kwargs) -> Union[typing.IO, None, io.BufferedReader, str, typing.BinaryIO]:
    """
    Copies data from one Google Cloud Storage bucket to another,
    without downloading it locally.

    Args:
        source_bucket: Source bucket name.
        dest_bucket: Destination bucket name.
        source_blob: Source blob name.
        gcp_credentials: Credentials to use for authentication with GCP.
        dest_blob: Destination blob name; if not provided, defaults to source_blob.
        timeout: The number of seconds the transport should wait
            for the server response. Can also be passed as a tuple
            (connect_timeout, read_timeout).
        project: Name of the project to use; overrides the
            gcp_credentials project if provided.
        **copy_kwargs: Additional keyword arguments to pass to
            `Bucket.copy_blob`.

    Returns:
        Destination blob name.

    Example:
        Copies blob from one bucket to another.
        ```python
        from prefect import flow
        from prefect_gcp import GcpCredentials
        from prefect_gcp.cloud_storage import cloud_storage_copy_blob

        @flow()
        def example_cloud_storage_copy_blob_flow():
            gcp_credentials = GcpCredentials(
                service_account_file="/path/to/service/account/keyfile.json")
            blob = cloud_storage_copy_blob(
                "source_bucket",
                "dest_bucket",
                "source_blob",
                gcp_credentials
            )
            return blob

        example_cloud_storage_copy_blob_flow()
        ```
    """
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
    """
    An enumeration class to represent different file formats,
    compression options for upload_from_dataframe

    Attributes:
        CSV: Representation for 'csv' file format with no compression
            and its related content type and suffix.

        CSV_GZIP: Representation for 'csv' file format with 'gzip' compression
            and its related content type and suffix.

        PARQUET: Representation for 'parquet' file format with no compression
            and its related content type and suffix.

        PARQUET_SNAPPY: Representation for 'parquet' file format
            with 'snappy' compression and its related content type and suffix.

        PARQUET_GZIP: Representation for 'parquet' file format
            with 'gzip' compression and its related content type and suffix.
    """
    CSV = ('csv', None, 'text/csv', '.csv')
    CSV_GZIP = ('csv', 'gzip', 'application/x-gzip', '.csv.gz')
    PARQUET = ('parquet', None, 'application/octet-stream', '.parquet')
    PARQUET_SNAPPY = ('parquet', 'snappy', 'application/octet-stream', '.snappy.parquet')
    PARQUET_GZIP = ('parquet', 'gzip', 'application/octet-stream', '.gz.parquet')

    @property
    def format(self):
        """The file format of the current instance."""
        return self.value[0]

    @property
    def compression(self):
        """The compression type of the current instance."""
        return self.value[1]

    @property
    def content_type(self):
        """The content type of the current instance."""
        return self.value[2]

    @property
    def suffix(self):
        """The suffix of the file format of the current instance."""
        return self.value[3]

    def fix_extension_with(self, gcs_blob_path: Union[pathlib.Path, str, pathlib.PurePath]) -> str:
        """Fix the extension of a GCS blob.

        Args:
            gcs_blob_path: The path to the GCS blob to be modified.

        Returns:
            The modified path to the GCS blob with the new extension.
        """
        gcs_blob_path = PurePosixPath(gcs_blob_path)
        folder = gcs_blob_path.parent
        filename = PurePosixPath(gcs_blob_path.stem).with_suffix(self.suffix)
        return str(folder.joinpath(filename))

class GcsBucket(WritableDeploymentStorage, WritableFileSystem, ObjectStorageBlock):
    """
    Block used to store data using GCP Cloud Storage Buckets.

    Note! `GcsBucket` in `prefect-gcp` is a unique block, separate from `GCS`
    in core Prefect. `GcsBucket` does not use `gcsfs` under the hood,
    instead using the `google-cloud-storage` package, and offers more configuration
    and functionality.

    Attributes:
        bucket: Name of the bucket.
        gcp_credentials: The credentials to authenticate with GCP.
        bucket_folder: A default path to a folder within the GCS bucket to use
            for reading and writing objects.

    Example:
        Load stored GCP Cloud Storage Bucket:
        ```python
        from prefect_gcp.cloud_storage import GcsBucket
        gcp_cloud_storage_bucket_block = GcsBucket.load("BLOCK_NAME")
        ```
    """
    _logo_url = 'https://cdn.sanity.io/images/3ugk85nk/production/10424e311932e31c477ac2b9ef3d53cefbaad708-250x250.png'
    _block_type_name = 'GCS Bucket'
    _documentation_url = 'https://docs.prefect.io/integrations/prefect-gcp'
    bucket = Field(..., description='Name of the bucket.')
    gcp_credentials = Field(default_factory=GcpCredentials, description='The credentials to authenticate with GCP.')
    bucket_folder = Field(default='', description='A default path to a folder within the GCS bucket to use for reading and writing objects.')

    @property
    def basepath(self):
        """
        Read-only property that mirrors the bucket folder.

        Used for deployment.
        """
        return self.bucket_folder

    @field_validator('bucket_folder')
    @classmethod
    def _bucket_folder_suffix(cls: Union[str, None, bytes], value: Union[str, bytes]) -> Union[str, bytes]:
        """
        Ensures that the bucket folder is suffixed with a forward slash.
        """
        if value != '' and (not value.endswith('/')):
            value = f'{value}/'
        return value

    def _resolve_path(self, path: Union[str, pathlib.PurePath, list[int]]) -> Union[str, pathlib.PurePath, list[int], None]:
        """
        A helper function used in write_path to join `self.bucket_folder` and `path`.

        Args:
            path: Name of the key, e.g. "file1". Each object in your
                bucket has a unique key (or key name).

        Returns:
            The joined path.
        """
        path = str(PurePosixPath(self.bucket_folder, path)) if self.bucket_folder else path
        if path in ['', '.', '/']:
            path = None
        return path

    @sync_compatible
    async def get_directory(self, from_path=None, local_path=None):
        """
        Copies a folder from the configured GCS bucket to a local directory.
        Defaults to copying the entire contents of the block's bucket_folder
        to the current working directory.

        Args:
            from_path: Path in GCS bucket to download from. Defaults to the block's
                configured bucket_folder.
            local_path: Local path to download GCS bucket contents to.
                Defaults to the current working directory.

        Returns:
            A list of downloaded file paths.
        """
        from_path = self.bucket_folder if from_path is None else self._resolve_path(from_path)
        if local_path is None:
            local_path = os.path.abspath('.')
        else:
            local_path = os.path.abspath(os.path.expanduser(local_path))
        project = self.gcp_credentials.project
        client = self.gcp_credentials.get_cloud_storage_client(project=project)
        blobs = await run_sync_in_worker_thread(client.list_blobs, self.bucket, prefix=from_path)
        file_paths = []
        for blob in blobs:
            blob_path = blob.name
            if blob_path[-1] == '/':
                continue
            relative_blob_path = os.path.relpath(blob_path, from_path)
            local_file_path = os.path.join(local_path, relative_blob_path)
            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
            with disable_run_logger():
                file_path = await cloud_storage_download_blob_to_file.fn(bucket=self.bucket, blob=blob_path, path=local_file_path, gcp_credentials=self.gcp_credentials)
                file_paths.append(file_path)
        return file_paths

    @sync_compatible
    async def put_directory(self, local_path=None, to_path=None, ignore_file=None):
        """
        Uploads a directory from a given local path to the configured GCS bucket in a
        given folder.

        Defaults to uploading the entire contents the current working directory to the
        block's bucket_folder.

        Args:
            local_path: Path to local directory to upload from.
            to_path: Path in GCS bucket to upload to. Defaults to block's configured
                bucket_folder.
            ignore_file: Path to file containing gitignore style expressions for
                filepaths to ignore.

        Returns:
            The number of files uploaded.
        """
        if local_path is None:
            local_path = os.path.abspath('.')
        else:
            local_path = os.path.expanduser(local_path)
        to_path = self.bucket_folder if to_path is None else self._resolve_path(to_path)
        included_files = None
        if ignore_file:
            with open(ignore_file, 'r') as f:
                ignore_patterns = f.readlines()
            included_files = filter_files(local_path, ignore_patterns)
        uploaded_file_count = 0
        for local_file_path in Path(local_path).rglob('*'):
            if included_files is not None and str(local_file_path.relative_to(local_path)) not in included_files:
                continue
            elif not local_file_path.is_dir():
                remote_file_path = str(PurePosixPath(to_path, local_file_path.relative_to(local_path)))
                local_file_content = local_file_path.read_bytes()
                with disable_run_logger():
                    await cloud_storage_upload_blob_from_string.fn(data=local_file_content, bucket=self.bucket, blob=remote_file_path, gcp_credentials=self.gcp_credentials)
                uploaded_file_count += 1
        return uploaded_file_count

    @sync_compatible
    async def read_path(self, path):
        """
        Read specified path from GCS and return contents. Provide the entire
        path to the key in GCS.

        Args:
            path: Entire path to (and including) the key.

        Returns:
            A bytes or string representation of the blob object.
        """
        path = self._resolve_path(path)
        with disable_run_logger():
            contents = await cloud_storage_download_blob_as_bytes.fn(bucket=self.bucket, blob=path, gcp_credentials=self.gcp_credentials)
        return contents

    @sync_compatible
    async def write_path(self, path, content):
        """
        Writes to an GCS bucket.

        Args:
            path: The key name. Each object in your bucket has a unique
                key (or key name).
            content: What you are uploading to GCS Bucket.

        Returns:
            The path that the contents were written to.
        """
        path = self._resolve_path(path)
        with disable_run_logger():
            await cloud_storage_upload_blob_from_string.fn(data=content, bucket=self.bucket, blob=path, gcp_credentials=self.gcp_credentials)
        return path

    def _join_bucket_folder(self, bucket_path: typing.Text='') -> Union[str, None]:
        """
        Joins the base bucket folder to the bucket path.

        NOTE: If a method reuses another method in this class, be careful to not
        call this  twice because it'll join the bucket folder twice.
        See https://github.com/PrefectHQ/prefect-aws/issues/141 for a past issue.
        """
        bucket_path = str(bucket_path)
        if self.bucket_folder != '' and bucket_path.startswith(self.bucket_folder):
            self.logger.info(f'Bucket path {bucket_path!r} is already prefixed with bucket folder {self.bucket_folder!r}; is this intentional?')
        bucket_path = str(PurePosixPath(self.bucket_folder) / bucket_path)
        if bucket_path in ['', '.', '/']:
            bucket_path = None
        return bucket_path

    @sync_compatible
    async def create_bucket(self, location=None, **create_kwargs):
        """
        Creates a bucket.

        Args:
            location: The location of the bucket.
            **create_kwargs: Additional keyword arguments to pass to the
                `create_bucket` method.

        Returns:
            The bucket object.

        Examples:
            Create a bucket.
            ```python
            from prefect_gcp.cloud_storage import GcsBucket

            gcs_bucket = GcsBucket(bucket="my-bucket")
            gcs_bucket.create_bucket()
            ```
        """
        self.logger.info(f'Creating bucket {self.bucket!r}.')
        client = self.gcp_credentials.get_cloud_storage_client()
        bucket = await run_sync_in_worker_thread(client.create_bucket, self.bucket, location=location, **create_kwargs)
        return bucket

    @sync_compatible
    async def get_bucket(self):
        """
        Returns the bucket object.

        Returns:
            The bucket object.

        Examples:
            Get the bucket object.
            ```python
            from prefect_gcp.cloud_storage import GcsBucket

            gcs_bucket = GcsBucket.load("my-bucket")
            gcs_bucket.get_bucket()
            ```
        """
        self.logger.info(f'Getting bucket {self.bucket!r}.')
        client = self.gcp_credentials.get_cloud_storage_client()
        bucket = await run_sync_in_worker_thread(client.get_bucket, self.bucket)
        return bucket

    @sync_compatible
    async def list_blobs(self, folder=''):
        """
        Lists all blobs in the bucket that are in a folder.
        Folders are not included in the output.

        Args:
            folder: The folder to list blobs from.

        Returns:
            A list of Blob objects.

        Examples:
            Get all blobs from a folder named "prefect".
            ```python
            from prefect_gcp.cloud_storage import GcsBucket

            gcs_bucket = GcsBucket.load("my-bucket")
            gcs_bucket.list_blobs("prefect")
            ```
        """
        client = self.gcp_credentials.get_cloud_storage_client()
        bucket_path = self._join_bucket_folder(folder)
        if bucket_path is None:
            self.logger.info(f'Listing blobs in bucket {self.bucket!r}.')
        else:
            self.logger.info(f'Listing blobs in folder {bucket_path!r} in bucket {self.bucket!r}.')
        blobs = await run_sync_in_worker_thread(client.list_blobs, self.bucket, prefix=bucket_path)
        return [blob for blob in blobs if not blob.name.endswith('/')]

    @sync_compatible
    async def list_folders(self, folder=''):
        """
        Lists all folders and subfolders in the bucket.

        Args:
            folder: List all folders and subfolders inside given folder.

        Returns:
            A list of folders.

        Examples:
            Get all folders from a bucket named "my-bucket".
            ```python
            from prefect_gcp.cloud_storage import GcsBucket

            gcs_bucket = GcsBucket.load("my-bucket")
            gcs_bucket.list_folders()
            ```

            Get all folders from a folder called years
            ```python
            from prefect_gcp.cloud_storage import GcsBucket

            gcs_bucket = GcsBucket.load("my-bucket")
            gcs_bucket.list_folders("years")
            ```
        """
        bucket_path = self._join_bucket_folder(folder)
        if bucket_path is None:
            self.logger.info(f'Listing folders in bucket {self.bucket!r}.')
        else:
            self.logger.info(f'Listing folders in {bucket_path!r} in bucket {self.bucket!r}.')
        blobs = await self.list_blobs(folder)
        folders = {str(PurePosixPath(blob.name).parent) for blob in blobs}
        return [folder for folder in folders if folder != '.']

    @sync_compatible
    async def download_object_to_path(self, from_path, to_path=None, **download_kwargs):
        """
        Downloads an object from the object storage service to a path.

        Args:
            from_path: The path to the blob to download; this gets prefixed
                with the bucket_folder.
            to_path: The path to download the blob to. If not provided, the
                blob's name will be used.
            **download_kwargs: Additional keyword arguments to pass to
                `Blob.download_to_filename`.

        Returns:
            The absolute path that the object was downloaded to.

        Examples:
            Download my_folder/notes.txt object to notes.txt.
            ```python
            from prefect_gcp.cloud_storage import GcsBucket

            gcs_bucket = GcsBucket.load("my-bucket")
            gcs_bucket.download_object_to_path("my_folder/notes.txt", "notes.txt")
            ```
        """
        if to_path is None:
            to_path = Path(from_path).name
        to_path = str(Path(to_path).absolute())
        bucket = await self.get_bucket()
        bucket_path = self._join_bucket_folder(from_path)
        blob = bucket.blob(bucket_path)
        self.logger.info(f'Downloading blob from bucket {self.bucket!r} path {bucket_path!r}to {to_path!r}.')
        await run_sync_in_worker_thread(blob.download_to_filename, filename=to_path, **download_kwargs)
        return Path(to_path)

    @sync_compatible
    async def download_object_to_file_object(self, from_path, to_file_object, **download_kwargs):
        """
        Downloads an object from the object storage service to a file-like object,
        which can be a BytesIO object or a BufferedWriter.

        Args:
            from_path: The path to the blob to download from; this gets prefixed
                with the bucket_folder.
            to_file_object: The file-like object to download the blob to.
            **download_kwargs: Additional keyword arguments to pass to
                `Blob.download_to_file`.

        Returns:
            The file-like object that the object was downloaded to.

        Examples:
            Download my_folder/notes.txt object to a BytesIO object.
            ```python
            from io import BytesIO
            from prefect_gcp.cloud_storage import GcsBucket

            gcs_bucket = GcsBucket.load("my-bucket")
            with BytesIO() as buf:
                gcs_bucket.download_object_to_file_object("my_folder/notes.txt", buf)
            ```

            Download my_folder/notes.txt object to a BufferedWriter.
            ```python
                from prefect_gcp.cloud_storage import GcsBucket

                gcs_bucket = GcsBucket.load("my-bucket")
                with open("notes.txt", "wb") as f:
                    gcs_bucket.download_object_to_file_object("my_folder/notes.txt", f)
            ```
        """
        bucket = await self.get_bucket()
        bucket_path = self._join_bucket_folder(from_path)
        blob = bucket.blob(bucket_path)
        self.logger.info(f'Downloading blob from bucket {self.bucket!r} path {bucket_path!r}to file object.')
        await run_sync_in_worker_thread(blob.download_to_file, file_obj=to_file_object, **download_kwargs)
        return to_file_object

    @sync_compatible
    async def download_folder_to_path(self, from_folder, to_folder=None, **download_kwargs):
        """
        Downloads objects *within* a folder (excluding the folder itself)
        from the object storage service to a folder.

        Args:
            from_folder: The path to the folder to download from; this gets prefixed
                with the bucket_folder.
            to_folder: The path to download the folder to. If not provided, will default
                to the current directory.
            **download_kwargs: Additional keyword arguments to pass to
                `Blob.download_to_filename`.

        Returns:
            The absolute path that the folder was downloaded to.

        Examples:
            Download my_folder to a local folder named my_folder.
            ```python
            from prefect_gcp.cloud_storage import GcsBucket

            gcs_bucket = GcsBucket.load("my-bucket")
            gcs_bucket.download_folder_to_path("my_folder", "my_folder")
            ```
        """
        if to_folder is None:
            to_folder = ''
        to_folder = Path(to_folder).absolute()
        blobs = await self.list_blobs(folder=from_folder)
        if len(blobs) == 0:
            self.logger.warning(f'No blobs were downloaded from bucket {self.bucket!r} path {from_folder!r}.')
            return to_folder
        bucket_folder = self._join_bucket_folder(from_folder)
        async_coros = []
        for blob in blobs:
            bucket_path = PurePosixPath(blob.name).relative_to(bucket_folder)
            if str(bucket_path).endswith('/'):
                continue
            to_path = to_folder / bucket_path
            to_path.parent.mkdir(parents=True, exist_ok=True)
            self.logger.info(f'Downloading blob from bucket {self.bucket!r} path {str(bucket_path)!r} to {to_path}.')
            async_coros.append(run_sync_in_worker_thread(blob.download_to_filename, filename=str(to_path), **download_kwargs))
        await asyncio.gather(*async_coros)
        return to_folder

    @sync_compatible
    async def upload_from_path(self, from_path, to_path=None, **upload_kwargs):
        """
        Uploads an object from a path to the object storage service.

        Args:
            from_path: The path to the file to upload from.
            to_path: The path to upload the file to. If not provided, will use
                the file name of from_path; this gets prefixed
                with the bucket_folder.
            **upload_kwargs: Additional keyword arguments to pass to
                `Blob.upload_from_filename`.

        Returns:
            The path that the object was uploaded to.

        Examples:
            Upload notes.txt to my_folder/notes.txt.
            ```python
            from prefect_gcp.cloud_storage import GcsBucket

            gcs_bucket = GcsBucket.load("my-bucket")
            gcs_bucket.upload_from_path("notes.txt", "my_folder/notes.txt")
            ```
        """
        if to_path is None:
            to_path = Path(from_path).name
        bucket_path = self._join_bucket_folder(to_path)
        bucket = await self.get_bucket()
        blob = bucket.blob(bucket_path)
        self.logger.info(f'Uploading from {from_path!r} to the bucket {self.bucket!r} path {bucket_path!r}.')
        await run_sync_in_worker_thread(blob.upload_from_filename, filename=from_path, **upload_kwargs)
        return bucket_path

    @sync_compatible
    async def upload_from_file_object(self, from_file_object, to_path, **upload_kwargs):
        """
        Uploads an object to the object storage service from a file-like object,
        which can be a BytesIO object or a BufferedReader.

        Args:
            from_file_object: The file-like object to upload from.
            to_path: The path to upload the object to; this gets prefixed
                with the bucket_folder.
            **upload_kwargs: Additional keyword arguments to pass to
                `Blob.upload_from_file`.

        Returns:
            The path that the object was uploaded to.

        Examples:
            Upload my_folder/notes.txt object to a BytesIO object.
            ```python
            from io import BytesIO
            from prefect_gcp.cloud_storage import GcsBucket

            gcs_bucket = GcsBucket.load("my-bucket")
            with open("notes.txt", "rb") as f:
                gcs_bucket.upload_from_file_object(f, "my_folder/notes.txt")
            ```

            Upload BufferedReader object to my_folder/notes.txt.
            ```python
            from io import BufferedReader
            from prefect_gcp.cloud_storage import GcsBucket

            gcs_bucket = GcsBucket.load("my-bucket")
            with open("notes.txt", "rb") as f:
                gcs_bucket.upload_from_file_object(
                    BufferedReader(f), "my_folder/notes.txt"
                )
            ```
        """
        bucket = await self.get_bucket()
        bucket_path = self._join_bucket_folder(to_path)
        blob = bucket.blob(bucket_path)
        self.logger.info(f'Uploading from file object to the bucket {self.bucket!r} path {bucket_path!r}.')
        await run_sync_in_worker_thread(blob.upload_from_file, from_file_object, **upload_kwargs)
        return bucket_path

    @sync_compatible
    async def upload_from_folder(self, from_folder, to_folder=None, **upload_kwargs):
        """
        Uploads files *within* a folder (excluding the folder itself)
        to the object storage service folder.

        Args:
            from_folder: The path to the folder to upload from.
            to_folder: The path to upload the folder to. If not provided, will default
                to bucket_folder or the base directory of the bucket.
            **upload_kwargs: Additional keyword arguments to pass to
                `Blob.upload_from_filename`.

        Returns:
            The path that the folder was uploaded to.

        Examples:
            Upload local folder my_folder to the bucket's folder my_folder.
            ```python
            from prefect_gcp.cloud_storage import GcsBucket

            gcs_bucket = GcsBucket.load("my-bucket")
            gcs_bucket.upload_from_folder("my_folder")
            ```
        """
        from_folder = Path(from_folder)
        bucket_folder = self._join_bucket_folder(to_folder or '') or ''
        num_uploaded = 0
        bucket = await self.get_bucket()
        async_coros = []
        for from_path in from_folder.rglob('**/*'):
            if from_path.is_dir():
                continue
            bucket_path = str(Path(bucket_folder) / from_path.relative_to(from_folder))
            self.logger.info(f'Uploading from {str(from_path)!r} to the bucket {self.bucket!r} path {bucket_path!r}.')
            blob = bucket.blob(bucket_path)
            async_coros.append(run_sync_in_worker_thread(blob.upload_from_filename, filename=from_path, **upload_kwargs))
            num_uploaded += 1
        await asyncio.gather(*async_coros)
        if num_uploaded == 0:
            self.logger.warning(f'No files were uploaded from {from_folder}.')
        return bucket_folder

    @sync_compatible
    async def upload_from_dataframe(self, df, to_path, serialization_format=DataFrameSerializationFormat.CSV_GZIP, **upload_kwargs):
        """
        Upload a Pandas DataFrame to Google Cloud Storage in various formats.

        This function uploads the data in a Pandas DataFrame to Google Cloud Storage
        in a specified format, such as .csv, .csv.gz, .parquet,
        .parquet.snappy, and .parquet.gz.

        Args:
            df: The Pandas DataFrame to be uploaded.
            to_path: The destination path for the uploaded DataFrame.
            serialization_format: The format to serialize the DataFrame into.
                When passed as a `str`, the valid options are:
                'csv', 'csv_gzip',  'parquet', 'parquet_snappy', 'parquet_gzip'.
                Defaults to `DataFrameSerializationFormat.CSV_GZIP`.
            **upload_kwargs: Additional keyword arguments to pass to the underlying
                `upload_from_dataframe` method.

        Returns:
            The path that the object was uploaded to.
        """
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