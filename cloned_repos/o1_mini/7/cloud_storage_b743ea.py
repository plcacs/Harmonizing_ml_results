"""Tasks for interacting with GCP Cloud Storage."""
import asyncio
import os
from enum import Enum
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Dict, List, Optional, Tuple, Union, Coroutine
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
    DataFrame = Any
try:
    from google.cloud.storage import Bucket
    from google.cloud.storage.blob import Blob
except ModuleNotFoundError:
    Bucket = Any
    Blob = Any


@task
@sync_compatible
async def cloud_storage_create_bucket(
    bucket: str,
    gcp_credentials: GcpCredentials,
    project: Optional[str] = None,
    location: Optional[str] = None,
    **create_kwargs: Any
) -> str:
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
        