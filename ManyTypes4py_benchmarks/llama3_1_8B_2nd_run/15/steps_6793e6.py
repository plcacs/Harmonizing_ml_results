"""
Prefect deployment steps for code storage and retrieval in S3 and S3
compatible services.
"""
from pathlib import Path, PurePosixPath
from typing import Dict, Optional
import boto3
from botocore.client import Config
from typing_extensions import TypedDict
from prefect._internal.compatibility.deprecated import deprecated_callable
from prefect.utilities.filesystem import filter_files, relative_path_to_current_platform

class PushToS3Output(TypedDict):
    """
    The output of the `push_to_s3` step.
    """

@deprecated_callable(start_date='Jun 2023', help='Use `PushToS3Output` instead.')
class PushProjectToS3Output(PushToS3Output):
    """Deprecated. Use `PushToS3Output` instead."""

class PullFromS3Output(TypedDict):
    """
    The output of the `pull_from_s3` step.
    """

@deprecated_callable(start_date='Jun 2023', help='Use `PullFromS3Output` instead.')
class PullProjectFromS3Output(PullFromS3Output):
    """Deprecated. Use `PullFromS3Output` instead.."""

@deprecated_callable(start_date='Jun 2023', help='Use `push_to_s3` instead.')
def push_project_to_s3(*args, **kwargs) -> None:
    """Deprecated. Use `push_to_s3` instead."""
    push_to_s3(*args, **kwargs)

def push_to_s3(
    bucket: str,
    folder: str,
    credentials: Optional[Dict[str, str]] = None,
    client_parameters: Optional[Dict[str, str]] = None,
    ignore_file: str = '.prefectignore'
) -> PushToS3Output:
    """
    Pushes the contents of the current working directory to an S3 bucket,
    excluding files and folders specified in the ignore_file.

    Args:
        bucket: The name of the S3 bucket where files will be uploaded.
        folder: The folder in the S3 bucket where files will be uploaded.
        credentials: A dictionary of AWS credentials (aws_access_key_id,
            aws_secret_access_key, aws_session_token) or MinIO credentials
            (minio_root_user, minio_root_password).
        client_parameters: A dictionary of additional parameters to pass to the boto3
            client.
        ignore_file: The name of the file containing ignore patterns.

    Returns:
        A dictionary containing the bucket and folder where files were uploaded.

    Examples:
        Push files to an S3 bucket:
        