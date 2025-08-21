"""
Prefect deployment steps for code storage and retrieval in S3 and S3
compatible services.
"""
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Optional, Set, Union
import boto3
from botocore.client import BaseClient, Config
from botocore.paginate import Paginator
from typing_extensions import TypedDict
from prefect._internal.compatibility.deprecated import deprecated_callable
from prefect.utilities.filesystem import filter_files, relative_path_to_current_platform


class PushToS3Output(TypedDict):
    """
    The output of the `push_to_s3` step.
    """
    bucket: str
    folder: str


@deprecated_callable(start_date='Jun 2023', help='Use `PushToS3Output` instead.')
class PushProjectToS3Output(PushToS3Output):
    """Deprecated. Use `PushToS3Output` instead."""


class PullFromS3Output(TypedDict):
    """
    The output of the `pull_from_s3` step.
    """
    bucket: str
    folder: str
    directory: str


@deprecated_callable(start_date='Jun 2023', help='Use `PullFromS3Output` instead.')
class PullProjectFromS3Output(PullFromS3Output):
    """Deprecated. Use `PullFromS3Output` instead.."""


class AwsClientParameters(TypedDict, total=False):
    api_version: str
    endpoint_url: str
    use_ssl: bool
    verify: Union[bool, str, None]
    config: Dict[str, Any]
    profile_name: str
    region_name: str


class CredentialDict(TypedDict, total=False):
    aws_access_key_id: str
    aws_secret_access_key: str
    aws_session_token: str
    minio_root_user: str
    minio_root_password: str
    profile_name: str
    region_name: str
    aws_client_parameters: AwsClientParameters


@deprecated_callable(start_date='Jun 2023', help='Use `push_to_s3` instead.')
def push_project_to_s3(*args: Any, **kwargs: Any) -> None:
    """Deprecated. Use `push_to_s3` instead."""
    push_to_s3(*args, **kwargs)


def push_to_s3(
    bucket: str,
    folder: str,
    credentials: Optional[CredentialDict] = None,
    client_parameters: Optional[AwsClientParameters] = None,
    ignore_file: Optional[str] = '.prefectignore',
) -> PushToS3Output:
    """
    Pushes the contents of the current working directory to an S3 bucket,
    excluding files and folders specified in the ignore_file.
    """
    s3: BaseClient = get_s3_client(credentials=credentials, client_parameters=client_parameters)
    local_path: Path = Path.cwd()
    included_files: Optional[Set[str]] = None
    if ignore_file and Path(ignore_file).exists():
        with open(ignore_file, 'r') as f:
            ignore_patterns: List[str] = f.readlines()
        included_files = set(filter_files(str(local_path), ignore_patterns))
    for local_file_path in local_path.expanduser().rglob('*'):
        if included_files is not None and str(local_file_path.relative_to(local_path)) not in included_files:
            continue
        elif not local_file_path.is_dir():
            remote_file_path: Path = Path(folder) / local_file_path.relative_to(local_path)
            s3.upload_file(str(local_file_path), bucket, str(remote_file_path.as_posix()))
    return {'bucket': bucket, 'folder': folder}


@deprecated_callable(start_date='Jun 2023', help='Use `pull_from_s3` instead.')
def pull_project_from_s3(*args: Any, **kwargs: Any) -> None:
    """Deprecated. Use `pull_from_s3` instead."""
    pull_from_s3(*args, **kwargs)


def pull_from_s3(
    bucket: str,
    folder: str,
    credentials: Optional[CredentialDict] = None,
    client_parameters: Optional[AwsClientParameters] = None,
) -> PullFromS3Output:
    """
    Pulls the contents of an S3 bucket folder to the current working directory.
    """
    s3: BaseClient = get_s3_client(credentials=credentials, client_parameters=client_parameters)
    local_path: Path = Path.cwd()
    paginator: Paginator = s3.get_paginator('list_objects_v2')
    for result in paginator.paginate(Bucket=bucket, Prefix=folder):
        for obj in result.get('Contents', []):
            remote_key: str = obj['Key']
            if remote_key[-1] == '/':
                continue
            target: PurePosixPath = PurePosixPath(
                local_path / relative_path_to_current_platform(remote_key).relative_to(folder)
            )
            Path.mkdir(Path(target.parent), parents=True, exist_ok=True)
            s3.download_file(bucket, remote_key, str(target))
    return {'bucket': bucket, 'folder': folder, 'directory': str(local_path)}


def get_s3_client(
    credentials: Optional[CredentialDict] = None,
    client_parameters: Optional[AwsClientParameters] = None,
) -> BaseClient:
    if credentials is None:
        credentials = {}
    if client_parameters is None:
        client_parameters = {}
    aws_access_key_id: Optional[str] = credentials.get('aws_access_key_id', credentials.get('minio_root_user', None))
    aws_secret_access_key: Optional[str] = credentials.get(
        'aws_secret_access_key', credentials.get('minio_root_password', None)
    )
    aws_session_token: Optional[str] = credentials.get('aws_session_token', None)
    profile_name: Optional[str] = credentials.get('profile_name', client_parameters.get('profile_name', None))
    region_name: Optional[str] = credentials.get('region_name', client_parameters.get('region_name', None))
    aws_client_parameters: AwsClientParameters = credentials.get('aws_client_parameters', client_parameters)  # type: ignore[assignment]
    api_version: Optional[str] = aws_client_parameters.get('api_version', None)  # type: ignore[assignment]
    endpoint_url: Optional[str] = aws_client_parameters.get('endpoint_url', None)  # type: ignore[assignment]
    use_ssl: bool = aws_client_parameters.get('use_ssl', True)  # type: ignore[assignment]
    verify: Union[bool, str, None] = aws_client_parameters.get('verify', None)  # type: ignore[assignment]
    config_params: Dict[str, Any] = aws_client_parameters.get('config', {})
    config: Config = Config(**config_params)
    session = boto3.Session(
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        aws_session_token=aws_session_token,
        profile_name=profile_name,
        region_name=region_name,
    )
    return session.client('s3', api_version=api_version, endpoint_url=endpoint_url, use_ssl=use_ssl, verify=verify, config=config)