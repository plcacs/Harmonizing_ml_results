from typing import List, Union, Optional, Dict, Any, AsyncGenerator, AsyncIterator, Generator, Tuple

import io
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import boto3
import pytest
from botocore.exceptions import ClientError, EndpointConnectionError
from moto import mock_s3
from prefect_aws import AwsCredentials, MinIOCredentials
from prefect_aws.client_parameters import AwsClientParameters
from prefect_aws.s3 import S3Bucket, acopy_objects, adownload_from_bucket, alist_objects, amove_objects, s3_copy, s3_download, s3_list_objects, s3_move, s3_upload
from prefect import flow

aws_clients: List[str] = ['aws_client_parameters_custom_endpoint', 'aws_client_parameters_empty', 'aws_client_parameters_public_bucket']

def s3_mock(monkeypatch, client_parameters: AwsClientParameters) -> Generator:
    ...

def client_parameters(request: Any) -> AwsClientParameters:
    ...

def bucket(s3_mock, request: Any) -> boto3.Bucket:
    ...

def bucket_2(s3_mock, request: Any) -> boto3.Bucket:
    ...

def object(bucket: boto3.Bucket, tmp_path: Path) -> Any:
    ...

def object_in_folder(bucket: boto3.Bucket, tmp_path: Path) -> Any:
    ...

def objects_in_folder(bucket: boto3.Bucket, tmp_path: Path) -> List[Any]:
    ...

def a_lot_of_objects(bucket: boto3.Bucket, tmp_path: Path) -> List[Any]:
    ...

async def test_s3_download_failed_with_wrong_endpoint_setup(object: Any, client_parameters: AwsClientParameters, aws_credentials: AwsCredentials) -> None:
    ...

async def test_s3_download(object: Any, client_parameters: AwsClientParameters, aws_credentials: AwsCredentials) -> None:
    ...

async def test_s3_download_object_not_found(object: Any, client_parameters: AwsClientParameters, aws_credentials: AwsCredentials) -> None:
    ...

async def test_s3_upload(bucket: boto3.Bucket, client_parameters: AwsClientParameters, tmp_path: Path, aws_credentials: AwsCredentials) -> None:
    ...

async def test_s3_copy(object: Any, bucket: boto3.Bucket, bucket_2: boto3.Bucket, aws_credentials: AwsCredentials) -> None:
    ...

async def test_s3_move(object: Any, bucket: boto3.Bucket, bucket_2: boto3.Bucket, aws_credentials: AwsCredentials) -> None:
    ...

async def test_move_object_to_nonexistent_bucket_fails(object: Any, bucket: boto3.Bucket, aws_credentials: AwsCredentials) -> None:
    ...

async def test_move_object_fail_cases(object: Any, bucket: boto3.Bucket, aws_credentials: AwsCredentials) -> None:
    ...

async def test_s3_list_objects(object: Any, client_parameters: AwsClientParameters, object_in_folder: Any, aws_credentials: AwsCredentials) -> None:
    ...

async def test_s3_list_objects_multiple_pages(a_lot_of_objects: List[Any], client_parameters: AwsClientParameters, aws_credentials: AwsCredentials) -> None:
    ...

async def test_s3_list_objects_prefix(object: Any, client_parameters: AwsClientParameters, object_in_folder: Any, aws_credentials: AwsCredentials) -> None:
    ...

async def test_s3_list_objects_prefix_slashes(object: Any, client_parameters: AwsClientParameters, objects_in_folder: List[Any], aws_credentials: AwsCredentials) -> None:
    ...

async def test_s3_list_objects_filter(object: Any, client_parameters: AwsClientParameters, aws_credentials: AwsCredentials) -> None:
    ...

def aws_creds_block() -> AwsCredentials:
    ...

def minio_creds_block() -> MinIOCredentials:
    ...

def s3() -> boto3.client:
    ...

def nested_s3_bucket_structure(s3: boto3.client, s3_bucket: S3Bucket, tmp_path: Path) -> None:
    ...

@pytest.fixture(params=['aws_credentials', 'minio_credentials'])
def s3_bucket(s3: boto3.client, request: Any, aws_creds_block: AwsCredentials, minio_creds_block: MinIOCredentials) -> S3Bucket:
    ...

def s3_bucket_with_file(s3_bucket: S3Bucket) -> Tuple[S3Bucket, str]:
    ...

async def test_read_write_roundtrip(s3_bucket: S3Bucket) -> None:
    ...

async def test_write_with_missing_directory_succeeds(s3_bucket: S3Bucket) -> None:
    ...

async def test_read_fails_does_not_exist(s3_bucket: S3Bucket) -> None:
    ...

@pytest.mark.parametrize('type_', [PureWindowsPath, PurePosixPath, str])
@pytest.mark.parametrize('delimiter', ['\\', '/'])
async def test_aws_bucket_folder(s3_bucket: S3Bucket, aws_creds_block: AwsCredentials, delimiter: str, type_: Any) -> None:
    ...

async def test_get_directory(nested_s3_bucket_structure: None, s3_bucket: S3Bucket, tmp_path: Path) -> None:
    ...

async def test_get_directory_respects_bucket_folder(nested_s3_bucket_structure: None, s3_bucket: S3Bucket, tmp_path: Path, aws_creds_block: AwsCredentials) -> None:
    ...

async def test_get_directory_respects_from_path(nested_s3_bucket_structure: None, s3_bucket: S3Bucket, tmp_path: Path, aws_creds_block: AwsCredentials) -> None:
    ...

async def test_put_directory(s3_bucket: S3Bucket, tmp_path: Path) -> None:
    ...

async def test_put_directory_respects_basepath(s3_bucket: S3Bucket, tmp_path: Path, aws_creds_block: AwsCredentials) -> None:
    ...

async def test_put_directory_with_ignore_file(s3_bucket: S3Bucket, tmp_path: Path, aws_creds_block: AwsCredentials) -> None:
    ...

async def test_put_directory_respects_local_path(s3_bucket: S3Bucket, tmp_path: Path, aws_creds_block: AwsCredentials) -> None:
    ...

def test_read_path_in_sync_context(s3_bucket_with_file: Tuple[S3Bucket, str]) -> None:
    ...

def test_write_path_in_sync_context(s3_bucket: S3Bucket) -> None:
    ...

def test_resolve_path(s3_bucket: S3Bucket) -> str:
    ...

class TestS3Bucket:

    @pytest.fixture(params=[AwsCredentials(), MinIOCredentials(minio_root_user='root', minio_root_password='password')])
    def credentials(self, request: Any) -> Union[AwsCredentials, MinIOCredentials]:
        ...

    @pytest.fixture
    def s3_bucket_empty(self, credentials: Union[AwsCredentials, MinIOCredentials], bucket: boto3.Bucket) -> S3Bucket:
        ...

    @pytest.fixture
    def s3_bucket_2_empty(self, credentials: Union[AwsCredentials, MinIOCredentials], bucket_2: boto3.Bucket) -> S3Bucket:
        ...

    @pytest.fixture
    def s3_bucket_with_object(self, s3_bucket_empty: S3Bucket, object: Any) -> S3Bucket:
        ...

    @pytest.fixture
    def s3_bucket_2_with_object(self, s3_bucket_2_empty: S3Bucket) -> S3Bucket:
        ...

    @pytest.fixture
    def s3_bucket_with_objects(self, s3_bucket_with_object: S3Bucket, object_in_folder: Any) -> S3Bucket:
        ...

    @pytest.fixture
    def s3_bucket_with_similar_objects(self, s3_bucket_with_objects: S3Bucket, objects_in_folder: List[Any]) -> S3Bucket:
        ...

    def test_credentials_are_correct_type(self, credentials: Union[AwsCredentials, MinIOCredentials]) -> None:
        ...

    @pytest.mark.parametrize('client_parameters', aws_clients[-1:])
    def test_list_objects_empty(self, s3_bucket_empty: S3Bucket, client_parameters: AwsClientParameters) -> None:
        ...

    @pytest.mark.parametrize('client_parameters', aws_clients[-1:])
    def test_list_objects_one(self, s3_bucket_with_object: S3Bucket, client_parameters: AwsClientParameters) -> None:
        ...

    @pytest.mark.parametrize('client_parameters', aws_clients[-1:])
    def test_list_objects(self, s3_bucket_with_objects: S3Bucket, client_parameters: AwsClientParameters) -> None:
        ...

    @pytest.mark.parametrize('client_parameters', aws_clients[-1:])
    def test_list_objects_with_params(self, s3_bucket_with_similar_objects: S3Bucket, client_parameters: AwsClientParameters) -> None:
        ...

    @pytest.mark.parametrize('to_path', [Path('to_path'), 'to_path', None])
    @pytest.mark.parametrize('client_parameters', aws_clients[-1:])
    def test_download_object_to_path(self, s3_bucket_with_object: S3Bucket, to_path: Optional[Union[Path, str]], client_parameters: AwsClientParameters, tmp_path: Path) -> None:
        ...

    @pytest.mark.parametrize('client_parameters', aws_clients[-1:])
    def test_download_object_to_file_object(self, s3_bucket_with_object: S3Bucket, client_parameters: AwsClientParameters, tmp_path: Path) -> None:
        ...

    @pytest.mark.parametrize('to_path', ['to_path', None])
    @pytest.mark.parametrize('client_parameters', aws_clients[-1:])
    def test_download_folder_to_path(self, s3_bucket_with_objects: S3Bucket, client_parameters: AwsClientParameters, tmp_path: Path, to_path: Optional[str]) -> None:
        ...

    @pytest.mark.parametrize('client_parameters', aws_clients[-1:])
    def test_download_object_with_bucket_folder(self, s3_bucket_empty: S3Bucket, client_parameters: AwsClientParameters, tmp_path: Path) -> None:
        ...

    @pytest.mark.parametrize('to_path', ['to_path', None])
    @pytest.mark.parametrize('client_parameters', aws_clients[-1:])
    def test_stream_from(self, s3_bucket_2_with_object: S3Bucket, s3_bucket_empty: S3Bucket, client_parameters: AwsClientParameters, to_path: Optional[str]) -> None:
        ...

    @pytest.mark.parametrize('to_path', ['new_object', None])
    @pytest.mark.parametrize('client_parameters', aws_clients[-1:])
    def test_upload_from_path(self, s3_bucket_empty: S3Bucket, client_parameters: AwsClientParameters, tmp_path: Path, to_path: Optional[str]) -> None:
        ...

    @pytest.mark.parametrize('client_parameters', aws_clients[-1:])
    def test_upload_from_file_object(self, s3_bucket_empty: S3Bucket, client_parameters: AwsClientParameters, tmp_path: Path) -> None:
        ...

    @pytest.mark.parametrize('client_parameters', aws_clients[-1:])
    def test_upload_from_folder(self, s3_bucket_empty: S3Bucket, client_parameters: AwsClientParameters, tmp_path: Path, caplog: Any) -> None:
        ...

    @pytest.mark.parametrize('client_parameters', aws_clients[-1:])
    def test_copy_object(self, s3_bucket_with_object: S3Bucket, s3_bucket_2_empty: S3Bucket) -> None:
        ...

    @pytest.mark.parametrize('client_parameters', aws_clients[-1:])
    @pytest.mark.parametrize('to_bucket, bucket_folder, expected_path', [(None, None, 'object'), (None, 'subfolder', 'subfolder/object'), ('bucket_2', None, 'object'), (None, None, 'object'), (None, 'subfolder', 'subfolder/object'), ('bucket_2', None, 'object')])
    def test_copy_subpaths(self, s3_bucket_with_object: S3Bucket, s3_bucket_2_empty: S3Bucket, to_bucket: Optional[Union[S3Bucket, str]], bucket_folder: Optional[str], expected_path: str) -> None:
        ...

    @pytest.mark.parametrize('client_parameters', aws_clients[-1:])
    def test_move_object_within_bucket(self, s3_bucket_with_object: S3Bucket) -> None:
        ...

    @pytest.mark.parametrize('client_parameters', aws_clients[-1:])
    def test_move_object_to_nonexistent_bucket_fails(self, s3_bucket_with_object: S3Bucket) -> None:
        ...

    @pytest.mark.parametrize('client_parameters', aws_clients[-1:])
    def test_move_object_onto_itself_fails(self, s3_bucket_with_object: S3Bucket) -> None:
        ...

    @pytest.mark.parametrize('client_parameters', aws_clients[-1:])
    def test_move_object_between_buckets(self, s3_bucket_with_object: S3Bucket, s3_bucket_2_empty: S3Bucket) -> None:
        ...

    @pytest.mark.parametrize('client_parameters', aws_clients[-1:])
    @pytest.mark.parametrize('to_bucket, bucket_folder, expected_path', [(None, None, 'object'), (None, 'subfolder', 'subfolder/object'), ('bucket_2', None, 'object'), (None, None, 'object'), (None, 'subfolder', 'subfolder/object'), ('bucket_2', None, 'object')])
    def test_move_subpaths(self, s3_bucket_with_object: S3Bucket, s3_bucket_2_empty: S3Bucket, to_bucket: Optional[Union[S3Bucket, str]], bucket_folder: Optional[str], expected_path: str) -> None:
        ...

    def test_round_trip_default_credentials(self) -> None:
        ...

    @pytest.mark.parametrize('client_parameters', [pytest.param('aws_client_parameters_custom_endpoint', marks=pytest.mark.is_public(False)), pytest.param('aws_client_parameters_custom_endpoint', marks=pytest.mark.is_public(True)), pytest.param('aws_client_parameters_empty', marks=pytest.mark.is_public(False)), pytest.param('aws_client_parameters_empty', marks=pytest.mark.is_public(True)), pytest.param('aws_client_parameters_public_bucket', marks=[pytest.mark.is_public(False), pytest.mark.xfail(reason='Bucket is not a public one')]), pytest.param('aws_client_parameters_public_bucket', marks=pytest.mark.is_public(True)])
    async def test_async_download_from_bucket(self, object: Any, client_parameters: AwsClientParameters, aws_credentials: AwsCredentials) -> None:
        ...

    @pytest.mark.parametrize('client_parameters', aws_clients)
    async def test_async_list_objects(self, object: Any, client_parameters: AwsClientParameters, aws_credentials: AwsCredentials) -> None:
        ...

    @pytest.mark.parametrize('client_parameters', aws_clients)
    async def test_async_copy_objects(self, object: Any, bucket: boto3.Bucket, bucket_2: boto3.Bucket, aws_credentials: AwsCredentials) -> None:
        ...

    @pytest.mark.parametrize('client_parameters', aws_clients)
    async def test_async_move_objects(self, object: Any, bucket: boto3.Bucket, bucket_2: boto3.Bucket, aws_credentials: AwsCredentials) -> None:
        ...
