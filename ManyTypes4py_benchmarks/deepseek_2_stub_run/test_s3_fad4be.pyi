```python
import io
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Optional, Union, List, Dict, BinaryIO, overload
from typing_extensions import Literal

import boto3
import pytest
from botocore.exceptions import ClientError, EndpointConnectionError
from moto import mock_s3
from prefect_aws import AwsCredentials, MinIOCredentials
from prefect_aws.client_parameters import AwsClientParameters
from prefect_aws.s3 import S3Bucket, acopy_objects, adownload_from_bucket, alist_objects, amove_objects, s3_copy, s3_download, s3_list_objects, s3_move, s3_upload
from prefect import flow

aws_clients: List[str] = ...

@pytest.fixture
def s3_mock(monkeypatch: Any, client_parameters: Any) -> Any: ...

@pytest.fixture
def client_parameters(request: Any) -> Any: ...

@pytest.fixture
def bucket(s3_mock: Any, request: Any) -> Any: ...

@pytest.fixture
def bucket_2(s3_mock: Any, request: Any) -> Any: ...

@pytest.fixture
def object(bucket: Any, tmp_path: Any) -> Any: ...

@pytest.fixture
def object_in_folder(bucket: Any, tmp_path: Any) -> Any: ...

@pytest.fixture
def objects_in_folder(bucket: Any, tmp_path: Any) -> Any: ...

@pytest.fixture
def a_lot_of_objects(bucket: Any, tmp_path: Any) -> Any: ...

@pytest.fixture
def aws_creds_block() -> AwsCredentials: ...

@pytest.fixture
def minio_creds_block() -> MinIOCredentials: ...

BUCKET_NAME: str = ...

@pytest.fixture
def s3() -> Any: ...

@pytest.fixture
def nested_s3_bucket_structure(s3: Any, s3_bucket: Any, tmp_path: Any) -> Any: ...

@pytest.fixture
def s3_bucket(s3: Any, request: Any, aws_creds_block: Any, minio_creds_block: Any) -> S3Bucket: ...

@pytest.fixture
def s3_bucket_with_file(s3_bucket: Any) -> Any: ...

class TestS3Bucket:
    @pytest.fixture
    def credentials(self, request: Any) -> Any: ...
    
    @pytest.fixture
    def s3_bucket_empty(self, credentials: Any, bucket: Any) -> S3Bucket: ...
    
    @pytest.fixture
    def s3_bucket_2_empty(self, credentials: Any, bucket_2: Any) -> S3Bucket: ...
    
    @pytest.fixture
    def s3_bucket_with_object(self, s3_bucket_empty: Any, object: Any) -> S3Bucket: ...
    
    @pytest.fixture
    def s3_bucket_2_with_object(self, s3_bucket_2_empty: Any) -> S3Bucket: ...
    
    @pytest.fixture
    def s3_bucket_with_objects(self, s3_bucket_with_object: Any, object_in_folder: Any) -> S3Bucket: ...
    
    @pytest.fixture
    def s3_bucket_with_similar_objects(self, s3_bucket_with_objects: Any, objects_in_folder: Any) -> S3Bucket: ...
```