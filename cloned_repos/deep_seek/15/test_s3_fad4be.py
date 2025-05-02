import io
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Dict, Generator, List, Optional, Tuple, Union, cast
from unittest.mock import MagicMock

import boto3
import pytest
from botocore.exceptions import ClientError, EndpointConnectionError
from moto import mock_s3
from prefect_aws import AwsCredentials, MinIOCredentials
from prefect_aws.client_parameters import AwsClientParameters
from prefect_aws.s3 import (
    S3Bucket,
    acopy_objects,
    adownload_from_bucket,
    alist_objects,
    amove_objects,
    s3_copy,
    s3_download,
    s3_list_objects,
    s3_move,
    s3_upload,
)
from prefect import flow

aws_clients: List[str] = [
    "aws_client_parameters_custom_endpoint",
    "aws_client_parameters_empty",
    "aws_client_parameters_public_bucket",
]


@pytest.fixture
def s3_mock(monkeypatch: pytest.MonkeyPatch, client_parameters: AwsClientParameters) -> Generator[None, None, None]:
    if client_parameters.endpoint_url:
        monkeypatch.setenv("MOTO_S3_CUSTOM_ENDPOINTS", client_parameters.endpoint_url)
    with mock_s3():
        yield


@pytest.fixture
def client_parameters(request: pytest.FixtureRequest) -> AwsClientParameters:
    client_parameters: AwsClientParameters = request.getfixturevalue(request.param)
    return client_parameters


@pytest.fixture
def bucket(s3_mock: None, request: pytest.FixtureRequest) -> boto3.resources.base.ServiceResource:
    s3: boto3.resources.base.ServiceResource = boto3.resource("s3")
    bucket: boto3.resources.base.ServiceResource = s3.Bucket("bucket")
    marker = request.node.get_closest_marker("is_public", None)
    if marker and marker.args[0]:
        bucket.create(ACL="public-read")
    else:
        bucket.create()
    return bucket


@pytest.fixture
def bucket_2(s3_mock: None, request: pytest.FixtureRequest) -> boto3.resources.base.ServiceResource:
    s3: boto3.resources.base.ServiceResource = boto3.resource("s3")
    bucket: boto3.resources.base.ServiceResource = s3.Bucket("bucket_2")
    marker = request.node.get_closest_marker("is_public", None)
    if marker and marker.args[0]:
        bucket.create(ACL="public-read")
    else:
        bucket.create()
    return bucket


@pytest.fixture
def object(bucket: boto3.resources.base.ServiceResource, tmp_path: Path) -> None:
    file: Path = tmp_path / "object.txt"
    file.write_text("TEST")
    with open(file, "rb") as f:
        return bucket.upload_fileobj(f, "object")


@pytest.fixture
def object_in_folder(bucket: boto3.resources.base.ServiceResource, tmp_path: Path) -> None:
    file: Path = tmp_path / "object_in_folder.txt"
    file.write_text("TEST OBJECT IN FOLDER")
    with open(file, "rb") as f:
        return bucket.upload_fileobj(f, "folder/object")


@pytest.fixture
def objects_in_folder(bucket: boto3.resources.base.ServiceResource, tmp_path: Path) -> List[None]:
    objects: List[None] = []
    for filename in [
        "folderobject/foo.txt",
        "folderobject/bar.txt",
        "folder/object/foo.txt",
        "folder/object/bar.txt",
    ]:
        file: Path = tmp_path / filename
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text("TEST OBJECTS IN FOLDER")
        with open(file, "rb") as f:
            filename_path: Path = Path(filename)
            obj: None = bucket.upload_fileobj(
                f, (filename_path.parent / filename_path.stem).as_posix()
            )
            objects.append(obj)
    return objects


@pytest.fixture
def a_lot_of_objects(bucket: boto3.resources.base.ServiceResource, tmp_path: Path) -> List[None]:
    objects: List[None] = []
    for i in range(0, 20):
        file: Path = tmp_path / f"object{i}.txt"
        file.write_text("TEST")
        with open(file, "rb") as f:
            objects.append(bucket.upload_fileobj(f, f"object{i}"))
    return objects


@pytest.mark.parametrize("client_parameters", ["aws_client_parameters_custom_endpoint"], indirect=True)
async def test_s3_download_failed_with_wrong_endpoint_setup(
    object: None, client_parameters: AwsClientParameters, aws_credentials: AwsCredentials
) -> None:
    client_parameters_wrong_endpoint: AwsClientParameters = AwsClientParameters(
        endpoint_url="http://something"
    )

    @flow
    async def test_flow() -> Any:
        return await s3_download(
            bucket="bucket",
            key="object",
            aws_credentials=aws_credentials,
            aws_client_parameters=client_parameters_wrong_endpoint,
        )

    with pytest.raises(EndpointConnectionError):
        await test_flow()


@pytest.mark.parametrize(
    "client_parameters",
    [
        pytest.param("aws_client_parameters_custom_endpoint", marks=pytest.mark.is_public(False)),
        pytest.param("aws_client_parameters_custom_endpoint", marks=pytest.mark.is_public(True)),
        pytest.param("aws_client_parameters_empty", marks=pytest.mark.is_public(False)),
        pytest.param("aws_client_parameters_empty", marks=pytest.mark.is_public(True)),
        pytest.param(
            "aws_client_parameters_public_bucket",
            marks=[pytest.mark.is_public(False), pytest.mark.xfail(reason="Bucket is not a public one")],
        ),
        pytest.param("aws_client_parameters_public_bucket", marks=pytest.mark.is_public(True)),
    ],
    indirect=True,
)
async def test_s3_download(
    object: None, client_parameters: AwsClientParameters, aws_credentials: AwsCredentials
) -> None:
    @flow
    async def test_flow() -> bytes:
        return await s3_download(
            bucket="bucket",
            key="object",
            aws_credentials=aws_credentials,
            aws_client_parameters=client_parameters,
        )

    result: bytes = await test_flow()
    assert result == b"TEST"


@pytest.mark.parametrize("client_parameters", aws_clients, indirect=True)
async def test_s3_download_object_not_found(
    object: None, client_parameters: AwsClientParameters, aws_credentials: AwsCredentials
) -> None:
    @flow
    async def test_flow() -> Any:
        return await s3_download(
            key="unknown_object",
            bucket="bucket",
            aws_credentials=aws_credentials,
            aws_client_parameters=client_parameters,
        )

    with pytest.raises(ClientError):
        await test_flow()


@pytest.mark.parametrize("client_parameters", aws_clients, indirect=True)
async def test_s3_upload(
    bucket: boto3.resources.base.ServiceResource,
    client_parameters: AwsClientParameters,
    tmp_path: Path,
    aws_credentials: AwsCredentials,
) -> None:
    @flow
    async def test_flow() -> Any:
        test_file: Path = tmp_path / "test.txt"
        test_file.write_text("NEW OBJECT")
        with open(test_file, "rb") as f:
            return await s3_upload(
                data=f.read(),
                bucket="bucket",
                key="new_object",
                aws_credentials=aws_credentials,
                aws_client_parameters=client_parameters,
            )

    await test_flow()
    stream: io.BytesIO = io.BytesIO()
    bucket.download_fileobj("new_object", stream)
    stream.seek(0)
    output: bytes = stream.read()
    assert output == b"NEW OBJECT"


@pytest.mark.parametrize("client_parameters", aws_clients, indirect=True)
async def test_s3_copy(
    object: None,
    bucket: boto3.resources.base.ServiceResource,
    bucket_2: boto3.resources.base.ServiceResource,
    aws_credentials: AwsCredentials,
) -> None:
    def read(bucket: boto3.resources.base.ServiceResource, key: str) -> bytes:
        stream: io.BytesIO = io.BytesIO()
        bucket.download_fileobj(key, stream)
        stream.seek(0)
        return stream.read()

    @flow
    async def test_flow() -> None:
        await s3_copy(
            source_path="object",
            target_path="subfolder/new_object",
            source_bucket_name="bucket",
            aws_credentials=aws_credentials,
            target_bucket_name="bucket_2",
        )
        await s3_copy(
            source_path="object",
            target_path="subfolder/new_object",
            source_bucket_name="bucket",
            aws_credentials=aws_credentials,
        )

    await test_flow()
    assert read(bucket_2, "subfolder/new_object") == b"TEST"
    assert read(bucket, "subfolder/new_object") == b"TEST"


@pytest.mark.parametrize("client_parameters", aws_clients, indirect=True)
async def test_s3_move(
    object: None,
    bucket: boto3.resources.base.ServiceResource,
    bucket_2: boto3.resources.base.ServiceResource,
    aws_credentials: AwsCredentials,
) -> None:
    def read(bucket: boto3.resources.base.ServiceResource, key: str) -> bytes:
        stream: io.BytesIO = io.BytesIO()
        bucket.download_fileobj(key, stream)
        stream.seek(0)
        return stream.read()

    @flow
    async def test_flow() -> None:
        await s3_move(
            source_path="object",
            target_path="subfolder/object_copy",
            source_bucket_name="bucket",
            aws_credentials=aws_credentials,
        )
        await s3_move(
            source_path="subfolder/object_copy",
            target_path="object_copy_2",
            source_bucket_name="bucket",
            target_bucket_name="bucket_2",
            aws_credentials=aws_credentials,
        )

    await test_flow()
    assert read(bucket_2, "object_copy_2") == b"TEST"
    with pytest.raises(ClientError):
        read(bucket, "object")
    with pytest.raises(ClientError):
        read(bucket, "subfolder/object_copy")


@pytest.mark.parametrize("client_parameters", aws_clients, indirect=True)
async def test_move_object_to_nonexistent_bucket_fails(
    object: None, bucket: boto3.resources.base.ServiceResource, aws_credentials: AwsCredentials
) -> None:
    def read(bucket: boto3.resources.base.ServiceResource, key: str) -> bytes:
        stream: io.BytesIO = io.BytesIO()
        bucket.download_fileobj(key, stream)
        stream.seek(0)
        return stream.read()

    @flow
    async def test_flow() -> None:
        await s3_move(
            source_path="object",
            target_path="subfolder/new_object",
            source_bucket_name="bucket",
            aws_credentials=aws_credentials,
            target_bucket_name="nonexistent-bucket",
        )

    with pytest.raises(ClientError):
        await test_flow()
    assert read(bucket, "object") == b"TEST"


@pytest.mark.parametrize("client_parameters", aws_clients, indirect=True)
async def test_move_object_fail_cases(
    object: None, bucket: boto3.resources.base.ServiceResource, aws_credentials: AwsCredentials
) -> None:
    def read(bucket: boto3.resources.base.ServiceResource, key: str) -> bytes:
        stream: io.BytesIO = io.BytesIO()
        bucket.download_fileobj(key, stream)
        stream.seek(0)
        return stream.read()

    @flow
    async def test_flow(
        source_path: str, target_path: str, source_bucket_name: str, target_bucket_name: str
    ) -> None:
        await s3_move(
            source_path=source_path,
            target_path=target_path,
            source_bucket_name=source_bucket_name,
            aws_credentials=aws_credentials,
            target_bucket_name=target_bucket_name,
        )

    with pytest.raises(ClientError):
        await test_flow(
            source_path="object",
            target_path="subfolder/new_object",
            source_bucket_name="bucket",
            target_bucket_name="nonexistent-bucket",
        )
    assert read(bucket, "object") == b"TEST"
    with pytest.raises(ClientError):
        await test_flow(
            source_path="object", target_path="object", source_bucket_name="bucket", target_bucket_name="bucket"
        )
    assert read(bucket, "object") == b"TEST"


@pytest.mark.parametrize("client_parameters", aws_clients, indirect=True)
async def test_s3_list_objects(
    object: None,
    client_parameters: AwsClientParameters,
    object_in_folder: None,
    aws_credentials: AwsCredentials,
) -> None:
    @flow
    async def test_flow() -> List[Dict[str, Any]]:
        return await s3_list_objects(
            bucket="bucket",
            aws_credentials=aws_credentials,
            aws_client_parameters=client_parameters,
        )

    objects: List[Dict[str, Any]] = await test_flow()
    assert len(objects) == 2
    assert [object["Key"] for object in objects] == ["folder/object", "object"]


@pytest.mark.parametrize("client_parameters", aws_clients, indirect=True)
async def test_s3_list_objects_multiple_pages(
    a_lot_of_objects: List[None], client_parameters: AwsClientParameters, aws_credentials: AwsCredentials
) -> None:
    @flow
    async def test_flow() -> List[Dict[str, Any]]:
        return await s3_list_objects(
            bucket="bucket",
            aws_credentials=aws_credentials,
            aws_client_parameters=client_parameters,
            page_size=2,
        )

    objects: List[Dict[str, Any]] = await test_flow()
    assert len(objects) == 20
    assert sorted([object["Key"] for object in objects]) == sorted(
        [f"object{i}" for i in range(0, 20)]
    )


@pytest.mark.parametrize("client_parameters", aws_clients, indirect=True)
async def test_s3_list_objects_prefix(
    object: None,
    client_parameters: AwsClientParameters,
    object_in_folder: None,
    aws_credentials: AwsCredentials,
) -> None:
    @flow
    async def test_flow() -> List[Dict[str, Any]]:
        return await s3_list_objects(
            bucket="bucket",
            prefix="folder",
            aws_credentials=aws_credentials,
            aws_client_parameters=client_parameters,
        )

    objects: List[Dict[str, Any]] = await test_flow()
    assert len(objects) == 1
    assert [object["Key"] for object in objects] == ["folder/object"]


@pytest.mark.parametrize("client_parameters", aws_clients, indirect=True)
async def test_s3_list_objects_prefix_slashes(
    object: None,
    client_parameters: AwsClientParameters,
    objects_in_folder: List[None],
    aws_credentials: AwsCredentials,
) -> None:
    @flow
    async def test_flow(slash: bool = False) -> List[Dict[str, Any]]:
        return await s3_list_objects(
            bucket="bucket",
            prefix="folder" + ("/" if slash else ""),
            aws_credentials=aws_credentials,
            aws_client_parameters=client_parameters,
        )

    objects: List[Dict[str, Any]] = await test_flow(slash=True)
    assert len(objects) == 2
    assert [object["Key"] for object in objects] == ["folder/object/bar", "folder/object/foo"]
    objects = await test_flow(slash=False)
    assert len(objects) == 4
    assert [object["Key"] for object in objects] == [
        "folder/object/bar",
        "folder/object/foo",
        "folderobject/bar",
        "folderobject/foo",
    ]


@pytest.mark.parametrize("client_parameters", aws_clients, indirect=True)
async def test_s3_list_objects_filter(
    object: None,
    client_parameters: AwsClientParameters,
    object_in_folder: None,
    aws_credentials: AwsCredentials,
) -> None:
    @flow
    async def test_flow() -> List[Dict[str, Any]]:
        return await s3_list_objects(
            bucket="bucket",
            jmespath_query="Contents[?Size > `10`][]",
            aws_credentials=aws_credentials,
            aws_client_parameters=client_parameters,
        )

    objects: List[Dict[str, Any]] = await test_flow()
    assert len(objects) == 1
    assert [object["Key"] for object in objects] == ["folder/object"]


@pytest.fixture
def aws_creds_block() -> AwsCredentials:
    return AwsCredentials(aws_access_key_id="testing", aws_secret_access_key="testing")


@pytest.fixture
def minio_creds_block() -> MinIOCredentials:
    return MinIOCredentials(minio_root_user="minioadmin", minio_root_password="minioadmin")


BUCKET_NAME: str = "test_bucket"


@pytest.fixture
def s3() -> Generator[boto3.client, None, None]:
    """Mock connection to AWS