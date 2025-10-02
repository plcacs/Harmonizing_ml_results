#!/usr/bin/env python3
from __future__ import annotations
import io
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Generator, Optional, Union
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

aws_clients: list[str] = [
    "aws_client_parameters_custom_endpoint",
    "aws_client_parameters_empty",
    "aws_client_parameters_public_bucket",
]


@pytest.fixture
def s3_mock(monkeypatch: pytest.MonkeyPatch, client_parameters: Any) -> Generator[None, None, None]:
    if client_parameters.endpoint_url:
        monkeypatch.setenv("MOTO_S3_CUSTOM_ENDPOINTS", client_parameters.endpoint_url)
    with mock_s3():
        yield


@pytest.fixture
def client_parameters(request: pytest.FixtureRequest) -> Any:
    cp = request.getfixturevalue(request.param)
    return cp


@pytest.fixture
def bucket(s3_mock: Any, request: pytest.FixtureRequest) -> Any:
    s3 = boto3.resource("s3")
    bucket_obj = s3.Bucket("bucket")
    marker = request.node.get_closest_marker("is_public", None)
    if marker and marker.args[0]:
        bucket_obj.create(ACL="public-read")
    else:
        bucket_obj.create()
    return bucket_obj


@pytest.fixture
def bucket_2(s3_mock: Any, request: pytest.FixtureRequest) -> Any:
    s3 = boto3.resource("s3")
    bucket_obj = s3.Bucket("bucket_2")
    marker = request.node.get_closest_marker("is_public", None)
    if marker and marker.args[0]:
        bucket_obj.create(ACL="public-read")
    else:
        bucket_obj.create()
    return bucket_obj


@pytest.fixture
def object(bucket: Any, tmp_path: Path) -> None:
    file = tmp_path / "object.txt"
    file.write_text("TEST")
    with open(file, "rb") as f:
        bucket.upload_fileobj(f, "object")
    # This fixture returns None


@pytest.fixture
def object_in_folder(bucket: Any, tmp_path: Path) -> None:
    file = tmp_path / "object_in_folder.txt"
    file.write_text("TEST OBJECT IN FOLDER")
    with open(file, "rb") as f:
        bucket.upload_fileobj(f, "folder/object")
    # This fixture returns None


@pytest.fixture
def objects_in_folder(bucket: Any, tmp_path: Path) -> list[Any]:
    objects: list[Any] = []
    for filename in ["folderobject/foo.txt", "folderobject/bar.txt", "folder/object/foo.txt", "folder/object/bar.txt"]:
        file = tmp_path / filename
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text("TEST OBJECTS IN FOLDER")
        with open(file, "rb") as f:
            filename_path = Path(filename)
            obj = bucket.upload_fileobj(f, (filename_path.parent / filename_path.stem).as_posix())
            objects.append(obj)
    return objects


@pytest.fixture
def a_lot_of_objects(bucket: Any, tmp_path: Path) -> list[Any]:
    objects: list[Any] = []
    for i in range(0, 20):
        file = tmp_path / f"object{i}.txt"
        file.write_text("TEST")
        with open(file, "rb") as f:
            objects.append(bucket.upload_fileobj(f, f"object{i}"))
    return objects


@pytest.mark.parametrize("client_parameters", ["aws_client_parameters_custom_endpoint"], indirect=True)
async def test_s3_download_failed_with_wrong_endpoint_setup(
    object: None, client_parameters: Any, aws_credentials: Any
) -> None:
    client_parameters_wrong_endpoint = AwsClientParameters(endpoint_url="http://something")

    @flow
    async def test_flow() -> bytes:
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
async def test_s3_download(object: None, client_parameters: Any, aws_credentials: Any) -> None:
    @flow
    async def test_flow() -> bytes:
        return await s3_download(
            bucket="bucket", key="object", aws_credentials=aws_credentials, aws_client_parameters=client_parameters
        )

    result: bytes = await test_flow()
    assert result == b"TEST"


@pytest.mark.parametrize("client_parameters", aws_clients, indirect=True)
async def test_s3_download_object_not_found(object: None, client_parameters: Any, aws_credentials: Any) -> None:
    @flow
    async def test_flow() -> bytes:
        return await s3_download(
            key="unknown_object",
            bucket="bucket",
            aws_credentials=aws_credentials,
            aws_client_parameters=client_parameters,
        )

    with pytest.raises(ClientError):
        await test_flow()


@pytest.mark.parametrize("client_parameters", aws_clients, indirect=True)
async def test_s3_upload(bucket: Any, client_parameters: Any, tmp_path: Path, aws_credentials: Any) -> None:
    @flow
    async def test_flow() -> None:
        test_file = tmp_path / "test.txt"
        test_file.write_text("NEW OBJECT")
        with open(test_file, "rb") as f:
            await s3_upload(
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
async def test_s3_copy(object: None, bucket: Any, bucket_2: Any, aws_credentials: Any) -> None:
    def read(bucket_obj: Any, key: str) -> bytes:
        stream = io.BytesIO()
        bucket_obj.download_fileobj(key, stream)
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
async def test_s3_move(object: None, bucket: Any, bucket_2: Any, aws_credentials: Any) -> None:
    def read(bucket_obj: Any, key: str) -> bytes:
        stream = io.BytesIO()
        bucket_obj.download_fileobj(key, stream)
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
async def test_move_object_to_nonexistent_bucket_fails(object: None, bucket: Any, aws_credentials: Any) -> None:
    def read(bucket_obj: Any, key: str) -> bytes:
        stream = io.BytesIO()
        bucket_obj.download_fileobj(key, stream)
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
async def test_move_object_fail_cases(object: None, bucket: Any, aws_credentials: Any) -> None:
    def read(bucket_obj: Any, key: str) -> bytes:
        stream = io.BytesIO()
        bucket_obj.download_fileobj(key, stream)
        stream.seek(0)
        return stream.read()

    @flow
    async def test_flow(
        source_path: str, target_path: str, source_bucket_name: str, target_bucket_name: Optional[str]
    ) -> None:
        await s3_move(
            source_path=source_path,
            target_path=target_path,
            source_bucket_name=source_bucket_name,
            aws_credentials=aws_credentials,
            target_bucket_name=target_bucket_name,
        )

    with pytest.raises(ClientError):
        await test_flow(source_path="object", target_path="subfolder/new_object", source_bucket_name="bucket", target_bucket_name="nonexistent-bucket")
    assert read(bucket, "object") == b"TEST"
    with pytest.raises(ClientError):
        await test_flow(source_path="object", target_path="object", source_bucket_name="bucket", target_bucket_name="bucket")
    assert read(bucket, "object") == b"TEST"


@pytest.mark.parametrize("client_parameters", aws_clients, indirect=True)
async def test_s3_list_objects(object: None, client_parameters: Any, object_in_folder: None, aws_credentials: Any) -> None:
    @flow
    async def test_flow() -> list[dict[str, Any]]:
        return await s3_list_objects(
            bucket="bucket", aws_credentials=aws_credentials, aws_client_parameters=client_parameters
        )

    objects: list[dict[str, Any]] = await test_flow()
    assert len(objects) == 2
    assert [obj["Key"] for obj in objects] == ["folder/object", "object"]


@pytest.mark.parametrize("client_parameters", aws_clients, indirect=True)
async def test_s3_list_objects_multiple_pages(a_lot_of_objects: list[Any], client_parameters: Any, aws_credentials: Any) -> None:
    @flow
    async def test_flow() -> list[dict[str, Any]]:
        return await s3_list_objects(
            bucket="bucket", aws_credentials=aws_credentials, aws_client_parameters=client_parameters, page_size=2
        )

    objects: list[dict[str, Any]] = await test_flow()
    assert len(objects) == 20
    expected_keys = sorted([f"object{i}" for i in range(0, 20)])
    obtained_keys = sorted([obj["Key"] for obj in objects])
    assert obtained_keys == expected_keys


@pytest.mark.parametrize("client_parameters", aws_clients, indirect=True)
async def test_s3_list_objects_prefix(object: None, client_parameters: Any, object_in_folder: None, aws_credentials: Any) -> None:
    @flow
    async def test_flow() -> list[dict[str, Any]]:
        return await s3_list_objects(
            bucket="bucket",
            prefix="folder",
            aws_credentials=aws_credentials,
            aws_client_parameters=client_parameters,
        )

    objects: list[dict[str, Any]] = await test_flow()
    assert len(objects) == 1
    assert [obj["Key"] for obj in objects] == ["folder/object"]


@pytest.mark.parametrize("client_parameters", aws_clients, indirect=True)
async def test_s3_list_objects_prefix_slashes(object: None, client_parameters: Any, objects_in_folder: list[Any], aws_credentials: Any) -> None:
    @flow
    async def test_flow(slash: bool = False) -> list[dict[str, Any]]:
        prefix = "folder" + ("/" if slash else "")
        return await s3_list_objects(
            bucket="bucket", prefix=prefix, aws_credentials=aws_credentials, aws_client_parameters=client_parameters
        )

    objects_with_slash = await test_flow(slash=True)
    assert len(objects_with_slash) == 2
    assert [obj["Key"] for obj in objects_with_slash] == ["folder/object/bar", "folder/object/foo"]
    objects_without_slash = await test_flow(slash=False)
    assert len(objects_without_slash) == 4
    assert [obj["Key"] for obj in objects_without_slash] == [
        "folder/object/bar",
        "folder/object/foo",
        "folderobject/bar",
        "folderobject/foo",
    ]


@pytest.mark.parametrize("client_parameters", aws_clients, indirect=True)
async def test_s3_list_objects_filter(object: None, client_parameters: Any, object_in_folder: None, aws_credentials: Any) -> None:
    @flow
    async def test_flow() -> list[dict[str, Any]]:
        return await s3_list_objects(
            bucket="bucket",
            jmespath_query="Contents[?Size > `10`][]",
            aws_credentials=aws_credentials,
            aws_client_parameters=client_parameters,
        )

    objects: list[dict[str, Any]] = await test_flow()
    assert len(objects) == 1
    assert [obj["Key"] for obj in objects] == ["folder/object"]


@pytest.fixture
def aws_creds_block() -> AwsCredentials:
    return AwsCredentials(aws_access_key_id="testing", aws_secret_access_key="testing")


@pytest.fixture
def minio_creds_block() -> MinIOCredentials:
    return MinIOCredentials(minio_root_user="minioadmin", minio_root_password="minioadmin")


BUCKET_NAME: str = "test_bucket"


@pytest.fixture
def s3() -> Generator[Any, None, None]:
    """Mock connection to AWS S3 with boto3 client."""
    with mock_s3():
        client = boto3.client(
            service_name="s3",
            region_name="us-east-1",
            aws_access_key_id="minioadmin",
            aws_secret_access_key="testing",
            aws_session_token="testing",
        )
        yield client


@pytest.fixture
def nested_s3_bucket_structure(s3: Any, s3_bucket: Any, tmp_path: Path) -> None:
    """Creates an S3 bucket with multiple files in a nested structure"""
    file = tmp_path / "object.txt"
    file.write_text("TEST")
    s3.upload_file(str(file), BUCKET_NAME, "object.txt")
    s3.upload_file(str(file), BUCKET_NAME, "level1/object_level1.txt")
    s3.upload_file(str(file), BUCKET_NAME, "level1/level2/object_level2.txt")
    s3.upload_file(str(file), BUCKET_NAME, "level1/level2/object2_level2.txt")
    file.unlink()
    assert not file.exists()


@pytest.fixture(params=["aws_credentials", "minio_credentials"])
def s3_bucket(
    s3: Any, request: pytest.FixtureRequest, aws_creds_block: AwsCredentials, minio_creds_block: MinIOCredentials
) -> S3Bucket:
    key: str = request.param
    if key == "aws_credentials":
        fs = S3Bucket(bucket_name=BUCKET_NAME, credentials=aws_creds_block)
    elif key == "minio_credentials":
        fs = S3Bucket(bucket_name=BUCKET_NAME, credentials=minio_creds_block)
    s3.create_bucket(Bucket=BUCKET_NAME)
    return fs


@pytest.fixture
def s3_bucket_with_file(s3_bucket: S3Bucket) -> tuple[S3Bucket, str]:
    key = s3_bucket.write_path("test.txt", content=b"hello")
    return (s3_bucket, key)


async def test_read_write_roundtrip(s3_bucket: S3Bucket) -> None:
    """
    Create an S3 bucket, instantiate S3Bucket block, write to and read from
    bucket.
    """
    key = await s3_bucket.write_path("test.txt", content=b"hello")
    assert await s3_bucket.read_path(key) == b"hello"


async def test_write_with_missing_directory_succeeds(s3_bucket: S3Bucket) -> None:
    """
    Create an S3 bucket, instantiate S3Bucket block, write to path with
    missing directory.
    """
    key = await s3_bucket.write_path("folder/test.txt", content=b"hello")
    assert await s3_bucket.read_path(key) == b"hello"


async def test_read_fails_does_not_exist(s3_bucket: S3Bucket) -> None:
    """
    Create an S3 bucket, instantiate S3Bucket block, assert read from
    nonexistent path fails.
    """
    with pytest.raises(ClientError):
        await s3_bucket.read_path("test_bucket/foo/bar")


@pytest.mark.parametrize("type_", [PureWindowsPath, PurePosixPath, str])
@pytest.mark.parametrize("delimiter", [r"\\", "/"])
async def test_aws_bucket_folder(
    s3_bucket: S3Bucket, aws_creds_block: AwsCredentials, delimiter: str, type_: type[Union[PureWindowsPath, PurePosixPath, str]]
) -> None:
    """Test the bucket folder functionality."""
    s3_bucket_block = S3Bucket(
        bucket_name=BUCKET_NAME, credentials=aws_creds_block, bucket_folder="subfolder/subsubfolder"
    )
    key = await s3_bucket_block.write_path("test.txt", content=b"hello")
    assert await s3_bucket_block.read_path("test.txt") == b"hello"
    expected: str = "subfolder/subsubfolder/test.txt"
    assert key == expected


async def test_get_directory(nested_s3_bucket_structure: None, s3_bucket: S3Bucket, tmp_path: Path) -> None:
    await s3_bucket.get_directory(local_path=str(tmp_path))
    assert (tmp_path / "object.txt").exists()
    assert (tmp_path / "level1" / "object_level1.txt").exists()
    assert (tmp_path / "level1" / "level2" / "object_level2.txt").exists()
    assert (tmp_path / "level1" / "level2" / "object2_level2.txt").exists()


async def test_get_directory_respects_bucket_folder(
    nested_s3_bucket_structure: None, s3_bucket: S3Bucket, tmp_path: Path, aws_creds_block: AwsCredentials
) -> None:
    s3_bucket_block = S3Bucket(bucket_name=BUCKET_NAME, credentials=aws_creds_block, bucket_folder="level1/level2")
    await s3_bucket_block.get_directory(local_path=str(tmp_path))
    assert len(list(tmp_path.glob("*"))) == 2
    assert (tmp_path / "object_level2.txt").exists()
    assert (tmp_path / "object2_level2.txt").exists()


async def test_get_directory_respects_from_path(
    nested_s3_bucket_structure: None, s3_bucket: S3Bucket, tmp_path: Path, aws_creds_block: AwsCredentials
) -> None:
    await s3_bucket.get_directory(local_path=str(tmp_path), from_path="level1")
    assert (tmp_path / "object_level1.txt").exists()
    assert (tmp_path / "level2" / "object_level2.txt").exists()
    assert (tmp_path / "level2" / "object2_level2.txt").exists()


async def test_put_directory(s3_bucket: S3Bucket, tmp_path: Path) -> None:
    (tmp_path / "file1.txt").write_text("FILE 1")
    (tmp_path / "file2.txt").write_text("FILE 2")
    (tmp_path / "folder1").mkdir()
    (tmp_path / "folder1" / "file3.txt").write_text("FILE 3")
    (tmp_path / "folder1" / "file4.txt").write_text("FILE 4")
    (tmp_path / "folder1" / "folder2").mkdir()
    (tmp_path / "folder1" / "folder2" / "file5.txt").write_text("FILE 5")
    uploaded_file_count: int = await s3_bucket.put_directory(local_path=str(tmp_path))
    assert uploaded_file_count == 5
    (tmp_path / "downloaded_files").mkdir()
    await s3_bucket.get_directory(local_path=str(tmp_path / "downloaded_files"))
    assert (tmp_path / "downloaded_files" / "file1.txt").exists()
    assert (tmp_path / "downloaded_files" / "file2.txt").exists()
    assert (tmp_path / "downloaded_files" / "folder1" / "file3.txt").exists()
    assert (tmp_path / "downloaded_files" / "folder1" / "file4.txt").exists()
    assert (tmp_path / "downloaded_files" / "folder1" / "folder2" / "file5.txt").exists()


async def test_put_directory_respects_basepath(s3_bucket: S3Bucket, tmp_path: Path, aws_creds_block: AwsCredentials) -> None:
    (tmp_path / "file1.txt").write_text("FILE 1")
    (tmp_path / "file2.txt").write_text("FILE 2")
    (tmp_path / "folder1").mkdir()
    (tmp_path / "folder1" / "file3.txt").write_text("FILE 3")
    (tmp_path / "folder1" / "file4.txt").write_text("FILE 4")
    (tmp_path / "folder1" / "folder2").mkdir()
    (tmp_path / "folder1" / "folder2" / "file5.txt").write_text("FILE 5")
    s3_bucket_block = S3Bucket(bucket_name=BUCKET_NAME, aws_credentials=aws_creds_block, basepath="subfolder")
    uploaded_file_count: int = await s3_bucket_block.put_directory(local_path=str(tmp_path))
    assert uploaded_file_count == 5
    (tmp_path / "downloaded_files").mkdir()
    await s3_bucket_block.get_directory(local_path=str(tmp_path / "downloaded_files"))
    assert (tmp_path / "downloaded_files" / "file1.txt").exists()
    assert (tmp_path / "downloaded_files" / "file2.txt").exists()
    assert (tmp_path / "downloaded_files" / "folder1" / "file3.txt").exists()
    assert (tmp_path / "downloaded_files" / "folder1" / "file4.txt").exists()
    assert (tmp_path / "downloaded_files" / "folder1" / "folder2" / "file5.txt").exists()


async def test_put_directory_with_ignore_file(s3_bucket: S3Bucket, tmp_path: Path, aws_creds_block: AwsCredentials) -> None:
    (tmp_path / "file1.txt").write_text("FILE 1")
    (tmp_path / "file2.txt").write_text("FILE 2")
    (tmp_path / "folder1").mkdir()
    (tmp_path / "folder1" / "file3.txt").write_text("FILE 3")
    (tmp_path / "folder1" / "file4.txt").write_text("FILE 4")
    (tmp_path / "folder1" / "folder2").mkdir()
    (tmp_path / "folder1" / "folder2" / "file5.txt").write_text("FILE 5")
    (tmp_path / ".prefectignore").write_text("folder2/*")
    uploaded_file_count: int = await s3_bucket.put_directory(
        local_path=str(tmp_path / "folder1"), ignore_file=str(tmp_path / ".prefectignore")
    )
    assert uploaded_file_count == 2
    (tmp_path / "downloaded_files").mkdir()
    await s3_bucket.get_directory(local_path=str(tmp_path / "downloaded_files"))
    assert (tmp_path / "downloaded_files" / "file3.txt").exists()
    assert (tmp_path / "downloaded_files" / "file4.txt").exists()
    assert not (tmp_path / "downloaded_files" / "folder2").exists()
    assert not (tmp_path / "downloaded_files" / "folder2" / "file5.txt").exists()


async def test_put_directory_respects_local_path(s3_bucket: S3Bucket, tmp_path: Path, aws_creds_block: AwsCredentials) -> None:
    (tmp_path / "file1.txt").write_text("FILE 1")
    (tmp_path / "file2.txt").write_text("FILE 2")
    (tmp_path / "folder1").mkdir()
    (tmp_path / "folder1" / "file3.txt").write_text("FILE 3")
    (tmp_path / "folder1" / "file4.txt").write_text("FILE 4")
    (tmp_path / "folder1" / "folder2").mkdir()
    (tmp_path / "folder1" / "folder2" / "file5.txt").write_text("FILE 5")
    uploaded_file_count: int = await s3_bucket.put_directory(local_path=str(tmp_path / "folder1"))
    assert uploaded_file_count == 3
    (tmp_path / "downloaded_files").mkdir()
    await s3_bucket.get_directory(local_path=str(tmp_path / "downloaded_files"))
    assert (tmp_path / "downloaded_files" / "file3.txt").exists()
    assert (tmp_path / "downloaded_files" / "file4.txt").exists()
    assert (tmp_path / "downloaded_files" / "folder2" / "file5.txt").exists()


def test_read_path_in_sync_context(s3_bucket_with_file: tuple[S3Bucket, str]) -> None:
    """Test that read path works in a sync context."""
    s3_bucket, key = s3_bucket_with_file
    content: bytes = s3_bucket.read_path(key)
    assert content == b"hello"


def test_write_path_in_sync_context(s3_bucket: S3Bucket) -> None:
    """Test that write path works in a sync context."""
    key = s3_bucket.write_path("test.txt", content=b"hello")
    content: bytes = s3_bucket.read_path(key)
    assert content == b"hello"


def test_resolve_path(s3_bucket: S3Bucket) -> None:
    resolved = s3_bucket._resolve_path("")
    assert resolved == ""


class TestS3Bucket:
    @pytest.fixture(params=[AwsCredentials(), MinIOCredentials(minio_root_user="root", minio_root_password="password")])
    def credentials(self, request: pytest.FixtureRequest) -> Generator[Any, None, Any]:
        with mock_s3():
            yield request.param

    @pytest.fixture
    def s3_bucket_empty(self, credentials: Any, bucket: Any) -> S3Bucket:
        _s3_bucket = S3Bucket(bucket_name="bucket", credentials=credentials)
        return _s3_bucket

    @pytest.fixture
    def s3_bucket_2_empty(self, credentials: Any, bucket_2: Any) -> S3Bucket:
        _s3_bucket = S3Bucket(bucket_name="bucket_2", credentials=credentials, bucket_folder="subfolder")
        return _s3_bucket

    @pytest.fixture
    def s3_bucket_with_object(self, s3_bucket_empty: S3Bucket, object: None) -> S3Bucket:
        _s3_bucket_with_object = s3_bucket_empty
        return _s3_bucket_with_object

    @pytest.fixture
    def s3_bucket_2_with_object(self, s3_bucket_2_empty: S3Bucket) -> S3Bucket:
        _s3_bucket_with_object = s3_bucket_2_empty
        s3_bucket_2_empty.write_path("object", content=b"TEST")
        return _s3_bucket_with_object

    @pytest.fixture
    def s3_bucket_with_objects(self, s3_bucket_with_object: S3Bucket, object_in_folder: None) -> S3Bucket:
        _s3_bucket_with_objects = s3_bucket_with_object
        return _s3_bucket_with_objects

    @pytest.fixture
    def s3_bucket_with_similar_objects(self, s3_bucket_with_objects: S3Bucket, objects_in_folder: list[Any]) -> S3Bucket:
        _s3_bucket_with_multiple_objects = s3_bucket_with_objects
        return _s3_bucket_with_multiple_objects

    def test_credentials_are_correct_type(self, credentials: Any) -> None:
        s3_bucket = S3Bucket(bucket_name="bucket", credentials=credentials)
        s3_bucket_parsed = S3Bucket.model_validate({"bucket_name": "bucket", "credentials": dict(credentials)})
        assert isinstance(s3_bucket.credentials, type(credentials))
        assert isinstance(s3_bucket_parsed.credentials, type(credentials))

    @pytest.mark.parametrize("client_parameters", aws_clients[-1:], indirect=True)
    def test_list_objects_empty(self, s3_bucket_empty: S3Bucket, client_parameters: Any) -> None:
        assert s3_bucket_empty.list_objects() == []

    @pytest.mark.parametrize("client_parameters", aws_clients[-1:], indirect=True)
    def test_list_objects_one(self, s3_bucket_with_object: S3Bucket, client_parameters: Any) -> None:
        objects = s3_bucket_with_object.list_objects()
        assert len(objects) == 1
        assert [obj["Key"] for obj in objects] == ["object"]

    @pytest.mark.parametrize("client_parameters", aws_clients[-1:], indirect=True)
    def test_list_objects(self, s3_bucket_with_objects: S3Bucket, client_parameters: Any) -> None:
        objects = s3_bucket_with_objects.list_objects()
        assert len(objects) == 2
        assert [obj["Key"] for obj in objects] == ["folder/object", "object"]

    @pytest.mark.parametrize("to_path", [Path("to_path"), "to_path", None])
    @pytest.mark.parametrize("client_parameters", aws_clients[-1:], indirect=True)
    def test_download_object_to_path(
        self, s3_bucket_with_object: S3Bucket, to_path: Optional[Union[Path, str]], client_parameters: Any, tmp_path: Path
    ) -> None:
        os.chdir(tmp_path)
        s3_bucket_with_object.download_object_to_path("object", to_path)
        if to_path is None:
            to_path = tmp_path / "object"
        to_path_converted = Path(to_path)
        assert to_path_converted.read_text() == "TEST"

    @pytest.mark.parametrize("client_parameters", aws_clients[-1:], indirect=True)
    def test_download_object_to_file_object(self, s3_bucket_with_object: S3Bucket, client_parameters: Any, tmp_path: Path) -> None:
        to_path = tmp_path / "object"
        with open(to_path, "wb") as f:
            s3_bucket_with_object.download_object_to_file_object("object", f)
        assert to_path.read_text() == "TEST"

    @pytest.mark.parametrize("to_path", [Path("to_path"), "to_path", None])
    @pytest.mark.parametrize("client_parameters", aws_clients[-1:], indirect=True)
    def test_download_folder_to_path(
        self, s3_bucket_with_objects: S3Bucket, client_parameters: Any, tmp_path: Path, to_path: Optional[Union[Path, str]]
    ) -> None:
        os.chdir(tmp_path)
        s3_bucket_with_objects.download_folder_to_path("folder", to_path)
        if to_path is None:
            to_path = ""
        to_path_converted = Path(to_path)
        assert (to_path_converted / "object").read_text() == "TEST OBJECT IN FOLDER"

    @pytest.mark.parametrize("client_parameters", aws_clients[-1:], indirect=True)
    def test_download_object_with_bucket_folder(
        self, s3_bucket_empty: S3Bucket, client_parameters: Any, tmp_path: Path
    ) -> None:
        """regression test for https://github.com/PrefectHQ/prefect/issues/12848"""
        s3_bucket_empty.bucket_folder = "some_folder"
        test_content: bytes = b"This is a test file."
        s3_bucket_empty.upload_from_file_object(io.BytesIO(test_content), to_path="testing.txt")
        objects = s3_bucket_empty.list_objects()
        assert len(objects) == 1
        assert objects[0]["Key"] == "some_folder/testing.txt"
        download_path: Path = tmp_path / "downloaded_test_file.txt"
        s3_bucket_empty.download_object_to_path("testing.txt", download_path)
        assert download_path.read_bytes() == test_content
        s3_bucket_empty.credentials.get_s3_client().delete_object(
            Bucket=s3_bucket_empty.bucket_name, Key=s3_bucket_empty._join_bucket_folder("testing.txt")
        )

    @pytest.mark.parametrize("to_path", ["to_path", None])
    @pytest.mark.parametrize("client_parameters", aws_clients[-1:], indirect=True)
    def test_stream_from(
        self,
        s3_bucket_2_with_object: S3Bucket,
        s3_bucket_empty: S3Bucket,
        client_parameters: Any,
        to_path: Optional[Union[str, Path]],
    ) -> None:
        path: str = s3_bucket_empty.stream_from(s3_bucket_2_with_object, "object", to_path)
        data: bytes = s3_bucket_empty.read_path(path)
        assert data == b"TEST"

    @pytest.mark.parametrize("to_path", ["new_object", None])
    @pytest.mark.parametrize("client_parameters", aws_clients[-1:], indirect=True)
    def test_upload_from_path(
        self, s3_bucket_empty: S3Bucket, client_parameters: Any, tmp_path: Path, to_path: Optional[Union[str, Path]]
    ) -> None:
        from_path: Path = tmp_path / "new_object"
        from_path.write_text("NEW OBJECT")
        s3_bucket_empty.upload_from_path(from_path, to_path)
        with io.BytesIO() as buf:
            s3_bucket_empty.download_object_to_file_object("new_object", buf)
            buf.seek(0)
            output: bytes = buf.read()
        assert output == b"NEW OBJECT"

    @pytest.mark.parametrize("client_parameters", aws_clients[-1:], indirect=True)
    def test_upload_from_file_object(self, s3_bucket_empty: S3Bucket, client_parameters: Any, tmp_path: Path) -> None:
        file_path: Path = tmp_path / "hello"
        with open(file_path, "wb") as f:
            f.write(b"NEW OBJECT")
        with open(file_path, "rb") as f:
            s3_bucket_empty.upload_from_file_object(f, "new_object")
        with io.BytesIO() as buf:
            s3_bucket_empty.download_object_to_file_object("new_object", buf)
            buf.seek(0)
            output: bytes = buf.read()
        assert output == b"NEW OBJECT"

    @pytest.mark.parametrize("client_parameters", aws_clients[-1:], indirect=True)
    def test_upload_from_folder(self, s3_bucket_empty: S3Bucket, client_parameters: Any, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        from_path: Path = tmp_path / "new_object"
        from_path.write_text("NEW OBJECT")
        other_path: Path = tmp_path / "other_object"
        other_path.write_text("OTHER OBJECT")
        folder_dir: Path = tmp_path / "folder"
        folder_dir.mkdir()
        folder_path: Path = folder_dir / "other_object"
        folder_path.write_text("FOLDER OBJECT")
        s3_bucket_empty.upload_from_folder(tmp_path)
        new_from_path: Path = tmp_path / "downloaded_new_object"
        s3_bucket_empty.download_object_to_path("new_object", new_from_path)
        assert new_from_path.read_text() == "NEW OBJECT"
        new_other_path: Path = tmp_path / "downloaded_other_object"
        s3_bucket_empty.download_object_to_path("other_object", new_other_path)
        assert new_other_path.read_text() == "OTHER OBJECT"
        new_folder_path: Path = tmp_path / "downloaded_folder_object"
        s3_bucket_empty.download_object_to_path("folder/other_object", new_folder_path)
        assert new_folder_path.read_text() == "FOLDER OBJECT"
        empty_folder: Path = tmp_path / "empty_folder"
        empty_folder.mkdir()
        s3_bucket_empty.upload_from_folder(empty_folder)
        for record in caplog.records:
            if "No files were uploaded from {empty_folder}" in record.message:
                break
        else:
            raise AssertionError("Files did upload")

    @pytest.mark.parametrize("client_parameters", aws_clients[-1:], indirect=True)
    def test_copy_object(self, s3_bucket_with_object: S3Bucket, s3_bucket_2_empty: S3Bucket) -> None:
        s3_bucket_with_object.copy_object("object", "object_copy_1")
        assert s3_bucket_with_object.read_path("object_copy_1") == b"TEST"
        s3_bucket_with_object.copy_object("object", "folder/object_copy_2")
        assert s3_bucket_with_object.read_path("folder/object_copy_2") == b"TEST"
        s3_bucket_with_object.copy_object(
            "object", s3_bucket_2_empty._resolve_path("object_copy_3"), to_bucket="bucket_2"
        )
        assert s3_bucket_2_empty.read_path("object_copy_3") == b"TEST"
        s3_bucket_with_object.copy_object("object", "object_copy_4", s3_bucket_2_empty)
        assert s3_bucket_2_empty.read_path("object_copy_4") == b"TEST"

    @pytest.mark.parametrize("client_parameters", aws_clients[-1:], indirect=True)
    @pytest.mark.parametrize(
        "to_bucket, bucket_folder, expected_path",
        [
            (None, None, "object"),
            (None, "subfolder", "subfolder/object"),
            ("bucket_2", None, "object"),
            (None, None, "object"),
            (None, "subfolder", "subfolder/object"),
            ("bucket_2", None, "object"),
        ],
    )
    def test_copy_subpaths(
        self,
        s3_bucket_with_object: S3Bucket,
        s3_bucket_2_empty: S3Bucket,
        to_bucket: Optional[Any],
        bucket_folder: Optional[str],
        expected_path: str,
        client_parameters: Any,
    ) -> None:
        if to_bucket is None:
            to_bucket = s3_bucket_2_empty
            if bucket_folder is not None:
                to_bucket.bucket_folder = bucket_folder
            else:
                to_bucket.bucket_folder = None
        key: str = s3_bucket_with_object.copy_object("object", "object", to_bucket=to_bucket)
        assert key == expected_path

    @pytest.mark.parametrize("client_parameters", aws_clients[-1:], indirect=True)
    def test_move_object_within_bucket(self, s3_bucket_with_object: S3Bucket, client_parameters: Any) -> None:
        s3_bucket_with_object.move_object("object", "object_copy_1")
        assert s3_bucket_with_object.read_path("object_copy_1") == b"TEST"
        with pytest.raises(ClientError):
            s3_bucket_with_object.read_path("object")

    @pytest.mark.parametrize("client_parameters", aws_clients[-1:], indirect=True)
    def test_move_object_to_nonexistent_bucket_fails(self, s3_bucket_with_object: S3Bucket, client_parameters: Any) -> None:
        with pytest.raises(ClientError):
            s3_bucket_with_object.move_object("object", "object_copy_1", to_bucket="nonexistent-bucket")
        assert s3_bucket_with_object.read_path("object") == b"TEST"

    @pytest.mark.parametrize("client_parameters", aws_clients[-1:], indirect=True)
    def test_move_object_onto_itself_fails(self, s3_bucket_with_object: S3Bucket, client_parameters: Any) -> None:
        with pytest.raises(ClientError):
            s3_bucket_with_object.move_object("object", "object")
        assert s3_bucket_with_object.read_path("object") == b"TEST"

    @pytest.mark.parametrize("client_parameters", aws_clients[-1:], indirect=True)
    def test_move_object_between_buckets(
        self, s3_bucket_with_object: S3Bucket, s3_bucket_2_empty: S3Bucket, client_parameters: Any
    ) -> None:
        s3_bucket_with_object.move_object("object", "object_copy_1", to_bucket=s3_bucket_2_empty)
        assert s3_bucket_2_empty.read_path("object_copy_1") == b"TEST"
        with pytest.raises(ClientError):
            s3_bucket_with_object.read_path("object")

    @pytest.mark.parametrize("client_parameters", aws_clients[-1:], indirect=True)
    @pytest.mark.parametrize(
        "to_bucket, bucket_folder, expected_path",
        [
            (None, None, "object"),
            (None, "subfolder", "subfolder/object"),
            ("bucket_2", None, "object"),
            (None, None, "object"),
            (None, "subfolder", "subfolder/object"),
            ("bucket_2", None, "object"),
        ],
    )
    def test_move_subpaths(
        self,
        s3_bucket_with_object: S3Bucket,
        s3_bucket_2_empty: S3Bucket,
        to_bucket: Optional[Any],
        bucket_folder: Optional[str],
        expected_path: str,
        client_parameters: Any,
    ) -> None:
        if to_bucket is None:
            to_bucket = s3_bucket_2_empty
            if bucket_folder is not None:
                to_bucket.bucket_folder = bucket_folder
            else:
                to_bucket.bucket_folder = None
        key: str = s3_bucket_with_object.move_object("object", "object", to_bucket=to_bucket)
        assert key == expected_path

    def test_round_trip_default_credentials(self) -> None:
        S3Bucket(bucket_name="round-trip-bucket").save("round-tripper")
        loaded: S3Bucket = S3Bucket.load("round-tripper")
        assert hasattr(loaded.credentials, "aws_access_key_id"), "`credentials` were not properly initialized"

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
    async def test_async_download_from_bucket(self, object: None, client_parameters: Any, aws_credentials: Any) -> None:
        @flow
        async def test_flow() -> bytes:
            return await adownload_from_bucket(
                bucket="bucket", key="object", aws_credentials=aws_credentials, aws_client_parameters=client_parameters
            )

        result: bytes = await test_flow()
        assert result == b"TEST"

    @pytest.mark.parametrize("client_parameters", aws_clients, indirect=True)
    async def test_async_list_objects(self, object: None, object_in_folder: None, client_parameters: Any, aws_credentials: Any) -> None:
        @flow
        async def test_flow() -> list[dict[str, Any]]:
            return await alist_objects(
                bucket="bucket", aws_credentials=aws_credentials, aws_client_parameters=client_parameters
            )

        objects: list[dict[str, Any]] = await test_flow()
        assert len(objects) == 2
        assert [obj["Key"] for obj in objects] == ["folder/object", "object"]

    @pytest.mark.parametrize("client_parameters", aws_clients, indirect=True)
    async def test_async_copy_objects(self, object: None, bucket: Any, bucket_2: Any, aws_credentials: Any) -> None:
        def read(bucket_obj: Any, key: str) -> bytes:
            stream = io.BytesIO()
            bucket_obj.download_fileobj(key, stream)
            stream.seek(0)
            return stream.read()

        @flow
        async def test_flow() -> None:
            await acopy_objects(
                source_path="object",
                target_path="subfolder/new_object",
                source_bucket_name="bucket",
                aws_credentials=aws_credentials,
                target_bucket_name="bucket_2",
            )

        await test_flow()
        assert read(bucket_2, "subfolder/new_object") == b"TEST"

    @pytest.mark.parametrize("client_parameters", aws_clients, indirect=True)
    async def test_async_move_objects(self, object: None, bucket: Any, bucket_2: Any, aws_credentials: Any) -> None:
        def read(bucket_obj: Any, key: str) -> bytes:
            stream = io.BytesIO()
            bucket_obj.download_fileobj(key, stream)
            stream.seek(0)
            return stream.read()

        @flow
        async def test_flow() -> None:
            await amove_objects(
                source_path="object",
                target_path="moved_object",
                source_bucket_name="bucket",
                target_bucket_name="bucket_2",
                aws_credentials=aws_credentials,
            )

        await test_flow()
        assert read(bucket_2, "moved_object") == b"TEST"
        with pytest.raises(ClientError):
            read(bucket, "object")
