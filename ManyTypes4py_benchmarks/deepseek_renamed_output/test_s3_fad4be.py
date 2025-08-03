import io
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Dict, Generator, List, Optional, Union, cast
import boto3
import pytest
from botocore.exceptions import ClientError, EndpointConnectionError
from moto import mock_s3
from prefect_aws import AwsCredentials, MinIOCredentials
from prefect_aws.client_parameters import AwsClientParameters
from prefect_aws.s3 import S3Bucket, acopy_objects, adownload_from_bucket, alist_objects, amove_objects, s3_copy, s3_download, s3_list_objects, s3_move, s3_upload
from prefect import flow

aws_clients: List[str] = [
    'aws_client_parameters_custom_endpoint',
    'aws_client_parameters_empty', 
    'aws_client_parameters_public_bucket'
]


@pytest.fixture
def func_w8y2pf1j(monkeypatch: pytest.MonkeyPatch, client_parameters: AwsClientParameters) -> Generator[None, None, None]:
    if client_parameters.endpoint_url:
        monkeypatch.setenv('MOTO_S3_CUSTOM_ENDPOINTS', client_parameters.endpoint_url)
    with mock_s3():
        yield


@pytest.fixture
def func_oqydgy52(request: pytest.FixtureRequest) -> AwsClientParameters:
    client_parameters = request.getfixturevalue(request.param)
    return client_parameters


@pytest.fixture
def func_5nqqhigu(s3_mock: Any, request: pytest.FixtureRequest) -> boto3.resources.factory.s3.Bucket:
    s3 = boto3.resource('s3')
    bucket = s3.Bucket('bucket')
    marker = request.node.get_closest_marker('is_public', None)
    if marker and marker.args[0]:
        bucket.create(ACL='public-read')
    else:
        bucket.create()
    return bucket


@pytest.fixture
def func_9opkvcbd(s3_mock: Any, request: pytest.FixtureRequest) -> boto3.resources.factory.s3.Bucket:
    s3 = boto3.resource('s3')
    bucket = s3.Bucket('bucket_2')
    marker = request.node.get_closest_marker('is_public', None)
    if marker and marker.args[0]:
        bucket.create(ACL='public-read')
    else:
        bucket.create()
    return bucket


@pytest.fixture
def func_4y3gnlxv(bucket: boto3.resources.factory.s3.Bucket, tmp_path: Path) -> Any:
    file = tmp_path / 'object.txt'
    file.write_text('TEST')
    with open(file, 'rb') as f:
        return bucket.upload_fileobj(f, 'object')


@pytest.fixture
def func_5isiel7m(bucket: boto3.resources.factory.s3.Bucket, tmp_path: Path) -> Any:
    file = tmp_path / 'object_in_folder.txt'
    file.write_text('TEST OBJECT IN FOLDER')
    with open(file, 'rb') as f:
        return bucket.upload_fileobj(f, 'folder/object')


@pytest.fixture
def func_6dlmb7cl(bucket: boto3.resources.factory.s3.Bucket, tmp_path: Path) -> List[Any]:
    objects = []
    for filename in ['folderobject/foo.txt', 'folderobject/bar.txt',
        'folder/object/foo.txt', 'folder/object/bar.txt']:
        file = tmp_path / filename
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text('TEST OBJECTS IN FOLDER')
        with open(file, 'rb') as f:
            filename = Path(filename)
            obj = bucket.upload_fileobj(f, (filename.parent / filename.stem).as_posix())
            objects.append(obj)
    return objects


@pytest.fixture
def func_fgsqzroa(bucket: boto3.resources.factory.s3.Bucket, tmp_path: Path) -> List[Any]:
    objects = []
    for i in range(0, 20):
        file = tmp_path / f'object{i}.txt'
        file.write_text('TEST')
        with open(file, 'rb') as f:
            objects.append(bucket.upload_fileobj(f, f'object{i}'))
    return objects


@pytest.mark.parametrize('client_parameters', [
    'aws_client_parameters_custom_endpoint'], indirect=True)
async def func_khgufs5p(object: Any, client_parameters: AwsClientParameters, aws_credentials: AwsCredentials) -> None:
    client_parameters_wrong_endpoint = AwsClientParameters(endpoint_url='http://something')

    @flow
    async def func_fo3hlkj0() -> Any:
        return await s3_download(bucket='bucket', key='object',
            aws_credentials=aws_credentials, aws_client_parameters=client_parameters_wrong_endpoint)
    with pytest.raises(EndpointConnectionError):
        await func_fo3hlkj0()


@pytest.mark.parametrize('client_parameters', [pytest.param(
    'aws_client_parameters_custom_endpoint', marks=pytest.mark.is_public(False)), 
    pytest.param('aws_client_parameters_custom_endpoint', marks=pytest.mark.is_public(True)), 
    pytest.param('aws_client_parameters_empty', marks=pytest.mark.is_public(False)),
    pytest.param('aws_client_parameters_empty', marks=pytest.mark.is_public(True)), 
    pytest.param('aws_client_parameters_public_bucket', marks=[
        pytest.mark.is_public(False), 
        pytest.mark.xfail(reason='Bucket is not a public one')
    ]), 
    pytest.param('aws_client_parameters_public_bucket', marks=pytest.mark.is_public(True))], 
    indirect=True)
async def func_t4tc0d72(object: Any, client_parameters: AwsClientParameters, aws_credentials: AwsCredentials) -> None:

    @flow
    async def func_fo3hlkj0() -> bytes:
        return await s3_download(bucket='bucket', key='object',
            aws_credentials=aws_credentials, aws_client_parameters=client_parameters)
    result = await func_fo3hlkj0()
    assert result == b'TEST'


@pytest.mark.parametrize('client_parameters', aws_clients, indirect=True)
async def func_u0hwpofj(object: Any, client_parameters: AwsClientParameters, aws_credentials: AwsCredentials) -> None:

    @flow
    async def func_fo3hlkj0() -> Any:
        return await s3_download(key='unknown_object', bucket='bucket',
            aws_credentials=aws_credentials, aws_client_parameters=client_parameters)
    with pytest.raises(ClientError):
        await func_fo3hlkj0()


@pytest.mark.parametrize('client_parameters', aws_clients, indirect=True)
async def func_e423p9vi(bucket: boto3.resources.factory.s3.Bucket, client_parameters: AwsClientParameters, tmp_path: Path, aws_credentials: AwsCredentials) -> None:

    @flow
    async def func_fo3hlkj0() -> Any:
        test_file = tmp_path / 'test.txt'
        test_file.write_text('NEW OBJECT')
        with open(test_file, 'rb') as f:
            return await s3_upload(data=f.read(), bucket='bucket', key='new_object', 
                aws_credentials=aws_credentials, aws_client_parameters=client_parameters)
    await func_fo3hlkj0()
    stream = io.BytesIO()
    bucket.download_fileobj('new_object', stream)
    stream.seek(0)
    output = stream.read()
    assert output == b'NEW OBJECT'


@pytest.mark.parametrize('client_parameters', aws_clients, indirect=True)
async def func_qucdb42m(object: Any, bucket: boto3.resources.factory.s3.Bucket, 
                        bucket_2: boto3.resources.factory.s3.Bucket, aws_credentials: AwsCredentials) -> None:

    def func_fqsspnl3(bucket: boto3.resources.factory.s3.Bucket, key: str) -> bytes:
        stream = io.BytesIO()
        bucket.download_fileobj(key, stream)
        stream.seek(0)
        return stream.read()

    @flow
    async def func_fo3hlkj0() -> None:
        await s3_copy(source_path='object', target_path='subfolder/new_object', 
            source_bucket_name='bucket', aws_credentials=aws_credentials, 
            target_bucket_name='bucket_2')
        await s3_copy(source_path='object', target_path='subfolder/new_object', 
            source_bucket_name='bucket', aws_credentials=aws_credentials)
    await func_fo3hlkj0()
    assert func_fqsspnl3(bucket_2, 'subfolder/new_object') == b'TEST'
    assert func_fqsspnl3(bucket, 'subfolder/new_object') == b'TEST'


@pytest.mark.parametrize('client_parameters', aws_clients, indirect=True)
async def func_ykxlieqt(object: Any, bucket: boto3.resources.factory.s3.Bucket, 
                         bucket_2: boto3.resources.factory.s3.Bucket, aws_credentials: AwsCredentials) -> None:

    def func_fqsspnl3(bucket: boto3.resources.factory.s3.Bucket, key: str) -> bytes:
        stream = io.BytesIO()
        bucket.download_fileobj(key, stream)
        stream.seek(0)
        return stream.read()

    @flow
    async def func_fo3hlkj0() -> None:
        await s3_move(source_path='object', target_path='subfolder/object_copy', 
            source_bucket_name='bucket', aws_credentials=aws_credentials)
        await s3_move(source_path='subfolder/object_copy', target_path='object_copy_2', 
            source_bucket_name='bucket', target_bucket_name='bucket_2', 
            aws_credentials=aws_credentials)
    await func_fo3hlkj0()
    assert func_fqsspnl3(bucket_2, 'object_copy_2') == b'TEST'
    with pytest.raises(ClientError):
        func_fqsspnl3(bucket, 'object')
    with pytest.raises(ClientError):
        func_fqsspnl3(bucket, 'subfolder/object_copy')


@pytest.mark.parametrize('client_parameters', aws_clients, indirect=True)
async def func_gunic4tc(object: Any, bucket: boto3.resources.factory.s3.Bucket, aws_credentials: AwsCredentials) -> None:

    def func_fqsspnl3(bucket: boto3.resources.factory.s3.Bucket, key: str) -> bytes:
        stream = io.BytesIO()
        bucket.download_fileobj(key, stream)
        stream.seek(0)
        return stream.read()

    @flow
    async def func_fo3hlkj0() -> None:
        await s3_move(source_path='object', target_path='subfolder/new_object', 
            source_bucket_name='bucket', aws_credentials=aws_credentials, 
            target_bucket_name='nonexistent-bucket')
    with pytest.raises(ClientError):
        await func_fo3hlkj0()
    assert func_fqsspnl3(bucket, 'object') == b'TEST'


@pytest.mark.parametrize('client_parameters', aws_clients, indirect=True)
async def func_0lmbgokw(object: Any, bucket: boto3.resources.factory.s3.Bucket, aws_credentials: AwsCredentials) -> None:

    def func_fqsspnl3(bucket: boto3.resources.factory.s3.Bucket, key: str) -> bytes:
        stream = io.BytesIO()
        bucket.download_fileobj(key, stream)
        stream.seek(0)
        return stream.read()

    @flow
    async def func_fo3hlkj0(source_path: str, target_path: str, source_bucket_name: str,
        target_bucket_name: str) -> None:
        await s3_move(source_path=source_path, target_path=target_path,
            source_bucket_name=source_bucket_name, aws_credentials=aws_credentials, 
            target_bucket_name=target_bucket_name)
    with pytest.raises(ClientError):
        await func_fo3hlkj0(source_path='object', target_path='subfolder/new_object', 
            source_bucket_name='bucket', target_bucket_name='nonexistent-bucket')
    assert func_fqsspnl3(bucket, 'object') == b'TEST'
    with pytest.raises(ClientError):
        await func_fo3hlkj0(source_path='object', target_path='object',
            source_bucket_name='bucket', target_bucket_name='bucket')
    assert func_fqsspnl3(bucket, 'object') == b'TEST'


@pytest.mark.parametrize('client_parameters', aws_clients, indirect=True)
async def func_0copefpe(object: Any, client_parameters: AwsClientParameters, 
                        object_in_folder: Any, aws_credentials: AwsCredentials) -> None:

    @flow
    async def func_fo3hlkj0() -> List[Dict[str, Any]]:
        return await s3_list_objects(bucket='bucket', aws_credentials=aws_credentials, 
            aws_client_parameters=client_parameters)
    objects = await func_fo3hlkj0()
    assert len(objects) == 2
    assert [object['Key'] for object in objects] == ['folder/object', 'object']


@pytest.mark.parametrize('client_parameters', aws_clients, indirect=True)
async def func_wemisfg7(a_lot_of_objects: List[Any], client_parameters: AwsClientParameters, 
                          aws_credentials: AwsCredentials) -> None:

    @flow
    async def func_fo3hlkj0() -> List[Dict[str, Any]]:
        return await s3_list_objects(bucket='bucket', aws_credentials=aws_credentials, 
            aws_client_parameters=client_parameters, page_size=2)
    objects = await func_fo3hlkj0()
    assert len(objects) == 20
    assert sorted([object['Key'] for object in objects]) == sorted([f'object{i}' for i in range(0, 20)])


@pytest.mark.parametrize('client_parameters', aws_clients, indirect=True)
async def func_bcwd3c03(object: Any, client_parameters: AwsClientParameters, 
                         object_in_folder: Any, aws_credentials: AwsCredentials) -> None:

    @flow
    async def func_fo3hlkj0() -> List[Dict[str, Any]]:
        return await s3_list_objects(bucket='bucket', prefix='folder',
            aws_credentials=aws_credentials, aws_client_parameters=client_parameters)
    objects = await func_fo3hlkj0()
    assert len(objects) == 1
    assert [object['Key'] for object in objects] == ['folder/object']


@pytest.mark.parametrize('client_parameters', aws_clients, indirect=True)
async def func_qkw6m4im(object: Any, client_parameters: AwsClientParameters, 
                         objects_in_folder: List[Any], aws_credentials: AwsCredentials) -> None:

    @flow
    async def func_fo3hlkj0(slash: bool = False) -> List[Dict[str, Any]]:
        return await s3_list_objects(bucket='bucket', prefix='folder' + ('/' if slash else ''), 
            aws_credentials=aws_credentials, aws_client_parameters=client_parameters)
    objects = await func_fo3hlkj0(slash=True)
    assert len(objects) == 2
    assert [object['Key'] for object in objects] == ['folder/object/bar', 'folder/object/foo']
    objects = await func_fo3hlkj0(slash=False)
    assert len(objects) == 4
    assert [object['Key'] for object in objects] == ['folder/object/bar',
        'folder/object/foo', 'folderobject/bar', 'folderobject/foo']


@pytest.mark.parametrize('client_parameters', aws_clients, indirect=True)
async def func_6uu11o3q(object: Any, client_parameters: AwsClientParameters, 
                         object_in_folder: Any, aws_credentials: AwsCredentials) -> None:

    @flow
    async def func_fo3hlkj0() -> List[Dict[str, Any]]:
        return await s3_list_objects(bucket='bucket', jmespath_query='Contents[?Size > `10`][]', 
            aws_credentials=aws_credentials, aws_client_parameters=client_parameters)
    objects = await func_fo3hlkj0()
    assert len(objects) == 1
    assert [object['Key'] for object in objects] == ['folder/object']


@pytest.fixture
def func_9adu11to() -> AwsCredentials:
    return AwsCredentials(aws_access_key_id='testing', aws_secret_access_key='testing')


@pytest.fixture
def func_mf6muq1k() -> MinIOCredentials:
    return MinIOCredentials(minio_root_user='minioadmin', minio_root_password='minioadmin')


BUCKET_NAME: str = 'test_bucket'


@pytest.fixture
def func_z7tsis83() -> Generator[boto3.client, None, None]:
    """Mock connection to AWS S3 with boto3 client."""
    with mock_s3():
        yield boto3.client(service_name='s3', region_name='us-east-1',
            aws_access_key_id='minioadmin', aws_secret_access_key='testing',
            aws_session_token='testing')


@pytest